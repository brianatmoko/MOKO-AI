"""
Test Integrasi — RAG 200MB Server ↔ Pipeline Retrieval (Phase 3.4)
==================================================================
Memverifikasi bahwa model RAG khusus 200MB (moko-rag.gguf, port 11437)
benar-benar TERINTEGRASI ke pipeline, bukan sekadar terkonfigurasi:

1. Engine bisa menembak server RAG khusus (port 11437), BUKAN server utama
   (port 11435) — via generate_rag().
2. engine.rag_available() mencerminkan status server RAG yang sebenarnya.
3. RetrievalLayer MENDISTILASI konteks lewat model RAG saat server RAG hidup.
4. RetrievalLayer FALLBACK ke konteks mentah saat server RAG mati
   (pipeline tidak boleh putus).
5. RetrievalLayer memakai default port RAG yang benar (11437, bukan 11435).
6. server_manager.start_rag_server() punya guard model-hilang & membangun
   perintah peluncuran yang benar (port/model/ctx).
7. start_servers() OTOMATIS mem-boot server RAG setelah server utama online.

Test ini TIDAK memerlukan server hidup: semua I/O jaringan/proses di-monkeypatch.

Jalankan:
    PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_rag_integration.py
"""

import os
import sys
import tempfile


# ─────────────────────────────────────────────────────────────────────────
# 1. Engine menembak server RAG khusus (port 11437), bukan server utama
# ─────────────────────────────────────────────────────────────────────────
def test_generate_rag_targets_rag_port():
    from moko_agents import llm_engine
    from moko_config import settings

    captured = {}

    class _FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "  KONTEKS RINGKAS  "}}]}

    def fake_post(url, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResp()

    orig_post = llm_engine.requests.post
    try:
        llm_engine.requests.post = fake_post
        out = llm_engine.engine.generate_rag("halo", system_prompt="sys")
    finally:
        llm_engine.requests.post = orig_post

    assert out == "KONTEKS RINGKAS", f"output tidak di-strip/benar: {out!r}"
    assert str(settings.MOKO_RAG_PORT) in captured["url"], (
        f"generate_rag harus menembak port RAG {settings.MOKO_RAG_PORT}, "
        f"nyatanya: {captured['url']}"
    )
    assert str(settings.MOKO_LLM_PORT) not in captured["url"], (
        f"generate_rag TIDAK boleh menembak port utama: {captured['url']}"
    )
    print("  ✅ Engine.generate_rag menembak server RAG khusus (port 11437).")


# ─────────────────────────────────────────────────────────────────────────
# 2. rag_available() mencerminkan status server RAG
# ─────────────────────────────────────────────────────────────────────────
def test_rag_available_reflects_status():
    from moko_agents import llm_engine
    from moko_inference import server_manager

    orig = server_manager.MokoLocalInferenceServer.get_server_status

    try:
        server_manager.MokoLocalInferenceServer.get_server_status = staticmethod(
            lambda port: "ok"
        )
        assert llm_engine.engine.rag_available() is True

        server_manager.MokoLocalInferenceServer.get_server_status = staticmethod(
            lambda port: "offline"
        )
        assert llm_engine.engine.rag_available() is False
    finally:
        server_manager.MokoLocalInferenceServer.get_server_status = orig
    print("  ✅ engine.rag_available() mengikuti status server RAG.")


# ─────────────────────────────────────────────────────────────────────────
# Helper: RetrievalLayer dengan KnowledgeLayer palsu (tanpa DiskManager)
# ─────────────────────────────────────────────────────────────────────────
def _make_retrieval_layer():
    from moko_agents.layers.retrieval_layer import RetrievalLayer

    class _FakeKnowledge:
        def search_facts(self, emb, top_k=5, domain="general"):
            # Kembalikan satu fakta ber-skor tinggi untuk domain "general" saja.
            if domain == "general":
                return [{"domain": "general", "text": "Bumi mengelilingi Matahari.", "score": 0.9}]
            return []

    layer = RetrievalLayer.__new__(RetrievalLayer)
    layer.knowledge_layer = _FakeKnowledge()
    from moko_config import settings
    layer.rag_port = getattr(settings, "MOKO_RAG_PORT", 11437)
    return layer


# ─────────────────────────────────────────────────────────────────────────
# 3. Distilasi via RAG saat server hidup
# ─────────────────────────────────────────────────────────────────────────
def test_retrieval_distills_when_rag_up():
    from moko_agents import llm_engine
    from moko_agents.layers import retrieval_layer as rl

    layer = _make_retrieval_layer()

    orig_embed = llm_engine.engine.get_embedding
    orig_avail = llm_engine.engine.rag_available
    orig_gen = llm_engine.engine.generate_rag
    captured = {}
    try:
        rl.engine.get_embedding = lambda q: [0.1, 0.2, 0.3]
        rl.engine.rag_available = lambda: True

        def fake_gen(prompt, system_prompt="", coop_params=None):
            captured["prompt"] = prompt
            return "RINGKAS: Bumi mengorbit Matahari."

        rl.engine.generate_rag = fake_gen

        out = layer.retrieve_context("apa itu tata surya?")
    finally:
        rl.engine.get_embedding = orig_embed
        rl.engine.rag_available = orig_avail
        rl.engine.generate_rag = orig_gen

    assert out == "RINGKAS: Bumi mengorbit Matahari.", f"distilasi tak dipakai: {out!r}"
    assert "Bumi mengelilingi Matahari" in captured["prompt"], (
        "fakta mentah harus diteruskan ke model RAG untuk didistilasi"
    )
    print("  ✅ RetrievalLayer mendistilasi konteks via model RAG saat server hidup.")


# ─────────────────────────────────────────────────────────────────────────
# 4. Fallback ke konteks mentah saat server RAG mati
# ─────────────────────────────────────────────────────────────────────────
def test_retrieval_falls_back_when_rag_down():
    from moko_agents import llm_engine
    from moko_agents.layers import retrieval_layer as rl

    layer = _make_retrieval_layer()

    orig_embed = llm_engine.engine.get_embedding
    orig_avail = llm_engine.engine.rag_available
    orig_gen = llm_engine.engine.generate_rag
    gen_called = {"n": 0}
    try:
        rl.engine.get_embedding = lambda q: [0.1, 0.2, 0.3]
        rl.engine.rag_available = lambda: False

        def fake_gen(*a, **k):
            gen_called["n"] += 1
            return "TIDAK BOLEH DIPANGGIL"

        rl.engine.generate_rag = fake_gen
        out = layer.retrieve_context("apa itu tata surya?")
    finally:
        rl.engine.get_embedding = orig_embed
        rl.engine.rag_available = orig_avail
        rl.engine.generate_rag = orig_gen

    assert "Bumi mengelilingi Matahari" in out, f"konteks mentah hilang: {out!r}"
    assert gen_called["n"] == 0, "generate_rag TIDAK boleh dipanggil saat server RAG mati"
    print("  ✅ RetrievalLayer fallback ke konteks mentah saat server RAG mati.")


# ─────────────────────────────────────────────────────────────────────────
# 5. Default port RAG yang benar (11437)
# ─────────────────────────────────────────────────────────────────────────
def test_retrieval_default_port():
    from moko_config import settings
    from moko_agents.layers.retrieval_layer import RetrievalLayer

    layer = RetrievalLayer.__new__(RetrievalLayer)
    layer.knowledge_layer = None
    # Panggil __init__ hanya untuk mengecek default port lewat jalur normal:
    RetrievalLayer.__init__(layer, None)
    assert layer.rag_port == settings.MOKO_RAG_PORT == 11437, (
        f"default port RAG salah: {layer.rag_port} (harus 11437)"
    )
    print("  ✅ RetrievalLayer memakai default port RAG yang benar (11437).")


# ─────────────────────────────────────────────────────────────────────────
# 6. start_rag_server(): guard model-hilang + perintah peluncuran benar
# ─────────────────────────────────────────────────────────────────────────
def test_start_rag_server_guard_and_cmd():
    from moko_inference import server_manager as sm
    from moko_config import settings

    S = sm.MokoLocalInferenceServer

    orig_path = settings.MODEL_RAG_LLM_PATH
    orig_status = S.get_server_status
    orig_popen = sm.subprocess.Popen
    orig_gpu = sm.get_gpu_layers

    try:
        # (a) Guard: model tidak ada → False, tanpa memanggil Popen.
        settings.MODEL_RAG_LLM_PATH = "/tmp/__tidak_ada_moko_rag__.gguf"
        S.get_server_status = staticmethod(lambda port: "offline")
        popen_calls = {"n": 0}
        sm.subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Popen tidak boleh dipanggil saat model RAG hilang")
        )
        assert S.start_rag_server() is False, "guard model-hilang gagal"

        # (b) Model ada → membangun perintah peluncuran yang benar.
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(b"GGUF")
        tmp.close()
        settings.MODEL_RAG_LLM_PATH = tmp.name
        sm.get_gpu_layers = lambda is_rag=False: 4

        captured = {}

        class _FakeProc:
            pid = 424242

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _FakeProc()

        sm.subprocess.Popen = fake_popen
        ok = S.start_rag_server()
        assert ok is True, "start_rag_server harus True saat model ada"

        cmd = captured["cmd"]
        assert str(settings.MOKO_RAG_PORT) in cmd, f"port RAG tak ada di cmd: {cmd}"
        assert tmp.name in cmd, f"path model RAG tak ada di cmd: {cmd}"
        assert str(settings.RAG_CONTEXT_WINDOW) in cmd, f"ctx RAG tak ada di cmd: {cmd}"
        os.unlink(tmp.name)
    finally:
        settings.MODEL_RAG_LLM_PATH = orig_path
        S.get_server_status = orig_status
        sm.subprocess.Popen = orig_popen
        sm.get_gpu_layers = orig_gpu
        # Bersihkan PID file uji agar tidak menipu status daemon.
        try:
            if os.path.exists(sm.RAG_PID_FILE):
                os.remove(sm.RAG_PID_FILE)
        except Exception:
            pass
    print("  ✅ start_rag_server: guard model-hilang + perintah peluncuran benar.")


# ─────────────────────────────────────────────────────────────────────────
# 7. start_servers() OTOMATIS mem-boot server RAG
# ─────────────────────────────────────────────────────────────────────────
def test_start_servers_auto_boots_rag():
    from moko_inference import server_manager as sm

    S = sm.MokoLocalInferenceServer

    orig_status = S.get_server_status
    orig_gpu = sm.get_gpu_layers
    orig_start_rag = S.start_rag_server

    boot = {"n": 0}
    try:
        # Server utama langsung "ok" → start_servers masuk cabang online.
        S.get_server_status = staticmethod(lambda port: "ok")
        sm.get_gpu_layers = lambda is_rag=False: 0  # CPU mode → lewati setup CUDA

        S.start_rag_server = classmethod(lambda cls: boot.__setitem__("n", boot["n"] + 1))

        result = S.start_servers()
        assert result is True, "start_servers harus True saat server utama online"
        assert boot["n"] == 1, "start_servers WAJIB memanggil start_rag_server sekali"
    finally:
        S.get_server_status = orig_status
        sm.get_gpu_layers = orig_gpu
        S.start_rag_server = orig_start_rag
    print("  ✅ start_servers() otomatis mem-boot server RAG setelah utama online.")


def main():
    print("\n🧪 Integrasi RAG 200MB ↔ Pipeline Retrieval (Phase 3.4)\n")
    test_generate_rag_targets_rag_port()
    test_rag_available_reflects_status()
    test_retrieval_distills_when_rag_up()
    test_retrieval_falls_back_when_rag_down()
    test_retrieval_default_port()
    test_start_rag_server_guard_and_cmd()
    test_start_servers_auto_boots_rag()
    print("\n✅ SEMUA ASSERT LULUS: RAG 200MB terintegrasi ke pipeline retrieval.\n")


if __name__ == "__main__":
    main()
