"""
Test Integrasi Menyeluruh — Phase 3.5 (Architecture Upgrade End-to-End)
======================================================================
Phase 3.5 adalah gerbang PENGUJIAN yang MENAUTKAN seluruh komponen Phase 3
(3.1 Intent Router → 3.2 Byte-Q → 3.3 Dispatcher → 3.4 RAG 200MB) menjadi
rantai lintas-komponen yang benar-benar bekerja bersama — bukan sekadar
lulus secara unit terpisah.

Rantai E2E yang diverifikasi (semua server-free / di-monkeypatch):

  1. Router → Dispatcher (3.1↔3.3): intent hasil IntentFirstRouter memilih
     model spesialis yang benar; switch_to mengaktifkannya dan mengekspos
     parameter generasi khas domain (temperature/context_window).
  2. Byte-Q dipreferensikan di registry (3.2↔3.3): saat ada saudara
     `.byteq.gguf`, dispatcher memilihnya (jembatan dequantize-before-load)
     dan mengoreksi anggaran VRAM (~40% lebih hemat). resolve_loadable_model
     meneruskan model GGUF biasa apa adanya.
  3. Skip-reload router-driven (3.3): percakapan multi-intent nyata (coding →
     math → general → coding) yang semuanya menunjuk file GGUF sama hanya
     memicu SATU load nyata (efisiensi VRAM lintas giliran).
  4. Router → CoreNode (3.1→3.3→generasi): suhu domain hasil routing benar-benar
     mengalir ke coop_params generasi utama CoreNode.quick_reply.
  5. RAG dalam pipeline (3.4): RetrievalLayer mendistilasi konteks via model RAG
     200MB saat server hidup, dan FALLBACK aman ke konteks mentah saat mati
     (pipeline tidak boleh putus).

Selain rantai E2E baru di atas, runner Phase 3.5 juga menjalankan ulang KETIGA
suite komponen (Byte-Q roundtrip, Dispatcher, RAG) sebagai regresi penuh Phase 3.

Jalankan (interpreter dengan 'gguf' + 'numpy' agar regresi Byte-Q ikut jalan):
    PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_phase35_integration.py
"""

import os
import tempfile


# Registry uji: temperature/ctx berbeda per domain, semua menunjuk file yang
# sama (mereplikasi kondisi produksi: 1 file GGUF untuk semua domain).
_SHARED_PATH = "/tmp/moko_phase35_shared_model.gguf"
TEST_REGISTRY = {
    "coding": {
        "path": _SHARED_PATH,
        "size_gb": 1.0,
        "domain": ["code", "programming", "software", "debug"],
        "temperature": 0.0,
        "context_window": 8192,
    },
    "math": {
        "path": _SHARED_PATH,
        "size_gb": 1.0,
        "domain": ["math", "physics", "formula"],
        "temperature": 0.2,
        "context_window": 2048,
    },
    "general": {
        "path": _SHARED_PATH,
        "size_gb": 1.0,
        "domain": ["general", "conversation", "personal"],
        "temperature": 0.7,
        "context_window": 4096,
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────
def _install_fake_engine_load():
    """Ganti engine.load_model agar tidak menyentuh server; hitung pemanggilan."""
    from moko_agents import llm_engine

    calls = {"load": []}

    def fake_load_model(model_path):
        calls["load"].append(model_path)
        return True

    llm_engine.engine.load_model = fake_load_model
    return calls


def _fresh_dispatcher():
    """Reset singleton dispatcher agar memakai TEST_REGISTRY."""
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = TEST_REGISTRY

    from moko_agents import model_dispatcher
    model_dispatcher._dispatcher_instance = None
    return model_dispatcher.get_dispatcher(verbose=False)


# ─────────────────────────────────────────────────────────────────────────
# 1. Router → Dispatcher: intent memilih model & mengaktifkan params domain
# ─────────────────────────────────────────────────────────────────────────
def test_router_selects_and_activates_domain_model():
    _install_fake_engine_load()
    disp = _fresh_dispatcher()

    from moko_agents.intent_router import IntentFirstRouter
    router = IntentFirstRouter(verbose=False)

    # (query, model_key harapan, temperature harapan dari TEST_REGISTRY)
    cases = [
        ("tulis fungsi python untuk sorting quicksort", "coding", 0.0),
        ("hitung integral dari x kuadrat", "math", 0.2),
        ("halo apa kabar", "general", 0.7),
    ]

    for query, expected_key, expected_temp in cases:
        manifest = router.classify(query)
        assert manifest.model_key == expected_key, (
            f"router salah rute '{query}': {manifest.model_key} != {expected_key}"
        )

        assert disp.switch_to(manifest.model_key) is True, (
            f"dispatcher gagal switch ke {manifest.model_key}"
        )

        active = disp.get_active_params()
        assert active["model_key"] == expected_key, (
            f"model aktif salah: {active} (harusnya {expected_key})"
        )
        assert active["temperature"] == expected_temp, (
            f"suhu domain {expected_key} salah: {active['temperature']} != {expected_temp}"
        )

    print("  ✅ Router memilih model spesialis yang benar & Dispatcher "
          "mengaktifkan parameter domainnya.")


# ─────────────────────────────────────────────────────────────────────────
# 2. Byte-Q dipreferensikan di registry (dequantize-before-load bridge)
# ─────────────────────────────────────────────────────────────────────────
def test_byteq_preferred_in_registry():
    from moko_config import settings
    from moko_agents import model_dispatcher
    from moko_tools.byteq_loader import resolve_loadable_model, BYTEQ_SUFFIX

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "domain_model.gguf")
        byteq = os.path.join(tmp, "domain_model.byteq.gguf")
        # Buat kedua file (isi tidak penting: registry hanya cek os.path.exists).
        with open(base, "wb") as f:
            f.write(b"GGUF-base")
        with open(byteq, "wb") as f:
            f.write(b"GGUF-byteq")

        settings.DOMAIN_MODEL_REGISTRY = {
            "coding": {
                "path": base,
                "size_gb": 1.0,
                "domain": ["code"],
                "temperature": 0.0,
                "context_window": 8192,
            },
        }
        model_dispatcher._dispatcher_instance = None
        disp = model_dispatcher.get_dispatcher(verbose=False)

        model = disp._models["coding"]
        assert model.path.endswith(BYTEQ_SUFFIX), (
            f"dispatcher tidak memilih saudara Byte-Q: {model.path}"
        )
        # Byte-Q menghemat ~40% VRAM → size_gb dikoreksi ke 0.6×.
        assert abs(model.size_gb - 0.6) < 1e-6, (
            f"anggaran VRAM Byte-Q tak dikoreksi: {model.size_gb} (harusnya 0.6)"
        )

        # resolve_loadable_model harus meneruskan GGUF biasa apa adanya.
        assert resolve_loadable_model(base) == base, (
            "GGUF biasa seharusnya diteruskan tanpa perubahan"
        )

    print("  ✅ Registry memilih model Byte-Q & mengoreksi VRAM; GGUF biasa "
          "diteruskan apa adanya.")


# ─────────────────────────────────────────────────────────────────────────
# 3. Skip-reload router-driven: percakapan multi-intent → 1 load nyata
# ─────────────────────────────────────────────────────────────────────────
def test_router_driven_conversation_reuses_shared_file():
    calls = _install_fake_engine_load()
    disp = _fresh_dispatcher()

    from moko_agents.intent_router import IntentFirstRouter
    router = IntentFirstRouter(verbose=False)

    conversation = [
        "tulis fungsi python untuk sorting quicksort",   # coding
        "hitung integral dari x kuadrat",                # math
        "halo apa kabar",                                # general
        "debug error di function main python",           # coding lagi
    ]
    for query in conversation:
        manifest = router.classify(query)
        disp.switch_to(manifest.model_key)

    assert len(calls["load"]) == 1, (
        f"file GGUF sama seharusnya dimuat 1x sepanjang percakapan, "
        f"nyatanya {len(calls['load'])}x: {calls['load']}"
    )
    assert disp._loaded_path == _SHARED_PATH
    print("  ✅ Percakapan multi-intent memakai ulang file model yang sama "
          "(hanya 1 load nyata).")


# ─────────────────────────────────────────────────────────────────────────
# 4. Router → CoreNode: suhu domain mengalir ke generasi utama
# ─────────────────────────────────────────────────────────────────────────
def test_router_domain_temperature_reaches_corenode_generation():
    _install_fake_engine_load()
    _fresh_dispatcher()

    from moko_agents.intent_router import IntentFirstRouter
    import moko_agents.core_node as core_node
    from moko_agents import auto_continue_engine as ace

    router = IntentFirstRouter(verbose=False)
    manifest = router.classify("tulis fungsi python untuk sorting quicksort")
    assert manifest.model_key == "coding"

    # Router tingkat-atas menghasilkan route_meta; paksa jalur DEEP (D5) agar
    # generasi utama (yang menerapkan suhu domain) benar-benar dieksekusi.
    route_meta = {
        "domain": manifest.domain,
        "model_key": manifest.model_key,
        "path": manifest.path,
        "complexity": "COMPLEX",
        "depth": "D5",
    }

    captured = {}

    def fake_stream(prompt, system_prompt="", coop_params=None, session_messages=None,
                    on_token=None, on_chunk=None, disable_timeout=False):
        captured["coop_params"] = coop_params
        return "jawaban uji"

    orig_stream = ace.auto_continue_engine.generate_complete_stream
    try:
        ace.auto_continue_engine.generate_complete_stream = fake_stream

        node = object.__new__(core_node.CoreNode)
        node.disk_mgr = None
        node.quick_reply("tulis fungsi python untuk sorting quicksort", route_meta=route_meta)
    finally:
        ace.auto_continue_engine.generate_complete_stream = orig_stream

    assert "coop_params" in captured, "generate_complete_stream tidak terpanggil"
    assert captured["coop_params"].get("temperature") == 0.0, (
        f"suhu domain coding (0.0) hasil routing tidak diterapkan: {captured['coop_params']}"
    )
    print("  ✅ Suhu domain hasil routing mengalir sampai ke coop_params "
          "generasi utama CoreNode.")


# ─────────────────────────────────────────────────────────────────────────
# Helper RetrievalLayer dengan KnowledgeLayer palsu (tanpa DiskManager)
# ─────────────────────────────────────────────────────────────────────────
def _make_retrieval_layer():
    from moko_agents.layers.retrieval_layer import RetrievalLayer
    from moko_config import settings

    class _FakeKnowledge:
        def search_facts(self, emb, top_k=5, domain="general"):
            if domain == "general":
                return [{"domain": "general", "text": "Matahari pusat tata surya.", "score": 0.9}]
            return []

    layer = RetrievalLayer.__new__(RetrievalLayer)
    layer.knowledge_layer = _FakeKnowledge()
    layer.rag_port = getattr(settings, "MOKO_RAG_PORT", 11437)
    return layer


# ─────────────────────────────────────────────────────────────────────────
# 5a. RAG dalam pipeline: distilasi saat server RAG hidup
# ─────────────────────────────────────────────────────────────────────────
def test_pipeline_rag_distills_when_up():
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
            return "RINGKAS: Matahari adalah pusat tata surya."

        rl.engine.generate_rag = fake_gen
        out = layer.retrieve_context("apa itu tata surya?")
    finally:
        rl.engine.get_embedding = orig_embed
        rl.engine.rag_available = orig_avail
        rl.engine.generate_rag = orig_gen

    assert out == "RINGKAS: Matahari adalah pusat tata surya.", f"distilasi tak dipakai: {out!r}"
    assert "Matahari pusat tata surya" in captured["prompt"], (
        "fakta mentah harus diteruskan ke model RAG untuk didistilasi"
    )
    print("  ✅ Pipeline mendistilasi konteks via model RAG 200MB saat server hidup.")


# ─────────────────────────────────────────────────────────────────────────
# 5b. RAG dalam pipeline: fallback aman saat server RAG mati
# ─────────────────────────────────────────────────────────────────────────
def test_pipeline_rag_falls_back_when_down():
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

    assert "Matahari pusat tata surya" in out, f"konteks mentah hilang: {out!r}"
    assert gen_called["n"] == 0, "generate_rag TIDAK boleh dipanggil saat server RAG mati"
    print("  ✅ Pipeline fallback aman ke konteks mentah saat server RAG mati "
          "(pipeline tidak putus).")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
def _run_e2e_chain():
    print("\n🧪 Phase 3.5 — Integrasi End-to-End Arsitektur (Router→Byte-Q→Dispatcher→RAG)\n")
    test_router_selects_and_activates_domain_model()
    test_byteq_preferred_in_registry()
    test_router_driven_conversation_reuses_shared_file()
    test_router_domain_temperature_reaches_corenode_generation()
    test_pipeline_rag_distills_when_up()
    test_pipeline_rag_falls_back_when_down()
    print("\n✅ RANTAI E2E LULUS: seluruh komponen Phase 3 bekerja bersama.\n")


def _run_component_regressions():
    """Jalankan ulang ketiga suite komponen Phase 3 sebagai regresi penuh."""
    print("━" * 70)
    print("  Regresi komponen Phase 3 (Byte-Q · Dispatcher · RAG)")
    print("━" * 70)

    from moko_tools import test_byteq_roundtrip
    test_byteq_roundtrip.main()

    from moko_agents import test_dispatcher_integration
    test_dispatcher_integration.main()

    from moko_agents import test_rag_integration
    test_rag_integration.main()


def main():
    _run_e2e_chain()
    _run_component_regressions()
    print("━" * 70)
    print("  ✅ PHASE 3.5 SELESAI: rantai E2E + regresi komponen semuanya LULUS.")
    print("━" * 70 + "\n")


if __name__ == "__main__":
    main()
