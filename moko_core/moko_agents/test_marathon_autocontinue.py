"""
Test Marathon / Auto-Continue Fix — Phase 3.7
=============================================
Reproduksi bug user: saat MOKO membuat kalkulator, jawaban berhenti di tengah
blok kode (mis. berakhir di `luas = p * l`) dan TIDAK dilanjutkan otomatis
("sistem maraton tidak aktif").

Root cause (sebelum fix):
  1. llm_engine.generate_stream MEMBUANG finish_reason dari SSE, sehingga
     auto_continue_engine tidak pernah tahu token habis ("length").
  2. Deteksi truncation hanya heuristik karakter (len >= 85% x num_predict*3
     + akhiran tanda baca) — output kode terpotong sering di bawah ambang.
  3. Fase 2 memanggil generate_complete (generate ULANG dari nol) lalu menebak
     overlap — sering gagal merge sehingga kode tetap tersangkut.

Fix yang diuji (semua server-free / engine di-monkeypatch):
  A. finish_reason=="length" memicu auto-continue meski heuristik lama gagal.
  B. Code-fence ``` ganjil (blok kode belum ditutup) juga memicu.
  C. Continuation dilakukan native via continue_generation dari partial
     (bukan regenerate), hasil tersambung mulus & di-stream via on_token.
  D. Jawaban tuntas (finish_reason=="stop", fence genap) TIDAK memicu.
  E. generate_stream benar-benar menangkap finish_reason dari payload SSE.

Jalankan:
    PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_marathon_autocontinue.py
"""

from moko_agents import llm_engine
from moko_agents.auto_continue_engine import auto_continue_engine

# Jawaban terpotong ala kasus user: blok kode belum ditutup, berhenti di tengah.
PARTIAL = (
    "Siap, Brian! Berikut kalkulator dengan rumus-rumusnya:\n\n"
    "```python\n"
    "class GameCalculator:\n"
    "    def calculate_rectangle(self, p, l):\n"
    "        try:\n"
    "            luas = p * l"
)
CONTINUATION = (
    "\n            return luas\n"
    "        except TypeError:\n"
    "            return None\n"
    "```\n\n"
    "Selesai — rumus: Luas = Panjang × Lebar."
)


def _patch_engine(stream_text, stream_finish, cont_chunks):
    """Pasang fake generate_stream + continue_generation; kembalikan call-log."""
    calls = {"continue": 0, "partials": []}

    def fake_generate_stream(prompt, system_prompt="", coop_params=None,
                             session_messages=None, on_token=None, stop_check=None):
        llm_engine.engine.last_stream_finish_reason = stream_finish
        if on_token:
            on_token(stream_text)
        return stream_text

    def fake_continue_generation(messages, partial_response, coop_params=None):
        calls["continue"] += 1
        calls["partials"].append(partial_response)
        chunk, reason = cont_chunks[min(calls["continue"] - 1, len(cont_chunks) - 1)]
        return chunk, reason

    llm_engine.engine.generate_stream = fake_generate_stream
    llm_engine.engine.continue_generation = fake_continue_generation
    return calls


# ─────────────────────────────────────────────────────────────────────────
# A+B+C. Kode terpotong (finish_reason="length" + fence ganjil) → dilanjutkan
# ─────────────────────────────────────────────────────────────────────────
def test_truncated_code_is_continued():
    calls = _patch_engine(PARTIAL, "length", [(CONTINUATION, "stop")])

    tokens = []
    result = auto_continue_engine.generate_complete_stream(
        prompt="buatkan saya kalkulator, dengan tambahan ada rumus rumusnya",
        system_prompt="Kamu MOKO.",
        coop_params={"num_predict": 2048, "enable_thinking": False},
        on_token=tokens.append,
        disable_timeout=True,
    )

    # Heuristik lama PASTI gagal di kasus ini (bukti bug):
    assert len(PARTIAL) < int(2048 * 3 * 0.85), "kasus uji harus di bawah ambang heuristik lama"

    assert calls["continue"] == 1, "auto-continue harus terpicu tepat 1x"
    assert calls["partials"][0] == PARTIAL, "harus melanjutkan dari partial asli (bukan regenerate)"
    assert result == PARTIAL + CONTINUATION, "hasil harus tersambung mulus"
    assert result.count("```") % 2 == 0, "blok kode harus tertutup"
    assert "".join(tokens) == result, "sambungan harus ikut di-stream via on_token"
    print("  ✅ Kode terpotong (finish_reason='length' + fence ganjil) dilanjutkan "
          "native dari partial dan tersambung mulus.")


# ─────────────────────────────────────────────────────────────────────────
# B. Fence ganjil saja (tanpa finish_reason) tetap memicu
# ─────────────────────────────────────────────────────────────────────────
def test_unclosed_fence_alone_triggers():
    calls = _patch_engine(PARTIAL, None, [(CONTINUATION, "stop")])

    result = auto_continue_engine.generate_complete_stream(
        prompt="buatkan kalkulator",
        coop_params={"num_predict": 2048, "enable_thinking": False},
        disable_timeout=True,
    )

    assert calls["continue"] == 1, "fence ``` ganjil harus cukup untuk memicu continue"
    assert result == PARTIAL + CONTINUATION
    print("  ✅ Blok kode ``` yang belum ditutup memicu auto-continue meski "
          "finish_reason tidak tersedia.")


# ─────────────────────────────────────────────────────────────────────────
# D. Jawaban tuntas → TIDAK memicu (tak ada false positive)
# ─────────────────────────────────────────────────────────────────────────
def test_complete_answer_not_continued():
    complete = "Luas persegi panjang = panjang × lebar. Contoh: 5 × 3 = 15."
    calls = _patch_engine(complete, "stop", [("HARUS TIDAK TERPANGGIL", "stop")])

    result = auto_continue_engine.generate_complete_stream(
        prompt="apa rumus luas persegi panjang?",
        coop_params={"num_predict": 512, "enable_thinking": False},
        disable_timeout=True,
    )

    assert calls["continue"] == 0, "jawaban tuntas tidak boleh memicu continue"
    assert result == complete
    print("  ✅ Jawaban tuntas (finish_reason='stop', fence genap) tidak memicu "
          "auto-continue (tanpa false positive).")


# ─────────────────────────────────────────────────────────────────────────
# E. generate_stream menangkap finish_reason dari payload SSE
# ─────────────────────────────────────────────────────────────────────────
def test_generate_stream_captures_finish_reason():
    import json

    sse_lines = [
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "halo "}, "finish_reason": None}]}).encode(),
        b'data: ' + json.dumps({"choices": [{"delta": {"content": "dunia"}, "finish_reason": "length"}]}).encode(),
        b'data: [DONE]',
    ]

    class FakeResp:
        status_code = 200
        def iter_lines(self):
            return iter(sse_lines)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    orig_post = llm_engine.requests.post
    orig_wait = llm_engine.engine._wait_until_ready
    try:
        llm_engine.requests.post = lambda *a, **k: FakeResp()
        llm_engine.engine._wait_until_ready = lambda: None

        out = llm_engine.engine.generate_stream(
            prompt="tes", coop_params={"num_predict": 8, "enable_thinking": False}
        )
        assert out == "halo dunia"
        assert llm_engine.engine.last_stream_finish_reason == "length", (
            f"finish_reason SSE harus tertangkap, nyatanya "
            f"{llm_engine.engine.last_stream_finish_reason!r}"
        )
    finally:
        llm_engine.requests.post = orig_post
        llm_engine.engine._wait_until_ready = orig_wait

    print("  ✅ generate_stream menangkap finish_reason='length' langsung dari "
          "payload SSE server.")


def main():
    print("\n🧪 Phase 3.7 — Marathon / Auto-Continue Fix (server-free)\n")
    test_generate_stream_captures_finish_reason()
    test_truncated_code_is_continued()
    test_unclosed_fence_alone_triggers()
    test_complete_answer_not_continued()
    print("\n✅ SEMUA LULUS: sistem maraton kembali aktif — kode terpotong "
          "dilanjutkan otomatis.\n")


if __name__ == "__main__":
    main()
