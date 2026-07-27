"""
Test Integrasi Menyeluruh — Phase 3.6 (Performance Optimization End-to-End)
==========================================================================
Phase 3.5 membuktikan seluruh komponen Phase 3 SALING TERHUBUNG (fungsional).
Phase 3.6 adalah gerbang PERFORMA: memverifikasi bahwa jalur-jalur optimasi
yang membuat arsitektur ini muat & cepat di RTX 2050 (4 GB) benar-benar
bekerja secara terukur — bukan sekadar "jalan".

Rantai optimasi yang diverifikasi (semua server-free / di-monkeypatch):

  1. Router fast-path (3.1): query berkata-kunci jelas diselesaikan pada
     Tier 0/1 (heuristik murni, TANPA embedding/LLM) → latensi kecil & tercatat.
  2. Enforcement anggaran VRAM (3.3): VRAMManager menegakkan batas usable
     (TOTAL−OS−KV = 3.0 GB) secara persis — alokasi yang melewati batas DITOLAK.
  3. Penghematan Byte-Q (3.2↔3.3): koreksi ~40% membuat model yang semula
     TIDAK muat menjadi muat (lebih banyak domain hidup dalam anggaran sama).
  4. Eviksi LRU/standby (3.3): saat VRAM penuh, _make_room melepas model
     standby/paling lama sehingga model baru muat & total tetap ≤ anggaran.
  5. Efisiensi skip-reload (3.3): percakapan panjang N-giliran yang berbagi
     satu file GGUF hanya memicu SATU load nyata (rasio reuse tinggi).
  6. Preload prediktif (3.3): preload_likely_next memanaskan model kandidat
     ke STANDBY lebih dulu (menyembunyikan latensi switch berikutnya).
  7. Telemetri switch (3.3): get_status melaporkan switch_count, ref_count,
     dan avg_switch_time_ms secara akurat untuk observability performa.

Runner Phase 3.6 juga menjalankan ulang seluruh gerbang Phase 3.5 (rantai E2E
+ regresi komponen Byte-Q/Dispatcher/RAG) sebagai regresi penuh Phase 3.

Jalankan (interpreter dengan 'gguf' + 'numpy' agar regresi Byte-Q ikut jalan):
    PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_phase36_integration.py
"""

import os
import tempfile


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


def _fresh_dispatcher(registry):
    """Reset singleton dispatcher agar memakai registry uji tertentu."""
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = registry

    from moko_agents import model_dispatcher
    model_dispatcher._dispatcher_instance = None
    return model_dispatcher.get_dispatcher(verbose=False)


def _registry_distinct(tmp):
    """Registry dengan file GGUF BERBEDA per domain (agar tiap load nyata)."""
    def mk(name, size):
        p = os.path.join(tmp, f"{name}.gguf")
        with open(p, "wb") as f:
            f.write(b"GGUF-" + name.encode())
        return {"path": p, "size_gb": size, "domain": [name]}

    return {
        "coding": mk("coding", 1.5),
        "math": mk("math", 1.5),
        "general": mk("general", 1.0),
    }


# ─────────────────────────────────────────────────────────────────────────
# 1. Router fast-path: Tier 0/1 heuristik, tanpa LLM/embedding
# ─────────────────────────────────────────────────────────────────────────
def test_router_fast_path_is_heuristic_and_fast():
    from moko_agents import llm_engine
    from moko_agents.intent_router import IntentFirstRouter

    # Jika Tier 2 (semantic) tersentuh, ia memanggil get_embedding → kita paksa
    # meledak agar test GAGAL jelas bila fast-path bocor ke jalur mahal.
    orig_embed = llm_engine.engine.get_embedding

    def _boom(*a, **k):
        raise AssertionError("Tier 2 semantic terpanggil — fast-path bocor ke LLM/embedding")

    router = IntentFirstRouter(verbose=False)
    try:
        llm_engine.engine.get_embedding = _boom

        cases = [
            "tulis fungsi python untuk sorting quicksort",  # coding (keyword kuat)
            "hitung integral dari x kuadrat",               # math
        ]
        for query in cases:
            res = router.classify(query)
            assert res.tier <= 1, (
                f"'{query}' seharusnya selesai di Tier 0/1 (heuristik), "
                f"nyatanya Tier {res.tier}"
            )
            assert res.latency_ms >= 0.0, "latency_ms harus tercatat"
            # Heuristik murni harus sangat cepat (ambang longgar untuk CI).
            assert res.latency_ms < 100.0, (
                f"fast-path terlalu lambat: {res.latency_ms:.2f}ms"
            )
    finally:
        llm_engine.engine.get_embedding = orig_embed

    print("  ✅ Router menyelesaikan query berkata-kunci di Tier 0/1 (heuristik, "
          "tanpa embedding/LLM) dengan latensi kecil.")


# ─────────────────────────────────────────────────────────────────────────
# 2. Enforcement anggaran VRAM: batas usable ditegakkan persis
# ─────────────────────────────────────────────────────────────────────────
def test_vram_budget_enforced_exactly():
    from moko_agents.model_dispatcher import VRAMManager

    vram = VRAMManager(verbose=False)
    # usable = TOTAL(4.0) − OS(0.5) − KV(0.5) = 3.0 GB
    assert abs(vram.usable_gb - 3.0) < 1e-9, f"usable_gb salah: {vram.usable_gb}"

    assert vram.allocate("a", 2.0) is True
    assert vram.can_fit(1.0) is True, "2.0+1.0=3.0 seharusnya pas muat"
    assert vram.can_fit(1.5) is False, "2.0+1.5=3.5 seharusnya TIDAK muat"
    assert vram.allocate("b", 1.5) is False, "alokasi melewati anggaran harus ditolak"
    assert vram.allocate("b", 1.0) is True, "2.0+1.0=3.0 seharusnya diterima"
    assert vram.can_fit(0.1) is False, "anggaran sudah penuh (3.0/3.0)"

    vram.deallocate("a")
    assert vram.can_fit(2.0) is True, "setelah dealokasi 'a', 1.0+2.0=3.0 muat lagi"

    print("  ✅ VRAMManager menegakkan anggaran usable (3.0 GB) secara persis: "
          "alokasi melampaui batas ditolak, dealokasi mengembalikan ruang.")


# ─────────────────────────────────────────────────────────────────────────
# 3. Penghematan Byte-Q (~40%): model yang tak muat jadi muat
# ─────────────────────────────────────────────────────────────────────────
def test_byteq_saving_makes_more_models_fit():
    from moko_config import settings
    from moko_agents import model_dispatcher
    from moko_agents.model_dispatcher import VRAMManager

    with tempfile.TemporaryDirectory() as tmp:
        registry = {}
        for name in ("coding", "math", "general"):
            base = os.path.join(tmp, f"{name}.gguf")
            byteq = os.path.join(tmp, f"{name}.byteq.gguf")
            with open(base, "wb") as f:
                f.write(b"base")
            with open(byteq, "wb") as f:
                f.write(b"byteq")
            registry[name] = {"path": base, "size_gb": 1.5, "domain": [name]}

        settings.DOMAIN_MODEL_REGISTRY = registry
        model_dispatcher._dispatcher_instance = None
        disp = model_dispatcher.get_dispatcher(verbose=False)

        # Setiap model dikoreksi ke 0.6× (1.5 → 0.9).
        corrected = [disp._models[k].size_gb for k in ("coding", "math", "general")]
        for s in corrected:
            assert abs(s - 0.9) < 1e-6, f"koreksi Byte-Q salah: {s} (harusnya 0.9)"

        full_sizes = [1.5, 1.5, 1.5]
        saving = (sum(full_sizes) - sum(corrected)) / sum(full_sizes)
        assert saving >= 0.4 - 1e-9, f"penghematan Byte-Q < 40%: {saving:.2%}"

        # Demonstrasi konkret: 3 model penuh TIDAK muat, versi Byte-Q MUAT.
        vram_full = VRAMManager(verbose=False)
        assert vram_full.allocate("coding", 1.5) is True
        assert vram_full.allocate("math", 1.5) is True
        assert vram_full.allocate("general", 1.5) is False, (
            "3 model penuh (4.5 GB) seharusnya tak muat dalam 3.0 GB"
        )

        vram_bq = VRAMManager(verbose=False)
        assert vram_bq.allocate("coding", 0.9) is True
        assert vram_bq.allocate("math", 0.9) is True
        assert vram_bq.allocate("general", 0.9) is True, (
            "3 model Byte-Q (2.7 GB) seharusnya muat dalam 3.0 GB"
        )

    print(f"  ✅ Byte-Q menghemat {saving:.0%} VRAM → 3 domain muat bersama "
          "(versi penuh tidak).")


# ─────────────────────────────────────────────────────────────────────────
# 4. Eviksi LRU/standby: VRAM penuh → lepas yang lama, total tetap ≤ anggaran
# ─────────────────────────────────────────────────────────────────────────
def test_vram_eviction_keeps_within_budget():
    from moko_agents.model_dispatcher import ModelState

    _install_fake_engine_load()
    with tempfile.TemporaryDirectory() as tmp:
        disp = _fresh_dispatcher(_registry_distinct(tmp))

        # coding(1.5) + math(1.5) = 3.0 → penuh. Keduanya file berbeda.
        assert disp.switch_to("coding") is True
        assert disp.switch_to("math") is True
        allocated = sum(disp.vram._allocated.values())
        assert abs(allocated - 3.0) < 1e-6, f"anggaran belum penuh: {allocated}"

        # general(1.0) tak muat → _make_room harus meng-eviksi standby (coding).
        assert disp.switch_to("general") is True

        assert disp._models["coding"].state == ModelState.UNLOADED, (
            "coding (standby/LRU) seharusnya di-eviksi untuk memberi ruang"
        )
        allocated_after = sum(disp.vram._allocated.values())
        assert allocated_after <= disp.vram.usable_gb + 1e-9, (
            f"total VRAM ({allocated_after}) melewati anggaran ({disp.vram.usable_gb})"
        )
        assert disp._active_model == "general"

    print("  ✅ Saat VRAM penuh, dispatcher meng-eviksi model standby/LRU "
          "sehingga model baru muat & total tetap dalam anggaran.")


# ─────────────────────────────────────────────────────────────────────────
# 5. Efisiensi skip-reload: percakapan panjang berbagi 1 file → 1 load
# ─────────────────────────────────────────────────────────────────────────
def test_skip_reload_reuse_ratio():
    shared = "/tmp/moko_phase36_shared.gguf"
    registry = {
        "coding": {"path": shared, "size_gb": 1.0, "domain": ["code"]},
        "math": {"path": shared, "size_gb": 1.0, "domain": ["math"]},
        "general": {"path": shared, "size_gb": 1.0, "domain": ["general"]},
    }
    calls = _install_fake_engine_load()
    disp = _fresh_dispatcher(registry)

    sequence = ["coding", "math", "general", "coding", "general", "math",
                "coding", "general"]  # 8 giliran, semua file sama
    for key in sequence:
        assert disp.switch_to(key) is True

    assert len(calls["load"]) == 1, (
        f"file sama seharusnya dimuat 1x, nyatanya {len(calls['load'])}x"
    )
    reuse_ratio = 1.0 - (len(calls["load"]) / len(sequence))
    assert reuse_ratio >= 0.8, f"rasio reuse terlalu rendah: {reuse_ratio:.0%}"
    assert disp.get_status()["switch_count"] == len(sequence)

    print(f"  ✅ Percakapan {len(sequence)} giliran berbagi 1 file → hanya 1 load "
          f"nyata (reuse {reuse_ratio:.0%}).")


# ─────────────────────────────────────────────────────────────────────────
# 6. Preload prediktif: memanaskan kandidat berikutnya ke STANDBY
# ─────────────────────────────────────────────────────────────────────────
def test_predictive_preload_warms_next_model():
    from moko_agents.model_dispatcher import ModelState

    calls = _install_fake_engine_load()
    with tempfile.TemporaryDirectory() as tmp:
        disp = _fresh_dispatcher(_registry_distinct(tmp))

        # coding → prediksi berikutnya = math (lihat preload_likely_next).
        disp.preload_likely_next("coding")

        math = disp._models["math"]
        assert math.state in (ModelState.STANDBY, ModelState.ACTIVE), (
            f"math seharusnya sudah dipanaskan ke STANDBY, nyatanya {math.state}"
        )
        assert math.ref_count == 0, "preload tidak boleh menandai model sebagai terpakai"
        assert disp._active_model is None, "preload tidak boleh mengaktifkan model"
        assert math.path in calls["load"], "engine.load_model harus dipanggil untuk math"

    print("  ✅ Preload prediktif memanaskan model kandidat berikutnya ke STANDBY "
          "(menyembunyikan latensi switch).")


# ─────────────────────────────────────────────────────────────────────────
# 7. Telemetri switch: switch_count / ref_count / avg_switch_time akurat
# ─────────────────────────────────────────────────────────────────────────
def test_switch_telemetry_is_accurate():
    _install_fake_engine_load()
    with tempfile.TemporaryDirectory() as tmp:
        disp = _fresh_dispatcher(_registry_distinct(tmp))

        for key in ["coding", "math", "coding", "general", "coding"]:
            disp.switch_to(key)

        status = disp.get_status()
        assert status["switch_count"] == 5, f"switch_count salah: {status['switch_count']}"
        assert status["models"]["coding"]["ref_count"] == 3, (
            f"ref_count coding salah: {status['models']['coding']['ref_count']}"
        )
        assert status["avg_switch_time_ms"] >= 0.0
        assert status["active_model"] == "coding"

    print("  ✅ Telemetri dispatcher (switch_count, ref_count, avg_switch_time) "
          "akurat untuk observability performa.")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
def _run_perf_chain():
    print("\n🧪 Phase 3.6 — Optimasi Performa End-to-End "
          "(Router · VRAM · Byte-Q · Eviksi · Skip-reload · Preload · Telemetri)\n")
    test_router_fast_path_is_heuristic_and_fast()
    test_vram_budget_enforced_exactly()
    test_byteq_saving_makes_more_models_fit()
    test_vram_eviction_keeps_within_budget()
    test_skip_reload_reuse_ratio()
    test_predictive_preload_warms_next_model()
    test_switch_telemetry_is_accurate()
    print("\n✅ RANTAI PERFORMA LULUS: jalur optimasi Phase 3 bekerja terukur.\n")


def _run_phase35_regression():
    """Jalankan ulang seluruh gerbang Phase 3.5 (E2E + regresi komponen)."""
    print("━" * 70)
    print("  Regresi penuh Phase 3.5 (rantai E2E + Byte-Q · Dispatcher · RAG)")
    print("━" * 70)
    from moko_agents import test_phase35_integration
    test_phase35_integration.main()


def main():
    _run_perf_chain()
    _run_phase35_regression()
    print("━" * 70)
    print("  ✅ PHASE 3.6 SELESAI: rantai performa + regresi Phase 3.5 semuanya LULUS.")
    print("━" * 70 + "\n")


if __name__ == "__main__":
    main()
