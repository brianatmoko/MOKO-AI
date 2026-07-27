"""
Test Integrasi — Multi-Model Dispatcher ↔ CoreNode (Phase 3.3)
=============================================================
Memverifikasi bahwa:

1. Dispatcher mengekspos parameter generasi khas domain (temperature,
   context_window, system_prompt) via get_active_params()/get_params_for().
2. Switch domain benar-benar mengubah parameter aktif (jembatan spesialisasi).
3. Reload TIDAK terjadi berulang saat beberapa domain menunjuk ke file GGUF
   yang sama (guard _loaded_path).
4. CoreNode.quick_reply menerapkan suhu domain ke coop_params generasi utama.

Test ini TIDAK memerlukan llama-server hidup: engine.load_model dan
generate_complete_stream di-monkeypatch.

Jalankan:
    PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_dispatcher_integration.py
"""

import os
import sys


# Registry uji: temperature berbeda per domain, semua menunjuk file yang sama
# (mereplikasi kondisi produksi saat ini: 1 file untuk semua domain).
_SAME_PATH = "/tmp/moko_test_shared_model.gguf"
TEST_REGISTRY = {
    "coding": {
        "path": _SAME_PATH,
        "size_gb": 1.0,
        "domain": ["code", "programming"],
        "temperature": 0.0,
        "context_window": 8192,
    },
    "math": {
        "path": _SAME_PATH,
        "size_gb": 1.0,
        "domain": ["math", "physics"],
        "temperature": 0.0,
        "context_window": 2048,
    },
    "general": {
        "path": _SAME_PATH,
        "size_gb": 1.0,
        "domain": ["general", "conversation"],
        "temperature": 0.7,
        "context_window": 4096,
    },
}


def _install_fake_engine():
    """Ganti engine.load_model agar tidak menyentuh server; hitung pemanggilan."""
    from moko_agents import llm_engine

    calls = {"load": []}

    def fake_load_model(model_path):
        calls["load"].append(model_path)
        return True

    llm_engine.engine.load_model = fake_load_model
    return calls


def test_domain_params_exposed():
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = TEST_REGISTRY

    from moko_agents.model_dispatcher import MultiModelDispatcher
    disp = MultiModelDispatcher(verbose=False)

    p_code = disp.get_params_for("coding")
    p_gen = disp.get_params_for("general")

    assert p_code["temperature"] == 0.0, f"coding temp salah: {p_code}"
    assert p_code["context_window"] == 8192, f"coding ctx salah: {p_code}"
    assert p_gen["temperature"] == 0.7, f"general temp salah: {p_gen}"
    assert disp.get_params_for("tidak_ada") == {}, "model tak dikenal harus {}"
    print("  ✅ Dispatcher mengekspos parameter khas domain dengan benar.")


def test_switch_changes_active_params():
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = TEST_REGISTRY
    _install_fake_engine()

    from moko_agents.model_dispatcher import MultiModelDispatcher
    disp = MultiModelDispatcher(verbose=False)

    assert disp.get_active_params() == {}, "belum switch → params aktif harus {}"

    disp.switch_to("coding")
    assert disp.get_active_params()["temperature"] == 0.0
    assert disp.get_active_params()["model_key"] == "coding"

    disp.switch_to("general")
    assert disp.get_active_params()["temperature"] == 0.7
    assert disp.get_active_params()["model_key"] == "general"
    print("  ✅ Switch domain benar-benar mengubah parameter generasi aktif.")


def test_same_file_skips_reload():
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = TEST_REGISTRY
    calls = _install_fake_engine()

    from moko_agents.model_dispatcher import MultiModelDispatcher
    disp = MultiModelDispatcher(verbose=False)

    # Tiga domain berbeda tapi file sama → hanya SATU kali load nyata.
    disp.switch_to("coding")
    disp.switch_to("math")
    disp.switch_to("general")
    disp.switch_to("coding")

    assert len(calls["load"]) == 1, (
        f"File sama seharusnya dimuat 1x, nyatanya {len(calls['load'])}x: {calls['load']}"
    )
    assert disp._loaded_path == _SAME_PATH
    print("  ✅ File model yang sama tidak dimuat ulang saat ganti domain.")


def test_core_node_applies_domain_temperature():
    """CoreNode.quick_reply harus meneruskan suhu domain ke coop_params generasi."""
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = TEST_REGISTRY
    _install_fake_engine()

    # Reset singleton dispatcher agar memakai registry uji.
    from moko_agents import model_dispatcher
    model_dispatcher._dispatcher_instance = None

    import moko_agents.core_node as core_node
    from moko_agents import auto_continue_engine as ace

    captured = {}

    def fake_stream(prompt, system_prompt="", coop_params=None, session_messages=None,
                    on_token=None, on_chunk=None, disable_timeout=False):
        captured["coop_params"] = coop_params
        return "jawaban uji"

    ace.auto_continue_engine.generate_complete_stream = fake_stream

    # Buat instance tanpa __init__ berat (hanya perlu method quick_reply).
    node = object.__new__(core_node.CoreNode)
    node.disk_mgr = None

    route_meta = {
        "domain": "coding",
        "model_key": "coding",
        "path": _SAME_PATH,
        "complexity": "COMPLEX",
        "depth": "D5",
    }
    node.quick_reply("tulis fungsi python untuk sorting", route_meta=route_meta)

    assert "coop_params" in captured, "generate_complete_stream tidak terpanggil"
    assert captured["coop_params"].get("temperature") == 0.0, (
        f"suhu domain coding (0.0) tidak diterapkan: {captured['coop_params']}"
    )
    print("  ✅ CoreNode menerapkan suhu domain ke pipeline generasi utama.")


def main():
    print("\n🧪 Integrasi Dispatcher ↔ CoreNode (Phase 3.3)\n")
    test_domain_params_exposed()
    test_switch_changes_active_params()
    test_same_file_skips_reload()
    test_core_node_applies_domain_temperature()
    print("\n✅ SEMUA ASSERT LULUS: Multi-Model Dispatcher terintegrasi ke CoreNode.\n")


if __name__ == "__main__":
    main()
