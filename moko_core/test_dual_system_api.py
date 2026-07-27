"""
Integration Test — MOKO Multi-Provider & API Failover Orchestration
===================================================================
Menguji ketahanan client API terhadap error 429 (rate limit), pemindaian status
koneksi pekerja, rotasi key otomatis, dan alur Mandor-Pekerja terintegrasi.

Jalankan langsung:  python test_dual_system_api.py   (dari folder moko_core)
"""
import sys
import os
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Pastikan modul moko_core dan moko_agents dapat diimpor
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moko_agents.dual_system.api_client import APIConfig, MokoAPIClient
from moko_agents.dual_system.worker_pool import WorkerPool
from moko_agents.dual_system.interaction_logger import InteractionLogger
from moko_agents.dual_system.orchestrator import DualSystemOrchestrator, VERDICT_COMMIT
from moko_agents.dual_system.runtime_guard import DualRuntimeGuard


class TestMokoAPIFailover(unittest.TestCase):

    def test_key_rotation_on_429(self):
        """Uji rotasi API key otomatis jika menerima HTTP 429."""
        config = APIConfig(
            name="test-gemini",
            provider="gemini",
            model_name="gemini-2.5-flash",
            api_keys=["FAIL_KEY_1", "SUCCESS_KEY_2"]
        )
        client = MokoAPIClient(config)

        # Mocking requests.post
        def mock_post(url, json, headers, timeout):
            # Cek key yang dikirim via query param key=...
            if "FAIL_KEY_1" in url:
                mock_res = MagicMock()
                mock_res.status_code = 429
                mock_res.text = "Rate limit exceeded"
                return mock_res
            elif "SUCCESS_KEY_2" in url:
                mock_res = MagicMock()
                mock_res.status_code = 200
                mock_res.json.return_value = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": "Pekerja Sukses!"}]
                        }
                    }]
                }
                return mock_res
            else:
                raise ValueError("Key tidak dikenal")

        with patch("requests.post", side_effect=mock_post):
            res = client.generate_text("halo", max_tokens=10)
            self.assertEqual(res, "Pekerja Sukses!")
            # Pastikan index key telah berotasi
            self.assertEqual(client.current_key_idx, 1)

    def test_worker_pool_scan(self):
        """Uji deteksi pekerja aktif saat scan koneksi."""
        pool = WorkerPool()
        pool.configs = [
            APIConfig(name="gemini-ok", provider="gemini", model_name="gemini-2.5-flash", api_keys=["KEY_1"]),
            APIConfig(name="openai-fail", provider="openai", model_name="gpt-4o-mini", api_keys=["KEY_2"])
        ]
        
        # Inisialisasi client manual
        pool.clients = {cfg.name: MokoAPIClient(cfg) for cfg in pool.configs}

        def mock_generate_text(prompt, system_prompt, max_tokens, timeout):
            if "gemini-ok" in prompt or "ping" in prompt:
                # Mock connection check prompt untuk gemini-ok
                client_name = getattr(self, "_current_client_name", "gemini-ok")
                if client_name == "gemini-ok":
                    return "pong"
            raise RuntimeError("429 Rate Limit")

        # Mock method generate_text per client
        pool.clients["gemini-ok"].generate_text = MagicMock(return_value="pong")
        pool.clients["openai-fail"].generate_text = MagicMock(side_effect=RuntimeError("Connection Timeout"))

        scan_results = pool.scan_workers()
        self.assertTrue(scan_results["gemini-ok"])
        self.assertFalse(scan_results["openai-fail"])
        self.assertIn("gemini-ok", pool.active_clients)
        self.assertNotIn("openai-fail", pool.active_clients)

    def test_interaction_logger(self):
        """Uji logger mencatat log sukses ke berkas JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "distill.jsonl"
            logger_inst = InteractionLogger(log_file_path=log_file)

            # Lolos guard
            success = logger_inst.log_sample(
                prompt="hitung kuadrat",
                thought="hitung kuadrat 3 adalah 9",
                code="def sq(x): return x*x",
                passed_guard=True
            )
            self.assertTrue(success)
            self.assertTrue(log_file.exists())

            # Gagal guard - tidak boleh dicatat
            success_fail = logger_inst.log_sample(
                prompt="hitung kuadrat",
                thought="hitung kuadrat 3 adalah 9",
                code="def sq(x): return x",
                passed_guard=False
            )
            self.assertFalse(success_fail)

            # Verifikasi isi file
            with open(log_file, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                self.assertEqual(data["prompt"], "hitung kuadrat")
                self.assertEqual(data["passed_guard"], True)

    def test_mandor_pekerja_orchestration(self):
        """Uji orkestrasi lengkap Mandor (planning) -> Pekerja (coding) -> Guard (verifikasi)."""
        pool = WorkerPool()
        mandor_cfg = APIConfig(name="mandor-ok", provider="openai", model_name="gpt-4o", api_keys=["M_KEY"])
        pekerja_cfg = APIConfig(name="pekerja-ok", provider="openai", model_name="gemini-2.5-flash", api_keys=["P_KEY"])
        pool.configs = [mandor_cfg, pekerja_cfg]
        pool.clients = {cfg.name: MokoAPIClient(cfg) for cfg in pool.configs}

        # Mock scan hasil
        pool.active_clients = ["mandor-ok", "pekerja-ok"]

        # Mock return Mandor
        mandor_plan = {
            "thought": "Merancang fungsi sum of squares",
            "target_module": "moko_generated_runtime.py",
            "test_module": "test_moko_generated.py",
            "folder_map": [],
            "steps": ["Step 1", "Step 2"],
            "outlines": {
                "moko_generated_runtime.py": "Buat fungsi moko_sum_of_squares",
                "test_moko_generated.py": "Buat test_moko_sum_of_squares"
            }
        }
        pool.clients["mandor-ok"].generate_text = MagicMock(return_value=json.dumps(mandor_plan))

        # Mock return Pekerja
        feature_code = (
            "<code>\n"
            "def moko_sum_of_squares(vals):\n"
            "    return sum(v * v for v in vals)\n"
            "</code>"
        )
        test_code = (
            "<code>\n"
            "from moko_generated_runtime import moko_sum_of_squares\n"
            "def test_moko_sum_of_squares():\n"
            "    assert moko_sum_of_squares([1, 2, 3]) == 14\n"
            "    print('MOKO_DUAL_TEST_PASSED')\n"
            "if __name__ == '__main__':\n"
            "    test_moko_sum_of_squares()\n"
            "</code>"
        )

        def mock_pekerja_generate(prompt, system_prompt, max_tokens=None):
            if "test_moko_generated" in prompt or "test" in prompt.lower():
                return test_code
            elif "moko_generated_runtime" in prompt:
                return feature_code
            return ""

        pool.clients["pekerja-ok"].generate_text = MagicMock(side_effect=mock_pekerja_generate)

        with tempfile.TemporaryDirectory() as tmp_ws:
            # Set target feature name pada Brain agar paritas dengan guard
            with patch("moko_agents.dual_system.brain_node.BrainNode.FEATURE_NAME", "moko_sum_of_squares"):
                orch = DualSystemOrchestrator(
                    workspace_dir=tmp_ws,
                    worker_pool=pool,
                    max_iterations=2
                )
                
                # Mock scan agar tidak melindas status client mock kita
                pool.scan_workers = MagicMock(return_value={"mandor-ok": True, "pekerja-ok": True})
                
                res = orch.run_loop("buatkan fungsi jumlah kuadrat")
                if not res.success:
                    print("\n--- MOCK RUN FAILED DETAILS ---")
                    print(f"Summary: {res.summary}")
                    print(f"Iterations: {res.iterations}")
                    for idx, trace in enumerate(res.traces):
                        print(f"Trace {idx} log:\n{trace.log}")
                        print(f"Trace {idx} verdict: {trace.guard_verdict}")
                        print(f"Trace {idx} summary: {trace.guard_summary}")
                self.assertTrue(res.success)
                self.assertEqual(res.iterations, 1)
                self.assertTrue((Path(tmp_ws) / "moko_generated_runtime.py").exists())
                self.assertTrue((Path(tmp_ws) / "test_moko_generated.py").exists())

    def test_free_gateways_integration(self):
        """Uji deteksi otomatis, prioritas, dan pemanggilan gateway API gratis (OmniRoute, 9Router, OpenCode)."""
        # Patch _fetch_gateway_models agar mengembalikan model default
        with patch.object(WorkerPool, "_fetch_gateway_models", return_value=["free-model-1"]):
            pool = WorkerPool()
            
            # Pastikan terdaftar otomatis
            self.assertIn("omniroute-free-model-1", pool.clients)
            self.assertIn("ninerouter-free-model-1", pool.clients)
            self.assertIn("opencode-free-model-1", pool.clients)
            
            omni = pool.clients["omniroute-free-model-1"]
            nine = pool.clients["ninerouter-free-model-1"]
            opencode = pool.clients["opencode-free-model-1"]
            
            self.assertEqual(omni.provider, "omniroute")
            self.assertEqual(nine.provider, "ninerouter")
            self.assertEqual(opencode.provider, "opencode")
            
            # Test 1: Prioritas saat omniroute aktif
            pool.active_clients = ["omniroute-free-model-1", "moko-local-coder"]
            mandor = pool.get_mandor()
            self.assertIsNotNone(mandor)
            self.assertEqual(mandor.name, "omniroute-free-model-1")
            
            # Test 2: Prioritas saat omniroute dan ninerouter aktif (omniroute harus lebih diprioritaskan)
            pool.active_clients = ["ninerouter-free-model-1", "omniroute-free-model-1", "moko-local-coder"]
            mandor = pool.get_mandor()
            self.assertEqual(mandor.name, "omniroute-free-model-1")
            
            # Test 3: Prioritas saat hanya ninerouter aktif
            pool.active_clients = ["ninerouter-free-model-1", "moko-local-coder"]
            mandor = pool.get_mandor()
            self.assertEqual(mandor.name, "ninerouter-free-model-1")
    
            # Test 4: Prioritas saat hanya opencode aktif
            pool.active_clients = ["opencode-free-model-1", "moko-local-coder"]
            mandor = pool.get_mandor()
            self.assertEqual(mandor.name, "opencode-free-model-1")
            
            # Test 5: Pengiriman request berhasil dialihkan ke _call_openai
            with patch.object(omni, "_call_openai", return_value="Respon OmniRoute") as mock_openai:
                res = omni.generate_text("halo")
                self.assertEqual(res, "Respon OmniRoute")
                mock_openai.assert_called_once()


if __name__ == "__main__":
    unittest.main()
