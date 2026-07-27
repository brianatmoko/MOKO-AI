import unittest

from moko_llm_runtime_guard import LLMRuntimeGuard, normalize_server_status


class MokoLLMRuntimeGuardTest(unittest.TestCase):
    def test_normalize_status_offline_not_false_positive(self) -> None:
        self.assertEqual(normalize_server_status({"status": "offline"}), "offline")

    def test_normalize_status_online_formats(self) -> None:
        self.assertEqual(normalize_server_status({"status": "online"}), "online")
        self.assertEqual(normalize_server_status({"state": "running"}), "online")
        self.assertEqual(normalize_server_status({"ready": True}), "online")

    def test_generate_uses_template_when_offline(self) -> None:
        guard = LLMRuntimeGuard(
            status_provider=lambda: {"status": "offline"},
            llm_generate=lambda _prompt: "print('llm')",
            fallback_generate=lambda _prompt: "print('template')",
        )

        result = guard.generate("buat kalkulator")
        self.assertEqual(result.source, "template")
        self.assertEqual(result.server_status, "offline")
        self.assertIn("MOKO SERVER OFFLINE", result.message)

    def test_generate_uses_template_when_llm_empty(self) -> None:
        guard = LLMRuntimeGuard(
            status_provider=lambda: {"status": "online"},
            llm_generate=lambda _prompt: "   ",
            fallback_generate=lambda _prompt: "print('template')",
        )

        result = guard.generate("buat kalkulator")
        self.assertEqual(result.source, "template")
        self.assertEqual(result.server_status, "online")
        self.assertEqual(result.used_fallback_reason, "empty_llm_output")


if __name__ == "__main__":
    unittest.main()
