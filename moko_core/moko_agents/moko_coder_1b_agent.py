"""
MOKO Coder 1B Agent — Dedicated Code Intelligence Agent
==========================================================
Agent ini adalah antarmuka tingkat tinggi antara model fine-tuned
MOKO Coder 1B (GGUF Q4_K_M) dan sistem MOKO IDE.

Cara kerja:
  1. Untuk task coding/math/security → gunakan model 1B lokal yang
     sudah di-fine-tune dengan knowledge khusus MOKO OS
  2. Jika model 1B tidak tersedia → graceful fallback ke engine utama
  3. Mendukung mode streaming untuk live code completion di IDE

Fitur:
  - code_completion(prefix, suffix)   → Inline code completion
  - explain_code(code)               → Penjelasan kode + arsitektur MOKO
  - generate_function(description)   → Buat fungsi dari deskripsi natural language
  - debug_code(code, error)          → Analisa bug dan saran perbaikan
  - moko_os_query(question)          → Tanya jawab khusus MOKO OS internals
"""

import requests
import time
import json
from typing import Optional, Generator
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

try:
    from moko_config import settings
    _CODER_PORT = getattr(settings, "MOKO_CODER_PORT", 11435)
    _CODER_API_URL = f"http://127.0.0.1:{_CODER_PORT}"
    _PROJECT_DIR = Path(settings.PROJECT_DIR) if hasattr(settings, "PROJECT_DIR") else Path(__file__).parent.parent.parent
except ImportError:
    _CODER_PORT = 11435
    _CODER_API_URL = "http://127.0.0.1:11435"
    _PROJECT_DIR = Path(__file__).parent.parent.parent

_CODER_GGUF_PATH = _PROJECT_DIR / "moko-coder-1b.gguf"


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS (MOKO OS–AWARE)
# ════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_BASE = """Kamu adalah MOKO Coder 1B — AI coding assistant khusus untuk MOKO OS.
MOKO OS adalah sistem operasi custom berbasis Linux yang ditulis dalam C++ dan Python.

Pengetahuan inti MOKO OS yang kamu miliki:
- Arsitektur multi-agent (Core Node, Prefrontal, Amygdala, Cerebellum, Insula, DMN)
- Pipeline inference lokal menggunakan llama.cpp dan llama-server
- Multi-model dispatcher dengan VRAM management untuk RTX 2050 (4GB)
- RAG system dengan nomic-embed-text embeddings
- Dual-system orchestrator (Mandor-Pekerja paradigm)
- MOKO IDE dengan LSP integration dan code_editor.cpp

Prinsip:
1. Berikan kode yang akurat, efisien, dan idiomatic C++ / Python
2. Gunakan pola arsitektur yang konsisten dengan codebase MOKO OS
3. Prioritaskan memory-efficiency karena resource terbatas (4GB VRAM, 16GB RAM)
4. Selalu tambahkan docstring dan komentar bahasa Indonesia yang informatif
"""

_SYSTEM_CODE_COMPLETION = _SYSTEM_PROMPT_BASE + """
Mode: Code Completion.
Lanjutkan kode yang diberikan secara natural dan logis.
Berikan HANYA kode pelengkap saja, tanpa penjelasan atau markdown.
"""

_SYSTEM_EXPLAIN = _SYSTEM_PROMPT_BASE + """
Mode: Code Explanation.
Jelaskan kode yang diberikan secara terstruktur:
1. Apa yang dilakukan kode ini
2. Bagaimana cara kerjanya (step by step)
3. Keterkaitan dengan arsitektur MOKO OS (jika relevan)
4. Potensi masalah atau area yang bisa dioptimasi
"""

_SYSTEM_GENERATE = _SYSTEM_PROMPT_BASE + """
Mode: Code Generation.
Buat implementasi lengkap berdasarkan deskripsi yang diberikan.
Sertakan:
- Header docstring
- Implementasi fungsi/class yang diminta
- Contoh penggunaan singkat di akhir sebagai komentar
Gunakan pola kode yang konsisten dengan MOKO OS codebase.
"""

_SYSTEM_DEBUG = _SYSTEM_PROMPT_BASE + """
Mode: Debug Assistant.
Analisa kode dan error yang diberikan:
1. Identifikasi root cause error
2. Jelaskan mengapa error terjadi
3. Berikan solusi perbaikan dengan kode yang sudah dikoreksi
4. Sarankan pencegahan untuk bug serupa
"""

_SYSTEM_MOKO_QUERY = _SYSTEM_PROMPT_BASE + """
Mode: MOKO OS Expert Query.
Jawab pertanyaan teknis tentang MOKO OS dengan akurat dan mendalam.
Jika pertanyaan berkaitan dengan arsitektur atau implementasi spesifik,
berikan referensi ke komponen MOKO yang relevan.
"""


# ════════════════════════════════════════════════════════════════════════════
# CODER AGENT CLASS
# ════════════════════════════════════════════════════════════════════════════

class MokoCoderAgent:
    """
    Agent wrapper untuk MOKO Coder 1B.
    
    Menghubungkan model fine-tuned ke sistem MOKO IDE dengan API yang
    bersih dan konsisten. Mendukung fallback otomatis jika model 1B
    tidak tersedia.
    """

    def __init__(self, api_url: str = _CODER_API_URL, timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout
        self._model_available: Optional[bool] = None
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0

    # ────────────────────────────────────────────────────────────────────────
    # HEALTH CHECK
    # ────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Cek apakah model 1B GGUF sudah ada dan server aktif."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval and self._model_available is not None:
            return self._model_available

        # 1. Cek apakah GGUF file sudah ada
        if not _CODER_GGUF_PATH.exists():
            self._model_available = False
            self._last_health_check = now
            return False

        # 2. Cek apakah llama-server sedang menjalankan model ini
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=2)
            self._model_available = resp.status_code == 200
        except Exception:
            self._model_available = False

        self._last_health_check = now
        return self._model_available

    def get_status(self) -> dict:
        """Kembalikan status lengkap model 1B."""
        gguf_exists = _CODER_GGUF_PATH.exists()
        gguf_size_mb = (_CODER_GGUF_PATH.stat().st_size / 1024**2) if gguf_exists else 0
        server_active = False

        if gguf_exists:
            try:
                resp = requests.get(f"{self.api_url}/health", timeout=2)
                server_active = resp.status_code == 200
            except Exception:
                pass

        return {
            "model_gguf_path": str(_CODER_GGUF_PATH),
            "gguf_exists": gguf_exists,
            "gguf_size_mb": round(gguf_size_mb, 1),
            "server_active": server_active,
            "api_url": self.api_url,
            "ready": gguf_exists and server_active,
        }

    # ────────────────────────────────────────────────────────────────────────
    # CORE API CALL
    # ────────────────────────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[list] = None,
    ) -> str:
        """
        Kirim request ke llama-server lokal menggunakan format ChatML.
        Mengembalikan string respons, atau string kosong jika gagal.
        """
        payload = {
            "model": "moko-coder-1b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            print(f"  ⚠️ [MokoCoderAgent] Timeout setelah {self.timeout}s")
            return ""
        except Exception as e:
            print(f"  ⚠️ [MokoCoderAgent] Error: {e}")
            return ""

    def _call_stream(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """
        Streaming version dari _call. Yield token per token.
        Berguna untuk live code completion di MOKO IDE.
        """
        payload = {
            "model": "moko-coder-1b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                pass
        except Exception as e:
            print(f"  ⚠️ [MokoCoderAgent] Stream error: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────────────

    def code_completion(
        self,
        prefix: str,
        suffix: str = "",
        language: str = "python",
        max_tokens: int = 256,
        stream: bool = False,
    ) -> str:
        """
        Inline code completion — melengkapi kode berdasarkan prefix dan suffix.
        
        Args:
            prefix: Kode di atas kursor
            suffix: Kode di bawah kursor (fill-in-the-middle)
            language: Bahasa pemrograman (python, cpp, etc.)
            max_tokens: Jumlah token maksimum yang di-generate
            stream: Jika True, kembalikan generator
            
        Returns:
            Kode pelengkap sebagai string
        """
        if suffix:
            prompt = (
                f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
            )
        else:
            prompt = f"Lanjutkan kode {language} berikut:\n\n```{language}\n{prefix}\n```\n\nPelengkap:"

        if stream:
            return self._call_stream(prompt, _SYSTEM_CODE_COMPLETION, max_tokens=max_tokens, temperature=0.0)

        return self._call(prompt, _SYSTEM_CODE_COMPLETION, max_tokens=max_tokens, temperature=0.0)

    def explain_code(self, code: str, language: str = "python") -> str:
        """
        Jelaskan kode yang diberikan, termasuk kaitannya dengan MOKO OS.
        
        Args:
            code: Kode yang ingin dijelaskan
            language: Bahasa pemrograman
            
        Returns:
            Penjelasan dalam bahasa Indonesia
        """
        prompt = f"Jelaskan kode {language} berikut:\n\n```{language}\n{code}\n```"
        return self._call(prompt, _SYSTEM_EXPLAIN, max_tokens=768, temperature=0.1)

    def generate_function(
        self,
        description: str,
        language: str = "python",
        context: str = "",
    ) -> str:
        """
        Generate fungsi/class dari deskripsi natural language.
        
        Args:
            description: Deskripsi fungsi yang diinginkan
            language: Target bahasa (python, cpp)
            context: Konteks tambahan (contoh: nama class yang sudah ada)
            
        Returns:
            Implementasi kode lengkap
        """
        context_section = f"\nKonteks:\n{context}\n" if context else ""
        prompt = (
            f"Buatkan implementasi {language} untuk kebutuhan berikut:\n\n"
            f"{description}"
            f"{context_section}\n\n"
            f"Berikan kode {language} yang lengkap dan siap pakai:"
        )
        return self._call(prompt, _SYSTEM_GENERATE, max_tokens=1024, temperature=0.05)

    def debug_code(self, code: str, error_msg: str, language: str = "python") -> str:
        """
        Analisa bug dan sarankan perbaikan.
        
        Args:
            code: Kode yang bermasalah
            error_msg: Pesan error yang muncul
            language: Bahasa pemrograman
            
        Returns:
            Analisis bug dan kode yang sudah diperbaiki
        """
        prompt = (
            f"Kode {language} berikut menghasilkan error:\n\n"
            f"```{language}\n{code}\n```\n\n"
            f"Error:\n```\n{error_msg}\n```\n\n"
            f"Tolong analisa dan perbaiki:"
        )
        return self._call(prompt, _SYSTEM_DEBUG, max_tokens=1024, temperature=0.1)

    def moko_os_query(self, question: str) -> str:
        """
        Tanya jawab tentang MOKO OS internals — domain knowledge khusus.
        
        Args:
            question: Pertanyaan teknis tentang MOKO OS
            
        Returns:
            Jawaban berdasarkan knowledge MOKO OS
        """
        return self._call(question, _SYSTEM_MOKO_QUERY, max_tokens=1024, temperature=0.1)

    def review_code(self, code: str, language: str = "python") -> str:
        """
        Code review — analisa kualitas, bug potential, dan saran perbaikan.
        
        Args:
            code: Kode yang ingin di-review
            language: Bahasa pemrograman
            
        Returns:
            Review terstruktur dengan saran perbaikan
        """
        system = _SYSTEM_PROMPT_BASE + """
Mode: Code Review.
Lakukan code review terstruktur:
1. Ringkasan apa yang kode lakukan
2. Potensi bug atau edge case yang tidak ditangani
3. Masalah performa atau efisiensi
4. Saran refactoring / perbaikan dengan contoh kode
5. Penilaian keseluruhan (1-10) dan alasannya
"""
        prompt = f"Lakukan code review untuk kode {language} berikut:\n\n```{language}\n{code}\n```"
        return self._call(prompt, system, max_tokens=1024, temperature=0.1)


# ════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ════════════════════════════════════════════════════════════════════════════

# Singleton instance — bisa di-import langsung oleh modul lain
coder_agent = MokoCoderAgent()


# ════════════════════════════════════════════════════════════════════════════
# CLI QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("MOKO Coder 1B Agent — Status Check")
    print("=" * 60)

    status = coder_agent.get_status()
    for key, val in status.items():
        print(f"  {key}: {val}")

    if not status["ready"]:
        print("\n⚠️  Model 1B belum siap.")
        if not status["gguf_exists"]:
            print("   → GGUF belum ada. Jalankan training pipeline terlebih dahulu:")
            print("   → python3 finetune/moko_trainer_v2.py --train --epochs 3")
            print("   → python3 finetune/moko_trainer_v2.py --gguf")
        else:
            print("   → GGUF ada tapi server tidak aktif. Pastikan llama-server berjalan.")
        sys.exit(1)

    print("\n✅ Model 1B SIAP! Menjalankan quick test...\n")

    # Quick smoke test
    test_cases = [
        ("MOKO OS Query", lambda: coder_agent.moko_os_query(
            "Apa itu MultiModelDispatcher di MOKO OS dan bagaimana cara kerjanya?"
        )),
        ("Code Completion", lambda: coder_agent.code_completion(
            "def load_model(path: str):\n    \"\"\"Load GGUF model dari disk.\"\"\"\n    ",
            language="python"
        )),
        ("Debug Code", lambda: coder_agent.debug_code(
            "result = engine.generate(prompt)\nprint(result.text)",
            "AttributeError: 'str' object has no attribute 'text'",
            language="python"
        )),
    ]

    for test_name, test_fn in test_cases:
        print(f"--- {test_name} ---")
        try:
            result = test_fn()
            preview = result[:200] + "..." if len(result) > 200 else result
            print(f"{preview}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
