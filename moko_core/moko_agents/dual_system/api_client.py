"""
MOKO Unified API Client — Client API Agnostik dengan Failover Otomatis
=====================================================================
Mendukung Google AI Studio (Gemini), OpenAI-compatible (DeepSeek, OpenRouter, dll.),
Cloudflare Workers AI, Custom API, dan Model Lokal MOKO (tanpa saklar/switch).

Provider yang didukung:
  - "gemini"     : Google AI Studio REST API
  - "openai"     : Format OpenAI-compatible (DeepSeek, OpenRouter, GLM, Kimi, dll.)
  - "cloudflare" : Cloudflare Workers AI
  - "custom"     : Custom HTTP endpoint (atomesus, free APIs, dll.)
  - "local"      : Model lokal MOKO (tanpa API) — mencoba HTTP daemon dulu,
                   fallback otomatis ke in-process GGUF engine jika daemon offline.

Tidak ada "saklar AI" (binary switch). Model lokal diperlakukan sebagai
worker biasa di dalam pool, sehingga sistem bisa beroperasi penuh
bahkan tanpa koneksi internet.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import requests

logger = logging.getLogger("moko_api_client")

# Port default MOKO local inference server (moko_server.py)
_MOKO_LOCAL_PORT = 11435
_MOKO_LOCAL_URL  = f"http://127.0.0.1:{_MOKO_LOCAL_PORT}"


@dataclass
class APIConfig:
    name: str
    provider: str                         # "gemini" | "openai" | "cloudflare" | "custom"
    model_name: str
    api_base: str = ""
    api_keys: list[str] = field(default_factory=list)
    extra_headers: dict = field(default_factory=dict)
    extra_payload: dict = field(default_factory=dict)
    timeout: int = 45
    is_mandor: bool = False
    enabled: bool = True


class MokoAPIClient:
    """Client API Tunggal yang membungkus satu provider dengan pool key miliknya sendiri."""

    def __init__(self, config: APIConfig) -> None:
        import os
        self.config = config
        self.name = config.name
        self.provider = config.provider.lower()
        self.model_name = config.model_name
        
        self.api_keys = []
        for key in (config.api_keys or [""]):
            if key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                self.api_keys.append(os.environ.get(env_var, ""))
            else:
                self.api_keys.append(key)
                
        self.current_key_idx = 0
        self.consecutive_failures = 0

    def rotate_key(self) -> None:
        """Pindah ke kunci API berikutnya dalam pool."""
        if len(self.api_keys) > 1:
            old_idx = self.current_key_idx
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            logger.warning(
                f"[{self.name}] Rotasi API key dari index {old_idx} ke {self.current_key_idx}"
            )

    def get_current_key(self) -> str:
        return self.api_keys[self.current_key_idx]

    def test_connection(self) -> bool:
        """Cek konektivitas cepat ke API dengan token minimal."""
        # Provider lokal: verifikasi via daemon HTTP atau keberadaan GGUF,
        # tidak perlu network request ke luar.
        if self.provider == "local":
            return self._test_local_connection()

        # Fast connection check untuk gateway lokal sebelum melakukan full generation
        if self.provider in ("omniroute", "ninerouter", "opencode"):
            try:
                base_url = self.config.api_base
                if base_url:
                    models_url = f"{base_url.rstrip('/')}/models"
                    headers = {}
                    current_key = self.get_current_key()
                    if current_key:
                        headers["Authorization"] = f"Bearer {current_key}"
                    if self.config.extra_headers:
                        headers.update(self.config.extra_headers)
                    
                    r = requests.get(models_url, headers=headers, timeout=2.0)
                    if r.status_code == 200:
                        logger.info(f"[{self.name}] Fast connection check sukses via /models.")
                        return True
            except Exception as e:
                logger.debug(f"[{self.name}] Fast connection check via /models gagal: {e}. Melanjutkan ke full check.")

        try:
            res = self.generate_text("ping", system_prompt="respond only with pong", max_tokens=5, timeout=10)
            return len(res.strip()) > 0
        except Exception as e:
            logger.warning(f"[{self.name}] Connection test gagal: {e}")
            return False

    def _test_local_connection(self) -> bool:
        """Verifikasi lokal: cek HTTP daemon atau keberadaan file GGUF di disk."""
        # 1. Cek daemon HTTP lokal (cepat)
        try:
            r = requests.get(f"{_MOKO_LOCAL_URL}/health", timeout=0.8)
            if r.status_code == 200:
                logger.info(f"[{self.name}] Local daemon ONLINE di port {_MOKO_LOCAL_PORT}.")
                return True
        except Exception:
            pass

        # 2. Cek keberadaan file GGUF di disk (in-process fallback akan bisa digunakan)
        try:
            from pathlib import Path
            from moko_config import settings
            candidate_paths = [
                getattr(settings, "MODEL_MOKO_GGUF_PATH", ""),
                str(Path(settings.PROJECT_DIR) / "MOKO-Coder-1.5B-Uncensored-F16.gguf"),
                str(Path(settings.PROJECT_DIR) / "MOKO-AI-4B-Q3_K_M.gguf"),
            ]
            for p in candidate_paths:
                if p and Path(p).exists():
                    logger.info(f"[{self.name}] GGUF ditemukan di disk: {Path(p).name}. Local worker READY (in-process).")
                    return True
        except Exception as e:
            logger.debug(f"[{self.name}] GGUF scan error: {e}")

        logger.warning(f"[{self.name}] Local worker tidak tersedia (daemon offline & GGUF tidak ditemukan).")
        return False

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: int | None = None
    ) -> str:
        """Kirim permintaan teks dengan mekanisme rotasi key internal."""
        timeout = timeout or self.config.timeout
        errors = []

        # Coba setiap key yang tersedia sebelum menyerah
        for _ in range(len(self.api_keys)):
            current_key = self.get_current_key()
            try:
                text = self._dispatch_request(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    api_key=current_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
                self.consecutive_failures = 0
                return text
            except Exception as e:
                errors.append(f"Key index {self.current_key_idx}: {str(e)}")
                self.consecutive_failures += 1
                # Jika error 429 atau network, langsung rotasi key
                self.rotate_key()

        # Jika sampai di sini, semua key di provider ini gagal
        raise RuntimeError(
            f"Provider '{self.name}' gagal mengeksekusi request setelah mencoba seluruh API key. "
            f"Detail error: {'; '.join(errors)}"
        )

    def _dispatch_request(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """Routing internal berdasarkan jenis provider."""
        if self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt, api_key, max_tokens, temperature, timeout)
        elif self.provider in ("openai", "omniroute", "ninerouter", "opencode", "openrouter"):
            return self._call_openai(prompt, system_prompt, api_key, max_tokens, temperature, timeout)
        elif self.provider == "cloudflare":
            return self._call_cloudflare(prompt, system_prompt, api_key, max_tokens, temperature, timeout)
        elif self.provider == "custom":
            return self._call_custom(prompt, system_prompt, api_key, max_tokens, temperature, timeout)
        elif self.provider == "local":
            return self._call_local(prompt, system_prompt, api_key, max_tokens, temperature, timeout)
        else:
            raise ValueError(f"Provider tidak dikenali: {self.provider}")

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """Memanggil REST API Google AI Studio."""
        # Menggunakan REST API endpoint v1beta
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System Instruction: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Paham. Saya akan mengikuti instruksi sistem tersebut."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)

        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if res.status_code == 429:
            raise RuntimeError("HTTP 429: Rate Limit Terlampaui")
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}: {res.text}")

        try:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gagal memparsing respon Gemini: {e}. Respon mentah: {res.text}")

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """Memanggil REST API OpenAI-compatible."""
        base_url = self.config.api_base or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)

        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if res.status_code == 429:
            raise RuntimeError("HTTP 429: Rate Limit Terlampaui")
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}: {res.text}")

        try:
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gagal memparsing respon OpenAI: {e}. Respon mentah: {res.text}")

    def _call_cloudflare(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """Memanggil Cloudflare Workers AI."""
        # Cloudflare format: https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
        base_url = self.config.api_base or "https://api.cloudflare.com/client/v4/accounts/default/ai/run"
        url = f"{base_url.rstrip('/')}/{self.model_name}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)

        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if res.status_code == 429:
            raise RuntimeError("HTTP 429: Rate Limit Terlampaui")
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}: {res.text}")

        try:
            data = res.json()
            if data.get("success") is False:
                errors = ", ".join(e.get("message", "") for e in data.get("errors", []))
                raise RuntimeError(f"Cloudflare error: {errors}")
            return data["result"]["response"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Gagal memparsing respon Cloudflare: {e}. Respon mentah: {res.text}")

    def _call_custom(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """Memanggil Custom HTTP endpoint (e.g. atomesus, free APIs)."""
        url = self.config.api_base
        if not url:
            raise ValueError(f"Provider '{self.name}' bertipe custom membutuhkan 'api_base' url.")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)

        res = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if res.status_code == 429:
            raise RuntimeError("HTTP 429: Rate Limit Terlampaui")
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}: {res.text}")

        # Fallback parsing default: coba OpenAI format, jika tidak coba parsing string langsung
        try:
            data = res.json()
            if isinstance(data, dict):
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                if "response" in data:
                    return data["response"]
                if "text" in data:
                    return data["text"]
            return str(data)
        except Exception:
            return res.text

    def _call_local(
        self,
        prompt: str,
        system_prompt: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: int
    ) -> str:
        """
        Memanggil model lokal MOKO.

        Strategi berlapis (tidak ada saklar — selalu mencoba yang paling efisien):
          1. HTTP daemon lokal (port 11435) — zero-copy inference, paling cepat
             jika moko_server.py sudah berjalan di background.
          2. In-process GGUF engine (llama-cpp-python) — selalu bisa digunakan
             selama file GGUF tersedia di disk, meskipun daemon tidak berjalan.

        Dengan demikian sistem tidak pernah bergantung penuh pada API eksternal.
        Model lokal adalah citizen kelas pertama di dalam worker pool.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ─── Strategi 1: HTTP Daemon Lokal ──────────────────────────────────
        try:
            health = requests.get(f"{_MOKO_LOCAL_URL}/health", timeout=0.8)
            if health.status_code == 200:
                payload = {
                    "model": self.model_name or "moko-local",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                res = requests.post(
                    f"{_MOKO_LOCAL_URL}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
                logger.debug(f"[{self.name}] Local daemon HTTP error {res.status_code}, fallback ke in-process.")
        except Exception as e:
            logger.debug(f"[{self.name}] Local daemon tidak tersedia ({e}), fallback ke in-process engine.")

        # ─── Strategi 2: In-Process GGUF Engine ─────────────────────────────
        try:
            from moko_inference.moko_engine import get_moko_engine
            engine = get_moko_engine()
            if engine is not None:
                content, _ = engine.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return content
            raise RuntimeError(
                "In-process GGUF engine tidak dapat diinisialisasi "
                "(file GGUF tidak ditemukan di workspace)."
            )
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python tidak terinstall dan daemon lokal offline. "
                "Install dengan: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.name}] Semua strategi inferensi lokal gagal: {e}"
            )
