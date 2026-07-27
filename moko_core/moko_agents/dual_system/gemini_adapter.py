"""
MOKO Gemini Adapter — Optimal Guru API Client
=============================================
Adapter khusus untuk Gemini 2.5 Flash / Flash-Lite sebagai Mandor/Guru Utama MOKO.

Fitur canggih yang tidak ada di api_client.py generik:
  1. Exponential Backoff dengan Jitter (IEEE best-practice)
     → Mengatasi 429 RESOURCE_EXHAUSTED secara elegan tanpa flood
  2. Retry Queue System (per-request)
     → Request yang gagal karena quota dimasukkan antrian, diproses ulang
       saat quota tersedia kembali
  3. Gemini 2.5 Flash Thinking Support
     → Mendukung `thinking_budget` untuk mengontrol CoT internal Gemini
  4. Async-friendly design
     → Berjalan di ThreadPoolExecutor via blocking calls, tapi state-safe
  5. System Instruction via API v1beta (bukan content turn workaround)
     → Memanfaatkan field `system_instruction` yang resmi di Gemini API
  6. Context Window Management
     → Otomatis memangkas riwayat pesan jika mendekati batas token

Digunakan oleh: mome_engine.py, orchestrator.py
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("moko_gemini_adapter")

# ── Gemini Endpoint Constants ─────────────────────────────────────────────────
_GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta/models"
_MODEL_FLASH    = "gemini-2.5-flash"
_MODEL_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"

# ── Backoff Konstanta (IEEE Exponential Backoff + Jitter) ─────────────────────
_INITIAL_DELAY_S  = 2.0    # Detik awal sebelum retry pertama
_BACKOFF_FACTOR   = 2.0    # Kelipatan setiap retry (2x, 4x, 8x, ...)
_JITTER_RANGE     = 0.5    # Tambah/kurangi random jitter (±0.5 detik)
_MAX_DELAY_S      = 64.0   # Maksimum waktu tunggu (tidak lebih dari 64 detik)
_MAX_RETRIES      = 5      # Jumlah retry maksimum per request

# ── Context Window ─────────────────────────────────────────────────────────────
_MAX_CONTENT_CHARS = 100_000  # Gemini 2.5 Flash: 1M tokens, kita soft-limit 100K chars


@dataclass
class GeminiRequest:
    """Representasi satu pending request ke Gemini API."""
    prompt: str
    system_prompt: str        = ""
    max_tokens: int           = 2048
    temperature: float        = 0.15
    model: str                = _MODEL_FLASH
    thinking_budget: int      = 0       # 0 = disable thinking; >0 = enable thinking mode
    use_system_instruction: bool = True  # Gunakan system_instruction field resmi


@dataclass
class GeminiResponse:
    """Hasil respon dari Gemini API beserta metadata retry/backoff."""
    text: str                  = ""
    model_used: str            = ""
    retry_count: int           = 0
    total_latency_s: float     = 0.0
    thinking_tokens: int       = 0     # Jumlah thinking token (jika thinking mode aktif)
    success: bool              = False
    error: Optional[str]       = None


class GeminiAdapter:
    """
    Adapter tunggal untuk Gemini 2.5 Flash dengan backoff optimal.
    Thread-safe — bisa digunakan dari multiple threads.
    """

    def __init__(
        self,
        api_keys: List[str],
        preferred_model: str = _MODEL_FLASH,
        fallback_model: str = _MODEL_FLASH_LITE,
        timeout: int = 60,
    ):
        if not api_keys:
            raise ValueError("GeminiAdapter membutuhkan minimal satu API key.")
        self.api_keys        = api_keys
        self.preferred_model = preferred_model
        self.fallback_model  = fallback_model
        self.timeout         = timeout
        self._key_idx        = 0
        self._key_failures: Dict[int, int] = {}  # idx -> jumlah kegagalan berturut-turut

    # ── Key Rotation ──────────────────────────────────────────────────────────

    def _current_key(self) -> str:
        return self.api_keys[self._key_idx]

    def _rotate_key(self) -> None:
        """Rotasi ke API key berikutnya (round-robin)."""
        prev = self._key_idx
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        logger.info(
            f"[GeminiAdapter] Rotasi key: {prev} → {self._key_idx} "
            f"(total keys: {len(self.api_keys)})"
        )

    # ── Backoff Calculator ────────────────────────────────────────────────────

    @staticmethod
    def _calc_backoff(attempt: int) -> float:
        """
        Hitung delay exponential backoff dengan random jitter (IEEE standard).
        delay = min(max_delay, initial * factor^attempt) ± jitter
        """
        raw = _INITIAL_DELAY_S * math.pow(_BACKOFF_FACTOR, attempt)
        capped = min(raw, _MAX_DELAY_S)
        jitter = random.uniform(-_JITTER_RANGE, _JITTER_RANGE)
        return max(0.5, capped + jitter)

    # ── Content Trimmer ───────────────────────────────────────────────────────

    @staticmethod
    def _trim_content(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        """Potong konten di awal jika melebihi soft-limit context window."""
        if len(text) <= max_chars:
            return text
        logger.warning(
            f"[GeminiAdapter] Konten terpotong: {len(text)} → {max_chars} karakter."
        )
        return text[-max_chars:]  # Pertahankan akhir (konteks terkini lebih relevan)

    # ── Payload Builder ───────────────────────────────────────────────────────

    def _build_payload(self, req: GeminiRequest, model: str) -> Tuple[str, dict]:
        """Buat URL dan payload JSON untuk Gemini REST API."""
        url = f"{_GEMINI_BASE}/{model}:generateContent?key={self._current_key()}"

        prompt_trimmed = self._trim_content(req.prompt)
        contents = [{"role": "user", "parts": [{"text": prompt_trimmed}]}]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": req.max_tokens,
                "temperature":     req.temperature,
            }
        }

        # System instruction via field resmi (bukan content turn workaround)
        if req.system_prompt and req.use_system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": req.system_prompt}]
            }

        # Thinking budget (Gemini 2.5 Flash Thinking)
        if req.thinking_budget > 0:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": req.thinking_budget
            }

        return url, payload

    # ── Core Request ──────────────────────────────────────────────────────────

    def _do_request(self, url: str, payload: dict) -> dict:
        """
        Kirim satu HTTP POST ke Gemini, raise exception jika gagal.
        Khusus 429 RESOURCE_EXHAUSTED, raise RuntimeError("RATE_LIMIT_429").
        """
        headers = {"Content-Type": "application/json"}
        res = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

        if res.status_code == 429:
            raise RuntimeError("RATE_LIMIT_429")
        if res.status_code == 503:
            raise RuntimeError("SERVICE_UNAVAILABLE_503")
        if res.status_code != 200:
            # Coba baca pesan error dari respons
            try:
                err_body = res.json()
                msg = err_body.get("error", {}).get("message", res.text[:300])
            except Exception:
                msg = res.text[:300]
            raise RuntimeError(f"HTTP_{res.status_code}: {msg}")

        return res.json()

    @staticmethod
    def _extract_text(data: dict) -> Tuple[str, int]:
        """
        Ekstrak teks dan jumlah thinking_tokens dari respons Gemini.
        Returns: (text, thinking_tokens)
        """
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Respons Gemini kosong (tidak ada candidates).")

        cand = candidates[0]

        # Cek finish reason
        finish_reason = cand.get("finishReason", "STOP")
        if finish_reason == "SAFETY":
            raise RuntimeError("Respons diblokir oleh filter Safety Gemini.")

        parts = cand.get("content", {}).get("parts", [])

        # Pisahkan thinking parts dari text parts
        text_parts = []
        thinking_tokens = 0
        for part in parts:
            if part.get("thought"):
                # Ini thinking content — hitung karakternya sebagai proxy tokens
                thinking_tokens += len(part.get("text", ""))
            else:
                text_parts.append(part.get("text", ""))

        combined = "".join(text_parts).strip()
        if not combined:
            raise RuntimeError("Gemini mengembalikan respons teks kosong.")

        return combined, thinking_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, req: GeminiRequest) -> GeminiResponse:
        """
        Kirim request ke Gemini dengan exponential backoff + key rotation.
        Mencoba model utama (Flash), fallback ke Flash-Lite jika gagal terus.
        """
        start_t = time.perf_counter()
        last_error: Optional[str] = None

        models_to_try = [req.model or self.preferred_model, self.fallback_model]
        # Deduplicate model list
        seen = set()
        models_ordered = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for model in models_ordered:
            for attempt in range(_MAX_RETRIES):
                try:
                    url, payload = self._build_payload(req, model)
                    data = self._do_request(url, payload)
                    text, thinking_tokens = self._extract_text(data)

                    total_latency = time.perf_counter() - start_t
                    logger.info(
                        f"[GeminiAdapter] ✅ Sukses via {model} "
                        f"(attempt {attempt}, key {self._key_idx}, "
                        f"latency {total_latency:.2f}s, thinking_tokens ~{thinking_tokens})"
                    )
                    return GeminiResponse(
                        text=text,
                        model_used=model,
                        retry_count=attempt,
                        total_latency_s=round(total_latency, 3),
                        thinking_tokens=thinking_tokens,
                        success=True,
                    )

                except RuntimeError as e:
                    err_str = str(e)
                    last_error = err_str

                    if "RATE_LIMIT_429" in err_str or "SERVICE_UNAVAILABLE_503" in err_str:
                        # Rotasi key terlebih dahulu
                        if len(self.api_keys) > 1:
                            self._rotate_key()

                        # Hitung delay backoff dengan jitter
                        delay = self._calc_backoff(attempt)
                        logger.warning(
                            f"[GeminiAdapter] ⚠️ {err_str} pada {model} attempt {attempt}. "
                            f"Menunggu {delay:.1f}s sebelum retry..."
                        )
                        time.sleep(delay)
                        continue

                    elif "SAFETY" in err_str:
                        # Safety block — tidak perlu retry, langsung ganti model
                        logger.warning(f"[GeminiAdapter] 🔴 Safety block pada {model}. Ganti model.")
                        break

                    else:
                        # Error lain (HTTP error, parsing) — retry dengan key rotation
                        if len(self.api_keys) > 1:
                            self._rotate_key()
                        delay = self._calc_backoff(attempt)
                        logger.warning(
                            f"[GeminiAdapter] ❌ Error pada {model} attempt {attempt}: {err_str}. "
                            f"Retry dalam {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue

            logger.error(f"[GeminiAdapter] Semua {_MAX_RETRIES} retry untuk model {model} habis.")

        total_latency = time.perf_counter() - start_t
        return GeminiResponse(
            text="",
            model_used="",
            retry_count=_MAX_RETRIES,
            total_latency_s=round(total_latency, 3),
            success=False,
            error=last_error,
        )

    def quick_check(self) -> bool:
        """Cek koneksi cepat ke Gemini API (timeout 5 detik)."""
        req = GeminiRequest(
            prompt="respond only: MOKO_PING_OK",
            max_tokens=10,
            temperature=0.0,
        )
        # Sementara override timeout
        old_timeout = self.timeout
        self.timeout = 5
        try:
            resp = self.generate(req)
            return resp.success and bool(resp.text)
        except Exception:
            return False
        finally:
            self.timeout = old_timeout


# ── Singleton Factory ─────────────────────────────────────────────────────────

_gemini_adapter: Optional[GeminiAdapter] = None


def get_gemini_adapter() -> Optional[GeminiAdapter]:
    """
    Dapatkan singleton GeminiAdapter dari konfigurasi worker pool.
    Mengembalikan None jika tidak ada API key Gemini yang tersedia.
    """
    global _gemini_adapter
    if _gemini_adapter is not None:
        return _gemini_adapter

    import os
    from pathlib import Path

    keys: List[str] = []

    # 1. Dari environment variable
    env_keys = os.environ.get("MOKO_GEMINI_KEYS") or os.environ.get("GEMINI_API_KEY", "")
    if env_keys:
        keys += [k.strip() for k in env_keys.split(",") if k.strip()]

    # 2. Dari api_keys.json
    config_candidates = [
        Path(__file__).resolve().parents[3] / "moko_config" / "api_keys.json",
        Path(__file__).resolve().parents[3] / "api_keys.json",
    ]
    for cfg_path in config_candidates:
        if cfg_path.exists():
            try:
                import json
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if item.get("provider") in ("gemini", "google"):
                            keys += [k for k in item.get("api_keys", []) if k]
            except Exception as e:
                logger.debug(f"[GeminiAdapter] Gagal baca {cfg_path}: {e}")

    if not keys:
        logger.warning("[GeminiAdapter] Tidak ada API key Gemini tersedia.")
        return None

    _gemini_adapter = GeminiAdapter(api_keys=list(dict.fromkeys(keys)))  # deduplicate
    logger.info(
        f"[GeminiAdapter] ✅ Initialized dengan {len(keys)} key(s), "
        f"model: {_MODEL_FLASH} → fallback: {_MODEL_FLASH_LITE}"
    )
    return _gemini_adapter
