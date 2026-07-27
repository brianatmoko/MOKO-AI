"""
MOKO Hybrid Chat Engine
=======================
Menjembatani panel chat MOKO OS langsung ke API gateway gratis
(OmniRouter, 9Router, OpenCode) dengan fallback otomatis ke model lokal.

Cara kerja:
1. Saat startup: scan semua worker di background thread (non-blocking).
2. Saat user chat: ambil mandor aktif (dari setting atau prioritas otomatis).
3. Streaming response langsung dari API eksternal ke UI.
4. Jika semua API offline → fallback mulus ke llm_engine lokal.

Ini adalah FAST-PATH — tidak melewati AnalystNode yang berat.
Hanya digunakan untuk pertanyaan umum (FAST_PATH / D0-D5).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("moko_hybrid_chat")


class HybridChatEngine:
    """Engine chat hybrid: API eksternal gratis + fallback lokal."""

    # Singleton state
    _pool = None
    _pool_lock = threading.Lock()
    _scan_done = False
    _scan_thread: Optional[threading.Thread] = None
    _last_scan_ts: float = 0.0
    _RESCAN_INTERVAL: float = 120.0  # rescan tiap 2 menit

    @classmethod
    def _get_pool(cls):
        """Inisialisasi WorkerPool (sekali, thread-safe)."""
        if cls._pool is None:
            with cls._pool_lock:
                if cls._pool is None:
                    try:
                        from moko_agents.dual_system.worker_pool import WorkerPool
                        cls._pool = WorkerPool()
                        logger.info("[HybridChat] WorkerPool initialized.")
                    except Exception as e:
                        logger.error(f"[HybridChat] Gagal init WorkerPool: {e}")
        return cls._pool

    @classmethod
    def start_background_scan(cls):
        """
        Jalankan scan_workers() di background thread saat startup.
        Non-blocking — UI tidak perlu menunggu.
        """
        pool = cls._get_pool()
        if pool is None:
            return

        now = time.time()
        # Jangan re-scan jika terlalu dekat dengan scan sebelumnya
        if cls._scan_done and (now - cls._last_scan_ts) < cls._RESCAN_INTERVAL:
            return

        # Hindari race condition: jangan mulai thread baru jika masih scan
        if cls._scan_thread and cls._scan_thread.is_alive():
            return

        def _do_scan():
            try:
                logger.info("[HybridChat] 🔍 Background scan worker dimulai...")
                results = pool.scan_workers()
                active = [n for n, ok in results.items() if ok]
                logger.info(
                    f"[HybridChat] ✅ Scan selesai. Worker aktif: {active or ['(tidak ada)']}"
                )
                cls._scan_done = True
                cls._last_scan_ts = time.time()
            except Exception as e:
                logger.error(f"[HybridChat] Scan error: {e}")

        cls._scan_thread = threading.Thread(target=_do_scan, daemon=True, name="MokoWorkerScan")
        cls._scan_thread.start()

    @classmethod
    def _pick_model_for_gateway(cls, worker) -> str:
        """
        Pilih model terbaik dari gateway secara otomatis.
        Jika model_name == 'auto', ambil model pertama dari /v1/models.
        """
        pool = cls._get_pool()
        if pool is None:
            return worker.model_name

        # Jika model sudah dikonfigurasi secara eksplisit, pakai itu
        if worker.model_name and worker.model_name not in ("auto", ""):
            return worker.model_name

        # Ambil dari daftar model aktif yang sudah terdaftar di pool untuk gateway ini
        provider = worker.provider
        gateway_workers = [
            c for name, c in pool.clients.items()
            if c.provider == provider and c.model_name not in ("auto", "")
        ]
        if gateway_workers:
            chosen = gateway_workers[0].model_name
            logger.debug(f"[HybridChat] Auto-pilih model '{chosen}' untuk {provider}")
            return chosen

        return worker.model_name

    @classmethod
    def get_mandor_worker(cls):
        """
        Ambil client mandor aktif dari WorkerPool.
        Prioritas: is_mandor=True > Gemini > OmniRoute > 9Router > OpenCode > Lokal.
        Returns None jika tidak ada worker aktif.
        """
        pool = cls._get_pool()
        if pool is None:
            return None

        # Pastikan scan sudah dimulai
        cls.start_background_scan()

        # Jika scan belum selesai, coba pakai siapa pun yang terdaftar
        if not pool.active_clients:
            # Coba quick-connect ke worker pertama yang enabled
            for name, client in pool.clients.items():
                if client.config.enabled and client.provider not in ("local",):
                    logger.info(f"[HybridChat] Scan belum selesai, mencoba {name} langsung...")
                    try:
                        ok = client.test_connection()
                        if ok:
                            pool.active_clients.append(name)
                            logger.info(f"[HybridChat] ✅ {name} tersedia!")
                            break
                    except Exception:
                        pass

        mandor = pool.get_mandor()
        if mandor and mandor.provider != "local":
            return mandor

        return None

    @classmethod
    def generate(
        cls,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        on_token: Optional[Callable[[str], None]] = None,
        session_messages: Optional[list] = None,
    ) -> str:
        """
        Generate respons via API eksternal. Fallback ke lokal jika gagal.

        Args:
            prompt: Pesan user
            system_prompt: System prompt
            max_tokens: Maks token output
            on_token: Callback streaming token per token
            session_messages: Riwayat percakapan (OpenAI format)

        Returns:
            Teks respons lengkap
        """
        mandor = cls.get_mandor_worker()
        if mandor is not None:
            # Tentukan model yang benar
            model_to_use = cls._pick_model_for_gateway(mandor)
            if model_to_use and model_to_use not in ("auto", ""):
                mandor.model_name = model_to_use

            logger.info(
                f"[HybridChat] 🚀 Mengirim ke '{mandor.name}' "
                f"(provider={mandor.provider}, model={mandor.model_name})"
            )

            # Build prompt dengan session messages jika ada
            full_prompt = prompt
            if session_messages:
                history_text = "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}"
                    for m in session_messages[-6:]  # Max 6 pesan terakhir
                    if m.get("content")
                )
                if history_text:
                    full_prompt = history_text + f"\nUser: {prompt}"

            try:
                response = mandor.generate_text(
                    prompt=full_prompt,
                    system_prompt=system_prompt or _build_moko_system_prompt(),
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                if response and response.strip():
                    if on_token:
                        on_token(response)
                    logger.info(
                        f"[HybridChat] ✅ Respons diterima dari '{mandor.name}' "
                        f"({len(response)} karakter)"
                    )
                    return response
                else:
                    logger.warning(f"[HybridChat] '{mandor.name}' mengembalikan respons kosong.")
            except Exception as e:
                logger.warning(f"[HybridChat] '{mandor.name}' gagal: {e}. Fallback ke lokal...")
                # Tandai worker ini gagal (opsional: mark inactive)
                pool = cls._get_pool()
                if pool and mandor.name in pool.active_clients:
                    pool.active_clients.remove(mandor.name)
                    mandor.consecutive_failures += 1

        # ── FALLBACK: Model Lokal ────────────────────────────────────────────
        logger.info("[HybridChat] 🔄 Menggunakan fallback model lokal...")
        return cls._local_fallback(prompt, system_prompt, max_tokens, on_token, session_messages)

    @classmethod
    def _local_fallback(
        cls,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
        session_messages: Optional[list],
    ) -> str:
        """Fallback ke llm_engine lokal."""
        try:
            from moko_agents.llm_engine import engine
            response = engine.generate_stream(
                prompt=prompt,
                system_prompt=system_prompt or _build_moko_system_prompt(),
                coop_params={"num_predict": max_tokens, "enable_thinking": False, "temperature": 0.3},
                session_messages=session_messages,
                on_token=on_token,
            )
            return response
        except Exception as e:
            logger.error(f"[HybridChat] Local fallback juga gagal: {e}")
            return "Maaf, semua mesin inferensi tidak tersedia saat ini."

    @classmethod
    def is_external_available(cls) -> bool:
        """True jika ada API eksternal aktif dan siap dipakai."""
        mandor = cls.get_mandor_worker()
        return mandor is not None and mandor.provider != "local"

    @classmethod
    def get_status(cls) -> dict:
        """Status ringkas untuk ditampilkan di UI/log."""
        pool = cls._get_pool()
        if pool is None:
            return {"status": "pool_unavailable", "active": [], "mandor": None}

        mandor = pool.get_mandor()
        return {
            "status": "ready",
            "scan_done": cls._scan_done,
            "active_workers": pool.active_clients,
            "total_workers": len(pool.clients),
            "mandor": mandor.name if mandor else None,
            "mandor_provider": mandor.provider if mandor else None,
        }


def _build_moko_system_prompt() -> str:
    """System prompt default MOKO untuk API eksternal."""
    return (
        "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian. "
        "Jawab dengan singkat, natural, dan percaya diri. "
        "Gunakan bahasa Indonesia kecuali diminta sebaliknya. "
        "Jangan menyebut nama model AI lain."
    )


# ── Singleton Instance ────────────────────────────────────────────────────────

hybrid_chat = HybridChatEngine()
