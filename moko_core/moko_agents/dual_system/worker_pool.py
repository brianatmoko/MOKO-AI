"""
MOKO Worker Pool — Pendaftaran, Pemindaian, dan Manajemen Pekerja API
====================================================================
Membaca konfigurasi penyedia API dari environment variables dan berkas JSON lokal.
Secara otomatis mendaftarkan model lokal MOKO sebagai pekerja di dalam pool
(tidak ada saklar — model lokal selalu ada sebagai fallback).

Hirarki Peran (sesuai riset 23_REVISI_MANDOR_API_MURID_LOKAL.md):
  - GURU/MANDOR  : External API (Gemini, OpenAI, dll.) — menulis kode, menghasilkan data latih
  - MURID/LOKAL  : MOKO-Coder 1.5B — belajar dari guru lewat distilasi, siap menggantikan
                   guru untuk tugas mudah seiring bertambahnya data SFT lokal.
  - GUARD        : runtime_guard.py — SELALU lokal, tidak berubah.

Pola ini identik dengan DeepSeek-R1 → distill ke 1.5B/8B:
satu guru besar mengajar murid kecil lewat contoh nyata.
"""
from __future__ import annotations

import json
import logging
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from moko_agents.dual_system.api_client import APIConfig, MokoAPIClient

logger = logging.getLogger("moko_worker_pool")


class WorkerPool:
    """Mengelola pool MokoAPIClient untuk Mandor dan Pekerja."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.configs: list[APIConfig] = []
        self.clients: dict[str, MokoAPIClient] = {}
        self.active_clients: list[str] = []  # Menyimpan nama client yang lulus scan

        # Tentukan path default ke api_keys.json di folder moko_config atau project root
        if config_path is None:
            project_dir = Path(__file__).resolve().parents[3]
            self.config_path = project_dir / "moko_config" / "api_keys.json"
            if not self.config_path.exists():
                self.config_path = project_dir / "api_keys.json"
        else:
            self.config_path = Path(config_path)

        self.load_configs()

    def _fetch_gateway_models(self, api_base: str, api_keys: list[str]) -> list[str]:
        """Menarik daftar model dari gateway secara dinamis."""
        try:
            headers = {}
            if api_keys and api_keys[0] and not api_keys[0].endswith("dummy-key"):
                headers["Authorization"] = f"Bearer {api_keys[0]}"
            models_url = f"{api_base.rstrip('/')}/models"
            r = requests.get(models_url, headers=headers, timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                models = []
                if "data" in data and isinstance(data["data"], list):
                    for m in data["data"]:
                        if "id" in m:
                            models.append(m["id"])
                return models
        except Exception as e:
            logger.debug(f"Gagal menarik model dari gateway {api_base}: {e}")
        return []

    def load_configs(self) -> None:
        """Muat konfigurasi dari file JSON dan Environment Variables."""
        configs_loaded = []

        # 1. Coba baca dari file JSON
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        for item in data:
                            if "name" in item and "provider" in item:
                                cfg = APIConfig(
                                    name=item["name"],
                                    provider=item["provider"],
                                    model_name=item.get("model_name", ""),
                                    api_base=item.get("api_base", ""),
                                    api_keys=item.get("api_keys", []),
                                    extra_headers=item.get("extra_headers", {}),
                                    extra_payload=item.get("extra_payload", {}),
                                    timeout=item.get("timeout", 45),
                                    is_mandor=item.get("is_mandor", False),
                                    enabled=item.get("enabled", True)
                                )
                                configs_loaded.append(cfg)
                logger.info(f"Berhasil memuat {len(configs_loaded)} konfigurasi dari {self.config_path.name}")
            except Exception as e:
                logger.error(f"Gagal membaca file konfigurasi {self.config_path}: {e}")

        # 2. Muat fallback otomatis dari Env Variables jika tidak ada config atau ingin ditambahkan
        # Format: MOKO_GEMINI_KEYS="key1,key2" dll.
        env_gemini_keys = os.environ.get("MOKO_GEMINI_KEYS") or os.environ.get("GEMINI_API_KEY")
        if env_gemini_keys:
            keys = [k.strip() for k in env_gemini_keys.split(",") if k.strip()]
            configs_loaded.append(
                APIConfig(
                    name="env-gemini-flash",
                    provider="gemini",
                    model_name="gemini-2.5-flash",
                    api_keys=keys
                )
            )

        env_openai_keys = os.environ.get("MOKO_OPENAI_KEYS") or os.environ.get("OPENAI_API_KEY")
        if env_openai_keys:
            keys = [k.strip() for k in env_openai_keys.split(",") if k.strip()]
            configs_loaded.append(
                APIConfig(
                    name="env-openai-compatible",
                    provider="openai",
                    model_name="gpt-4o-mini",
                    api_base=os.environ.get("MOKO_OPENAI_BASE", "https://api.openai.com/v1"),
                    api_keys=keys
                )
            )

        self.configs = configs_loaded
        
        # Inisialisasi client objects
        self.clients = {}
        for cfg in self.configs:
            self.clients[cfg.name] = MokoAPIClient(cfg)

        # ─── Otomatis daftarkan model lokal MOKO sebagai pekerja tetap ────────────────────
        # Ini memastikan tidak ada "saklar AI" — model lokal selalu ada di pool
        # sehingga sistem bisa offline-first kapanpun diperlukan.
        local_name = "moko-local-coder"
        if local_name not in self.clients:
            local_cfg = APIConfig(
                name=local_name,
                provider="local",
                model_name="moko-coder-1.5b",
                api_keys=["local-no-key"],  # Tidak digunakan oleh provider lokal
                timeout=120,               # Lebih longgar: inferensi CPU bisa lambat
            )
            self.configs.append(local_cfg)
            self.clients[local_name] = MokoAPIClient(local_cfg)
            logger.info(f"[WorkerPool] Model lokal '{local_name}' terdaftar otomatis di pool.")

        # ─── Otomatis daftarkan OmniRoute, 9Router, dan OpenCode ─────────────────────
        # 1. OmniRoute
        omni_api_base = os.environ.get("MOKO_OMNIROUTE_BASE") or os.environ.get("OMNIROUTE_API_BASE") or "http://localhost:20128/v1"
        omni_keys_env = os.environ.get("MOKO_OMNIROUTE_KEYS") or os.environ.get("OMNIROUTE_API_KEY")
        omni_keys = [k.strip() for k in omni_keys_env.split(",") if k.strip()] if omni_keys_env else ["omni-free-dummy-key"]
        
        omni_models = self._fetch_gateway_models(omni_api_base, omni_keys)
        if not omni_models:
            omni_models = [os.environ.get("MOKO_OMNIROUTE_MODEL") or "gpt-4o-mini"]
        
        for m_name in omni_models:
            cfg_name = f"omniroute-{m_name}"
            if cfg_name not in self.clients:
                omni_cfg = APIConfig(
                    name=cfg_name,
                    provider="omniroute",
                    model_name=m_name,
                    api_base=omni_api_base,
                    api_keys=omni_keys,
                    timeout=30,
                    enabled=True
                )
                self.configs.append(omni_cfg)
                self.clients[cfg_name] = MokoAPIClient(omni_cfg)
                logger.info(f"[WorkerPool] OmniRoute '{cfg_name}' terdaftar otomatis di pool.")

        # 2. 9Router
        nine_api_base = os.environ.get("MOKO_NINEROUTE_BASE") or os.environ.get("NINEROUTE_API_BASE") or "http://localhost:20130/v1"
        nine_keys_env = os.environ.get("MOKO_NINEROUTE_KEYS") or os.environ.get("NINEROUTE_API_KEY")
        nine_keys = [k.strip() for k in nine_keys_env.split(",") if k.strip()] if nine_keys_env else ["nine-free-dummy-key"]
        
        nine_models = self._fetch_gateway_models(nine_api_base, nine_keys)
        if not nine_models:
            nine_models = [os.environ.get("MOKO_NINEROUTE_MODEL") or "gpt-4o-mini"]
            
        for m_name in nine_models:
            cfg_name = f"ninerouter-{m_name}"
            if cfg_name not in self.clients:
                nine_cfg = APIConfig(
                    name=cfg_name,
                    provider="ninerouter",
                    model_name=m_name,
                    api_base=nine_api_base,
                    api_keys=nine_keys,
                    timeout=30,
                    enabled=True
                )
                self.configs.append(nine_cfg)
                self.clients[cfg_name] = MokoAPIClient(nine_cfg)
                logger.info(f"[WorkerPool] 9Router '{cfg_name}' terdaftar otomatis di pool.")

        # 3. OpenCode
        opencode_api_base = os.environ.get("MOKO_OPENCODE_BASE") or os.environ.get("OPENCODE_API_BASE") or "http://localhost:4096/v1"
        opencode_keys_env = os.environ.get("MOKO_OPENCODE_KEYS") or os.environ.get("OPENCODE_API_KEY")
        opencode_keys = [k.strip() for k in opencode_keys_env.split(",") if k.strip()] if opencode_keys_env else ["opencode-free-dummy-key"]
        
        opencode_models = self._fetch_gateway_models(opencode_api_base, opencode_keys)
        if not opencode_models:
            opencode_models = [os.environ.get("MOKO_OPENCODE_MODEL") or "opencode-default"]
            
        for m_name in opencode_models:
            cfg_name = f"opencode-{m_name}"
            if cfg_name not in self.clients:
                opencode_cfg = APIConfig(
                    name=cfg_name,
                    provider="opencode",
                    model_name=m_name,
                    api_base=opencode_api_base,
                    api_keys=opencode_keys,
                    timeout=30,
                    enabled=True
                )
                self.configs.append(opencode_cfg)
                self.clients[cfg_name] = MokoAPIClient(opencode_cfg)
                logger.info(f"[WorkerPool] OpenCode '{cfg_name}' terdaftar otomatis di pool.")

    def scan_workers(self) -> dict[str, bool]:
        """Memindai status koneksi seluruh pekerja secara paralel.
        
        Mengembalikan dictionary nama client -> boolean status keaktifan.
        """
        results = {}
        if not self.clients:
            logger.warning("Tidak ada pekerja terkonfigurasi untuk dipindai.")
            return results

        logger.info(f"Memulai pemindaian koneksi untuk {len(self.clients)} pekerja...")
        
        # Lakukan ping/test_connection paralel hanya untuk worker yang enabled
        enabled_clients = {name: client for name, client in self.clients.items() if client.config.enabled}
        if not enabled_clients:
            logger.warning("Tidak ada pekerja teraktifkan untuk dipindai.")
            self.active_clients = []
            return results

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_name = {
                executor.submit(client.test_connection): name
                for name, client in enabled_clients.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    is_ok = future.result()
                    results[name] = is_ok
                except Exception as e:
                    logger.error(f"Error memindai koneksi {name}: {e}")
                    results[name] = False
        
        # Mark disabled clients as False in results
        for name in self.clients:
            if name not in results:
                results[name] = False

        # Update active client list
        self.active_clients = [name for name, is_ok in results.items() if is_ok]
        logger.info(
            f"Pemindaian selesai. Pekerja aktif ({len(self.active_clients)}/{len(self.clients)}): "
            f"{', '.join(self.active_clients)}"
        )
        return results

    def get_mandor(self) -> MokoAPIClient | None:
        """
        Mengambil client yang berperan sebagai Guru/Mandor.

        Prioritas (sesuai riset 23_REVISI_MANDOR_API_MURID_LOKAL.md):
          0. Manual Selection: Jika ada worker aktif dengan is_mandor=True, pakai itu.
          1. External API (Gemini, OpenAI, DeepSeek, dll.) — mereka adalah GURU
             yang benar-benar mengerjakan kode dan menghasilkan data latih.
          2. Local model MOKO — fallback saat offline/quota habis.
        """
        active_list = self.get_active_workers()

        # 0. Manual Mandor Selection
        manual_mandor = [w for w in active_list if w.config.is_mandor]
        if manual_mandor:
            return manual_mandor[0]

        # Cari external API aktif (bukan provider 'local')
        external = [w for w in active_list if w.provider != "local"]
        if external:
            # 1. Prioritaskan Gemini (gratis, context window besar)
            gemini_workers = [w for w in external if w.provider == "gemini"]
            if gemini_workers:
                return gemini_workers[0]
            
            # 2. Prioritaskan free gateways: omniroute, ninerouter, opencode
            free_gateways = [w for w in external if w.provider in ("omniroute", "ninerouter", "opencode")]
            if free_gateways:
                # prioritaskan berdasarkan urutan: omniroute, lalu ninerouter, lalu opencode
                for gateway_provider in ("omniroute", "ninerouter", "opencode"):
                    matches = [w for w in free_gateways if w.provider == gateway_provider]
                    if matches:
                        return matches[0]
                return free_gateways[0]

            return external[0]

        # Fallback: model lokal jadi mandor saat semua API eksternal mati
        local = [w for w in active_list if w.provider == "local"]
        if local:
            logger.info(
                "[WorkerPool] Semua external API tidak aktif. "
                "Menggunakan model lokal sebagai Mandor (offline mode)."
            )
            return local[0]

        # Last resort: klien terdaftar pertama meskipun belum di-scan
        if self.clients:
            return next(iter(self.clients.values()))
        return None

    def get_pekerja_candidates(self) -> list[MokoAPIClient]:
        """
        Daftar pekerja yang tersedia untuk mengeksekusi kode (Worker role).
        Mengembalikan semua worker aktif kecuali Mandor,
        lalu Mandor sendiri jika tidak ada worker lain,
        lalu model lokal sebagai fallback terakhir.
        """
        active = self.get_active_workers()
        mandor = self.get_mandor()
        # Filter mandor keluar agar pekerja berbeda dari mandor (jika memungkinkan)
        non_mandor = [w for w in active if mandor is None or w.name != mandor.name]
        if non_mandor:
            return non_mandor
        # Jika cuma ada satu worker (mandor itu sendiri), dia juga yang mengerjakan
        if active:
            return active
        return []

    def get_active_workers(self) -> list[MokoAPIClient]:
        """Mengembalikan daftar instansi client yang berstatus aktif/siap."""
        return [self.clients[name] for name in self.active_clients if name in self.clients]

    def get_worker_pool_by_role(self, role: str) -> list[MokoAPIClient]:
        """Mendapatkan daftar pekerja berdasarkan kriteria role/kebutuhan."""
        active = self.get_active_workers()
        # Saat ini kita perlakukan semua pekerja aktif sebagai pool siap pakai.
        return active

    # ── System Mode ────────────────────────────────────────────────────────────

    @staticmethod
    def get_system_mode() -> str:
        """
        Membaca mode sistem dari moko_config/moko_settings.json.
        Returns:
            "agent"    — Multi-Agent collaboration mode (default)
            "rotation" — API Rotation mode (auto-switch on rate-limit)
        """
        settings_candidates = [
            Path("../moko_config/moko_settings.json"),
            Path("moko_config/moko_settings.json"),
            Path(__file__).parent.parent.parent / "moko_config" / "moko_settings.json",
        ]
        for path in settings_candidates:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return data.get("system_mode", "agent")
                except Exception:
                    pass
        return "agent"  # default

    def call_with_rotation(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """
        API Rotation Mode: Panggil API aktif pertama. Jika gagal (429 / timeout / error),
        langsung retry ke API berikutnya dalam pool (Opsi A: retry pada request yang sama).
        Jika semua API eksternal habis, fallback ke model lokal.

        Hanya digunakan saat system_mode == "rotation".
        """
        active = self.get_active_workers()
        external = [w for w in active if w.provider != "local"]
        local_workers = [w for w in active if w.provider == "local"]

        rotation_targets = external if external else local_workers

        last_error = None
        for worker in rotation_targets:
            try:
                logger.info(f"[Rotation] Mencoba API: {worker.name} ({worker.provider})")
                result = worker.generate_text(prompt, system_prompt, max_tokens=max_tokens)
                logger.info(f"[Rotation] Berhasil via: {worker.name}")
                return result
            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = any(kw in err_str for kw in [
                    "429", "rate_limit", "rate limit", "quota", "ratelimit",
                    "resourceexhausted", "resource_exhausted", "too many requests",
                    "timeout", "timed out", "timedout",
                ])
                if is_rate_limit:
                    logger.warning(f"[Rotation] {worker.name} kena rate-limit/timeout. Beralih ke API berikutnya...")
                    last_error = exc
                    continue
                else:
                    # Non-rate-limit error — still rotate but log as error
                    logger.error(f"[Rotation] {worker.name} error: {exc}. Mencoba API berikutnya...")
                    last_error = exc
                    continue

        # All external APIs failed — try local fallback
        if local_workers and rotation_targets != local_workers:
            logger.warning("[Rotation] Semua external API gagal. Fallback ke model lokal.")
            try:
                return local_workers[0].generate_text(prompt, system_prompt, max_tokens=max_tokens)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"[Rotation] Semua API gagal. Error terakhir: {last_error}"
        )
