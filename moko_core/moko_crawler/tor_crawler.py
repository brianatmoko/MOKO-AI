"""
moko_crawler/tor_crawler.py
============================
Core crawler engine:
- TorSession: HTTP session via SOCKS5 Tor proxy
- TorCircuitRotator: Ganti circuit Tor via ControlPort
- PageFetcher: Fetch + parse satu halaman
"""

import random
import time
import logging
import socket
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse

from .config import TorConfig, CrawlerConfig, USER_AGENTS, TOR_CONFIG, CRAWLER_CONFIG
from .parser import get_parser

log = logging.getLogger("moko_crawler.tor_crawler")


# ── Cek dependencies ──────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import socks  # PySocks
    except ImportError:
        missing.append("PySocks")
    if missing:
        raise ImportError(
            f"Dependensi kurang: {', '.join(missing)}. "
            f"Jalankan: pip install requests[socks] PySocks"
        )

_check_deps()


# ── Tor Circuit Rotator ───────────────────────────────────────────────────────
class TorCircuitRotator:
    """
    Gunakan ControlPort Tor untuk request new identity (IP baru).
    Butuh Tor ControlPort aktif dan password (jika dikonfigurasi).
    """

    def __init__(self, cfg: TorConfig = TOR_CONFIG):
        self._cfg = cfg
        self._request_count = 0
        self._stem_available = False
        self._controller = None
        self._lock = __import__("threading").Lock()

        try:
            import stem
            import stem.control
            self._stem_available = True
            log.info("stem library tersedia - IP rotation aktif")
        except ImportError:
            log.warning("stem tidak terinstall - IP rotation dinonaktifkan. "
                        "Install: pip install stem")

    def _connect_controller(self):
        if not self._stem_available:
            return False
        try:
            from stem.control import Controller
            from stem import Signal

            self._controller = Controller.from_port(
                address=self._cfg.control_host,
                port=self._cfg.control_port
            )
            if self._cfg.control_password:
                self._controller.authenticate(self._cfg.control_password)
            else:
                self._controller.authenticate()
            log.info("Terhubung ke Tor ControlPort")
            return True
        except Exception as e:
            log.warning(f"Tidak bisa connect ke Tor ControlPort: {e}")
            return False

    def record_request(self):
        """Panggil setiap selesai request. Auto-rotate jika perlu."""
        self._request_count += 1
        if self._request_count % self._cfg.new_circuit_every == 0:
            self.rotate()

    def rotate(self) -> bool:
        """Paksa Tor untuk ganti circuit (new IP)."""
        if not self._stem_available:
            return False
        with self._lock:
            try:
                from stem.control import Controller
                from stem import Signal

                if self._controller is None:
                    if not self._connect_controller():
                        return False

                self._controller.signal(Signal.NEWNYM)
                log.info(f"🔄 Tor circuit dirotasi (request #{self._request_count})")
                time.sleep(2)  # Tor butuh ~2 detik untuk circuit baru
                return True
            except Exception as e:
                log.error(f"Gagal rotate circuit: {e}")
                self._controller = None  # Reset untuk reconnect
                return False


# ── Tor HTTP Session ──────────────────────────────────────────────────────────
class TorSession:
    """
    HTTP session yang merutekan semua traffic melalui Tor SOCKS5 proxy.
    Satu instance per worker thread.
    """

    def __init__(self, cfg: TorConfig = TOR_CONFIG, worker_id: int = 0):
        import requests
        self._cfg = cfg
        self._worker_id = worker_id
        self._session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        proxy_url = f"socks5h://{self._cfg.socks_host}:{self._cfg.socks_port}"
        self._session.proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }
        self._session.verify = False   # .onion tidak punya valid SSL cert
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        ua = random.choice(USER_AGENTS)
        self._session.headers.update({
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "DNT": "1",
        })

    def get(self, url: str, **kwargs) -> Optional[object]:
        """HTTP GET dengan timeout dan retry. Return response atau None."""
        self._rotate_user_agent()
        try:
            resp = self._session.get(
                url,
                timeout=(self._cfg.connect_timeout, self._cfg.read_timeout),
                allow_redirects=True,
                **kwargs
            )
            return resp
        except Exception as e:
            log.debug(f"[Worker {self._worker_id}] GET failed ({url}): {type(e).__name__}: {e}")
            return None

    def close(self):
        self._session.close()


# ── Page Fetcher ──────────────────────────────────────────────────────────────
class PageFetcher:
    """
    Fetch + parse satu halaman .onion.
    Digunakan oleh setiap worker.
    """

    def __init__(self, tor_cfg: TorConfig = TOR_CONFIG,
                 crawl_cfg: CrawlerConfig = CRAWLER_CONFIG,
                 worker_id: int = 0):
        self._tcfg = tor_cfg
        self._ccfg = crawl_cfg
        self._worker_id = worker_id
        self._session = TorSession(tor_cfg, worker_id)
        self._parser  = get_parser(detect_lang=True)

    def fetch(self, url: str, depth: int) -> Optional[Dict]:
        """
        Fetch + parse URL.
        Return dict hasil parse, atau None jika gagal.
        """
        attempt = 0
        backoff = self._ccfg.retry_backoff

        while attempt < self._ccfg.retry_attempts:
            attempt += 1
            log.debug(f"[W{self._worker_id}] Fetch attempt {attempt}: {url}")

            resp = self._session.get(url, stream=False)
            if resp is None:
                log.debug(f"[W{self._worker_id}] No response ({url}), backoff {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            # Status check
            if resp.status_code == 429:
                log.warning(f"[W{self._worker_id}] Rate limited ({url}), sleeping 30s")
                time.sleep(30)
                continue

            if resp.status_code >= 500:
                log.warning(f"[W{self._worker_id}] Server error {resp.status_code} ({url})")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if resp.status_code >= 400:
                # 4xx = tidak ada halaman, tidak usah retry
                return {
                    "url": url, "depth": depth,
                    "status_code": resp.status_code,
                    "title": "", "description": "", "language": "unknown",
                    "text_content": "", "html_content": None,
                    "links": [], "onion_links": [],
                    "content_size": 0,
                    "success": False,
                    "error": f"HTTP {resp.status_code}",
                }

            # Cek content-type
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                log.debug(f"[W{self._worker_id}] Skip non-HTML ({content_type}): {url}")
                return None

            # Baca konten (dengan batas ukuran)
            try:
                # Gunakan iter_content untuk batasi ukuran
                chunks = []
                total = 0
                for chunk in resp.iter_content(chunk_size=8192, decode_unicode=False):
                    total += len(chunk)
                    chunks.append(chunk)
                    if total >= self._ccfg.max_content_size:
                        log.debug(f"[W{self._worker_id}] Content too large, truncating: {url}")
                        break
                raw_bytes = b"".join(chunks)
            except Exception:
                raw_bytes = resp.content

            # Decode dengan encoding detection
            encoding = resp.encoding or "utf-8"
            try:
                html = raw_bytes.decode(encoding, errors="replace")
            except (LookupError, UnicodeDecodeError):
                html = raw_bytes.decode("utf-8", errors="replace")

            content_size = len(raw_bytes)

            # Parse
            parsed = self._parser.parse(html, url)
            parsed["url"]          = url
            parsed["depth"]        = depth
            parsed["status_code"]  = resp.status_code
            parsed["content_size"] = content_size
            parsed["html_content"] = html if self._ccfg.save_html else None
            parsed["success"]      = True
            parsed["error"]        = None

            log.info(
                f"[W{self._worker_id}] ✓ {resp.status_code} | "
                f"{len(parsed['onion_links'])} links | "
                f"{content_size//1024}KB | {url}"
            )
            return parsed

        # Semua retry habis
        return {
            "url": url, "depth": depth,
            "status_code": 0,
            "title": "", "description": "", "language": "unknown",
            "text_content": "", "html_content": None,
            "links": [], "onion_links": [],
            "content_size": 0,
            "success": False,
            "error": "Max retries exceeded",
        }

    def close(self):
        self._session.close()


# ── Tor Connectivity Test ─────────────────────────────────────────────────────
def test_tor_connection(cfg: TorConfig = TOR_CONFIG) -> bool:
    """Test apakah Tor SOCKS5 proxy bisa diakses."""
    try:
        sock = socket.create_connection(
            (cfg.socks_host, cfg.socks_port), timeout=10
        )
        sock.close()
        log.info(f"✅ Tor SOCKS5 proxy dapat dijangkau di {cfg.socks_host}:{cfg.socks_port}")

        # Coba fetch check.torproject.org
        import requests
        proxy = f"socks5h://{cfg.socks_host}:{cfg.socks_port}"
        try:
            r = requests.get(
                "https://check.torproject.org/api/ip",
                proxies={"https": proxy, "http": proxy},
                timeout=30, verify=False
            )
            if r.status_code == 200:
                data = r.json()
                is_tor = data.get("IsTor", False)
                ip     = data.get("IP", "unknown")
                log.info(f"✅ IP via Tor: {ip} | IsTor: {is_tor}")
                return is_tor
        except Exception as e:
            log.warning(f"Tidak bisa verifikasi via check.torproject.org: {e}")
            return True  # SOCKS accessible, assume OK

    except Exception as e:
        log.error(f"❌ Tor SOCKS5 tidak bisa dijangkau: {e}")
        return False
