"""
moko_crawler/config.py
======================
Konfigurasi global untuk sistem Tor Onion Deep Crawler.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ── Direktori root ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_BASE_DIR, "crawler_data")
LOG_DIR  = os.path.join(DATA_DIR, "logs")
DB_PATH  = os.path.join(DATA_DIR, "onion_crawler.db")
SEED_FILE = os.path.join(DATA_DIR, "seeds.txt")


# ── Konfigurasi Tor ─────────────────────────────────────────────────────────
@dataclass
class TorConfig:
    socks_host: str = "127.0.0.1"
    socks_port: int = 9050
    control_host: str = "127.0.0.1"
    control_port: int = 9051
    control_password: str = ""          # isi jika ControlPort punya password
    new_circuit_every: int = 50         # rotate circuit setiap N request
    connect_timeout: int = 30           # detik
    read_timeout: int = 60              # detik


# ── Konfigurasi Crawling ────────────────────────────────────────────────────
@dataclass
class CrawlerConfig:
    max_depth: int = 5                  # kedalaman maksimum per seed
    max_pages_per_domain: int = 500     # batas halaman per domain .onion
    max_total_pages: int = 50_000       # batas total halaman seluruh crawler
    num_workers: int = 10               # jumlah worker async concurrent
    rate_limit_delay: float = 2.0       # detik antar request per domain
    retry_attempts: int = 3             # maksimum retry jika gagal
    retry_backoff: float = 5.0          # detik awal backoff (exponential)
    checkpoint_every: int = 100         # simpan checkpoint setiap N halaman
    save_html: bool = False             # simpan full HTML (makan storage)
    follow_external_onion: bool = True  # ikuti link ke domain .onion lain
    respect_robots: bool = False        # robots.txt di onion jarang valid
    max_content_size: int = 5_000_000  # 5MB maks per halaman


# ── Konfigurasi Storage ─────────────────────────────────────────────────────
@dataclass
class StorageConfig:
    db_path: str = DB_PATH
    wal_mode: bool = True               # SQLite WAL mode (lebih cepat concurrent write)
    cache_size_kb: int = 65536          # 64MB SQLite cache
    page_size: int = 4096


# ── User-Agent rotation ──────────────────────────────────────────────────────
USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/109.0 Firefox/116.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/116.0",
    "Wget/1.21.3",
    "curl/7.88.1",
]

# ── Pola URL yang di-skip (hanya skema non-HTTP yang tidak bisa di-crawl) ────
# Semua URL .onion diizinkan — tidak ada filter konten atau tipe file.
# Hanya skip skema yang memang bukan HTTP dan tidak bisa di-fetch.
SKIP_URL_PATTERNS: List[str] = [
    r"^mailto:", r"^javascript:", r"^data:", r"^ftp:",
]

# ── Singleton instance default ───────────────────────────────────────────────
TOR_CONFIG     = TorConfig()
CRAWLER_CONFIG = CrawlerConfig()
STORAGE_CONFIG = StorageConfig()
