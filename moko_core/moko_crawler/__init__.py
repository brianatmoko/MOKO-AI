"""
moko_crawler — Tor Onion Deep Crawler
======================================
Modul crawling jaringan Tor untuk MOKO_OS Project.

Komponen utama:
- config          : Konfigurasi global
- storage         : SQLite storage backend
- url_manager     : URL frontier & deduplication
- parser          : HTML parser & link extractor
- tor_crawler     : Tor session & page fetcher
- scheduler       : Worker pool & orchestrator
- search          : Orchestrator pencarian berantai (chain fallback)
- search_providers: Daftar provider search engine (Ahmia, DuckDuckGo, Custom...)
- cli             : Command-line interface

Penggunaan cepat:
    from moko_core.moko_crawler.storage import CrawlDatabase
    from moko_core.moko_crawler.url_manager import URLFrontier
    from moko_core.moko_crawler.scheduler import CrawlerScheduler

    db = CrawlDatabase()
    frontier = URLFrontier(db)
    frontier.add_seed("http://example.onion")

    scheduler = CrawlerScheduler(db, frontier)
    scheduler.run()
"""

__version__ = "1.1.0"
__author__  = "MOKO_OS Project"

from .config import (
    TorConfig, CrawlerConfig, StorageConfig,
    TOR_CONFIG, CRAWLER_CONFIG, STORAGE_CONFIG,
    DATA_DIR, DB_PATH, SEED_FILE, LOG_DIR,
)
from .storage import CrawlDatabase
from .url_manager import URLFrontier, DomainRateLimiter, normalize_url, extract_domain
from .parser import ContentParser, get_parser
from .tor_crawler import TorSession, PageFetcher, TorCircuitRotator, test_tor_connection
from .scheduler import CrawlerScheduler, CrawlerStats
from .search import search_onion
from .search_providers import (
    BaseSearchEngine,
    TorchSearchEngine,
    AhmiaSearchEngine,
    DuckDuckGoHtmlEngine,
    CustomSearchEngine,
)

__all__ = [
    # Config
    "TorConfig", "CrawlerConfig", "StorageConfig",
    "TOR_CONFIG", "CRAWLER_CONFIG", "STORAGE_CONFIG",
    "DATA_DIR", "DB_PATH", "SEED_FILE", "LOG_DIR",
    # Storage
    "CrawlDatabase",
    # URL
    "URLFrontier", "DomainRateLimiter", "normalize_url", "extract_domain",
    # Parser
    "ContentParser", "get_parser",
    # Tor
    "TorSession", "PageFetcher", "TorCircuitRotator", "test_tor_connection",
    # Scheduler
    "CrawlerScheduler", "CrawlerStats",
    # Search
    "search_onion",
    # Search Providers
    "BaseSearchEngine",
    "TorchSearchEngine",
    "AhmiaSearchEngine",
    "DuckDuckGoHtmlEngine",
    "CustomSearchEngine",
]

