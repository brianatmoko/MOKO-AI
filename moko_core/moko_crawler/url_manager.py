"""
moko_crawler/url_manager.py
============================
URL Frontier dengan bloom filter deduplication dan per-domain rate limiting.
"""

import re
import time
import hashlib
import threading
import logging
from collections import defaultdict
from typing import Optional, Set, List, Tuple
from urllib.parse import urlparse, urljoin, urldefrag

from .config import SKIP_URL_PATTERNS, CRAWLER_CONFIG

log = logging.getLogger("moko_crawler.url_manager")


# ── Bloom Filter sederhana (tidak butuh library external) ───────────────────
class SimpleBloomFilter:
    """Bloom filter ringan berbasis bitarray + SHA256."""

    def __init__(self, capacity: int = 5_000_000, error_rate: float = 0.01):
        import math
        self.capacity   = capacity
        self.error_rate = error_rate
        n, p = capacity, error_rate
        m = -int((n * math.log(p)) / (math.log(2) ** 2))
        k = max(1, int((m / n) * math.log(2)))
        self._size   = m
        self._k      = k
        self._bits   = bytearray(math.ceil(m / 8))
        self._count  = 0
        self._lock   = threading.Lock()

    def _hashes(self, item: str) -> List[int]:
        result = []
        h1 = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        for i in range(self._k):
            result.append((h1 + i * h2) % self._size)
        return result

    def add(self, item: str):
        with self._lock:
            for idx in self._hashes(item):
                byte_idx = idx // 8
                bit_idx  = idx % 8
                self._bits[byte_idx] |= (1 << bit_idx)
            self._count += 1

    def __contains__(self, item: str) -> bool:
        for idx in self._hashes(item):
            byte_idx = idx // 8
            bit_idx  = idx % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    @property
    def count(self) -> int:
        return self._count


# ── URL Normalizer ───────────────────────────────────────────────────────────
def normalize_url(url: str) -> Optional[str]:
    """Normalisasi URL, return None jika tidak valid."""
    try:
        url, _ = urldefrag(url)          # hilangkan #fragment
        url = url.strip()
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        host = parsed.netloc.lower()
        if not host:
            return None
        # pastikan host adalah .onion atau subdomain .onion
        if not (host.endswith(".onion") or ".onion:" in host):
            return None
        path = parsed.path or "/"
        query = parsed.query
        # susun ulang URL yang bersih
        norm = f"{parsed.scheme}://{host}{path}"
        if query:
            norm += f"?{query}"
        return norm
    except Exception:
        return None


def extract_domain(url: str) -> Optional[str]:
    try:
        return urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return None


def resolve_url(base: str, href: str) -> Optional[str]:
    """Resolve relative URL terhadap base URL."""
    try:
        joined = urljoin(base, href.strip())
        return normalize_url(joined)
    except Exception:
        return None


# ── Skip-Pattern Checker ─────────────────────────────────────────────────────
_SKIP_RE = re.compile("|".join(SKIP_URL_PATTERNS), re.IGNORECASE)

def should_skip(url: str) -> bool:
    return bool(_SKIP_RE.search(url))


# ── Per-Domain Rate Limiter ──────────────────────────────────────────────────
class DomainRateLimiter:
    """Token bucket per domain."""

    def __init__(self, delay: float = CRAWLER_CONFIG.rate_limit_delay):
        self._last_access: dict = defaultdict(float)
        self._delay = delay
        self._lock  = threading.Lock()

    def wait(self, domain: str):
        """Block sampai domain boleh diakses lagi."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_access[domain]
            wait_sec = max(0.0, self._delay - elapsed)
            self._last_access[domain] = now + wait_sec

        if wait_sec > 0:
            time.sleep(wait_sec)

    def set_delay(self, delay: float):
        self._delay = delay


# ── URL Frontier ─────────────────────────────────────────────────────────────
class URLFrontier:
    """
    Mengelola URL queue dengan:
    - Bloom filter deduplication (in-memory)
    - Sinkronisasi dengan database queue (persistent)
    - Per-domain stats
    """

    def __init__(self, db, cfg=CRAWLER_CONFIG):
        self._db    = db
        self._cfg   = cfg
        self._seen  = SimpleBloomFilter(capacity=10_000_000)
        self._lock  = threading.Lock()

        # Tandai URL yang sudah ada di DB sebagai "seen"
        self._preload_seen()

    def _preload_seen(self):
        """Load URL yang sudah ada di pages/queue ke bloom filter."""
        log.info("Preloading seen URLs into bloom filter...")
        conn = self._db._get_conn()
        cur  = conn.cursor()
        count = 0
        for table in ("pages", "queue"):
            cur.execute(f"SELECT url FROM {table}")
            for (url,) in cur:
                self._seen.add(url)
                count += 1
        cur.close()
        log.info(f"Preloaded {count} URLs into bloom filter (size: {self._seen.count})")

    def add_seed(self, url: str) -> bool:
        """Tambahkan seed URL ke frontier."""
        norm = normalize_url(url)
        if not norm:
            log.warning(f"Invalid seed URL: {url}")
            return False
        domain = extract_domain(norm)
        return self._enqueue(norm, domain, depth=0)

    def add_discovered(self, urls: List[Tuple[str, int]]) -> int:
        """
        Tambahkan URLs yang ditemukan saat crawling.
        urls: [(url, depth), ...]
        Returns jumlah URL baru yang berhasil di-enqueue.
        """
        added = 0
        batch = []
        for url, depth in urls:
            if depth > self._cfg.max_depth:
                continue
            norm = normalize_url(url)
            if not norm or should_skip(norm):
                continue
            domain = extract_domain(norm)
            if not domain:
                continue

            with self._lock:
                if norm in self._seen:
                    continue
                # Cek batas per domain
                if self._db.get_domain_page_count(domain) >= self._cfg.max_pages_per_domain:
                    continue
                self._seen.add(norm)

            batch.append((norm, domain, depth))
            added += 1

        if batch:
            self._db.enqueue_batch(batch)

        return added

    def _enqueue(self, url: str, domain: str, depth: int) -> bool:
        with self._lock:
            if url in self._seen:
                return False
            self._seen.add(url)
        result = self._db.enqueue_url(url, domain, depth)
        return result

    def get_batch(self, size: int = 50) -> List[dict]:
        return self._db.dequeue_batch(size)

    def mark_done(self, url: str):
        self._db.mark_queue_done(url)

    def mark_failed(self, url: str):
        self._db.mark_queue_failed(url)

    @property
    def seen_count(self) -> int:
        return self._seen.count
