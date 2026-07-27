"""
moko_crawler/scheduler.py
==========================
Multi-threaded crawler scheduler dengan worker pool, rate limiting,
checkpoint auto-save, dan live progress stats.
"""

import time
import threading
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable
from queue import Queue, Empty

from .config import CrawlerConfig, TorConfig, TOR_CONFIG, CRAWLER_CONFIG
from .storage import CrawlDatabase
from .url_manager import URLFrontier, DomainRateLimiter, extract_domain
from .tor_crawler import PageFetcher, TorCircuitRotator

log = logging.getLogger("moko_crawler.scheduler")


# ── Live Stats ────────────────────────────────────────────────────────────────
class CrawlerStats:
    def __init__(self):
        self._lock       = threading.Lock()
        self.pages_ok    = 0
        self.pages_fail  = 0
        self.links_found = 0
        self.start_time  = time.monotonic()

    def record_success(self, link_count: int):
        with self._lock:
            self.pages_ok    += 1
            self.links_found += link_count

    def record_failure(self):
        with self._lock:
            self.pages_fail += 1

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def pages_per_sec(self) -> float:
        elapsed = self.elapsed
        return (self.pages_ok + self.pages_fail) / elapsed if elapsed > 0 else 0

    def summary(self) -> str:
        return (
            f"✓{self.pages_ok} ✗{self.pages_fail} "
            f"🔗{self.links_found} "
            f"⚡{self.pages_per_sec:.2f}p/s "
            f"⏱{self.elapsed:.0f}s"
        )


# ── Worker ────────────────────────────────────────────────────────────────────
class CrawlerWorker:
    """
    Satu worker thread yang memproses URL dari queue.
    """

    def __init__(self, worker_id: int, db: CrawlDatabase,
                 frontier: URLFrontier, rate_limiter: DomainRateLimiter,
                 stats: CrawlerStats, rotator: TorCircuitRotator,
                 tor_cfg: TorConfig, crawl_cfg: CrawlerConfig,
                 stop_event: threading.Event):
        self._id          = worker_id
        self._db          = db
        self._frontier    = frontier
        self._rate_limiter = rate_limiter
        self._stats       = stats
        self._rotator     = rotator
        self._stop_event  = stop_event
        self._fetcher     = PageFetcher(tor_cfg, crawl_cfg, worker_id)
        self._cfg         = crawl_cfg

    def process_item(self, item: dict):
        """Proses satu URL dari queue."""
        url    = item["url"]
        domain = item["domain"]
        depth  = item["depth"]

        # Rate limit per domain
        self._rate_limiter.wait(domain)

        # Cek stop
        if self._stop_event.is_set():
            self._frontier.mark_failed(url)
            return

        # Fetch
        result = self._fetcher.fetch(url, depth)
        self._rotator.record_request()

        if result is None:
            self._frontier.mark_failed(url)
            self._stats.record_failure()
            return

        if result.get("success"):
            # Simpan halaman
            self._db.save_page(
                url           = url,
                domain        = domain,
                depth         = depth,
                status_code   = result["status_code"],
                title         = result.get("title", ""),
                description   = result.get("description", ""),
                language      = result.get("language", "unknown"),
                text_content  = result.get("text_content", ""),
                html_content  = result.get("html_content"),
                content_size  = result.get("content_size", 0),
                outlink_count = len(result.get("onion_links", [])),
            )

            # Update domain stats
            self._db.upsert_domain(domain, success=True, title=result.get("title",""))

            # Simpan links
            onion_links = result.get("onion_links", [])
            if onion_links:
                self._db.save_links(url, [(l[0], l[1]) for l in onion_links])

            # Enqueue link baru yang ditemukan
            if depth < self._cfg.max_depth:
                new_urls = [(l[0], depth + 1) for l in onion_links]
                added = self._frontier.add_discovered(new_urls)
                log.debug(f"[W{self._id}] +{added} URLs baru dari {url}")

            self._frontier.mark_done(url)
            self._stats.record_success(len(onion_links))

        else:
            # Simpan error
            err = result.get("error", "Unknown")
            self._db.save_error(url, domain, "fetch_error", err)
            self._db.upsert_domain(domain, success=False)
            self._frontier.mark_failed(url)
            self._stats.record_failure()

    def close(self):
        self._fetcher.close()


# ── Scheduler / Orchestrator ──────────────────────────────────────────────────
class CrawlerScheduler:
    """
    Orchestrator utama crawler:
    - Feed URL dari DB queue ke worker pool
    - Handle stop/resume
    - Print progress
    - Auto checkpoint
    """

    def __init__(self, db: CrawlDatabase, frontier: URLFrontier,
                 crawl_cfg: CrawlerConfig = CRAWLER_CONFIG,
                 tor_cfg: TorConfig = TOR_CONFIG,
                 on_checkpoint: Optional[Callable] = None):
        self._db           = db
        self._frontier     = frontier
        self._cfg          = crawl_cfg
        self._tor_cfg      = tor_cfg
        self._on_checkpoint = on_checkpoint
        self._stats        = CrawlerStats()
        self._rate_limiter = DomainRateLimiter(delay=crawl_cfg.rate_limit_delay)
        self._rotator      = TorCircuitRotator(tor_cfg)
        self._stop_event   = threading.Event()
        self._workers: List[CrawlerWorker] = []

        # Handle SIGINT untuk graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        log.warning(f"\n⚠️  Signal {sig} diterima — menghentikan crawler secara graceful...")
        self._stop_event.set()

    def _create_workers(self, num: int) -> List[CrawlerWorker]:
        workers = []
        for i in range(num):
            w = CrawlerWorker(
                worker_id    = i,
                db           = self._db,
                frontier     = self._frontier,
                rate_limiter = self._rate_limiter,
                stats        = self._stats,
                rotator      = self._rotator,
                tor_cfg      = self._tor_cfg,
                crawl_cfg    = self._cfg,
                stop_event   = self._stop_event,
            )
            workers.append(w)
        return workers

    def _print_progress(self):
        """Print progress ke console setiap 10 detik."""
        while not self._stop_event.is_set():
            db_stats = self._db.get_stats()
            queue_stats = db_stats.get("queue", {})
            pending = queue_stats.get("pending", 0) + queue_stats.get("processing", 0)
            log.info(
                f"📊 {self._stats.summary()} | "
                f"DB: {db_stats['total_pages']} pages, {db_stats['total_domains']} domains | "
                f"Queue pending: {pending}"
            )
            time.sleep(10)

    def _checkpoint(self, total_crawled: int):
        """Simpan state checkpoint."""
        self._db.save_state("last_checkpoint", {
            "total_crawled": total_crawled,
            "timestamp": time.time(),
            "stats": {
                "pages_ok": self._stats.pages_ok,
                "pages_fail": self._stats.pages_fail,
                "links_found": self._stats.links_found,
            }
        })
        log.info(f"💾 Checkpoint disimpan ({total_crawled} halaman)")
        if self._on_checkpoint:
            self._on_checkpoint(total_crawled)

    def run(self):
        """
        Jalankan crawler loop utama.
        Fetch URL dari DB queue, distribusikan ke worker pool.
        """
        log.info(f"🚀 Memulai crawler dengan {self._cfg.num_workers} workers...")
        log.info(f"   Max depth: {self._cfg.max_depth}")
        log.info(f"   Max pages: {self._cfg.max_total_pages}")
        log.info(f"   Rate limit: {self._cfg.rate_limit_delay}s/domain")

        workers = self._create_workers(self._cfg.num_workers)
        self._workers = workers

        # Progress printer thread
        progress_thread = threading.Thread(target=self._print_progress, daemon=True)
        progress_thread.start()

        total_crawled = 0
        last_checkpoint = 0
        empty_cycles = 0

        with ThreadPoolExecutor(max_workers=self._cfg.num_workers) as executor:
            active_futures = {}

            while not self._stop_event.is_set():
                # Cek total limit
                if total_crawled >= self._cfg.max_total_pages:
                    log.info(f"✅ Batas max_total_pages ({self._cfg.max_total_pages}) tercapai")
                    break

                # Isi batch dari DB queue
                items = self._frontier.get_batch(size=self._cfg.num_workers * 3)

                if not items:
                    # Queue kosong
                    if not active_futures:
                        empty_cycles += 1
                        if empty_cycles >= 5:
                            log.info("📭 Queue kosong dan tidak ada worker aktif — selesai.")
                            break
                    time.sleep(3)
                    continue

                empty_cycles = 0

                # Submit ke worker pool
                for item in items:
                    # Pilih worker round-robin
                    worker = workers[total_crawled % len(workers)]
                    fut = executor.submit(worker.process_item, item)
                    active_futures[fut] = item["url"]
                    total_crawled += 1

                # Collect completed futures
                done_futs = [f for f in list(active_futures.keys()) if f.done()]
                for fut in done_futs:
                    del active_futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        log.error(f"Worker exception: {e}")

                # Checkpoint
                if total_crawled - last_checkpoint >= self._cfg.checkpoint_every:
                    self._checkpoint(total_crawled)
                    last_checkpoint = total_crawled

            # Wait semua yang pending
            log.info("⏳ Menunggu worker yang masih aktif...")
            for fut in as_completed(list(active_futures.keys()), timeout=60):
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"Worker exception: {e}")

        # Cleanup
        self._stop_event.set()
        for w in workers:
            w.close()

        # Final stats
        db_stats = self._db.get_stats()
        log.info("=" * 60)
        log.info(f"✅ CRAWLING SELESAI")
        log.info(f"   Total pages   : {db_stats['total_pages']}")
        log.info(f"   Total domains : {db_stats['total_domains']}")
        log.info(f"   Total links   : {db_stats['total_links']}")
        log.info(f"   Total errors  : {db_stats['total_errors']}")
        log.info(f"   Waktu total   : {self._stats.elapsed:.0f}s")
        log.info(f"   Rate          : {self._stats.pages_per_sec:.2f} pages/sec")
        log.info("=" * 60)

        return db_stats
