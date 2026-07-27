"""
moko_crawler/storage.py
========================
SQLite storage backend — thread-safe dengan WAL mode.
Menyimpan: pages, links, domain stats, crawl queue, errors.
"""

import sqlite3
import threading
import time
import json
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple

from .config import STORAGE_CONFIG, StorageConfig

log = logging.getLogger("moko_crawler.storage")


# ── Schema SQL ───────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT UNIQUE NOT NULL,
    domain        TEXT NOT NULL,
    depth         INTEGER DEFAULT 0,
    status_code   INTEGER,
    title         TEXT,
    description   TEXT,
    language      TEXT,
    text_content  TEXT,
    html_content  TEXT,
    content_size  INTEGER DEFAULT 0,
    outlink_count INTEGER DEFAULT 0,
    crawled_at    REAL,
    created_at    REAL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url  TEXT NOT NULL,
    target_url  TEXT NOT NULL,
    anchor_text TEXT,
    discovered_at REAL DEFAULT (unixepoch('now')),
    UNIQUE(source_url, target_url)
);

CREATE TABLE IF NOT EXISTS domains (
    domain         TEXT PRIMARY KEY,
    pages_crawled  INTEGER DEFAULT 0,
    pages_failed   INTEGER DEFAULT 0,
    first_seen     REAL DEFAULT (unixepoch('now')),
    last_crawled   REAL,
    is_alive       INTEGER DEFAULT 1,
    title          TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT UNIQUE NOT NULL,
    domain      TEXT NOT NULL,
    depth       INTEGER DEFAULT 0,
    priority    INTEGER DEFAULT 0,
    added_at    REAL DEFAULT (unixepoch('now')),
    status      TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    domain      TEXT NOT NULL,
    error_type  TEXT,
    error_msg   TEXT,
    attempt_no  INTEGER DEFAULT 1,
    occurred_at REAL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS crawler_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_domain ON pages(domain);
CREATE INDEX IF NOT EXISTS idx_pages_crawled ON pages(crawled_at);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_url);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_url);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_queue_domain ON queue(domain);
CREATE INDEX IF NOT EXISTS idx_errors_domain ON errors(domain);
"""


class CrawlDatabase:
    """Thread-safe SQLite database wrapper untuk crawler."""

    def __init__(self, cfg: StorageConfig = STORAGE_CONFIG):
        self.db_path = cfg.db_path
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    # ── Koneksi per-thread ──────────────────────────────────────────────────
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ── Init ─────────────────────────────────────────────────────────────────
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()
        log.info(f"Database initialized: {self.db_path}")

    # ── Pages ─────────────────────────────────────────────────────────────────
    def save_page(self, url: str, domain: str, depth: int,
                  status_code: int, title: str, description: str,
                  language: str, text_content: str,
                  html_content: Optional[str], content_size: int,
                  outlink_count: int) -> bool:
        """Simpan halaman yang sudah di-crawl. Return True jika INSERT, False jika UPDATE."""
        now = time.time()
        try:
            with self._cursor() as cur:
                cur.execute("""
                    INSERT INTO pages
                        (url, domain, depth, status_code, title, description,
                         language, text_content, html_content, content_size,
                         outlink_count, crawled_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(url) DO UPDATE SET
                        status_code   = excluded.status_code,
                        title         = excluded.title,
                        description   = excluded.description,
                        language      = excluded.language,
                        text_content  = excluded.text_content,
                        html_content  = excluded.html_content,
                        content_size  = excluded.content_size,
                        outlink_count = excluded.outlink_count,
                        crawled_at    = excluded.crawled_at
                """, (url, domain, depth, status_code, title, description,
                      language, text_content, html_content, content_size,
                      outlink_count, now))
                return cur.lastrowid is not None
        except Exception as e:
            log.error(f"save_page error ({url}): {e}")
            return False

    def is_page_crawled(self, url: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM pages WHERE url=? AND crawled_at IS NOT NULL", (url,))
            return cur.fetchone() is not None

    def get_page_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pages WHERE crawled_at IS NOT NULL")
            return cur.fetchone()[0]

    # ── Links ─────────────────────────────────────────────────────────────────
    def save_links(self, source_url: str, links: List[Tuple[str, str]]):
        """Simpan batch links (target_url, anchor_text)."""
        now = time.time()
        rows = [(source_url, t, a, now) for t, a in links]
        with self._cursor() as cur:
            cur.executemany("""
                INSERT OR IGNORE INTO links (source_url, target_url, anchor_text, discovered_at)
                VALUES (?,?,?,?)
            """, rows)

    # ── Domains ──────────────────────────────────────────────────────────────
    def upsert_domain(self, domain: str, success: bool = True, title: str = ""):
        now = time.time()
        with self._cursor() as cur:
            if success:
                cur.execute("""
                    INSERT INTO domains (domain, pages_crawled, last_crawled, title)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        pages_crawled = pages_crawled + 1,
                        last_crawled  = excluded.last_crawled,
                        title = COALESCE(NULLIF(excluded.title,''), domains.title)
                """, (domain, now, title))
            else:
                cur.execute("""
                    INSERT INTO domains (domain, pages_failed)
                    VALUES (?, 1)
                    ON CONFLICT(domain) DO UPDATE SET
                        pages_failed = pages_failed + 1
                """, (domain,))

    def get_domain_page_count(self, domain: str) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT pages_crawled FROM domains WHERE domain=?", (domain,))
            row = cur.fetchone()
            return row[0] if row else 0

    def get_all_domains(self) -> List[Dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM domains ORDER BY pages_crawled DESC")
            return [dict(r) for r in cur.fetchall()]

    # ── Queue ─────────────────────────────────────────────────────────────────
    def enqueue_url(self, url: str, domain: str, depth: int, priority: int = 0) -> bool:
        try:
            with self._cursor() as cur:
                cur.execute("""
                    INSERT OR IGNORE INTO queue (url, domain, depth, priority)
                    VALUES (?, ?, ?, ?)
                """, (url, domain, depth, priority))
                return cur.rowcount > 0
        except Exception:
            return False

    def enqueue_batch(self, items: List[Tuple[str, str, int]]):
        """Batch enqueue [(url, domain, depth), ...]."""
        rows = [(u, d, dep, 0) for u, d, dep in items]
        with self._cursor() as cur:
            cur.executemany("""
                INSERT OR IGNORE INTO queue (url, domain, depth, priority)
                VALUES (?,?,?,?)
            """, rows)

    def dequeue_batch(self, limit: int = 50) -> List[Dict]:
        """Ambil batch URL dari queue dengan status='pending'."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT id, url, domain, depth FROM queue
                WHERE status='pending'
                ORDER BY priority DESC, id ASC
                LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                ids = tuple(r["id"] for r in rows)
                placeholder = ",".join("?" * len(ids))
                cur.execute(
                    f"UPDATE queue SET status='processing' WHERE id IN ({placeholder})",
                    ids
                )
            return rows

    def mark_queue_done(self, url: str):
        with self._cursor() as cur:
            cur.execute("UPDATE queue SET status='done' WHERE url=?", (url,))

    def mark_queue_failed(self, url: str):
        with self._cursor() as cur:
            cur.execute("UPDATE queue SET status='failed' WHERE url=?", (url,))

    def get_queue_size(self) -> Dict[str, int]:
        with self._cursor() as cur:
            cur.execute("SELECT status, COUNT(*) as cnt FROM queue GROUP BY status")
            return {r["status"]: r["cnt"] for r in cur.fetchall()}

    def is_url_queued_or_done(self, url: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM queue WHERE url=?", (url,))
            return cur.fetchone() is not None

    # ── Errors ────────────────────────────────────────────────────────────────
    def save_error(self, url: str, domain: str, error_type: str,
                   error_msg: str, attempt_no: int = 1):
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO errors (url, domain, error_type, error_msg, attempt_no)
                VALUES (?,?,?,?,?)
            """, (url, domain, error_type, str(error_msg)[:1000], attempt_no))

    # ── State ─────────────────────────────────────────────────────────────────
    def save_state(self, key: str, value: Any):
        val = json.dumps(value) if not isinstance(value, str) else value
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO crawler_state (key, value) VALUES (?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, val))

    def load_state(self, key: str, default=None) -> Any:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM crawler_state WHERE key=?", (key,))
            row = cur.fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                return row[0]

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pages WHERE crawled_at IS NOT NULL")
            total_pages = cur.fetchone()[0]

            cur.execute("SELECT COUNT(DISTINCT domain) FROM pages")
            total_domains = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM links")
            total_links = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM errors")
            total_errors = cur.fetchone()[0]

            queue_stats = self.get_queue_size()

        return {
            "total_pages": total_pages,
            "total_domains": total_domains,
            "total_links": total_links,
            "total_errors": total_errors,
            "queue": queue_stats,
        }

    # ── Export ────────────────────────────────────────────────────────────────
    def export_pages(self, output_path: str, fmt: str = "jsonl"):
        import os
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT url, domain, title, language, text_content, crawled_at FROM pages WHERE crawled_at IS NOT NULL")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            if fmt == "jsonl":
                for row in cur:
                    record = {
                        "url": row[0], "domain": row[1], "title": row[2],
                        "language": row[3], "text": row[4], "crawled_at": row[5]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            elif fmt == "csv":
                import csv
                writer = csv.writer(f)
                writer.writerow(["url","domain","title","language","text_content","crawled_at"])
                for row in cur:
                    writer.writerow(row)
                    count += 1

        cur.close()
        log.info(f"Exported {count} pages to {output_path}")
        return count
