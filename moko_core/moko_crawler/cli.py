"""
moko_crawler/cli.py
====================
Command-line interface untuk Tor Onion Deep Crawler.

Perintah:
  start    -- Mulai crawling baru
  resume   -- Lanjut dari checkpoint
  add-seed -- Tambah seed URL
  stats    -- Tampilkan statistik
  domains  -- Tampilkan daftar domain
  export   -- Export data ke file
  test-tor -- Test koneksi Tor
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import List

# ── Setup logging sebelum import modul lain ───────────────────────────────────
def _setup_logging(verbose: bool = False, log_file: str = None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=handlers)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("stem").setLevel(logging.WARNING)


log = logging.getLogger("moko_crawler.cli")


# ── Helper: Load seeds dari file ──────────────────────────────────────────────
def load_seeds(seed_file: str) -> List[str]:
    seeds = []
    if not os.path.exists(seed_file):
        log.warning(f"Seed file tidak ditemukan: {seed_file}")
        return seeds
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                seeds.append(line)
    log.info(f"Loaded {len(seeds)} seed URLs dari {seed_file}")
    return seeds


# ── Inisialisasi komponen ─────────────────────────────────────────────────────
def _init_components(args):
    from .config import (
        TorConfig, CrawlerConfig, StorageConfig,
        DATA_DIR, LOG_DIR, DB_PATH, SEED_FILE
    )
    from .storage import CrawlDatabase
    from .url_manager import URLFrontier
    from .tor_crawler import test_tor_connection

    # Override config dari args
    tor_cfg = TorConfig(
        socks_host      = getattr(args, "tor_host", "127.0.0.1"),
        socks_port      = getattr(args, "tor_port", 9050),
        control_port    = getattr(args, "control_port", 9051),
        control_password = getattr(args, "control_password", ""),
    )
    crawl_cfg = CrawlerConfig(
        max_depth            = getattr(args, "depth", 5),
        num_workers          = getattr(args, "workers", 10),
        max_pages_per_domain = getattr(args, "max_per_domain", 500),
        max_total_pages      = getattr(args, "max_total", 50_000),
        rate_limit_delay     = getattr(args, "delay", 2.0),
        save_html            = getattr(args, "save_html", False),
    )

    db_path = getattr(args, "db", None) or DB_PATH
    storage_cfg = StorageConfig(db_path=db_path)

    db = CrawlDatabase(storage_cfg)
    frontier = URLFrontier(db, crawl_cfg)

    return tor_cfg, crawl_cfg, storage_cfg, db, frontier


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: test-tor
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_test_tor(args):
    _setup_logging(args.verbose)
    from .config import TorConfig
    from .tor_crawler import test_tor_connection

    tor_cfg = TorConfig(
        socks_host = args.tor_host,
        socks_port = args.tor_port,
    )
    ok = test_tor_connection(tor_cfg)
    if ok:
        print("✅ Koneksi Tor berhasil — siap crawling!")
        sys.exit(0)
    else:
        print("❌ Koneksi Tor GAGAL. Pastikan 'tor' daemon berjalan.")
        print("   Install : sudo apt install tor")
        print("   Start   : sudo systemctl start tor  OR  tor &")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: start
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_start(args):
    from .config import SEED_FILE, LOG_DIR
    _setup_logging(args.verbose, os.path.join(LOG_DIR, "crawler.log"))

    tor_cfg, crawl_cfg, storage_cfg, db, frontier = _init_components(args)

    from .tor_crawler import test_tor_connection
    if not args.no_tor_check:
        log.info("Memeriksa koneksi Tor...")
        if not test_tor_connection(tor_cfg):
            log.error("❌ Tor tidak bisa dijangkau! Gunakan --no-tor-check untuk skip.")
            sys.exit(1)

    # Load seeds
    seed_file = args.seeds or SEED_FILE
    seeds = load_seeds(seed_file)

    # Tambahkan seed dari args langsung
    if args.seed_url:
        seeds.extend(args.seed_url)

    if not seeds:
        log.error("Tidak ada seed URL! Gunakan --seeds atau --seed-url")
        sys.exit(1)

    # Enqueue seeds ke frontier
    log.info(f"Menambahkan {len(seeds)} seed URLs...")
    added = 0
    for url in seeds:
        if frontier.add_seed(url):
            added += 1
    log.info(f"Berhasil enqueue {added} seeds ke queue")

    # Jalankan scheduler
    from .scheduler import CrawlerScheduler
    scheduler = CrawlerScheduler(db, frontier, crawl_cfg, tor_cfg)
    scheduler.run()


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: resume
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_resume(args):
    from .config import LOG_DIR
    _setup_logging(args.verbose, os.path.join(LOG_DIR, "crawler.log"))

    tor_cfg, crawl_cfg, storage_cfg, db, frontier = _init_components(args)

    # Cek apakah ada data yang bisa di-resume
    stats = db.get_stats()
    queue_stats = stats.get("queue", {})
    pending = queue_stats.get("pending", 0)
    # Reset 'processing' yang tertinggal ke 'pending'
    conn = db._get_conn()
    conn.execute("UPDATE queue SET status='pending' WHERE status='processing'")
    conn.commit()
    pending += queue_stats.get("processing", 0)

    log.info(f"📂 Melanjutkan crawling dari checkpoint...")
    log.info(f"   Pages crawled sebelumnya: {stats['total_pages']}")
    log.info(f"   Queue pending: {pending}")

    if pending == 0 and stats["total_pages"] == 0:
        log.error("Tidak ada data untuk dilanjutkan! Gunakan 'start' terlebih dahulu.")
        sys.exit(1)

    if not args.no_tor_check:
        from .tor_crawler import test_tor_connection
        if not test_tor_connection(tor_cfg):
            log.error("❌ Tor tidak bisa dijangkau!")
            sys.exit(1)

    from .scheduler import CrawlerScheduler
    scheduler = CrawlerScheduler(db, frontier, crawl_cfg, tor_cfg)
    scheduler.run()


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: add-seed
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_add_seed(args):
    _setup_logging(False)
    _, _, _, db, frontier = _init_components(args)

    urls = args.urls
    added = 0
    for url in urls:
        url = url.strip()
        if not url:
            continue
        if frontier.add_seed(url):
            log.info(f"✅ Ditambahkan: {url}")
            added += 1
        else:
            log.warning(f"⚠️  Skip (invalid/duplikat): {url}")

    print(f"\nBerhasil menambahkan {added} dari {len(urls)} seed URLs.")

    # Opsional: simpan ke seeds.txt
    if args.save:
        from .config import SEED_FILE
        seed_file = args.seeds or SEED_FILE
        with open(seed_file, "a", encoding="utf-8") as f:
            for url in urls:
                f.write(url.strip() + "\n")
        print(f"Tersimpan ke {seed_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: stats
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_stats(args):
    _setup_logging(False)
    _, _, _, db, _ = _init_components(args)

    stats = db.get_stats()
    checkpoint = db.load_state("last_checkpoint", {})

    print("\n" + "=" * 55)
    print("  🧅 MOKO Onion Crawler — Statistik")
    print("=" * 55)
    print(f"  Total halaman crawled : {stats['total_pages']:,}")
    print(f"  Total domain          : {stats['total_domains']:,}")
    print(f"  Total links ditemukan : {stats['total_links']:,}")
    print(f"  Total errors          : {stats['total_errors']:,}")
    print("")
    print("  Queue:")
    for status, count in stats.get("queue", {}).items():
        print(f"    {status:<12}: {count:,}")
    if checkpoint:
        ts = checkpoint.get("timestamp", 0)
        import datetime
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  Checkpoint terakhir   : {dt}")
        print(f"  Halaman saat itu      : {checkpoint.get('total_crawled', 0):,}")
    print("=" * 55 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: domains
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_domains(args):
    _setup_logging(False)
    _, _, _, db, _ = _init_components(args)

    domains = db.get_all_domains()
    limit = args.limit or 50

    print(f"\n{'DOMAIN':<50} {'PAGES':>8} {'FAILED':>8} {'LAST CRAWL'}")
    print("-" * 80)
    for d in domains[:limit]:
        import datetime
        lc = d.get("last_crawled")
        lc_str = datetime.datetime.fromtimestamp(lc).strftime("%m/%d %H:%M") if lc else "never"
        print(f"{d['domain']:<50} {d['pages_crawled']:>8,} {d['pages_failed']:>8,} {lc_str}")
    print(f"\nTotal: {len(domains)} domains\n")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: export
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_export(args):
    _setup_logging(False)
    _, _, _, db, _ = _init_components(args)

    from .config import DATA_DIR
    fmt = args.format or "jsonl"
    output = args.output or os.path.join(DATA_DIR, f"export.{fmt}")

    log.info(f"Export ke {output} (format: {fmt})...")
    count = db.export_pages(output, fmt=fmt)
    print(f"✅ Berhasil export {count:,} halaman ke: {output}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: search-start
# ═══════════════════════════════════════════════════════════════════════════════
def cmd_search_start(args):
    from .config import LOG_DIR
    _setup_logging(args.verbose, os.path.join(LOG_DIR, "crawler.log"))

    tor_cfg, crawl_cfg, storage_cfg, db, frontier = _init_components(args)

    from .tor_crawler import test_tor_connection
    if not args.no_tor_check:
        log.info("Memeriksa koneksi Tor...")
        if not test_tor_connection(tor_cfg):
            log.error("❌ Tor tidak bisa dijangkau! Gunakan --no-tor-check untuk skip.")
            sys.exit(1)

    from .search import search_onion
    
    # Lakukan pencarian onion
    query = args.query
    limit = args.limit
    use_onion_engine = args.use_onion_engine
    
    # Ambil list engines dari parameter
    engine_names = None
    if getattr(args, "engines", None):
        engine_names = [name.strip() for name in args.engines.split(",") if name.strip()]
    
    log.info(f"Memulai pencarian berantai untuk: '{query}'...")
    seeds = search_onion(
        query,
        limit=limit,
        use_onion_engine=use_onion_engine,
        tor_cfg=tor_cfg,
        engine_names=engine_names
    )
    
    if not seeds:
        log.error("❌ Tidak ada hasil pencarian atau gagal menghubungi search engine.")
        sys.exit(1)
        
    print("\n--- TOP SITES FOUND ---")
    for idx, url in enumerate(seeds, 1):
        print(f"  {idx}. {url}")
    print("-----------------------\n")
    
    # Enqueue seeds hasil pencarian
    log.info(f"Menambahkan {len(seeds)} hasil pencarian ke database queue...")
    added = 0
    for url in seeds:
        if frontier.add_seed(url):
            added += 1
    log.info(f"Berhasil enqueue {added} seeds hasil pencarian.")
    
    if added == 0:
        # Mungkin sudah pernah di-crawl, mari cek apakah queue bersih
        stats = db.get_stats()
        queue_stats = stats.get("queue", {})
        pending = queue_stats.get("pending", 0)
        if pending == 0:
            log.warning("Semua URL pencarian ini sudah selesai di-crawl sebelumnya. Gunakan 'stats' or 'export'.")
            sys.exit(0)

    # Jalankan scheduler untuk crawl deep situs-situs tersebut (dan link rujukan langsungnya)
    from .scheduler import CrawlerScheduler
    scheduler = CrawlerScheduler(db, frontier, crawl_cfg, tor_cfg)
    scheduler.run()


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moko_crawler",
        description="🧅 MOKO Tor Onion Deep Crawler — Sistem crawling jaringan Tor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Test koneksi Tor
  python -m moko_crawler test-tor

  # Mulai crawling dengan seed default
  python -m moko_crawler start

  # Cari 5 situs onion teratas untuk suatu query, lalu crawl situs tersebut dan outlinks-nya (depth 1)
  python -m moko_crawler search-start --query "hacker" --limit 5 --depth 1

  # Batasi urutan search engine tertentu (contoh: torch saja, atau torch lalu ahmia)
  python -m moko_crawler search-start --query "security" --engines torch,ahmia

  # Lanjut dari checkpoint
  python -m moko_crawler resume

  # Lihat statistik
  python -m moko_crawler stats

  # Export data
  python -m moko_crawler export --format jsonl --output /tmp/crawl.jsonl
        """
    )

    # Global args
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mode verbose (debug logging)")
    parser.add_argument("--db", default=None, metavar="PATH",
                        help="Path database SQLite")
    parser.add_argument("--tor-host", default="127.0.0.1",
                        help="Tor SOCKS5 host (default: 127.0.0.1)")
    parser.add_argument("--tor-port", type=int, default=9050,
                        help="Tor SOCKS5 port (default: 9050)")
    parser.add_argument("--control-port", type=int, default=9051,
                        help="Tor ControlPort (default: 9051)")
    parser.add_argument("--control-password", default="",
                        help="Password Tor ControlPort")

    sub = parser.add_subparsers(title="perintah", dest="command")
    sub.required = True

    # ── test-tor ─────────────────────────────────────────────────────────────
    p_test = sub.add_parser("test-tor", help="Test koneksi Tor")
    p_test.set_defaults(func=cmd_test_tor)

    # ── start ─────────────────────────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Mulai crawling baru")
    p_start.add_argument("--seeds", metavar="FILE",
                         help="Path file berisi seed URLs (satu per baris)")
    p_start.add_argument("--seed-url", nargs="*", default=[],
                         help="Seed URL langsung (bisa beberapa)")
    p_start.add_argument("--workers", type=int, default=10,
                         help="Jumlah worker concurrent (default: 10)")
    p_start.add_argument("--depth", type=int, default=5,
                         help="Kedalaman crawl maksimum (default: 5)")
    p_start.add_argument("--delay", type=float, default=2.0,
                         help="Delay antar request per domain detik (default: 2.0)")
    p_start.add_argument("--max-per-domain", type=int, default=500,
                         help="Max halaman per domain (default: 500)")
    p_start.add_argument("--max-total", type=int, default=50_000,
                         help="Max total halaman (default: 50000)")
    p_start.add_argument("--save-html", action="store_true",
                         help="Simpan full HTML (makan storage)")
    p_start.add_argument("--no-tor-check", action="store_true",
                         help="Skip test koneksi Tor di awal")
    p_start.set_defaults(func=cmd_start)

    # ── search-start ──────────────────────────────────────────────────────────
    p_search = sub.add_parser("search-start", help="Cari keyword onion di Torch/Ahmia/DDG lalu crawl situs terpilih")
    p_search.add_argument("--query", "-q", required=True,
                          help="Kata kunci pencarian onion")
    p_search.add_argument("--limit", type=int, default=5,
                          help="Batas maksimal situs teratas hasil pencarian (default: 5)")
    p_search.add_argument("--engines", default=None,
                          help="Daftar search engine dipisahkan koma (pilihan: torch, ahmia, duckduckgo, custom)")
    p_search.add_argument("--use-onion-engine", action="store_true",
                          help="Gunakan onion engine jika tersedia (seperti Ahmia onion)")
    p_search.add_argument("--workers", type=int, default=10,
                          help="Jumlah worker concurrent (default: 10)")
    p_search.add_argument("--depth", type=int, default=1,
                          help="Kedalaman crawl maksimum (default: 1, disarankan rendah agar fokus ke rujukan langsung)")
    p_search.add_argument("--delay", type=float, default=2.0,
                          help="Delay antar request per domain detik (default: 2.0)")
    p_search.add_argument("--max-per-domain", type=int, default=500,
                         help="Max halaman per domain (default: 500)")
    p_search.add_argument("--max-total", type=int, default=50_000,
                         help="Max total halaman (default: 50000)")
    p_search.add_argument("--save-html", action="store_true",
                         help="Simpan full HTML (makan storage)")
    p_search.add_argument("--no-tor-check", action="store_true",
                         help="Skip test koneksi Tor di awal")
    p_search.set_defaults(func=cmd_search_start)


    # ── resume ────────────────────────────────────────────────────────────────
    p_resume = sub.add_parser("resume", help="Lanjut crawling dari checkpoint")
    p_resume.add_argument("--workers", type=int, default=10)
    p_resume.add_argument("--depth", type=int, default=5)
    p_resume.add_argument("--delay", type=float, default=2.0)
    p_resume.add_argument("--max-per-domain", type=int, default=500)
    p_resume.add_argument("--max-total", type=int, default=50_000)
    p_resume.add_argument("--save-html", action="store_true")
    p_resume.add_argument("--no-tor-check", action="store_true")
    p_resume.set_defaults(func=cmd_resume)

    # ── add-seed ──────────────────────────────────────────────────────────────
    p_add = sub.add_parser("add-seed", help="Tambah seed URL ke queue")
    p_add.add_argument("urls", nargs="+", help="URL .onion yang ingin ditambahkan")
    p_add.add_argument("--seeds", metavar="FILE", help="File seeds.txt untuk disimpan")
    p_add.add_argument("--save", action="store_true",
                       help="Simpan URL ke file seeds.txt")
    p_add.set_defaults(func=cmd_add_seed)

    # ── stats ─────────────────────────────────────────────────────────────────
    p_stats = sub.add_parser("stats", help="Tampilkan statistik crawling")
    p_stats.set_defaults(func=cmd_stats)

    # ── domains ───────────────────────────────────────────────────────────────
    p_domains = sub.add_parser("domains", help="Tampilkan daftar domain yang ditemukan")
    p_domains.add_argument("--limit", type=int, default=50,
                           help="Jumlah domain yang ditampilkan (default: 50)")
    p_domains.set_defaults(func=cmd_domains)

    # ── export ────────────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export data ke file")
    p_export.add_argument("--format", choices=["jsonl", "csv"], default="jsonl",
                          help="Format output (default: jsonl)")
    p_export.add_argument("--output", metavar="FILE",
                          help="Path file output")
    p_export.set_defaults(func=cmd_export)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

