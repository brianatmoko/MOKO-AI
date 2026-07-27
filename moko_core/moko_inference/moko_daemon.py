"""
MOKO Daemon — Sovereign Inference Server Launcher
==================================================
Phase 18: Memastikan MOKO Server Sovereign selalu hidup secara persisten
sebagai daemon background, 100% MOKO - NO llama-server dependency.

USAGE:
  # Nyalakan daemon (idempotent — aman dipanggil berulang):
  python moko_daemon.py start

  # Cek status:
  python moko_daemon.py status

  # Matikan daemon:
  python moko_daemon.py stop

  # Mulai ulang:
  python moko_daemon.py restart

Untuk autostart saat login, tambahkan di ~/.bashrc:
  # Auto-start MOKO inference daemon
  python /path/to/moko_daemon.py start --quiet
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

# Tambahkan project path ke sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
MOKO_CORE   = PROJECT_ROOT / "moko_core"
sys.path.insert(0, str(MOKO_CORE))

from moko_config import settings

MOKO_PID_FILE  = "/tmp/moko_server.pid"
RAG_PID_FILE   = "/tmp/moko_rag_server.pid"
DAEMON_LOG     = "/tmp/moko_daemon.log"


def _read_pid(pid_file: str):
    try:
        if os.path.exists(pid_file):
            with open(pid_file) as f:
                return int(f.read().strip())
    except Exception:
        pass
    return None


def _is_running(pid) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_http(port: int) -> str:
    """Returns 'ok', 'loading', atau 'offline'."""
    try:
        import requests
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        if r.status_code == 200:
            return "ok"
        elif r.status_code == 503:
            data = r.json()
            if "error" in data and "Loading model" in data["error"].get("message", ""):
                return "loading"
    except Exception:
        pass
    return "offline"


def cmd_status(quiet=False):
    """Tampilkan status server."""
    moko_pid    = _read_pid(MOKO_PID_FILE)
    moko_http   = _check_http(settings.MOKO_LLM_PORT)
    moko_alive  = _is_running(moko_pid)

    # Phase 3.4: server RAG khusus (200MB, port 11437)
    rag_port    = getattr(settings, "MOKO_RAG_PORT", 11437)
    rag_pid     = _read_pid(RAG_PID_FILE)
    rag_http    = _check_http(rag_port)
    rag_alive   = _is_running(rag_pid)

    if not quiet:
        print("=" * 50)
        print("  MOKO Daemon — Server Status (Sovereign)")
        print("=" * 50)
        moko_icon = "✅" if moko_http == "ok" else ("⏳" if moko_http == "loading" else "🔴")
        print(f"  {moko_icon} MOKO Server (:{settings.MOKO_LLM_PORT}) PID={moko_pid}  proc={'running' if moko_alive else 'dead'}  http={moko_http.upper()}")
        rag_icon = "✅" if rag_http == "ok" else ("⏳" if rag_http == "loading" else "🔴")
        print(f"  {rag_icon} RAG Server  (:{rag_port}) PID={rag_pid}  proc={'running' if rag_alive else 'dead'}  http={rag_http.upper()}")
        print("=" * 50)

    return {
        "moko": {"pid": moko_pid, "proc": moko_alive, "http": moko_http},
        "rag":  {"pid": rag_pid,  "proc": rag_alive,  "http": rag_http},
    }


def cmd_start(quiet=False):
    """Nyalakan server jika belum aktif (idempotent)."""
    status = cmd_status(quiet=True)
    moko_http = status["moko"]["http"]

    if moko_http in ("ok", "loading"):
        if not quiet:
            print("✅ MOKO Server sudah aktif. Tidak ada yang perlu dilakukan.")
        return

    if not quiet:
        print("🚀 Menyalakan MOKO Server Sovereign daemon...")

    # Panggil start_servers dari MokoLocalInferenceServer
    from moko_inference.server_manager import MokoLocalInferenceServer
    MokoLocalInferenceServer.start_servers()

    if not quiet:
        # Tunggu sedikit lalu tampilkan status akhir
        time.sleep(2)
        cmd_status()


def cmd_stop(quiet=False):
    """Matikan server."""
    if not quiet:
        print("🛑 Mematikan MOKO Server daemon...")
    from moko_inference.server_manager import MokoLocalInferenceServer
    MokoLocalInferenceServer.stop_servers()
    if not quiet:
        print("✅ Server dihentikan.")


def cmd_restart(quiet=False):
    cmd_stop(quiet=quiet)
    time.sleep(2)
    cmd_start(quiet=quiet)


def cmd_warmup(quiet=False):
    """
    Tunggu sampai server benar-benar OK (bukan loading).
    """
    if not quiet:
        print("⏳ Menunggu MOKO Server warm-up selesai...")
    timeout = 240
    start   = time.time()
    while time.time() - start < timeout:
        moko_http = _check_http(settings.MOKO_LLM_PORT)
        if moko_http == "ok":
            if not quiet:
                print(f"✅ MOKO Server siap dalam {time.time()-start:.1f} detik!")
            return True
        if not quiet:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] MOKO: {moko_http.upper()}")
        time.sleep(3)
    print("❌ Timeout: MOKO Server tidak siap dalam 4 menit.")
    return False


def main():
    parser = argparse.ArgumentParser(description="MOKO Daemon — Sovereign Inference Server Manager")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "warmup"],
                        help="Perintah daemon")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output (cocok untuk autostart di .bashrc)")
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(quiet=args.quiet)
    elif args.command == "stop":
        cmd_stop(quiet=args.quiet)
    elif args.command == "restart":
        cmd_restart(quiet=args.quiet)
    elif args.command == "status":
        cmd_status(quiet=args.quiet)
    elif args.command == "warmup":
        ok = cmd_warmup(quiet=args.quiet)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
