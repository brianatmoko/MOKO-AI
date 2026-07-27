"""
moko_ramdisk.py — Modul Manajemen RAM Disk MOKO
================================================
Mengurusi loading dan unloading model AI (GGUF langsung) ke RAM sistem.

CATATAN (v4 — Fix):
  Model sebelumnya menggunakan blob hash Ollama (~9.5 GB, MOKO 7B).
  Sekarang menggunakan file GGUF langsung (~2.5 GB, MOKO3.5 4B Q4_K_M).
  RAM disk cukup 3500 MB, proses server harus dimatikan SEBELUM unload
  agar file tidak dalam keadaan 'busy'.
"""
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional
from moko_config import settings

# ─── Konfigurasi ─────────────────────────────────────────────────────────────

# Path RAM disk
RAM_DISK_PATH = settings.MOKO_RAM_DISK_PATH

# Ukuran RAM disk yang cukup untuk model GGUF saat ini
# MOKO3.5-4B Q4_K_M ≈ 2.5 GB + embedder GGUF ≈ 300 MB + ruang napas
RAM_DISK_SIZE_MB = 3500

# File GGUF yang akan dimuat ke RAM (path di SSD sumber)
GGUF_SOURCES: List[Path] = []


def _resolve_gguf_sources() -> List[Path]:
    """
    Cari semua file GGUF yang dikenal dari settings.
    Cek GGUF_MODEL_PATH dan EMBED_GGUF_PATH jika ada.
    """
    sources = []
    # Model utama MOKO
    try:
        qwen_path = Path(getattr(settings, "MODEL_MOKO_GGUF_PATH", ""))
        if qwen_path.exists() and qwen_path.stat().st_size > 0:
            sources.append(qwen_path)
    except Exception:
        pass

    # Embedder GGUF (jika ada)
    try:
        embed_path = Path(getattr(settings, "MODEL_EMBEDDER_GGUF_PATH", ""))
        if embed_path.exists() and embed_path.stat().st_size > 0:
            sources.append(embed_path)
    except Exception:
        pass

    # Fallback: cari GGUF di folder project
    if not sources:
        project_dir = Path("/home/brianatmokoo/Documents/Linux/MOKO_OS_Project")
        for f in project_dir.glob("*.gguf"):
            if f.stat().st_size > 100_000_000:  # > 100 MB
                sources.append(f)

    return sources


# ─── Fungsi Cek Status ────────────────────────────────────────────────────────

def is_ram_disk_active() -> bool:
    """Cek apakah RAM disk MOKO sudah ter-mount."""
    try:
        result = subprocess.run(
            ["mountpoint", "-q", str(RAM_DISK_PATH)],
            capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False


def is_gguf_in_ram() -> bool:
    """Cek apakah setidaknya satu file GGUF sudah ada di RAM disk."""
    if not is_ram_disk_active():
        return False
    try:
        for f in Path(RAM_DISK_PATH).glob("*.gguf"):
            if f.stat().st_size > 100_000_000:
                return True
    except Exception:
        pass
    return False


def get_ram_disk_usage_mb() -> dict:
    """Baca penggunaan RAM disk saat ini (MB)."""
    try:
        result = subprocess.run(
            ["df", "-m", str(RAM_DISK_PATH)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            return {
                "total_mb": int(parts[1]),
                "used_mb":  int(parts[2]),
                "free_mb":  int(parts[3]),
            }
    except Exception:
        pass
    return {"total_mb": 0, "used_mb": 0, "free_mb": 0}


def get_active_model_path() -> str:
    """
    Kembalikan path model GGUF yang aktif:
    - Jika ada di RAM disk → gunakan RAM disk path
    - Jika tidak → gunakan SSD path langsung
    """
    if is_gguf_in_ram():
        for f in Path(RAM_DISK_PATH).glob("*.gguf"):
            if f.stat().st_size > 100_000_000:
                return str(f)
    # Fallback ke path SSD
    sources = _resolve_gguf_sources()
    if sources:
        return str(sources[0])
    return ""




# ─── Load ke RAM ─────────────────────────────────────────────────────────────

def warmup_to_ram(
    on_progress: Optional[Callable[[str, float], None]] = None
) -> bool:
    """
    Muat file GGUF ke RAM disk secara asinkron.

    Langkah:
    1. Mount tmpfs (3500 MB)
    2. Salin file GGUF dari SSD ke RAM disk
    3. Update path di env jika ada

    Returns: True (thread sudah dimulai)
    """
    def _do_copy():
        try:
            _log(on_progress, "🔍 Mencari file model GGUF...", 0.0)
            sources = _resolve_gguf_sources()

            if not sources:
                _log(on_progress, "❌ Tidak ada file GGUF ditemukan! Pastikan settings.GGUF_MODEL_PATH sudah benar.", 0.0)
                return

            # Hitung total ukuran
            total_size = sum(f.stat().st_size for f in sources)
            total_mb   = total_size // (1024 * 1024)
            _log(on_progress, f"📦 {len(sources)} file GGUF ditemukan ({total_mb} MB total)", 5.0)

            # Cek apakah RAM cukup
            import psutil
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            needed   = total_mb + 500  # ruang napas 500 MB
            if avail_mb < needed:
                _log(on_progress,
                     f"⚠️ RAM tidak cukup! Butuh ≈{needed} MB, tersedia {avail_mb} MB. "
                     f"Tutup aplikasi lain dulu.", 0.0)
                return

            # Mount RAM disk jika belum
            if not is_ram_disk_active():
                _log(on_progress, f"🔧 Membuat RAM disk {RAM_DISK_SIZE_MB} MB...", 8.0)
                result = subprocess.run(
                    ["sudo", "-S", "mount", "-t", "tmpfs",
                     "-o", f"size={RAM_DISK_SIZE_MB}M,mode=0755",
                     "tmpfs", str(RAM_DISK_PATH)],
                    input="brian2756\n",
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0:
                    _log(on_progress, f"❌ Gagal mount RAM disk: {result.stderr[:150]}", 0.0)
                    return
                _log(on_progress, "✅ RAM disk ter-mount.", 10.0)
            else:
                _log(on_progress, "ℹ️  RAM disk sudah aktif, skip mount.", 10.0)

            # Salin file GGUF
            Path(RAM_DISK_PATH).mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(sources):
                dst = Path(RAM_DISK_PATH) / src.name
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    pct = 10.0 + ((idx + 1) / len(sources)) * 85.0
                    _log(on_progress, f"ℹ️  {src.name} sudah ada di RAM, skip.", pct)
                    continue
                size_mb = src.stat().st_size // (1024 * 1024)
                pct_start = 10.0 + (idx / len(sources)) * 85.0
                _log(on_progress, f"📋 Menyalin {src.name} ({size_mb} MB) ke RAM...", pct_start)
                shutil.copy2(str(src), str(dst))
                pct_end = 10.0 + ((idx + 1) / len(sources)) * 85.0
                _log(on_progress, f"✅ {src.name} berhasil disalin ke RAM!", pct_end)

            _log(on_progress, "🚀 RAM Disk siap! Model AI kini berjalan dari RAM (ultra-cepat).", 100.0)

        except Exception as e:
            _log(on_progress, f"❌ Error warmup: {e}", 0.0)

    t = threading.Thread(target=_do_copy, daemon=True)
    t.start()
    return True


# ─── Unload dari RAM ─────────────────────────────────────────────────────────

def stop_ram_disk(on_progress: Optional[Callable[[str], None]] = None) -> bool:
    """
    Melepas RAM disk dan membebaskan memori.

    Langkah PENTING:
    1. Hentikan server llama (MOKO & Embedder) secara instan menggunakan SIGKILL
    2. Baru umount RAM disk secara cepat
    """
    try:
        # Langkah 1: Hentikan server llama jika ada
        _log_simple(on_progress, "🛑 Menghentikan inference server secara instan...")
        _kill_server_by_pidfile("/tmp/moko_qwen_server.pid", "MOKO", on_progress)
        _kill_server_by_pidfile("/tmp/moko_embed_server.pid", "Embedder", on_progress)
        import time; time.sleep(0.2)  # Kurangi delay karena SIGKILL melepas file handle secara instan

        # Langkah 2: Umount RAM disk
        _log_simple(on_progress, "💿 Melepas RAM disk dari memori...")
        result = subprocess.run(
            ["sudo", "-S", "umount", str(RAM_DISK_PATH)],
            input="brian2756\n",
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _log_simple(on_progress, f"✅ RAM dibebaskan (~{RAM_DISK_SIZE_MB} MB kembali ke sistem).")
            return True
        else:
            err = result.stderr.strip()
            # Jika sudah tidak ter-mount → OK
            if "not mounted" in err or "no mount point" in err.lower():
                _log_simple(on_progress, "ℹ️  RAM disk memang sudah tidak aktif.")
                return True
            _log_simple(on_progress, f"❌ Gagal umount: {err[:150]}")
            return False

    except Exception as e:
        _log_simple(on_progress, f"❌ Error stop_ram_disk: {e}")
        return False


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _log(cb, msg: str, pct: float):
    if cb:
        try:
            cb(msg, pct)
        except TypeError:
            cb(msg)


def _log_simple(cb, msg: str):
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def _kill_server_by_pidfile(pidfile: str, name: str, on_progress=None):
    """Matikan proses secara instan menggunakan SIGKILL."""
    import signal as sig_mod
    try:
        if not Path(pidfile).exists():
            return
        pid = int(Path(pidfile).read_text().strip())
        try:
            # Kirim SIGKILL agar langsung mati instan tanpa delay
            os.kill(pid, sig_mod.SIGKILL)
            _log_simple(on_progress, f"  🔴 {name} server (PID {pid}) dihentikan secara instan.")
        except ProcessLookupError:
            pass  # proses sudah tidak ada
        # Hapus PID file
        Path(pidfile).unlink(missing_ok=True)
    except Exception as e:
        _log_simple(on_progress, f"  ⚠️ Gagal hentikan {name}: {e}")
