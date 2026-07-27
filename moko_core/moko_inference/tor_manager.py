"""
MOKO Tor Manager
================
Mengelola instalasi dan startup daemon Tor secara otomatis sebelum
crawling dark web dimulai.

Strategi:
1. Cek apakah binary 'tor' sudah terinstall
2. Jika belum → install via apt-get (butuh sudo tanpa password, atau password di settings)
3. Cek apakah proxy SOCKS5h sudah mendengarkan di port 9050/9150
4. Jika belum → jalankan tor sebagai proses background (nohup)
5. Tunggu hingga Tor berhasil membangun circuit (max 60 detik)
6. Verifikasi koneksi dengan request ke check.torproject.org

Semua langkah dilaporkan via callback log agar GUI bisa
menampilkan progress bar / teks status.
"""

import os
import time
import socket
import shutil
import subprocess
import threading
from typing import Callable, Optional


# ─── Password sudo (disesuaikan dengan ramdisk.py) ────────────────────────────
_SUDO_PASS = "brian2756"

# ─── Port yang dicek ──────────────────────────────────────────────────────────
TOR_PORTS = [9050, 9150]

# ─── PID file untuk proses tor yang dijalankan MOKO ──────────────────────────
TOR_PID_FILE = "/tmp/moko_tor.pid"


def _log(cb: Optional[Callable], msg: str, color: str = "#ff00ff"):
    """Emit log ke callback jika ada."""
    if cb:
        try:
            cb(msg, color)
        except TypeError:
            try:
                cb(msg)
            except Exception:
                pass
    else:
        print(f"[TorManager] {msg}")


def _is_tor_binary_installed() -> bool:
    """Cek apakah binary 'tor' tersedia di PATH."""
    return shutil.which("tor") is not None


def _is_tor_port_active(timeout: float = 0.5) -> tuple:
    """
    Cek apakah SOCKS5 proxy Tor sudah mendengarkan.
    Returns (active: bool, port: int)
    """
    for port in TOR_PORTS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(("127.0.0.1", port))
            s.close()
            return True, port
        except Exception:
            continue
    return False, 0


def _install_tor(log_cb=None) -> bool:
    """
    Install Tor via apt-get menggunakan sudo.
    Returns True jika berhasil.
    """
    _log(log_cb, "📦 Menginstall Tor... (membutuhkan akses sudo)", "#ffaa00")
    try:
        result = subprocess.run(
            ["sudo", "-S", "apt-get", "install", "-y", "tor"],
            input=f"{_SUDO_PASS}\n",
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and _is_tor_binary_installed():
            _log(log_cb, "✅ Tor berhasil diinstall!", "#00ff88")
            return True
        else:
            err = (result.stderr or "").strip()[:200]
            _log(log_cb, f"❌ Gagal install Tor: {err}", "#ff4444")
            return False
    except subprocess.TimeoutExpired:
        _log(log_cb, "❌ Install Tor timeout (>120 detik).", "#ff4444")
        return False
    except Exception as e:
        _log(log_cb, f"❌ Error install Tor: {e}", "#ff4444")
        return False


def _start_tor_process(log_cb=None) -> bool:
    """
    Jalankan Tor sebagai proses background menggunakan nohup.
    Menunggu hingga port 9050 aktif (max 60 detik).
    Returns True jika berhasil.
    """
    _log(log_cb, "🧅 Menghidupkan Tor daemon di background...", "#ff00ff")

    try:
        # Coba via systemctl dulu (lebih stabil)
        try:
            r = subprocess.run(
                ["sudo", "-S", "systemctl", "start", "tor"],
                input=f"{_SUDO_PASS}\n",
                capture_output=True, text=True, timeout=15
            )
            if r.returncode == 0:
                _log(log_cb, "✅ Tor dijalankan via systemctl.", "#00ff88")
                # Tunggu port aktif
                return _wait_for_tor_port(log_cb, max_wait=30)
        except Exception:
            pass

        # Fallback: jalankan tor langsung sebagai proses
        _log(log_cb, "🔄 Systemctl gagal — mencoba jalankan tor langsung...", "#ffaa00")
        tor_bin = shutil.which("tor") or "tor"
        proc = subprocess.Popen(
            ["sudo", "-S", tor_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Kirim password ke stdin (untuk sudo -S)
        try:
            proc.stdin.write(f"{_SUDO_PASS}\n".encode())
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass

        # Simpan PID
        try:
            with open(TOR_PID_FILE, "w") as f:
                f.write(str(proc.pid))
        except Exception:
            pass

        return _wait_for_tor_port(log_cb, max_wait=60)

    except Exception as e:
        _log(log_cb, f"❌ Gagal menjalankan Tor: {e}", "#ff4444")
        return False


def _wait_for_tor_port(log_cb=None, max_wait: int = 60) -> bool:
    """
    Tunggu hingga port 9050/9150 aktif, dengan animasi loading.
    Returns True jika berhasil dalam max_wait detik.
    """
    dots = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    for i in range(max_wait):
        active, port = _is_tor_port_active()
        if active:
            _log(log_cb, f"✅ Tor SOCKS5 proxy aktif di port {port}!", "#00ff88")
            return True
        spinner = dots[i % len(dots)]
        _log(log_cb, f"{spinner} Menunggu Tor circuit terbentuk... ({i+1}/{max_wait}s)", "#ff88ff")
        time.sleep(1)

    _log(log_cb, "❌ Tor tidak aktif setelah menunggu. Cek instalasi Tor.", "#ff4444")
    return False


def _verify_tor_connection(log_cb=None) -> bool:
    """
    Verifikasi bahwa koneksi Tor bekerja dengan mengakses check.torproject.org
    melalui SOCKS5h proxy.
    Returns True jika verifikasi berhasil.
    """
    active, port = _is_tor_port_active()
    if not active:
        return False

    _log(log_cb, "🔍 Memverifikasi koneksi Tor...", "#ff88ff")
    try:
        import requests
        proxies = {
            "http": f"socks5h://127.0.0.1:{port}",
            "https": f"socks5h://127.0.0.1:{port}",
        }
        r = requests.get(
            "https://check.torproject.org/api/ip",
            proxies=proxies,
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            is_tor = data.get("IsTor", False)
            ip = data.get("IP", "?")
            if is_tor:
                _log(log_cb, f"🎉 Tor AKTIF! IP anonim: {ip}", "#00ff88")
                return True
            else:
                _log(log_cb, f"⚠️ Terhubung ke Tor port, tapi IP {ip} bukan Tor exit node.", "#ffaa00")
                return True  # Port aktif = cukup untuk crawl
    except Exception as e:
        _log(log_cb, f"⚠️ Verifikasi Tor gagal ({e}) — tetap lanjut crawl.", "#ffaa00")

    return True  # Port aktif = anggap OK


def ensure_tor_ready(log_cb: Optional[Callable] = None) -> bool:
    """
    ENTRY POINT UTAMA — pastikan Tor siap sebelum dark web crawling.

    Langkah otomatis:
    1. Cek binary tor → install jika belum ada
    2. Cek port 9050/9150 → start daemon jika belum aktif
    3. Verifikasi koneksi Tor
    4. Return True jika siap, False jika gagal

    Args:
        log_cb: callback(msg: str, color: str) untuk progress log ke GUI
    """
    _log(log_cb, "🧅 [Tor Manager] Memeriksa status Tor...", "#ff88ff")

    # ── Langkah 1: Cek dan install binary ─────────────────────────────────────
    if not _is_tor_binary_installed():
        _log(log_cb, "⚠️ Tor belum terinstall. Menginstall otomatis...", "#ffaa00")
        success = _install_tor(log_cb)
        if not success:
            _log(log_cb,
                 "❌ [Tor Manager] Gagal install Tor. Dark web crawling akan terbatas ke Ahmia clearnet saja.",
                 "#ff4444")
            return False
    else:
        _log(log_cb, "✅ Binary Tor ditemukan: " + (shutil.which("tor") or "tor"), "#00ff88")

    # ── Langkah 2: Cek dan start daemon ───────────────────────────────────────
    active, port = _is_tor_port_active()
    if active:
        _log(log_cb, f"✅ Tor proxy sudah aktif di port {port}.", "#00ff88")
    else:
        _log(log_cb, "🔄 Tor proxy belum aktif. Menghidupkan daemon...", "#ffaa00")
        started = _start_tor_process(log_cb)
        if not started:
            _log(log_cb,
                 "❌ [Tor Manager] Tor daemon gagal start. Dark web crawling akan terbatas.",
                 "#ff4444")
            return False

    # ── Langkah 3: Verifikasi koneksi ─────────────────────────────────────────
    ok = _verify_tor_connection(log_cb)
    if ok:
        _log(log_cb, "🟢 [Tor Manager] Tor SIAP. Memulai dark web crawling...", "#00ff88")
    return ok
