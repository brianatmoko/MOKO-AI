"""
MOKO GC Tuner — Python Memory & Garbage Collection Optimizer
=============================================================
Modul ini dikerjakan satu kali saat startup untuk mengatur:

1. GC Threshold — default Python (700, 10, 10) terlalu agresif untuk
   proses long-running dengan banyak objek sementara (numpy array, ctypes buffer).
   Menaikkan threshold gen-0 mengurangi frekuensi koleksi yang tidak perlu.

2. GC Generation Management — menjadwalkan koleksi gen-2 (full sweep)
   hanya setelah ingest selesai, bukan di tengah-tengah operasi.

3. Memory Trim — membebaskan memori kosong kembali ke OS setelah
   operasi besar selesai (via ctypes malloc_trim pada Linux).

4. RSS Monitor — membantu mendeteksi memory leak sesi panjang.

Penggunaan:
    from moko_memory.gc_tuner import gc_tuner
    gc_tuner.apply()           # panggil sekali di startup
    gc_tuner.post_ingest()     # panggil setelah batch ingest selesai
    gc_tuner.report()          # cetak laporan memori saat ini
"""

import gc
import os
import sys
import ctypes
import threading
import time
from typing import Optional


class MokoGCTuner:
    """
    Mengoptimalkan Python GC dan penggunaan memori untuk proses MOKO yang
    berjalan lama (daemon + learning worker + LLM inference).
    """

    # GC thresholds optimal untuk long-running AI workload:
    # Gen-0: 2000 (default 700) — kurangi minor GC saat numpy/ctypes aktif
    # Gen-1: 30   (default 10)  — sedikit lebih jarang dari default
    # Gen-2: 20   (default 10)  — full sweep hanya saat perlu
    GC_THRESHOLD = (2000, 30, 20)

    def __init__(self):
        self._applied = False
        self._baseline_rss_mb: Optional[float] = None
        self._lock = threading.Lock()

    def apply(self):
        """
        Terapkan semua optimasi GC dan memori.
        Aman dipanggil berkali-kali (idempotent).
        """
        with self._lock:
            if self._applied:
                return

            # 1. Atur threshold GC
            gc.set_threshold(*self.GC_THRESHOLD)
            print(f"[GCTuner] ✅ GC threshold diatur: {self.GC_THRESHOLD}")

            # 2. Nonaktifkan GC selama startup (akan diaktifkan kembali setelah init)
            # Ini mencegah koleksi di tengah loading centroid dan model
            gc.disable()
            print("[GCTuner] ⏸️  GC dinonaktifkan sementara selama startup...")

            # 3. Simpan baseline RSS
            self._baseline_rss_mb = self._get_rss_mb()

            self._applied = True

    def enable(self):
        """Aktifkan kembali GC setelah startup selesai."""
        gc.enable()
        rss = self._get_rss_mb()
        print(f"[GCTuner] ▶️  GC diaktifkan kembali. RSS saat ini: {rss:.1f} MB")

    def post_ingest(self, force_gen2: bool = False):
        """
        Dipanggil setelah setiap batch ingest selesai.
        Menjalankan GC gen-0 + gen-1, dan optionally gen-2 (full sweep).
        Kemudian trim heap ke OS.
        """
        # Gen-0 + Gen-1 collection
        gc.collect(0)
        gc.collect(1)

        if force_gen2:
            gc.collect(2)
            print("[GCTuner] 🧹 Full GC (gen-2) selesai.")

        # Trim malloc arena ke OS (Linux only, tidak fatal jika gagal)
        self._malloc_trim()

    def post_learning_session(self):
        """
        Dipanggil setelah TopicLearningWorker selesai.
        Membersihkan semua generasi dan trim heap.
        """
        collected = gc.collect(2)
        self._malloc_trim()
        rss_after = self._get_rss_mb()
        print(f"[GCTuner] 🧹 Post-learning GC: {collected} objek dikumpulkan. "
              f"RSS: {rss_after:.1f} MB")

    def report(self) -> dict:
        """Laporan memori saat ini."""
        rss = self._get_rss_mb()
        baseline = self._baseline_rss_mb or rss
        gc_counts = gc.get_count()
        gc_thresholds = gc.get_threshold()
        return {
            "rss_mb": round(rss, 1),
            "baseline_mb": round(baseline, 1),
            "delta_mb": round(rss - baseline, 1),
            "gc_counts_gen0_1_2": gc_counts,
            "gc_thresholds": gc_thresholds,
            "gc_enabled": gc.isenabled(),
        }

    def print_report(self):
        """Cetak laporan memori yang dapat dibaca manusia."""
        r = self.report()
        print(
            f"[GCTuner] 📊 Memori: RSS={r['rss_mb']} MB | "
            f"Delta dari baseline: {r['delta_mb']:+.1f} MB | "
            f"GC counts={r['gc_counts_gen0_1_2']} | "
            f"GC {'ON' if r['gc_enabled'] else 'OFF'}"
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_rss_mb() -> float:
        """Baca RSS (Resident Set Size) proses ini dari /proc/self/status."""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _malloc_trim():
        """
        Kembalikan halaman memori kosong ke OS menggunakan malloc_trim(0).
        Hanya efektif di Linux dengan glibc. Tidak fatal jika tidak tersedia.
        """
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.malloc_trim(0)
        except Exception:
            pass  # Tidak tersedia di macOS atau sistem non-glibc


# ── Singleton ─────────────────────────────────────────────────────────────────
gc_tuner = MokoGCTuner()
