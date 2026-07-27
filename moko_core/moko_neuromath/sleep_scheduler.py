"""
MOKO NeuroMath: Sleep Scheduler — Biological Sleep Cycle Manager
================================================================
Berdasarkan: Sleep-Wake Cycle Regulation (Borbely, 1982)
             Two-Process Model of Sleep Regulation (Process S & C)
             Memory Consolidation During Sleep (Stickgold, 2005)

Di otak manusia, kapan seseorang "harus tidur" dikontrol oleh 2 proses:
  Process S (Sleep Pressure): Semakin lama terjaga → semakin banyak
                              adenosine terkumpul → semakin butuh tidur.
  Process C (Circadian): Ritme sirkadian tubuh mengatur waktu tidur.

Di MOKO, SleepScheduler menentukan kapan Epistemic Forager harus
berhenti belajar (AWAKE mode) dan beralih ke konsolidasi (SLEEP mode).

Tiga kondisi yang memicu Sleep:
  1. ARTICLE COUNT: Sudah membaca N artikel → waktunya tidur
  2. TIME ELAPSED: Sudah aktif M menit tanpa tidur → waktunya tidur
  3. RAM PRESSURE: RAM > threshold% → tidur paksa agar sistem stabil

Setelah sleep selesai, scheduler direset dan Forager bisa lanjut belajar.
"""

import time
import psutil
from pathlib import Path
from typing import Optional

from moko_config import settings


# ── Konstanta ─────────────────────────────────────────────────────────────────

# Jumlah artikel yang harus dibaca sebelum sleep cycle dimulai
ARTICLES_BEFORE_SLEEP = 5

# Waktu maksimum (detik) antara siklus tidur, walau artikel belum terpenuhi
MAX_AWAKE_SECONDS = 10 * 60  # 10 menit

# Batas RAM yang memaksa sleep (system stability)
RAM_SLEEP_THRESHOLD_PCT = 80.0

# Minimum chunk yang di-encode sebelum sleep (anti-terlalu-sering-sleep)
MIN_CHUNKS_BEFORE_SLEEP = 10


class SleepScheduler:
    """
    Mengatur siklus AWAKE → SLEEP → AWAKE pada Epistemic Forager.

    Cara pakai:
        scheduler = SleepScheduler()
        
        # Setelah setiap artikel selesai dibaca:
        scheduler.tick_article()
        
        # Setelah setiap chunk di-encode:
        scheduler.tick_chunk()
        
        # Cek apakah waktunya tidur:
        if scheduler.should_sleep():
            emit_sleep_trigger()
            scheduler.reset()
    """

    def __init__(self,
                 articles_before_sleep: int = ARTICLES_BEFORE_SLEEP,
                 max_awake_seconds: int = MAX_AWAKE_SECONDS,
                 ram_threshold: float = RAM_SLEEP_THRESHOLD_PCT):
        self.articles_before_sleep = articles_before_sleep
        self.max_awake_seconds     = max_awake_seconds
        self.ram_threshold         = ram_threshold

        # Counters
        self._articles_read  = 0
        self._chunks_encoded = 0
        self._awake_since    = time.time()
        self._sleep_reason   = ""

    # ── Tick Methods ──────────────────────────────────────────────────────────

    def tick_article(self):
        """Panggil setelah satu artikel selesai diproses."""
        self._articles_read += 1

    def tick_chunk(self):
        """Panggil setelah satu chunk berhasil di-encode ke Omni-Index."""
        self._chunks_encoded += 1

    # ── Decision Logic ────────────────────────────────────────────────────────

    def should_sleep(self) -> bool:
        """
        Return True jika kondisi sleep terpenuhi.

        Prioritas pemeriksaan:
          1. RAM pressure (paling kritis — langsung sleep)
          2. Article count threshold
          3. Time elapsed threshold
        """
        # Guard: jangan sleep terlalu dini
        if self._chunks_encoded < MIN_CHUNKS_BEFORE_SLEEP and self._articles_read < 2:
            return False

        # 1. RAM Pressure (System S.O.S.)
        ram_pct = psutil.virtual_memory().percent
        if ram_pct >= self.ram_threshold:
            self._sleep_reason = f"RAM PRESSURE ({ram_pct:.1f}% ≥ {self.ram_threshold}%)"
            return True

        # 2. Article count (Process S analog — sleep pressure accumulation)
        if self._articles_read >= self.articles_before_sleep:
            self._sleep_reason = f"ARTICLE_COUNT ({self._articles_read} artikel terbaca)"
            return True

        # 3. Time elapsed (Process C analog — circadian rhythm)
        elapsed = time.time() - self._awake_since
        if elapsed >= self.max_awake_seconds:
            self._sleep_reason = f"TIME_ELAPSED ({elapsed/60:.1f} menit aktif)"
            return True

        return False

    def get_sleep_reason(self) -> str:
        """Kembalikan alasan sleep yang terakhir aktif."""
        return self._sleep_reason

    def get_status(self) -> dict:
        """Status lengkap untuk UI / logging."""
        elapsed = time.time() - self._awake_since
        ram_pct = psutil.virtual_memory().percent
        return {
            "articles_read":         self._articles_read,
            "chunks_encoded":        self._chunks_encoded,
            "awake_seconds":         round(elapsed, 1),
            "awake_minutes":         round(elapsed / 60, 2),
            "ram_pct":               round(ram_pct, 1),
            "articles_until_sleep":  max(0, self.articles_before_sleep - self._articles_read),
            "seconds_until_sleep":   max(0, self.max_awake_seconds - elapsed),
            "sleep_reason":          self._sleep_reason,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """
        Reset semua counter setelah sleep selesai.
        Dipanggil oleh SleepWorker saat done_signal emitted.
        """
        self._articles_read  = 0
        self._chunks_encoded = 0
        self._awake_since    = time.time()
        self._sleep_reason   = ""

    def force_sleep(self, reason: str = "manual"):
        """Paksa sleep di siklus berikutnya (untuk testing atau manual trigger)."""
        self._articles_read = self.articles_before_sleep
        self._sleep_reason  = f"FORCED ({reason})"


# ── Singleton ─────────────────────────────────────────────────────────────────
sleep_scheduler = SleepScheduler()
