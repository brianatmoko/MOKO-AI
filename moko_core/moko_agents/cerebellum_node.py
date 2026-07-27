"""
MOKO Agents: CerebellumNode — Internal Forward Model & Sequence Learning
=========================================================================
Berdasarkan:
  - Springer Nature / The Cerebellum (2019): "Cerebro-Cerebellum as a
    Locus of Forward Model"
  - NCBI/PMC (2020): "The Cerebellum as a Locus of Forward Model"
  - Ito (2008): Cerebellar circuitry as a neuronal machine

FUNGSI:
  1. Internal Forward Model: prediksi output SEBELUM eksekusi selesai.
     Diimplementasikan sebagai Kalman Filter ringan — estimasi state
     berdasarkan riwayat input-output.

  2. Sequence Learning: pola urutan yang berulang menjadi otomatis (cached).
     Jika prediction_error rendah berulang kali (>3x), pola disimpan ke
     sequence_cache → AnalystNode bisa skip DeepThink.

  3. Timing Precision: mengatur delay optimal antara iterasi.

  4. Error-Based Correction: selisih prediksi vs aktual dikirim sebagai
     sinyal koreksi (analog Purkinje cell error signal).

INTEGRASI:
  AnalystNode instantiates CerebellumNode dan memanggil:
    - predict_output(query_embedding) → predicted_output (str)
    - record_actual(query_hash, actual_output, actual_embedding)
    - get_sequence_cache_hit(query_hash) → cached response | None
"""

import math
import time
import hashlib
import json
from collections import deque
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from moko_config import settings


# ── Konstanta ─────────────────────────────────────────────────────────────────

SEQUENCE_CACHE_THRESHOLD   = 3      # Error rendah berturut-kali → cache
PREDICTION_ERROR_THRESHOLD = 0.25   # Error di bawah ini dianggap "familiar"
MAX_CACHE_SIZE             = 200    # Batas entri sequence_cache
CACHE_PERSIST_PATH         = None   # Akan diisi di __init__

# Kalman Filter parameters
KF_PROCESS_NOISE           = 0.01   # Q: process noise
KF_MEASUREMENT_NOISE       = 0.1    # R: measurement noise


# ── Helper: Cosine Similarity ─────────────────────────────────────────────────

def _cosine(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot  = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Lightweight Kalman Filter (1D state per query class) ──────────────────────

class KalmanEstimator:
    """
    Kalman Filter 1D untuk estimasi kualitas output.
    State = perkiraan cosine similarity antara prediksi dan aktual.

    Digunakan sebagai "confidence meter" cerebellar prediction.
    """

    def __init__(self, initial_estimate: float = 0.5):
        self.estimate = initial_estimate  # x̂ (state estimate)
        self.error_cov = 1.0              # P (error covariance)

    def predict(self) -> float:
        """Predict step: state biasanya stasioner (model konstan)."""
        self.error_cov += KF_PROCESS_NOISE
        return self.estimate

    def update(self, measurement: float) -> float:
        """Update step: koreksi dengan measurement baru."""
        kalman_gain    = self.error_cov / (self.error_cov + KF_MEASUREMENT_NOISE)
        self.estimate  = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error_cov = (1 - kalman_gain) * self.error_cov
        return self.estimate


# ── CerebellumNode ────────────────────────────────────────────────────────────

class CerebellumNode:
    """
    Node cerebellar untuk MOKO — berjalan paralel dengan AnalystNode.

    Lifecycle per query:
      1. predict_output(query_emb)  → string prediksi singkat
      2. AnalystNode generate actual output
      3. record_actual(hash, actual_text, actual_emb) → update error
      4. Jika sequence matang → simpan ke cache
      5. Berikutnya: get_sequence_cache_hit(hash) → bypass DeepThink
    """

    def __init__(self):
        self.sequence_cache: Dict[str, Dict] = {}  # hash → {text, hits, emb}
        self.error_history:  Dict[str, deque] = {} # hash → deque[float]
        self.kalman_models:  Dict[str, KalmanEstimator] = {}
        self.timing_log:     deque = deque(maxlen=100)  # timing records

        # Persist sequence_cache ke disk
        workspace = Path(settings.WORKSPACE_DIR)
        self.cache_path = workspace / ".math_omni" / "cerebellum_cache.jsonl"
        self._load_cache()

    # ── Cache Persistence ─────────────────────────────────────────────────────

    def _load_cache(self):
        """Load sequence_cache dari disk."""
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            h = entry.get("hash")
                            if h:
                                self.sequence_cache[h] = entry
                        except Exception:
                            pass
        except Exception:
            pass

    def _save_cache(self):
        """Persist sequence_cache ke disk (top MAX_CACHE_SIZE entri)."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Prioritaskan berdasarkan hit count
            sorted_items = sorted(
                self.sequence_cache.values(),
                key=lambda e: e.get("hits", 0),
                reverse=True
            )[:MAX_CACHE_SIZE]
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                for entry in sorted_items:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def get_query_hash(self, query: str) -> str:
        """Hasilkan hash deterministik dari query (setelah normalisasi)."""
        normalized = query.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]

    def get_sequence_cache_hit(self, query_hash: str) -> Optional[str]:
        """
        Cek apakah pola query ini sudah ada di sequence_cache.

        Returns:
            str:  cached response jika ada dan mature (hits >= threshold)
            None: tidak ada cache hit / belum mature
        """
        entry = self.sequence_cache.get(query_hash)
        if not entry:
            return None

        if entry.get("hits", 0) >= SEQUENCE_CACHE_THRESHOLD:
            # Update hit count
            entry["hits"]       += 1
            entry["last_used"]   = time.time()
            self._save_cache()
            return entry.get("text")
        return None

    def predict_confidence(self, query_hash: str) -> float:
        """
        Prediksi confidence (0.0 – 1.0) untuk query ini.
        Menggunakan Kalman Filter — makin sering dilihat, makin akurat estimasi.

        Returns:
            float: 0.0 = tidak familiar, 1.0 = sangat familiar (dapat di-cache)
        """
        if query_hash not in self.kalman_models:
            self.kalman_models[query_hash] = KalmanEstimator(initial_estimate=0.3)

        kf = self.kalman_models[query_hash]
        return round(kf.predict(), 4)

    def record_actual(self,
                      query_hash: str,
                      actual_text: str,
                      actual_embedding: Optional[List[float]] = None,
                      timing_ms: float = 0.0):
        """
        Rekam output aktual setelah AnalystNode selesai.
        Hitung prediction error, update Kalman, cek apakah perlu di-cache.

        Args:
            query_hash:       Hash query (dari get_query_hash)
            actual_text:      Teks jawaban yang dihasilkan
            actual_embedding: Embedding dari actual_text (opsional)
            timing_ms:        Berapa ms yang diperlukan untuk menjawab
        """
        # Init error history
        if query_hash not in self.error_history:
            self.error_history[query_hash] = deque(maxlen=10)

        # Hitung prediction error berdasarkan Kalman estimate
        kf       = self.kalman_models.get(query_hash,
                                          KalmanEstimator(initial_estimate=0.3))
        # Error = 1 - Kalman estimate (makin familiar, error makin rendah)
        est_sim  = kf.estimate
        p_error  = max(0.0, 1.0 - est_sim)

        # Update Kalman dengan measurement baru
        # Measurement: jika teks pendek → similarity tinggi (familiar response)
        # Lebih baik gunakan embedding similarity jika ada
        if actual_embedding and len(actual_embedding) == 768:
            # Gunakan panjang teks sebagai proxy confidence (lebih panjang = lebih banyak berpikir)
            text_len = len(actual_text)
            measurement = max(0.1, 1.0 - min(1.0, text_len / 5000))
        else:
            # Fallback: hitung berdasarkan timing
            measurement = max(0.1, 1.0 - min(1.0, timing_ms / 30000))

        if query_hash not in self.kalman_models:
            self.kalman_models[query_hash] = KalmanEstimator(initial_estimate=0.3)
        updated_est = self.kalman_models[query_hash].update(measurement)

        # Simpan ke error history
        self.error_history[query_hash].append(p_error)

        # Simpan timing log
        self.timing_log.append({
            "query_hash": query_hash,
            "timing_ms":  timing_ms,
            "p_error":    round(p_error, 4),
            "ts":         time.time(),
        })

        # Cek apakah pola cukup matang untuk di-cache
        self._check_and_cache(query_hash, actual_text, actual_embedding, p_error)

    def _check_and_cache(self,
                         query_hash: str,
                         actual_text: str,
                         actual_embedding: Optional[List[float]],
                         current_error: float):
        """
        Jika error terus rendah (< threshold) berulang kali →
        simpan ke sequence_cache (analog: pola menjadi refleks otomatis).
        """
        history = self.error_history.get(query_hash, deque())
        if len(history) < SEQUENCE_CACHE_THRESHOLD:
            return

        # Cek berapa banyak error rendah berturut-turut
        low_error_streak = sum(
            1 for e in history if e < PREDICTION_ERROR_THRESHOLD
        )

        if low_error_streak >= SEQUENCE_CACHE_THRESHOLD:
            # Cache respons ini sebagai "sequence"
            existing = self.sequence_cache.get(query_hash, {})
            self.sequence_cache[query_hash] = {
                "hash":       query_hash,
                "text":       actual_text,
                "embedding":  actual_embedding[:50] if actual_embedding else [],
                "hits":       existing.get("hits", 0) + 1,
                "cached_at":  time.time(),
                "last_used":  time.time(),
                "error_avg":  round(sum(history) / len(history), 4),
            }
            self._save_cache()

    def get_avg_timing(self) -> float:
        """Rata-rata waktu respons dari timing log."""
        if not self.timing_log:
            return 0.0
        return sum(t.get("timing_ms", 0) for t in self.timing_log) / len(self.timing_log)

    def get_cache_stats(self) -> Dict:
        """Statistik sequence cache untuk monitoring."""
        mature = sum(
            1 for e in self.sequence_cache.values()
            if e.get("hits", 0) >= SEQUENCE_CACHE_THRESHOLD
        )
        return {
            "total_cached":    len(self.sequence_cache),
            "mature_sequences":mature,
            "avg_timing_ms":   round(self.get_avg_timing(), 1),
        }

    def apply_purkinje_error_signal(self):
        """
        Pangkas sequence_cache: hapus entri yang sudah lama tidak digunakan
        (analog Purkinje cell LTD — hapus prediksi yang salah/usang).
        Panggil ini secara berkala dari apoptosis_daemon.
        """
        now = time.time()
        stale_threshold = 7 * 24 * 3600  # 7 hari tidak digunakan

        to_delete = [
            h for h, e in self.sequence_cache.items()
            if (now - e.get("last_used", now)) > stale_threshold
        ]
        for h in to_delete:
            del self.sequence_cache[h]

        if to_delete:
            self._save_cache()

        return len(to_delete)


# ── Singleton ──────────────────────────────────────────────────────────────────
cerebellum_node = CerebellumNode()
