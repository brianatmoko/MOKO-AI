"""
MOKO NeuroMath: Thalamus Gate & Thalamus Node
================================================
Berdasarkan:
  - Frontiers in Neurology (2025): "Thalamus and consciousness: a
    systematic review on thalamic nuclei associated with consciousness"
  - Sherman & Guillery (2006): Thalamic Gating Theory
  - Lisman & Grace (2005): Novelty Detection in Hippocampus
  - MIT (2024): "Thalamic Relay: Input Gating and Gain Control"

DUA KOMPONEN:

1. ThalamusGate (SUDAH ADA — unchanged):
   Filter novelty sederhana berdasarkan cosine similarity.
   Menentukan apakah chunk perlu di-encode ke Omni-Index.

2. ThalamusNode (BARU — Fase 2 Neurosains):
   Hub integrasi kortiko-subkortikal sejati dengan:
   - firing_mode: "tonic" (transmisi setia) | "burst" (amplifikasi novelty)
   - attention_gain: multiplier kekuatan sinyal input (0.5 – 2.0)
   - consciousness_threshold: ambang batas untuk masuk ke processing sadar
   - spindle_coordinator: interface ke SleepConsolidationWorker
   - sensory_sharpening: resolusi representasi input yang relevan

   Thalamus biologis:
   - Tonic Mode: processing normal, transmisi setia, threshold rendah
   - Burst Mode: novelty tinggi / arousal tinggi → amplifikasi 1.5-2x
     (analog: kamera yang tiba-tiba zoom ke objek penting)
"""

import json
import math
import struct
from pathlib import Path
from typing import Optional

from moko_config import settings
from moko_neuromath.locus_coeruleus import locus_coeruleus


# ── Konstanta ────────────────────────────────────────────────────────────────

# Jika cosine similarity dengan chunk terdekatan MELEBIHI threshold ini,
# konten dianggap "sudah diketahui" → skip (tidak di-encode ulang).
# 0.85 = 85% mirip → familiar. Turunkan jika ingin lebih banyak encoding.
NOVELTY_THRESHOLD = 0.85

# Jumlah maksimum kandidat dari Omni-Index yang di-sample untuk perbandingan.
# Semakin besar → lebih akurat, tapi lebih lambat. 50 adalah sweet-spot.
SAMPLE_SIZE = 50

# Jika Omni-Index memiliki kurang dari N entri, skip filter (cold start mode).
# Otak bayi tidak bisa mendeteksi novelty karena tidak punya referensi.
COLD_START_MIN = 30


# ── Helper: Cosine Similarity ─────────────────────────────────────────────────

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity antara dua vektor float. Tanpa numpy, murni Python."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _fp16_to_fp32(fp16_bytes: bytes) -> list[float]:
    """Konversi FP16 bytes (1536 byte = 768 float16) ke float list."""
    import numpy as np
    arr = np.frombuffer(fp16_bytes, dtype=np.float16)
    return arr.astype(np.float32).tolist()


class ThalamusGate:
    """
    Biological novelty filter untuk pipeline belajar MOKO.

    Cara kerja:
      1. Terima embedding teks baru (768 dim float list)
      2. Sample N entri dari Omni-Index
      3. Hitung cosine similarity dengan setiap sample
      4. Jika max similarity > NOVELTY_THRESHOLD → bukan novelty → skip
      5. Jika max similarity < NOVELTY_THRESHOLD → novelty → encode
    
    Cost: ~0ms (murni aritmetika Python, tanpa I/O LLM)
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.omni_dir = Path(workspace_dir or settings.WORKSPACE_DIR) / ".moko_crypto"
        self._entry_count_cache: int = -1
        self._cache_dirty: bool = True

    def _count_entries(self) -> int:
        """
        Hitung total entri dari _domain_meta.json di setiap domain .moko_crypto.
        Jauh lebih cepat dibanding traversal file.
        """
        if not self.omni_dir.exists():
            return 0
        import json
        total = 0
        for meta_file in self.omni_dir.rglob("_domain_meta.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                total += meta.get("entry_count", 0)
            except Exception:
                pass
        return total

    def _sample_embeddings(self, n: int = SAMPLE_SIZE) -> list[list[float]]:
        """
        Sample N vektor FP16 dari .moko_crypto/domain/bucket/sub_bucket/vectors.f16.
        Menggunakan stratified sampling dari berbagai domain.
        """
        import numpy as np
        if not self.omni_dir.exists():
            return []

        FP16_RECORD_SIZE = 1536  # 768 float16 × 2 bytes
        vec_files = list(self.omni_dir.rglob("vectors.f16"))
        if not vec_files:
            return []

        samples = []
        per_file = max(1, n // len(vec_files))
        for vf in vec_files:
            try:
                data = vf.read_bytes()
                record_count = len(data) // FP16_RECORD_SIZE
                if record_count == 0:
                    continue
                for i in range(min(per_file, record_count)):
                    start = i * FP16_RECORD_SIZE
                    fp16_bytes = data[start:start + FP16_RECORD_SIZE]
                    if len(fp16_bytes) == FP16_RECORD_SIZE:
                        samples.append(_fp16_to_fp32(fp16_bytes))
            except Exception:
                continue

            if len(samples) >= n:
                break

        return samples[:n]

    def get_novelty_score(self, text_embedding: list[float]) -> float:
        """
        Hitung skor novelty konten baru terhadap Omni-Index yang ada.

        Returns:
            float: 0.0 = sangat familiar (sudah sangat diketahui)
                   1.0 = benar-benar baru (tidak ada yang mirip)

        Formula: novelty = 1 - max_cosine_similarity(text_emb, known_embeddings)
        """
        entry_count = self._count_entries()

        # Cold start: belum cukup data untuk membandingkan → anggap semua novel
        if entry_count < COLD_START_MIN:
            return 1.0

        samples = self._sample_embeddings()
        if not samples:
            return 1.0

        max_similarity = max(
            _cosine_similarity(text_embedding, s) for s in samples
        )
        novelty = 1.0 - max_similarity
        return round(max(0.0, min(1.0, novelty)), 4)

    def is_novel(self, text_embedding: list[float],
                 threshold: float = NOVELTY_THRESHOLD) -> bool:
        """
        Putuskan apakah konten ini layak di-encode ke Omni-Index.

        Args:
            text_embedding: Embedding 768-dim dari chunk teks baru.
            threshold: Ambang batas novelty. Default NOVELTY_THRESHOLD (0.85).
                       novelty_score > (1 - threshold) → novel → encode.
                       novelty_score < (1 - threshold) → familiar → skip.

        Returns:
            True  = Konten baru, perlu di-encode (lanjutkan pipeline)
            False = Sudah diketahui, skip (hemat resource)
        """
        # Modulasi threshold menggunakan Locus Coeruleus (Noradrenalin)
        mod_threshold = locus_coeruleus.modulate_novelty_threshold(threshold)
        novelty_score = self.get_novelty_score(text_embedding)
        # Konversi: novelty > (1-threshold) berarti similarity < threshold
        # Equivalen: hanya encode jika similarity tertinggi di bawah threshold
        return novelty_score > (1.0 - mod_threshold)

    def get_arousal_level(self, novelty_score: float) -> str:
        """
        Konversi novelty score ke Arousal Level untuk Math-Omni.
        Analog: semakin baru sesuatu → semakin tinggi arousal di otak.

        Returns:
            "1" = Tenang (familiar, novelty < 0.3)
            "2" = Sedang (agak baru, novelty 0.3-0.7)
            "3" = Tinggi (sangat baru, novelty > 0.7)
        """
        if novelty_score > 0.7:
            return "3"
        elif novelty_score > 0.3:
            return "2"
        else:
            return "1"


# ── Singleton (ThalamusGate ─ backward compat) ────────────────────────────
thalamus_gate = ThalamusGate()


# ─────────────────────────────────────────────────────────────────────────────
# MODUL 2.4 — THALAMUS NODE SEJATI
# ─────────────────────────────────────────────────────────────────────────────

class ThalamusNode:
    """
    Thalamus Node Sejati — Hub integrasi kortiko-subkortikal MOKO.

    Berbeda dari ThalamusGate (hanya filter novelty),
    ThalamusNode mengatur:
      1. Firing mode (tonic vs burst)
      2. Attention gain (penguat/pelemah sinyal)
      3. Consciousness gating (apakah input masuk ke processing sadar?)
      4. Spindle coordination (sinkronisasi dengan SleepConsolidation)
      5. Sensory sharpening (meningkatkan resolusi representasi relevan)

    Cara pakai:
        gate_result = thalamus_node.gate_input(
            embedding=emb,
            novelty_score=novelty,
            arousal_score=0.7,       # dari AmygdalaNode
            current_goal=query,
        )
        if gate_result["pass"]:
            strength = gate_result["gain"]  # multiplier untuk LTP
    """

    def __init__(self):
        self.firing_mode          : str   = "tonic"   # "tonic" | "burst"
        self.attention_gain       : float = 1.0        # Multiplier sinyal (0.5–2.0)
        self.consciousness_threshold: float = 0.20     # Ambang batas novelty untuk masuk
        self._spindle_active      : bool  = False      # True saat sleep spindle berjalan
        self._burst_timer         : float = 0.0        # Kapan burst dimulai
        self._burst_duration      : float = 1.5        # Detik burst setelah novelty tinggi

        # History untuk adaptasi threshold
        self._recent_novelties    : list  = []         # Novelty scores terakhir (N=50)
        self._gain_history        : list  = []         # Gain history

    # ── Firing Mode ─────────────────────────────────────────────────────────────

    def _update_firing_mode(self, novelty_score: float, arousal_score: float = 0.5):
        """
        Perbarui firing mode berdasarkan novelty dan arousal:
          - Burst mode: novelty > 0.7 ATAU arousal > 0.75
          - Tonic mode: kondisi normal
        """
        import time as _time
        now = _time.time()

        if novelty_score > 0.7 or arousal_score > 0.75:
            self.firing_mode  = "burst"
            self._burst_timer = now
        elif (now - self._burst_timer) > self._burst_duration:
            # Kembali ke tonic setelah burst_duration selesai
            self.firing_mode = "tonic"

    def _compute_attention_gain(self, novelty_score: float,
                                 arousal_score: float = 0.5) -> float:
        """
        Hitung attention_gain:
          - Tonic mode: gain = 1.0 (transmisi setia)
          - Burst mode: gain = 1.0 + 0.8 * novelty_score (amplifikasi)
          - Sleep spindle: gain = 0.3 (supresé cortical excitability)
        """
        if self._spindle_active:
            return 0.3  # Thalamic gating saat sleep

        if self.firing_mode == "burst":
            gain = 1.0 + 0.8 * novelty_score + 0.4 * arousal_score
            return round(min(2.5, gain), 3)
        else:
            # Tonic: gain dimodulasi oleh attention baseline
            gain = 0.8 + 0.4 * novelty_score
            return round(min(1.5, gain), 3)

    # ── Consciousness Gating ──────────────────────────────────────────────

    def _adapt_consciousness_threshold(self):
        """
        Adaptasikan consciousness_threshold berdasarkan recent novelty.
        Jika rata-rata novelty tinggi (banyak hal baru), naikkan threshold
        (sistem lebih selektif untuk menghindari overload).
        Jika novelty rendah (sistem familiar), turunkan threshold
        (lebih sensitif terhadap hal baru).
        """
        if len(self._recent_novelties) < 10:
            return
        avg_novelty = sum(self._recent_novelties[-20:]) / min(20, len(self._recent_novelties))
        # Adaptasi: threshold berkisar 0.10 – 0.45
        self.consciousness_threshold = round(0.10 + 0.35 * avg_novelty, 3)

    def is_conscious_input(self, novelty_score: float) -> bool:
        """
        Putuskan apakah input ini layak masuk ke processing sadar.
        Input di bawah consciousness_threshold diproses secara "otomatis"
        (analog subconscious processing).
        """
        return novelty_score >= self.consciousness_threshold

    # ── Spindle Coordinator ─────────────────────────────────────────────────

    def begin_sleep_spindle(self):
        """
        Aktifkan sleep spindle mode (dipanggil oleh SleepConsolidationWorker).
        Saat spindle aktif, attention_gain turun ke 0.3 — korteks
        "dimatikan sementara" agar hippocampal replay bisa berjalan tanpa
        interferensi dari input baru.
        """
        self._spindle_active = True

    def end_sleep_spindle(self):
        """Nonaktifkan sleep spindle mode."""
        self._spindle_active = False
        self.firing_mode     = "tonic"
        self.attention_gain  = 1.0

    @property
    def spindle_active(self) -> bool:
        return self._spindle_active

    # ── Main Gate API ────────────────────────────────────────────────────────

    def gate_input(self,
                   embedding: list,
                   novelty_score: float,
                   arousal_score: float = 0.5,
                   current_goal: str = "") -> dict:
        """
        Main gating function: putuskan apakah dan seberapa kuat input
        diproses oleh sistem MOKO.

        Args:
            embedding:    Embedding 768-dim dari input baru
            novelty_score: 0.0 (sangat familiar) – 1.0 (benar-benar baru)
            arousal_score: 0.0 – 1.0 dari AmygdalaNode
            current_goal:  Query/goal aktif untuk sensory sharpening

        Returns:
            dict:
                "pass":        bool  — apakah input lolos ke processing
                "gain":        float — multiplier kekuatan sinyal
                "firing_mode": str   — "tonic" | "burst"
                "conscious":   bool  — apakah masuk ke conscious processing
                "sharpening":  float — resolusi enhancement factor
        """
        # Update tracking
        self._recent_novelties.append(novelty_score)
        if len(self._recent_novelties) > 100:
            self._recent_novelties = self._recent_novelties[-100:]

        # Adaptasi threshold
        self._adapt_consciousness_threshold()

        # Update firing mode
        self._update_firing_mode(novelty_score, arousal_score)

        # Hitung gain
        gain = self._compute_attention_gain(novelty_score, arousal_score)
        self.attention_gain = gain

        # Sensory sharpening: boost relevansi jika ada goal aktif
        sharpening = 1.0
        if current_goal and novelty_score > 0.4:
            # Sederhana: boost proporsional novelty x goal length factor
            goal_factor = min(1.0, len(current_goal) / 200)
            sharpening  = 1.0 + 0.5 * novelty_score * goal_factor

        # Pass / No-pass decision
        # Saat sleep spindle: blok semua input baru (replay mode)
        if self._spindle_active:
            return {
                "pass":        False,
                "gain":        0.3,
                "firing_mode": "spindle",
                "conscious":   False,
                "sharpening":  0.5,
            }

        # Normal gating: pass jika novelty di atas consciousness_threshold
        passes = novelty_score >= self.consciousness_threshold

        return {
            "pass":        passes,
            "gain":        gain,
            "firing_mode": self.firing_mode,
            "conscious":   self.is_conscious_input(novelty_score),
            "sharpening":  round(sharpening, 3),
        }

    def get_status(self) -> dict:
        """Status thalamus untuk UI/monitoring."""
        return {
            "firing_mode":             self.firing_mode,
            "attention_gain":          self.attention_gain,
            "consciousness_threshold": self.consciousness_threshold,
            "spindle_active":          self._spindle_active,
        }


# ── Singleton (ThalamusNode ─ Fase 2) ──────────────────────────────────────
thalamus_node = ThalamusNode()
