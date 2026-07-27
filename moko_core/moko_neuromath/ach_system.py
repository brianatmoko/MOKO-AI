"""
MOKO NeuroMath: Acetylcholine System (ACh / Basal Forebrain) — Neuromodulation
=============================================================================
Berdasarkan:
  - Hasselmo (2006): Role of Acetylcholine in Learning and Memory
  - Yu & Dayan (2005): Acetylcholine in Expected Uncertainty
  - ACh levels dictate encoding vs retrieval mode

FUNGSI:
  1. ACh Level: Mengatur sensitivitas terhadap kejutan/novelty.
  2. Attention Gate: Multiplier untuk attention gain di ThalamusNode.
  3. Encoding Boost: Multiplier untuk Hebbian learning rate (LTP rate).
"""

import time
from typing import Optional


class AcetylcholineSystem:
    """
    AcetylcholineSystem — Mengatur tingkat kewaspadaan kognitif, 
    modulasi atensi, dan penguat encoding memori baru.
    """

    def __init__(self):
        self.ach_level: float = 0.5   # Default alert state
        self.last_updated: float = time.time()

    def update_ach(self, novelty_score: float):
        """
        Update level Asetilkolin (ACh) berdasarkan novelty yang dibaca.
        Novelty tinggi -> Lepaskan ACh (kewaspadaan meningkat).
        """
        # Meluruhkan secara alami ke baseline 0.4
        elapsed = time.time() - self.last_updated
        decay = 0.8 ** (elapsed / 30.0)  # half-life ~30 detik
        self.ach_level = 0.4 + (self.ach_level - 0.4) * decay
        
        # Boost berdasarkan novelty
        if novelty_score > 0.6:
            # ACh naik tajam untuk novelty tinggi
            boost = (novelty_score - 0.6) * 0.5
            self.ach_level = min(1.0, self.ach_level + boost)
            
        self.last_updated = time.time()

    def enter_sleep_mode(self):
        """ACh diturunkan saat tidur kognitif (NREM sleep consolidation mode)."""
        self.ach_level = 0.15
        self.last_updated = time.time()

    def exit_sleep_mode(self):
        """Reset ACh kembali ke normal setelah bangun tidur."""
        self.ach_level = 0.5
        self.last_updated = time.time()

    @property
    def attention_gate(self) -> float:
        """
        Attention Gate Multiplier (0.5 s.d. 1.5):
        Digunakan oleh ThalamusNode untuk memperkuat/mempersempit gain atensi.
        ACh tinggi -> Multiplier tinggi -> Gating Thalamus lebih selektif.
        """
        # ACh 0.0 -> 0.5, ACh 0.5 -> 1.0, ACh 1.0 -> 1.5
        return round(0.5 + self.ach_level, 4)

    @property
    def encoding_boost(self) -> float:
        """
        Encoding Boost Multiplier (0.8 s.d. 1.8):
        Digunakan oleh HebbLinker untuk memperkuat nilai LTP (laju belajar).
        ACh tinggi -> LTP diperkuat -> ingatan baru terserap lebih tajam.
        ACh rendah (sleep) -> LTP normal/rendah -> cocok untuk decay/homeostasis.
        """
        return round(0.8 + self.ach_level, 4)

    def get_status(self) -> dict:
        return {
            "ach_level":      round(self.ach_level, 4),
            "attention_gate": self.attention_gate,
            "encoding_boost": self.encoding_boost
        }


# Singleton Instance
ach_system = AcetylcholineSystem()
