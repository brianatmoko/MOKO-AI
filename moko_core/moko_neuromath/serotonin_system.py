"""
MOKO NeuroMath: Serotonin System (Raphe Nuclei) — Emotional Homeostasis
======================================================================
Berdasarkan:
  - Serotonin & Cognitive Flexibility (Doya, 2002; Cools et al., 2008)
  - Serotonin in Delay Gratification (Miyazaki et al., 2011)
  - Mood Baseline & Valence modulation

FUNGSI:
  1. Serotonin Level: Homeostasis kognitif berdasarkan keberhasilan respons.
  2. Patience Score: Mengatur "delay gratification" — waktu tunggu kognitif.
  3. Mood Baseline: Memodulasi default sentiment valence di AmygdalaNode.
  4. Cognitive Flexibility: Memodulasi tingkat eksplorasi (temperatur Softmax).
"""

import json
import time
from pathlib import Path
from typing import Optional

from moko_config import settings

# ── Konstanta Homeostasis ─────────────────────────────────────────────────────
SEROTONIN_DECAY = 0.05       # Laju peluruhan alami level serotonin
SUCCESS_BOOST   = 0.10       # Kenaikan serotonin setelah respons sukses/akurat
FAILURE_PENALTY = 0.15       # Penurunan serotonin saat terjadi error/inhibisi


class SerotoninNode:
    """
    SerotoninNode — Mengatur tingkat ketahanan emosional (mood), 
    kesabaran (patience), dan fleksibilitas kognitif.
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.state_path = Path(
            workspace_dir or settings.WORKSPACE_DIR
        ) / ".math_omni" / "serotonin_state.json"
        
        self._load_state()

    def _load_state(self):
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.serotonin_level = float(raw.get("serotonin_level", 0.70))
                self.success_streak   = int(raw.get("success_streak", 0))
                return
            except Exception:
                pass
        
        # Default state
        self.serotonin_level = 0.70
        self.success_streak   = 0

    def _save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({
                    "serotonin_level": round(self.serotonin_level, 4),
                    "success_streak":  self.success_streak,
                    "updated_at":      time.time()
                }, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    # ── Public APIs ───────────────────────────────────────────────────────────

    def record_outcome(self, success: bool):
        """Catat hasil akhir dari evaluasi kognitif (PFC/ACC)."""
        # Meluruhkan secara alami
        self.serotonin_level = max(0.1, self.serotonin_level - SEROTONIN_DECAY)

        if success:
            self.serotonin_level = min(1.0, self.serotonin_level + SUCCESS_BOOST)
            self.success_streak += 1
        else:
            self.serotonin_level = max(0.1, self.serotonin_level - FAILURE_PENALTY)
            self.success_streak = 0
            
        self._save_state()

    @property
    def patience_score(self) -> float:
        """
        Delay Gratification (0.0 s.d. 1.0):
        Tingkat kesabaran MOKO. Jika serotonin tinggi -> sabar -> 
        siap menunggu/menambah passing komputasi kognitif yang mendalam.
        Jika serotonin rendah (pessimistic) -> terburu-buru -> kurangi passes kognitif.
        """
        # Kesabaran meningkat seiring tingginya serotonin
        return round(0.2 + 0.8 * self.serotonin_level, 4)

    @property
    def mood_baseline(self) -> float:
        """
        Mood Baseline (-1.0 s.d. +1.0):
        Memodulasi valence di AmygdalaNode.
        Serotonin > 0.6 -> mood positif (boost sentiment ke arah senang).
        Serotonin < 0.4 -> mood negatif (tarik sentiment ke arah sedih/curiga).
        """
        # Map: 0.0 -> -0.5, 0.5 -> 0.0, 1.0 -> +0.5
        mood = (self.serotonin_level - 0.5) * 1.0
        return round(max(-0.5, min(0.5, mood)), 4)

    @property
    def cognitive_flexibility(self) -> float:
        """
        Fleksibilitas Kognitif (0.0 s.d. 1.0):
        Mempengaruhi temperatur Softmax pada pemilihan aksi BasalGanglia.
        Serotonin optimal (~0.7) -> fleksibilitas tinggi -> siap eksplorasi.
        Serotonin sangat rendah -> kaku -> terpaku pada kueri yang sama (bias konfirmasi).
        """
        # Serotonin tinggi meningkatkan toleransi perubahan topik
        return round(0.1 + 0.9 * self.serotonin_level, 4)

    def get_status(self) -> dict:
        return {
            "serotonin_level":      round(self.serotonin_level, 4),
            "patience_score":       self.patience_score,
            "mood_baseline":        self.mood_baseline,
            "cognitive_flexibility":self.cognitive_flexibility,
            "success_streak":       self.success_streak
        }


# Singleton Instance
serotonin_node = SerotoninNode()
