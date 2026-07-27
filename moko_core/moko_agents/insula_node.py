"""
MOKO Agent: InsulaNode — Interoception, Cognitive Monitoring & Salience Map
==========================================================================
Berdasarkan:
  - Stanford Medicine (2024), Menon et al.: Insular cortex as a hub for saliency and cognitive control.
  - PMC (2024): Interoceptive training impacts neural circuit connectivity.
  - Examining the neural correlates of error awareness: Insula activation in error consciousness.
"""

import time
from typing import Dict, List, Any, Optional
from moko_cpu.governor import CPUGovernor

class InsulaNode:
    """
    InsulaNode — Hub Interoception Kognitif & Fisik MOKO OS.
    Menerapkan:
      - hardware_interoception: integrasi suhu CPU, RAM, VRAM.
      - cognitive_interoception: uncertainty, surprisal, conflict.
      - salience_map: menghitung kegunaan & urgensi kueri input.
      - error_awareness: melacak discrepancy (expected vs actual).
    """
    def __init__(self):
        self.hardware_interoception: Dict[str, float] = {}
        self.cognitive_interoception: Dict[str, float] = {
            "surprisal": 0.0,
            "conflict": 0.0,
            "error_rate": 0.0
        }
        self.salience_history: List[float] = []
        self._expected_outcome: Any = None
        self._actual_outcome: Any = None

    def update_interoception(self, conflict_score: float = 0.0, surprisal_score: float = 0.0):
        """Update hardware & cognitive states."""
        # 1. Hardware Interoception
        try:
            vitals = CPUGovernor.read_vitals()
            self.hardware_interoception = {
                "cpu_temp": vitals.get("cpu_temp", 50.0),
                "cpu_pct": vitals.get("cpu_pct", 20.0),
                "ram_pct": vitals.get("ram_pct", 30.0),
                "vram_used": vitals.get("vram_used", 1000.0)
            }
        except Exception:
            self.hardware_interoception = {
                "cpu_temp": 50.0,
                "cpu_pct": 20.0,
                "ram_pct": 30.0,
                "vram_used": 1000.0
            }

        # 2. Cognitive Interoception
        self.cognitive_interoception["conflict"] = conflict_score
        self.cognitive_interoception["surprisal"] = surprisal_score

    def compute_salience(self, query: str, novelty_score: float, arousal_score: float) -> float:
        """
        Menghitung salience_score (0.0 - 1.0) untuk kueri saat ini.
        Semakin penting/mendesak kueri, semakin tinggi salience score-nya.
        """
        # Novelty & Arousal weight
        salience = 0.4 * novelty_score + 0.4 * arousal_score
        
        # Goal Relevance (panjang/kompleksitas query)
        goal_len = len(query)
        goal_factor = min(1.0, goal_len / 150)
        salience += 0.2 * goal_factor

        # Hardware Strain penalty (jika sistem sangat panas/kritis, turunkan salience)
        temp = self.hardware_interoception.get("cpu_temp", 50.0)
        if temp > 80.0:
            salience *= 0.7  # Kurangi fokus kognitif saat hardware mendidih
            
        salience_score = round(max(0.0, min(1.0, salience)), 4)
        self.salience_history.append(salience_score)
        if len(self.salience_history) > 50:
            self.salience_history.pop(0)
            
        return salience_score

    def set_expectation(self, expectation: Any):
        """Mencatat outcome yang diekspektasikan oleh PFC/ACC."""
        self._expected_outcome = expectation

    def record_actual_outcome(self, actual: Any) -> float:
        """
        Mencatat outcome aktual dan menghitung awareness discrepancy.
        """
        self._actual_outcome = actual
        if not self._expected_outcome or not self._actual_outcome:
            return 0.0
            
        # Perbandingan sederhana
        if str(self._expected_outcome).strip().lower() == str(self._actual_outcome).strip().lower():
            discrepancy = 0.0
        else:
            # Hitung perbedaan string kasar
            exp_str = str(self._expected_outcome).strip().lower()
            act_str = str(self._actual_outcome).strip().lower()
            discrepancy = 1.0 - (len(set(exp_str.split()) & set(act_str.split())) / max(1, len(set(exp_str.split()) | set(act_str.split()))))
            
        # Update cognitive error rate
        alpha = 0.1
        self.cognitive_interoception["error_rate"] = round(
            (1.0 - alpha) * self.cognitive_interoception["error_rate"] + alpha * discrepancy, 4
        )
        return round(discrepancy, 4)

    def get_status(self) -> Dict:
        return {
            "hardware": self.hardware_interoception,
            "cognitive": self.cognitive_interoception,
            "avg_salience": round(sum(self.salience_history)/max(1, len(self.salience_history)), 3) if self.salience_history else 0.0
        }

# Singleton instance
insula_node = InsulaNode()
