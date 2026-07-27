"""
MOKO NeuroMath: FEP Engine & Predictive Coding Hierarchy
=========================================================
Berdasarkan:
  - Karl Friston's Free Energy Principle (FEP)
  - Causal Cognitive Architecture (BICA), Schneider 2023-2024
  - Predictive Coding Hierarchy: Multi-level processing & error cascade
"""

import json
import re
import math
from typing import Dict, Tuple, Any, Optional
from moko_agents.llm_engine import engine
from moko_config import settings

class PredictiveLevel:
    """Representasi satu level dalam hierarki prediktif."""
    def __init__(self, name: str, update_rate: float = 0.15):
        self.name = name
        self.prior_belief = 0.5  # Prior ekspektasi kecocokan (0.0 - 1.0)
        self.prediction = 0.5
        self.error = 0.0
        self.update_rate = update_rate

    def update(self, actual_outcome: float):
        """Update prior belief berdasarkan error prediksi (actual - prediction)."""
        self.error = round(actual_outcome - self.prediction, 4)
        self.prior_belief = round(max(0.0, min(1.0, self.prior_belief + self.update_rate * self.error)), 4)
        self.prediction = self.prior_belief


class PredictiveHierarchy:
    """
    PredictiveHierarchy — Hierarki Prediktif 5-Level untuk MOKO OS.
    Tingkat hierarki:
      1. Lexical / Token Level (struktur teks mikro)
      2. Syntactic / Logic Route Level (logic, arousal, depth)
      3. Semantic / Concept Level (kesesuaian makna semantik)
      4. Executive / Goal Level (goal compliance & aturan PFC)
      5. World / Context Level (koherensi global & konsistensi jangka panjang)
    """
    def __init__(self):
        self.levels = {
            1: PredictiveLevel("Lexical", update_rate=0.2),
            2: PredictiveLevel("Syntactic", update_rate=0.15),
            3: PredictiveLevel("Semantic", update_rate=0.15),
            4: PredictiveLevel("Executive", update_rate=0.1),
            5: PredictiveLevel("World", update_rate=0.05)
        }

    def compute_hierarchical_surprisal(
        self,
        text: str,
        formula_instruction: str,
        logic_route: Tuple[str, str, str],
        pfc_inhibited: bool,
        semantic_similarity: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Menghitung surprisal di setiap level dan melakukan cascading error ke atas.
        Mengembalikan: (total_surprisal, details_dict)
        """
        # --- Level 1: Lexical ---
        words_instr = set(formula_instruction.lower().split())
        words_text = set(text.lower().split())
        lexical_match = len(words_instr & words_text) / max(1, len(words_instr))
        self.levels[1].update(lexical_match)
        err1 = abs(self.levels[1].error)

        # --- Level 2: Syntactic ---
        # Sederhana: jika logic route valid (misal: "E", "2", "D9"), score 1.0, else 0.0
        logic_match = 1.0 if (logic_route and len(logic_route) == 3 and logic_route[0] in "ABCDEFGH") else 0.0
        # Cascade error level 1 memodulasi prediksi level 2
        self.levels[2].prediction = max(0.0, min(1.0, self.levels[2].prior_belief - 0.1 * err1))
        self.levels[2].update(logic_match)
        err2 = abs(self.levels[2].error)

        # --- Level 3: Semantic ---
        self.levels[3].prediction = max(0.0, min(1.0, self.levels[3].prior_belief - 0.1 * err2))
        self.levels[3].update(semantic_similarity)
        err3 = abs(self.levels[3].error)

        # --- Level 4: Executive (Inhibisi PFC) ---
        executive_match = 0.0 if pfc_inhibited else 1.0
        self.levels[4].prediction = max(0.0, min(1.0, self.levels[4].prior_belief - 0.1 * err3))
        self.levels[4].update(executive_match)
        err4 = abs(self.levels[4].error)

        # --- Level 5: World ---
        world_match = 1.0 - abs(err4)
        self.levels[5].prediction = max(0.0, min(1.0, self.levels[5].prior_belief - 0.1 * err4))
        self.levels[5].update(world_match)
        err5 = abs(self.levels[5].error)

        # Total surprisal: weighted average of absolute errors
        weights = {1: 0.1, 2: 0.15, 3: 0.25, 4: 0.3, 5: 0.2}
        total_surprisal = sum(abs(self.levels[lvl].error) * weights[lvl] for lvl in range(1, 6))

        # Record cognitive surprisal to Locus Coeruleus
        from moko_neuromath.locus_coeruleus import locus_coeruleus
        locus_coeruleus.record_surprisal(total_surprisal)

        details = {
            "Level 1 (Lexical)": {"prior": self.levels[1].prior_belief, "error": self.levels[1].error},
            "Level 2 (Syntactic)": {"prior": self.levels[2].prior_belief, "error": self.levels[2].error},
            "Level 3 (Semantic)": {"prior": self.levels[3].prior_belief, "error": self.levels[3].error},
            "Level 4 (Executive)": {"prior": self.levels[4].prior_belief, "error": self.levels[4].error},
            "Level 5 (World)": {"prior": self.levels[5].prior_belief, "error": self.levels[5].error},
        }

        return round(total_surprisal, 4), details


# Singleton instance
predictive_hierarchy = PredictiveHierarchy()


class FEPEngine:
    """
    Karl Friston's Free Energy Principle Engine.
    Mengukur tingkat "Surprisal" (Ambiguitas/Error) dari penerapan sebuah rumus logika terhadap teks.
    """
    
    EVALUATION_PROMPT = """
Kamu adalah Korteks Prefrontal (FEP Evaluator).
Sebuah Teks Jurnal telah dianalisis menggunakan RUMUS LOGIKA tertentu.

Teks Asli:
"{text}"

Rumus Logika yang Diterapkan:
"{instruction}"

Tugasmu: Evaluasi apakah rumus logika ini berhasil memecahkan teks tersebut tanpa kontradiksi atau ambiguitas (Surprisal).
Berikan skoring 'Surprisal' dari 0.0 hingga 1.0.
- 0.0 = Sempurna, sangat cocok, tidak ada ambiguitas. Rumus ini sempurna untuk teks ini.
- 1.0 = Sangat buruk, banyak kontradiksi, rumus sama sekali tidak nyambung dengan kebutuhan teks.

KEMBALIKAN HANYA OBJEK JSON (tanpa markdown tambahan):
{{
  "surprisal_score": 0.45,
  "reason": "Alasan singkat..."
}}
"""

    SATISFACTION_THRESHOLD = 0.25

    @classmethod
    def calculate_free_energy(cls, text: str, formula_instruction: str, coop_params: dict = None) -> Tuple[float, bool, str]:
        """
        Menghitung Surprisal (F) dan menentukan apakah MOKO "Puas".
        Returns: (surprisal_score, is_satisfied, reason)
        """
        prompt = cls.EVALUATION_PROMPT.format(text=text[:1000], instruction=formula_instruction)
        
        try:
            # Menggunakan model analis karena butuh akurasi logis
            response = engine.generate_text(prompt, "Return JSON only.", model_override=settings.MODEL_ANALYST, coop_params=coop_params)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return 1.0, False, "Gagal mengekstrak JSON dari evaluator FEP."
                
            data = json.loads(json_match.group(0))
            surprisal = float(data.get("surprisal_score", 1.0))
            reason = data.get("reason", "Tidak ada alasan.")
            
            # Record cognitive surprisal to Locus Coeruleus
            from moko_neuromath.locus_coeruleus import locus_coeruleus
            locus_coeruleus.record_surprisal(surprisal)
            
            is_satisfied = surprisal <= cls.SATISFACTION_THRESHOLD
            
            return surprisal, is_satisfied, reason
            
        except Exception as e:
            return 1.0, False, f"FEP Engine Error: {str(e)}"

    @classmethod
    def calculate_free_energy_fast(cls, chunk_emb: list, formula_emb: list) -> float:
        """
        Surprisal = 1 - cosine_similarity(chunk_emb, formula_emb)
        0.0 = rumus sangat cocok (tidak ada kejutan)
        1.0 = rumus tidak cocok (banyak kejutan)
        
        Ini menggantikan LLM call untuk evaluasi FEP saat AWAKE mode.
        LLM FEP tetap tersedia untuk analisis mendalam di SLEEP mode.
        """
        if not chunk_emb or not formula_emb or len(chunk_emb) != len(formula_emb):
            return 1.0
        dot = sum(a * b for a, b in zip(chunk_emb, formula_emb))
        mag1 = math.sqrt(sum(a * a for a in chunk_emb))
        mag2 = math.sqrt(sum(b * b for b in formula_emb))
        if mag1 == 0 or mag2 == 0:
            return 1.0
        sim = dot / (mag1 * mag2)
        surprisal = round(1.0 - sim, 4)
        
        # Record cognitive surprisal to Locus Coeruleus
        from moko_neuromath.locus_coeruleus import locus_coeruleus
        locus_coeruleus.record_surprisal(surprisal)
        
        return surprisal
