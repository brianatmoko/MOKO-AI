"""
MOKO NeuroMath: Synergy Router
================================
Berdasarkan: Bernstein's Motor Synergies Theory (1967)
             "Degrees of Freedom Problem" & Synergy Recombination

Ketika tidak ada formula Math-Omni yang persis cocok dengan situasi baru,
SynergyRouter TIDAK memanggil LLM dari nol.

Ia mengambil 2-3 formula yang paling mendekati dari berbagai kategori,
lalu menggabungkannya menjadi satu "Synergy" (instruksi gabungan baru)
secara instan — persis seperti otak mengambil pola gerakan yang sudah dikenal
dan merekombinasikannya menjadi gerakan penyelamatan baru saat jatuh.

TIGA FASE BERNSTEIN (diimplementasikan secara matematis):
  1. FREEZE: Batasi pencarian ke kategori logika yang paling yakin (DOF minimum)
  2. RELEASE: Perluas ke kateogori tetangga jika hasil pertama kurang
  3. EXPLOIT: Gabungkan semua kandidat menjadi Synergy yang efisien
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from moko_agents.llm_engine import engine
from moko_memory.math_omni import math_omni, fp32_to_qev, qev_dot
from moko_config import settings

SYNERGY_MIN_CANDIDATES = 2  # Minimal 2 formula untuk merekombinasi
SYNERGY_FREEZE_THRESHOLD = 0.50  # Jika skor terbaik < ini, mulai "Release DOF"

SYNERGY_PROMPT = """
Kamu adalah MOKO Synergy Recombinator.
Kamu diberikan BEBERAPA KERANGKA LOGIKA yang berbeda.
Tugasmu adalah MENGGABUNGKAN kerangka-kerangka tersebut menjadi satu
instruksi tunggal yang sinergis untuk menjawab situasi baru ini.

SITUASI BARU (Pertanyaan User):
"{query}"

KERANGKA LOGIKA TERSEDIA:
{formula_list}

Gabungkan menjadi satu instruksi berpikir yang mencakup kelebihan setiap kerangka.
Instruksi harus KONKRET, ACTIONABLE, dan sesuai konteks pertanyaan.

KEMBALIKAN HANYA OBJEK JSON:
{{
  "synergy_instruction": "Instruksi gabungan sinergis yang mencakup semua kerangka...",
  "dominant_logic": "E",
  "reasoning": "Mengapa kombinasi ini tepat..."
}}
"""


class SynergyRouter:
    """
    Mesin Rekombinasi Synergy — Respons AI terhadap situasi benar-benar baru.
    """

    def __init__(self):
        self.omni_dir = Path(settings.WORKSPACE_DIR) / ".math_omni"

    def _search_multi_category(self, query_emb: List[float],
                                excluded_logic: str = "") -> List[Dict]:
        """
        Cari kandidat formula dari SEMUA kategori logika (A-H),
        kecuali kategori yang sudah diuji dan gagal.
        Ini adalah fase "Release DOF" Bernstein.
        """
        all_logic = ["A", "B", "C", "D", "E", "F", "G", "H"]
        candidates = []
        query_qev = fp32_to_qev(query_emb)

        for logic in all_logic:
            if logic == excluded_logic:
                continue
            for bin_file in self.omni_dir.glob(f"{logic}*.bin"):
                try:
                    db_qev = bin_file.read_bytes()
                    score = qev_dot(query_qev, db_qev)
                    if score > 0.15:  # Filter yang terlalu tidak relevan
                        formula_id = bin_file.stem
                        candidates.append({
                            "formula_id": formula_id,
                            "score": round(score, 4),
                            "logic": logic
                        })
                except:
                    pass

        # Urutkan berdasarkan skor, ambil top-3
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:3]

    def _enrich_with_instructions(self, candidates: List[Dict]) -> List[Dict]:
        """Tambahkan instruksi dari sidecar untuk setiap kandidat."""
        sidecar = self.omni_dir / "formula_sidecar.jsonl"
        index = {}
        if sidecar.exists():
            with open(sidecar, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        index[data.get("id", "")] = data.get("instruction", "")
                    except:
                        pass

        for c in candidates:
            c["instruction"] = index.get(c["formula_id"], "")

        return [c for c in candidates if c["instruction"]]

    def recombine(self, query: str, query_emb: List[float],
                  failed_logic: str = "") -> Optional[Dict]:
        """
        Lakukan Synergy Recombination.
        Dipanggil saat pencarian Math-Omni utama gagal memuaskan FEP Engine.

        Returns:
            {synergy_instruction, formula_ids_used, dominant_logic} atau None jika gagal
        """
        # Fase 1: FREEZE — Coba kategori primer dulu
        candidates = self._search_multi_category(query_emb, excluded_logic=failed_logic)

        if len(candidates) < SYNERGY_MIN_CANDIDATES:
            return None  # Tidak cukup bahan untuk Synergy

        candidates = self._enrich_with_instructions(candidates)
        if len(candidates) < SYNERGY_MIN_CANDIDATES:
            return None

        # Susun daftar formula untuk prompt LLM
        formula_list_str = ""
        for i, c in enumerate(candidates, 1):
            formula_list_str += f"\n{i}. [{c['formula_id']} | Skor: {c['score']:.2f}]\n   {c['instruction']}\n"

        # Fase 3: EXPLOIT — Rekombinasi
        prompt = SYNERGY_PROMPT.format(query=query, formula_list=formula_list_str)
        response = engine.generate_text(prompt, "Return JSON only.",
                                         model_override=settings.MODEL_ANALYST)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group(0))
            synergy_instruction = data.get("synergy_instruction", "")
            dominant_logic = data.get("dominant_logic", candidates[0]["logic"])

            if not synergy_instruction:
                return None

            return {
                "synergy_instruction": synergy_instruction,
                "formula_ids_used": [c["formula_id"] for c in candidates],
                "dominant_logic": dominant_logic,
                "candidate_scores": [c["score"] for c in candidates],
            }
        except:
            return None


# Singleton
synergy_router = SynergyRouter()
