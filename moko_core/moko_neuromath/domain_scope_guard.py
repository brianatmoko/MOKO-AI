"""
MOKO OS — Domain Scope Guard
=============================
Sistem verifikasi domain untuk mendeteksi ketidakcocokan antara domain pertanyaan
user dengan domain asal data OMNI yang ditarik (cross-domain contamination).
Mencegah data fisika teoretis bocor ke pertanyaan otomotif, dsb.
"""

import re
from typing import List, Dict, Tuple

class DomainScopeGuard:
    # Definisi kedekatan/relasi antar domain
    # Nilai penalty: 0.0 (cocok sempurna) s.d. 1.0 (sangat tidak cocok)
    DOMAIN_RELATIONS = {
        "math": {"physics": 0.10, "code": 0.15, "reasoning": 0.10},
        "physics": {"math": 0.10, "code": 0.20, "reasoning": 0.15},
        "code": {"math": 0.15, "physics": 0.20, "reasoning": 0.10},
        "reasoning": {"math": 0.10, "physics": 0.15, "code": 0.10},
        "lexical": {"reasoning": 0.20},
    }

    @staticmethod
    def infer_chunk_domain(chunk: Dict) -> str:
        """Menebak domain dari sebuah OMNI chunk berdasarkan sumber atau isinya."""
        source = str(chunk.get("source", "")).lower()
        file_path = str(chunk.get("file", "")).lower()
        text = str(chunk.get("text", "")).lower()

        # Cek berdasarkan nama sumber/file
        if "arxiv" in source or "arxiv" in file_path:
            # ArXiv bisa math, physics, atau AI (reasoning/code)
            if any(k in text for k in ["physics", "quantum", "gravity", "relativity", "mechanics"]):
                return "physics"
            if any(k in text for k in ["math", "theorem", "lemma", "algebra", "calculus"]):
                return "math"
            return "reasoning"  # AI/CS/Reasoning

        if "python" in source or "mdn" in source or "devdocs" in source or "github" in source:
            return "code"
        if "semanticscholar" in source or "openalex" in source:
            if any(k in text for k in ["quantum", "gravity", "thermodynamics", "physics"]):
                return "physics"
            if any(k in text for k in ["algorithm", "neural", "deep learning", "complexity"]):
                return "code"
            return "reasoning"

        # Cek heuristik keyword di teks chunk
        code_kws = ["def ", "function", "class ", "import ", "const ", "var ", "void ", "public class"]
        if any(kw in text for kw in code_kws):
            return "code"

        math_kws = ["\\lambda", "integral", "derivative", "matrix", "equation", "theorem"]
        if any(kw in text for kw in math_kws):
            return "math"

        # Default fallback
        return "general"

    def evaluate_and_sanitize(
        self,
        query: str,
        query_meta: Dict,
        omni_results: List[Dict],
        current_confidence: float
    ) -> Tuple[List[Dict], float]:
        """
        Evaluasi kecocokan domain antara query user dan hasil pencarian OMNI.
        Menerapkan penalty pada score chunk dan confidence jika terjadi ketidakcocokan.
        """
        if not omni_results:
            return [], current_confidence

        target_domain = (query_meta or {}).get("domain", "general")
        if target_domain == "general":
            # Jika domain target general, kita tidak menerapkan strict domain matching
            return omni_results, current_confidence

        sanitized_results = []
        max_adjusted_score = 0.0

        for chunk in omni_results:
            chunk_domain = self.infer_chunk_domain(chunk)
            score = chunk.get("score", 0.0)

            # Hitung penalty
            penalty = 0.0
            if chunk_domain != "general" and chunk_domain != target_domain:
                # Cek relasi
                rel = self.DOMAIN_RELATIONS.get(target_domain, {})
                if chunk_domain in rel:
                    penalty = rel[chunk_domain]  # Penalty ringan (adjacent domain)
                else:
                    penalty = 0.35  # Penalty berat (mismatch total)

            # Terapkan penalty pada score chunk
            adjusted_score = max(0.0, score - penalty)
            
            # Jika adjusted score masih layak, simpan
            if adjusted_score > 0.30:  # Threshold kelayakan chunk setelah penalty
                new_chunk = chunk.copy()
                new_chunk["score"] = adjusted_score
                new_chunk["domain_scope_verified"] = (penalty == 0.0)
                new_chunk["inferred_domain"] = chunk_domain
                sanitized_results.append(new_chunk)
                if adjusted_score > max_adjusted_score:
                    max_adjusted_score = adjusted_score

        # Urutkan ulang berdasarkan score baru
        sanitized_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Sesuaikan confidence global berdasarkan max adjusted score
        adjusted_confidence = min(current_confidence, max_adjusted_score)

        return sanitized_results, adjusted_confidence

# Singleton instance
domain_scope_guard = DomainScopeGuard()
