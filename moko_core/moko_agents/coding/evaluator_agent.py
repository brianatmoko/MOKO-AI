"""
MOKO Evaluator Agent (Thread 4)
===============================
Fokus: Melakukan code review, penilaian kualitas, dan estimasi kompleksitas kode.
Neuro-symbolic: Menghitung metrik kompleksitas secara matematis + review kualitatif LLM.
"""

import re
from typing import Dict, Tuple

class EvaluatorAgent:
    """
    Agen spesialis evaluasi kualitas kode.
    Menghitung metrik kompleksitas, duplikasi, kepatuhan arsitektur,
    dan memberikan score penilaian (1-10) serta feedback perbaikan.
    """

    def __init__(self, coder_agent=None):
        self.coder = coder_agent
        self.system_prompt = """Kamu adalah MOKO Evaluator Agent — sub-agent spesialis penilaian kualitas kode dan code review.
Tugas kamu adalah mengevaluasi kode yang diberikan:
1. Berikan nilai kualitas keseluruhan (1-10).
2. Analisis keterbacaan, kebersihan kode (clean code), dan potensi edge-case.
3. Berikan saran refactoring nyata untuk meningkatkan keterbacaan atau performa.
Berikan feedback secara konstruktif dan teknis tinggi.
"""

    def estimate_complexity(self, code: str) -> Tuple[int, str]:
        """
        Estimasi kompleksitas siklomatis (cyclomatic complexity) sederhana.
        Menghitung control flow statements: if, for, while, except, with, and, or.
        """
        # Standar basis kompleksitas = 1
        complexity = 1
        
        # Cari control keywords
        keywords = [r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bexcept\b', r'\bwith\b', r'\band\b', r'\bor\b', r'\bcase\b']
        for kw in keywords:
            matches = re.findall(kw, code)
            complexity += len(matches)
            
        # Rating klasifikasi
        if complexity <= 5:
            rating = "Rendah (Sangat baik, mudah dimengerti)"
        elif complexity <= 10:
            rating = "Sedang (Baik, pertimbangkan untuk membagi jika bertambah besar)"
        elif complexity <= 20:
            rating = "Tinggi (Rumit, direkomendasikan refactoring/modularisasi)"
        else:
            rating = "Sangat Tinggi (Sangat kompleks, risiko bug tinggi, wajib refactoring)"
            
        return complexity, rating

    def get_code_stats(self, code: str) -> Dict:
        """Menghitung data statistik dasar dari kode."""
        lines = code.split("\n")
        total_lines = len(lines)
        
        # Cari baris komentar
        comment_lines = 0
        blank_lines = 0
        for line in lines:
            line_str = line.strip()
            if not line_str:
                blank_lines += 1
            elif line_str.startswith("#") or line_str.startswith("//") or line_str.startswith("/*") or line_str.startswith("*"):
                comment_lines += 1
                
        code_lines = total_lines - comment_lines - blank_lines
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "comment_ratio": f"{round((comment_lines / (total_lines + 0.1)) * 100, 1)}%"
        }

    def process(self, code: str, language: str = "python") -> Dict:
        """
        Mengevaluasi kode.
        """
        results = {
            "stats": self.get_code_stats(code),
            "complexity_score": 1,
            "complexity_rating": "",
            "review": "",
            "score": 5.0
        }
        
        # 1. Hitung metrik kompleksitas
        comp_val, comp_rate = self.estimate_complexity(code)
        results["complexity_score"] = comp_val
        results["complexity_rating"] = comp_rate
        
        # 2. Panggil LLM untuk review kualitatif
        if self.coder:
            prompt = (
                f"Lakukan evaluasi dan code review untuk kode {language} berikut:\n\n"
                f"```\n{code}\n```\n\n"
                f"Statistik Kode:\n{results['stats']}\n"
                f"Nilai Kompleksitas Siklomatis Estimasi: {comp_val} ({comp_rate})"
            )
            response = self.coder._call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.1
            )
            results["review"] = response
            
            # Cari score rating (e.g. 8/10 atau "Score: 8")
            score_match = re.search(r'\b(?:score|penilaian|nilai|rating)\b:?\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?', response, re.IGNORECASE)
            if score_match:
                results["score"] = float(score_match.group(1))

        return results
