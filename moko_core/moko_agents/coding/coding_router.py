"""
MOKO Coding Router
==================
Heuristic router ultra-cepat (<1ms) untuk menentukan thread/agent
pemrograman yang sesuai untuk memproses query user.
"""

import re
from typing import Tuple

class CodingRouter:
    """
    Router berbasis aturan (rule-based) berkecepatan tinggi.
    Mencocokkan pola bahasa alami dan kode untuk mengidentifikasi intent pemrograman.
    """

    def __init__(self):
        # Pola reguler untuk setiap intent agen
        self.patterns = {
            "SYNTAX": re.compile(
                r"\b(syntax|sintaks|format|indent|pep8|linter|autoformat|lint|kurung|bracket|semicolon|titik koma|spasi|tab)\b", 
                re.IGNORECASE
            ),
            "ERROR": re.compile(
                r"\b(error|bug|debug|crash|traceback|exception|runtime|segfault|core dump|failed|gagal|salah|fix|perbaiki|penyebab|mengapa|kenapa)\b", 
                re.IGNORECASE
            ),
            "EVALUATE": re.compile(
                r"\b(review|evaluasi|nilai|kualitas|readability|kejelasan|static analysis|analisa statis|audit|keamanan|vulnerability|celah)\b", 
                re.IGNORECASE
            ),
            "SYSTEM": re.compile(
                r"\b(optimize|optimasi|performa|performance|speed|cepat|lambat|slow|leak|bocor|memory|ram|vram|cpu|cache|thread|parallel|throughput|low-level|profiling)\b", 
                re.IGNORECASE
            ),
            # Default fallback jika tidak cocok adalah GENERATE (pembuatan kode baru)
            "GENERATE": re.compile(
                r"\b(buat|tulis|write|generate|implement|implementasikan|create|bikin|coding|code)\b", 
                re.IGNORECASE
            )
        }

    def route(self, query: str) -> Tuple[str, float]:
        """
        Menentukan agent yang paling cocok untuk query tertentu.
        
        Returns:
            Tuple[str, float]: (Agent Name, Confidence Score)
        """
        scores = {intent: 0.0 for intent in self.patterns}
        
        # Hitung kecocokan keyword/pattern
        for intent, pattern in self.patterns.items():
            matches = pattern.findall(query)
            if matches:
                # Berikan bobot berdasarkan jumlah kecocokan
                scores[intent] = len(matches) * 1.0

        # Jika query mengandung output compiler/traceback secara eksplisit
        if "Traceback (most recent call last)" in query or "error:" in query.lower() or "exception:" in query.lower():
            scores["ERROR"] += 5.0
            
        # Jika query secara spesifik menanyakan tentang syntax error/kesalahan sintaks
        if re.search(r'\bsyntax\s+error\b|\bsintaks\s+error\b|\bkesalahan\s+sintaks\b', query, re.IGNORECASE):
            scores["SYNTAX"] += 3.0
            
        # Jika query berisi code blocks kosong atau perintah format eksplisit
        if "```" in query and any(w in query.lower() for w in ["format", "rapikan", "bersihkan"]):
            scores["SYNTAX"] += 3.0

        # Cari score tertinggi
        best_intent = "GENERATE"  # default fallback
        best_score = 0.0
        
        for intent, score in scores.items():
            if score > best_score:
                best_score = score
                best_intent = intent
                
        # Normalisasi confidence score sederhana
        total_score = sum(scores.values())
        confidence = 1.0
        if total_score > 0:
            confidence = min(best_score / (total_score + 0.1) + 0.3, 1.0)
            
        return best_intent, confidence
