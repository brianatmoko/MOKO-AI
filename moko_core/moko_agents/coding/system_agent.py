"""
MOKO System Agent (Thread 5)
============================
Fokus: Optimasi sistem, efisiensi memori (VRAM/RAM), performa algoritma,
dan optimasi runtime untuk hardware mobile/RTX 2050.
"""

import re
from typing import Dict, Tuple

class SystemAgent:
    """
    Agen spesialis optimasi performa sistem.
    Menganalisis penggunaan resource, bottleneck I/O, kompleksitas memori,
    dan memberikan instruksi optimasi low-level.
    """

    def __init__(self, coder_agent=None):
        self.coder = coder_agent
        self.system_prompt = """Kamu adalah MOKO System Agent — sub-agent spesialis optimasi performa dan arsitektur sistem.
Tugas kamu adalah menganalisis kode untuk menemukan bottleneck performa, kebocoran memori (memory leak), penggunaan memori yang berlebihan, dan alokasi resource tidak efisien.
Fokus pada batasan hardware RTX 2050 (4GB VRAM) dan CPU multi-core.
Berikan rekomendasi perbaikan berbasis data:
1. Analisis Big O (Time & Space complexity).
2. Potensi bottleneck (e.g. disk write berulang, overhead serialization, lock contention).
3. Kode teroptimasi yang siap menggantikan kode lama.
"""

    def check_loop_depth(self, code: str) -> Tuple[int, str]:
        """Analisis loop bersarang (nested loops) untuk mendeteksi O(N^2) atau lebih tinggi."""
        max_depth = 0
        current_depth = 0
        lines = code.split("\n")
        
        # Cari indentasi loop untuk menghitung kedalaman loop bersarang (Python)
        loop_indents = []
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Bersihkan loop indents yang sudah dilewati
            loop_indents = [ind for ind in loop_indents if ind < indent]
            current_depth = len(loop_indents)
            
            if stripped.startswith("for ") or stripped.startswith("while "):
                loop_indents.append(indent)
                current_depth = len(loop_indents)
                if current_depth > max_depth:
                    max_depth = current_depth
                    
        # Untuk C++ (curly braces loop counting sederhana)
        cpp_loop_matches = re.findall(r'\b(?:for|while)\s*\(.*?\)\s*\{', code)
        if len(cpp_loop_matches) > 1 and max_depth == 0:
            # Estimasi kasar jika C++
            max_depth = len(cpp_loop_matches) # fallback saja
            
        if max_depth >= 3:
            msg = f"Kedalaman loop bersarang tinggi ({max_depth} level). Potensi waktu eksekusi eksponensial/polinomial tinggi."
        elif max_depth == 2:
            msg = "Loop bersarang 2 level. Kompleksitas kemungkinan O(N^2). Pastikan jumlah data kecil."
        else:
            msg = "Kedalaman loop optimal (1 level atau tanpa loop)."
            
        return max_depth, msg

    def process(self, code: str, language: str = "python") -> Dict:
        """
        Mengoptimalkan kode untuk performa sistem.
        """
        results = {
            "loop_depth": 0,
            "loop_warning": "",
            "optimization_review": "",
            "optimized_code": code
        }
        
        # 1. Analisis loop bersarang secara simbolik
        depth, msg = self.check_loop_depth(code)
        results["loop_depth"] = depth
        results["loop_warning"] = msg
        
        # 2. Panggil LLM untuk optimasi performa komprehensif
        if self.coder:
            prompt = (
                f"Optimalkan performa kode {language} berikut untuk runtime yang efisien:\n\n"
                f"```\n{code}\n```\n\n"
                f"Hasil analisa loop awal:\nKedalaman loop: {depth} level\nAnalisis: {msg}"
            )
            response = self.coder._call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.05
            )
            results["optimization_review"] = response
            
            # Ekstrak kode teroptimasi dari markdown block
            code_block_match = re.search(r'```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)\n```', response, re.DOTALL)
            if code_block_match:
                results["optimized_code"] = code_block_match.group(1)

        return results
