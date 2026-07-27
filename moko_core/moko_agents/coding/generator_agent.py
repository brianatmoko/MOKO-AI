"""
MOKO Generator Agent (Thread 3)
===============================
Fokus: Membuat modul, kelas, dan fungsi pemrograman baru berdasarkan spesifikasi natural language.
"""

import re
from typing import Dict

class GeneratorAgent:
    """
    Agen spesialis penulisan kode baru (Code Generation).
    Menerjemahkan instruksi natural language menjadi kode bersih, efisien,
    dan sesuai standar arsitektur MOKO OS.
    """

    def __init__(self, coder_agent=None):
        self.coder = coder_agent
        self.system_prompt = """Kamu adalah MOKO Generator Agent — sub-agent spesialis penulisan kode baru.
Tugas kamu adalah menerjemahkan instruksi, rancangan, atau deskripsi natural language menjadi kode pemrograman utuh (Python / C++).
Aturan wajib:
1. Kode harus memiliki tipe data yang jelas (type hinting di Python, statically typed di C++).
2. Tulis docstring dan komentar bahasa Indonesia yang informatif.
3. Selalu pertimbangkan performa dan penggunaan memori yang efisien.
4. Berikan HANYA kode di dalam block markdown ``` agar mudah di-copy oleh orchestrator.
"""

    def process(self, description: str, language: str = "python", context: str = "") -> Dict:
        """
        Membuat kode baru berdasarkan deskripsi.
        """
        results = {
            "generated_code": "",
            "raw_response": ""
        }
        
        if self.coder:
            context_str = f"\nKonteks proyek saat ini:\n{context}\n" if context else ""
            prompt = (
                f"Tulis kode {language} lengkap berdasarkan deskripsi berikut:\n\n"
                f"{description}\n"
                f"{context_str}\n"
                f"Pastikan kode siap digunakan, modular, dan bersih."
            )
            response = self.coder._call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.05
            )
            results["raw_response"] = response
            
            # Ekstrak kode dari markdown block
            code_block_match = re.search(r'```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)\n```', response, re.DOTALL)
            if code_block_match:
                results["generated_code"] = code_block_match.group(1)
            else:
                results["generated_code"] = response

        return results
