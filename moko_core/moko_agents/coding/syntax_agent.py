"""
MOKO Syntax Agent (Thread 1)
============================
Fokus: Memeriksa, merapikan, dan memvalidasi sintaksis kode (sintaksis C++/Python).
Neuro-symbolic: Menggabungkan AST compiler check dengan LLM perbaikan sintaks.
"""

import ast
import re
from typing import Dict, Tuple

class SyntaxAgent:
    """
    Agen spesialis sintaksis. Memastikan kode bebas dari error kompilasi/parsing,
    memperbaiki indentasi, bracket balancing, dan standard style.
    """

    def __init__(self, coder_agent=None):
        self.coder = coder_agent  # Instance MokoCoderAgent dari singleton jika dipass
        
        self.system_prompt = """Kamu adalah MOKO Syntax Agent — sub-agent spesialis sintaksis pemrograman.
Tugas utama kamu adalah memeriksa aturan sintaksis, membenarkan indentasi, memperbaiki tanda baca kode (semicolon, bracket, quotes), dan merapikan layout kode agar bersih dan valid.
Jangan menjelaskan konsep logika program. Fokus HANYA pada pembetulan sintaksis yang rusak.
Berikan penjelasan singkat mengenai perbaikan sintaksis apa saja yang kamu lakukan.
"""

    def check_python_syntax(self, code: str) -> Tuple[bool, str]:
        """Validasi sintaksis Python menggunakan engine AST bawaan."""
        try:
            ast.parse(code)
            return True, "Sintaksis Python Valid."
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (Line {e.lineno}, Col {e.offset})\nContext: {e.text.strip() if e.text else 'None'}"

    def check_brackets_balanced(self, code: str) -> Tuple[bool, str]:
        """Pemeriksaan bracket balancing cepat (kurung kurawal, siku, bulat)."""
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        lines = code.split("\n")
        
        for idx, line in enumerate(lines, 1):
            # Abaikan string literal dan komentar
            clean_line = re.sub(r'".*?"|\'.*?\'|//.*|#.*', '', line)
            for char in clean_line:
                if char in mapping.values():
                    stack.append((char, idx))
                elif char in mapping.keys():
                    if not stack:
                        return False, f"Kelebihan bracket '{char}' di baris {idx}"
                    top, start_idx = stack.pop()
                    if mapping[char] != top:
                        return False, f"Mismatch bracket: '{char}' di baris {idx} tidak cocok dengan '{top}' dari baris {start_idx}"
        
        if stack:
            top, start_idx = stack.pop()
            return False, f"Bracket '{top}' di baris {start_idx} tidak pernah ditutup"
            
        return True, "Semua bracket seimbang (balanced)."

    def process(self, code: str, language: str = "python") -> Dict:
        """
        Memproses pemeriksaan dan perbaikan kode.
        """
        results = {
            "valid": True,
            "checks": {},
            "repaired_code": code,
            "explanation": "Tidak ada perbaikan sintaksis yang diperlukan."
        }
        
        # 1. Jalankan symbolic checks
        if language.lower() == "python":
            py_ok, py_msg = self.check_python_syntax(code)
            results["checks"]["ast_compile"] = py_msg
            if not py_ok:
                results["valid"] = False
                
        bracket_ok, bracket_msg = self.check_brackets_balanced(code)
        results["checks"]["brackets"] = bracket_msg
        if not bracket_ok:
            results["valid"] = False

        # 2. Jika ada error, panggil LLM untuk memperbaikinya
        if not results["valid"] and self.coder:
            prompt = f"Perbaiki sintaksis kode {language} berikut agar kompilasinya berhasil:\n\n```\n{code}\n```\n\nDetail error:\n{results['checks']}"
            repaired = self.coder._call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.0
            )
            results["repaired_code"] = repaired
            results["explanation"] = "Memperbaiki bracket mismatch/syntax error menggunakan LLM."

        return results
