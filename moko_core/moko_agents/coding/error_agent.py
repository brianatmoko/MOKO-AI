"""
MOKO Error Agent (Thread 2)
===========================
Fokus: Menganalisis log error, traceback python/C++, dan membetulkan bug logika/runtime.
"""

import re
from typing import Dict, Tuple

class ErrorAgent:
    """
    Agen spesialis penanganan bug dan crash.
    Menganalisis traceback dan runtime error, memberikan analisis penyebab,
    serta merumuskan solusi perbaikan kode.
    """

    def __init__(self, coder_agent=None):
        self.coder = coder_agent
        self.system_prompt = """Kamu adalah MOKO Error Agent — sub-agent spesialis debugging dan perbaikan error runtime.
Tugas kamu adalah menganalisis error log, traceback, compiler error, atau bug perilaku pada program.
Lakukan hal berikut:
1. Identifikasi tipe error (e.g. AttributeError, NullPointerException, Segfault).
2. Jelaskan penyebab error tersebut berdasarkan baris kode yang bermasalah.
3. Berikan perbaikan kode yang bersih dan teruji.
Sajikan analisis secara langsung dan profesional tanpa basa-basi umum.
"""

    def parse_python_traceback(self, error_msg: str) -> Dict[str, str]:
        """Ekstraksi tipe error, baris error, dan file dari python traceback."""
        info = {"type": "Unknown", "line": "Unknown", "file": "Unknown", "message": ""}
        
        # Cari baris error terakhir (e.g. ValueError: invalid literal)
        last_line_match = re.search(r'^([a-zA-Z0-9_]+Error|Exception):\s*(.*)$', error_msg, re.MULTILINE)
        if last_line_match:
            info["type"] = last_line_match.group(1)
            info["message"] = last_line_match.group(2)
            
        # Cari lokasi error terakhir (e.g. File "script.py", line 42, in <module>)
        file_line_matches = re.findall(r'File "([^"]+)", line (\d+)', error_msg)
        if file_line_matches:
            # Ambil yang paling bawah (biasanya merupakan root cause di crash traceback)
            info["file"], info["line"] = file_line_matches[-1]
            
        return info

    def process(self, code: str, error_msg: str, language: str = "python") -> Dict:
        """
        Menganalisis bug dan mereparasi kode.
        """
        results = {
            "parsed_info": {},
            "root_cause": "Tidak terdeteksi otomatis.",
            "repaired_code": code,
            "solution": ""
        }
        
        # 1. Parse log error jika python
        if "traceback" in error_msg.lower() or language.lower() == "python":
            parsed = self.parse_python_traceback(error_msg)
            results["parsed_info"] = parsed
            if parsed["type"] != "Unknown":
                results["root_cause"] = f"Terjadi error '{parsed['type']}' pada file '{parsed['file']}' baris {parsed['line']}: {parsed['message']}"

        # 2. Kirim ke LLM untuk debugging & perbaikan komprehensif
        if self.coder:
            prompt = (
                f"Analisis dan perbaiki bug berikut:\n\n"
                f"Bahasa: {language}\n"
                f"Kode:\n```\n{code}\n```\n\n"
                f"Error Message/Log:\n{error_msg}\n\n"
                f"Hasil Parse Lokasi Error:\n{results['root_cause']}"
            )
            response = self.coder._call(
                prompt=prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.1
            )
            results["solution"] = response
            
            # Ekstrak kode yang diperbaiki jika ada di dalam markdown blocks
            code_block_match = re.search(r'```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)\n```', response, re.DOTALL)
            if code_block_match:
                results["repaired_code"] = code_block_match.group(1)

        return results
