"""
MOKO Puzzle Planner
===================
Modul untuk merancang blueprint struktur file, folder, dan dependensi
dari sebuah kueri pembuatan aplikasi/proyek multi-file kompleks.
"""
import re
import json
import os
from typing import List, Dict, Any
from moko_agents.llm_engine import engine

class PuzzlePlanner:
    def __init__(self, session_id: str):
        self.session_id = session_id

    def generate_blueprint(self, spec: str, on_breath = None) -> Dict[str, Any]:
        """
        Menganalisis kueri user dan menghasilkan blueprint proyek terstruktur (JSON).
        """
        if on_breath:
            on_breath("💡 [Puzzle Planner] Menganalisis kueri untuk merancang skema proyek...")

        sys_prompt = (
            "You are a Principal Software Architect. "
            "You design project structures for web applications. "
            "Your task is to break down the user request into a modular file plan. "
            "You MUST output a valid JSON object only. No markdown formatting, no explanation."
        )

        prompt = (
            f"User Request:\n{spec}\n\n"
            "TUGAS: Pecah request di atas menjadi komponen file terpisah (e.g. index.html, style.css, script.js, dsb).\n"
            "Kembalikan skema dalam format JSON dengan format persis berikut:\n"
            "{\n"
            "  \"project_name\": \"nama_project_singkat\",\n"
            "  \"files\": [\n"
            "    {\n"
            "      \"path\": \"relative/path/to/file.ext\",\n"
            "      \"type\": \"html|css|javascript|python\",\n"
            "      \"description\": \"penjelasan singkat peran file ini\",\n"
            "      \"depends_on\": [\"path/file/dependensi.ext\"]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "PENTING: Hanya kembalikan raw JSON. JANGAN gunakan ```json ... ``` wrapper. "
            "Pastikan relative path file diletakkan di dalam subfolder proyek baru (misal 'my_app/index.html')."
        )

        raw_res = engine.generate_text(
            prompt=prompt,
            system_prompt=sys_prompt,
            coop_params={"num_predict": 1000, "enable_thinking": False}
        )

        # Parsing JSON secara aman
        blueprint = self._clean_and_parse_json(raw_res)
        
        if on_breath:
            on_breath(f"📋 [Puzzle Planner] Blueprint selesai. Menemukan {len(blueprint.get('files', []))} file untuk ditulis.")
            for f in blueprint.get("files", []):
                on_breath(f"  📎 {f['path']} ({f['description']})")

        return blueprint

    def _clean_and_parse_json(self, text: str) -> Dict[str, Any]:
        """Pembersihan string agar JSON parse sukses."""
        cleaned = text.strip()
        # Bersihkan markdown wrappers jika ada
        cleaned = re.sub(r'^```[a-zA-Z0-9]*\n', '', cleaned)
        cleaned = re.sub(r'\n```$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception as e:
            # Fallback jika parsing gagal: sediakan struktur dasar minimal
            # agar program tidak crash
            print(f"JSON Parse error in Puzzle Planner: {e}. Raw: {text}")
            return {
                "project_name": "moko_puzzle_app",
                "files": [
                    {
                        "path": "moko_puzzle_app/index.html",
                        "type": "html",
                        "description": "HTML entry point (Fallback)",
                        "depends_on": []
                    }
                ]
            }
