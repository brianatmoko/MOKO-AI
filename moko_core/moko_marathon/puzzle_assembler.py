"""
MOKO Puzzle Assembler
=====================
Orkestrator utama eksekusi proyek multi-file.
Menyusun urutan penulisan file berdasarkan dependensi (Topological Sort),
memanggil Marathon Pit Stop per file, dan menjalankan cross-file verification.
"""
import os
import time
from typing import List, Dict, Any
from moko_marathon.marathon_pitstop import MarathonPitStop, CodeSegment
from moko_marathon.cross_verifier import CrossVerifier

class PuzzleAssembler:
    def __init__(self, session_id: str, workspace_dir: str):
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.pit_runner = MarathonPitStop(session_id)
        self.cross_verifier = CrossVerifier(workspace_dir)

    def build_project(self, blueprint: Dict[str, Any], on_breath = None) -> str:
        """
        Membangun proyek berdasarkan blueprint terstruktur.
        """
        project_name = blueprint.get("project_name", "moko_puzzle_app")
        files = blueprint.get("files", [])

        if on_breath:
            on_breath(f"🧩 [Puzzle Assembler] Memulai assembly proyek '{project_name}'...")

        # 1. Topological Sort untuk menyelesaikan dependensi
        sorted_files = self._topological_sort(files)
        if on_breath:
            on_breath("🧬 [Puzzle Assembler] Urutan kompilasi terhitung:")
            for f in sorted_files:
                on_breath(f"  ➡️ {f['path']}")

        # 2. Eksekusi penulisan per file menggunakan Marathon Pit Stop
        for file_info in sorted_files:
            file_path = os.path.join(self.workspace_dir, file_info["path"])
            
            # Buat segmen dinamis berdasarkan tipe file
            segments = self._build_segments_for_file(file_info)
            
            if on_breath:
                on_breath(f"\n📂 [Puzzle Assembler] Memproses file: {file_info['path']}...")

            # Run Marathon Pit Stop untuk menulis file ini
            # Gunakan penulisan modular
            self.pit_runner.run_code_marathon(
                spec=f"Buat file {file_info['path']} ({file_info['description']}) untuk proyek {project_name}.",
                segments=segments,
                output_path=file_path,
                max_pit_retries=2,
                on_breath=on_breath
            )

        # 3. Cross-file Verification di akhir
        if on_breath:
            on_breath("\n🔍 [Puzzle Assembler] Menjalankan Cross-file Verification...")
            
        cross_res = self.cross_verifier.verify_project_references(blueprint)
        if cross_res.ok:
            success_msg = f"🏆 [PUZZLE COMPLETE] Proyek '{project_name}' sukses dirakit dengan {len(files)} file sinkron!"
            if on_breath:
                on_breath(success_msg)
            return success_msg
        else:
            fail_msg = f"⚠️ [PUZZLE WARNING] Perakitan selesai tapi ada peringatan cross-reference:\n" + "\n".join([f"- {e}" for e in cross_res.errors])
            if on_breath:
                on_breath(fail_msg)
            return fail_msg

    def _topological_sort(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Urutkan file berdasarkan dependensi (file mandiri diproses terlebih dahulu)."""
        adj = {}
        in_degree = {}
        file_map = {}

        # Inisialisasi
        for f in files:
            path = f["path"]
            file_map[path] = f
            adj[path] = []
            in_degree[path] = 0

        # Bangun graf
        for f in files:
            path = f["path"]
            deps = f.get("depends_on", [])
            for dep in deps:
                # Normalisasi path
                dep_path = dep
                # Jika dependensi terdaftar di proyek, tambahkan edge
                if dep_path in adj:
                    adj[dep_path].append(path)
                    in_degree[path] += 1

        # Cari node dengan in-degree 0 (tidak punya dependensi)
        queue = [p for p in in_degree if in_degree[p] == 0]
        sorted_paths = []

        while queue:
            node = queue.pop(0)
            sorted_paths.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Tambahkan sisa file jika ada siklus dependensi sirkular (fallback)
        for path in file_map:
            if path not in sorted_paths:
                sorted_paths.append(path)

        return [file_map[p] for p in sorted_paths]

    def _build_segments_for_file(self, file_info: Dict[str, Any]) -> List[CodeSegment]:
        """Membuat segmen instruksi khusus berdasarkan ekstensi/tipe file."""
        path = file_info["path"]
        ftype = file_info["type"]
        desc = file_info["description"]
        ext = os.path.splitext(path)[1].lower()

        if ftype == "html" or ext == ".html":
            return [
                CodeSegment(
                    name="HTML_PAGE",
                    prompt=f"Tulis kode HTML lengkap untuk {path} ({desc}). Pastikan untuk melink file CSS dan script JS yang ada di blueprint menggunakan tag <link href=\"...\"> dan <script src=\"...\">. Jangan menulis CSS/JS inline.",
                    verify_types=["html_structure"]
                )
            ]
        elif ftype == "css" or ext == ".css":
            return [
                CodeSegment(
                    name="CSS_STYLES",
                    prompt=f"Tulis kode CSS lengkap untuk {path} ({desc}). Desain glassmorphism dark space premium dengan neon glows. Seimbangkan semua bracket.",
                    verify_types=["css_braces"]
                )
            ]
        elif ftype == "javascript" or ext == ".js":
            # Cek jika file berisi matematika
            v_types = ["js_syntax"]
            if "math" in path.lower() or "formula" in path.lower():
                v_types.extend(["math_lorenz", "math_rossler", "math_aizawa"])
            return [
                CodeSegment(
                    name="JS_CODE",
                    prompt=f"Tulis kode Javascript lengkap untuk {path} ({desc}). Ini adalah file .js STANDALONE — JANGAN gunakan <script> tags. Tulis pure JavaScript saja. Pastikan semua variabel dan fungsi dideklarasikan dengan benar.",
                    verify_types=v_types
                )
            ]
        elif ftype == "python" or ext == ".py":
            return [
                CodeSegment(
                    name="PYTHON_SCRIPT",
                    prompt=(
                        f"Tulis kode Python LENGKAP untuk {path} ({desc}).\n"
                        f"ATURAN PENTING:\n"
                        f"- Ini adalah file Python (.py) — tulis HANYA kode Python, BUKAN HTML/CSS/JS\n"
                        f"- Gunakan Python standard library saja (os, sys, re, dll.)\n"
                        f"- Untuk menulis file HTML, gunakan Python string dan f.write()\n"
                        f"- Awali file dengan #!/usr/bin/env python3 dan encoding comment\n"
                        f"- Tutup dengan if __name__ == '__main__': main()"
                    ),
                    verify_types=["python_syntax"],
                    num_predict=2000
                )
            ]
        else:
            return [
                CodeSegment(
                    name="RAW_FILE",
                    prompt=f"Tulis isi file untuk {path} ({desc}) secara lengkap tanpa markdown wrappers.",
                    verify_types=[]
                )
            ]
