"""
MOKO Cross Verifier
===================
Modul untuk melakukan verifikasi keterkaitan antar-file (cross-reference)
dalam proyek multi-file Puzzle System.
"""
import os
import re
from typing import List, Dict, Any
from moko_marathon.code_verifier import VerifyResult

class CrossVerifier:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def verify_project_references(self, blueprint: Dict[str, Any]) -> VerifyResult:
        """
        Memeriksa semua referensi antar-file yang tertera di blueprint.
        Memastikan:
        1. File-file target benar-benar ada di disk.
        2. Tag rujukan (link/script) di HTML sesuai dengan file yang dideklarasikan.
        """
        errors = []
        project_name = blueprint.get("project_name", "moko_puzzle_app")
        files = blueprint.get("files", [])

        # Kumpulkan semua path file yang seharusnya ada
        expected_paths = {os.path.join(self.workspace_dir, f["path"]) for f in files}

        # 1. Cek keberadaan file di disk
        for file_info in files:
            full_path = os.path.join(self.workspace_dir, file_info["path"])
            if not os.path.exists(full_path):
                errors.append(f"Cross-File Error: File '{file_info['path']}' tidak ditemukan di disk.")
                continue

            # Baca isi file untuk analisis cross-reference
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                errors.append(f"Cross-File Error: Gagal membaca file '{file_info['path']}': {str(e)}")
                continue

            # 2. Jika tipe HTML, cek tag link/script
            if file_info["type"] == "html":
                # Cari src di <script>
                script_srcs = re.findall(r'<script\b[^>]*\bsrc=[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
                for src in script_srcs:
                    # Rujukan lokal/relatif
                    if not src.startswith("http") and not src.startswith("//"):
                        ref_path = os.path.normpath(os.path.join(os.path.dirname(full_path), src))
                        if ref_path not in expected_paths and not os.path.exists(ref_path):
                            errors.append(
                                f"Cross-File Error: HTML '{file_info['path']}' merujuk script '{src}' "
                                f"yang tidak terdaftar di proyek atau tidak ditemukan."
                            )

                # Cari href di <link rel="stylesheet">
                link_hrefs = re.findall(r'<link\b[^>]*\bhref=[\'"]([^\'"]+)[\'"]', content, re.IGNORECASE)
                for href in link_hrefs:
                    if not href.startswith("http") and not href.startswith("//"):
                        ref_path = os.path.normpath(os.path.join(os.path.dirname(full_path), href))
                        if ref_path not in expected_paths and not os.path.exists(ref_path):
                            errors.append(
                                f"Cross-File Error: HTML '{file_info['path']}' merujuk stylesheet '{href}' "
                                f"yang tidak terdaftar di proyek atau tidak ditemukan."
                            )

        return VerifyResult(ok=len(errors) == 0, errors=errors)
