"""
MOKO Test Runner Sandbox
========================
Modul untuk mengeksekusi file unit test secara otomatis dalam sandbox terisolasi,
menangkap asersi gagal, traceback run-time, dan mengembalikan log kesalahan
untuk disuapkan ke loop self-healing MOKO OS.
"""
import subprocess
import shutil
import os
from typing import Dict, Any, List

class TestRunner:
    def __init__(self):
        self.node_path = shutil.which("node")
        self.python_path = shutil.which("python3") or shutil.which("python")

    def execute_test(self, test_file_path: str) -> Dict[str, Any]:
        """
        Menjalankan pengujian (Python atau JS) dan mengembalikan hasil detail.
        """
        if not os.path.exists(test_file_path):
            return {
                "ok": False,
                "errors": [f"File test tidak ditemukan di path: {test_file_path}"],
                "output": ""
            }

        ext = os.path.splitext(test_file_path)[1].lower()
        if ext == ".py":
            return self._run_python_test(test_file_path)
        elif ext == ".js":
            return self._run_javascript_test(test_file_path)
        else:
            return {
                "ok": False,
                "errors": [f"Tipe file test tidak didukung: {ext}"],
                "output": ""
            }

    def _run_python_test(self, path: str) -> Dict[str, Any]:
        """Jalankan unit test Python."""
        if not self.python_path:
            return {"ok": False, "errors": ["Interpreter Python tidak ditemukan di PATH."], "output": ""}
            
        try:
            # Gunakan subprocess run untuk mengeksekusi script test secara mandiri
            res = subprocess.run(
                [self.python_path, path],
                text=True,
                capture_output=True,
                timeout=5,
                cwd=os.path.dirname(path)
            )
            ok = res.returncode == 0
            errors = []
            
            if not ok:
                # Tangkap baris error/traceback terakhir
                lines = res.stderr.splitlines()
                # Ekstrak traceback krusial
                relevant_err = [line for line in lines if line.strip()]
                errors = relevant_err[-4:] if len(relevant_err) > 4 else relevant_err
                if not errors:
                    errors = [res.stdout.strip()]
            
            return {
                "ok": ok,
                "errors": errors,
                "output": res.stdout + "\n" + res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "errors": ["Pengujian Python timeout (melampaui 5 detik)."], "output": ""}
        except Exception as e:
            return {"ok": False, "errors": [f"Gagal menjalankan test Python: {str(e)}"], "output": ""}

    def _run_javascript_test(self, path: str) -> Dict[str, Any]:
        """Jalankan unit test Javascript menggunakan runner bawaan node."""
        if not self.node_path:
            return {"ok": False, "errors": ["Runtime Node.js tidak ditemukan di PATH."], "output": ""}

        try:
            # Jalankan: node --test path/to/file.js (jika didukung) atau node path/to/file.js
            # Untuk kompatibilitas luas, jalankan script langsung dengan asersi manual
            res = subprocess.run(
                [self.node_path, path],
                text=True,
                capture_output=True,
                timeout=5,
                cwd=os.path.dirname(path)
            )
            ok = res.returncode == 0
            errors = []
            
            if not ok:
                lines = res.stderr.splitlines()
                relevant_err = [line for line in lines if line.strip()]
                errors = relevant_err[-4:] if len(relevant_err) > 4 else relevant_err
                if not errors:
                    errors = [res.stdout.strip()]
                    
            return {
                "ok": ok,
                "errors": errors,
                "output": res.stdout + "\n" + res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "errors": ["Pengujian JS timeout (melampaui 5 detik)."], "output": ""}
        except Exception as e:
            return {"ok": False, "errors": [f"Gagal menjalankan test JS: {str(e)}"], "output": ""}
