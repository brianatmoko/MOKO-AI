"""
MOKO Git Sandbox Manager
========================
Modul untuk mengelola repositori Git lokal sementara di dalam subdirektori proyek.
Menyediakan kemampuan commit otomatis segmen yang stabil dan rollback otomatis
jika LLM mengalami jalan buntu (dead-end) dalam loop perbaikan.
"""
import subprocess
import shutil
import os

class GitSandboxManager:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.git_path = shutil.which("git")

    def initialize_repo(self) -> bool:
        """Inisialisasi repositori Git kosong di folder proyek."""
        if not self.git_path:
            return False
            
        try:
            # Pastikan folder proyek ada
            os.makedirs(self.project_dir, exist_ok=True)
            
            # Jika sudah ada .git, abaikan
            if os.path.exists(os.path.join(self.project_dir, ".git")):
                return True
                
            res = subprocess.run([self.git_path, "init"], cwd=self.project_dir, capture_output=True, text=True)
            if res.returncode == 0:
                # Konfigurasi username dan email lokal agar commit tidak eror
                subprocess.run([self.git_path, "config", "user.name", "MokoAgent"], cwd=self.project_dir)
                subprocess.run([self.git_path, "config", "user.email", "moko@os.local"], cwd=self.project_dir)
                return True
            return False
        except Exception as e:
            print(f"Git init error: {e}")
            return False

    def commit_segment(self, segment_name: str) -> bool:
        """Commit status proyek setelah segmen lolos verifikasi."""
        if not self.git_path or not os.path.exists(os.path.join(self.project_dir, ".git")):
            return False
            
        try:
            # Git add
            subprocess.run([self.git_path, "add", "."], cwd=self.project_dir, capture_output=True)
            
            # Git commit
            msg = f"feat: compile segment {segment_name} successfully"
            res = subprocess.run([self.git_path, "commit", "-m", msg], cwd=self.project_dir, capture_output=True, text=True)
            return res.returncode == 0
        except Exception as e:
            print(f"Git commit error: {e}")
            return False

    def rollback_to_last_commit(self) -> bool:
        """Rollback seluruh perubahan kembali ke commit terakhir (HEAD)."""
        if not self.git_path or not os.path.exists(os.path.join(self.project_dir, ".git")):
            return False
            
        try:
            # Cek apakah ada commit sama sekali
            check_commits = subprocess.run([self.git_path, "log", "-1"], cwd=self.project_dir, capture_output=True)
            if check_commits.returncode != 0:
                # Belum ada commit pertama sama sekali, bersihkan file-file tidak terlacak saja
                subprocess.run([self.git_path, "clean", "-fd"], cwd=self.project_dir, capture_output=True)
                return True

            # Reset hard ke HEAD
            res = subprocess.run([self.git_path, "reset", "--hard", "HEAD"], cwd=self.project_dir, capture_output=True, text=True)
            # Bersihkan file untracked
            subprocess.run([self.git_path, "clean", "-fd"], cwd=self.project_dir, capture_output=True)
            return res.returncode == 0
        except Exception as e:
            print(f"Git rollback error: {e}")
            return False
