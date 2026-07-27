import subprocess
from moko_puzzles.base_puzzle import BasePuzzle

class OSControlPuzzle(BasePuzzle):
    name = "os_control"
    description = "Linux OS System Controller"
    version = "1.0.0"

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        os_keywords = [
            "cek disk", "ruang penyimpanan", "perintah bash", "eksekusi perintah",
            "uptime", "cek ram", "proses yang berjalan", "df -h", "free -h"
        ]
        if any(kw in q for kw in os_keywords):
            return 0.90
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        
        # Mapping command aman secara sederhana
        cmd = ""
        if "disk" in q or "df -h" in q or "penyimpanan" in q:
            cmd = "df -h / | tail -n 1"
        elif "ram" in q or "free -h" in q or "memori" in q:
            cmd = "free -h"
        elif "uptime" in q:
            cmd = "uptime -p"
        elif "proses" in q:
            cmd = "ps aux --sort=-%cpu | head -n 6"
        else:
            # Jika user meminta eksekusi perintah kustom, batasi perintah dasar saja
            cmd = "echo 'Akses perintah dibatasi.'"
            
        try:
            res = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=5)
            output = f"Command yang Dijalankan: {cmd}\nSTDOUT:\n{res.stdout}\n"
            if res.stderr:
                output += f"STDERR:\n{res.stderr}\n"
            confidence = 1.0
        except Exception as e:
            output = f"Gagal mengeksekusi perintah '{cmd}': {e}"
            confidence = 0.0
            
        return {
            "facts": f"=== SISTEM LINUX OS DATA ===\n{output}",
            "confidence": confidence,
            "metadata": {"command": cmd}
        }
