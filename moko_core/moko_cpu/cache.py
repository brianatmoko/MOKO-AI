import json
import hashlib
from pathlib import Path
from moko_config import settings

class InstantCache:
    """
    Sistem Memori Refleks.
    Menghindari LLM memproses pertanyaan yang sama dua kali.
    Jika ada di cache, jawab seketika (0.1 detik).
    """
    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "reflex_memory.json"
        
        if self.cache_file.exists():
            try:
                self.memory = json.loads(self.cache_file.read_text())
            except Exception:
                self.memory = {}
        else:
            self.memory = {}

    def _hash(self, text: str) -> str:
        # Menghapus spasi dan huruf besar/kecil agar lebih toleran
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_response(self, question: str) -> str:
        """Mengambil jawaban dari memori refleks jika ada."""
        q_hash = self._hash(question)
        return self.memory.get(q_hash, None)

    def store_response(self, question: str, answer: str):
        """Menyimpan jawaban baru ke memori refleks."""
        q_hash = self._hash(question)
        self.memory[q_hash] = answer
        
        # Batasi memori refleks hingga 1000 entri untuk efisiensi RAM/Disk
        if len(self.memory) > 1000:
            # Hapus 10% entri lama secara kasar
            keys_to_remove = list(self.memory.keys())[:100]
            for k in keys_to_remove:
                del self.memory[k]
                
        try:
            self.cache_file.write_text(json.dumps(self.memory, indent=2))
        except Exception as e:
            print(f"[CACHE ERROR] Gagal menyimpan memori refleks: {e}")

# Global access point
reflex_cache = InstantCache()
