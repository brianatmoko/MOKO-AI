"""
MOKO Speculative Pre-loader — Sistem Input Pra-Fix
===================================================
Menebak dan menyiapkan context sebelum user menekan tombol Send / Enter.

Cara Kerja:
1. Saat user mengetik di keyboard, UI memanggil suggest_speculation(text).
2. Debounce handler mendeteksi jika user berhenti mengetik sejenak (>400ms).
3. Jika input > 6 karakter, background task berjalan untuk menghitung embedding
   dan mencari memori sesi / Omni Index berdasarkan teks parsial tersebut.
4. Hasil pencarian disimpan di RAM cache.
5. Saat user klik Send, CognitiveWorker langsung mengambil cache tersebut.
   Waktu tunggu loading pre-processing terpangkas menjadi 0 ms!
"""
import time
from PyQt6.QtCore import QRunnable, QThreadPool
from moko_agents.router import router
from moko_agents.llm_engine import engine
from moko_memory.disk_manager import DiskManager
from moko_memory.conv_buffer import session_buffer
from moko_config import settings

# Global disk manager instance (di-share)
_disk_mgr = DiskManager(settings.WORKSPACE_DIR)

class SpeculativeCache:
    def __init__(self):
        self._cached_prompt = ""
        self._cached_data = None
        self._ts = 0.0
        self._is_running = False

    def put(self, prompt: str, data: dict):
        self._cached_prompt = prompt
        self._cached_data = data
        self._ts = time.time()
        self._is_running = False

    def get(self, prompt: str) -> dict:
        """
        Ambil context spekulatif jika prompt yang dikirim mirip 
        atau cocok dengan tebakan input parsial sebelumnya.
        """
        now = time.time()
        # Cache valid selama 15 detik
        if now - self._ts > 15.0:
            return None
        
        # Cek kecocokan (apakah prompt user mengandung draft parsial)
        p_clean = prompt.strip().lower()
        c_clean = self._cached_prompt.strip().lower()
        
        if c_clean and (c_clean in p_clean or p_clean in c_clean):
            return self._cached_data
        return None

    def set_running(self, state: bool):
        self._is_running = state

    def is_running(self) -> bool:
        return self._is_running


# Singleton cache
speculative_cache = SpeculativeCache()


class SpeculationTask(QRunnable):
    def __init__(self, partial_text: str):
        super().__init__()
        self.text = partial_text

    def run(self):
        try:
            # 1. Routing cepat
            path, reason, route_meta = router.classify_intent(self.text)
            
            # Jika hanya sapaan/FAST_PATH pendek, skip speculative embedding
            if path == "FAST_PATH" and route_meta.get("domain") == "personal" and len(self.text) <= 15:
                speculative_cache.set_running(False)
                return

            # 2. Hitung embedding di background thread
            emb = engine.get_embedding(self.text)
            if not emb or len(emb) != 768:
                speculative_cache.set_running(False)
                return

            # 3. Cari memori sesi
            session_context = session_buffer.get_relevant_context(emb, top_k=3)

            # 4. Cari Omni Index
            omni_result = _disk_mgr.omni_first_search(
                emb, top_k=7, target_domain=route_meta.get("domain")
            )

            # Simpan hasil kalkulasi berat ini ke cache spekulatif
            result_data = {
                "emb": emb,
                "session_context": session_context,
                "omni_result": omni_result,
                "route_meta": route_meta,
                "path": path,
            }
            speculative_cache.put(self.text, result_data)
            print(f"[SPECULATION] ✅ Pre-processing siap untuk: \"{self.text[:30]}...\"")

        except Exception as e:
            print(f"[SPECULATION ERROR] {e}")
            speculative_cache.set_running(False)


def trigger_speculation(text: str):
    """Memicu speculative background loading jika belum berjalan."""
    txt = text.strip()
    if len(txt) < 8 or speculative_cache.is_running():
        return

    # Jangan spekulasi jika user sedang mengetik slash command
    if txt.startswith('/'):
        return

    speculative_cache.set_running(True)
    task = SpeculationTask(txt)
    QThreadPool.globalInstance().start(task)
