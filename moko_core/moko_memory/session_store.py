"""
MOKO Persistent Session Store — Memori Percakapan Lintas Sesi
=============================================================
Menyimpan riwayat percakapan ke disk dalam format JSONL agar
tidak hilang saat aplikasi ditutup dan dibuka kembali.

Format file: .moko_session.jsonl
  - Satu baris per giliran percakapan (user + moko)
  - Append-only untuk performa I/O maksimal
  - Rotasi otomatis saat melebihi MAX_TURNS

Keamanan:
  - Max MAX_TURNS giliran per file. Selebihnya dirotasi ke .old
  - Setiap baris divalidasi JSON sebelum dimuat (skip jika rusak)
  - Thread-safe: menggunakan threading.Lock()
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional


MAX_TURNS       = 500   # Maksimum giliran tersimpan sebelum rotasi
EXPORT_LIMIT    = 15    # Default jumlah giliran terakhir untuk diinjeksi ke prompt


class SessionStore:
    """
    Persistent store untuk riwayat percakapan MOKO.

    Cara pakai:
        store = SessionStore(path)
        store.save_turn("siapa kamu?", "Saya MOKO.")
        turns = store.load_recent(20)
        context_str = store.export_for_llm(10)
        messages = store.get_openai_messages(10)  # Format OpenAI API
    """

    def __init__(self, session_path: Path):
        self.path = Path(session_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: List[Dict] = []  # In-memory cache setelah load
        self._loaded = False

    # ─── Private ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load dari disk sekali saja saat pertama kali diakses."""
        if not self._loaded:
            self._cache = self._read_all()
            self._loaded = True

    def _read_all(self) -> List[Dict]:
        """Baca semua giliran dari file JSONL, skip baris yang rusak."""
        turns = []
        if not self.path.exists():
            return turns
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Validasi struktur minimum
                        if "user" in obj and "moko" in obj:
                            turns.append(obj)
                    except json.JSONDecodeError:
                        continue  # Skip baris rusak
        except Exception as e:
            print(f"[SessionStore] Gagal membaca sesi: {e}")
        return turns

    def _rotate_if_needed(self):
        """Rotasi file ke .old jika melebihi MAX_TURNS."""
        if len(self._cache) >= MAX_TURNS:
            old_path = self.path.with_suffix(".jsonl.old")
            try:
                import shutil
                shutil.move(str(self.path), str(old_path))
                # Pertahankan 50 giliran terakhir di file baru
                last_50 = self._cache[-50:]
                self._cache = last_50
                self._rewrite_cache()
                print(f"[SessionStore] Rotasi sesi: {len(last_50)} giliran tersisa, lama disimpan ke {old_path.name}")
            except Exception as e:
                print(f"[SessionStore] Gagal rotasi sesi: {e}")

    def _rewrite_cache(self):
        """Tulis ulang seluruh cache ke file (setelah rotasi)."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for turn in self._cache:
                    f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[SessionStore] Gagal menulis ulang sesi: {e}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def save_turn(self, user_text: str, moko_text: str):
        """
        Simpan satu giliran percakapan ke disk secara thread-safe.
        Append-only untuk I/O cepat.
        """
        with self._lock:
            self._ensure_loaded()
            self._rotate_if_needed()

            record = {
                "ts":   time.time(),
                "user": user_text.strip(),
                "moko": moko_text.strip()[:2000]  # Batasi agar file tidak terlalu besar
            }
            self._cache.append(record)

            # Append ke file — tidak perlu rewrite seluruh file
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[SessionStore] Gagal menyimpan giliran: {e}")

    def load_recent(self, n: int = 20) -> List[Dict]:
        """Kembalikan N giliran terakhir dari memori cache."""
        with self._lock:
            self._ensure_loaded()
            return self._cache[-n:] if self._cache else []

    def export_for_llm(self, n: int = EXPORT_LIMIT) -> str:
        """
        Format N giliran terakhir sebagai teks yang siap diinjeksi
        ke system prompt LLM sebagai konteks sesi.
        """
        turns = self.load_recent(n)
        if not turns:
            return ""

        lines = ["=== RIWAYAT PERCAKAPAN SESI SEBELUMNYA ==="]
        for t in turns:
            user_prev = t["user"][:300]
            moko_prev = t["moko"][:600]
            lines.append(f"User: {user_prev}")
            lines.append(f"MOKO: {moko_prev}")
            lines.append("")

        lines.append(
            "INSTRUKSI: Gunakan riwayat di atas untuk menjawab secara konsisten. "
            "Ingat nama, fakta, dan konteks yang sudah pernah dibahas. "
            "Jika user merujuk ke sesuatu dari percakapan sebelumnya, kamu HARUS ingat."
        )
        return "\n".join(lines)

    def get_openai_messages(self, n: int = EXPORT_LIMIT) -> List[Dict]:
        """
        Kembalikan N giliran terakhir dalam format OpenAI messages array:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        
        Format ini digunakan oleh Auto-Continue Engine untuk mengirim
        riwayat lengkap ke LLM saat melanjutkan generasi yang terpotong.
        """
        turns = self.load_recent(n)
        messages = []
        for t in turns:
            messages.append({"role": "user",      "content": t["user"]})
            messages.append({"role": "assistant", "content": t["moko"]})
        return messages

    @property
    def turn_count(self) -> int:
        """Jumlah giliran yang tersimpan di cache."""
        with self._lock:
            self._ensure_loaded()
            return len(self._cache)

    def clear(self):
        """Hapus seluruh sesi (reset percakapan). Tidak bisa di-undo."""
        with self._lock:
            self._cache = []
            self._loaded = True
            try:
                if self.path.exists():
                    self.path.unlink()
                print("[SessionStore] Sesi berhasil direset.")
            except Exception as e:
                print(f"[SessionStore] Gagal menghapus sesi: {e}")

    def get_last_moko_response(self) -> Optional[str]:
        """Kembalikan respons MOKO terakhir (berguna untuk Auto-Continue)."""
        with self._lock:
            self._ensure_loaded()
            if self._cache:
                return self._cache[-1].get("moko", "")
            return None

    def update_last_moko_response(self, new_moko_text: str):
        """
        Update teks MOKO di giliran TERAKHIR — digunakan oleh Auto-Continue
        untuk menggabungkan partial + continuation menjadi satu entri utuh.
        """
        with self._lock:
            self._ensure_loaded()
            if not self._cache:
                return
            self._cache[-1]["moko"] = new_moko_text.strip()[:2000]
            # Tulis ulang file untuk sinkronisasi
            self._rewrite_cache()


# ── Singleton Session Store ───────────────────────────────────────────────────
# Diinisialisasi dengan path dari settings saat pertama kali diimport.
# Lazy init: path diisi oleh cognitive_worker saat pertama digunakan.
_store_instance: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Singleton getter — inisialisasi sekali, digunakan di mana-mana."""
    global _store_instance
    if _store_instance is None:
        from moko_config import settings
        session_path = Path(settings.WORKSPACE_DIR) / ".moko_session.jsonl"
        _store_instance = SessionStore(session_path)
        print(f"[SessionStore] Sesi dimuat dari: {session_path}")
        print(f"[SessionStore] Riwayat percakapan tersimpan: {_store_instance.turn_count} giliran")
    return _store_instance
