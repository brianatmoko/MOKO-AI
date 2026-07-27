"""
MOKO WAL Manager — Write-Ahead Log dengan Rotasi & Checkpoint
=============================================================
Memisahkan logika WAL dari DiskManager agar lebih mudah diuji
dan dikelola secara independen.

Fitur:
  - WAL rotate: setelah N entri, pindahkan ke .wal.bak dan mulai baru
  - Checkpoint: setelah semua entri berhasil di-ingest ke RSA, truncate WAL
  - WAL size limit: cegah file WAL tumbuh tak terbatas di RAM/disk
  - Thread-safe via Lock

Masalah yang diselesaikan:
  Sebelumnya WAL hanya ditambahi (append-only) tanpa pernah dibersihkan.
  Setelah restart, semua entri WAL di-replay ulang meskipun sudah ada di RSA,
  menyebabkan RAM naik terus dan duplikasi data potensial.
"""

import os
import json
import threading
import shutil
from pathlib import Path
from typing import List, Dict, Optional


# Maksimum entri sebelum WAL di-rotate ke file backup
WAL_MAX_ENTRIES = 500

# Maksimum ukuran file WAL dalam byte sebelum di-rotate
WAL_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


class WALManager:
    """
    Mengelola siklus hidup Write-Ahead Log:
      1. Append  — tulis entri baru ke WAL aktif
      2. Rotate  — jika WAL terlalu besar, pindah ke .bak
      3. Checkpoint — setelah berhasil tersimpan ke RSA, hapus WAL
      4. Replay  — saat restart, baca WAL dan kirim ulang ke DiskManager

    Setiap instance terikat ke satu workspace_path.
    """

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.wal_path = self.workspace_path / ".moko_wal.jsonl"
        self.bak_path = self.workspace_path / ".moko_wal.bak.jsonl"
        self._lock = threading.Lock()
        self._entry_count = 0

        # Hitung jumlah entri saat init (tanpa baca seluruh file)
        if self.wal_path.exists():
            try:
                with open(self.wal_path, "r", encoding="utf-8") as f:
                    self._entry_count = sum(1 for line in f if line.strip())
            except Exception:
                self._entry_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def append(
        self,
        source_name: str,
        text: str,
        vector: List[float],
        domain: str = "general",
        valence: float = 0.0,
        arousal: float = 0.5,
        memory_type: str = "semantic",
        consolidated_count: int = 0,
    ) -> bool:
        """Tulis satu entri ke WAL. Thread-safe."""
        record = {
            "source_name": source_name,
            "text": text,
            "vector": vector,
            "domain": domain,
            "valence": valence,
            "arousal": arousal,
            "memory_type": memory_type,
            "consolidated_count": consolidated_count,
        }
        try:
            with self._lock:
                with open(self.wal_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._entry_count += 1

                # Auto-rotate jika WAL terlalu besar
                if self._should_rotate():
                    self._rotate_nolock()

            return True
        except Exception as e:
            print(f"[WALManager] ⚠️ Gagal menulis WAL: {e}")
            return False

    def replay(self) -> List[Dict]:
        """
        Baca semua entri WAL aktif (dan .bak jika ada) untuk di-replay
        saat restart. Mengembalikan list record dict.
        """
        records = []
        for path in [self.bak_path, self.wal_path]:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WALManager] ⚠️ Gagal membaca {path.name}: {e}")
        return records

    def checkpoint(self):
        """
        Hapus WAL setelah semua entri berhasil tersimpan ke RSA.
        Dipanggil oleh DiskManager setelah ingest sukses.
        """
        with self._lock:
            try:
                if self.wal_path.exists():
                    os.remove(self.wal_path)
                if self.bak_path.exists():
                    os.remove(self.bak_path)
                self._entry_count = 0
                print("[WALManager] ✅ WAL checkpoint: file WAL dibersihkan.")
            except Exception as e:
                print(f"[WALManager] ⚠️ Gagal checkpoint WAL: {e}")

    def get_stats(self) -> Dict:
        """Statistik WAL untuk monitoring."""
        wal_size = self.wal_path.stat().st_size if self.wal_path.exists() else 0
        bak_size = self.bak_path.stat().st_size if self.bak_path.exists() else 0
        return {
            "entry_count": self._entry_count,
            "wal_size_mb": round(wal_size / 1024 / 1024, 2),
            "bak_size_mb": round(bak_size / 1024 / 1024, 2),
            "total_size_mb": round((wal_size + bak_size) / 1024 / 1024, 2),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _should_rotate(self) -> bool:
        """Cek apakah WAL perlu di-rotate (dipanggil dalam lock)."""
        if self._entry_count >= WAL_MAX_ENTRIES:
            return True
        if self.wal_path.exists():
            return self.wal_path.stat().st_size >= WAL_MAX_BYTES
        return False

    def _rotate_nolock(self):
        """Rotate WAL → .bak (dipanggil dalam lock yang sudah dipegang)."""
        try:
            if self.bak_path.exists():
                # Gabungkan .bak lama dengan .wal sebelum rotate
                with open(self.bak_path, "a", encoding="utf-8") as bak:
                    if self.wal_path.exists():
                        with open(self.wal_path, "r", encoding="utf-8") as wal:
                            bak.write(wal.read())
                if self.wal_path.exists():
                    os.remove(self.wal_path)
            else:
                # Pindahkan .wal menjadi .bak
                shutil.move(str(self.wal_path), str(self.bak_path))

            self._entry_count = 0
            print(f"[WALManager] 🔄 WAL di-rotate ke {self.bak_path.name} "
                  f"(ukuran: {self.bak_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"[WALManager] ⚠️ Gagal rotate WAL: {e}")
