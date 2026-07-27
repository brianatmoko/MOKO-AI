"""
MOKO NeuroMath: Mutation Queue — Deferred Formula Mutation System
================================================================
Berdasarkan: Synaptic Tagging & Capture (Frey & Morris, 1997)
             Late-LTP Consolidation (Bhatt & Bhattacharya, 2009)

Di otak, "Synaptic Tagging" adalah mekanisme di mana sinapsis yang
mengalami prediksi error (surprisal tinggi) diberi TAG sementara,
tapi TIDAK langsung diubah. Perubahan permanen (konsolidasi) baru
terjadi selama tidur ketika protein plastisitas tersedia.

Ini mencegah otak "menulis ulang" memori setiap saat — yang akan
menyebabkan catastrophic forgetting.

MutationQueue di MOKO mengimplementasikan mekanisme ini:
  AWAKE mode: Formula yang gagal FEP → hanya di-TAG (dicatat ke queue)
  SLEEP mode: Queue di-proses secara batch → mutasi sesungguhnya terjadi

Keuntungan:
  1. Tidak ada LLM call saat AWAKE (belajar cepat)
  2. Mutasi dilakukan secara batch (efisien)
  3. Mencegah modifikasi berulang formula yang sama (dedup)
"""

import json
import time
from pathlib import Path
from typing import Optional

from moko_config import settings


class MutationQueue:
    """
    Antrian deferred untuk mutasi formula Math-Omni.

    Entri antrian:
        formula_id   : ID formula yang perlu dimutasi
        chunk_text   : Teks chunk yang menyebabkan surprisal
        fep_score    : Skor surprisal (0.0–1.0); semakin tinggi semakin perlu dimutasi
        domain       : Domain konten (untuk konteks mutasi)
        timestamp    : Kapan ini dicatat
        status       : "pending" | "processed" | "failed"
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.queue_path = Path(
            workspace_dir or settings.WORKSPACE_DIR
        ) / ".math_omni" / "mutation_queue.jsonl"
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ──────────────────────────────────────────────────────────────────

    def enqueue(self, formula_id: str, chunk_text: str,
                fep_score: float, domain: str = ""):
        """
        Tambahkan formula ke antrian mutasi.

        Dipanggil saat AWAKE mode menemukan surprisal > threshold.
        Tidak ada LLM call di sini — hanya tulis file.

        Deduplication: jika formula_id sudah ada di antrian (status=pending),
        update chunk_text dan fep_score jika skor baru lebih tinggi.
        """
        # Cek apakah sudah ada di antrian
        existing = self._load_queue()
        for entry in existing:
            if (entry.get("formula_id") == formula_id
                    and entry.get("status") == "pending"):
                # Update jika skor baru lebih tinggi (semakin perlu dimutasi)
                if fep_score > entry.get("fep_score", 0):
                    entry["fep_score"]  = round(fep_score, 4)
                    entry["chunk_text"] = chunk_text[:500]
                    entry["updated_at"] = time.time()
                self._save_queue(existing)
                return

        # Entry baru
        entry = {
            "formula_id":  formula_id,
            "chunk_text":  chunk_text[:500],  # Truncate untuk hemat disk
            "fep_score":   round(fep_score, 4),
            "domain":      domain,
            "timestamp":   time.time(),
            "status":      "pending",
        }
        with open(self.queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_pending(self, limit: int = 20) -> list[dict]:
        """
        Ambil antrian mutasi yang belum diproses.
        Diurutkan berdasarkan fep_score (tertinggi = paling perlu dimutasi).

        Args:
            limit: Maksimum entri yang diambil per batch sleep.

        Returns:
            List entri pending, diurutkan fep_score descending.
        """
        queue = self._load_queue()
        pending = [e for e in queue if e.get("status") == "pending"]
        pending.sort(key=lambda e: e.get("fep_score", 0), reverse=True)
        return pending[:limit]

    def count_pending(self) -> int:
        """Jumlah entri yang menunggu diproses."""
        queue = self._load_queue()
        return sum(1 for e in queue if e.get("status") == "pending")

    # ── Update Status ─────────────────────────────────────────────────────────

    def mark_processed(self, formula_id: str):
        """Tandai formula sebagai selesai dimutasi."""
        self._update_status(formula_id, "processed")

    def mark_failed(self, formula_id: str):
        """Tandai formula sebagai gagal dimutasi (retry nanti)."""
        self._update_status(formula_id, "failed")

    def _update_status(self, formula_id: str, new_status: str):
        queue = self._load_queue()
        for entry in queue:
            if (entry.get("formula_id") == formula_id
                    and entry.get("status") == "pending"):
                entry["status"]       = new_status
                entry["processed_at"] = time.time()
                break
        self._save_queue(queue)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def prune_old(self, max_age_days: int = 7):
        """
        Hapus entri yang sudah diproses atau gagal lebih dari N hari lalu.
        Panggil dari apoptosis daemon atau setelah sleep cycle.
        """
        cutoff = time.time() - (max_age_days * 86400)
        queue  = self._load_queue()
        surviving = [
            e for e in queue
            if e.get("status") == "pending"
            or e.get("processed_at", time.time()) > cutoff
        ]
        self._save_queue(surviving)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_queue(self) -> list[dict]:
        if not self.queue_path.exists():
            return []
        entries = []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        return entries

    def _save_queue(self, entries: list[dict]):
        with open(self.queue_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── Singleton ─────────────────────────────────────────────────────────────────
mutation_queue = MutationQueue()
