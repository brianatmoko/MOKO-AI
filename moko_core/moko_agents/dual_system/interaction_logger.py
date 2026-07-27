"""
MOKO Interaction Logger — Perekam Sesi untuk Distilasi Pengetahuan
===================================================================
Mencatat hasil interaksi yang berhasil lolos verifikasi (runtime_guard)
untuk ditabulasikan ke dalam format dataset SFT offline.

Field yang dicatat (sesuai riset 23_REVISI_MANDOR_API_MURID_LOKAL.md):
  - prompt          : permintaan asli pengguna
  - reasoning_trace : alur pikir Guru API / BrainNode (<thought>...)</thought>)
  - code_output     : kode yang dihasilkan dan lolos Guard
  - passed_guard    : apakah verifier menyetujui (SELALU True saat dicatat)
  - task_complexity : estimasi kompleksitas 0.0-1.0 (untuk kurikulum SFT)
  - task_category   : kategori tugas (coding, math, general)
  - source          : asal data ("guru_api" | "lokal") — berguna untuk weight
  - timestamp       : waktu UNIX epoch saat interaksi terjadi
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("moko_interaction_logger")


@dataclass
class DistillSample:
    prompt: str
    reasoning_trace: str
    code_output: str
    passed_guard: bool
    task_complexity: float = 0.5
    task_category: str = "general"      # coding | math | general
    source: str = "guru_api"            # guru_api | lokal
    timestamp: float = 0.0


class InteractionLogger:
    """Mencatat interaksi terpilih ke berkas dataset JSONL."""

    def __init__(self, log_file_path: str | Path | None = None) -> None:
        if log_file_path is None:
            # Cari root repositori
            project_dir = Path(__file__).resolve().parents[3]
            self.log_path = project_dir / "distill_dataset" / "moko_distill_samples.jsonl"
        else:
            self.log_path = Path(log_file_path)

        # Inisialisasi direktori
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Gagal membuat direktori log dataset: {e}")

    def log_sample(
        self,
        prompt: str,
        thought: str,
        code: str,
        passed_guard: bool,
        task_complexity: float = 0.5,
        task_category: str = "general",
        source: str = "guru_api"
    ) -> bool:
        """Tambahkan satu sampel log jika memenuhi syarat lolos verifikasi (passed_guard)."""
        # Hanya catat data yang valid/lolos unit test (Verifiable Rewards)
        if not passed_guard:
            logger.info("Sampel diabaikan karena tidak lolos runtime_guard.")
            return False

        sample = DistillSample(
            prompt=prompt,
            reasoning_trace=thought,
            code_output=code,
            passed_guard=passed_guard,
            task_complexity=task_complexity,
            task_category=task_category,
            source=source,
            timestamp=time.time()
        )

        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(sample)) + "\n")
            logger.info(f"Berhasil mencatat sampel baru ke {self.log_path}")

            # Auto-kompresi dataset jika file > 50MB (Riset 25)
            try:
                from moko_agents.dual_system.dataset_compressor import get_compressor
                comp = get_compressor()
                comp.auto_compress_if_large(self.log_path, threshold_mb=50.0)
            except Exception as e:
                logger.debug(f"Auto-compression skipped: {e}")

            return True
        except Exception as e:
            logger.error(f"Gagal menulis sampel ke {self.log_path}: {e}")
            return False
