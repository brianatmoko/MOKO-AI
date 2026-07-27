"""
MOKO Dataset Compressor — Zstd Streaming Pipeline
==================================================
Kompresi lossless dataset training menggunakan Zstandard.

Fitur:
  1. Streaming Compression — Kompresi JSONL tanpa load seluruh file ke RAM
  2. Streaming Decompression — Iterasi baris demi baris dari .jsonl.zst
  3. Auto-compression threshold — Otomatis kompres jika file > N MB
  4. Integrity verification — SHA256 checksum setelah kompresi
  5. Backward compatible — Bisa baca JSONL biasa sekaligus .jsonl.zst

Diperlukan:
  pip install zstandard  (atau: pip3 install zstandard)

Digunakan oleh:
  - interaction_logger.py    → kompres distill dataset JSONL secara otomatis
  - moko_trainer_v2.py       → baca dataset terkompresi saat training
  - disk_manager.py          → archival memori ke SSD
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("moko_dataset_compressor")

# ── Coba import zstandard ─────────────────────────────────────────────────────
try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False
    logger.warning(
        "[DatasetCompressor] zstandard tidak tersedia. "
        "Install dengan: pip install zstandard. "
        "Akan fallback ke gzip."
    )


class MokoDatasetCompressor:
    """
    Kompresi dan dekompresi dataset JSONL menggunakan Zstd atau gzip (fallback).

    Penggunaan:
        comp = MokoDatasetCompressor()
        info = comp.compress("dataset.jsonl", "dataset.jsonl.zst")
        for sample in comp.stream_read("dataset.jsonl.zst"):
            print(sample)
    """

    # ── Konfigurasi default ───────────────────────────────────────────────────
    DEFAULT_LEVEL    = 5       # Level 5: balance antara kecepatan dan ratio (1-22)
    CHUNK_SIZE       = 65536   # 64KB per chunk saat streaming dekompresi
    AUTO_COMPRESS_MB = 50      # Otomatis kompres jika dataset > 50MB
    _ZSTD_SUFFIX     = ".zst"
    _GZIP_SUFFIX     = ".gz"

    def __init__(self, level: int = DEFAULT_LEVEL, threads: int = 2):
        self.level   = level
        self.threads = threads
        self._backend = "zstd" if _ZSTD_AVAILABLE else "gzip"
        logger.info(f"[DatasetCompressor] Backend: {self._backend}, Level: {level}")

    # ── Kompresi ─────────────────────────────────────────────────────────────

    def compress(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        remove_original: bool = False,
    ) -> dict:
        """
        Kompresi file JSONL → .jsonl.zst (atau .jsonl.gz jika zstd tidak ada).

        Args:
            input_path:      Path ke file JSONL sumber.
            output_path:     Path output. Jika None, tambahkan suffix secara otomatis.
            remove_original: Hapus file asli setelah kompresi berhasil.

        Returns:
            dict dengan info: original_mb, compressed_mb, ratio, lines, checksum
        """
        input_path  = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"[DatasetCompressor] File tidak ditemukan: {input_path}")

        # Tentukan nama output
        if output_path is None:
            suffix = self._ZSTD_SUFFIX if self._backend == "zstd" else self._GZIP_SUFFIX
            output_path = input_path.with_suffix(input_path.suffix + suffix)
        output_path = Path(output_path)

        start_t      = time.perf_counter()
        original_size = input_path.stat().st_size
        line_count   = 0
        sha256       = hashlib.sha256()

        logger.info(f"[DatasetCompressor] Mengkompresi: {input_path.name} ({original_size//1024}KB)")

        if self._backend == "zstd":
            cctx = zstd.ZstdCompressor(level=self.level, threads=self.threads)
            with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
                with cctx.stream_writer(fout) as writer:
                    for line in fin:
                        writer.write(line)
                        sha256.update(line)
                        if line.strip():
                            line_count += 1
        else:
            import gzip
            with open(input_path, "rb") as fin, gzip.open(output_path, "wb", compresslevel=self.level) as fout:
                for line in fin:
                    fout.write(line)
                    sha256.update(line)
                    if line.strip():
                        line_count += 1

        compressed_size = output_path.stat().st_size
        ratio           = original_size / max(compressed_size, 1)
        elapsed         = time.perf_counter() - start_t

        result = {
            "input":          str(input_path),
            "output":         str(output_path),
            "original_mb":    round(original_size  / 1024**2, 3),
            "compressed_mb":  round(compressed_size / 1024**2, 3),
            "ratio":          round(ratio, 2),
            "lines":          line_count,
            "checksum_sha256": sha256.hexdigest(),
            "elapsed_s":      round(elapsed, 2),
            "backend":        self._backend,
        }

        logger.info(
            f"[DatasetCompressor] ✅ Selesai: {result['original_mb']}MB → "
            f"{result['compressed_mb']}MB (ratio {ratio:.1f}x) | "
            f"{line_count} baris | {elapsed:.1f}s"
        )

        if remove_original:
            input_path.unlink()
            logger.info(f"[DatasetCompressor] File asli dihapus: {input_path}")

        return result

    # ── Dekompresi Streaming ──────────────────────────────────────────────────

    def stream_read(
        self,
        path: str | Path,
        max_lines: Optional[int] = None,
        skip_invalid: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator: baca file JSONL atau .jsonl.zst per baris tanpa load penuh ke RAM.

        Args:
            path:        Path ke file (.jsonl atau .jsonl.zst)
            max_lines:   Batas maksimum baris yang dibaca (None = semua)
            skip_invalid: Skip baris yang bukan JSON valid

        Yields:
            dict — satu sample JSONL per iterasi
        """
        path    = Path(path)
        suffix  = "".join(path.suffixes).lower()
        count   = 0

        if not path.exists():
            raise FileNotFoundError(f"[DatasetCompressor] File tidak ditemukan: {path}")

        if ".zst" in suffix and _ZSTD_AVAILABLE:
            yield from self._stream_zstd(path, max_lines, skip_invalid)

        elif ".gz" in suffix:
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if max_lines and count >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        if not skip_invalid:
                            raise

        else:
            # Baca JSONL biasa
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_lines and count >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        if not skip_invalid:
                            raise

    def _stream_zstd(self, path: Path, max_lines: Optional[int], skip_invalid: bool):
        """Streaming dekompresi Zstd tanpa load seluruh file ke RAM."""
        dctx    = zstd.ZstdDecompressor()
        count   = 0
        buffer  = b""

        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                while True:
                    chunk = reader.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line.decode("utf-8"))
                            count += 1
                            if max_lines and count >= max_lines:
                                return
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            if not skip_invalid:
                                raise

    # ── Auto-Compress (untuk interaction_logger) ──────────────────────────────

    def auto_compress_if_large(
        self,
        path: str | Path,
        threshold_mb: float = AUTO_COMPRESS_MB,
        remove_original: bool = False,
    ) -> Optional[dict]:
        """
        Kompres file secara otomatis jika ukurannya melebihi threshold_mb.
        Dipanggil oleh interaction_logger.py setelah log_sample().

        Returns:
            info dict jika dikompresi, None jika di bawah threshold.
        """
        path = Path(path)
        if not path.exists():
            return None

        size_mb = path.stat().st_size / 1024**2
        if size_mb < threshold_mb:
            return None

        logger.info(
            f"[DatasetCompressor] Auto-compress dipicu: "
            f"{path.name} ({size_mb:.1f}MB > {threshold_mb}MB threshold)"
        )
        return self.compress(path, remove_original=remove_original)

    # ── Batch Write (untuk training data generation) ─────────────────────────

    def write_compressed_jsonl(
        self,
        records: List[Dict],
        output_path: str | Path,
        append: bool = False,
    ) -> dict:
        """
        Tulis daftar dict langsung ke file .jsonl.zst tanpa file perantara.

        Args:
            records:     List of dicts untuk ditulis.
            output_path: Path output .jsonl.zst
            append:      Jika True, append ke file yang ada (mode 'ab').

        Returns:
            info dict dengan total bytes ditulis dan jumlah records.
        """
        output_path = Path(output_path)
        mode        = "ab" if append and output_path.exists() else "wb"
        total_bytes = 0

        if self._backend == "zstd":
            cctx = zstd.ZstdCompressor(level=self.level, threads=self.threads)
            with open(output_path, mode) as fout:
                with cctx.stream_writer(fout) as writer:
                    for rec in records:
                        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
                        writer.write(line)
                        total_bytes += len(line)
        else:
            import gzip
            with gzip.open(output_path, "at" if append else "wt", encoding="utf-8") as fout:
                for rec in records:
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    fout.write(line)
                    total_bytes += len(line.encode("utf-8"))

        return {
            "output":    str(output_path),
            "records":   len(records),
            "bytes_raw": total_bytes,
            "backend":   self._backend,
        }

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def count_lines(self, path: str | Path) -> int:
        """Hitung jumlah baris (sample) dalam file JSONL atau .jsonl.zst."""
        return sum(1 for _ in self.stream_read(path))

    def verify_integrity(self, original: str | Path, compressed: str | Path) -> bool:
        """
        Verifikasi integritas kompresi dengan membandingkan checksum per-baris.
        """
        try:
            orig_hash = hashlib.sha256()
            comp_hash = hashlib.sha256()

            with open(original, "rb") as f:
                for line in f:
                    if line.strip():
                        orig_hash.update(line)

            for sample in self.stream_read(compressed):
                comp_hash.update((json.dumps(sample, ensure_ascii=False) + "\n").encode())

            match = orig_hash.hexdigest() == comp_hash.hexdigest()
            if not match:
                logger.warning(
                    f"[DatasetCompressor] ⚠️ Integrity check GAGAL: "
                    f"{Path(original).name} vs {Path(compressed).name}"
                )
            return match
        except Exception as e:
            logger.error(f"[DatasetCompressor] Error integrity check: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────
_compressor_instance: Optional[MokoDatasetCompressor] = None


def get_compressor() -> MokoDatasetCompressor:
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = MokoDatasetCompressor()
    return _compressor_instance
