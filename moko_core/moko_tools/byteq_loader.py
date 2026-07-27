"""
MOKO Byte-Q Loader — Dequantize-Before-Load Bridge
===================================================
Menjembatani Byte-Q ke jalur inferensi nyata (llama.cpp / llama-server).

llama.cpp tidak mengenal format Byte-Q (4-state {-1,0,+1,2} + optimal levels),
sehingga model .byteq.gguf tidak bisa dimuat langsung. Modul ini melakukan
rekonstruksi ("dequantize-before-load"): membaca container Byte-Q, mengembalikan
weights ke FP16 menggunakan levels per-tensor, lalu menulis GGUF standar F16 yang
bisa dimuat penuh oleh llama.cpp.

Format container Byte-Q (dihasilkan quantize_domain_models.py):
  - Metadata global:
      byteq.version              : STRING
      byteq.quantized_tensors    : [STRING]  (nama tensor yang dikuantisasi)
  - Per tensor terkuantisasi <name>:
      tensor <name>              : I8 (indeks {-1,0,+1,2}, shape asli)
      byteq.levels.<name>        : [FLOAT32] (4 optimal levels)
  - Tensor lain disalin apa adanya.
"""

import os
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    gguf = None


# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA FORMAT
# ─────────────────────────────────────────────────────────────────────────────
BYTEQ_VERSION = "1"
BYTEQ_VERSION_KEY = "byteq.version"
BYTEQ_TENSORS_KEY = "byteq.quantized_tensors"
BYTEQ_LEVELS_PREFIX = "byteq.levels."
BYTEQ_SUFFIX = ".byteq.gguf"


def byteq_decode_fast(indices: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Dekode indeks Byte-Q {-1,0,+1,2} → FP32 secara vektor (cepat).

    Mapping: -1 → levels[0], 0 → levels[1], +1 → levels[2], +2 → levels[3]
    """
    levels = np.asarray(levels, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.int64) + 1  # geser ke 0..3
    idx = np.clip(idx, 0, len(levels) - 1)
    return levels[idx].astype(np.float32)


def is_byteq_model(path) -> bool:
    """Deteksi apakah sebuah path merupakan model Byte-Q."""
    p = str(path)
    if p.endswith(BYTEQ_SUFFIX):
        return True
    if not os.path.isfile(p):
        return False
    if gguf is None:
        return False
    try:
        reader = gguf.GGUFReader(p)
        return reader.get_field(BYTEQ_VERSION_KEY) is not None
    except Exception:
        return False


def _copy_kv(reader, writer, skip_keys):
    """Salin seluruh metadata KV dari reader ke writer, kecuali skip_keys."""
    for name, field in reader.fields.items():
        if name in skip_keys:
            continue
        # Lewati pseudo-field header GGUF (GGUF.version/tensor_count/kv_count):
        # writer akan menuliskannya sendiri, menyalinnya menyebabkan duplikat.
        if name.startswith("GGUF."):
            continue
        if not field.types:
            continue
        main_type = field.types[0]
        value = field.contents()
        try:
            if main_type == gguf.GGUFValueType.ARRAY:
                writer.add_array(name, value)
            else:
                writer.add_key_value(name, value, main_type)
        except Exception:
            # Metadata yang tidak bisa disalin dilewati agar tidak menggagalkan build.
            continue


def dequantize_to_gguf(byteq_path, out_path=None, force: bool = False) -> str:
    """
    Rekonstruksi container Byte-Q menjadi GGUF standar F16 yang bisa dimuat llama.cpp.

    Args:
        byteq_path: Path ke model .byteq.gguf.
        out_path:   Path output GGUF F16 (default: <name>.dequant.gguf di samping input).
        force:      Paksa rebuild walau cache sudah ada & lebih baru.

    Returns:
        Path ke GGUF F16 hasil rekonstruksi.
    """
    if gguf is None:
        raise RuntimeError("Library 'gguf' tidak tersedia. Jalankan 'pip install gguf'.")

    byteq_path = Path(byteq_path)
    if not byteq_path.is_file():
        raise FileNotFoundError(f"Byte-Q model tidak ditemukan: {byteq_path}")

    if out_path is None:
        out_path = byteq_path.with_suffix("").with_suffix(".dequant.gguf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache: gunakan hasil sebelumnya jika lebih baru dari sumber.
    if (
        not force
        and out_path.is_file()
        and out_path.stat().st_mtime >= byteq_path.stat().st_mtime
        and out_path.stat().st_size > 0
    ):
        print(f"[BYTE-Q] Cache valid, memakai: {out_path}")
        return str(out_path)

    print(f"[BYTE-Q] Rekonstruksi {byteq_path.name} → {out_path.name}")
    reader = gguf.GGUFReader(byteq_path)

    version_field = reader.get_field(BYTEQ_VERSION_KEY)
    if version_field is None:
        raise ValueError(f"Bukan container Byte-Q (metadata '{BYTEQ_VERSION_KEY}' tidak ada): {byteq_path}")

    tensors_field = reader.get_field(BYTEQ_TENSORS_KEY)
    quantized_names = set(tensors_field.contents()) if tensors_field is not None else set()

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else "llama"

    writer = gguf.GGUFWriter(out_path, arch=arch)

    # Salin metadata, kecuali arch (sudah di-set writer) dan metadata Byte-Q internal.
    skip = {gguf.Keys.General.ARCHITECTURE, BYTEQ_VERSION_KEY, BYTEQ_TENSORS_KEY}
    skip |= {f for f in reader.fields if f.startswith(BYTEQ_LEVELS_PREFIX)}
    _copy_kv(reader, writer, skip)

    n_reconstructed = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name in quantized_names:
            levels_field = reader.get_field(f"{BYTEQ_LEVELS_PREFIX}{name}")
            if levels_field is None:
                # Tidak ada levels → salin apa adanya (fallback aman).
                writer.add_tensor(name, tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)
                continue
            levels = np.array(levels_field.contents(), dtype=np.float32)
            decoded = byteq_decode_fast(tensor.data, levels).astype(np.float16)
            decoded = decoded.reshape(tuple(int(x) for x in tensor.shape))
            writer.add_tensor(name, decoded)
            n_reconstructed += 1
        else:
            writer.add_tensor(name, tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[BYTE-Q] Selesai: {n_reconstructed} tensor direkonstruksi → {out_path}")
    return str(out_path)


def resolve_loadable_model(model_path, cache_dir=None) -> str:
    """
    Kembalikan path model yang bisa dimuat llama.cpp.

    - Jika model_path adalah container Byte-Q → dequantize ke cache dan kembalikan
      path GGUF F16 hasil rekonstruksi.
    - Selain itu → kembalikan model_path apa adanya.
    """
    if not is_byteq_model(model_path):
        return str(model_path)

    byteq_path = Path(model_path)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / (byteq_path.stem.replace(".byteq", "") + ".dequant.gguf")
    else:
        out_path = None
    return dequantize_to_gguf(byteq_path, out_path=out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOKO Byte-Q → GGUF F16 dequantizer")
    parser.add_argument("input", help="Path ke model .byteq.gguf")
    parser.add_argument("--output", help="Path output GGUF F16", default=None)
    parser.add_argument("--force", action="store_true", help="Paksa rebuild cache")
    args = parser.parse_args()

    result = dequantize_to_gguf(args.input, out_path=args.output, force=args.force)
    print(f"Output: {result}")
