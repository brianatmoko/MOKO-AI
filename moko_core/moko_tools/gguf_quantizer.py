"""
MOKO GGUF Quantizer — Kompres model dari BF16 ke format hemat
================================================================
Menggunakan llama_cpp.llama_model_quantize() Python API langsung.
Tidak butuh binary llama-quantize eksternal.

Target: MOKO-AI-4B-CryptoCore-BF16.gguf (7.9GB) → Q2_K (~1.5GB)
"""

import sys
import os
import struct
import time
import ctypes
from pathlib import Path

# Tambahkan venv ke path
VENV_SITE = Path(__file__).parent.parent / "venv" / "lib" / "python3.12" / "site-packages"
sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Quant type mapping (dari llama.h)
QUANT_TYPES = {
    "Q4_K_M": 15,
    "Q3_K_M": 12,   # LLAMA_FTYPE_MOSTLY_Q3_K_M
    "Q2_K":   10,   # LLAMA_FTYPE_MOSTLY_Q2_K
    "IQ2_XXS": 19,  # LLAMA_FTYPE_MOSTLY_IQ2_XXS
    "Q5_K_M": 17,
    "Q6_K":   20,
    "Q8_0":   8,
}


def quantize_gguf(input_path: str, output_path: str, quant_type: str = "Q2_K") -> bool:
    """
    Re-quantize GGUF dari BF16 ke format yang lebih kecil.
    Menggunakan llama_cpp Python API.
    """
    from llama_cpp import (
        llama_model_quantize,
        llama_model_quantize_default_params,
        llama_backend_init,
        llama_backend_free,
    )

    if quant_type not in QUANT_TYPES:
        print(f"[Quantizer] Tipe tidak valid. Pilihan: {list(QUANT_TYPES.keys())}")
        return False

    quant_id = QUANT_TYPES[quant_type]

    print(f"[MOKO Quantizer] Input:  {input_path}")
    print(f"[MOKO Quantizer] Output: {output_path}")
    print(f"[MOKO Quantizer] Target: {quant_type} (type={quant_id})")
    print(f"[MOKO Quantizer] Memulai quantisasi... (bisa memakan 5-15 menit)")

    llama_backend_init()

    params = llama_model_quantize_default_params()
    params.ftype = quant_id
    params.nthread = os.cpu_count() or 4
    params.allow_requantize = True
    params.quantize_output_tensor = False  # Jangan kuantisasi output tensor (jaga akurasi output)
    params.only_copy = False
    params.pure = False

    t0 = time.perf_counter()
    ret = llama_model_quantize(
        input_path.encode(),
        output_path.encode(),
        ctypes.byref(params)
    )
    elapsed = time.perf_counter() - t0

    llama_backend_free()

    if ret == 0:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"[MOKO Quantizer] ✅ Selesai dalam {elapsed:.1f}s")
        print(f"[MOKO Quantizer] Output size: {size_mb:.0f}MB ({size_mb/1024:.2f}GB)")
        return True
    else:
        print(f"[MOKO Quantizer] ❌ Gagal (return code: {ret})")
        return False


def inject_moko_metadata(gguf_path: str) -> bool:
    """
    Baca GGUF dan modifikasi metadata untuk MOKO branding.
    PERINGATAN: Ini memodifikasi file — selalu backup dulu.
    """
    import hashlib
    import struct

    # Ambil SHA256 4KB header untuk fingerprint baru
    with open(gguf_path, "rb") as f:
        header = f.read(4096)
    fingerprint = hashlib.sha256(header).hexdigest()

    print(f"[MOKO Quantizer] Fingerprint Q2K: {fingerprint[:32]}...")

    # Simpan fingerprint ke sidecar JSON
    import json
    sidecar = {
        "model": Path(gguf_path).name,
        "quantization": "Byte-Q (IQ2_XXS)",
        "moko.identity": "MOKO-RAG-v1",
        "moko.role": "RAG-Bridge",
        "moko.description": "MOKO RAG Bridge — Ultra-efficient 2-bit quantization.",
        "moko.base_architecture": "qwen2.5-1.5b-instruct",
    }
    sidecar_path = gguf_path.replace(".gguf", "_moko_meta.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[MOKO Quantizer] ✅ Metadata disimpan ke: {sidecar_path}")
    return True


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).parent.parent.parent
    
    # Input: MOKO-Coder-1.5B-Uncensored-F16.gguf (3.4GB)
    INPUT  = str(PROJECT_DIR / "MOKO-Coder-1.5B-Uncensored-F16.gguf")
    # Output: MOKO-RAG-1.5B-ByteQ.gguf (Target < 500MB)
    OUTPUT = str(PROJECT_DIR / "MOKO-RAG-1.5B-ByteQ.gguf")
    TARGET = "Q2_K"

    if len(sys.argv) > 1:
        TARGET = sys.argv[1].upper()

    print("=" * 60)
    print("  MOKO GGUF Quantizer — BF16 → Compact")
    print("=" * 60)

    if not os.path.exists(INPUT):
        print(f"ERROR: Input tidak ditemukan: {INPUT}")
        sys.exit(1)

    input_gb = os.path.getsize(INPUT) / 1024**3
    print(f"Input size: {input_gb:.2f}GB")
    print(f"Target: {TARGET}")

    expected_sizes = {"Q2_K": "~1.5GB", "Q3_K_M": "~1.8GB", "Q4_K_M": "~2.6GB", "Q8_0": "~4.1GB"}
    print(f"Expected output: {expected_sizes.get(TARGET, '?')}")
    print()

    ok = quantize_gguf(INPUT, OUTPUT, TARGET)
    if ok:
        inject_moko_metadata(OUTPUT)
        print()
        print("✅ Quantisasi selesai. File siap dipakai.")
        print(f"   Model: {OUTPUT}")
