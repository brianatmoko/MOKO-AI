"""
MOKO Byte-Q Automation Pipeline
===============================
Otomatisasi kuantisasi Byte-Q untuk domain-specialized models (Phase 4.1).
Menerapkan Lloyd's algorithm dan Huffman coding ke model GGUF, lalu menyimpannya
dalam container Byte-Q yang REKONSTRUKTIBEL (bisa dimuat kembali oleh
moko_tools/byteq_loader.py → GGUF F16 → llama.cpp).

Container yang dihasilkan (<name>.byteq.gguf):
  - Metadata global:
      byteq.version            : STRING  ("1")
      byteq.quantized_tensors  : [STRING] (daftar tensor terkuantisasi)
  - Per tensor terkuantisasi <t>:
      tensor <t>               : I8 (indeks {-1,0,+1,2}, shape asli)
      byteq.levels.<t>         : [FLOAT32] (4 optimal levels dari Lloyd)
  - Tensor lain (embedding/norm/bias) disalin apa adanya.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    print("Error: 'gguf' library tidak ditemukan. Jalankan 'pip install gguf'.")
    sys.exit(1)

from moko_tools.byteq_quantizer import byteq_quantize
from moko_tools.byteq_loader import (
    BYTEQ_VERSION,
    BYTEQ_VERSION_KEY,
    BYTEQ_TENSORS_KEY,
    BYTEQ_LEVELS_PREFIX,
    _copy_kv,
)

# Tensor float yang layak dikuantisasi Byte-Q
_FLOAT_TYPES = (
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
)


def quantize_model(input_path: str, output_path: str = None):
    """
    Kuantisasi model GGUF menggunakan Byte-Q menjadi container rekonstruktibel.
    """
    input_path = Path(input_path)
    if not output_path:
        # Hasilkan nama <stem>.byteq.gguf
        output_path = input_path.with_suffix("").with_name(input_path.stem + ".byteq.gguf")
    else:
        output_path = Path(output_path)

    print(f"🚀 Memulai Byte-Q Quantization: {input_path.name}")
    t_start = time.time()

    # 1. Baca GGUF sumber
    reader = gguf.GGUFReader(input_path)
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else "llama"

    writer = gguf.GGUFWriter(output_path, arch=arch)

    # 2. Salin metadata KV (kecuali arch yang sudah di-set writer)
    _copy_kv(reader, writer, skip_keys={gguf.Keys.General.ARCHITECTURE})

    # 3. Proses tensors
    total_params = 0
    quantized_params = 0
    quantized_names = []

    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        shape = tuple(int(x) for x in tensor.shape)

        total_params += int(np.prod(shape))

        is_weight = "weight" in name.lower() and len(shape) >= 2
        is_float = tensor.tensor_type in _FLOAT_TYPES

        if is_weight and is_float:
            # Konversi ke float32 untuk processing
            w_fp32 = np.array(data, dtype=np.float32)

            # Jalankan Byte-Q (Lloyd optimal levels)
            res = byteq_quantize(w_fp32, use_lloyd=True)

            # Simpan indeks {-1,0,+1,2} sebagai tensor I8 + levels sebagai metadata.
            encoded = res.weights_quantized.astype(np.int8).reshape(shape)
            writer.add_tensor(name, encoded, raw_dtype=gguf.GGMLQuantizationType.I8)
            writer.add_array(
                f"{BYTEQ_LEVELS_PREFIX}{name}",
                [float(x) for x in np.asarray(res.levels, dtype=np.float32).tolist()],
            )
            quantized_names.append(name)

            quantized_params += int(np.prod(shape))
            print(f"  ✅ Quantized {name}: MSE={res.mse:.8f}, Compression={res.total_compression:.1f}x")
        else:
            # Salin tensor tanpa perubahan (embedding, norm, bias, dsb.)
            writer.add_tensor(name, data, raw_shape=shape, raw_dtype=tensor.tensor_type)

    # 4. Metadata Byte-Q global
    writer.add_key_value(BYTEQ_VERSION_KEY, BYTEQ_VERSION, gguf.GGUFValueType.STRING)
    writer.add_array(BYTEQ_TENSORS_KEY, quantized_names)

    # 5. Tulis output
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    t_end = time.time()
    pct = (quantized_params / total_params * 100) if total_params else 0.0
    print(f"\n✨ Selesai! Model Byte-Q disimpan di: {output_path}")
    print(f"⏱️ Waktu: {t_end - t_start:.1f}s")
    print(f"📊 Progress: {pct:.1f}% params dioptimasi via Byte-Q ({len(quantized_names)} tensor)")
    print(f"ℹ️ Muat via engine (otomatis dequantize) atau: python -m moko_tools.byteq_loader {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MOKO Byte-Q Model Quantizer")
    parser.add_argument("input", help="Path ke model GGUF (FP16/BF16)")
    parser.add_argument("--output", help="Path output", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} tidak ditemukan.")
        return

    quantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
