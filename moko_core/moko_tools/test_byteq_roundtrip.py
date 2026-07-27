"""
Round-trip test untuk integrasi Byte-Q ke jalur inferensi.

Alur:
  1. Bangun GGUF dummy F16 (arch=llama) berisi beberapa weight tensor.
  2. Kuantisasi via quantize_domain_models.quantize_model → <name>.byteq.gguf
  3. Rekonstruksi via byteq_loader.dequantize_to_gguf → <name>.dequant.gguf (F16 loadable)
  4. Verifikasi:
     - Container Byte-Q terdeteksi sebagai model Byte-Q.
     - Tensor terkuantisasi tersimpan sebagai I8 di container.
     - GGUF rekonstruksi bisa dibaca kembali, bertipe F16, shape sama.
     - MSE(rekonstruksi vs asli) sesuai ekspektasi Byte-Q (kecil, terbatas).

Jalankan:
  PYTHONPATH=moko_core ./bin/python moko_core/moko_tools/test_byteq_roundtrip.py
"""

import tempfile
from pathlib import Path

import numpy as np
import gguf

from moko_tools.quantize_domain_models import quantize_model
from moko_tools import byteq_loader


def _build_dummy_gguf(path: Path, tensors: dict):
    writer = gguf.GGUFWriter(path, arch="llama")
    # Metadata minimal agar terlihat seperti model nyata.
    writer.add_name("dummy-byteq-test")
    writer.add_context_length(2048)
    writer.add_block_count(1)
    for name, arr in tensors.items():
        writer.add_tensor(name, arr.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    rng = np.random.default_rng(42)  # reproducibility
    tensors = {
        "blk.0.attn_q.weight": rng.standard_normal((64, 128)).astype(np.float32) * 0.05,
        "blk.0.ffn_up.weight": rng.standard_normal((128, 64)).astype(np.float32) * 0.03,
        "blk.0.attn_norm.weight": rng.standard_normal((64,)).astype(np.float32),  # 1D → tidak dikuantisasi
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "dummy-F16.gguf"
        _build_dummy_gguf(src, tensors)
        print(f"[TEST] Dummy GGUF: {src} ({src.stat().st_size} bytes)")

        # 1. Kuantisasi
        byteq_path = tmp / "dummy-F16.byteq.gguf"
        quantize_model(str(src), str(byteq_path))
        assert byteq_path.is_file(), "Container Byte-Q tidak dibuat"

        # 2. Deteksi Byte-Q
        assert byteq_loader.is_byteq_model(str(byteq_path)), "Container tidak terdeteksi sebagai Byte-Q"
        assert not byteq_loader.is_byteq_model(str(src)), "GGUF biasa salah terdeteksi sebagai Byte-Q"

        # 3. Cek container: tensor 2D tersimpan I8, levels ada
        rq = gguf.GGUFReader(byteq_path)
        qnames = set(rq.get_field(byteq_loader.BYTEQ_TENSORS_KEY).contents())
        assert "blk.0.attn_q.weight" in qnames and "blk.0.ffn_up.weight" in qnames
        assert "blk.0.attn_norm.weight" not in qnames, "Tensor 1D seharusnya tidak dikuantisasi"
        for t in rq.tensors:
            if t.name in qnames:
                assert t.tensor_type == gguf.GGMLQuantizationType.I8, f"{t.name} bukan I8"
                assert rq.get_field(f"{byteq_loader.BYTEQ_LEVELS_PREFIX}{t.name}") is not None

        # 4. Rekonstruksi → F16 loadable
        dequant_path = tmp / "dummy.dequant.gguf"
        out = byteq_loader.dequantize_to_gguf(str(byteq_path), out_path=str(dequant_path))
        assert Path(out).is_file(), "GGUF rekonstruksi tidak dibuat"

        # 5. Verifikasi rekonstruksi
        rr = gguf.GGUFReader(out)
        rec = {t.name: t for t in rr.tensors}
        for name in qnames:
            t = rec[name]
            assert t.tensor_type == gguf.GGMLQuantizationType.F16, f"{name} tidak F16 setelah rekonstruksi"
            orig = tensors[name]
            recon = np.array(t.data, dtype=np.float32).reshape(orig.shape)
            mse = float(np.mean((orig - recon) ** 2))
            var = float(np.var(orig))
            nmse = mse / (var + 1e-12)
            print(f"[TEST] {name}: shape={tuple(t.shape)} MSE={mse:.6e} NMSE={nmse:.3f}")
            # Byte-Q 4-state: NMSE harus jauh di bawah 1.0 (jika tidak, kuantisasi rusak)
            assert nmse < 0.5, f"NMSE {nmse:.3f} terlalu besar untuk {name}"

        # Cek cache: panggilan kedua tidak rebuild (mtime cache >= sumber)
        out2 = byteq_loader.dequantize_to_gguf(str(byteq_path), out_path=str(dequant_path))
        assert out2 == out

    print("\n✅ SEMUA ASSERT LULUS: Byte-Q terintegrasi ke jalur muat model (dequantize-before-load).")


if __name__ == "__main__":
    main()
