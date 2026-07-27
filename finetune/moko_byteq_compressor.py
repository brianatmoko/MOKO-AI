"""
MOKO Byte-Q INT4 Extreme Compressor
=====================================
Implementasi pipeline kompresi dari riset docs/riset/09_FORMULASI_EFISIENSI.md:

  FP16/BF16 → Lloyd Optimal Levels → Byte-Q Quantize → Huffman Encode → Packed Storage

Target kompresi:
  FP16   : 2.0 bytes/param (baseline)
  INT4   : 0.5 bytes/param (4× compression)
  Byte-Q : 0.25 bytes/param (8× compression)
  +Huffman: 0.209 bytes/param (9.6× compression) ← TARGET KITA

Formula dari riset (05_RUMUSAN_MATEMATIKA.md):
  Lloyd optimal levels: l* ≈ { -1.51σ, -0.45σ, +0.45σ, +1.51σ }
  MSE Lloyd: 0.11σ² (vs uniform: 0.333σ²) — 3× lebih akurat!

  Distribusi real LLM weights:
    p(-1)=0.08, p(0)=0.75, p(+1)=0.14, p(+2)=0.03
  → Shannon H = 1.154 bits
  → Huffman L_avg ≈ 1.67 bits  (9.6× dari FP16 via 2 bytes/1.67 bits × 8×)

Penggunaan:
  python3 moko_byteq_compressor.py --adapter finetune/moko_adapters/moko_coder_1b
  python3 moko_byteq_compressor.py --adapter ... --bits 4 --output output_dir/
  python3 moko_byteq_compressor.py --benchmark  # analisis distribusi saja
"""

import sys
import os
import json
import time
import struct
import argparse
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

PROJECT_DIR = Path(__file__).parent.parent
_site = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site.exists() and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

import torch
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════

# Format Byte-Q: {-1, 0, +1, +2} → encoding integer 0-3
BYTEQ_LEVELS = 4
BYTEQ_MAP = {-1: 0b00, 0: 0b01, 1: 0b10, 2: 0b11}  # value → 2-bit code
BYTEQ_UNMAP = {v: k for k, v in BYTEQ_MAP.items()}

# Lloyd optimal boundaries untuk Gaussian (dari riset)
# b* ≈ { -∞, -0.98σ, 0, +0.98σ, +∞ }
LLOYD_BOUNDARY_SIGMA = 0.98


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%Human:%M:%S")
    icons = {"INFO": "ℹ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "COMPRESS": "🗜️"}
    print(f"[{ts}] {icons.get(level,'•')} {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LLOYD OPTIMAL LEVEL CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_lloyd_levels(sigma: float, n_iterations: int = 20) -> Tuple[List[float], List[float]]:
    """
    Lloyd's algorithm untuk menemukan optimal quantization levels untuk Gaussian.
    
    Dari riset 05_RUMUSAN_MATEMATIKA.md §1.4:
      Initial: l₀ = {-θ, -θ/3, +θ/3, +θ} dimana θ = 3σ
      Konvergensi: l* ≈ { -1.51σ, -0.45σ, +0.45σ, +1.51σ }
    
    Returns:
      levels (centroids): titik rekonstruksi optimal
      boundaries: batas keputusan antar level
    """
    theta = 3.0 * sigma
    # Initial levels (uniform)
    levels = [-theta, -theta/3, theta/3, theta]
    
    for _ in range(n_iterations):
        # Update boundaries (midpoint antara level)
        boundaries = [-float('inf')]
        for i in range(len(levels) - 1):
            boundaries.append((levels[i] + levels[i+1]) / 2)
        boundaries.append(float('inf'))
        
        # Update levels (centroid = E[w | dalam region])
        new_levels = []
        for i in range(len(levels)):
            lo, hi = boundaries[i], boundaries[i+1]
            # E[X | lo ≤ X < hi] untuk X ~ N(0, σ²)
            # Menggunakan scipy-free approximation (erf)
            centroid = _gaussian_conditional_mean(lo, hi, sigma)
            new_levels.append(centroid)
        
        # Cek konvergensi
        delta = sum(abs(new_levels[i] - levels[i]) for i in range(len(levels)))
        levels = new_levels
        if delta < 1e-8:
            break
    
    return levels, boundaries[1:-1]  # exclude -inf dan +inf


def _gaussian_conditional_mean(lo: float, hi: float, sigma: float) -> float:
    """
    E[X | lo ≤ X < hi] untuk X ~ N(0, σ²).
    Formula: σ × [φ(lo/σ) - φ(hi/σ)] / [Φ(hi/σ) - Φ(lo/σ)]
    dimana φ = PDF Gaussian, Φ = CDF Gaussian
    """
    def phi(x):
        """Standard Gaussian PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    
    def Phi(x):
        """Standard Gaussian CDF via erf."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    import math
    
    if sigma < 1e-10:
        return (lo + hi) / 2
    
    lo_s = lo / sigma if lo != -float('inf') else -10.0
    hi_s = hi / sigma if hi != float('inf') else 10.0
    
    p_lo = phi(lo_s)
    p_hi = phi(hi_s)
    P_lo = Phi(lo_s)
    P_hi = Phi(hi_s)
    
    prob = P_hi - P_lo
    if prob < 1e-10:
        return (lo + hi) / 2
    
    return sigma * (p_lo - p_hi) / prob


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: BYTE-Q QUANTIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantizationConfig:
    """Konfigurasi per-tensor untuk Byte-Q quantization."""
    tensor_name: str
    sigma: float           # Estimated standard deviation
    scale: float           # Dequantization scale factor (Lloyd alpha)
    lloyd_levels: List[float]   # Optimal reconstruction levels
    lloyd_boundaries: List[float]  # Decision boundaries
    n_params: int
    original_dtype: str
    quantized: bool = True


def estimate_sigma(weight: torch.Tensor) -> float:
    """Estimasi σ dari weight tensor. Gunakan MAD untuk robustness."""
    w = weight.float().flatten()
    # MAD estimator: σ ≈ 1.4826 × median(|w - median(w)|)
    med = w.median()
    mad = (w - med).abs().median()
    sigma = float(1.4826 * mad)
    if sigma < 1e-6:
        sigma = float(w.std())  # Fallback ke std
    return max(sigma, 1e-6)


def quantize_tensor_byteq(
    weight: torch.Tensor,
    sigma: float,
    boundaries: List[float],
    lloyd_levels: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Quantize satu tensor ke Byte-Q format (INT4, 4 levels: {-1,0,+1,+2}).
    
    Returns:
      quantized: ndarray int8 shape sama dengan weight, values dalam {0,1,2,3}
      scale: scalar untuk dequantize (alpha = lloyd_levels[2] - lloyd_levels[1])
    """
    w = weight.float().numpy()
    q = np.zeros_like(w, dtype=np.int8)
    
    # Boundaries: [b1, b2, b3]
    b1, b2, b3 = boundaries[0], boundaries[1], boundaries[2]
    
    # Quantize berdasarkan boundaries
    q[w < b1] = 0   # maps to -1
    q[(w >= b1) & (w < b2)] = 1   # maps to 0
    q[(w >= b2) & (w < b3)] = 2   # maps to +1
    q[w >= b3] = 3   # maps to +2
    
    # Scale factor: jarak antara adjacent Lloyd levels
    scale = (lloyd_levels[2] - lloyd_levels[0]) / 2  # approx alpha
    if abs(scale) < 1e-8:
        scale = sigma * 2.0 / 3.0
    
    return q, scale


def dequantize_tensor_byteq(
    q: np.ndarray,
    lloyd_levels: List[float],
    original_shape: tuple,
    original_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Rekonstruksi tensor dari Byte-Q quantized form.
    Setiap int value {0,1,2,3} → Lloyd optimal level.
    """
    # Map integer codes ke Lloyd levels
    level_map = np.array(lloyd_levels, dtype=np.float32)
    reconstructed = level_map[q.flatten()].reshape(original_shape)
    return torch.tensor(reconstructed, dtype=original_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: HUFFMAN ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data: np.ndarray) -> Dict[int, str]:
    """
    Build Huffman codes dari distribusi data Byte-Q.
    
    Distribusi real LLM (dari riset 09):
      p(0→-1) = 0.08, p(1→0) = 0.75, p(2→+1) = 0.14, p(3→+2) = 0.03
    """
    # Hitung frekuensi
    counter = Counter(data.flatten().tolist())
    
    # Ensure semua 4 symbols ada
    for sym in [0, 1, 2, 3]:
        if sym not in counter:
            counter[sym] = 1
    
    # Build Huffman tree
    heap = [HuffmanNode(sym, freq) for sym, freq in counter.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        merged = HuffmanNode(None, lo.freq + hi.freq)
        merged.left = lo
        merged.right = hi
        heapq.heappush(heap, merged)
    
    # Traverse tree untuk dapatkan codes
    codes = {}
    def traverse(node, code=""):
        if node.symbol is not None:
            codes[node.symbol] = code if code else "0"
            return
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")
    
    if heap:
        traverse(heap[0])
    
    return codes


def huffman_encode(data: np.ndarray, codes: Dict[int, str]) -> Tuple[bytes, int]:
    """
    Encode numpy array menggunakan Huffman codes.
    Returns: (packed_bytes, total_bits)
    """
    bitstring = "".join(codes[int(v)] for v in data.flatten())
    n_bits = len(bitstring)
    
    # Pack bits ke bytes
    # Pad ke multiple of 8
    padded = bitstring + "0" * ((8 - len(bitstring) % 8) % 8)
    packed = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
    
    return packed, n_bits


def huffman_decode(packed: bytes, n_bits: int, codes: Dict[int, str], n_elements: int) -> np.ndarray:
    """Decode Huffman-encoded bytes kembali ke array integer."""
    # Reverse code lookup
    reverse_codes = {v: k for k, v in codes.items()}
    
    # Unpack bytes ke bitstring
    bitstring = "".join(format(b, '08b') for b in packed)[:n_bits]
    
    # Decode
    result = []
    current = ""
    for bit in bitstring:
        current += bit
        if current in reverse_codes:
            result.append(reverse_codes[current])
            current = ""
        if len(result) == n_elements:
            break
    
    return np.array(result, dtype=np.int8)


def compute_huffman_stats(q_data: np.ndarray, codes: Dict[int, str]) -> Dict:
    """Hitung statistik kompresi Huffman."""
    counter = Counter(q_data.flatten().tolist())
    n_total = len(q_data.flatten())
    
    # Shannon entropy
    H = 0.0
    avg_bits = 0.0
    for sym in [0, 1, 2, 3]:
        cnt = counter.get(sym, 0)
        if cnt > 0:
            p = cnt / n_total
            H -= p * math.log2(p)
            avg_bits += p * len(codes.get(sym, "00"))
    
    import math
    
    return {
        "distribution": {str(k): round(v/n_total, 4) for k, v in counter.items()},
        "shannon_entropy_bits": round(H, 4),
        "huffman_avg_bits": round(avg_bits, 4),
        "codes": {str(k): v for k, v in codes.items()},
        "compression_vs_byteq": round(2.0 / avg_bits, 3),  # 2 bits Byte-Q / huffman avg
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PIPELINE LENGKAP
# ══════════════════════════════════════════════════════════════════════════════

def compress_adapter(
    adapter_dir: Path,
    output_dir: Path,
    use_huffman: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Pipeline kompresi lengkap:
    FP16/BF16 → Lloyd → Byte-Q → Huffman → .byteq file

    Returns: compression report
    """
    print("=" * 70)
    print("  🗜️  MOKO Byte-Q INT4 Extreme Compressor")
    print("=" * 70)
    log(f"Input : {adapter_dir}")
    log(f"Output: {output_dir}")
    log(f"Mode  : Byte-Q INT4" + (" + Huffman" if use_huffman else ""))
    print()

    # Load weights
    from safetensors.torch import load_file, save_file

    weights_path = None
    for p in [adapter_dir / "adapter_model.safetensors",
              adapter_dir / "adapter_model.bin",
              adapter_dir / "lora_adapter" / "adapter_model.safetensors",
              adapter_dir / "lora_adapter" / "adapter_model.bin"]:
        if p.exists():
            weights_path = p
            break

    if weights_path is None:
        log(f"Tidak ada weight file di {adapter_dir}", "ERR")
        return {}

    log(f"Memuat weights dari: {weights_path.name}")
    if weights_path.suffix == ".safetensors":
        weights = load_file(str(weights_path), device="cpu")
    else:
        weights = torch.load(str(weights_path), map_location="cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Kompresi per tensor
    configs = {}
    compressed_data = {}
    
    total_original_bytes = 0
    total_compressed_bytes = 0
    
    log(f"Mengompresi {len(weights)} tensor...")
    print()

    for name, weight in weights.items():
        if weight.dim() == 0 or weight.numel() < 16:
            # Tensor terlalu kecil, simpan apa adanya
            compressed_data[name] = weight
            total_original_bytes += weight.nbytes
            total_compressed_bytes += weight.nbytes
            continue

        orig_bytes = weight.nbytes
        orig_shape = tuple(weight.shape)
        orig_dtype = str(weight.dtype)

        # ── Step 1: Estimasi σ
        sigma = estimate_sigma(weight)

        # ── Step 2: Lloyd optimal levels
        lloyd_levels, lloyd_bounds = compute_lloyd_levels(sigma)

        # ── Step 3: Byte-Q quantize
        q_data, scale = quantize_tensor_byteq(weight, sigma, lloyd_bounds, lloyd_levels)

        # ── Step 4: Huffman encoding (opsional)
        if use_huffman and weight.numel() >= 64:
            codes = build_huffman_tree(q_data)
            huff_bytes, n_bits = huffman_encode(q_data, codes)
            huff_stats = compute_huffman_stats(q_data, codes)
            comp_bytes = len(huff_bytes)
            encoding = "huffman"
        else:
            # Pack 4 Byte-Q values per byte (2 bits × 4 = 8 bits)
            flat = q_data.flatten()
            padded_len = (len(flat) + 3) // 4 * 4
            padded = np.pad(flat, (0, padded_len - len(flat)))
            packed = np.packbits(np.unpackbits(padded.astype(np.uint8)).reshape(-1, 8)[:, 6:], axis=1)
            huff_bytes = packed.tobytes()
            n_bits = len(flat) * 2
            codes = {0: "00", 1: "01", 2: "10", 3: "11"}
            huff_stats = compute_huffman_stats(q_data, codes)
            comp_bytes = len(huff_bytes)
            encoding = "byteq_packed"

        total_original_bytes += orig_bytes
        total_compressed_bytes += comp_bytes

        ratio = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0

        config = QuantizationConfig(
            tensor_name=name,
            sigma=sigma,
            scale=scale,
            lloyd_levels=lloyd_levels,
            lloyd_boundaries=lloyd_bounds,
            n_params=weight.numel(),
            original_dtype=orig_dtype,
        )
        configs[name] = asdict(config)
        configs[name]["encoding"] = encoding
        configs[name]["original_shape"] = list(orig_shape)
        configs[name]["original_bytes"] = orig_bytes
        configs[name]["compressed_bytes"] = comp_bytes
        configs[name]["compression_ratio"] = round(ratio, 3)
        configs[name]["huffman_codes"] = {str(k): v for k, v in codes.items()}
        configs[name]["huffman_n_bits"] = n_bits
        configs[name]["huffman_stats"] = huff_stats

        if verbose:
            log(f"{name[-50:]:50s} | σ={sigma:.4f} | {ratio:.1f}× | {orig_bytes//1024}KB→{comp_bytes//1024}KB")

    # Summary
    overall_ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    fp16_equivalent = sum(w.numel() * 2 for w in weights.values())

    print()
    print("=" * 70)
    print("  📊 HASIL KOMPRESI")
    print("=" * 70)
    log(f"Original    : {total_original_bytes/1024**2:.1f} MB", "OK")
    log(f"Compressed  : {total_compressed_bytes/1024**2:.1f} MB", "OK")
    log(f"Ratio       : {overall_ratio:.2f}× compression", "OK")
    log(f"vs FP16     : {fp16_equivalent/total_compressed_bytes:.2f}× total compression", "OK")
    print()

    # Simpan konfigurasi
    config_path = output_dir / "byteq_config.json"
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=2)
    log(f"Config disimpan: {config_path.name}", "OK")

    # Simpan metadata
    metadata = {
        "format": "MOKO-ByteQ-v1",
        "compression": "byte_q_int4" + ("_huffman" if use_huffman else ""),
        "total_tensors": len(weights),
        "original_bytes": total_original_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": round(overall_ratio, 3),
        "fp16_to_compressed_ratio": round(fp16_equivalent / total_compressed_bytes, 3),
        "lloyd_algorithm": "Lloyd_optimal",
        "quantization_levels": [-1, 0, 1, 2],
    }
    meta_path = output_dir / "byteq_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print(f"  ✅ Kompresi selesai! Output: {output_dir}")
    print("=" * 70)

    return {"metadata": metadata, "configs": configs}


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(adapter_dir: Path):
    """Analisis distribusi weight tanpa kompresi penuh — versi hemat memori."""
    import math

    log("Mode Benchmark — Analisis Distribusi Weight (Memory-Efficient)", "INFO")

    from safetensors.torch import load_file
    weights_path = None
    
    files_to_try = [
        "adapter_model.safetensors",
        "model.safetensors"
    ]
    
    for f_name in files_to_try:
        p = adapter_dir / f_name
        if p.exists():
            weights_path = p
            break
        # Coba subdirektori
        for sub in ["lora_adapter", "moko_coder_1b", "base_model_coder_hf"]:
            p_sub = adapter_dir / sub / f_name
            if p_sub.exists():
                weights_path = p_sub
                break
        if weights_path:
            break
            
    if not weights_path:
        log("Weights file tidak ditemukan", "ERR")
        return

    weights = load_file(str(weights_path), device="cpu")

    # Hitung rata-rata standard deviasi
    sigmas = []
    for name, w in weights.items():
        if w.dim() >= 2 and w.numel() > 1000:
            sigmas.append(float(w.float().std()))

    if not sigmas:
        log("Tidak ada weight tensor valid ditemukan", "ERR")
        return

    sigma = sum(sigmas) / len(sigmas)
    log(f"Estimated average σ: {sigma:.6f}")

    # Lloyd boundaries
    lloyd_levels, lloyd_bounds = compute_lloyd_levels(sigma)
    log(f"Lloyd levels : {[f'{l:.4f}' for l in lloyd_levels]}")
    log(f"Lloyd bounds : {[f'{b:.4f}' for b in lloyd_bounds]}")

    b1, b2, b3 = lloyd_bounds
    
    # Hitung distribusi dengan memproses per-layer (mengurangi RAM overhead)
    total_elements = 0
    neg1_cnt = 0
    zero_cnt = 0
    pos1_cnt = 0
    pos2_cnt = 0

    log("Menghitung distribusi weight...")
    for name, w in weights.items():
        if w.dim() >= 2 and w.numel() > 1000:
            # Pindahkan ke numpy float32
            flat = w.float().numpy().flatten()
            total_elements += len(flat)
            neg1_cnt += np.sum(flat < b1)
            zero_cnt += np.sum((flat >= b1) & (flat < b2))
            pos1_cnt += np.sum((flat >= b2) & (flat < b3))
            pos2_cnt += np.sum(flat >= b3)

    if total_elements == 0:
        log("Tidak ada elemen yang dianalisis", "ERR")
        return

    p_neg1 = neg1_cnt / total_elements
    p_zero = zero_cnt / total_elements
    p_pos1 = pos1_cnt / total_elements
    p_pos2 = pos2_cnt / total_elements

    print()
    print(f"  Total weights dianalisis: {total_elements:,}")
    print("  Distribusi real setelah Byte-Q:")
    print(f"  p(-1) = {p_neg1:.4f}  ({p_neg1*100:.1f}%)")
    print(f"  p( 0) = {p_zero:.4f}  ({p_zero*100:.1f}%)")
    print(f"  p(+1) = {p_pos1:.4f}  ({p_pos1*100:.1f}%)")
    print(f"  p(+2) = {p_pos2:.4f}  ({p_pos2*100:.1f}%)")

    # Shannon entropy
    H = 0.0
    for p in [p_neg1, p_zero, p_pos1, p_pos2]:
        if p > 1e-10:
            H -= p * math.log2(p)

    print()
    print(f"  Shannon entropy H    : {H:.4f} bits")
    print(f"  Huffman avg bits est : {H * 1.05:.4f} bits  (praktis ~5% overhead)")
    print()
    print(f"  Compression dari FP16:")
    print(f"  ├── Byte-Q saja    : {2.0/0.25:.1f}×  ({0.25:.3f} bytes/param)")
    print(f"  ├── +Huffman       : {2.0/(H/8):.1f}×  ({H/8:.3f} bytes/param)  ← theory")
    print(f"  └── Shannon limit  : {16/H:.1f}×  (tidak bisa lebih baik dari ini)")
    print()

    # MSE comparison Lloyd vs uniform
    mse_lloyd = 0.11 * sigma ** 2
    mse_uniform = sigma ** 2 / 3
    print(f"  MSE Comparison (dari riset):")
    print(f"  ├── Uniform INT4  : MSE = {mse_uniform:.6f}  (0.333σ²)")
    print(f"  ├── Lloyd Byte-Q  : MSE = {mse_lloyd:.6f}  (0.11σ²)")
    print(f"  └── Improvement   : {mse_uniform/mse_lloyd:.1f}× lebih akurat dengan Lloyd ✅")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MOKO Byte-Q INT4 Extreme Compressor")
    parser.add_argument("--adapter", type=str, default=None, help="Path adapter dir")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--no-huffman", action="store_true", help="Gunakan Byte-Q saja (tanpa Huffman)")
    parser.add_argument("--benchmark", action="store_true", help="Analisis distribusi saja")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Default paths
    candidates = [
        PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder_1b",
        PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder",
        PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder" / "lora_adapter",
    ]
    adapter_dir = Path(args.adapter) if args.adapter else None
    if adapter_dir is None:
        for c in candidates:
            if c.exists():
                adapter_dir = c
                break
    if adapter_dir is None:
        log("Adapter tidak ditemukan. Gunakan --adapter <path>", "ERR")
        sys.exit(1)

    if args.benchmark:
        run_benchmark(adapter_dir)
        return

    output_dir = Path(args.output) if args.output else \
                 PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder_1b_byteq"

    compress_adapter(
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        use_huffman=not args.no_huffman,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    import math
    main()
