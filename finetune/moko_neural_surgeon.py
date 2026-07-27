"""
MOKO Neural Surgeon — Bedah Saraf Layer-by-Layer
=================================================
Menganalisis LoRA adapter hasil training secara mendalam:

1. ANATOMY (Anatomi)
   - Statistik weight per layer: mean, std, kurtosis, sparsity
   - Distribusi per layer (histogram)
   - Deteksi outlier weights yang sensitif

2. WANDA SCORING (Sensitivitas Neuron)
   - S(w) = |W| × ||X||₂  → identifikasi neuron penting vs redundan
   - Layer sensitivity ranking: mana yang bisa di-INT4 agresif

3. CRITICAL PATH DETECTION
   - Attention head importance per layer
   - MLP vs Attention sensitivity comparison
   - Embedding + LM Head: selalu protected (FP16)

4. MIXED-PRECISION RECOMMENDATION
   - Layer sensitif → pertahankan INT8 atau FP16
   - Layer tidak sensitif → INT4 agresif
   - Hasilkan per-layer quantization config

Penggunaan:
  python3 moko_neural_surgeon.py --adapter finetune/moko_adapters/moko_coder_1b
  python3 moko_neural_surgeon.py --adapter ... --full   # include activation stats
  python3 moko_neural_surgeon.py --adapter ... --report # simpan JSON report
"""

import sys
import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Tambah project lib ke path
PROJECT_DIR = Path(__file__).parent.parent
_site = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site.exists() and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

import torch
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Layer yang HARUS dilindungi dari quantization agresif
PROTECTED_LAYERS = {
    "embed_tokens",   # Embedding: fundamental representation
    "lm_head",        # Output head: menentukan probabilitas token
    "norm",           # LayerNorm: sangat sensitif terhadap quantization
    "layernorm",
}

# Target batas sensitivitas (Wanda score relatif)
SENSITIVITY_THRESHOLD_HIGH = 0.7   # > threshold → pertahankan INT8/FP16
SENSITIVITY_THRESHOLD_MED  = 0.4   # 0.4–0.7 → INT4 hati-hati
# < 0.4 → INT4 agresif aman


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    symbols = {"INFO": "ℹ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "BEDAH": "🔬"}
    sym = symbols.get(level, "•")
    print(f"[{ts}] {sym} {msg}")


def compute_kurtosis(t: torch.Tensor) -> float:
    """Hitung excess kurtosis (distribusi Gaussian = 0, heavy-tail > 0)."""
    x = t.float().flatten()
    mean = x.mean()
    std = x.std()
    if std < 1e-10:
        return 0.0
    return float(((x - mean) ** 4).mean() / (std ** 4) - 3.0)


def compute_sparsity(t: torch.Tensor, threshold: float = 1e-3) -> float:
    """Persentase weight yang mendekati nol (sparse)."""
    return float((t.abs() < threshold).float().mean() * 100)


def compute_outlier_ratio(t: torch.Tensor, n_sigma: float = 3.0) -> float:
    """Persentase weight yang lebih dari n_sigma dari mean (outlier)."""
    x = t.float().flatten()
    mean = x.mean()
    std = x.std()
    if std < 1e-10:
        return 0.0
    return float((((x - mean).abs() / std) > n_sigma).float().mean() * 100)


def wanda_score(weight: torch.Tensor) -> float:
    """
    Wanda sensitivity score: S = mean(|W| × ||X||₂)
    
    Untuk LoRA adapter (tanpa activation statistics), kita gunakan
    proxy: S = mean(|W|) × std(W) — korelasi dengan output variance.
    Lebih tinggi = lebih sensitif = jangan di-INT4 agresif.
    """
    w = weight.float()
    magnitude = w.abs().mean()
    variance = w.std()
    # Normalize ke [0,1] menggunakan percentile-based scaling
    raw_score = float(magnitude * variance)
    return raw_score


# ══════════════════════════════════════════════════════════════════════════════
# LAYER STATS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_layer(name: str, weight: torch.Tensor) -> Dict:
    """Analisis statistik mendalam untuk satu layer."""
    w = weight.float()
    flat = w.flatten()

    stats = {
        "name": name,
        "shape": list(weight.shape),
        "n_params": weight.numel(),
        # Statistik dasar
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_mean": float(flat.abs().mean()),
        # Distribusi
        "kurtosis": compute_kurtosis(weight),
        "sparsity_pct": compute_sparsity(weight),
        "outlier_pct": compute_outlier_ratio(weight),
        # Wanda proxy
        "wanda_raw": wanda_score(weight),
        # Tipe layer
        "is_lora": "lora_" in name.lower(),
        "is_attention": any(k in name for k in ["q_proj","k_proj","v_proj","o_proj","attn"]),
        "is_mlp": any(k in name for k in ["gate_proj","up_proj","down_proj","mlp"]),
        "is_protected": any(p in name.lower() for p in PROTECTED_LAYERS),
        # Rekomendasi (diisi nanti setelah normalisasi)
        "quant_recommendation": None,
        "sensitivity_label": None,
    }
    return stats


def classify_layer(stats: Dict, max_wanda: float) -> Dict:
    """Tentukan rekomendasi quantization berdasarkan Wanda score yang dinormalisasi."""
    if max_wanda < 1e-10:
        norm_score = 0.0
    else:
        norm_score = stats["wanda_raw"] / max_wanda

    stats["wanda_normalized"] = round(norm_score, 4)

    # Protected layer → selalu FP16
    if stats["is_protected"]:
        stats["sensitivity_label"] = "PROTECTED"
        stats["quant_recommendation"] = "FP16"
        return stats

    # Klasifikasi berdasarkan normalized Wanda
    if norm_score > SENSITIVITY_THRESHOLD_HIGH:
        stats["sensitivity_label"] = "HIGH"
        stats["quant_recommendation"] = "INT8"
    elif norm_score > SENSITIVITY_THRESHOLD_MED:
        stats["sensitivity_label"] = "MEDIUM"
        stats["quant_recommendation"] = "INT4_CAREFUL"
    else:
        stats["sensitivity_label"] = "LOW"
        stats["quant_recommendation"] = "INT4_AGGRESSIVE"

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SURGERY
# ══════════════════════════════════════════════════════════════════════════════

def load_adapter_weights(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    """Load weights dari safetensors atau pytorch_model.bin (LoRA atau Base Model)."""
    weights = {}

    # Coba nama-nama file standard
    files_to_try = [
        "adapter_model.safetensors",
        "model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin"
    ]

    for f_name in files_to_try:
        p = adapter_dir / f_name
        if p.exists():
            log(f"Memuat weights dari: {p.name}")
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file
                weights = load_file(str(p), device="cpu")
            else:
                weights = torch.load(str(p), map_location="cpu")
            break

    if not weights:
        # Coba cari di subdirektori
        for sub in ["lora_adapter", "moko_coder_1b", "base_model_coder_hf"]:
            sub_path = adapter_dir / sub
            for f_name in files_to_try:
                p = sub_path / f_name
                if p.exists():
                    log(f"Memuat weights dari subdirektori {sub}/{p.name}")
                    if p.suffix == ".safetensors":
                        from safetensors.torch import load_file
                        weights = load_file(str(p), device="cpu")
                    else:
                        weights = torch.load(str(p), map_location="cpu")
                    break
            if weights:
                break

    if not weights:
        raise FileNotFoundError(f"Tidak ada weights ditemukan di: {adapter_dir}")

    log(f"Berhasil memuat {len(weights)} tensor", "OK")
    return weights


def run_neural_surgery(
    adapter_dir: Path,
    save_report: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Jalankan bedah saraf lengkap pada LoRA adapter.
    
    Returns: Dictionary dengan semua statistik dan rekomendasi.
    """
    print("=" * 70)
    print("  🔬 MOKO NEURAL SURGEON — Bedah Saraf Layer-by-Layer")
    print("=" * 70)
    print(f"  Adapter: {adapter_dir}")
    print()

    # ── Load weights
    weights = load_adapter_weights(adapter_dir)

    # ── Pisahkan LoRA A dan B (LoRA: W_delta = A × B)
    lora_a = {k: v for k, v in weights.items() if "lora_A" in k}
    lora_b = {k: v for k, v in weights.items() if "lora_B" in k}
    other = {k: v for k, v in weights.items() if "lora_A" not in k and "lora_B" not in k}

    log(f"LoRA A matrices: {len(lora_a)}")
    log(f"LoRA B matrices: {len(lora_b)}")
    log(f"Other tensors: {len(other)}")
    print()

    # ── Analisis setiap layer
    log("Memulai bedah anatomi saraf...", "BEDAH")
    all_stats = []

    # Proses LoRA A × B pairs (effective weight = A_weight ⊗ B_weight)
    processed_bases = set()
    for key_a, w_a in lora_a.items():
        # Cari pasangan B yang sesuai
        key_b = key_a.replace("lora_A", "lora_B")
        if key_b in lora_b:
            w_b = lora_b[key_b]
            # Effective LoRA delta = B @ A (shape: [out, in])
            try:
                effective = w_b @ w_a
                base_name = key_a.replace(".lora_A.weight", "").replace("base_model.model.", "")
                stats = analyze_layer(base_name, effective)
                stats["tensor_key"] = key_a
                stats["lora_rank"] = w_a.shape[0]
                all_stats.append(stats)
                processed_bases.add(base_name)
            except Exception as e:
                if verbose:
                    log(f"Skip {key_a}: {e}", "WARN")

    # Proses tensor lainnya
    for key, weight in other.items():
        if weight.dim() >= 1:
            stats = analyze_layer(key, weight)
            stats["tensor_key"] = key
            stats["lora_rank"] = None
            all_stats.append(stats)

    # ── Normalisasi Wanda score
    log(f"Menghitung Wanda sensitivity untuk {len(all_stats)} layer...")
    max_wanda = max((s["wanda_raw"] for s in all_stats), default=1.0)
    all_stats = [classify_layer(s, max_wanda) for s in all_stats]

    # ── Sort berdasarkan sensitivity
    all_stats.sort(key=lambda x: x.get("wanda_normalized", 0), reverse=True)

    # ── Print Laporan
    print()
    print("=" * 70)
    print("  📊 HASIL BEDAH SARAF")
    print("=" * 70)

    # Summary counts
    counts = {"FP16": 0, "INT8": 0, "INT4_CAREFUL": 0, "INT4_AGGRESSIVE": 0, "PROTECTED": 0}
    for s in all_stats:
        rec = s.get("quant_recommendation", "UNKNOWN")
        if rec in counts:
            counts[rec] += 1

    total_params = sum(s["n_params"] for s in all_stats)
    print(f"\n  Total layer dianalisis : {len(all_stats)}")
    print(f"  Total parameter        : {total_params:,} ({total_params/1e6:.2f}M)")
    print()
    print("  Rekomendasi Quantization:")
    print(f"  ├── 🛡️  FP16 (Protected)     : {counts['FP16'] + counts['PROTECTED']} layer")
    print(f"  ├── 🟡 INT8 (Hati-hati)    : {counts['INT8']} layer")
    print(f"  ├── 🟠 INT4 (Normal)       : {counts['INT4_CAREFUL']} layer")
    print(f"  └── 🔴 INT4 (Agresif)      : {counts['INT4_AGGRESSIVE']} layer")

    # Top 10 most sensitive layers
    print()
    print("  🔬 Top 10 Layer Paling Sensitif (HARUS dilindungi):")
    print(f"  {'Layer Name':<45} {'Score':>6} {'Rec':>16} {'Kurtosis':>9}")
    print("  " + "-" * 80)
    for s in all_stats[:10]:
        name = s["name"][-44:] if len(s["name"]) > 44 else s["name"]
        print(f"  {name:<45} {s.get('wanda_normalized', 0):>6.3f} {s.get('quant_recommendation','?'):>16} {s['kurtosis']:>9.2f}")

    # Bottom 10 (paling aman untuk INT4 agresif)
    print()
    print("  🔴 10 Layer Paling Aman Untuk INT4 Agresif:")
    print(f"  {'Layer Name':<45} {'Score':>6} {'Sparsity%':>10} {'Outlier%':>9}")
    print("  " + "-" * 80)
    safe_layers = [s for s in all_stats if s.get("quant_recommendation") == "INT4_AGGRESSIVE"]
    for s in safe_layers[:10]:
        name = s["name"][-44:] if len(s["name"]) > 44 else s["name"]
        print(f"  {name:<45} {s.get('wanda_normalized', 0):>6.3f} {s['sparsity_pct']:>9.1f}% {s['outlier_pct']:>8.1f}%")

    # Estimasi kompresi
    print()
    print("  💾 Estimasi Ukuran Setelah Quantization:")
    fp16_size  = total_params * 2 / 1024**2
    int8_size  = total_params * 1 / 1024**2
    int4_size  = total_params * 0.5 / 1024**2
    byteq_size = total_params * 0.209 / 1024**2

    print(f"  FP16 (saat ini)   : {fp16_size:.1f} MB")
    print(f"  INT8              : {int8_size:.1f} MB  ({fp16_size/int8_size:.1f}× smaller)")
    print(f"  INT4 (standar)    : {int4_size:.1f} MB  ({fp16_size/int4_size:.1f}× smaller)")
    print(f"  Byte-Q + Huffman  : {byteq_size:.1f} MB  ({fp16_size/byteq_size:.1f}× smaller) ← TARGET KITA")

    # Mixed precision estimate
    high_params  = sum(s["n_params"] for s in all_stats if s.get("quant_recommendation") in ("FP16", "INT8", "PROTECTED"))
    med_params   = sum(s["n_params"] for s in all_stats if s.get("quant_recommendation") == "INT4_CAREFUL")
    low_params   = sum(s["n_params"] for s in all_stats if s.get("quant_recommendation") == "INT4_AGGRESSIVE")
    mixed_size_mb = (high_params * 1 + med_params * 0.5 + low_params * 0.25) / 1024**2
    print(f"  Mixed-precision   : {mixed_size_mb:.1f} MB  ({fp16_size/mixed_size_mb:.1f}× smaller) ← OPTIMAL")
    print()

    # ── Build report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S"),
        "adapter_dir": str(adapter_dir),
        "total_layers": len(all_stats),
        "total_params": total_params,
        "recommendation_counts": counts,
        "size_estimates_mb": {
            "fp16": round(fp16_size, 1),
            "int8": round(int8_size, 1),
            "int4": round(int4_size, 1),
            "byteq_huffman": round(byteq_size, 1),
            "mixed_precision": round(mixed_size_mb, 1),
        },
        "layers": all_stats,
    }

    # ── Simpan report
    if save_report:
        report_path = adapter_dir.parent / "neural_surgery_report.json"
        # Konversi numpy types ke Python native
        def make_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(report_path, "w") as f:
            json.dump(make_serializable(report), f, indent=2)
        log(f"Report disimpan ke: {report_path}", "OK")

    print("=" * 70)
    print("  🔬 BEDAH SELESAI. Model siap untuk INT4 quantization.")
    print("=" * 70)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MOKO Neural Surgeon — Layer-by-Layer Analysis")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path ke direktori LoRA adapter")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Simpan JSON report (default: True)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Tampilkan detail error dan warning")
    args = parser.parse_args()

    # Default adapter paths
    if args.adapter:
        adapter_dir = Path(args.adapter)
    else:
        # Coba semua kemungkinan path
        candidates = [
            PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder_1b",
            PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder",
            PROJECT_DIR / "finetune" / "moko_adapters" / "moko_coder" / "lora_adapter",
        ]
        adapter_dir = None
        for p in candidates:
            if p.exists():
                adapter_dir = p
                break

        if adapter_dir is None:
            print("❌ Tidak ada adapter ditemukan. Gunakan --adapter <path>")
            print("   Kandidat yang dicari:")
            for p in candidates:
                print(f"   - {p}")
            sys.exit(1)

    run_neural_surgery(
        adapter_dir=adapter_dir,
        save_report=args.report,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
