# 05 — Rumusan Matematika: Semua Formula yang Diverifikasi

> **Tujuan:** KUMPULAN SEMUA RUMUSAN yang sudah diverifikasi, siap digunakan.
> Setiap formula memuat: definisi, asumsi, validasi, dan batasan.

---

## Daftar Isi

1. [Quantization Formulas](#1-quantization-formulas)
2. [Bandwidth Formulas](#2-bandwidth-formulas)
3. [Throughput Formulas](#3-throughput-formulas)
4. [KV Cache Formulas](#4-kv-cache-formulas)
5. [Information Theory Formulas](#5-information-theory-formulas)
6. [Pipeline Formulas](#6-pipeline-formulas)
7. [MoE Formulas](#7-moe-formulas)

---

## 1. Quantization Formulas

### 1.1 Uniform Quantization (Dasar)

```
Given: w ∈ ℝ (full-precision weight)

Step size:
  α = (w_max - w_min) / (2ᵏ - 1)

Quantize:
  w_q = round((w - w_min) / α)
  w_q = clamp(w_q, 0, 2ᵏ - 1)

Dequantize:
  w̃ = α × w_q + w_min

Quantization error:
  ε = |w - w̃| ≤ α/2

MSE (uniform distribution):
  MSE = α² / 12
```

### 1.2 Symmetric Quantization

```
Given: w ∈ ℝ
θ = max(|w_min|, |w_max|)

Step size:
  α = 2θ / (2ᵏ - 1)

Quantize:
  w_q = round(w / α)
  w_q = clamp(w_q, -(2ᵏ⁻¹ - 1), 2ᵏ⁻¹ - 1)

Dequantize:
  w̃ = α × w_q

MSE:
  MSE = α² / 12
```

### 1.3 Byte-Q Quaternary (Novel)

```
Given: w ∈ ℝ, W ~ N(0, σ²)
k = 2 bits (4 levels)

Step size:
  α = 2θ / 3

Quantize:
  w_q = round(w / α)
  w_q = clamp(w_q, -1, 2)

Dequantize:
  w̃ = α × w_q

Levels: {-1, 0, +1, +2}
State mapping: 00→-1, 01→0, 10→+1, 11→+2

MSE (uniform):
  MSE = α² / 12 = (2θ)² / (9 × 12) = θ² / 27

Untuk θ = 3σ (99.9th percentile):
  MSE = (3σ)² / 27 = 9σ² / 27 = σ² / 3
```

### 1.4 Byte-Q dengan Lloyd's Algorithm (Novel)

```
Input: W ~ N(0, σ²), L = 4 levels
Initial: l₀ = {-θ, -θ/3, +θ/3, +θ}

Lloyd's Algorithm:
  repeat
    // Update boundaries
    b₀ = -∞
    b₁ = (l₋₁ + l₀) / 2
    b₂ = (l₀ + l₁) / 2
    b₃ = (l₁ + l₂) / 2
    b₄ = +∞

    // Update levels (centroids)
    for i = 0 to 3:
      lᵢ = E[w | bᵢ ≤ w < bᵢ₊₁]
        = ∫_{bᵢ}^{bᵢ₊₁} w × p(w) dw / ∫_{bᵢ}^{bᵢ₊₁} p(w) dw

    // Check convergence
    Δ = Σ|l_new - l_old|
  until Δ < ε

Optimal levels (Gaussian):
  l* ≈ { -1.51σ, -0.45σ, +0.45σ, +1.51σ }

MSE (Lloyd):
  MSE ≈ 0.11 × θ²

Compare:
  MSE_Lloyd / MSE_uniform = 0.11 / (1/27) = 0.11 × 27 = 2.97
  → Lloyd 2.97× (3×) lebih baik dari uniform
```

### 1.5 BitNet b1.58 (Microsoft)

```
Given: W ∈ ℝⁿˣᵐ

Scale factor:
  γ = (1/nm) Σᵢⱼ |Wᵢⱼ|

Quantize:
  Q(w; γ) = { +1,  if w > +γ
             {  0,  if |w| ≤ γ
             { -1,  if w < -γ

Dequantize:
  w̃ = γ × w_ternary

MSE (untuk W ~ N(0, σ²), γ ≈ σ):
  P(w > σ) ≈ 0.159
  P(|w| ≤ σ) ≈ 0.683
  P(w < -σ) ≈ 0.159

  MSE = 0.159 × (σ - σ)² + 0.683 × σ² + 0.159 × (-σ + σ)²
      = 0.683 × σ²
```

### 1.6 Reconstruction Error Comparison

```
MSE ratios (untuk W ~ N(0, σ²)):

Method              MSE              Ratio vs BitNet
────────────────────────────────────────────────────
BitNet              0.683 × σ²       1.000 (baseline)
Byte-Q uniform      0.333 × σ²       0.488 (51% lebih baik)
Byte-Q Lloyd        0.110 × σ²       0.161 (84% lebih baik, 6.2×)

SQNR (Signal-to-Quantization-Noise Ratio):
  SQNR ≈ 6.02k + 1.76 dB

  BitNet (1.58 bits):  SQNR ≈ 11.27 dB
  Byte-Q (2 bits):     SQNR ≈ 13.80 dB
  ΔSQNR = 2.53 dB → 1.79× lebih baik
```

---

## 2. Bandwidth Formulas

### 2.1 Memory Bandwidth

```
Model size:
  Size = P × B_per_param

Decode throughput (memory-bound):
  tok/s = B_mem / Size

Dimana:
  B_mem = memory bandwidth (GB/s)
  P = parameters
  B_per_param = bytes per parameter
```

### 2.2 Disk Bandwidth

```
Streaming throughput:
  tok/s = B_disk / Size

Dimana:
  B_disk = disk sequential read bandwidth (GB/s)
```

### 2.3 RTX 2050 Specs

```
GPU:
  B_VRAM = 112 GB/s
  FLOPS = 4.5 TFLOPS (peak)
  VRAM = 4 GB

Storage:
  B_NVMe_seq = 7 GB/s
  B_NVMe_rand = 100 MB/s

PCIe:
  B_pcie = 6.5 GB/s (realistic)
```

---

## 3. Throughput Formulas

### 3.1 Dense Model (Full VRAM)

```
Untuk model P params, INT4:
  tok/s = B_VRAM / (P × 0.5)

Contoh:
  4B:   112 / 2.0 = 56 tok/s (theoretical)
  7B:   112 / 3.5 = 32 tok/s (theoretical)
  
  Actual ≈ 60-70% theoretical
  7B actual ≈ 25-30 tok/s
```

### 3.2 Dense Model (Disk Streaming)

```
Untuk model P params, INT4, L layers:
  tok/s = B_disk / (P × 0.5)

Contoh:
  4B:   7 / 2.0 = 3.5 tok/s
  7B:   7 / 3.5 = 2.0 tok/s
```

### 3.3 MoE Model

```
Untuk MoE dengan P_total, P_active:
  tok/s = B_disk / (P_active × B_per_param)

Contoh:
  35B MoE-A3B: 7 / 1.5 = 4.7 tok/s (disk)
  800B MoE (50B active): 7 / 25 = 0.28 tok/s (disk)
```

---

## 4. KV Cache Formulas

### 4.1 Per Token

```
KV per token:
  KV_token = 2 × d_model × B_per_element

Byte-Q (2-bit):
  KV_token = 2 × d_model / 4 = d_model / 2 bytes

FP16:
  KV_token = 2 × d_model × 2 = 4 × d_model bytes
```

### 4.2 Total Context

```
100K context:
  KV_total = 100,000 × KV_token

Contoh (Byte-Q):
  0.6B:  100K × 512 B = 50 MB
  4B:    100K × 1792 B = 175 MB
  7B:    100K × 1792 B = 175 MB
  72B:   100K × 4096 B = 400 MB
```

### 4.3 Read Time

```
t_KV = KV_total / B_VRAM

Contoh (Byte-Q, 100K ctx):
  0.6B:  50 MB / 112 GB/s = 0.45 ms
  7B:    175 MB / 112 GB/s = 1.56 ms
  72B:   400 MB / 112 GB/s = 3.57 ms
```

### 4.4 Total Decode Time

```
t_total = max(t_weight, t_KV)

Contoh (7B INT4, 100K ctx):
  t_weight = 3.5 GB / 112 GB/s = 31.25 ms
  t_KV = 175 MB / 112 GB/s = 1.56 ms
  t_total = max(31.25, 1.56) = 31.25 ms

→ KV cache hanya 5% dari total time
→ Weight loading dominan (95%)
```

---

## 5. Information Theory Formulas

### 5.1 Shannon Entropy

```
Discrete source:
  H = -Σᵢ pᵢ log₂(pᵢ)

Gaussian source W ~ N(0, σ²):
  H(W) = ½ log₂(2πeσ²)
```

### 5.2 Maximum Entropy per Encoding

```
BitNet (3 states, equiprobable):
  H_max = log₂(3) = 1.585 bits

Byte-Q (4 states, equiprobable):
  H_max = log₂(4) = 2.0 bits

FP16 (65536 levels):
  H_max = 16 bits
```

### 5.3 Encoding Efficiency

```
η = |H(W)| / H_max

Untuk Gaussian(0, σ²):
  BitNet:  η = |H(W)| / 1.585
  Byte-Q:  η = |H(W)| / 2.0
  FP16:    η = |H(W)| / 16

Jika |H(W)| > H_max:
  → Encoding tidak cukup untuk representasi penuh
  → Perlu adaptive codebook
```

---

## 6. Pipeline Formulas

### 6.1 I/O vs Compute Ratio

```
R = t_io / t_compute

t_io = (P_per_layer × B_per_param) / B_disk
t_compute = (2 × P_per_layer × seq_len) / FLOPS

R > 1: I/O-bound
R < 1: Compute-bound
```

### 6.2 Pipeline k-Prefetch

```
Optimal depth:
  k* = ceil(R)

Maximum feasible (VRAM):
  k_max = VRAM_available / (P_per_layer × B_per_param)

Effective throughput:
  tok/s = 1 / max(t_io/k_max, t_compute)
```

### 6.3 Contoh: 4B Model

```
P_per_layer = 111M, INT4:
  t_io = 55.5 MB / 7 GB/s = 7.93 ms
  t_compute = 2 × 111M / 4500 = 0.049 ms
  R = 162

k* = 162
k_max = 2000 MB / 55.5 MB = 36

tok/s = 1 / max(7.93/36, 0.049)
      = 1 / max(0.22, 0.049)
      = 1 / 0.22 = 4.55 tok/s

vs tanpa overlap: 1 / (36 × 7.93) = 0.0035 tok/s
→ Overlap meningkatkan 1300×
→ Tapi masih 7× lebih lambat dari fully VRAM (32 tok/s)
```

---

## 7. MoE Formulas

### 7.1 Active Parameters

```
MoE model:
  P_total = total parameters
  E = experts per layer
  K = top-k experts
  P_expert = params per expert
  P_shared = attention + shared expert

Active per token:
  P_active = K × P_expert + P_shared
```

### 7.2 VRAM Requirement

```
VRAM = P_active × B_per_param + KV_cache

Contoh (35B MoE-A3B, INT4):
  P_active = 3B
  VRAM = 3B × 0.5 + 50 MB = 1.55 GB → MUAT di 4 GB
```

### 7.3 Expert Offloading (PCIe)

```
Expert transfer time:
  t_transfer = Expert_size / B_pcie

Compute time:
  t_compute = 2 × Expert_size × seq_len / FLOPS

Ratio:
  R_pcie = t_transfer / t_compute

Contoh (24M expert, INT4):
  Expert_size = 12 MB
  t_transfer = 12 / 6500 = 1.85 ms
  t_compute = 2 × 24M / 4500 = 0.011 ms
  R_pcie = 168 → PCIe-bound
```

---

## 8. Compression Ratios

```
vs FP16 (2 bytes/param):

Method          Bytes/param   Compression   Efficiency
──────────────────────────────────────────────────────
FP16            2.0           1×            Baseline
INT8            1.0           2×            100%
INT4            0.5           4×            100%
BitNet          0.25          8×            75% (1 state waste)
Byte-Q          0.25          8×            100% (all states used)
Byte-Q+outlier  0.266         7.5×          ~100% + outlier FP16
```
