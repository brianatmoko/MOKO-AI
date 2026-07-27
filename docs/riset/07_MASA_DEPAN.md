# 07 — Prediksi Masa Depan: Rencana dan Aspirasi

> **Tujuan:** Dokumen ini memuat SEMUA rencana masa depan, aspirasi, dan riset yang belum selesai.
> Dipisah menjadi: (A) yang FEASIBLE, (B) yang ASPIRATIONAL, (C) yang TIDAK MUNGKIN.

---

## Daftar Isi

1. [Rencana Feasible (Bisa Dilakukan)](#1-rencana-feasible)
2. [Rencana Aspirational (Perlu Riset)](#2-rencana-aspirational)
3. [Yang TIDAK MUNGKIN (Fisika Melarang)](#3-yang-tidak-mungkin)
4. [Prioritas Riset](#4-prioritas-riset)

---

## 1. Rencana Feasible

### 1.1 Intent-First Router (Tinggi Prioritas)

```
Apa: Implementasi 8-class intent router
Mengapa: Routing accuracy ~90% (dari ~70%)
Effort: 1-2 minggu
Risk: Rendah

Spesifikasi:
  Priority chain:
    1. COMMAND    → regex pattern matching
    2. HOWTO      → step-by-step keyword detection
    3. LEXICAL    → exact keyword match (BM25)
    4. MATH       → math expression detection
    5. CODE       → code keyword detection
    6. PERSONAL   → user history lookup
    7. CHITCHAT   → greeting/casual detection
    8. FACTUAL    → semantic similarity (default)

  Scoped search:
    - Intent → domain mapping
    - Max 3 domains per query
    - Domain governance rules

Expected result:
  - Q01 (howto query) → HOWTO intent + BROWSING_PATH
  - Q04 (math query) → MATH intent + NeuroMath
  - Q13 (code query) → CODE intent + code domain
```

### 1.2 Byte-Q Implementation (Tinggi Prioritas)

```
Apa: Implementasi quaternary quantization {-1,0,+1,+2}
Mengapa: 6.2× lebih akurat dari BitNet, 100% state utilization
Effort: 2-3 minggu
Risk: Sedang (perlu training/finetuning)

Langkah:
  1. Implementasi Lloyd's algorithm untuk optimal levels
  2. Quantize existing model weights
  3. Benchmark: MSE vs BitNet, vs INT4
  4. If MSE bagus → finetune dengan quantization-aware training

Target:
  - Model 7B Byte-Q: 1.75 GB (vs 3.5 GB INT4)
  - VRAM 4GB: bisa muat 16B model (vs 8B INT4)
  - Throughput: ~64 tok/s theoretical (vs 32 tok/s INT4)
```

### 1.3 Optimal Control Scheduler (Sedang Prioritas)

```
Apa: Implementasi scheduler berbasis optimal control
Mengapa: Lebih baik dari heuristic (FlexGen)
Effort: 2-3 minggu
Risk: Sedang

Spesifikasi:
  Input:
    - Model architecture (layers, params per layer)
    - Hardware specs (VRAM, disk BW, compute)
    - Query characteristics (seq length, active params)
  
  Output:
    - Optimal pipeline depth (k*)
    - Load/unload schedule per layer
    - Prefetch strategy

  Constraints:
    - VRAM capacity
    - Disk bandwidth
    - Compute throughput

Expected result:
  - Optimal throughput untuk model yang tidak muat di VRAM
  - Proven optimality (bukan heuristic)
```

### 1.4 Scoped Domain Search (Rendah Prioritas)

```
Apa: Batasi pencarian ke 1-3 domain per query
Mengapa: Mengurangi noise, meningkatkan relevance
Effort: 1 minggu
Risk: Rendah

Spesifikasi:
  Intent → Domain mapping:
    COMMAND → general
    HOWTO → general, code
    LEXICAL → lexical
    MATH → math
    CODE → code
    PERSONAL → personal
    CHITCHAT → general
    FACTUAL → all (fallback)

Expected result:
  - Search latency: -50% (dari 8 domain ke 1-3)
  - Relevance: +20% (less noise)
```

---

## 2. Rencana Aspirational

### 2.1 PPR dari RAM Disk (Perlu Riset)

```
Apa: Pre-load parameter dari RAM disk berdasarkan prediksi
Mengapa: Mengurangi latency inference
Effort: 3-6 bulan
Risk: Tinggi

Analisis feasibility:
  RAM disk BW: >50 GB/s (NVMe 7× lebih lambat)
  Keystroke interval: 50-100 ms
  RAM access: <1 ms
  
  → Dari RAM disk: FEASIBLE (latency cukup)
  → Dari NVMe: TIDAK FEASIBLE (latency melebihi keystroke)

Riset yang diperlukan:
  1. Keystroke prediction model (next-character prediction)
  2. Token prediction dari partial input
  3. Expert routing prediction dari token sequence
  4. Pre-loading schedule

Status: Konseptual, belum ada implementasi
```

### 2.2 Parameter Streaming untuk 70B (Perlu Riset)

```
Apa: Run 70B model di 4GB VRAM melalui streaming
Mengapa: Kualitas jauh lebih baik dari 7B
Effort: 6-12 bulan
Risk: Tinggi

Analisis:
  70B INT4 = 35 GB
  NVMe 7 GB/s
  Time per token: 35 / 7 = 5 detik
  → Tidak interaktif (5 detik per token)

  Dengan pipeline overlap:
    Per layer: 35B/80 = 437.5 MB → 62.5 ms I/O
    k_max = 2000 MB / 437.5 MB = 4.6 → k = 4
    Effective: 62.5 / 4 = 15.6 ms per layer
    Total: 80 × 15.6 = 1.25 detik per token
    → 0.8 tok/s (marginal interaktif)

Status: Matematika menunjukkan BATAS 0.8-2 tok/s untuk 70B
```

### 2.3 Adaptive Codebook (Perlu Riset)

```
Apa: Level quantization tidak uniform berdasarkan weight distribution
Mengapa: 3× lebih akurat dari uniform (Lloyd's algorithm)
Effort: 1-2 bulan
Risk: Sedang

Riset yang diperlukan:
  1. Analisis weight distribution per layer
  2. Lloyd's algorithm convergence untuk distribusi berbeda
  3. Mixed codebook (level berbeda per layer)
  4. Hardware implementation (dequantization kernel)

Status: Formula sudah ada, perlu implementasi
```

---

## 3. Yang TIDAK MUNGKIN

### 3.1 800B di 4GB VRAM secara Interaktif

```
Klaim: "800B model bisa jalan di 4GB VRAM"
Realita:
  800B INT4 = 400 GB
  Active 5% = 40B = 20 GB
  NVMe 7 GB/s → 20/7 = 2.86 detik per token
  → 0.35 tok/s (TIDAK interaktif)

  Even dengan optimal overlap:
    Per layer: 20B/80 = 250 MB → 35.7 ms I/O
    k_max = 2000/250 = 8
    Effective: 35.7/8 = 4.46 ms per layer
    Total: 80 × 4.46 = 357 ms → 2.8 tok/s
    
    TAPI: ini tanpa mempertimbangkan shared params
    (attention, embedding) yang harus tetap di VRAM

  Verdict: 1-5 tok/s adalah BATAS FISIKA
  → Bukan interaktif untuk chatting
```

### 3.2 Voltage Scaling via Software

```
Klaim: "V_DD bisa diturunkan dari 0.7V ke 0.35V"
Realita:
  V_DD dikontrol oleh VRM (voltage regulator module)
  Software TIDAK BISA mengubah V_DD
  0.35V di bawah V_th transistor 7nm (0.3-0.4V)
  → Transistor tidak switch

  Verdict: Impossible via software
```

### 3.3 Zero Disk Latency via Overlap

```
Klaim: "Disk latency bisa dihilangkan dengan pipeline overlap"
Realita:
  I/O per layer: 69 ms (4B model)
  Compute per layer: 0.049 ms
  Ratio: 1408× I/O lebih lambat
  
  k* = 1408 layer prefetch needed
  k_max = 36 (VRAM constraint)
  
  → Overlap hanya mengurangi 36× dari 1408×
  → Masih 39× I/O-bound

  Verdict: Overlap membantu TAPI tidak bisa menghilangkan I/O bottleneck
```

### 3.4 4M× Efficiency vs GPT-4

```
Klaim: "MOKO 4 juta kali lebih efisien dari GPT-4"
Realita:
  MEI formula: (C × P × S) / (V × W × T)
  Unit: tokens²·params / (GB·W·s·°C)
  → Bukan metrik efisiensi yang dikenal
  
  Perbandingan valid:
    GPT-4: 1.8T params, 8×H100, high quality
    MOKO: 4B params, 1×RTX 2050, limited quality
  
    Per quality-adjusted token:
      GPT-4 lebih efisien (selesaikan tugas dalam lebih sedikit token)

  Verdict: Metrik invalid, perbandingan tidak apple-to-apple
```

---

## 4. Prioritas Riset

### 4.1 Prioritas Tinggi (Feasible, High Impact)

| # | Rencana | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| 1 | Intent-first router | 1-2 minggu | Tinggi | Rendah |
| 2 | Byte-Q implementation | 2-3 minggu | Tinggi | Sedang |
| 3 | Scoped domain search | 1 minggu | Sedang | Rendah |
| 4 | Optimal control scheduler | 2-3 minggu | Sedang | Sedang |

### 4.2 Prioritas Sedang (Perlu Riset)

| # | Rencana | Effort | Impact | Risk |
|---|---------|--------|--------|------|
| 5 | PPR dari RAM disk | 3-6 bulan | Tinggi | Tinggi |
| 6 | Adaptive codebook | 1-2 bulan | Sedang | Sedang |
| 7 | Parameter streaming 70B | 6-12 bulan | Tinggi | Tinggi |

### 4.3 Tidak Dilakukan (Fisika Melarang)

| # | Klaim | Alasan |
|---|-------|--------|
| 1 | 800B interaktif di 4GB | I/O bottleneck: 1-5 tok/s max |
| 2 | Voltage scaling via software | Hardware-controlled |
| 3 | Zero disk latency | Physics: I/O >> compute |
| 4 | 4M× efficiency vs GPT-4 | Invalid metric |
| 5 | 2342 tok/s untuk 7B | FLOPs formula salah |

---

## 5. Roadmap Realistis

```
Bulan 1-2: Crypto Rollback + Intent Router
  ├── R2 Fase 3-6: Cleanup crypto
  ├── Intent-first router implementation
  └── Scoped search implementation

Bulan 3-4: Byte-Q + Optimization
  ├── Byte-Q quantization implementation
  ├── Lloyd's algorithm untuk optimal levels
  ├── Benchmark MSE vs INT4
  └── Optimal control scheduler

Bulan 5-6: Evaluation + Next Steps
  ├── Full system benchmark
  ├── Accuracy evaluation
  ├── Latency evaluation
  └── Decide: PPR atau parameter streaming

Realistis untuk 4GB VRAM:
  ✅ 7B INT4 fully in VRAM: ~30 tok/s
  ✅ 35B MoE-A3B fully in VRAM: ~43 tok/s
  ✅ Byte-Q 7B: ~64 tok/s (theoretical)
  ✅ Intent-first router: ~90% accuracy
  ❌ Anything >70B: lambat (1-5 tok/s)
```
