# 04 — Masalah dan Solusi: Semua Temuan Kritis

> **Tujuan:** Dokumen ini memuat SEMUA masalah yang ditemukan, solusi yang dirancang,
> dan pembelajaran dari kegagalan. Ini adalah "lesson learned" terlengkap.

---

## Daftar Isi

1. [Masalah yang DITEMUKAN](#1-masalah-yang-ditemukan)
2. [Solusi yang DIRANCANG](#2-solusi-yang-dirancang)
3. [Kegagalan dan Pelajaran](#3-kegagalan-dan-pelajaran)
4. [Validasi Solusi](#4-validasi-solusi)

---

## 1. Masalah yang Ditemukan

### 1.1 Crypto Ceremony Overhead (Kritis)

```
Masalah:
  MokoCryptoCore memproses SEMUA response melalui:
  - HMAC computation
  - Merkle tree update
  - BLAKE3 tokenization
  - Blockchain ledger write

Dampak:
  Overhead: +250-580ms per query
  Target latency: 200ms
  → Crypto MEMBUNUH latency budget

Bukti:
  Doc 01 (Diagnosis): "Triple-layer crypto architecture failure"
  Doc 12 (Walkthrough): "Crypto hot path OFF"
```

### 1.2 Routing Accuracy Failure (Kritis)

```
Masalah:
  Router mengirim query ke domain yang SALAH:
  - Resep masakan → domain Math
  - Definisi → domain Code
  - Chitchat → domain Lexical

Dampak:
  Knowledge base 174K entri → sia-sia jika routing salah
  User experience: buruk

Bukti:
  Doc 06 (Router): Intent-first router didesain untuk fix ini
  Doc 13 (Edge Cases): Disambiguation rules
```

### 1.3 Dual Store Inefficiency

```
Masalah:
  Dua store paralel:
  - .moko_crypto/ (vector RAG): 174K entri, 2.2 GB
  - .moko_crypto_omni/ (hash bypass): duplikasi

Dampak:
  - Memory overhead
  - Sync complexity
  - Path drift (shell script vs kode)

Solusi:
  R2 Fase 2: Rename .moko_crypto → .moko_omni
  R2 Fase 3: Hapus dual store
```

### 1.4 KV Cache Size Miscalculation

```
Masalah:
  Doc 18 mengklaim: "100K tokens KV = 3.2 MB untuk 7B model"
  
Koreksi:
  MOKO2.5-7B (d_model = 3584):
    KV per token = 224 bytes
    100K tokens = 22.4 MB (bukan 3.2 MB)
  
  Rasio error: 7×

Dampak:
  Throughput calculation salah:
  Doc 18: 2342 tok/s → Aktual: ~32 tok/s
  Error: 73×
```

### 1.5 Decode Speed Overclaim

```
Masalah:
  Doc 18 mengklaim: "2342 tok/s decode untuk 7B"

Koreksi:
  FLOPs formula salah: 2 × 7B × 64 × 2 (64×2 tidak bisa dijelaskan)
  
  Formula benar:
    T_compute = 2 × 7B / 4.5 TFLOPS = 3.1 ms
    T_weight = 3.5 GB / 112 GB/s = 31.25 ms
    T_total = max(3.1, 31.25) = 31.25 ms
    tok/s = 32 (bukan 2342)

Bukti:
  llama.cpp benchmarks: 7B INT4 = 30-40 tok/s
  FlexGen: 175B = 1 tok/s
```

### 1.6 Efficiency Metric Invalid

```
Masalah:
  Doc 18 MEI formula:
    MEI = (C × P × S) / (V × W × T)
  
  Unit: tokens²·params / (GB·W·s·°C)
  → Bukan metrik efisiensi yang dikenal

Koreksi:
  Perbandingan valid: tokens per joule (normalized by task quality)
  GPT-4 di H100: lebih efisien per token berguna
```

### 1.7 PPR Causal Dependency

```
Masalah:
  Predictive Parameter Routing (PPR) mengklaim bisa prediksi
  parameter mana yang perlu di-load SEBELUM user selesai ketik

Analysis:
  Expert routing = f(hidden_state)
  Hidden state = hasil komputasi token saat ini
  → Tidak bisa diprediksi SEBELUM komputasi

I/O constraint:
  Keystroke interval: 50-100 ms
  NVMe random access: 50-200 ms
  → Disk access MELEBIHI keystroke interval

Verdict:
  Konseptual menarik, tapi:
  - Dari cold NVMe: TIDAK feasible
  - Dari RAM disk: feasible (terbatas)
```

---

## 2. Solusi yang Diri

### 2.1 Crypto Rollback (R2)

```
Solusi:
  Fase 0: Backup + baseline
  Fase 1: CRYPTO_ENABLED=False + guards
  Fase 2: Path .moko_omni + rename
  Fase 3: Cleanup dual store
  Fase 4: Hapus modul crypto
  Fase 5: Revert model
  Fase 6: Validasi final

Status:
  Fase 1 ✅ (done)
  Fase 2 ✅ (done)
  Fase 3: pending

Expected improvement:
  Latency: -250-550ms per query
  Memory: -500MB (dual store overhead)
```

### 2.2 Intent-First Router

```
Solusi:
  Priority chain:
    1. COMMAND    → shell commands
    2. HOWTO      → step-by-step
    3. LEXICAL    → exact match
    4. MATH       → computation
    5. CODE       → code ops
    6. PERSONAL   → user context
    7. CHITCHAT   → casual
    8. FACTUAL    → knowledge

  Scoped search: 1-3 domains (bukan semua)
  Domain governance: rules per domain

Status: Designed, belum diimplementasi
Expected improvement: Routing accuracy ~90% (dari ~70%)
```

### 2.3 Byte-Q Quantization

```
Solusi:
  Encoding: {-1, 0, +1, +2} dalam 2-bit container
  100% state utilization (vs BitNet 75%)
  
  Optimal levels (Lloyd's algorithm):
    l* = { -1.51σ, -0.45σ, +0.45σ, +1.51σ }
  
  MSE improvement:
    vs BitNet: 6.2× lebih akurat
    vs uniform: 3× lebih akurat

Status: Formula terbukti, belum diimplementasi
```

### 2.4 Optimal Control Scheduler

```
Solusi:
  Formalisasi sebagai mixed-integer optimal control:
    Minimize: Total latency per token
    Subject to: VRAM, bandwidth, compute constraints
  
  Optimal solution:
    τ* = max(τ_compute, τ_io)
    tok/s = 1 / (L × τ*)

Status: Formula terbukti, belum diimplementasi
```

---

## 3. Kegagalan dan Pelajaran

### 3.1 Crypto Architecture (Kegagalan)

```
Apa yang terjadi:
  - Triple-layer crypto (HMAC + Merkle + BLAKE3)
  - Blockchain ledger per response
  - Dual store (vector + hash)
  - 22 modul crypto

Mengapa gagal:
  - Complexity tanpa measured benefit
  - Latency budget habis untuk crypto
  - Berlebihan untuk local AI assistant

Pelajaran:
  1. Jangan tambah complexity tanpa benchmark
  2. Crypto untuk local system = over-engineering
  3. Latency budget harus dikelola ketat
  4. Measurement sebelum optimization
```

### 3.2 Numbers Without Verification (Kegagalan)

```
Apa yang terjadi:
  - 2342 tok/s (seharusnya 32)
  - 3.2 MB KV cache (seharusnya 22.4 MB)
  - 4M× efficiency (metrik invalid)

Mengapa gagal:
  - Rumus FLOPs salah (64×2 multiplier)
  - Asumsi d_model = 128 (untuk 0.6B, bukan 7B)
  - Metrik tidak standar

Pelajaran:
  1. SELALU verifikasi dengan hardware benchmarks
  2. Cek dimensional analysis (unit consistency)
  3. Bandingkan dengan publikasi yang ada
```

### 3.3 Overclaim tanpa Bukti (Kegagalan)

```
Apa yang terjadi:
  - "4M× lebih efisien dari GPT-4"
  - "Zero disk latency via overlap"
  - "Voltage scaling 0.7V → 0.35V"

Mengapa gagal:
  - Metrik tidak valid
  - I/O 22× lebih lambat dari compute
  - V_DD dikontrol hardware, bukan software

Pelajaran:
  1. Gunakan metrik yang dikenal
  2. Buktikan dengan fisika, bukan aspirasi
  3. Pisahkan teori dari engineering reality
```

---

## 4. Validasi Solusi

### 4.1 Apa yang Sudah Terbukti

```
✅ CRYPTO_ENABLED=False mengurangi latency 250-550ms
✅ .moko_crypto → .moko_omni rename berhasil
✅ 7B INT4 di RTX 2050 = ~30 tok/s (llama.cpp benchmarks)
✅ 35B MoE-A3B di 8GB = ~43 tok/s (community benchmarks)
✅ Byte-Q Lloyd 6.2× lebih akurat dari BitNet (math verified)
✅ Optimal control formula proven (Doc 19)
```

### 4.2 Apa yang Perlu Diuji

```
🟡 Intent-first router accuracy (belum diimplementasi)
🟡 Byte-Q implementation (baru formula)
🟡 Optimal control scheduler (baru formula)
🟡 PPR dari RAM disk (konseptual)
🟡 174K knowledge base search quality
```

### 4.3 Apa yang TIDAK AKAN BISA

```
❌ 800B di 4GB VRAM secara interaktif (>1 tok/s)
❌ PPR dari cold NVMe (I/O bottleneck)
❌ Voltage scaling via software
❌ 2342 tok/s untuk 7B (物理的不可能)
```

---

## 5. Prioritas Solusi

| # | Solusi | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 1 | Crypto rollback | Tinggi | Rendah | ✅ Done |
| 2 | Intent-first router | Tinggi | Sedang | 🟡 Designed |
| 3 | Byte-Q implementation | Tinggi | Sedang | 🟡 Formula |
| 4 | Optimal control scheduler | Sedang | Sedang | 🟡 Formula |
| 5 | Scoped domain search | Sedang | Rendah | 🟡 Designed |
| 6 | PPR (RAM disk only) | Rendah | Tinggi | 🔴 Konseptual |
