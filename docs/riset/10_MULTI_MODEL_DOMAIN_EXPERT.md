# 10 — Multi-Model Domain Expert: Sistem yang Seperti Manusia

> **Tujuan:** Konsep arsitektur di mana VRAM 4GB diperlakukan seperti otak manusia —
> tidak mencoba jadi ahli SEMUA bidang, tapi menjadi ahli di BIDANG SPESIFIK.
> Setiap model = 1 orang ahli. Router = manajer yang mengarahkan ke ahli yang tepat.

---

## Daftar Isi

1. [Analogi: Otak Manusia](#1-analogi-otak-manusia)
2. [MoDEM: Bukti Konsep Sudah Ada](#2-modem-bukti-konsep-sudah-ada)
3. [Arsitektur MOKO Multi-Model](#3-arsitektur-moko-multi-model)
4. [Router sebagai Manajer](#4-router-sebagai-manajer)
5. [RAG + AI Agent untuk Direktori](#5-rag--ai-agent-untuk-direktori)
6. [Formulasi Efisiensi](#6-formulasi-efisiensi)
7. [Perbandingan dengan Pendekatan Lain](#7-perbandingan-dengan-pendekatan-lain)
8. [Spesifikasi Model](#8-spesifikasi-model)

---

## 1. Analogi: Otak Manusia

```
Otak manusia:
  - Tidak ada "area" yang bisa SEMUA bidang
  - Area Broca → bahasa
  - Area Wernicke → pemahaman
  - Area Motorik → pergerakan
  - Area Visual → penglihatan

  → Setiap area SPESIALIS
  → Komunikasi antar area (corpus callosum)
  → Efisien: tidak semua area aktif sekaligus

MOKO Multi-Model:
  - Model Coding → expert programming
  - Model Math → expert reasoning
  - Model Security → expert hacking/defense
  - Model General → expert chitchat/general

  → Setiap model SPESIALIS
  → Router mengarahkan (corpus callosum)
  → Efisien: tidak semua model aktif sekaligus
```

### Mengapa Ini Lebih Baik dari Model Besar

```
Model besar (70B):
  ✅ Bisa semua bidang
  ❌ Butuh 35 GB VRAM (INT4)
  ❌ Lambat: ~5 tok/s dari disk
  ❌ Generalist: tidak ahli di satu bidang

Multi-Model (4 × 1-2B):
  ✅ Total VRAM: 4 × 0.5 GB = 2 GB (INT4)
  ✅ Cepat: ~50 tok/s per model
  ✅ Specialist: expert di bidang masing-masing
  ✅ Muat di 4 GB VRAM
```

---

## 2. MoDEM: Bukti Konsep Sudah Ada

### 2.1 Paper: arXiv:2410.07490

```
Title: "MoDEM: Mixture of Domain Expert Models"
Authors: Simonds, Kurniawan, Lau (2024)

Konsep:
  - BERT-based router → direct ke domain expert
  - Expert models: health, math, science
  - Each model: fine-tuned untuk 1 domain

Hasil:
  - Outperforms general-purpose models of comparable size
  - Superior performance-to-cost ratio
  - "Paradigm shift: ecosystems of smaller, specialized models"

Quote penting:
  "Rather than focusing solely on creating increasingly large,
   general-purpose models, the future of AI may lie in developing
   ecosystems of smaller, highly specialized models coupled with
   sophisticated routing systems."
```

### 2.2 Apa yang BELUM dilakukan MoDEM

```
MoDEM:
  - Router: BERT-based (butuh model terpisah)
  - Expert: text-only
  - RAG: tidak disebut
  - Agent: tidak ada
  - Directory management: tidak ada

MOKO Multi-Model:
  - Router: rule-based + semantic (tidak perlu model terpisah)
  - Expert: text + code + math + security
  - RAG: integrated (174K knowledge base)
  - Agent: AI agent untuk directory management
  - Directory management: automated file organization
```

---

## 3. Arsitektur MOKO Multi-Model

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                            │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: INTENT ROUTER (Manajer)                       │
│  - Deteksi intent: coding? math? security? general?     │
│  - Rule-based + semantic similarity                     │
│  - Output: domain label + confidence                    │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: MODEL DISPATCHER (Lift)                       │
│  - Load model yang sesuai ke VRAM                       │
│  - Unload model sebelumnya (jika perlu)                 │
│  - Manage VRAM allocation                               │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: DOMAIN EXPERT MODELS (Ahli)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  CODING  │ │   MATH   │ │ SECURITY │ │ GENERAL  │  │
│  │  1.5B    │ │  1.0B    │ │  1.0B    │ │  2.0B    │  │
│  │ Q4_K_M   │ │ Q4_K_M   │ │ Q4_K_M   │ │ Q4_K_M   │  │
│  │ 0.75 GB  │ │ 0.5 GB   │ │ 0.5 GB   │ │ 1.0 GB   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: RAG + KNOWLEDGE BASE (Perpustakaan)           │
│  - .moko_omni/ (174K entries)                           │
│  - Domain-specific knowledge per model                  │
│  - Vector search + BM25 hybrid                          │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 5: AI AGENT (Sekretaris)                         │
│  - File management:组织, backup, cleanup                │
│  - Directory structure optimization                     │
│  - Knowledge ingestion pipeline                         │
│  - Session management                                   │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    USER OUTPUT                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 VRAM Allocation

```
RTX 2050: 4 GB VRAM

Strategy: 1 model aktif + 1 model standby

Scenario A: Coding query
  Active:  Coding model (0.75 GB)
  Standby: General model (1.0 GB) — preload
  KV cache: 0.5 GB
  OS: 0.5 GB
  Total: 2.75 GB (69% utilization)

Scenario B: Math query
  Active:  Math model (0.5 GB)
  Standby: General model (1.0 GB) — preload
  KV cache: 0.5 GB
  OS: 0.5 GB
  Total: 2.5 GB (63% utilization)

Scenario C: Security query
  Active:  Security model (0.5 GB)
  Standby: Coding model (0.75 GB) — preload
  KV cache: 0.5 GB
  OS: 0.5 GB
  Total: 2.25 GB (56% utilization)
```

### 3.3 Model Switching

```
Switching time:
  -unload old model: ~100 ms (VRAM writeback)
  -load new model: ~500 ms (from SSD)
  -total: ~600 ms

Optimization:
  - Keep 2 models in VRAM (active + standby)
  - Predict next model berdasarkan conversation pattern
  - Pre-load standby model saat idle

  → Effective switching: ~100 ms (hanya unload)
```

---

## 4. Router sebagai Manajer

### 4.1 Intent Detection (Tanpa Model)

```
Rule-based detection:

CODING keywords:
  function, class, def, import, git, debug, compile,
  algorithm, data structure, API, database, SQL,
  python, javascript, rust, golang, etc.

MATH keywords:
  calculate, solve, prove, theorem, formula, equation,
  integral, derivative, matrix, vector, probability,
  algebra, calculus, geometry, etc.

SECURITY keywords:
  exploit, vulnerability, penetration, firewall, encrypt,
  hack, crack, CVE, buffer overflow, SQL injection,
  XSS, CSRF, malware, reverse engineering, etc.

GENERAL keywords:
  hello, thanks, help, explain, what is, how to,
  recommend, compare, etc.

Confidence scoring:
  score(domain) = Σ keyword_match × weight
  selected_domain = argmax(score)
  
  if max(score) < threshold:
    → fallback ke GENERAL model
```

### 4.2 Semantic Fallback

```
Jika rule-based confidence < threshold:
  → Gunakan embedding similarity

  Query embedding = embed(query)
  Domain centroids = {centroid_coding, centroid_math, ...}
  
  similarity(domain) = cosine(query_embedding, centroid[domain])
  selected_domain = argmax(similarity)

  → Lebih akurat tapi lebih lambat
  → Hanya dipakai sebagai fallback
```

### 4.3 Multi-Intent Handling

```
Query: "tulis fungsi Python untuk menghitung integral"

Intent detection:
  coding_score = 0.8 (ada "fungsi", "Python")
  math_score = 0.6 (ada "integral")
  
  → Dual intent: CODING + MATH
  → Strategy: coding model dengan math context

Execution:
  1. Load coding model
  2. Inject math knowledge dari RAG
  3. Generate code dengan mathematical correctness
```

---

## 5. RAG + AI Agent untuk Direktori

### 5.1 AI Agent: Sekretaris Digital

```
Fungsi Agent:
  1. File Organization
     - Auto-rename files berdasarkan konten
     - Move files ke direktori yang benar
     - Create directory structure otomatis

  2. Knowledge Ingestion
     - Monitor direktori untuk file baru
     - Auto-ingest ke knowledge base
     - Update vector index

  3. Backup Management
     - Auto-backup knowledge base
     - Version control untuk model weights
     - Cleanup old files

  4. Session Management
     - Save/load conversation history
     - Context switching antar session
     - Memory consolidation
```

### 5.2 Directory Structure

```
.moko/
├── models/
│   ├── coding/
│   │   ├── moko-coding-1.5B-Q4_K_M.gguf
│   │   └── moko-coding-1.5B-Q4_K_M.meta.json
│   ├── math/
│   │   ├── moko-math-1.0B-Q4_K_M.gguf
│   │   └── moko-math-1.0B-Q4_K_M.meta.json
│   ├── security/
│   │   ├── moko-security-1.0B-Q4_K_M.gguf
│   │   └── moko-security-1.0B-Q4_K_M.meta.json
│   └── general/
│       ├── moko-general-2.0B-Q4_K_M.gguf
│       └── moko-general-2.0B-Q4_K_M.meta.json
├── knowledge/
│   ├── coding/
│   │   ├── code/ (1,687 entries)
│   │   └── cybersecurity/ (978 entries)
│   ├── math/
│   │   ├── math/ (1,051 entries)
│   │   └── physics/ (660 entries)
│   ├── security/
│   │   └── cyberoffensive/ (929 entries)
│   └── general/
│       ├── general/ (1,222 entries)
│       ├── lexical/ (10,830 entries)
│       └── personal/ (759 entries)
├── sessions/
│   ├── session_001.jsonl
│   └── session_002.jsonl
├── cache/
│   ├── embeddings/
│   └── responses/
└── config/
    ├── router_rules.json
    └── model_registry.json
```

### 5.3 Agent Workflow

```
Trigger: File baru ditambahkan ke .moko/

Workflow:
  1. Detect file type (extension + content analysis)
  2. Classify domain (coding/math/security/general)
  3. Extract knowledge (chunking + embedding)
  4. Store ke domain-specific directory
  5. Update vector index
  6. Log activity

Contoh:
  File: "buffer_overflow_exploit.py"
  → Domain: security
  → Action: move ke .moko/knowledge/security/
  → Embed & store: .moko_omni/security/
  → Update index
```

---

## 6. Formulasi Efisiensi

### 6.1 Multi-Model vs Single Large Model

```
Single 70B model (INT4):
  Size: 35 GB
  VRAM needed: 35 GB
  tok/s: ~5 (dari disk)
  Quality: generalist

Multi-Model (4 × small):
  Total size: 2.75 GB
  VRAM needed: 2.75 GB (1 aktif)
  tok/s: ~50 (dari VRAM)
  Quality: specialist

Efficiency ratio:
  tok/s per GB: 50/0.75 = 66.7 (multi) vs 5/35 = 0.14 (single)
  → 476× lebih efisien per GB
```

### 6.2 Quality per Resource

```
Definisi:
  η = Quality_score / Resources_used

Single 70B:
  Quality: 85% (generalist)
  Resources: 35 GB VRAM
  η = 85/35 = 2.43

Multi-Model (4 × 1.5B):
  Quality: 92% (specialist, di domainnya)
  Resources: 0.75 GB VRAM (aktif)
  η = 92/0.75 = 122.7

Ratio: 122.7/2.43 = 50.5× lebih efisien
```

### 6.3 Latency Budget

```
Single 70B (disk streaming):
  t_load: 35 GB / 7 GB/s = 5000 ms
  t_infer: 1000 ms (5 tok/s × 200 tokens)
  t_total: 6000 ms

Multi-Model (VRAM):
  t_load: 0 ms (sudah di VRAM)
  t_infer: 100 ms (50 tok/s × 200 tokens)
  t_total: 100 ms

Speedup: 6000/100 = 60×
```

---

## 7. Perbandingan dengan Pendekatan Lain

### 7.1 MoE (Mixture of Experts)

```
MoE:
  - 1 model besar dengan banyak expert internal
  - Router DALAM model
  - Expert sharing weights
  - Contoh: MOKO3.5-35B-A3B

Multi-Model MOKO:
  - Banyak model TERPISAH
  - Router LUAR model (system-level)
  - Expert TIDAK sharing weights
  - Contoh: 4 × 1-2B models

Perbedaan:
  MoE:   semua model load sekaligus (sparse activation)
  MOKO:  hanya 1 model load (model switching)

  MoE:   routing per TOKEN
  MOKO:  routing per QUERY

  MoE:   butuh VRAM untuk semua expert
  MOKO:  butuh VRAM untuk 1 model saja
```

### 7.2 Ensemble

```
Ensemble:
  - Semua model run sekaligus
  - Gabungkan output (voting/averaging)
  - Redundan tapi akurat

Multi-Model MOKO:
  - Hanya 1 model run
  - Pilih model BERDASARKAN intent
  - Efisien tapi bergantung pada router

  → MOKO: lebih efisien, tapi butuh router akurat
```

### 7.3 Cascade

```
Cascade:
  - Model kecil dulu → jika tidak yakin, model besar
  - Fallback strategy

Multi-Model MOKO:
  - Router pilih model langsung
  - Tidak ada cascading
  - Lebih cepat, tapi router harus benar

  → MOKO: lebih cepat, tapi zero tolerance untuk routing error
```

---

## 8. Spesifikasi Model

### 8.1 Coding Expert

```
Name: moko-coder-1.5B
Base: MOKO2.5-1.5B (atau fine-tuned dari MOKO3.5-4B)
Size: 1.5B params
Quantization: Q4_K_M (0.75 GB)
Training data:
  - GitHub code (Python, JavaScript, Rust, Go)
  - Stack Overflow answers
  - Security vulnerability databases
  - Algorithm textbooks
Specialization:
  - Code generation
  - Debugging
  - Code review
  - Security analysis
  - System design
```

### 8.2 Math Expert

```
Name: moko-math-1.0B
Base: MOKO2.5-1.5B (fine-tuned)
Size: 1.0B params
Quantization: Q4_K_M (0.5 GB)
Training data:
  - Mathematical proofs
  - Textbook problems
  - Competition math (AMC, AIME)
  - Engineering formulas
  - Statistics
Specialization:
  - Mathematical reasoning
  - Formula derivation
  - Problem solving
  - Numerical computation
  - Symbolic math (SymPy)
```

### 8.3 Security Expert

```
Name: moko-security-1.0B
Base: MOKO2.5-1.5B (fine-tuned)
Size: 1.0B params
Quantization: Q4_K_M (0.5 GB)
Training data:
  - CVE database
  - Penetration testing guides
  - Malware analysis
  - Reverse engineering
  - Cryptography
Specialization:
  - Vulnerability analysis
  - Exploit writing
  - Security audit
  - Threat modeling
  - Incident response
```

### 8.4 General Expert

```
Name: moko-general-2.0B
Base: MOKO3.5-4B (quantized aggressively)
Size: 2.0B params
Quantization: Q4_K_M (1.0 GB)
Training data:
  - General conversation
  - Knowledge base (174K entries)
  - Web content
  - Documentation
Specialization:
  - General Q&A
  - Conversation
  - Knowledge retrieval
  - Explanation
  - Creative writing
```

---

## 9. Roadmap Implementasi

```
Fase 1: Router Development (1-2 minggu)
  - Rule-based intent detection
  - Semantic fallback
  - Multi-intent handling

Fase 2: Model Collection (2-3 minggu)
  - Download base models (MOKO2.5-1.5B, MOKO3.5-4B)
  - Fine-tune untuk setiap domain
  - Quantize ke Q4_K_M

Fase 3: System Integration (2-3 minggu)
  - Model dispatcher
  - VRAM manager
  - Knowledge base per domain

Fase 4: AI Agent (1-2 minggu)
  - File organization
  - Knowledge ingestion
  - Session management

Fase 5: Testing & Optimization (1-2 minggu)
  - Accuracy per domain
  - Latency measurement
  - VRAM utilization
```

---

## 10. Key Insight

```
Manusia tidak mencoba jadi expert di SEMUA bidang.
Manusia SPECIALIZE dan bekerja sama.

AI juga seharusnya begitu.

4GB VRAM = 1 otak kecil
Bukan: 1 otak besar yang bisa semua
Tapi: 4 otak kecil yang masing-masing expert

Router = corpus callosum
RAG = memori jangka panjang
Agent = sistem saraf otonom

→ Paradigma baru: AI yang seperti manusia
→ Bukan AI yang seperti komputer
```
