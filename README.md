# MOKO AI — Sistem Operasi Kecerdasan Buatan Otonom

<p align="center">
  <img src="https://img.shields.io/badge/status-aktif-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Platform">
</p>

> **MOKO** bukan sekadar chatbot. MOKO adalah **sistem operasi kognitif** — sebuah ekosistem AI otonom yang memiliki kesadaran buatan, sistem memori jangka panjang, penalaran matematis mendalam, kemampuan coding agen otonom, mesin pencari darkweb, dan antarmuka IDE native.  
>
> Proyek ini lahir dari visi bahwa AI lokal harus bisa menjadi **mitra berpikir yang utuh** — bukan sekadar API call ke cloud.

---

## Daftar Isi

- [Ikhtisar](#ikhtisar)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Komponen Utama](#komponen-utama)
  - [MOKO Core Engine](#1-moko-core-engine)
  - [Dual-System Architecture](#2-dual-system-architecture-system-1--system-2)
  - [NeuroMath Engine](#3-neuromath-engine)
  - [Marathon Engine](#4-marathon-engine)
  - [RAG & Sistem Memori](#5-rag--sistem-memori)
  - [MOKO IDE](#6-moko-ide)
  - [Akselerasi Native (C++/Rust)](#7-akselerasi-native-crust)
  - [Sistem Keamanan](#8-sistem-keamanan)
  - [Pipeline Fine-Tuning](#9-pipeline-fine-tuning)
- [Cara Menjalankan](#cara-menjalankan)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Struktur Proyek](#struktur-proyek)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

---

## Ikhtisar

MOKO AI adalah **sistem kecerdasan buatan otonom berbasis agen** yang dirancang untuk berjalan sepenuhnya secara lokal. Sistem ini bukan sekadar LLM wrapper — ia memiliki:

| Kemampuan | Deskripsi |
|-----------|-----------|
| **Kesadaran Kognitif** | Sistem Ganda (System 1 cepat + System 2 lambat) terinspirasi psikologi kognitif Kahneman |
| **Penalaran Matematis** | Mesin matematika simbolik, kalkulus, aljabar linear, fisika, dan pembuktian teorema |
| **Coding Otonom** | Agen coding yang bisa menulis, mengevaluasi, memperbaiki, dan menguji kode secara mandiri |
| **Memori Jangka Panjang** | Vector store, RSA storage, WAL logging, dan sistem memori distribusi omni-domain |
| **Pencarian Darkweb** | Crawler Tor terintegrasi dengan onion search |
| **Marathon Reasoning** | Penalaran rantai panjang dengan kompresi konteks otomatis |
| **Fine-Tuning Pipeline** | Pipeline lengkap untuk fine-tune model coding dari Qwen2.5-1.5B |
| **IDE Native** | C++ Qt6 IDE dengan LSP client, AI assistant, dan syntax highlighting |
| **Puzzle System** | Sistem teka-teki lintas domain (matematika, fisika, logika, kode, KBBI, OS control) |

---

## Arsitektur Sistem

```
                         ┌──────────────────────────────────────┐
                         │         MOKO OS Launcher              │
                         │  ./moko.sh (auto) / ./moko_cli.sh     │
                         └──────────┬───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              ┌─────▼──────┐                 ┌──────▼─────┐
              │  CLI Mode  │                 │  GUI Mode  │
              │ (Terminal) │                 │ (Qt6 IDE)  │
              └─────┬──────┘                 └──────┬─────┘
                    │                               │
              ┌─────▼───────────────────────────────▼─────┐
              │              MOKO Core Engine               │
              │                                            │
              │  ┌─────────────┐    ┌──────────────────┐   │
              │  │ Intent Router│───▶│ Cognitive Router │   │
              │  └─────────────┘    └────────┬─────────┘   │
              │                               │            │
              │       ┌───────────────────────┼──────────┐ │
              │       ▼                       ▼          ▼ │
              │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │
              │  │  Agen   │  │ Dual Sys │  │ Marathon │  │
              │  │ Coding  │  │ Orkestra │  │  Engine  │  │
              │  └─────────┘  └──────────┘  └──────────┘  │
              │       │            │              │        │
              │       ▼            ▼              ▼        │
              │  ┌─────────────────────────────────────┐   │
              │  │        NeuroMath Engine              │   │
              │  │   (Matematika · Fisika · Logika)     │   │
              │  └─────────────────────────────────────┘   │
              │       │            │              │        │
              │       ▼            ▼              ▼        │
              │  ┌─────────────────────────────────────┐   │
              │  │     Sistem Memori & RAG              │   │
              │  │  (Vector DB · RSA · Omni Storage)    │   │
              │  └─────────────────────────────────────┘   │
              │       │                                    │
              │       ▼                                    │
              │  ┌─────────────────────────────────────┐   │
              │  │  Akselerasi Native (C++ / Rust)      │   │
              │  └─────────────────────────────────────┘   │
              └────────────────────────────────────────────┘
```

---

## Komponen Utama

### 1. MOKO Core Engine

Inti dari sistem. Berjalan di `moko_core/` dan menyediakan seluruh layanan AI.

#### Cognitive Router (`moko_core/moko_agents/router.py`)
Router utama yang mengklasifikasikan setiap input pengguna menggunakan 3-tier routing:
- **Tier 0-A (<1ms)**: Slash commands dan exact pattern matching
- **Tier 0-B (~15ms)**: Semantic Vector Router — cosine similarity terhadap centroid domain
- **Tier 0-C**: Rule-based keyword fallback

#### Intent Router (`moko_core/moko_agents/intent_router.py`)
Mengklasifikasikan kueri ke dalam intent:
- `CODING` — pertanyaan pemrograman
- `MATH` — matematika dan fisika
- `DARKWEB` — pencarian darkweb/tor
- `GENERAL` — pengetahuan umum
- `PERSONAL` — percakapan personal
- `SECURITY` — keamanan siber
- `REASONING` — penalaran logis
- `OS_CONTROL` — kontrol sistem operasi

#### Multi-Agent System (`moko_core/moko_agents/`)
Sistem agen kognitif yang terinspirasi dari arsitektur otak manusia:

| Agen | Fungsi |
|------|--------|
| **Prefrontal Node** | Perencanaan dan pengambilan keputusan eksekutif |
| **Amygdala Node** | Deteksi urgensi dan respons emosional |
| **Insula Node** | Kesadaran diri dan metakognisi |
| **Cerebellum Node** | Koordinasi gerakan kognitif (eksekusi cepat) |
| **Basal Ganglia** | Pemilihan aksi dan kebiasaan |
| **DMN Node** | Default Mode Network — melamun dan introspeksi |
| **ACC Node** | Anterior Cingulate Cortex — deteksi konflik |
| **HPA Axis** | Respons stres dan adaptasi |
| **Locus Coeruleus** | Modulasi perhatian dan kewaspadaan |

---

### 2. Dual-System Architecture (System 1 & System 2)

Terinspirasi dari kerangka *Thinking, Fast and Slow* karya Daniel Kahneman.

**System 1 (Cepat / Intuitif)**:
- Eksekusi langsung oleh `ExecutorNode`
- Untuk tugas rutin, pertanyaan sederhana, pencarian fakta
- Latensi rendah, tanpa penalaran mendalam

**System 2 (Lambat / Analitis)**:
- Melibatkan `BrainNode` untuk perencanaan
- `DualRuntimeGuard` untuk verifikasi
- Iteratif: Plan → Execute → Review → Re-plan jika gagal
- Untuk tugas kompleks: coding, debugging, matematika, keamanan

Alur Dual-System:
```
                  ┌──────────┐
                  │  Query   │
                  └────┬─────┘
                       │
              ┌────────▼────────┐
              │  Intent Router  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Butuh System 2?│──Tidak──▶ System 1 (Executor)
              └────────┬────────┘
                       │ Ya
              ┌────────▼────────┐
              │  Brain (Plan)   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Executor (Act)  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Guard (Review) │──Lolos──▶ Commit
              └────────┬────────┘
                       │ Gagal
                       ▼
                  Re-plan (Brain)
```

File kunci: `moko_core/moko_agents/dual_system/orchestrator.py`

---

### 3. NeuroMath Engine

**Ini adalah jantung matematis MOKO.** Bukan sekadar kalkulator — ini adalah sistem penalaran matematis yang terinspirasi dari neurosains kognitif.

`moko_core/moko_neuromath/` berisi **60+ modul** yang mencakup:

| Modul | Fungsi |
|-------|--------|
| **PureMathEngine** | Matematika murni: aljabar, kalkulus, trigonometri, logaritma |
| **ExactMathEngine** | Perhitungan presisi tinggi dengan verifikasi |
| **ComputerMathEngine** | Matematika komputer: floating-point, numerik, error analysis |
| **CSMathEngine** | Struktur data, algoritma, kompleksitas |
| **AppliedFormulaEngine** | Formula terapan dari berbagai bidang |
| **FormalReasoningEngine** | Penalaran formal dan logika matematika |
| **StepLogicProver** | Pembuktian langkah-demi-langkah |
| **SymbolicRegression** | Regresi simbolik untuk menemukan formula dari data |
| **SymbolicSynthesizer** | Sintesis formula baru dari pola |
| **ProgramVerificationEngine** | Verifikasi program dengan logika matematis |
| **ActiveInference** | Mutasi formula berdasarkan Free Energy Principle (FEP) |
| **FEPEngine** | Free Energy Principle — minimisasi surprisal |
| **UncertaintyEngine** | Kuantifikasi ketidakpastian |
| **MCTSReasoner** | Monte Carlo Tree Search untuk penalaran |
| **CognitiveMap** | Peta kognitif untuk navigasi abstrak |
| **DimensionalSynthesis** | Sintesis lintas dimensi dan domain |
| **QuantumSimulator** | Simulasi quantum computing dasar |
| **TuringConsciousnessEngine** | Eksperimen kesadaran buatan berbasis Turing |
| **FormulaCritic** | Kritik dan evaluasi formula |
| **FormulaGenesisEngine** | Penemuan formula baru secara otonom |
| **AppliedMathTrainer** | Pelatihan matematika terapan |
| **SleepConsolidation** | Konsolidasi memori matematis (seperti tidur pada otak) |

---

### 4. Marathon Engine

Sistem penalaran rantai panjang (`moko_core/moko_marathon/`) yang memungkinkan MOKO memecahkan masalah kompleks melalui langkah-langkah bertahap.

```
                 ┌──────────────────┐
                 │    Pertanyaan    │
                 └────────┬─────────┘
                          │
              ┌───────────▼───────────┐
              │  Context Pager        │
              │  (memori konteks)     │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Langkah 1: Analisis  │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Langkah 2: Penalaran │
              └───────────┬───────────┘
                          │
                     ──▶ ... ◀──
                          │
              ┌───────────▼───────────┐
              │  Kompresi Semantik    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Jawaban Akhir      │
              └───────────────────────┘
```

Komponen:
- **MarathonRunner** — koordinator siklus penalaran
- **ContextPager** — manajemen memori konteks (paging)
- **SemanticCompressor** — kompresi konteks otomatis agar tidak melebihi limit LLM
- **PuzzleAssembler** — perakit teka-teki dari berbagai domain
- **CodeAssembler** — perakit kode dari fragmen
- **CodeVerifier** — verifikasi kode yang dihasilkan
- **CrossVerifier** — verifikasi silang antar domain
- **MarathonCodeSentinel** — pengawas kualitas kode maraton
- **MarathonPitstop** — titik istirahat untuk evaluasi tengah jalan
- **SecurityAuditor** — audit keamanan kode yang dihasilkan
- **GitManager** — integrasi git untuk tracking perubahan

---

### 5. RAG & Sistem Memori

MOKO memiliki sistem memori berlapis yang memungkinkannya mengingat dan mengambil pengetahuan dari berbagai sumber.

`moko_core/moko_memory/`:

| Komponen | Fungsi |
|----------|--------|
| **OmniVectorStore** | Vector database multi-domain (code, math, security, general, dll.) |
| **RSAStorage** | Enkripsi dan penyimpanan berbasis RSA |
| **WALManager** | Write-Ahead Logging untuk integritas data |
| **KVCacheManager** | Manajemen KV-cache untuk inferensi LLM |
| **NeuralWorkingMemory** | Memori kerja neural (short-term) |
| **MultiDomainStorage** | Penyimpanan terdistribusi per domain pengetahuan |
| **SessionStore** | Penyimpanan sesi percakapan |
| **DiskManager** | Manajemen penyimpanan disk |
| **GCTuner** | Garbage collection tuner untuk memori |
| **SearchCache** | Cache pencarian untuk mempercepat retrieval |
| **MokoRAGRetriever** | Retrieval RAG utama |
| **MokoEmbedEngine** | Mesin embedding lokal |
| **BinaryKnowledgeCodec** | Codec pengetahuan biner untuk kompresi |
| **OmniHashEncoder** | Encoding hash multi-domain |
| **OmniStorage** | Penyimpanan omni-domain terdistribusi |
| **MathOmni** | Omni storage khusus matematika |
| **ConvBuffer** | Buffer percakapan untuk context window |
| **Ramdisk** | Ramdisk untuk akses cepat |
| **HDCContext** | Hyperdimensional Computing context |

Sistem RAG terdistribusi di port:
- Port `11437` — server RAG utama
- Auto-boot saat MOKO IDE menyala
- Fallback aman jika server mati

Pipeline RAG:
```
Query → Intent Classification → Domain Selection → Vector Retrieval
  → Context Assembly → LLM Generation → Response
```

---

### 6. MOKO IDE

MOKO memiliki dua antarmuka IDE:

#### A. MOKO IDE C++ (Native)
- Dibangun dengan **Qt6** (C++)
- 21 file source di `moko_ide_cpp/`
- Fitur: code editor, syntax highlighting, terminal, chat panel, stress test

#### B. MOKO IDE Python
- Antarmuka terminal (`moko_core/moko_os.py`)
- Dual-mode: CLI dan GUI (PyQt6)
- Integrasi LSP Client (`moko_core/moko_lsp/lsp_client.py`)
  - Dukungan bahasa: Python, JavaScript, TypeScript, JSON, CSS, HTML
  - Fitur: diagnostics, autocomplete, go-to-definition
  - Fallback aman jika language server tidak tersedia
- Code Structure Analysis (`moko_core/moko_utils/code_structure.py`)
  - Pengecekan pasangan tag HTML/XML secara real-time
  - Deteksi bracket imbalance `()[]{}`
  - Wave underline + status bar indicator

---

### 7. Akselerasi Native (C++/Rust)

Untuk performa maksimal, MOKO memiliki beberapa komponen native:

#### C++ Native (`moko_core/moko_native/`)
- Compile dengan `build.sh`
- Benchmark: `bench.py`
- Integrasi Python via cpp_loader

#### C++ Core Kernel (`moko_core/moko_cpp_kernel/`)
- `moko_kernel.cpp` — kernel utama
- `simd_math.hpp` — operasi matematika SIMD
- `thread_pool.hpp` — thread pool untuk paralelisasi
- `mmap_io.hpp` — memory-mapped I/O
- `libmoko_core.so` — shared library terkompilasi

#### C++ Core (`moko_core/moko_cpp_core/`)
- `moko_cpp_core.cpp` — core C++ untuk operasi berat
- `libmoko_cpp.so` — shared library

#### Rust Core (`moko_core/moko_rust_core/`)
- Module Rust dengan Cargo
- Operasi yang membutuhkan memory safety tinggi

---

### 8. Sistem Keamanan

`moko_core/moko_security/`:

| Modul | Fungsi |
|-------|--------|
| **RedTeamFuzzer** | Fuzzer keamanan otomatis — menguji celah keamanan |
| **BlueTeamDefender** | Defender otomatis — melindungi dari serangan |
| **GaussianNoiseEngine** | Menambahkan noise Gaussian untuk privasi diferensial |

---

### 9. Pipeline Fine-Tuning

`finetune/` berisi pipeline lengkap untuk fine-tuning model coding:

**Base Model**: Qwen2.5-1.5B-Instruct (HF format, ~3GB)
**Output**: LoRA adapter → GGUF untuk llama.cpp

```
python3 moko_finetune.py --prepare    # Download base model dari HuggingFace
python3 moko_finetune.py --build      # Bangun dataset coding
python3 moko_finetune.py --train      # Jalankan fine-tuning
python3 moko_finetune.py --status     # Cek status training
```

**Dataset** (di `finetune/moko_datasets/`):
- `moko_coder_dataset.jsonl` — 12.000+ sampah coding
- `moko_algo_dataset.jsonl` — dataset algoritma
- `moko_security_dataset.jsonl` — dataset keamanan siber
- `moko_reasoning_dataset.jsonl` — dataset penalaran
- `moko_multiturn_dataset.jsonl` — dataset multi-turn percakapan
- `moko_os_code_dataset.jsonl` — dataset kode sistem operasi
- `moko_programming_dataset.jsonl` — dataset pemrograman umum
- `moko_docs_dataset.jsonl` — dataset dokumentasi
- `moko_hex_encoder.jsonl` — dataset encoding heksadesimal
- `moko_distill_dataset.jsonl` — dataset knowledge distillation
- `moko_cpp_qt_dataset.jsonl` — dataset Qt/C++
- `moko_ide_integration.jsonl` — dataset integrasi IDE
- `moko_algo_dataset.jsonl` — dataset algoritma lanjutan

**Pipeline Lengkap**:
```
Download HF Model → Build Dataset → LoRA Training → Convert to GGUF → Load di llama.cpp
```

---

## Cara Menjalankan

### Prasyarat

```bash
# 1. Clone repositori
git clone https://github.com/brianatmoko/MOKO-AI.git
cd MOKO-AI

# 2. Buat virtual environment
python3 -m venv moko_core/venv --upgrade-deps
source moko_core/venv/bin/activate

# 3. Install dependensi
# (disesuaikan dengan kebutuhan, belum ada requirements.txt terpusat)
pip install numpy  # core dependency

# 4. (Opsional) Download model untuk fine-tuning
cd finetune
python3 moko_finetune.py --prepare
```

### Menjalankan MOKO OS

```bash
# Terminal CLI Mode (otomatis)
./moko.sh

# Paksa CLI Mode
./moko.sh --cli

# GUI Mode (butuh Qt6 dan display server)
./moko.sh --gui

# Daemon Mode (web API)
./moko.sh --daemon

# CLI-only launcher
./moko_cli.sh
```

### Mode CLI

```
./moko.sh --cli
```

Setelah masuk CLI:
- Ketik pertanyaan apa saja untuk memulai percakapan
- `/ai on` — aktifkan local LLM (butuh model dijalankan)
- `/ai off` — nonaktifkan local LLM
- `exit` — keluar

### Mode GUI

```
./moko.sh --gui
```

MOKO akan:
1. Menjalankan local LLM server di background
2. Meluncurkan Native C++ IDE (jika sudah dikompilasi)
3. Fallback ke Python GUI jika binary IDE tidak ditemukan

### Mode Daemon

```
./moko.sh --daemon
```

Web API akan berjalan di `http://127.0.0.1:8000`

---

## Persyaratan Sistem

| Komponen | Minimal | Rekomendasi |
|----------|---------|-------------|
| **CPU** | 4-core | 8-core + (untuk inferensi lokal) |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | - | RTX 2050 4GB+ (untuk fine-tuning) |
| **Storage** | 1 GB (source) | 10 GB+ (dengan model) |
| **OS** | Linux (kernel 5.x+) | Linux (kernel 6.x+) |
| **Python** | 3.12 | 3.12 |
| **Qt** | - | Qt6 (untuk IDE) |
| **C++ Compiler** | g++ 11+ | g++ 13+ (untuk native) |
| **Rust** | - | Rust 1.70+ (untuk rust core) |

### Catatan Hardware
- **Fine-tuning**: Diuji pada RTX 2050 4GB VRAM + 16GB RAM — berjalan lambat tapi stabil
- **Inferensi lokal**: Butuh model GGUF + llama.cpp. Setup otomatis oleh MOKO Inference Server
- **Tanpa GPU**: MOKO tetap berfungsi penuh dalam mode CLI dengan API LLM eksternal

---

## Struktur Proyek

```
MOKO-AI/
├── moko.sh                     # Launcher utama (auto-detect mode)
├── moko_cli.sh                 # Launcher CLI-only
├── moko_launcher.sh            # Launcher alternatif
├── moko-warmup.sh              # Script warmup sistem
├── moko-models.sh              # Manajemen model
├── moko-fix-limits.sh          # Fix system limits
├── build_ide.sh                # Build MOKO IDE C++
├── build_kernel.sh             # Build kernel C++
│
├── moko_core/                  # ★ INTI SISTEM
│   ├── moko_os.py              # Entry point CLI
│   ├── moko_config/            # Konfigurasi sistem
│   ├── moko_agents/            # Sistem multi-agen kognitif
│   │   ├── router.py           # Cognitive Router utama
│   │   ├── intent_router.py    # Intent classifier
│   │   ├── dual_system/        # System 1 + System 2
│   │   ├── coding/             # Agen coding (7 agen)
│   │   ├── software_builder/   # Software builder agent
│   │   └── ...                 # 30+ node agen kognitif
│   ├── moko_neuromath/         # ★ Mesin matematika (60+ modul)
│   ├── moko_marathon/          # Marathon reasoning engine
│   ├── moko_memory/            # Sistem memori & RAG
│   ├── moko_inference/         # LLM inference server
│   ├── moko_crawler/           # Web & Tor crawler
│   ├── moko_security/          # Sistem keamanan
│   ├── moko_super_learning/    # Super learning engine
│   ├── moko_native/            # Akselerasi C++ / Rust
│   ├── moko_cpp_kernel/        # C++ kernel
│   ├── moko_cpp_core/          # C++ core library
│   ├── moko_rust_core/         # Rust core module
│   ├── moko_lsp/               # LSP client untuk IDE
│   ├── moko_puzzles/           # Sistem teka-teki
│   ├── moko_cpu/               # CPU scheduler & governor
│   ├── moko_tools/             # Tools (quantisasi, encoding, dll.)
│   ├── moko_benchmark/         # Benchmark sistem
│   └── moko_utils/             # Utilitas (UI, text, code)
│
├── moko_ide_cpp/               # MOKO IDE C++ (Qt6)
│   ├── main.cpp                # Entry point
│   ├── moko_window.h/cpp       # Window utama
│   ├── code_editor.h/cpp       # Code editor
│   ├── chat_widget.h/cpp       # AI Chat panel
│   ├── terminal_widget.h/cpp   # Terminal terintegrasi
│   ├── syntax_highlighter.*    # Syntax highlighting
│   ├── helper_engine.*         # AI helper engine
│   ├── find_bar.*              # Find & replace
│   ├── settings_dialog.*       # Dialog pengaturan
│   ├── graphify_dialog.*       # Dialog graph visualisasi
│   └── stress_test_dialog.*    # Stress test dialog
│
├── finetune/                   # Pipeline fine-tuning
│   ├── moko_finetune.py        # Script utama
│   ├── moko_trainer_v2.py      # Trainer v2
│   ├── moko_trainer_v3_7b.py   # Trainer v3 (7B model)
│   ├── moko_data_factory.py    # Data factory
│   ├── moko_distill_engine.py  # Knowledge distillation
│   ├── moko_byteq_compressor.py# ByteQ compression
│   ├── moko_hex_encoder.py     # Hex encoding
│   ├── moko_datasets/          # 13 dataset coding
│   └── ...                     # 20+ script pendukung
│
├── moko_config/                # Konfigurasi
│   ├── api_keys.json           # API keys (OpenRouter, dll.)
│   └── moko_settings.json      # Pengaturan sistem
│
├── docs/                       # Dokumentasi
│   ├── RINGKASAN_PROGRES.md    # Ringkasan progres
│   ├── STRUKTUR_FOLDER_MOKO.md # Struktur folder detail
│   └── riset/                  # 19 dokumen riset teknis
│
└── share/                      # File sharing (man pages, dll.)
```

---

## Catatan Teknis Penting

### API Keys
Konfigurasi API keys ada di `moko_config/api_keys.json`. Secara default MOKO menggunakan:
- **Openroute/OpenRouter** — untuk akses LLM cloud gratis (Llama 3.1 8B, Gemma 3 12B, dll.)
- **Local LLM** — via llama.cpp server di port lokal
- **OpenCode** — API lokal untuk development

### Model Metadata
- `MOKO-AI-4B-CryptoCore-Q3_K_M_moko_meta.json` — metadata untuk model 4B CryptoCore (Q3_K_M)
- `MOKO-AI-4B-CryptoCore-BF16_moko_header.json` — header untuk model BF16
- `MOKO-RAG-1.5B-ByteQ_moko_meta.json` — metadata untuk model RAG 1.5B dengan ByteQ

### Konsep Kunci

**ByteQ Quantization**: Teknik kuantisasi ekstrem 2-bit Lloyd yang dikembangkan sendiri. Mampu mengompresi tensor hingga ~6.8x dengan MSE sangat kecil (~1e-4). Implementasi di `moko_tools/byteq_quantizer.py`.

**Omni Storage**: Sistem penyimpanan vektor multi-domain yang mengorganisir pengetahuan ke dalam 8 domain: code, math, security, general, finance, personal, programming, reasoning. Setiap domain memiliki centroid embedding untuk retrieval yang presisi.

**Cognitive Template**: Template kognitif untuk berbagai jenis penalaran. Sistem belajar dari pengalaman dan menyempurnakan template-nya seiring waktu.

**Marathon Pitstop**: Mekanisme "istirahat" di tengah penalaran panjang untuk mengevaluasi progres, melakukan kompresi konteks, dan memutuskan apakah perlu melanjutkan atau cukup.

---

## Visi & Masa Depan

MOKO lahir dari keyakinan bahwa **kecerdasan buatan yang otonom, lokal, dan dapat dipercaya** bukanlah mimpi. Bahwa kita tidak harus bergantung pada API cloud raksasa untuk memiliki asisten berpikir yang cerdas.

### Yang sudah tercapai:
- ✅ Sistem multi-agen kognitif dengan arsitektur neural terinspirasi otak
- ✅ Mesin matematika dengan 60+ modul penalaran
- ✅ Dual-System (cepat + lambat) dengan loop iteratif
- ✅ RAG multi-domain dengan vector store lokal
- ✅ Pipeline fine-tuning dari Qwen ke MOKO Coder
- ✅ IDE C++ native dengan LSP dan AI integration
- ✅ Marathon reasoning untuk masalah kompleks
- ✅ Darkweb crawling via Tor
- ✅ Puzzle system lintas domain

### Yang perlu dilanjutkan:
- ⏳ Integrasi penuh antara semua komponen
- ⏳ Optimalisasi performa dan memori
- ⏳ Dokumentasi API dan developer guide
- ⏳ Unit testing yang komprehensif
- ⏳ Packaging dan distribusi yang mudah
- ⏳ Model MOKO Coder yang sudah di-fine-tune
- ⏳ GUI yang lebih matang

---

## Kontribusi

Proyek ini adalah **open-source** dan sangat membutuhkan kontributor. Jika Anda tertarik dengan AI otonom, sistem multi-agen, matematika komputasional, atau pengembangan IDE:

1. **Fork** repositori ini
2. **Clone** fork Anda
3. **Baca** dokumentasi di `docs/` untuk memahami arsitektur
4. **Mulai** dari issue atau area yang Anda kuasai
5. **Kirim** Pull Request

Area yang paling membutuhkan bantuan:
- Dokumentasi dan contoh penggunaan
- Testing dan QA
- Optimisasi performa
- Packaging (pip installable)
- Frontend GUI yang lebih baik

> *"MOKO adalah proyek ambisius yang dibangun dengan sumber daya terbatas. Tapi saya percaya fondasinya kuat. Saya menitipkannya pada siapa pun yang percaya bahwa AI lokal yang cerdas dan otonom adalah masa depan yang layak diperjuangkan."*
>
> — Brian Atmoko, Pembuat MOKO

---

## Lisensi

Proyek ini dilisensikan di bawah **MIT License** — silakan digunakan, dimodifikasi, dan disebarluaskan.

---

<p align="center">
  <b>MOKO AI</b><br>
  <i>Bukan sekadar chatbot. Sistem operasi kognitif.</i><br>
  <a href="https://github.com/brianatmoko/MOKO-AI">github.com/brianatmoko/MOKO-AI</a>
</p>
