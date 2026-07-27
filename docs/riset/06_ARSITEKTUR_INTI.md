# 06 — Arsitektur Inti: Desain Sistem MOKO

> **⚠️ PEMBERITAHUAN TRANSISI: NO-CRYPTO ARCHITECTURE**
> Mulai Juli 2026, MOKO OS telah sepenuhnya beralih dari arsitektur berbasis kriptografi (MokoCryptoCore, Blockchain, Merkle) menuju arsitektur **Omni-Knowledge** yang murni. Semua komponen kripto telah dihapus untuk meningkatkan efisiensi dan kecepatan inferensi.

> **Tujuan:** Arsitektur lengkap MOKO OS — apa yang sudah dibangun, apa yang akan dibangun,
> dan bagaimana semua komponen saling terhubung.

---

## 1. Arsitektur Saat Ini

```
┌─────────────────────────────────────────────────────────┐
│                    MOKO OS Stack                         │
├─────────────────────────────────────────────────────────┤
│  Layer 5: UI (PyQt5/Cyberpunk theme)                    │
│  Layer 4: Cognitive Engine (Router + Analyst + Core)     │
│  Layer 3: Memory (Omni Index + RAG)                     │
│  Layer 2: Inference (LLaMA.cpp + MOKO-AI-4B)             │
│  Layer 1: Hardware (RTX 2050 4GB + NVMe)                │
└─────────────────────────────────────────────────────────┘
```

### 1.1 Layer 5: UI

```
Components:
  - main_window.py: Cyberpunk-themed GUI
  - omni_worker.py: Background knowledge ingestion
  - hw_worker.py: Hardware monitoring

Status: ✅ Working
```

### 1.2 Layer 4: Cognitive Engine

```
Components:
  - router.py: 3-tier routing (rule → semantic → keyword)
  - core_node.py: Main inference pipeline
  - analyst_node.py: Deep analysis (DUAL mode)
  - cognitive_executive.py: Orchestrator

Current routing:
  Tier 1: Rule-based (regex patterns)
  Tier 2: Semantic centroid (cosine similarity)
  Tier 3: Keyword fallback

Target routing (Doc 06):
  Intent-first: COMMAND → HOWTO → LEXICAL → MATH → CODE → PERSONAL → CHITCHAT → FACTUAL

Status: 🟡 Working, accuracy needs improvement
```

### 1.3 Layer 3: Memory

```
Components:
  - rsa_storage.py: Vector storage engine
  - disk_manager.py: Disk I/O manager
  - omni_hash_encoder.py: Hash-based encoding
  - omni_vector_store.py: Vector store

Storage:
  .moko_omni/
    ├── code/          1,687 entries
    ├── cyberoffensive/ 929 entries
    ├── cybersecurity/  978 entries
    ├── general/       1,222 entries
    ├── general_sub_1/  674 entries
    ├── lexical/      10,830 entries
    ├── math/          1,051 entries
    ├── personal/       759 entries
    ├── physics/        660 entries
    └── test_domain/      3 entries
  Total: 174,418 entries

Status: ✅ Working (174K entries searchable)
```

### 1.4 Layer 2: Inference

```
Components:
  - LLaMA.cpp server (port 11435): Main inference
  - Embedding server (port 11436): nomic-embed-text
  - Ollama (port 11434): Fallback

Models:
  - MOKO-AI-4B Q3_K_M (2.1 GB): Inference
  - MOKO-AI-4B BF16 (8.0 GB): Identity reference
  - nomic-embed-text: Embedding

Status: ✅ Working
```

### 1.5 Layer 1: Hardware

```
GPU: RTX 2050 Mobile
  VRAM: 4 GB GDDR6
  Bandwidth: 112 GB/s
  FLOPS: 4.5 TFLOPS (peak)

Storage: NVMe SSD
  Sequential: 7 GB/s
  Random: 100 MB/s

RAM: varies
  Disk: fallback storage
  RAM disk: /mnt/moko_ram (optional)
  /dev/shm: /dev/shm/moko_omni (fallback)
```

---

## 2. Arsitektur Target (Post-Rollback)

### 2.1 Perubahan

```
Crypto layer: DIHAPUS
  - MokoCryptoCore: ❌ removed
  - Blockchain ledger: ❌ removed
  - HMAC ceremony: ❌ removed
  - Merkle tree: ❌ removed

Router: DITINGKATKAN
  - Intent-first: 8 priority classes
  - Scoped search: 1-3 domains
  - Domain governance rules

Storage: DISATUKAN
  - .moko_omni/ (single store)
  - Hapus dual store
```

### 2.2 Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: UI (unchanged)                                │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Intent-First Router → Scoped Search           │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Omni Index (single store, .moko_omni/)        │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Inference (unchanged)                         │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Hardware (unchanged)                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Query Flow (Current)

```
User Input
  ↓
Router (3-tier)
  ├── Tier 1: Rule match → direct response
  ├── Tier 2: Semantic centroid → domain selection
  └── Tier 3: Keyword fallback
  ↓
Domain Selected
  ↓
RSAStorage.search()
  ↓
Results ranked by similarity
  ↓
Core Node generates response
  ↓
Response to User
```

### 3.2 Query Flow (Target)

```
User Input
  ↓
Intent Detection (8 classes)
  ├── COMMAND → execute directly
  ├── HOWTO → step-by-step from knowledge
  ├── LEXICAL → exact keyword search
  ├── MATH → compute via NeuroMath
  ├── CODE → code generation/retrieval
  ├── PERSONAL → user history
  ├── CHITCHAT → casual response
  └── FACTUAL → scoped knowledge search
  ↓
Domain Governance Rules
  ↓
Scoped Search (1-3 domains)
  ↓
RSAStorage.search()
  ↓
Results ranked
  ↓
Core Node generates response
  ↓
Response to User
```

---

## 4. NeuroMath Subsystem

```
Components:
  - mcts_reasoner.py: Monte Carlo Tree Search
  - applied_formula_engine.py: 100+ engineering formulas
  - formula_genesis_engine.py: Formula discovery
  - step_logic_prover.py: Z3-powered verifier
  - crypto_dispatch.py: Intent parsing

Status: ✅ Implemented (77 modules)
Usage: Mathematical reasoning, formula verification
```

---

## 5. Token Stream System (T0)

```
Components:
  - TokenStreamPager: Context window management
  - conv_buffer: Conversation buffer
  - semantic_compressor_v2: Message compression

Settings:
  TOKEN_STREAM_ENABLED = True
  TOKEN_STREAM_RESERVE_TOKENS = 512
  TOKEN_STREAM_RECENT_TURNS = 2
  SEMANTIC_COMPRESSOR_V2 = True
  SEMANTIC_COMPRESS_MAX_TOKENS = 96

Status: ✅ Implemented
```

---

## 6. Rollback Status

```
R2 Fase 1 ✅: CRYPTO_ENABLED=False + guards
R2 Fase 2 ✅: Path .moko_omni + rename
R2 Fase 3:    Cleanup dual store (pending)
R2 Fase 4:    Hapus modul crypto (pending)
R2 Fase 5:    Revert model (pending)
R2 Fase 6:    Validasi final (pending)
```

---

## 7. File Map

```
moko_core/
├── moko_config/
│   └── settings.py          # Global settings
├── moko_agents/
│   ├── router.py            # Routing engine
│   ├── core_node.py         # Main inference
│   └── analyst_node.py      # Deep analysis
├── moko_memory/
│   ├── rsa_storage.py       # Vector storage
│   └── disk_manager.py      # Disk I/O
├── moko_neuromath/
│   ├── mcts_reasoner.py     # MCTS
│   └── ... (77 modules)
├── moko_marathon/
│   └── context_pager.py     # Token stream
├── moko_ui/
│   ├── main_window.py       # GUI
│   └── workers/
│       ├── omni_worker.py   # Knowledge ingest
│       └── cognitive_worker.py
└── moko_inference/
    └── server_manager.py    # LLaMA.cpp management
```
