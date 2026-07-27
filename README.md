# MOKO AI — Autonomous Cognitive Operating System

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/platform-linux-lightgrey" alt="Platform">
  <a href="https://github.com/brianatmoko/MOKO-AI/actions"><img src="https://github.com/brianatmoko/MOKO-AI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="README.id.md"><img src="https://img.shields.io/badge/Bahasa-Indonesia-red" alt="ID"></a>
</p>

> **MOKO** is not just a chatbot. It is a **cognitive operating system** — an autonomous AI ecosystem with artificial consciousness, long-term memory, deep mathematical reasoning, autonomous coding agents, darkweb search, and a native IDE interface.
>
> This project was born from the belief that local AI should be a **complete thinking partner** — not just a cloud API call.

---

## Overview

MOKO is an **agent-based autonomous AI system** designed to run fully locally. It is not just an LLM wrapper — it provides:

| Capability | Description |
|-----------|-------------|
| **Cognitive Awareness** | Dual-System architecture (Fast System 1 + Slow System 2) inspired by Kahneman's cognitive psychology |
| **Mathematical Reasoning** | Symbolic math engine: calculus, linear algebra, physics, theorem proving |
| **Autonomous Coding** | Coding agents that write, evaluate, fix, and test code independently |
| **Long-Term Memory** | Vector store, RSA storage, WAL logging, omni-domain distributed memory |
| **Darkweb Search** | Integrated Tor crawler with onion search |
| **Marathon Reasoning** | Long-chain reasoning with automatic context compression |
| **Fine-Tuning Pipeline** | Complete pipeline to fine-tune coding models from Qwen2.5-1.5B |
| **Native IDE** | C++ Qt6 IDE with LSP client, AI assistant, and syntax highlighting |
| **Puzzle System** | Cross-domain puzzles (math, physics, logic, code, OS control) |

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Core Components](#core-components)
  - [MOKO Core Engine](#1-moko-core-engine)
  - [Dual-System Architecture](#2-dual-system-architecture)
  - [NeuroMath Engine](#3-neuromath-engine)
  - [Marathon Engine](#4-marathon-engine)
  - [RAG & Memory System](#5-rag--memory-system)
  - [MOKO IDE](#6-moko-ide)
  - [Native Acceleration (C++/Rust)](#7-native-acceleration-crust)
  - [Security System](#8-security-system)
  - [Fine-Tuning Pipeline](#9-fine-tuning-pipeline)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## System Architecture

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
              │  │  Coding │  │ Dual Sys │  │ Marathon │  │
              │  │ Agents  │  │Orchestra │  │  Engine  │  │
              │  └─────────┘  └──────────┘  └──────────┘  │
              │       │            │              │        │
              │       ▼            ▼              ▼        │
              │  ┌─────────────────────────────────────┐   │
              │  │        NeuroMath Engine              │   │
              │  │  (Math · Physics · Logic · CS)       │   │
              │  └─────────────────────────────────────┘   │
              │       │            │              │        │
              │       ▼            ▼              ▼        │
              │  ┌─────────────────────────────────────┐   │
              │  │     Memory System & RAG              │   │
              │  │  (Vector DB · RSA · Omni Storage)    │   │
              │  └─────────────────────────────────────┘   │
              │       │                                    │
              │       ▼                                    │
              │  ┌─────────────────────────────────────┐   │
              │  │  Native Acceleration (C++ / Rust)    │   │
              │  └─────────────────────────────────────┘   │
              └────────────────────────────────────────────┘
```

---

## Core Components

### 1. MOKO Core Engine

The heart of the system, located in `moko_core/`, provides all AI services.

#### Cognitive Router (`moko_core/moko_agents/router.py`)
Routes every user input using a 3-tier architecture:
- **Tier 0-A (<1ms)**: Slash commands and exact pattern matching
- **Tier 0-B (~15ms)**: Semantic Vector Router — cosine similarity against domain centroids
- **Tier 0-C**: Rule-based keyword fallback

#### Intent Router (`moko_core/moko_agents/intent_router.py`)
Classifies queries into intents:
- `CODING` — programming questions
- `MATH` — mathematics and physics
- `DARKWEB` — darkweb/tor search
- `GENERAL` — general knowledge
- `PERSONAL` — personal conversation
- `SECURITY` — cybersecurity
- `REASONING` — logical reasoning
- `OS_CONTROL` — operating system control

#### Multi-Agent System (`moko_core/moko_agents/`)
A cognitive agent system inspired by the human brain architecture:

| Agent | Function |
|------|----------|
| **Prefrontal Node** | Planning and executive decision-making |
| **Amygdala Node** | Urgency detection and emotional response |
| **Insula Node** | Self-awareness and metacognition |
| **Cerebellum Node** | Cognitive movement coordination (fast execution) |
| **Basal Ganglia** | Action selection and habits |
| **DMN Node** | Default Mode Network — daydreaming and introspection |
| **ACC Node** | Anterior Cingulate Cortex — conflict detection |
| **HPA Axis** | Stress response and adaptation |
| **Locus Coeruleus** | Attention modulation and alertness |

---

### 2. Dual-System Architecture

Inspired by *Thinking, Fast and Slow* by Daniel Kahneman.

**System 1 (Fast / Intuitive)**:
- Direct execution by `ExecutorNode`
- For routine tasks, simple questions, fact retrieval
- Low latency, no deep reasoning

**System 2 (Slow / Analytical)**:
- Involves `BrainNode` for planning
- `DualRuntimeGuard` for verification
- Iterative: Plan → Execute → Review → Re-plan if failed
- For complex tasks: coding, debugging, math, security

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
              │  Need System 2? │──No───▶ System 1 (Executor)
              └────────┬────────┘
                       │ Yes
              ┌────────▼────────┐
              │  Brain (Plan)   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Executor (Act)  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Guard (Review) │──Pass──▶ Commit
              └────────┬────────┘
                       │ Fail
                       ▼
                  Re-plan (Brain)
```

Key file: `moko_core/moko_agents/dual_system/orchestrator.py`

---

### 3. NeuroMath Engine

**This is MOKO's mathematical core.** Not just a calculator — it is a mathematical reasoning system inspired by cognitive neuroscience.

`moko_core/moko_neuromath/` contains **60+ modules** including:

| Module | Function |
|-------|----------|
| **PureMathEngine** | Pure math: algebra, calculus, trigonometry, logarithms |
| **ExactMathEngine** | High-precision computation with verification |
| **ComputerMathEngine** | Computer math: floating-point, numerical, error analysis |
| **CSMathEngine** | Data structures, algorithms, complexity |
| **AppliedFormulaEngine** | Applied formulas from various fields |
| **FormalReasoningEngine** | Formal logic and mathematical reasoning |
| **StepLogicProver** | Step-by-step theorem proving |
| **SymbolicRegression** | Discover formulas from data |
| **SymbolicSynthesizer** | Synthesize new formulas from patterns |
| **ProgramVerificationEngine** | Program verification with mathematical logic |
| **ActiveInference** | Formula mutation using Free Energy Principle |
| **FEPEngine** | Free Energy Principle — surprisal minimization |
| **UncertaintyEngine** | Uncertainty quantification |
| **MCTSReasoner** | Monte Carlo Tree Search for reasoning |
| **CognitiveMap** | Cognitive map for abstract navigation |
| **DimensionalSynthesis** | Cross-dimensional and cross-domain synthesis |
| **QuantumSimulator** | Basic quantum computing simulation |
| **TuringConsciousnessEngine** | Turing-based artificial consciousness experiments |
| **FormulaCritic** | Formula critique and evaluation |
| **FormulaGenesisEngine** | Autonomous formula discovery |
| **AppliedMathTrainer** | Applied mathematics training |
| **SleepConsolidation** | Mathematical memory consolidation (sleep-like) |

---

### 4. Marathon Engine

Long-chain reasoning system (`moko_core/moko_marathon/`) that enables MOKO to solve complex problems through step-by-step iteration.

```
                 ┌──────────────────┐
                 │    Question      │
                 └────────┬─────────┘
                          │
              ┌───────────▼───────────┐
              │  Context Pager        │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 1: Analysis     │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Step 2: Reasoning    │
              └───────────┬───────────┘
                          │
                     ──▶ ... ◀──
                          │
              ┌───────────▼───────────┐
              │  Semantic Compression │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Final Answer       │
              └───────────────────────┘
```

Components:
- **MarathonRunner** — reasoning cycle coordinator
- **ContextPager** — context memory management (paging)
- **SemanticCompressor** — automatic context compression to avoid LLM limit
- **PuzzleAssembler** — cross-domain puzzle assembly
- **CodeAssembler** — code fragment assembly
- **CodeVerifier** — generated code verification
- **CrossVerifier** — cross-domain verification
- **MarathonCodeSentinel** — marathon code quality guard
- **MarathonPitstop** — mid-way evaluation checkpoint
- **SecurityAuditor** — code security audit
- **GitManager** — git integration for change tracking

---

### 5. RAG & Memory System

MOKO has a layered memory system that enables knowledge retrieval from multiple sources.

`moko_core/moko_memory/`:

| Component | Function |
|----------|----------|
| **OmniVectorStore** | Multi-domain vector database (code, math, security, general, etc.) |
| **RSAStorage** | RSA-based encrypted storage |
| **WALManager** | Write-Ahead Logging for data integrity |
| **KVCacheManager** | KV-cache management for LLM inference |
| **NeuralWorkingMemory** | Neural short-term working memory |
| **MultiDomainStorage** | Distributed domain-specific storage |
| **SessionStore** | Conversation session storage |
| **DiskManager** | Storage management |
| **GCTuner** | Memory garbage collection tuner |
| **SearchCache** | Search cache for faster retrieval |
| **MokoRAGRetriever** | Primary RAG retrieval |
| **MokoEmbedEngine** | Local embedding engine |
| **BinaryKnowledgeCodec** | Binary knowledge codec for compression |
| **OmniHashEncoder** | Multi-domain hash encoding |
| **OmniStorage** | Distributed omni-domain storage |
| **MathOmni** | Math-specific omni storage |
| **ConvBuffer** | Conversation buffer for context window |
| **Ramdisk** | Fast-access ramdisk |
| **HDCContext** | Hyperdimensional Computing context |

Distributed RAG at port `11437` with auto-boot and safe fallback.

---

### 6. MOKO IDE

MOKO has two IDE interfaces:

#### A. MOKO IDE C++ (Native)
- Built with **Qt6** (C++)
- 21 source files in `moko_ide_cpp/`
- Features: code editor, syntax highlighting, terminal, chat panel, stress test

#### B. MOKO IDE Python
- Terminal interface (`moko_core/moko_os.py`)
- Dual-mode: CLI and GUI (PyQt6)
- LSP Client integration (`moko_core/moko_lsp/lsp_client.py`)
  - Languages: Python, JavaScript, TypeScript, JSON, CSS, HTML
  - Features: diagnostics, autocomplete, go-to-definition
  - Safe fallback when server unavailable
- Code Structure Analysis (`moko_core/moko_utils/code_structure.py`)
  - Real-time HTML/XML tag pair checking
  - Bracket imbalance detection `()[]{}`
  - Wave underline + status bar indicator

---

### 7. Native Acceleration (C++/Rust)

Multiple native components for maximum performance:

#### C++ Native (`moko_core/moko_native/`)
- Compile with `build.sh`, benchmark with `bench.py`

#### C++ Core Kernel (`moko_core/moko_cpp_kernel/`)
- `moko_kernel.cpp` — main kernel
- `simd_math.hpp` — SIMD math operations
- `thread_pool.hpp` — parallelization thread pool
- `mmap_io.hpp` — memory-mapped I/O
- `libmoko_core.so` — compiled shared library

#### Rust Core (`moko_core/moko_rust_core/`)
- Rust module with Cargo, focused on safe memory operations

---

### 8. Security System

`moko_core/moko_security/`:

| Module | Function |
|-------|----------|
| **RedTeamFuzzer** | Automated security fuzzer — finds vulnerabilities |
| **BlueTeamDefender** | Automated defender — protects against attacks |
| **GaussianNoiseEngine** | Gaussian noise for differential privacy |

---

### 9. Fine-Tuning Pipeline

`finetune/` contains a complete pipeline for fine-tuning coding models:

**Base Model**: Qwen2.5-1.5B-Instruct (HF format, ~3GB)
**Output**: LoRA adapter → GGUF for llama.cpp

```
python3 moko_finetune.py --prepare    # Download base model from HuggingFace
python3 moko_finetune.py --build      # Build coding dataset
python3 moko_finetune.py --train      # Start fine-tuning
python3 moko_finetune.py --status     # Check training status
```

**Datasets** (in `finetune/moko_datasets/`):
- 13 datasets covering: coding algorithms, security, reasoning, multi-turn conversations, OS code, programming, documentation, hex encoding, knowledge distillation, Qt/C++, IDE integration

**Pipeline**:
```
Download HF Model → Build Dataset → LoRA Training → Convert to GGUF → Load in llama.cpp
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/brianatmoko/MOKO-AI.git
cd MOKO-AI

# 2. Virtual environment
python3 -m venv moko_core/venv --upgrade-deps
source moko_core/venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
./moko.sh                      # Auto-detect mode
./moko.sh --cli                # Force CLI mode
./moko.sh --gui                # GUI mode (needs Qt6 + display)
./moko.sh --daemon             # Web API mode (http://127.0.0.1:8000)
./moko_cli.sh                  # CLI-only launcher
```

### CLI Commands

```
./moko.sh --cli
```

- Type any question to start a conversation
- `/ai on` — enable local LLM
- `/ai off` — disable local LLM
- `exit` — quit

### Daemon API

```bash
./moko.sh --daemon
curl http://127.0.0.1:8000
```

---

## Requirements

| Component | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 4-core | 8-core+ (for local inference) |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | - | RTX 2050 4GB+ (for fine-tuning) |
| **Storage** | 1 GB (source) | 10 GB+ (with models) |
| **OS** | Linux (kernel 5.x+) | Linux (kernel 6.x+) |
| **Python** | 3.12 | 3.12 |
| **Qt** | - | Qt6 (for IDE) |
| **C++** | g++ 11+ | g++ 13+ (for native) |
| **Rust** | - | Rust 1.70+ (for rust core) |

### Hardware Notes
- **Fine-tuning**: Tested on RTX 2050 4GB VRAM + 16GB RAM — slow but stable
- **Local inference**: Needs GGUF model + llama.cpp. Auto-setup by MOKO Inference Server
- **No GPU**: MOKO works fully in CLI mode with external LLM API

---

## Project Structure

```
MOKO-AI/
├── moko.sh                     # Main launcher (auto-detect)
├── moko_cli.sh                 # CLI-only launcher
├── moko_launcher.sh            # Alternative launcher
├── moko-warmup.sh              # System warmup
├── moko-models.sh              # Model management
├── moko-fix-limits.sh          # System limits fix
├── build_ide.sh                # Build C++ IDE
├── build_kernel.sh             # Build C++ kernel
├── setup.py                    # pip install setup
├── requirements.txt            # Python dependencies
│
├── moko_core/                  # ★ SYSTEM CORE
│   ├── moko_os.py              # CLI entry point
│   ├── moko_config/            # System configuration
│   ├── moko_agents/            # Multi-agent cognitive system
│   │   ├── router.py           # Main cognitive router
│   │   ├── intent_router.py    # Intent classifier
│   │   ├── dual_system/        # System 1 + System 2
│   │   ├── coding/             # 7 coding agents
│   │   ├── software_builder/   # Software builder agent
│   │   └── 30+ cognitive node agents
│   ├── moko_neuromath/         # ★ Math engine (60+ modules)
│   ├── moko_marathon/          # Marathon reasoning engine
│   ├── moko_memory/            # Memory & RAG system
│   ├── moko_inference/         # LLM inference server
│   ├── moko_crawler/           # Web & Tor crawler
│   ├── moko_security/          # Security system
│   ├── moko_super_learning/    # Super learning engine
│   ├── moko_native/            # C++/Rust acceleration
│   ├── moko_cpp_kernel/        # C++ kernel
│   ├── moko_cpp_core/          # C++ core library
│   ├── moko_rust_core/         # Rust core module
│   ├── moko_lsp/               # LSP client
│   ├── moko_puzzles/           # Puzzle system
│   ├── moko_cpu/               # CPU scheduler & governor
│   ├── moko_tools/             # Tools (quantization, encoding, etc.)
│   ├── moko_benchmark/         # System benchmarks
│   └── moko_utils/             # Utilities
│
├── moko_ide_cpp/               # MOKO IDE C++ (Qt6)
│   ├── main.cpp                # Entry point
│   ├── moko_window.*           # Main window
│   ├── code_editor.*           # Code editor
│   ├── chat_widget.*           # AI chat panel
│   ├── terminal_widget.*       # Integrated terminal
│   ├── syntax_highlighter.*    # Syntax highlighting
│   ├── helper_engine.*         # AI helper
│   ├── find_bar.*              # Find & replace
│   ├── settings_dialog.*       # Settings
│   ├── graphify_dialog.*       # Graph visualization
│   └── stress_test_dialog.*    # Stress test
│
├── finetune/                   # Fine-tuning pipeline
│   ├── moko_finetune.py        # Main script
│   ├── moko_trainer_v2.py      # Trainer v2
│   ├── moko_trainer_v3_7b.py   # Trainer v3 (7B)
│   ├── moko_data_factory.py    # Data factory
│   ├── moko_distill_engine.py  # Knowledge distillation
│   ├── moko_byteq_compressor.py# ByteQ compression
│   ├── moko_datasets/          # 13 coding datasets
│   └── 20+ supporting scripts
│
├── moko_config/                # Configuration
│   ├── api_keys.json           # API keys (OpenRouter, etc.)
│   └── moko_settings.json      # System settings
│
├── docs/                       # Documentation
│   ├── RINGKASAN_PROGRES.md    # Progress summary (ID)
│   ├── STRUKTUR_FOLDER_MOKO.md # Detailed structure (ID)
│   └── riset/                  # 19 technical research documents
│
└── .github/workflows/          # CI/CD
    └── ci.yml                  # GitHub Actions
```

---

## Key Technical Concepts

### ByteQ Quantization
An extreme 2-bit Lloyd quantization technique developed in-house. Compresses tensors ~6.8x with very low MSE (~1e-4). Implementation in `moko_tools/byteq_quantizer.py`.

### Omni Storage
Multi-domain vector storage organizing knowledge into 8 domains: code, math, security, general, finance, personal, programming, reasoning. Each domain has embedding centroids for precise retrieval.

### Cognitive Templates
Cognitive templates for various reasoning types. The system learns from experience and refines its templates over time.

### Marathon Pitstop
A "rest" mechanism mid-reasoning to evaluate progress, compress context, and decide whether to continue or conclude.

### API Configuration
API keys are in `moko_config/api_keys.json`. Default providers:
- **OpenRouter** — Free cloud LLM access (Llama 3.1 8B, Gemma 3 12B, etc.)
- **Local LLM** — via llama.cpp server on local port
- **OpenCode** — Local development API

---

## Roadmap

### Achieved
- ✅ Multi-agent cognitive system with brain-inspired neural architecture
- ✅ Math engine with 60+ reasoning modules
- ✅ Dual-System (fast + slow) with iterative loop
- ✅ Multi-domain RAG with local vector store
- ✅ Fine-tuning pipeline (Qwen → MOKO Coder)
- ✅ Native C++ IDE with LSP and AI integration
- ✅ Marathon reasoning for complex problems
- ✅ Darkweb crawling via Tor
- ✅ Cross-domain puzzle system

### Needs Work
- Full integration between all components
- Performance and memory optimization
- API documentation and developer guide
- Comprehensive unit testing
- Easy packaging and distribution
- Pre-fine-tuned MOKO Coder model
- Mature GUI

---

## Contributing

This project is **open-source** and needs contributors. If you're interested in autonomous AI, multi-agent systems, computational math, or IDE development:

1. **Fork** this repository
2. **Clone** your fork
3. **Read** the docs in `docs/` to understand the architecture
4. **Start** with a [good first issue](https://github.com/brianatmoko/MOKO-AI/issues)
5. **Submit** a Pull Request

### Areas Needing Help
- Documentation and usage examples
- Testing and QA
- Performance optimization
- Packaging (`pip install` ready)
- Frontend/GUI improvement

> *"MOKO is an ambitious project built with limited resources. But I believe the foundation is solid. I entrust it to anyone who believes that intelligent, autonomous local AI is a future worth fighting for."*
>
> — Brian Atmoko, Creator of MOKO

---

## License

MIT License — free to use, modify, and distribute.

---

<p align="center">
  <b>MOKO AI</b><br>
  <i>Not just a chatbot. A cognitive operating system.</i><br>
  <a href="https://github.com/brianatmoko/MOKO-AI">github.com/brianatmoko/MOKO-AI</a>
</p>
