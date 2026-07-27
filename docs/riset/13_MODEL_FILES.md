# 13 — Model Files Documentation

> **Tujuan:** Dokumentasi lengkap semua file model GGUF, termasuk
> ukuran, tujuan, path, dan cara penggunaannya.

---

## Daftar Isi

1. [Model Registry](#1-model-registry)
2. [File Details](#2-file-details)
3. [Path Configuration](#3-path-configuration)
4. [Size Analysis](#4-size-analysis)

---

## 1. Model Registry

| File | Size | Quant | Purpose | Status |
|------|------|-------|---------|--------|
| `MOKO-AI-4B-Q3_K_M.gguf` | 2.2 GB | Q3_K_M | Main Inference Backbone | **ACTIVE** |
| `MOKO-Coder-1.5B-Uncensored-F16.gguf` | 3.4 GB | F16 | Specialized Coding Agent | **ACTIVE** |
| `moko-rag.gguf` | 981 MB | Q2_K | RAG Bridge Agent (200MB VRAM target) | **ACTIVE** |

---

## 2. File Details

### 2.1 MOKO-AI-4B-Q3_K_M.gguf

```yaml
Size: 2.2 GB
Quantization: Q3_K_M (3-bit, k-quant medium)
Architecture: Qwen2.5 (MOKO Rebranded)
Purpose: Main inference backbone for general and logic tasks.
Usage: Primary model for Agent 1 (Output Agent).
Location: Project root directory
```

### 2.2 MOKO-Coder-1.5B-Uncensored-F16.gguf

```yaml
Size: 3.4 GB
Quantization: F16 (Float16)
Architecture: Qwen2.5-1.5B (Uncensored)
Purpose: Specialized coding, math, and technical analysis.
Usage: Domain expert for complex technical queries.
Location: Project root directory
```

### 2.3 moko-rag.gguf

```yaml
Size: 981 MB
Quantization: Q2_K (2-bit, k-quant)
Architecture: Qwen2.5-1.5B (MOKO Rebranded)
Purpose: RAG Bridge Agent (Agent 2 - Data Manager).
Usage: Extremely efficient context processing for Omni knowledge extraction.
Target: 200MB VRAM footprint via partial offloading.
Location: Project root directory
```

---

## 3. Path Configuration

### settings.py

```python
# Project directory (auto-detected)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Main Models
MOKO_Q3_MODEL_PATH = str(PROJECT_DIR / "MOKO-AI-4B-Q3_K_M.gguf")
MODEL_MOKO_GGUF_PATH = MOKO_Q3_MODEL_PATH

# Domain Models
DOMAIN_MODEL_REGISTRY = {
    "coding": {"path": str(PROJECT_DIR / "MOKO-Coder-1.5B-Uncensored-F16.gguf"), ...},
    ...
}

# RAG Model
MODEL_RAG_LLM_PATH = str(PROJECT_DIR / "moko-rag.gguf")
```

---

## 4. Size Analysis

### Disk Usage
```
Total model files:     ~6.6 GB
  MOKO-AI 4B:          2.2 GB
  MOKO-Coder 1.5B:     3.4 GB
  MOKO-RAG:            0.98 GB
```

### VRAM Usage (Runtime - Multi-Agent Bridge)
```
Main LLM (Agent 1):    ~2.5 GB (Loaded)
RAG LLM (Agent 2):     ~200 MB (Budgeted via 4 GPU Layers)
KV Cache & Overhead:   ~0.8 GB
Total VRAM:            ~3.5 GB (Safe for 4GB GPUs like RTX 2050)
```
