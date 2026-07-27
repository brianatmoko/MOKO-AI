# 11 — Data Inventory & Reconstruction Plan

> **⚠️ PEMBERITAHUAN TRANSISI: NO-CRYPTO ARCHITECTURE**
> Sistem kripto telah dinonaktifkan secara permanen. Fokus saat ini adalah pada optimasi memori Omni dan integrasi model MOKO AI (non-Qwen rebranding).

> **Tujuan:** Inventaris lengkap semua data, file, dan modul yang dibutuhkan
> untuk rekonstruksi MOKO OS dari state saat ini.

---

## Daftar Isi

1. [Status Saat Ini](#1-status-saat-ini)
2. [Model Files](#2-model-files)
3. [Knowledge Base](#3-knowledge-base)
4. [Core Modules](#4-core-modules)
5. [Inject Scripts](#5-inject-scripts)
6. [Cleanup Plan (R2 Fase 3)](#6-cleanup-plan-r2-fase-3)
7. [Reconstruction Priority](#7-reconstruction-priority)

---

## 1. Status Saat Ini

### Completed (R2 Fase 1-4)
- ✅ `CRYPTO_ENABLED=False` di settings.py
- ✅ `OMNI_DIR` updated ke `.moko_omni`
- ✅ Directory renamed: `.moko_crypto` → `.moko_omni`
- ✅ 10 research documents in `docs/riset/`
- ✅ Cleanup old crypto directories (.moko_crypto_*)
- ✅ Delete unused large GGUF models (>15GB saved)
- ✅ Cleanup legacy crypto modules (rm crypto_*.py, etc)
- ✅ Rebranding: Remove MOKO, Implement MOKO Identity

### Pending (R2 Fase 5)
- ❌ Consolidate scattered inject scripts
- ✅ Update settings.py for RAG 200MB (Completed)
- ✅ Intent-First Router upgrade (Completed via Multi-Agent Bridge)

---

## 2. Model Files

### Active Models (Inference)
```
MOKO-AI-4B-Q3_K_M.gguf                2.2 GB  ← Main backbone
MOKO-Coder-1.5B-Uncensored-F16.gguf   3.4 GB  ← Coding expert
moko-rag.gguf                         0.98 GB ← RAG Bridge (Agent 2)
```

### Reference Models (Identity)
```
None (Old BF16 deleted to save space)
```

### Total Disk Usage: ~6.6 GB (models only)

### Model Path Configuration (settings.py)
```python
MOKO_Q3_MODEL_PATH   = "MOKO-AI-4B-Q3_K_M.gguf"    # Inference
MODEL_MOKO_GGUF_PATH = MOKO_Q3_MODEL_PATH           # Alias
MODEL_RAG_LLM_PATH   = "moko-rag.gguf"              # RAG Bridge
```

---

## 3. Knowledge Base

### Active: `.moko_omni/` (2.2 GB, 174K entries)
```
.code/          1,687 entries  ← Programming, software engineering
.cyberoffensive/  929 entries  ← Hacking, exploits, offensive security
.cybersecurity/   978 entries  ← Defense, hardening, compliance
.general/       1,222 entries  ← General knowledge
.general_sub_1/  674 entries  ← Sub-domain general
.lexical/      10,830 entries  ← Words, definitions, KBBI
.math/          1,051 entries  ← Mathematics, formulas
.personal/       759 entries  ← User preferences, personal data
.physics/        660 entries  ← Physics, engineering
.test_domain/      3 entries  ← Test data
```

### Old Directories (Cleaned)
```
.moko_crypto_backup/   (Deleted)
.moko_crypto_cache/    (Deleted)
.moko_crypto_omni/     (Deleted)
```

### Cache: `.moko_cache/`
```
session_chains/        ← Session chain data
```

---

## 4. Core Modules

### Main Entry Point
```
moko_core/moko_os.py              ← Main loop, imports all modules
```

### Configuration
```
moko_core/moko_config/settings.py ← Global settings (UPDATED: CRYPTO_ENABLED=False)
```

### Agents (Routing & Processing)
```
moko_core/moko_agents/router.py           ← Intent router (needs upgrade)
moko_core/moko_agents/analyst_node.py     ← Analyst phase
moko_core/moko_agents/core_node.py        ← Core phase
moko_core/moko_agents/llm_engine.py       ← LLM interface
moko_core/moko_agents/moko_crypto_core.py ← DISABLED (crypto bypass)
```

### Memory System
```
moko_core/moko_memory/disk_manager.py        ← Disk operations
moko_core/moko_memory/multi_domain_storage.py ← Multi-domain storage
moko_core/moko_memory/rsa_storage.py         ← RSA storage (disabled)
moko_core/moko_memory/crypto_*.py            ← Crypto modules (disabled)
```

### Math Engine
```
moko_core/moko_neuromath/               ← 77 modules (Math-Brain Engine v7)
moko_core/moko_neuromath/__init__.py    ← Module registry
moko_core/moko_neuromath/exact_math_engine.py ← Core math engine
```

### Security
```
moko_core/moko_security/               ← Security modules
moko_core/moko_security/crypto_gateway.py ← Gateway (disabled)
```

### Tools
```
moko_core/moko_tools/gguf_quantizer.py ← GGUF quantization tools
moko_core/moko_tools/gguf_editor.py    ← GGUF editing tools
```

### UI
```
moko_core/moko_ui/terminal_ui.py       ← Terminal interface
moko_core/moko_ui/main_window.py       ← GUI window
```

---

## 5. Inject Scripts

### Root Level (Scattered)
```
inject_hacking_course.py    ← Inject hacking course data
inject_kbbi.py              ← Inject KBBI dictionary
inject_math_curriculum.py   ← Inject math curriculum
inject_math_extended.py     ← Inject extended math
inject_motor_cc_formula.py  ← Inject motor CC formulas
inject_wikipedia.py         ← Inject Wikipedia data
build_omni.py               ← Build omni database
```

### Purpose
- These scripts populate `.moko_omni/` with knowledge
- They should be consolidated or documented

---

## 6. Cleanup Plan (R2 Fase 3)

### Step 1: Verify No Dependencies
```bash
# Check if any code references old directories
grep -r "moko_crypto_backup" moko_core/
grep -r "moko_crypto_cache" moko_core/
grep -r "moko_crypto_omni" moko_core/
```

### Step 2: Archive (Optional)
```bash
# If user wants to keep backups
tar -czf .moko_crypto_backup.tar.gz .moko_crypto_backup/
```

### Step 3: Delete Old Directories
```bash
rm -rf .moko_crypto_backup/
rm -rf .moko_crypto_cache/
rm -rf .moko_crypto_omni/
```

### Step 4: Update Settings
```python
# Remove any references to old directories in settings.py
# Ensure OMNI_DIR = WORKSPACE_DIR / ".moko_omni" (already done)
```

---

## 7. Reconstruction Priority

### Phase 1: Cleanup (Immediate)
1. Delete old crypto directories (704 MB savings)
2. Update settings.py if needed
3. Verify system still works

### Phase 2: Documentation (1-2 days)
1. Document model file purposes
2. Document inject scripts
3. Create reconstruction roadmap

### Phase 3: Architecture Upgrade (1-2 weeks)
1. Intent-First Router upgrade
2. Byte-Q quantization implementation
3. Multi-Model dispatcher

### Phase 4: New Features (2-4 weeks)
1. Domain-specialized models
2. RAG + AI Agent for directory management
3. Knowledge ingestion pipeline

---

## 8. Quick Reference

### Key Files to Monitor
| File | Purpose | Status |
|------|---------|--------|
| `settings.py` | Global config | ✅ Updated |
| `router.py` | Intent routing | ⚠️ Needs upgrade |
| `moko_os.py` | Main entry | ✅ Working |
| `llm_engine.py` | LLM interface | ✅ Working |
| `disk_manager.py` | Disk ops | ✅ Working |

### Key Directories
| Directory | Size | Purpose | Action |
|-----------|------|---------|--------|
| `.moko_omni/` | 2.2 GB | Active knowledge base | Keep |
| `.moko_crypto_backup/` | 704 MB | Old backup | Delete |
| `.moko_crypto_cache/` | 52 KB | Old cache | Delete |
| `.moko_crypto_omni/` | 164 KB | Old omni | Delete |
| `docs/riset/` | ~50 KB | Research docs | Keep |

### Disk Space Summary
```
Total project: ~12.5 GB
  Models:      ~10.1 GB (81%)
  Knowledge:    ~2.2 GB (17%)
  Code/docs:    ~0.1 GB (<1%)
  Free (approx): ~0.1 GB
```
