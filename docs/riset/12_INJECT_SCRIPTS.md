# 12 — Inject Scripts Consolidated

> **Tujuan:** Dokumentasi lengkap semua script injeksi data ke knowledge base.
> Setiap script mengisi domain spesifik di `.moko_omni/`.

---

## Daftar Isi

1. [Overview](#1-overview)
2. [Script Details](#2-script-details)
3. [Dependency Map](#3-dependency-map)
4. [Execution Order](#4-execution-order)

---

## 1. Overview

| Script | Domain | Source | Entries | Size |
|--------|--------|--------|---------|------|
| `inject_hacking_course.py` | cybersecurity | Ethical-Hacking-Course-Bank/ | ~50 files | 2.2 KB |
| `inject_kbbi.py` | lexical | kbbi.csv (32 MB) | ~10,000+ words | 5.4 KB |
| `inject_math_curriculum.py` | math | Hardcoded curriculum | ~100 topics | 59 KB |
| `inject_math_extended.py` | math | Extended curriculum | ~150 topics | 68 KB |
| `inject_motor_cc_formula.py` | math | Motor CC formulas | 1 formula | 2.5 KB |
| `inject_wikipedia.py` | multiple | Wikipedia API | ~100 articles | 14.8 KB |

---

## 2. Script Details

### 2.1 inject_hacking_course.py

```python
# Purpose: Inject ethical hacking course materials
# Source: Ethical-Hacking-Course-Bank-main/ (md/txt files)
# Domain: cybersecurity
# Method: Split by paragraphs, embed with local engine

# Usage:
python inject_hacking_course.py

# Dependencies:
from moko_agents.llm_engine import engine
from moko_memory.disk_manager import DiskManager
```

**Flow:**
1. Scan `Ethical-Hacking-Course-Bank-main/` for .md/.txt files
2. Split content into chunks (paragraph-based)
3. Embed each chunk with local engine
4. Store via DiskManager

---

### 2.2 inject_kbbi.py

```python
# Purpose: Inject KBBI (Kamus Besar Bahasa Indonesia) dictionary
# Source: kbbi.csv (32 MB)
# Domain: lexical
# Method: CSV → semantic text → embed → RSAStorage

# Usage:
python inject_kbbi.py

# Dependencies:
from moko_agents.llm_engine import engine
from moko_memory.rsa_storage import RSAStorage
from moko_cpu.governor import CPUGovernor
```

**Features:**
- RAM disk support (/dev/shm/moko_omni)
- CPU thermal guard (breathe every 50 words)
- Auto-sync to SSD every 500 words
- Resume from last index (kbbi_state.json)

---

### 2.3 inject_math_curriculum.py

```python
# Purpose: Inject math curriculum (algebra, calculus, etc.)
# Source: Hardcoded MATH_CURRICULUM array
# Domain: math
# Method: Topic → content → embed → RSAStorage

# Usage:
python inject_math_curriculum.py

# Dependencies:
from moko_agents.llm_engine import engine
from moko_memory.rsa_storage import RSAStorage
```

**Topics covered:**
- Aritmatika & Teori Bilangan
- Aljabar Elementer & Menengah
- Trigonometri
- Kalkulus (Differential & Integral)
- Statistika & Probabilitas
- Geometri & Analisis Riil

---

### 2.4 inject_math_extended.py

```python
# Purpose: Inject extended math curriculum (foundational to advanced)
# Source: Hardcoded MATH_CURRICULUM array
# Domain: math
# Method: Topic → content → embed → RSAStorage

# Usage:
python inject_math_extended.py

# Dependencies:
from moko_agents.llm_engine import engine
from moko_memory.rsa_storage import RSAStorage
```

**Topics covered:**
- Fondasi Aritmatika (membilang, nilai tempat)
- Operasi Dasar (penjumlahan, pengurangan, perkalian, pembagian)
- Pecahan & Desimal
- Persen & Rasio
- Geometri Dasar
- Pengukuran

---

### 2.5 inject_motor_cc_formula.py

```python
# Purpose: Inject motorcycle CC (cylinder capacity) formula
# Source: Hardcoded formula text
# Domain: math
# Method: Text → embed → DiskManager

# Usage:
python inject_motor_cc_formula.py

# Dependencies:
from moko_agents.llm_engine import engine
from moko_memory.disk_manager import DiskManager
```

**Formula:**
```
V = (π/4) × d² × s × N

V = Kapasitas silinder (cc/cm³)
d = Diameter bore (cm)
s = Panjang stroke (cm)
N = Jumlah silinder
```

---

### 2.6 inject_wikipedia.py

```python
# Purpose: Inject curated Wikipedia articles
# Source: Wikipedia API (live fetch)
# Domain: multiple (programming, math, physics, security)
# Method: Fetch → chunk → embed → DiskManager

# Usage:
python inject_wikipedia.py [--domain programming] [--lang en]

# Dependencies:
import requests
from moko_agents.llm_engine import engine
from moko_memory.disk_manager import DiskManager
```

**Domains covered:**
- programming: 18 articles (Python, C++, Rust, etc.)
- math: 15 articles (Algebra, Calculus, etc.)
- physics: 12 articles (Quantum, Relativity, etc.)
- security: 10 articles (Cryptography, Hacking, etc.)

---

## 3. Dependency Map

```
inject_hacking_course.py
  ├── moko_agents.llm_engine
  └── moko_memory.disk_manager

inject_kbbi.py
  ├── moko_agents.llm_engine
  ├── moko_memory.rsa_storage
  └── moko_cpu.governor

inject_math_curriculum.py
  ├── moko_agents.llm_engine
  └── moko_memory.rsa_storage

inject_math_extended.py
  ├── moko_agents.llm_engine
  └── moko_memory.rsa_storage

inject_motor_cc_formula.py
  ├── moko_agents.llm_engine
  └── moko_memory.disk_manager

inject_wikipedia.py
  ├── requests (external)
  ├── moko_agents.llm_engine
  └── moko_memory.disk_manager
```

---

## 4. Execution Order

### Recommended Order:
1. `inject_kbbi.py` — Foundation (lexical database)
2. `inject_math_curriculum.py` — Core math knowledge
3. `inject_math_extended.py` — Extended math knowledge
4. `inject_wikipedia.py` — General knowledge
5. `inject_hacking_course.py` — Security knowledge
6. `inject_motor_cc_formula.py` — Specific formula

### Prerequisites:
- LLM server running (port 11435 for inference, 11436 for embedding)
- `.moko_omni/` directory exists
- For inject_kbbi.py: kbbi.csv in project root

### Notes:
- Scripts 2-4 use RSAStorage (older API)
- Scripts 1, 5-6 use DiskManager (newer API)
- All scripts require embedding engine
