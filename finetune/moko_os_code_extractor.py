"""
MOKO OS Code Extractor — Training Data dari Source Code MOKO
=============================================================
Mengekstrak source code dari project MOKO OS sendiri dan mengubahnya
menjadi training data yang kaya konteks.

Strategi:
  1. Scan semua file Python dan C++ di project MOKO OS
  2. Untuk setiap function/class yang cukup kompleks, generate Q&A:
     - Q: "Bagaimana cara mengimplementasikan X?"
     - A: [code yang sebenarnya dari MOKO OS]
  3. Juga extract docstrings, comments sebagai penjelasan
  4. Output: JSONL siap untuk fine-tuning

Hasilnya: Model 1B tahu MOKO OS dari dalam — bukan hafal, tapi benar-benar mengerti.

Usage:
  python3 moko_os_code_extractor.py              # Ekstrak semua
  python3 moko_os_code_extractor.py --python     # Python saja
  python3 moko_os_code_extractor.py --cpp        # C++ saja
  python3 moko_os_code_extractor.py --stats      # Statistik
"""

import os
import re
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR  = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR  = FINETUNE_DIR / "moko_datasets"
OUTPUT_FILE  = DATASET_DIR / "moko_os_code_dataset.jsonl"

# Direktori yang akan di-scan
SCAN_DIRS = [
    PROJECT_DIR / "moko_core",
    PROJECT_DIR / "moko_ide_cpp",
    PROJECT_DIR / "finetune",
]

# Pattern untuk skip
SKIP_DIRS = {
    "__pycache__", "venv", ".git", "node_modules",
    "build", "dist", ".vscode", ".junie",
    "llama_bin", "base_model", "base_model_hf",
    "moko_adapters", "moko_datasets",
}
SKIP_FILES = {
    "test_", "_test.", ".pyc", "conftest",
}

# Mimimum panjang kode untuk diekstrak
MIN_FUNC_LINES = 5
MIN_CLASS_LINES = 10
MAX_CODE_LINES  = 150  # Jangan terlalu panjang (context limit 2048 token)


# ═══════════════════════════════════════════════════════════════════════════
# MOKO CODER SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

MOKO_CODER_SYSTEM = """You are MOKO Coder, an expert AI coding assistant built exclusively for MOKO OS and MOKO IDE.

IDENTITY:
- Name: MOKO Coder
- Version: 1.0.0
- Platform: MOKO OS (custom Linux-based OS)
- Built for: MOKO IDE — AI-powered development environment
- Motto: "Kode yang efisien, solusi yang cerdas."

CORE EXPERTISE:
1. Code Generation — Write clean, complete, runnable code
2. MOKO OS Integration — Deep knowledge of MOKO OS APIs and architecture
3. C/C++ & Python — Primary languages of MOKO OS development
4. Bug Detection & Fix — Identify and resolve errors with explanation
5. Documentation — Generate docstrings and inline comments"""


# ═══════════════════════════════════════════════════════════════════════════
# PYTHON CODE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class PythonExtractor:
    """Ekstrak functions dan classes dari file Python."""

    # Pattern untuk mendeteksi fungsi dan class
    FUNC_PATTERN = re.compile(
        r'^(    |\t)*(async\s+)?def\s+(\w+)\s*\(', re.MULTILINE
    )
    CLASS_PATTERN = re.compile(
        r'^class\s+(\w+)[\s:(]', re.MULTILINE
    )
    DOCSTRING_PATTERN = re.compile(
        r'"""(.*?)"""', re.DOTALL
    )

    def extract_functions(self, source: str, filepath: Path) -> List[Dict]:
        """Ekstrak semua fungsi dari source Python."""
        samples = []
        lines = source.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Cari definisi fungsi
            func_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
            if func_match:
                indent   = len(func_match.group(1))
                func_name = func_match.group(3)

                # Skip private/dunder functions kecuali yang penting
                if func_name.startswith("__") and func_name not in (
                    "__init__", "__str__", "__repr__", "__call__",
                    "__enter__", "__exit__", "__len__", "__getitem__",
                ):
                    i += 1
                    continue

                # Extract body hingga indentasi kembali ke awal
                func_lines = [lines[i]]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # Kosong atau lebih dalam dari indent — masih bagian fungsi
                    if not next_line.strip() or len(next_line) - len(next_line.lstrip()) > indent:
                        func_lines.append(next_line)
                        j += 1
                    else:
                        break

                func_code = "\n".join(func_lines).rstrip()
                func_lines_count = len([l for l in func_lines if l.strip()])

                if func_lines_count >= MIN_FUNC_LINES:
                    # Potong jika terlalu panjang
                    if len(func_lines) > MAX_CODE_LINES:
                        func_code = "\n".join(func_lines[:MAX_CODE_LINES]) + "\n    # ... (truncated)"

                    # Ekstrak docstring jika ada
                    docstring = self._extract_docstring(func_code)

                    # Generate training sample
                    sample = self._make_python_sample(
                        func_name=func_name,
                        func_code=func_code.strip(),
                        docstring=docstring,
                        filepath=filepath,
                        is_async="async" in (func_match.group(2) or ""),
                    )
                    if sample:
                        samples.append(sample)

                i = j
                continue

            i += 1

        return samples

    def extract_classes(self, source: str, filepath: Path) -> List[Dict]:
        """Ekstrak class definitions dari source Python."""
        samples = []
        lines = source.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            class_match = re.match(r'^class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(1)

                # Extract class body
                class_lines = [lines[i]]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip() or next_line[0] in (' ', '\t'):
                        class_lines.append(next_line)
                        j += 1
                    else:
                        break

                class_code = "\n".join(class_lines).rstrip()
                class_lines_count = len([l for l in class_lines if l.strip()])

                if class_lines_count >= MIN_CLASS_LINES:
                    # Potong jika terlalu panjang
                    if len(class_lines) > MAX_CODE_LINES:
                        class_code = "\n".join(class_lines[:MAX_CODE_LINES]) + "\n    # ... (truncated)"

                    docstring = self._extract_docstring(class_code)
                    sample = self._make_class_sample(
                        class_name=class_name,
                        class_code=class_code.strip(),
                        docstring=docstring,
                        filepath=filepath,
                    )
                    if sample:
                        samples.append(sample)

                i = j
                continue

            i += 1

        return samples

    def _extract_docstring(self, code: str) -> str:
        """Ekstrak docstring pertama dari kode."""
        match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"'''(.*?)'''", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _make_python_sample(
        self, func_name: str, func_code: str, docstring: str,
        filepath: Path, is_async: bool,
    ) -> Optional[Dict]:
        """Buat training sample dari fungsi Python."""
        # Buat beberapa variasi pertanyaan
        module_name = filepath.stem
        is_moko = "moko" in str(filepath).lower()

        question_variants = [
            f"Implementasikan fungsi Python `{func_name}` untuk digunakan di {module_name}.",
            f"Tulis fungsi {'async ' if is_async else ''}`{func_name}` dalam Python.",
            f"Buat fungsi Python `{func_name}`{'yang asynchronous' if is_async else ''}.",
        ]

        if docstring:
            question_variants.append(
                f"Buat fungsi Python yang: {docstring[:200]}"
            )

        if is_moko:
            question_variants.append(
                f"Bagaimana implementasi `{func_name}` di modul {module_name} MOKO OS?"
            )

        user_prompt = random.choice(question_variants)
        response    = f"```python\n{func_code}\n```"

        if docstring:
            response = f"{docstring}\n\n```python\n{func_code}\n```"

        return {
            "messages": [
                {"role": "system",    "content": MOKO_CODER_SYSTEM},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "moko_os_code",
                "file":   str(filepath.name),
                "type":   "python_function",
                "name":   func_name,
            }
        }

    def _make_class_sample(
        self, class_name: str, class_code: str, docstring: str, filepath: Path
    ) -> Optional[Dict]:
        """Buat training sample dari class Python."""
        module_name = filepath.stem
        is_moko     = "moko" in str(filepath).lower()

        question_variants = [
            f"Implementasikan class Python `{class_name}`.",
            f"Tulis class `{class_name}` di Python.",
            f"Buat implementasi class `{class_name}` untuk {module_name}.",
        ]

        if docstring:
            question_variants.append(
                f"Buat class Python yang: {docstring[:200]}"
            )

        if is_moko:
            question_variants.append(
                f"Bagaimana implementasi class `{class_name}` di modul MOKO {module_name}?"
            )

        user_prompt = random.choice(question_variants)
        response    = f"```python\n{class_code}\n```"

        if docstring:
            response = f"{docstring}\n\n```python\n{class_code}\n```"

        return {
            "messages": [
                {"role": "system",    "content": MOKO_CODER_SYSTEM},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "moko_os_code",
                "file":   str(filepath.name),
                "type":   "python_class",
                "name":   class_name,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# C++ CODE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class CppExtractor:
    """Ekstrak functions dan classes dari file C++."""

    # Match: return_type ClassName::method_name(params) atau function_name(params)
    FUNC_PATTERN = re.compile(
        r'^[\w\s\*\&:<>]+\s+(?:(\w+)::)?(\w+)\s*\([^;{]*\)\s*(?:const\s*)?\{',
        re.MULTILINE
    )
    CLASS_PATTERN = re.compile(
        r'^(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{',
        re.MULTILINE
    )

    def extract_functions(self, source: str, filepath: Path) -> List[Dict]:
        """Ekstrak fungsi-fungsi dari source C++."""
        samples = []
        lines   = source.splitlines()

        for match in self.FUNC_PATTERN.finditer(source):
            class_name = match.group(1) or ""
            func_name  = match.group(2)

            # Skip konstruktor trivial dan destruktor kecil
            if func_name.startswith("~"):
                continue

            # Cari body dengan bracket matching
            start_pos = match.end() - 1  # posisi '{'
            body = self._extract_braced_block(source, start_pos)

            if not body:
                continue

            # Hitung baris kode
            body_lines = body.splitlines()
            code_lines = len([l for l in body_lines if l.strip()])

            if code_lines < MIN_FUNC_LINES or code_lines > MAX_CODE_LINES:
                continue

            # Nama lengkap fungsi
            full_name = f"{class_name}::{func_name}" if class_name else func_name

            # Signature = dari match.start() sampai '{'
            match_start_line = source[:match.start()].count('\n')
            sig_end          = match.start() + match.end() - match.start()
            signature_code   = match.group(0).strip()

            full_code = signature_code + "\n" + body

            # Cari comment sebelum fungsi
            comment = self._extract_preceding_comment(source, match.start())

            sample = self._make_cpp_sample(
                func_name=full_name,
                func_code=full_code[:2000],  # Limit karakter
                comment=comment,
                filepath=filepath,
            )
            if sample:
                samples.append(sample)

        return samples

    def _extract_braced_block(self, source: str, open_brace_pos: int) -> Optional[str]:
        """Ekstrak konten dalam {} dengan bracket matching."""
        if open_brace_pos >= len(source) or source[open_brace_pos] != '{':
            return None

        depth  = 0
        start  = open_brace_pos
        end    = start

        for i in range(start, min(start + 5000, len(source))):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if depth != 0:
            return None

        return source[start:end + 1]

    def _extract_preceding_comment(self, source: str, pos: int) -> str:
        """Ekstrak komentar yang tepat sebelum fungsi."""
        before = source[:pos].rstrip()

        # Cari block comment /** ... */
        block_match = re.search(r'/\*\*?(.*?)\*/', before, re.DOTALL)
        if block_match and len(before) - block_match.end() < 5:
            return block_match.group(1).strip().replace(" * ", "").replace("* ", "")

        # Cari line comments // ...
        lines = before.splitlines()
        comment_lines = []
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith("//"):
                comment_lines.insert(0, stripped[2:].strip())
            elif stripped:
                break

        return " ".join(comment_lines)

    def _make_cpp_sample(
        self, func_name: str, func_code: str, comment: str, filepath: Path
    ) -> Optional[Dict]:
        """Buat training sample dari fungsi C++."""
        filename    = filepath.name
        is_moko     = "moko" in func_name.lower() or "moko" in str(filepath).lower()

        question_variants = [
            f"Implementasikan fungsi C++ `{func_name}` untuk {filename}.",
            f"Tulis implementasi C++ untuk `{func_name}`.",
            f"Buat fungsi C++ `{func_name}`.",
        ]

        if comment:
            question_variants.append(
                f"Buat fungsi C++ yang: {comment[:200]}"
            )

        if is_moko:
            question_variants.append(
                f"Bagaimana implementasi `{func_name}` di MOKO IDE/OS dalam C++?"
            )

        user_prompt = random.choice(question_variants)
        response    = f"```cpp\n{func_code}\n```"

        if comment:
            response = f"{comment}\n\n```cpp\n{func_code}\n```"

        return {
            "messages": [
                {"role": "system",    "content": MOKO_CODER_SYSTEM},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "source": "moko_os_code",
                "file":   str(filepath.name),
                "type":   "cpp_function",
                "name":   func_name,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# MOKO OS KNOWLEDGE SAMPLES — Hardcoded high-value knowledge
# ═══════════════════════════════════════════════════════════════════════════

def generate_moko_architecture_samples() -> List[Dict]:
    """
    Generate hardcoded samples tentang arsitektur MOKO OS yang penting.
    Ini knowledge yang tidak bisa diekstrak dari kode saja.
    """
    samples = [
        {
            "q": "Apa itu MOKO OS dan bagaimana arsitektur utamanya?",
            "a": """MOKO OS adalah sistem operasi berbasis Linux yang didesain khusus untuk pengembangan dengan AI. Arsitektur utamanya:

**Core Components:**
```
moko_core/
├── moko_agents/          # AI agent system
│   ├── dual_system/      # System 1 (fast) + System 2 (deliberate)
│   ├── llm_engine.py     # Interface ke llama-server
│   ├── router.py         # Intent routing
│   └── orchestrator.py   # Agent orchestration
├── moko_inference/       # LLM inference engine
│   ├── server_manager.py # llama-server lifecycle
│   └── moko_server.py    # Server wrapper
└── moko_memory/          # Persistent memory system
```

**Key Design Principles:**
1. Local-first — semua AI berjalan lokal, tidak ada cloud dependency
2. Dual-system thinking — System 1 untuk respons cepat, System 2 untuk reasoning mendalam
3. GGUF models — format optimal untuk CPU/GPU inference
4. Agent-based — setiap kemampuan diimplementasikan sebagai agent terpisah"""
        },
        {
            "q": "Bagaimana cara MOKO Orchestrator mendistribusikan task ke agent yang tepat?",
            "a": """MOKO Orchestrator menggunakan intent routing multi-layer:

```python
# Simplified flow di orchestrator.py
class MokoOrchestrator:
    def process(self, user_input: str) -> str:
        # 1. Classify intent
        intent = self.intent_router.classify(user_input)
        
        # 2. System 1: Fast response untuk intent sederhana
        if intent.confidence > 0.9 and intent.is_simple:
            return self.system1.respond(user_input)
        
        # 3. System 2: Deep thinking untuk complex tasks
        context = self.retrieval_layer.get_context(user_input)
        return self.system2.reason(user_input, context=context)
```

**Intent Categories:**
- `coding` → software_builder_agent, llm_engine
- `memory` → moko_memory, learning_manager  
- `analysis` → analyst_node, neuro_symbolic_cbc
- `math` → math_query_amplifier, moko_neuromath
- `general` → core_node → llm_engine"""
        },
        {
            "q": "Bagaimana cara menambahkan model AI baru ke MOKO OS?",
            "a": """Untuk menambahkan model baru ke MOKO OS:

**Step 1: Tambahkan model ke konfigurasi**
```python
# moko_config/settings.py
MODEL_REGISTRY = {
    "moko-coder-1b": {
        "path": "/path/to/moko-coder-1b.gguf",
        "port": 11436,
        "ctx_size": 2048,
        "n_gpu_layers": -1,  # Semua layer ke GPU
        "role": "coding",
    },
    # ... model lainnya
}
```

**Step 2: Daftarkan di model dispatcher**
```python
# moko_agents/model_dispatcher.py
class ModelDispatcher:
    def dispatch_by_role(self, role: str, prompt: str) -> str:
        if role == "coding":
            return self.call_model("moko-coder-1b", prompt)
        # ...
```

**Step 3: Start server**
```python
# moko_inference/server_manager.py
MokoLocalInferenceServer.start_servers(
    model_path="moko-coder-1b.gguf",
    port=11436,
    ctx_size=2048,
)
```"""
        },
        {
            "q": "Bagaimana MOKO IDE berkomunikasi dengan AI backend?",
            "a": """MOKO IDE (C++/Qt) berkomunikasi dengan MOKO Core (Python) via:

**1. REST API** — untuk single requests
```cpp
// code_editor.cpp
QNetworkRequest request(QUrl("http://127.0.0.1:11434/chat/completions"));
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

QJsonObject payload;
payload["messages"] = messagesArray;
payload["max_tokens"] = 512;
payload["stream"] = false;

QNetworkReply* reply = networkManager->post(request, 
    QJsonDocument(payload).toJson());
```

**2. Streaming** — untuk real-time output
```cpp
// Streaming response dari llama-server
connect(reply, &QNetworkReply::readyRead, this, [this, reply]() {
    QByteArray data = reply->readAll();
    // Parse SSE: "data: {...}\n\n"
    parseStreamingChunk(data);
});
```

**3. IPC Python** — untuk operasi kompleks yang butuh MOKO agents
```cpp
QProcess* mokoProcess = new QProcess();
mokoProcess->start("python3", {"-m", "moko_core.moko_os", "--query", prompt});
```"""
        },
        {
            "q": "Bagaimana cara kerja RAG (Retrieval-Augmented Generation) di MOKO OS?",
            "a": """MOKO RAG menggunakan model embedding kecil (moko-rag.gguf ~1GB) untuk semantic search:

```python
# moko_agents/layers/retrieval_layer.py
class RetrievalLayer:
    def get_context(self, query: str, top_k: int = 3) -> str:
        # 1. Convert query ke embedding vector
        query_embedding = self.engine.get_embedding(query)
        
        # 2. Cari dokumen paling relevan (cosine similarity)
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k,
            threshold=0.7
        )
        
        # 3. Format konteks untuk injection ke prompt
        context_parts = []
        for doc in results:
            context_parts.append(f"[Source: {doc.source}]\\n{doc.content}")
        
        return "\\n\\n".join(context_parts)
    
    def index_document(self, content: str, source: str):
        # Chunking + embedding + store
        chunks = self.chunker.split(content, chunk_size=512, overlap=50)
        for chunk in chunks:
            embedding = self.engine.get_embedding(chunk)
            self.vector_store.add(embedding, chunk, source)
```

**RAG Server**: Berjalan di port 11437, terpisah dari model utama (port 11434), 
menggunakan model ringan ~1GB agar tidak memakan VRAM berlebihan."""
        },
    ]

    result = []
    for s in samples:
        result.append({
            "messages": [
                {"role": "system",    "content": MOKO_CODER_SYSTEM},
                {"role": "user",      "content": s["q"]},
                {"role": "assistant", "content": s["a"]},
            ],
            "metadata": {
                "source": "moko_architecture",
                "type":   "knowledge",
            }
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════
# FILE SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class MokoCodeScanner:
    """Scanner utama yang mengkoordinasikan ekstraksi dari semua file."""

    def __init__(self, verbose: bool = True):
        self.verbose      = verbose
        self.py_extractor = PythonExtractor()
        self.cpp_extractor= CppExtractor()
        self.stats: Dict  = {
            "files_scanned":   0,
            "py_functions":    0,
            "py_classes":      0,
            "cpp_functions":   0,
            "total_samples":   0,
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def _should_skip(self, path: Path) -> bool:
        """Cek apakah file/direktori harus dilewati."""
        for part in path.parts:
            if part in SKIP_DIRS:
                return True
        name = path.name
        return any(skip in name for skip in SKIP_FILES)

    def scan_python_file(self, filepath: Path) -> List[Dict]:
        """Scan satu file Python."""
        if self._should_skip(filepath):
            return []
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
            if len(source.strip()) < 100:
                return []

            funcs   = self.py_extractor.extract_functions(source, filepath)
            classes = self.py_extractor.extract_classes(source, filepath)

            self.stats["py_functions"] += len(funcs)
            self.stats["py_classes"]   += len(classes)
            return funcs + classes
        except Exception as e:
            self._log(f"⚠️  Error scanning {filepath.name}: {e}")
            return []

    def scan_cpp_file(self, filepath: Path) -> List[Dict]:
        """Scan satu file C++ atau header."""
        if self._should_skip(filepath):
            return []
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
            if len(source.strip()) < 100:
                return []

            funcs = self.cpp_extractor.extract_functions(source, filepath)
            self.stats["cpp_functions"] += len(funcs)
            return funcs
        except Exception as e:
            self._log(f"⚠️  Error scanning {filepath.name}: {e}")
            return []

    def scan_all(
        self,
        include_python: bool = True,
        include_cpp: bool = True,
        max_per_file: int = 10,
    ) -> List[Dict]:
        """Scan semua direktori dan kumpulkan training samples."""
        all_samples = []

        for scan_dir in SCAN_DIRS:
            if not scan_dir.exists():
                self._log(f"⚠️  Direktori tidak ada: {scan_dir}")
                continue

            self._log(f"📂 Scanning: {scan_dir.relative_to(PROJECT_DIR)}")

            # Python files
            if include_python:
                for filepath in sorted(scan_dir.rglob("*.py")):
                    if self._should_skip(filepath):
                        continue
                    samples = self.scan_python_file(filepath)
                    if samples:
                        # Limit per file agar tidak ada file yang dominan
                        samples = samples[:max_per_file]
                        all_samples.extend(samples)
                        self.stats["files_scanned"] += 1
                        self._log(f"  ✅ {filepath.name}: {len(samples)} samples")

            # C++ files
            if include_cpp:
                for ext in ("*.cpp", "*.h", "*.hpp", "*.cc"):
                    for filepath in sorted(scan_dir.rglob(ext)):
                        if self._should_skip(filepath):
                            continue
                        samples = self.scan_cpp_file(filepath)
                        if samples:
                            samples = samples[:max_per_file]
                            all_samples.extend(samples)
                            self.stats["files_scanned"] += 1
                            self._log(f"  ✅ {filepath.name}: {len(samples)} C++ samples")

        # Tambah hardcoded MOKO architecture knowledge
        arch_samples = generate_moko_architecture_samples()
        all_samples.extend(arch_samples)
        self._log(f"\n  ✅ MOKO Architecture: {len(arch_samples)} knowledge samples added")

        self.stats["total_samples"] = len(all_samples)
        return all_samples


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def show_stats(scanner: MokoCodeScanner = None):
    """Tampilkan statistik ekstraksi."""
    print("\n" + "=" * 55)
    print("  MOKO OS Code Extractor — Stats")
    print("=" * 55)

    if scanner:
        s = scanner.stats
        print(f"\n  Files scanned:    {s['files_scanned']:>6,}")
        print(f"  Python functions: {s['py_functions']:>6,}")
        print(f"  Python classes:   {s['py_classes']:>6,}")
        print(f"  C++ functions:    {s['cpp_functions']:>6,}")
        print(f"  Total samples:    {s['total_samples']:>6,}")

    if OUTPUT_FILE.exists():
        count = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8") if _.strip())
        size  = OUTPUT_FILE.stat().st_size / 1024 / 1024
        print(f"\n  Output file: {OUTPUT_FILE.name}")
        print(f"  Samples:     {count:,}")
        print(f"  Size:        {size:.1f} MB")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak source code MOKO OS sebagai training data"
    )
    parser.add_argument("--python",    action="store_true", help="Hanya Python files")
    parser.add_argument("--cpp",       action="store_true", help="Hanya C++ files")
    parser.add_argument("--output",    type=str, default=None, help="Output file path")
    parser.add_argument("--max-per-file", type=int, default=10,
                        help="Maksimum samples per file (default: 10)")
    parser.add_argument("--no-append", action="store_true", help="Overwrite output file")
    parser.add_argument("--stats",     action="store_true", help="Tampilkan statistik")
    parser.add_argument("--quiet",     action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    print("\n" + "=" * 60)
    print("  MOKO OS Code Extractor")
    print("  Mengekstrak source code MOKO OS sebagai training data")
    print("=" * 60 + "\n")

    # Tentukan mode ekstraksi
    include_python = not args.cpp  # default include both, unless --cpp only
    include_cpp    = not args.python
    if args.python and args.cpp:  # keduanya eksplisit → include both
        include_python = include_cpp = True

    output = Path(args.output) if args.output else OUTPUT_FILE

    scanner = MokoCodeScanner(verbose=not args.quiet)
    samples = scanner.scan_all(
        include_python=include_python,
        include_cpp=include_cpp,
        max_per_file=args.max_per_file,
    )

    # Shuffle untuk distribusi yang merata
    random.shuffle(samples)

    # Simpan ke file
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if args.no_append else "w"  # Selalu overwrite karena re-scan

    with open(output, write_mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ Ekstraksi selesai!")
    show_stats(scanner)
    print(f"   Output: {output}")
    print(f"   Total:  {len(samples):,} training samples dari source code MOKO OS")


if __name__ == "__main__":
    main()
