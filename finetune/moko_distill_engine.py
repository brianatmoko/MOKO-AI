"""
MOKO Distill Engine — Teacher-Student Distillation
====================================================
Menggunakan MOKO-AI-4B yang sudah ada sebagai "guru" untuk generate
training data bagi MOKO Coder 1B "murid".

Strategi:
  1. Jalankan MOKO-AI-4B via llama-server (sudah berjalan di sistem)
  2. Kirim prompt coding yang beragam ke model 4B
  3. Simpan Q&A pairs ke JSONL sebagai dataset training
  4. Dataset ini kemudian dipakai untuk fine-tune model 1B

Keunggulan vs dataset publik:
  - Knowledge MOKO-specific (API, path, arsitektur)
  - Bahasa campuran Indonesia-Inggris sesuai style MOKO
  - Contoh coding sesuai standar MOKO OS

Usage:
  python3 moko_distill_engine.py --mode quick   # 500 samples cepat
  python3 moko_distill_engine.py --mode full    # 5000 samples lengkap
  python3 moko_distill_engine.py --mode moko    # MOKO-specific knowledge
  python3 moko_distill_engine.py --test         # Test koneksi ke model 4B
  python3 moko_distill_engine.py --status       # Cek status dataset
"""

import os
import sys
import json
import time
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator
import requests

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR  = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR  = FINETUNE_DIR / "moko_datasets"
OUTPUT_FILE  = DATASET_DIR / "moko_distill_dataset.jsonl"
LOG_DIR      = FINETUNE_DIR / "logs"

# Port llama-server yang sudah berjalan (sesuai konfigurasi MOKO)
LLAMA_SERVER_URL = "http://127.0.0.1:11434"
LLAMA_SERVER_ALT = "http://127.0.0.1:8080"


# ═══════════════════════════════════════════════════════════════════════════
# MOKO CODER IDENTITY — System prompt untuk model 1B
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
2. Code Completion — Intelligent autocomplete and continuation
3. Bug Detection & Fix — Identify and resolve errors with explanation
4. Refactoring — Improve code quality while preserving behavior
5. MOKO OS Integration — Deep knowledge of MOKO OS APIs and architecture
6. C/C++ & Python — Primary languages of MOKO OS development
7. Documentation — Generate docstrings and inline comments
8. Testing — Write unit tests and integration tests

RULES:
1. Always output COMPLETE, RUNNABLE code — never truncate
2. Use proper error handling
3. Follow MOKO OS coding conventions
4. Keep explanations concise — code first
5. Never output placeholder code (no "# TODO" without implementation)
6. Use type hints in Python
7. Include proper includes/imports"""


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES — Beragam untuk variety training data
# ═══════════════════════════════════════════════════════════════════════════

# Template untuk tiap kategori
PROMPT_CATEGORIES = {

    "python_basics": [
        "Buat fungsi Python untuk menghitung {algo} menggunakan {method}.",
        "Tulis class Python yang mengimplementasikan {data_structure} dari scratch.",
        "Buat decorator Python untuk {use_case}.",
        "Implementasikan {pattern} design pattern dalam Python.",
        "Tulis fungsi async Python untuk {async_task}.",
        "Buat generator Python yang menghasilkan {sequence}.",
        "Implementasikan context manager Python untuk {resource}.",
    ],

    "cpp_moko": [
        "Tulis kelas C++ untuk {component} dalam sistem operasi.",
        "Buat fungsi C++ yang menghandle {syscall} system call.",
        "Implementasikan {ds} data structure dalam C++ dengan memory safety.",
        "Tulis C++ code untuk {ipc} inter-process communication.",
        "Buat C++ class untuk manajemen {resource} di kernel space.",
        "Implementasikan thread-safe {pattern} dalam C++.",
    ],

    "debugging": [
        "Kode Python ini punya bug:\n```python\n{buggy_code}\n```\nTemukan dan perbaiki bug-nya.",
        "Apa yang salah dengan kode C++ ini:\n```cpp\n{buggy_cpp}\n```",
        "Kode ini crash dengan error: {error}. Bagaimana cara memperbaikinya?\n```python\n{code}\n```",
    ],

    "refactoring": [
        "Refactor kode Python ini agar lebih clean dan efisien:\n```python\n{code}\n```",
        "Optimalkan fungsi ini untuk performa lebih baik:\n```python\n{slow_code}\n```",
        "Ubah kode ini agar lebih Pythonic:\n```python\n{code}\n```",
    ],

    "moko_specific": [
        "Bagaimana cara membuat agent baru di MOKO OS yang terintegrasi dengan orchestrator?",
        "Jelaskan arsitektur dual-system MOKO OS dan cara kerjanya.",
        "Bagaimana MOKO IDE berkomunikasi dengan llama-server untuk inference?",
        "Buat komponen MOKO OS yang mengimplementasikan {feature}.",
        "Bagaimana cara menambahkan plugin baru ke MOKO IDE?",
        "Tulis kode untuk mengintegrasi model AI baru ke dalam ekosistem MOKO.",
        "Bagaimana cara kerja RAG layer di MOKO OS?",
        "Buat script untuk {task} dalam konteks MOKO OS development.",
    ],

    "algorithms": [
        "Implementasikan {algo} algorithm dengan kompleksitas waktu optimal.",
        "Buat solusi untuk masalah {problem} menggunakan dynamic programming.",
        "Implementasikan {algo} menggunakan {lang} dengan penjelasan complexity.",
        "Tulis binary search tree lengkap dengan insert, delete, dan search.",
        "Implementasikan graph algorithm untuk {graph_problem}.",
    ],

    "api_design": [
        "Desain REST API untuk {system} dengan endpoint yang lengkap.",
        "Buat FastAPI server dengan authentication dan rate limiting.",
        "Implementasikan WebSocket server untuk {realtime_feature}.",
        "Tulis Python client untuk mengakses {service} API.",
    ],

    "testing": [
        "Tulis unit tests lengkap untuk fungsi {function_desc}.",
        "Buat integration test untuk {component}.",
        "Implementasikan pytest fixtures untuk {scenario}.",
        "Tulis mock objects untuk {dependency}.",
    ],

    "system_programming": [
        "Tulis C program yang menggunakan {syscall} untuk {task}.",
        "Implementasikan memory pool allocator dalam C.",
        "Buat file I/O handler yang aman dengan error handling penuh.",
        "Implementasikan signal handler untuk {signal} dalam C.",
        "Tulis kode untuk manajemen proses menggunakan fork() dan exec().",
    ],
}

# Variable substitutions untuk template
TEMPLATE_VARS = {
    "algo": ["factorial", "fibonacci", "quicksort", "merge sort", "binary search",
             "BFS", "DFS", "dijkstra", "A* pathfinding", "KMP string matching"],
    "method": ["iteration", "recursion", "memoization", "dynamic programming", "divide and conquer"],
    "data_structure": ["linked list", "stack", "queue", "hash map", "binary tree",
                       "heap", "trie", "graph adjacency list", "circular buffer"],
    "use_case": ["timing function execution", "caching return values (memoize)",
                 "retry on exception", "rate limiting", "logging function calls",
                 "validation input parameters"],
    "pattern": ["Singleton", "Factory", "Observer", "Command", "Strategy",
                "Builder", "Decorator", "Proxy", "Iterator"],
    "async_task": ["fetching multiple URLs concurrently", "processing a queue of tasks",
                   "reading multiple files simultaneously", "polling an API endpoint"],
    "sequence": ["prime numbers", "Fibonacci numbers", "powers of 2",
                 "permutations of a list", "infinite counter with reset"],
    "resource": ["database connections", "file handles", "network sockets",
                 "temporary directories", "GPU memory"],
    "component": ["process scheduler", "memory manager", "file system driver",
                  "network interface", "device driver", "IPC handler"],
    "syscall": ["read/write", "mmap", "fork/exec", "socket", "ioctl", "epoll"],
    "ds": ["lock-free queue", "ring buffer", "hash table", "red-black tree"],
    "ipc": ["shared memory", "message queue", "pipe", "Unix socket", "semaphore"],
    "graph_problem": ["shortest path", "minimum spanning tree", "topological sort",
                      "cycle detection", "strongly connected components"],
    "problem": ["knapsack", "longest common subsequence", "coin change",
                "matrix chain multiplication", "edit distance"],
    "lang": ["Python", "C++", "Go"],
    "system": ["user management", "inventory tracking", "task management",
               "real-time monitoring", "file metadata"],
    "realtime_feature": ["live chat", "real-time notifications", "collaborative editing",
                         "live system metrics streaming"],
    "service": ["GitHub", "OpenAI", "weather data", "stock prices"],
    "function_desc": ["sorting a list of dictionaries by multiple keys",
                      "parsing and validating JSON config files",
                      "concurrent file downloader with retry logic"],
    "scenario": ["database with test data", "mock file system", "fake HTTP server"],
    "dependency": ["database connection", "external API client", "file system"],
    "signal": ["SIGTERM", "SIGINT", "SIGUSR1"],
    "feature": ["hot-reload plugin system", "sandboxed code execution",
                "real-time syntax checking", "multi-cursor editing"],
    "task": ["benchmarking model inference speed", "batch processing dataset files",
             "automating build pipeline", "monitoring system resources"],
    "ipc2": ["shared memory", "pipes", "sockets"],
}

# Buggy code samples untuk debugging exercises
BUGGY_PYTHON_SAMPLES = [
    {
        "code": """def binary_search(arr, target):
    left, right = 0, len(arr)  # Bug: harus len(arr) - 1
    while left < right:  # Bug: harus left <= right
        mid = left + right // 2  # Bug: preseden operator salah
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # Bug: harus mid + 1 (infinite loop)
        else:
            right = mid  # Bug: harus mid - 1
    return -1""",
        "error": "Infinite loop dan hasil tidak akurat"
    },
    {
        "code": """class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()  # Bug: tidak cek empty
    
    def peek(self):
        return self.items[-1]  # Bug: tidak cek empty
    
    def is_empty(self):
        return len(self.items) = 0  # Bug: SyntaxError, = bukan ==
    
    def size(self):
        return len(self.items)""",
        "error": "SyntaxError dan tidak ada empty check"
    },
    {
        "code": """def flatten_list(nested):
    result = []
    for item in nested:
        if type(item) == list:  # Bug: harus isinstance
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Bug: tidak handle tuple, dict, atau nested structures lain""",
        "error": "Tidak robust untuk berbagai tipe data"
    },
]

SLOW_CODE_SAMPLES = [
    """# Mencari angka yang muncul lebih dari sekali - O(n²)
def find_duplicates(nums):
    duplicates = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and nums[i] not in duplicates:
                duplicates.append(nums[i])
    return duplicates""",

    """# String concatenation dalam loop - sangat lambat
def build_csv(data):
    result = ""
    for row in data:
        line = ""
        for i, val in enumerate(row):
            if i > 0:
                line = line + ","
            line = line + str(val)
        result = result + line + "\\n"
    return result""",
]


# ═══════════════════════════════════════════════════════════════════════════
# DISTILLATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MokoDistillEngine:
    """Engine untuk distillasi knowledge dari model 4B ke dataset training 1B."""

    def __init__(self, server_url: str = None, verbose: bool = True):
        self.verbose    = verbose
        self.server_url = server_url or self._detect_server()
        self.generated  = 0
        self.failed     = 0
        self._seen_hashes: set = set()  # Deduplication

    def _detect_server(self) -> str:
        """Deteksi otomatis port llama-server yang aktif."""
        candidates = [
            "http://127.0.0.1:11434",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:11435",
            "http://127.0.0.1:8081",
        ]
        for url in candidates:
            try:
                r = requests.get(f"{url}/health", timeout=2)
                if r.status_code in (200, 404):  # 404 = server ada tapi endpoint beda
                    self._log(f"Server terdeteksi di: {url}")
                    return url
            except Exception:
                continue

        # Coba endpoint /v1/models
        for url in candidates:
            try:
                r = requests.get(f"{url}/v1/models", timeout=2)
                if r.status_code == 200:
                    self._log(f"Server terdeteksi di: {url}")
                    return url
            except Exception:
                continue

        self._log("⚠️  Tidak bisa deteksi server otomatis, pakai default 11434", "WARNING")
        return LLAMA_SERVER_URL

    def _log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def test_connection(self) -> bool:
        """Test koneksi ke llama-server."""
        self._log(f"Testing connection ke {self.server_url}...")
        try:
            # Test /health
            try:
                r = requests.get(f"{self.server_url}/health", timeout=5)
                if r.status_code == 200:
                    self._log("✅ Server ONLINE (/health)")
                    return True
            except Exception:
                pass

            # Test dengan simple completion
            payload = {
                "messages": [
                    {"role": "user", "content": "Say: MOKO OK"}
                ],
                "max_tokens": 10,
                "temperature": 0.0,
            }
            r = requests.post(
                f"{self.server_url}/chat/completions",
                json=payload, timeout=15
            )
            if r.status_code == 200:
                resp = r.json()
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                self._log(f"✅ Server ONLINE — response: '{content[:50]}'")
                return True
            else:
                self._log(f"❌ Server error: HTTP {r.status_code}", "ERROR")
                return False
        except requests.exceptions.ConnectionError:
            self._log(f"❌ Tidak bisa connect ke {self.server_url}", "ERROR")
            self._log("   Pastikan MOKO AI server sudah jalan (tombol ▶ START di Status Panel)", "ERROR")
            return False
        except Exception as e:
            self._log(f"❌ Error: {e}", "ERROR")
            return False

    def _build_prompt(self, template: str) -> str:
        """Isi variabel template dengan nilai acak."""
        import re
        result = template
        for var, values in TEMPLATE_VARS.items():
            placeholder = "{" + var + "}"
            if placeholder in result:
                result = result.replace(placeholder, random.choice(values))
        return result

    def _call_teacher(self, user_prompt: str, max_tokens: int = 600,
                      temperature: float = 0.7) -> Optional[str]:
        """Panggil model 4B (guru) untuk generate respons."""
        # System prompt yang memandu model 4B untuk mengajar model 1B
        teacher_system = """You are an expert software engineer and coding instructor.
Generate clear, complete, well-documented code solutions.
- Always include complete, runnable code
- Add brief inline comments for non-obvious parts
- Include type hints in Python
- Handle edge cases properly
- Keep explanations concise but informative
- Use English for code, comments can be in English or Indonesian"""

        payload = {
            "messages": [
                {"role": "system", "content": teacher_system},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            r = requests.post(
                f"{self.server_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            if r.status_code == 200:
                choices = r.json().get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            elif r.status_code == 503:
                self._log("Model sedang loading... tunggu 10 detik", "WARNING")
                time.sleep(10)
                return None
            else:
                self._log(f"HTTP {r.status_code}: {r.text[:100]}", "WARNING")
                return None
        except requests.exceptions.Timeout:
            self._log("Timeout — model mungkin sibuk", "WARNING")
            return None
        except Exception as e:
            self._log(f"Error calling teacher: {e}", "WARNING")
            return None

    def _is_quality_response(self, response: str) -> bool:
        """Filter respons berkualitas rendah."""
        if not response:
            return False
        if len(response.strip()) < 50:
            return False
        # Harus ada kode (code block atau keyword kode)
        code_indicators = ["```", "def ", "class ", "function ", "import ",
                           "const ", "let ", "var ", "int ", "void ", "#include"]
        if not any(ind in response for ind in code_indicators):
            # Boleh jika itu penjelasan konsep MOKO-specific
            moko_keywords = ["MOKO", "orchestrator", "agent", "llama-server",
                             "inference", "RAG", "plugin"]
            if not any(kw in response for kw in moko_keywords):
                return False
        # Junk filters
        junk = ["I cannot", "I can't help", "As an AI language model",
                "I don't have access", "I'm unable to"]
        if any(j in response for j in junk):
            return False
        return True

    def _dedup_check(self, prompt: str) -> bool:
        """True jika prompt belum pernah dipakai (unique)."""
        key = hashlib.md5(prompt.strip().lower().encode()).hexdigest()
        if key in self._seen_hashes:
            return False
        self._seen_hashes.add(key)
        return True

    def _make_sample(self, user_prompt: str, assistant_response: str) -> dict:
        """Buat satu training sample dalam format ChatML."""
        return {
            "messages": [
                {"role": "system",    "content": MOKO_CODER_SYSTEM},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }

    def generate_batch(
        self,
        category: str,
        count: int,
        temperature: float = 0.7,
        max_tokens: int = 600,
    ) -> List[dict]:
        """Generate 'count' samples dari kategori tertentu."""
        samples = []
        templates = PROMPT_CATEGORIES.get(category, [])

        if not templates:
            self._log(f"Kategori tidak dikenal: {category}", "WARNING")
            return []

        attempts = 0
        max_attempts = count * 3  # Toleransi failure

        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)

            # Handle kategori debugging/refactoring yang butuh code samples
            if category == "debugging" and "{buggy_code}" in template:
                buggy = random.choice(BUGGY_PYTHON_SAMPLES)
                user_prompt = template.replace("{buggy_code}", buggy["code"])\
                                      .replace("{error}", buggy["error"])\
                                      .replace("{code}", buggy["code"])
            elif category == "refactoring" and ("{code}" in template or "{slow_code}" in template):
                code = random.choice(SLOW_CODE_SAMPLES)
                user_prompt = template.replace("{slow_code}", code).replace("{code}", code)
            else:
                user_prompt = self._build_prompt(template)

            # Dedup check
            if not self._dedup_check(user_prompt):
                continue

            # Call teacher model
            response = self._call_teacher(user_prompt, max_tokens=max_tokens,
                                          temperature=temperature)

            if response and self._is_quality_response(response):
                samples.append(self._make_sample(user_prompt, response))
                self.generated += 1
                if self.verbose and len(samples) % 10 == 0:
                    self._log(f"  {category}: {len(samples)}/{count} samples")
            else:
                self.failed += 1
                time.sleep(0.5)  # Beri jeda sebelum retry

        return samples

    def generate_moko_knowledge(self, count: int = 100) -> List[dict]:
        """
        Generate MOKO-specific knowledge samples — ini yang paling berharga.
        Mencakup arsitektur, API, dan cara kerja MOKO OS/IDE.
        """
        moko_questions = [
            # Architecture
            "Jelaskan arsitektur dual-system di MOKO OS: bagaimana System 1 (fast) dan System 2 (deliberate) bekerja bersama?",
            "Bagaimana cara kerja MOKO Agent Orchestrator? Jelaskan flow dari user input sampai respons.",
            "Apa peran masing-masing node dalam MOKO brain: prefrontal, amygdala, insula, cerebellum?",
            "Jelaskan bagaimana MOKO RAG (Retrieval-Augmented Generation) bekerja untuk menambah konteks.",
            "Bagaimana MOKO IDE berkomunikasi dengan MOKO Core via IPC atau REST API?",

            # Development
            "Bagaimana cara membuat custom agent baru untuk MOKO OS? Berikan template kodenya.",
            "Buat plugin MOKO IDE untuk {feature} dengan interface yang benar.",
            "Bagaimana cara menambahkan model AI baru ke MOKO model dispatcher?",
            "Tulis unit test untuk MOKO agent menggunakan framework test yang ada.",
            "Bagaimana cara kerja MOKO super learning untuk continuous improvement?",

            # C++ MOKO IDE
            "Buat komponen C++ untuk MOKO IDE yang mengimplementasikan {component}.",
            "Bagaimana cara menambahkan syntax highlighting untuk bahasa baru di MOKO IDE?",
            "Implementasikan code completion provider di MOKO IDE menggunakan LSP.",
            "Tulis C++ code untuk MOKO IDE window management menggunakan Qt framework.",

            # Integration
            "Bagaimana MOKO OS menghandle multiple AI model secara concurrent?",
            "Tulis script Python untuk benchmark kecepatan inference model di MOKO.",
            "Bagaimana cara menggunakan MOKO RAG untuk mencari informasi dari codebase?",
            "Implementasikan MOKO tool baru yang mengintegrasikan external API.",

            # Troubleshooting
            "Bagaimana cara debug jika llama-server tidak bisa start di MOKO OS?",
            "Apa yang harus dilakukan jika MOKO agent tidak bisa connect ke inference server?",
        ]

        samples = []
        random.shuffle(moko_questions)
        questions = (moko_questions * ((count // len(moko_questions)) + 2))[:count * 2]

        for q in questions:
            if len(samples) >= count:
                break

            # Isi variabel template
            user_prompt = self._build_prompt(q)
            if not self._dedup_check(user_prompt):
                continue

            # MOKO questions butuh konteks lebih panjang
            response = self._call_teacher(
                user_prompt,
                max_tokens=800,
                temperature=0.5,  # Lebih deterministik untuk knowledge
            )

            if response and self._is_quality_response(response):
                samples.append(self._make_sample(user_prompt, response))
                self.generated += 1
                self._log(f"  MOKO knowledge: {len(samples)}/{count}")
            else:
                self.failed += 1
                time.sleep(1)

        return samples

    def run(
        self,
        mode: str = "quick",
        output_file: Path = None,
        append: bool = True,
    ) -> int:
        """
        Jalankan distillasi lengkap.

        mode:
          "quick"  — ~500 samples, semua kategori
          "full"   — ~5000 samples, distribusi merata
          "moko"   — fokus MOKO-specific knowledge
          "coding" — fokus general coding skills
        """
        output_file = output_file or OUTPUT_FILE
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        # Test koneksi dulu
        if not self.test_connection():
            self._log("❌ Tidak bisa connect ke model 4B. Jalankan MOKO AI dulu!", "ERROR")
            return 0

        # Konfigurasi per mode
        configs = {
            "quick": {
                "python_basics":   30,
                "cpp_moko":        20,
                "debugging":       20,
                "refactoring":     15,
                "moko_specific":   50,  # Prioritas MOKO knowledge
                "algorithms":      30,
                "api_design":      20,
                "testing":         15,
                "system_programming": 15,
                "moko_knowledge":  100,  # Dedicated MOKO generation
            },
            "full": {
                "python_basics":   300,
                "cpp_moko":        200,
                "debugging":       200,
                "refactoring":     150,
                "moko_specific":   300,
                "algorithms":      300,
                "api_design":      150,
                "testing":         150,
                "system_programming": 200,
                "moko_knowledge":  500,
            },
            "moko": {
                "moko_specific":   200,
                "moko_knowledge":  300,
                "cpp_moko":        100,
            },
            "coding": {
                "python_basics":   200,
                "algorithms":      200,
                "debugging":       150,
                "refactoring":     150,
                "api_design":      100,
                "testing":         100,
                "system_programming": 100,
            },
        }

        plan = configs.get(mode, configs["quick"])
        total_target = sum(plan.values())
        self._log(f"Mode: {mode} — Target: {total_target} samples")
        self._log(f"Output: {output_file}")
        self._log("=" * 60)

        # Load existing hashes jika append mode
        existing_count = 0
        if append and output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            s = json.loads(line)
                            msgs = s.get("messages", [])
                            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                            self._dedup_check(user_msg)  # Masukkan ke seen set
                            existing_count += 1
                        except Exception:
                            pass
            self._log(f"Resuming: {existing_count} samples sudah ada")

        # Buka file untuk append
        write_mode = "a" if (append and output_file.exists()) else "w"
        all_new = 0

        with open(output_file, write_mode, encoding="utf-8") as f:
            for category, count in plan.items():
                self._log(f"\n▶ Generating [{category}]: {count} samples...")

                if category == "moko_knowledge":
                    samples = self.generate_moko_knowledge(count)
                else:
                    samples = self.generate_batch(
                        category=category,
                        count=count,
                        temperature=0.7 if category not in ("moko_specific",) else 0.5,
                    )

                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    f.flush()
                    all_new += 1

                self._log(f"  ✅ [{category}]: {len(samples)} samples saved")
                time.sleep(0.5)  # Beri napas ke server

        total = existing_count + all_new
        self._log("\n" + "=" * 60)
        self._log(f"✅ Distillasi selesai!")
        self._log(f"   Generated:  {self.generated}")
        self._log(f"   Failed:     {self.failed}")
        self._log(f"   New saved:  {all_new}")
        self._log(f"   Total file: {total} samples")
        self._log(f"   Output:     {output_file}")
        self._log("=" * 60)

        return all_new


# ═══════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════

def show_status():
    """Tampilkan status dataset distillasi."""
    print("\n" + "=" * 55)
    print("  MOKO Distill Engine — Status")
    print("=" * 55)

    files_to_check = [
        OUTPUT_FILE,
        DATASET_DIR / "moko_coder_dataset.jsonl",
        DATASET_DIR / "moko_coder_hex.jsonl",
        PROJECT_DIR / "distill_dataset" / "moko_distill_samples.jsonl",
    ]

    print("\n📊 Dataset Files:")
    for f in files_to_check:
        if f.exists():
            count = sum(1 for line in open(f, encoding="utf-8") if line.strip())
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  ✅ {f.name:45s} {count:>6,} samples  {size_mb:>6.1f} MB")
        else:
            print(f"  ❌ {f.name}")

    print()

    # Check server
    print("🤖 Teacher Model (MOKO-AI-4B):")
    engine = MokoDistillEngine(verbose=False)
    if engine.test_connection():
        print(f"  ✅ Server ONLINE di {engine.server_url}")
    else:
        print(f"  ❌ Server OFFLINE — jalankan MOKO AI dulu!")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MOKO Distill Engine — Generate training data dari model 4B"
    )
    parser.add_argument(
        "--mode", choices=["quick", "full", "moko", "coding"],
        default="quick",
        help="Mode distillasi (default: quick ~500 samples)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path output file JSONL (default: moko_datasets/moko_distill_dataset.jsonl)"
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="URL llama-server (default: auto-detect)"
    )
    parser.add_argument(
        "--no-append", action="store_true",
        help="Overwrite file lama (default: append)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test koneksi ke model 4B saja"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Tampilkan status dataset"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override jumlah sample per kategori"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print("\n" + "=" * 60)
    print("  MOKO Distill Engine")
    print("  Teacher: MOKO-AI-4B  →  Student Dataset: MOKO Coder 1B")
    print("=" * 60 + "\n")

    engine = MokoDistillEngine(
        server_url=args.server,
        verbose=True,
    )

    if args.test:
        ok = engine.test_connection()
        sys.exit(0 if ok else 1)

    output = Path(args.output) if args.output else None
    count  = engine.run(
        mode=args.mode,
        output_file=output,
        append=not args.no_append,
    )

    if count > 0:
        print(f"\n✅ Berhasil generate {count} samples baru!")
        print(f"   Gunakan dataset ini untuk training:")
        print(f"   python3 finetune/moko_trainer_v2.py --dataset distill")
    else:
        print("\n❌ Tidak ada sample yang berhasil di-generate.")
        print("   Pastikan MOKO AI server sudah running!")


if __name__ == "__main__":
    main()
