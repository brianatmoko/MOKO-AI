"""
MOKO Mathematical Query Amplifier (MQA)
========================================
Berdasarkan riset:
  - SAAS: Solving Ability Amplification Strategy (Upstage AI, 2024)
  - SCoT: Structured Chain-of-Thought for Code Generation (2024-2025)
  - PoT: Program-of-Thought Paradigm

Filosofi:
  Input manusia = informasi rendah densitas
  Output MQA   = representasi matematis tinggi densitas

Proses:
  1. Classify domain matematika dari input
  2. Inject definisi formal domain tersebut
  3. Ekstrak boundary conditions dan error cases
  4. Buat SCoT template (Sequential → Branch → Loop)
  5. Auto-generate verification target

Hasilnya adalah "prompt yang diperkuat" yang dikirim ke LLM sebagai ganti
input mentah, sehingga bahkan model kecil 4B bisa menghasilkan output
dengan keakuratan matematis yang jauh lebih tinggi.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN TAXONOMY
# Setiap domain memiliki: definisi formal, boundary conditions, oracle hints
# ─────────────────────────────────────────────────────────────────────────────

class MathDomain(Enum):
    STATISTICS      = "statistics"
    LINEAR_ALGEBRA  = "linear_algebra"
    CALCULUS        = "calculus"
    NUMBER_THEORY   = "number_theory"
    GRAPH_THEORY    = "graph_theory"
    COMBINATORICS   = "combinatorics"
    GEOMETRY        = "geometry"
    CRYPTOGRAPHY    = "cryptography"
    SORTING         = "sorting"
    DYNAMIC_PROG    = "dynamic_programming"
    STRING_PROC     = "string_processing"
    DATA_STRUCTURE  = "data_structure"
    PROBABILITY     = "probability"
    OPTIMIZATION    = "optimization"
    SPATIAL_GRAPHICS = "spatial_graphics"
    TENSOR_MANIFOLD  = "tensor_manifold"
    GENERAL_CODING  = "general_coding"


# Domain signatures: kata kunci yang mengindikasikan domain
DOMAIN_SIGNATURES: Dict[MathDomain, List[str]] = {
    MathDomain.STATISTICS: [
        "rata-rata", "mean", "median", "modus", "mode", "variance", "variansi",
        "standar deviasi", "standard deviation", "distribusi", "distribution",
        "regresi", "regression", "korelasi", "correlation", "histogram",
        "weighted", "berbobot", "statistik", "statistic", "probabilitas rata-rata"
    ],
    MathDomain.LINEAR_ALGEBRA: [
        "matriks", "matrix", "vektor", "vector", "determinan", "determinant",
        "eigenvalue", "eigenvector", "invers", "transpose", "dot product",
        "cross product", "norm", "orthogonal", "linear transformation",
        "sistem persamaan linear", "gaussian elimination"
    ],
    MathDomain.CALCULUS: [
        "turunan", "derivative", "integral", "limit", "diferensial",
        "differential", "gradien", "gradient", "jacobian", "hessian",
        "taylor series", "fourier", "laplace", "ode", "pde"
    ],
    MathDomain.NUMBER_THEORY: [
        "prima", "prime", "bilangan, faktor", "gcd", "lcm", "fpb", "kpk",
        "modular", "kongruensi", "congruence", "fibonacci", "pangkat",
        "power", "akar", "root", "logaritma", "logarithm", "faktorial", "factorial"
    ],
    MathDomain.GRAPH_THEORY: [
        "graf", "graph", "node", "edge", "vertex", "path", "cycle",
        "dijkstra", "bfs", "dfs", "tree", "pohon", "connected", "weighted graph",
        "shortest path", "minimum spanning tree", "topological sort"
    ],
    MathDomain.SORTING: [
        "urutkan", "sort", "sorting", "bubble sort", "merge sort", "quick sort",
        "heap sort", "insertion sort", "selection sort", "order", "ascending",
        "descending", "komparator", "comparator"
    ],
    MathDomain.DYNAMIC_PROG: [
        "dynamic programming", "dp", "memoization", "tabulation",
        "knapsack", "longest common subsequence", "lcs", "fibonacci dp",
        "optimal substructure", "overlapping subproblems", "coin change"
    ],
    MathDomain.STRING_PROC: [
        "string", "substring", "palindrome", "anagram", "regex", "pattern matching",
        "kmp", "rabin karp", "suffix array", "trie", "edit distance",
        "levenshtein", "tokenize", "parse string"
    ],
    MathDomain.PROBABILITY: [
        "probabilitas", "probability", "peluang", "chance", "bayesian",
        "prior", "posterior", "likelihood", "expected value", "monte carlo",
        "random", "sampling", "hypothesis test"
    ],
    MathDomain.OPTIMIZATION: [
        "optimasi", "optimization", "minimize", "maximize", "linear programming",
        "gradient descent", "convex", "local minimum", "global minimum",
        "objective function", "constraint"
    ],
    MathDomain.GEOMETRY: [
        "geometri", "geometry", "titik", "point", "garis", "line",
        "lingkaran", "circle", "segitiga", "triangle", "luas", "area",
        "keliling", "perimeter", "jarak", "distance", "euclidean"
    ],
    MathDomain.CRYPTOGRAPHY: [
        "enkripsi", "encryption", "dekripsi", "decryption", "hash",
        "rsa", "aes", "cipher", "kunci", "key", "digital signature",
        "sha", "md5", "xor encryption"
    ],
    MathDomain.LINEAR_ALGEBRA: [
        "matriks", "matrix", "vektor", "vector"
    ],
    MathDomain.DATA_STRUCTURE: [
        "linked list", "stack", "queue", "heap", "binary tree", "bst",
        "hash table", "dictionary", "set", "deque", "priority queue"
    ],
    MathDomain.SPATIAL_GRAPHICS: [
        "3d", "prototipe 3d", "mekanisme 3d", "kamera", "camera", "camera move",
        "rotasi", "quaternion", "matrix4x4", "webgl", "three.js", "canvas 2d",
        "game web", "web game", "raycast", "spline", "bezier", "kinematika",
        "fisika 3d", "transformasi", "game engine", "rendering"
    ],
    MathDomain.TENSOR_MANIFOLD: [
        "tensor", "manifold", "lie group", "lie algebra", "so(3)", "se(3)",
        "rodrigues", "lagrangian", "euler-lagrange", "hamiltonian", "gaussian curvature",
        "mean curvature", "laplace-beltrami", "diferensial geometri", "geometri diferensial",
        "super rumit", "super kompleks", "matematika murni", "level tengah"
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# FORMAL DEFINITIONS LIBRARY
# Definisi matematis formal per domain — ini yang membedakan MOKO dengan
# IDE biasa. Ini adalah "matematika yang diinjeksikan ke dalam prompt"
# ─────────────────────────────────────────────────────────────────────────────

FORMAL_DEFINITIONS: Dict[MathDomain, Dict] = {
    MathDomain.STATISTICS: {
        "definitions": [
            "Mean: μ = (1/n) Σᵢ xᵢ",
            "Weighted Mean: W = Σ(wᵢ·xᵢ) / Σwᵢ",
            "Variance: σ² = (1/n) Σᵢ (xᵢ - μ)²",
            "Standard Deviation: σ = √σ²",
            "Median: nilai tengah setelah diurutkan",
            "Mode: nilai yang paling sering muncul",
        ],
        "boundary_conditions": [
            "n ≠ 0 (input tidak boleh kosong)",
            "Σwᵢ ≠ 0 untuk weighted mean",
            "wᵢ ≥ 0 untuk semua bobot",
            "Semua nilai harus numerik (float atau int)",
        ],
        "error_cases": [
            "ZeroDivisionError jika input kosong",
            "ZeroDivisionError jika jumlah bobot = 0",
            "TypeError jika ada nilai non-numerik",
            "ValueError jika panjang weights ≠ panjang values",
        ],
        "oracle_code": """
import statistics
# Oracle untuk verifikasi:
# statistics.mean([1,2,3]) == 2.0
# statistics.variance([1,2,3]) ≈ 1.0
""",
    },
    MathDomain.SORTING: {
        "definitions": [
            "Sorting: penyusunan elemen dalam urutan tertentu",
            "Comparison-based sort lower bound: Ω(n log n)",
            "Stable sort: mempertahankan urutan relatif elemen sama",
            "In-place sort: O(1) extra space",
            "Merge Sort: T(n) = 2T(n/2) + O(n) → O(n log n)",
            "Quick Sort average: O(n log n), worst: O(n²)",
        ],
        "boundary_conditions": [
            "Input list bisa kosong [] → return []",
            "Input list satu elemen → return as-is",
            "Elemen bisa duplikat",
            "Elemen harus comparable (mendukung operator < dan >)",
        ],
        "error_cases": [
            "TypeError jika elemen tidak comparable",
            "Rekursi infinite jika pivot selection buruk (quick sort)",
        ],
        "oracle_code": """
# Oracle: sorted() Python sebagai ground truth
# assert sorted([3,1,2]) == [1,2,3]
# assert sorted([]) == []
""",
    },
    MathDomain.DYNAMIC_PROG: {
        "definitions": [
            "DP: dekomposisi masalah ke submasalah yang overlapping",
            "Optimal Substructure: solusi optimal terdiri dari solusi submasalah optimal",
            "Memoization (top-down): cache hasil rekursi",
            "Tabulation (bottom-up): isi tabel dari base case ke atas",
            "Knapsack: f(i,w) = max(f(i-1,w), vᵢ + f(i-1,w-wᵢ))",
            "LCS: dp[i][j] = dp[i-1][j-1]+1 jika a[i]==b[j]",
        ],
        "boundary_conditions": [
            "Base case harus jelas didefinisikan",
            "State space harus finite",
            "Subproblem graph harus acyclic (DAG)",
        ],
        "error_cases": [
            "RecursionError jika base case salah (tanpa memoization)",
            "MemoryError jika state space terlalu besar",
            "KeyError jika cache diakses sebelum diisi",
        ],
        "oracle_code": """
# Oracle contoh LCS:
# assert lcs("ABCBDAB", "BDCAB") == 4
# assert lcs("", "ABC") == 0
""",
    },
    MathDomain.NUMBER_THEORY: {
        "definitions": [
            "GCD: Algoritma Euclidean — gcd(a,b) = gcd(b, a mod b)",
            "LCM: lcm(a,b) = |a·b| / gcd(a,b)",
            "Prima: bilangan > 1 yang hanya habis dibagi 1 dan dirinya",
            "Sieve of Eratosthenes: O(n log log n) untuk semua prima ≤ n",
            "Modular Arithmetic: (a+b) mod m = ((a mod m) + (b mod m)) mod m",
            "Euler's Totient: φ(n) = n · Π(1 - 1/p) untuk p | n",
        ],
        "boundary_conditions": [
            "gcd(0, x) = gcd(x, 0) = x",
            "lcm(0, x) = 0",
            "Prima selalu > 1 (1 bukan prima, 2 prima terkecil)",
            "Modulus m ≠ 0",
        ],
        "error_cases": [
            "ZeroDivisionError pada lcm jika salah satu = 0",
            "ValueError jika input negatif untuk beberapa fungsi",
        ],
        "oracle_code": """
import math
# Oracle: math.gcd(48, 18) == 6
# Oracle: math.lcm(4, 6) == 12
""",
    },
    MathDomain.GRAPH_THEORY: {
        "definitions": [
            "Graf G = (V, E), V = vertices, E = edges",
            "BFS: kunjungi level demi level, kompleksitas O(V+E)",
            "DFS: kunjungi sedalam mungkin, kompleksitas O(V+E)",
            "Dijkstra: single-source shortest path, O((V+E) log V) dengan priority queue",
            "Minimum Spanning Tree: subset E dengan total bobot minimum",
        ],
        "boundary_conditions": [
            "Graf bisa kosong (V=0)",
            "Bisa ada self-loop atau multi-edge",
            "Dijkstra hanya valid untuk bobot non-negatif",
            "MST hanya ada di graf connected",
        ],
        "error_cases": [
            "Infinite loop jika deteksi cycle diabaikan di DFS",
            "KeyError jika node tidak ada di adjacency list",
        ],
        "oracle_code": """
# Oracle Dijkstra: shortest path dari 0 ke semua node di weighted graph
# Verifikasi: jarak harus ≥ 0 dan ≤ INF
""",
    },
    MathDomain.LINEAR_ALGEBRA: {
        "definitions": [
            "Matriks A berukuran m×n: m baris, n kolom",
            "Perkalian matriks C = A·B: C[i][j] = Σₖ A[i][k]·B[k][j]",
            "Syarat perkalian: kolom A = baris B",
            "Transpos Aᵀ[i][j] = A[j][i]",
            "Determinan 2×2: det([[a,b],[c,d]]) = ad - bc",
            "Invers: A·A⁻¹ = I, hanya ada jika det(A) ≠ 0",
        ],
        "boundary_conditions": [
            "Kolom A harus sama dengan baris B untuk perkalian",
            "det(A) ≠ 0 untuk invers",
            "Matriks persegi untuk determinan dan invers",
        ],
        "error_cases": [
            "ValueError: dimensi tidak kompatibel untuk perkalian",
            "LinAlgError: matriks singular (det=0) untuk invers",
        ],
        "oracle_code": """
import numpy as np
# Oracle: np.dot(A, B) sebagai ground truth
# Oracle: np.linalg.det(A) untuk determinan
""",
    },
    MathDomain.STRING_PROC: {
        "definitions": [
            "Palindrome: s = reverse(s)",
            "Anagram: semua karakter sama, urutan berbeda",
            "Edit Distance (Levenshtein): min operasi insert/delete/replace",
            "KMP: pattern matching O(n+m) — precompute failure function",
            "Trie: prefix tree O(m) search, m = panjang string",
        ],
        "boundary_conditions": [
            "Empty string '' adalah palindrome",
            "Single char selalu palindrome",
            "Case sensitivity harus eksplisit didefinisikan",
        ],
        "error_cases": [
            "TypeError jika input bukan string",
            "IndexError jika out-of-bounds pada manual indexing",
        ],
        "oracle_code": """
# Oracle palindrome: s == s[::-1]
# Oracle anagram: sorted(s1) == sorted(s2)
""",
    },
    MathDomain.GENERAL_CODING: {
        "definitions": [
            "Fungsi harus memiliki tipe input/output yang jelas",
            "Single Responsibility: satu fungsi, satu tanggung jawab",
            "DRY: Don't Repeat Yourself",
        ],
        "boundary_conditions": [
            "Handle None/null input",
            "Handle empty collections",
            "Handle overflow untuk integer besar",
        ],
        "error_cases": [
            "TypeError: tipe input salah",
            "ValueError: nilai di luar range valid",
            "IndexError: akses index di luar bounds",
        ],
        "oracle_code": "# Verifikasi dengan unit test assertion",
    },
    MathDomain.SPATIAL_GRAPHICS: {
        "definitions": [
            "3D Vector Normalization: v_hat = v / ||v||",
            "Quaternion Rotation: q = (cos(θ/2), u_x*sin(θ/2), u_y*sin(θ/2), u_z*sin(θ/2)) — membebas Gimbal Lock",
            "Quaternion SLERP: Slerp(q1, q2, t) = (sin((1-t)θ)/sin(θ))q1 + (sin(tθ)/sin(θ))q2 — interpolasi mulus kamera 3D",
            "Camera LookAt Matrix: Z_axis = normalize(eye - target), X_axis = normalize(cross(up, Z)), Y_axis = cross(Z, X)",
            "Raycasting Möller–Trumbore: e1 = v1-v0, e2 = v2-v0, h = cross(dir, e2), a = dot(e1, h) — deteksi tabrakan 3D real-time",
            "Cubic Bezier Spline: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃ — pergerakan kamera & jalur mekanis mulus",
        ],
        "boundary_conditions": [
            "Normalisasi vektor tidak boleh membagi dengan nol (||v|| > 1e-7)",
            "Quaternion harus selalu dinormalisasi (||q|| = 1.0)",
            "Parameter interpolasi t harus berada pada rentang [0.0, 1.0]",
            "Ray-Triangle parallel check: |a| > 1e-7 untuk menghindari deviasi pembagian nol",
        ],
        "error_cases": [
            "ZeroDivisionError jika panjang vektor nol",
            "Gimbal Lock jika menggunakan Euler angles biasa alih-alih Quaternion",
            "NaN / Infinity pada proyeksi jika z_near <= 0",
        ],
        "oracle_code": """
from moko_neuromath.spatial_math_engine import Vector3D, Quaternion, Raycast3D
# Oracle SGM Engine siap memverifikasi transformasi 3D & kinematika kamera
""",
    },
    MathDomain.TENSOR_MANIFOLD: {
        "definitions": [
            "Lie Algebra so(3) Skew-Symmetric Matrix: ω_hat = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]",
            "Lie Group SO(3) Exponential Map (Rodrigues): exp(ω_hat) = I + (sin θ / θ)ω_hat + ((1-cos θ)/θ²)ω_hat²",
            "Analytical Lagrangian Mechanics: d/dt( ∂L / ∂q_dot ) - ∂L / ∂q = 0 di mana L = T - V",
            "Gaussian Curvature K = (LN - M²) / (EG - F²) pada Manifold Permukaan 3D r(u,v)",
            "Mean Curvature H = (EN + GL - 2FM) / (2(EG - F²))",
        ],
        "boundary_conditions": [
            "Sudut θ pada Rodrigues' formula jika θ < 1e-9 gunakan batas limit Taylor I + ω_hat",
            "Denominator First & Second Fundamental Forms (EG - F²) > 1e-7 (bebas dari singularitas)",
            "General koordinat q harus independen dan kontinu terderivasi dua kali",
        ],
        "error_cases": [
            "Singularitas parametrik saat permukaan mengalami self-intersection",
            "Numerical drift jika menggunakan integrasi Euler biasa alih-alih Lie Group Exp Map",
        ],
        "oracle_code": """
from moko_neuromath.tensor_manifold_engine import LieGroupSO3, LagrangianMechanics, DifferentialGeometry
# Oracle TMG Engine siap memverifikasi tensor manifold & fisika analitis
""",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SCOT TEMPLATE ENGINE
# Berdasarkan: "Structured Chain-of-Thought Prompting for Code Generation"
# Paksa LLM untuk reasoning dalam pola: Sequential → Branch → Loop
# sebelum menulis kode. Terbukti meningkatkan HumanEval accuracy.
# ─────────────────────────────────────────────────────────────────────────────

SCOT_TEMPLATE = """
[MOKO STRUCTURED REASONING — BEFORE WRITING CODE, COMPLETE THIS TEMPLATE]

STEP 1 [SEQUENTIAL — Define I/O Contract]:
  - Function name  : <nama_fungsi>
  - Input params   : <param: tipe>
  - Return type    : <tipe_return>
  - Pre-conditions : <apa yang HARUS benar sebelum fungsi dipanggil>
  - Post-conditions: <apa yang HARUS benar setelah fungsi selesai>

STEP 2 [BRANCH — Error & Edge Cases]:
  - JIKA <kondisi_error_1> → raise <ExceptionType>("<pesan>")
  - JIKA <kondisi_edge_1> → return <nilai_default>
  - JIKA <kondisi_edge_2> → <penanganan_khusus>

STEP 3 [LOOP — Core Algorithm]:
  - Struktur iterasi : for/while <kondisi>
  - Transformasi    : setiap iterasi melakukan <operasi>
  - Akumulasi       : hasil dikumpulkan di <variabel>
  - Terminasi       : loop berhenti saat <kondisi_berhenti>

STEP 4 [VERIFY — Mathematical Check]:
  - Formula/invariant : <persamaan atau properti yang harus selalu benar>
  - Test case 1       : <input> → expected <output>
  - Test case 2       : <edge_input> → expected <edge_output>

[NOW WRITE THE COMPLETE, VERIFIED CODE BELOW]
"""


@dataclass
class AmplifiedQuery:
    """
    Hasil dari proses amplifikasi query.
    Berisi semua komponen untuk membangun prompt yang diperkuat secara matematis.
    """
    original_input: str
    detected_domains: List[MathDomain]
    formal_definitions: List[str]
    boundary_conditions: List[str]
    error_cases: List[str]
    scot_template: str
    oracle_hints: List[str]
    amplified_prompt: str
    hash_id: str  # Fingerprint untuk caching

    @property
    def complexity_score(self) -> float:
        """
        Skor kompleksitas query (0.0 - 1.0).
        Tinggi = perlu lebih banyak reasoning.
        """
        base = len(self.detected_domains) * 0.2
        depth = len(self.formal_definitions) * 0.05
        return min(1.0, base + depth)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AMPLIFIER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MathQueryAmplifier:
    """
    Inti sistem MQA.
    
    Mengubah input mentah bahasa manusia menjadi representasi matematis
    formal yang memaksimalkan akurasi output LLM.
    
    Pipeline:
      raw_input → classify → extract_formulas → build_scot → assemble_prompt
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._cache: Dict[str, AmplifiedQuery] = {}

    # ── 1. DOMAIN CLASSIFIER ──────────────────────────────────────────────

    def classify_domains(self, text: str) -> List[MathDomain]:
        """
        Klasifikasi domain matematika dari teks input.
        Gunakan keyword matching + tf-like scoring.
        
        Return: list domain yang terdeteksi, diurutkan berdasarkan skor.
        """
        text_lower = text.lower()
        domain_scores: Dict[MathDomain, int] = {}

        for domain, keywords in DOMAIN_SIGNATURES.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                domain_scores[domain] = score

        # Deteksi sinyal coding umum
        coding_patterns = [
            r'\bfungsi\b', r'\bfunction\b', r'\bdef\b', r'\bclass\b',
            r'\bimplementasikan\b', r'\bprogram\b', r'\bkode\b', r'\bcode\b',
            r'\balgoritma\b', r'\balgorithm\b', r'\bbuat\b.*\byang\b',
        ]
        coding_signal = sum(1 for p in coding_patterns if re.search(p, text_lower))
        if coding_signal > 0 and not domain_scores:
            domain_scores[MathDomain.GENERAL_CODING] = coding_signal

        # Urutkan berdasarkan skor tertinggi
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_domains:
            return [MathDomain.GENERAL_CODING]
        
        # Ambil domain dengan skor > 0, maks 3 domain utama
        return [d for d, _ in sorted_domains[:3]]

    # ── 2. FORMAL KNOWLEDGE EXTRACTOR ────────────────────────────────────

    def extract_domain_knowledge(
        self, domains: List[MathDomain]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Ekstrak semua pengetahuan formal dari domains yang terdeteksi.
        
        Return: (definitions, boundary_conditions, error_cases, oracle_hints)
        """
        definitions: List[str] = []
        boundaries: List[str] = []
        errors: List[str] = []
        oracles: List[str] = []

        seen = set()
        for domain in domains:
            if domain not in FORMAL_DEFINITIONS:
                domain = MathDomain.GENERAL_CODING
            
            knowledge = FORMAL_DEFINITIONS[domain]
            
            for d in knowledge.get("definitions", []):
                if d not in seen:
                    definitions.append(d)
                    seen.add(d)
            
            for b in knowledge.get("boundary_conditions", []):
                if b not in seen:
                    boundaries.append(b)
                    seen.add(b)
            
            for e in knowledge.get("error_cases", []):
                if e not in seen:
                    errors.append(e)
                    seen.add(e)
            
            oracle = knowledge.get("oracle_code", "").strip()
            if oracle and oracle not in oracles:
                oracles.append(oracle)

        return definitions, boundaries, errors, oracles

    # ── 3. SCOT TEMPLATE BUILDER ──────────────────────────────────────────

    def build_scot_section(self, original_input: str, domains: List[MathDomain]) -> str:
        """
        Bangun template SCoT yang spesifik untuk request ini.
        Template ini akan diisi oleh LLM sebelum menulis kode.
        """
        domain_names = ", ".join(d.value for d in domains)
        
        # Deteksi apakah ini request coding
        is_coding_request = any([
            re.search(r'\b(buat|create|implement|tulis|write|fungsi|function|kode|code|program)\b',
                      original_input.lower()),
            any(d in domains for d in [
                MathDomain.SORTING, MathDomain.DYNAMIC_PROG,
                MathDomain.STRING_PROC, MathDomain.GRAPH_THEORY,
                MathDomain.GENERAL_CODING
            ])
        ])
        
        if not is_coding_request:
            # Untuk non-coding, gunakan reasoning template yang lebih singkat
            return f"""
[MOKO MATHEMATICAL REASONING]
Domain: {domain_names}
Reasoning approach:
  1. Identifikasi formula relevan
  2. Substitusi nilai
  3. Hitung step-by-step
  4. Verifikasi dengan oracle/definisi formal
"""
        
        return SCOT_TEMPLATE

    # ── 4. PROMPT ASSEMBLER ───────────────────────────────────────────────

    def assemble_amplified_prompt(
        self,
        original_input: str,
        domains: List[MathDomain],
        definitions: List[str],
        boundaries: List[str],
        errors: List[str],
        oracles: List[str],
        scot: str,
    ) -> str:
        """
        Rakit semua komponen menjadi prompt yang diperkuat secara matematis.
        
        Ini adalah implementasi dari konsep "input (bahasa manusia) → matematika"
        yang menjadi fondasi sistem komputasi tingkat tinggi.
        """
        domain_str = " | ".join(f"[{d.value.upper()}]" for d in domains)
        
        # Bangun seksi definisi
        def_section = "\n".join(f"  • {d}" for d in definitions) if definitions else "  • (general programming)"
        
        # Bangun seksi boundary
        bound_section = "\n".join(f"  ⚠ {b}" for b in boundaries) if boundaries else "  ⚠ handle empty/null input"
        
        # Bangun seksi error
        error_section = "\n".join(f"  ✗ {e}" for e in errors) if errors else "  ✗ TypeError, ValueError"
        
        # Oracle hints
        oracle_section = "\n".join(oracles) if oracles else ""
        
        amplified = f"""
╔══════════════════════════════════════════════════════════════════
║ MOKO MATHEMATICAL QUERY AMPLIFIER — Amplified Reasoning Context
╠══════════════════════════════════════════════════════════════════
║ DOMAIN DETECTED   : {domain_str}
║ ORIGINAL REQUEST  : {original_input}
╠══════════════════════════════════════════════════════════════════
║ § FORMAL MATHEMATICAL DEFINITIONS:
{def_section}
║
║ § BOUNDARY CONDITIONS (MUST ENFORCE):
{bound_section}
║
║ § ERROR CASES (MUST HANDLE):
{error_section}
╚══════════════════════════════════════════════════════════════════

{scot}

---
VERIFICATION ORACLE HINTS:
{oracle_section if oracle_section else "# Use assert statements to verify output correctness"}

---
USER REQUEST (answer this completely, respecting all mathematical constraints above):
{original_input}
""".strip()

        return amplified

    # ── 5. MAIN AMPLIFY METHOD ────────────────────────────────────────────

    def amplify(self, raw_input: str, force: bool = False) -> AmplifiedQuery:
        """
        Entry point utama MQA.
        
        Mengambil input mentah dan mengembalikan AmplifiedQuery yang siap
        dikirim ke LLM sebagai pengganti input biasa.
        
        Args:
            raw_input: teks input asli dari user
            force: abaikan cache, proses ulang
            
        Return:
            AmplifiedQuery dengan semua komponen matematis
        """
        # Cache check
        input_hash = hashlib.md5(raw_input.encode()).hexdigest()[:12]
        if not force and input_hash in self._cache:
            return self._cache[input_hash]

        if self.verbose:
            print(f"[MQA] Processing: '{raw_input[:60]}...'")

        # Step 1: Classify domain
        domains = self.classify_domains(raw_input)
        if self.verbose:
            print(f"[MQA] Domains: {[d.value for d in domains]}")

        # Step 2: Extract formal knowledge
        definitions, boundaries, errors, oracles = self.extract_domain_knowledge(domains)

        # Step 3: Build SCoT template
        scot = self.build_scot_section(raw_input, domains)

        # Step 4: Inject Repo Map context untuk domain kode/matematika kompleks
        repo_context_section = ""
        needs_repo_context = any(d.value in [
            "spatial_graphics", "tensor_manifold", "linear_algebra",
            "general_coding", "graph_theory", "optimization"
        ] for d in domains)
        if needs_repo_context:
            try:
                from moko_agents.repo_mapper import get_repo_mapper
                mapper = get_repo_mapper()
                repo_ctx = mapper.get_context_for_query(raw_input, max_symbols=12)
                if repo_ctx:
                    repo_context_section = "\n\n" + repo_ctx
            except Exception:
                pass

        # Step 5: Assemble final amplified prompt
        amplified = self.assemble_amplified_prompt(
            raw_input, domains, definitions, boundaries, errors, oracles, scot
        ) + repo_context_section

        # Package result
        result = AmplifiedQuery(
            original_input=raw_input,
            detected_domains=domains,
            formal_definitions=definitions,
            boundary_conditions=boundaries,
            error_cases=errors,
            scot_template=scot,
            oracle_hints=oracles,
            amplified_prompt=amplified,
            hash_id=input_hash,
        )

        self._cache[input_hash] = result
        return result

    def get_stats(self) -> Dict:
        """Statistik penggunaan MQA."""
        return {
            "cached_queries": len(self._cache),
            "domains_covered": len(DOMAIN_SIGNATURES),
            "formal_definitions_total": sum(
                len(v.get("definitions", [])) for v in FORMAL_DEFINITIONS.values()
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE — digunakan oleh analyst_node.py
# ─────────────────────────────────────────────────────────────────────────────

_mqa_instance: Optional[MathQueryAmplifier] = None

def get_mqa(verbose: bool = False) -> MathQueryAmplifier:
    """Get atau buat singleton MQA instance."""
    global _mqa_instance
    if _mqa_instance is None:
        _mqa_instance = MathQueryAmplifier(verbose=verbose)
    return _mqa_instance


def amplify_query(raw_input: str) -> str:
    """
    Shortcut function untuk amplify query dan langsung return string prompt.
    Gunakan ini dari analyst_node.py.
    """
    mqa = get_mqa()
    result = mqa.amplify(raw_input)
    return result.amplified_prompt


def amplify_if_coding(raw_input: str) -> Tuple[str, bool]:
    """
    Amplify hanya jika query adalah coding/math request.
    Return: (prompt_to_use, was_amplified)
    
    Untuk query percakapan biasa (salam, opini, dsb.) → kembalikan original.
    Untuk query math/coding → kembalikan amplified version.
    """
    mqa = get_mqa()
    result = mqa.amplify(raw_input)
    
    # Jika hanya GENERAL_CODING dengan skor rendah dan tidak ada math domain
    non_trivial = any(
        d not in [MathDomain.GENERAL_CODING]
        for d in result.detected_domains
    )
    
    has_coding_keywords = bool(re.search(
        r'\b(buat|create|implement|tulis|write|fungsi|function|kode|code|'
        r'program|algoritma|algorithm|debug|error|fix|perbaiki)\b',
        raw_input.lower()
    ))
    
    should_amplify = non_trivial or has_coding_keywords
    
    if should_amplify:
        return result.amplified_prompt, True
    else:
        return raw_input, False
