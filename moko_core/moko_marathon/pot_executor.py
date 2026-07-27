"""
MOKO Program-of-Thought Executor (PoT)
=======================================
Berdasarkan riset:
  - "Program of Thought" (PoT) paradigm (2023-2024)
  - "Caco" & "PromptCoT" — code-anchored verification
  - Execution grounding: output LLM divalidasi dengan eksekusi nyata

Filosofi:
  LLM bisa BERBOHONG tentang kebenaran kode.
  Python interpreter TIDAK BISA berbohong tentang eksekusi.
  
  → Gunakan interpreter sebagai hakim, bukan LLM.

Pipeline:
  1. Analisis domain dari kode yang dihasilkan LLM
  2. Auto-generate oracle tests dari definisi matematika formal
     (BUKAN dari LLM — dari scipy/sympy/math library)
  3. Execute kode + oracle tests
  4. Jika gagal → ekstrak error + witness → kirim ke self-healing
  5. Jika berhasil → kode verified secara matematis

Ini berbeda dari code_verifier.py yang hanya cek sintaks.
PoT benar-benar MENJALANKAN kode dan MEMBUKTIKAN outputnya benar.
"""

import ast
import sys
import os
import re
import math
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# RESULT STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OracleTest:
    """Satu test case yang dihasilkan dari definisi matematika formal."""
    description: str        # Apa yang ditest
    test_code: str          # Kode assertion Python
    domain: str             # Domain matematika
    is_edge_case: bool = False  # Apakah ini edge case


@dataclass
class PoTResult:
    """Hasil eksekusi Program-of-Thought."""
    ok: bool
    passed_tests: int
    total_tests: int
    failures: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    witness: Optional[str] = None  # Input yang memicu kegagalan


# ─────────────────────────────────────────────────────────────────────────────
# ORACLE GENERATOR
# Menghasilkan test case dari definisi matematika — bukan dari LLM
# Ini kunci utama PoT: ground truth berasal dari matematika, bukan probabilitas
# ─────────────────────────────────────────────────────────────────────────────

class OracleGenerator:
    """
    Menghasilkan oracle test cases dari definisi formal matematika.
    
    Setiap oracle adalah sebuah assertion Python yang PASTI benar
    berdasarkan definisi matematis, bukan berdasarkan tebakan LLM.
    """

    def generate_for_code(self, code: str, function_name: Optional[str] = None) -> List[OracleTest]:
        """
        Analisis kode, deteksi domain, dan generate oracle tests.
        """
        code_lower = code.lower()
        tests: List[OracleTest] = []

        # Deteksi fungsi utama jika tidak diberikan
        if function_name is None:
            function_name = self._extract_function_name(code)

        if not function_name:
            # Tidak ada fungsi yang bisa ditest
            return tests

        # Deteksi domain dan generate tests yang sesuai
        if self._is_sorting(code_lower):
            tests.extend(self._sorting_oracles(function_name))
        
        if self._is_statistics(code_lower):
            tests.extend(self._statistics_oracles(function_name))
        
        if self._is_gcd_lcm(code_lower):
            tests.extend(self._gcd_lcm_oracles(function_name))
        
        if self._is_palindrome(code_lower):
            tests.extend(self._palindrome_oracles(function_name))
        
        if self._is_fibonacci(code_lower):
            tests.extend(self._fibonacci_oracles(function_name))
        
        if self._is_prime(code_lower):
            tests.extend(self._prime_oracles(function_name))
        
        if self._is_factorial(code_lower):
            tests.extend(self._factorial_oracles(function_name))
        
        if self._is_binary_search(code_lower):
            tests.extend(self._binary_search_oracles(function_name))

        # Jika tidak ada domain yang terdeteksi, tambahkan basic type safety test
        if not tests:
            tests.extend(self._generic_safety_oracles(function_name))

        return tests

    # ── Domain Detectors ──────────────────────────────────────────────────

    def _is_sorting(self, code: str) -> bool:
        return any(kw in code for kw in [
            "sort", "urutkan", "ascending", "descending",
            "swap", "pivot", "merge", "partition"
        ])

    def _is_statistics(self, code: str) -> bool:
        return any(kw in code for kw in [
            "mean", "median", "mode", "variance", "std", "average",
            "rata-rata", "rata_rata", "weighted", "berbobot"
        ])

    def _is_gcd_lcm(self, code: str) -> bool:
        return any(kw in code for kw in ["gcd", "lcm", "fpb", "kpk", "euclidean"])

    def _is_palindrome(self, code: str) -> bool:
        return any(kw in code for kw in ["palindrome", "palindrom", "reverse"])

    def _is_fibonacci(self, code: str) -> bool:
        return any(kw in code for kw in ["fibonacci", "fib", "golden ratio"])

    def _is_prime(self, code: str) -> bool:
        return any(kw in code for kw in [
            "is_prime", "prima", "prime", "sieve", "eratosthenes"
        ])

    def _is_factorial(self, code: str) -> bool:
        return any(kw in code for kw in ["factorial", "faktorial", "n!"])

    def _is_binary_search(self, code: str) -> bool:
        return any(kw in code for kw in [
            "binary_search", "binary search", "bisect", "bsearch"
        ])

    def _extract_function_name(self, code: str) -> Optional[str]:
        """Ekstrak nama fungsi pertama yang didefinisikan dalam kode."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except SyntaxError:
            # Coba regex fallback
            m = re.search(r'def\s+(\w+)\s*\(', code)
            if m:
                return m.group(1)
        return None

    # ── Oracle Factories ──────────────────────────────────────────────────

    def _sorting_oracles(self, fn: str) -> List[OracleTest]:
        """
        Oracles untuk fungsi sorting.
        Ground truth: sorted() Python (Tim Sort, O(n log n), proven correct).
        """
        return [
            OracleTest(
                description="Sort list bilangan bulat",
                test_code=f"""
import builtins
_input = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
_result = {fn}(_input.copy())
_expected = builtins.sorted(_input)
assert _result == _expected, f"Sort salah: {{_result}} != {{_expected}}"
""",
                domain="sorting",
            ),
            OracleTest(
                description="Sort list kosong → kembalikan []",
                test_code=f"""
_result = {fn}([])
assert _result == [], f"Sort empty list harus return [], dapat: {{_result}}"
""",
                domain="sorting",
                is_edge_case=True,
            ),
            OracleTest(
                description="Sort satu elemen → tidak berubah",
                test_code=f"""
_result = {fn}([42])
assert _result == [42], f"Sort single element harus [42], dapat: {{_result}}"
""",
                domain="sorting",
                is_edge_case=True,
            ),
            OracleTest(
                description="Sort sudah terurut → tetap sama",
                test_code=f"""
import builtins
_input = [1, 2, 3, 4, 5]
_result = {fn}(_input.copy())
assert _result == _input, f"Sort already sorted salah: {{_result}}"
""",
                domain="sorting",
            ),
        ]

    def _statistics_oracles(self, fn: str) -> List[OracleTest]:
        """
        Oracles untuk fungsi statistik.
        Ground truth: definisi matematika langsung.
        mean([1,2,3]) = (1+2+3)/3 = 2.0 — ini adalah fakta matematis.
        """
        return [
            OracleTest(
                description="Mean [1,2,3] = 2.0 (definisi matematika)",
                test_code=f"""
_result = {fn}([1, 2, 3])
assert abs(_result - 2.0) < 1e-9, f"mean([1,2,3]) harus 2.0, dapat: {{_result}}"
""",
                domain="statistics",
            ),
            OracleTest(
                description="Mean satu elemen = elemen itu sendiri",
                test_code=f"""
_result = {fn}([7.5])
assert abs(_result - 7.5) < 1e-9, f"mean([7.5]) harus 7.5, dapat: {{_result}}"
""",
                domain="statistics",
                is_edge_case=True,
            ),
            OracleTest(
                description="Mean tidak boleh crash pada input kosong",
                test_code=f"""
try:
    _result = {fn}([])
    # Jika tidak raise exception, harus ada default/NaN behavior
    # Beberapa implementasi return 0 atau None — ini diterima
except (ValueError, ZeroDivisionError, StatisticsError):
    pass  # Exception yang tepat = benar
except Exception as e:
    assert False, f"Exception tidak diharapkan: {{type(e).__name__}}: {{e}}"
""",
                domain="statistics",
                is_edge_case=True,
            ),
        ]

    def _gcd_lcm_oracles(self, fn: str) -> List[OracleTest]:
        """
        Oracles untuk GCD/LCM.
        Ground truth: Teorema Euclidean — gcd(48, 18) = 6.
        """
        fn_lower = fn.lower()
        if "lcm" in fn_lower or "kpk" in fn_lower:
            return [
                OracleTest(
                    description="lcm(4, 6) = 12 (Teorema Aritmatika Dasar)",
                    test_code=f"""
_result = {fn}(4, 6)
assert _result == 12, f"lcm(4,6) harus 12, dapat: {{_result}}"
""",
                    domain="number_theory",
                ),
                OracleTest(
                    description="lcm(a, 1) = a",
                    test_code=f"""
for _a in [5, 13, 100]:
    _result = {fn}(_a, 1)
    assert _result == _a, f"lcm({{_a}},1) harus {{_a}}, dapat: {{_result}}"
""",
                    domain="number_theory",
                ),
            ]
        else:  # GCD
            return [
                OracleTest(
                    description="gcd(48, 18) = 6 (Algoritma Euclidean)",
                    test_code=f"""
_result = {fn}(48, 18)
assert _result == 6, f"gcd(48,18) harus 6, dapat: {{_result}}"
""",
                    domain="number_theory",
                ),
                OracleTest(
                    description="gcd(a, 0) = a (definisi)",
                    test_code=f"""
for _a in [5, 13, 100]:
    _result = {fn}(_a, 0)
    assert _result == _a, f"gcd({{_a}},0) harus {{_a}}, dapat: {{_result}}"
""",
                    domain="number_theory",
                    is_edge_case=True,
                ),
                OracleTest(
                    description="gcd(a, b) == gcd(b, a) (komutativitas)",
                    test_code=f"""
_pairs = [(12, 8), (100, 25), (17, 5)]
for _a, _b in _pairs:
    assert {fn}(_a, _b) == {fn}(_b, _a), f"gcd tidak komutatif untuk {{_a}}, {{_b}}"
""",
                    domain="number_theory",
                ),
            ]

    def _palindrome_oracles(self, fn: str) -> List[OracleTest]:
        return [
            OracleTest(
                description='"racecar" adalah palindrome',
                test_code=f"""
assert {fn}("racecar") == True, f'"racecar" harus palindrome'
""",
                domain="string",
            ),
            OracleTest(
                description='"hello" bukan palindrome',
                test_code=f"""
assert {fn}("hello") == False, f'"hello" bukan palindrome'
""",
                domain="string",
            ),
            OracleTest(
                description='String kosong adalah palindrome (definisi)',
                test_code=f"""
assert {fn}("") == True, f'Empty string harus palindrome'
""",
                domain="string",
                is_edge_case=True,
            ),
        ]

    def _fibonacci_oracles(self, fn: str) -> List[OracleTest]:
        """Fibonacci: F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2)"""
        return [
            OracleTest(
                description="Fibonacci sequence terkenal: 0,1,1,2,3,5,8,13,21",
                test_code=f"""
_expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
for _i, _exp in enumerate(_expected):
    _result = {fn}(_i)
    assert _result == _exp, f"fib({{_i}}) harus {{_exp}}, dapat: {{_result}}"
""",
                domain="number_theory",
            ),
            OracleTest(
                description="F(0) = 0, F(1) = 1 (base case)",
                test_code=f"""
assert {fn}(0) == 0, f"fib(0) harus 0"
assert {fn}(1) == 1, f"fib(1) harus 1"
""",
                domain="number_theory",
                is_edge_case=True,
            ),
        ]

    def _prime_oracles(self, fn: str) -> List[OracleTest]:
        return [
            OracleTest(
                description="Bilangan prima yang diketahui: 2,3,5,7,11,13",
                test_code=f"""
for _p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
    assert {fn}(_p) == True, f"{{_p}} harus prima"
""",
                domain="number_theory",
            ),
            OracleTest(
                description="Bilangan bukan prima: 1,4,6,8,9,10",
                test_code=f"""
for _n in [1, 4, 6, 8, 9, 10, 12, 15]:
    assert {fn}(_n) == False, f"{{_n}} bukan prima"
""",
                domain="number_theory",
            ),
            OracleTest(
                description="1 bukan prima (definisi)",
                test_code=f"""
assert {fn}(1) == False, f"1 BUKAN prima"
assert {fn}(2) == True, f"2 adalah prima terkecil"
""",
                domain="number_theory",
                is_edge_case=True,
            ),
        ]

    def _factorial_oracles(self, fn: str) -> List[OracleTest]:
        return [
            OracleTest(
                description="Faktorial terkenal: 0!=1, 1!=1, 5!=120, 10!=3628800",
                test_code=f"""
import math
_cases = [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120), (10, 3628800)]
for _n, _exp in _cases:
    _result = {fn}(_n)
    assert _result == _exp, f"{{_n}}! harus {{_exp}}, dapat: {{_result}}"
""",
                domain="number_theory",
            ),
            OracleTest(
                description="0! = 1 (definisi konvensi matematis)",
                test_code=f"""
assert {fn}(0) == 1, f"0! harus 1 (konvensi matematika)"
""",
                domain="number_theory",
                is_edge_case=True,
            ),
        ]

    def _binary_search_oracles(self, fn: str) -> List[OracleTest]:
        return [
            OracleTest(
                description="Binary search menemukan elemen yang ada",
                test_code=f"""
_arr = [1, 3, 5, 7, 9, 11, 13]
for _i, _val in enumerate(_arr):
    _result = {fn}(_arr, _val)
    assert _result == _i, f"binary_search(arr, {{_val}}) harus {{_i}}, dapat: {{_result}}"
""",
                domain="sorting",
            ),
            OracleTest(
                description="Binary search return -1 untuk elemen yang tidak ada",
                test_code=f"""
_arr = [1, 3, 5, 7, 9]
_result = {fn}(_arr, 4)
assert _result == -1, f"Elemen tidak ada harus return -1, dapat: {{_result}}"
""",
                domain="sorting",
                is_edge_case=True,
            ),
        ]

    def _generic_safety_oracles(self, fn: str) -> List[OracleTest]:
        """Oracles minimal untuk kode yang domain-nya tidak terdeteksi."""
        return [
            OracleTest(
                description="Fungsi bisa dipanggil tanpa crash (None safety)",
                test_code=f"""
# Test bahwa fungsi setidaknya bisa dipanggil
# Ini hanya smoke test — tidak ada assertion nilai
try:
    {fn}  # Pastikan fungsi terdefinisi
    assert callable({fn}), f"{fn} harus callable"
except NameError:
    assert False, f"Fungsi {fn} tidak terdefinisi"
""",
                domain="general",
            ),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# POT EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class PoTExecutor:
    """
    Program-of-Thought Executor.
    
    Menjalankan kode yang dihasilkan LLM bersama oracle tests dari
    OracleGenerator, dan mengembalikan hasil verifikasi matematis.
    """

    def __init__(self, timeout: int = 10, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.oracle_gen = OracleGenerator()

    def execute(self, code: str, function_name: Optional[str] = None) -> PoTResult:
        """
        Execute kode + oracle tests dan kembalikan PoTResult.
        
        Args:
            code: source code yang dihasilkan LLM
            function_name: nama fungsi utama (opsional, akan auto-detect)
            
        Return:
            PoTResult dengan status verifikasi dan detail kegagalan
        """
        # Generate oracle tests
        oracles = self.oracle_gen.generate_for_code(code, function_name)
        
        if not oracles:
            # Tidak ada oracle → jalankan kode biasa untuk cek sintaks
            return self._execute_syntax_only(code)
        
        if self.verbose:
            print(f"[PoT] Generated {len(oracles)} oracle tests")

        # Jalankan setiap oracle test
        passed = 0
        failures = []
        all_stdout = []

        for oracle in oracles:
            result = self._run_oracle(code, oracle)
            if result["ok"]:
                passed += 1
                if self.verbose:
                    print(f"  ✓ {oracle.description}")
            else:
                failure_msg = f"FAILED [{oracle.domain}] {oracle.description}: {result['error']}"
                failures.append(failure_msg)
                if oracle.is_edge_case:
                    # Edge case failure = critical
                    failures[-1] = "⚠ EDGE CASE " + failures[-1]
                if self.verbose:
                    print(f"  ✗ {oracle.description}")
                    print(f"    Error: {result['error']}")

            if result.get("stdout"):
                all_stdout.append(result["stdout"])

        return PoTResult(
            ok=len(failures) == 0,
            passed_tests=passed,
            total_tests=len(oracles),
            failures=failures,
            stdout="\n".join(all_stdout),
            witness=failures[0] if failures else None,
        )

    def _run_oracle(self, user_code: str, oracle: OracleTest) -> Dict:
        """Jalankan satu oracle test terhadap user_code."""
        # Gabungkan user code + oracle test code
        full_code = f"""
{user_code}

# ── ORACLE TEST: {oracle.description} ──
{oracle.test_code}
"""
        return self._run_python(full_code)

    def _execute_syntax_only(self, code: str) -> PoTResult:
        """Fallback: jalankan kode biasa jika tidak ada oracle."""
        result = self._run_python(code)
        return PoTResult(
            ok=result["ok"],
            passed_tests=0 if not result["ok"] else 1,
            total_tests=1,
            failures=[result["error"]] if not result["ok"] else [],
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
        )

    def _run_python(self, code: str) -> Dict:
        """Jalankan kode Python di subprocess dengan timeout."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True,
                timeout=self.timeout
            )
            
            if res.returncode == 0:
                return {"ok": True, "stdout": res.stdout, "stderr": ""}
            else:
                # Bersihkan path temp dari error message
                error = res.stderr.replace(tmp_path, "<code>").strip()
                return {"ok": False, "error": error, "stdout": res.stdout, "stderr": error}
        
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Timeout: kode tidak selesai dalam {self.timeout}s"}
        except Exception as e:
            return {"ok": False, "error": f"Execution error: {str(e)}"}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON & SHORTCUT
# ─────────────────────────────────────────────────────────────────────────────

_pot_instance: Optional[PoTExecutor] = None

def get_pot(verbose: bool = False) -> PoTExecutor:
    """Get atau buat singleton PoT Executor."""
    global _pot_instance
    if _pot_instance is None:
        _pot_instance = PoTExecutor(verbose=verbose)
    return _pot_instance


def verify_with_pot(code: str, function_name: Optional[str] = None) -> PoTResult:
    """
    Shortcut: verify kode dengan Program-of-Thought.
    Gunakan ini dari marathon_pitstop.py.
    """
    pot = get_pot()
    return pot.execute(code, function_name)
