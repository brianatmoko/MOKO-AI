"""
MOKO Architect-Editor Pipeline — Industrial-Grade Code Generation
================================================================
Berdasarkan: Aider "Architect Mode" (Paul Gauthier, 2023)
             OpenHands Self-Healing Execution Loop
             AlphaCode Competitive Programming Pipeline (DeepMind)

Tiga Fase Kognitif Profesional:
  1. ARCHITECT  → Perencanaan arsitektur & matematika tingkat tinggi
  2. EDITOR     → Implementasi kode presisi dari rencana arsitektur
  3. SELF-HEAL  → Eksekusi sandboxed + verifikasi otomatis + perbaikan mandiri

Berbeda dari prototipe: Kode TIDAK langsung disajikan kepada user.
Kode dites dan diverifikasi secara mandiri sampai benar, BARU disajikan.
"""

import re
import ast
import sys
import time
import traceback
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ArchitectPlan:
    """Rencana arsitektur yang dihasilkan oleh Fase Architect"""
    problem_statement: str
    math_foundations: List[str]     # Rumus matematika yang digunakan
    algorithm_outline: List[str]    # Garis besar algoritma
    data_structures: List[str]      # Struktur data yang diperlukan
    edge_cases: List[str]           # Kasus tepi yang harus ditangani
    verification_tests: List[str]   # Test case yang harus lulus
    complexity_note: str            # O(n) complexity, catatan performa

    def to_prompt_context(self) -> str:
        parts = []
        parts.append("╔═══════════════════════════════════════════════════════════")
        parts.append("║  MOKO ARCHITECT PLAN (Fase 1 dari 3)")
        parts.append("╠═══════════════════════════════════════════════════════════")
        parts.append(f"║ Problem: {self.problem_statement}")
        parts.append("╠═══════════════════════════════════════════════════════════")
        if self.math_foundations:
            parts.append("║ MATHEMATICAL FOUNDATIONS:")
            for f in self.math_foundations:
                parts.append(f"║   • {f}")
        if self.algorithm_outline:
            parts.append("║ ALGORITHM OUTLINE:")
            for i, step in enumerate(self.algorithm_outline, 1):
                parts.append(f"║   {i}. {step}")
        if self.data_structures:
            parts.append("║ DATA STRUCTURES:")
            for ds in self.data_structures:
                parts.append(f"║   • {ds}")
        if self.edge_cases:
            parts.append("║ EDGE CASES TO HANDLE:")
            for ec in self.edge_cases:
                parts.append(f"║   ⚠ {ec}")
        if self.verification_tests:
            parts.append("║ VERIFICATION TARGETS:")
            for t in self.verification_tests:
                parts.append(f"║   ✓ {t}")
        if self.complexity_note:
            parts.append(f"║ COMPLEXITY: {self.complexity_note}")
        parts.append("╚═══════════════════════════════════════════════════════════")
        return "\n".join(parts)


@dataclass
class EditorResult:
    """Hasil implementasi kode dari Fase Editor"""
    code: str
    language: str
    entry_point: str
    test_harness: str   # Kode test untuk self-healing


@dataclass
class ExecutionResult:
    """Hasil eksekusi sandboxed dari kode yang dihasilkan"""
    success: bool
    stdout: str
    stderr: str
    error_type: Optional[str]
    exec_time_ms: float
    healed: bool = False
    heal_attempts: int = 0


class SandboxExecutor:
    """Menjalankan kode Python di lingkungan sandboxed (subprocess) tanpa menginfeksi proses utama"""

    MAX_TIMEOUT_SEC = 8

    @staticmethod
    def run_python(code: str, test_harness: str = "") -> ExecutionResult:
        """Eksekusi kode + test harness dalam subprocess terisolasi"""
        full_code = code.strip()
        if test_harness:
            full_code += "\n\n# === MOKO SELF-VERIFICATION HARNESS ===\n" + test_harness.strip()

        t0 = time.time()
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as tmp:
                tmp.write(full_code)
                tmp_path = tmp.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True,
                timeout=SandboxExecutor.MAX_TIMEOUT_SEC,
                cwd=os.path.dirname(tmp_path)
            )
            os.unlink(tmp_path)

            elapsed = (time.time() - t0) * 1000
            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    stdout=result.stdout[:2000],
                    stderr="",
                    error_type=None,
                    exec_time_ms=elapsed
                )
            else:
                error_lines = result.stderr.strip().split('\n')
                error_type = None
                for line in reversed(error_lines):
                    m = re.match(r'^(\w+Error|Exception):', line.strip())
                    if m:
                        error_type = m.group(1)
                        break
                return ExecutionResult(
                    success=False,
                    stdout=result.stdout[:500],
                    stderr=result.stderr[:1000],
                    error_type=error_type,
                    exec_time_ms=elapsed
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, stdout="", stderr="Timeout: eksekusi melebihi batas waktu.",
                error_type="TimeoutError", exec_time_ms=(time.time() - t0) * 1000
            )
        except Exception as e:
            return ExecutionResult(
                success=False, stdout="", stderr=str(e),
                error_type=type(e).__name__, exec_time_ms=(time.time() - t0) * 1000
            )


class ArchitectEditorPipeline:
    """
    Pipeline tiga fase profesional untuk menghasilkan kode yang terverifikasi.
    Tidak menyajikan kode ke user sebelum terbukti benar secara eksekusi.
    """

    MAX_HEAL_ATTEMPTS = 3

    def __init__(self, llm_caller=None, verbose: bool = True):
        """
        Args:
            llm_caller: fungsi callable(prompt: str) -> str untuk memanggil LLM.
                        Jika None, pipeline berjalan dalam mode deterministik saja.
        """
        self.llm_caller = llm_caller
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  ⚙️  [AEP] {msg}")

    def _extract_python_code(self, text: str) -> str:
        """Ekstrak blok kode Python dari respons LLM"""
        # Cari blok ```python ... ``` atau ``` ... ```
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
        # Fallback: ambil teks keseluruhan jika ada def atau class
        if 'def ' in text or 'class ' in text:
            return text.strip()
        return ""

    def _validate_syntax(self, code: str) -> Tuple[bool, str]:
        """Validasi sintaks Python tanpa menjalankan kode"""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError on line {e.lineno}: {e.msg}"

    def build_architect_plan(self, query: str, repo_context: str = "") -> ArchitectPlan:
        """
        Fase 1: Architect — Susun rencana matematika & arsitektur.
        Jika LLM tersedia, gunakan LLM. Jika tidak, susun rencana deterministik.
        """
        self._log(f"Fase 1 (Architect): Menyusun rencana untuk '{query[:60]}...'")

        if self.llm_caller:
            architect_prompt = f"""\
Kamu adalah MOKO Architect — seorang insinyur matematika dan software senior.
Analisis request berikut dan susun rencana arsitektur TERSTRUKTUR (BUKAN kode langsung).

{repo_context}

REQUEST: {query}

Balas HANYA dalam format ini (isi dalam kurung sudut):
PROBLEM: <ringkasan masalah dalam 1 kalimat>
MATH: <rumus kunci 1> | <rumus kunci 2> | <dst>
STEPS: <langkah 1> | <langkah 2> | <langkah 3> | <dst>
STRUCTS: <struktur data 1> | <dst>
EDGES: <edge case 1> | <edge case 2>
TESTS: <test assertion 1> | <test assertion 2>
COMPLEXITY: <notasi Big-O dan catatan performa>
"""
            raw = self.llm_caller(architect_prompt)
            return self._parse_architect_response(raw, query)
        else:
            # Mode deterministik tanpa LLM
            return self._build_deterministic_plan(query)

    def _parse_architect_response(self, raw: str, original_query: str) -> ArchitectPlan:
        """Parse respons LLM Architect menjadi ArchitectPlan terstruktur"""
        def extract(key: str) -> List[str]:
            m = re.search(rf'^{key}:\s*(.+)$', raw, re.MULTILINE | re.IGNORECASE)
            if m:
                return [s.strip() for s in m.group(1).split('|') if s.strip()]
            return []

        problem = extract("PROBLEM")
        return ArchitectPlan(
            problem_statement=problem[0] if problem else original_query,
            math_foundations=extract("MATH"),
            algorithm_outline=extract("STEPS"),
            data_structures=extract("STRUCTS"),
            edge_cases=extract("EDGES"),
            verification_tests=extract("TESTS"),
            complexity_note=extract("COMPLEXITY")[0] if extract("COMPLEXITY") else ""
        )

    def _build_deterministic_plan(self, query: str) -> ArchitectPlan:
        """Susun rencana deterministik berbasis keyword"""
        q = query.lower()
        math = []
        steps = []
        edges = []
        tests = []
        complexity = "O(n) atau lebih baik"

        if any(k in q for k in ['sort', 'urutkan', 'sorted']):
            math = ["Comparison-based lower bound: Ω(n log n)", "Merge Sort: T(n) = 2T(n/2) + O(n)"]
            steps = ["Validasi input", "Pilih algoritma sort", "Handle edge cases", "Kembalikan hasil"]
            edges = ["input kosong []", "satu elemen", "elemen duplikat", "semua elemen sama"]
            tests = ["sorted([3,1,2]) == [1,2,3]", "sorted([]) == []"]
            complexity = "O(n log n) average"
        elif any(k in q for k in ['prime', 'prima', 'bilangan prima']):
            math = ["Sieve of Eratosthenes: O(n log log n)", "Trial division: O(√n)"]
            steps = ["Sieve initialization", "Cross out composites", "Collect primes"]
            edges = ["n < 2 → []", "n = 2 → [2]"]
            tests = ["primes(10) == [2,3,5,7]", "primes(1) == []"]
            complexity = "O(n log log n) Sieve"
        elif any(k in q for k in ['matrix', 'matriks']):
            math = ["Matrix multiply: C[i][j] = Σ A[i][k] * B[k][j]", "Dimensi: kolom A = baris B"]
            steps = ["Cek dimensi", "Init result matrix", "Triple loop compute", "Return"]
            edges = ["dimensi tidak kompatibel → ValueError", "matriks kosong"]
            tests = ["matmul([[1,0],[0,1]], A) == A (identity)"]
            complexity = "O(n³) naïve, O(n^2.37) Strassen"
        else:
            steps = ["Parse input", "Validasi constraints", "Eksekusi core logic", "Verifikasi output", "Return hasil"]
            edges = ["input None/kosong", "overflow integer", "tipe data salah"]
            tests = ["output type benar", "edge case tidak crash"]

        return ArchitectPlan(
            problem_statement=query,
            math_foundations=math,
            algorithm_outline=steps,
            data_structures=["list", "dict", "set"] if not math else ["list"],
            edge_cases=edges,
            verification_tests=tests,
            complexity_note=complexity
        )

    def generate_code(self, plan: ArchitectPlan, query: str, repo_context: str = "") -> EditorResult:
        """
        Fase 2: Editor — Implementasikan kode dari rencana arsitektur.
        """
        self._log("Fase 2 (Editor): Mengimplementasikan kode dari rencana...")

        if self.llm_caller:
            editor_prompt = f"""\
Kamu adalah MOKO Editor — developer Python senior yang mengimplementasikan rencana arsitektur.

{plan.to_prompt_context()}

{repo_context}

REQUEST ASLI: {query}

Tulis implementasi Python LENGKAP dan PROFESIONAL yang:
1. Mengikuti SEMUA langkah dalam arsitektur plan
2. Menangani SEMUA edge cases yang disebutkan
3. Menggunakan type hints dan docstring yang jelas
4. Bisa langsung dijalankan

PENTING: Tulis HANYA kode Python (tanpa penjelasan panjang), dalam blok ```python ... ```
"""
            raw_code = self.llm_caller(editor_prompt)
            code = self._extract_python_code(raw_code)
            if not code:
                code = raw_code.strip()
        else:
            code = self._generate_template_code(plan, query)

        # Buat test harness
        test_harness = self._build_test_harness(plan)

        return EditorResult(
            code=code,
            language="python",
            entry_point="main",
            test_harness=test_harness
        )

    def _generate_template_code(self, plan: ArchitectPlan, query: str) -> str:
        """Template kode minimal deterministik jika tidak ada LLM"""
        q = query.lower()
        if any(k in q for k in ['sort', 'urutkan']):
            return """\
from typing import List

def sort_elements(data: List) -> List:
    \"\"\"Mengurutkan elemen menggunakan merge sort O(n log n).\"\"\"
    if len(data) <= 1:
        return list(data)
    mid = len(data) // 2
    left = sort_elements(data[:mid])
    right = sort_elements(data[mid:])
    return _merge(left, right)

def _merge(left: List, right: List) -> List:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
        return f'# TODO: Implement solution for: {query}\n'

    def _build_test_harness(self, plan: ArchitectPlan) -> str:
        """Bangun test harness otomatis dari verification_tests plan"""
        lines = []
        lines.append("import sys")
        lines.append("_all_passed = True")
        for test in plan.verification_tests:
            lines.append(f"try:")
            lines.append(f"    assert {test}, 'FAIL: {test}'")
            lines.append(f"    print('  ✅ PASS: {test}')")
            lines.append(f"except Exception as _e:")
            lines.append(f"    print(f'  ❌ FAIL: {test} — {{_e}}')")
            lines.append(f"    _all_passed = False")
        lines.append("if not _all_passed: sys.exit(1)")
        return "\n".join(lines)

    def self_heal(self, editor_result: EditorResult, plan: ArchitectPlan, query: str) -> Tuple[EditorResult, ExecutionResult]:
        """
        Fase 3: Self-Healing Loop — Eksekusi + Verifikasi + Perbaiki Mandiri.
        Kode tidak disajikan ke user sampai benar atau mencapai batas upaya.
        """
        self._log("Fase 3 (Self-Heal): Memulai verifikasi & perbaikan mandiri...")

        code = editor_result.code
        heal_attempts = 0
        final_exec: Optional[ExecutionResult] = None

        for attempt in range(self.MAX_HEAL_ATTEMPTS + 1):
            # 1. Validasi sintaks dulu (instant)
            syntax_ok, syntax_err = self._validate_syntax(code)
            if not syntax_ok:
                self._log(f"  Attempt {attempt+1}: Syntax error — {syntax_err}")
                if self.llm_caller and attempt < self.MAX_HEAL_ATTEMPTS:
                    code = self._heal_syntax(code, syntax_err)
                    heal_attempts += 1
                    continue
                else:
                    final_exec = ExecutionResult(False, "", syntax_err, "SyntaxError", 0.0, heal_attempts > 0, heal_attempts)
                    break

            # 2. Eksekusi sandboxed
            exec_result = SandboxExecutor.run_python(code, editor_result.test_harness)
            final_exec = exec_result

            if exec_result.success:
                self._log(f"  Attempt {attempt+1}: ✅ Sukses ({exec_result.exec_time_ms:.1f}ms)")
                final_exec.healed = heal_attempts > 0
                final_exec.heal_attempts = heal_attempts
                break
            else:
                self._log(f"  Attempt {attempt+1}: ❌ {exec_result.error_type} — {exec_result.stderr[:80]}")
                if self.llm_caller and attempt < self.MAX_HEAL_ATTEMPTS:
                    code = self._heal_runtime(code, exec_result, plan)
                    heal_attempts += 1
                else:
                    final_exec.healed = heal_attempts > 0
                    final_exec.heal_attempts = heal_attempts
                    break

        healed_result = EditorResult(
            code=code,
            language=editor_result.language,
            entry_point=editor_result.entry_point,
            test_harness=editor_result.test_harness
        )

        return healed_result, final_exec

    def _heal_syntax(self, code: str, error_msg: str) -> str:
        """Perbaiki error sintaks via LLM"""
        if not self.llm_caller:
            return code
        prompt = f"""\
Perbaiki kode Python berikut yang memiliki syntax error:
ERROR: {error_msg}

KODE:
```python
{code}
```
Balas HANYA dengan kode yang sudah diperbaiki dalam blok ```python ... ```
"""
        return self._extract_python_code(self.llm_caller(prompt)) or code

    def _heal_runtime(self, code: str, exec_result: ExecutionResult, plan: ArchitectPlan) -> str:
        """Perbaiki error runtime via LLM"""
        if not self.llm_caller:
            return code
        prompt = f"""\
Perbaiki kode Python berikut yang gagal saat eksekusi:
ERROR TYPE: {exec_result.error_type}
STDERR: {exec_result.stderr[:600]}

KODE:
```python
{code}
```

CATATAN ARSITEKTUR:
{plan.to_prompt_context()}

Balas HANYA dengan kode yang sudah diperbaiki dalam blok ```python ... ```
"""
        return self._extract_python_code(self.llm_caller(prompt)) or code

    def run_full_pipeline(self, query: str, repo_context: str = "") -> Dict[str, Any]:
        """
        Jalankan tiga fase pipeline secara penuh dan kembalikan hasil terverifikasi.
        """
        t0 = time.time()

        # Fase 1: Architect
        plan = self.build_architect_plan(query, repo_context)

        # Fase 2: Editor
        editor_result = self.generate_code(plan, query, repo_context)

        # Fase 3: Self-Heal
        final_result, exec_result = self.self_heal(editor_result, plan, query)

        elapsed = (time.time() - t0) * 1000

        return {
            "query": query,
            "plan": plan,
            "code": final_result.code,
            "execution": exec_result,
            "elapsed_ms": elapsed,
            "status": "verified" if exec_result.success else "unverified",
            "heal_attempts": exec_result.heal_attempts if exec_result else 0
        }
