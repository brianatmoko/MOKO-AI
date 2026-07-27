"""
MOKO Adversarial Self-Refinement Loop (ASR)
============================================
Berdasarkan riset:
  - "Adversarial CoT" — Generator-Discriminator Loop (2024-2025)
  - "Self-Correction via Feedback" paradigm
  - Meta-prompting: kritik output sendiri sebelum deliver ke user

Filosofi:
  LLM yang baik bukan yang langsung benar pada percobaan pertama.
  LLM yang baik adalah yang tahu kapan dirinya salah dan bisa memperbaiki diri.
  
  Discriminator (deterministik) >> Discriminator (LLM lain)
  karena deterministik tidak bisa ditipu oleh statistical pattern.

Pipeline:
  Generator (LLM) → Draft Output
       ↓
  Discriminator (deterministik):
    • Z3 SMT verification (batas logika)
    • PoT oracle execution (kebenaran matematis)
    • SCoT template validation (kelengkapan struktur)
    • Boundary condition check (robustness)
       ↓
  PASS → Deliver ke user
  FAIL → Feedback spesifik → Generator → (loop max N kali)
"""

import re
import ast
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum

# Import pilar lain
try:
    from moko_core.moko_marathon.pot_executor import verify_with_pot, PoTResult
    _pot_available = True
except ImportError:
    _pot_available = False

try:
    from moko_core.moko_marathon.code_verifier import run_all_verifications, VerifyResult
    _verifier_available = True
except ImportError:
    _verifier_available = False

try:
    from moko_core.moko_agents.math_query_amplifier import MathQueryAmplifier, MathDomain
    _mqa_available = True
except ImportError:
    _mqa_available = False

try:
    from moko_core.moko_marathon.security_auditor import SecurityAuditor
    _security_available = True
except ImportError:
    _security_available = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class DiscriminatorVerdict(Enum):
    PASS    = "pass"       # Kode lolos semua checks
    WARN    = "warn"       # Ada masalah minor, bisa dikirim dengan catatan
    FAIL    = "fail"       # Kode gagal, perlu regenerasi
    UNKNOWN = "unknown"    # Tidak bisa diverifikasi


@dataclass
class DiscriminatorReport:
    """Laporan lengkap dari Discriminator."""
    verdict: DiscriminatorVerdict
    overall_ok: bool
    
    # Hasil per komponen
    syntax_ok: bool = True
    syntax_errors: List[str] = field(default_factory=list)
    
    pot_ok: bool = True
    pot_passed: int = 0
    pot_total: int = 0
    pot_failures: List[str] = field(default_factory=list)
    
    boundary_ok: bool = True
    boundary_issues: List[str] = field(default_factory=list)
    
    scot_ok: bool = True
    scot_issues: List[str] = field(default_factory=list)
    
    security_ok: bool = True
    security_violations: List[str] = field(default_factory=list)
    
    # Feedback untuk Generator (jika FAIL)
    regeneration_prompt: str = ""
    
    # Score (0.0 = total fail, 1.0 = perfect)
    quality_score: float = 0.0


@dataclass
class RefinementResult:
    """Hasil akhir dari proses adversarial refinement."""
    final_output: str           # Output terbaik yang dihasilkan
    iterations: int             # Berapa kali loop dijalankan
    converged: bool             # Apakah berhasil pass semua checks
    final_report: DiscriminatorReport
    history: List[Dict]         # Riwayat setiap iterasi


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR — Deterministik, tidak bisa ditipu oleh LLM
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator:
    """
    Discriminator deterministik untuk Adversarial Self-Refinement.
    
    Menggunakan 4 lapisan verifikasi:
    1. Syntax check (AST parse)
    2. PoT Oracle execution (kebenaran matematis)
    3. Boundary condition analysis (robustness)
    4. SCoT structure validation (kelengkapan reasoning)
    
    TIDAK menggunakan LLM untuk menilai output LLM.
    Semua penilaian bersifat deterministik dan verifiable.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, code_output: str, original_request: str = "") -> DiscriminatorReport:
        """
        Evaluasi output LLM secara deterministik.
        
        Args:
            code_output: output dari Generator (LLM)
            original_request: request asli user (untuk konteks)
            
        Return:
            DiscriminatorReport dengan verdict dan feedback
        """
        report = DiscriminatorReport(
            verdict=DiscriminatorVerdict.UNKNOWN,
            overall_ok=False,
        )

        # Ekstrak code blocks dari output
        code_blocks = self._extract_code_blocks(code_output)
        
        if not code_blocks:
            # Tidak ada kode → kemungkinan jawaban teks biasa
            report.verdict = DiscriminatorVerdict.PASS
            report.overall_ok = True
            report.quality_score = 0.7  # Teks tanpa kode — kita tidak bisa memverifikasi
            return report

        # Analisis semua code blocks
        python_blocks = [(lang, code) for lang, code in code_blocks if lang in ["python", "py", ""]]
        
        if not python_blocks:
            # Kode non-Python (JavaScript, dll) — verifikasi syntax saja
            report.verdict = DiscriminatorVerdict.WARN
            report.overall_ok = True
            report.quality_score = 0.6
            return report

        # Ambil block Python terbesar/utama
        main_code = max(python_blocks, key=lambda x: len(x[1]))[1]

        # ── Layer 1: Syntax Check ──────────────────────────────────────
        report.syntax_ok, report.syntax_errors = self._check_syntax(main_code)
        
        # ── Layer 2: PoT Oracle Execution ─────────────────────────────
        if report.syntax_ok and _pot_available:
            pot_result = verify_with_pot(main_code)
            report.pot_ok = pot_result.ok
            report.pot_passed = pot_result.passed_tests
            report.pot_total = pot_result.total_tests
            report.pot_failures = pot_result.failures
        else:
            report.pot_ok = True  # Skip jika PoT tidak tersedia
            report.pot_total = 0

        # ── Layer 3: Boundary Condition Analysis ──────────────────────
        report.boundary_ok, report.boundary_issues = self._check_boundaries(main_code)
        
        # ── Layer 4: SCoT Structure Validation ────────────────────────
        report.scot_ok, report.scot_issues = self._check_scot_compliance(code_output)

        # ── Layer 5: Security & Vulnerability Audit ──────────────────
        if report.syntax_ok and _security_available:
            auditor = SecurityAuditor()
            audit_ext = "py"
            if python_blocks:
                audit_ext = "py"
            elif any(lang in ["js", "javascript"] for lang, _ in code_blocks):
                audit_ext = "js"
            elif any(lang in ["c", "cpp"] for lang, _ in code_blocks):
                audit_ext = "cpp"
            
            audit_res = auditor.audit_code(main_code, f"moko_draft.{audit_ext}")
            report.security_ok = audit_res.ok
            report.security_violations = [str(v) for v in audit_res.violations]
        else:
            report.security_ok = True
            report.security_violations = []

        # ── Aggregate Score ───────────────────────────────────────────
        report.quality_score = self._compute_score(report)
        
        # ── Determine Verdict ─────────────────────────────────────────
        # Check if there are HIGH severity security violations
        has_high_security_risk = any("[HIGH]" in v for v in report.security_violations)

        if not report.syntax_ok:
            report.verdict = DiscriminatorVerdict.FAIL
            report.overall_ok = False
        elif not report.pot_ok and report.pot_total > 0:
            report.verdict = DiscriminatorVerdict.FAIL
            report.overall_ok = False
        elif has_high_security_risk:
            report.verdict = DiscriminatorVerdict.FAIL
            report.overall_ok = False
        elif not report.boundary_ok and len(report.boundary_issues) > 2:
            report.verdict = DiscriminatorVerdict.WARN
            report.overall_ok = True  # WARN = bisa dikirim dengan catatan
        else:
            report.verdict = DiscriminatorVerdict.PASS
            report.overall_ok = True

        # ── Generate Regeneration Prompt jika FAIL ────────────────────
        if not report.overall_ok:
            report.regeneration_prompt = self._build_regen_prompt(
                report, original_request
            )

        if self.verbose:
            print(f"[Discriminator] Verdict: {report.verdict.value}, Score: {report.quality_score:.2f}")
            if report.pot_total > 0:
                print(f"  PoT: {report.pot_passed}/{report.pot_total} passed")

        return report

    # ── Layer Implementations ─────────────────────────────────────────

    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Ekstrak semua code blocks dari markdown output."""
        blocks = []
        # Match ```language\ncode\n```
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        for lang, code in matches:
            blocks.append((lang.lower().strip(), code.strip()))
        
        # Juga coba ekstrak kode Python tanpa markdown
        if not blocks and re.search(r'\bdef\s+\w+\s*\(', text):
            blocks.append(("python", text))
        
        return blocks

    def _check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Layer 1: Parse Python AST."""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"SyntaxError pada baris {e.lineno}: {e.msg}"]

    def _check_boundaries(self, code: str) -> Tuple[bool, List[str]]:
        """
        Layer 3: Analisis boundary conditions.
        Cek apakah kode menangani kondisi batas yang umum.
        """
        issues = []
        
        # Cek apakah ada division tanpa guard
        if re.search(r'\/\s*\w+', code) and not re.search(r'if.*==\s*0|try|except.*Zero', code, re.DOTALL):
            # Ada pembagian tapi tidak ada guard
            # Ini WARN bukan FAIL — mungkin sudah divalidasi di tempat lain
            if re.search(r'def\s+\w+.*:\n', code):
                issues.append("Pembagian ditemukan tanpa explicit zero-division guard")
        
        # Cek apakah ada list indexing tanpa bounds check
        if re.search(r'\w+\[\w+\]', code) and not re.search(r'if.*len|try|IndexError', code, re.DOTALL):
            issues.append("Indexing ditemukan tanpa explicit bounds check")
        
        # Cek apakah ada None check untuk parameter
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.args.args:
                    fn_body = ast.dump(node)
                    has_none_check = 'None' in fn_body or 'is None' in fn_body or 'if not' in fn_body
                    if not has_none_check and len(node.args.args) > 0:
                        issues.append(
                            f"Fungsi '{node.name}' tidak ada None/empty check di awal"
                        )
        except Exception:
            pass
        
        return len(issues) == 0, issues

    def _check_scot_compliance(self, full_output: str) -> Tuple[bool, List[str]]:
        """
        Layer 4: Validasi apakah LLM mengikuti SCoT template.
        SCoT compliance = LLM menunjukkan reasoning sebelum kode.
        """
        issues = []
        output_lower = full_output.lower()
        
        # Cek apakah ada reasoning sebelum kode
        has_reasoning = any([
            "step 1" in output_lower,
            "step1" in output_lower,
            "pre-condition" in output_lower,
            "precondition" in output_lower,
            "base case" in output_lower,
            "edge case" in output_lower,
            "input:" in output_lower and "output:" in output_lower,
            re.search(r'(pertama|first|langkah 1|1\.|step 1)\s*[:.]', output_lower) is not None,
        ])
        
        if not has_reasoning and len(full_output) > 200:
            issues.append("Output tidak menunjukkan structured reasoning sebelum kode (SCoT tidak diikuti)")
        
        # Cek apakah ada test cases / verifikasi di output
        has_verification = any([
            "assert" in full_output,
            "test" in output_lower and "case" in output_lower,
            "example" in output_lower and ("→" in full_output or "->" in full_output),
        ])
        
        if not has_verification:
            issues.append("Tidak ada test case atau verifikasi di output")
        
        return len(issues) == 0, issues

    def _compute_score(self, report: DiscriminatorReport) -> float:
        """Hitung quality score 0.0 - 1.0."""
        score = 0.0
        weights = {
            "syntax":   0.25,  # Syntax harus benar
            "pot":      0.35,  # Mathematical correctness paling penting
            "boundary": 0.10,  # Boundary handling penting tapi tidak kritis
            "scot":     0.10,  # Structured reasoning bagus tapi opsional
            "security": 0.20,  # Security / exploit safety check
        }
        
        if report.syntax_ok:
            score += weights["syntax"]
        
        if report.pot_total > 0:
            pot_ratio = report.pot_passed / report.pot_total
            score += weights["pot"] * pot_ratio
        else:
            # Tidak ada oracle → assume neutral
            score += weights["pot"] * 0.5
        
        if report.boundary_ok:
            score += weights["boundary"]
        elif len(report.boundary_issues) <= 1:
            score += weights["boundary"] * 0.5
        
        if report.scot_ok:
            score += weights["scot"]

        if report.security_ok:
            score += weights["security"]
        else:
            # Deduction based on severity of security violations
            high_count = sum(1 for v in report.security_violations if "[HIGH]" in v)
            med_count = sum(1 for v in report.security_violations if "[MEDIUM]" in v)
            penalty = (high_count * 0.10) + (med_count * 0.05)
            score += max(0.0, weights["security"] - penalty)
        
        return min(1.0, score)

    def _build_regen_prompt(self, report: DiscriminatorReport, original_request: str) -> str:
        """Buat prompt feedback spesifik untuk Generator."""
        issues_text = []
        
        if report.syntax_errors:
            issues_text.append("SYNTAX ERRORS:\n" + "\n".join(f"  • {e}" for e in report.syntax_errors))
        
        if report.pot_failures:
            issues_text.append("MATHEMATICAL ORACLE FAILURES:\n" + "\n".join(
                f"  • {f}" for f in report.pot_failures[:3]
            ))
        
        if report.boundary_issues:
            issues_text.append("BOUNDARY CONDITION GAPS:\n" + "\n".join(
                f"  • {i}" for i in report.boundary_issues[:3]
            ))

        if report.security_violations:
            issues_text.append("SECURITY VULNERABILITIES DETECTED:\n" + "\n".join(
                f"  • {v}" for v in report.security_violations[:3]
            ))

        feedback = "\n\n".join(issues_text)
        
        return f"""
[MOKO ADVERSARIAL DISCRIMINATOR FEEDBACK]
Kode yang kamu hasilkan GAGAL verifikasi matematis dengan detail berikut:

{feedback}

INSTRUKSI PERBAIKAN:
1. Perbaiki semua syntax errors yang tertulis di atas
2. Pastikan semua oracle test berikut ini LULUS:
{chr(10).join(f"   • {f}" for f in report.pot_failures[:3])}
3. Tambahkan penanganan untuk semua boundary conditions
4. Gunakan SCoT template: Sequential → Branch → Loop → Verify

REQUEST ASLI:
{original_request}

Hasilkan ulang kode yang benar secara matematis:
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL REFINER — Orchestrator loop
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialRefiner:
    """
    Orchestrator untuk Adversarial Self-Refinement Loop.
    
    Mengelola interaksi antara Generator (LLM) dan Discriminator (deterministik)
    hingga output memenuhi standar matematis atau mencapai batas iterasi.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.discriminator = Discriminator(verbose=verbose)

    def refine(
        self,
        initial_output: str,
        generator_fn: Callable[[str], str],
        original_request: str = "",
    ) -> RefinementResult:
        """
        Jalankan adversarial refinement loop.
        
        Args:
            initial_output: output awal dari Generator
            generator_fn: fungsi yang menghasilkan output baru dari prompt
            original_request: request asli user
            
        Return:
            RefinementResult dengan output final dan riwayat iterasi
        """
        current_output = initial_output
        history = []
        converged = False

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[ASR] Iteration {iteration + 1}/{self.max_iterations}")

            # Discriminator mengevaluasi
            report = self.discriminator.evaluate(current_output, original_request)
            
            history.append({
                "iteration": iteration + 1,
                "output_length": len(current_output),
                "verdict": report.verdict.value,
                "quality_score": report.quality_score,
                "pot_passed": f"{report.pot_passed}/{report.pot_total}",
            })

            if report.overall_ok:
                # Output sudah baik — selesai
                converged = True
                if self.verbose:
                    print(f"[ASR] Converged at iteration {iteration + 1} ✓")
                break

            # Output belum cukup baik — minta Generator regenerasi
            if self.verbose:
                print(f"[ASR] Score: {report.quality_score:.2f}, requesting regeneration...")
                for failure in report.pot_failures[:2]:
                    print(f"  ✗ {failure}")

            # Generate ulang dengan feedback dari Discriminator
            regen_prompt = report.regeneration_prompt
            try:
                current_output = generator_fn(regen_prompt)
            except Exception as e:
                if self.verbose:
                    print(f"[ASR] Generator error: {e}")
                break

        # Evaluasi final
        final_report = self.discriminator.evaluate(current_output, original_request)

        return RefinementResult(
            final_output=current_output,
            iterations=len(history),
            converged=converged,
            final_report=final_report,
            history=history,
        )

    def evaluate_only(self, output: str, request: str = "") -> DiscriminatorReport:
        """
        Hanya evaluasi tanpa regenerasi (gunakan untuk monitoring).
        """
        return self.discriminator.evaluate(output, request)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON & SHORTCUT
# ─────────────────────────────────────────────────────────────────────────────

_asr_instance: Optional[AdversarialRefiner] = None

def get_asr(max_iterations: int = 3, verbose: bool = False) -> AdversarialRefiner:
    """Get atau buat singleton ASR instance."""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = AdversarialRefiner(max_iterations=max_iterations, verbose=verbose)
    return _asr_instance


def quick_discriminate(code_output: str, request: str = "") -> DiscriminatorReport:
    """
    Shortcut: evaluasi output tanpa refinement loop.
    Gunakan untuk monitoring kualitas output.
    """
    return Discriminator().evaluate(code_output, request)
