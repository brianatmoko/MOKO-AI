"""
MOKO ARMS Orchestrator — Applied Real-World Math Solver (Full Pipeline)
=======================================================================
Mengintegrasikan semua komponen NeuroMath untuk menyelesaikan soal
matematika terapan nyata dari narasi bahasa manusia.

Pipeline:
  Narasi Teks
    └─▶ StoryMathParser (L0)          : ekstraksi problem terstruktur
         └─▶ AppliedFormulaEngine (L2): LOOKUP → DERIVE → SYNTHESIZE
              └─▶ ARMSSolution        : jawaban terformat + langkah kerja

Ini adalah Layer 3 dari ARMS (Applied Real-world Math Solver System).
"""

import time
import math
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class SolveStatus(Enum):
    SUCCESS     = "success"       # Jawaban lengkap ditemukan
    PARTIAL     = "partial"       # Bisa diketahui beberapa variabel, target tidak
    PARSE_FAIL  = "parse_fail"    # Parser tidak bisa mengekstrak problem
    NO_FORMULA  = "no_formula"    # Tidak ada formula yang cocok
    COMPUTE_ERR = "compute_error" # Error saat komputasi


@dataclass
class ARMSSolution:
    """Hasil penyelesaian soal oleh ARMS Orchestrator."""
    # Input
    original_text: str
    domain: str
    domain_confidence: float

    # Variabel yang diekstrak
    known_vars: Dict[str, float]       # Simbol → nilai SI
    target_symbol: str
    target_description: str

    # Hasil
    status: SolveStatus
    result_value: Optional[float]
    result_unit: str
    result_symbol: str

    # Reasoning trace
    solution_steps: List[str] = field(default_factory=list)
    formula_used: str = ""
    formula_source: str = ""           # LOOKUP / DERIVED / SYNTHESIZED

    # Metadata
    elapsed_ms: float = 0.0
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    partial_results: Dict[str, float] = field(default_factory=dict)

    def is_success(self) -> bool:
        return self.status == SolveStatus.SUCCESS

    def pretty_print(self) -> str:
        """Format jawaban yang bisa dibaca manusia."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"🔢 MOKO ARMS — Penyelesaian Soal Matematika Terapan")
        lines.append("=" * 60)
        lines.append(f"\n📝 Soal: {self.original_text[:200]}")
        lines.append(f"\n🎯 Domain: {self.domain} (keyakinan: {self.domain_confidence:.0%})")

        lines.append(f"\n📊 Variabel yang Diketahui:")
        for sym, val in self.known_vars.items():
            lines.append(f"   {sym} = {val:.6g}")

        lines.append(f"\n❓ Dicari: {self.target_description or self.target_symbol}")

        lines.append(f"\n{'✅ JAWABAN' if self.is_success() else '⚠️ STATUS'}: "
                     f"{self.status.value.upper()}")

        if self.is_success() and self.result_value is not None:
            lines.append(f"\n📌 {self.result_symbol} = {self.result_value:.6g} {self.result_unit}")
            lines.append(f"   [Sumber: {self.formula_source}]")
            if self.formula_used:
                lines.append(f"   Rumus: {self.formula_used}")

        if self.solution_steps:
            lines.append(f"\n📖 Langkah Penyelesaian:")
            for step in self.solution_steps:
                lines.append(f"   {step}")

        if self.partial_results:
            lines.append(f"\n🔍 Variabel Antara yang Ditemukan:")
            for sym, val in self.partial_results.items():
                lines.append(f"   {sym} = {val:.6g}")

        if self.warnings:
            lines.append(f"\n⚠️ Peringatan:")
            for w in self.warnings:
                lines.append(f"   • {w}")

        lines.append(f"\n⏱️  Waktu: {self.elapsed_ms:.1f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)


class ARMSOrchestrator:
    """
    Orchestrator utama ARMS — menghubungkan semua komponen pipeline.

    Urutan resolusi:
      1. StoryMathParser: ekstraksi variabel dari teks
      2. AppliedFormulaEngine: LOOKUP → DERIVE → SYNTHESIZE
      3. Format hasil menjadi ARMSSolution terstruktur
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._parser = None
        self._engine = None
        self._init_components()

    def _init_components(self):
        """Inisialisasi komponen secara lazy."""
        try:
            from moko_neuromath.story_math_parser import get_story_parser
            self._parser = get_story_parser(verbose=self.verbose)
        except Exception as e:
            self._warn(f"StoryMathParser init failed: {e}")

        try:
            from moko_neuromath.applied_formula_engine import get_formula_engine
            self._engine = get_formula_engine(verbose=self.verbose)
        except Exception as e:
            self._warn(f"AppliedFormulaEngine init failed: {e}")

    def _warn(self, msg: str):
        if self.verbose:
            print(f"  [ARMS] ⚠️  {msg}")

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [ARMS] {msg}")

    # ── MAIN SOLVE ────────────────────────────────────────────────────────

    def solve(self, text: str) -> ARMSSolution:
        """
        Entry point utama: solve soal dari teks narasi.

        Args:
            text: Teks soal matematika terapan

        Returns:
            ARMSSolution dengan hasil lengkap
        """
        t_start = time.time()
        warnings = []

        # ── Step 1: Parse teks ──────────────────────────────────────────
        if self._parser is None:
            return self._make_error_solution(text, SolveStatus.PARSE_FAIL,
                                              "StoryMathParser tidak tersedia",
                                              time.time() - t_start)

        try:
            parsed = self._parser.parse(text)
        except Exception as e:
            return self._make_error_solution(text, SolveStatus.PARSE_FAIL,
                                              f"Parse error: {e}",
                                              time.time() - t_start)

        self._log(f"Parsed: domain={parsed.domain}, known={list(parsed.known.keys())}, "
                  f"unknown={parsed.unknown}")

        # ── Step 2: Ekstrak nilai SI dari ParsedProblem ─────────────────
        known_si: Dict[str, float] = {}
        for sym, mv in parsed.known.items():
            known_si[sym] = mv.si_value

        # Tentukan target
        target_sym = parsed.unknown[0] if parsed.unknown else None
        target_desc = parsed.unknown_descriptions[0] if parsed.unknown_descriptions else ""

        if not target_sym or target_sym == "?":
            warnings.append("Target variabel tidak terdeteksi dari teks")

        # ── Step 3: Resolve dengan Formula Engine ───────────────────────
        if self._engine is None:
            return self._make_error_solution(text, SolveStatus.NO_FORMULA,
                                              "AppliedFormulaEngine tidak tersedia",
                                              time.time() - t_start)

        solution = None
        try:
            solution = self._engine.resolve(
                domain=parsed.domain,
                known_si=known_si,
                target_symbol=target_sym,
                target_description=target_desc
            )
        except Exception as e:
            warnings.append(f"Engine resolve error: {traceback.format_exc(limit=2)}")

        elapsed = (time.time() - t_start) * 1000

        # ── Step 4: Build ARMSSolution ──────────────────────────────────
        if solution is not None and not math.isnan(solution.result_value):
            return ARMSSolution(
                original_text=text,
                domain=parsed.domain,
                domain_confidence=parsed.domain_confidence,
                known_vars=known_si,
                target_symbol=target_sym or solution.result_symbol,
                target_description=target_desc,
                status=SolveStatus.SUCCESS,
                result_value=solution.result_value,
                result_unit=solution.result_unit,
                result_symbol=solution.result_symbol,
                solution_steps=solution.steps,
                formula_used=solution.formula.formula_str,
                formula_source=solution.source.value,
                elapsed_ms=elapsed,
                confidence=parsed.domain_confidence * 0.9,
                warnings=warnings,
                partial_results={k: v for k, v in known_si.items()},
            )

        # ── Fallback: jawaban partial ────────────────────────────────────
        if known_si:
            return ARMSSolution(
                original_text=text,
                domain=parsed.domain,
                domain_confidence=parsed.domain_confidence,
                known_vars=known_si,
                target_symbol=target_sym or "?",
                target_description=target_desc,
                status=SolveStatus.PARTIAL,
                result_value=None,
                result_unit="",
                result_symbol=target_sym or "?",
                solution_steps=[
                    f"Domain terdeteksi: {parsed.domain}",
                    f"Variabel yang diekstrak: {list(known_si.keys())}",
                    f"Target: {target_sym} — tidak dapat diselesaikan dengan rumus yang tersedia",
                    "Coba berikan lebih banyak variabel atau ubah pertanyaan.",
                ],
                elapsed_ms=elapsed,
                confidence=0.3,
                warnings=warnings,
                partial_results=known_si,
            )

        return self._make_error_solution(text, SolveStatus.PARSE_FAIL,
                                          "Tidak ada variabel yang diekstrak dari teks",
                                          elapsed)

    def solve_structured(
        self,
        domain: str,
        known: Dict[str, float],
        target: str,
        story_text: str = ""
    ) -> ARMSSolution:
        """
        Solve langsung dari variabel terstruktur (tanpa NLP parsing).
        Digunakan untuk evaluasi dari dataset terstruktur.

        Args:
            domain: Domain teknik
            known: Dict variabel yang diketahui {simbol: nilai SI}
            target: Simbol variabel yang dicari
            story_text: Opsional — teks soal untuk keperluan display

        Returns:
            ARMSSolution
        """
        t_start = time.time()

        if self._engine is None:
            return self._make_error_solution(story_text, SolveStatus.NO_FORMULA,
                                              "AppliedFormulaEngine tidak tersedia", 0)
        try:
            solution = self._engine.resolve(
                domain=domain,
                known_si=known,
                target_symbol=target,
            )
        except Exception as e:
            return self._make_error_solution(story_text, SolveStatus.COMPUTE_ERR,
                                              str(e), 0)

        elapsed = (time.time() - t_start) * 1000

        if solution and not math.isnan(solution.result_value):
            return ARMSSolution(
                original_text=story_text,
                domain=domain,
                domain_confidence=1.0,
                known_vars=known,
                target_symbol=target,
                target_description=target,
                status=SolveStatus.SUCCESS,
                result_value=solution.result_value,
                result_unit=solution.result_unit,
                result_symbol=solution.result_symbol,
                solution_steps=solution.steps,
                formula_used=solution.formula.formula_str,
                formula_source=solution.source.value,
                elapsed_ms=elapsed,
                confidence=0.95,
            )

        return ARMSSolution(
            original_text=story_text,
            domain=domain,
            domain_confidence=1.0,
            known_vars=known,
            target_symbol=target,
            target_description=target,
            status=SolveStatus.NO_FORMULA,
            result_value=None,
            result_unit="",
            result_symbol=target,
            solution_steps=[f"Tidak ditemukan formula untuk {target} dari {list(known.keys())}"],
            elapsed_ms=elapsed,
            confidence=0.0,
        )

    def _make_error_solution(
        self, text: str, status: SolveStatus,
        error_msg: str, elapsed: float
    ) -> ARMSSolution:
        return ARMSSolution(
            original_text=text,
            domain="unknown",
            domain_confidence=0.0,
            known_vars={},
            target_symbol="?",
            target_description="",
            status=status,
            result_value=None,
            result_unit="",
            result_symbol="?",
            solution_steps=[f"Error: {error_msg}"],
            elapsed_ms=elapsed * 1000 if elapsed < 1 else elapsed,
            confidence=0.0,
            warnings=[error_msg],
        )

    def batch_solve(self, texts: List[str]) -> List[ARMSSolution]:
        """Solve banyak soal sekaligus."""
        return [self.solve(t) for t in texts]


# ── EVALUASI JAWABAN ──────────────────────────────────────────────────────────

class AnswerEvaluator:
    """
    Membandingkan jawaban ARMS dengan jawaban referensi (ground truth).
    Digunakan dalam training loop untuk menilai kebenaran AI.
    """

    @staticmethod
    def evaluate(
        solution: ARMSSolution,
        expected_value: float,
        tolerance_pct: float = 1.0
    ) -> Tuple[bool, float, str]:
        """
        Evaluasi jawaban ARMS vs expected.

        Returns:
            (is_correct, percent_error, verdict_msg)
        """
        if not solution.is_success():
            return False, 100.0, f"ARMS gagal: {solution.status.value}"

        if solution.result_value is None:
            return False, 100.0, "Jawaban kosong"

        if expected_value == 0:
            is_ok = abs(solution.result_value) < 1e-9
            return is_ok, 0.0 if is_ok else 100.0, "OK" if is_ok else "Hasil tidak nol"

        pct_err = abs((solution.result_value - expected_value) / expected_value) * 100.0
        is_correct = pct_err <= tolerance_pct

        if is_correct:
            verdict = f"✅ BENAR (error: {pct_err:.3f}%)"
        elif pct_err <= tolerance_pct * 5:
            verdict = f"⚠️ HAMPIR BENAR (error: {pct_err:.2f}%)"
        else:
            verdict = f"❌ SALAH (error: {pct_err:.2f}%, harapan: {expected_value:.6g})"

        return is_correct, round(pct_err, 4), verdict

    @staticmethod
    def score_solution(solution: ARMSSolution, expected_value: float, tolerance_pct: float = 1.0) -> float:
        """
        Hitung skor [0.0 - 1.0] untuk jawaban ARMS.
        Skor bertingkat:
          1.0  = Jawaban tepat (dalam toleransi)
          0.5  = Jawaban partial (domain, variabel benar, tapi target tidak solved)
          0.1  = Status PARTIAL dengan variabel terekstrak
          0.0  = Gagal total
        """
        if not solution.is_success():
            if solution.status == SolveStatus.PARTIAL and solution.partial_results:
                return 0.1
            return 0.0

        if solution.result_value is None:
            return 0.0

        is_correct, pct_err, _ = AnswerEvaluator.evaluate(solution, expected_value, tolerance_pct)

        if is_correct:
            return 1.0
        elif pct_err <= tolerance_pct * 10:
            return 0.7  # Hampir benar
        elif pct_err <= tolerance_pct * 50:
            return 0.3  # Agak jauh
        else:
            return 0.05  # Salah tapi ada upaya


# ── SINGLETON ──────────────────────────────────────────────────────────────────

_orchestrator_instance: Optional[ARMSOrchestrator] = None

def get_orchestrator(verbose: bool = False) -> ARMSOrchestrator:
    """Return singleton ARMS Orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ARMSOrchestrator(verbose=verbose)
    return _orchestrator_instance


if __name__ == "__main__":
    orch = ARMSOrchestrator(verbose=True)

    test_cases = [
        "Piston diameter 80mm pada tekanan 12 bar. Berapa gaya yang dihasilkan?",
        "Suhu udara 25°C, berapakah kecepatan suara di udara?",
        "Modal Rp 10.000.000 dengan bunga 8% per tahun selama 5 tahun. Hitunglah FV.",
        "Mesin 4 silinder, bore 80mm, stroke 90mm. Berapa kapasitas mesin dalam cc?",
    ]

    for text in test_cases:
        print(f"\n{'='*70}")
        sol = orch.solve(text)
        print(sol.pretty_print())
