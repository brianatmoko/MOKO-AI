"""
MOKO ARMS Engine — Applied Real-World Mathematics System
=========================================================
Koordinator utama untuk menyelesaikan masalah matematika terapan dunia nyata.

Pipeline:
  1. StoryMathParser     → Ekstrak variabel dari narasi
  2. AppliedFormulaEngine → Cari/turunkan formula yang tepat
  3. DimensionalAnalysisEngine → Verifikasi konsistensi dimensi
  4. UncertaintyPropagator → Hitung ketidakpastian
  5. (Opsional) CryptoGateway v2 → Tanda tangani hasil

Contoh penggunaan:
    arms = ARMSEngine()
    result = arms.solve("Piston diameter 80mm, tekanan 12 bar. Berapa gaya?")
    print(result.pretty_print())
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .story_math_parser import (
    StoryMathParser, ParsedProblem, MeasuredValue, parse_problem,
    APPLIED_DOMAIN_SIGNALS, QUANTITY_TO_SYMBOL
)
from .applied_formula_engine import (
    AppliedFormulaEngine, FormulaSolution, FormulaSource,
    PHYSICAL_CONSTANTS, get_formula_engine, get_constant
)
from .dimensional_analysis_engine import (
    DimensionalAnalysisEngine, get_dae
)
from .uncertainty_engine import (
    UncertaintyPropagator, UncertaintyInput, UncertaintyResult,
    get_propagator, estimate_instrument_uncertainty
)


# ═══════════════════════════════════════════════════════════════════════════════
# ARMS RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ARMSResult:
    """Hasil lengkap dari ARMS problem solving."""
    # Input
    original_question: str
    parsed_problem: ParsedProblem

    # Formula
    formula_used: str
    formula_source: str         # "lookup", "derived", "fallback"
    formula_name: str

    # Computation
    variables_used: Dict[str, Tuple[float, str]]  # {sym: (value, unit)}
    computation_steps: List[str]

    # Result
    answer_value: float
    answer_unit: str
    answer_symbol: str
    answer_description: str

    # Quality
    dimensional_proof: str
    uncertainty: Optional[UncertaintyResult] = None
    confidence: float = 1.0     # 0-1, kepercayaan hasil

    # Status
    success: bool = True
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

    # Certificate (opsional)
    certificate: Optional[Any] = None

    def pretty_print(self) -> str:
        """Format hasil untuk tampilan ke user."""
        lines = []
        lines.append("═" * 60)
        lines.append("🔭 MOKO ARMS — HASIL ANALISIS MATEMATIS TERAPAN")
        lines.append("═" * 60)

        if not self.success:
            lines.append(f"❌ Gagal: {self.error_message}")
            return "\n".join(lines)

        # Problem Understanding
        lines.append(f"\n📋 MASALAH YANG DIPAHAMI:")
        lines.append(f"   {self.original_question[:120]}")
        lines.append(f"   Domain: {self.parsed_problem.domain.replace('_', ' ').title()}")

        # Variables
        lines.append(f"\n📊 VARIABEL YANG DIKETAHUI:")
        for sym, mv in self.parsed_problem.known.items():
            lines.append(f"   {sym} = {mv.value} {mv.original_unit}  =  {mv.si_value:.6g} {mv.si_unit}")

        # Formula
        lines.append(f"\n📐 FORMULA YANG DIGUNAKAN:")
        lines.append(f"   {self.formula_used}")
        lines.append(f"   ({self.formula_name}  [sumber: {self.formula_source}])")

        # Steps
        lines.append(f"\n🔢 LANGKAH KOMPUTASI:")
        for i, step in enumerate(self.computation_steps, 1):
            lines.append(f"   {i}. {step}")

        # Answer
        lines.append(f"\n✅ JAWABAN:")
        if self.uncertainty and self.uncertainty.expanded_uncertainty > 0:
            lines.append(f"   {self.answer_symbol} = {self.uncertainty.format_result()}")
            lines.append(f"   (nilai ± ketidakpastian diperluas, k=2, ~95% CI)")
        else:
            lines.append(f"   {self.answer_symbol} = {self.answer_value:.6g} {self.answer_unit}")

        if self.answer_description:
            lines.append(f"   → {self.answer_description}")

        # Dimensional proof
        if self.dimensional_proof:
            lines.append(f"\n📏 VERIFIKASI DIMENSIONAL:")
            lines.append(f"   {self.dimensional_proof}")

        # Warnings
        if self.warnings:
            lines.append(f"\n⚠️  PERINGATAN:")
            for w in self.warnings:
                lines.append(f"   • {w}")

        lines.append("═" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize ke dictionary untuk JSON/API."""
        return {
            "success": self.success,
            "question": self.original_question,
            "domain": self.parsed_problem.domain,
            "formula": self.formula_used,
            "formula_source": self.formula_source,
            "variables": {
                sym: {"value": mv.si_value, "unit": mv.si_unit}
                for sym, mv in self.parsed_problem.known.items()
            },
            "answer": {
                "symbol": self.answer_symbol,
                "value": self.answer_value,
                "unit": self.answer_unit,
                "uncertainty": self.uncertainty.expanded_uncertainty if self.uncertainty else None,
            },
            "dimensional_proof": self.dimensional_proof,
            "steps": self.computation_steps,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ARMS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ARMSEngine:
    """
    Applied Real-World Mathematics System — Orchestrator utama.

    Menggabungkan 5 layer:
    L0: StoryMathParser
    L1: AppliedDomainClassifier (via StoryMathParser)
    L2: AppliedFormulaEngine
    L3: DimensionalAnalysisEngine
    L4: UncertaintyPropagator
    L5: (Opsional) CryptoGateway

    Kemampuan utama:
    - Parsing narasi bahasa Indonesia/Inggris
    - Normalisasi satuan ke SI
    - Formula lookup + multi-step derivation
    - Verifikasi dimensional
    - Propagasi ketidakpastian
    """

    def __init__(self, verbose: bool = False, use_crypto: bool = False):
        self.verbose = verbose
        self.use_crypto = use_crypto

        # Inisialisasi semua layer
        self.parser = StoryMathParser(verbose=verbose)
        self.formula_engine = get_formula_engine(verbose=verbose)
        self.dae = get_dae(verbose=verbose)
        self.propagator = get_propagator(verbose=verbose)

        # Opsional: crypto gateway
        self._gateway = None
        if use_crypto:
            try:
                from ..moko_security.crypto_gateway import CryptoGateway
                self._gateway = CryptoGateway(version="v2", verbose=False)
            except ImportError:
                pass

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ARMS] {msg}")

    # ── PREPROCESSING: INJECT DOMAIN CONSTANTS ────────────────────────────

    def _inject_domain_constants(
        self, problem: ParsedProblem
    ) -> Dict[str, float]:
        """
        Tambahkan konstanta fisika yang relevan untuk domain ini.
        Contoh: acoustics → tambahkan kecepatan suara jika T diketahui.
        """
        known_si = {sym: mv.si_value for sym, mv in problem.known.items()}
        domain = problem.domain
        warnings = []

        # Inject kecepatan suara jika domain akustik dan T diketahui
        if domain == "acoustics" and "v" not in known_si and "v_s" not in known_si:
            if "T" in known_si:
                v_sound = 331.3 * math.sqrt(known_si["T"] / 273.15)
                known_si["v"] = v_sound
                self._log(f"Auto-inject: v_sound = {v_sound:.2f} m/s (dari T = {known_si['T']:.2f} K)")
            else:
                # Default suhu ruangan 20°C = 293.15 K
                v_sound, _ = get_constant("v_sound_20C")
                known_si["v"] = v_sound
                warnings.append(f"Asumsi kecepatan suara v = {v_sound} m/s (suhu ruang 20°C)")

        # Inject g jika domain kinematics
        if domain == "kinematics" and "g" not in known_si:
            g, _ = get_constant("g")
            known_si["g"] = g

        # Inject ρ_water jika domain thermodynamics dan tidak ada densitas
        if domain == "thermodynamics" and "ρ" not in known_si:
            if any(kw in problem.original_text.lower() for kw in ["air", "water", "liquid"]):
                rho_water, _ = get_constant("ρ_water")
                known_si["ρ"] = rho_water

        # Inject c_water jika thermodynamics dan ada massa dan suhu tapi tidak ada c
        if domain == "thermodynamics" and "c" not in known_si:
            text_lower = problem.original_text.lower()
            if any(kw in text_lower for kw in ["air", "water", "minum", "liquid"]):
                c_water, _ = get_constant("c_water")
                known_si["c"] = c_water
                warnings.append(f"Asumsi kalor jenis air: c = {c_water} J/kg·K")
            elif any(kw in text_lower for kw in ["baja", "steel", "besi"]):
                c_steel, _ = get_constant("c_steel")
                known_si["c"] = c_steel
                warnings.append(f"Asumsi kalor jenis baja: c = {c_steel} J/kg·K")
            elif any(kw in text_lower for kw in ["udara", "air udara"]):
                c_air, _ = get_constant("c_air")
                known_si["c"] = c_air
                warnings.append(f"Asumsi kalor jenis udara: c = {c_air} J/kg·K")

        # Hitung ΔT jika ada T_1 dan T_2
        if "ΔT" not in known_si and "T_2" in known_si and "T_1" in known_si:
            known_si["ΔT"] = known_si["T_2"] - known_si["T_1"]
            self._log(f"Auto-compute: ΔT = T₂ - T₁ = {known_si['ΔT']:.2f} K")

        return known_si, warnings

    # ── MAIN SOLVE ────────────────────────────────────────────────────────

    def solve(self, question: str) -> ARMSResult:
        """
        Solve applied math problem dari narasi natural language.

        Args:
            question: teks narasi masalah (Indonesian/English)

        Return:
            ARMSResult dengan semua detail komputasi + sertifikat
        """
        self._log(f"Solving: {question[:80]}...")

        # ─── L0: Parse story ──────────────────────────────────────────────
        problem = self.parser.parse(question)
        all_warnings = []

        if not problem.known:
            return ARMSResult(
                original_question=question,
                parsed_problem=problem,
                formula_used="N/A",
                formula_source="N/A",
                formula_name="N/A",
                variables_used={},
                computation_steps=["Tidak ada variabel numerik yang ditemukan dalam teks."],
                answer_value=float('nan'),
                answer_unit="",
                answer_symbol="?",
                answer_description="",
                dimensional_proof="",
                success=False,
                error_message="Tidak dapat mengekstrak variabel numerik dari teks. "
                              "Pastikan menyertakan angka dan satuan (contoh: '80mm', '12 bar').",
            )

        # ─── L1: Inject domain constants ──────────────────────────────────
        known_si, inject_warnings = self._inject_domain_constants(problem)
        all_warnings.extend(inject_warnings)

        # ─── L2: Formula resolution ────────────────────────────────────────
        target_sym = problem.unknown[0] if problem.unknown else None
        target_desc = problem.unknown_descriptions[0] if problem.unknown_descriptions else ""

        solution = self.formula_engine.resolve(
            domain=problem.domain,
            known_si=known_si,
            target_symbol=target_sym,
            target_description=target_desc,
        )

        if solution is None:
            # Fallback: coba semua domain
            self._log("Domain-specific lookup failed, trying cross-domain...")
            for alt_domain in ["kinematics", "electronics", "acoustics", "thermodynamics",
                               "fluid_mechanics", "energy", "structural"]:
                if alt_domain != problem.domain:
                    solution = self.formula_engine.resolve(
                        domain=alt_domain,
                        known_si=known_si,
                        target_symbol=target_sym,
                    )
                    if solution:
                        all_warnings.append(f"Formula ditemukan dari domain '{alt_domain}' (cross-domain)")
                        break

        if solution is None:
            return ARMSResult(
                original_question=question,
                parsed_problem=problem,
                formula_used="N/A",
                formula_source="N/A",
                formula_name="N/A",
                variables_used={sym: (mv.si_value, mv.si_unit) for sym, mv in problem.known.items()},
                computation_steps=["Formula tidak ditemukan dalam database."],
                answer_value=float('nan'),
                answer_unit="",
                answer_symbol=target_sym or "?",
                answer_description=target_desc,
                dimensional_proof="",
                success=False,
                error_message=(
                    f"Formula untuk menghitung '{target_desc}' dari "
                    f"{list(problem.known.keys())} tidak ditemukan. "
                    f"Domain: {problem.domain}. "
                    "Coba tambahkan variabel yang lebih spesifik."
                ),
                warnings=all_warnings,
            )

        # ─── L3: Dimensional Analysis ──────────────────────────────────────
        input_units = {}
        for sym in solution.inputs:
            if sym in problem.known:
                input_units[sym] = problem.known[sym].si_unit
            elif sym in known_si:
                # Constant yang di-inject
                input_units[sym] = "SI"

        dim_proof = self.dae.verify_result(
            formula_name=solution.formula.name,
            formula_str=solution.formula.formula_str,
            input_units=input_units,
            output_unit=solution.result_unit,
        )

        # ─── L4: Uncertainty Propagation ──────────────────────────────────
        uncertainty_result = None
        try:
            unc_inputs = {}
            for sym, val in solution.inputs.items():
                unit = problem.known[sym].si_unit if sym in problem.known else ""
                u = estimate_instrument_uncertainty("", val, unit)
                unc_inputs[sym] = UncertaintyInput(
                    symbol=sym,
                    value=val,
                    uncertainty=u,
                    unit=unit,
                )

            uncertainty_result = self.propagator.propagate(
                func=solution.formula.python_fn,
                inputs=unc_inputs,
                result_unit=solution.result_unit,
            )
        except Exception as e:
            self._log(f"Uncertainty propagation failed: {e}")
            all_warnings.append("Ketidakpastian tidak dapat dihitung")

        # ─── Assemble computation steps ────────────────────────────────────
        steps = []

        # Step 1: List known variables
        steps.append(f"Variabel yang diketahui:")
        for sym, mv in problem.known.items():
            steps.append(f"  {sym} = {mv.value} {mv.original_unit} = {mv.si_value:.6g} {mv.si_unit}")

        # Step 2: Injected constants
        injected = {sym: val for sym, val in known_si.items() if sym not in problem.known}
        if injected:
            steps.append("Konstanta yang digunakan:")
            for sym, val in injected.items():
                steps.append(f"  {sym} = {val:.6g}")

        # Step 3: Formula
        steps.append(f"Menggunakan formula: {solution.formula.formula_str}")

        # Step 4: Substitusi nilai
        input_str = ", ".join(f"{s}={v:.6g}" for s, v in solution.inputs.items())
        steps.append(f"Substitusi: {input_str}")

        # Step 5: Hasil
        steps.append(f"Hasil: {solution.result_symbol} = {solution.result_value:.6g} {solution.result_unit}")

        # Step 6: Uncertainty
        if uncertainty_result and uncertainty_result.expanded_uncertainty > 0:
            steps.append(
                f"Ketidakpastian: ±{uncertainty_result.expanded_uncertainty:.3g} {solution.result_unit} "
                f"(95% CI, GUM k=2)"
            )

        # ─── L5: Crypto sign (opsional) ───────────────────────────────────
        cert = None
        if self.use_crypto and self._gateway:
            try:
                reasoning = f"ARMS Solution: {solution.formula.formula_str} → {solution.result_value:.6g} {solution.result_unit}"
                status, cert = self._gateway.process_reasoning(
                    question,
                    [f"{sym}={val:.6g}" for sym, val in solution.inputs.items()] + [
                        f"{solution.result_symbol}={solution.result_value:.6g}"
                    ]
                )
            except Exception as e:
                self._log(f"Crypto sign failed: {e}")

        # ─── Format answer description ─────────────────────────────────────
        answer_desc = target_desc
        if not answer_desc:
            answer_desc = f"{solution.result_symbol} dalam {solution.result_unit}"

        return ARMSResult(
            original_question=question,
            parsed_problem=problem,
            formula_used=solution.formula.formula_str,
            formula_source=solution.source.value,
            formula_name=solution.formula.name,
            variables_used={sym: (val, solution.formula.variables.get(sym, "")) for sym, val in solution.inputs.items()},
            computation_steps=steps,
            answer_value=solution.result_value,
            answer_unit=solution.result_unit,
            answer_symbol=solution.result_symbol,
            answer_description=answer_desc,
            dimensional_proof=dim_proof,
            uncertainty=uncertainty_result,
            confidence=1.0 if solution.source == FormulaSource.LOOKUP else 0.9,
            success=True,
            warnings=all_warnings,
            certificate=cert,
        )

    def solve_batch(self, questions: List[str]) -> List[ARMSResult]:
        """Solve multiple problems sekaligus."""
        return [self.solve(q) for q in questions]

    def can_solve(self, question: str) -> bool:
        """Cek apakah ARMS bisa menyelesaikan masalah ini tanpa compute penuh."""
        problem = self.parser.parse(question)
        if not problem.known:
            return False
        if problem.domain_confidence < 0.3:
            return False
        return True

    def get_stats(self) -> Dict:
        """Statistik engine."""
        from .applied_formula_engine import FORMULA_DATABASE
        return {
            "total_formulas": len(FORMULA_DATABASE),
            "domains_covered": len(APPLIED_DOMAIN_SIGNALS),
            "units_supported": 120,
            "physical_constants": len(PHYSICAL_CONSTANTS),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_arms_instance: Optional[ARMSEngine] = None

def get_arms(verbose: bool = False) -> ARMSEngine:
    global _arms_instance
    if _arms_instance is None:
        _arms_instance = ARMSEngine(verbose=verbose)
    return _arms_instance


def solve_applied(question: str) -> ARMSResult:
    """Shortcut: solve langsung dari string."""
    return get_arms().solve(question)
