"""
MOKO Uncertainty Engine — Layer 4 of ARMS
==========================================
Propagasi ketidakpastian pengukuran sesuai standar GUM
(Guide to the Expression of Uncertainty in Measurement, BIPM 2008).

Di dunia nyata, semua pengukuran punya ketidakpastian.
"Diameter 80mm" sebenarnya adalah "80mm ± 0.1mm" (toleransi alat ukur).
Bagaimana ketidakpastian ini mempengaruhi hasil akhir?

Dua metode:
  1. Analitik (GUM Supplement 1): derivasi parsial → σ_f = √(Σ(∂f/∂xᵢ × σᵢ)²)
  2. Monte Carlo: sampling distribusi input, hitung distribusi output

Referensi:
  - BIPM/IEC/IFCC/ISO/IUPAC/IUPAP/OIML, GUM:1995 + Supplement 1 (2008)
  - NIST Technical Note 1297: Guidelines for Evaluating and Expressing the 
    Uncertainty of NIST Measurement Results
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple


@dataclass
class UncertaintyInput:
    """Input variabel dengan ketidakpastian pengukuran."""
    symbol: str
    value: float
    uncertainty: float         # Ketidakpastian standar (1σ)
    unit: str = ""
    distribution: str = "normal"  # "normal", "uniform", "triangular"

    @property
    def relative_uncertainty(self) -> float:
        """Ketidakpastian relatif (%)."""
        if self.value == 0:
            return float('inf')
        return abs(self.uncertainty / self.value) * 100


@dataclass
class UncertaintyResult:
    """Hasil propagasi ketidakpastian."""
    result_value: float
    standard_uncertainty: float    # u(y) = σ (1σ, ~68%)
    expanded_uncertainty: float    # U = k×u(y), k=2 → 95% CI
    coverage_factor: float = 2.0   # k = 2 → ~95% confidence
    unit: str = ""
    method: str = "analytic"       # "analytic" atau "monte_carlo"
    contributions: Dict[str, float] = field(default_factory=dict)  # kontribusi per variabel
    dominant_source: str = ""      # variabel yang paling mempengaruhi ketidakpastian

    @property
    def relative_uncertainty_percent(self) -> float:
        if self.result_value == 0:
            return float('inf')
        return (self.standard_uncertainty / abs(self.result_value)) * 100

    def summary(self) -> str:
        return (
            f"Hasil: {self.result_value:.6g} {self.unit}\n"
            f"Ketidakpastian standar: u(y) = ±{self.standard_uncertainty:.3g} {self.unit}\n"
            f"Ketidakpastian diperluas: U = ±{self.expanded_uncertainty:.3g} {self.unit} "
            f"(k={self.coverage_factor}, ~95% CI)\n"
            f"Ketidakpastian relatif: {self.relative_uncertainty_percent:.2f}%\n"
            f"Sumber dominan: {self.dominant_source}\n"
            f"Metode: {self.method}"
        )

    def format_result(self, sig_figs: int = 4) -> str:
        """Format hasil dengan angka penting yang tepat."""
        # Tentukan jumlah desimal dari uncertainty
        if self.standard_uncertainty > 0:
            decimals = max(0, -int(math.floor(math.log10(self.standard_uncertainty))) + 1)
        else:
            decimals = sig_figs
        fmt = f".{decimals}f"
        return f"({self.result_value:{fmt}} ± {self.expanded_uncertainty:{fmt}}) {self.unit}"


class UncertaintyPropagator:
    """
    Propagasi ketidakpastian pengukuran menggunakan metode GUM.

    Penggunaan:
        propagator = UncertaintyPropagator()

        result = propagator.propagate_analytic(
            func=lambda v: v["P"] * v["A"],
            inputs={
                "P": UncertaintyInput("P", 1.2e6, 1e4, "Pa"),   # 12 bar ± 0.1 bar
                "A": UncertaintyInput("A", 5.027e-3, 5e-5, "m²"),
            },
            result_unit="N"
        )
        print(result.summary())
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Uncertainty] {msg}")

    # ── METODE 1: ANALITIK (GUM) ──────────────────────────────────────────

    def propagate_analytic(
        self,
        func: Callable[[Dict[str, float]], float],
        inputs: Dict[str, 'UncertaintyInput'],
        result_unit: str = "",
        h_factor: float = 1e-6,  # step size untuk diferensiasi numerik
    ) -> UncertaintyResult:
        """
        Propagasi ketidakpastian dengan diferensiasi parsial numerik.

        Formula GUM:
            u²(y) = Σᵢ (∂f/∂xᵢ)² × u²(xᵢ)
            u(y)  = √(u²(y))

        ∂f/∂xᵢ dihitung numerik: (f(x+h) - f(x-h)) / (2h)
        """
        # Nilai tengah (best estimate)
        nominal = {sym: inp.value for sym, inp in inputs.items()}
        y_nominal = func(nominal)

        contributions = {}
        variance_sum = 0.0

        for sym, inp in inputs.items():
            if inp.uncertainty == 0:
                contributions[sym] = 0.0
                continue

            # Partial derivative numerik (central difference)
            h = max(abs(inp.value) * h_factor, h_factor)
            x_plus  = {s: v for s, v in nominal.items()}
            x_minus = {s: v for s, v in nominal.items()}
            x_plus[sym]  = inp.value + h
            x_minus[sym] = inp.value - h

            try:
                df_dxi = (func(x_plus) - func(x_minus)) / (2 * h)
            except Exception:
                df_dxi = 0.0

            sensitivity = df_dxi * inp.uncertainty
            variance_contrib = sensitivity ** 2
            variance_sum += variance_contrib
            contributions[sym] = abs(sensitivity)
            self._log(f"  ∂f/∂{sym} = {df_dxi:.4g}, u({sym}) = {inp.uncertainty:.4g}, contribution = {abs(sensitivity):.4g}")

        u_y = math.sqrt(variance_sum)
        k = 2.0  # coverage factor untuk ~95% CI (normal distribution)
        U_y = k * u_y

        # Tentukan sumber ketidakpastian dominan
        dominant = max(contributions, key=contributions.get) if contributions else ""

        self._log(f"u(y) = {u_y:.4g}, U(y) = ±{U_y:.4g} {result_unit}")

        return UncertaintyResult(
            result_value=y_nominal,
            standard_uncertainty=u_y,
            expanded_uncertainty=U_y,
            coverage_factor=k,
            unit=result_unit,
            method="analytic_GUM",
            contributions=contributions,
            dominant_source=dominant,
        )

    # ── METODE 2: MONTE CARLO ─────────────────────────────────────────────

    def propagate_monte_carlo(
        self,
        func: Callable[[Dict[str, float]], float],
        inputs: Dict[str, 'UncertaintyInput'],
        result_unit: str = "",
        n_samples: int = 10_000,
    ) -> UncertaintyResult:
        """
        Propagasi ketidakpastian menggunakan Monte Carlo simulation.

        Lebih akurat untuk fungsi non-linear. Lebih lambat.

        Untuk setiap sampel:
          1. Sample setiap xᵢ dari distribusinya (normal/uniform/triangular)
          2. Hitung y = f(x₁, x₂, ..., xₙ)
        Setelah N sampel:
          - u(y) = std(y_samples)
          - U(y) = 2 × u(y)  (k=2)
        """
        results = []

        for _ in range(n_samples):
            sampled = {}
            for sym, inp in inputs.items():
                if inp.distribution == "normal":
                    x = random.gauss(inp.value, inp.uncertainty)
                elif inp.distribution == "uniform":
                    x = random.uniform(inp.value - inp.uncertainty * math.sqrt(3),
                                       inp.value + inp.uncertainty * math.sqrt(3))
                elif inp.distribution == "triangular":
                    x = random.triangular(inp.value - inp.uncertainty * math.sqrt(6),
                                          inp.value + inp.uncertainty * math.sqrt(6),
                                          inp.value)
                else:
                    x = random.gauss(inp.value, inp.uncertainty)
                sampled[sym] = x

            try:
                y = func(sampled)
                if math.isfinite(y):
                    results.append(y)
            except Exception:
                pass

        if not results:
            return UncertaintyResult(
                result_value=func({sym: inp.value for sym, inp in inputs.items()}),
                standard_uncertainty=0,
                expanded_uncertainty=0,
                unit=result_unit,
                method="monte_carlo_failed",
            )

        y_mean = sum(results) / len(results)
        y_std = math.sqrt(sum((y - y_mean)**2 for y in results) / len(results))
        k = 2.0
        U = k * y_std

        self._log(f"MC: N={len(results)}, mean={y_mean:.6g}, std={y_std:.4g}")

        return UncertaintyResult(
            result_value=y_mean,
            standard_uncertainty=y_std,
            expanded_uncertainty=U,
            coverage_factor=k,
            unit=result_unit,
            method=f"monte_carlo_N={n_samples}",
        )

    # ── AUTO METHOD ───────────────────────────────────────────────────────

    def propagate(
        self,
        func: Callable[[Dict[str, float]], float],
        inputs: Dict[str, 'UncertaintyInput'],
        result_unit: str = "",
        force_mc: bool = False,
        n_mc: int = 5000,
    ) -> UncertaintyResult:
        """
        Auto-pilih metode terbaik.

        Analitik lebih cepat dan cukup untuk kebanyakan kasus.
        Monte Carlo lebih akurat untuk fungsi sangat non-linear.
        """
        # Cek apakah ada ketidakpastian yang signifikan
        has_uncertainty = any(inp.uncertainty > 0 for inp in inputs.values())
        if not has_uncertainty:
            nominal = {sym: inp.value for sym, inp in inputs.items()}
            return UncertaintyResult(
                result_value=func(nominal),
                standard_uncertainty=0.0,
                expanded_uncertainty=0.0,
                unit=result_unit,
                method="no_uncertainty",
            )

        if force_mc:
            return self.propagate_monte_carlo(func, inputs, result_unit, n_mc)
        else:
            return self.propagate_analytic(func, inputs, result_unit)


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUMENT UNCERTAINTY DATABASE
# Ketidakpastian khas untuk alat ukur umum
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_UNCERTAINTIES: Dict[str, Dict] = {
    "micrometer": {
        "unit": "m", "typical_u": 1e-5,
        "description": "Mikrometer sekrup, resolusi 0.01mm, u ≈ ±0.01mm"
    },
    "vernier_caliper": {
        "unit": "m", "typical_u": 5e-5,
        "description": "Jangka sorong 0.05mm, u ≈ ±0.05mm"
    },
    "ruler": {
        "unit": "m", "typical_u": 5e-4,
        "description": "Penggaris ±0.5mm"
    },
    "digital_thermometer": {
        "unit": "K", "typical_u": 0.5,
        "description": "Termometer digital ±0.5°C"
    },
    "pressure_gauge": {
        "unit": "Pa", "typical_u": 0.005,
        "description": "Manometer ±0.5% FS"
    },
    "multimeter_voltage": {
        "unit": "V", "typical_u": 0.005,
        "description": "Multimeter digital ±0.5% reading"
    },
    "multimeter_current": {
        "unit": "A", "typical_u": 0.01,
        "description": "Multimeter digital ±1% reading"
    },
    "digital_scale": {
        "unit": "kg", "typical_u": 5e-4,
        "description": "Timbangan digital ±0.5g"
    },
    "stopwatch": {
        "unit": "s", "typical_u": 0.01,
        "description": "Stopwatch digital ±0.01s"
    },
}


def estimate_instrument_uncertainty(quantity_type: str, value: float, unit: str) -> float:
    """
    Estimasi ketidakpastian khas berdasarkan jenis kuantitas.
    Gunakan ini ketika user tidak menyebutkan toleransi.
    """
    instrument_map = {
        "m": ("ruler", 0.001),         # ±1mm default
        "kg": ("digital_scale", 0.001), # ±1g
        "K": ("digital_thermometer", 0.5),
        "°C": ("digital_thermometer", 0.5),
        "Pa": ("pressure_gauge", value * 0.005),  # ±0.5%
        "N": (None, value * 0.01),     # ±1%
        "V": ("multimeter_voltage", value * 0.005),
        "A": ("multimeter_current", value * 0.01),
        "Ω": (None, value * 0.01),
        "Hz": (None, value * 0.001),   # ±0.1%
        "s": ("stopwatch", 0.01),
    }
    if unit in instrument_map:
        _, u = instrument_map[unit]
        return u
    return value * 0.01  # Default ±1%


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_propagator_instance: Optional[UncertaintyPropagator] = None

def get_propagator(verbose: bool = False) -> UncertaintyPropagator:
    global _propagator_instance
    if _propagator_instance is None:
        _propagator_instance = UncertaintyPropagator(verbose=verbose)
    return _propagator_instance
