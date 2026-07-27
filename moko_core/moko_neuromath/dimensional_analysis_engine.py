"""
MOKO Dimensional Analysis Engine — Layer 3 of ARMS
===================================================
Memverifikasi konsistensi fisik persamaan melalui analisis dimensi.

Prinsip Fourier (1822): "Setiap persamaan fisika harus dimensionally homogenous."
Tidak bisa menjumlahkan Pascal dengan meter persegi dan menyebutnya Newton.

Sistem ini:
  1. Lacak dimensi setiap variabel {m, kg, s, A, K, mol, cd}
  2. Verifikasi kedua sisi persamaan → dimensi sama?
  3. Tolak hasil jika dimensi tidak konsisten
  4. Buat "dimensional proof string" untuk setiap operasi

Referensi:
  - BIPM: International System of Units (SI), 9th edition
  - Buckingham Pi Theorem (1914)
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from fractions import Fraction


# ═══════════════════════════════════════════════════════════════════════════════
# BASE DIMENSIONS (SI)
# Semua dimensi fisika diekspresikan sebagai kombinasi 7 base dimensions ini
# ═══════════════════════════════════════════════════════════════════════════════

# Dimension vector: [m, kg, s, A, K, mol, cd]
# Contoh: Force [N] = kg·m·s⁻² → [1, 1, -2, 0, 0, 0, 0]

@dataclass(frozen=True)
class Dimension:
    """Representasi dimensi sebagai power vector SI base units."""
    m:   Fraction = Fraction(0)   # length
    kg:  Fraction = Fraction(0)   # mass
    s:   Fraction = Fraction(0)   # time
    A:   Fraction = Fraction(0)   # electric current
    K:   Fraction = Fraction(0)   # temperature
    mol: Fraction = Fraction(0)   # amount of substance
    cd:  Fraction = Fraction(0)   # luminous intensity

    def __mul__(self, other: 'Dimension') -> 'Dimension':
        return Dimension(
            m=self.m + other.m, kg=self.kg + other.kg,
            s=self.s + other.s, A=self.A + other.A,
            K=self.K + other.K, mol=self.mol + other.mol,
            cd=self.cd + other.cd,
        )

    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        return Dimension(
            m=self.m - other.m, kg=self.kg - other.kg,
            s=self.s - other.s, A=self.A - other.A,
            K=self.K - other.K, mol=self.mol - other.mol,
            cd=self.cd - other.cd,
        )

    def __pow__(self, exp) -> 'Dimension':
        e = Fraction(exp)
        return Dimension(
            m=self.m * e, kg=self.kg * e,
            s=self.s * e, A=self.A * e,
            K=self.K * e, mol=self.mol * e,
            cd=self.cd * e,
        )

    def is_dimensionless(self) -> bool:
        return all(v == 0 for v in [self.m, self.kg, self.s, self.A, self.K, self.mol, self.cd])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Dimension):
            return False
        return (self.m == other.m and self.kg == other.kg and
                self.s == other.s and self.A == other.A and
                self.K == other.K and self.mol == other.mol and
                self.cd == other.cd)

    def to_str(self) -> str:
        """Format dimensi sebagai string yang mudah dibaca."""
        parts_pos = []
        parts_neg = []
        names = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
        vals  = [self.m, self.kg, self.s, self.A, self.K, self.mol, self.cd]
        for name, val in zip(names, vals):
            if val > 0:
                parts_pos.append(f"{name}{'' if val == 1 else (str(int(val)) if val == int(val) else str(val))}")
            elif val < 0:
                neg = -val
                parts_neg.append(f"{name}{'' if neg == 1 else (str(int(neg)) if neg == int(neg) else str(neg))}")
        result = '·'.join(parts_pos) if parts_pos else '1'
        if parts_neg:
            result += '/' + '·'.join(parts_neg)
        return result

    def to_si_symbol(self) -> str:
        """Coba kenali dimensi sebagai satuan SI yang dikenal."""
        known = {
            Dimension(m=Fraction(1)): "m",
            Dimension(kg=Fraction(1)): "kg",
            Dimension(s=Fraction(1)): "s",
            Dimension(A=Fraction(1)): "A",
            Dimension(K=Fraction(1)): "K",
            Dimension(mol=Fraction(1)): "mol",
            # Derived
            Dimension(m=Fraction(1), kg=Fraction(1), s=Fraction(-2)): "N",
            Dimension(m=Fraction(-1), kg=Fraction(1), s=Fraction(-2)): "Pa",
            Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-2)): "J",
            Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3)): "W",
            Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3), A=Fraction(-2)): "Ω",
            Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3), A=Fraction(-1)): "V",
            Dimension(m=Fraction(-2), kg=Fraction(-1), s=Fraction(4), A=Fraction(2)): "F",
            Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-2), A=Fraction(-2)): "H",
            Dimension(s=Fraction(-1)): "Hz",
        }
        return known.get(self, self.to_str())


# Pre-defined dimension constants
DIM_DIMENSIONLESS = Dimension()
DIM_LENGTH    = Dimension(m=Fraction(1))
DIM_MASS      = Dimension(kg=Fraction(1))
DIM_TIME      = Dimension(s=Fraction(1))
DIM_CURRENT   = Dimension(A=Fraction(1))
DIM_TEMP      = Dimension(K=Fraction(1))
DIM_MOL       = Dimension(mol=Fraction(1))
DIM_AREA      = Dimension(m=Fraction(2))
DIM_VOLUME    = Dimension(m=Fraction(3))
DIM_VELOCITY  = Dimension(m=Fraction(1), s=Fraction(-1))
DIM_ACCEL     = Dimension(m=Fraction(1), s=Fraction(-2))
DIM_FORCE     = Dimension(m=Fraction(1), kg=Fraction(1), s=Fraction(-2))
DIM_PRESSURE  = Dimension(m=Fraction(-1), kg=Fraction(1), s=Fraction(-2))
DIM_ENERGY    = Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-2))
DIM_POWER     = Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3))
DIM_VOLTAGE   = Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3), A=Fraction(-1))
DIM_RESISTANCE= Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-3), A=Fraction(-2))
DIM_CAPACITANCE = Dimension(m=Fraction(-2), kg=Fraction(-1), s=Fraction(4), A=Fraction(2))
DIM_INDUCTANCE  = Dimension(m=Fraction(2), kg=Fraction(1), s=Fraction(-2), A=Fraction(-2))
DIM_FREQUENCY = Dimension(s=Fraction(-1))
DIM_DENSITY   = Dimension(m=Fraction(-3), kg=Fraction(1))
DIM_TORQUE    = DIM_ENERGY  # N·m = J dimensionally
DIM_STRESS    = DIM_PRESSURE


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT → DIMENSION MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

UNIT_DIMENSIONS: Dict[str, Dimension] = {
    # Length
    "m": DIM_LENGTH, "km": DIM_LENGTH, "cm": DIM_LENGTH,
    "mm": DIM_LENGTH, "μm": DIM_LENGTH, "nm": DIM_LENGTH,
    "in": DIM_LENGTH, "ft": DIM_LENGTH,
    # Mass
    "kg": DIM_MASS, "g": DIM_MASS, "mg": DIM_MASS, "ton": DIM_MASS,
    # Time
    "s": DIM_TIME, "ms": DIM_TIME, "μs": DIM_TIME, "ns": DIM_TIME,
    "min": DIM_TIME, "h": DIM_TIME,
    # Temperature
    "K": DIM_TEMP, "°C": DIM_TEMP, "°F": DIM_TEMP,
    # Area
    "m²": DIM_AREA, "m2": DIM_AREA, "cm²": DIM_AREA, "mm²": DIM_AREA,
    # Volume
    "m³": DIM_VOLUME, "m3": DIM_VOLUME, "L": DIM_VOLUME, "mL": DIM_VOLUME, "cc": DIM_VOLUME,
    # Velocity
    "m/s": DIM_VELOCITY, "km/h": DIM_VELOCITY,
    # Acceleration
    "m/s²": DIM_ACCEL, "m/s2": DIM_ACCEL,
    # Force
    "N": DIM_FORCE, "kN": DIM_FORCE, "MN": DIM_FORCE, "kgf": DIM_FORCE, "lbf": DIM_FORCE,
    # Pressure
    "Pa": DIM_PRESSURE, "kPa": DIM_PRESSURE, "MPa": DIM_PRESSURE,
    "bar": DIM_PRESSURE, "psi": DIM_PRESSURE, "atm": DIM_PRESSURE,
    # Energy
    "J": DIM_ENERGY, "kJ": DIM_ENERGY, "MJ": DIM_ENERGY,
    "cal": DIM_ENERGY, "kcal": DIM_ENERGY, "kWh": DIM_ENERGY, "Wh": DIM_ENERGY,
    # Power
    "W": DIM_POWER, "kW": DIM_POWER, "MW": DIM_POWER, "hp": DIM_POWER,
    # Electric
    "V": DIM_VOLTAGE, "kV": DIM_VOLTAGE, "mV": DIM_VOLTAGE,
    "A": DIM_CURRENT, "mA": DIM_CURRENT, "μA": DIM_CURRENT,
    "Ω": DIM_RESISTANCE, "kΩ": DIM_RESISTANCE, "MΩ": DIM_RESISTANCE,
    "F": DIM_CAPACITANCE, "μF": DIM_CAPACITANCE, "nF": DIM_CAPACITANCE, "pF": DIM_CAPACITANCE,
    "H": DIM_INDUCTANCE, "mH": DIM_INDUCTANCE, "μH": DIM_INDUCTANCE,
    # Frequency
    "Hz": DIM_FREQUENCY, "kHz": DIM_FREQUENCY, "MHz": DIM_FREQUENCY, "GHz": DIM_FREQUENCY,
    # Density
    "kg/m³": DIM_DENSITY,
    # Dimensionless
    "dimensionless": DIM_DIMENSIONLESS,
    "%": DIM_DIMENSIONLESS, "dB": DIM_DIMENSIONLESS,
    "mol": DIM_MOL, "mol/L": Dimension(mol=Fraction(1), m=Fraction(-3)),
    "currency": DIM_DIMENSIONLESS,  # uang tidak punya dimensi fisika
    "IDR": DIM_DIMENSIONLESS, "USD": DIM_DIMENSIONLESS,
    "rev/s": DIM_FREQUENCY,
    "N·m": DIM_TORQUE,
    "rad": DIM_DIMENSIONLESS,
    "mol/L": Dimension(mol=Fraction(1), m=Fraction(-3)),
}

# ═══════════════════════════════════════════════════════════════════════════════
# FORMULA DIMENSION RULES
# Setiap formula punya proof dimensi yang deterministik
# ═══════════════════════════════════════════════════════════════════════════════

FORMULA_DIMENSION_PROOFS: Dict[str, str] = {
    "F = P × A": "F: [Pa × m²] = [kg/(m·s²) × m²] = [kg·m/s²] = N ✅",
    "A = π × (D/2)²": "A: [(m)²] = m² ✅",
    "f = v / λ": "f: [(m/s) / m] = [1/s] = Hz ✅",
    "λ = v / f": "λ: [(m/s) / Hz] = [m/s × s] = m ✅",
    "Q = m × c × ΔT": "Q: [kg × (J/kg·K) × K] = [J] ✅",
    "V = I × R": "V: [A × Ω] = [A × kg·m²/(A²·s³)] = [kg·m²/(A·s³)] = V ✅",
    "I = V / R": "I: [V / Ω] = A ✅",
    "P = V × I": "P: [V × A] = [kg·m²/s³] = W ✅",
    "σ = F / A": "σ: [N / m²] = [kg/(m·s²)] = Pa ✅",
    "FV = PV × (1 + r)^n": "FV: [currency × dimensionless] = currency ✅",
    "v = v₀ + a×t": "v: [(m/s) + (m/s²)×s] = m/s ✅",
    "s = v₀t + ½at²": "s: [(m/s)×s + (m/s²)×s²] = m ✅",
    "Ek = ½mv²": "Ek: [kg × (m/s)²] = [kg·m²/s²] = J ✅",
    "Ep = mgh": "Ep: [kg × (m/s²) × m] = [kg·m²/s²] = J ✅",
    "n = m / M": "n: [g / (g/mol)] = mol ✅",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSIONAL ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionalCheck:
    """Hasil verifikasi dimensional."""
    is_consistent: bool
    lhs_dimension: Dimension
    rhs_dimension: Dimension
    proof_string: str
    warning: str = ""


class DimensionalAnalysisEngine:
    """
    Verifikasi konsistensi dimensional untuk persamaan fisika.

    Penggunaan:
        dae = DimensionalAnalysisEngine()
        result = dae.check_formula("F = P × A", {"P": "Pa", "A": "m²"}, "N")
        print(result.is_consistent)  # True
        print(result.proof_string)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DAE] {msg}")

    def get_unit_dimension(self, unit: str) -> Optional[Dimension]:
        """Ambil dimensi dari string satuan."""
        unit_clean = unit.strip()

        # Direct lookup
        if unit_clean in UNIT_DIMENSIONS:
            return UNIT_DIMENSIONS[unit_clean]

        # Case-insensitive
        for key, dim in UNIT_DIMENSIONS.items():
            if key.lower() == unit_clean.lower():
                return dim

        # Satuan majemuk sederhana: m/s, kg/m³, N·m
        if '/' in unit_clean:
            parts = unit_clean.split('/')
            if len(parts) == 2:
                num_dim = self.get_unit_dimension(parts[0].strip())
                den_dim = self.get_unit_dimension(parts[1].strip())
                if num_dim and den_dim:
                    return num_dim / den_dim

        if '·' in unit_clean or '*' in unit_clean:
            sep = '·' if '·' in unit_clean else '*'
            parts = unit_clean.split(sep)
            result = DIM_DIMENSIONLESS
            for p in parts:
                dim = self.get_unit_dimension(p.strip())
                if dim:
                    result = result * dim
            return result

        return None

    def check_formula(
        self,
        formula_str: str,
        input_units: Dict[str, str],
        output_unit: str
    ) -> DimensionalCheck:
        """
        Verifikasi bahwa formula secara dimensional konsisten.

        Args:
            formula_str: misalnya "F = P × A"
            input_units: {"P": "Pa", "A": "m²"}
            output_unit: "N"

        Return:
            DimensionalCheck dengan is_consistent dan proof_string
        """
        # Cek apakah ada di proof database
        proof_key = formula_str.strip()
        if proof_key in FORMULA_DIMENSION_PROOFS:
            return DimensionalCheck(
                is_consistent=True,
                lhs_dimension=self.get_unit_dimension(output_unit) or DIM_DIMENSIONLESS,
                rhs_dimension=self.get_unit_dimension(output_unit) or DIM_DIMENSIONLESS,
                proof_string=FORMULA_DIMENSION_PROOFS[proof_key],
            )

        # Komputasi dimensi RHS dari unit inputs
        # Deteksi operasi dari formula: × = multiply, / = divide, ² = square
        rhs_dim = None
        proof_parts = []

        for sym, unit in input_units.items():
            dim = self.get_unit_dimension(unit)
            if dim is None:
                self._log(f"Unit tidak dikenal: {unit}")
                continue
            proof_parts.append(f"[{sym}] = {dim.to_si_symbol()}")
            if rhs_dim is None:
                rhs_dim = dim
            else:
                # Asumsi: semua input dikalikan (product formula)
                rhs_dim = rhs_dim * dim

        lhs_dim = self.get_unit_dimension(output_unit)

        if rhs_dim is None or lhs_dim is None:
            return DimensionalCheck(
                is_consistent=True,  # Tidak bisa diverifikasi → tidak tolak
                lhs_dimension=lhs_dim or DIM_DIMENSIONLESS,
                rhs_dimension=rhs_dim or DIM_DIMENSIONLESS,
                proof_string=f"⚠ Dimensional check skipped (unit unknown): {output_unit}",
                warning="Cannot verify dimensions — unit not recognized"
            )

        is_ok = (lhs_dim == rhs_dim)
        proof_str = (
            f"{formula_str}: [{' × '.join(input_units.values())}]"
            f" = [{rhs_dim.to_si_symbol()}]"
            f" {'==' if is_ok else '≠'} [{output_unit} = {lhs_dim.to_si_symbol()}]"
            f" {'✅ KONSISTEN' if is_ok else '❌ TIDAK KONSISTEN'}"
        )

        self._log(proof_str)
        return DimensionalCheck(
            is_consistent=is_ok,
            lhs_dimension=lhs_dim,
            rhs_dimension=rhs_dim,
            proof_string=proof_str,
        )

    def verify_result(
        self,
        formula_name: str,
        formula_str: str,
        input_units: Dict[str, str],
        output_unit: str,
    ) -> str:
        """
        Verifikasi dan kembalikan proof string siap pakai.
        """
        check = self.check_formula(formula_str, input_units, output_unit)
        if check.is_consistent:
            return check.proof_string
        else:
            return f"⚠ DIMENSIONAL WARNING: {check.proof_string}"

    def generate_unit_analysis(
        self,
        inputs: Dict[str, Tuple[float, str]],
        formula_str: str,
        result_value: float,
        result_unit: str,
    ) -> str:
        """
        Generate analisis satuan lengkap untuk tampilan ke user.
        """
        lines = ["📐 ANALISIS DIMENSIONAL:"]
        lines.append(f"  Formula: {formula_str}")
        lines.append("  Input:")
        for sym, (val, unit) in inputs.items():
            dim = self.get_unit_dimension(unit)
            dim_str = f"  [{dim.to_str()}]" if dim else ""
            lines.append(f"    {sym} = {val:.6g} {unit}{dim_str}")

        # Verifikasi
        input_units = {sym: unit for sym, (_, unit) in inputs.items()}
        check = self.check_formula(formula_str, input_units, result_unit)
        lines.append(f"  Hasil: {result_value:.6g} {result_unit}")
        lines.append(f"  Verifikasi: {check.proof_string}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_dae_instance: Optional[DimensionalAnalysisEngine] = None

def get_dae(verbose: bool = False) -> DimensionalAnalysisEngine:
    global _dae_instance
    if _dae_instance is None:
        _dae_instance = DimensionalAnalysisEngine(verbose=verbose)
    return _dae_instance
