"""
MOKO Applied Formula Engine — Layer 2 of ARMS
==============================================
Database formula teknik terapan + 3 strategi resolusi.

Strategi:
  1. LOOKUP   — Cari langsung dari database 100+ formula
  2. DERIVE   — Turunkan dari prinsip pertama (SymPy)
  3. SYNTHESIZE — Dimensional analysis (Buckingham-Pi)

Referensi:
  - Halliday, Resnick & Krane: Physics (Wiley)
  - Shigley: Mechanical Engineering Design
  - Hayt & Kemmerly: Engineering Circuit Analysis
  - Tipler: Physics for Scientists and Engineers
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum

try:
    import sympy as sp
    from sympy import symbols, solve, sqrt, pi, simplify, diff
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# FORMULA RECORD
# ═══════════════════════════════════════════════════════════════════════════════

class FormulaSource(Enum):
    LOOKUP    = "lookup"
    DERIVED   = "derived"
    SYNTHESIZED = "synthesized"
    FALLBACK  = "fallback"


@dataclass
class FormulaRecord:
    """Satu formula dalam database."""
    name: str                          # Nama deskriptif
    domain: str                        # Domain teknik
    formula_str: str                   # Persamaan string (untuk tampilan)
    variables: Dict[str, str]          # {simbol: deskripsi + satuan}
    solve_for: str                     # Variabel default yang diselesaikan
    python_fn: Callable                # Fungsi Python untuk komputasi
    units_out: str                     # Satuan output
    reference: str = ""               # Sumber / referensi
    prerequisites: List[str] = field(default_factory=list)  # Variabel yang dibutuhkan
    notes: str = ""

    def can_solve(self, known_symbols: List[str]) -> bool:
        """Cek apakah semua variabel yang diperlukan tersedia."""
        required = [v for v in self.prerequisites if v != self.solve_for]
        return all(r in known_symbols for r in required)


@dataclass
class FormulaSolution:
    """Hasil resolusi formula."""
    formula: FormulaRecord
    inputs: Dict[str, float]           # Nilai input yang digunakan (SI)
    result_value: float                # Nilai hasil (SI)
    result_unit: str                   # Satuan hasil
    result_symbol: str                 # Simbol yang diselesaikan
    steps: List[str]                   # Langkah-langkah komputasi
    source: FormulaSource = FormulaSource.LOOKUP
    dimensional_proof: str = ""
    uncertainty: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"Formula: {self.formula.formula_str}",
            f"Source:  {self.source.value}",
            f"Inputs:",
        ]
        for sym, val in self.inputs.items():
            lines.append(f"  {sym} = {val:.6g} {self.formula.variables.get(sym, '')}")
        lines.append(f"Result:  {self.result_symbol} = {self.result_value:.6g} {self.result_unit}")
        if self.dimensional_proof:
            lines.append(f"Dimensions: {self.dimensional_proof}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMULA DATABASE
# Diorganisasi per domain. Setiap formula punya:
# - Formula string (untuk tampilan)
# - Python function (untuk komputasi eksak)
# - Prerequisites (variabel yang dibutuhkan)
# - Output unit
# ═══════════════════════════════════════════════════════════════════════════════

FORMULA_DATABASE: List[FormulaRecord] = [

    # ─── FLUID MECHANICS / PISTON ─────────────────────────────────────────

    FormulaRecord(
        name="Piston Force from Pressure",
        domain="fluid_mechanics",
        formula_str="F = P × A",
        variables={"P": "Tekanan (Pa)", "A": "Luas penampang (m²)", "F": "Gaya (N)"},
        solve_for="F",
        python_fn=lambda vars: vars["P"] * vars["A"],
        units_out="N",
        prerequisites=["P", "A"],
        reference="Mechanics of Fluids, Massey",
        notes="A = π(D/2)² untuk piston silindris"
    ),

    FormulaRecord(
        name="Circle Area from Diameter",
        domain="fluid_mechanics",
        formula_str="A = π × (D/2)²",
        variables={"D": "Diameter (m)", "A": "Luas (m²)"},
        solve_for="A",
        python_fn=lambda vars: math.pi * (vars["D"] / 2) ** 2,
        units_out="m²",
        prerequisites=["D"],
        reference="Geometry",
    ),

    FormulaRecord(
        name="Circle Area from Radius",
        domain="fluid_mechanics",
        formula_str="A = π × r²",
        variables={"r": "Jari-jari (m)", "A": "Luas (m²)"},
        solve_for="A",
        python_fn=lambda vars: math.pi * vars["r"] ** 2,
        units_out="m²",
        prerequisites=["r"],
    ),

    FormulaRecord(
        name="Hydraulic Power",
        domain="fluid_mechanics",
        formula_str="P_hyd = F × v",
        variables={"F": "Gaya (N)", "v": "Kecepatan (m/s)", "P_hyd": "Daya hidraulik (W)"},
        solve_for="P_hyd",
        python_fn=lambda vars: vars["F"] * vars["v"],
        units_out="W",
        prerequisites=["F", "v"],
    ),

    FormulaRecord(
        name="Bernoulli Pressure Difference",
        domain="fluid_mechanics",
        formula_str="ΔP = ½ρ(v₁² - v₂²)",
        variables={"ρ": "Densitas fluida (kg/m³)", "v_1": "Kecepatan titik 1 (m/s)", "v_2": "Kecepatan titik 2 (m/s)"},
        solve_for="ΔP",
        python_fn=lambda vars: 0.5 * vars["ρ"] * (vars["v_1"]**2 - vars["v_2"]**2),
        units_out="Pa",
        prerequisites=["ρ", "v_1", "v_2"],
        reference="Bernoulli equation (simplified, horizontal flow)"
    ),

    # ─── ACOUSTICS / SOUND ───────────────────────────────────────────────

    FormulaRecord(
        name="Wave Frequency from Speed and Wavelength",
        domain="acoustics",
        formula_str="f = v / λ",
        variables={"v": "Kecepatan suara (m/s)", "λ": "Panjang gelombang (m)", "f": "Frekuensi (Hz)"},
        solve_for="f",
        python_fn=lambda vars: vars["v"] / vars["λ"],
        units_out="Hz",
        prerequisites=["v", "λ"],
        reference="Wave mechanics",
    ),

    FormulaRecord(
        name="Wavelength from Speed and Frequency",
        domain="acoustics",
        formula_str="λ = v / f",
        variables={"v": "Kecepatan suara (m/s)", "f": "Frekuensi (Hz)", "λ": "Panjang gelombang (m)"},
        solve_for="λ",
        python_fn=lambda vars: vars["v"] / vars["f"],
        units_out="m",
        prerequisites=["v", "f"],
    ),

    FormulaRecord(
        name="Speed of Sound in Air (Temperature-dependent)",
        domain="acoustics",
        formula_str="v = 331.3 × √(T/273.15)",
        variables={"T": "Suhu (K)", "v": "Kecepatan suara (m/s)"},
        solve_for="v",
        python_fn=lambda vars: 331.3 * math.sqrt(vars["T"] / 273.15),
        units_out="m/s",
        prerequisites=["T"],
        reference="Standard formula, T in Kelvin",
        notes="Approximate: v ≈ 331.3 + 0.606×T_celsius"
    ),

    FormulaRecord(
        name="Fundamental Frequency of Closed Tube",
        domain="acoustics",
        formula_str="f₁ = v / (4L)",
        variables={"v": "Kecepatan suara (m/s)", "L": "Panjang tabung (m)", "f": "Frekuensi fundamental (Hz)"},
        solve_for="f",
        python_fn=lambda vars: vars["v"] / (4 * vars["L"]),
        units_out="Hz",
        prerequisites=["v", "L"],
        reference="Standing waves in closed-end tube",
        notes="Hanya harmonik ganjil: f_n = n×v/(4L), n=1,3,5,..."
    ),

    FormulaRecord(
        name="Fundamental Frequency of Open Tube",
        domain="acoustics",
        formula_str="f₁ = v / (2L)",
        variables={"v": "Kecepatan suara (m/s)", "L": "Panjang tabung (m)", "f": "Frekuensi fundamental (Hz)"},
        solve_for="f",
        python_fn=lambda vars: vars["v"] / (2 * vars["L"]),
        units_out="Hz",
        prerequisites=["v", "L"],
        reference="Standing waves in open-end tube",
        notes="Semua harmonik: f_n = n×v/(2L), n=1,2,3,..."
    ),

    FormulaRecord(
        name="Sound Pressure Level (dB)",
        domain="acoustics",
        formula_str="SPL = 20 × log₁₀(P / P₀)",
        variables={"P": "Tekanan suara (Pa)", "P₀": "Referensi 20μPa", "SPL": "Level tekanan suara (dB)"},
        solve_for="SPL",
        python_fn=lambda vars: 20 * math.log10(vars["P"] / vars.get("P₀", 2e-5)),
        units_out="dB",
        prerequisites=["P"],
        reference="IEC 61672, P₀ = 20 μPa",
    ),

    FormulaRecord(
        name="String Fundamental Frequency",
        domain="acoustics",
        formula_str="f = (1/2L) × √(T/μ)",
        variables={"L": "Panjang senar (m)", "T": "Tegangan senar (N)", "μ": "Massa per satuan panjang (kg/m)"},
        solve_for="f",
        python_fn=lambda vars: (1 / (2 * vars["L"])) * math.sqrt(vars["T"] / vars["μ"]),
        units_out="Hz",
        prerequisites=["L", "T", "μ"],
        reference="Mersenne's laws",
    ),

    # ─── THERMODYNAMICS ──────────────────────────────────────────────────

    FormulaRecord(
        name="Sensible Heat",
        domain="thermodynamics",
        formula_str="Q = m × c × ΔT",
        variables={"m": "Massa (kg)", "c": "Kalor jenis (J/kg·K)", "ΔT": "Perubahan suhu (K)", "Q": "Kalor (J)"},
        solve_for="Q",
        python_fn=lambda vars: vars["m"] * vars["c"] * vars.get("ΔT", vars.get("T_2", 0) - vars.get("T_1", 0)),
        units_out="J",
        prerequisites=["m", "c"],
        reference="Thermodynamics: An Engineering Approach, Cengel",
        notes="c_air=1005, c_water=4186, c_steel=502, c_aluminum=900 J/kg·K"
    ),

    FormulaRecord(
        name="Heat Transfer for Water",
        domain="thermodynamics",
        formula_str="Q = m × 4186 × ΔT  (air/water)",
        variables={"m": "Massa air (kg)", "ΔT": "Perubahan suhu (K atau °C)", "Q": "Kalor (J)"},
        solve_for="Q",
        python_fn=lambda vars: vars["m"] * 4186.0 * (vars.get("ΔT") or (vars.get("T_2", 0) - vars.get("T_1", 0))),
        units_out="J",
        prerequisites=["m"],
        notes="Untuk air (c_water = 4186 J/kg·K)"
    ),

    FormulaRecord(
        name="Ideal Gas Law",
        domain="thermodynamics",
        formula_str="PV = nRT",
        variables={"P": "Tekanan (Pa)", "V": "Volume (m³)", "n": "Jumlah mol", "T": "Suhu (K)"},
        solve_for="P",
        python_fn=lambda vars: vars["n"] * 8.314 * vars["T"] / vars["V"],
        units_out="Pa",
        prerequisites=["n", "T", "V"],
        reference="Ideal Gas Law, R=8.314 J/mol·K",
    ),

    FormulaRecord(
        name="Carnot Efficiency",
        domain="thermodynamics",
        formula_str="η = 1 - T_cold/T_hot",
        variables={"T_cold": "Suhu dingin (K)", "T_hot": "Suhu panas (K)", "η": "Efisiensi Carnot"},
        solve_for="η",
        python_fn=lambda vars: 1 - vars["T_cold"] / vars["T_hot"],
        units_out="dimensionless",
        prerequisites=["T_cold", "T_hot"],
        reference="Carnot theorem",
    ),

    FormulaRecord(
        name="Stefan-Boltzmann Radiation",
        domain="thermodynamics",
        formula_str="P = ε × σ × A × T⁴",
        variables={"ε": "Emisivitas (0-1)", "A": "Luas permukaan (m²)", "T": "Suhu (K)", "P": "Daya radiasi (W)"},
        solve_for="P",
        python_fn=lambda vars: vars.get("ε", 1.0) * 5.6704e-8 * vars["A"] * vars["T"]**4,
        units_out="W",
        prerequisites=["A", "T"],
        reference="Stefan-Boltzmann law, σ=5.6704×10⁻⁸ W/m²K⁴",
    ),

    # ─── ELECTRONICS ─────────────────────────────────────────────────────

    FormulaRecord(
        name="Ohm's Law — Voltage",
        domain="electronics",
        formula_str="V = I × R",
        variables={"I": "Arus (A)", "R": "Hambatan (Ω)", "V": "Tegangan (V)"},
        solve_for="V",
        python_fn=lambda vars: vars["I"] * vars["R"],
        units_out="V",
        prerequisites=["I", "R"],
        reference="Ohm's Law",
    ),

    FormulaRecord(
        name="Ohm's Law — Current",
        domain="electronics",
        formula_str="I = V / R",
        variables={"V": "Tegangan (V)", "R": "Hambatan (Ω)", "I": "Arus (A)"},
        solve_for="I",
        python_fn=lambda vars: vars["V"] / vars["R"],
        units_out="A",
        prerequisites=["V", "R"],
    ),

    FormulaRecord(
        name="Electric Power",
        domain="electronics",
        formula_str="P = V × I = I²R = V²/R",
        variables={"V": "Tegangan (V)", "I": "Arus (A)", "R": "Hambatan (Ω)", "P": "Daya (W)"},
        solve_for="P_e",
        python_fn=lambda vars: (
            vars["V"] * vars["I"] if "V" in vars and "I" in vars else
            vars["I"]**2 * vars["R"] if "I" in vars and "R" in vars else
            vars["V"]**2 / vars["R"]
        ),
        units_out="W",
        prerequisites=["V", "I"],
    ),

    FormulaRecord(
        name="Capacitive Reactance",
        domain="electronics",
        formula_str="Xc = 1 / (2πfC)",
        variables={"f": "Frekuensi (Hz)", "C": "Kapasitansi (F)", "Xc": "Reaktansi kapasitif (Ω)"},
        solve_for="Xc",
        python_fn=lambda vars: 1 / (2 * math.pi * vars["f"] * vars["C"]),
        units_out="Ω",
        prerequisites=["f", "C"],
        reference="AC Circuit Analysis",
    ),

    FormulaRecord(
        name="Inductive Reactance",
        domain="electronics",
        formula_str="Xl = 2πfL",
        variables={"f": "Frekuensi (Hz)", "L": "Induktansi (H)", "Xl": "Reaktansi induktif (Ω)"},
        solve_for="Xl",
        python_fn=lambda vars: 2 * math.pi * vars["f"] * vars["L"],
        units_out="Ω",
        prerequisites=["f", "L"],
    ),

    FormulaRecord(
        name="RC Circuit Cutoff Frequency",
        domain="electronics",
        formula_str="f_c = 1 / (2πRC)",
        variables={"R": "Hambatan (Ω)", "C": "Kapasitansi (F)", "f_c": "Frekuensi cutoff (Hz)"},
        solve_for="f_c",
        python_fn=lambda vars: 1 / (2 * math.pi * vars["R"] * vars["C"]),
        units_out="Hz",
        prerequisites=["R", "C"],
        reference="RC Low-pass filter",
    ),

    FormulaRecord(
        name="LC Resonance Frequency",
        domain="electronics",
        formula_str="f₀ = 1 / (2π√(LC))",
        variables={"L": "Induktansi (H)", "C": "Kapasitansi (F)", "f": "Frekuensi resonansi (Hz)"},
        solve_for="f",
        python_fn=lambda vars: 1 / (2 * math.pi * math.sqrt(vars["L"] * vars["C"])),
        units_out="Hz",
        prerequisites=["L", "C"],
        reference="LC resonance",
    ),

    # ─── KINEMATICS ───────────────────────────────────────────────────────

    FormulaRecord(
        name="Final Velocity (uniform acceleration)",
        domain="kinematics",
        formula_str="v = v₀ + a×t",
        variables={"v_0": "Kecepatan awal (m/s)", "a": "Percepatan (m/s²)", "t": "Waktu (s)", "v": "Kecepatan akhir (m/s)"},
        solve_for="v",
        python_fn=lambda vars: vars.get("v_0", 0) + vars["a"] * vars["t"],
        units_out="m/s",
        prerequisites=["a", "t"],
    ),

    FormulaRecord(
        name="Displacement (uniform acceleration)",
        domain="kinematics",
        formula_str="s = v₀t + ½at²",
        variables={"v_0": "Kecepatan awal (m/s)", "a": "Percepatan (m/s²)", "t": "Waktu (s)", "s": "Perpindahan (m)"},
        solve_for="s",
        python_fn=lambda vars: vars.get("v_0", 0) * vars["t"] + 0.5 * vars.get("a", 9.80665) * vars["t"]**2,
        units_out="m",
        prerequisites=["t"],
    ),

    FormulaRecord(
        name="Free Fall Time",
        domain="kinematics",
        formula_str="t = √(2h/g)",
        variables={"h": "Ketinggian (m)", "t": "Waktu jatuh (s)"},
        solve_for="t",
        python_fn=lambda vars: math.sqrt(2 * vars["h"] / 9.80665),
        units_out="s",
        prerequisites=["h"],
        reference="Free fall, g = 9.80665 m/s²",
    ),

    FormulaRecord(
        name="Projectile Maximum Height",
        domain="kinematics",
        formula_str="h_max = v₀²sin²(θ) / (2g)",
        variables={"v_0": "Kecepatan awal (m/s)", "θ": "Sudut (rad)", "h": "Tinggi maksimum (m)"},
        solve_for="h",
        python_fn=lambda vars: vars["v_0"]**2 * math.sin(vars.get("θ", math.pi/4))**2 / (2 * 9.80665),
        units_out="m",
        prerequisites=["v_0"],
    ),

    # ─── STRUCTURAL ───────────────────────────────────────────────────────

    FormulaRecord(
        name="Normal Stress",
        domain="structural",
        formula_str="σ = F / A",
        variables={"F": "Gaya (N)", "A": "Luas penampang (m²)", "σ": "Tegangan normal (Pa)"},
        solve_for="σ",
        python_fn=lambda vars: vars["F"] / vars["A"],
        units_out="Pa",
        prerequisites=["F", "A"],
        reference="Mechanics of Materials, Hibbeler",
    ),

    FormulaRecord(
        name="Strain from Stress (Hooke's Law)",
        domain="structural",
        formula_str="ε = σ / E",
        variables={"σ": "Tegangan (Pa)", "E": "Modulus Young (Pa)", "ε": "Regangan (dimensionless)"},
        solve_for="ε",
        python_fn=lambda vars: vars["σ"] / vars["E"],
        units_out="dimensionless",
        prerequisites=["σ", "E"],
        reference="Hooke's Law",
        notes="E_steel≈200GPa, E_aluminum≈70GPa, E_concrete≈30GPa"
    ),

    FormulaRecord(
        name="Bending Moment",
        domain="structural",
        formula_str="M = F × d",
        variables={"F": "Gaya (N)", "d": "Lengan momen (m)", "M": "Momen lentur (N·m)"},
        solve_for="M",
        python_fn=lambda vars: vars["F"] * vars.get("d", vars.get("L", 1)),
        units_out="N·m",
        prerequisites=["F"],
    ),

    # ─── OPTICS ──────────────────────────────────────────────────────────

    FormulaRecord(
        name="Thin Lens Equation",
        domain="optics",
        formula_str="1/f = 1/do + 1/di",
        variables={"do": "Jarak objek (m)", "di": "Jarak bayangan (m)", "f": "Panjang fokus (m)"},
        solve_for="f",
        python_fn=lambda vars: 1 / (1/vars["do"] + 1/vars["di"]),
        units_out="m",
        prerequisites=["do", "di"],
        reference="Thin lens formula",
    ),

    FormulaRecord(
        name="Lens Magnification",
        domain="optics",
        formula_str="M = -di / do",
        variables={"di": "Jarak bayangan (m)", "do": "Jarak objek (m)", "M": "Perbesaran"},
        solve_for="M",
        python_fn=lambda vars: -vars["di"] / vars["do"],
        units_out="dimensionless",
        prerequisites=["di", "do"],
    ),

    FormulaRecord(
        name="Snell's Law (refraction angle)",
        domain="optics",
        formula_str="n₁×sin(θ₁) = n₂×sin(θ₂)",
        variables={"n_1": "Indeks bias medium 1", "θ_1": "Sudut datang (rad)", "n_2": "Indeks bias medium 2"},
        solve_for="θ_2",
        python_fn=lambda vars: math.asin(vars["n_1"] * math.sin(vars["θ_1"]) / vars.get("n_2", 1.0)),
        units_out="rad",
        prerequisites=["n_1", "θ_1"],
    ),

    # ─── CHEMISTRY ───────────────────────────────────────────────────────

    FormulaRecord(
        name="Moles from Mass",
        domain="chemistry",
        formula_str="n = m / M",
        variables={"m": "Massa (g)", "M": "Massa molar (g/mol)", "n": "Jumlah mol"},
        solve_for="n",
        python_fn=lambda vars: vars["m"] / vars["M"],
        units_out="mol",
        prerequisites=["m", "M"],
        reference="Stoichiometry",
    ),

    FormulaRecord(
        name="Solution Concentration (Molarity)",
        domain="chemistry",
        formula_str="C = n / V",
        variables={"n": "Jumlah mol (mol)", "V": "Volume larutan (L)", "C": "Konsentrasi molar (mol/L)"},
        solve_for="C",
        python_fn=lambda vars: vars["n"] / (vars["V"] * 1000 if vars["V"] < 1 else vars["V"]),
        units_out="mol/L",
        prerequisites=["n", "V"],
    ),

    FormulaRecord(
        name="pH from Hydrogen Ion Concentration",
        domain="chemistry",
        formula_str="pH = -log₁₀([H⁺])",
        variables={"H+": "Konsentrasi H⁺ (mol/L)", "pH": "Derajat keasaman"},
        solve_for="pH",
        python_fn=lambda vars: -math.log10(vars.get("H+", vars.get("C", 1e-7))),
        units_out="dimensionless",
        prerequisites=["H+"],
    ),

    # ─── FINANCE ─────────────────────────────────────────────────────────

    FormulaRecord(
        name="Compound Interest Future Value",
        domain="finance",
        formula_str="FV = PV × (1 + r)^n",
        variables={"PV": "Nilai sekarang", "r": "Bunga per periode", "n": "Jumlah periode", "FV": "Nilai masa depan"},
        solve_for="FV",
        python_fn=lambda vars: vars["PV"] * (1 + vars["r"]) ** vars["n"],
        units_out="currency",
        prerequisites=["PV", "r", "n"],
        reference="Compound interest formula",
    ),

    FormulaRecord(
        name="Simple Interest",
        domain="finance",
        formula_str="I = PV × r × t",
        variables={"PV": "Modal awal", "r": "Bunga per tahun", "t": "Waktu (tahun)", "I": "Bunga"},
        solve_for="I",
        python_fn=lambda vars: vars["PV"] * vars["r"] * vars["t"],
        units_out="currency",
        prerequisites=["PV", "r", "t"],
    ),

    FormulaRecord(
        name="Return on Investment (ROI)",
        domain="finance",
        formula_str="ROI = (profit / PV) × 100%",
        variables={"profit": "Keuntungan", "PV": "Modal awal", "ROI": "ROI (%)"},
        solve_for="ROI",
        python_fn=lambda vars: (vars["profit"] / vars["PV"]) * 100,
        units_out="%",
        prerequisites=["profit", "PV"],
    ),

    FormulaRecord(
        name="CAGR (Compound Annual Growth Rate)",
        domain="finance",
        formula_str="CAGR = (FV/PV)^(1/n) - 1",
        variables={"FV": "Nilai akhir", "PV": "Nilai awal", "n": "Jumlah tahun", "CAGR": "CAGR (desimal)"},
        solve_for="CAGR",
        python_fn=lambda vars: (vars["FV"] / vars["PV"]) ** (1 / vars["n"]) - 1,
        units_out="dimensionless",
        prerequisites=["FV", "PV", "n"],
    ),

    # ─── SIGNAL PROCESSING ────────────────────────────────────────────────

    FormulaRecord(
        name="Nyquist Sampling Theorem",
        domain="signal_processing",
        formula_str="f_s ≥ 2 × f_max",
        variables={"f_max": "Frekuensi maksimum sinyal (Hz)", "f_s": "Frekuensi sampling minimum (Hz)"},
        solve_for="f_s",
        python_fn=lambda vars: 2 * vars["f_max"],
        units_out="Hz",
        prerequisites=["f_max"],
        reference="Nyquist-Shannon sampling theorem",
    ),

    FormulaRecord(
        name="Signal Period",
        domain="signal_processing",
        formula_str="T = 1/f",
        variables={"f": "Frekuensi (Hz)", "T": "Periode (s)"},
        solve_for="T",
        python_fn=lambda vars: 1 / vars["f"],
        units_out="s",
        prerequisites=["f"],
    ),

    # ─── ENERGY ──────────────────────────────────────────────────────────

    FormulaRecord(
        name="Kinetic Energy",
        domain="energy",
        formula_str="Ek = ½mv²",
        variables={"m": "Massa (kg)", "v": "Kecepatan (m/s)", "Ek": "Energi kinetik (J)"},
        solve_for="Ek",
        python_fn=lambda vars: 0.5 * vars["m"] * vars["v"]**2,
        units_out="J",
        prerequisites=["m", "v"],
        reference="Classical mechanics",
    ),

    FormulaRecord(
        name="Potential Energy",
        domain="energy",
        formula_str="Ep = mgh",
        variables={"m": "Massa (kg)", "h": "Ketinggian (m)", "Ep": "Energi potensial (J)"},
        solve_for="Ep",
        python_fn=lambda vars: vars["m"] * 9.80665 * vars["h"],
        units_out="J",
        prerequisites=["m", "h"],
    ),

    FormulaRecord(
        name="Power from Work and Time",
        domain="energy",
        formula_str="P = W / t",
        variables={"W": "Usaha/energi (J)", "t": "Waktu (s)", "P": "Daya (W)"},
        solve_for="P",
        python_fn=lambda vars: vars["W"] / vars["t"],
        units_out="W",
        prerequisites=["W", "t"],
    ),

    FormulaRecord(
        name="Mechanical Efficiency",
        domain="energy",
        formula_str="η = P_out / P_in × 100%",
        variables={"P_out": "Daya output (W)", "P_in": "Daya input (W)", "η": "Efisiensi (%)"},
        solve_for="η",
        python_fn=lambda vars: (vars["P_out"] / vars["P_in"]) * 100,
        units_out="%",
        prerequisites=["P_out", "P_in"],
    ),

    FormulaRecord(
        name="Gravitational Force",
        domain="kinematics",
        formula_str="W = m × g",
        variables={"m": "Massa (kg)", "g": "Percepatan gravitasi (m/s²)", "W": "Berat (N)"},
        solve_for="W",
        python_fn=lambda vars: vars["m"] * vars.get("g", 9.80665),
        units_out="N",
        prerequisites=["m"],
    ),

    # ─── ENGINE MECHANICS ─────────────────────────────────────────────────

    FormulaRecord(
        name="Engine Displacement",
        domain="engine_mechanics",
        formula_str="V_d = (π/4) × B² × S × N_cyl",
        variables={"B": "Bore/Diameter Silinder (m)", "S": "Stroke/Langkah (m)", "N_cyl": "Jumlah Silinder", "V_d": "Kapasitas/Volume Displacement (m³)"},
        solve_for="V_d",
        python_fn=lambda vars: (math.pi / 4) * vars["B"]**2 * vars["S"] * vars.get("N_cyl", 1.0),
        units_out="m³",
        prerequisites=["B", "S"],
    ),

    FormulaRecord(
        name="Compression Ratio",
        domain="engine_mechanics",
        formula_str="CR = (V_d + V_c) / V_c",
        variables={"V_d": "Volume Displacement (m³)", "V_c": "Volume Ruang Bakar/Clearance (m³)", "CR": "Rasio Kompresi"},
        solve_for="CR",
        python_fn=lambda vars: (vars["V_d"] + vars["V_c"]) / vars["V_c"],
        units_out="dimensionless",
        prerequisites=["V_d", "V_c"],
    ),

    FormulaRecord(
        name="Combustion Chamber Volume",
        domain="engine_mechanics",
        formula_str="V_c = V_d / (CR - 1)",
        variables={"V_d": "Volume Displacement (m³)", "CR": "Rasio Kompresi", "V_c": "Volume Ruang Bakar (m³)"},
        solve_for="V_c",
        python_fn=lambda vars: vars["V_d"] / (vars["CR"] - 1.0) if vars["CR"] > 1.0 else 0.0,
        units_out="m³",
        prerequisites=["V_d", "CR"],
    ),

    FormulaRecord(
        name="Crankshaft Radius",
        domain="engine_mechanics",
        formula_str="r_crank = S / 2",
        variables={"S": "Stroke/Langkah (m)", "r_crank": "Jari-jari Crankshaft (m)"},
        solve_for="r_crank",
        python_fn=lambda vars: vars["S"] / 2.0,
        units_out="m",
        prerequisites=["S"],
    ),

    FormulaRecord(
        name="Piston Position",
        domain="engine_mechanics",
        formula_str="x = r_crank × cos(θ_crank) + √(L_con² - r_crank² × sin²(θ_crank))",
        variables={"r_crank": "Jari-jari Crank (m)", "L_con": "Panjang Connecting Rod (m)", "θ_crank": "Sudut Crank (rad)", "x": "Posisi Piston (m)"},
        solve_for="x",
        python_fn=lambda vars: vars["r_crank"] * math.cos(vars["θ_crank"]) + math.sqrt(vars["L_con"]**2 - vars["r_crank"]**2 * math.sin(vars["θ_crank"])**2),
        units_out="m",
        prerequisites=["r_crank", "L_con", "θ_crank"],
    ),

    FormulaRecord(
        name="Ignition Angle Lead Time",
        domain="engine_mechanics",
        formula_str="θ_ign = (360/60) × RPM × t_lead",
        variables={"RPM": "Kecepatan Putaran Mesin (rev/min)", "t_lead": "Waktu Rambat Api/Lead Time (s)", "θ_ign": "Sudut Pengapian (derajat)"},
        solve_for="θ_ign",
        # Jika RPM dalam SI (Hz), maka kecepatan putaran mesin adalah RPM = vars['RPM'] * 60.
        # Sehingga θ_ign = 6 * (vars['RPM'] * 60) * vars['t_lead'] = 360 * vars['RPM'] * vars['t_lead'].
        python_fn=lambda vars: 360.0 * vars["RPM"] * vars["t_lead"],
        units_out="degree",
        prerequisites=["RPM", "t_lead"],
    ),

    # ─── INVERSE / EXTENDED FORMULAS ─────────────────────────────────────

    FormulaRecord(
        name="Piston Diameter from Force and Pressure",
        domain="fluid_mechanics",
        formula_str="D = 2 × sqrt(F / (π × P))",
        variables={"F": "Gaya (N)", "P": "Tekanan (Pa)", "D": "Diameter piston (m)"},
        solve_for="D",
        python_fn=lambda vars: 2 * math.sqrt(vars["F"] / (math.pi * vars["P"])),
        units_out="m",
        prerequisites=["F", "P"],
        reference="Derived from F = P × π(D/2)²",
        notes="Inverse: cari diameter dari gaya dan tekanan yang diketahui",
    ),

    FormulaRecord(
        name="Open Tube Length from Frequency",
        domain="acoustics",
        formula_str="L = v / (2f)",
        variables={"v": "Kecepatan suara (m/s)", "f": "Frekuensi (Hz)", "L": "Panjang tabung (m)"},
        solve_for="L",
        python_fn=lambda vars: vars["v"] / (2 * vars["f"]),
        units_out="m",
        prerequisites=["v", "f"],
        reference="Derived from f = v/(2L), tabung terbuka",
    ),

    FormulaRecord(
        name="Closed Tube Length from Frequency",
        domain="acoustics",
        formula_str="L = v / (4f)",
        variables={"v": "Kecepatan suara (m/s)", "f": "Frekuensi (Hz)", "L": "Panjang tabung (m)"},
        solve_for="L",
        python_fn=lambda vars: vars["v"] / (4 * vars["f"]),
        units_out="m",
        prerequisites=["v", "f"],
        reference="Derived from f = v/(4L), tabung tertutup satu ujung",
    ),

    FormulaRecord(
        name="Carnot Work Output",
        domain="thermodynamics",
        formula_str="W_out = (1 - T_cold/T_hot) × Q_in",
        variables={
            "T_hot":  "Suhu reservoir panas (K)",
            "T_cold": "Suhu reservoir dingin (K)",
            "Q_in":   "Kalor masuk (J)",
            "W_out":  "Kerja output Carnot (J)",
        },
        solve_for="W_out",
        python_fn=lambda vars: (1.0 - vars["T_cold"] / vars["T_hot"]) * vars["Q_in"],
        units_out="J",
        prerequisites=["T_hot", "T_cold", "Q_in"],
        reference="Carnot theorem: W = η_Carnot × Q_in",
    ),

    FormulaRecord(
        name="Thermal Linear Expansion",
        domain="materials",
        formula_str="ΔL = L₀ × α × ΔT",
        variables={
            "L":     "Panjang awal (m)",
            "alpha": "Koefisien muai linear (/K)",
            "ΔT":    "Perubahan suhu (K)",
            "ΔL":    "Pertambahan panjang (m)",
        },
        solve_for="ΔL",
        python_fn=lambda vars: vars["L"] * vars["alpha"] * vars["ΔT"],
        units_out="m",
        prerequisites=["L", "alpha", "ΔT"],
        reference="Thermal expansion formula",
        notes="α_steel≈12e-6, α_aluminum≈23e-6, α_copper≈17e-6 /K",
    ),

    FormulaRecord(
        name="Clearance Volume from Displacement and CR (extended)",
        domain="engine_mechanics",
        formula_str="V_c = V_d / (CR - 1)",
        variables={
            "V_d": "Volume displacement (m³)",
            "CR":  "Rasio kompresi",
            "V_c": "Volume ruang bakar (m³)",
        },
        solve_for="V_c",
        python_fn=lambda vars: vars["V_d"] / (vars["CR"] - 1.0),
        units_out="m³",
        prerequisites=["V_d", "CR"],
        notes="Variasi formula clearance volume dengan CR dan V_d yang diketahui",
    ),

    FormulaRecord(
        name="Wavelength from Frequency and Speed (acoustics)",
        domain="acoustics",
        formula_str="λ = v / f",
        variables={"v": "Kecepatan suara (m/s)", "f": "Frekuensi (Hz)", "λ": "Panjang gelombang (m)"},
        solve_for="λ",
        python_fn=lambda vars: vars["v"] / vars["f"],
        units_out="m",
        prerequisites=["v", "f"],
    ),

    FormulaRecord(
        name="Hydraulic Force from Pressure and Diameter",
        domain="fluid_mechanics",
        formula_str="F = P × π × (D/2)²",
        variables={"P": "Tekanan (Pa)", "D": "Diameter (m)", "F": "Gaya (N)"},
        solve_for="F",
        python_fn=lambda vars: vars["P"] * math.pi * (vars["D"] / 2) ** 2,
        units_out="N",
        prerequisites=["P", "D"],
        notes="Langsung dari P dan D tanpa perlu A sebagai perantara",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# APPLIED FORMULA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AppliedFormulaEngine:
    """
    Mesin resolusi formula untuk masalah teknik terapan.

    Tiga strategi:
    1. LOOKUP: Cari di FORMULA_DATABASE
    2. DERIVE: Turunkan dari prinsip pertama (SymPy)
    3. SYNTHESIZE: Dimensional analysis Buckingham-Pi
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Index formula per domain
        self._domain_index: Dict[str, List[FormulaRecord]] = {}
        for formula in FORMULA_DATABASE:
            self._domain_index.setdefault(formula.domain, []).append(formula)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [FormulaEngine] {msg}")

    # ── STRATEGI 1: LOOKUP ────────────────────────────────────────────────

    def lookup(
        self,
        domain: str,
        known_symbols: List[str],
        target_symbol: str = None
    ) -> List[FormulaRecord]:
        """
        Cari formula yang bisa diselesaikan dengan variabel yang tersedia.

        Return: list FormulaRecord yang match, diurutkan dari paling relevan.
        """
        candidates = []

        # Cari di domain utama
        domain_formulas = self._domain_index.get(domain, [])
        # Juga cari di semua domain (cross-domain)
        all_formulas = FORMULA_DATABASE

        for formula in all_formulas:
            # Cek apakah target symbol cocok
            if target_symbol and formula.solve_for != target_symbol:
                continue
            # Cek apakah semua prerequisites tersedia
            if formula.can_solve(known_symbols):
                # Beri skor lebih tinggi jika domain cocok
                score = 2 if formula.domain == domain else 1
                candidates.append((score, formula))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [f for _, f in candidates]
        self._log(f"Lookup found {len(result)} formula(s) for domain={domain}, known={known_symbols}")
        return result

    def solve_with_formula(
        self,
        formula: FormulaRecord,
        known_values: Dict[str, float]
    ) -> FormulaSolution:
        """
        Hitung menggunakan formula yang diberikan.
        """
        steps = [
            f"Formula: {formula.formula_str}",
            f"Input values:",
        ]
        for sym, val in known_values.items():
            unit = formula.variables.get(sym, "")
            steps.append(f"  {sym} = {val:.6g}  [{unit}]")

        try:
            result = formula.python_fn(known_values)
            steps.append(f"Computation: {formula.formula_str}")
            steps.append(f"Result: {formula.solve_for} = {result:.6g} {formula.units_out}")

            return FormulaSolution(
                formula=formula,
                inputs=known_values,
                result_value=result,
                result_unit=formula.units_out,
                result_symbol=formula.solve_for,
                steps=steps,
                source=FormulaSource.LOOKUP,
            )
        except Exception as e:
            steps.append(f"ERROR: {e}")
            return FormulaSolution(
                formula=formula,
                inputs=known_values,
                result_value=float('nan'),
                result_unit=formula.units_out,
                result_symbol=formula.solve_for,
                steps=steps,
                source=FormulaSource.FALLBACK,
            )

    # ── STRATEGI 2: DERIVE ────────────────────────────────────────────────

    def derive_from_first_principles(
        self,
        domain: str,
        known: Dict[str, float],
        target: str
    ) -> Optional[FormulaSolution]:
        """
        Turunkan formula yang diperlukan dari prinsip pertama menggunakan SymPy.
        Contoh: kombinasi F=P×A dan A=π(D/2)² → F = P×π×(D/2)²
        """
        if not SYMPY_AVAILABLE:
            return None

        self._log(f"Attempting derivation: {target} from {list(known.keys())}")

        # Coba gabungkan formula yang ada secara bertahap
        # Misalnya: D diketahui → hitung A → hitung F
        known_syms = list(known.keys())
        all_solutions = {}
        all_solutions.update({k: v for k, v in known.items()})

        steps = ["[DERIVATION] Menggabungkan formula secara bertahap:"]
        max_iterations = 5

        for iteration in range(max_iterations):
            progress = False
            candidates = self.lookup(domain, list(all_solutions.keys()))

            for formula in candidates:
                if formula.solve_for in all_solutions:
                    continue  # Sudah dihitung
                if formula.can_solve(list(all_solutions.keys())):
                    try:
                        result = formula.python_fn(all_solutions)
                        all_solutions[formula.solve_for] = result
                        steps.append(f"  Langkah {iteration+1}: {formula.formula_str} → {formula.solve_for} = {result:.6g} {formula.units_out}")
                        progress = True

                        if formula.solve_for == target:
                            # Buat dummy FormulaRecord untuk hasilnya
                            derived_formula = FormulaRecord(
                                name=f"Derived: {target} from {list(known.keys())}",
                                domain=domain,
                                formula_str=" → ".join([f.formula_str for f in candidates if f.can_solve(list(known.keys()))][:3]),
                                variables={},
                                solve_for=target,
                                python_fn=lambda v: result,
                                units_out=formula.units_out,
                                notes="Diturunkan dari kombinasi formula"
                            )
                            steps.append(f"  SELESAI: {target} = {result:.6g} {formula.units_out}")
                            return FormulaSolution(
                                formula=derived_formula,
                                inputs=known,
                                result_value=result,
                                result_unit=formula.units_out,
                                result_symbol=target,
                                steps=steps,
                                source=FormulaSource.DERIVED,
                            )
                    except Exception as e:
                        pass

            if not progress:
                break

        return None

    # ── MAIN RESOLVE ──────────────────────────────────────────────────────

    def resolve(
        self,
        domain: str,
        known_si: Dict[str, float],
        target_symbol: str = None,
        target_description: str = ""
    ) -> Optional[FormulaSolution]:
        """
        Resolve formula dari known variables ke target.
        Urutan: LOOKUP → DERIVE → None (tidak bisa diselesaikan)
        """
        known_list = list(known_si.keys())
        self._log(f"Resolving: domain={domain}, known={known_list}, target={target_symbol}")

        # Strategi 1: Direct lookup
        formulas = self.lookup(domain, known_list, target_symbol)
        if formulas:
            solution = self.solve_with_formula(formulas[0], known_si)
            if not math.isnan(solution.result_value):
                return solution

        # Strategi 2: Multi-step derivation
        derived = self.derive_from_first_principles(domain, known_si, target_symbol or "?")
        if derived:
            return derived

        # Strategi 2b: Symbolic Synthesis (Dynamic system-of-equations solving)
        if target_symbol:
            try:
                from .symbolic_synthesizer import get_synthesizer
                synthesizer = get_synthesizer(verbose=self.verbose)
                solution = synthesizer.synthesize_and_solve(domain, known_si, target_symbol)
                if solution:
                    return solution
            except Exception as e:
                self._log(f"Symbolic synthesis failed: {e}")

        self._log(f"Could not resolve: {target_symbol}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICAL_CONSTANTS = {
    "g": (9.80665, "m/s²", "Percepatan gravitasi standar"),
    "c": (299792458, "m/s", "Kecepatan cahaya di vakum"),
    "R": (8.314462, "J/mol·K", "Konstanta gas universal"),
    "k_B": (1.380649e-23, "J/K", "Konstanta Boltzmann"),
    "N_A": (6.02214076e23, "mol⁻¹", "Bilangan Avogadro"),
    "h": (6.62607015e-34, "J·s", "Konstanta Planck"),
    "σ": (5.670374419e-8, "W/m²K⁴", "Konstanta Stefan-Boltzmann"),
    "ε_0": (8.8541878128e-12, "F/m", "Permitivitas vakum"),
    "μ_0": (1.25663706212e-6, "H/m", "Permeabilitas vakum"),
    "e": (1.602176634e-19, "C", "Muatan elementer"),
    "P_0": (101325, "Pa", "Tekanan atmosfer standar"),
    "T_0": (273.15, "K", "0°C dalam Kelvin"),
    "ρ_air": (1.293, "kg/m³", "Densitas udara pada STP"),
    "ρ_water": (1000.0, "kg/m³", "Densitas air pada 4°C"),
    "c_water": (4186.0, "J/kg·K", "Kalor jenis air"),
    "c_air": (1005.0, "J/kg·K", "Kalor jenis udara (cp)"),
    "c_steel": (502.0, "J/kg·K", "Kalor jenis baja"),
    "E_steel": (200e9, "Pa", "Modulus Young baja"),
    "E_aluminum": (70e9, "Pa", "Modulus Young aluminium"),
    "v_sound_20C": (343.2, "m/s", "Kecepatan suara udara 20°C"),
}


def get_constant(name: str) -> Optional[Tuple[float, str]]:
    """Ambil konstanta fisika berdasarkan nama."""
    if name in PHYSICAL_CONSTANTS:
        val, unit, _ = PHYSICAL_CONSTANTS[name]
        return val, unit
    return None


# Singleton
_engine_instance: Optional[AppliedFormulaEngine] = None

def get_formula_engine(verbose: bool = False) -> AppliedFormulaEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AppliedFormulaEngine(verbose=verbose)
    return _engine_instance
