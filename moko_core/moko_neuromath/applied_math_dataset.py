"""
MOKO Applied Math Dataset — Real-World Problem Bank
=====================================================
Bank soal matematika terapan nyata untuk melatih AI MOKO menjawab
pertanyaan matematika dari berbagai domain teknik & sains terapan.

Setiap soal mencakup:
  - Narasi bahasa manusia (Bahasa Indonesia & Inggris)
  - Known variables + nilai + satuan
  - Target variabel + jawaban yang benar
  - Langkah-langkah penyelesaian
  - Tag untuk curriculum tracking

3 Level Kesulitan:
  EASY   — 1 langkah formula langsung
  MEDIUM — 2-3 langkah derivasi bertingkat
  HARD   — Sistem persamaan / symbolic synthesis
"""

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class Difficulty(Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class MathProblem:
    """Satu soal matematika terapan nyata."""
    id: str
    domain: str
    difficulty: Difficulty
    story_text: str                    # Narasi soal
    known_vars: Dict[str, float]       # Variabel yang diketahui (sudah dalam SI)
    target_symbol: str                 # Simbol variabel yang dicari
    expected_answer: float             # Jawaban benar (SI)
    expected_unit: str                 # Satuan jawaban
    tolerance_pct: float               # Toleransi kebenaran (%)
    solution_steps: List[str]          # Langkah penyelesaian
    tags: List[str] = field(default_factory=list)
    formula_hint: str = ""             # Rumus yang relevan
    difficulty_score: float = 1.0     # 1.0 = easy, 2.0 = medium, 3.0 = hard
    source: str = "MOKO Applied Math Dataset v1.0"

    def check_answer(self, answer: float) -> Tuple[bool, float]:
        """
        Cek apakah jawaban benar (dalam toleransi).
        Return: (is_correct, percent_error)
        """
        if self.expected_answer == 0:
            is_correct = abs(answer) < 1e-9
            return is_correct, 0.0 if is_correct else 100.0
        pct_err = abs((answer - self.expected_answer) / self.expected_answer) * 100.0
        return pct_err <= self.tolerance_pct, round(pct_err, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "difficulty": self.difficulty.value,
            "story_text": self.story_text,
            "known_vars": self.known_vars,
            "target_symbol": self.target_symbol,
            "expected_answer": self.expected_answer,
            "expected_unit": self.expected_unit,
            "tolerance_pct": self.tolerance_pct,
            "solution_steps": self.solution_steps,
            "tags": self.tags,
            "formula_hint": self.formula_hint,
        }


@dataclass
class ProblemDataset:
    """Koleksi soal-soal matematika terapan."""
    problems: List[MathProblem] = field(default_factory=list)
    version: str = "1.0"

    def get_by_domain(self, domain: str) -> List[MathProblem]:
        return [p for p in self.problems if p.domain == domain]

    def get_by_difficulty(self, difficulty: Difficulty) -> List[MathProblem]:
        return [p for p in self.problems if p.difficulty == difficulty]

    def get_by_tags(self, tag: str) -> List[MathProblem]:
        return [p for p in self.problems if tag in p.tags]

    def sample(self, n: int = 10, domain: str = None,
               difficulty: Difficulty = None,
               exclude_ids: List[str] = None) -> List[MathProblem]:
        """Ambil sampel soal secara acak."""
        pool = self.problems
        if domain:
            pool = [p for p in pool if p.domain == domain]
        if difficulty:
            pool = [p for p in pool if p.difficulty == difficulty]
        if exclude_ids:
            pool = [p for p in pool if p.id not in exclude_ids]
        return random.sample(pool, min(n, len(pool)))

    def stats(self) -> Dict[str, Any]:
        """Statistik dataset."""
        from collections import Counter
        return {
            "total": len(self.problems),
            "by_domain": dict(Counter(p.domain for p in self.problems)),
            "by_difficulty": dict(Counter(p.difficulty.value for p in self.problems)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER — SOAL PER DOMAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _build_fluid_mechanics_problems() -> List[MathProblem]:
    problems = []

    # --- EASY ---
    D = 0.08  # 80mm piston
    P = 12e5  # 12 bar
    A = math.pi * (D/2)**2
    F = P * A
    problems.append(MathProblem(
        id="FM-001",
        domain="fluid_mechanics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah silinder hidrolik memiliki piston berdiameter 80 mm. "
            "Tekanan fluida di dalam silinder adalah 12 bar. "
            "Hitunglah gaya yang dihasilkan piston."
        ),
        known_vars={"D": D, "P": P, "A": A},
        target_symbol="F",
        expected_answer=F,
        expected_unit="N",
        tolerance_pct=0.5,
        solution_steps=[
            "1. Hitung luas penampang piston: A = π(D/2)² = π(0.08/2)² = 5.027×10⁻³ m²",
            "2. Konversi tekanan: P = 12 bar = 12×10⁵ Pa = 1.2×10⁶ Pa",
            f"3. Hitung gaya: F = P × A = 1.2×10⁶ × {A:.6g} = {F:.6g} N",
        ],
        tags=["piston", "hydraulic", "force", "pressure"],
        formula_hint="F = P × A",
        difficulty_score=1.0,
    ))

    # EASY — radius version
    r = 0.05  # 50mm radius
    P2 = 8e5  # 8 bar
    A2 = math.pi * r**2
    F2 = P2 * A2
    problems.append(MathProblem(
        id="FM-002",
        domain="fluid_mechanics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Pompa hidraulik menghasilkan tekanan 8 bar pada piston berbentuk lingkaran "
            "dengan jari-jari 50 mm. Berapa gaya total yang bekerja pada piston?"
        ),
        known_vars={"r": r, "P": P2, "A": A2},
        target_symbol="F",
        expected_answer=F2,
        expected_unit="N",
        tolerance_pct=0.5,
        solution_steps=[
            "1. Luas penampang: A = πr² = π × (0.05)² = 7.854×10⁻³ m²",
            f"2. Gaya: F = P × A = 8×10⁵ × {A2:.6g} = {F2:.6g} N",
        ],
        tags=["piston", "hydraulic", "force"],
        formula_hint="F = P × A = P × πr²",
        difficulty_score=1.0,
    ))

    # MEDIUM — cari D dari F dan P yang diketahui
    F_target = 50_000  # 50 kN
    P_target = 20e5    # 20 bar
    A_target = F_target / P_target
    D_target = 2 * math.sqrt(A_target / math.pi)
    problems.append(MathProblem(
        id="FM-003",
        domain="fluid_mechanics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sistem hidrolik harus menghasilkan gaya 50 kN pada tekanan kerja 20 bar. "
            "Berapakah diameter minimum piston yang diperlukan?"
        ),
        known_vars={"F": F_target, "P": P_target},
        target_symbol="D",
        expected_answer=D_target,
        expected_unit="m",
        tolerance_pct=1.0,
        solution_steps=[
            "1. F = P × A  →  A = F/P = 50000 / (20×10⁵) = 0.025 m²",
            "2. A = π(D/2)²  →  D = 2√(A/π) = 2√(0.025/π)",
            f"3. D = {D_target:.4f} m = {D_target*100:.2f} cm",
        ],
        tags=["piston", "hydraulic", "inverse", "diameter"],
        formula_hint="F = P × A, A = π(D/2)²  →  D = 2√(F/(πP))",
        difficulty_score=2.0,
    ))

    # MEDIUM — Bernoulli
    rho = 1000.0  # air
    v1 = 2.0      # m/s
    v2 = 5.0      # m/s
    dP = 0.5 * rho * (v1**2 - v2**2)
    problems.append(MathProblem(
        id="FM-004",
        domain="fluid_mechanics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Air mengalir dalam pipa horizontal. Di titik 1 kecepatan aliran 2 m/s, "
            "di titik 2 kecepatan aliran 5 m/s. Densitas air 1000 kg/m³. "
            "Hitung perbedaan tekanan antara dua titik tersebut (ΔP = P₁ - P₂)."
        ),
        known_vars={"ρ": rho, "v_1": v1, "v_2": v2},
        target_symbol="ΔP",
        expected_answer=dP,
        expected_unit="Pa",
        tolerance_pct=1.0,
        solution_steps=[
            "Persamaan Bernoulli (aliran horizontal): P₁ + ½ρv₁² = P₂ + ½ρv₂²",
            "ΔP = P₁ - P₂ = ½ρ(v₂² - v₁²)",
            f"ΔP = ½ × 1000 × (5² - 2²) = ½ × 1000 × 21 = {dP:.2f} Pa",
            f"ΔP = {dP:.2f} Pa = {dP/1000:.2f} kPa",
        ],
        tags=["bernoulli", "flow", "pressure difference"],
        formula_hint="ΔP = ½ρ(v₂² - v₁²)",
        difficulty_score=2.0,
    ))

    return problems


def _build_acoustics_problems() -> List[MathProblem]:
    problems = []

    # EASY — kecepatan suara
    T = 25 + 273.15  # 25°C dalam Kelvin
    v_sound = 331.3 * math.sqrt(T / 273.15)
    problems.append(MathProblem(
        id="AC-001",
        domain="acoustics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Pada suhu udara 25°C, berapakah kecepatan suara di udara?"
        ),
        known_vars={"T": T},
        target_symbol="v",
        expected_answer=v_sound,
        expected_unit="m/s",
        tolerance_pct=0.5,
        solution_steps=[
            "Konversi suhu: T = 25°C + 273.15 = 298.15 K",
            f"v = 331.3 × √(T/273.15) = 331.3 × √(298.15/273.15) = {v_sound:.2f} m/s",
        ],
        tags=["sound speed", "temperature", "air"],
        formula_hint="v = 331.3 × √(T/273.15), T dalam Kelvin",
        difficulty_score=1.0,
    ))

    # EASY — frekuensi dari v dan λ
    v_s = 343.2
    lam = 0.5
    f_freq = v_s / lam
    problems.append(MathProblem(
        id="AC-002",
        domain="acoustics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Gelombang suara merambat di udara (20°C) dengan kecepatan 343.2 m/s. "
            "Jika panjang gelombangnya 0.5 m, berapakah frekuensi gelombang tersebut?"
        ),
        known_vars={"v": v_s, "λ": lam},
        target_symbol="f",
        expected_answer=f_freq,
        expected_unit="Hz",
        tolerance_pct=0.5,
        solution_steps=[
            "f = v / λ = 343.2 / 0.5 = 686.4 Hz",
        ],
        tags=["frequency", "wavelength", "wave"],
        formula_hint="f = v / λ",
        difficulty_score=1.0,
    ))

    # MEDIUM — tabung tertutup
    v_t = 340.0
    L_tube = 0.85
    f_closed = v_t / (4 * L_tube)
    problems.append(MathProblem(
        id="AC-003",
        domain="acoustics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah tabung orgel tertutup satu ujung memiliki panjang 0.85 m. "
            "Kecepatan suara di udara 340 m/s. "
            "Tentukan frekuensi nada dasar yang dihasilkan tabung tersebut."
        ),
        known_vars={"v": v_t, "L": L_tube},
        target_symbol="f",
        expected_answer=f_closed,
        expected_unit="Hz",
        tolerance_pct=0.5,
        solution_steps=[
            "Tabung tertutup: hanya harmonik ganjil, nada dasar pada n=1",
            f"f₁ = v / (4L) = 340 / (4 × 0.85) = {f_closed:.2f} Hz",
        ],
        tags=["closed tube", "standing wave", "organ pipe", "resonance"],
        formula_hint="f₁ = v / (4L) untuk tabung tertutup satu ujung",
        difficulty_score=2.0,
    ))

    # MEDIUM — cari panjang tabung dari f dan v
    v_t2 = 340.0
    f_target2 = 440.0  # nada A4
    L_needed = v_t2 / (2 * f_target2)
    problems.append(MathProblem(
        id="AC-004",
        domain="acoustics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Seorang pengrajin alat musik ingin membuat tabung terbuka (flute) "
            "yang menghasilkan nada A4 (440 Hz). Kecepatan suara 340 m/s. "
            "Berapakah panjang tabung yang diperlukan?"
        ),
        known_vars={"v": v_t2, "f": f_target2},
        target_symbol="L",
        expected_answer=L_needed,
        expected_unit="m",
        tolerance_pct=1.0,
        solution_steps=[
            "Tabung terbuka: f = v / (2L)",
            f"L = v / (2f) = 340 / (2 × 440) = {L_needed:.4f} m = {L_needed*100:.2f} cm",
        ],
        tags=["open tube", "flute", "resonance", "inverse"],
        formula_hint="f = v / (2L)  →  L = v / (2f)",
        difficulty_score=2.0,
    ))

    return problems


def _build_thermodynamics_problems() -> List[MathProblem]:
    problems = []

    # EASY — kalor sensibel
    m_w = 2.0      # 2 kg air
    c_w = 4186.0   # J/kg·K
    dT = 50.0      # ΔT = 50°C
    Q_heat = m_w * c_w * dT
    problems.append(MathProblem(
        id="TH-001",
        domain="thermodynamics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Diperlukan energi panas untuk memanaskan 2 kg air dari 20°C menjadi 70°C. "
            "Kalor jenis air 4186 J/(kg·K). Hitunglah energi panas yang diperlukan."
        ),
        known_vars={"m": m_w, "c": c_w, "ΔT": dT},
        target_symbol="Q",
        expected_answer=Q_heat,
        expected_unit="J",
        tolerance_pct=0.5,
        solution_steps=[
            "ΔT = 70 - 20 = 50 K",
            f"Q = m × c × ΔT = 2 × 4186 × 50 = {Q_heat:.0f} J = {Q_heat/1000:.2f} kJ",
        ],
        tags=["heat", "water", "specific heat", "temperature"],
        formula_hint="Q = m × c × ΔT",
        difficulty_score=1.0,
    ))

    # EASY — efisiensi Carnot
    T_cold = 300.0  # 27°C = 300K
    T_hot  = 600.0  # 327°C = 600K
    eta_c  = 1 - T_cold / T_hot
    problems.append(MathProblem(
        id="TH-002",
        domain="thermodynamics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Mesin Carnot beroperasi antara reservoir panas pada 600 K "
            "dan reservoir dingin pada 300 K. "
            "Berapakah efisiensi termal maksimum mesin tersebut?"
        ),
        known_vars={"T_hot": T_hot, "T_cold": T_cold},
        target_symbol="η",
        expected_answer=eta_c,
        expected_unit="dimensionless (or %)",
        tolerance_pct=0.5,
        solution_steps=[
            "η_Carnot = 1 - T_cold / T_hot",
            f"η = 1 - 300/600 = 1 - 0.5 = {eta_c:.2f} = {eta_c*100:.0f}%",
        ],
        tags=["carnot", "efficiency", "heat engine", "thermodynamics"],
        formula_hint="η = 1 - T_cold / T_hot",
        difficulty_score=1.0,
    ))

    # MEDIUM — Gas Ideal
    n = 1.5   # mol
    R = 8.314
    T_gas = 350  # K
    V_gas = 0.025  # 25L = 0.025 m³
    P_gas = n * R * T_gas / V_gas
    problems.append(MathProblem(
        id="TH-003",
        domain="thermodynamics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah tangki 25 liter berisi 1.5 mol gas ideal pada suhu 350 K. "
            "Hitunglah tekanan gas dalam tangki tersebut (konstanta gas R = 8.314 J/mol·K)."
        ),
        known_vars={"n": n, "T": T_gas, "V": V_gas},
        target_symbol="P",
        expected_answer=P_gas,
        expected_unit="Pa",
        tolerance_pct=0.5,
        solution_steps=[
            "Persamaan gas ideal: PV = nRT",
            f"P = nRT/V = (1.5 × 8.314 × 350) / 0.025",
            f"P = {n*R*T_gas:.2f} / 0.025 = {P_gas:.2f} Pa = {P_gas/1000:.2f} kPa",
        ],
        tags=["ideal gas", "pressure", "temperature", "volume"],
        formula_hint="PV = nRT  →  P = nRT/V",
        difficulty_score=2.0,
    ))

    return problems


def _build_electronics_problems() -> List[MathProblem]:
    problems = []

    # EASY — Ohm's Law
    I = 2.5   # A
    R = 10.0  # Ω
    V_ohm = I * R
    problems.append(MathProblem(
        id="EL-001",
        domain="electronics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah resistor 10 ohm dialiri arus listrik 2.5 ampere. "
            "Hitunglah tegangan (beda potensial) pada ujung-ujung resistor tersebut."
        ),
        known_vars={"I": I, "R": R},
        target_symbol="V",
        expected_answer=V_ohm,
        expected_unit="V",
        tolerance_pct=0.5,
        solution_steps=[
            "V = I × R = 2.5 × 10 = 25 V",
        ],
        tags=["ohm", "voltage", "resistance", "current"],
        formula_hint="V = I × R (Hukum Ohm)",
        difficulty_score=1.0,
    ))

    # EASY — Electric Power
    V_p = 220.0  # V
    I_p = 5.0    # A
    P_el = V_p * I_p
    problems.append(MathProblem(
        id="EL-002",
        domain="electronics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah peralatan listrik beroperasi pada tegangan 220 V "
            "dan menarik arus 5 A. Berapakah daya yang dikonsumsi?"
        ),
        known_vars={"V": V_p, "I": I_p},
        target_symbol="P_e",
        expected_answer=P_el,
        expected_unit="W",
        tolerance_pct=0.5,
        solution_steps=[
            "P = V × I = 220 × 5 = 1100 W = 1.1 kW",
        ],
        tags=["power", "electricity", "watt"],
        formula_hint="P = V × I",
        difficulty_score=1.0,
    ))

    # MEDIUM — RC cutoff frequency
    R_rc = 1000.0  # 1kΩ
    C_rc = 10e-9   # 10nF
    f_cut = 1 / (2 * math.pi * R_rc * C_rc)
    problems.append(MathProblem(
        id="EL-003",
        domain="electronics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah filter RC low-pass terdiri dari resistor 1 kΩ dan kapasitor 10 nF. "
            "Tentukan frekuensi cutoff (-3dB) filter tersebut."
        ),
        known_vars={"R": R_rc, "C": C_rc},
        target_symbol="f_c",
        expected_answer=f_cut,
        expected_unit="Hz",
        tolerance_pct=1.0,
        solution_steps=[
            "f_c = 1 / (2πRC)",
            f"f_c = 1 / (2π × 1000 × 10×10⁻⁹)",
            f"f_c = 1 / (2π × 10⁻⁵) = {f_cut:.2f} Hz ≈ {f_cut/1000:.2f} kHz",
        ],
        tags=["RC filter", "cutoff frequency", "low-pass", "electronics"],
        formula_hint="f_c = 1 / (2πRC)",
        difficulty_score=2.0,
    ))

    # MEDIUM — LC Resonance
    L_lc = 1e-3   # 1 mH
    C_lc = 100e-9 # 100 nF
    f_res = 1 / (2 * math.pi * math.sqrt(L_lc * C_lc))
    problems.append(MathProblem(
        id="EL-004",
        domain="electronics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah rangkaian resonan LC memiliki induktor 1 mH dan kapasitor 100 nF. "
            "Pada frekuensi berapa rangkaian ini beresonansi?"
        ),
        known_vars={"L": L_lc, "C": C_lc},
        target_symbol="f",
        expected_answer=f_res,
        expected_unit="Hz",
        tolerance_pct=1.0,
        solution_steps=[
            "f₀ = 1 / (2π√(LC))",
            f"f₀ = 1 / (2π × √(10⁻³ × 10⁻⁷))",
            f"f₀ = 1 / (2π × √(10⁻¹⁰)) = {f_res:.2f} Hz ≈ {f_res/1000:.2f} kHz",
        ],
        tags=["LC resonance", "inductor", "capacitor", "resonance"],
        formula_hint="f₀ = 1 / (2π√(LC))",
        difficulty_score=2.0,
    ))

    return problems


def _build_kinematics_problems() -> List[MathProblem]:
    problems = []

    # EASY — Free fall time
    h_ff = 45.0  # 45m
    g = 9.80665
    t_fall = math.sqrt(2 * h_ff / g)
    problems.append(MathProblem(
        id="KN-001",
        domain="kinematics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah bola jatuh bebas dari ketinggian 45 meter. "
            "Berapa lama waktu yang diperlukan bola untuk mencapai tanah "
            "(abaikan hambatan udara, g = 9.8 m/s²)?"
        ),
        known_vars={"h": h_ff},
        target_symbol="t",
        expected_answer=t_fall,
        expected_unit="s",
        tolerance_pct=1.0,
        solution_steps=[
            "h = ½gt²  →  t = √(2h/g)",
            f"t = √(2 × 45 / 9.80665) = √(9.175) = {t_fall:.4f} s",
        ],
        tags=["free fall", "gravity", "time", "height"],
        formula_hint="t = √(2h/g)",
        difficulty_score=1.0,
    ))

    # EASY — final velocity
    v0 = 0.0
    a_acc = 4.0  # m/s²
    t_acc = 8.0  # s
    v_final = v0 + a_acc * t_acc
    problems.append(MathProblem(
        id="KN-002",
        domain="kinematics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah mobil bergerak dari keadaan diam dengan percepatan konstan 4 m/s². "
            "Berapa kecepatan mobil setelah 8 detik?"
        ),
        known_vars={"v_0": v0, "a": a_acc, "t": t_acc},
        target_symbol="v",
        expected_answer=v_final,
        expected_unit="m/s",
        tolerance_pct=0.5,
        solution_steps=[
            "v = v₀ + at = 0 + 4 × 8 = 32 m/s",
        ],
        tags=["acceleration", "velocity", "kinematics"],
        formula_hint="v = v₀ + at",
        difficulty_score=1.0,
    ))

    # MEDIUM — projectile max height
    v0_proj = 40.0  # m/s
    theta = math.radians(45)
    h_max = v0_proj**2 * math.sin(theta)**2 / (2 * g)
    problems.append(MathProblem(
        id="KN-003",
        domain="kinematics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah bola dilempar dengan kecepatan awal 40 m/s pada sudut 45° "
            "terhadap horizontal. Berapakah tinggi maksimum yang dicapai bola "
            "(g = 9.80665 m/s²)?"
        ),
        known_vars={"v_0": v0_proj, "θ": theta},
        target_symbol="h",
        expected_answer=h_max,
        expected_unit="m",
        tolerance_pct=1.0,
        solution_steps=[
            "Komponen vertikal: v_y = v₀ × sin(45°) = 40 × 0.7071 = 28.28 m/s",
            "Tinggi maks: h = v_y² / (2g)",
            f"h = (28.28)² / (2 × 9.80665) = {h_max:.4f} m",
        ],
        tags=["projectile", "height", "trigonometry"],
        formula_hint="h_max = v₀²sin²(θ) / (2g)",
        difficulty_score=2.0,
    ))

    return problems


def _build_engine_mechanics_problems() -> List[MathProblem]:
    problems = []

    # EASY — Engine displacement
    B = 0.08   # 80mm bore
    S = 0.09   # 90mm stroke
    N = 4      # 4 cylinders
    V_d = (math.pi / 4) * B**2 * S * N
    problems.append(MathProblem(
        id="EM-001",
        domain="engine_mechanics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Mesin bensin 4 silinder memiliki diameter silinder (bore) 80 mm "
            "dan langkah piston (stroke) 90 mm. "
            "Berapakah kapasitas mesin (displacement) dalam cc?"
        ),
        known_vars={"B": B, "S": S, "N_cyl": N},
        target_symbol="V_d",
        expected_answer=V_d,
        expected_unit="m³",
        tolerance_pct=0.5,
        solution_steps=[
            "V_d = (π/4) × B² × S × N_cyl",
            f"V_d = (π/4) × (0.08)² × 0.09 × 4 = {V_d:.6g} m³",
            f"     = {V_d * 1e6:.2f} cc",
        ],
        tags=["engine", "displacement", "cc", "bore", "stroke"],
        formula_hint="V_d = (π/4) × B² × S × N_cyl",
        difficulty_score=1.0,
    ))

    # MEDIUM — Compression Ratio
    B2 = 0.075  # bore
    S2 = 0.085  # stroke
    N2 = 1
    V_d2 = (math.pi / 4) * B2**2 * S2 * N2
    V_c2 = V_d2 / (10.5 - 1)  # CR = 10.5:1
    CR_check = (V_d2 + V_c2) / V_c2
    problems.append(MathProblem(
        id="EM-002",
        domain="engine_mechanics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah silinder mesin memiliki bore 75mm, stroke 85mm, "
            "dan rasio kompresi 10.5:1. "
            "Berapakah volume ruang bakar (clearance volume) dalam cc?"
        ),
        known_vars={"B": B2, "S": S2, "N_cyl": N2, "CR": 10.5},
        target_symbol="V_c",
        expected_answer=V_c2,
        expected_unit="m³",
        tolerance_pct=1.0,
        solution_steps=[
            f"1. V_d = (π/4) × B² × S = (π/4) × (0.075)² × 0.085 = {V_d2:.6g} m³",
            f"2. CR = (V_d + V_c) / V_c  →  V_c = V_d / (CR - 1)",
            f"3. V_c = {V_d2:.6g} / (10.5 - 1) = {V_c2:.6g} m³ = {V_c2*1e6:.2f} cc",
        ],
        tags=["compression ratio", "clearance volume", "engine"],
        formula_hint="V_c = V_d / (CR - 1)",
        difficulty_score=2.0,
    ))

    # MEDIUM — Ignition timing
    RPM_ign = 3000 / 60  # konversi ke rev/s (stored as SI)
    t_lead = 2.3e-3  # 2.3 ms flame travel time
    theta_ign = 360.0 * RPM_ign * t_lead
    problems.append(MathProblem(
        id="EM-003",
        domain="engine_mechanics",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Mesin berputar pada 3000 RPM. Waktu rambat api dari busi ke seluruh "
            "ruang bakar adalah 2.3 ms. Berapa derajat sebelum TMA (sudut pengapian) "
            "yang harus disetel?"
        ),
        known_vars={"RPM": RPM_ign, "t_lead": t_lead},
        target_symbol="θ_ign",
        expected_answer=theta_ign,
        expected_unit="degree",
        tolerance_pct=1.0,
        solution_steps=[
            "RPM = 3000 rev/min = 50 rev/s",
            "θ_ign = 360° × RPM (rev/s) × t_lead",
            f"θ_ign = 360 × 50 × 0.0023 = {theta_ign:.2f}°",
        ],
        tags=["ignition timing", "RPM", "flame speed", "engine"],
        formula_hint="θ_ign = 360 × RPM(rev/s) × t_lead",
        difficulty_score=2.0,
    ))

    return problems


def _build_energy_problems() -> List[MathProblem]:
    problems = []

    # EASY — Kinetic energy
    m_ke = 1500.0  # kg
    v_ke = 25.0    # m/s
    Ek = 0.5 * m_ke * v_ke**2
    problems.append(MathProblem(
        id="EN-001",
        domain="energy",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah kendaraan bermassa 1500 kg bergerak dengan kecepatan 25 m/s "
            "(sekitar 90 km/jam). Berapakah energi kinetik kendaraan tersebut?"
        ),
        known_vars={"m": m_ke, "v": v_ke},
        target_symbol="Ek",
        expected_answer=Ek,
        expected_unit="J",
        tolerance_pct=0.5,
        solution_steps=[
            "Ek = ½mv² = ½ × 1500 × 25²",
            f"Ek = ½ × 1500 × 625 = {Ek:.0f} J = {Ek/1000:.1f} kJ",
        ],
        tags=["kinetic energy", "mass", "velocity"],
        formula_hint="Ek = ½mv²",
        difficulty_score=1.0,
    ))

    # EASY — Potential energy
    m_pe = 80.0   # kg
    h_pe = 50.0   # m
    Ep = m_pe * 9.80665 * h_pe
    problems.append(MathProblem(
        id="EN-002",
        domain="energy",
        difficulty=Difficulty.EASY,
        story_text=(
            "Seorang pendaki bermassa 80 kg naik ke puncak bukit setinggi 50 meter. "
            "Berapa besar energi potensial yang tersimpan (g = 9.80665 m/s²)?"
        ),
        known_vars={"m": m_pe, "h": h_pe},
        target_symbol="Ep",
        expected_answer=Ep,
        expected_unit="J",
        tolerance_pct=0.5,
        solution_steps=[
            "Ep = mgh = 80 × 9.80665 × 50",
            f"Ep = {Ep:.2f} J = {Ep/1000:.3f} kJ",
        ],
        tags=["potential energy", "gravity", "height"],
        formula_hint="Ep = mgh",
        difficulty_score=1.0,
    ))

    # MEDIUM — Mechanical Efficiency
    P_out = 7500.0  # W
    P_in = 9000.0   # W
    eta_mech = (P_out / P_in) * 100
    problems.append(MathProblem(
        id="EN-003",
        domain="energy",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Sebuah motor listrik menggunakan daya input 9 kW "
            "dan menghasilkan daya output mekanik 7.5 kW. "
            "Berapakah efisiensi mekanik motor tersebut?"
        ),
        known_vars={"P_in": P_in, "P_out": P_out},
        target_symbol="η",
        expected_answer=eta_mech,
        expected_unit="%",
        tolerance_pct=0.5,
        solution_steps=[
            "η = (P_out / P_in) × 100%",
            f"η = (7500 / 9000) × 100% = {eta_mech:.2f}%",
        ],
        tags=["efficiency", "motor", "power"],
        formula_hint="η = P_out / P_in × 100%",
        difficulty_score=2.0,
    ))

    return problems


def _build_finance_problems() -> List[MathProblem]:
    problems = []

    # EASY — Compound Interest
    PV = 10_000_000  # Rp 10 juta
    r = 0.08         # 8% per tahun
    n = 5            # 5 tahun
    FV = PV * (1 + r) ** n
    problems.append(MathProblem(
        id="FN-001",
        domain="finance",
        difficulty=Difficulty.EASY,
        story_text=(
            "Modal Rp 10.000.000 diinvestasikan selama 5 tahun dengan bunga majemuk 8% per tahun. "
            "Berapakah nilai akhir investasi tersebut?"
        ),
        known_vars={"PV": PV, "r": r, "n": n},
        target_symbol="FV",
        expected_answer=FV,
        expected_unit="IDR",
        tolerance_pct=0.1,
        solution_steps=[
            "FV = PV × (1 + r)^n",
            f"FV = 10.000.000 × (1 + 0.08)^5",
            f"FV = 10.000.000 × {(1+r)**n:.6f} = {FV:,.0f}",
        ],
        tags=["compound interest", "future value", "investment"],
        formula_hint="FV = PV × (1 + r)^n",
        difficulty_score=1.0,
    ))

    # MEDIUM — ROI
    profit = 3_500_000
    modal = 10_000_000
    roi = (profit / modal) * 100
    problems.append(MathProblem(
        id="FN-002",
        domain="finance",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Seorang pengusaha menginvestasikan modal Rp 10.000.000 dan "
            "mendapatkan keuntungan bersih Rp 3.500.000 dalam setahun. "
            "Berapakah Return on Investment (ROI) usaha tersebut?"
        ),
        known_vars={"profit": profit, "PV": modal},
        target_symbol="ROI",
        expected_answer=roi,
        expected_unit="%",
        tolerance_pct=0.1,
        solution_steps=[
            "ROI = (Keuntungan / Modal) × 100%",
            f"ROI = (3.500.000 / 10.000.000) × 100% = {roi:.1f}%",
        ],
        tags=["ROI", "profit", "investment", "percentage"],
        formula_hint="ROI = (profit / PV) × 100%",
        difficulty_score=2.0,
    ))

    return problems


def _build_structural_problems() -> List[MathProblem]:
    problems = []

    # EASY — Normal Stress
    F_struct = 50_000  # N (50 kN)
    A_struct = 0.0025  # 50mm × 50mm = 0.0025 m²
    sigma = F_struct / A_struct
    problems.append(MathProblem(
        id="ST-001",
        domain="structural",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah batang baja berpenampang persegi 50mm × 50mm "
            "menerima beban tarik aksial 50 kN. "
            "Hitunglah tegangan normal (stress) pada batang tersebut."
        ),
        known_vars={"F": F_struct, "A": A_struct},
        target_symbol="σ",
        expected_answer=sigma,
        expected_unit="Pa",
        tolerance_pct=0.5,
        solution_steps=[
            "A = 50mm × 50mm = 0.05 × 0.05 = 0.0025 m²",
            "σ = F / A = 50000 / 0.0025",
            f"σ = {sigma:.2e} Pa = {sigma/1e6:.2f} MPa",
        ],
        tags=["stress", "structural", "steel", "tension"],
        formula_hint="σ = F / A",
        difficulty_score=1.0,
    ))

    # MEDIUM — Strain from Hooke's Law
    E_steel = 200e9  # Pa
    sigma_2 = 120e6  # 120 MPa
    eps = sigma_2 / E_steel
    problems.append(MathProblem(
        id="ST-002",
        domain="structural",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Batang baja (Modulus Young = 200 GPa) mengalami tegangan 120 MPa. "
            "Hitunglah regangan (strain) yang terjadi, dan jika panjang batang 2 meter, "
            "berapa perpanjangan yang terjadi?"
        ),
        known_vars={"σ": sigma_2, "E": E_steel},
        target_symbol="ε",
        expected_answer=eps,
        expected_unit="dimensionless",
        tolerance_pct=0.5,
        solution_steps=[
            "ε = σ / E = 120×10⁶ / 200×10⁹",
            f"ε = {eps:.2e} (adimensional)",
            f"Perpanjangan: ΔL = ε × L = {eps:.2e} × 2 m = {eps*2*1000:.3f} mm",
        ],
        tags=["strain", "Hooke's law", "Young modulus", "deformation"],
        formula_hint="ε = σ / E",
        difficulty_score=2.0,
    ))

    return problems


def _build_optics_problems() -> List[MathProblem]:
    problems = []

    # EASY — Thin Lens
    do = 0.3   # 30 cm
    di = 0.6   # 60 cm
    f_lens = 1 / (1/do + 1/di)
    problems.append(MathProblem(
        id="OP-001",
        domain="optics",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebuah benda diletakkan 30 cm di depan lensa konvergen. "
            "Bayangan terbentuk 60 cm di belakang lensa. "
            "Tentukan jarak fokus lensa tersebut."
        ),
        known_vars={"do": do, "di": di},
        target_symbol="f",
        expected_answer=f_lens,
        expected_unit="m",
        tolerance_pct=0.5,
        solution_steps=[
            "1/f = 1/do + 1/di = 1/0.3 + 1/0.6",
            f"1/f = {1/do:.4f} + {1/di:.4f} = {1/do + 1/di:.4f}",
            f"f = {f_lens:.4f} m = {f_lens*100:.1f} cm",
        ],
        tags=["lens", "focal length", "optics", "image"],
        formula_hint="1/f = 1/do + 1/di",
        difficulty_score=1.0,
    ))

    return problems


def _build_chemistry_problems() -> List[MathProblem]:
    problems = []

    # EASY — mol dari massa
    m_chem = 36.0  # g NaCl
    M_chem = 58.44  # g/mol NaCl
    n_mol = m_chem / M_chem
    problems.append(MathProblem(
        id="CH-001",
        domain="chemistry",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sebanyak 36 gram garam dapur (NaCl, massa molar = 58.44 g/mol) "
            "dilarutkan dalam air. Berapa banyak mol NaCl yang terlarut?"
        ),
        known_vars={"m": m_chem, "M": M_chem},
        target_symbol="n",
        expected_answer=n_mol,
        expected_unit="mol",
        tolerance_pct=0.5,
        solution_steps=[
            "n = m / M = 36 / 58.44",
            f"n = {n_mol:.4f} mol",
        ],
        tags=["mole", "stoichiometry", "NaCl", "chemistry"],
        formula_hint="n = m / M",
        difficulty_score=1.0,
    ))

    # EASY — pH
    H_plus = 0.001  # 10⁻³ mol/L (HCl 0.001M)
    pH_val = -math.log10(H_plus)
    problems.append(MathProblem(
        id="CH-002",
        domain="chemistry",
        difficulty=Difficulty.EASY,
        story_text=(
            "Larutan HCl memiliki konsentrasi ion H⁺ sebesar 10⁻³ mol/L. "
            "Hitunglah nilai pH larutan tersebut."
        ),
        known_vars={"H+": H_plus},
        target_symbol="pH",
        expected_answer=pH_val,
        expected_unit="dimensionless",
        tolerance_pct=0.1,
        solution_steps=[
            "pH = -log₁₀([H⁺])",
            f"pH = -log₁₀(10⁻³) = 3",
        ],
        tags=["pH", "acid", "concentration", "chemistry"],
        formula_hint="pH = -log₁₀([H⁺])",
        difficulty_score=1.0,
    ))

    return problems


def _build_materials_problems() -> List[MathProblem]:
    problems = []

    # MEDIUM — Thermal expansion
    L0 = 2.0        # m
    alpha = 12e-6   # /K baja
    dT_mat = 100.0  # K
    dL = L0 * alpha * dT_mat
    problems.append(MathProblem(
        id="MT-001",
        domain="materials",
        difficulty=Difficulty.MEDIUM,
        story_text=(
            "Rel kereta api dari baja memiliki panjang 2 m pada suhu awal. "
            "Koefisien muai linear baja = 12 × 10⁻⁶ /K. "
            "Jika suhu naik 100°C, berapa pertambahan panjang rel tersebut?"
        ),
        known_vars={"L": L0, "alpha": alpha, "ΔT": dT_mat},
        target_symbol="ΔL",
        expected_answer=dL,
        expected_unit="m",
        tolerance_pct=0.5,
        solution_steps=[
            "ΔL = L₀ × α × ΔT",
            f"ΔL = 2.0 × 12×10⁻⁶ × 100 = {dL:.6f} m = {dL*1000:.3f} mm",
        ],
        tags=["thermal expansion", "steel", "materials"],
        formula_hint="ΔL = L₀ × α × ΔT",
        difficulty_score=2.0,
    ))

    return problems


def _build_signal_processing_problems() -> List[MathProblem]:
    problems = []

    # EASY — Nyquist
    f_max = 20_000  # 20 kHz audio
    f_nyq = 2 * f_max
    problems.append(MathProblem(
        id="SP-001",
        domain="signal_processing",
        difficulty=Difficulty.EASY,
        story_text=(
            "Sinyal audio memiliki frekuensi tertinggi 20 kHz. "
            "Berdasarkan teorema Nyquist-Shannon, berapakah frekuensi sampling "
            "minimum yang diperlukan untuk merekam sinyal tersebut tanpa aliasing?"
        ),
        known_vars={"f_max": f_max},
        target_symbol="f_s",
        expected_answer=f_nyq,
        expected_unit="Hz",
        tolerance_pct=0.5,
        solution_steps=[
            "Teorema Nyquist: f_s ≥ 2 × f_max",
            f"f_s = 2 × 20000 = {f_nyq} Hz = {f_nyq/1000} kHz",
        ],
        tags=["Nyquist", "sampling", "audio", "digital signal"],
        formula_hint="f_s ≥ 2 × f_max",
        difficulty_score=1.0,
    ))

    return problems


# ═══════════════════════════════════════════════════════════════════════════════
# HARD PROBLEMS — Multi-step / Symbolic
# ═══════════════════════════════════════════════════════════════════════════════

def _build_hard_problems() -> List[MathProblem]:
    problems = []

    # HARD — Cari D dari gaya, tekanan, tapi D tidak diketahui langsung
    F_hrd = 75_000  # N
    P_hrd = 25e5   # 25 bar
    D_ans = 2 * math.sqrt(F_hrd / (math.pi * P_hrd))
    problems.append(MathProblem(
        id="HRD-001",
        domain="fluid_mechanics",
        difficulty=Difficulty.HARD,
        story_text=(
            "Sistem rem hidrolik membutuhkan gaya pengereman 75 kN pada tekanan sistem 25 bar. "
            "Hitunglah diameter piston caliper yang diperlukan untuk mencapai gaya tersebut. "
            "Berikan jawaban dalam milimeter."
        ),
        known_vars={"F": F_hrd, "P": P_hrd},
        target_symbol="D",
        expected_answer=D_ans,
        expected_unit="m",
        tolerance_pct=1.0,
        solution_steps=[
            "F = P × A  →  A = F/P = 75000 / (25×10⁵) = 0.03 m²",
            "A = π(D/2)²  →  D = 2√(A/π) = 2√(0.03/π)",
            f"D = {D_ans:.6f} m = {D_ans*1000:.2f} mm",
        ],
        tags=["hydraulic", "brake", "force", "inverse", "hard"],
        formula_hint="F = P × πD²/4  →  D = 2√(F/(πP))",
        difficulty_score=3.0,
    ))

    # HARD — Multi-step termodinamika (Carnot + output power)
    T_h = 800  # K
    T_c = 300  # K
    Q_in = 500_000  # J input heat
    eta_hard = 1 - T_c / T_h
    W_out = eta_hard * Q_in
    problems.append(MathProblem(
        id="HRD-002",
        domain="thermodynamics",
        difficulty=Difficulty.HARD,
        story_text=(
            "Mesin Carnot beroperasi antara suhu 800 K dan 300 K. "
            "Mesin menerima kalor 500 kJ dari sumber panas per siklus. "
            "Berapakah kerja mekanik yang dihasilkan per siklus?"
        ),
        known_vars={"T_hot": T_h, "T_cold": T_c, "Q_in": Q_in},
        target_symbol="W_out",
        expected_answer=W_out,
        expected_unit="J",
        tolerance_pct=0.5,
        solution_steps=[
            "Langkah 1: Efisiensi Carnot",
            "η = 1 - T_cold/T_hot = 1 - 300/800 = 0.625 = 62.5%",
            "Langkah 2: Kerja Output",
            f"W_out = η × Q_in = 0.625 × 500000 = {W_out:.0f} J = {W_out/1000:.1f} kJ",
        ],
        tags=["carnot", "work", "heat engine", "efficiency", "hard"],
        formula_hint="η = 1 - T_c/T_h, W = η × Q_in",
        difficulty_score=3.0,
    ))

    # HARD — Engine: cari CR dari V_d yang diketahui dan V_c yang dicari dari spek torsi
    V_d_h = 500e-6    # 500cc = 500 cm³
    CR_h  = 11.0
    V_c_h = V_d_h / (CR_h - 1)
    problems.append(MathProblem(
        id="HRD-003",
        domain="engine_mechanics",
        difficulty=Difficulty.HARD,
        story_text=(
            "Mesin motor 500cc dirancang dengan rasio kompresi 11:1. "
            "Hitunglah (a) volume displacement dalam cc, "
            "(b) volume ruang bakar (clearance volume) dalam cc, "
            "dan (c) volume total silinder saat piston di BDC."
        ),
        known_vars={"V_d": V_d_h, "CR": CR_h},
        target_symbol="V_c",
        expected_answer=V_c_h,
        expected_unit="m³",
        tolerance_pct=0.5,
        solution_steps=[
            f"(a) V_d = 500 cc = {V_d_h:.2e} m³",
            "V_c = V_d / (CR - 1) = 500 / (11 - 1) = 500 / 10 = 50 cc",
            f"V_c = {V_c_h:.2e} m³ = {V_c_h*1e6:.2f} cc",
            f"(c) V_total = V_d + V_c = 500 + 50 = 550 cc",
        ],
        tags=["compression ratio", "clearance volume", "engine", "hard"],
        formula_hint="V_c = V_d / (CR - 1), V_total = V_d + V_c",
        difficulty_score=3.0,
    ))

    return problems


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset() -> ProblemDataset:
    """Bangun dan return dataset lengkap."""
    all_problems: List[MathProblem] = []

    all_problems.extend(_build_fluid_mechanics_problems())
    all_problems.extend(_build_acoustics_problems())
    all_problems.extend(_build_thermodynamics_problems())
    all_problems.extend(_build_electronics_problems())
    all_problems.extend(_build_kinematics_problems())
    all_problems.extend(_build_engine_mechanics_problems())
    all_problems.extend(_build_energy_problems())
    all_problems.extend(_build_finance_problems())
    all_problems.extend(_build_structural_problems())
    all_problems.extend(_build_optics_problems())
    all_problems.extend(_build_chemistry_problems())
    all_problems.extend(_build_materials_problems())
    all_problems.extend(_build_signal_processing_problems())
    all_problems.extend(_build_hard_problems())

    return ProblemDataset(problems=all_problems)


# ── PROBLEM GENERATOR (Random Variations) ─────────────────────────────────────

class ProblemVariationGenerator:
    """
    Buat variasi soal dengan nilai yang dirandomize.
    Memastikan dataset tidak pernah habis untuk training.
    """

    @staticmethod
    def vary_fluid_piston(seed: int = None) -> MathProblem:
        if seed:
            random.seed(seed)
        D_mm = random.randint(40, 200)  # mm
        P_bar = random.uniform(5, 50)    # bar
        D = D_mm / 1000
        P = P_bar * 1e5
        A = math.pi * (D/2)**2
        F = P * A
        return MathProblem(
            id=f"FM-GEN-{D_mm}mm-{P_bar:.0f}bar",
            domain="fluid_mechanics",
            difficulty=Difficulty.MEDIUM,
            story_text=(
                f"Sebuah aktuator hidrolik berdiameter {D_mm} mm bekerja pada tekanan "
                f"{P_bar:.1f} bar. Hitunglah gaya yang dihasilkan."
            ),
            known_vars={"D": D, "P": P, "A": A},
            target_symbol="F",
            expected_answer=F,
            expected_unit="N",
            tolerance_pct=1.0,
            solution_steps=[
                f"A = π(D/2)² = π({D_mm/2}mm)² = {A:.6g} m²",
                f"F = P × A = {P:.2e} × {A:.6g} = {F:.2f} N",
            ],
            tags=["piston", "hydraulic", "generated"],
            formula_hint="F = P × A",
            difficulty_score=1.5,
        )

    @staticmethod
    def vary_compound_interest(seed: int = None) -> MathProblem:
        if seed:
            random.seed(seed)
        PV = random.choice([5e6, 10e6, 15e6, 20e6, 50e6])
        r = random.choice([0.05, 0.06, 0.07, 0.08, 0.10, 0.12])
        n = random.choice([3, 5, 7, 10])
        FV = PV * (1 + r)**n
        return MathProblem(
            id=f"FN-GEN-PV{PV/1e6:.0f}m-r{r*100:.0f}pct-n{n}yr",
            domain="finance",
            difficulty=Difficulty.EASY,
            story_text=(
                f"Modal Rp {PV:,.0f} diinvestasikan dengan bunga majemuk "
                f"{r*100:.0f}% per tahun selama {n} tahun. Hitung nilai akhirnya."
            ),
            known_vars={"PV": PV, "r": r, "n": n},
            target_symbol="FV",
            expected_answer=FV,
            expected_unit="IDR",
            tolerance_pct=0.1,
            solution_steps=[
                f"FV = PV × (1+r)^n = {PV:,.0f} × (1+{r})^{n}",
                f"FV = {PV:,.0f} × {(1+r)**n:.6f} = {FV:,.2f}",
            ],
            tags=["compound interest", "finance", "generated"],
            formula_hint="FV = PV × (1 + r)^n",
        )

    @staticmethod
    def generate_batch(n: int = 20) -> List[MathProblem]:
        """Generate n soal variasi acak."""
        batch = []
        generators = [
            ProblemVariationGenerator.vary_fluid_piston,
            ProblemVariationGenerator.vary_compound_interest,
        ]
        for i in range(n):
            gen = generators[i % len(generators)]
            try:
                batch.append(gen(seed=random.randint(1, 9999)))
            except Exception:
                pass
        return batch


# ── SINGLETON ──────────────────────────────────────────────────────────────────

_dataset_instance: Optional[ProblemDataset] = None

def get_dataset(force_rebuild: bool = False) -> ProblemDataset:
    """Return singleton dataset (lazy build)."""
    global _dataset_instance
    if _dataset_instance is None or force_rebuild:
        _dataset_instance = build_dataset()
    return _dataset_instance


if __name__ == "__main__":
    ds = build_dataset()
    stats = ds.stats()
    print(f"\n📚 MOKO Applied Math Dataset v1.0")
    print(f"   Total soal: {stats['total']}")
    print(f"   Per domain: {stats['by_domain']}")
    print(f"   Per difficulty: {stats['by_difficulty']}")
    print()
    for p in ds.problems[:3]:
        print(f"  [{p.id}] {p.difficulty.value.upper()} | {p.story_text[:70]}...")
        print(f"       Target: {p.target_symbol} = {p.expected_answer:.4g} {p.expected_unit}")
