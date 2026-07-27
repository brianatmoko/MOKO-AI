"""
MOKO Dimensional Synthesis Engine
===================================
Menggunakan Buckingham-Pi Theorem untuk menurunkan bentuk rumus
dari dimensi variabel saja — tanpa data, tanpa template.

Ini adalah pendekatan Level 3 (Discovery) murni:
  Input : variabel + dimensi fisik masing-masing
  Output: daftar kandidat rumus yang konsisten secara dimensional

Contoh:
  Variables: F [M L T⁻²], P [M L⁻¹ T⁻²], A [L²]
  Buckingham: F = k × P^a × A^b
  Dimensional solve: k=1, a=1, b=1
  Kandidat: F = P × A ✓

Referensi:
  - Buckingham (1914), "On Physically Similar Systems"
  - AI Feynman: Udrescu & Tegmark (2019)
  - Dimensional Analysis & Similitude, Pankhurst (1964)
"""

import math
import itertools
import fractions
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ── DIMENSI FISIK ─────────────────────────────────────────────────────────────

# 7 dimensi dasar SI
BASE_DIMS = ['M', 'L', 'T', 'θ', 'I', 'N', 'J']
# M=massa, L=panjang, T=waktu, θ=suhu, I=arus, N=mol, J=intensitas cahaya

# Tabel dimensi variabel fisika umum
DIMENSION_TABLE: Dict[str, Dict[str, int]] = {
    # ── Mekanika ──────────────────────────────────────────────────────
    "F":    {"M": 1, "L": 1,  "T": -2},                    # Gaya [N]
    "m":    {"M": 1},                                        # Massa [kg]
    "a":    {"L": 1, "T": -2},                               # Percepatan [m/s²]
    "v":    {"L": 1, "T": -1},                               # Kecepatan [m/s]
    "v_s":  {"L": 1, "T": -1},                               # Kecepatan suara
    "L":    {"L": 1},                                        # Panjang [m]
    "t":    {"T": 1},                                        # Waktu [s]
    "x":    {"L": 1},                                        # Posisi [m]
    "h":    {"L": 1},                                        # Ketinggian [m]
    "d":    {"L": 1},                                        # Jarak [m]
    "r":    {"L": 1},                                        # Jari-jari [m]
    "g":    {"L": 1, "T": -2},                               # Gravitasi [m/s²]
    "E_k":  {"M": 1, "L": 2, "T": -2},                      # Energi kinetik [J]
    "E_p":  {"M": 1, "L": 2, "T": -2},                      # Energi potensial [J]
    "W":    {"M": 1, "L": 2, "T": -2},                      # Kerja [J]
    "Ek":   {"M": 1, "L": 2, "T": -2},                      # Alias energi kinetik
    "Ep":   {"M": 1, "L": 2, "T": -2},                      # Alias energi potensial
    "P_mech": {"M": 1, "L": 2, "T": -3},                    # Daya mekanik [W]
    "I_momen": {"M": 1, "L": 2},                             # Momen inersia [kg⋅m²]
    "τ":    {"M": 1, "L": 2, "T": -2},                      # Torsi [N⋅m]
    "ω":    {"T": -1},                                       # Kecepatan angular [rad/s]
    "α_ang": {"T": -2},                                      # Percepatan angular [rad/s²]
    "p":    {"M": 1, "L": 1, "T": -1},                      # Momentum [kg⋅m/s]
    # ── Fluida ────────────────────────────────────────────────────────
    "P":    {"M": 1, "L": -1, "T": -2},                     # Tekanan [Pa]
    "ΔP":   {"M": 1, "L": -1, "T": -2},                     # Beda tekanan
    "A":    {"L": 2},                                        # Luas area [m²]
    "V_vol": {"L": 3},                                       # Volume [m³]
    "ρ":    {"M": 1, "L": -3},                               # Densitas [kg/m³]
    "Q_flow": {"L": 3, "T": -1},                             # Debit volumetrik [m³/s]
    "η":    {"M": 1, "L": -1, "T": -1},                     # Viskositas dinamik [Pa⋅s]
    "D":    {"L": 1},                                        # Diameter [m]
    "Re":   {},                                              # Reynolds number [tak berdimensi]
    # ── Termodinamika ─────────────────────────────────────────────────
    "T_temp": {"θ": 1},                                      # Suhu [K]
    "T_hot":  {"θ": 1},
    "T_cold": {"θ": 1},
    "ΔT":     {"θ": 1},                                      # Beda suhu [K]
    "Q_heat": {"M": 1, "L": 2, "T": -2},                    # Kalor [J]
    "Q_in":   {"M": 1, "L": 2, "T": -2},
    "c_heat": {"L": 2, "T": -2, "θ": -1},                   # Kapasitas kalor [J/kg⋅K]
    "k_cond": {"M": 1, "L": 1, "T": -3, "θ": -1},           # Konduktivitas termal
    "S":      {"M": 1, "L": 2, "T": -2, "θ": -1},           # Entropi [J/K]
    "R_gas":  {"M": 1, "L": 2, "T": -2, "θ": -1, "N": -1}, # Konstanta gas [J/mol⋅K]
    "n_mol":  {"N": 1},                                      # Jumlah mol
    # ── Elektronika ───────────────────────────────────────────────────
    "V_e":   {"M": 1, "L": 2, "T": -3, "I": -1},            # Tegangan [V]
    "V":     {"M": 1, "L": 2, "T": -3, "I": -1},            # Alias tegangan
    "I_e":   {"I": 1},                                       # Arus [A]
    "I":     {"I": 1},                                       # Alias arus
    "R_e":   {"M": 1, "L": 2, "T": -3, "I": -2},            # Resistansi [Ω]
    "R":     {"M": 1, "L": 2, "T": -3, "I": -2},            # Alias resistansi
    "C":     {"M": -1, "L": -2, "T": 4, "I": 2},            # Kapasitansi [F]
    "L_ind": {"M": 1, "L": 2, "T": -2, "I": -2},            # Induktansi [H]
    "P_e":   {"M": 1, "L": 2, "T": -3},                     # Daya listrik [W]
    "f":     {"T": -1},                                      # Frekuensi [Hz]
    # ── Akustik / Optik ───────────────────────────────────────────────
    "λ":     {"L": 1},                                       # Panjang gelombang [m]
    "f_opt": {"T": -1},                                      # Frekuensi optik
    # ── Material ──────────────────────────────────────────────────────
    "σ":    {"M": 1, "L": -1, "T": -2},                     # Tegangan material [Pa]
    "ε":    {},                                              # Regangan [tak berdimensi]
    "E_mod": {"M": 1, "L": -1, "T": -2},                    # Modulus elastisitas [Pa]
    "ΔL":   {"L": 1},                                        # Perubahan panjang [m]
    "alpha": {"θ": -1},                                      # Koefisien muai [/K]
}


@dataclass
class DimVector:
    """Vektor dimensi: dictionary basis dimensi → eksponen."""
    dims: Dict[str, int] = field(default_factory=dict)

    def __add__(self, other: 'DimVector') -> 'DimVector':
        result = dict(self.dims)
        for k, v in other.dims.items():
            result[k] = result.get(k, 0) + v
        return DimVector({k: v for k, v in result.items() if v != 0})

    def __mul__(self, scalar) -> 'DimVector':
        return DimVector({k: v * scalar for k, v in self.dims.items()})

    def __eq__(self, other) -> bool:
        return self.dims == other.dims

    def is_dimensionless(self) -> bool:
        return all(v == 0 for v in self.dims.values()) or len(self.dims) == 0

    def to_string(self) -> str:
        parts = []
        for dim in BASE_DIMS:
            exp = self.dims.get(dim, 0)
            if exp == 1:
                parts.append(dim)
            elif exp != 0:
                parts.append(f"{dim}^{exp}")
        return " ".join(parts) if parts else "1"


@dataclass
class DimensionalFormula:
    """Satu kandidat formula yang konsisten secara dimensional."""
    expression: str           # Ekspresi formula: "Y = c × X1^a × X2^b"
    exponents: Dict[str, fractions.Fraction]  # Eksponen setiap variabel
    pi_groups: List[str]      # Grup tak-berdimensi (Buckingham Pi)
    dimensional_check: bool   # True jika dimensi konsisten
    source: str = "dimensional_synthesis"

    def is_valid(self) -> bool:
        return self.dimensional_check


@dataclass
class SynthesisResult:
    """Hasil dari Dimensional Synthesis."""
    target: str
    variables: List[str]
    candidate_formulas: List[DimensionalFormula]
    pi_groups: List[str]
    n_pi_groups: int           # Jumlah grup tak-berdimensi
    dimensional_matrix: List[List[int]]
    explanation: str


class DimensionalSynthesisEngine:
    """
    Engine untuk sintesis rumus berdasarkan analisis dimensional.

    Algoritma Buckingham-Pi:
    1. Tentukan variabel (n) dan dimensi dasar (k)
    2. Hitung jumlah Pi-groups = n - k
    3. Pilih variabel "repeating" (basis)
    4. Bentuk setiap Pi-group dari variabel repeating + satu variabel lain
    5. Setiap Pi-group adalah kombinasi tak-berdimensi

    Output: kandidat rumus yang secara dimensional konsisten.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._dim_table = DIMENSION_TABLE

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DimSynth] {msg}")

    def get_dim_vector(self, var_name: str) -> Optional[DimVector]:
        """Ambil vektor dimensi untuk variabel."""
        # Direct lookup
        if var_name in self._dim_table:
            return DimVector(dict(self._dim_table[var_name]))
        # Alias lookup (misal: T_temp → T, T_hot, T_cold semua punya θ)
        for alias in [var_name.lower(), var_name.upper()]:
            if alias in self._dim_table:
                return DimVector(dict(self._dim_table[alias]))
        return None

    def _build_dimensional_matrix(
        self,
        variables: List[str],
        dims_used: List[str]
    ) -> List[List[int]]:
        """
        Bangun matriks dimensional:
        Baris = dimensi dasar yang digunakan
        Kolom = variabel
        Entry[i][j] = eksponen dimensi i dalam variabel j
        """
        matrix = []
        for dim in dims_used:
            row = []
            for var in variables:
                dv = self.get_dim_vector(var)
                row.append(dv.dims.get(dim, 0) if dv else 0)
            matrix.append(row)
        return matrix

    def _gaussian_elimination(
        self, matrix: List[List[fractions.Fraction]], n_cols: int
    ) -> int:
        """
        Gaussian elimination untuk cari rank matriks.
        Return: rank matriks
        """
        m = len(matrix)
        row = 0
        rank = 0
        for col in range(n_cols):
            # Cari pivot
            pivot = None
            for r in range(row, m):
                if matrix[r][col] != 0:
                    pivot = r
                    break
            if pivot is None:
                continue
            # Swap
            matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
            # Normalize
            pivot_val = matrix[row][col]
            matrix[row] = [x / pivot_val for x in matrix[row]]
            # Eliminate
            for r in range(m):
                if r != row and matrix[r][col] != 0:
                    factor = matrix[r][col]
                    matrix[r] = [matrix[r][c] - factor * matrix[row][c]
                                  for c in range(len(matrix[r]))]
            row += 1
            rank += 1
        return rank

    def synthesize(
        self,
        target_var: str,
        input_vars: List[str],
        known_dims: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> SynthesisResult:
        """
        Sintesis kandidat rumus dari analisis dimensional.

        Args:
            target_var: Variabel yang ingin dicari (Y)
            input_vars: Variabel yang diketahui [X1, X2, ...]
            known_dims: Override dimensi untuk variabel yang tidak ada di table

        Returns:
            SynthesisResult dengan daftar kandidat formula
        """
        # Update tabel dengan dimensi yang diberikan
        if known_dims:
            for var, dims in known_dims.items():
                self._dim_table[var] = dims

        all_vars = [target_var] + input_vars

        # Kumpulkan dimensi yang digunakan
        dims_used = set()
        var_dims = {}
        for var in all_vars:
            dv = self.get_dim_vector(var)
            if dv:
                var_dims[var] = dv
                dims_used.update(dv.dims.keys())
            else:
                var_dims[var] = DimVector({})
                self._log(f"Peringatan: dimensi {var} tidak diketahui, asumsikan tak berdimensi")

        dims_used = [d for d in BASE_DIMS if d in dims_used]
        n = len(all_vars)    # Total variabel
        k = len(dims_used)   # Dimensi dasar yang digunakan

        self._log(f"Variabel: {all_vars}")
        self._log(f"Dimensi: {dims_used}")
        self._log(f"n={n}, k={k}, Pi-groups = {n-k}")

        # Bangun matriks dimensional
        dim_matrix = self._build_dimensional_matrix(all_vars, dims_used)

        # Konversi ke Fraction untuk presisi
        frac_matrix = [[fractions.Fraction(x) for x in row] for row in dim_matrix]
        rank = self._gaussian_elimination([row[:] for row in frac_matrix], n)

        n_pi = n - rank  # Jumlah Pi-groups

        explanation_lines = [
            f"Variabel: {all_vars}",
            f"Dimensi yang digunakan: {dims_used}",
            f"n = {n} variabel, rank(D) = {rank}",
            f"Jumlah Pi-groups = {n} - {rank} = {n_pi}",
        ]

        # ── Generasi Kandidat Formula ─────────────────────────────────────
        candidates = []

        # Metode 1: Power Law — Y = c × X1^a × X2^b × ...
        # Selesaikan: sum(a_i × dim_vec(X_i)) = dim_vec(Y)
        target_dv = var_dims[target_var]
        power_law_candidates = self._solve_power_law(
            target_var, target_dv, input_vars, var_dims, dims_used
        )
        candidates.extend(power_law_candidates)

        # Metode 2: Additive — Y = X1 + X2 (jika dimensi sama)
        same_dim_vars = [v for v in input_vars
                         if var_dims.get(v) == target_dv and not target_dv.is_dimensionless()]
        if len(same_dim_vars) >= 2:
            expr = " + ".join(same_dim_vars[:3])
            candidates.append(DimensionalFormula(
                expression=f"{target_var} = {expr}",
                exponents={v: fractions.Fraction(1) for v in same_dim_vars[:3]},
                pi_groups=[],
                dimensional_check=True,
                source="additive_synthesis",
            ))

        # Metode 3: Pi-groups (jika ada)
        if n_pi > 0:
            pi_groups = self._compute_pi_groups(all_vars, var_dims, dims_used, rank)
            explanation_lines.append(f"Pi-groups: {pi_groups}")
        else:
            pi_groups = []

        explanation = "\n".join(explanation_lines)

        return SynthesisResult(
            target=target_var,
            variables=input_vars,
            candidate_formulas=candidates,
            pi_groups=pi_groups,
            n_pi_groups=n_pi,
            dimensional_matrix=dim_matrix,
            explanation=explanation,
        )

    def _solve_power_law(
        self,
        target: str,
        target_dv: DimVector,
        input_vars: List[str],
        var_dims: Dict[str, DimVector],
        dims_used: List[str]
    ) -> List[DimensionalFormula]:
        """
        Cari eksponen power law: target = c × X1^a × X2^b × ...
        Dengan constraint: dimensi kedua sisi harus sama.

        Untuk masalah dengan dimensi sederhana, solusi seringkali unik.
        """
        results = []
        n_in = len(input_vars)

        # Bangun sistem persamaan linear:
        # Untuk setiap dimensi d: sum(a_i × dim_d(X_i)) = dim_d(target)
        # Variabel yang dicari: a_1, a_2, ..., a_n

        # Encode sebagai matriks A × x = b
        A = []
        b = []
        for dim in dims_used:
            row = [var_dims[v].dims.get(dim, 0) for v in input_vars]
            rhs = target_dv.dims.get(dim, 0)
            A.append([fractions.Fraction(x) for x in row])
            b.append(fractions.Fraction(rhs))

        # Coba cari solusi dengan Gaussian elimination (augmented matrix)
        if not A:
            return results

        n_eq = len(A)
        n_var = len(input_vars)

        # Augmented matrix [A | b]
        aug = [A[i] + [b[i]] for i in range(n_eq)]

        # Gaussian elimination
        row_ptr = 0
        pivot_cols = []
        for col in range(n_var):
            # Find pivot
            piv = None
            for r in range(row_ptr, n_eq):
                if aug[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                continue
            aug[row_ptr], aug[piv] = aug[piv], aug[row_ptr]
            piv_val = aug[row_ptr][col]
            aug[row_ptr] = [x / piv_val for x in aug[row_ptr]]
            for r in range(n_eq):
                if r != row_ptr and aug[r][col] != 0:
                    fac = aug[r][col]
                    aug[r] = [aug[r][c] - fac * aug[row_ptr][c]
                               for c in range(len(aug[r]))]
            pivot_cols.append(col)
            row_ptr += 1

        free_cols = [c for c in range(n_var) if c not in pivot_cols]
        n_free = len(free_cols)

        # Ekstrak solusi
        # Jika ada free variables, generate beberapa kombinasi
        free_value_sets = [[fractions.Fraction(0)] * n_free]
        for i in range(n_free):
            row = [fractions.Fraction(0)] * n_free
            row[i] = fractions.Fraction(1)
            free_value_sets.append(row)

        for free_vals in free_value_sets:
            exponents = {}
            valid = True

            # Assign free variable values
            for i, fc in enumerate(free_cols):
                exponents[input_vars[fc]] = free_vals[i] if i < len(free_vals) else fractions.Fraction(0)

            # Solve pivot variables back-substitute
            for i, pc in enumerate(pivot_cols):
                if i >= row_ptr:
                    break
                # Find the row with this pivot
                for r in range(n_eq):
                    if aug[r][pc] == 1 and all(
                        aug[r][c] == 0 for c in pivot_cols if c != pc
                    ):
                        val = aug[r][-1]
                        for fc_idx, fc in enumerate(free_cols):
                            val -= aug[r][fc] * free_vals[fc_idx]
                        exponents[input_vars[pc]] = val
                        break

            if not valid:
                continue

            # Build formula string
            parts = []
            for v in input_vars:
                exp = exponents.get(v, fractions.Fraction(0))
                if exp == 0:
                    continue
                elif exp == 1:
                    parts.append(v)
                elif exp == fractions.Fraction(1, 2):
                    parts.append(f"√{v}")
                elif exp == fractions.Fraction(-1, 2):
                    parts.append(f"1/√{v}")
                elif exp > 0:
                    parts.append(f"{v}^{exp}")
                else:
                    parts.append(f"{v}^({exp})")

            if not parts:
                formula_str = f"{target} = c (konstanta)"
            else:
                formula_str = f"{target} = c × {'×'.join(parts)}"

            # Verifikasi dimensi
            lhs = target_dv
            rhs = DimVector({})
            for v, exp in exponents.items():
                dv = var_dims.get(v, DimVector({}))
                for dim, exp_dim in dv.dims.items():
                    rhs.dims[dim] = rhs.dims.get(dim, 0) + int(exp * exp_dim)
            rhs.dims = {k: v for k, v in rhs.dims.items() if v != 0}

            is_valid = (lhs == rhs)

            results.append(DimensionalFormula(
                expression=formula_str,
                exponents={v: e for v, e in exponents.items()},
                pi_groups=[],
                dimensional_check=is_valid,
                source="power_law_synthesis",
            ))

        return [r for r in results if r.dimensional_check][:5]  # Max 5 kandidat

    def _compute_pi_groups(
        self, all_vars: List[str],
        var_dims: Dict[str, DimVector],
        dims_used: List[str],
        rank: int
    ) -> List[str]:
        """Hitung Pi-groups dengan pemilihan variabel repeating."""
        if rank >= len(all_vars):
            return []

        # Pilih variabel repeating (rank pertama yang bisa membentuk matriks full rank)
        repeating = all_vars[:min(rank, len(all_vars))]
        remaining  = all_vars[rank:]

        pi_groups = []
        for i, var in enumerate(remaining):
            pi_str = f"π_{i+1} = {var} / ({' × '.join(repeating)})"
            pi_groups.append(pi_str)

        return pi_groups

    def explain(self, target: str, inputs: List[str]) -> str:
        """
        Tampilkan penjelasan analisis dimensional dalam format yang mudah dibaca.
        """
        result = self.synthesize(target, inputs)
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  🔬 MOKO Dimensional Synthesis")
        lines.append(f"  Mencari: {target} = f({', '.join(inputs)})")
        lines.append(f"{'='*60}")
        lines.append(f"\n{result.explanation}")
        lines.append(f"\n  Kandidat Formula ({len(result.candidate_formulas)} ditemukan):")
        for i, cand in enumerate(result.candidate_formulas, 1):
            icon = "✅" if cand.dimensional_check else "❌"
            lines.append(f"  {icon} [{i}] {cand.expression}")
            lines.append(f"       Sumber: {cand.source}")
        if not result.candidate_formulas:
            lines.append("  ⚠️  Tidak ada kandidat yang konsisten secara dimensional.")
            lines.append("  Hint: Tambahkan konstanta fisika (g, R_gas, c_light) sebagai input.")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ── INTEGRASI DENGAN FORMULA ENGINE ───────────────────────────────────────────

class DimensionalFallback:
    """
    Digunakan sebagai fallback di AppliedFormulaEngine:
    Jika tidak ada formula di database → coba sintesis dari dimensi.
    """

    def __init__(self, verbose: bool = False):
        self._engine = DimensionalSynthesisEngine(verbose=verbose)

    def attempt_synthesis(
        self,
        target: str,
        known_vars: Dict[str, float],
        domain: str
    ) -> Optional[str]:
        """
        Coba sintesis formula dari variabel yang tersedia.
        Return: ekspresi formula string jika berhasil, None jika tidak.
        """
        inputs = list(known_vars.keys())
        result = self._engine.synthesize(target, inputs)

        if result.candidate_formulas:
            best = result.candidate_formulas[0]
            return best.expression

        return None


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_dim_engine: Optional[DimensionalSynthesisEngine] = None

def get_dim_synthesis_engine(verbose: bool = False) -> DimensionalSynthesisEngine:
    global _dim_engine
    if _dim_engine is None:
        _dim_engine = DimensionalSynthesisEngine(verbose=verbose)
    return _dim_engine


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*65)
    print("  MOKO Dimensional Synthesis Engine — Self Test")
    print("="*65)

    engine = DimensionalSynthesisEngine(verbose=True)

    tests = [
        ("F",     ["P", "A"],       "F = P × A (Hukum Pascal)"),
        ("Ek",    ["m", "v"],       "Ek = ½mv² (Energi Kinetik)"),
        ("F",     ["m", "a"],       "F = ma (Hukum Newton II)"),
        ("V",     ["I", "R"],       "V = IR (Hukum Ohm)"),
        ("Q_heat",["m", "c_heat", "ΔT"], "Q = mcΔT (Kalor spesifik)"),
    ]

    print()
    for target, inputs, expected in tests:
        result = engine.synthesize(target, inputs)
        has_valid = any(c.dimensional_check for c in result.candidate_formulas)
        icon = "✅" if has_valid else "❌"
        best = result.candidate_formulas[0].expression if result.candidate_formulas else "Tidak ditemukan"
        print(f"  {icon} Target: {expected}")
        print(f"       Kandidat: {best}")
        print()

    # Demo explain
    print(engine.explain("F", ["P", "A"]))
