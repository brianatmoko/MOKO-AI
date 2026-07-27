"""
MOKO Pattern Conjecturer (Ramanujan-style)
===========================================
Menganalisis deret angka/sekuensial numerik untuk menemukan rumus eksplisit (closed-form).
Berguna untuk mengenali pola deret bilangan integer, deret pertumbuhan, deret fraksional,
serta memformulasikan hubungan non-linier dari data masukan sekuensial.

Metode deteksi:
  1. Finite Differences (Linear, Kuadratik, Kubik, dst.)
  2. Geometric Ratios (Deret Eksponensial/Geometris)
  3. Pola khusus (Fibonacci, Faktorial, Kuadrat Sempurna)
  4. Continued Fractions (untuk hampiran rasional)

Lompatan ke Level 3 (Discovery) murni berbasis matematika diskrit.
"""

import math
import fractions
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class PatternResult:
    formula_str: str          # Representasi rumus: e.g., "a(n) = n^2 + n"
    latex_str: str            # Format LaTeX
    confidence: float         # Tingkat keyakinan (0.0 - 1.0)
    pattern_type: str         # "polynomial", "geometric", "factorial", "fibonacci", "constant"
    next_values: List[float]  # Prediksi 3 nilai berikutnya
    sympy_expr_str: str       # Ekspresi string untuk SymPy


class PatternConjecturer:
    """
    Mendeteksi pola dari sekuens angka numerik dan menyusun formula konjektur.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Conjecturer] {msg}")

    def analyze(self, seq: List[float]) -> Optional[PatternResult]:
        """
        Menganalisis sekuens angka dan menyusun rumus.
        Seq diasumsikan diindeks mulai dari n = 1, 2, 3, ...
        """
        if len(seq) < 4:
            self._log("Sekuens terlalu pendek untuk analisis terpercaya (min 4 elemen)")
            return None

        # Hapus floating point noise kecil
        seq = [round(x, 9) for x in seq]

        # 1. Cek konstanta
        if all(x == seq[0] for x in seq):
            val = seq[0]
            val_str = str(int(val)) if val.is_integer() else f"{val:.4g}"
            return PatternResult(
                formula_str=f"a(n) = {val_str}",
                latex_str=f"a_n = {val_str}",
                confidence=1.0,
                pattern_type="constant",
                next_values=[val, val, val],
                sympy_expr_str=f"{val_str}"
            )

        # 2. Cek Finite Differences (Polinomial)
        poly_res = self._check_finite_differences(seq)
        if poly_res:
            return poly_res

        # 3. Cek Geometric/Exponential Ratio
        geom_res = self._check_geometric_ratio(seq)
        if geom_res:
            return geom_res

        # 4. Cek Pola Khusus (Faktorial, Fibonacci)
        special_res = self._check_special_patterns(seq)
        if special_res:
            return special_res

        return None

    def _check_finite_differences(self, seq: List[float]) -> Optional[PatternResult]:
        """
        Metode Finite Differences untuk mendeteksi orde polinomial:
        - Orde 1: Linear (a(n) = d*n + c)
        - Orde 2: Kuadratik (a(n) = A*n^2 + B*n + C)
        - Orde 3: Kubik
        """
        n_len = len(seq)
        working_seq = list(seq)
        diffs = [working_seq]
        
        # Max orde yang ditest tergantung panjang sekuens
        max_order = min(4, n_len - 2)

        for order in range(1, max_order + 1):
            next_diff = []
            for i in range(len(diffs[-1]) - 1):
                next_diff.append(round(diffs[-1][i+1] - diffs[-1][i], 9))
            diffs.append(next_diff)
            
            # Cek jika selisihnya konstan
            if len(next_diff) >= 2 and all(x == next_diff[0] for x in next_diff):
                # Polinomial dengan orde 'order' terdeteksi!
                coefs = self._solve_polynomial_coefficients(seq, order)
                if coefs:
                    return self._build_poly_result(coefs, seq)

        return None

    def _solve_polynomial_coefficients(self, seq: List[float], order: int) -> Optional[List[float]]:
        """
        Selesaikan koefisien polinomial menggunakan sistem linear sederhana.
        Menghasilkan [c_k, ..., c_1, c_0] untuk sum(c_i * n^i)
        """
        # Kita selesaikan sistem persamaan linear:
        # Untuk n = 1, 2, ..., order+1:
        # c_k * n^k + ... + c_1 * n + c_0 = seq[n-1]
        
        # Implementasi Gaussian Elimination sederhana
        n_eq = order + 1
        if len(seq) < n_eq:
            return None

        # Matriks augmented [A | B]
        matrix = []
        for row in range(1, n_eq + 1):
            eq = []
            for col in range(order, -1, -1):
                eq.append(float(row ** col))
            eq.append(float(seq[row - 1]))
            matrix.append(eq)

        # Gaussian Elimination
        for i in range(n_eq):
            # Pivot
            pivot = matrix[i][i]
            if abs(pivot) < 1e-9:
                return None
            for j in range(i, n_eq + 1):
                matrix[i][j] /= pivot
            for k in range(n_eq):
                if k != i:
                    factor = matrix[k][i]
                    for j in range(i, n_eq + 1):
                        matrix[k][j] -= factor * matrix[i][j]

        # Ambil hasil
        coefs = [matrix[r][-1] for r in range(n_eq)]
        return coefs

    def _build_poly_result(self, coefs: List[float], seq: List[float]) -> PatternResult:
        """Menyusun representasi string dari koefisien polinomial."""
        order = len(coefs) - 1
        parts = []
        latex_parts = []
        sympy_parts = []

        for i, c in enumerate(coefs):
            power = order - i
            if abs(c) < 1e-9:
                continue

            # Format tanda dan koefisien
            c_round = round(c, 5)
            c_str = ""
            if power == 0 or abs(c_round - 1.0) > 1e-5:
                if abs(c_round - int(c_round)) < 1e-9:
                    c_str = str(int(c_round))
                else:
                    c_str = f"{c_round:.4g}"

            if c_str == "1" and power > 0:
                c_str = ""
            if c_str == "-1" and power > 0:
                c_str = "-"

            # Tambahkan tanda tambah jika bukan suku pertama
            sign = ""
            if parts and not c_str.startswith("-"):
                sign = " + "
            elif parts and c_str.startswith("-"):
                sign = " - "
                c_str = c_str[1:]

            term = c_str
            latex_term = c_str
            sympy_term = c_str if c_str else "1"

            if power > 0:
                term += "n"
                latex_term += "n"
                sympy_term = f"{sympy_term}*n" if sympy_term not in ("", "-") else f"{sympy_term}n"
                if power > 1:
                    term += f"^{power}"
                    latex_term += f"^{power}"
                    sympy_term += f"**{power}"

            parts.append(f"{sign}{term}")
            latex_parts.append(f"{sign}{latex_term}")
            sympy_parts.append(f"{sign if sign != ' - ' else '-'}{sympy_term}")

        formula_str = f"a(n) = {''.join(parts)}"
        latex_str = f"a_n = {''.join(latex_parts)}"
        
        # Hapus spasi/tanda aneh di awal sympy
        sympy_expr = "".join(sympy_parts).strip()
        if sympy_expr.startswith("+ "):
            sympy_expr = sympy_expr[2:]

        # Hitung prediksi berikutnya
        next_n = [len(seq) + 1, len(seq) + 2, len(seq) + 3]
        next_vals = []
        for n in next_n:
            val = sum(c * (n ** (order - i)) for i, c in enumerate(coefs))
            next_vals.append(val)

        return PatternResult(
            formula_str=formula_str,
            latex_str=latex_str,
            confidence=0.95,
            pattern_type="polynomial",
            next_values=next_vals,
            sympy_expr_str=sympy_expr
        )

    def _check_geometric_ratio(self, seq: List[float]) -> Optional[PatternResult]:
        """Mendeteksi deret geometris/eksponensial: a(n) = a * r^(n-1)."""
        if any(x == 0 for x in seq):
            return None

        ratios = []
        for i in range(len(seq) - 1):
            ratios.append(round(seq[i+1] / seq[i], 9))

        if len(ratios) >= 2 and all(r == ratios[0] for r in ratios):
            r = ratios[0]
            a = seq[0]
            
            # Buat representasi pembilang/penyebut untuk presisi
            r_frac = fractions.Fraction(r).limit_denominator(1000)
            a_frac = fractions.Fraction(a).limit_denominator(1000)

            a_str = str(a_frac) if a_frac.denominator > 1 else str(a_frac.numerator)
            r_str = str(r_frac) if r_frac.denominator > 1 else str(r_frac.numerator)

            if r_frac.denominator > 1:
                r_display = f"({r_str})"
            else:
                r_display = r_str

            formula_str = f"a(n) = {a_str} × {r_display}^(n-1)"
            latex_str = f"a_n = {a_str} \\cdot {r_display}^{{n-1}}"
            sympy_expr = f"{a_frac} * ({r_frac})**(n-1)"

            next_vals = [seq[-1] * r, seq[-1] * (r**2), seq[-1] * (r**3)]

            return PatternResult(
                formula_str=formula_str,
                latex_str=latex_str,
                confidence=0.98,
                pattern_type="geometric",
                next_values=next_vals,
                sympy_expr_str=sympy_expr
            )

        return None

    def _check_special_patterns(self, seq: List[float]) -> Optional[PatternResult]:
        """Mengecek deret faktorial (n!) atau Fibonacci."""
        n_len = len(seq)
        
        # 1. Faktorial check: seq = [1, 2, 6, 24, 120, ...]
        factorials = [math.factorial(n) for n in range(1, n_len + 1)]
        if seq == factorials:
            next_n = n_len + 1
            return PatternResult(
                formula_str="a(n) = n!",
                latex_str="a_n = n!",
                confidence=1.0,
                pattern_type="factorial",
                next_values=[float(math.factorial(next_n)), float(math.factorial(next_n+1)), float(math.factorial(next_n+2))],
                sympy_expr_str="factorial(n)"
            )

        # 2. Fibonacci check: seq = [1, 1, 2, 3, 5, 8, 13, ...] atau pergeseran
        fib = [1, 1]
        for _ in range(n_len + 5):
            fib.append(fib[-1] + fib[-2])

        # Cek kecocokan sub-list
        for offset in range(5):
            sub_fib = fib[offset:offset+n_len]
            if seq == sub_fib:
                next_vals = fib[offset+n_len:offset+n_len+3]
                next_vals_f = [float(x) for x in next_vals]
                offset_str = f"+{offset-1}" if offset-1 > 0 else (f"-{1-offset}" if offset-1 < 0 else "")
                return PatternResult(
                    formula_str=f"a(n) = F(n{offset_str})",
                    latex_str=f"a_n = F_{{n{offset_str}}}",
                    confidence=0.99,
                    pattern_type="fibonacci",
                    next_values=next_vals_f,
                    sympy_expr_str=f"fibonacci(n{offset_str})"
                )

        return None


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_conjecturer_instance: Optional[PatternConjecturer] = None

def get_pattern_conjecturer(verbose: bool = False) -> PatternConjecturer:
    global _conjecturer_instance
    if _conjecturer_instance is None:
        _conjecturer_instance = PatternConjecturer(verbose=verbose)
    return _conjecturer_instance


# ── DEMO & TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    con = PatternConjecturer(verbose=True)

    tests = [
        [1, 3, 5, 7, 9, 11],          # 2n - 1
        [2, 6, 12, 20, 30, 42],       # n^2 + n
        [3, 9, 27, 81, 243],          # 3^n
        [1, 2, 6, 24, 120],           # n!
        [1, 2, 3, 5, 8, 13],          # F(n+1)
        [5, 5, 5, 5, 5],              # Constant
    ]

    print("=== MOKO Pattern Conjecturer Self Test ===")
    for seq in tests:
        res = con.analyze(seq)
        if res:
            print(f"Sekuens   : {seq}")
            print(f"Pola      : {res.pattern_type.upper()}")
            print(f"Rumus     : {res.formula_str}")
            print(f"Prediksi  : {res.next_values}")
            print(f"SymPy     : {res.sympy_expr_str}")
            print("-" * 50)
        else:
            print(f"Sekuens   : {seq} -> Pola tidak terdeteksi")
            print("-" * 50)
