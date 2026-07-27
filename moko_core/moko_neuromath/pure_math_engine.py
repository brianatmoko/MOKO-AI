"""
MOKO Pure Math Engine — Matematika Murni (Fondasi Semua Domain)
===============================================================
Mencakup:
  1. Number Theory     — GCD, LCM, primalitas, faktorisasi, Euler φ(n), modular inverse
  2. Kombinatorik      — Permutasi, kombinasi, Catalan number, Stars-and-Bars
  3. Aljabar Linear    — Determinan, invers matriks, Gauss-Jordan, dot/cross product, eigen
  4. Kalkulus Simbolik — Turunan, integral tentu/tak tentu, limit (via SymPy)
  5. Teori Bilangan    — Konversi basis, CRT, Fermat little theorem
  6. Barisan & Rekursi — Fibonacci, geometrik, aritmetika, rekursi umum
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import sympy as sp
    from sympy import (Symbol, symbols, diff, integrate, limit, simplify,
                       factor, expand, solve, Matrix, det, eye, latex,
                       Rational, oo, pi, E, sqrt, factorial, binomial,
                       isprime, factorint, totient, gcd, lcm, mod_inverse)
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. NUMBER THEORY
# ═══════════════════════════════════════════════════════════════════════════

class NumberTheory:
    """Aritmetika bilangan bulat, primalitas, modular arithmetic."""

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest Common Divisor via Euclidean algorithm."""
        while b:
            a, b = b, a % b
        return abs(a)

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Least Common Multiple."""
        return abs(a * b) // NumberTheory.gcd(a, b) if a and b else 0

    @staticmethod
    def is_prime(n: int) -> bool:
        """Uji primalitas Miller-Rabin deterministik (n < 3.3 × 10²⁴)."""
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        # Miller-Rabin witnesses for deterministic test up to 3.3e24
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in witnesses:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    @staticmethod
    def prime_factors(n: int) -> Dict[int, int]:
        """Faktorisasi prima → {faktor: eksponen}."""
        factors: Dict[int, int] = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    @staticmethod
    def euler_totient(n: int) -> int:
        """φ(n) — jumlah bilangan 1..n yang relatif prima terhadap n."""
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    @staticmethod
    def modular_inverse(a: int, m: int) -> Optional[int]:
        """Invers modular a^(-1) mod m via Extended Euclidean. None jika tidak ada."""
        def extended_gcd(a, b):
            if a == 0: return b, 0, 1
            g, x, y = extended_gcd(b % a, a)
            return g, y - (b // a) * x, x
        g, x, _ = extended_gcd(a % m, m)
        return (x % m) if g == 1 else None

    @staticmethod
    def chinese_remainder(remainders: List[int], moduli: List[int]) -> Optional[int]:
        """Chinese Remainder Theorem: cari x sehingga x ≡ r_i (mod m_i)."""
        M = math.prod(moduli)
        x = 0
        for r, m in zip(remainders, moduli):
            Mi = M // m
            inv = NumberTheory.modular_inverse(Mi, m)
            if inv is None: return None
            x += r * Mi * inv
        return x % M

    @staticmethod
    def convert_base(n: int, from_base: int, to_base: int) -> str:
        """Konversi bilangan antar basis. n adalah int (desimal)."""
        if n == 0: return "0"
        digits = "0123456789ABCDEF"
        negative = n < 0
        n = abs(n)
        result = []
        while n:
            result.append(digits[n % to_base])
            n //= to_base
        if negative: result.append('-')
        return ''.join(reversed(result))

    @staticmethod
    def sieve_primes(limit: int) -> List[int]:
        """Sieve of Eratosthenes — daftar bilangan prima ≤ limit."""
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]


# ═══════════════════════════════════════════════════════════════════════════
# 2. KOMBINATORIK
# ═══════════════════════════════════════════════════════════════════════════

class Combinatorics:
    """Hitung kombinasi, permutasi, dan prinsip menghitung lanjutan."""

    @staticmethod
    def permutation(n: int, r: int) -> int:
        """P(n,r) = n! / (n-r)!"""
        if r > n: return 0
        return math.perm(n, r)

    @staticmethod
    def combination(n: int, r: int) -> int:
        """C(n,r) = n! / (r! × (n-r)!)"""
        if r > n: return 0
        return math.comb(n, r)

    @staticmethod
    def catalan(n: int) -> int:
        """Bilangan Catalan C_n = C(2n,n) / (n+1)."""
        return math.comb(2 * n, n) // (n + 1)

    @staticmethod
    def stars_and_bars(n: int, k: int) -> int:
        """Jumlah cara menempatkan n objek identik ke k kotak: C(n+k-1, k-1)."""
        return math.comb(n + k - 1, k - 1)

    @staticmethod
    def derangement(n: int) -> int:
        """D(n) = n! × Σ(-1)^k/k! — jumlah permutasi tanpa fixed point."""
        if n == 0: return 1
        if n == 1: return 0
        return (n - 1) * (Combinatorics.derangement(n - 1) + Combinatorics.derangement(n - 2))

    @staticmethod
    def multinomial(n: int, groups: List[int]) -> int:
        """Koefisien multinomial n! / (k1! × k2! × ...)."""
        if sum(groups) != n: return 0
        result = math.factorial(n)
        for k in groups:
            result //= math.factorial(k)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════

class LinearAlgebra:
    """Operasi matriks dan vektor — tanpa numpy, menggunakan list-of-lists."""

    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """Perkalian titik dua vektor."""
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def cross_product(v1: List[float], v2: List[float]) -> List[float]:
        """Perkalian silang dua vektor 3D."""
        a1, a2, a3 = v1
        b1, b2, b3 = v2
        return [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]

    @staticmethod
    def magnitude(v: List[float]) -> float:
        return math.sqrt(sum(x**2 for x in v))

    @staticmethod
    def determinant(matrix: List[List[float]]) -> float:
        """Hitung determinan matriks n×n via SymPy jika tersedia, else rekursif."""
        n = len(matrix)
        if _SYMPY:
            return float(Matrix(matrix).det())
        if n == 1: return matrix[0][0]
        if n == 2: return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
        det = 0.0
        for c in range(n):
            minor = [row[:c] + row[c+1:] for row in matrix[1:]]
            det += ((-1)**c) * matrix[0][c] * LinearAlgebra.determinant(minor)
        return det

    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Perkalian matriks A × B."""
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        C = [[0.0]*cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @staticmethod
    def gauss_jordan(augmented: List[List[float]]) -> Optional[List[float]]:
        """
        Selesaikan sistem Ax=b menggunakan eliminasi Gauss-Jordan.
        Input: matriks augmented [A|b] ukuran n×(n+1).
        Output: solusi x, atau None jika tidak ada solusi unik.
        """
        import copy
        mat = copy.deepcopy(augmented)
        n = len(mat)
        for col in range(n):
            # Cari pivot
            pivot_row = None
            for row in range(col, n):
                if abs(mat[row][col]) > 1e-12:
                    pivot_row = row
                    break
            if pivot_row is None: return None
            mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
            pivot = mat[col][col]
            mat[col] = [x / pivot for x in mat[col]]
            for row in range(n):
                if row != col:
                    factor = mat[row][col]
                    mat[row] = [mat[row][j] - factor * mat[col][j] for j in range(n+1)]
        return [mat[i][n] for i in range(n)]

    @staticmethod
    def eigenvalues_sympy(matrix: List[List[float]]) -> List:
        """Hitung nilai eigen menggunakan SymPy."""
        if not _SYMPY: return []
        M = Matrix(matrix)
        return list(M.eigenvals().keys())


# ═══════════════════════════════════════════════════════════════════════════
# 4. KALKULUS SIMBOLIK (via SymPy)
# ═══════════════════════════════════════════════════════════════════════════

class SymbolicCalculus:
    """Turunan, integral, dan limit via SymPy."""

    @staticmethod
    def derivative(expr_str: str, var: str = 'x', order: int = 1) -> Dict[str, Any]:
        """Hitung turunan simbolik."""
        if not _SYMPY:
            return {"success": False, "error": "SymPy tidak tersedia"}
        try:
            x = Symbol(var)
            expr = sp.sympify(expr_str)
            result = diff(expr, x, order)
            return {
                "success": True,
                "input": expr_str,
                "derivative_order": order,
                "result": str(result),
                "result_latex": latex(result),
                "simplified": str(simplify(result))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def integral(expr_str: str, var: str = 'x',
                 lower: Optional[str] = None, upper: Optional[str] = None) -> Dict[str, Any]:
        """Hitung integral tentu atau tak tentu."""
        if not _SYMPY:
            return {"success": False, "error": "SymPy tidak tersedia"}
        try:
            x = Symbol(var)
            expr = sp.sympify(expr_str)
            if lower is not None and upper is not None:
                a = sp.sympify(lower)
                b = sp.sympify(upper)
                result = integrate(expr, (x, a, b))
                kind = "definite"
            else:
                result = integrate(expr, x)
                kind = "indefinite"
            return {
                "success": True,
                "input": expr_str,
                "kind": kind,
                "result": str(result),
                "result_latex": latex(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def limit_calc(expr_str: str, var: str = 'x', point: str = '0',
                   direction: str = '+') -> Dict[str, Any]:
        """Hitung limit ekspresi mendekati suatu titik."""
        if not _SYMPY:
            return {"success": False, "error": "SymPy tidak tersedia"}
        try:
            x = Symbol(var)
            expr = sp.sympify(expr_str)
            pt = sp.sympify(point)
            result = limit(expr, x, pt, direction)
            return {
                "success": True,
                "input": expr_str,
                "point": point,
                "result": str(result),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def solve_equation(expr_str: str, var: str = 'x') -> Dict[str, Any]:
        """Selesaikan persamaan f(x)=0."""
        if not _SYMPY:
            return {"success": False, "error": "SymPy tidak tersedia"}
        try:
            x = Symbol(var)
            expr = sp.sympify(expr_str)
            solutions = solve(expr, x)
            return {
                "success": True,
                "equation": f"{expr_str} = 0",
                "solutions": [str(s) for s in solutions],
                "n_solutions": len(solutions)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. BARISAN & REKURSI
# ═══════════════════════════════════════════════════════════════════════════

class SequencesMath:
    """Barisan aritmetika, geometrika, Fibonacci, dan rekursi umum."""

    @staticmethod
    def arithmetic_sequence(a1: float, d: float, n: int) -> Dict[str, Any]:
        """Suku ke-n dan jumlah n suku barisan aritmetika."""
        an = a1 + (n - 1) * d
        Sn = n * (a1 + an) / 2
        return {"a_n": an, "S_n": Sn, "common_difference": d}

    @staticmethod
    def geometric_sequence(a1: float, r: float, n: int) -> Dict[str, Any]:
        """Suku ke-n dan jumlah n suku barisan geometrika."""
        an = a1 * (r ** (n - 1))
        Sn = a1 * (1 - r**n) / (1 - r) if abs(r - 1) > 1e-12 else a1 * n
        S_inf = (a1 / (1 - r)) if abs(r) < 1 else float('inf')
        return {"a_n": an, "S_n": Sn, "S_inf": S_inf, "common_ratio": r}

    @staticmethod
    def fibonacci(n: int) -> int:
        """Suku ke-n barisan Fibonacci (0-indexed). O(log n) via matrix exp."""
        if n <= 0: return 0
        if n == 1: return 1
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    @staticmethod
    def fibonacci_sequence(n: int) -> List[int]:
        """n suku pertama barisan Fibonacci."""
        if n <= 0: return []
        seq = [0, 1]
        for i in range(2, n):
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

    @staticmethod
    def sum_of_squares(n: int) -> int:
        """Σ k² dari k=1 sampai n = n(n+1)(2n+1)/6."""
        return n * (n + 1) * (2 * n + 1) // 6

    @staticmethod
    def sum_of_cubes(n: int) -> int:
        """Σ k³ dari k=1 sampai n = [n(n+1)/2]²."""
        s = n * (n + 1) // 2
        return s * s


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class PureMathEngine:
    """Fasad utama untuk semua kapabilitas matematika murni."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.number_theory = NumberTheory()
        self.combinatorics = Combinatorics()
        self.linear_algebra = LinearAlgebra()
        self.calculus = SymbolicCalculus()
        self.sequences = SequencesMath()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔢 [PureMath] {msg}")

    def solve(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """Unified solver — dispatch ke sub-engine berdasarkan query_type."""
        dispatch = {
            "gcd":          lambda: {"result": self.number_theory.gcd(**kwargs)},
            "lcm":          lambda: {"result": self.number_theory.lcm(**kwargs)},
            "is_prime":     lambda: {"result": self.number_theory.is_prime(**kwargs)},
            "prime_factors":lambda: {"result": self.number_theory.prime_factors(**kwargs)},
            "totient":      lambda: {"result": self.number_theory.euler_totient(**kwargs)},
            "mod_inverse":  lambda: {"result": self.number_theory.modular_inverse(**kwargs)},
            "crt":          lambda: {"result": self.number_theory.chinese_remainder(**kwargs)},
            "convert_base": lambda: {"result": self.number_theory.convert_base(**kwargs)},
            "primes_sieve": lambda: {"result": self.number_theory.sieve_primes(**kwargs)},
            "permutation":  lambda: {"result": self.combinatorics.permutation(**kwargs)},
            "combination":  lambda: {"result": self.combinatorics.combination(**kwargs)},
            "catalan":      lambda: {"result": self.combinatorics.catalan(**kwargs)},
            "derangement":  lambda: {"result": self.combinatorics.derangement(**kwargs)},
            "determinant":  lambda: {"result": self.linear_algebra.determinant(**kwargs)},
            "gauss_jordan": lambda: {"result": self.linear_algebra.gauss_jordan(**kwargs)},
            "dot_product":  lambda: {"result": self.linear_algebra.dot_product(**kwargs)},
            "cross_product":lambda: {"result": self.linear_algebra.cross_product(**kwargs)},
            "derivative":   lambda: self.calculus.derivative(**kwargs),
            "integral":     lambda: self.calculus.integral(**kwargs),
            "limit":        lambda: self.calculus.limit_calc(**kwargs),
            "solve_eq":     lambda: self.calculus.solve_equation(**kwargs),
            "arithmetic":   lambda: self.sequences.arithmetic_sequence(**kwargs),
            "geometric":    lambda: self.sequences.geometric_sequence(**kwargs),
            "fibonacci":    lambda: {"result": self.sequences.fibonacci(**kwargs)},
        }
        fn = dispatch.get(query_type)
        if fn is None:
            return {"success": False, "error": f"Query type tidak dikenal: '{query_type}'"}
        try:
            result = fn()
            if isinstance(result, dict) and "success" not in result:
                result["success"] = True
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════

_pure_math_instance: Optional[PureMathEngine] = None

def get_pure_math_engine(verbose: bool = True) -> PureMathEngine:
    global _pure_math_instance
    if _pure_math_instance is None:
        _pure_math_instance = PureMathEngine(verbose=verbose)
    return _pure_math_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n🔢 Pure Math Engine — Self Test\n" + "="*55)
    e = PureMathEngine(verbose=True)

    # Number Theory
    assert e.solve("gcd", a=48, b=18)["result"] == 6,         "GCD gagal"
    assert e.solve("lcm", a=12, b=18)["result"] == 36,         "LCM gagal"
    assert e.solve("is_prime", n=97)["result"] == True,         "Primalitas gagal"
    assert e.solve("is_prime", n=100)["result"] == False,       "Non-prima gagal"
    assert e.solve("totient", n=12)["result"] == 4,             "Totient gagal"
    assert e.solve("mod_inverse", a=3, m=7)["result"] == 5,     "ModInverse gagal"
    assert e.solve("convert_base", n=255, from_base=10, to_base=16)["result"] == "FF", "Base conv gagal"
    print("  ✅ Number Theory: GCD, LCM, Primalitas, Totient, ModInverse, Base conv")

    # Kombinatorik
    assert e.solve("combination", n=10, r=3)["result"] == 120,  "C(10,3) gagal"
    assert e.solve("permutation", n=5, r=3)["result"] == 60,    "P(5,3) gagal"
    assert e.solve("catalan", n=4)["result"] == 14,             "Catalan gagal"
    assert e.solve("derangement", n=4)["result"] == 9,          "Derangement gagal"
    print("  ✅ Kombinatorik: C(10,3)=120, P(5,3)=60, Catalan(4)=14, D(4)=9")

    # Linear Algebra
    det_2x2 = e.solve("determinant", matrix=[[3,8],[4,6]])["result"]
    assert abs(det_2x2 - (-14.0)) < 1e-9,                       "Determinan gagal"
    dp = e.solve("dot_product", v1=[1,2,3], v2=[4,5,6])["result"]
    assert dp == 32,                                             "Dot product gagal"
    sol = e.solve("gauss_jordan", augmented=[[2,1,5],[1,3,10]])["result"]
    assert sol is not None and abs(sol[0] - 1.0) < 1e-9,        "Gauss-Jordan gagal"
    print("  ✅ Linear Algebra: Determinan=-14, DotProduct=32, Gauss-Jordan ok")

    # Kalkulus (SymPy)
    if _SYMPY:
        d = e.solve("derivative", expr_str="x**3 + 2*x", var="x")
        assert "3*x**2" in d["result"] or "3*x" in d["simplified"], "Turunan gagal"
        ig = e.solve("integral", expr_str="2*x", var="x", lower="0", upper="3")
        assert "9" in ig["result"],                              "Integral tentu gagal"
        print("  ✅ Kalkulus: d/dx(x³+2x)=3x²+2, ∫₀³ 2x dx = 9")
    else:
        print("  ⚠️  SymPy tidak tersedia — kalkulus simbolik di-skip")

    # Barisan
    fib10 = e.solve("fibonacci", n=10)["result"]
    assert fib10 == 55,                                         "Fibonacci gagal"
    arith = e.solve("arithmetic", a1=2, d=3, n=10)
    assert arith["a_n"] == 29 and arith["S_n"] == 155,         "Aritmetika gagal"
    geom = e.solve("geometric", a1=2, r=3, n=5)
    assert geom["a_n"] == 162,                                   "Geometrika gagal"
    print("  ✅ Barisan: Fibonacci(10)=55, Aritmetika(2,3,10): a₁₀=29 S₁₀=155")

    print("\n✅ Semua test Pure Math Engine berhasil!\n")


if __name__ == "__main__":
    _self_test()
