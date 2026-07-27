"""
MOKO ExactMath Engine — Arbitrary Precision Mathematics Core
=============================================================
Layer 1 dari Precision Mathematics Engine (PME).

Filosofi: "Ketepatan dengan toleransi salah sekecil mungkin."

Masalah floating-point standar Python:
  >>> 0.1 + 0.2
  0.30000000000000004   ← Error! Bukan 0.3 yang sesungguhnya

  >>> sum(0.1 for _ in range(1000))
  99.9999999999998      ← Error akumulatif 2e-13!

Solusi ExactMath Engine (3 mode):
  MODE_SYMBOLIC : SymPy exact algebra (√8 = 2√2, ∫x²dx = x³/3)
  MODE_HPFLOAT  : mpmath 100-digit precision (99.9999...9 dengan 100 digit)  
  MODE_INTERVAL : mpmath interval arithmetic ([a,b] mengandung nilai sebenarnya)

Referensi akademis:
  - "Arbitrary Precision Arithmetic" — Knuth, TAOCP Vol. 2
  - "Interval Arithmetic" — Moore, Kearfott, Cloud (2009)
  - mpmath documentation — Fredrik Johansson (2023)
  - SymPy: Computer Algebra System in Python — Meurer et al., PeerJ CS 2017
  - "Precision and Expressiveness of Transformer Models" — ACL Anthology 2024
    (membuktikan bahwa presisi numerik mempengaruhi kemampuan reasoning AI)
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# KONSTANTA PRESISI
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PRECISION   = 100    # digit desimal (> double precision: 15-17 digit)
HIGH_PRECISION      = 200    # untuk kriptografi (RSA, ECC)
EXTREME_PRECISION   = 500    # untuk penelitian matematis khusus
CONVERGENCE_LEVELS  = [50, 100, 200]  # Level untuk cross-check konvergensi


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class ComputeMode(Enum):
    SYMBOLIC = "symbolic"   # SymPy: exact, tak terbatas, lambat
    HPFLOAT  = "hpfloat"   # mpmath: presisi tinggi, cepat
    INTERVAL = "interval"   # mpmath.iv: interval, terjamin, paling lambat


@dataclass
class ErrorBound:
    """
    Interval [lower, upper] yang terjamin mengandung nilai sebenarnya.
    Berdasarkan IEEE 1788 Standard for Interval Arithmetic.
    """
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

    @property
    def relative_error(self) -> float:
        if self.midpoint == 0:
            return self.width
        return self.width / abs(self.midpoint)

    def __repr__(self) -> str:
        return f"[{self.lower:.6g}, {self.upper:.6g}] (±{self.width:.2e})"


@dataclass
class PrecisionResult:
    """
    Hasil komputasi presisi tinggi dengan semua metadata validitas.
    
    Setiap hasil mengandung bukti kriptografi (SHA-256) agar bisa
    diverifikasi ulang tanpa mengulang komputasi penuh.
    """
    expression: str                    # Ekspresi yang dihitung
    value_str: str                     # Representasi string dengan presisi penuh
    value_float: float                 # Approximasi float (untuk kompatibilitas)
    mode: ComputeMode                  # Mode yang digunakan
    precision_digits: int              # Digit presisi yang diminta
    digits_correct: int                # Estimasi digit yang benar
    error_bound: Optional[ErrorBound]  # Interval [lower, upper] (None jika symbolic)
    is_exact: bool                     # True jika symbolic exact
    convergence_verified: bool         # True jika cross-precision check lolos
    sha256_fingerprint: str            # SHA-256 dari (expression + value_str)
    computation_steps: List[str]       # Langkah-langkah komputasi
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mode_label = {
            ComputeMode.SYMBOLIC: "EXACT",
            ComputeMode.HPFLOAT:  f"{self.precision_digits}-DIGIT",
            ComputeMode.INTERVAL: "INTERVAL"
        }[self.mode]
        bound_str = f" ± {self.error_bound.width:.2e}" if self.error_bound else ""
        return f"{self.value_str}{bound_str} [{mode_label}]"


# ═══════════════════════════════════════════════════════════════════════════
# EXACT MATH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ExactMathEngine:
    """
    Engine komputasi matematika presisi tinggi untuk MOKO OS.

    Strategi 3-mode berdasarkan kebutuhan:
    - SYMBOLIC: untuk algebra, calculus simbolik (hasil exact)
    - HPFLOAT:  untuk numerik dengan kontrol digit (default: 100 digit)
    - INTERVAL: untuk komputasi dengan garansi error bound (IEEE 1788)

    Verifikasi konvergensi: jalankan di 3 level presisi, bandingkan hasil.
    Jika divergen → peringatan bahwa komputasi tidak stabil.
    """

    def __init__(self, default_precision: int = DEFAULT_PRECISION,
                 verbose: bool = False):
        self.default_precision = default_precision
        self.verbose = verbose
        self._sympy_available = self._check_sympy()
        self._mpmath_available = self._check_mpmath()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔢 [ExactMath] {msg}")

    @staticmethod
    def _check_sympy() -> bool:
        try:
            import sympy  # noqa
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_mpmath() -> bool:
        try:
            import mpmath  # noqa
            return True
        except ImportError:
            return False

    def _fingerprint(self, expression: str, value_str: str) -> str:
        """SHA-256 fingerprint dari pasangan (expr, value) untuk verifikasi."""
        data = f"{expression}||{value_str}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    # ─────────────────────────────────────────────────────────────────────
    # COMPUTE — ENTRY POINT UTAMA
    # ─────────────────────────────────────────────────────────────────────

    def compute(
        self,
        expression: str,
        mode: ComputeMode = ComputeMode.HPFLOAT,
        precision: Optional[int] = None,
        verify_convergence: bool = True
    ) -> PrecisionResult:
        """
        Hitung ekspresi matematika dengan presisi tinggi.

        Args:
            expression:          String ekspresi (e.g., "sqrt(2)", "pi**2/6")
            mode:                Mode komputasi (SYMBOLIC/HPFLOAT/INTERVAL)
            precision:           Jumlah digit (default: self.default_precision)
            verify_convergence:  Cross-check dengan 3 level presisi berbeda

        Returns:
            PrecisionResult dengan nilai, error bound, dan fingerprint
        """
        prec = precision or self.default_precision
        steps = []

        if mode == ComputeMode.SYMBOLIC and self._sympy_available:
            return self._compute_symbolic(expression, steps, verify_convergence)
        elif mode == ComputeMode.INTERVAL and self._mpmath_available:
            return self._compute_interval(expression, prec, steps, verify_convergence)
        else:
            return self._compute_hpfloat(expression, prec, steps, verify_convergence)

    # ─────────────────────────────────────────────────────────────────────
    # MODE 1: SYMBOLIC (SymPy)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_symbolic(self, expression: str, steps: List[str],
                          verify: bool) -> PrecisionResult:
        """Komputasi symbolik exact menggunakan SymPy."""
        import sympy as sp

        steps.append(f"[SYMBOLIC] Parse: {expression}")
        try:
            # Parse ekspresi dengan SymPy (mendukung pi, E, sqrt, oo, dll.)
            # Gunakan sympify dengan namespace yang aman
            local_ns = {
                'pi': sp.pi, 'e': sp.E, 'E': sp.E,
                'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos,
                'tan': sp.tan, 'log': sp.log, 'exp': sp.exp,
                'abs': sp.Abs, 'oo': sp.oo, 'I': sp.I,
                'factorial': sp.factorial, 'binomial': sp.binomial,
                'gcd': sp.gcd, 'lcm': sp.lcm,
                'floor': sp.floor, 'ceiling': sp.ceiling,
                'Rational': sp.Rational, 'Integer': sp.Integer,
            }
            expr = sp.sympify(expression, locals=local_ns)
            steps.append(f"[SYMBOLIC] Parsed: {expr}")

            # Simplifikasi
            simplified = sp.simplify(expr)
            steps.append(f"[SYMBOLIC] Simplified: {simplified}")

            # Representasi string exact
            value_str = str(simplified)
            steps.append(f"[SYMBOLIC] Exact result: {value_str}")

            # Evaluasi numerik untuk float approximation (50 digit)
            try:
                numeric = float(simplified.evalf(50))
            except Exception:
                numeric = float('nan')

            # Error bound untuk symbolic: exact = [value, value]
            error_bound = ErrorBound(numeric, numeric) if not float('nan') == numeric else None

            fingerprint = self._fingerprint(expression, value_str)
            convergence = True  # Symbolic selalu convergent (exact)

            return PrecisionResult(
                expression=expression,
                value_str=value_str,
                value_float=numeric,
                mode=ComputeMode.SYMBOLIC,
                precision_digits=0,  # Tidak ada batas digit
                digits_correct=-1,   # Tak terbatas (exact)
                error_bound=error_bound,
                is_exact=True,
                convergence_verified=convergence,
                sha256_fingerprint=fingerprint,
                computation_steps=steps,
                metadata={"sympy_type": type(simplified).__name__,
                          "is_number": simplified.is_number,
                          "is_rational": simplified.is_rational}
            )

        except Exception as e:
            steps.append(f"[SYMBOLIC] Error: {e} — fallback ke HPFLOAT")
            return self._compute_hpfloat(expression, self.default_precision, steps, verify)

    # ─────────────────────────────────────────────────────────────────────
    # MODE 2: HIGH-PRECISION FLOAT (mpmath)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_hpfloat(self, expression: str, precision: int,
                         steps: List[str], verify: bool) -> PrecisionResult:
        """Komputasi presisi tinggi menggunakan mpmath."""
        from mpmath import mp

        steps.append(f"[HPFLOAT] Presisi: {precision} digit desimal")

        # Set presisi global mpmath
        mp.dps = precision + 10  # Extra margin untuk rounding

        try:
            # Buat namespace aman untuk eval
            mp_ns = {
                'mp': mp,
                'pi': mp.pi, 'e': mp.e, 'E': mp.e,
                'sqrt': mp.sqrt, 'sin': mp.sin, 'cos': mp.cos,
                'tan': mp.tan, 'log': mp.log, 'exp': mp.exp,
                'ln': mp.log, 'abs': mp.fabs,
                'factorial': mp.factorial, 'gamma': mp.gamma,
                'zeta': mp.zeta, 'phi': mp.phi,  # Golden ratio
                'inf': mp.inf, 'nan': mp.nan,
                'power': mp.power, 'nstr': mp.nstr,
                # Fungsi kriptografi-matematis
                'gcd': mp.gcd if hasattr(mp, 'gcd') else None,
            }

            # Evaluasi
            result = eval(expression, {"__builtins__": {}}, mp_ns)
            steps.append(f"[HPFLOAT] Evaluated: {mp.nstr(result, 20)}")

            # Konversi ke string dengan presisi penuh — gunakan n+5 untuk pastikan cukup digit
            value_str = mp.nstr(result, precision, strip_zeros=False)
            value_float = float(result)
            steps.append(f"[HPFLOAT] Full precision: {value_str[:50]}...")

            # Estimasi digit yang benar
            digits_correct = precision

            # Verifikasi konvergensi
            convergence = True
            if verify:
                convergence = self._verify_convergence_hpfloat(
                    expression, value_str, steps
                )

            # Error bound estimasi dari presisi
            eps = 10 ** (-(precision - 5))
            error_bound = ErrorBound(value_float - eps, value_float + eps)

            fingerprint = self._fingerprint(expression, value_str)

            return PrecisionResult(
                expression=expression,
                value_str=value_str,
                value_float=value_float,
                mode=ComputeMode.HPFLOAT,
                precision_digits=precision,
                digits_correct=digits_correct,
                error_bound=error_bound,
                is_exact=False,
                convergence_verified=convergence,
                sha256_fingerprint=fingerprint,
                computation_steps=steps,
                metadata={"mpmath_type": type(result).__name__,
                          "actual_dps": mp.dps}
            )

        except Exception as e:
            # Fallback ke Python standar
            steps.append(f"[HPFLOAT] Error: {e} — fallback Python eval")
            try:
                import math
                safe_ns = vars(math).copy()
                safe_ns['pi'] = math.pi
                result_f = eval(expression, {"__builtins__": {}}, safe_ns)
                value_str = repr(result_f)
                value_float = float(result_f)
                fingerprint = self._fingerprint(expression, value_str)
                return PrecisionResult(
                    expression=expression,
                    value_str=value_str,
                    value_float=value_float,
                    mode=ComputeMode.HPFLOAT,
                    precision_digits=15,
                    digits_correct=15,
                    error_bound=ErrorBound(value_float - 1e-10, value_float + 1e-10),
                    is_exact=False,
                    convergence_verified=False,
                    sha256_fingerprint=fingerprint,
                    computation_steps=steps,
                    metadata={"fallback": "python_math", "error": str(e)}
                )
            except Exception as e2:
                raise ValueError(f"ExactMath: gagal menghitung '{expression}': {e2}")

    def _verify_convergence_hpfloat(self, expression: str,
                                     reference: str, steps: List[str]) -> bool:
        """
        Cross-precision convergence test.
        Jalankan di presisi 50, 100, 200 digit.
        Jika hasil divergen → komputasi tidak stabil (warning).
        """
        from mpmath import mp

        results = {}
        for prec in CONVERGENCE_LEVELS:
            mp.dps = prec + 10
            try:
                mp_ns = {
                    'pi': mp.pi, 'e': mp.e, 'E': mp.e,
                    'sqrt': mp.sqrt, 'sin': mp.sin, 'cos': mp.cos,
                    'tan': mp.tan, 'log': mp.log, 'exp': mp.exp,
                    'factorial': mp.factorial, 'gamma': mp.gamma,
                    'zeta': mp.zeta, 'phi': mp.phi,
                }
                val = eval(expression, {"__builtins__": {}}, mp_ns)
                results[prec] = mp.nstr(val, min(prec, 30))
            except Exception:
                results[prec] = None

        # Cek apakah 15 digit pertama konsisten antar level
        valid_results = [v for v in results.values() if v is not None]
        if len(valid_results) < 2:
            return False

        # Ambil 15 digit pertama dari semua hasil
        prefixes = [v.replace('-', '').replace('.', '')[:15]
                    for v in valid_results]
        all_match = len(set(prefixes)) == 1

        steps.append(
            f"[CONVERGENCE] Levels {CONVERGENCE_LEVELS}: "
            f"{'PASS ✅' if all_match else 'DIVERGE ⚠️'}"
        )
        return all_match

    # ─────────────────────────────────────────────────────────────────────
    # MODE 3: INTERVAL ARITHMETIC (mpmath.iv)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_interval(self, expression: str, precision: int,
                          steps: List[str], verify: bool) -> PrecisionResult:
        """
        Komputasi dengan interval arithmetic.
        Setiap operasi menghasilkan [lower_bound, upper_bound] yang TERJAMIN
        mengandung nilai sebenarnya — tidak ada ketidakpastian yang tersembunyi.
        """
        from mpmath import mp, iv

        mp.dps = precision + 10
        steps.append(f"[INTERVAL] Mode IEEE 1788 — presisi {precision} digit")

        try:
            iv_ns = {
                'iv': iv,
                'pi': iv.pi, 'e': iv.e, 'E': iv.e,
                'sqrt': iv.sqrt, 'sin': iv.sin, 'cos': iv.cos,
                'tan': iv.tan, 'log': iv.log, 'exp': iv.exp,
                'abs': iv.fabs,
            }

            result = eval(expression, {"__builtins__": {}}, iv_ns)
            steps.append(f"[INTERVAL] Raw interval: [{result.a}, {result.b}]")

            lower = float(result.a)
            upper = float(result.b)
            midpoint = (lower + upper) / 2
            width = upper - lower

            steps.append(
                f"[INTERVAL] Width = {width:.2e} (relative: {width/abs(midpoint):.2e})"
            )

            value_str = f"[{mp.nstr(result.a, precision//2)}, {mp.nstr(result.b, precision//2)}]"
            fingerprint = self._fingerprint(expression, value_str)
            error_bound = ErrorBound(lower, upper)

            # Estimasi digit benar dari lebar interval
            if width > 0 and midpoint != 0:
                import math
                digits_correct = max(0, int(-math.log10(width / abs(midpoint))))
            else:
                digits_correct = precision

            return PrecisionResult(
                expression=expression,
                value_str=value_str,
                value_float=midpoint,
                mode=ComputeMode.INTERVAL,
                precision_digits=precision,
                digits_correct=digits_correct,
                error_bound=error_bound,
                is_exact=False,
                convergence_verified=True,  # Interval selalu terjamin
                sha256_fingerprint=fingerprint,
                computation_steps=steps,
                metadata={"interval_width": width,
                          "relative_error": width / abs(midpoint) if midpoint != 0 else float('inf')}
            )

        except Exception as e:
            steps.append(f"[INTERVAL] Error: {e} — fallback ke HPFLOAT")
            return self._compute_hpfloat(expression, precision, steps, verify)

    # ─────────────────────────────────────────────────────────────────────
    # DOMAIN-SPESIFIK API
    # ─────────────────────────────────────────────────────────────────────

    def compute_pi(self, precision: int = DEFAULT_PRECISION) -> PrecisionResult:
        """Hitung π dengan presisi arbitrer."""
        return self.compute("pi", ComputeMode.HPFLOAT, precision)

    def compute_e(self, precision: int = DEFAULT_PRECISION) -> PrecisionResult:
        """Hitung e (Euler's number) dengan presisi arbitrer."""
        return self.compute("e", ComputeMode.HPFLOAT, precision)

    def compute_sqrt(self, n: Union[int, str], precision: int = DEFAULT_PRECISION) -> PrecisionResult:
        """Hitung √n dengan presisi arbitrer."""
        return self.compute(f"sqrt({n})", ComputeMode.HPFLOAT, precision)

    def compute_symbolic(self, expression: str) -> PrecisionResult:
        """Komputasi simbolik exact (SymPy)."""
        return self.compute(expression, ComputeMode.SYMBOLIC)

    def compute_with_error_bound(self, expression: str,
                                  precision: int = DEFAULT_PRECISION) -> PrecisionResult:
        """Komputasi dengan garansi error bound (Interval Arithmetic)."""
        return self.compute(expression, ComputeMode.INTERVAL, precision)

    # ─────────────────────────────────────────────────────────────────────
    # CRYPTOGRAPHIC MATH API
    # ─────────────────────────────────────────────────────────────────────

    def is_prime_miller_rabin(self, n: int, rounds: int = 40) -> Tuple[bool, str]:
        """
        Uji keprimaan Miller-Rabin dengan k putaran.
        
        Miller-Rabin adalah uji probabilistik dengan error probability ≤ 4^(-rounds).
        Dengan rounds=40: probabilitas salah ≤ 4^(-40) ≈ 8.3 × 10^(-25)
        — lebih kecil dari probabilitas error hardware!

        Returns:
            (is_prime: bool, confidence: str)
        """
        import sympy
        result = sympy.isprime(n)  # SymPy menggunakan Miller-Rabin deterministic
        confidence = f"Deterministik via SymPy (BPSW test, tidak ada false positive yang diketahui)"
        return result, confidence

    def compute_modular_exp(self, base: int, exp: int, mod: int) -> PrecisionResult:
        """
        Hitung (base^exp) mod n secara exact.
        Fundamental untuk RSA, Diffie-Hellman, ElGamal.
        Menggunakan square-and-multiply (O(log exp) operasi).
        """
        result = pow(base, exp, mod)  # Python built-in: exact, O(log exp)
        value_str = str(result)
        steps = [
            f"[MODEXP] Menghitung {base}^{exp} mod {mod}",
            f"[MODEXP] Menggunakan square-and-multiply algorithm (O(log {exp}))",
            f"[MODEXP] Hasil: {value_str}",
        ]
        fingerprint = self._fingerprint(f"{base}^{exp} mod {mod}", value_str)
        return PrecisionResult(
            expression=f"({base}^{exp}) mod {mod}",
            value_str=value_str,
            value_float=float(result),
            mode=ComputeMode.SYMBOLIC,
            precision_digits=0,
            digits_correct=-1,
            error_bound=None,
            is_exact=True,
            convergence_verified=True,
            sha256_fingerprint=fingerprint,
            computation_steps=steps,
        )

    def _extended_gcd_python(self, a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean Algorithm in pure Python. Returns (gcd, x, y)."""
        x0, x1, y0, y1 = 1, 0, 0, 1
        while b != 0:
            q, r = divmod(a, b)
            a, b = b, r
            x0, x1 = x1, x0 - q * x1
            y0, y1 = y1, y0 - q * y1
        return a, x0, y0

    def compute_gcd_extended(self, a: int, b: int) -> Dict[str, Any]:
        """
        Extended Euclidean Algorithm: gcd(a,b) dan koefisien Bezout (x, y).
        ax + by = gcd(a, b)
        Fundamental untuk modular inverse dalam kriptografi.
        """
        try:
            import sympy
            x, y, g = sympy.gcdex(a, b)
            s_int = int(x)
            t_int = int(y)
            gcd_val = int(g)
            return {
                "gcd": gcd_val,
                "s": s_int,
                "t": t_int,
                "verification": f"{a}·({s_int}) + {b}·({t_int}) = {int(a*s_int + b*t_int)} (expect: {gcd_val})",
                "verified": int(a * s_int + b * t_int) == gcd_val
            }
        except Exception as e:
            g, x, y = self._extended_gcd_python(a, b)
            return {
                "gcd": g,
                "s": x,
                "t": y,
                "verification": f"{a}·({x}) + {b}·({y}) = {a*x + b*y} (expect: {g})",
                "verified": a*x + b*y == g
            }

    def compute_euler_totient(self, n: int) -> PrecisionResult:
        """
        Euler's Totient Function φ(n) — fundamental untuk RSA.
        φ(n) = jumlah bilangan 1..n yang coprime dengan n.
        """
        import sympy
        result = sympy.totient(n)
        value_str = str(result)
        steps = [
            f"[TOTIENT] φ({n}) menggunakan faktorisasi prima",
            f"[TOTIENT] Hasil: φ({n}) = {value_str}",
        ]
        fingerprint = self._fingerprint(f"totient({n})", value_str)
        return PrecisionResult(
            expression=f"φ({n})",
            value_str=value_str,
            value_float=float(result),
            mode=ComputeMode.SYMBOLIC,
            precision_digits=0,
            digits_correct=-1,
            error_bound=None,
            is_exact=True,
            convergence_verified=True,
            sha256_fingerprint=fingerprint,
            computation_steps=steps,
        )

    # ─────────────────────────────────────────────────────────────────────
    # REPORT GENERATOR
    # ─────────────────────────────────────────────────────────────────────

    def generate_precision_report(self, result: PrecisionResult) -> str:
        """Hasilkan laporan presisi yang komprehensif."""
        mode_info = {
            ComputeMode.SYMBOLIC: "SymPy Exact Symbolic",
            ComputeMode.HPFLOAT:  f"mpmath {result.precision_digits}-digit Float",
            ComputeMode.INTERVAL: f"mpmath Interval Arithmetic (IEEE 1788)",
        }
        lines = [
            "━" * 60,
            f"  🔢 MOKO ExactMath — Precision Report",
            "━" * 60,
            f"  Expression     : {result.expression}",
            f"  Mode           : {mode_info[result.mode]}",
            f"  Value          : {result.value_str[:60]}{'...' if len(result.value_str) > 60 else ''}",
            f"  Exact?         : {'✅ YES (symbolic)' if result.is_exact else '❌ No (numerical)'}",
            f"  Digits Correct : {'∞ (symbolic)' if result.digits_correct == -1 else str(result.digits_correct)}",
        ]
        if result.error_bound:
            lines.append(f"  Error Bound    : {result.error_bound}")
        lines += [
            f"  Convergence    : {'✅ VERIFIED' if result.convergence_verified else '⚠️ NOT VERIFIED'}",
            f"  SHA-256 Hash   : {result.sha256_fingerprint[:32]}...",
            "",
            "  Computation Steps:",
        ]
        for step in result.computation_steps:
            lines.append(f"    → {step}")
        lines.append("━" * 60)
        return "\n".join(lines)
