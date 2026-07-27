"""
MOKO MathCAS Engine — Symbolic Mathematics Core
================================================
Mesin komputasi simbolik deterministik berbasis SymPy.
LLM TIDAK menghitung — engine ini yang menghitung.

Arsitektur Neuro-Symbolic:
  LLM (Translator) → CAS Engine (Calculator) → Verifier → Response

Referensi:
  - AlphaGeometry 2 (DeepMind): neuro-symbolic hybrid
  - rStar-Math (Microsoft): code-verified reasoning steps
  - ILAC Framework: LLM + CAS integration
"""
import re
import math
import traceback
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

try:
    import sympy
    from sympy import (
        symbols, Symbol, Integer, Rational, Float, pi, E, I, oo,
        sin, cos, tan, log, ln, exp, sqrt, Abs, factorial,
        integrate, diff, limit, summation, product as sym_product,
        solve, simplify, expand, factor, cancel, apart,
        Matrix, det, Eq, Ne, Lt, Gt, Le, Ge,
        binomial, gamma, zeta, fibonacci,
        series, Derivative, Integral,
        latex, pretty, N as sympy_N,
        Sum, Product,
        gcd as sympy_gcd, lcm as sympy_lcm,
        isprime, nextprime, prevprime, prime, primerange,
        factorint, divisors,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application,
        convert_xor, implicit_multiplication,
        function_exponentiation
    )
    from sympy.stats import Normal, Exponential, P, E as stats_E
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class MathDomain(Enum):
    """Klasifikasi domain matematika."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    COMBINATORICS = "combinatorics"
    CRYPTOGRAPHY = "cryptography"
    UNKNOWN = "unknown"


class CASResult:
    """Hasil dari komputasi CAS."""
    __slots__ = (
        'success', 'symbolic_result', 'numeric_result',
        'latex_form', 'steps', 'domain', 'error',
        'confidence', 'execution_ms'
    )

    def __init__(self, success: bool = False, symbolic_result: str = "",
                 numeric_result: float = None, latex_form: str = "",
                 steps: list = None, domain: str = "unknown",
                 error: str = "", confidence: float = 0.0,
                 execution_ms: float = 0.0):
        self.success = success
        self.symbolic_result = symbolic_result
        self.numeric_result = numeric_result
        self.latex_form = latex_form
        self.steps = steps or []
        self.domain = domain
        self.error = error
        self.confidence = confidence
        self.execution_ms = execution_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "symbolic": self.symbolic_result,
            "numeric": self.numeric_result,
            "latex": self.latex_form,
            "steps": self.steps,
            "domain": self.domain,
            "error": self.error,
            "confidence": self.confidence,
            "exec_ms": self.execution_ms
        }

    def to_prompt_injection(self) -> str:
        """Generate text yang bisa diinjeksi ke LLM prompt sebagai ground truth."""
        if not self.success:
            return ""
        parts = [f"[CAS VERIFIED RESULT — GROUND TRUTH]"]
        parts.append(f"Domain: {self.domain}")
        parts.append(f"Symbolic: {self.symbolic_result}")
        if self.numeric_result is not None:
            parts.append(f"Numeric: {self.numeric_result}")
        if self.latex_form:
            parts.append(f"LaTeX: {self.latex_form}")
        if self.steps:
            parts.append("Steps:")
            for i, step in enumerate(self.steps, 1):
                parts.append(f"  {i}. {step}")
        parts.append(f"Confidence: {self.confidence:.2f}")
        parts.append("[END CAS RESULT — LLM HARUS MENGGUNAKAN ANGKA INI, JANGAN MENGHITUNG ULANG]")
        return "\n".join(parts)


class MathCASEngine:
    """
    Engine CAS utama. Menangani:
    1. Parsing ekspresi dari string
    2. Komputasi simbolik (integral, turunan, solve, simplify)
    3. Komputasi numerik (aritmetika, evaluasi)
    4. Verifikasi (CAS result vs LLM result)
    """

    # Common variable symbols
    _DEFAULT_SYMBOLS = {}

    # Pattern untuk mendeteksi operasi matematika
    _CALC_PATTERNS = [
        # Aljabar
        (r'(?:selesaikan|solve|cari\s+(?:nilai|x))\s+(.+)', MathDomain.ALGEBRA),
        (r'(?:faktorkan|factor)\s+(.+)', MathDomain.ALGEBRA),
        (r'(?:sederhanakan|simplify)\s+(.+)', MathDomain.ALGEBRA),
        (r'(?:expand|uraikan|jabarkan)\s+(.+)', MathDomain.ALGEBRA),

        # Kalkulus
        (r'(?:integral|∫|integralkan)\s+(.+)', MathDomain.CALCULUS),
        (r'(?:turunan|derivative|diferensial|d/dx)\s+(.+)', MathDomain.CALCULUS),
        (r'(?:limit|lim)\s+(.+)', MathDomain.CALCULUS),

        # Trigonometri
        (r'(?:sin|cos|tan|sec|csc|cot)\s*\(?\s*(\d+)', MathDomain.TRIGONOMETRY),

        # Statistik
        (r'(?:rata-rata|mean|average|median|modus|mode|std|standar\s*deviasi)\s+(.+)', MathDomain.STATISTICS),
        (r'(?:probabilitas|probability|peluang)\s+(.+)', MathDomain.STATISTICS),

        # Teori Bilangan
        (r'(?:prima|prime)\s+(.+)', MathDomain.NUMBER_THEORY),
        (r'(?:fpb|gcd|faktor\s+persekutuan)\s+(.+)', MathDomain.NUMBER_THEORY),
        (r'(?:kpk|lcm|kelipatan\s+persekutuan)\s+(.+)', MathDomain.NUMBER_THEORY),

        # Aljabar Linear
        (r'(?:determinan|det|matriks|matrix)\s+(.+)', MathDomain.LINEAR_ALGEBRA),

        # Kombinatorika
        (r'(?:kombinasi|combination|C)\s*\(\s*(\d+)\s*,\s*(\d+)', MathDomain.COMBINATORICS),
        (r'(?:permutasi|permutation|P)\s*\(\s*(\d+)\s*,\s*(\d+)', MathDomain.COMBINATORICS),
        (r'(\d+)\s*(?:faktorial|!)', MathDomain.COMBINATORICS),

        # Kriptografi Matematika
        (r'rsa\s+(?:keypair|kunci)', MathDomain.CRYPTOGRAPHY),
        (r'rsa\s+(?:encrypt|enkripsi|decrypt|dekripsi)', MathDomain.CRYPTOGRAPHY),
        (r'diffie[\s-]*hellman', MathDomain.CRYPTOGRAPHY),
        (r'(?:totient|phi)\s*\(?\s*\d+', MathDomain.CRYPTOGRAPHY),
        (r'modular\s+inverse', MathDomain.CRYPTOGRAPHY),
        (r'extended\s+gcd', MathDomain.CRYPTOGRAPHY),
        (r'birthday\s+(?:attack|problem)', MathDomain.CRYPTOGRAPHY),

        # Aritmetika dasar
        (r'(?:akar|sqrt|√)\s*(?:kuadrat\s+(?:dari\s+)?)?\(?(\d+)\)?', MathDomain.ARITHMETIC),
        (r'(\d+)\s*(?:pangkat|power|\^)\s*(\d+)', MathDomain.ARITHMETIC),
        (r'(\d+)\s*(?:mod|modulo|%)\s*(\d+)', MathDomain.NUMBER_THEORY),
        (r'(\d+[\.\d]*)\s*[\+\-\*\/\^]\s*(\d+[\.\d]*)', MathDomain.ARITHMETIC),
    ]

    # Map bahasa Indonesia → SymPy function
    _ID_FUNC_MAP = {
        "sin": "sin", "cos": "cos", "tan": "tan",
        "akar": "sqrt", "akar kuadrat": "sqrt",
        "pangkat": "**", "kuadrat": "**2", "kubik": "**3",
        "logaritma": "log", "log": "log", "ln": "ln",
        "absolut": "Abs", "mutlak": "Abs",
        "faktorial": "factorial",
        "pi": "pi", "euler": "E",
    }

    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for MathCAS Engine. Install with: pip install sympy")
        # Initialize default symbols
        self._DEFAULT_SYMBOLS = {
            name: symbols(name) for name in 'x y z a b c n m k t r s p q'.split()
        }
        self._transformations = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor, function_exponentiation)
        )

    def classify_domain(self, query: str) -> MathDomain:
        """Klasifikasi domain matematika dari query."""
        q = query.lower().strip()
        for pattern, domain in self._CALC_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return domain
        # Heuristic fallback
        if any(w in q for w in ['integral', 'turunan', 'limit', 'deret', 'series']):
            return MathDomain.CALCULUS
        if any(w in q for w in ['persamaan', 'solve', 'selesaikan', 'cari x', 'cari nilai']):
            return MathDomain.ALGEBRA
        if any(w in q for w in ['matriks', 'matrix', 'determinan', 'eigenvalue']):
            return MathDomain.LINEAR_ALGEBRA
        if any(w in q for w in ['rata-rata', 'mean', 'median', 'probabilitas', 'peluang']):
            return MathDomain.STATISTICS
        if any(w in q for w in ['prima', 'prime', 'fpb', 'gcd', 'kpk', 'lcm', 'faktor']):
            return MathDomain.NUMBER_THEORY
        if any(w in q for w in ['rsa', 'diffie', 'hellman', 'totient', 'modular inverse',
                                 'extended gcd', 'birthday attack', 'kunci publik',
                                 'private key', 'keypair', 'enkripsi asimetris']):
            return MathDomain.CRYPTOGRAPHY
        # Check for pure arithmetic expressions
        if re.match(r'^[\d\s\+\-\*\/\^\(\)\.\,]+$', q.strip()):
            return MathDomain.ARITHMETIC
        return MathDomain.UNKNOWN

    def can_compute(self, query: str) -> bool:
        """Apakah query ini bisa dikomputasi secara deterministik oleh CAS?"""
        domain = self.classify_domain(query)
        return domain != MathDomain.UNKNOWN

    def compute(self, query: str) -> CASResult:
        """
        Main entry point. Parse query → compute → return CASResult.
        Dispatches to domain-specific handlers.
        """
        import time
        t0 = time.time()

        try:
            domain = self.classify_domain(query)
            q = query.lower().strip()

            if domain == MathDomain.ARITHMETIC:
                result = self._compute_arithmetic(q)
            elif domain == MathDomain.ALGEBRA:
                result = self._compute_algebra(q)
            elif domain == MathDomain.CALCULUS:
                result = self._compute_calculus(q)
            elif domain == MathDomain.TRIGONOMETRY:
                result = self._compute_trigonometry(q)
            elif domain == MathDomain.STATISTICS:
                result = self._compute_statistics(q)
            elif domain == MathDomain.NUMBER_THEORY:
                result = self._compute_number_theory(q)
            elif domain == MathDomain.LINEAR_ALGEBRA:
                result = self._compute_linear_algebra(q)
            elif domain == MathDomain.COMBINATORICS:
                result = self._compute_combinatorics(q)
            elif domain == MathDomain.CRYPTOGRAPHY:
                result = self._compute_cryptography(q)
            else:
                # Fallback: try to parse as raw expression
                result = self._compute_raw_expression(q)

            result.domain = domain.value
            result.execution_ms = (time.time() - t0) * 1000.0
            return result

        except Exception as e:
            return CASResult(
                success=False,
                error=f"CAS Engine Error: {type(e).__name__}: {str(e)}",
                execution_ms=(time.time() - t0) * 1000.0
            )

    def verify_llm_answer(self, cas_result: CASResult, llm_answer: str) -> Tuple[bool, float, str]:
        """
        Verifikasi jawaban LLM terhadap hasil CAS.
        Returns: (is_correct, confidence, explanation)
        """
        if not cas_result.success:
            return True, 0.5, "CAS tidak bisa memverifikasi (no ground truth)"

        # Extract numbers from LLM answer
        llm_numbers = re.findall(r'[-+]?\d*\.?\d+(?:/\d+)?', llm_answer)

        if cas_result.numeric_result is not None:
            cas_val = float(cas_result.numeric_result)
            for num_str in llm_numbers:
                try:
                    if '/' in num_str:
                        parts = num_str.split('/')
                        llm_val = float(parts[0]) / float(parts[1])
                    else:
                        llm_val = float(num_str)
                    # Tolerance check
                    if abs(cas_val) < 1e-10:
                        if abs(llm_val) < 1e-6:
                            return True, 1.0, "Verified: both zero"
                    elif abs(cas_val - llm_val) / max(abs(cas_val), 1e-10) < 0.001:
                        return True, 1.0, f"Verified: CAS={cas_val}, LLM={llm_val}"
                except (ValueError, ZeroDivisionError):
                    continue

            return False, 0.0, f"MISMATCH: CAS={cas_val}, LLM numbers found={llm_numbers}"

        # Symbolic comparison
        cas_sym = cas_result.symbolic_result.replace(" ", "")
        llm_clean = llm_answer.replace(" ", "")
        if cas_sym in llm_clean:
            return True, 0.9, "Symbolic match found in LLM output"

        return True, 0.5, "Unable to definitively verify"

    # ─── Domain-Specific Compute Methods ───

    def _compute_arithmetic(self, query: str) -> CASResult:
        """Komputasi aritmetika: +, -, *, /, ^, sqrt, factorial, mod."""
        steps = []

        # Handle Indonesian math terms
        expr_str = self._translate_id_to_sympy(query)

        # Try to extract and evaluate a math expression
        # First, try to find a pure math expression in the query
        expr_match = re.search(
            r'(?:hitung|berapa|hasil(?:\s+dari)?|calculate|compute|=)?\s*([\d\s\+\-\*\/\^\(\)\.\,\%]+)',
            expr_str, re.IGNORECASE
        )

        if expr_match:
            raw_expr = expr_match.group(1).strip()
        else:
            raw_expr = expr_str

        # Clean up expression
        raw_expr = raw_expr.replace('^', '**').replace(',', '.').replace('%', ' % ')
        raw_expr = re.sub(r'\s+', ' ', raw_expr).strip()

        # Handle sqrt
        sqrt_match = re.search(r'(?:sqrt|akar)\s*\(?(\d+(?:\.\d+)?)\)?', query.lower())
        if sqrt_match:
            val = float(sqrt_match.group(1))
            sym_result = sqrt(sympy.Integer(int(val)) if val == int(val) else sympy.Float(val))
            num_result = float(sympy_N(sym_result))
            steps.append(f"√{int(val) if val == int(val) else val} = {sym_result}")
            if sym_result != num_result:
                steps.append(f"≈ {num_result}")
            return CASResult(
                success=True,
                symbolic_result=str(sym_result),
                numeric_result=num_result,
                latex_form=latex(sym_result),
                steps=steps,
                confidence=1.0
            )

        # Handle factorial
        fact_match = re.search(r'(\d+)\s*(?:faktorial|!)', query.lower())
        if fact_match:
            n = int(fact_match.group(1))
            result = factorial(n)
            steps.append(f"{n}! = {result}")
            return CASResult(
                success=True,
                symbolic_result=str(result),
                numeric_result=float(result),
                latex_form=f"{n}!",
                steps=steps,
                confidence=1.0
            )

        # General arithmetic parse
        try:
            parsed = parse_expr(raw_expr, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            simplified = simplify(parsed)
            steps.append(f"Ekspresi: {raw_expr}")
            steps.append(f"Hasil: {simplified}")

            try:
                numeric = float(sympy_N(simplified))
            except (TypeError, ValueError):
                numeric = None

            return CASResult(
                success=True,
                symbolic_result=str(simplified),
                numeric_result=numeric,
                latex_form=latex(simplified),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Parse error: {e}")

    def _compute_algebra(self, query: str) -> CASResult:
        """Aljabar: solve, factor, simplify, expand."""
        steps = []
        q = query.lower()
        x, y, z = symbols('x y z')

        # Detect operation
        if any(w in q for w in ['selesaikan', 'solve', 'cari nilai', 'cari x']):
            return self._solve_equation(query, steps)
        elif any(w in q for w in ['faktorkan', 'factor']):
            return self._factor_expression(query, steps)
        elif any(w in q for w in ['sederhanakan', 'simplify']):
            return self._simplify_expression(query, steps)
        elif any(w in q for w in ['expand', 'uraikan', 'jabarkan']):
            return self._expand_expression(query, steps)
        else:
            return self._solve_equation(query, steps)

    def _solve_equation(self, query: str, steps: list) -> CASResult:
        """Solve equations."""
        x, y, z = symbols('x y z')
        expr_str = self._extract_math_expression(query)
        steps.append(f"Persamaan: {expr_str}")

        try:
            # Handle "= 0" format
            if '=' in expr_str:
                parts = expr_str.split('=')
                lhs = parse_expr(parts[0].strip(), local_dict=self._DEFAULT_SYMBOLS,
                                 transformations=self._transformations)
                rhs = parse_expr(parts[1].strip(), local_dict=self._DEFAULT_SYMBOLS,
                                 transformations=self._transformations)
                eq = Eq(lhs, rhs)
                solutions = solve(eq, x)
            else:
                parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                    transformations=self._transformations)
                solutions = solve(parsed, x)

            steps.append(f"Solusi: x = {solutions}")
            sol_str = str(solutions)

            numeric = None
            if solutions and len(solutions) == 1:
                try:
                    numeric = float(sympy_N(solutions[0]))
                except (TypeError, ValueError):
                    pass

            return CASResult(
                success=True,
                symbolic_result=sol_str,
                numeric_result=numeric,
                latex_form=latex(solutions),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Solve error: {e}", steps=steps)

    def _factor_expression(self, query: str, steps: list) -> CASResult:
        """Factor expressions."""
        expr_str = self._extract_math_expression(query)
        steps.append(f"Ekspresi: {expr_str}")
        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            factored = factor(parsed)
            steps.append(f"Faktorisasi: {factored}")
            return CASResult(
                success=True,
                symbolic_result=str(factored),
                latex_form=latex(factored),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Factor error: {e}", steps=steps)

    def _simplify_expression(self, query: str, steps: list) -> CASResult:
        """Simplify expressions."""
        expr_str = self._extract_math_expression(query)
        steps.append(f"Ekspresi: {expr_str}")
        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            simplified = simplify(parsed)
            steps.append(f"Disederhanakan: {simplified}")
            return CASResult(
                success=True,
                symbolic_result=str(simplified),
                latex_form=latex(simplified),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Simplify error: {e}", steps=steps)

    def _expand_expression(self, query: str, steps: list) -> CASResult:
        """Expand expressions."""
        expr_str = self._extract_math_expression(query)
        steps.append(f"Ekspresi: {expr_str}")
        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            expanded = expand(parsed)
            steps.append(f"Dijabarkan: {expanded}")
            return CASResult(
                success=True,
                symbolic_result=str(expanded),
                latex_form=latex(expanded),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Expand error: {e}", steps=steps)

    def _compute_calculus(self, query: str) -> CASResult:
        """Kalkulus: integral, turunan, limit, deret."""
        steps = []
        q = query.lower()
        x = symbols('x')

        if any(w in q for w in ['integral', '∫', 'integralkan']):
            return self._compute_integral(query, steps)
        elif any(w in q for w in ['turunan', 'derivative', 'diferensial', 'd/dx']):
            return self._compute_derivative(query, steps)
        elif any(w in q for w in ['limit', 'lim']):
            return self._compute_limit(query, steps)
        elif any(w in q for w in ['deret', 'series', 'taylor']):
            return self._compute_series(query, steps)
        else:
            return CASResult(success=False, error="Unrecognized calculus operation")

    def _compute_integral(self, query: str, steps: list) -> CASResult:
        """Compute integrals (definite and indefinite)."""
        x = symbols('x')
        expr_str = self._extract_math_expression(query)
        steps.append(f"Integran: {expr_str}")

        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)

            # Check for definite integral bounds
            bounds = re.findall(
                r'(?:dari|from)\s*([-\d\.]+)\s*(?:ke|sampai|to|hingga)\s*([-\d\.]+)',
                query.lower()
            )

            if bounds:
                a, b = float(bounds[0][0]), float(bounds[0][1])
                steps.append(f"Batas: [{a}, {b}]")
                result = integrate(parsed, (x, a, b))
                steps.append(f"∫[{a},{b}] {expr_str} dx = {result}")
                try:
                    numeric = float(sympy_N(result))
                except (TypeError, ValueError):
                    numeric = None
            else:
                result = integrate(parsed, x)
                steps.append(f"∫ {expr_str} dx = {result} + C")
                numeric = None

            return CASResult(
                success=True,
                symbolic_result=str(result),
                numeric_result=numeric,
                latex_form=latex(result),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Integral error: {e}", steps=steps)

    def _compute_derivative(self, query: str, steps: list) -> CASResult:
        """Compute derivatives."""
        x = symbols('x')
        expr_str = self._extract_math_expression(query)
        steps.append(f"Fungsi: {expr_str}")

        # Check for nth derivative
        order_match = re.search(r'(?:ke|order|orde)\s*(\d+)', query.lower())
        order = int(order_match.group(1)) if order_match else 1

        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            result = diff(parsed, x, order)
            steps.append(f"d{'ⁿ' if order > 1 else ''}/dx{'ⁿ' if order > 1 else ''} ({expr_str}) = {result}")
            try:
                numeric = float(sympy_N(result))
            except (TypeError, ValueError):
                numeric = None
            return CASResult(
                success=True,
                symbolic_result=str(result),
                numeric_result=numeric,
                latex_form=latex(result),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Derivative error: {e}", steps=steps)

    def _compute_limit(self, query: str, steps: list) -> CASResult:
        """Compute limits."""
        x = symbols('x')
        expr_str = self._extract_math_expression(query)

        # Extract point
        point_match = re.search(
            r'(?:x|untuk)\s*(?:→|->|mendekati|menuju)\s*([-\d\.]+|inf|infinity|∞|tak\s*hingga)',
            query.lower()
        )
        if point_match:
            p = point_match.group(1)
            if p in ('inf', 'infinity', '∞', 'tak hingga'):
                point = oo
            else:
                point = float(p)
        else:
            point = 0

        steps.append(f"lim(x→{point}) {expr_str}")

        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            result = limit(parsed, x, point)
            steps.append(f"= {result}")
            try:
                numeric = float(sympy_N(result))
            except (TypeError, ValueError):
                numeric = None
            return CASResult(
                success=True,
                symbolic_result=str(result),
                numeric_result=numeric,
                latex_form=latex(result),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Limit error: {e}", steps=steps)

    def _compute_series(self, query: str, steps: list) -> CASResult:
        """Compute Taylor/Maclaurin series."""
        x = symbols('x')
        expr_str = self._extract_math_expression(query)

        # Extract order
        order_match = re.search(r'(?:orde|order|hingga|sampai)\s*(\d+)', query.lower())
        n_terms = int(order_match.group(1)) if order_match else 5

        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            result = series(parsed, x, 0, n_terms)
            steps.append(f"Deret Taylor {expr_str} di x=0 hingga orde {n_terms}:")
            steps.append(f"= {result}")
            return CASResult(
                success=True,
                symbolic_result=str(result),
                latex_form=latex(result),
                steps=steps,
                confidence=1.0
            )
        except Exception as e:
            return CASResult(success=False, error=f"Series error: {e}", steps=steps)

    def _compute_trigonometry(self, query: str) -> CASResult:
        """Komputasi trigonometri."""
        steps = []
        # Extract function and angle
        trig_match = re.search(
            r'(sin|cos|tan|sec|csc|cot)\s*\(?(\d+(?:\.\d+)?)\s*(derajat|degree|°|radian|rad)?\)?',
            query.lower()
        )
        if not trig_match:
            return CASResult(success=False, error="Could not parse trigonometric expression")

        func_name = trig_match.group(1)
        angle_val = float(trig_match.group(2))
        unit = trig_match.group(3) or "derajat"

        # Convert to radians if in degrees
        if unit in ('derajat', 'degree', '°'):
            angle_rad = sympy.rad(angle_val)
            steps.append(f"{angle_val}° = {angle_rad} radian")
        else:
            angle_rad = sympy.Float(angle_val)

        func_map = {'sin': sin, 'cos': cos, 'tan': tan}
        if func_name in func_map:
            result = func_map[func_name](angle_rad)
            simplified = simplify(result)
            numeric = float(sympy_N(simplified))
            steps.append(f"{func_name}({angle_val}{'°' if unit in ('derajat', 'degree', '°') else ''}) = {simplified}")
            steps.append(f"≈ {numeric}")
            return CASResult(
                success=True,
                symbolic_result=str(simplified),
                numeric_result=numeric,
                latex_form=latex(simplified),
                steps=steps,
                confidence=1.0
            )

        return CASResult(success=False, error=f"Unsupported trig function: {func_name}")

    def _compute_statistics(self, query: str) -> CASResult:
        """Statistik: mean, median, mode, std dev."""
        steps = []
        q = query.lower()

        # Extract numbers
        numbers = [float(n) for n in re.findall(r'[-+]?\d+(?:\.\d+)?', q)]
        if len(numbers) < 2:
            return CASResult(success=False, error="Need at least 2 numbers for statistics")

        steps.append(f"Data: {numbers}")

        if any(w in q for w in ['rata-rata', 'mean', 'average', 'rerata']):
            result = sum(numbers) / len(numbers)
            steps.append(f"Mean = Σ/n = {sum(numbers)}/{len(numbers)} = {result}")
            return CASResult(success=True, symbolic_result=str(result),
                             numeric_result=result, steps=steps, confidence=1.0)

        elif any(w in q for w in ['median']):
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            if n % 2 == 0:
                result = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
            else:
                result = sorted_nums[n // 2]
            steps.append(f"Sorted: {sorted_nums}")
            steps.append(f"Median = {result}")
            return CASResult(success=True, symbolic_result=str(result),
                             numeric_result=result, steps=steps, confidence=1.0)

        elif any(w in q for w in ['standar deviasi', 'std', 'standard deviation', 'simpangan baku']):
            mean_val = sum(numbers) / len(numbers)
            variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
            result = math.sqrt(variance)
            steps.append(f"Mean = {mean_val}")
            steps.append(f"Variance = {variance}")
            steps.append(f"Std Dev = √{variance} = {result}")
            return CASResult(success=True, symbolic_result=str(result),
                             numeric_result=result, steps=steps, confidence=1.0)

        elif any(w in q for w in ['varians', 'variance']):
            mean_val = sum(numbers) / len(numbers)
            result = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
            steps.append(f"Mean = {mean_val}")
            steps.append(f"Variance = {result}")
            return CASResult(success=True, symbolic_result=str(result),
                             numeric_result=result, steps=steps, confidence=1.0)

        return CASResult(success=False, error="Unrecognized statistics operation")

    def _compute_number_theory(self, query: str) -> CASResult:
        """Teori bilangan: prima, gcd, lcm, faktorisasi."""
        steps = []
        q = query.lower()
        numbers = [int(n) for n in re.findall(r'\d+', q)]

        if not numbers:
            return CASResult(success=False, error="No numbers found in query")

        if any(w in q for w in ['prima', 'prime']):
            if any(w in q for w in ['apakah', 'is', 'cek', 'check']):
                n = numbers[0]
                is_p = isprime(n)
                steps.append(f"Apakah {n} prima? {'Ya' if is_p else 'Tidak'}")
                if not is_p:
                    steps.append(f"Faktorisasi: {factorint(n)}")
                return CASResult(
                    success=True, symbolic_result=str(is_p),
                    steps=steps, confidence=1.0
                )
            else:
                n = numbers[0]
                primes = list(primerange(2, n + 1))
                steps.append(f"Bilangan prima ≤ {n}: {primes}")
                return CASResult(
                    success=True, symbolic_result=str(primes),
                    steps=steps, confidence=1.0
                )

        elif any(w in q for w in ['fpb', 'gcd', 'faktor persekutuan']):
            if len(numbers) >= 2:
                from math import gcd
                from functools import reduce
                result = reduce(gcd, numbers)
                steps.append(f"FPB({', '.join(map(str, numbers))}) = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )

        elif any(w in q for w in ['kpk', 'lcm', 'kelipatan persekutuan']):
            if len(numbers) >= 2:
                from math import gcd, lcm
                from functools import reduce
                result = reduce(lcm, numbers)
                steps.append(f"KPK({', '.join(map(str, numbers))}) = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )

        elif any(w in q for w in ['faktorisasi', 'faktor prima', 'factorize']):
            n = numbers[0]
            factors = factorint(n)
            steps.append(f"Faktorisasi prima {n}: {factors}")
            factor_str = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items())
            steps.append(f"{n} = {factor_str}")
            return CASResult(
                success=True, symbolic_result=factor_str,
                steps=steps, confidence=1.0
            )

        elif any(w in q for w in ['pembagi', 'divisor']):
            n = numbers[0]
            divs = divisors(n)
            steps.append(f"Pembagi dari {n}: {divs}")
            return CASResult(
                success=True, symbolic_result=str(divs),
                steps=steps, confidence=1.0
            )

        return CASResult(success=False, error="Unrecognized number theory operation")

    def _compute_linear_algebra(self, query: str) -> CASResult:
        """Aljabar linear: determinan, invers, eigenvalue."""
        steps = []
        q = query.lower()

        # Try to extract matrix from query
        # Format: [[1,2],[3,4]] or {1,2;3,4}
        mat_match = re.search(r'\[\[(.*?)\]\]', query)
        if not mat_match:
            mat_match = re.search(r'\{(.*?)\}', query)

        if not mat_match:
            return CASResult(success=False, error="Could not parse matrix from query")

        try:
            mat_str = mat_match.group(0)
            # Parse matrix
            mat_str = mat_str.replace('{', '[[').replace('}', ']]').replace(';', '],[')
            mat = Matrix(eval(mat_str))
            steps.append(f"Matriks: {mat}")

            if 'determinan' in q or 'det' in q:
                result = det(mat)
                steps.append(f"det(A) = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )
            elif 'invers' in q or 'inverse' in q:
                result = mat.inv()
                steps.append(f"A⁻¹ = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    latex_form=latex(result), steps=steps, confidence=1.0
                )
            elif 'eigenvalue' in q or 'eigen' in q:
                eigenvals = mat.eigenvals()
                steps.append(f"Eigenvalues: {eigenvals}")
                return CASResult(
                    success=True, symbolic_result=str(eigenvals),
                    steps=steps, confidence=1.0
                )
            else:
                # Default: show det
                result = det(mat)
                steps.append(f"det(A) = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )
        except Exception as e:
            return CASResult(success=False, error=f"Linear algebra error: {e}", steps=steps)

    def _compute_combinatorics(self, query: str) -> CASResult:
        """Kombinatorika: C(n,r), P(n,r), factorial."""
        steps = []
        q = query.lower()
        numbers = [int(n) for n in re.findall(r'\d+', q)]

        if any(w in q for w in ['kombinasi', 'combination', 'c(']):
            if len(numbers) >= 2:
                n, r = numbers[0], numbers[1]
                result = int(binomial(n, r))
                steps.append(f"C({n},{r}) = {n}! / ({r}! × ({n}-{r})!) = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )
        elif any(w in q for w in ['permutasi', 'permutation', 'p(']):
            if len(numbers) >= 2:
                n, r = numbers[0], numbers[1]
                result = int(factorial(n) / factorial(n - r))
                steps.append(f"P({n},{r}) = {n}! / ({n}-{r})! = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )
        elif any(w in q for w in ['faktorial', '!']):
            if numbers:
                n = numbers[0]
                result = int(factorial(n))
                steps.append(f"{n}! = {result}")
                return CASResult(
                    success=True, symbolic_result=str(result),
                    numeric_result=float(result), steps=steps, confidence=1.0
                )

        return CASResult(success=False, error="Unrecognized combinatorics operation")

    def _compute_cryptography(self, query: str) -> CASResult:
        """
        Komputasi kriptografi matematika secara deterministik:
        RSA keypair, RSA encrypt/decrypt, Diffie-Hellman, Euler Totient,
        Modular Inverse, Extended GCD, Birthday Attack probability.
        100% akurat — ZERO LLM hallucination.
        """
        import hashlib
        steps: list = []
        q = query.lower()

        def _is_prime_simple(n: int) -> bool:
            if n < 2: return False
            if n == 2: return True
            if n % 2 == 0: return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0: return False
            return True

        def _ext_gcd(a: int, b: int):
            if a == 0: return b, 0, 1
            g, x, y = _ext_gcd(b % a, a)
            return g, y - (b // a) * x, x

        # ── 1. RSA Keypair ──────────────────────────────────────────────────
        if 'keypair' in q or ('kunci' in q and 'rsa' in q):
            p_m = re.search(r'\bp\s*=\s*(\d+)', q)
            qq_m = re.search(r'\bq\s*=\s*(\d+)', q)
            e_m = re.search(r'\be\s*=\s*(\d+)', q)
            p = int(p_m.group(1)) if p_m else 61
            qv = int(qq_m.group(1)) if qq_m else 53
            e = int(e_m.group(1)) if e_m else 65537
            if not _is_prime_simple(p):
                return CASResult(success=False, error=f"{p} bukan bilangan prima")
            if not _is_prime_simple(qv):
                return CASResult(success=False, error=f"{qv} bukan bilangan prima")
            if p == qv:
                return CASResult(success=False, error="p dan q harus berbeda")
            n = p * qv
            phi_n = (p - 1) * (qv - 1)
            g, _, _ = _ext_gcd(e, phi_n)
            if g != 1:
                return CASResult(success=False, error=f"e={e} tidak relatif prima dengan φ(n)={phi_n}")
            _, x, _ = _ext_gcd(e, phi_n)
            d = x % phi_n
            steps += [
                f"p = {p} (prima ✓),  q = {qv} (prima ✓)",
                f"n  = p × q = {p} × {qv} = {n}",
                f"φ(n) = (p-1)(q-1) = {p-1} × {qv-1} = {phi_n}",
                f"e  = {e}  → gcd(e, φ(n)) = 1 ✓",
                f"d  = e⁻¹ mod φ(n) = {d}  [Extended Euclidean]",
                f"Public Key  = (e={e}, n={n})",
                f"Private Key = (d={d}, n={n})",
            ]
            sym = (
                f"RSA({p},{qv}): n={n}, φ(n)={phi_n}, "
                f"PublicKey=({e},{n}), PrivateKey=({d},{n})"
            )
            return CASResult(
                success=True,
                symbolic_result=sym,
                latex_form=(
                    f"n={n},\\; \\phi(n)={phi_n},\\;"
                    f"\\; e={e},\\; d={d}"
                ),
                steps=steps, confidence=1.0
            )

        # ── 2. RSA Encrypt ──────────────────────────────────────────────────
        if 'encrypt' in q or 'enkripsi' in q:
            m_m = re.search(r'\bm\s*=\s*(\d+)', q)
            e_m = re.search(r'\be\s*=\s*(\d+)', q)
            n_m = re.search(r'\bn\s*=\s*(\d+)', q)
            if m_m and e_m and n_m:
                m, e, n = int(m_m.group(1)), int(e_m.group(1)), int(n_m.group(1))
                c = pow(m, e, n)
                steps += [
                    f"RSA Enkripsi: C = M^e mod n",
                    f"C = {m}^{e} mod {n}",
                    f"C = {c}",
                ]
                return CASResult(
                    success=True,
                    symbolic_result=str(c),
                    numeric_result=float(c),
                    latex_form=f"C = {m}^{{{e}}} \\pmod{{{n}}} = {c}",
                    steps=steps, confidence=1.0
                )

        # ── 3. RSA Decrypt ──────────────────────────────────────────────────
        if 'decrypt' in q or 'dekripsi' in q:
            c_m = re.search(r'\bc\s*=\s*(\d+)', q)
            d_m = re.search(r'\bd\s*=\s*(\d+)', q)
            n_m = re.search(r'\bn\s*=\s*(\d+)', q)
            if c_m and d_m and n_m:
                c, d, n = int(c_m.group(1)), int(d_m.group(1)), int(n_m.group(1))
                m = pow(c, d, n)
                steps += [
                    f"RSA Dekripsi: M = C^d mod n",
                    f"M = {c}^{d} mod {n}",
                    f"M = {m} (plaintext asli)",
                ]
                return CASResult(
                    success=True,
                    symbolic_result=str(m),
                    numeric_result=float(m),
                    latex_form=f"M = {c}^{{{d}}} \\pmod{{{n}}} = {m}",
                    steps=steps, confidence=1.0
                )

        # ── 4. Diffie-Hellman ───────────────────────────────────────────────
        if 'diffie' in q or ('dh' in q and ('key' in q or 'kunci' in q or 'p=' in q)):
            p_m = re.search(r'\bp\s*=\s*(\d+)', q)
            g_m = re.search(r'\bg\s*=\s*(\d+)', q)
            a_m = re.search(r'\ba\s*=\s*(\d+)', q)
            b_m = re.search(r'\bb\s*=\s*(\d+)', q)
            if p_m and g_m and a_m and b_m:
                p  = int(p_m.group(1))
                g  = int(g_m.group(1))
                a  = int(a_m.group(1))
                b  = int(b_m.group(1))
                A  = pow(g, a, p)
                B  = pow(g, b, p)
                s1 = pow(B, a, p)   # shared from Alice
                s2 = pow(A, b, p)   # shared from Bob
                assert s1 == s2, "Shared secret mismatch — bukan DH valid"
                steps += [
                    f"Parameter: p={p} (modulus), g={g} (generator)",
                    f"Alice private key: a={a}",
                    f"Bob   private key: b={b}",
                    f"Alice public key: A = g^a mod p = {g}^{a} mod {p} = {A}",
                    f"Bob   public key: B = g^b mod p = {g}^{b} mod {p} = {B}",
                    f"Shared Secret (Alice): B^a mod p = {B}^{a} mod {p} = {s1}",
                    f"Shared Secret (Bob):   A^b mod p = {A}^{b} mod {p} = {s2}",
                    f"Verifikasi: s1 == s2 → {'✓ IDENTIK' if s1==s2 else '✗ GAGAL'}",
                ]
                return CASResult(
                    success=True,
                    symbolic_result=(
                        f"DH(p={p},g={g}): A={A}, B={B}, "
                        f"SharedSecret={s1}"
                    ),
                    numeric_result=float(s1),
                    latex_form=(
                        f"A={A},\\; B={B},\\;"
                        f"\\; S={s1}"
                    ),
                    steps=steps, confidence=1.0
                )

        # ── 5. Euler's Totient ──────────────────────────────────────────────
        tot_m = re.search(r'(?:totient|phi|\u03c6)\s*\(?\s*(\d+)\)?', q)
        if tot_m:
            n = int(tot_m.group(1))
            try:
                import sympy as _sp
                result_val = int(_sp.totient(n))
            except Exception:
                # Fallback pure-python
                result_val = n
                tmp = n
                p = 2
                while p * p <= tmp:
                    if tmp % p == 0:
                        while tmp % p == 0:
                            tmp //= p
                        result_val -= result_val // p
                    p += 1
                if tmp > 1:
                    result_val -= result_val // tmp
            steps += [
                f"Euler's Totient φ({n})",
                f"φ(n) = jumlah bilangan 1 ≤ k ≤ n yang gcd(k,n)=1",
                f"φ({n}) = {result_val}",
            ]
            return CASResult(
                success=True,
                symbolic_result=str(result_val),
                numeric_result=float(result_val),
                latex_form=f"\\phi({n}) = {result_val}",
                steps=steps, confidence=1.0
            )

        # ── 6. Modular Inverse ──────────────────────────────────────────────
        modinv_m = re.search(
            r'(?:modular\s+inverse|invers\s+modular)\s+(?:of\s+)?(\d+)\s+(?:mod|modulo)\s+(\d+)', q
        )
        if modinv_m:
            a, m = int(modinv_m.group(1)), int(modinv_m.group(2))
            g, x, _ = _ext_gcd(a, m)
            if g != 1:
                return CASResult(success=False, error=f"Invers tidak ada: gcd({a},{m})={g} ≠ 1")
            inv = x % m
            steps += [
                f"Modular Inverse: {a}⁻¹ mod {m}",
                f"Extended GCD({a}, {m}): gcd={g}, x={x}",
                f"{a}⁻¹ mod {m} = {inv}",
                f"Verifikasi: {a} × {inv} mod {m} = {(a * inv) % m} (harus = 1)",
            ]
            return CASResult(
                success=True,
                symbolic_result=str(inv),
                numeric_result=float(inv),
                latex_form=f"{a}^{{-1}} \\pmod{{{m}}} = {inv}",
                steps=steps, confidence=1.0
            )

        # ── 7. Extended GCD ─────────────────────────────────────────────────
        egcd_m = re.search(
            r'extended\s+gcd\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', q
        )
        if egcd_m:
            a, b = int(egcd_m.group(1)), int(egcd_m.group(2))
            g, x, y = _ext_gcd(a, b)
            steps += [
                f"Extended GCD({a}, {b})",
                f"gcd  = {g}",
                f"x    = {x}  (sehingga {a}×{x} + {b}×{y} = {g})",
                f"y    = {y}",
                f"Verifikasi: {a}×{x} + {b}×{y} = {a*x + b*y} (harus = {g})",
            ]
            return CASResult(
                success=True,
                symbolic_result=f"gcd={g}, x={x}, y={y}",
                latex_form=f"\\gcd({a},{b})={g},\\; {a}x+{b}y={g}",
                steps=steps, confidence=1.0
            )

        # ── 8. Birthday Attack Probability ──────────────────────────────────
        bday_m = re.search(
            r'birthday.*?(\d+)\s*bit.*?(\d+)\s*(?:attempts|percobaan)', q
        )
        if bday_m:
            import math as _math
            bits, attempts = int(bday_m.group(1)), int(bday_m.group(2))
            space = 2 ** bits
            prob = 1 - _math.exp(-(attempts**2) / (2 * space))
            steps += [
                f"Birthday Attack: hash {bits}-bit, {attempts} percobaan",
                f"Ruang hash: 2^{bits} = {space:.2e}",
                f"P ≈ 1 - e^(-n²/2H)  dimana H={space:.2e}, n={attempts}",
                f"P ≈ {prob:.6e}",
            ]
            return CASResult(
                success=True,
                symbolic_result=f"{prob:.8f}",
                numeric_result=prob,
                latex_form=f"P \\approx {prob:.6e}",
                steps=steps, confidence=1.0
            )

        return CASResult(
            success=False,
            error="Parameter kriptografi tidak lengkap atau operasi tidak dikenali. "
                  "Contoh: 'rsa keypair p=61 q=53', 'diffie hellman p=23 g=5 a=6 b=15', "
                  "'totient 100', 'modular inverse of 3 mod 7'"
        )

    def _compute_raw_expression(self, query: str) -> CASResult:
        """Fallback: try to parse and evaluate any raw expression."""
        expr_str = self._translate_id_to_sympy(query)
        # Remove non-math text
        expr_str = re.sub(r'[a-zA-Z]{3,}', '', expr_str)
        expr_str = expr_str.replace('^', '**').strip()

        if not expr_str:
            return CASResult(success=False, error="No evaluable expression found")

        try:
            parsed = parse_expr(expr_str, local_dict=self._DEFAULT_SYMBOLS,
                                transformations=self._transformations)
            result = simplify(parsed)
            try:
                numeric = float(sympy_N(result))
            except (TypeError, ValueError):
                numeric = None
            return CASResult(
                success=True, symbolic_result=str(result),
                numeric_result=numeric,
                latex_form=latex(result),
                steps=[f"{expr_str} = {result}"],
                confidence=0.8
            )
        except Exception as e:
            return CASResult(success=False, error=f"Raw parse failed: {e}")

    # ─── Helper Methods ───

    def _translate_id_to_sympy(self, text: str) -> str:
        """Translate Indonesian math terms to SymPy-parseable format."""
        result = text.lower()
        id_replacements = {
            r'\bakar kuadrat dari\b': 'sqrt',
            r'\bakar kuadrat\b': 'sqrt',
            r'\bakar dari\b': 'sqrt',
            r'\bakar\b': 'sqrt',
            r'\bpangkat\b': '**',
            r'\bkali\b': '*',
            r'\bbagi\b': '/',
            r'\btambah\b': '+',
            r'\bkurang\b': '-',
            r'\bper\b': '/',
            r'\bdikali\b': '*',
            r'\bdibagi\b': '/',
            r'\bditambah\b': '+',
            r'\bdikurangi\b': '-',
        }
        for pattern, replacement in id_replacements.items():
            result = re.sub(pattern, replacement, result)
        return result

    def _extract_math_expression(self, query: str) -> str:
        """Extract the mathematical expression from a natural language query."""
        q = query.lower()
        # Remove operation keywords and common descriptive nouns
        remove_patterns = [
            r'(?:hitung|berapa|hasil|calculate|compute)\s*(?:dari\s+)?',
            r'(?:selesaikan|solve|cari\s+(?:nilai|x))\s+',
            r'(?:integral|∫|integralkan)\s+(?:dari\s+)?',
            r'(?:turunan|derivative|diferensial|d/dx)\s+(?:dari\s+)?',
            r'(?:limit|lim)\s+(?:dari\s+)?',
            r'(?:faktorkan|factor)\s+',
            r'(?:sederhanakan|simplify)\s+',
            r'(?:expand|uraikan|jabarkan)\s+',
            r'(?:dari|from)\s+\d+\s+(?:ke|sampai|to)\s+\d+',
            r'(?:terhadap|respect\s+to)\s+\w+',
            r'(?:untuk|for)\s+',
            r'(?:x|untuk)\s*(?:→|->|mendekati|menuju)\s*(?:[-\d\.]+|inf|infinity|∞|tak\s*hingga)',
            r'\b(?:persamaan|fungsi|ekspresi|soal|nilai|hasil|dari|tentukan)\b\s*',
        ]
        result = q
        for pat in remove_patterns:
            result = re.sub(pat, '', result, flags=re.IGNORECASE)

        # Translate Indonesian
        result = self._translate_id_to_sympy(result)
        # Replace ^ with **
        result = result.replace('^', '**')
        result = result.strip()
        return result if result else q



# ─── Global Singleton ───
math_cas = MathCASEngine() if SYMPY_AVAILABLE else None
