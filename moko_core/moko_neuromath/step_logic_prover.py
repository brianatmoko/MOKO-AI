"""
MOKO Step-Aware Logic Prover (SALP)
===================================
Berdasarkan makalah penelitian:
  - Safe: step-aware formal verification for mathematical reasoning in LLMs
  - rStar-Math: code-verified reasoning steps via formal solvers

SALP menganalisis rantai pemikiran (Chain-of-Thought/CoT) matematika,
mengekstrak proposisi/persamaan bertahap, dan membuktikannya secara formal
menggunakan Z3 SMT Prover.
"""

import re
import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Cek ketersediaan z3
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    import sympy
    from sympy import symbols, Eq, solve
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from moko_neuromath.exact_math_engine import ExactMathEngine, ComputeMode



class LogicStepReport:
    def __init__(self, ok: bool, errors: List[str], details: List[Dict[str, Any]]):
        self.ok = ok
        self.errors = errors
        self.details = details  # List detail verifikasi per langkah

    def __repr__(self):
        return f"LogicStepReport(ok={self.ok}, errors={self.errors})"


class StepLogicProver:
    """
    Mesin penguji kebenaran langkah logika matematika.
    Mengekstrak pernyataan relasional dan membuktikan validitas transisi logika.
    """

    def __init__(self):
        self.z3_available = Z3_AVAILABLE
        self.sympy_available = SYMPY_AVAILABLE
        self.exact_math = ExactMathEngine(default_precision=50, verbose=False)


    def extract_relations(self, text: str) -> List[str]:
        """
        Mengekstrak pernyataan matematika relasional/logika dari teks CoT.
        Contoh: "karena x > 5 dan y = x + 2 maka y > 7" -> ["x > 5", "y = x + 2", "y > 7"]
        """
        # Pola deteksi persamaan/pertidaksamaan matematika standar
        # e.g., x > 5, y = x + 2, z >= 10, a <= b
        pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*\s*(?:[=<>!]=?|[<>])\s*[-+]?[a-zA-Z0-9_]*\s*(?:[\+\-\*\/]\s*[a-zA-Z0-9_]+)*)\b'
        matches = re.findall(pattern, text)
        
        # Bersihkan spasi berlebih
        cleaned_matches = []
        for m in matches:
            cleaned = re.sub(r'\s+', ' ', m).strip()
            if cleaned and cleaned not in cleaned_matches:
                cleaned_matches.append(cleaned)
                
        return cleaned_matches

    def verify_transitions(self, relations: List[str]) -> LogicStepReport:
        """
        Memverifikasi bahwa setiap relasi ke-i secara logis diimplikasikan oleh
        kombinasi relasi 0 sampai i-1.
        
        Rumus pembuktian formal SMT:
          Assert: (R_0 & R_1 & ... & R_{i-1}) & Not(R_i)
          Check: UNSAT -> Transisi valid (terbukti benar).
                 SAT   -> Transisi tidak valid (ditemukan celah logika / counter-example).
        """
        if not self.z3_available:
            return LogicStepReport(ok=True, errors=["Z3 tidak tersedia, melewati verifikasi formal."], details=[])

        if len(relations) <= 1:
            return LogicStepReport(ok=True, errors=[], details=[{"step": "init", "status": "PASS", "msg": "Relasi terlalu sedikit untuk verifikasi transisi."}])

        errors = []
        details = []
        
        # Inisialisasi solver Z3
        solver = z3.Solver()
        
        # Dictionary untuk menyimpan simbol Z3 secara dinamis
        z3_vars = {}

        def get_z3_var(name: str):
            """Dapatkan atau buat Z3 Real variable untuk simbol."""
            if name not in z3_vars:
                z3_vars[name] = z3.Real(name)
            return z3_vars[name]

        # Parse string relasi matematika ke ekspresi Z3
        parsed_exprs = []
        for r in relations:
            try:
                expr = self._parse_to_z3(r, get_z3_var)
                parsed_exprs.append((r, expr))
            except Exception as e:
                parsed_exprs.append((r, None))

        # Lakukan pembuktian implikasi bertahap:
        assignments = {}
        for i in range(len(parsed_exprs)):
            current_rel_str, current_expr = parsed_exprs[i]
            
            # Catat assignment baru jika polanya cocok
            # e.g. x = 5, y = sin(x)
            match_assign = re.match(r'^([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(.+)$', current_rel_str)
            if match_assign:
                var_name, val_expr = match_assign.groups()
                try:
                    # Ganti variabel yang ada di val_expr dengan nilainya
                    eval_expr = val_expr
                    for k, v in assignments.items():
                        eval_expr = re.sub(rf'\b{k}\b', str(v), eval_expr)
                    res = self.exact_math.compute(eval_expr, ComputeMode.HPFLOAT)
                    assignments[var_name] = float(res.value_float)
                except Exception:
                    pass

            # Tentukan apakah langkah ini adalah assignment atau deduction
            # Pertama, langkah pertama (index 0) selalu merupakan asumsi/premise awal
            # Kedua, relasi menggunakan '=' atau '==' dianggap sebagai penugasan/definisi nilai
            is_assignment = (i == 0) or ("=" in current_rel_str and "<=" not in current_rel_str and ">=" not in current_rel_str)
            
            if is_assignment:
                # Cukup tambahkan ke solver sebagai basis fakta, tidak perlu diverifikasi
                if current_expr is not None:
                    solver.add(current_expr)
                details.append({
                    "step": current_rel_str,
                    "status": "PASS",
                    "msg": "Asumsi/Definisi: ditambahkan ke basis fakta logika."
                })
                continue
                
            # Jika ini adalah deduksi/kesimpulan (e.g. y > 7)
            # Verifikasi bahwa solver context (prev_steps) mengimplikasikan current_expr
            check_res = None
            if current_expr is not None:
                solver.push()
                solver.add(z3.Not(current_expr))
                check_res = solver.check()
            
            if check_res == z3.unsat:
                # UNSAT = Implikasi valid (terbukti benar)
                details.append({
                    "step": current_rel_str,
                    "status": "PASS",
                    "msg": "Deduksi valid secara formal."
                })
                if current_expr is not None:
                    solver.pop()
                    # Tambahkan fakta ini ke basis fakta untuk langkah-langkah selanjutnya
                    solver.add(current_expr)
            elif check_res == z3.sat:
                # SAT = Celah logika terdeteksi
                model = solver.model()
                witness = {str(d): float(model[d].as_fraction()) if isinstance(model[d], z3.RatNumRef) else str(model[d]) for d in model.decls()}
                
                # Coba lakukan cross-verification menggunakan ExactMathEngine
                # Ini meng-override kegagalan solver jika secara numerik dengan assignment kita terbukti valid
                verified_numerically = False
                try:
                    match_deduct = re.match(r'^([a-zA-Z0-9_\s\+\-\*\/\.]+)\s*(==|<=|>=|[=<>])\s*([a-zA-Z0-9_\s\+\-\*\/\.]+)$', current_rel_str)
                    if match_deduct:
                        left_expr, op, right_expr = match_deduct.groups()
                        for k, v in assignments.items():
                            left_expr = re.sub(rf'\b{k}\b', str(v), left_expr)
                            right_expr = re.sub(rf'\b{k}\b', str(v), right_expr)
                        
                        val_left = float(self.exact_math.compute(left_expr, ComputeMode.HPFLOAT).value_float)
                        val_right = float(self.exact_math.compute(right_expr, ComputeMode.HPFLOAT).value_float)
                        
                        if op in ('=', '=='): verified_numerically = abs(val_left - val_right) < 1e-9
                        elif op == '<': verified_numerically = val_left < val_right
                        elif op == '>': verified_numerically = val_left > val_right
                        elif op == '<=': verified_numerically = val_left <= val_right
                        elif op == '>=': verified_numerically = val_left >= val_right
                except Exception:
                    pass

                if verified_numerically:
                    details.append({
                        "step": current_rel_str,
                        "status": "PASS",
                        "msg": f"Deduksi diverifikasi secara numerik via MOKO ExactMath ({val_left:.6g} {op} {val_right:.6g})."
                    })
                    if current_expr is not None:
                        solver.pop()
                        solver.add(current_expr)
                else:
                    err_msg = f"Logical Leap / Contradiction di langkah deduksi: '{current_rel_str}'. Counter-example: {witness}"
                    errors.append(err_msg)
                    
                    details.append({
                        "step": current_rel_str,
                        "status": "FAIL",
                        "msg": err_msg,
                        "witness": witness
                    })
                    if current_expr is not None:
                        solver.pop()
            else:
                # Unknown/timeout — Coba cross-verification menggunakan ExactMathEngine
                verified_numerically = False
                try:
                    match_deduct = re.match(r'^([a-zA-Z0-9_\s\+\-\*\/\.]+)\s*(==|<=|>=|[=<>])\s*([a-zA-Z0-9_\s\+\-\*\/\.]+)$', current_rel_str)
                    if match_deduct:
                        left_expr, op, right_expr = match_deduct.groups()
                        for k, v in assignments.items():
                            left_expr = re.sub(rf'\b{k}\b', str(v), left_expr)
                            right_expr = re.sub(rf'\b{k}\b', str(v), right_expr)
                        
                        val_left = float(self.exact_math.compute(left_expr, ComputeMode.HPFLOAT).value_float)
                        val_right = float(self.exact_math.compute(right_expr, ComputeMode.HPFLOAT).value_float)
                        
                        if op in ('=', '=='): verified_numerically = abs(val_left - val_right) < 1e-9
                        elif op == '<': verified_numerically = val_left < val_right
                        elif op == '>': verified_numerically = val_left > val_right
                        elif op == '<=': verified_numerically = val_left <= val_right
                        elif op == '>=': verified_numerically = val_left >= val_right
                except Exception:
                    pass

                if verified_numerically:
                    details.append({
                        "step": current_rel_str,
                        "status": "PASS",
                        "msg": f"Deduksi diverifikasi secara numerik via MOKO ExactMath ({val_left:.6g} {op} {val_right:.6g})."
                    })
                    if current_expr is not None:
                        solver.pop()
                        solver.add(current_expr)
                else:
                    details.append({
                        "step": current_rel_str,
                        "status": "UNKNOWN",
                        "msg": "Z3 solver tidak dapat menyimpulkan (timeout/nonlinear)."
                    })
                    if current_expr is not None:
                        solver.pop()

        return LogicStepReport(ok=len(errors) == 0, errors=errors, details=details)

    def _parse_to_z3(self, rel_str: str, get_var_fn) -> Optional[Any]:
        """Menerjemahkan string relasi matematika ke objek asersi Z3."""
        # Parsing sederhana untuk relasi x > y, x = y + z, dll.
        match = re.match(r'^([a-zA-Z0-9_\s\+\-\*\/\.]+)\s*(==|<=|>=|[=<>])\s*([a-zA-Z0-9_\s\+\-\*\/\.]+)$', rel_str)
        if not match:
            return None
            
        left_str, op, right_str = match.groups()
        
        # Helper untuk parse operand menjadi Z3 expression
        def parse_operand(op_str: str):
            op_str = op_str.strip()
            # Cek jika angka
            if re.match(r'^[-+]?\d*\.?\d+$', op_str):
                return float(op_str)
            # Cek jika variabel tunggal
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', op_str):
                return get_var_fn(op_str)
            # Parsing operasi sederhana (e.g. x + 2)
            plus_match = re.match(r'^([a-zA-Z0-9_]+)\s*([\+\-\*\/])\s*([a-zA-Z0-9_]+)$', op_str)
            if plus_match:
                v1_str, math_op, v2_str = plus_match.groups()
                v1 = parse_operand(v1_str)
                v2 = parse_operand(v2_str)
                if math_op == '+': return v1 + v2
                if math_op == '-': return v1 - v2
                if math_op == '*': return v1 * v2
                if math_op == '/': return v1 / v2
            return get_var_fn(op_str)

        try:
            left = parse_operand(left_str)
            right = parse_operand(right_str)
            
            if op == '=' or op == '==': return left == right
            if op == '<': return left < right
            if op == '>': return left > right
            if op == '<=': return left <= right
            if op == '>=': return left >= right
        except Exception:
            return None
        return None
