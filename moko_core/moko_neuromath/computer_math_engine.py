"""
MOKO Foundations of Computing Engine (FCE) — Early Computer Mathematics
========================================================================
Komponen matematika dasar MOKO OS. Menghubungkan matematika murni dengan
akar logika komputer menggunakan formal verification (Z3 SMT Solver).

Kemampuan:
  1. Boolean Equivalence: Membuktikan keabsahan persamaan aljabar boolean.
  2. Circuit Verification: Memverifikasi logika sirkuit digital (Half/Full Adder).
  3. Bit-Vector Arithmetic: Menyelesaikan bitwise logic dan pembuktian overflow.
  4. Turing Machine Simulator: Menjalankan dan memvalidasi transisi mesin Turing.
"""

import re
import time
from typing import Dict, Any, List, Tuple, Optional
from z3 import *

class ComputerMathEngine:
    """
    Engine matematika logika dan biner menggunakan Z3.
    Mendukung pembuktian formal formal untuk logika gerbang dan bit-vector.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔌 [FCE Engine] {msg}")

    # ── 1. BOOLEAN ALGEBRA EQUIVALENCE PROVER ──────────────────────────────

    def verify_boolean_equivalence(self, expr1_str: str, expr2_str: str) -> Dict[str, Any]:
        """
        Buktikan formal ekuivalensi dua ekspresi aljabar boolean.
        Contoh: "A & B | ~A & B" dan "B"
        Mendukung simbol: & | ^ ~ ! (dan kata: and or xor not)
        """
        t0 = time.time()

        # Normalize ke simbol operator standar Python/Z3
        expr1 = self._normalize_bool_expr(expr1_str)
        expr2 = self._normalize_bool_expr(expr2_str)

        # Temukan semua nama variabel kapital (A-Z, atau nama kata tunggal)
        vars_found = sorted(list(set(re.findall(r'\b[A-Z][a-z0-9_]*\b|\b[A-Z]+\b', expr1 + " " + expr2))))
        # Fallback: cari semua identifier bukan keyword
        if not vars_found:
            vars_found = sorted(list(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr1 + " " + expr2))))
            vars_found = [v for v in vars_found if v not in ("True", "False")]

        self._log(f"Membuktikan ekuivalensi: '{expr1}' == '{expr2}' | var: {vars_found}")

        # Buat Z3 Bool variables (mendukung & | ^ ~ langsung via operator overloading)
        z3_vars = {v: Bool(v) for v in vars_found}
        eval_ns = {"__builtins__": {}, **z3_vars}

        try:
            z3_expr1 = eval(expr1, eval_ns)
            z3_expr2 = eval(expr2, eval_ns)

            s = Solver()
            s.add(Not(z3_expr1 == z3_expr2))
            chk = s.check()
            latency = (time.time() - t0) * 1000

            if chk == unsat:
                return {
                    "success": True,
                    "equivalent": True,
                    "proof_type": "tautology",
                    "vars_checked": vars_found,
                    "counterexample": None,
                    "latency_ms": latency,
                }
            else:
                model = s.model()
                counterexample = {str(k): bool(model[k]) for k in model.decls()}
                return {
                    "success": True,
                    "equivalent": False,
                    "proof_type": "satisfiable_counterexample",
                    "vars_checked": vars_found,
                    "counterexample": counterexample,
                    "latency_ms": latency,
                }
        except Exception as e:
            return {
                "success": False,
                "equivalent": False,
                "error": str(e),
                "latency_ms": (time.time() - t0) * 1000,
            }

    # ── 2. LOGIC CIRCUIT VERIFIER (Full Adder) ──────────────────────────────

    def verify_full_adder(self, sum_expr: str, carry_expr: str) -> Dict[str, Any]:
        """
        Verifikasi formal sirkuit Full Adder.
        Spec resmi:
          Sum   = A ^ B ^ Cin
          Carry = (A & B) | (Cin & (A ^ B))
        """
        a, b, cin = Bools('A B Cin')
        eval_ns = {"__builtins__": {}, "A": a, "B": b, "Cin": cin}

        try:
            impl_sum   = eval(self._normalize_bool_expr(sum_expr),   eval_ns)
            impl_carry = eval(self._normalize_bool_expr(carry_expr), eval_ns)

            spec_sum   = a ^ b ^ cin
            spec_carry = (a & b) | (cin & (a ^ b))

            s = Solver()
            s.add(Or(Not(impl_sum == spec_sum), Not(impl_carry == spec_carry)))

            if s.check() == unsat:
                return {"success": True, "verified": True,
                        "msg": "✅ Sirkuit Full Adder terverifikasi 100% Benar secara formal."}
            else:
                model = s.model()
                cex = {str(k): bool(model[k]) for k in model.decls()}
                return {"success": True, "verified": False,
                        "msg": "❌ Sirkuit Full Adder memiliki bug!",
                        "counterexample": cex}
        except Exception as e:
            return {"success": False, "verified": False, "error": str(e)}

    # ── 3. BIT-VECTOR ARITHMETIC SOLVER ────────────────────────────────────

    def verify_bit_trick_power_of_two(self, num_bits: int = 8) -> Dict[str, Any]:
        """
        Buktikan formal trik biner: x & (x - 1) == 0 membuktikan
        bahwa x adalah bilangan pangkat dua (atau nol).
        """
        t0 = time.time()
        x = BitVec('x', num_bits)
        
        # Spec: x adalah pangkat dua -> popcount(x) == 1
        # Dalam SMT, kita membuktikan properti: jika (x & (x - 1)) == 0,
        # maka tidak ada bit lain yang aktif kecuali satu bit.
        
        s = Solver()
        # Buktikan untuk semua x > 0:
        # Pemicu: (x & (x - 1)) == 0 dan x != 0
        condition = And(x & (x - 1) == 0, x != 0)
        
        # Kita buktikan bahwa x hanya memiliki 1 bit bernilai 1.
        # Caranya: jika popcount != 1, cari counterexample
        # Untuk menyederhanakan tanpa menulis loop popcount di SMT:
        # Kita assert negasi: s.add(condition, Not(Or([x == (1 << i) for i in range(num_bits)])))
        power_of_twos = Or([x == (1 << i) for i in range(num_bits)])
        s.add(condition, Not(power_of_twos))

        chk = s.check()
        latency = (time.time() - t0) * 1000

        if chk == unsat:
            return {
                "success": True,
                "verified": True,
                "msg": f"Trik biner 'x & (x - 1) == 0' terbukti valid untuk {num_bits}-bit integer.",
                "latency_ms": latency
            }
        else:
            return {
                "success": True,
                "verified": False,
                "counterexample": str(s.model()),
                "latency_ms": latency
            }

    # ── 4. TURING MACHINE EMULATOR ──────────────────────────────────────────

    def run_turing_machine(
        self,
        tape: str,
        initial_state: str,
        rules: List[Dict[str, str]],  # [{'state': 'q0', 'read': '1', 'write': '0', 'dir': 'R', 'next': 'q1'}]
        max_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Simulasi mesin turing klasik dengan pita satu dimensi.
        """
        tape_list = list(tape) if tape else ['_']
        head_pos = 0
        current_state = initial_state
        
        steps = []
        step_count = 0
        halted = False

        # Bangun indeks aturan untuk pencarian cepat
        rule_map = {}
        for r in rules:
            rule_map[(r['state'], r['read'])] = r

        while step_count < max_steps:
            # Baca simbol saat ini
            if head_pos < 0:
                tape_list.insert(0, '_')
                head_pos = 0
            elif head_pos >= len(tape_list):
                tape_list.append('_')

            current_symbol = tape_list[head_pos]
            steps.append({
                "step": step_count,
                "state": current_state,
                "head": head_pos,
                "tape": "".join(tape_list),
            })

            # Cari aturan transisi
            rule = rule_map.get((current_state, current_symbol))
            if not rule:
                # Halt jika tidak ada transisi
                halted = True
                break

            # Tulis simbol baru
            tape_list[head_pos] = rule['write']
            
            # Gerakkan head
            if rule['dir'].upper() == 'R':
                head_pos += 1
            elif rule['dir'].upper() == 'L':
                head_pos -= 1

            # Transisi state
            current_state = rule['next']
            step_count += 1

            if current_state.lower() in ("halt", "q_halt", "reject", "accept"):
                halted = True
                steps.append({
                    "step": step_count,
                    "state": current_state,
                    "head": head_pos,
                    "tape": "".join(tape_list),
                })
                break

        return {
            "success": True,
            "final_tape": "".join(tape_list).strip('_'),
            "final_state": current_state,
            "steps_count": step_count,
            "halted": halted,
            "execution_trace": steps
        }

    # ── HELPERS ─────────────────────────────────────────────────────────────

    def _normalize_bool_expr(self, expr: str) -> str:
        """
        Normalisasi ekspresi boolean agar kompatibel dengan Z3 Bool operator overloading.
        Z3 Bool mendukung langsung: & (AND), | (OR), ^ (XOR), ~ (NOT).
        Hanya perlu mengganti kata kunci tekstual.
        """
        s = expr.strip()
        # Kata kunci tekstual -> operator simbol (harus sebelum simbol replacement)
        s = re.sub(r'\bnot\b',  '~',  s, flags=re.IGNORECASE)
        s = re.sub(r'\band\b',  '&',  s, flags=re.IGNORECASE)
        s = re.sub(r'\bor\b',   '|',  s, flags=re.IGNORECASE)
        s = re.sub(r'\bxor\b',  '^',  s, flags=re.IGNORECASE)
        # '!' -> '~'
        s = s.replace('!', '~')
        # Bereskan spasi ganda
        s = re.sub(r'\s+', ' ', s).strip()
        return s


# ── SINGLETON ──────────────────────────────────────────────────────────────
_fce_instance: Optional[ComputerMathEngine] = None

def get_fce_engine(verbose: bool = True) -> ComputerMathEngine:
    global _fce_instance
    if _fce_instance is None:
        _fce_instance = ComputerMathEngine(verbose=verbose)
    return _fce_instance


# ── SELF-TEST ──────────────────────────────────────────────────────────────
def _self_test():
    print("\n🧪 Foundations of Computing Engine (FCE) — Self Test\n")
    fce = ComputerMathEngine(verbose=True)

    # 1. Test Boolean Prover: (A & B) | (~A & B) == B
    print("  [Test 1] Uji ekuivalensi aljabar boolean...")
    res = fce.verify_boolean_equivalence(" (A & B) | (~A & B) ", " B ")
    assert res["success"] and res["equivalent"], "Test 1 Gagal!"
    print(f"  ✅ Sukses: Tautologi terbukti dalam {res['latency_ms']:.2f}ms")

    # 2. Test Boolean Prover (Salah): A | B == A & B
    print("\n  [Test 2] Uji ekuivalensi boolean (bernilai salah)...")
    res2 = fce.verify_boolean_equivalence("A | B", "A & B")
    assert res2["success"] and not res2["equivalent"], "Test 2 Gagal!"
    print(f"  ✅ Sukses: Counterexample ditemukan -> {res2['counterexample']}")

    # 3. Test Full Adder Circuit
    print("\n  [Test 3] Uji verifikasi sirkuit Full Adder...")
    res3 = fce.verify_full_adder(
        sum_expr="A ^ B ^ Cin",
        carry_expr="(A & B) | (Cin & (A ^ B))"
    )
    assert res3["success"] and res3["verified"], "Test 3 Gagal!"
    print(f"  ✅ Sukses: Sirkuit Full Adder terbukti 100% valid.")

    # 4. Test Turing Machine
    print("\n  [Test 4] Uji simulasi Mesin Turing (Inkrementasi Biner)...")
    # Aturan: increment biner (misal pita '101' -> '110')
    # q0: cari ujung kanan pita
    # q1: increment dengan carry
    rules = [
        {'state': 'q0', 'read': '0', 'write': '0', 'dir': 'R', 'next': 'q0'},
        {'state': 'q0', 'read': '1', 'write': '1', 'dir': 'R', 'next': 'q0'},
        {'state': 'q0', 'read': '_', 'write': '_', 'dir': 'L', 'next': 'q1'}, # Ujung kanan
        
        {'state': 'q1', 'read': '0', 'write': '1', 'dir': 'L', 'next': 'halt'}, # Tulis 1, selesai
        {'state': 'q1', 'read': '1', 'write': '0', 'dir': 'L', 'next': 'q1'},   # Tulis 0, carry ke kiri
        {'state': 'q1', 'read': '_', 'write': '1', 'dir': 'L', 'next': 'halt'}, # Tambah 1 di paling depan
    ]
    res4 = fce.run_turing_machine("1011", "q0", rules)
    assert res4["success"] and res4["final_tape"] == "1100", f"Test 4 Gagal! Hasil: {res4['final_tape']}"
    print(f"  ✅ Sukses: Mesin Turing menyelesaikan '1011' + 1 = '{res4['final_tape']}' dalam {res4['steps_count']} langkah.")

    print("\n✅ Semua test FCE Engine berhasil!\n")


if __name__ == "__main__":
    _self_test()
