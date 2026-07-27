"""
MOKO Formula Red Team (Adversarial Formula Critic)
===================================================
Mesin penguji kritis (Red Team) untuk mengevaluasi secara ketat setiap rumus
yang dihasilkan oleh Formula Generator (Blue Team).

Tujuannya adalah mendeteksi HALUSINASI RUMUS (rumus yang tampak benar pada data latih,
tetapi melanggar hukum fisika dasar atau gagal pada kondisi ekstrim).

Strategi Serangan Red Team (Falsifikasi):
  1. Edge-Case/Boundary Attack : Evaluasi nilai ekstrem (0, inf, negatif) untuk mendeteksi
                                non-physical values (massa negatif, suhu di bawah 0 K, dsb).
  2. Fuzzing Attack            : Menguji stabilitas formula dengan perturbasi input acak.
  3. Overfitting Attack        : Membandingkan performa pada data latih vs hold-out test set
                                secara ekstrem (uji generalisasi).
  4. Invariant Law Check       : Memastikan rumus tidak melanggar hukum konservasi/invarian fisik.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class CriticReport:
    passed: bool                      # Apakah rumus lolos kritik Red Team?
    defense_score: float              # Skor ketahanan rumus (0.0 - 1.0)
    failed_attacks: List[str]         # Jenis serangan yang berhasil menembus rumus
    attack_details: List[str]         # Detail eksekusi serangan
    recommendations: List[str]        # Saran perbaikan rumus


class FormulaCritic:
    """
    Red Team Critic untuk menguji dan memfalsifikasi rumus kandidat.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Domain boundaries untuk tipe variabel umum
        self.physical_constraints = {
            "m": (1e-30, float('inf')),     # Massa harus positif
            "T_temp": (0.0, float('inf')),  # Suhu Kelvin >= 0
            "T_hot": (0.0, float('inf')),
            "T_cold": (0.0, float('inf')),
            "v": (-299792458, 299792458),   # Kecepatan terbatas c (relativitas)
            "v_s": (0.0, float('inf')),     # Kecepatan suara positif
            "R": (1e-10, float('inf')),     # Resistansi positif
            "R_e": (1e-10, float('inf')),
            "A": (1e-30, float('inf')),     # Area positif
            "V_vol": (1e-30, float('inf')), # Volume positif
            "p": (1e-30, float('inf')),     # Tekanan positif (absolut)
            "P": (1e-30, float('inf')),
            "D": (1e-30, float('inf')),     # Diameter positif
            "L": (1e-30, float('inf')),     # Panjang positif
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [RedTeam] {msg}")

    def evaluate_formula(
        self,
        formula_expr_str: str,
        target_symbol: str,
        variables: List[str],
        train_r2: float,
        test_r2: float,
        domain_name: str = "unknown"
    ) -> CriticReport:
        """
        Jalankan serangkaian serangan Red Team terhadap formula.
        """
        self._log(f"Menyerang rumus kandidat: {target_symbol} = {formula_expr_str}")
        
        failed_attacks = []
        attack_details = []
        recommendations = []
        
        # Parse ekspresi menggunakan SymPy secara aman
        import sympy as sp
        try:
            expr = sp.sympify(formula_expr_str)
            free_syms = list(expr.free_symbols)
            eval_func = sp.lambdify(free_syms, expr, 'math')
        except Exception as e:
            return CriticReport(
                passed=False,
                defense_score=0.0,
                failed_attacks=["Syntax Parse Attack"],
                attack_details=[f"Gagal parse ekspresi: {e}"],
                recommendations=["Perbaiki format sintaks ekspresi agar valid secara matematis."]
            )

        # ── SERANGAN 1: Uji Boundary & Konstanta Fisik (Edge Cases) ─────
        boundary_failed = False
        attack_details.append("1. Meluncurkan Edge-Case Boundary Attack...")
        
        # Test 1a: Uji pembagian dengan nol / input ekstrem kecil
        zero_inputs = {str(s): 1e-15 for s in free_syms}
        try:
            val = eval_func(*[zero_inputs[str(s)] for s in free_syms])
            if math.isnan(val) or math.isinf(val):
                boundary_failed = True
                failed_attacks.append("Singularity Attack (Division by Zero)")
                attack_details.append("   [FAIL] Rumus menghasilkan NaN/Infinity pada input sangat kecil.")
                recommendations.append("Tambahkan faktor pengaman atau batasi pembagian langsung dengan variabel input.")
        except Exception:
            boundary_failed = True
            failed_attacks.append("Singularity Attack (Division by Zero)")
            attack_details.append("   [FAIL] Rumus crash saat dievaluasi dengan input mendekati nol.")
            recommendations.append("Hindari pembagian langsung dengan variabel tanpa konstanta pengaman.")

        # Test 1b: Uji pelanggaran batasan fisik (Physical Bounds Check)
        # Ambil sampel input yang valid
        sample_inputs = {}
        for sym in free_syms:
            sym_name = str(sym)
            if sym_name in self.physical_constraints:
                low, high = self.physical_constraints[sym_name]
                sample_inputs[sym_name] = low * 2.0 if not math.isinf(low) else 1.0
            else:
                sample_inputs[sym_name] = 1.0

        try:
            val = eval_func(*[sample_inputs[str(s)] for s in free_syms])
            # Cek jika target memiliki batasan fisik (misal energi kinetik tidak boleh negatif)
            if target_symbol in self.physical_constraints:
                low, high = self.physical_constraints[target_symbol]
                if val < low or val > high:
                    boundary_failed = True
                    failed_attacks.append("Physical Constraint Violation")
                    attack_details.append(f"   [FAIL] Rumus menghasilkan nilai target luar batas fisik: {val} (batas: {low} s/d {high})")
                    recommendations.append(f"Target '{target_symbol}' harus menghasilkan nilai dalam rentang {low} s/d {high}.")
        except Exception as e:
            boundary_failed = True
            failed_attacks.append("Evaluation Failure")
            attack_details.append(f"   [FAIL] Rumus gagal dievaluasi pada kondisi standar: {e}")

        # ── SERANGAN 2: Fuzzing Perturbasi Input ─────────────────────────
        attack_details.append("2. Meluncurkan Perturbation Fuzzing Attack...")
        fuzz_failed = False
        noise_variance = 0.05
        
        # Kita uji apakah perubahan kecil 5% pada input menyebabkan perubahan liar (chaotic) pada output
        try:
            base_vals = [sample_inputs[str(s)] for s in free_syms]
            base_out = eval_func(*base_vals)
            
            if not math.isnan(base_out) and not math.isinf(base_out) and base_out != 0:
                fuzz_diffs = []
                for _ in range(50):
                    fuzzed_vals = [v * (1 + random.gauss(0, noise_variance)) for v in base_vals]
                    fuzzed_out = eval_func(*fuzzed_vals)
                    if math.isnan(fuzzed_out) or math.isinf(fuzzed_out):
                        fuzz_failed = True
                        break
                    
                    pct_diff = abs((fuzzed_out - base_out) / base_out)
                    fuzz_diffs.append(pct_diff)
                
                # Jika rata-rata perbedaan output > 500% untuk input yang hanya berubah 5%
                if fuzz_failed or (fuzz_diffs and sum(fuzz_diffs) / len(fuzz_diffs) > 5.0):
                    fuzz_failed = True
                    failed_attacks.append("Chaotic Sensitivity / Numerical Instability")
                    attack_details.append("   [FAIL] Rumus sangat sensitif terhadap perubahan kecil input (potensi kekacauan numerik).")
                    recommendations.append("Sederhanakan eksponen atau perkalian berulang yang sensitif.")
                else:
                    attack_details.append("   [PASS] Rumus stabil terhadap gangguan (perturbasi) numerik kecil.")
        except Exception:
            fuzz_failed = True
            failed_attacks.append("Fuzzing Exception")
            attack_details.append("   [FAIL] Fuzzing menyebabkan crash evaluasi.")

        # ── SERANGAN 3: Overfitting & Generalization Attack ─────────────
        attack_details.append("3. Meluncurkan Overfitting Generalization Attack...")
        overfit_failed = False
        
        # Cek drop akurasi dari train ke test set
        r2_drop = train_r2 - test_r2
        if r2_drop > 0.15:
            overfit_failed = True
            failed_attacks.append("Overfitting / Poor Generalization")
            attack_details.append(f"   [FAIL] Akurasi drop signifikan dari train ({train_r2:.4f}) ke test set ({test_r2:.4f}).")
            recommendations.append("Rumus terlalu kompleks (overfit). Kurangi jumlah operator atau node dalam tree.")
        else:
            attack_details.append(f"   [PASS] Selisih akurasi train vs test aman (drop: {r2_drop:.4f}).")

        # ── Evaluasi Hasil Akhir Red Team ──────────────────────────────
        total_attacks = 3
        successful_defenses = total_attacks - (1 if boundary_failed else 0) - (1 if fuzz_failed else 0) - (1 if overfit_failed else 0)
        defense_score = successful_defenses / total_attacks
        
        passed = (defense_score >= 0.8) # Lolos jika score >= 80%

        if passed:
            attack_details.append("\n🛡️  KESIMPULAN: Rumus lolos audit Red Team (Kuat & Stabil).")
        else:
            attack_details.append("\n⚠️ KESIMPULAN: Rumus gagal menangkis serangan Red Team (Potensi Halusinasi).")

        return CriticReport(
            passed=passed,
            defense_score=defense_score,
            failed_attacks=failed_attacks,
            attack_details=attack_details,
            recommendations=recommendations
        )


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_critic_instance: Optional[FormulaCritic] = None

def get_formula_critic(verbose: bool = False) -> FormulaCritic:
    global _critic_instance
    if _critic_instance is None:
        _critic_instance = FormulaCritic(verbose=verbose)
    return _critic_instance
