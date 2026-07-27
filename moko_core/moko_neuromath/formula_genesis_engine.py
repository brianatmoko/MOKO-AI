"""
MOKO Formula Genesis Engine (FGE) - Orchestrator
================================================
Mengintegrasikan semua komponen penciptaan & penemuan rumus (FGE):
  1. Symbolic Regression Engine (dari data ke rumus numerik)
  2. Dimensional Synthesis Engine (dari dimensi variabel ke rumus)
  3. Pattern Conjecturer (dari deret angka ke formula)
  4. Formula Validator (memverifikasi keabsahan rumus)

Menyimpan rumus yang berhasil ditemukan secara persisten ke:
  .math_omni/discovered_formulas.json
Dan mendaftarkannya secara dinamis ke AppliedFormulaEngine.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Absolute paths
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MATH_OMNI_DIR = os.path.join(_BASE_DIR, ".math_omni")
_DISCOVERED_FILE = os.path.join(_MATH_OMNI_DIR, "discovered_formulas.json")


class FormulaGenesisEngine:
    """
    Orchestrator utama untuk memicu penciptaan dan penemuan rumus baru di MOKO.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._init_components()
        self.discovered_db: List[Dict[str, Any]] = []
        self.load_discovered_formulas()

    def _init_components(self):
        """Lazy init komponen FGE."""
        from moko_neuromath.symbolic_regression import get_discoverer
        from moko_neuromath.dimensional_synthesis import get_dim_synthesis_engine
        from moko_neuromath.pattern_conjecturer import get_pattern_conjecturer
        from moko_neuromath.formula_validator import get_formula_validator
        from moko_neuromath.formula_critic import get_formula_critic

        self.sr_discoverer = get_discoverer(verbose=self.verbose)
        self.dim_engine = get_dim_synthesis_engine(verbose=self.verbose)
        self.pattern_con = get_pattern_conjecturer(verbose=self.verbose)
        self.validator = get_formula_validator(verbose=self.verbose)
        self.critic = get_formula_critic(verbose=self.verbose)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [FGE] {msg}")

    def load_discovered_formulas(self):
        """Memuat rumus yang pernah ditemukan sebelumnya dari file JSON."""
        if not os.path.exists(_DISCOVERED_FILE):
            return
        try:
            with open(_DISCOVERED_FILE, "r", encoding="utf-8") as f:
                self.discovered_db = json.load(f)
            self._log(f"Berhasil memuat {len(self.discovered_db)} rumus hasil discovery dari disk.")
            self._register_all_to_engine()
        except Exception as e:
            self._log(f"Gagal memuat discovered formulas: {e}")

    def save_discovered_formulas(self):
        """Menyimpan database rumus hasil discovery ke disk."""
        os.makedirs(_MATH_OMNI_DIR, exist_ok=True)
        try:
            with open(_DISCOVERED_FILE, "w", encoding="utf-8") as f:
                json.dump(self.discovered_db, f, indent=2, ensure_ascii=False)
            self._log("Database rumus hasil discovery berhasil disimpan.")
        except Exception as e:
            self._log(f"Gagal menyimpan discovered formulas: {e}")

    def _register_all_to_engine(self):
        """Mendaftarkan seluruh rumus yang termuat ke AppliedFormulaEngine."""
        from moko_neuromath.applied_formula_engine import get_formula_engine, FormulaRecord, FORMULA_DATABASE
        engine = get_formula_engine()

        for f_data in self.discovered_db:
            # Cegah duplikasi
            if any(f.name == f_data["name"] for f in FORMULA_DATABASE):
                continue

            # Rekonstruksi python_fn dari sympy_str (atau fallback) secara dinamis
            import sympy as sp
            expr_str = f_data.get("sympy_str")
            if not expr_str:
                # Fallback replacement
                raw_expr = f_data["formula_str"].split("=")[-1].strip()
                expr_str = raw_expr.replace("×", "*").replace("^", "**")
            
            expr = sp.sympify(expr_str)
            free_syms = list(expr.free_symbols)
            eval_func = sp.lambdify(free_syms, expr, 'math')

            def make_python_fn(f_syms, ev_func):
                return lambda vars: ev_func(*[vars[str(s)] for s in f_syms])

            prereqs = f_data["prerequisites"]
            
            # Buat record baru
            rec = FormulaRecord(
                name=f_data["name"],
                domain=f_data["domain"],
                formula_str=f_data["formula_str"],
                variables={v: "Variabel terekstrak" for v in prereqs + [f_data["solve_for"]]},
                solve_for=f_data["solve_for"],
                python_fn=make_python_fn(free_syms, eval_func),
                units_out=f_data.get("units_out", "unknown"),
                reference=f_data.get("reference", "Discovered"),
                prerequisites=prereqs,
                notes=f_data.get("notes", "")
            )

            # Masukkan ke database global
            FORMULA_DATABASE.append(rec)
            # Update index singleton engine
            engine._domain_index.setdefault(rec.domain, []).append(rec)

    def discover_from_data(
        self,
        data_X: List[Dict[str, float]],
        data_Y: List[float],
        target_symbol: str,
        domain: str,
        var_names: Optional[List[str]] = None,
        units_out: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """
        Alur Discovery 1: Menemukan rumus dari data eksperimental/observasional.
        """
        # Split data untuk hold-out validation set (80% train, 20% test)
        split_idx = int(len(data_X) * 0.8)
        train_X, test_X = data_X[:split_idx], data_X[split_idx:]
        train_Y, test_Y = data_Y[:split_idx], data_Y[split_idx:]

        # Jalankan Symbolic Regression
        results = self.sr_discoverer.discover(train_X, train_Y, target_name=target_symbol, domain=domain)
        if not results:
            self._log("Symbolic Regression tidak menghasilkan rumus apa pun.")
            return None

        best_res = results[0]
        self._log(f"Rumus kandidat terbaik dari SR: {target_symbol} = {best_res.formula_str} (R²={best_res.r_squared:.4f})")

        # Jalankan validasi
        val_res = self.validator.validate_formula(
            formula_expr_str=best_res.sympy_str,
            target_symbol=target_symbol,
            test_X=test_X,
            test_Y=test_Y,
            target_domain=domain
        )

        # Jalankan kritik Red Team (anti-halusinasi / adversarial audit)
        critic_report = self.critic.evaluate_formula(
            formula_expr_str=best_res.sympy_str,
            target_symbol=target_symbol,
            variables=best_res.variables_used,
            train_r2=best_res.r_squared,
            test_r2=val_res.r_squared,
            domain_name=domain
        )

        # Hanya simpan jika lolos kritik Red Team dan validasi dasar terpenuhi
        if val_res.is_valid and critic_report.passed:
            new_formula = {
                "name": f"Discovered: {target_symbol} = {best_res.formula_str}",
                "domain": domain,
                "formula_str": f"{target_symbol} = {best_res.formula_str}",
                "sympy_str": best_res.sympy_str,
                "solve_for": target_symbol,
                "prerequisites": best_res.variables_used,
                "units_out": units_out,
                "reference": "Discovered via MOKO FGE (Red Team Certified)",
                "notes": f"R2={best_res.r_squared:.5f}, defense_score={critic_report.defense_score:.2f}",
                "timestamp": time.time()
            }
            
            # Update database lokal dan disk
            self.discovered_db.append(new_formula)
            self.save_discovered_formulas()
            self._register_all_to_engine()
            
            self._log(f"🎉 SUKSES: Rumus baru '{target_symbol} = {best_res.formula_str}' lolos audit Red Team & terdaftar di MOKO.")
            return new_formula

        self._log("Gagal audit Red Team / validasi kelayakan. Rumus ditolak untuk menghindari halusinasi.")
        if critic_report.failed_attacks:
            self._log(f"Serangan Red Team yang berhasil menembus: {critic_report.failed_attacks}")
        return None

    def discover_from_sequence(
        self,
        sequence: List[float],
        target_symbol: str,
        domain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Alur Discovery 2: Menemukan rumus deret / sekuens.
        """
        res = self.pattern_con.analyze(sequence)
        if not res or res.confidence < 0.90:
            return None

        new_formula = {
            "name": f"Discovered Sequence: {target_symbol} = {res.formula_str}",
            "domain": domain,
            "formula_str": f"{res.formula_str.replace('a(n)', target_symbol)}",
            "sympy_str": res.sympy_expr_str,
            "solve_for": target_symbol,
            "prerequisites": ["n"],
            "units_out": "integer",
            "reference": f"Sequence Pattern ({res.pattern_type})",
            "notes": f"Confidence={res.confidence:.2f}",
            "timestamp": time.time()
        }

        self.discovered_db.append(new_formula)
        self.save_discovered_formulas()
        self._register_all_to_engine()
        return new_formula


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_fge_instance: Optional[FormulaGenesisEngine] = None

def get_fge(verbose: bool = False) -> FormulaGenesisEngine:
    global _fge_instance
    if _fge_instance is None:
        _fge_instance = FormulaGenesisEngine(verbose=verbose)
    return _fge_instance
