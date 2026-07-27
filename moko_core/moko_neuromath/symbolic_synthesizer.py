"""
MOKO Symbolic Synthesizer — Layer 2b of ARMS
=============================================
Penyelesaian sistem persamaan simultan secara dinamis menggunakan SymPy.
Sistem dinyatakan siap jika bisa menyusun rumus dan menyelesaikannya sendiri.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import sympy as sp

from .applied_formula_engine import (
    FormulaRecord, FormulaSolution, FormulaSource, FORMULA_DATABASE
)

class SymbolicSynthesizer:
    """
    Sintesis dan penyelesaian sistem persamaan simultan secara dinamis.
    Menerima variabel yang diketahui (known) dan mencari variabel target.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [SymbolicSynthesizer] {msg}")

    def get_sympy_equation(self, formula: FormulaRecord, syms: Dict[str, sp.Symbol]) -> Optional[sp.Eq]:
        """
        Dapatkan sp.Eq (persamaan SymPy) untuk FormulaRecord.
        Mencoba lookup tabel, jika tidak ada, lakukan fallback parsing atau mock execution.
        """
        # Tabel representasi simbolik teoretis
        formula_exprs = {
            "Engine Displacement": sp.Eq(syms["V_d"], (sp.pi / 4) * syms["B"]**2 * syms["S"] * syms.get("N_cyl", 1.0)),
            "Compression Ratio": sp.Eq(syms["CR"], (syms["V_d"] + syms["V_c"]) / syms["V_c"]),
            "Combustion Chamber Volume": sp.Eq(syms["V_c"], syms["V_d"] / (syms["CR"] - 1.0)),
            "Crankshaft Radius": sp.Eq(syms["r_crank"], syms["S"] / 2.0),
            "Piston Position": sp.Eq(syms["x"], syms["r_crank"] * sp.cos(syms.get("theta_crank", sp.Symbol("θ_crank"))) + sp.sqrt(syms["L_con"]**2 - syms["r_crank"]**2 * sp.sin(syms.get("theta_crank", sp.Symbol("θ_crank")))**2)),
            "Ignition Angle Lead Time": sp.Eq(syms["theta_ign"], 360.0 * syms["RPM"] * syms["t_lead"]),
            
            # Physics / General
            "Piston Force from Pressure": sp.Eq(syms["F"], syms["P"] * syms["A"]),
            "Circle Area from Diameter": sp.Eq(syms["A"], (sp.pi / 4) * syms["D"]**2),
            "Circle Area from Radius": sp.Eq(syms["A"], sp.pi * syms["r"]**2),
            "Wave Frequency from Speed and Wavelength": sp.Eq(syms["f"], syms["v"] / syms.get("λ", sp.Symbol("λ"))),
            "Wavelength from Speed and Frequency": sp.Eq(syms.get("λ", sp.Symbol("λ")), syms["v"] / syms["f"]),
            "Speed of Sound in Air (Temperature-dependent)": sp.Eq(syms["v"], 331.3 * sp.sqrt(syms["T"] / 273.15)),
            "Fundamental Frequency of Closed Tube": sp.Eq(syms["f"], syms["v"] / (4 * syms["L"])),
            "Fundamental Frequency of Open Tube": sp.Eq(syms["f"], syms["v"] / (2 * syms["L"])),
            "Sensible Heat": sp.Eq(syms["Q"], syms["m"] * syms["c"] * syms.get("ΔT", sp.Symbol("ΔT"))),
            "Ohm's Law — Voltage": sp.Eq(syms["V"], syms["I"] * syms["R"]),
            "Ohm's Law — Current": sp.Eq(syms["I"], syms["V"] / syms["R"]),
            "Kinetic Energy": sp.Eq(sp.Symbol("Ek"), 0.5 * syms["m"] * syms["v"]**2),
            "Potential Energy": sp.Eq(sp.Symbol("Ep"), syms["m"] * 9.80665 * syms["h"]),
            "Compound Interest Future Value": sp.Eq(syms["FV"], syms["PV"] * (1 + syms["r"])**syms["n"]),
        }

        if formula.name in formula_exprs:
            return formula_exprs[formula.name]

        # Fallback 1: Parse formula_str
        try:
            eq_str = formula.formula_str.replace('×', '*').replace('²', '**2').replace('π', 'pi').replace('√', 'sqrt')
            parts = eq_str.split('=')
            if len(parts) == 2:
                lhs = sp.parsing.sympy_parser.parse_expr(parts[0].strip(), local_dict=syms)
                rhs = sp.parsing.sympy_parser.parse_expr(parts[1].strip(), local_dict=syms)
                return sp.Eq(lhs, rhs)
        except Exception:
            pass

        # Fallback 2: Mocking math functions & execute python_fn
        try:
            class MockMath:
                pi = sp.pi
                @staticmethod
                def sqrt(x): return sp.sqrt(x)
                @staticmethod
                def sin(x): return sp.sin(x)
                @staticmethod
                def cos(x): return sp.cos(x)
                @staticmethod
                def log10(x): return sp.log(x) / sp.log(10)

            import builtins
            orig_import = builtins.__import__
            def mock_import(name, *args, **kwargs):
                if name == 'math':
                     return MockMath
                return orig_import(name, *args, **kwargs)
            builtins.__import__ = mock_import
            try:
                rhs = formula.python_fn(syms)
                builtins.__import__ = orig_import
                return sp.Eq(syms[formula.solve_for], rhs)
            except Exception:
                builtins.__import__ = orig_import
        except Exception:
            pass

        return None

    def synthesize_and_solve(
        self,
        domain: str,
        known: Dict[str, float],
        target: str
    ) -> Optional[FormulaSolution]:
        """
        Kumpulkan semua rumus dari domain, susun sistem persamaan,
        selesaikan secara dinamis menggunakan SymPy.
        """
        self._log(f"Synthesizing formulas for target: '{target}' from known: {list(known.keys())}")

        # 1. Kumpulkan rumus relevan dari domain utama (atau cross-domain jika relevan)
        relevant_formulas = [
            f for f in FORMULA_DATABASE 
            if f.domain == domain or f.domain in ("engine_mechanics", "fluid_mechanics", "kinematics", "energy")
        ]

        # 2. Definisikan semua simbol SymPy yang dibutuhkan
        all_var_names = set()
        for f in relevant_formulas:
            all_var_names.update(f.variables.keys())
            all_var_names.add(f.solve_for)
        all_var_names.update(known.keys())
        all_var_names.add(target)

        # Standardisasi karakter khusus agar kompatibel dengan penamaan variabel python/sympy
        sp_syms = {}
        for name in all_var_names:
            sp_name = name.replace('λ', 'lambda_').replace('Δ', 'delta_').replace('θ', 'theta_').replace('η', 'eta_')
            sp_syms[name] = sp.Symbol(sp_name)

        # 3. Bangun persamaan dari rumus-rumus
        eqs = []
        used_records = []
        for f in relevant_formulas:
            eq = self.get_sympy_equation(f, sp_syms)
            if eq is not None:
                eqs.append(eq)
                used_records.append(f)

        # 4. Tambahkan persamaan dari variabel yang diketahui
        known_eqs = []
        for k, v in known.items():
            if k in sp_syms:
                known_eqs.append(sp.Eq(sp_syms[k], v))

        all_eqs = eqs + known_eqs
        self._log(f"Total equations in system: {len(all_eqs)}")

        # 5. Cari variabel bebas yang perlu diselesaikan
        symbols_in_eqs = set()
        for eq in all_eqs:
            symbols_in_eqs.update(eq.free_symbols)

        # Hanya selesaikan variabel yang ada dalam persamaan dan tidak diketahui nilainya secara langsung
        known_symbols = {sp_syms[k] for k in known.keys() if k in sp_syms}
        vars_to_solve = list(symbols_in_eqs - known_symbols)

        if sp_syms[target] not in symbols_in_eqs:
            self._log(f"Target symbol '{target}' is not present in the system equations.")
            return None

        # 6. Selesaikan sistem persamaan simultan secara simbolik/numerik
        try:
            self._log(f"Solving system for variables: {vars_to_solve}")
            solutions = sp.solve(all_eqs, vars_to_solve, dict=True)
            self._log(f"Found solutions: {len(solutions)}")
        except Exception as e:
            self._log(f"SymPy solver failed: {e}")
            return None

        if not solutions:
            return None

        # Pilih solusi pertama yang memiliki nilai riil/positif untuk target variabel
        target_sp = sp_syms[target]
        result_value = None

        for sol in solutions:
            if target_sp in sol:
                val = sol[target_sp]
                # Evaluasi ke float jika itu ekspresi simbolik
                try:
                    val_float = float(val.evalf())
                    # Validasi fisik sederhana: hindari nilai imaginer
                    if not math.isnan(val_float) and not math.isinf(val_float):
                        # Filter nilai masuk akal jika ada (misal volume/cc harus positif)
                        if target in ("V_d", "V_c", "CR", "B", "S", "r_crank") and val_float <= 0:
                            continue
                        result_value = val_float
                        break
                except Exception:
                    pass

        if result_value is None:
            # Coba ambil langsung dari dict jika sudah terevaluasi numeric
            for sol in solutions:
                if target_sp in sol:
                    try:
                        result_value = float(sol[target_sp])
                        break
                    except Exception:
                        pass

        if result_value is None:
            self._log("Could not extract a valid numerical value from the solutions.")
            return None

        # 7. Bangun langkah-langkah komputasi dinamis
        steps = [
            "[DYNAMIC SYNTHESIS] MOKO merumuskan sistem persamaan mekanika:",
        ]
        for f in used_records:
            steps.append(f"  • {f.name}: {f.formula_str}")
        steps.append("Nilai variabel input:")
        for k, v in known.items():
            steps.append(f"  • {k} = {v:.6g}")

        steps.append(f"Menyelesaikan sistem persamaan secara simultan untuk target: '{target}'")
        steps.append(f"Hasil: {target} = {result_value:.6g}")

        # Buat dummy FormulaRecord untuk menampung hasil sintesis
        synthesized_formula = FormulaRecord(
            name=f"Synthesized: {target} from {list(known.keys())}",
            domain=domain,
            formula_str=" & ".join([f.formula_str for f in used_records if any(sym in f.variables for sym in known)]),
            variables={k: "" for k in known.keys()},
            solve_for=target,
            python_fn=lambda vars: result_value,
            units_out="m³" if target in ("V_d", "V_c") else "dimensionless" if target == "CR" else "m" if target in ("B", "S", "r_crank", "x") else "degree" if target == "θ_ign" else "SI",
            notes="Disintesis secara dinamis dari prinsip-prinsip mesin terapan."
        )

        return FormulaSolution(
            formula=synthesized_formula,
            inputs=known,
            result_value=result_value,
            result_unit=synthesized_formula.units_out,
            result_symbol=target,
            steps=steps,
            source=FormulaSource.SYNTHESIZED,
        )

# Global Instance
_synthesizer_instance: Optional[SymbolicSynthesizer] = None

def get_synthesizer(verbose: bool = False) -> SymbolicSynthesizer:
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = SymbolicSynthesizer(verbose=verbose)
    return _synthesizer_instance
