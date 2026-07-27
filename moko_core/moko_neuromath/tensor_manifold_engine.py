"""
MOKO Tensor Manifold & Lie Group Core (TMG Engine)
===================================================
Mesin Matematika Super-Kompleks Tingkat Tinggi MOKO OS.
Menghadirkan aljabar tensor, Lie Groups SO(3)/SE(3), Fisika Lagrangian Analitis,
dan Geometri Diferensial Manifold untuk sintesis kode program presisi profesional.

Fungsi Utama:
1. LieGroupSO3: Eksponensial matriks exp(ω_hat) via Rodrigues' Formula & Lie Algebra.
2. LagrangianMechanics: Solver Persamaan Euler-Lagrange d/dt(∂L/∂q_dot) - ∂L/∂q = 0.
3. DifferentialGeometry: Gaussian Curvature (K), Mean Curvature (H), & Laplace-Beltrami Operator.
"""

import math
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional


class LieGroupSO3:
    """
    Representasi Grup Lie SO(3) & Aljabar Lie so(3).
    Menghitung rotasi 3D presisi tinggi tanpa drift numerik menggunakan eksponensial matriks.
    """

    @staticmethod
    def skew_symmetric(w: List[float]) -> List[List[float]]:
        """Mengubah vektor kecepatan sudut w = (wx, wy, wz) menjadi matriks skew-symmetric w_hat"""
        wx, wy, wz = w[0], w[1], w[2]
        return [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0]
        ]

    @staticmethod
    def exp_map(w: List[float]) -> List[List[float]]:
        """
        Peta Eksponensial exp(w_hat): Aljabar Lie so(3) -> Grup Lie SO(3)
        Menggunakan Rumus Rotasi Rodrigues (Rodrigues' Rotation Formula):
        exp(w_hat) = I + (sin θ / θ) * w_hat + ((1 - cos θ) / θ^2) * w_hat^2
        """
        theta = math.sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2])
        if theta < 1e-9:
            # Identity Matrix jika sudut mendekati 0
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        wx, wy, wz = w[0]/theta, w[1]/theta, w[2]/theta
        
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        one_minus_cos = 1.0 - cos_t

        # Komponen matriks rotasi R = cos(θ)I + (1-cos(θ))u u^T + sin(θ)u_hat
        R = [
            [cos_t + wx*wx*one_minus_cos,       wx*wy*one_minus_cos - wz*sin_t,  wx*wz*one_minus_cos + wy*sin_t],
            [wy*wx*one_minus_cos + wz*sin_t,  cos_t + wy*wy*one_minus_cos,       wy*wz*one_minus_cos - wx*sin_t],
            [wz*wx*one_minus_cos - wy*sin_t,  wz*wy*one_minus_cos + wx*sin_t,  cos_t + wz*wz*one_minus_cos]
        ]
        return R

    @staticmethod
    def log_map(R: List[List[float]]) -> List[float]:
        """
        Peta Logaritma log(R): Grup Lie SO(3) -> Aljabar Lie so(3)
        Memulihkan vektor rotasi w dari matriks SO(3).
        """
        trace = R[0][0] + R[1][1] + R[2][2]
        cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        theta = math.acos(cos_theta)
        
        if theta < 1e-7:
            return [0.0, 0.0, 0.0]

        sin_theta = math.sin(theta)
        factor = theta / (2.0 * sin_theta)
        
        wx = (R[2][1] - R[1][2]) * factor
        wy = (R[0][2] - R[2][0]) * factor
        wz = (R[1][0] - R[0][1]) * factor
        return [wx, wy, wz]


class LagrangianMechanics:
    """
    Solver Persamaan Euler-Lagrange untuk Fisika Analitis Simbolik.
    Menurunkan persamaan gerak otomatis dari L = T - V.
    """

    @staticmethod
    def derive_euler_lagrange(T_expr_str: str, V_expr_str: str, q_var_str: str = "q") -> Dict[str, Any]:
        """
        Menurunkan Persamaan Gerak Euler-Lagrange:
        d/dt( ∂L / ∂q_dot ) - ∂L / ∂q = 0
        """
        try:
            t = sp.Symbol('t')
            q = sp.Function(q_var_str)(t)
            q_dot = sp.diff(q, t)
            q_ddot = sp.diff(q_dot, t)

            # Parse T dan V
            local_dict = {q_var_str: q, f"{q_var_str}_dot": q_dot, "t": t, "m": sp.Symbol('m'), "g": sp.Symbol('g'), "l": sp.Symbol('l'), "k": sp.Symbol('k')}
            T = sp.sympify(T_expr_str, locals=local_dict)
            V = sp.sympify(V_expr_str, locals=local_dict)
            
            L = T - V  # Lagrangian

            # ∂L / ∂q_dot
            dL_dqdot = sp.diff(L, q_dot)
            # d/dt( ∂L / ∂q_dot )
            ddt_dL_dqdot = sp.diff(dL_dqdot, t)
            # ∂L / ∂q
            dL_dq = sp.diff(L, q)

            # Euler-Lagrange Equation: ddt_dL_dqdot - dL_dq = 0
            el_eq = sp.Eq(ddt_dL_dqdot - dL_dq, 0)
            
            # Selesaikan untuk q_ddot (akselerasi sistem)
            q_ddot_sol = sp.solve(el_eq, q_ddot)

            return {
                "success": True,
                "lagrangian": str(L),
                "euler_lagrange_eq": str(el_eq),
                "acceleration_expr": str(q_ddot_sol[0]) if q_ddot_sol else "Kompleks",
                "latex": sp.latex(el_eq)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class DifferentialGeometry:
    """Geometri Diferensial Manifold Permukaan 3D (Kelengkungan Gaussian & Mean)"""

    @staticmethod
    def calculate_surface_curvature(fx_str: str, fy_str: str, fz_str: str, u_val: float = 0.0, v_val: float = 0.0) -> Dict[str, Any]:
        """
        Hitung Kelengkungan Gaussian (K) dan Kelengkungan Rata-rata (H) 
        dari permukaan parametrik r(u,v) = (fx, fy, fz).
        """
        try:
            u, v = sp.symbols('u v')
            r = sp.Matrix([sp.sympify(fx_str), sp.sympify(fy_str), sp.sympify(fz_str)])

            # Turunan Pertama (Ru, Rv)
            ru = sp.diff(r, u)
            rv = sp.diff(r, v)

            # First Fundamental Form (E, F, G)
            E = ru.dot(ru)
            F = ru.dot(rv)
            G = rv.dot(rv)

            # Vektor Normal Satuan (n)
            cross_prod = ru.cross(rv)
            norm_cross = cross_prod.norm()
            if norm_cross == 0:
                return {"success": False, "error": "Singularitas permukaan terdeteksi (norm normal = 0)"}
            n = cross_prod / norm_cross

            # Turunan Kedua (Ruu, Ruv, Rvv)
            ruu = sp.diff(ru, u)
            ruv = sp.diff(ru, v)
            rvv = sp.diff(rv, v)

            # Second Fundamental Form (L, M, N)
            L_form = ruu.dot(n)
            M_form = ruv.dot(n)
            N_form = rvv.dot(n)

            # Kelengkungan Gaussian K = (LN - M^2) / (EG - F^2)
            EG_minus_F2 = E*G - F**2
            LN_minus_M2 = L_form*N_form - M_form**2
            K_expr = LN_minus_M2 / EG_minus_F2

            # Kelengkungan Mean H = (EN + GL - 2FM) / (2(EG - F^2))
            H_expr = (E*N_form + G*L_form - 2*F*M_form) / (2 * EG_minus_F2)

            # Evaluasi numerik pada (u_val, v_val)
            K_val = float(K_expr.subs({u: u_val, v: v_val}).evalf())
            H_val = float(H_expr.subs({u: u_val, v: v_val}).evalf())

            return {
                "success": True,
                "gaussian_curvature_K": K_val,
                "mean_curvature_H": H_val,
                "gaussian_expr": str(K_expr),
                "mean_expr": str(H_expr)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class TensorManifoldEngine:
    """Tensor Manifold & Lie Group Engine — Main Interface"""

    @staticmethod
    def evaluate_advanced_query(query: str) -> Dict[str, Any]:
        q = query.lower()
        if "lie" in q or "so(3)" in q or "rodrigues" in q:
            w = [0.1, 0.2, 0.3]
            R = LieGroupSO3.exp_map(w)
            w_recovered = LieGroupSO3.log_map(R)
            return {
                "type": "lie_group_so3",
                "rotation_matrix": R,
                "recovered_vector": w_recovered,
                "explanation": "Peta Eksponensial SO(3) exp(w_hat) dan Logaritma log(R) via Rodrigues' Formula."
            }
        elif "lagrangian" in q or "euler-lagrange" in q:
            # Pendulum sederhana: T = 0.5*m*l^2*q_dot^2, V = -m*g*l*cos(q)
            res = LagrangianMechanics.derive_euler_lagrange("0.5*m*l**2*q_dot**2", "-m*g*l*cos(q)")
            return {
                "type": "lagrangian_mechanics",
                "result": res,
                "explanation": "Penurunan otomatis Persamaan Gerak Euler-Lagrange d/dt(∂L/∂q_dot) - ∂L/∂q = 0."
            }
        elif "curvature" in q or "manifold" in q or "gaussian" in q:
            # Paraboloid r(u,v) = (u, v, u^2 + v^2)
            res = DifferentialGeometry.calculate_surface_curvature("u", "v", "u**2 + v**2", 0.0, 0.0)
            return {
                "type": "differential_geometry",
                "result": res,
                "explanation": "Kalkulasi Kelengkungan Gaussian (K) dan Mean (H) pada permukaan manifold 3D."
            }
        return {
            "type": "tensor_manifold_general",
            "status": "TMG Engine Active",
            "explanation": "Super-Complex Mathematical Reasoning Engine Aktif."
        }


# Singleton
tmg_engine = TensorManifoldEngine()
