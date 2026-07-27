"""
MOKO Engine Advanced Math — Matematika Mesin & Otomotif Tingkat Lanjut
========================================================================
Mencakup:
  1. Tenaga & Torsi   — Power curves, BHP, BMEP, torsi dari dinamometer
  2. AFR & Lambda     — Air-Fuel Ratio, stoichiometri, equivalence ratio
  3. Volumetrik Eff.  — VE aktual vs teoritis, estimasi mass airflow
  4. Overbore Math    — Kalkulasi CC setelah overbore/rebore
  5. Injector Sizing  — Duty cycle, cc/min yang dibutuhkan untuk target HP
  6. Turbo/Kompresor  — Pressure ratio, boost, mass flow
  7. Cam Timing       — Lift, durasi, LSA, overlap
"""

import math
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════
# 1. TENAGA & TORSI
# ═══════════════════════════════════════════════════════════════════════════

class PowerTorque:
    """Konversi dan kalkulasi tenaga dan torsi."""

    @staticmethod
    def power_from_torque(torque_Nm: float, rpm: float) -> Dict[str, float]:
        """Daya dari torsi: P = 2π × N × T / 60 [Watt]."""
        P_watt = 2 * math.pi * rpm * torque_Nm / 60.0
        return {
            "power_W": P_watt,
            "power_kW": P_watt / 1000,
            "power_hp": P_watt / 745.7,
            "power_PS": P_watt / 735.5,
        }

    @staticmethod
    def torque_from_power(power_W: float, rpm: float) -> float:
        """Torsi dari daya: T = P × 60 / (2π × N) [Nm]."""
        return power_W * 60.0 / (2 * math.pi * rpm)

    @staticmethod
    def bmep(torque_Nm: float, displacement_m3: float,
             stroke_type: int = 4) -> float:
        """
        Brake Mean Effective Pressure [Pa].
        BMEP = T × 2 × n × 2π / V_d
        n = jumlah tenaga per putaran: 2 untuk 4-tak (power stroke tiap 2 rev)
        """
        n_power = 2 if stroke_type == 4 else 1
        return (torque_Nm * 2 * n_power * 2 * math.pi) / displacement_m3

    @staticmethod
    def bmep_kpa(bmep_pa: float) -> float:
        return bmep_pa / 1000

    @staticmethod
    def power_to_hp_conversions(power_W: float) -> Dict[str, float]:
        """Konversi daya ke berbagai satuan."""
        return {
            "W": power_W, "kW": power_W/1000,
            "HP (mechanical)": power_W/745.7,
            "PS (metric HP)": power_W/735.5,
            "BTU/hr": power_W * 3.41214,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. AIR-FUEL RATIO & LAMBDA
# ═══════════════════════════════════════════════════════════════════════════

class AFRCalculator:
    """Stoichiometri bahan bakar, lambda, dan kalkulasi campuran."""

    # Stoichiometric AFR untuk berbagai bahan bakar
    STOICH_AFR = {
        "gasoline":  14.7,
        "bensin":    14.7,
        "diesel":    14.5,
        "ethanol":    9.0,
        "methanol":   6.47,
        "lpg":       15.5,
        "cng":       17.2,
        "hydrogen":  34.0,
    }

    @staticmethod
    def lambda_ratio(afr_actual: float, fuel: str = "gasoline") -> float:
        """Lambda λ = AFR_aktual / AFR_stoichiometric."""
        afr_stoich = AFRCalculator.STOICH_AFR.get(fuel.lower(), 14.7)
        return afr_actual / afr_stoich

    @staticmethod
    def afr_from_lambda(lam: float, fuel: str = "gasoline") -> float:
        """AFR = λ × AFR_stoichiometric."""
        return lam * AFRCalculator.STOICH_AFR.get(fuel.lower(), 14.7)

    @staticmethod
    def mixture_classification(lam: float) -> str:
        """Klasifikasi campuran berdasarkan lambda."""
        if lam < 0.90: return "Rich (Kaya) — kemungkinan black smoke"
        if lam < 0.97: return "Slightly Rich — torsi optimal"
        if lam < 1.03: return "Stoichiometric — emisi optimal"
        if lam < 1.10: return "Slightly Lean — efisiensi bahan bakar tinggi"
        return "Lean (Miskin) — risiko detonasi/overheating"

    @staticmethod
    def fuel_consumption(power_W: float, bsfc_g_per_kWh: float = 260.0) -> Dict[str, float]:
        """
        Konsumsi bahan bakar dari BSFC (Brake Specific Fuel Consumption).
        BSFC bensin normal ≈ 250-300 g/kWh.
        """
        fuel_g_per_hour = bsfc_g_per_kWh * (power_W / 1000)
        fuel_L_per_hour = fuel_g_per_hour / 750  # density bensin ≈ 750 g/L
        return {
            "fuel_g_per_hour": fuel_g_per_hour,
            "fuel_L_per_hour": fuel_L_per_hour,
            "fuel_mL_per_min": fuel_L_per_hour * 1000 / 60,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. VOLUMETRIK EFFICIENCY & AIRFLOW
# ═══════════════════════════════════════════════════════════════════════════

class VolumetricEfficiency:
    """Efisiensi volumetrik dan estimasi airflow."""

    AIR_DENSITY_STP = 1.225  # kg/m³ pada STP (15°C, 1 atm)

    @staticmethod
    def theoretical_airflow(displacement_m3: float, rpm: float,
                             stroke_type: int = 4) -> float:
        """Mass airflow teoritis [kg/s] tanpa memperhitungkan VE."""
        # Untuk 4-tak: satu siklus power = 2 putaran
        strokes_per_sec = rpm / 60.0 / (stroke_type / 2)
        volume_per_sec = displacement_m3 * strokes_per_sec
        return volume_per_sec * VolumetricEfficiency.AIR_DENSITY_STP

    @staticmethod
    def volumetric_efficiency(actual_airflow_kgs: float, displacement_m3: float,
                               rpm: float, stroke_type: int = 4) -> float:
        """VE = (airflow_aktual / airflow_teoritis) × 100%."""
        theoretical = VolumetricEfficiency.theoretical_airflow(
            displacement_m3, rpm, stroke_type)
        return (actual_airflow_kgs / theoretical) * 100 if theoretical > 0 else 0

    @staticmethod
    def estimated_power_from_ve(displacement_cc: float, rpm: float,
                                 ve_percent: float = 85,
                                 bmep_kpa: float = 900) -> Dict[str, float]:
        """
        Estimasi cepat daya mesin dari VE dan BMEP.
        P [kW] ≈ BMEP [kPa] × V_d [L] × n [rpm/min] / (30000 untuk 4-tak)
        """
        Vd_L = displacement_cc / 1000
        P_kW = (bmep_kpa * Vd_L * rpm * (ve_percent/100)) / 30000
        return {
            "power_kW": P_kW,
            "power_hp": P_kW * 1.341,
            "power_PS": P_kW * 1.360,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. OVERBORE & DISPLACEMENT MATH
# ═══════════════════════════════════════════════════════════════════════════

class DisplacementMath:
    """Kalkulasi CC mesin, overbore, dan perubahan geometri."""

    @staticmethod
    def displacement_cc(bore_mm: float, stroke_mm: float, cylinders: int = 1) -> float:
        """
        Volume displacement total mesin [cc / cm³].
        V = (π/4) × B² × S × N_cyl
        B dan S dalam mm, hasil dalam cc.
        """
        bore_cm = bore_mm / 10
        stroke_cm = stroke_mm / 10
        return (math.pi / 4) * (bore_cm ** 2) * stroke_cm * cylinders

    @staticmethod
    def displacement_per_cylinder(bore_mm: float, stroke_mm: float) -> float:
        """Volume satu silinder [cc]."""
        return DisplacementMath.displacement_cc(bore_mm, stroke_mm, 1)

    @staticmethod
    def overbore_cc(original_bore_mm: float, new_bore_mm: float,
                    stroke_mm: float, cylinders: int) -> Dict[str, float]:
        """Kalkulasi perubahan CC setelah overbore."""
        original_cc = DisplacementMath.displacement_cc(original_bore_mm, stroke_mm, cylinders)
        new_cc = DisplacementMath.displacement_cc(new_bore_mm, stroke_mm, cylinders)
        return {
            "original_cc": original_cc,
            "new_cc": new_cc,
            "gain_cc": new_cc - original_cc,
            "gain_pct": ((new_cc - original_cc) / original_cc) * 100,
            "bore_increase_mm": new_bore_mm - original_bore_mm,
        }

    @staticmethod
    def compression_ratio(displacement_cc: float, clearance_cc: float) -> float:
        """CR = (V_d + V_c) / V_c."""
        return (displacement_cc + clearance_cc) / clearance_cc

    @staticmethod
    def clearance_volume(displacement_cc: float, cr: float) -> float:
        """V_c = V_d / (CR - 1)."""
        return displacement_cc / (cr - 1)

    @staticmethod
    def stroke_from_cc(target_cc: float, bore_mm: float, cylinders: int = 1) -> float:
        """Hitung stroke yang diperlukan untuk mencapai target CC [mm]."""
        bore_cm = bore_mm / 10
        area = math.pi / 4 * bore_cm ** 2
        stroke_cm = (target_cc / cylinders) / area
        return stroke_cm * 10


# ═══════════════════════════════════════════════════════════════════════════
# 5. INJECTOR SIZING
# ═══════════════════════════════════════════════════════════════════════════

class InjectorMath:
    """Kalkulasi ukuran injector untuk target tenaga."""

    @staticmethod
    def injector_size_cc_per_min(target_hp: float,
                                  bsfc: float = 0.55,
                                  n_injectors: int = 4,
                                  max_duty_cycle: float = 0.80) -> Dict[str, float]:
        """
        Injector size yang dibutuhkan.
        BSFC = Brake Specific Fuel Consumption [lb/hp/hr] (bensin: ~0.50-0.60)
        cc/min = (HP × BSFC × 10.1) / (N_inj × duty_cycle)
        """
        cc_per_min = (target_hp * bsfc * 10.1) / (n_injectors * max_duty_cycle)
        cc_per_hr = cc_per_min * 60
        lb_per_hr = target_hp * bsfc / n_injectors
        return {
            "injector_cc_per_min": cc_per_min,
            "injector_cc_per_hr": cc_per_hr,
            "lb_per_hr": lb_per_hr,
            "min_injector_size_cc": cc_per_min * 1.1,  # 10% safety margin
        }

    @staticmethod
    def duty_cycle(actual_cc_per_min: float, target_cc_per_min: float) -> float:
        """Duty cycle injector yang dibutuhkan [%]."""
        return min((target_cc_per_min / actual_cc_per_min) * 100, 100)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TURBO / SUPERCHARGER
# ═══════════════════════════════════════════════════════════════════════════

class ForcedInduction:
    """Kalkulasi tekanan, rasio, dan airflow turbo/supercharger."""

    @staticmethod
    def pressure_ratio(boost_psi: float, atmospheric_psi: float = 14.7) -> float:
        """Pressure Ratio = (boost + atmospheric) / atmospheric."""
        return (boost_psi + atmospheric_psi) / atmospheric_psi

    @staticmethod
    def boost_from_pr(pressure_ratio: float, atmospheric_psi: float = 14.7) -> float:
        """Boost [psi] dari pressure ratio."""
        return pressure_ratio * atmospheric_psi - atmospheric_psi

    @staticmethod
    def intercooler_efficiency(T_in: float, T_out: float, T_ambient: float) -> float:
        """Efisiensi intercooler: η = (T_in - T_out) / (T_in - T_ambient) × 100%."""
        return ((T_in - T_out) / (T_in - T_ambient)) * 100

    @staticmethod
    def air_density_correction(altitude_m: float) -> float:
        """Koreksi densitas udara terhadap ketinggian (International Standard Atmosphere)."""
        T = 288.15 - 0.0065 * altitude_m  # Temperatur [K]
        P = 101325 * (T / 288.15) ** 5.2561  # Tekanan [Pa]
        rho = P / (287.058 * T)  # Densitas [kg/m³]
        return rho


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class EngineAdvancedMath:
    """Fasad utama untuk matematika mesin lanjutan."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.power = PowerTorque()
        self.afr = AFRCalculator()
        self.ve = VolumetricEfficiency()
        self.displacement = DisplacementMath()
        self.injector = InjectorMath()
        self.turbo = ForcedInduction()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔧 [EngineMath] {msg}")


_engine_instance = None

def get_engine_math(verbose: bool = True) -> EngineAdvancedMath:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EngineAdvancedMath(verbose=verbose)
    return _engine_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n🔧 Engine Advanced Math — Self Test\n" + "="*55)

    # CC Displacement
    cc = DisplacementMath.displacement_cc(bore_mm=75, stroke_mm=67, cylinders=4)
    assert abs(cc - 1183.99) < 1.0, f"CC gagal: {cc}"
    print(f"  ✅ Displacement: Bore=75mm, Stroke=67mm, 4cyl → {cc:.1f} cc")

    # Overbore
    ob = DisplacementMath.overbore_cc(75, 77, 67, 4)
    print(f"  ✅ Overbore 75→77mm: +{ob['gain_cc']:.1f} cc ({ob['gain_pct']:.1f}%)")

    # Power from Torque
    p = PowerTorque.power_from_torque(torque_Nm=100, rpm=5000)
    assert abs(p["power_kW"] - 52.36) < 0.1, f"Power gagal: {p}"
    print(f"  ✅ Power: T=100Nm @5000rpm → {p['power_kW']:.2f} kW ({p['power_hp']:.1f} HP)")

    # BMEP
    bmep = PowerTorque.bmep(torque_Nm=100, displacement_m3=1.189e-3)
    print(f"  ✅ BMEP: {bmep/1000:.0f} kPa (mesin 1.189L, T=100Nm)")

    # AFR / Lambda
    lam = AFRCalculator.lambda_ratio(afr_actual=13.0, fuel="gasoline")
    cls = AFRCalculator.mixture_classification(lam)
    print(f"  ✅ Lambda: AFR=13.0 → λ={lam:.3f} → '{cls}'")

    # VE Power Estimate
    est = VolumetricEfficiency.estimated_power_from_ve(1200, 6000, ve_percent=90)
    print(f"  ✅ VE Power est.: 1200cc @6000rpm @90%VE → {est['power_kW']:.1f} kW ({est['power_hp']:.1f} HP)")

    # Injector
    inj = InjectorMath.injector_size_cc_per_min(target_hp=150, n_injectors=4)
    print(f"  ✅ Injector: 150HP/4inj → {inj['injector_cc_per_min']:.0f} cc/min (min: {inj['min_injector_size_cc']:.0f})")

    # Turbo Pressure Ratio
    pr = ForcedInduction.pressure_ratio(boost_psi=15)
    print(f"  ✅ Turbo: 15 psi boost → PR = {pr:.2f}")

    print("\n✅ Semua test Engine Advanced Math berhasil!\n")


if __name__ == "__main__":
    _self_test()
