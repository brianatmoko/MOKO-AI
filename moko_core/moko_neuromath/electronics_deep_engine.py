"""
MOKO Electronics Deep Engine — Elektronika Lanjutan
=====================================================
Mencakup:
  1. Hukum Kirchhoff (KVL / KCL) & Mesh/Node Analysis
  2. Rangkaian AC — Impedansi, Reaktansi, Daya Kompleks
  3. Transistor BJT — Bias, Titik Q, Penguatan
  4. Op-Amp — Gain, Bandwidth, Konfigurasi
  5. Filter Pasif — RC, RL, RLC cutoff & respons
  6. Power Supply — Efisiensi, Ripple, Regulasi
  7. ADC/DAC & Sensor — Resolusi, SNR, Sensitivitas
"""

import math
from typing import Dict, List, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════
# 1. HUKUM DASAR
# ═══════════════════════════════════════════════════════════════════════════

class OhmKirchhoff:
    """Hukum Ohm dan Kirchhoff untuk rangkaian DC dan AC."""

    @staticmethod
    def ohm(V: Optional[float] = None, I: Optional[float] = None,
            R: Optional[float] = None) -> Dict[str, float]:
        """Hukum Ohm: V = I × R. Berikan 2, dapatkan 1."""
        if V is None and I is not None and R is not None:
            return {"V": I * R, "unit": "V"}
        if I is None and V is not None and R is not None:
            return {"I": V / R, "unit": "A"}
        if R is None and V is not None and I is not None and I != 0:
            return {"R": V / I, "unit": "Ω"}
        raise ValueError("Berikan tepat 2 dari 3 variabel: V, I, R")

    @staticmethod
    def power(V: Optional[float] = None, I: Optional[float] = None,
              R: Optional[float] = None, P: Optional[float] = None) -> Dict[str, float]:
        """Daya listrik: P = V×I = I²×R = V²/R."""
        if V is not None and I is not None:
            return {"P": V * I, "unit": "W"}
        if I is not None and R is not None:
            return {"P": I**2 * R, "unit": "W"}
        if V is not None and R is not None and R != 0:
            return {"P": V**2 / R, "unit": "W"}
        if P is not None and R is not None and R != 0:
            return {"I": math.sqrt(P / R), "V": math.sqrt(P * R)}
        raise ValueError("Kombinasi variabel tidak valid untuk kalkulasi daya")

    @staticmethod
    def resistors_series(resistances: List[float]) -> float:
        """Resistor seri: R_total = R1 + R2 + ..."""
        return sum(resistances)

    @staticmethod
    def resistors_parallel(resistances: List[float]) -> float:
        """Resistor paralel: 1/R_total = 1/R1 + 1/R2 + ..."""
        if any(r == 0 for r in resistances):
            return 0.0
        return 1.0 / sum(1.0 / r for r in resistances)

    @staticmethod
    def voltage_divider(Vin: float, R1: float, R2: float) -> Dict[str, float]:
        """Pembagi tegangan: Vout = Vin × R2 / (R1 + R2)."""
        Rtotal = R1 + R2
        return {
            "Vout": Vin * R2 / Rtotal,
            "V_R1": Vin * R1 / Rtotal,
            "I": Vin / Rtotal,
        }

    @staticmethod
    def current_divider(Iin: float, R1: float, R2: float) -> Dict[str, float]:
        """Pembagi arus: I1 = Iin × R2 / (R1 + R2)."""
        Rtotal = R1 + R2
        return {
            "I1": Iin * R2 / Rtotal,
            "I2": Iin * R1 / Rtotal,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. RANGKAIAN AC
# ═══════════════════════════════════════════════════════════════════════════

class ACCircuits:
    """Impedansi, reaktansi, daya kompleks, dan faktor daya untuk AC."""

    @staticmethod
    def capacitive_reactance(f: float, C: float) -> float:
        """Reaktansi kapasitif Xc = 1 / (2πfC) [Ω]."""
        return 1.0 / (2 * math.pi * f * C)

    @staticmethod
    def inductive_reactance(f: float, L: float) -> float:
        """Reaktansi induktif XL = 2πfL [Ω]."""
        return 2 * math.pi * f * L

    @staticmethod
    def impedance_series_RLC(R: float, f: float, L: float, C: float) -> Dict[str, float]:
        """Impedansi rangkaian seri RLC: Z = √(R² + (XL - Xc)²)."""
        Xc = ACCircuits.capacitive_reactance(f, C)
        XL = ACCircuits.inductive_reactance(f, L)
        X_net = XL - Xc
        Z = math.sqrt(R**2 + X_net**2)
        phase_angle = math.degrees(math.atan2(X_net, R))
        return {
            "Z": Z, "XL": XL, "Xc": Xc, "X_net": X_net,
            "phase_angle_deg": phase_angle,
            "power_factor": math.cos(math.radians(phase_angle))
        }

    @staticmethod
    def resonant_frequency(L: float, C: float) -> float:
        """Frekuensi resonansi: f0 = 1 / (2π√(LC)) [Hz]."""
        return 1.0 / (2 * math.pi * math.sqrt(L * C))

    @staticmethod
    def quality_factor(f0: float, R: float, L: float) -> float:
        """Faktor kualitas Q = 2πf0L / R (untuk rangkaian seri)."""
        return (2 * math.pi * f0 * L) / R

    @staticmethod
    def apparent_power(V_rms: float, I_rms: float) -> float:
        """Daya semu S = V_rms × I_rms [VA]."""
        return V_rms * I_rms

    @staticmethod
    def real_power(S: float, power_factor: float) -> float:
        """Daya nyata P = S × cos(φ) [W]."""
        return S * power_factor

    @staticmethod
    def rms_voltage(V_peak: float) -> float:
        """Tegangan RMS: V_rms = V_peak / √2."""
        return V_peak / math.sqrt(2)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSISTOR BJT
# ═══════════════════════════════════════════════════════════════════════════

class TransistorBJT:
    """Kalkulasi transistor Bipolar Junction Transistor (BJT)."""

    @staticmethod
    def dc_operating_point(Vcc: float, R1: float, R2: float,
                            Rc: float, Re: float, beta: float = 100,
                            Vbe: float = 0.7) -> Dict[str, float]:
        """
        Titik kerja (Q-point) bias pembagi tegangan.
        Konfigurasi paling umum untuk amplifier BJT.
        """
        Vth = Vcc * R2 / (R1 + R2)       # Tegangan Thevenin
        Rth = (R1 * R2) / (R1 + R2)      # Resistansi Thevenin
        Ib = (Vth - Vbe) / (Rth + (beta + 1) * Re)
        Ic = beta * Ib
        Ie = Ic + Ib
        Vce = Vcc - Ic * Rc - Ie * Re
        Vc = Vcc - Ic * Rc
        Ve = Ie * Re
        Vb = Vth - Ib * Rth
        saturated = Vce < 0.2
        cut_off = Ic < 1e-12
        return {
            "Ib_uA": Ib * 1e6,
            "Ic_mA": Ic * 1e3,
            "Ie_mA": Ie * 1e3,
            "Vce": Vce,
            "Vc": Vc,
            "Vb": Vb,
            "Ve": Ve,
            "region": "saturation" if saturated else ("cut-off" if cut_off else "active"),
            "stable": not saturated and not cut_off
        }

    @staticmethod
    def current_gain(Ic: float, Ib: float) -> float:
        """β (beta) = Ic / Ib — penguatan arus DC."""
        return Ic / Ib if Ib != 0 else float('inf')

    @staticmethod
    def alpha_from_beta(beta: float) -> float:
        """α = β / (β + 1) — rasio arus kolektor-emitter."""
        return beta / (beta + 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. OP-AMP
# ═══════════════════════════════════════════════════════════════════════════

class OpAmp:
    """Kalkulasi rangkaian op-amp ideal."""

    @staticmethod
    def inverting_gain(Rf: float, Rin: float) -> Dict[str, float]:
        """Gain inverting: Av = -Rf/Rin."""
        Av = -Rf / Rin
        return {"gain": Av, "gain_dB": 20 * math.log10(abs(Av))}

    @staticmethod
    def non_inverting_gain(Rf: float, R1: float) -> Dict[str, float]:
        """Gain non-inverting: Av = 1 + Rf/R1."""
        Av = 1 + Rf / R1
        return {"gain": Av, "gain_dB": 20 * math.log10(abs(Av))}

    @staticmethod
    def summing_amplifier(input_voltages: List[float], input_resistors: List[float],
                           Rf: float) -> float:
        """Penjumlah inverting: Vout = -Rf × Σ(Vin_i / Rin_i)."""
        return -Rf * sum(v / r for v, r in zip(input_voltages, input_resistors))

    @staticmethod
    def differentiator(Rf: float, C: float, f: float) -> float:
        """Gain diferensiator pada frekuensi f: Av = -2πf × Rf × C."""
        return -2 * math.pi * f * Rf * C

    @staticmethod
    def integrator(Rin: float, C: float, f: float) -> float:
        """Gain integrator pada frekuensi f: Av = -1/(2πf × Rin × C)."""
        return -1.0 / (2 * math.pi * f * Rin * C)

    @staticmethod
    def gbw_bandwidth(gain: float, GBW: float) -> float:
        """Bandwidth dari gain-bandwidth product: BW = GBW / |Av|."""
        return GBW / abs(gain) if gain != 0 else float('inf')

    @staticmethod
    def comparator_threshold(Vref: float, R1: float, R2: float, Vcc: float) -> Dict[str, float]:
        """Histeresis komparator: threshold atas/bawah."""
        V_upper = Vref + (Vcc - Vref) * R1 / (R1 + R2)
        V_lower = Vref * R2 / (R1 + R2)
        return {"V_upper": V_upper, "V_lower": V_lower, "hysteresis": V_upper - V_lower}


# ═══════════════════════════════════════════════════════════════════════════
# 5. FILTER PASIF
# ═══════════════════════════════════════════════════════════════════════════

class PassiveFilter:
    """RC, RL, dan RLC filter: frekuensi cutoff dan respons."""

    @staticmethod
    def rc_lowpass_cutoff(R: float, C: float) -> Dict[str, float]:
        """Low-pass filter: fc = 1/(2πRC)."""
        fc = 1.0 / (2 * math.pi * R * C)
        tau = R * C
        return {"fc_Hz": fc, "tau_s": tau, "omega_c": 2 * math.pi * fc}

    @staticmethod
    def rl_lowpass_cutoff(R: float, L: float) -> Dict[str, float]:
        """RL Low-pass: fc = R / (2πL)."""
        fc = R / (2 * math.pi * L)
        return {"fc_Hz": fc, "tau_s": L / R}

    @staticmethod
    def rlc_bandpass(R: float, L: float, C: float) -> Dict[str, float]:
        """Band-pass RLC: f0, bandwidth, Q factor."""
        f0 = 1.0 / (2 * math.pi * math.sqrt(L * C))
        BW = R / (2 * math.pi * L)
        Q = f0 / BW
        return {"f0_Hz": f0, "bandwidth_Hz": BW, "Q": Q,
                "f_lower": f0 - BW/2, "f_upper": f0 + BW/2}

    @staticmethod
    def gain_at_freq(fc: float, f: float, filter_type: str = "lowpass") -> float:
        """Gain (rasio) filter pada frekuensi f."""
        ratio = f / fc
        if filter_type == "lowpass":
            return 1.0 / math.sqrt(1 + ratio**2)
        elif filter_type == "highpass":
            return ratio / math.sqrt(1 + ratio**2)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. ADC & SENSOR
# ═══════════════════════════════════════════════════════════════════════════

class ADCSensor:
    """Kalkulasi ADC, DAC, dan sensor analog."""

    @staticmethod
    def adc_resolution(bits: int, Vref: float) -> Dict[str, float]:
        """Resolusi ADC: ΔV = Vref / 2^n."""
        levels = 2 ** bits
        step = Vref / levels
        return {
            "n_bits": bits,
            "levels": levels,
            "step_size_V": step,
            "step_size_mV": step * 1000,
        }

    @staticmethod
    def adc_digital_output(V_in: float, Vref: float, bits: int) -> int:
        """Output digital ADC: D = round(V_in / Vref × 2^n)."""
        return round(V_in / Vref * (2 ** bits))

    @staticmethod
    def snr_bits(bits: int) -> float:
        """SNR teoritis ADC ideal: SNR = 6.02×n + 1.76 dB."""
        return 6.02 * bits + 1.76

    @staticmethod
    def thermistor_temperature(R: float, R0: float = 10000,
                                T0: float = 298.15, B: float = 3950) -> float:
        """Temperatur dari resistansi thermistor NTC via persamaan Steinhart-Hart.
        T = 1 / (1/T0 + (1/B) × ln(R/R0)) [Kelvin]"""
        import math
        T = 1.0 / (1.0/T0 + (1.0/B) * math.log(R/R0))
        return T - 273.15  # Konversi ke Celsius


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class ElectronicsDeepEngine:
    """Fasad utama untuk semua kapabilitas elektronika lanjutan."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.ohm = OhmKirchhoff()
        self.ac = ACCircuits()
        self.bjt = TransistorBJT()
        self.opamp = OpAmp()
        self.filter = PassiveFilter()
        self.adc = ADCSensor()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  ⚡ [ElecDeep] {msg}")


_elec_instance = None

def get_electronics_engine(verbose: bool = True) -> ElectronicsDeepEngine:
    global _elec_instance
    if _elec_instance is None:
        _elec_instance = ElectronicsDeepEngine(verbose=verbose)
    return _elec_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n⚡ Electronics Deep Engine — Self Test\n" + "="*55)

    # Ohm's Law
    res = OhmKirchhoff.ohm(V=12.0, R=4.0)
    assert abs(res["I"] - 3.0) < 1e-9, "Ohm gagal"
    print("  ✅ Ohm: I = V/R = 12/4 = 3A")

    # Voltage Divider
    vd = OhmKirchhoff.voltage_divider(Vin=10.0, R1=3000, R2=7000)
    assert abs(vd["Vout"] - 7.0) < 1e-9, "Voltage divider gagal"
    print("  ✅ Voltage Divider: Vout = 10×7k/(3k+7k) = 7V")

    # AC Reactance
    Xc = ACCircuits.capacitive_reactance(f=1000, C=1e-6)
    assert abs(Xc - 159.15) < 0.1, "Xc gagal"
    print(f"  ✅ Reaktansi kapasitif Xc(1kHz, 1µF) = {Xc:.2f} Ω")

    # BJT Q-point
    q = TransistorBJT.dc_operating_point(
        Vcc=12, R1=100000, R2=22000, Rc=3300, Re=1000, beta=100)
    assert q["region"] == "active", f"BJT Q-point gagal: {q}"
    print(f"  ✅ BJT Q-point: Ic={q['Ic_mA']:.2f}mA, Vce={q['Vce']:.2f}V, Region={q['region']}")

    # Op-Amp Gain
    g = OpAmp.inverting_gain(Rf=10000, Rin=1000)
    assert abs(g["gain"] - (-10.0)) < 1e-9, "Op-Amp gain gagal"
    print(f"  ✅ Op-Amp Inverting: Av = -Rf/Rin = -10x ({g['gain_dB']:.1f} dB)")

    # RC Filter
    fc = PassiveFilter.rc_lowpass_cutoff(R=1000, C=1e-6)
    assert abs(fc["fc_Hz"] - 159.15) < 0.1, "RC filter gagal"
    print(f"  ✅ RC Low-pass: fc = 1/(2π×1k×1µF) = {fc['fc_Hz']:.2f} Hz")

    # ADC Resolution
    adc = ADCSensor.adc_resolution(bits=10, Vref=3.3)
    assert adc["levels"] == 1024, "ADC gagal"
    print(f"  ✅ ADC 10-bit (3.3V): {adc['levels']} levels, step={adc['step_size_mV']:.2f}mV")

    print("\n✅ Semua test Electronics Deep Engine berhasil!\n")


if __name__ == "__main__":
    _self_test()
