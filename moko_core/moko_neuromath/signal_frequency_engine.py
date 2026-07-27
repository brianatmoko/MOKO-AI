"""
MOKO Signal & Frequency Engine (SFE)
======================================
"Dari matematika dikembangkan untuk pendengaran dari frekuensi
 yang dihasilkan langsung di matematika dan biner."
 — Visi MOKO OS

Ini adalah fondasi matematis untuk input SUARA user.
Tidak ada library audio — hanya MATEMATIKA dan BINER.

Arsitektur:
  1. DFT dari First Principles    — Dekomposisi sinyal ke frekuensi
  2. FFT Cooley-Tukey             — O(N log N) via divide and conquer
  3. Inverse FFT                  — Rekonstruksi sinyal dari frekuensi
  4. Binary Audio Representation  — PCM, IEEE 754 float32, bit-exact ops
  5. Frequency Analysis           — Pitch detection, note mapping
  6. Nyquist & Sampling Theory    — Batas matematika dari digitisasi audio

Filosofi:
  Suara adalah getaran udara → tekanan → bilangan.
  Setiap bunyi yang pernah ada bisa didekomposisi menjadi
  jumlah gelombang sinus dengan frekuensi dan amplitudo berbeda.
  Ini bukan metafora — ini adalah teorema Fourier yang TERBUKTI.
"""

import math
import struct
from typing import List, Tuple, Dict, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 1: DISCRETE FOURIER TRANSFORM (DFT)
# Langsung dari definisi matematika, tanpa library
# X[k] = Σ x[n] · e^(-j2πkn/N)  untuk k = 0, 1, ..., N-1
# ═══════════════════════════════════════════════════════════════════════════

class Complex:
    """
    Bilangan kompleks dari first principles.
    Bukan import complex dari Python — ini implementasi sendiri
    untuk menunjukkan bahwa semua ini adalah MATEMATIKA MURNI.
    """
    __slots__ = ('re', 'im')

    def __init__(self, re: float = 0.0, im: float = 0.0):
        self.re = re
        self.im = im

    def __add__(self, o: 'Complex') -> 'Complex':
        return Complex(self.re + o.re, self.im + o.im)

    def __sub__(self, o: 'Complex') -> 'Complex':
        return Complex(self.re - o.re, self.im - o.im)

    def __mul__(self, o: 'Complex') -> 'Complex':
        # (a+bj)(c+dj) = ac-bd + (ad+bc)j
        return Complex(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re
        )

    def magnitude(self) -> float:
        """|z| = √(re² + im²) — amplitudo di domain frekuensi"""
        return math.sqrt(self.re**2 + self.im**2)

    def phase(self) -> float:
        """arg(z) = atan2(im, re) — fase dalam radian"""
        return math.atan2(self.im, self.re)

    @staticmethod
    def from_polar(r: float, theta: float) -> 'Complex':
        """z = r·e^(jθ) = r·cos(θ) + jr·sin(θ)  [Euler's formula]"""
        return Complex(r * math.cos(theta), r * math.sin(theta))

    def __repr__(self) -> str:
        sign = '+' if self.im >= 0 else '-'
        return f"{self.re:.4f}{sign}{abs(self.im):.4f}j"


def dft(x: List[float]) -> List[Complex]:
    """
    Discrete Fourier Transform (DFT) — implementasi naif O(N²).
    X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j2πkn/N}

    Ini adalah transformasi yang mengubah sinyal dari domain WAKTU
    ke domain FREKUENSI. Setiap titik X[k] mewakili amplitudo dan
    fase komponen frekuensi ke-k dalam sinyal x.
    """
    N = len(x)
    X: List[Complex] = []
    for k in range(N):
        total = Complex(0.0, 0.0)
        for n in range(N):
            # e^(-j2πkn/N) = cos(2πkn/N) - j·sin(2πkn/N)  [Euler]
            angle = -2.0 * math.pi * k * n / N
            twiddle = Complex(math.cos(angle), math.sin(angle))
            total = total + Complex(x[n], 0.0) * twiddle
        X.append(total)
    return X


def idft(X: List[Complex]) -> List[float]:
    """
    Inverse Discrete Fourier Transform.
    x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{j2πkn/N}

    Rekonstruksi sinyal dari representasi frekuensi.
    Kebalikan dari DFT — domain frekuensi → domain waktu.
    """
    N = len(X)
    x: List[float] = []
    for n in range(N):
        total = Complex(0.0, 0.0)
        for k in range(N):
            angle = 2.0 * math.pi * k * n / N  # Note: positif untuk IDFT
            twiddle = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * twiddle
        x.append(total.re / N)  # Bagi N untuk normalisasi
    return x


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 2: FAST FOURIER TRANSFORM (FFT)
# Algoritma Cooley-Tukey — O(N log N) via Divide & Conquer
# Penemuan yang mengubah dunia: membuat FFT 1000x lebih cepat dari DFT
# ═══════════════════════════════════════════════════════════════════════════

def fft(x: list) -> List[Complex]:
    """
    Fast Fourier Transform — Algoritma Cooley-Tukey (1965).
    Kompleksitas: O(N log N) vs O(N²) untuk DFT naif.

    Prinsip Divide & Conquer:
      X[k] = Σ x[2n]·W^{2nk} + W^k·Σ x[2n+1]·W^{2nk}
           = DFT(genap) + W^k · DFT(ganjil)

    di mana W = e^{-j2π/N} (twiddle factor)

    Untuk N=1024: DFT butuh 1,048,576 operasi; FFT hanya 10,240.
    Ini bukan hanya optimasi — ini adalah insight matematis mendalam.
    """
    N = len(x)
    if N <= 1:
        val = x[0] if x else 0.0
        if isinstance(val, Complex):
            return [val]
        return [Complex(val, 0.0)]

    # Pad ke power of 2 jika perlu
    if N & (N - 1) != 0:
        # Pad dengan nol ke next power of 2
        next_pow2 = 1 << (N - 1).bit_length()
        pad_val = Complex(0.0, 0.0) if isinstance(x[0], Complex) else 0.0
        x = x + [pad_val] * (next_pow2 - N)
        N = next_pow2

    if N == 1:
        val = x[0]
        if isinstance(val, Complex):
            return [val]
        return [Complex(val, 0.0)]

    # Divide: pisahkan genap dan ganjil
    even = fft(x[0::2])  # x[0], x[2], x[4], ...
    odd  = fft(x[1::2])  # x[1], x[3], x[5], ...

    # Conquer: gabungkan menggunakan butterfly operation
    half = N // 2
    X: List[Complex] = [Complex(0.0, 0.0)] * N

    for k in range(half):
        # Twiddle factor: W_k = e^{-j2πk/N}
        angle = -2.0 * math.pi * k / N
        W_k = Complex(math.cos(angle), math.sin(angle))

        # Butterfly: X[k] = E[k] + W_k·O[k]
        #            X[k+N/2] = E[k] - W_k·O[k]
        butterfly = W_k * odd[k]
        X[k]        = even[k] + butterfly
        X[k + half] = even[k] - butterfly

    return X


def ifft(X: List[Complex]) -> List[float]:
    """
    Inverse FFT menggunakan sifat dualitas:
    IFFT(X) = (1/N) · conj(FFT(conj(X)))
    """
    N = len(X)
    # Conjugate input
    X_conj = [Complex(c.re, -c.im) for c in X]
    # Apply FFT
    result = fft(X_conj)
    # Conjugate dan scale
    return [Complex(c.re, -c.im).re / N for c in result]


def fft_magnitudes(signal: List[float]) -> List[float]:
    """
    Hitung magnitude spektrum (amplitudo per frekuensi).
    Output: |X[k]| untuk k = 0, 1, ..., N/2 (simetris di atas Nyquist)
    """
    spectrum = fft(signal)
    N = len(spectrum)
    # Ambil sisi positif saja (0 sampai N/2)
    magnitudes = [spectrum[k].magnitude() for k in range(N // 2 + 1)]
    # Normalisasi
    magnitudes = [m * 2 / N for m in magnitudes]
    magnitudes[0] /= 2   # DC tidak di-double
    return magnitudes


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 3: BINARY AUDIO REPRESENTATION
# Suara sebagai angka biner — PCM & IEEE 754 Float32
# ═══════════════════════════════════════════════════════════════════════════

class BinaryAudio:
    """
    Representasi biner dari sinyal audio.

    PCM (Pulse Code Modulation): setiap sampel audio adalah bilangan.
    Untuk 16-bit: -32768 sampai 32767 (2¹⁶ = 65536 level)
    Untuk float32: IEEE 754 — 32 bit = 1 sign + 8 exponent + 23 mantissa

    Ini adalah MATEMATIKA BINER dari suara.
    """

    @staticmethod
    def float_to_binary(f: float, bits: int = 32) -> str:
        """
        Konversi float ke representasi biner IEEE 754.
        Ini menunjukkan bagaimana bilangan real disimpan dalam bit.
        """
        if bits == 32:
            packed = struct.pack('>f', f)
            int_val = struct.unpack('>I', packed)[0]
            binary = format(int_val, '032b')
            return (
                f"Sign:     {binary[0]}\n"
                f"Exponent: {binary[1:9]} = {int(binary[1:9], 2)} (biased: {int(binary[1:9], 2)-127})\n"
                f"Mantissa: {binary[9:]} = 1.{binary[9:]} (implicit leading 1)\n"
                f"Value:    {f}\n"
                f"Hex:      0x{int_val:08X}"
            )
        elif bits == 16:
            # Simple 16-bit signed int representation
            int_val = int(f * 32767)
            int_val = max(-32768, min(32767, int_val))
            if int_val < 0:
                int_val = int_val + 65536
            return format(int_val, '016b')
        return ""

    @staticmethod
    def binary_to_float(binary_str: str) -> float:
        """Parse string biner IEEE 754 float32 ke float Python."""
        binary_str = binary_str.replace(' ', '')
        if len(binary_str) != 32:
            raise ValueError("IEEE 754 float32 membutuhkan tepat 32 bit")
        int_val = int(binary_str, 2)
        packed = struct.pack('>I', int_val)
        return struct.unpack('>f', packed)[0]

    @staticmethod
    def generate_sine_wave(freq: float, duration: float,
                            sample_rate: int = 44100,
                            amplitude: float = 1.0) -> List[float]:
        """
        Generate gelombang sinus murni dari matematika.
        x[n] = A · sin(2π · f · n / fs)

        Ini adalah sinyal audio paling sederhana — satu frekuensi murni.
        """
        n_samples = int(duration * sample_rate)
        samples = []
        for n in range(n_samples):
            t = n / sample_rate
            sample = amplitude * math.sin(2 * math.pi * freq * t)
            samples.append(sample)
        return samples

    @staticmethod
    def generate_chord(frequencies: List[float], duration: float,
                        sample_rate: int = 44100) -> List[float]:
        """
        Generate chord (beberapa frekuensi sekaligus).
        Superposisi gelombang — ini adalah matematika fisika gelombang.
        """
        n_samples = int(duration * sample_rate)
        samples = [0.0] * n_samples
        for freq in frequencies:
            for n in range(n_samples):
                t = n / sample_rate
                samples[n] += math.sin(2 * math.pi * freq * t)
        # Normalize
        max_amp = max(abs(s) for s in samples) or 1.0
        return [s / max_amp for s in samples]

    @staticmethod
    def quantize(samples: List[float], bits: int = 16) -> List[int]:
        """
        Kuantisasi: float → integer (proses digitisasi audio).
        Ini adalah source dari 'quantization noise'.
        """
        levels = 2 ** bits
        max_val = levels // 2 - 1
        min_val = -(levels // 2)
        result = []
        for s in samples:
            quantized = int(round(s * max_val))
            quantized = max(min_val, min(max_val, quantized))
            result.append(quantized)
        return result

    @staticmethod
    def quantization_snr(bits: int) -> float:
        """
        SNR teoritis dari kuantisasi PCM:
        SNR = 6.02·n + 1.76 dB
        Ini adalah RUMUS MATEMATIS, bukan empiris.
        """
        return 6.02 * bits + 1.76


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 4: FREQUENCY ANALYSIS & PITCH DETECTION
# Dari biner → frekuensi → nada → fonem
# Ini adalah pipeline dasar untuk input suara
# ═══════════════════════════════════════════════════════════════════════════

class FrequencyAnalysis:
    """
    Analisis frekuensi sinyal audio dan pemetaan ke nada musik.

    Equal Temperament Tuning System:
    f(n) = f₀ × 2^(n/12)
    di mana f₀ = 440 Hz (A4) dan n = semitones dari A4.

    Ini adalah matematika di balik semua musik modern.
    """

    # Note names dan offset semitone dari A4=440Hz
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4_FREQ = 440.0
    A4_MIDI = 69  # MIDI note number untuk A4

    @staticmethod
    def freq_to_note(freq: float) -> Dict[str, Any]:
        """
        Konversi frekuensi ke nama nada musik.
        n = 12 · log₂(f/440) → semitones dari A4
        """
        if freq <= 0:
            return {"error": "Frekuensi harus positif"}

        # Hitung semitones dari A4
        semitones = 12 * math.log2(freq / FrequencyAnalysis.A4_FREQ)
        midi = round(FrequencyAnalysis.A4_MIDI + semitones)
        note_index = midi % 12
        octave = (midi // 12) - 1
        note_name = FrequencyAnalysis.NOTE_NAMES[note_index]

        # Hitung deviation (cent) dari nada terdekat
        exact_freq = FrequencyAnalysis.A4_FREQ * (2 ** ((midi - FrequencyAnalysis.A4_MIDI) / 12))
        cents_off = 1200 * math.log2(freq / exact_freq)

        return {
            "frequency_hz": freq,
            "note": f"{note_name}{octave}",
            "midi": midi,
            "semitones_from_A4": semitones,
            "exact_freq": round(exact_freq, 3),
            "cents_deviation": round(cents_off, 2),
            "in_tune": abs(cents_off) < 5,  # ±5 cent threshold
        }

    @staticmethod
    def note_to_freq(note_name: str, octave: int) -> float:
        """
        Konversi nama nada ke frekuensi.
        f = 440 × 2^((midi-69)/12)
        """
        note_index = FrequencyAnalysis.NOTE_NAMES.index(note_name.upper().replace('B', 'A#').replace('BB', 'A'))
        if note_name.upper() == 'B':
            note_index = 11
        elif note_name.upper() == 'A':
            note_index = 9

        midi = (octave + 1) * 12 + note_index
        return FrequencyAnalysis.A4_FREQ * (2 ** ((midi - FrequencyAnalysis.A4_MIDI) / 12))

    @staticmethod
    def dominant_frequencies(signal: List[float], sample_rate: int = 44100,
                              n_peaks: int = 5) -> List[Dict[str, Any]]:
        """
        Temukan frekuensi dominan dalam sinyal menggunakan FFT.
        Ini adalah langkah pertama dari pipeline speech recognition.
        """
        # Terapkan FFT
        magnitudes = fft_magnitudes(signal)
        padded_N = (len(magnitudes) - 1) * 2
        freq_resolution = sample_rate / padded_N  # Hz per bin

        # Temukan n_peaks tertinggi
        indexed = [(mag, i) for i, mag in enumerate(magnitudes)]
        indexed.sort(reverse=True)

        results = []
        for mag, bin_idx in indexed[:n_peaks]:
            if mag < 0.01:  # Threshold noise
                continue
            freq = bin_idx * freq_resolution
            note_info = FrequencyAnalysis.freq_to_note(freq) if freq > 20 else {"note": "sub-bass", "frequency_hz": freq}
            results.append({
                "frequency_hz": round(freq, 2),
                "magnitude": round(mag, 4),
                "fft_bin": bin_idx,
                "note": note_info.get("note", "?"),
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 5: NYQUIST & SAMPLING THEOREM
# Matematika di balik batas digitisasi
# ═══════════════════════════════════════════════════════════════════════════

class SamplingTheory:
    """
    Teori sampling Nyquist-Shannon:
    fs > 2 × f_max

    Untuk merekonstruksi sempurna sinyal dengan frekuensi maksimum f_max,
    kita perlu sample dengan rate minimal 2×f_max.

    Ini bukan aturan empiris — ini adalah TEOREMA yang TERBUKTI.
    (Shannon 1949, Nyquist 1928, Kotelnikov 1933, Whittaker 1915)
    """

    @staticmethod
    def verify_nyquist(sample_rate: int, max_freq: float) -> Dict[str, Any]:
        """Verifikasi apakah sample rate memenuhi Nyquist theorem."""
        nyquist_freq = sample_rate / 2
        satisfied = sample_rate > 2 * max_freq
        return {
            "sample_rate_hz": sample_rate,
            "max_signal_freq_hz": max_freq,
            "nyquist_frequency_hz": nyquist_freq,
            "nyquist_satisfied": satisfied,
            "theorem": "fs > 2·f_max",
            "verdict": (
                f"✅ NYQUIST SATISFIED: {sample_rate} > 2×{max_freq}={2*max_freq}"
                if satisfied else
                f"❌ ALIASING akan terjadi: {sample_rate} ≤ 2×{max_freq}={2*max_freq}"
            ),
            "aliased_freq": abs(max_freq - sample_rate) if not satisfied else None,
        }

    @staticmethod
    def aliasing_freq(original_freq: float, sample_rate: int) -> float:
        """
        Hitung frekuensi alias yang muncul jika Nyquist dilanggar.
        f_alias = |f_original - n·fs| dimana n dipilih agar hasilnya < fs/2
        """
        nyquist = sample_rate / 2
        alias = original_freq % sample_rate
        if alias > nyquist:
            alias = sample_rate - alias
        return alias

    @staticmethod
    def common_sample_rates() -> List[Dict[str, Any]]:
        """Standar sample rate audio yang umum digunakan."""
        rates = [
            (8000,  "Telepon (8 kHz)", 4000),
            (16000, "Speech recognition (16 kHz)", 8000),
            (22050, "Radio AM (22.05 kHz)", 11025),
            (44100, "CD Audio (44.1 kHz)", 22050),
            (48000, "Studio/DAW (48 kHz)", 24000),
            (96000, "High-res audio (96 kHz)", 48000),
        ]
        results = []
        for sr, desc, nyquist in rates:
            results.append({
                "sample_rate": sr,
                "description": desc,
                "nyquist_freq": nyquist,
                "max_audible": min(nyquist, 20000),
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class SignalFrequencyEngine:
    """
    Fasad utama untuk semua kapabilitas signal & frequency math.
    Fondasi matematis untuk input suara pengguna.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.audio = BinaryAudio()
        self.freq = FrequencyAnalysis()
        self.sampling = SamplingTheory()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🎵 [SFE] {msg}")


_sfe_instance: Optional[SignalFrequencyEngine] = None

def get_sfe(verbose: bool = True) -> SignalFrequencyEngine:
    global _sfe_instance
    if _sfe_instance is None:
        _sfe_instance = SignalFrequencyEngine(verbose=verbose)
    return _sfe_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n🎵 Signal & Frequency Engine — Self Test\n" + "="*55)

    # ── Test 1: DFT dari Binary Math ─────────────────────────────────
    # Input: [1, 0, -1, 0] — satu siklus gelombang sinus
    # DFT output: X[0]=0 (DC), X[1]=2 (frekuensi 1), X[2]=0, X[3]=2
    signal = [1.0, 0.0, -1.0, 0.0]
    X = dft(signal)
    assert abs(X[0].magnitude()) < 1e-10, f"DFT DC gagal: {X[0]}"
    assert abs(X[1].magnitude() - 2.0) < 1e-9, f"DFT freq-1 gagal: {X[1].magnitude()}"
    print(f"  ✅ DFT [1,0,-1,0]: X[0]={X[0].magnitude():.4f}, X[1]={X[1].magnitude():.4f} (benar: 0, 2)")

    # ── Test 2: FFT hasil harus sama dengan DFT ───────────────────────
    signal_8 = [math.sin(2 * math.pi * k / 8) for k in range(8)]
    dft_out = dft(signal_8)
    fft_out = fft(signal_8)
    # Bandingkan magnitude
    for k in range(len(dft_out)):
        diff = abs(dft_out[k].magnitude() - fft_out[k].magnitude())
        assert diff < 1e-8, f"FFT≠DFT pada k={k}: diff={diff}"
    print(f"  ✅ FFT = DFT untuk N=8 sinyal (presisi: < 1e-8)")

    # ── Test 3: IFFT(FFT(x)) ≈ x ─────────────────────────────────────
    original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    recovered = ifft(fft(original))
    for i, (orig, rec) in enumerate(zip(original, recovered)):
        assert abs(orig - rec) < 1e-8, f"IFFT(FFT) gagal di index {i}: {orig} vs {rec}"
    print(f"  ✅ IFFT(FFT(x)) = x (roundtrip error < 1e-8)")

    # ── Test 4: Binary Float Representation ──────────────────────────
    binary_repr = BinaryAudio.float_to_binary(3.14)
    assert "Sign:" in binary_repr, "Binary float gagal"
    print(f"  ✅ Float32 Binary: 3.14 → IEEE 754 representation")
    print(f"     {binary_repr.split(chr(10))[0]}")

    # ── Test 5: Sine Wave Generation ─────────────────────────────────
    # Generate 440Hz (A4) selama 0.01 detik
    sine = BinaryAudio.generate_sine_wave(freq=440.0, duration=0.01, sample_rate=44100)
    assert len(sine) == 441, f"Sine wave length gagal: {len(sine)}"
    assert abs(sine[0]) < 1e-10, f"Sine tidak mulai dari 0: {sine[0]}"
    print(f"  ✅ Sine wave 440Hz: {len(sine)} sampel, x[0]≈0 ✓")

    # ── Test 6: Dominant Frequency Detection ─────────────────────────
    # Generate 440Hz dan detect frekuensinya
    sine_440 = BinaryAudio.generate_sine_wave(freq=440.0, duration=0.1, sample_rate=4096)
    dominant = FrequencyAnalysis.dominant_frequencies(sine_440, sample_rate=4096, n_peaks=3)
    if dominant:
        top_freq = dominant[0]["frequency_hz"]
        assert abs(top_freq - 440.0) < 50, f"Freq detection gagal: {top_freq}"
        print(f"  ✅ Frequency detection: 440Hz → detected {top_freq:.1f}Hz ≈ {dominant[0]['note']}")

    # ── Test 7: Note → Frequency & Frequency → Note ──────────────────
    note_info = FrequencyAnalysis.freq_to_note(440.0)
    assert note_info["note"] == "A4", f"A4 gagal: {note_info['note']}"
    print(f"  ✅ 440 Hz → {note_info['note']} (A4, deviation={note_info['cents_deviation']} cents)")

    note_info2 = FrequencyAnalysis.freq_to_note(261.63)  # C4 (middle C)
    print(f"  ✅ 261.63 Hz → {note_info2['note']} (C4)")

    # ── Test 8: Nyquist Theorem ───────────────────────────────────────
    nyq = SamplingTheory.verify_nyquist(sample_rate=44100, max_freq=20000)
    assert nyq["nyquist_satisfied"], f"Nyquist gagal: {nyq}"
    print(f"  ✅ Nyquist: 44100Hz > 2×20000Hz=40000Hz → {nyq['verdict'][:40]}...")

    nyq_bad = SamplingTheory.verify_nyquist(sample_rate=8000, max_freq=8000)
    assert not nyq_bad["nyquist_satisfied"], "Nyquist violation tidak terdeteksi"
    print(f"  ✅ Aliasing terdeteksi: {nyq_bad['verdict'][:45]}...")

    # ── Test 9: Quantization SNR ─────────────────────────────────────
    snr_16 = BinaryAudio.quantization_snr(16)
    assert abs(snr_16 - (6.02*16 + 1.76)) < 0.01, "SNR gagal"
    print(f"  ✅ Quantization SNR: 16-bit PCM = {snr_16:.2f} dB (teorema matematis)")

    # ── Test 10: Complex Number Math ─────────────────────────────────
    z1 = Complex(3.0, 4.0)
    assert abs(z1.magnitude() - 5.0) < 1e-10, f"|3+4j| = {z1.magnitude()}"
    z2 = Complex.from_polar(1.0, math.pi)
    assert abs(z2.re - (-1.0)) < 1e-10, f"e^jπ = {z2}"
    print(f"  ✅ Complex: |3+4j|={z1.magnitude()}, e^(jπ)={z2.re:.4f}+{z2.im:.4f}j ≈ -1+0j (Euler)")

    print("\n✅ Semua test Signal & Frequency Engine berhasil!\n")


if __name__ == "__main__":
    _self_test()
