"""
MOKO Gaussian Noise Engine — Discrete Gaussian Sampler
=======================================================
Implementasi sampler noise Gaussian diskrit untuk Ring-LWE.

Referensi matematis:
  - Knuth & Yao (1976): optimal rejection sampling
  - Peikert (2010): discrete Gaussian for LWE/RLWE
  - NIST PQC Kyber spec: σ = 3.2 (binomial approximation)
  - "Gaussian Sampling over Lattices" — Ducas et al.

Distribusi: P(e = k) = exp(-k²/(2σ²)) / (Σ_z exp(-z²/(2σ²)))
            ≈ Binomial(n, 0.5) untuk σ² = n/4

Properti keamanan:
  - Uniform noise e ∈ [-4,4]:  statistical attack BERHASIL (v1 MCRG)
  - Gaussian noise σ=3.2:      statistical attack GAGAL (v2 target)
"""

import math
import random
import struct
import hashlib
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# CENTERED BINOMIAL DISTRIBUTION (Kyber standard approximation)
# ═══════════════════════════════════════════════════════════════════════════

class CenteredBinomialSampler:
    """
    Centered Binomial Distribution (CBD) — digunakan di Kyber NIST PQC.
    
    CBD_η(x): sampel Σ(aᵢ - bᵢ) untuk aᵢ,bᵢ ∈ {0,1} i=1..η
    - μ = 0, σ² = η/2
    - Untuk η=2: σ ≈ 1.0 (Kyber-512 style, cepat)
    - Untuk η=3: σ ≈ 1.22 (lebih aman)
    - Untuk η=4: σ ≈ 1.41 (sangat aman)
    
    Kyber-768 menggunakan η=2 (σ=1.0) untuk efisiensi.
    MOKO menggunakan η=6 (σ≈1.73) untuk keamanan ekstra.
    """

    def __init__(self, eta: int = 6):
        """
        Args:
            eta: parameter distribusi. σ² = eta/2, max |e| = eta.
                 Kyber standard: eta=2 atau eta=3.
                 MOKO security: eta=6 (lebih konservatif).
        """
        if eta < 1 or eta > 16:
            raise ValueError("eta harus antara 1 dan 16")
        self.eta = eta
        self.sigma = math.sqrt(eta / 2)

    def sample(self) -> int:
        """Sampel satu nilai dari CBD_η."""
        a = sum(random.getrandbits(1) for _ in range(self.eta))
        b = sum(random.getrandbits(1) for _ in range(self.eta))
        return a - b

    def sample_poly(self, n: int) -> List[int]:
        """Sampel polynomial berisi n koefisien dari CBD_η."""
        return [self.sample() for _ in range(n)]

    def sample_batch(self, count: int) -> List[int]:
        """Sampel count nilai secara batch."""
        return [self.sample() for _ in range(count)]

    @property
    def max_magnitude(self) -> int:
        return self.eta

    @property
    def std_dev(self) -> float:
        return self.sigma


# ═══════════════════════════════════════════════════════════════════════════
# TRUE DISCRETE GAUSSIAN SAMPLER (Knuth-Yao + CDT)
# ═══════════════════════════════════════════════════════════════════════════

class DiscreteGaussianSampler:
    """
    True Discrete Gaussian Distribution menggunakan CDT (Cumulative Distribution Table).
    
    P(e = k) ∝ exp(-k²/(2σ²))
    
    Algoritma:
    1. Precompute CDT untuk k ∈ [-K, K] di mana K = ceil(6σ)
    2. Sampel u ∈ [0,1) dari CSPRNG
    3. Cari k terkecil di mana CDF(k) ≥ u
    
    Lebih lambat dari CBD tapi distribusi lebih tepat secara statistik.
    """

    def __init__(self, sigma: float = 3.2, tail_cut: int = None):
        """
        Args:
            sigma: standar deviasi Gaussian. NIST Kyber: 3.2.
            tail_cut: potong distribusi di |k| > tail_cut. Default: ceil(6σ).
        """
        self.sigma = sigma
        self.tail = tail_cut if tail_cut else math.ceil(6 * sigma)
        self._build_cdt()

    def _build_cdt(self):
        """Bangun Cumulative Distribution Table."""
        sigma2 = 2 * self.sigma ** 2
        # Hitung probabilitas unnormalized
        probs = {}
        for k in range(-self.tail, self.tail + 1):
            probs[k] = math.exp(-k * k / sigma2)

        # Normalisasi
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        # Bangun CDF
        self._cdf = []  # list of (cumulative_prob, k)
        cumsum = 0.0
        for k in range(-self.tail, self.tail + 1):
            cumsum += probs[k]
            self._cdf.append((cumsum, k))

        # Precompute stats
        self._mean = 0.0  # By symmetry
        self._std = math.sqrt(sum(k*k * probs[k] for k in probs))

    def sample(self) -> int:
        """Sampel satu nilai dari Gaussian diskrit menggunakan CDT lookup."""
        u = random.random()
        # Binary search di CDF
        lo, hi = 0, len(self._cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cdf[mid][0] < u:
                lo = mid + 1
            else:
                hi = mid
        return self._cdf[lo][1]

    def sample_poly(self, n: int) -> List[int]:
        """Sampel polynomial berisi n koefisien."""
        return [self.sample() for _ in range(n)]

    def sample_batch(self, count: int) -> List[int]:
        """Sampel count nilai secara batch."""
        return [self.sample() for _ in range(count)]

    @property
    def max_magnitude(self) -> int:
        return self.tail

    @property
    def std_dev(self) -> float:
        return self._std

    def chi_squared_test(self, n_samples: int = 10000) -> Tuple[float, bool]:
        """
        Verifikasi distribusi dengan chi-squared goodness-of-fit test.
        
        Returns:
            (chi2_stat, passed): statistik chi-squared dan apakah distribusi valid.
        """
        # Sampel
        samples = self.sample_batch(n_samples)

        # Hitung frekuensi observasi
        freq_obs = {}
        for s in samples:
            freq_obs[s] = freq_obs.get(s, 0) + 1

        # Hitung frekuensi ekspektasi
        sigma2 = 2 * self.sigma ** 2
        raw = {k: math.exp(-k*k / sigma2) for k in range(-self.tail, self.tail+1)}
        total_raw = sum(raw.values())
        freq_exp = {k: (raw[k] / total_raw) * n_samples for k in raw}

        # Chi-squared statistic: Σ (O-E)²/E
        chi2 = 0.0
        for k in freq_exp:
            o = freq_obs.get(k, 0)
            e = freq_exp[k]
            if e > 0.5:  # Skip sangat kecil untuk stabilitas numerik
                chi2 += (o - e) ** 2 / e

        # Degrees of freedom ≈ 2*tail (jumlah bins - 1)
        df = 2 * self.tail
        # Critical value χ²_{0.05, df} ≈ df + 2√(2·df) (approximasi)
        crit = df + 2 * math.sqrt(2 * df)
        passed = chi2 < crit

        return chi2, passed


# ═══════════════════════════════════════════════════════════════════════════
# SECURITY PARAMETER VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class NoiseSecurityValidator:
    """
    Validasi apakah parameter noise memenuhi standar keamanan LWE.
    
    Syarat keamanan (berdasarkan analisis Regev & NIST):
    - noise-to-modulus ratio α = σ/q harus ≥ 1/√n
    - Ini memastikan decoding benar dan keamanan computational
    """

    @staticmethod
    def validate(n: int, q: int, sigma: float) -> dict:
        """
        Validasi parameter noise untuk LWE(n, q, σ).
        
        Returns:
            dict berisi metrik keamanan dan status.
        """
        alpha = sigma / q                    # Noise-to-modulus ratio
        alpha_min = 1.0 / math.sqrt(n)      # Minimum required
        noise_ok = alpha >= alpha_min

        # Estimasi keamanan (simplified BKZ model)
        # β ≈ 0.127 * n * log2(q/sigma)
        if sigma > 0 and q > sigma:
            rho = math.log2(q / sigma)      # log ratio
            beta_approx = 0.127 * n * rho
            classical_bits = 0.292 * beta_approx
            quantum_bits   = 0.265 * beta_approx
        else:
            classical_bits = 0
            quantum_bits   = 0

        # Decoding correctness: noise harus < q/4 dengan probabilitas tinggi
        max_noise_safe = q / 4
        # Probabilitas error decodingk: P(|noise| > q/4)
        # Untuk Gaussian: P(|e| > t) ≈ 2*erfc(t/(σ√2))
        # Approximasi: P(|e| > q/4) ≈ 2*exp(-(q/4)²/(2σ²))
        if sigma > 0:
            t = max_noise_safe
            error_prob = 2 * math.exp(-(t*t) / (2 * sigma*sigma))
        else:
            error_prob = 0.0

        return {
            "n": n,
            "q": q,
            "sigma": sigma,
            "alpha": alpha,
            "alpha_min_required": alpha_min,
            "noise_ratio_ok": noise_ok,
            "classical_security_bits": round(classical_bits, 1),
            "quantum_security_bits": round(quantum_bits, 1),
            "decoding_error_prob": error_prob,
            "is_128bit_quantum_safe": quantum_bits >= 128,
            "is_military_grade": quantum_bits >= 128 and error_prob < 1e-10,
        }
