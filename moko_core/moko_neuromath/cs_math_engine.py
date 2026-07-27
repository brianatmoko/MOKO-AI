"""
MOKO CS Math Engine — Matematika Ilmu Komputer
================================================
Mencakup:
  1. Kompleksitas Algoritma — Big-O, Big-Θ, Master Theorem, recurrence
  2. Teori Graf            — Shortest path cost, MST weight, spanning trees
  3. Kriptografi Matematika — RSA keypair, Diffie-Hellman, hash probability
  4. Memori & Storage      — Bit/Byte konversi, RAID parity, cache math
  5. Jaringan Komputer     — Bandwidth-delay product, throughput, jitter
  6. Probabilitas Hashing  — Birthday problem, collision probability
"""

import math
from typing import Dict, List, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════
# 1. KOMPLEKSITAS ALGORITMA
# ═══════════════════════════════════════════════════════════════════════════

class AlgorithmComplexity:
    """Analisis Big-O, Master Theorem, dan operasi kompleksitas."""

    # Kamus Big-O → kompleksitas nama
    BIG_O_NAMES = {
        "O(1)":         "Constant — operasi langsung (array access, hash lookup)",
        "O(log n)":     "Logarithmic — binary search, balanced BST",
        "O(n)":         "Linear — linear scan, single loop",
        "O(n log n)":   "Linearithmic — merge sort, heap sort, FFT",
        "O(n²)":        "Quadratic — bubble sort, nested loops, naive matrix mult",
        "O(n³)":        "Cubic — naive matrix multiplication 3 loops",
        "O(2^n)":       "Exponential — brute-force subsets, recursive Fibonacci naive",
        "O(n!)":        "Factorial — permutasi semua, Traveling Salesman brute-force",
    }

    @staticmethod
    def master_theorem(a: int, b: int, k: int,
                       p: float = 0.0) -> Dict[str, str]:
        """
        Master Theorem untuk T(n) = a·T(n/b) + Θ(n^k · log^p(n)).

        Case 1: log_b(a) > k → T(n) = Θ(n^log_b(a))
        Case 2: log_b(a) = k → T(n) = Θ(n^k · log^(p+1)(n))
        Case 3: log_b(a) < k → T(n) = Θ(n^k · log^p(n))
        """
        if a <= 0 or b <= 1:
            return {"error": "a harus > 0 dan b harus > 1"}

        log_b_a = math.log(a) / math.log(b)
        recurrence = f"T(n) = {a}·T(n/{b}) + Θ(n^{k}" + (f"·log^{p}(n))" if p else ")")

        if log_b_a > k + 1e-9:
            complexity = f"Θ(n^{log_b_a:.4g})"
            case = "Case 1"
            explanation = f"log_{b}({a}) = {log_b_a:.4g} > {k} → dominated oleh subproblem"
        elif abs(log_b_a - k) < 1e-9:
            if p == 0:
                complexity = f"Θ(n^{k}·log(n))"
            elif p == -1:
                complexity = f"Θ(n^{k}·log(log(n)))"
            elif p < -1:
                complexity = f"Θ(n^{k})"
            else:
                complexity = f"Θ(n^{k}·log^{p+1}(n))"
            case = "Case 2"
            explanation = f"log_{b}({a}) = {log_b_a:.4g} = {k} → balanced"
        else:
            complexity = f"Θ(n^{k}" + (f"·log^{p}(n))" if p else ")")
            case = "Case 3"
            explanation = f"log_{b}({a}) = {log_b_a:.4g} < {k} → dominated oleh work function"

        return {
            "recurrence": recurrence,
            "log_b_a": round(log_b_a, 6),
            "case": case,
            "complexity": complexity,
            "explanation": explanation,
        }

    @staticmethod
    def compare_growth(n: int) -> Dict[str, float]:
        """Bandingkan nilai aktual berbagai kompleksitas untuk n tertentu."""
        safe_log = math.log2(n) if n > 0 else 0
        return {
            "n": n,
            "O(1)": 1,
            "O(log n)": round(safe_log, 2),
            "O(n)": n,
            "O(n log n)": round(n * safe_log, 2),
            "O(n²)": n**2,
            "O(n³)": n**3,
            "O(2^n)": f"2^{n} (sangat besar)" if n > 30 else 2**n,
        }

    @staticmethod
    def fibonacci_naive_calls(n: int) -> int:
        """Jumlah pemanggilan rekursif Fibonacci naif = 2^n - 1 (approx)."""
        # Exact: T(n) = T(n-1) + T(n-2) + 1
        # Untuk n besar: ≈ φ^n, dimana φ = (1+√5)/2
        phi = (1 + math.sqrt(5)) / 2
        return round(phi ** n / math.sqrt(5))


# ═══════════════════════════════════════════════════════════════════════════
# 2. KRIPTOGRAFI MATEMATIKA
# ═══════════════════════════════════════════════════════════════════════════

class CryptoMath:
    """Matematika di balik kriptografi modern."""

    @staticmethod
    def rsa_keypair(p: int, q: int, e: int = 65537) -> Dict[str, Any]:
        """
        Generate RSA keypair dari dua bilangan prima p dan q.
        n = p×q, φ(n) = (p-1)(q-1), d = e^(-1) mod φ(n)
        """
        # Validasi input
        def is_prime_simple(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True

        if not is_prime_simple(p): return {"error": f"{p} bukan bilangan prima"}
        if not is_prime_simple(q): return {"error": f"{q} bukan bilangan prima"}
        if p == q: return {"error": "p dan q harus berbeda"}

        n = p * q
        phi_n = (p - 1) * (q - 1)

        # Hitung d = e^(-1) mod φ(n)
        def extended_gcd(a, b):
            if a == 0: return b, 0, 1
            g, x, y = extended_gcd(b % a, a)
            return g, y - (b // a) * x, x

        g, _, _ = extended_gcd(e, phi_n)
        if g != 1:
            return {"error": f"e={e} tidak relatif prima dengan φ(n)={phi_n}"}

        _, x, _ = extended_gcd(e, phi_n)
        d = x % phi_n

        return {
            "p": p, "q": q,
            "n": n,
            "phi_n": phi_n,
            "e": e,
            "d": d,
            "public_key": f"({e}, {n})",
            "private_key": f"({d}, {n})",
            "key_bits": n.bit_length(),
        }

    @staticmethod
    def rsa_encrypt(m: int, e: int, n: int) -> int:
        """RSA enkripsi: C = m^e mod n."""
        return pow(m, e, n)

    @staticmethod
    def rsa_decrypt(c: int, d: int, n: int) -> int:
        """RSA dekripsi: M = c^d mod n."""
        return pow(c, d, n)

    @staticmethod
    def diffie_hellman(p: int, g: int, a: int, b: int) -> Dict[str, int]:
        """
        Diffie-Hellman key exchange.
        A = g^a mod p, B = g^b mod p
        Shared secret = A^b mod p = B^a mod p
        """
        A = pow(g, a, p)
        B = pow(g, b, p)
        shared_from_A = pow(B, a, p)
        shared_from_B = pow(A, b, p)
        assert shared_from_A == shared_from_B, "DH kunci tidak cocok"
        return {
            "public_A": A,
            "public_B": B,
            "shared_secret": shared_from_A,
            "secure": shared_from_A == shared_from_B,
        }

    @staticmethod
    def birthday_attack_probability(hash_bits: int, n_attempts: int) -> float:
        """
        Probabilitas collision setelah n_attempts (Birthday Problem).
        P ≈ 1 - e^(-n²/(2×2^bits))
        """
        space = 2 ** hash_bits
        exp_val = -(n_attempts ** 2) / (2 * space)
        return 1 - math.exp(exp_val)

    @staticmethod
    def hash_space(hash_bits: int) -> Dict[str, str]:
        """Ukuran ruang hash untuk berbagai panjang bit."""
        space = 2 ** hash_bits
        return {
            "hash_bits": hash_bits,
            "total_values": f"2^{hash_bits}",
            "hex_chars": hash_bits // 4,
            "birthday_bound": f"≈ 2^{hash_bits//2} (akar kuadrat ruang)",
            "approximate_decimal_digits": round(hash_bits * math.log10(2)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. MEMORI & STORAGE
# ═══════════════════════════════════════════════════════════════════════════

class MemoryMath:
    """Konversi unit penyimpanan, RAID, dan cache math."""

    @staticmethod
    def storage_convert(value: float, from_unit: str, to_unit: str) -> float:
        """Konversi antar satuan storage (SI dan binary)."""
        units_in_bytes = {
            "bit": 0.125, "B": 1, "KB": 1000, "KiB": 1024,
            "MB": 1e6, "MiB": 1024**2, "GB": 1e9, "GiB": 1024**3,
            "TB": 1e12, "TiB": 1024**4, "PB": 1e15,
        }
        fb = units_in_bytes.get(from_unit)
        tb = units_in_bytes.get(to_unit)
        if fb is None or tb is None:
            raise ValueError(f"Satuan tidak dikenal: {from_unit} atau {to_unit}")
        return value * fb / tb

    @staticmethod
    def raid_overhead(total_drives: int, raid_level: int,
                       drive_size_GB: float) -> Dict[str, float]:
        """Kalkulasi overhead dan kapasitas RAID."""
        if raid_level == 0:
            usable = total_drives * drive_size_GB
            overhead = 0
        elif raid_level == 1:
            usable = drive_size_GB  # Mirror — hanya 1 drive usable
            overhead = (total_drives - 1) * drive_size_GB
        elif raid_level == 5:
            usable = (total_drives - 1) * drive_size_GB
            overhead = drive_size_GB  # 1 drive parity
        elif raid_level == 6:
            usable = (total_drives - 2) * drive_size_GB
            overhead = 2 * drive_size_GB  # 2 drive parity
        elif raid_level == 10:
            usable = (total_drives // 2) * drive_size_GB
            overhead = (total_drives // 2) * drive_size_GB
        else:
            return {"error": f"RAID {raid_level} tidak didukung"}

        efficiency = (usable / (total_drives * drive_size_GB)) * 100
        return {
            "raid_level": raid_level,
            "total_drives": total_drives,
            "usable_GB": usable,
            "overhead_GB": overhead,
            "efficiency_pct": efficiency,
        }

    @staticmethod
    def cache_performance(hit_rate: float, cache_time_ns: float,
                           mem_time_ns: float) -> Dict[str, float]:
        """
        AMAT (Average Memory Access Time).
        AMAT = hit_rate × t_cache + (1 - hit_rate) × t_mem
        """
        amat = hit_rate * cache_time_ns + (1 - hit_rate) * mem_time_ns
        speedup = mem_time_ns / amat
        return {
            "AMAT_ns": amat,
            "speedup_vs_no_cache": speedup,
            "hit_rate_pct": hit_rate * 100,
            "miss_rate_pct": (1 - hit_rate) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. JARINGAN KOMPUTER
# ═══════════════════════════════════════════════════════════════════════════

class NetworkMath:
    """Kalkulasi bandwidth, latency, dan performa jaringan."""

    @staticmethod
    def bandwidth_delay_product(bandwidth_bps: float, rtt_s: float) -> Dict[str, float]:
        """BDP = Bandwidth × RTT — jumlah data 'in flight' [bits]."""
        bdp_bits = bandwidth_bps * rtt_s
        return {
            "BDP_bits": bdp_bits,
            "BDP_bytes": bdp_bits / 8,
            "BDP_KB": bdp_bits / 8 / 1000,
        }

    @staticmethod
    def transfer_time(file_size_MB: float, bandwidth_mbps: float,
                       latency_ms: float = 0) -> Dict[str, float]:
        """Waktu transfer file."""
        transfer_s = (file_size_MB * 8) / bandwidth_mbps
        total_s = transfer_s + latency_ms / 1000
        return {
            "transfer_time_s": transfer_s,
            "total_with_latency_s": total_s,
            "effective_throughput_mbps": (file_size_MB * 8) / total_s,
        }

    @staticmethod
    def packet_loss_impact(goodput_mbps: float, loss_rate: float) -> float:
        """Goodput dengan packet loss: throughput_efektif ≈ goodput × (1 - loss)."""
        return goodput_mbps * (1 - loss_rate)

    @staticmethod
    def jitter_impact(base_latency_ms: float, jitter_ms: float,
                       buffer_ms: float = 50) -> Dict[str, Any]:
        """Analisis dampak jitter untuk streaming/VoIP."""
        max_latency = base_latency_ms + jitter_ms
        playable = max_latency <= buffer_ms
        return {
            "max_latency_ms": max_latency,
            "buffer_ms": buffer_ms,
            "playable": playable,
            "verdict": "OK untuk VoIP/streaming" if playable else "Buffer overflow — kualitas buruk",
        }

    @staticmethod
    def subnet_hosts(prefix_len: int) -> Dict[str, int]:
        """Kalkulasi jumlah host dan alamat dari subnet mask."""
        host_bits = 32 - prefix_len
        total = 2 ** host_bits
        usable = max(0, total - 2)  # Kurangi network & broadcast
        return {
            "prefix": f"/{prefix_len}",
            "total_addresses": total,
            "usable_hosts": usable,
            "subnet_mask_bits": prefix_len,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class CSMathEngine:
    """Fasad utama untuk matematika ilmu komputer."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.complexity = AlgorithmComplexity()
        self.crypto = CryptoMath()
        self.memory = MemoryMath()
        self.network = NetworkMath()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  💻 [CSMath] {msg}")


_cs_instance = None

def get_cs_math_engine(verbose: bool = True) -> CSMathEngine:
    global _cs_instance
    if _cs_instance is None:
        _cs_instance = CSMathEngine(verbose=verbose)
    return _cs_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n💻 CS Math Engine — Self Test\n" + "="*55)

    # Master Theorem
    # T(n) = 2T(n/2) + n → O(n log n) (merge sort)
    mt = AlgorithmComplexity.master_theorem(a=2, b=2, k=1)
    assert "n log" in mt["complexity"] or "log" in mt["complexity"], f"Master Theorem gagal: {mt}"
    print(f"  ✅ Master Theorem T(n)=2T(n/2)+n → {mt['complexity']} ({mt['case']})")

    # Master Theorem Case 1: T(n) = 8T(n/2) + n²
    mt2 = AlgorithmComplexity.master_theorem(a=8, b=2, k=2)
    assert "n^3" in mt2["complexity"], f"Master Theorem Case 1 gagal: {mt2}"
    print(f"  ✅ Master Theorem T(n)=8T(n/2)+n² → {mt2['complexity']} ({mt2['case']})")

    # Growth comparison
    growth = AlgorithmComplexity.compare_growth(n=16)
    assert growth["O(log n)"] == 4.0, "Growth compare gagal"
    print(f"  ✅ Growth @n=16: O(log n)=4, O(n)=16, O(n²)=256, O(n log n)=64")

    # RSA
    rsa = CryptoMath.rsa_keypair(p=61, q=53)
    assert rsa["n"] == 3233, f"RSA n gagal: {rsa}"
    assert rsa["phi_n"] == 3120, f"RSA phi gagal"
    # Test encrypt/decrypt
    m = 42
    c = CryptoMath.rsa_encrypt(m, rsa["e"], rsa["n"])
    m2 = CryptoMath.rsa_decrypt(c, rsa["d"], rsa["n"])
    assert m == m2, f"RSA encrypt/decrypt gagal: {m} != {m2}"
    print(f"  ✅ RSA(61,53): n={rsa['n']}, φ(n)={rsa['phi_n']}, d={rsa['d']} | Encrypt→Decrypt: {m}✓")

    # Diffie-Hellman
    dh = CryptoMath.diffie_hellman(p=23, g=5, a=6, b=15)
    assert dh["secure"], "DH gagal"
    print(f"  ✅ Diffie-Hellman(p=23,g=5): shared_secret={dh['shared_secret']}")

    # Memory
    gb = MemoryMath.storage_convert(1, "GiB", "MB")
    assert abs(gb - 1073.741824) < 0.001, f"Storage convert gagal: {gb}"
    print(f"  ✅ Storage: 1 GiB = {gb:.3f} MB")

    raid = MemoryMath.raid_overhead(total_drives=5, raid_level=5, drive_size_GB=1000)
    assert raid["usable_GB"] == 4000, f"RAID5 gagal: {raid}"
    print(f"  ✅ RAID5 (5×1TB): usable={raid['usable_GB']/1000:.0f} TB, efisiensi={raid['efficiency_pct']:.0f}%")

    # Network
    bdp = NetworkMath.bandwidth_delay_product(bandwidth_bps=1e9, rtt_s=0.1)
    assert abs(bdp["BDP_bits"] - 1e8) < 1, "BDP gagal"
    print(f"  ✅ BDP: 1Gbps × 100ms RTT = {bdp['BDP_KB']:.0f} KB in-flight")

    subnet = NetworkMath.subnet_hosts(prefix_len=24)
    assert subnet["usable_hosts"] == 254, "Subnet gagal"
    print(f"  ✅ Subnet /24: {subnet['total_addresses']} total, {subnet['usable_hosts']} usable hosts")

    print("\n✅ Semua test CS Math Engine berhasil!\n")


if __name__ == "__main__":
    _self_test()
