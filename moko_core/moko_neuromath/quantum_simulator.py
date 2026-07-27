"""
MOKO Quantum Computing Simulator
=================================
Simulator Quantum Register murni Python (berbasis state-vector kompleks)
untuk mendemonstrasikan fenomena superposisi, entanglement (keterikatan),
gerbang logika kuantum, dan cara kerja Shor's Algorithm (pemecah RSA).
"""

import math
import random
import cmath
from typing import List, Tuple, Dict, Any


class QuantumRegister:
    """
    Simulator Register Qubit dengan representasi State Vector Kompleks.
    Mendukung superposisi penuh dan entanglement untuk N qubit.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.num_states = 1 << num_qubits  # 2^N states
        
        # State vector diinisialisasi ke basis ground state |00...0>
        # state_vector[0] = 1.0 (amplitudo probabilitas 100% untuk basis 0)
        self.state_vector = [complex(0.0, 0.0)] * self.num_states
        self.state_vector[0] = complex(1.0, 0.0)

    def apply_gate_1(self, matrix: List[List[complex]], target_qubit: int):
        """Terapkan gerbang 1-qubit (matriks 2x2) pada qubit target."""
        new_vector = [complex(0.0, 0.0)] * self.num_states
        
        # Iterasi seluruh basis state vector
        for state in range(self.num_states):
            amplitude = self.state_vector[state]
            if abs(amplitude) < 1e-9:
                continue
                
            # Dapatkan nilai bit (0 atau 1) dari target_qubit pada state saat ini
            bit_val = (state >> target_qubit) & 1
            
            # Hitung kontribusi gerbang kuantum ke dua state basis baru
            # state_0: target qubit bernilai 0
            # state_1: target qubit bernilai 1
            state_0 = state & ~(1 << target_qubit)
            state_1 = state | (1 << target_qubit)
            
            if bit_val == 0:
                new_vector[state_0] += amplitude * matrix[0][0]
                new_vector[state_1] += amplitude * matrix[1][0]
            else:
                new_vector[state_0] += amplitude * matrix[0][1]
                new_vector[state_1] += amplitude * matrix[1][1]
                
        self.state_vector = new_vector
        self._normalize()

    def apply_hadamard(self, target_qubit: int):
        """Terapkan gerbang Hadamard (H) -> Membuat Qubit Superposisi."""
        s = 1.0 / math.sqrt(2.0)
        h_matrix = [
            [complex(s, 0.0), complex(s, 0.0)],
            [complex(s, 0.0), complex(-s, 0.0)]
        ]
        self.apply_gate_1(h_matrix, target_qubit)

    def apply_pauli_x(self, target_qubit: int):
        """Terapkan gerbang Pauli-X (Quantum NOT Gate)."""
        x_matrix = [
            [complex(0.0, 0.0), complex(1.0, 0.0)],
            [complex(1.0, 0.0), complex(0.0, 0.0)]
        ]
        self.apply_gate_1(x_matrix, target_qubit)

    def apply_pauli_z(self, target_qubit: int):
        """Terapkan gerbang Pauli-Z (Phase Flip Gate)."""
        z_matrix = [
            [complex(1.0, 0.0), complex(0.0, 0.0)],
            [complex(0.0, 0.0), complex(-1.0, 0.0)]
        ]
        self.apply_gate_1(z_matrix, target_qubit)

    def apply_cnot(self, control_qubit: int, target_qubit: int):
        """Terapkan gerbang Controlled-NOT (CNOT) -> Membuat Entanglement."""
        new_vector = [complex(0.0, 0.0)] * self.num_states
        
        for state in range(self.num_states):
            amplitude = self.state_vector[state]
            if abs(amplitude) < 1e-9:
                continue
                
            # Cek apakah bit control bernilai 1
            control_val = (state >> control_qubit) & 1
            if control_val == 1:
                # Balik bit target
                target_state = state ^ (1 << target_qubit)
                new_vector[target_state] += amplitude
            else:
                new_vector[state] += amplitude
                
        self.state_vector = new_vector
        self._normalize()

    def measure(self) -> str:
        """
        Lakukan pengukuran (Measurement).
        Menyebabkan runtuhnya fungsi gelombang (wavefunction collapse)
        secara acak berdasarkan probabilitas kuantum amplitudo.
        """
        probabilities = [abs(amp) ** 2 for amp in self.state_vector]
        r = random.uniform(0.0, 1.0)
        
        cumulative = 0.0
        collapsed_state = 0
        for state, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                collapsed_state = state
                break
                
        # Keruntuhan total (Collapse): Set amplitudo state terpilih ke 1.0, yang lain 0.0
        self.state_vector = [complex(0.0, 0.0)] * self.num_states
        self.state_vector[collapsed_state] = complex(1.0, 0.0)
        
        # Kembalikan representasi string biner
        return f"{collapsed_state:0{self.num_qubits}b}"

    def _normalize(self):
        """Normalisasi state vector agar jumlah probabilitas selalu = 1.0."""
        total_probability = sum(abs(amp) ** 2 for amp in self.state_vector)
        if total_probability > 1e-9:
            factor = 1.0 / math.sqrt(total_probability)
            self.state_vector = [amp * factor for amp in self.state_vector]

    def get_state_expression(self) -> str:
        """Mengembalikan ekspresi matematis dalam notasi Dirac (Ket Notation)."""
        terms = []
        for state in range(self.num_states):
            amplitude = self.state_vector[state]
            if abs(amplitude) < 1e-4:
                continue
                
            # Format bilangan kompleks
            real = amplitude.real
            imag = amplitude.imag
            bin_str = f"{state:0{self.num_qubits}b}"
            
            if abs(imag) < 1e-4:
                coeff = f"{real:+.3f}" if abs(real - 1.0) > 1e-4 and abs(real + 1.0) > 1e-4 else ("+" if real > 0 else "-")
                if coeff in ("+", "-"):
                    terms.append(f"{coeff}|{bin_str}>")
                else:
                    terms.append(f"{coeff}|{bin_str}>")
            else:
                terms.append(f"({real:+.3f}{imag:+.3f}j)|{bin_str}>")
                
        expr = " ".join(terms)
        if expr.startswith("+"):
            expr = expr[1:]
        return expr


def simulate_quantum_entanglement() -> str:
    """Simulasi pembuatan Bell State (Entanglement): (|00> + |11>) / sqrt(2)"""
    reg = QuantumRegister(2)
    report = []
    report.append("Inisialisasi Qubit Register: |00>")
    report.append(f"  State Vector Awal: {reg.get_state_expression()}")
    
    # 1. Terapkan Hadamard ke Qubit 0
    reg.apply_hadamard(0)
    report.append("\nTerapkan Gerbang Hadamard (H) pada Qubit 0 (Superposisi):")
    report.append(f"  State Vector: {reg.get_state_expression()}")
    
    # 2. Terapkan CNOT (Qubit 0 -> Qubit 1)
    reg.apply_cnot(0, 1)
    report.append("\nTerapkan Gerbang CNOT (Control: Qubit 0, Target: Qubit 1) -> Entanglement:")
    report.append(f"  State Vector (Bell State): {reg.get_state_expression()}")
    
    # 3. Lakukan Pengukuran berulang-ulang
    report.append("\nMelakukan 10x Pengukuran Kuantum (Wavefunction Collapse):")
    counts = {"00": 0, "11": 0, "01": 0, "10": 0}
    
    # Simpan backup state vector Bell State
    saved_vector = list(reg.state_vector)
    
    for i in range(10):
        # Restore state vector sebelum diukur kembali
        reg.state_vector = list(saved_vector)
        res = reg.measure()
        counts[res] += 1
        report.append(f"  Pengukuran #{i+1}: Kolaps ke basis |{res}>")
        
    report.append(f"\nDistribusi Pengukuran: {counts}")
    report.append("Catatan: Terbukti tidak pernah terjadi basis |01> atau |10> karena kedua qubit terikat (entangled) mutlak!")
    return "\n".join(report)


def simulate_shor_period_finding(n: int, a: int) -> str:
    """
    Simulasi matematis untuk Shor's Algorithm bagian Period Finding.
    Shor's Algorithm memecahkan faktor prima n dari a^r = 1 (mod n) dengan mencari periode r.
    """
    report = []
    report.append(f"=== SIMULASI SHOR'S ALGORITMA: Period Finding ===")
    report.append(f"Tujuan: Memecahkan kunci RSA dengan memfaktorkan N = {n} menggunakan basis a = {a}")
    
    # Cek FPB
    fpb = math.gcd(a, n)
    if fpb > 1:
        report.append(f"  Faktor langsung ditemukan lewat FPB klasik: gcd({a}, {n}) = {fpb}")
        return "\n".join(report)
        
    # Cari periode r secara klasik/simulasi kuantum
    # Kuantum menggunakan Quantum Fourier Transform (QFT) untuk menemukan r
    report.append(f"  1. Membuat Register Kuantum Evaluasi Modulo f(x) = {a}^x mod {n}")
    
    found_r = None
    table = []
    for x in range(1, n + 2):
        val = (a ** x) % n
        table.append(f"f({x}) = {val}")
        if val == 1 and found_r is None:
            found_r = x
            
    report.append(f"  2. Hasil Spektrum Evaluasi Kuantum (QFT Periode Search):")
    report.append(f"     " + ", ".join(table[:10]) + " ...")
    
    if found_r is None:
        report.append("  [ERROR] Periode r tidak ditemukan.")
        return "\n".join(report)
        
    report.append(f"  3. QFT mendeteksi frekuensi resonansi periode: r = {found_r}")
    
    # Hitung faktor prima
    if found_r % 2 != 0:
        report.append(f"  ⚠️ Periode r ({found_r}) ganjil. Shor's algoritma harus diulang dengan basis 'a' yang berbeda.")
        return "\n".join(report)
        
    val_minus = (a ** (found_r // 2)) - 1
    val_plus  = (a ** (found_r // 2)) + 1
    
    p = math.gcd(val_minus, n)
    q = math.gcd(val_plus, n)
    
    report.append(f"  4. Hitung Faktor Pembagi Terbesar (FPB) dari Modulo:")
    report.append(f"     p = gcd({a}^{found_r//2} - 1, {n}) = gcd({val_minus}, {n}) = {p}")
    report.append(f"     q = gcd({a}^{found_r//2} + 1, {n}) = gcd({val_plus}, {n}) = {q}")
    
    if p * q == n and p > 1 and q > 1:
        report.append(f"\n  ✅ RSA Terpecahkan! Faktor prima dari N={n} adalah p={p} dan q={q}.")
    else:
        report.append(f"\n  ❌ Faktor gagal diperoleh secara deterministik. Ganti basis 'a'.")
        
    return "\n".join(report)


def run_quantum_report(query_text: str) -> str:
    """Generate visual quantum computing report for MOKO OS L0 bypass."""
    q_lower = query_text.lower()
    
    report = []
    report.append("=====================================================================")
    report.append("               MOKO OS — SOVEREIGN QUANTUM SIMULATOR")
    report.append("=====================================================================")
    
    if "shor" in q_lower or "rsa" in q_lower:
        # Jalankan period finding simulator
        report.append(simulate_shor_period_finding(15, 7))
    else:
        # Jalankan register superposisi/entanglement simulator
        report.append(simulate_quantum_entanglement())
        
    report.append("=====================================================================")
    return "\n".join(report)
