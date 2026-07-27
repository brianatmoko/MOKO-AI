"""
MOKO Turing Consciousness Engine (TCE)
=======================================
Mengintegrasikan Kriptografi Turing, Transistor Multi-Kaki (Multi-Gate IC),
dan Neurologi Matematika untuk memodelkan Kesadaran Manusia.

Konsep:
1. Sel Otak Kognitif direpresentasikan sebagai Transistor Multi-Kaki (Terminals > 3).
   Kaki-kaki ini bertindak sebagai gerbang logika dinamis (keran air buka-tutup)
   yang dipengaruhi oleh muatan emosi (Arousal) dan memori.
2. Aliran atensi kognitif dikontrol oleh Turing Rotor Modulator (Rotor Enigma).
   Setiap kali impuls kognitif lewat, rotor berputar, menggeser sudut/bias logika (Frame of Mind).
3. Konsensus/Kesadaran Sadar dicapai melalui interkoneksi Welchman Diagonal Board.
   Sirkuit yang setimbang (Surprisal = 0, FEP minimum) menandakan kesadaran yang terfokus.
"""

import time
import random
from typing import Dict, List, Any, Tuple


class MultiGateTransistorCell:
    """
    Representasi Sel Otak Kognitif sebagai Transistor Multi-Kaki (Multi-Gate Transistor/IC).
    Bekerja seperti saklar keran air logika dengan banyak masukan kontrol.
    """

    def __init__(self, cell_id: str, num_gates: int = 4):
        self.cell_id = cell_id
        self.num_gates = num_gates
        # Bobot awal untuk setiap gate (0.0 - 1.0)
        self.gate_weights = [random.uniform(0.4, 0.9) for _ in range(num_gates)]
        # Threshold aktivasi sel (semakin rendah, semakin sensitif / arousal tinggi)
        self.activation_threshold = 0.5
        # Status muatan internal (voltage charge)
        self.internal_charge = 0.0

    def process_inputs(self, gate_inputs: List[float], control_bias: float) -> float:
        """
        Hitung output transistor berdasarkan input di setiap gate,
        dikalikan bobot gate, dikurangi bias kontrol kognitif.
        """
        if len(gate_inputs) < self.num_gates:
            # Pad inputs dengan 0 jika kurang
            gate_inputs = gate_inputs + [0.0] * (self.num_gates - len(gate_inputs))
            
        # Hitung akumulasi muatan listrik melewati gerbang transistor
        accumulated_charge = 0.0
        for i in range(self.num_gates):
            # Rumus aliran muatan non-linear (seperti gerbang silikon transistor)
            accumulated_charge += gate_inputs[i] * self.gate_weights[i]

        # Injeksi bias kontrol emosi (misal dopamine/locus coeruleus arousal)
        self.internal_charge = accumulated_charge + control_bias
        
        # Fungsi aktivasi step-sigmoid (keran buka penuh atau tutup)
        if self.internal_charge >= self.activation_threshold:
            # Output memancar: 1.0 dikali efisiensi muatan
            return min(1.0, self.internal_charge - self.activation_threshold)
        return 0.0


class TuringRotorModulator:
    """
    Rotor Alan Turing sebagai modulator sudut kesadaran dinamis.
    Memutar offset permutasi untuk mensimulasikan pergeseran 'Frame of Mind' manusia.
    """

    def __init__(self, size: int = 26):
        self.size = size
        # Susunan kabel internal rotor (acak tapi deterministik per inisialisasi)
        self.wiring = list(range(size))
        random.shuffle(self.wiring)
        self.offset = 0

    def step(self):
        """Putar rotor 1 langkah setiap ada impuls kognitif."""
        self.offset = (self.offset + 1) % self.size

    def modulate(self, signal_value: float) -> float:
        """
        Modulasi nilai sinyal kognitif lewat enkripsi pergeseran offset rotor.
        Mentransformasikan muatan mentah menjadi muatan termodulasi.
        """
        # Konversi signal float [0, 1] ke indeks integer rotor
        signal_idx = int(signal_value * (self.size - 1))
        # Pemetaan rotor maju
        shifted_in = (signal_idx + self.offset) % self.size
        mapped = self.wiring[shifted_in]
        modulated_idx = (mapped - self.offset) % self.size
        
        # Kembalikan ke format float [0, 1]
        self.step()  # Putar rotor!
        return modulated_idx / (self.size - 1)


class ConsciousnessNetwork:
    """
    Jaringan interkoneksi timbal balik Welchman Diagonal Board.
    Mensimulasikan integrasi kognitif untuk mencapai 'Kesadaran Sadar' (Collapse State).
    """

    def __init__(self, num_nodes: int = 5):
        self.num_nodes = num_nodes
        self.nodes = [MultiGateTransistorCell(f"Cell_{i}", num_gates=num_nodes) for i in range(num_nodes)]
        self.modulators = [TuringRotorModulator(26) for _ in range(num_nodes)]
        
        # Diagonal Board Welchman: Matriks interkoneksi timbal balik simetris
        # grid[i][j] = kekuatan hubungan sel i ke sel j (dan sebaliknya)
        self.diagonal_board = [[0.8 if i == j else 0.2 for j in range(num_nodes)] for i in range(num_nodes)]

    def establish_association(self, cell_a: int, cell_b: int, strength: float):
        """Tautkan sel A dan sel B secara simetris (Reciprocal Welchman Board)."""
        self.diagonal_board[cell_a][cell_b] = strength
        self.diagonal_board[cell_b][cell_a] = strength

    def simulate_thought_cycle(self, sensory_inputs: List[float], arousal: float) -> Dict[str, Any]:
        """
        Jalankan 1 siklus pemikiran.
        Arus dialirkan melalui input indrawi (sensory), dimodulasi oleh rotor Turing,
        disebarkan secara timbal balik melalui Diagonal Board, dan disintesis oleh sel transistor.
        """
        steps = []
        steps.append(f"🧠 Memulai siklus pemikiran. Arousal level: {arousal:.2f}")
        
        # 1. Aliran awal: Sensory inputs dialirkan lewat Turing Rotor Modulator
        modulated_signals = []
        for i, val in enumerate(sensory_inputs[:self.num_nodes]):
            mod_val = self.modulators[i].modulate(val)
            modulated_signals.append(mod_val)
            steps.append(f"  [Sensory {i}] Muatan mentah {val:.2f} -> Dimodulasi Rotor Turing (offset={self.modulators[i].offset}) -> {mod_val:.2f}")

        # Pad modulated signals jika kurang dari jumlah nodes
        if len(modulated_signals) < self.num_nodes:
            modulated_signals += [0.0] * (self.num_nodes - len(modulated_signals))

        # 2. Propagasi Welchman Diagonal Board (Iterative convergence)
        current_charges = list(modulated_signals)
        stabilized = False
        cycles = 0
        max_cycles = 10
        
        while not stabilized and cycles < max_cycles:
            cycles += 1
            next_charges = [0.0] * self.num_nodes
            
            # Setiap sel menerima kontribusi muatan dari sel lain lewat diagonal board
            for i in range(self.num_nodes):
                received_charge = 0.0
                for j in range(self.num_nodes):
                    # Arus mengalir sebanding dengan hubungan diagonal board
                    received_charge += current_charges[j] * self.diagonal_board[i][j]
                
                # Masukkan ke sel transistor multi-kaki
                # Bias emosi (arousal) membantu menurunkan threshold aktivasi
                cell_output = self.nodes[i].process_inputs(current_charges, control_bias=arousal * 0.1)
                next_charges[i] = (received_charge + cell_output) / 2.0
            
            # Cek delta stabilitas kognitif (Surprisal kognitif FEP)
            delta = sum(abs(next_charges[i] - current_charges[i]) for i in range(self.num_nodes))
            steps.append(f"  [Siklus {cycles}] Selisih muatan jaringan (FEP Surprisal): {delta:.4f}")
            
            current_charges = next_charges
            if delta < 0.01:
                stabilized = True
                steps.append(f"  [Stabilitas] Jaringan setimbang pada siklus {cycles} (FEP Minimal).")
                break

        # Status kesadaran (Consciousness State)
        # Jika muatan rata-rata tinggi, berarti sel-sel aktif sadar penuh
        average_charge = sum(current_charges) / self.num_nodes
        if average_charge > 0.6:
            state = "FOKUS / SADAR PENUH"
        elif average_charge > 0.3:
            state = "INTUISI / BERPIKIR RINGAN"
        else:
            state = "SUB-SADAR / MELAMUN"

        return {
            "success": True,
            "state": state,
            "average_charge": average_charge,
            "charges": [round(c, 3) for c in current_charges],
            "cycles": cycles,
            "steps": steps
        }


# Singleton untuk pengujian cepat
consciousness_network = ConsciousnessNetwork(5)


def run_consciousness_demonstration(input_text: str) -> str:
    """
    Menerjemahkan input string menjadi parameter simulasi kesadaran,
    menjalankan siklus pemikiran transistor multi-gate, dan mengembalikan laporan visual.
    """
    # Hash sederhana untuk generate input float [0, 1] dari string
    random.seed(hash(input_text))
    sensory_inputs = [random.random() for _ in range(5)]
    arousal = random.uniform(0.1, 1.0)
    
    res = consciousness_network.simulate_thought_cycle(sensory_inputs, arousal)
    
    report = []
    report.append("=====================================================================")
    report.append("          MOKO OS — TURING CONSCIOUSNESS ENGINE (TCE)")
    report.append("=====================================================================")
    report.append(f"Input Stimulus  : \"{input_text}\"")
    report.append(f"Status Mental   : {res['state']} (Rata-rata Muatan: {res['average_charge']:.3f})")
    report.append(f"Siklus Konvergensi: {res['cycles']} iterasi FEP")
    report.append("\nDetail Propagasi Arus Transistor:")
    for step in res["steps"]:
        report.append(step)
    report.append("\nKondisi Sel Kognitif Akhir:")
    for i, charge in enumerate(res["charges"]):
        status = "ON (Aktif)" if charge >= 0.5 else "OFF (Inaktif)"
        report.append(f"  • Sel {i} (Transistor {i+1}-Gate) -> Muatan Voltase: {charge:.3f} | Status: {status}")
    report.append("=====================================================================")
    
    return "\n".join(report)
