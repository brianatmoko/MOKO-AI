import time
from moko_cpu.governor import CPUGovernor

class LocusCoeruleus:
    """
    MOKO NeuroMath: Locus Coeruleus — Norepinephrine System
    ======================================================
    Biological modulator of attentional focus and sensory gating.
    
    Noradrenalin (Norepinephrine) dilepaskan ketika otak mengalami:
      1. Stres fisik/hardware (CPU panas, CPU load tinggi, RAM sempit)
      2. Stres kognitif (Surprisal/Prediction Error tinggi dari evaluasi FEP)
      
    Efek pelepasan noradrenalin:
      - Norepinephrine tinggi -> Atensi terfokus (Freeze DOF, kurangi search space)
      - Gating Thalamus lebih ketat -> Tingkatkan selektivitas Novelty (tolak redundant)
    """

    def __init__(self):
        self.recent_surprisal: float = 0.0
        self.last_updated: float = time.time()

    def record_surprisal(self, score: float):
        """Mencatat nilai surprisal kognitif terbaru."""
        self.recent_surprisal = float(score)
        self.last_updated = time.time()

    def calculate_norepinephrine(self) -> float:
        """
        Menghitung level Norepinephrine (NE) dalam rentang 0.0 (tenang) s.d. 1.0 (sangat panik).
        """
        # Meluruhkan secara eksponensial seiring waktu jika tidak ada surprisal baru
        elapsed = time.time() - self.last_updated
        decayed_surprisal = self.recent_surprisal * (0.9 ** (elapsed / 30.0)) # Half-life ~3 menit

        # 1. Ambil tanda vital hardware
        try:
            vitals = CPUGovernor.read_vitals()
            cpu_temp = vitals.get("cpu_temp", 50.0)
            cpu_pct = vitals.get("cpu_pct", 10.0)
            ram_pct = vitals.get("ram_pct", 50.0)
        except Exception:
            cpu_temp, cpu_pct, ram_pct = 50.0, 10.0, 50.0

        # 2. Faktor stres hardware (0.0 s.d. 1.0)
        # Mulai stress jika suhu > 60°C, maks di 85°C
        temp_stress = max(0.0, min(1.0, (cpu_temp - 60.0) / 25.0))
        # Mulai stress jika CPU usage > 60%
        cpu_stress = max(0.0, min(1.0, (cpu_pct - 60.0) / 35.0))
        # Mulai stress jika RAM > 75%
        ram_stress = max(0.0, min(1.0, (ram_pct - 75.0) / 20.0))

        # Stres hardware keseluruhan adalah nilai maksimum stres organ
        hw_stress = max(temp_stress, cpu_stress, ram_stress)

        # 3. Gabungkan stres kognitif dan stres fisik
        ne_level = max(hw_stress, decayed_surprisal)
        return round(ne_level, 4)

    def modulate_arousal(self, base_arousal: str) -> int:
        """
        Menyesuaikan tingkat arousal kognitif (1 s.d. 3) berdasarkan level Norepinephrine.
        """
        ne = self.calculate_norepinephrine()
        
        # Jika noradrenalin sangat tinggi, paksa arousal 3 (Fokus Kritis / Freeze DOF)
        if ne >= 0.70:
            return 3
        # Jika sedang, paksa arousal 2 (Normal / Release Partial)
        elif ne >= 0.40:
            return 2
        # Jika rendah, ikuti base arousal bawaan logika pertanyaan
        else:
            try:
                return int(base_arousal)
            except Exception:
                return 2

    def modulate_novelty_threshold(self, base_threshold: float) -> float:
        """
        Menyesuaikan threshold penyerapan Novelty di Thalamus Gate.
        
        Norepinephrine tinggi -> Selektivitas meningkat.
        Artinya kita menurunkan threshold (misal dari 0.85 ke 0.65) agar data yang
        hanya mirip 70% pun sudah ditolak karena dianggap tidak cukup novel untuk
        menyelamatkan RAM/CPU.
        """
        ne = self.calculate_norepinephrine()
        # Mengurangi threshold maksimal 0.20 saat NE = 1.0
        modulated = base_threshold - (ne * 0.20)
        return round(max(0.40, modulated), 4)


# Singleton Instance
locus_coeruleus = LocusCoeruleus()
