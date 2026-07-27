"""
MOKO NeuroMath: Cortisol System (HPA Axis) — Stress Modulator
=============================================================
Berdasarkan:
  - HPA Axis Activation & Cortisol (Frontiers in Endocrinology, 2023)
  - Lupien et al. (2009): Effects of stress and cortisol on cognitive function
  - Acute stress (adaptive search) vs Chronic stress (hippocampal damage)

FUNGSI:
  1. Cortisol Level: Akumulasi stres fisik (hardware) dan stres kognitif.
  2. Acute Stress Response: Mempercepat pengambilan keputusan saat terdesak.
  3. Chronic Stress Penalty: Menekan plastisitas LTP Hebbian saat stres berkepanjangan.
"""

import time
from typing import Optional


class CortisolSystem:
    """
    CortisolSystem (HPA Axis) — Mengelola tingkat stres sistem MOKO.
    Stres dipicu oleh beban hardware (RAM/CPU/Suhu) dan kebingungan kognitif (ACC conflict).
    """

    def __init__(self):
        self.cortisol_level: float = 0.2  # Healthy baseline
        self.chronic_timer: float = 0.0   # Berapa lama stres tinggi dialami
        self.last_updated: float = time.time()

    def update_stress(self, conflict_score: float, vitals: dict):
        """
        Hitung tingkat stres sistem berdasarkan vitals hardware dan konflik kognitif.
        """
        now = time.time()
        elapsed = now - self.last_updated
        
        # 1. Meluruhkan cortisol secara alami jika tidak ada stressor
        decay_factor = 0.85 ** (elapsed / 60.0)  # half-life 60 detik
        self.cortisol_level = max(0.1, self.cortisol_level * decay_factor)

        # 2. Hitung stressor fisik (hardware)
        cpu_temp = vitals.get("cpu_temp", 50.0)
        cpu_pct = vitals.get("cpu_pct", 10.0)
        ram_pct = vitals.get("ram_pct", 50.0)
        
        hw_stress = 0.0
        if cpu_temp > 75.0:
            hw_stress = max(hw_stress, (cpu_temp - 75.0) / 15.0)
        if cpu_pct > 80.0:
            hw_stress = max(hw_stress, (cpu_pct - 80.0) / 20.0)
        if ram_pct > 85.0:
            hw_stress = max(hw_stress, (ram_pct - 85.0) / 15.0)

        # 3. Hitung stressor kognitif (ACC conflict)
        cognitive_stress = 0.0
        if conflict_score > 0.80:
            # Konflik tinggi berkepanjangan memicu stres kognitif
            cognitive_stress = (conflict_score - 0.80) * 1.5

        # 4. Akumulasikan stres
        total_stressor = max(hw_stress, cognitive_stress)
        if total_stressor > 0.1:
            self.cortisol_level = min(2.0, self.cortisol_level + total_stressor * 0.15)

        # 5. Deteksi stres kronis (cortisol > 1.2 bertahan lama)
        if self.cortisol_level > 1.2:
            if self.chronic_timer == 0.0:
                self.chronic_timer = now
        else:
            self.chronic_timer = 0.0

        self.last_updated = now

    def cool_down(self):
        """Meredakan stres secara drastis (dipanggil saat MOKO tidur kognitif)."""
        self.cortisol_level = max(0.15, self.cortisol_level * 0.3)
        self.chronic_timer = 0.0
        self.last_updated = time.time()

    @property
    def is_acute_stress(self) -> bool:
        """Stres akut terdeteksi (cortisol > 0.85). Mengarahkan sistem untuk lebih defensif/cepat."""
        return self.cortisol_level > 0.85

    @property
    def plasticity_penalty(self) -> float:
        """
        Chronic Stress Penalty (0.1 s.d. 1.0):
        Stres kronis merusak dendrit di Hippocampus.
        Jika cortisol > 1.2 selama lebih dari 10 detik kognitif (simulasi), 
        turunkan laju LTP Hebbian secara proporsional.
        """
        if self.cortisol_level > 1.2 and self.chronic_timer > 0.0:
            duration = time.time() - self.chronic_timer
            if duration > 10.0:  # Stres tinggi dialami terus-menerus
                # Semakin tinggi stres, semakin besar penalti plastisitas
                penalty = max(0.1, 2.0 - self.cortisol_level)
                return round(penalty, 4)
        return 1.0

    def get_status(self) -> dict:
        return {
            "cortisol_level":      round(self.cortisol_level, 4),
            "is_acute_stress":     self.is_acute_stress,
            "plasticity_penalty":  self.plasticity_penalty,
            "chronic_stress_time": round(time.time() - self.chronic_timer, 1) if self.chronic_timer > 0.0 else 0.0
        }


# Singleton Instance
cortisol_system = CortisolSystem()
