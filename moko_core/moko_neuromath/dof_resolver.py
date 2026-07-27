"""
MOKO NeuroMath: Degrees of Freedom (DOF) Resolver
==================================================
Berdasarkan: Bernstein's Degrees of Freedom Problem (1967)

Ketika MOKO menghadapi pertanyaan baru, ada ratusan kemungkinan rumus 
di Math-Omni yang bisa digunakan (terlalu banyak DOF).
Mencoba semuanya akan menyebabkan "Cognitive Overload" (terlalu lambat).

DOF Resolver meniru cara otak mengatasi ini:
  1. FREEZE (Pemula/Kondisi Kritis): Kunci pencarian hanya pada kategori
     logika dominan. Abaikan yang lain.
  2. RELEASE (Biasa/Eksplorasi): Buka sedikit demi sedikit kategori 
     tetangga jika tidak ada solusi yang memuaskan.
  3. EXPLOIT (Master/Synergy): Gunakan kombinasi dari seluruh kategori
     (dikelola oleh SynergyRouter).
"""

from typing import List, Tuple
from moko_memory.math_omni import LOGIC_MAP

# Peta Kedekatan Kategori (Manakah kategori yang berdekatan sifatnya?)
LOGIC_NEIGHBORS = {
    "A": ["E", "D"],  # Faktual dekat dengan Kausal & Definisional
    "B": ["F", "G"],  # Empatik dekat dengan Filosofis & Instruktif
    "C": ["F", "E"],  # Kreatif dekat dengan Filosofis & Kausal
    "D": ["A", "E"],  # Definisional dekat dengan Faktual & Kausal
    "E": ["A", "C"],  # Kausal dekat dengan Faktual & Kreatif
    "F": ["B", "C"],  # Filosofis dekat dengan Empatik & Kreatif
    "G": ["B", "A"],  # Instruktif dekat dengan Empatik & Faktual
}


class DOFResolver:
    """
    Mengontrol seberapa "luas" pencarian Math-Omni dilakukan.
    """

    @staticmethod
    def get_search_space(primary_logic: str, arousal_level: int) -> List[str]:
        """
        Menentukan kategori logika apa saja yang boleh dicari (melepas DOF).
        
        Args:
            primary_logic: Kategori utama hasil analisis (A-H)
            arousal_level: 1 (Tenang), 2 (Sedang), 3 (Tinggi/Kritis)
            
        Returns:
            List kategori logika yang diizinkan untuk dicari.
        """
        primary = primary_logic.upper()
        if primary not in LOGIC_MAP:
            primary = "A"

        if arousal_level >= 3:
            # FREEZE DOF: Situasi kritis, jangan buang waktu. 
            # Kunci pencarian hanya pada kategori utama.
            return [primary]
            
        elif arousal_level == 2:
            # RELEASE DOF (Partial): Situasi normal.
            # Buka pencarian ke kategori utama dan 1 tetangga terdekat.
            neighbors = LOGIC_NEIGHBORS.get(primary, [])
            space = [primary]
            if neighbors:
                space.append(neighbors[0])
            return space
            
        else:
            # RELEASE DOF (Full) / EXPLOIT: Situasi santai.
            # Otak punya banyak waktu untuk mengeksplorasi dan berkreasi.
            neighbors = LOGIC_NEIGHBORS.get(primary, [])
            return [primary] + neighbors


# Singleton
dof_resolver = DOFResolver()
