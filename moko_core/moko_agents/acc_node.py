"""
MOKO Agent: ACCNode — Anterior Cingulate Cortex (Conflict & Error Monitor)
========================================================================
Tugas:
1. Conflict Monitoring: Mendeteksi jika terdapat beberapa memori yang bersaing 
   dengan nilai kecocokan (score) yang berdekatan.
2. Effort Allocation: Menentukan jumlah pass (1 hingga 3) secara dinamis 
   berdasarkan konflik kognitif, menggantikan kontrol termal murni.
3. Post-Error Slowing: Memperlambat langkah kognitif berikutnya jika terdeteksi 
   kegagalan/error untuk meningkatkan akurasi.
"""

from typing import List

class ACCNode:
    def __init__(self):
        self.post_error_slowing = False
        self.last_conflict_score = 0.0

    def monitor_conflict(self, memory_scores: List[float]) -> float:
        """
        Menghitung skor konflik berdasarkan kedekatan nilai skor memori teratas.
        Jika selisih skor memori teratas tipis -> konflik tinggi (kebingungan).
        """
        if not memory_scores or len(memory_scores) < 2:
            self.last_conflict_score = 0.0
            return 0.0

        # Urutkan secara descending
        sorted_scores = sorted(memory_scores, reverse=True)
        s0 = sorted_scores[0]
        s1 = sorted_scores[1]

        # Selisih skor teratas
        diff = s0 - s1
        
        # Jika diff mendekati 0, konflik mendekati 1.0
        # Jika diff lebar (misal > 0.4), konflik rendah
        conflict = max(0.0, 1.0 - (diff * 2.5))
        self.last_conflict_score = round(min(1.0, conflict), 3)
        return self.last_conflict_score

    def allocate_effort(self, conflict_score: float, base_passes: int) -> int:
        """
        Menentukan pass count secara adaptif berdasarkan tingkat konflik.
        """
        # Batasi base_passes awal
        passes = base_passes
        
        if conflict_score > 0.8:
            # Konflik tinggi -> tambah iterasi analisis demi akurasi (max 3)
            passes = min(3, base_passes + 1)
        elif conflict_score < 0.2:
            # Konflik sangat rendah -> kurangi iterasi untuk efisiensi (min 1)
            passes = max(1, base_passes - 1)
            
        return passes

    def trigger_error(self):
        """Picu perlambatan kognitif pasca error."""
        self.post_error_slowing = True

    def reset_error_state(self):
        """Reset status perlambatan kognitif."""
        self.post_error_slowing = False
