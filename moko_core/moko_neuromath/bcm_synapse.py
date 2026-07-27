import json
import os
from pathlib import Path
from moko_config import settings

class BCMSynapse:
    """
    Bienenstock-Cooper-Munro (BCM) Theory & Oja's Rule.
    Mengelola plastisitas sinaptik (LTP/LTD) dan Sliding Threshold untuk Math-Omni.
    """
    
    LTP_DELTA = 0.10   # Long-Term Potentiation (Penguatan memori saat puas)
    LTD_DELTA = -0.05  # Long-Term Depression (Pelemahan memori saat error/tidak puas)
    MAX_WEIGHT = 5.0   # Oja's Rule: Batas atas bobot agar tidak meledak tak terbatas
    
    @classmethod
    def get_synaptic_weights(cls) -> dict:
        """
        Membaca bobot sinapsis dari sidecar.
        Mengembalikan dict: {formula_id: weight_float}
        """
        sidecar_path = Path(settings.WORKSPACE_DIR) / ".math_omni" / "formula_sidecar.jsonl"
        weights = {}
        if not sidecar_path.exists():
            return weights
            
        with open(sidecar_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    fid = data.get("formula_id") or data.get("id")
                    # Bobot default untuk formula baru adalah 1.0
                    w = float(data.get("synaptic_weight", 1.0))
                    if fid:
                        weights[fid] = w
                except:
                    pass
        return weights

    @classmethod
    def apply_plasticity(cls, formula_id: str, is_satisfied: bool):
        """
        Menerapkan LTP atau LTD pada bobot formula.
        Menulis ulang seluruh file sidecar (karena JSONL).
        """
        sidecar_path = Path(settings.WORKSPACE_DIR) / ".math_omni" / "formula_sidecar.jsonl"
        if not sidecar_path.exists():
            return

        lines = []
        with open(sidecar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                
                fid = data.get("formula_id") or data.get("id")
                if fid == formula_id:
                    w = float(data.get("synaptic_weight", 1.0))
                    
                    if is_satisfied:
                        w += cls.LTP_DELTA
                    else:
                        w += cls.LTD_DELTA
                        
                    # Oja's Rule (Normalisasi maks)
                    w = min(w, cls.MAX_WEIGHT)
                    
                    data["synaptic_weight"] = round(w, 4)
                    
                f.write(json.dumps(data) + "\n")

    @classmethod
    def compute_sliding_threshold(cls) -> float:
        """
        Menghitung $\\theta_M$ (Sliding Threshold) berdasarkan BCM Theory.
        Threshold adalah rata-rata (atau persentil) dari aktivitas seluruh sinapsis.
        """
        weights = cls.get_synaptic_weights()
        if not weights:
            return 0.5 # Default minimum threshold
            
        avg_weight = sum(weights.values()) / len(weights)
        # BCM Sliding Threshold: rumus sinapsis mati jika bobot < 50% dari rata-rata otak
        theta_m = avg_weight * 0.50
        
        # Oja's absolute bottom clamp
        return max(0.2, theta_m)
