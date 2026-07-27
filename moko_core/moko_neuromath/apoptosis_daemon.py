import os
import glob
from pathlib import Path
from moko_config import settings
from moko_neuromath.bcm_synapse import BCMSynapse

class ApoptosisDaemon:
    """
    Synaptic Pruning / Sang Malaikat Maut.
    Berjalan di background untuk menghapus sel-sel memori (rumus) yang mati (W < Theta_M).
    """

    @classmethod
    def execute_pruning(cls) -> list:
        """
        Mengeksekusi pemangkasan sel.
        Mengembalikan daftar formula_id yang dihancurkan.
        """
        pruned_ids = []
        theta_m = BCMSynapse.compute_sliding_threshold()
        weights = BCMSynapse.get_synaptic_weights()
        
        math_omni_dir = Path(settings.WORKSPACE_DIR) / ".math_omni"
        if not math_omni_dir.exists():
            return []

        for fid, w in weights.items():
            if w < theta_m:
                # Kematian Sel: Hapus file .bin
                target_file = math_omni_dir / f"{fid}.bin"
                if target_file.exists():
                    try:
                        os.remove(target_file)
                        pruned_ids.append(fid)
                    except:
                        pass
                        
        # Bersihkan dari sidecar
        if pruned_ids:
            cls._clean_sidecar(pruned_ids, math_omni_dir / "formula_sidecar.jsonl")
            
        return pruned_ids

    @classmethod
    def _clean_sidecar(cls, pruned_ids: list, sidecar_path: Path):
        import json
        if not sidecar_path.exists(): return
        
        lines = []
        with open(sidecar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                fid = data.get("formula_id") or data.get("id")
                if fid not in pruned_ids:
                    f.write(line)
