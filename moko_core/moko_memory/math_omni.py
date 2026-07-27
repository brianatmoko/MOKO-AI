"""
MOKO Math-Omni — Otak Kanan (Memori Prosedural)
================================================
Sistem penyimpanan rumusan kognitif MOKO yang bekerja paralel dengan Omni-RSA.

STRUKTUR:
    .math_omni/
        formula_sidecar.jsonl    ← Index semua rumus
        A1-B2-C0-LOG1.bin        ← QEV vektor trigger rumus (384 byte)
        A1-B2-C0-LOG2.bin
        B1-B1-D3-LOG1.bin
        ...

SISTEM ID (Nama File = Peta Hierarki):
    Segmen 1 — Tipe Logika (1 huruf):
        A = Faktual/Analitik
        B = Empatik/Emosional
        C = Sintesis/Kreatif
        D = Definisional (kamus, arti)
        E = Kausal (sebab-akibat)
        F = Filosofis/Reflektif
        G = Instruktif (perintah, tutorial)

    Segmen 2 — Level Arousal (1 digit):
        1 = Tenang/Rendah (sapaan, pertanyaan ringan)
        2 = Sedang/Netral (diskusi biasa)
        3 = Tinggi/Urgent (masalah serius, pertanyaan mendalam)

    Segmen 3 — Kedalaman (1 huruf + 1 digit):
        D0 = Permukaan (jawaban singkat)
        D5 = Sedang (2-4 paragraf)
        D9 = Mendalam (analisis lengkap, multi-paragraf)

    Segmen 4 — Nomor Urut Rumus (LOG + angka):
        LOG1, LOG2, LOG3 ... (MOKO menambah sendiri saat buat rumus baru)

    Contoh: A2-3-D5-LOG1 = Logika Faktual, Arousal Sedang, Kedalaman Sedang, Rumus ke-1

NAVIGASI (Glob Prefix Mengerucut):
    glob("A*.bin")          → filter tipe logika (dari 200 ke ~30 file)
    glob("A2*.bin")         → filter + arousal (dari 30 ke ~10 file)
    glob("A2-3-D5*.bin")    → filter + kedalaman (dari 10 ke 3 file)
    dot_product(3 kandidat) → cocokkan QEV, selesai!

RASA PUAS (Satisfaction Threshold):
    score > 0.85 → PUAS → pakai rumus lama
    score ≤ 0.85 → TIDAK PUAS → LLM buat rumus baru, simpan otomatis
"""

import json
import struct
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict


SATISFACTION_THRESHOLD = 0.85   # Ambang batas "rasa puas"
QEV_BYTES = 384                  # Ukuran vektor QEV (biner, 384 byte)

# Mapping: prefix segmen → label manusia
LOGIC_MAP = {
    "A": "Faktual/Analitik",
    "B": "Empatik/Emosional",
    "C": "Sintesis/Kreatif",
    "D": "Definisional",
    "E": "Kausal/Sebab-Akibat",
    "F": "Filosofis/Reflektif",
    "G": "Instruktif/Tutorial",
}

AROUSAL_MAP = {"1": "Tenang", "2": "Sedang", "3": "Urgent"}
DEPTH_MAP   = {"D0": "Singkat", "D5": "Sedang", "D9": "Mendalam"}


def fp32_to_qev(fp32_vec: List[float]) -> bytes:
    """Kompresi vektor 768-D float32 → QEV biner 384 byte (2 nilai per byte)."""
    n = min(len(fp32_vec), 768)
    result = bytearray(384)
    table = [-2, -1, 0, 1, 2]

    def quantize(v: float) -> int:
        if v >= 1.0:   return 4
        if v >= 0.5:   return 3
        if v >= 0.0:   return 2
        if v >= -0.5:  return 1
        return 0

    for i in range(0, n, 2):
        idx = i // 2
        hi = quantize(fp32_vec[i])
        lo = quantize(fp32_vec[i + 1]) if (i + 1) < n else 2
        result[idx] = (hi << 3) | lo

    return bytes(result)


def qev_dot(q: bytes, db: bytes) -> float:
    """Hitung kemiripan dua QEV (biner). Kembalikan -1.0 hingga 1.0."""
    if not q or not db or len(q) < QEV_BYTES or len(db) < QEV_BYTES:
        return 0.0
    score = 0
    table = [-2, -1, 0, 1, 2]
    for i in range(QEV_BYTES):
        hi_q = (q[i] >> 3) & 0x07
        lo_q =  q[i]       & 0x07
        hi_d = (db[i] >> 3) & 0x07
        lo_d =  db[i]       & 0x07
        q0 = table[hi_q] if hi_q < 5 else 0
        q1 = table[lo_q] if lo_q < 5 else 0
        d0 = table[hi_d] if hi_d < 5 else 0
        d1 = table[lo_d] if lo_d < 5 else 0
        score += q0 * d0 + q1 * d1
    max_score = QEV_BYTES * 8
    return score / max_score if max_score > 0 else 0.0


class MathOmni:
    """
    Otak Kanan MOKO — Gudang Rumusan Kognitif.

    Cara pakai:
        omni = MathOmni(Path(".math_omni"))
        result = omni.search(user_fp32_vec)
        # → {"formula_id": "A2-3-D5-LOG1", "instruction": "...", "satisfied": True/False}
    """

    def __init__(self, base_dir: Path):
        self.base_dir  = base_dir
        self.sidecar   = base_dir / "formula_sidecar.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index: List[Dict] = []
        self._load_index()

    # ══════════════════════════════════════════════════════════════════
    # INDEX
    # ══════════════════════════════════════════════════════════════════

    def _load_index(self):
        """Membaca formula_sidecar.jsonl ke dalam RAM (cepat, kecil)."""
        self._index = []
        if not self.sidecar.exists():
            return
        with open(self.sidecar, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index.append(json.loads(line))
                except Exception:
                    pass

    def _save_entry_to_index(self, entry: Dict):
        """Menambahkan satu entri formula ke formula_sidecar.jsonl."""
        with open(self.sidecar, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._index.append(entry)

    # ══════════════════════════════════════════════════════════════════
    # NAVIGASI (Glob Prefix Mengerucut)
    # ══════════════════════════════════════════════════════════════════

    def _glob_candidates(self, logic: str, arousal: str, depth: str) -> List[Path]:
        """
        Navigasi Kode Pos: glob mengerucut dari level 1 → 2 → 3.
        Setiap level memangkas kandidat tanpa membaca isi file.
        """
        # Level 1: Tipe Logika (misal: "A")
        candidates = list(self.base_dir.glob(f"{logic}*.bin"))
        if not candidates:
            return []

        # Level 2: + Arousal (misal: "A2")
        prefix2 = f"{logic}{arousal}"
        candidates = [p for p in candidates if p.stem.startswith(prefix2)]
        if not candidates:
            # Arousal tidak ditemukan → fallback ke semua arousal dalam tipe logika
            candidates = list(self.base_dir.glob(f"{logic}*.bin"))

        # Level 3: + Depth (misal: "A2-3-D5")
        prefix3 = f"{logic}{arousal}-{depth}"
        narrowed = [p for p in candidates if p.stem.startswith(prefix3)]
        if narrowed:
            return narrowed
        return candidates   # Fallback ke level 2 jika level 3 kosong

    # ══════════════════════════════════════════════════════════════════
    # SEARCH UTAMA
    # ══════════════════════════════════════════════════════════════════

    def search(
        self,
        user_fp32: List[float],
        logic: str = "A",
        arousal: str = "2",
        depth: str = "D5"
    ) -> Dict:
        """
        Cari rumus yang paling cocok menggunakan Glob Prefix + QEV Dot Product.

        Returns:
            {
                "formula_id":   str,    # ID rumus terpilih
                "instruction":  str,    # Instruksi rumus untuk LLM
                "score":        float,  # Skor kecocokan QEV
                "logic_label":  str,    # Label manusia tipe logika
            }
        """
        user_qev = fp32_to_qev(user_fp32)

        # Glob mengerucut
        candidates = self._glob_candidates(logic, arousal, depth)

        best_score  = 0.0
        best_path   = None

        for path in candidates:
            try:
                db_qev = path.read_bytes()
                score  = qev_dot(user_qev, db_qev)
                if score > best_score:
                    best_score = score
                    best_path  = path
            except Exception:
                pass

        if best_path:
            formula_id  = best_path.stem
            entry       = self._find_entry(formula_id)
            instruction = entry.get("instruction", "") if entry else ""
            weight      = float(entry.get("synaptic_weight", 1.0)) if entry else 1.0
            # Update usage counter
            if entry:
                entry["usage_count"] = entry.get("usage_count", 0) + 1
        else:
            formula_id  = ""
            instruction = ""
            weight      = 1.0

        return {
            "formula_id":  formula_id,
            "instruction": instruction,
            "score":       round(best_score, 4),
            "logic_label": LOGIC_MAP.get(logic, logic),
            "synaptic_weight": weight,
        }

    def _find_entry(self, formula_id: str) -> Optional[Dict]:
        for e in self._index:
            fid = e.get("id") or e.get("formula_id")
            if fid == formula_id:
                return e
        return None

    # ══════════════════════════════════════════════════════════════════
    # SIMPAN RUMUS BARU (dijalankan LLM saat Tidak Puas)
    # ══════════════════════════════════════════════════════════════════

    def save_formula(
        self,
        logic: str,
        arousal: str,
        depth: str,
        trigger_text: str,
        instruction: str,
        fp32_vec: List[float]
    ) -> str:
        """
        Simpan rumus baru ke Math-Omni (dipanggil saat MOKO tidak puas).
        ID otomatis dihasilkan: A2-3-D5-LOG4 (menambah nomor setelah LOG terbesar).

        Returns: formula_id yang baru dibuat
        """
        # Cari nomor LOG terbesar yang sudah ada di kategori ini
        prefix = f"{logic}{arousal}-{depth}-LOG"
        existing = list(self.base_dir.glob(f"{prefix}*.bin"))
        max_n = 0
        for p in existing:
            try:
                n = int(p.stem.replace(prefix, "").replace("-", ""))
                max_n = max(max_n, n)
            except Exception:
                pass

        new_n     = max_n + 1
        formula_id = f"{logic}{arousal}-{depth}-LOG{new_n}"
        bin_path   = self.base_dir / f"{formula_id}.bin"

        # Simpan QEV ke file biner
        qev = fp32_to_qev(fp32_vec)
        bin_path.write_bytes(qev)

        # Tambahkan ke index
        entry = {
            "id":              formula_id,
            "logic":           logic,
            "arousal":         arousal,
            "depth":           depth,
            "trigger_text":    trigger_text,
            "instruction":     instruction,
            "usage_count":     0,
            "synaptic_weight": 1.0,
            "created_at":      time.time(),
        }
        self._save_entry_to_index(entry)
        print(f"[MathOmni] ✨ Rumus baru tersimpan: {formula_id}")
        return formula_id

    def rewrite_formula(
        self,
        formula_id: str,
        trigger_text: str,
        instruction: str,
        fp32_vec: List[float]
    ) -> bool:
        """
        Menulis ulang formula kognitif yang usang/tidak berkualitas.
        Memperbarui berkas biner .bin dan index di formula_sidecar.jsonl.
        Mereset synaptic_weight kembali ke 1.0 (segar kembali) dan memperbarui timestamp.
        """
        bin_path = self.base_dir / f"{formula_id}.bin"
        
        # 1. Simpan QEV baru ke file biner
        try:
            qev = fp32_to_qev(fp32_vec)
            bin_path.write_bytes(qev)
        except Exception as e:
            print(f"[MathOmni] ❌ Gagal menulis biner rewrite {formula_id}: {e}")
            return False

        # 2. Perbarui memori di RAM index dan sidecar file
        entry = self._find_entry(formula_id)
        if not entry:
            # Jika tidak ada di index (fallback), buat entry baru
            parts = formula_id.split("-")
            logic = parts[0][0] if len(parts) > 0 else "A"
            arousal = parts[0][1] if len(parts) > 0 and len(parts[0]) > 1 else "2"
            depth = parts[1] if len(parts) > 1 else "D5"
            entry = {
                "id":              formula_id,
                "logic":           logic,
                "arousal":         arousal,
                "depth":           depth,
                "trigger_text":    trigger_text,
                "instruction":     instruction,
                "usage_count":     0,
                "synaptic_weight": 1.0,
                "created_at":      time.time(),
            }
            self._save_entry_to_index(entry)
            return True

        # Update entry fields
        entry["trigger_text"] = trigger_text
        entry["instruction"] = instruction
        entry["synaptic_weight"] = 1.0  # Reset bobot
        entry["created_at"] = time.time()
        
        # Tulis ulang sidecar file
        if not self.sidecar.exists():
            return False
            
        try:
            lines = []
            with open(self.sidecar, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            with open(self.sidecar, "w", encoding="utf-8") as f:
                for line in lines:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    fid = data.get("id") or data.get("formula_id")
                    if fid == formula_id:
                        data["trigger_text"] = trigger_text
                        data["instruction"] = instruction
                        data["synaptic_weight"] = 1.0
                        data["created_at"] = entry["created_at"]
                        data["id"] = formula_id
                        if "formula_id" in data:
                            data["formula_id"] = formula_id
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            print(f"[MathOmni] 🔄 Rumus {formula_id} berhasil ditulis ulang (rewritten)!")
            return True
        except Exception as e:
            print(f"[MathOmni] ❌ Gagal menulis ulang sidecar rewrite {formula_id}: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════
    # INFO
    # ══════════════════════════════════════════════════════════════════

    @property
    def formula_count(self) -> int:
        return len(self._index)

    def list_formulas(self) -> List[Dict]:
        return list(self._index)


# ── Singleton Instance ─────────────────────────────────────────────────────────
from moko_config import settings
_math_omni_path = settings.WORKSPACE_DIR / ".math_omni"
math_omni = MathOmni(_math_omni_path)
