"""
MOKO NeuroMath: CLS Consolidator
==================================
Berdasarkan: Complementary Learning Systems Theory
             McClelland, McNaughton & O'Reilly, 1995
             + Memory Consolidation during Sleep (Stickgold, 2005)

Proses Konsolidasi Asinkron: Hippocampus → Neocortex
(MOKO "Tidur Malam" — memadatkan pengalaman mentah menjadi prinsip abstrak)

CARA KERJA:
  1. Memindai entri terbaru di Omni-Index (.moko_omni/) — seperti Hippocampus
     memindai memori episodik dari hari ini.
  2. Mengelompokkan entri yang memiliki rute RSA (kode pos) yang berdekatan
     menjadi "cluster pengalaman".
  3. Setiap cluster dikirim ke LLM untuk diekstrak menjadi satu prinsip
     abstrak (formula Math-Omni baru) — seperti Neocortex mengekstrak
     pengetahuan umum dari banyak episode spesifik.
  4. Prinsip baru disimpan ke Math-Omni dengan bobot sinapsis awal yang
     sedikit lebih tinggi dari formula manual (karena berasal dari pengalaman nyata).

CATATAN: Jalankan ini sebagai proses terpisah (cron job atau QTimer background)
         JANGAN jalankan saat MOKO sedang aktif menjawab user.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict

from moko_agents.llm_engine import engine
from moko_memory.math_omni import math_omni
from moko_config import settings

MAX_ENTRIES_PER_RUN = 20  # Ambil N entri Omni terbaru per sesi konsolidasi
MIN_CLUSTER_SIZE = 2       # Minimal 2 entri sejenis untuk layak dikonsolidasi

CONSOLIDATION_PROMPT = """
Kamu adalah sistem Konsolidasi Memori MOKO (CLS Consolidator).
Berikut adalah kumpulan pengalaman faktual (Hippocampal Memory) yang MOKO rekam:

{cluster_text}

Tugasmu adalah:
1. Ekstrak satu PRINSIP ABSTRAK yang mendasari seluruh pengalaman di atas.
2. Tentukan Kategori Logika yang paling tepat (A/B/C/D/E/F/G/H).
3. Tulis instruksi berpikir generik berdasarkan prinsip tersebut.

KEMBALIKAN HANYA OBJEK JSON:
{{
  "prinsip_abstrak": "Prinsip yang bisa diterapkan ke situasi serupa di masa depan...",
  "logic": "E",
  "arousal": "2",
  "depth": "D9",
  "instruction": "Instruksi cara berpikir generik..."
}}
"""


class CLSConsolidator:
    """
    Daemon Konsolidasi Memori: Malam hari AI MOKO.
    """

    def __init__(self):
        self.omni_dir = Path(settings.WORKSPACE_DIR) / ".moko_crypto"
        self.log_path = Path(settings.WORKSPACE_DIR) / ".math_omni" / "cls_consolidation_log.jsonl"

    def _get_recent_omni_entries(self) -> List[Dict]:
        """Ambil N entri Omni terbaru yang belum dikonsolidasi."""
        all_entries = []
        if not self.omni_dir.exists():
            return []

        # Pindai semua file WAL atau index di .moko_omni
        wal_path = Path(settings.WORKSPACE_DIR) / ".moko_wal.jsonl"
        if not wal_path.exists():
            return []

        with open(wal_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                except:
                    pass

        # Ambil N terbaru, filter yang belum dikonsolidasi
        already_consolidated = self._get_consolidated_keys()
        fresh = [e for e in all_entries if e.get("key") not in already_consolidated]

        # Urutkan berdasarkan timestamp terbaru
        fresh.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return fresh[:MAX_ENTRIES_PER_RUN]

    def _get_consolidated_keys(self) -> set:
        """Baca log konsolidasi untuk menghindari duplikasi."""
        keys = set()
        if not self.log_path.exists():
            return keys
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        keys.update(data.get("consolidated_keys", []))
                    except:
                        pass
        return keys

    def _cluster_by_route(self, entries: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Kelompokkan entri berdasarkan prefix rute RSA (2 digit pertama kode pos).
        Ini mengelompokkan memori yang "sejenis secara tematik".
        """
        clusters: Dict[str, List[Dict]] = {}
        for entry in entries:
            route = str(entry.get("route", entry.get("postal_route", "ZZ")))
            prefix = route[:2] if len(route) >= 2 else route
            if prefix not in clusters:
                clusters[prefix] = []
            clusters[prefix].append(entry)
        return clusters

    def run_consolidation(self) -> List[str]:
        """
        Eksekusi konsolidasi satu sesi.
        Mengembalikan daftar formula_id yang baru tercipta.
        """
        new_formula_ids = []
        entries = self._get_recent_omni_entries()

        if not entries:
            print("[CLS] Tidak ada memori baru untuk dikonsolidasi.")
            return []

        clusters = self._cluster_by_route(entries)
        print(f"[CLS] Ditemukan {len(entries)} entri dalam {len(clusters)} cluster.")

        for prefix, cluster_entries in clusters.items():
            if len(cluster_entries) < MIN_CLUSTER_SIZE:
                continue  # Belum cukup pengalaman untuk membentuk prinsip

            # Susun teks cluster
            cluster_text = "\n\n---\n\n".join([
                e.get("text", e.get("chunk", ""))[:500]
                for e in cluster_entries[:5]
            ])

            # LLM: Ekstrak prinsip abstrak
            prompt = CONSOLIDATION_PROMPT.format(cluster_text=cluster_text)
            response = engine.generate_text(prompt, "Return JSON only.",
                                             model_override=settings.MODEL_ANALYST)

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                print(f"[CLS] Cluster {prefix}: Gagal mengekstrak JSON.")
                continue

            try:
                data = json.loads(json_match.group(0))
                prinsip = data.get("prinsip_abstrak", "")
                logic = data.get("logic", "A")
                arousal = str(data.get("arousal", "2"))
                depth = data.get("depth", "D5")
                instruction = data.get("instruction", "")

                if not instruction:
                    continue

                # Cari embedding untuk prinsip
                emb = engine.get_embedding(prinsip)
                if len(emb) != 768:
                    continue

                # Simpan ke Math-Omni dengan initial weight lebih tinggi
                formula_id = math_omni.save_formula(
                    logic, arousal, depth, prinsip, instruction, emb
                )

                # Set bobot sinapsis awal lebih tinggi (1.5) karena dari pengalaman nyata
                self._boost_initial_weight(formula_id)

                new_formula_ids.append(formula_id)
                print(f"[CLS] ✨ Prinsip baru dari cluster {prefix}: {formula_id}")

                # Catat ke log konsolidasi
                self._log_consolidation(
                    formula_id,
                    [e.get("key", "") for e in cluster_entries]
                )

            except Exception as e:
                print(f"[CLS] Error di cluster {prefix}: {e}")
                continue

        print(f"[CLS] Konsolidasi selesai. {len(new_formula_ids)} formula baru.")
        return new_formula_ids

    def _boost_initial_weight(self, formula_id: str):
        """Tingkatkan bobot awal formula hasil konsolidasi (karena lebih terpercaya)."""
        sidecar = Path(settings.WORKSPACE_DIR) / ".math_omni" / "formula_sidecar.jsonl"
        if not sidecar.exists(): return
        lines = sidecar.read_text(encoding='utf-8').splitlines()
        with open(sidecar, 'w', encoding='utf-8') as f:
            for line in lines:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("id") == formula_id:
                    data["synaptic_weight"] = 1.5  # CLS Boost
                    data["source"] = "cls_consolidation"
                f.write(json.dumps(data) + "\n")

    def _log_consolidation(self, formula_id: str, source_keys: list):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "formula_id": formula_id,
                "consolidated_keys": source_keys,
                "timestamp": time.time()
            }) + "\n")


# Singleton
cls_consolidator = CLSConsolidator()
