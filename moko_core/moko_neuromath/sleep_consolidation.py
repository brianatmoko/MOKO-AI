"""
MOKO NeuroMath: Sleep Consolidation Worker — Fase 2
=====================================================
Berdasarkan:
  - Neuron (2025): "Large sharp-wave ripples promote hippocampo-cortical
    memory reactivation and consolidation during sleep"
  - Cell Press (2024): "Prefrontal cortical ripples mediate top-down
    suppression of hippocampal reactivation during sleep memory consolidation"
  - Tononi & Cirelli: Synaptic Homeostasis Hypothesis (SHY)
  - McClelland et al. (1995): Complementary Learning Systems (CLS) Theory

CARA KERJA (4 Fase, analog NREM sleep):
  Fase 1 — SWR Replay (Sharp-Wave Ripple Simulation):
    Ambil memori terbaru dari WAL, kelompokkan yang mirip (sim > 0.6),
    perkuat LTP di setiap cluster (Hebbian Re-linking).

  Fase 2 — Synaptic Homeostasis (SHY Downscaling):
    Terapkan global multiplicative LTD (W *= 0.98) pada semua Hebbian
    assemblies KECUALI yang baru di-replay (dilindungi dari pruning).

  Fase 3 — Cross-Domain Interleaving:
    Sample acak dari berbagai domain, jalankan Hebbian linking lintas-domain
    untuk membangun koneksi abstrak (analog schema formation).

  Fase 4 — Memory Transfer (HPC → Neokorteks):
    Memori yang sudah di-replay > 3 kali ditandai sebagai "consolidated"
    di sidecar JSON. Ini analog dengan transfer hippocampus -> neokorteks.

Trigger: Dijalankan dari SleepWorker (existing) ketika sleep_scheduler
         mengaktifkan sleep cycle.
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from moko_config import settings

# Import SOM untuk Adam-adaptive Hebb learning rate
try:
    from moko_neuromath.self_optimization_math import OptimizationMath
    _SOM_OK = True
except ImportError:
    _SOM_OK = False
    OptimizationMath = None


# ── Konstanta ─────────────────────────────────────────────────────────────────

SWR_SIMILARITY_THRESHOLD   = 0.60   # Batas sim untuk cluster replay
SWR_MAX_ENTRIES            = 50     # Ambil maks N entri dari WAL per sesi
HOMEOSTASIS_SCALE_FACTOR   = 0.98   # W_new = W_old * 0.98 (LTD global)
CONSOLIDATED_REPLAY_COUNT  = 3      # Berapa kali replay sebelum "consolidated"
MIN_CLUSTER_SIZE           = 2      # Minimal entri dalam cluster untuk di-replay
INTERLEAVE_SAMPLE_PER_DOMAIN = 3    # Sample per domain untuk interleaving


# ── Helper: Cosine Similarity (tanpa numpy) ────────────────────────────────────

def _cosine(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot  = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ── Kelas Utama ────────────────────────────────────────────────────────────────

class SleepConsolidationWorker:
    """
    Worker konsolidasi memori — dijalankan saat sistem idle / sleep cycle.

    Cara pakai:
        worker = SleepConsolidationWorker()
        report = worker.run_full_consolidation()
        print(report)
    """

    def __init__(self):
        workspace = Path(settings.WORKSPACE_DIR)
        self.wal_path          = workspace / ".moko_wal.jsonl"
        self.hebb_path         = workspace / ".math_omni" / "hebb_assemblies.jsonl"
        self.sidecar_dir       = workspace / ".moko_rsa"
        self.consolidation_log = workspace / ".math_omni" / "sleep_consolidation_log.jsonl"

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def run_full_consolidation(self) -> Dict:
        """
        Jalankan semua 4 fase konsolidasi.
        Mengembalikan laporan hasil lengkap.
        """
        report = {
            "timestamp":        time.time(),
            "fase1_swr":        {},
            "fase2_homeostasis":{},
            "fase3_interleave": {},
            "fase4_transfer":   {},
        }

        print("[SLEEP] ═══════════════════════════════════")
        print("[SLEEP] Memulai Siklus Konsolidasi Memori")
        print("[SLEEP] ═══════════════════════════════════")

        # Fase 1 — SWR Replay
        report["fase1_swr"] = self._fase1_swr_replay()

        # Fase 2 — Synaptic Homeostasis
        report["fase2_homeostasis"] = self._fase2_synaptic_homeostasis(
            protected_keys=report["fase1_swr"].get("replayed_keys", set())
        )

        # Fase 3 — Cross-Domain Interleaving
        report["fase3_interleave"] = self._fase3_cross_domain_interleave()

        # Fase 4 — Memory Transfer (HPC → Neokorteks)
        report["fase4_transfer"] = self._fase4_memory_transfer()

        # --- Fase 4: Cognitive Map Update ---
        try:
            from moko_neuromath.cognitive_map import cognitive_map_builder
            from moko_neuromath.hebb_linker import hebb_linker
            
            # Ambil assemblies yang ada untuk memperbarui graf konsep
            assemblies = hebb_linker._load_assemblies()
            for a in assemblies:
                route = a.get("omni_route")
                fid = a.get("formula_id")
                w = a.get("weight", 0.0)
                if route and fid and w > 0.1:
                    cognitive_map_builder.add_concept_link(route, fid, w)
            
            # Deteksi skema konseptual baru
            schemas = cognitive_map_builder.detect_schemas(threshold=0.5)
            print(f"[SLEEP] Cognitive Map ter-update. Terdeteksi {len(schemas)} skema konseptual aktif.")
            report["fase4_transfer"]["schemas_detected"] = len(schemas)
        except Exception as e:
            print(f"[SLEEP] Gagal memperbarui peta kognitif: {e}")

        # --- Fase 5 (SUKES): Weight Adapter Fine-Tuning ---
        try:
            from moko_neuromath.sleep_finetuner import get_finetuner
            finetuner = get_finetuner(verbose=True)
            tune_res = finetuner.consolidate(dry_run=False)
            print(f"[SLEEP] SUKES Fine-tuning: status={tune_res.get('status')} | samples={tune_res.get('samples_count')}")
            report["fase5_sukes"] = tune_res
        except Exception as e:
            print(f"[SLEEP] Gagal menjalankan SUKES fine-tuning: {e}")
            report["fase5_sukes"] = {"status": "error", "error": str(e)}

        # --- Fase 6: LoRA Weight Update — MOKO belajar data MOKO baru ---
        try:
            import subprocess, sys
            from pathlib import Path
            
            lora_trainer_path = Path(settings.WORKSPACE_DIR) / "finetune" / "lora_trainer.py"
            finetune_venv_python = Path(settings.WORKSPACE_DIR) / "finetune" / "venv" / "bin" / "python"
            
            if lora_trainer_path.exists() and finetune_venv_python.exists():
                print("[SLEEP] Fase 6: Memicu LoRA Weight Update MOKO...")
                # Jalankan di background agar tidak blokir sleep cycle
                proc = subprocess.Popen(
                    [str(finetune_venv_python), str(lora_trainer_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True  # Detach dari proses induk
                )
                print(f"[SLEEP] LoRA Trainer berjalan di background (PID: {proc.pid})")
                report["fase6_lora"] = {"status": "triggered", "pid": proc.pid}
            else:
                report["fase6_lora"] = {"status": "skipped", "reason": "finetune pipeline belum tersedia"}
        except Exception as e:
            print(f"[SLEEP] Gagal memicu LoRA trainer: {e}")
            report["fase6_lora"] = {"status": "error", "error": str(e)}

        # Simpan log
        self._log_consolidation(report)

        total_new = (
            report["fase1_swr"].get("clusters_replayed", 0) +
            report["fase3_interleave"].get("cross_links_formed", 0) +
            report["fase4_transfer"].get("memories_consolidated", 0)
        )
        print(f"[SLEEP] ✅ Konsolidasi selesai. {total_new} operasi kognitif baru.")
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1: SHARP-WAVE RIPPLE (SWR) REPLAY
    # ══════════════════════════════════════════════════════════════════════════

    def _fase1_swr_replay(self) -> Dict:
        """
        Simulasi Sharp-Wave Ripple:
        Ambil entri WAL terbaru, cluster yang mirip, perkuat Hebbian linking.
        """
        print("[SLEEP][Fase 1] SWR Replay — Membaca WAL...")

        entries = self._read_wal_entries(SWR_MAX_ENTRIES)
        if not entries:
            print("[SLEEP][Fase 1] WAL kosong. Skip.")
            return {"clusters_replayed": 0, "entries_processed": 0, "replayed_keys": set()}

        # Cluster berdasarkan similaritas embedding
        clusters = self._cluster_by_similarity(entries, SWR_SIMILARITY_THRESHOLD)

        replayed_keys  = set()
        clusters_count = 0

        for cluster_entries in clusters:
            if len(cluster_entries) < MIN_CLUSTER_SIZE:
                continue

            # Perkuat Hebbian linking di dalam cluster (SWR Replay)
            self._strengthen_cluster_hebb(cluster_entries)
            clusters_count += 1

            for e in cluster_entries:
                key = e.get("key", e.get("route", ""))
                if key:
                    replayed_keys.add(key)

            print(f"[SLEEP][Fase 1] Cluster {clusters_count}: "
                  f"{len(cluster_entries)} entri di-replay (SWR).")

        result = {
            "entries_processed": len(entries),
            "clusters_found":    len(clusters),
            "clusters_replayed": clusters_count,
            "replayed_keys":     replayed_keys,
        }
        print(f"[SLEEP][Fase 1] ✓ {clusters_count} cluster di-replay dari "
              f"{len(entries)} entri WAL.")
        return result

    def _read_wal_entries(self, limit: int = SWR_MAX_ENTRIES) -> List[Dict]:
        """Baca N entri terbaru dari WAL."""
        if not self.wal_path.exists():
            return []
        entries = []
        try:
            with open(self.wal_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return []
        # Urutkan terbaru, ambil N teratas
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return entries[:limit]

    def _cluster_by_similarity(self, entries: List[Dict],
                                threshold: float) -> List[List[Dict]]:
        """
        Greedy clustering berdasarkan cosine similarity embedding.
        Setiap entri bergabung ke cluster pertama yang memiliki
        similarity > threshold terhadap centroid cluster.
        """
        clusters: List[List[Dict]] = []
        centroid_embs: List[List[float]] = []

        for entry in entries:
            emb = entry.get("embedding") or entry.get("vector")
            if not emb or len(emb) < 10:
                # Tidak ada embedding, masukkan ke cluster terpisah
                clusters.append([entry])
                centroid_embs.append([0.0])
                continue

            assigned = False
            for i, centroid in enumerate(centroid_embs):
                if centroid == [0.0]:
                    continue
                sim = _cosine(emb, centroid)
                if sim >= threshold:
                    clusters[i].append(entry)
                    # Update centroid (moving average)
                    n = len(clusters[i])
                    centroid_embs[i] = [
                        (c * (n - 1) + e) / n
                        for c, e in zip(centroid, emb)
                    ]
                    assigned = True
                    break

            if not assigned:
                clusters.append([entry])
                centroid_embs.append(list(emb))

        return clusters

    def _strengthen_cluster_hebb(self, cluster_entries: List[Dict]):
        """
        Perkuat semua pasangan Hebbian di dalam cluster:
        Analog neuron yang firing bersama → wire together lebih kuat.
        """
        if not self.hebb_path.exists():
            return

        # Kumpulkan route IDs dari semua entri cluster
        routes = [
            str(e.get("route", e.get("postal_route", e.get("key", ""))))
            for e in cluster_entries
        ]
        routes = [r for r in routes if r]

        if len(routes) < 2:
            return

        # Load assemblies
        assemblies = []
        try:
            with open(self.hebb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            assemblies.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return

        # Buat set pasangan route yang ada di cluster
        cluster_route_set = set(routes)

        # ── Adam-Adaptive Hebb Learning Rate (SOM Integration) ───────────────
        # LR setiap link dihitung secara adaptif menggunakan satu step Adam.
        # Link yang sering di-replay (gradient tinggi berulang) → momentum tinggi → LR terkontrol.
        # Link yang jarang di-replay → momentum rendah → LR lebih agresif (boost lebih besar).
        # Ini mencegah saturasi dan memastikan konvergensi Banach.
        DEFAULT_ETA = 0.08  # Fallback jika SOM tidak tersedia
        ADAM_LR     = 0.08  # Base learning rate Adam
        ADAM_BETA1  = 0.90  # Momen pertama (mean)
        ADAM_BETA2  = 0.99  # Momen kedua (variance)
        ADAM_EPS    = 1e-8

        for a in assemblies:
            if a.get("omni_route") in cluster_route_set:
                old_w        = a.get("weight", 0.1)
                replay_count = a.get("replay_count", 0) + 1

                # Hitung gradient proxy: seberapa jauh bobot dari target (1.0)
                # g = dL/dw di mana L = (1 - w)^2 → g = -2*(1 - w)
                grad = -(1.0 - old_w)  # Gradient turun ke arah w=1.0

                if _SOM_OK and replay_count >= 1:
                    # Ambil Adam moments dari metadata assembly
                    adam_m = a.get("adam_m", 0.0)
                    adam_v = a.get("adam_v", 0.0)
                    t      = replay_count

                    # Satu step Adam
                    new_params, new_m, new_v = OptimizationMath.adam_step(
                        params=[old_w],
                        grads=[grad],
                        m=[adam_m],
                        v=[adam_v],
                        t=t,
                        lr=ADAM_LR,
                        beta1=ADAM_BETA1,
                        beta2=ADAM_BETA2,
                        eps=ADAM_EPS
                    )
                    # Adam meng-update ke arah minimum loss (w→1.0)
                    # new_params[0] = w_old + |Adam_step| (selalu naik karena grad negatif)
                    new_w    = min(1.0, max(old_w, new_params[0]))
                    eta_used = abs(new_w - old_w)

                    # Simpan moments untuk sesi berikutnya
                    a["adam_m"] = round(new_m[0], 6)
                    a["adam_v"] = round(new_v[0], 8)
                else:
                    # Fallback: gunakan eta tetap
                    eta_used = DEFAULT_ETA
                    new_w    = min(1.0, old_w + eta_used * (1.0 - old_w))

                a["weight"]        = round(new_w, 4)
                a["last_replayed"] = time.time()
                a["replay_count"]  = replay_count

        # Simpan balik
        try:
            with open(self.hebb_path, 'w', encoding='utf-8') as f:
                for a in assemblies:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2: SYNAPTIC HOMEOSTASIS (SHY DOWNSCALING)
    # ══════════════════════════════════════════════════════════════════════════

    def _fase2_synaptic_homeostasis(self, protected_keys: set) -> Dict:
        """
        Terapkan global multiplicative LTD:
          W_new = W_old * HOMEOSTASIS_SCALE_FACTOR
        Kecuali link yang baru saja di-replay (dilindungi).
        Mencegah saturasi sinapsis → sistem tidak "lupa kontras".
        """
        print("[SLEEP][Fase 2] Synaptic Homeostasis — LTD Global...")

        if not self.hebb_path.exists():
            print("[SLEEP][Fase 2] Hebb assemblies kosong. Skip.")
            return {"links_scaled": 0, "links_protected": 0}

        assemblies = []
        try:
            with open(self.hebb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            assemblies.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return {"links_scaled": 0, "links_protected": 0}

        scaled    = 0
        protected = 0

        for a in assemblies:
            route = a.get("omni_route", "")
            # Lindungi link yang baru di-replay
            if route in protected_keys:
                protected += 1
                continue
            # Juga lindungi link yang baru di-replay hari ini
            last_replayed = a.get("last_replayed", 0)
            if (time.time() - last_replayed) < 3600:  # < 1 jam yang lalu
                protected += 1
                continue
            # Terapkan LTD global
            old_w = a.get("weight", 0.1)
            new_w = old_w * HOMEOSTASIS_SCALE_FACTOR
            a["weight"] = round(max(0.005, new_w), 4)
            scaled += 1

        # Simpan
        try:
            with open(self.hebb_path, 'w', encoding='utf-8') as f:
                for a in assemblies:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        except Exception:
            pass

        print(f"[SLEEP][Fase 2] ✓ {scaled} link di-downscale, "
              f"{protected} link dilindungi dari homeostasis.")

        # Terapkan Synaptic Pruning & E/I Balance scaling dari HebbLinker
        try:
            from moko_neuromath.hebb_linker import hebb_linker
            pruned_count = hebb_linker.apply_pruning(min_weight=0.05, inactive_days=30.0)
            ei_scaling_factor = hebb_linker.monitor_and_scale_ei_balance()
            print(f"[SLEEP][Fase 2] Plastisitas Struktural: {pruned_count} link dipangkas, E/I Scaling factor: {ei_scaling_factor:.2f}")
        except Exception as e:
            print(f"[SLEEP][Fase 2] Gagal menerapkan plastisitas struktural: {e}")

        return {"links_scaled": scaled, "links_protected": protected}

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3: CROSS-DOMAIN INTERLEAVING
    # ══════════════════════════════════════════════════════════════════════════

    def _fase3_cross_domain_interleave(self) -> Dict:
        """
        Ambil sampel dari berbagai domain RSA, cari koneksi lintas-domain.
        Analog: memory interleaving saat NREM mencegah catastrophic forgetting
        dan mendukung generalisasi.
        """
        print("[SLEEP][Fase 3] Cross-Domain Interleaving...")

        # Kumpulkan semua sidecar JSON domain
        cross_links = 0

        if not self.sidecar_dir.exists():
            print("[SLEEP][Fase 3] RSA sidecar directory tidak ada. Skip.")
            return {"cross_links_formed": 0}

        domain_samples: Dict[str, List[Dict]] = {}

        # Baca sampel dari setiap domain sidecar
        for sidecar_file in self.sidecar_dir.glob("*_sidecar.jsonl"):
            domain_name = sidecar_file.stem.replace("_sidecar", "")
            samples = []
            try:
                with open(sidecar_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                samples.append(json.loads(line))
                            except Exception:
                                pass
            except Exception:
                continue

            # Ambil sampel terbaru (N per domain)
            samples.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            domain_samples[domain_name] = samples[:INTERLEAVE_SAMPLE_PER_DOMAIN]

        if len(domain_samples) < 2:
            print("[SLEEP][Fase 3] Kurang dari 2 domain tersedia. Skip.")
            return {"cross_links_formed": 0}

        # Cross-link: bandingkan antar domain
        domain_names = list(domain_samples.keys())

        assemblies = []
        if self.hebb_path.exists():
            try:
                with open(self.hebb_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                assemblies.append(json.loads(line))
                            except Exception:
                                pass
            except Exception:
                pass

        existing_keys = {a.get("key", "") for a in assemblies}

        for i in range(len(domain_names)):
            for j in range(i + 1, len(domain_names)):
                dom_a = domain_names[i]
                dom_b = domain_names[j]

                for entry_a in domain_samples[dom_a]:
                    for entry_b in domain_samples[dom_b]:
                        route_a = str(entry_a.get("route", entry_a.get("postal_route", "")))
                        route_b = str(entry_b.get("route", entry_b.get("postal_route", "")))

                        if not route_a or not route_b:
                            continue

                        # Bentuk cross-domain link baru
                        cross_key = f"XDOMAIN:{dom_a}:{route_a}|{dom_b}:{route_b}"
                        if cross_key in existing_keys:
                            continue

                        # Link dengan bobot awal rendah (baru terbentuk)
                        assemblies.append({
                            "key":            cross_key,
                            "omni_route":     f"{dom_a}:{route_a}",
                            "formula_id":     f"{dom_b}:{route_b}",
                            "weight":         0.03,
                            "link_type":      "cross_domain",
                            "domain_a":       dom_a,
                            "domain_b":       dom_b,
                            "activation_count": 1,
                            "last_activated": time.time(),
                            "created_at":     time.time(),
                        })
                        existing_keys.add(cross_key)
                        cross_links += 1

        # Simpan
        if cross_links > 0:
            try:
                with open(self.hebb_path, 'w', encoding='utf-8') as f:
                    for a in assemblies:
                        f.write(json.dumps(a, ensure_ascii=False) + "\n")
            except Exception:
                pass

        print(f"[SLEEP][Fase 3] ✓ {cross_links} koneksi lintas-domain baru terbentuk.")
        return {"cross_links_formed": cross_links, "domains_connected": len(domain_names)}

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 4: MEMORY TRANSFER (HPC → NEOKORTEKS)
    # ══════════════════════════════════════════════════════════════════════════

    def _fase4_memory_transfer(self) -> Dict:
        """
        Tandai memori yang sudah di-replay >= CONSOLIDATED_REPLAY_COUNT
        sebagai "consolidated" di sidecar JSON.
        Analog transfer Hippocampus → Neokorteks (long-term stable memory).
        """
        print("[SLEEP][Fase 4] Memory Transfer HPC → Neokorteks...")

        if not self.hebb_path.exists():
            print("[SLEEP][Fase 4] Tidak ada Hebb assemblies. Skip.")
            return {"memories_consolidated": 0}

        # Load assemblies untuk cek replay_count
        assemblies = []
        try:
            with open(self.hebb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            assemblies.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return {"memories_consolidated": 0}

        # Identifikasi yang sudah matang untuk "transfer"
        mature_routes = set()
        for a in assemblies:
            if a.get("replay_count", 0) >= CONSOLIDATED_REPLAY_COUNT:
                route = a.get("omni_route", "")
                if route and "XDOMAIN" not in route:
                    mature_routes.add(route)

        if not mature_routes:
            print("[SLEEP][Fase 4] Belum ada memori yang matang untuk transfer.")
            return {"memories_consolidated": 0}

        # Update sidecar JSON: tandai sebagai consolidated
        consolidated_count = 0
        if self.sidecar_dir.exists():
            for sidecar_file in self.sidecar_dir.glob("*_sidecar.jsonl"):
                try:
                    lines = []
                    with open(sidecar_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                route = str(entry.get("route",
                                            entry.get("postal_route", "")))
                                if route in mature_routes:
                                    prev = entry.get("consolidated_count", 0)
                                    entry["consolidated_count"] = prev + 1
                                    entry["memory_type"]        = "semantic"
                                    entry["hpc_transfer_ts"]    = time.time()
                                    consolidated_count += 1
                                lines.append(json.dumps(entry, ensure_ascii=False))
                            except Exception:
                                lines.append(line)
                    with open(sidecar_file, 'w', encoding='utf-8') as f:
                        for l in lines:
                            f.write(l + "\n")
                except Exception:
                    continue

        print(f"[SLEEP][Fase 4] ✓ {consolidated_count} entri memori "
              f"ditransfer ke Neokorteks (tagged as 'semantic + consolidated').")
        return {
            "memories_consolidated": consolidated_count,
            "mature_routes":         len(mature_routes),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    def _log_consolidation(self, report: Dict):
        """Catat hasil konsolidasi ke log file."""
        try:
            log_entry = {
                "timestamp":           report["timestamp"],
                "entries_processed":   report["fase1_swr"].get("entries_processed", 0),
                "clusters_replayed":   report["fase1_swr"].get("clusters_replayed", 0),
                "links_scaled":        report["fase2_homeostasis"].get("links_scaled", 0),
                "cross_links":         report["fase3_interleave"].get("cross_links_formed", 0),
                "memories_transferred":report["fase4_transfer"].get("memories_consolidated", 0),
            }
            self.consolidation_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.consolidation_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_last_consolidation_stats(self) -> Optional[Dict]:
        """Kembalikan statistik konsolidasi terakhir."""
        if not self.consolidation_log.exists():
            return None
        last = None
        try:
            with open(self.consolidation_log, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            last = json.loads(line)
                        except Exception:
                            pass
        except Exception:
            pass
        return last


# ── Singleton ──────────────────────────────────────────────────────────────────
sleep_consolidation_worker = SleepConsolidationWorker()
