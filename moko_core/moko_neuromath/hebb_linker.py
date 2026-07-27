"""
MOKO NeuroMath: Hebb Linker + STDP
=====================================
Berdasarkan:
  - Hebb's Cell Assembly Theory (1949): Δw_ij = η · x_i · y_j
  - Spike-Timing-Dependent Plasticity (Bi & Poo, 1998; STDP):
    Jika pre fires SEBELUM post (Δt > 0): LTP — perkuat sinapsis
    Jika pre fires SETELAH post (Δt < 0): LTD — lemahkan sinapsis
    Magnitude: Δw = A_plus * exp(-Δt / tau_plus)  jika Δt > 0
               Δw = A_minus * exp(Δt / tau_minus)  jika Δt < 0

Di MOKO, "waktu" analog dengan POSISI dalam reasoning chain:
  - Jika konsep A muncul di posisi 0 dan B di posisi 1 (Δpos = 1):
    → LTP: A mengaktifkan B (A-before-B → perkuat A→B)
  - Jika A di posisi 2 dan B di posisi 0 (Δpos = -2):
    → LTD: B sudah ada sebelum A → lemahkan A→B, perkuat B→A

Timing buffer menyimpan urutan aktivasi dalam satu reasoning session.
FireTogether masih tersedia untuk ko-aktifasi sederhana (backward compat).
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple
from moko_config import settings

ETA               = 0.05    # Learning rate (η) — seberapa cepat hubungan menguat
DECAY_RATE        = 0.005   # Peluruhan pasif per-waktu (Forgetting Rate)
ASSEMBLY_THRESHOLD= 0.75   # Ambang batas: jika Assembly cukup kuat, percayai tanpa LLM

# ── STDP Constants (Bi & Poo, 1998) ──────────────────────────────────────────
STDP_A_PLUS   = 0.08    # Amplitudo LTP (pre-before-post)
STDP_A_MINUS  = 0.04    # Amplitudo LTD (post-before-pre)
STDP_TAU_PLUS = 4.0     # Time constant LTP (dalam satuan "posisi" reasoning)
STDP_TAU_MINUS= 4.0     # Time constant LTD
TIMING_BUFFER_SIZE = 20 # Ukuran buffer urutan aktivasi


class HebbLinker:
    """
    Mencatat dan memperkuat tautan antara omni_chunk_id dan formula_id.
    Kini mendukung STDP: kekuatan LTP/LTD ditentukan oleh urutan temporal
    aktivasi dalam satu reasoning session.
    """

    def __init__(self):
        self.assembly_path  = Path(settings.WORKSPACE_DIR) / ".math_omni" / "hebb_assemblies.jsonl"
        # timing_buffer: urutan (node_id, timestamp_posisi) dalam session aktif
        self._timing_buffer: deque = deque(maxlen=TIMING_BUFFER_SIZE)
        self._session_pos: int = 0   # Posisi sekuensial dalam reasoning chain

    def _load_assemblies(self) -> list:
        if not self.assembly_path.exists():
            return []
        assemblies = []
        with open(self.assembly_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        assemblies.append(json.loads(line))
                    except:
                        pass
        return assemblies

    def _save_assemblies(self, assemblies: list):
        with open(self.assembly_path, 'w', encoding='utf-8') as f:
            for a in assemblies:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # ── STDP Timing Buffer ────────────────────────────────────────────────────

    def begin_session(self):
        """Reset timing buffer untuk sesi reasoning baru."""
        self._timing_buffer.clear()
        self._session_pos = 0

    def record_activation(self, node_id: str):
        """
        Rekam bahwa node_id baru saja diaktifkan di posisi ini
        dalam reasoning chain. Panggil untuk setiap memori/formula
        yang diambil selama satu sesi.
        """
        self._timing_buffer.append((node_id, self._session_pos))
        self._session_pos += 1

    def stdp_update(self, pre_id: str, post_id: str) -> float:
        """
        Hitung perubahan bobot STDP antara pre_id dan post_id
        berdasarkan posisi relatif di timing_buffer.

        Δt = pos(post) - pos(pre)  (dalam satuan posisi reasoning)
        Δw = +A_plus  * exp(-Δt / tau_plus)   jika Δt > 0  (LTP)
           = -A_minus * exp( Δt / tau_minus)  jika Δt < 0  (LTD)
           = +ETA/2                            jika Δt == 0 (ko-aktifasi)

        Returns:
            float: delta_weight (bisa positif/negatif)
        """
        buf_dict = {nid: pos for nid, pos in self._timing_buffer}
        pos_pre  = buf_dict.get(pre_id)
        pos_post = buf_dict.get(post_id)

        if pos_pre is None or pos_post is None:
            return ETA  # fallback ke Hebb biasa

        delta_t = pos_post - pos_pre  # positif = pre before post (LTP)

        if delta_t > 0:
            dw = STDP_A_PLUS  * math.exp(-delta_t / STDP_TAU_PLUS)
        elif delta_t < 0:
            dw = -STDP_A_MINUS * math.exp( delta_t / STDP_TAU_MINUS)
        else:
            dw = ETA / 2.0

        return round(dw, 5)

    def apply_stdp_session(self):
        """
        Setelah sesi selesai, terapkan STDP untuk semua pasangan node
        yang diaktifkan berurutan dalam timing_buffer.
        Ini menggantikan fire_together untuk sesi yang punya konteks temporal.
        """
        buf = list(self._timing_buffer)
        if len(buf) < 2:
            return

        assemblies   = self._load_assemblies()
        assembly_map = {a.get("key"): a for a in assemblies}

        # Iterasi semua pasangan (pre, post) yang berurutan dalam buffer
        for i in range(len(buf)):
            for j in range(i + 1, min(i + 5, len(buf))):  # Jangkauan 5 posisi
                pre_id  = buf[i][0]
                post_id = buf[j][0]
                dw      = self.stdp_update(pre_id, post_id)

                key      = f"{pre_id}|{post_id}"
                existing = assembly_map.get(key)
                if existing:
                    old_w = existing.get("weight", 0.05)
                    new_w = min(1.0, max(0.005, old_w + dw))
                    existing["weight"]          = round(new_w, 4)
                    existing["last_activated"]  = time.time()
                    existing["activation_count"]= existing.get("activation_count", 1) + 1
                    existing["stdp_delta"]       = round(dw, 5)
                else:
                    init_w = max(0.005, ETA + dw)
                    new_entry = {
                        "key":              key,
                        "omni_route":       pre_id,
                        "formula_id":       post_id,
                        "weight":           round(init_w, 4),
                        "activation_count": 1,
                        "last_activated":   time.time(),
                        "created_at":       time.time(),
                        "stdp_delta":       round(dw, 5),
                        "link_type":        "stdp",
                    }
                    assemblies.append(new_entry)
                    assembly_map[key] = new_entry

        self._save_assemblies(assemblies)

    # ── Klasik Fire-Together (backward compatible) ────────────────────────────

    def fire_together(self, omni_route: str, formula_id: str, eta_override: float = None):
        """
        Ko-aktifasi klasik Hebbian (backward compatible).
        Untuk sesi dengan timing, gunakan record_activation + apply_stdp_session.
        
        Args:
            eta_override: Opsional. Multiplier dari Fase 3 (ACh encoding_boost × Cortisol
                          plasticity_penalty). Jika None, gunakan ETA default.
        """
        if not omni_route or not formula_id:
            return

        # Tentukan learning rate efektif
        effective_eta = ETA * (eta_override if eta_override is not None else 1.0)
        effective_eta = max(0.001, min(0.5, effective_eta))  # clamp aman

        assemblies = self._load_assemblies()
        key = f"{omni_route}|{formula_id}"

        existing = next((a for a in assemblies if a.get("key") == key), None)
        if existing:
            # LTP: perkuat tautan (Oja's normalization) dengan eta efektif
            old_w = existing.get("weight", 0.1)
            new_w = min(1.0, old_w + effective_eta * (1.0 - old_w))
            existing["weight"]          = round(new_w, 4)
            existing["last_activated"]  = time.time()
            existing["activation_count"]= existing.get("activation_count", 1) + 1
        else:
            # Tautan baru: gunakan eta efektif sebagai bobot awal
            assemblies.append({
                "key":              key,
                "omni_route":       omni_route,
                "formula_id":       formula_id,
                "weight":           round(effective_eta, 4),
                "activation_count": 1,
                "last_activated":   time.time(),
                "created_at":       time.time(),
            })

        self._save_assemblies(assemblies)

    def get_strongest_formula(self, omni_route: str) -> str | None:
        """
        Untuk omni_route tertentu, cari formula_id yang memiliki
        tautan Hebbianl paling kuat. Kembalikan None jika tidak ada yang kuat.
        """
        assemblies = self._load_assemblies()
        candidates = [a for a in assemblies if a.get("omni_route") == omni_route]

        if not candidates:
            return None

        best = max(candidates, key=lambda a: a.get("weight", 0))
        if best.get("weight", 0) >= ASSEMBLY_THRESHOLD:
            return best.get("formula_id")
        return None

    def apply_decay(self):
        """
        Peluruhan pasif: hubungan yang lama tidak diaktifkan akan melemah.
        Panggil ini dari apoptosis_daemon secara berkala.
        """
        now = time.time()
        assemblies = self._load_assemblies()
        surviving = []

        for a in assemblies:
            elapsed_hours = (now - a.get("last_activated", now)) / 3600.0
            w = a.get("weight", 0.1)
            w -= DECAY_RATE * elapsed_hours
            if w > 0.01:
                a["weight"] = round(max(0.01, w), 4)
                surviving.append(a)
            # Else: tautan mati, tidak disimpan (Natural Forgetting)

        self._save_assemblies(surviving)

    def apply_pruning(self, min_weight: float = 0.05, inactive_days: float = 30.0) -> int:
        """
        Synaptic Pruning: Hapus hubungan Hebbian yang bobotnya jatuh di bawah batas minimum
        atau sudah tidak diaktifkan dalam jangka waktu yang sangat lama.
        Mengembalikan jumlah link yang berhasil dipangkas.
        """
        now = time.time()
        assemblies = self._load_assemblies()
        pruned_count = 0
        surviving = []

        for a in assemblies:
            last_act = a.get("last_activated", now)
            elapsed_days = (now - last_act) / (24.0 * 3600.0)

            # Prune jika weight < min_weight ATAU tidak aktif > inactive_days
            if a.get("weight", 0.0) < min_weight or elapsed_days > inactive_days:
                pruned_count += 1
            else:
                surviving.append(a)

        if pruned_count > 0:
            self._save_assemblies(surviving)

        return pruned_count

    def monitor_and_scale_ei_balance(self) -> float:
        """
        E/I Balance Monitor: Mengukur rasio LTP (eksitasi) dan LTD (inhibisi)
        pada assemblies dan mengembalikan scaling factor yang diterapkan.
        """
        assemblies = self._load_assemblies()
        if not assemblies:
            return 1.0

        weights = [a.get("weight", 0.0) for a in assemblies]
        avg_weight = sum(weights) / len(weights)

        scaling_factor = 1.0
        if avg_weight > 0.7:
            # Over-excitation -> downscale
            scaling_factor = 0.90
        elif avg_weight < 0.1:
            # Over-inhibition/depression -> boost
            scaling_factor = 1.10

        if scaling_factor != 1.0:
            for a in assemblies:
                new_w = min(1.0, max(0.005, a.get("weight", 0.0) * scaling_factor))
                a["weight"] = round(new_w, 4)
            self._save_assemblies(assemblies)

        return scaling_factor


# Singleton
hebb_linker = HebbLinker()

