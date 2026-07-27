"""
MOKO NeuroMath: Basal Ganglia System — Reinforcement Learning & Action Selection
================================================================================
Berdasarkan:
  - Schultz et al. (1997): Dopamine Reward Prediction Error (RPE)
  - Sutton & Barto (1998): Temporal Difference Learning & Q-Learning
  - Basal Ganglia loops (Striatum, Nucleus Accumbens, VTA)

FUNGSI:
  1. Striatum (Q-table): Menyimpan nilai kegunaan (Q-value) untuk aksi kognitif (domain, query).
  2. Nucleus Accumbens: Pengumpul reward kognitif (curiosity balance).
  3. RPE Calculator: delta = actual_reward - predicted_reward.
  4. Habit Cache: Jika RPE terus-menerus mendekati 0 untuk kueri tertentu, 
     pola tersebut di-cache sebagai habit kognitif otomatis (bypass LLM).
  5. Softmax Action Selector: Pemilihan aksi berbasis temperatur eksplorasi.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from moko_config import settings
from moko_neuromath.dopamine_scheduler import DOMAINS, DopamineScheduler

# ── Konstanta Reinforcement Learning ──────────────────────────────────────────
LEARNING_RATE_ALPHA = 0.15   # Laju update Q-learning (α)
DISCOUNT_FACTOR_GAMMA= 0.90  # Gamma (γ) untuk future reward discount
RPE_HABIT_THRESHOLD = 0.05   # Toleransi RPE mendekati 0 untuk habituation
HABIT_STREAK_TARGET = 3      # Berapa kali RPE rendah berturut-turut sebelum masuk habit cache
VISIT_COOLDOWN_SEC  = 300    # Detik minimal antar kunjungan domain yang sama
MAX_HABIT_SIZE      = 200


class BasalGangliaSystem(DopamineScheduler):
    """
    Basal Ganglia System — Peningkatan dari DopamineScheduler dengan reinforcement learning.
    Menerapkan Q-learning, RPE, dan Habit Cache.
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        # Inisialisasi state path
        workspace = Path(workspace_dir or settings.WORKSPACE_DIR)
        self.state_path = workspace / ".math_omni" / "basal_ganglia_state.json"
        self.habit_path = workspace / ".math_omni" / "habit_cache.jsonl"
        
        # Inisialisasi internal cache
        self.habit_cache: Dict[str, Dict] = {}
        self.rpe_streak: Dict[str, int] = {}  # query_hash -> streak count
        
        # Last actions
        self._last_domain: Optional[str] = None
        self._last_query: Optional[str] = None

        # Load Q-table & state
        self._state = self._load_state()
        self._load_habits()

    # ── State Management ──────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        """Load state, buat baru jika belum ada."""
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                # Validasi Q-table
                if "q_table" not in raw:
                    raw["q_table"] = {}
                if "nucleus_accumbens_balance" not in raw:
                    raw["nucleus_accumbens_balance"] = 0.0
                
                # Pastikan semua domain terdaftar di Q-table
                for domain, cfg in DOMAINS.items():
                    if domain not in raw["q_table"]:
                        raw["q_table"][domain] = {q: cfg["initial_score"] for q in cfg["queries"]}
                    else:
                        # Pastikan kueri baru masuk
                        for q in cfg["queries"]:
                            if q not in raw["q_table"][domain]:
                                raw["q_table"][domain][q] = cfg["initial_score"]
                return raw
            except Exception:
                pass

        # State Baru
        q_table = {}
        for domain, cfg in DOMAINS.items():
            q_table[domain] = {q: cfg["initial_score"] for q in cfg["queries"]}

        return {
            "q_table": q_table,
            "scores": {d: cfg["initial_score"] for d, cfg in DOMAINS.items()},
            "visit_counts": {d: 0 for d in DOMAINS},
            "last_visited": {d: 0.0 for d in DOMAINS},
            "nucleus_accumbens_balance": 0.0,
            "total_articles_read": 0,
        }

    def _save_state(self):
        """Simpan state ke disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass

    # ── Habit Cache Persistence ───────────────────────────────────────────────

    def _load_habits(self):
        """Muat habit cache dari disk."""
        if not self.habit_path.exists():
            return
        try:
            with open(self.habit_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            qh = entry.get("hash")
                            if qh:
                                self.habit_cache[qh] = entry
                        except Exception:
                            pass
        except Exception:
            pass

    def _save_habits(self):
        """Simpan habit cache ke disk."""
        try:
            self.habit_path.parent.mkdir(parents=True, exist_ok=True)
            sorted_habits = sorted(
                self.habit_cache.values(),
                key=lambda e: e.get("use_count", 0),
                reverse=True
            )[:MAX_HABIT_SIZE]
            with open(self.habit_path, "w", encoding="utf-8") as f:
                for entry in sorted_habits:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── Action Selection (Softmax over Q-values) ──────────────────────────────

    def get_next_domain(self, temperature: float = 0.5) -> Tuple[str, str]:
        """
        Pilih domain dan query berikutnya menggunakan Softmax atas Q-values.
        Modulasi temperatur (eksplorasi) dari SerotoninNode memengaruhi randomness.
        """
        now = time.time()
        q_table = self._state["q_table"]
        last_visited = self._state["last_visited"]

        # Filter domain dengan cooldown
        eligible_domains = [
            d for d in DOMAINS
            if (now - last_visited.get(d, 0)) > VISIT_COOLDOWN_SEC
        ]
        if not eligible_domains:
            eligible_domains = list(DOMAINS.keys())  # Fallback: bypass cooldown

        # 1. Hitung nilai Q rata-rata untuk setiap eligible domain
        domain_q_values = {}
        for d in eligible_domains:
            queries_q = q_table.get(d, {})
            if queries_q:
                domain_q_values[d] = sum(queries_q.values()) / len(queries_q)
            else:
                domain_q_values[d] = 0.5

        # 2. Terapkan Softmax untuk memilih domain
        chosen_domain = self._softmax_select(domain_q_values, temperature)

        # 3. Terapkan Softmax untuk memilih query dalam domain terpilih
        queries_q = q_table[chosen_domain]
        chosen_query = self._softmax_select(queries_q, temperature)

        # Simpan aksi terakhir
        self._last_domain = chosen_domain
        self._last_query = chosen_query

        return chosen_domain, chosen_query

    def _softmax_select(self, q_dict: Dict[str, float], temperature: float) -> str:
        """Helper Softmax Action Selection."""
        keys = list(q_dict.keys())
        q_vals = [q_dict[k] for k in keys]
        
        # Atasi temperatur mendekati 0 (pilih murni argmax/greedy)
        if temperature < 0.05:
            max_idx = q_vals.index(max(q_vals))
            return keys[max_idx]

        # Hitung probabilitas Softmax: exp(Q(a)/T) / sum(exp(Q(b)/T))
        try:
            exp_vals = [math.exp(q / temperature) for q in q_vals]
            sum_exp = sum(exp_vals)
            probs = [ev / sum_exp for ev in exp_vals]
            return random.choices(keys, weights=probs, k=1)[0]
        except OverflowError:
            # Fallback jika eksponensial meledak
            max_idx = q_vals.index(max(q_vals))
            return keys[max_idx]

    # ── Reward Prediction Error (RPE) & Q-Learning ────────────────────────────

    def update_reward(self, domain: str, novelty_score: float):
        """
        Update Q-value berdasarkan novelty (sebagai reward R)
        dan hitung Reward Prediction Error (RPE).
        """
        if not self._last_query or not self._last_domain:
            # Fallback jika dipanggil langsung tanpa get_next_domain
            effective_dom = domain if domain else "general"
            self._last_domain = effective_dom
            cfg = DOMAINS.get(effective_dom, {})
            self._last_query = cfg["queries"][0] if cfg.get("queries") else "general"

        dom = self._last_domain if self._last_domain else "general"
        qry = self._last_query if self._last_query else "general"

        # Pastikan domain dan query ada di Q-table untuk mencegah KeyError
        if dom not in self._state["q_table"]:
            self._state["q_table"][dom] = {}
        if qry not in self._state["q_table"][dom]:
            self._state["q_table"][dom][qry] = 0.5

        # Ambil perkiraan reward sebelumnya (predicted reward)
        predicted = self._state["q_table"][dom].get(qry, 0.5)

        # Reward aktual (novelty = reward)
        actual = novelty_score

        # Hitung RPE: delta = actual - predicted
        rpe = actual - predicted

        # TD Update: Q(s,a) = Q(s,a) + alpha * RPE
        new_q = predicted + LEARNING_RATE_ALPHA * rpe
        self._state["q_table"][dom][qry] = round(max(0.01, min(2.0, new_q)), 4)

        # Perbarui legacy score untuk kecocokan backward
        if self._state["q_table"][dom]:
            self._state["scores"][dom] = sum(self._state["q_table"][dom].values()) / len(self._state["q_table"][dom])

        # Akumulasikan reward di Nucleus Accumbens

        self._state["nucleus_accumbens_balance"] = round(
            self._state.get("nucleus_accumbens_balance", 0.0) + actual, 4
        )
        self._state["visit_counts"][dom] = self._state["visit_counts"].get(dom, 0) + 1
        self._state["last_visited"][dom] = time.time()
        self._state["total_articles_read"] += 1

        self._save_state()

        # ── Plastisitas Hebbian Tergantung RPE ──
        self._apply_rpe_to_hebbian(dom, rpe)

    def _apply_rpe_to_hebbian(self, domain: str, rpe: float):
        """
        Modulasi bobot Hebbian berdasarkan RPE:
          RPE > 0 (Lebih baik dari ekspektasi) -> LTP Boost
          RPE < 0 (Lebih buruk dari ekspektasi) -> LTD Pelemahan
        """
        try:
            from moko_neuromath.hebb_linker import hebb_linker
            assemblies = hebb_linker._load_assemblies()
            modified = False
            
            # Cari link Hebbian yang berkaitan dengan domain ini
            for a in assemblies:
                route = a.get("omni_route", "")
                if domain.lower() in route.lower() or route.lower() in domain.lower():
                    old_w = a.get("weight", 0.1)
                    # LTP/LTD proporsional dengan RPE
                    dw = rpe * 0.05
                    new_w = min(1.0, max(0.005, old_w + dw))
                    a["weight"] = round(new_w, 4)
                    a["last_activated"] = time.time()
                    modified = True
            
            if modified:
                hebb_linker._save_assemblies(assemblies)
        except Exception:
            pass

    # ── Habit Cache Management ────────────────────────────────────────────────

    def get_habit_hit(self, query_hash: str) -> Optional[str]:
        """Periksa apakah kueri ini merupakan kebiasaan otomatis (habit cache hit)."""
        entry = self.habit_cache.get(query_hash)
        if entry:
            entry["use_count"] = entry.get("use_count", 0) + 1
            entry["last_used"] = time.time()
            self._save_habits()
            return entry.get("response")
        return None

    def record_actual_for_habit(self, query_hash: str, query_text: str, response_text: str):
        """
        Evaluasi kueri untuk dimasukkan ke habit cache.
        Jika RPE kueri saat ini sangat rendah (dekat 0, artinya model sudah 
        sepenuhnya memperkirakan hasil), tambah streak. Setelah N-streak -> jadikan habit.
        """
        # Ambil RPE kueri terakhir
        if not self._last_domain or not self._last_query:
            return
            
        predicted = self._state["q_table"][self._last_domain].get(self._last_query, 0.5)
        # Sederhanakan: asumsikan kueri berulang stabil memiliki RPE -> 0
        rpe = 0.0  # default
        
        # Hitung deviasi ekspektasi: jika nilai Q sudah tinggi (>0.80)
        if predicted > 0.80:
            streak = self.rpe_streak.get(query_hash, 0) + 1
            self.rpe_streak[query_hash] = streak
            
            if streak >= HABIT_STREAK_TARGET:
                # Masukkan ke habit cache
                self.habit_cache[query_hash] = {
                    "hash": query_hash,
                    "query": query_text,
                    "response": response_text,
                    "use_count": 1,
                    "created_at": time.time(),
                    "last_used": time.time()
                }
                self._save_habits()
                self.rpe_streak[query_hash] = 0  # reset
        else:
            self.rpe_streak[query_hash] = 0  # reset

    # ── Diagnostics & Helpers ─────────────────────────────────────────────────

    def get_status_report(self) -> dict:
        """Legacy status report + Q-learning parameters."""
        scores = {}
        for d, queries in self._state["q_table"].items():
            scores[d] = sum(queries.values()) / len(queries)
            
        return {
            "scores": scores,
            "visit_counts": dict(self._state["visit_counts"]),
            "total_articles_read": self._state.get("total_articles_read", 0),
            "nucleus_accumbens_balance": self._state.get("nucleus_accumbens_balance", 0.0),
            "habit_cache_size": len(self.habit_cache)
        }

    def apply_boredom_decay(self):
        """Habituation: Turunkan Q-table secara bertahap saat tidur kognitif."""
        DECAY = 0.12
        for domain in self._state["q_table"]:
            for query in self._state["q_table"][domain]:
                old_q = self._state["q_table"][domain][query]
                self._state["q_table"][domain][query] = round(max(0.1, old_q * (1.0 - DECAY)), 4)
        
        #legacy score decay
        for domain in self._state["scores"]:
            self._state["scores"][domain] = round(max(0.1, self._state["scores"][domain] * (1.0 - DECAY)), 4)
            
        self._save_state()


# Singleton Instance
basal_ganglia = BasalGangliaSystem()
