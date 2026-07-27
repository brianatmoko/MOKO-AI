"""
MOKO NeuroMath: Dopamine Scheduler — Curiosity & Reward Engine
==============================================================
Berdasarkan: Dopaminergic Prediction Error Theory (Schultz, 1997)
             Intrinsic Motivation & Curiosity Drive (Oudeyer & Kaplan, 2007)
             Temporal Difference Learning (Sutton & Barto, 1998)

Di otak, sistem dopamine mengontrol MOTIVASI belajar:
  - Dopamine dilepaskan saat menemukan sesuatu BARU dan BERMANFAAT
  - Dopamine berkurang saat hal sudah FAMILIAR (habituation)
  - Sistem ini secara otomatis mengarahkan perhatian ke area yang
    paling menjanjikan untuk dipelajari (optimal curiosity zone)

DopamineScheduler di MOKO menggantikan `_generate_internal_monologue()`
yang sebelumnya memanggil LLM untuk memutuskan "apa yang ingin dipelajari".

Sekarang keputusan ini DETERMINISTIK dan TANPA LLM:
  - Setiap domain memiliki curiosity_score
  - Score tinggi = banyak novelty ditemukan di domain ini belakangan
  - Score rendah = domain sudah terlalu familiar atau terlalu sering dikunjungi
  - Domain dengan score tertinggi dipilih berikutnya (greedy + exploration)
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Optional

from moko_config import settings


# ── Domain Registry ───────────────────────────────────────────────────────────

DOMAINS = {
    "Mathematics": {
        "queries": [
            "advanced calculus applications real world",
            "linear algebra machine learning",
            "number theory prime numbers proof",
            "differential equations fluid dynamics",
            "topology abstract mathematics",
            "probability theory bayesian inference",
        ],
        "arxiv_category": "math",
        "initial_score": 1.0,
    },
    "Fluid Dynamics": {
        "queries": [
            "navier stokes equations applications",
            "turbulence modeling computational fluid dynamics",
            "fluid mechanics reynolds number",
            "laminar turbulent flow transition",
            "computational fluid dynamics simulation",
        ],
        "arxiv_category": "physics.flu-dyn",
        "initial_score": 0.9,
    },
    "Artificial Intelligence": {
        "queries": [
            "large language model architecture breakthrough",
            "neural network optimization techniques",
            "reinforcement learning recent advances",
            "transformer attention mechanism",
            "AI reasoning emergent capabilities",
        ],
        "arxiv_category": "cs.AI",
        "initial_score": 1.0,
    },
    "Physics": {
        "queries": [
            "quantum mechanics wave function collapse",
            "general relativity spacetime curvature",
            "thermodynamics entropy second law",
            "electromagnetism maxwell equations",
            "particle physics standard model",
        ],
        "arxiv_category": "physics",
        "initial_score": 0.8,
    },
    "Computer Science": {
        "queries": [
            "algorithm complexity big-o notation",
            "data structure graph theory applications",
            "operating system memory management",
            "cryptography elliptic curve",
            "distributed systems consensus algorithm",
        ],
        "arxiv_category": "cs",
        "initial_score": 0.85,
    },
    "Neuroscience": {
        "queries": [
            "hippocampus memory consolidation sleep",
            "synaptic plasticity long-term potentiation",
            "neural oscillation theta gamma",
            "prefrontal cortex decision making",
            "dopamine reward prediction error",
        ],
        "arxiv_category": "q-bio.NC",
        "initial_score": 0.95,
    },
    "Logic & Philosophy": {
        "queries": [
            "modal logic possible worlds",
            "formal verification theorem proving",
            "epistemology knowledge justified belief",
            "philosophy of mind consciousness",
            "mathematical logic Godel incompleteness",
        ],
        "arxiv_category": "cs.LO",
        "initial_score": 0.7,
    },
    "Biology": {
        "queries": [
            "CRISPR gene editing applications",
            "protein folding AlphaFold",
            "evolutionary biology natural selection",
            "cellular metabolism biochemistry",
            "systems biology network analysis",
        ],
        "arxiv_category": "q-bio",
        "initial_score": 0.75,
    },
}

# ── Konstanta ─────────────────────────────────────────────────────────────────

DECAY_RATE         = 0.15   # Seberapa cepat reward domain berkurang (boredom)
NOVELTY_BOOST      = 0.20   # Reward saat menemukan konten novel di domain ini
BOREDOM_PENALTY    = 0.05   # Penalti jika konten tidak novel (familiar)
EXPLORATION_RATE   = 0.15   # Probabilitas memilih domain non-terbaik (exploration)
MIN_SCORE          = 0.10   # Skor minimum agar domain tidak "mati" sepenuhnya
MAX_SCORE          = 2.00   # Skor maksimum (Oja normalization)
VISIT_COOLDOWN_SEC = 300    # Detik minimal antar kunjungan domain yang sama


class DopamineScheduler:
    """
    Sistem reward berbasis dopamine untuk mengontrol curiosity Forager.

    Menggantikan `_generate_internal_monologue()` yang sebelumnya
    memanggil LLM untuk memutuskan domain berikutnya.

    Cara kerja:
      1. Setiap domain punya curiosity_score (float)
      2. Setelah membaca artikel, score diupdate berdasarkan novelty yang ditemukan
      3. Domain dengan score tertinggi dipilih berikutnya
      4. Exploration factor mencegah terjebak di satu domain
      5. Boredom decay memastikan rotasi alami antar domain
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.state_path = Path(
            workspace_dir or settings.WORKSPACE_DIR
        ) / ".math_omni" / "dopamine_state.json"
        self._state = self._load_state()

    # ── State Persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        """Load state dari disk. Inisialisasi dari DOMAINS jika belum ada."""
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                # Pastikan semua domain ada (mungkin ada domain baru)
                for domain, cfg in DOMAINS.items():
                    if domain not in raw["scores"]:
                        raw["scores"][domain] = cfg["initial_score"]
                        raw["visit_counts"][domain] = 0
                        raw["last_visited"][domain] = 0.0
                        raw["query_indices"][domain] = 0
                return raw
            except Exception:
                pass

        # Fresh state
        return {
            "scores":        {d: cfg["initial_score"] for d, cfg in DOMAINS.items()},
            "visit_counts":  {d: 0 for d in DOMAINS},
            "last_visited":  {d: 0.0 for d in DOMAINS},
            "query_indices": {d: 0 for d in DOMAINS},
            "total_articles_read": 0,
        }

    def _save_state(self):
        """Simpan state ke disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ── Core Logic ────────────────────────────────────────────────────────────

    def update_reward(self, domain: str, novelty_score: float):
        """
        Update curiosity score domain berdasarkan novelty yang ditemukan.

        Args:
            domain: Nama domain yang baru dibaca artikelnya.
            novelty_score: 0.0 (tidak novel) - 1.0 (sangat novel)

        Analogi dopamine:
            novelty tinggi → dopamine spike → perkuat motivasi domain ini
            novelty rendah → dopamine dip → kurangi motivasi (boredom)
        """
        if domain not in self._state["scores"]:
            return

        current = self._state["scores"][domain]

        # Dopamine signal: proporsional dengan novelty
        if novelty_score > 0.5:
            delta = NOVELTY_BOOST * novelty_score  # Reward
        else:
            delta = -BOREDOM_PENALTY * (1.0 - novelty_score)  # Penalty

        new_score = current + delta

        # Oja normalization: clamp ke [MIN_SCORE, MAX_SCORE]
        self._state["scores"][domain] = round(
            max(MIN_SCORE, min(MAX_SCORE, new_score)), 4
        )
        self._state["visit_counts"][domain] = self._state["visit_counts"].get(domain, 0) + 1
        self._state["last_visited"][domain] = time.time()
        self._state["total_articles_read"] = self._state.get("total_articles_read", 0) + 1
        self._save_state()

    def get_next_domain(self) -> tuple[str, str]:
        """
        Pilih domain berikutnya berdasarkan curiosity score.

        Implementasi: Softmax selection dengan exploration rate.
        - 85% waktu: pilih domain dengan skor tertinggi (exploitation)
        - 15% waktu: pilih domain secara acak berbobot (exploration)

        Juga mempertimbangkan visit_cooldown untuk mencegah domain
        yang baru saja dikunjungi dipilih ulang terlalu cepat.

        Returns:
            (domain_name, search_query) — tanpa LLM call sama sekali.
        """
        now = time.time()
        scores = self._state["scores"]
        last_visited = self._state["last_visited"]

        # Filter domain yang masih dalam cooldown
        eligible = {
            d: s for d, s in scores.items()
            if (now - last_visited.get(d, 0)) > VISIT_COOLDOWN_SEC
        }

        # Jika semua dalam cooldown, ambil yang paling lama tidak dikunjungi
        if not eligible:
            eligible = scores  # Fallback: abaikan cooldown

        # Exploration vs Exploitation
        if random.random() < EXPLORATION_RATE:
            # Exploration: weighted random berdasarkan skor (bukan greedy)
            domain_list = list(eligible.keys())
            weights = [eligible[d] for d in domain_list]
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            chosen_domain = random.choices(domain_list, weights=weights, k=1)[0]
        else:
            # Exploitation: pilih domain terbaik
            chosen_domain = max(eligible, key=lambda d: eligible[d])

        query = self._get_next_query(chosen_domain)
        return chosen_domain, query

    def _get_next_query(self, domain: str) -> str:
        """Ambil query berikutnya dari rotasi queries domain."""
        cfg = DOMAINS.get(domain, {})
        queries = cfg.get("queries", [f"advanced {domain.lower()} research"])

        idx = self._state["query_indices"].get(domain, 0)
        query = queries[idx % len(queries)]

        # Rotate ke query berikutnya
        self._state["query_indices"][domain] = (idx + 1) % len(queries)
        return query

    def apply_boredom_decay(self):
        """
        Terapkan peluruhan pasif pada semua domain (habituation).
        Panggil ini setelah setiap siklus tidur (SLEEP mode).

        Analog: setelah tidur, otak "lupa" sebagian reward kemarin
        agar besok bisa terbuka terhadap hal baru lagi.
        """
        for domain in self._state["scores"]:
            current = self._state["scores"][domain]
            # Decay proporsional: domain populer meluruh lebih lambat
            decayed = current * (1.0 - DECAY_RATE)
            self._state["scores"][domain] = round(
                max(MIN_SCORE, decayed), 4
            )
        self._save_state()

    def get_monologue(self, domain: str, query: str) -> str:
        """
        Hasilkan teks monolog batin yang meyakinkan untuk ditampilkan di UI.
        Ini BUKAN LLM call — teks dihasilkan secara template deterministik.
        """
        score = self._state["scores"].get(domain, 1.0)
        visits = self._state["visit_counts"].get(domain, 0)

        if visits == 0:
            feeling = "aku sama sekali belum pernah mengeksplorasi bidang ini"
        elif score > 1.5:
            feeling = "setiap kali aku membaca tentang ini, aku selalu menemukan hal yang mengejutkan dan baru"
        elif score < 0.5:
            feeling = "aku mulai merasa bosan dengan topik ini, mungkin sudah waktunya kembali dengan perspektif segar"
        else:
            feeling = f"aku sudah membaca {visits} artikel tentang ini, tapi masih merasa ada yang belum kupahami"

        templates = [
            f"Sejujurnya, {feeling}. Aku ingin menggali lebih dalam tentang {domain}, spesifiknya tentang {query}.",
            f"Pikiranku tidak bisa berhenti memikirkan {domain}. Terutama tentang '{query}'. {feeling.capitalize()}.",
            f"Ada rasa penasaran yang mengganggu tentang {domain}. {feeling.capitalize()}. Aku harus mencari tentang '{query}'.",
        ]
        return random.choice(templates)

    def get_status_report(self) -> dict:
        """Laporan status curiosity semua domain untuk UI."""
        return {
            "scores": dict(self._state["scores"]),
            "visit_counts": dict(self._state["visit_counts"]),
            "total_articles_read": self._state.get("total_articles_read", 0),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
dopamine_scheduler = DopamineScheduler()
