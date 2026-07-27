"""
MOKO Agent: DefaultModeNetwork — Introspection, Self-reflection & Mental Simulation
===================================================================================
Berdasarkan:
  - Nature Neuroscience (2025): Default Mode Network (DMN) cytoarchitecture & signal flow.
  - Raichle (2015): Systematic overview of Default Mode Network.
  - Smallwood et al. (2021): Mind-wandering as a cognitive resource.
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from moko_config import settings
from moko_agents.llm_engine import engine

# Import SOM untuk entropy-guided topic selection
try:
    from moko_neuromath.self_optimization_math import InformationTheory
    _SOM_OK = True
except ImportError:
    _SOM_OK = False
    InformationTheory = None

class DefaultModeNetwork:
    """
    Default Mode Network (DMN) — Modul Simulasi Introspeksi Internal.
    Aktif ketika sistem terdeteksi idle (mind-wandering).
    """
    def __init__(self, workspace_dir: Optional[str] = None):
        workspace = Path(workspace_dir or settings.WORKSPACE_DIR)
        self.state_path = workspace / ".math_omni" / "dmn_state.json"
        
        # Load state
        self._state = self._load_state()
        
    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if "self_model" not in state:
                    state["self_model"] = {}
                if "autobiographical_index" not in state:
                    state["autobiographical_index"] = []
                if "theory_of_mind" not in state:
                    state["theory_of_mind"] = {}
                return state
            except Exception:
                pass
                
        return {
            "self_model": {
                "total_memories_processed": 0,
                "error_rate": 0.0,
                "success_count": 0,
                "fail_count": 0,
                "domain_distribution": {},
                "last_active": time.time()
            },
            "autobiographical_index": [
                {"timestamp": time.time(), "event": "MOKO DMN Node Initialized."}
            ],
            "theory_of_mind": {
                "user_favorite_topics": [],
                "complexity_preference": "medium",
                "interaction_count": 0
            }
        }
        
    def save_state(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def record_activity(self, success: bool = True, domain: str = ""):
        """Mencatat aktivitas dari kognisi luar ke self-model."""
        self._state["self_model"]["total_memories_processed"] += 1
        if success:
            self._state["self_model"]["success_count"] += 1
        else:
            self._state["self_model"]["fail_count"] += 1
            
        total = self._state["self_model"]["success_count"] + self._state["self_model"]["fail_count"]
        if total > 0:
            self._state["self_model"]["error_rate"] = round(self._state["self_model"]["fail_count"] / total, 4)
            
        if domain:
            dist = self._state["self_model"]["domain_distribution"]
            dist[domain] = dist.get(domain, 0) + 1
            
        self._state["self_model"]["last_active"] = time.time()
        self.save_state()

    def record_user_query(self, query: str):
        """Mengidentifikasi topik kueri user untuk membangun Theory of Mind."""
        self._state["theory_of_mind"]["interaction_count"] += 1
        words = [w.lower() for w in query.split() if len(w) > 4 and w not in ["adalah", "untuk", "dalam", "dengan", "yang"]]
        # Track kata kunci teratas yang paling sering ditanyakan
        fav_topics = self._state["theory_of_mind"].get("user_favorite_topics", [])
        for w in words:
            if w not in fav_topics:
                fav_topics.append(w)
        if len(fav_topics) > 10:
            fav_topics = fav_topics[-10:]
        self._state["theory_of_mind"]["user_favorite_topics"] = fav_topics
        
        # Kompleksitas
        if len(query) > 100 or any(w in query.lower() for w in ["jelaskan", "bagaimana", "mengapa"]):
            self._state["theory_of_mind"]["complexity_preference"] = "high"
        else:
            self._state["theory_of_mind"]["complexity_preference"] = "medium"
            
        self.save_state()

    def log_event(self, event_description: str):
        """Mencatat peristiwa penting ke Autobiographical Index."""
        self._state["autobiographical_index"].append({
            "timestamp": time.time(),
            "event": event_description
        })
        # Batasi autobiographical index ke 100 entri
        if len(self._state["autobiographical_index"]) > 100:
            self._state["autobiographical_index"] = self._state["autobiographical_index"][-100:]
        self.save_state()

    def _entropy_weighted_topic_selection(self, fav_topics: list) -> str:
        """
        Pilih topik introspeksi berdasarkan Shannon Entropy dari distribusi domain DMN.

        Strategi:
          - Domain yang underexplored (frekuensi rendah) mendapat bobot lebih tinggi
          - Ini memaksimalkan mutual information dari siklus introspeksi berikutnya
          - Analog: hippocampal-driven memory reactivation memprioritaskan memori langka

        Fallback ke random.choice jika SOM tidak tersedia atau distribusi kosong.
        """
        if not _SOM_OK or not fav_topics:
            return random.choice(fav_topics) if fav_topics else "matematika"

        domain_dist = self._state["self_model"].get("domain_distribution", {})

        if not domain_dist or len(domain_dist) < 2:
            # Distribusi belum cukup kaya untuk analisis entropi — pakai random
            return random.choice(fav_topics)

        # Hitung "rarity score" per topik berdasarkan domain yang paling jarang
        # Topic dengan domain jarang = lebih berharga untuk dieksplorasi
        total_queries = sum(domain_dist.values())

        # Map fav_topics ke domain terdekat (prefix matching)
        topic_weights = []
        for topic in fav_topics:
            # Cari apakah topik ini ada di domain yang dikenal
            topic_lower = topic.lower()
            matched_count = 0
            for domain, count in domain_dist.items():
                if domain.lower() in topic_lower or topic_lower in domain.lower():
                    matched_count = count
                    break

            if matched_count > 0:
                # Topik yang domain-nya sudah banyak dikunjungi → bobot rendah (sudah jenuh)
                freq = matched_count / total_queries
                # Rarity = 1 - freq → topik jarang lebih disukai
                weight = max(0.05, 1.0 - freq)
            else:
                # Domain baru / tidak dikenal → bobot maksimum (sangat informatif)
                weight = 1.0

            topic_weights.append(weight)

        # Hitung entropi distribusi bobot untuk log
        total_w = sum(topic_weights)
        if total_w > 0 and _SOM_OK:
            probs = [w / total_w for w in topic_weights]
            try:
                h = InformationTheory.entropy(probs, base=2.0)
            except Exception:
                h = 0.0
        else:
            h = 0.0

        # Weighted random selection
        rand_val = random.uniform(0, total_w)
        cumulative = 0.0
        selected = fav_topics[-1]  # Default ke topik terakhir
        for topic, weight in zip(fav_topics, topic_weights):
            cumulative += weight
            if rand_val <= cumulative:
                selected = topic
                break

        return selected

    def run_introspection_cycle(self, disk_manager) -> Optional[Dict[str, Any]]:
        """
        Menjalankan satu siklus introspeksi kognitif (mind-wandering).
        Membaca memori secara acak, mensimulasikan penalaran 'bagaimana-jika',
        dan mengembalikan wawasan baru yang dihasilkan.
        """
        # Ambil sampel memori dari DiskManager/RSA — dipilih secara entropy-weighted
        fav_topics = self._state["theory_of_mind"].get("user_favorite_topics", ["kecerdasan", "logika", "sistem", "otak", "memori"])
        if not fav_topics:
            fav_topics = ["logika", "sains", "komputer", "matematika"]

        # ── Entropy-Guided Topic Selection (SOM Integration) ────────────────
        # Prioritaskan topik dari domain yang paling underexplored (mutual info tinggi)
        search_query = self._entropy_weighted_topic_selection(fav_topics)
        emb = engine.get_embedding(search_query)
        if len(emb) != 768:
            return None
            
        results = disk_manager.search_memory(emb, top_k=3)
        if not results:
            return None
            
        memories = [r["text"] for r in results]
        
        # Mental Simulation Prompt
        combined_memories = "\n".join([f"- {m}" for m in memories])
        prompt = (
            f"Kamu adalah MOKO Default Mode Network (DMN Simulator).\n"
            f"Di bawah ini adalah beberapa memori internal yang sedang mengambang di pikiran bawah sadarmu:\n"
            f"{combined_memories}\n\n"
            f"Tugasmu: Jalankan introspeksi kreatif. Hubungkan memori-memori di atas dan buatlah satu pertanyaan hipotesis 'Bagaimana jika...' yang menantang, "
            f"lahu jawab sendiri pertanyaan tersebut secara mendalam. Tuliskan analisis pemikiranmu.\n"
            f"Kembalikan respons dalam format JSON (tanpa markdown):\n"
            f"{{\n"
            f"  \"mental_scenario\": \"Pertanyaan bagaimana-jika...\",\n"
            f"  \"insight\": \"Jawaban / analisis mendalam...\"\n"
            f"}}"
        )
        
        try:
            response = engine.generate_text(prompt, "Return JSON only.", model_override=settings.MODEL_ANALYST)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                scenario = data.get("mental_scenario", "")
                insight = data.get("insight", "")
                
                # Log event
                self.log_event(f"Introspeksi skenario: '{scenario[:60]}...'")
                
                # Simpan wawasan introspeksi baru ini ke Omni-Index secara otonom
                insight_text = f"[DMN INTROSPECTION] Skenario: {scenario}\nWawasan: {insight}"
                insight_emb = engine.get_embedding(insight)
                if len(insight_emb) == 768:
                    disk_manager.ingest_chunk(
                        "dmn_introspection.txt",
                        insight_text,
                        insight_emb,
                        source_type="introspection",
                        domain="general"
                    )
                
                return {
                    "scenario": scenario,
                    "insight": insight,
                    "timestamp": time.time()
                }
        except Exception as e:
            self.log_event(f"Gagal dalam siklus introspeksi: {str(e)}")
            
        return None

# Singleton DMN instance
dmn_node = DefaultModeNetwork()
