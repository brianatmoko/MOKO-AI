import re
from moko_puzzles.base_puzzle import BasePuzzle
from moko_agents.llm_engine import engine
from moko_neuromath.math_cas_engine import math_cas
from moko_neuromath.mcts_reasoner import MCTSMathReasoner

_PERSONAL_MARKERS = [
    "kenalin", "kenalkan", "nama saya", "nama aku", "namaku",
    "siapa kamu", "apa kabar", "halo", "hai", "selamat", "siapa moko"
]

class MathPuzzle(BasePuzzle):
    name = "math_lookup"
    description = "MOKO Mathematical Domain Explorer & Solver"
    version = "2.0.0"

    def __init__(self):
        super().__init__()
        self._rsa_math = None
        self._reasoner = None

    def _get_rsa_math(self):
        if self._rsa_math is None:
            from moko_memory.rsa_storage import RSAStorage
            self._rsa_math = RSAStorage(domain="math")
        return self._rsa_math

    def _get_reasoner(self):
        if self._reasoner is None:
            # Lambda wrapper to run generation through engine
            def llm_gen(prompt, system):
                return engine.generate_text(
                    prompt=prompt,
                    system_prompt=system,
                    coop_params={"num_predict": 180, "enable_thinking": False}
                )
            self._reasoner = MCTSMathReasoner(llm_generate_fn=llm_gen, cas_engine=math_cas)
        return self._reasoner

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        
        if any(marker in q for marker in _PERSONAL_MARKERS):
            return 0.0
            
        physics_keywords = [
            "fisika", "gaya", "kecepatan", "energi", "massa", "newton", 
            "relativitas", "kuantum", "termodinamika", "elektromagnetik", 
            "gravitasi", "hukum newton", "gelombang", "frekuensi", "optik", 
            "cahaya", "bunyi", "listrik", "magnet"
        ]
        
        if any(pk in q for pk in physics_keywords):
            return 0.0

        # Check if CAS engine recognizes mathematical intent
        if math_cas and math_cas.can_compute(q):
            return 0.98

        math_keywords = [
            "hitung", "rumus", "matematika", "integral", "turunan", 
            "persamaan", "aljabar", "geometri", "probabilitas", "statistik", 
            "kalkulus", "logaritma", "perkalian", "pembagian", "pertambahan", 
            "pengurangan", "penjumlahan", "akar kuadrat", "faktorial", "prima",
            "fpb", "kpk", "gcd", "lcm", "matriks", "determinan", "kombinasi",
            "permutasi"
        ]
        
        if any(kw in q for kw in math_keywords):
            return 0.90
            
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        target_query = self._clean_query(q)
        
        # 1. Run MCTS Reasoner (which handles CAS internally first)
        reasoner = self._get_reasoner()
        mcts_result = reasoner.reason(target_query)
        
        # 2. Extract facts from local memory (RSA) as supplementary context
        emb = engine.get_embedding(target_query)
        rsa_facts = []
        if emb and len(emb) == 768:
            rsa_math = self._get_rsa_math()
            search_results = rsa_math.search(emb, top_k=2, n_probe=8)
            if search_results:
                for res in search_results:
                    if res.get("score", 0.0) >= 0.40:
                        rsa_facts.append(res['text'])

        # 3. Combine results
        cas_data = mcts_result.get("cas_result", {})
        cas_success = mcts_result.get("cas_verified", False)
        
        fact_lines = []
        if cas_success and cas_data.get("success"):
            fact_lines.append(f"Hasil Komputasi CAS (Deterministik): {mcts_result['answer']}")
            if cas_data.get("steps"):
                fact_lines.append("Langkah Komputasi:")
                for step in cas_data["steps"]:
                    fact_lines.append(f"  - {step}")
        else:
            # Fallback to MCTS steps
            fact_lines.append(f"Hasil Penawaran MCTS: {mcts_result['answer']}")
            if mcts_result.get("steps"):
                fact_lines.append("Langkah Penalaran (MCTS):")
                for s in mcts_result["steps"]:
                    fact_lines.append(f"  - {s.description} (Reward: {s.reward:.2f})")
                    if s.code_result:
                        fact_lines.append(f"    [Kode Terverifikasi: {s.code_result.strip()}]")

        if rsa_facts:
            fact_lines.append("\nReferensi Teori Tambahan:")
            for fact in rsa_facts:
                fact_lines.append(f"  - {fact}")

        # Boosted confidence
        boosted_confidence = 0.98 if cas_success else max(mcts_result.get("trajectory_score", 0.0), 0.85)

        return {
            "facts": "\n".join(fact_lines),
            "confidence": boosted_confidence,
            "metadata": {
                "target": target_query,
                "complexity": mcts_result.get("complexity", "unknown"),
                "iterations": mcts_result.get("iterations", 0),
                "time_ms": mcts_result.get("time_ms", 0.0),
                "cas_verified": cas_success,
                "cas_prompt_injection": mcts_result.get("cas_prompt_injection", "")
            }
        }

    def _clean_query(self, query_str: str) -> str:
        clean = query_str.replace("?", "").replace("!", "")
        stop_phrases = [
            "tolong jelaskan", "apa rumus", "bagaimana cara menghitung",
            "berapa hasil dari", "hitunglah", "rumus untuk", "cari rumus",
            "jelaskan rumus", "apa itu", "bagaimana konsep"
        ]
        for phrase in stop_phrases:
            clean = re.sub(r'\b' + re.escape(phrase) + r'\b', '', clean)
            
        clean = clean.strip()
        return clean if clean else query_str

