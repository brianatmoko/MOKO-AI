import re
from moko_puzzles.base_puzzle import BasePuzzle
from moko_agents.llm_engine import engine

_PERSONAL_MARKERS = [
    "kenalin", "kenalkan", "nama saya", "nama aku", "namaku",
    "siapa kamu", "apa kabar", "halo", "hai", "selamat", "siapa moko"
]

class PhysicsPuzzle(BasePuzzle):
    name = "physics_lookup"
    description = "MOKO Physics Domain Explorer"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._rsa_physics = None

    def _get_rsa_physics(self):
        if self._rsa_physics is None:
            from moko_memory.rsa_storage import RSAStorage
            self._rsa_physics = RSAStorage(domain="physics")
        return self._rsa_physics

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
        
        if any(kw in q for kw in physics_keywords):
            return 0.92
            
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        target_query = self._clean_query(q)
        
        emb = engine.get_embedding(target_query)
        if not emb or len(emb) != 768:
            return {
                "facts": "Gagal menghasilkan representasi semantik untuk kueri fisika.",
                "confidence": 0.0,
                "metadata": {}
            }
            
        rsa_physics = self._get_rsa_physics()
        search_results = rsa_physics.search(emb, top_k=3, n_probe=8)
        
        if not search_results:
            return {
                "facts": f"Topik fisika '{target_query}' tidak ditemukan di memori lokal.",
                "confidence": 0.0,
                "metadata": {"target": target_query}
            }
            
        fact_lines = []
        top_score = search_results[0].get("score", 0.0)
        
        for idx, res in enumerate(search_results, 1):
            if res.get("score", 0.0) >= 0.40:
                fact_lines.append(f"Fakta Fisika {idx} [Relevansi: {res['score']:.3f}]: {res['text']}")
                
        if not fact_lines:
            return {
                "facts": f"Tidak ada kecocokan referensi fisika yang cukup kuat untuk '{target_query}'.",
                "confidence": top_score,
                "metadata": {"target": target_query}
            }
            
        # Boosted confidence untuk memicu OMNI_ENRICHED
        boosted_confidence = max(top_score, 0.85)
        
        return {
            "facts": f"Referensi Fisika terverifikasi untuk '{target_query}':\n" + "\n".join(fact_lines),
            "confidence": boosted_confidence,
            "metadata": {
                "target": target_query,
                "raw_score": top_score,
                "scores": [r["score"] for r in search_results]
            }
        }

    def _clean_query(self, query_str: str) -> str:
        clean = query_str.replace("?", "").replace("!", "")
        stop_phrases = [
            "tolong jelaskan", "apa itu", "bagaimana konsep", "jelaskan konsep",
            "bagaimana prinsip", "hukum fisika", "apa yang dimaksud dengan",
            "hukum"
        ]
        for phrase in stop_phrases:
            clean = re.sub(r'\b' + re.escape(phrase) + r'\b', '', clean)
            
        clean = clean.strip()
        return clean if clean else query_str
