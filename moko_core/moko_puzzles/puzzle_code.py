import re
from moko_puzzles.base_puzzle import BasePuzzle
from moko_agents.llm_engine import engine

_PERSONAL_MARKERS = [
    "kenalin", "kenalkan", "nama saya", "nama aku", "namaku",
    "siapa kamu", "apa kabar", "halo", "hai", "selamat", "siapa moko"
]

class CodePuzzle(BasePuzzle):
    name = "code_lookup"
    description = "MOKO Coding & Programming Domain Explorer"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._rsa_code = None

    def _get_rsa_code(self):
        if self._rsa_code is None:
            from moko_memory.rsa_storage import RSAStorage
            self._rsa_code = RSAStorage(domain="code")
        return self._rsa_code

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        
        if any(marker in q for marker in _PERSONAL_MARKERS):
            return 0.0
            
        code_keywords = [
            "kode", "code", "program", "fungsi", "algoritma", "python", 
            "javascript", "bug", "error", "syntax", "class", "variable", 
            "loop", "array", "html", "css", "c++", "java", "database", 
            "sql", "git", "docker", "json", "api", "routing", "compilation", 
            "compiler"
        ]
        
        if any(kw in q for kw in code_keywords):
            return 0.92
            
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        target_query = self._clean_query(q)
        
        emb = engine.get_embedding(target_query)
        if not emb or len(emb) != 768:
            return {
                "facts": "Gagal menghasilkan representasi semantik untuk kueri coding.",
                "confidence": 0.0,
                "metadata": {}
            }
            
        rsa_code = self._get_rsa_code()
        search_results = rsa_code.search(emb, top_k=3, n_probe=8)
        
        if not search_results:
            return {
                "facts": f"Topik coding '{target_query}' tidak ditemukan di memori lokal.",
                "confidence": 0.0,
                "metadata": {"target": target_query}
            }
            
        fact_lines = []
        top_score = search_results[0].get("score", 0.0)
        
        for idx, res in enumerate(search_results, 1):
            if res.get("score", 0.0) >= 0.40:
                fact_lines.append(f"Fakta Code {idx} [Relevansi: {res['score']:.3f}]: {res['text']}")
                
        if not fact_lines:
            return {
                "facts": f"Tidak ada kecocokan referensi coding yang cukup kuat untuk '{target_query}'.",
                "confidence": top_score,
                "metadata": {"target": target_query}
            }
            
        # Boosted confidence untuk memicu OMNI_ENRICHED
        boosted_confidence = max(top_score, 0.85)
        
        return {
            "facts": f"Referensi Coding terverifikasi untuk '{target_query}':\n" + "\n".join(fact_lines),
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
            "bagaimana cara membuat", "buatkan kode", "bagaimana menulis",
            "contoh program", "cara mengatasi bug", "kenapa error",
            "bagaimana cara", "bagaimana membuat", "bagaimana", "buatlah"
        ]
        for phrase in stop_phrases:
            clean = re.sub(r'\b' + re.escape(phrase) + r'\b', '', clean)
            
        clean = clean.strip()
        return clean if clean else query_str
