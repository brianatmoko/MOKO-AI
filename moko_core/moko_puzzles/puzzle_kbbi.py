import re
from moko_puzzles.base_puzzle import BasePuzzle
from moko_agents.llm_engine import engine

# Kata-kata yang menandakan obrolan personal (tidak butuh KBBI)
_PERSONAL_MARKERS = [
    "kenalin", "kenalkan", "nama saya", "nama aku", "namaku",
    "siapa kamu", "apa kabar", "halo", "hai", "selamat", "siapa moko"
]

class KBBIPuzzle(BasePuzzle):
    name = "kbbi_lookup"
    description = "KBBI Lexical Database Explorer"
    version = "1.2.0"

    def __init__(self):
        super().__init__()
        self._rsa_lexical = None  # Lazy-load agar tidak double WAL saat import

    def _get_rsa_lexical(self):
        """Lazy-load RSAStorage domain lexical — mencegah double DiskManager init."""
        if self._rsa_lexical is None:
            from moko_memory.rsa_storage import RSAStorage
            self._rsa_lexical = RSAStorage(domain="lexical")
        return self._rsa_lexical

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        
        # Obrolan personal — tidak perlu KBBI lookup sama sekali
        if any(marker in q for marker in _PERSONAL_MARKERS):
            return 0.0
        
        # Kata kunci pencarian makna/kamus — sangat cocok
        kbbi_keywords = ["arti", "definisi", "makna", "kamus", "apa itu", "artinya"]
        if any(kw in q for kw in kbbi_keywords):
            return 0.95
            
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        
        # Ekstrak kata target dari kalimat (misal: "apa arti dari sistem" -> "sistem")
        target_word = self._clean_query(q)
        
        # Cari embedding kata target
        word_emb = engine.get_embedding(target_word)
        
        if not word_emb or len(word_emb) != 768:
            return {
                "facts": "Gagal menghasilkan representasi semantik untuk kata target.",
                "confidence": 0.0,
                "metadata": {}
            }
            
        # Cari di OMNI spesifik domain 'lexical' untuk kecepatan optimal
        rsa_lexical = self._get_rsa_lexical()
        search_results = rsa_lexical.search(word_emb, top_k=3, n_probe=8)
        
        if not search_results:
            return {
                "facts": f"Kata '{target_word}' tidak ditemukan di database KBBI lokal.",
                "confidence": 0.0,
                "metadata": {"target": target_word}
            }
            
        # Susun fakta dari hasil pencarian
        fact_lines = []
        top_score = search_results[0].get("score", 0.0)
        
        for idx, res in enumerate(search_results, 1):
            if res.get("score", 0.0) >= 0.40:
                fact_lines.append(f"Definisi {idx}: {res['text']}")
                
        if not fact_lines:
            return {
                "facts": f"Tidak ada kecocokan makna yang cukup kuat untuk '{target_word}'.",
                "confidence": top_score,
                "metadata": {"target": target_word}
            }
            
        # Naikkan confidence ke 0.80 (di atas threshold OMNI_ENRICHED=0.72)
        # karena KBBI adalah sumber definitif faktual yang akurat
        boosted_confidence = max(top_score, 0.80)
        
        return {
            "facts": f"Hasil Kamus Besar Bahasa Indonesia (KBBI) untuk kata '{target_word}':\n" + "\n".join(fact_lines),
            "confidence": boosted_confidence,
            "metadata": {
                "target": target_word,
                "raw_score": top_score,
                "scores": [r["score"] for r in search_results]
            }
        }

    def _clean_query(self, query_str: str) -> str:
        """Membersihkan query untuk menyisakan kata kunci yang dicari."""
        clean = query_str
        # Hilangkan tanda tanya
        clean = clean.replace("?", "").replace("!", "")
        # Hilangkan kata tanya penolong
        stop_phrases = [
            "apa arti kata", "apa arti dari", "apa arti", "artinya", 
            "tolong artikan", "apa definisi dari", "apa definisi", 
            "makna kata", "makna dari", "apakah itu", "apa itu", "kamus"
        ]
        for phrase in stop_phrases:
            clean = re.sub(r'\b' + re.escape(phrase) + r'\b', '', clean)
            
        clean = clean.strip()
        # Jika hasil kosong, fallback ke query asli
        return clean if clean else query_str
