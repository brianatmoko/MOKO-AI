import re
from moko_puzzles.base_puzzle import BasePuzzle
from moko_agents.llm_engine import engine

_PERSONAL_MARKERS = [
    "kenalin", "kenalkan", "nama saya", "nama aku", "namaku",
    "siapa kamu", "apa kabar", "halo", "hai", "selamat", "siapa moko"
]

class GeneralPuzzle(BasePuzzle):
    name = "general_lookup"
    description = "MOKO General Knowledge & Fallback Explorer"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._rsa_general = None
        self._rsa_general_sub_1 = None

    def _get_rsa_general(self):
        if self._rsa_general is None:
            from moko_memory.rsa_storage import RSAStorage
            self._rsa_general = RSAStorage(domain="general")
        return self._rsa_general

    def _get_rsa_general_sub_1(self):
        if self._rsa_general_sub_1 is None:
            from moko_memory.rsa_storage import RSAStorage, get_domain_path
            # Hanya load jika direktori sub_1 memang ada
            sub1_path = get_domain_path("general_sub_1")
            if sub1_path.exists():
                self._rsa_general_sub_1 = RSAStorage(domain="general_sub_1")
        return self._rsa_general_sub_1

    def evaluate_suitability(self, query: str) -> float:
        q = query.lower().strip()
        
        # Obrolan personal tidak boleh dijawab lewat RAG/puzzle
        if any(marker in q for marker in _PERSONAL_MARKERS):
            return 0.0
            
        # Fallback suitability moderat agar didahului oleh puzzle domain yang lebih spesifik
        return 0.55

    def execute(self, query: str, context: dict) -> dict:
        q = query.lower().strip()
        target_query = self._clean_query(q)
        
        emb = engine.get_embedding(target_query)
        if not emb or len(emb) != 768:
            return {
                "facts": "Gagal menghasilkan representasi semantik untuk kueri umum.",
                "confidence": 0.0,
                "metadata": {}
            }
            
        # Cari di domain general
        rsa_gen = self._get_rsa_general()
        search_results = rsa_gen.search(emb, top_k=3, n_probe=8)
        
        # Cari di domain general_sub_1 jika aktif
        rsa_sub1 = self._get_rsa_general_sub_1()
        if rsa_sub1:
            try:
                sub1_results = rsa_sub1.search(emb, top_k=3, n_probe=8)
                if sub1_results:
                    search_results.extend(sub1_results)
            except Exception as e:
                print(f"[GeneralPuzzle] Error searching general_sub_1: {e}")
                
        # Urutkan berdasarkan score
        search_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        search_results = search_results[:3]
        
        if not search_results:
            return {
                "facts": f"Topik umum '{target_query}' tidak ditemukan di memori lokal.",
                "confidence": 0.0,
                "metadata": {"target": target_query}
            }
            
        fact_lines = []
        top_score = search_results[0].get("score", 0.0)
        
        for idx, res in enumerate(search_results, 1):
            if res.get("score", 0.0) >= 0.40:
                domain_src = res.get("domain", "general")
                fact_lines.append(f"Fakta Umum ({domain_src}) {idx} [Relevansi: {res['score']:.3f}]: {res['text']}")
                
        if not fact_lines:
            return {
                "facts": f"Tidak ada kecocokan referensi umum yang cukup kuat untuk '{target_query}'.",
                "confidence": top_score,
                "metadata": {"target": target_query}
            }
            
        # Jangan boost confidence secara artificial — gunakan skor asli dari pencarian vektor.
        # Boosting ke 0.80 menyebabkan entri memori rusak memicu OMNI_ENRICHED sehingga
        # jawaban mentah (termasuk blockchain footer) tampil langsung ke user tanpa filter LLM.
        # Biarkan OmniDirectAnswer memutuskan mode berdasarkan skor nyata.

        return {
            "facts": f"Referensi Umum terverifikasi untuk '{target_query}':\n" + "\n".join(fact_lines),
            "confidence": top_score,  # Skor asli, BUKAN boosted
            "metadata": {
                "target": target_query,
                "raw_score": top_score,
                "scores": [r["score"] for r in search_results]
            }
        }

    def _clean_query(self, query_str: str) -> str:
        clean = query_str.replace("?", "").replace("!", "")
        stop_phrases = [
            "tolong jelaskan tentang", "jelaskan tentang", "apa yang dimaksud dengan",
            "apa itu", "tolong jelaskan", "siapa", "dimana", "kapan"
        ]
        for phrase in stop_phrases:
            clean = re.sub(r'\b' + re.escape(phrase) + r'\b', '', clean)
            
        clean = clean.strip()
        return clean if clean else query_str
