"""
MOKO RAG Retriever
==================
Unified retrieval interface untuk MOKO RAG system.

Pipeline:
  query (str)
    → translate_query_id_to_en()          [ID→EN keyword translation, <1ms]
    → moko_embed_engine.embed()           [local, <20ms]
    → OmniHashEncoder.encode()            [SimHash routing]
    → OmniVectorStore.search_by_hamming() [Hamming + FP16 cosine]
    → List[RagChunk]                      [teks yang siap digunakan]

Usage:
    from moko_memory.moko_rag_retriever import MokoRagRetriever
    retriever = MokoRagRetriever()
    chunks = retriever.retrieve("how to prevent sql injection", top_k=5)
    context = retriever.format_context(chunks)
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np


# ─── Indonesian → English Query Translation ────────────────────────────────────
# Kamus terjemahan kata kunci penting Bahasa Indonesia → Bahasa Inggris
# Digunakan untuk menjembatani gap bahasa saat mencari di OMNI index (English)
_ID_TO_EN: Dict[str, str] = {
    # ── Matematika ─────────────────────────────────────────────────────────────
    "integral":          "integral calculus",
    "turunan":           "derivative differentiation",
    "diferensial":       "differential calculus",
    "limit":             "limit calculus",
    "matriks":           "matrix linear algebra",
    "vektor":            "vector linear algebra",
    "determinan":        "determinant matrix",
    "invers":            "inverse matrix",
    "eigenvalue":        "eigenvalue eigenvector",
    "persamaan":         "equation",
    "pertidaksamaan":    "inequality",
    "fungsi":            "function",
    "grafik":            "graph plot",
    "statistik":         "statistics",
    "probabilitas":      "probability",
    "distribusi":        "distribution",
    "rata-rata":         "mean average",
    "median":            "median",
    "simpangan":         "standard deviation",
    "regresi":           "regression",
    "turunan parsial":   "partial derivative",
    "deret":             "series sequence",
    "bilangan prima":    "prime number",
    "faktorial":         "factorial",
    "kombinasi":         "combination combinatorics",
    "permutasi":         "permutation",
    "logaritma":         "logarithm",
    "eksponen":          "exponent",
    "akar":              "square root",
    "pangkat":           "power exponent",
    "trigonometri":      "trigonometry",
    "sinus":             "sine trigonometry",
    "cosinus":           "cosine trigonometry",
    "tangen":            "tangent trigonometry",
    "geometri":          "geometry",
    "luas":              "area",
    "volume":            "volume",
    "keliling":          "perimeter",
    "gram schmidt":      "gram schmidt orthogonalization",
    "hitung":            "calculate compute",
    "rumus":             "formula",
    # ── Pemrograman / Kode ─────────────────────────────────────────────────────
    "pemrograman":       "programming",
    "kode":              "code",
    "program":           "program code",
    "algoritma":         "algorithm",
    "struktur data":     "data structure",
    "antrian":           "queue data structure",
    "tumpukan":          "stack data structure",
    "pohon":             "tree data structure",
    "graf":              "graph data structure",
    "pengurutan":        "sorting algorithm",
    "pencarian":         "search algorithm",
    "rekursi":           "recursion recursive",
    "iterasi":           "iteration loop",
    "kelas":             "class object oriented",
    "objek":             "object class",
    "warisan":           "inheritance",
    "polimorfisme":      "polymorphism",
    "enkapsulasi":       "encapsulation",
    "antarmuka":         "interface",
    "fungsi anonim":     "lambda anonymous function",
    "penutupan":         "closure",
    "dekorator":         "decorator",
    "generator":         "generator iterator",
    "pola desain":       "design pattern",
    "singleton":         "singleton design pattern",
    "pabrik":            "factory design pattern",
    "pengamat":          "observer design pattern",
    "strategi":          "strategy design pattern",
    "basis data":        "database",
    "kueri":             "query database",
    "sql":               "sql database",
    "tabel":             "table database",
    "indeks":            "index database",
    "api":               "api rest endpoint",
    "server":            "server backend",
    "klien":             "client frontend",
    "kompilasi":         "compile compiler",
    "debug":             "debug debugging",
    "pengujian":         "testing unit test",
    "perulangan":        "loop iteration",
    "kondisi":           "condition if else",
    "variabel":          "variable",
    "tipe data":         "data type",
    "string":            "string text",
    "larik":             "array list",
    "kamus":             "dictionary hashmap",
    "himpunan":          "set collection",
    "pengecualian":      "exception error handling",
    "kesalahan":         "error exception",
    "berkas":            "file filesystem",
    "memori":            "memory",
    "proses":            "process thread",
    "benang":            "thread concurrency",
    "konkurensi":        "concurrency parallel",
    "jaringan":          "network",
    "protokol":          "protocol",
    "soket":             "socket network",
    "keamanan":          "security",
    "enkripsi":          "encryption",
    "hash":              "hash hashing",
    "otentikasi":        "authentication",
    "otorisasi":         "authorization",
    # ── Keamanan Siber ─────────────────────────────────────────────────────────
    "serangan":          "attack exploit",
    "injeksi":           "injection attack",
    "injeksi sql":       "sql injection attack",
    "skrip lintas situs":"xss cross site scripting",
    "pemalsuan permintaan": "csrf request forgery",
    "eskalasi hak":      "privilege escalation",
    "pindai port":       "port scan nmap",
    "pentest":           "penetration testing",
    "kerentanan":        "vulnerability exploit",
    "malware":           "malware virus",
    "ransomware":        "ransomware",
    "phishing":          "phishing social engineering",
    "firewall":          "firewall network security",
    "kriptografi":       "cryptography",
    "sertifikat":        "certificate ssl tls",
    "kata sandi":        "password authentication",
    # ── Umum / AI / Sistem ─────────────────────────────────────────────────────
    "kecerdasan buatan": "artificial intelligence AI",
    "pembelajaran mesin":"machine learning",
    "pembelajaran mendalam": "deep learning neural network",
    "jaringan saraf":    "neural network",
    "model":             "model training",
    "pelatihan":         "training model",
    "prediksi":          "prediction inference",
    "klasifikasi":       "classification",
    "klasterisasi":      "clustering",
    "sistem operasi":    "operating system",
    "kernel":            "kernel operating system",
    "prosesor":          "processor cpu",
    "memori":            "memory ram",
    "penyimpanan":       "storage disk",
    "virtualisasi":      "virtualization",
    "kontainer":         "container docker",
    "jelaskan":          "explain",
    "bagaimana":         "how to",
    "apa itu":           "what is",
    "cara":              "how to method",
    "contoh":            "example",
    "perbedaan":         "difference between",
    "bandingkan":        "compare comparison",
    "manfaat":           "benefit advantage",
    "kekurangan":        "disadvantage limitation",
    "implementasi":      "implementation",
    "optimasi":          "optimization",
    "performa":          "performance optimization",
}

def translate_query_id_to_en(query: str) -> str:
    """
    Terjemahkan kata kunci penting dari Bahasa Indonesia ke Bahasa Inggris
    sebelum melakukan embedding untuk pencarian di OMNI index.

    Hanya menambahkan terjemahan — kata asli tetap dipertahankan agar
    pencarian juga menangkap data yang mungkin mengandung kata Indonesia.

    Contoh:
        "jelaskan integral" → "jelaskan integral explain integral calculus"
        "algoritma pengurutan" → "algoritma pengurutan algorithm sorting algorithm"
    """
    query_lower = query.lower()
    extra_terms = []

    # Cek frasa dua-kata dulu (lebih spesifik), lalu satu kata
    for id_word, en_word in _ID_TO_EN.items():
        if id_word in query_lower and en_word not in query_lower:
            extra_terms.append(en_word)

    if extra_terms:
        translated = query + " " + " ".join(extra_terms)
        return translated
    return query



@dataclass
class RagChunk:
    """Satu potongan memori yang ditemukan dari OMNI index."""
    text:    str
    source:  str
    domain:  str
    score:   float
    folder:  str
    valence: float = 0.0
    arousal: float = 0.5


class MokoRagRetriever:
    """
    Retriever utama MOKO RAG — thread-safe, always-available (offline-capable).

    Strategy:
    - Terjemahkan query ID→EN dulu (jika mengandung kata Indonesia)
    - Gunakan moko_embed_engine (bge-small atau fallback) untuk embedding
    - Cari di semua domain yang ada, atau di domain tertentu
    - Threshold skor diturunkan ke 0.10 agar hasil tidak terbuang
    - Hasil di-deduplikasi berdasarkan 20 karakter pertama teks
    """

    # Threshold minimum score
    MIN_SCORE = 0.10
    # Domain yang dicari jika domain=None (dari yang paling besar)
    DEFAULT_SEARCH_DOMAINS = ["code", "math", "security", "general", "science", "history", "finance", "health"]

    def __init__(self):
        from moko_config import settings
        from moko_memory.omni_hash_encoder import get_omni_encoder
        from moko_memory.omni_vector_store import OmniVectorStore
        self._omni_root = Path(settings.OMNI_DIR)
        self._encoder   = get_omni_encoder()
        self._stores: Dict[str, OmniVectorStore] = {}  # lazy-loaded per domain

    def _get_store(self, domain: str):
        """Lazy-load OmniVectorStore per domain."""
        from moko_memory.omni_vector_store import OmniVectorStore
        if domain not in self._stores:
            self._stores[domain] = OmniVectorStore(domain, base_path=self._omni_root)
        return self._stores[domain]

    def _active_domains(self) -> List[str]:
        """Daftar domain yang memiliki data di .moko_omni/."""
        if not self._omni_root.exists():
            return []
        return [
            d.name for d in self._omni_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

    def embed_query(self, query: str, use_model: bool = True) -> np.ndarray:
        """
        Embed query menggunakan moko_embed_engine (local, offline-capable).
        Query akan diterjemahkan ID→EN terlebih dahulu jika mengandung kata Indonesia.
        """
        from moko_memory.moko_embed_engine import embed
        translated = translate_query_id_to_en(query)
        if translated != query:
            print(f"[RAG] Query diterjemahkan: {query!r} → {translated!r}")
        return embed(translated, use_model=use_model)


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        max_hamming: int = 48,      # Dinaikkan dari 32 → 48 untuk kompensasi tanpa linear fallback
        min_score: Optional[float] = None,
    ) -> List[RagChunk]:
        """
        Cari top-k chunk paling relevan untuk query dari OMNI index.

        Args:
            query       : Teks query
            top_k       : Jumlah hasil yang dikembalikan
            domain      : Domain spesifik (None = cari semua domain)
            max_hamming : Batas Hamming distance (makin besar = lebih banyak kandidat)
            min_score   : Override threshold minimum (default: MIN_SCORE)

        Returns:
            List[RagChunk] diurutkan berdasarkan skor tertinggi
        """
        threshold = min_score if min_score is not None else self.MIN_SCORE
        t_start = time.time()

        # 1. Tentukan domain yang akan dicari dengan prioritas cerdas (Cognitive Domain Routing)
        if domain:
            search_domains = [domain]
        else:
            active = self._active_domains()
            
            # Deteksi domain prioritas secara cerdas dari keyword query
            priority_domain = None
            q_lower = query.lower()
            
            # Kata kunci matematika
            if any(kw in q_lower for kw in [
                "integral", "turunan", "matriks", "vektor", "limit", "persamaan", "calculus", 
                "algebra", "eigen", "determinant", "math", "schmidt", "gram", "euler", "theorem", 
                "formula", "decomposition", "factorization", "equation", "probability", "statistics", 
                "discrete", "numerical", "geometry", "topology"
            ]):
                priority_domain = "math"
            # Kata kunci keamanan siber
            elif any(kw in q_lower for kw in [
                "injection", "sql", "exploit", "security", "vulnerability", "attack", "phishing", 
                "bypass", "forging", "ticket", "kerberos", "buffer overflow", "cve", "vulnerable", 
                "defense", "offensive", "hacking", "ethical", "payload", "mitre", "att&ck", 
                "privilege escalation", "sandbox", "mitigation", "threat"
            ]):
                priority_domain = "security"
            # Kata kunci coding/programming
            elif any(kw in q_lower for kw in [
                "code", "class", "def ", "import", "sorting", "algorithm", "quicksort", "binary search", 
                "program", "programming", "database", "asyncio", "rust", "python", "javascript", "c++", 
                "cpp", "java", "typescript", "go", "goroutine", "docker", "kubernetes", "nginx", 
                "rest api", "json", "git", "event loop", "web socket", "server", "client"
            ]):
                priority_domain = "code"

            # Susun daftar domain pencarian, prioritaskan yang cocok teratas
            ordered_defaults = list(self.DEFAULT_SEARCH_DOMAINS)
            if priority_domain and priority_domain in ordered_defaults:
                ordered_defaults.remove(priority_domain)
                ordered_defaults.insert(0, priority_domain)
                
            search_domains = [d for d in ordered_defaults if d in active]
            search_domains += [d for d in active if d not in search_domains]

        # PRE-COMPUTE: Translate query sekali & embed dua vektor (blake2b + neural) sekali di luar loop
        # Ini menghilangkan translate/embed duplikat yang terjadi N kali (1x per domain)
        _translated_query = translate_query_id_to_en(query)
        if _translated_query != query:
            print(f"[RAG] Query diterjemahkan: {query!r} → {_translated_query!r}")
        from moko_memory.moko_embed_engine import embed as _embed_fn
        # Blake2b vec: digunakan untuk domain code/security/math (ruang vektor sinkron dengan ingest)
        _vec_blake2b      = _embed_fn(_translated_query, use_model=False)
        _vec_blake2b_list = _vec_blake2b.tolist()
        _addr_blake2b     = self._encoder.encode(_translated_query, _vec_blake2b_list)
        # Neural vec: digunakan untuk domain general/finance/dll
        _vec_neural       = _embed_fn(_translated_query, use_model=True)
        _vec_neural_list  = _vec_neural.tolist()
        _addr_neural      = self._encoder.encode(_translated_query, _vec_neural_list)

        # 2. Cari di setiap domain dengan embedding yang konsisten (blake2b vs neural)
        all_results: List[RagChunk] = []
        for dom in search_domains:
            try:
                store = self._get_store(dom)
                if not store.has_data():
                    continue

                # Pilih vektor yang sesuai: blake2b untuk domain yg di-ingest tanpa neural model,
                # neural untuk domain general. Keduanya sudah di-precompute sebelum loop.
                if dom in ("code", "security", "math"):
                    vec_list = _vec_blake2b_list
                    addr     = _addr_blake2b
                else:
                    vec_list = _vec_neural_list
                    addr     = _addr_neural

                raw = store.search_by_hamming(
                    addr=addr,
                    fp32_query=vec_list,
                    top_k=top_k,
                    max_hamming=max_hamming,
                    n_probe_extra=1,
                )
                for r in raw:
                    score = float(r.get("score", 0.0))
                    if score >= threshold:
                        all_results.append(RagChunk(
                            text    = r.get("text", ""),
                            source  = r.get("source", "?"),
                            domain  = r.get("domain", dom),
                            score   = score,
                            folder  = r.get("folder", "?"),
                            valence = float(r.get("valence", 0.0)),
                            arousal = float(r.get("arousal", 0.5)),
                        ))
            except Exception as e:
                print(f"[MokoRagRetriever] Warn: domain '{dom}' error: {e}")
                continue




        # 4b. Fallback: Keyword search on source/filename if no semantic results found
        if not all_results:
            # Pecah query menjadi keywords individual
            keywords = [kw for kw in re.findall(r"\w+", query.lower()) if len(kw) > 2]
            
            # Tambahkan keyword English dari terjemahan (agar cocok dengan nama file English)
            translated_q = translate_query_id_to_en(query)
            if translated_q != query:
                en_words = [kw for kw in re.findall(r"\w+", translated_q.lower()) if len(kw) > 2]
                keywords = list(set(keywords + en_words))  # merge dan deduplikasi
            
            if keywords:
                for dom in search_domains:
                    try:
                        store = self._get_store(dom)
                        raw_kw = store.keyword_search(keywords, top_k=top_k)
                        for r in raw_kw:
                            all_results.append(RagChunk(
                                text    = r.get("text", ""),
                                source  = r.get("source", "?"),
                                domain  = r.get("domain", dom),
                                score   = r.get("score", 0.70), # default keyword match score
                                folder  = r.get("folder", "?"),
                                valence = float(r.get("valence", 0.0)),
                                arousal = float(r.get("arousal", 0.5)),
                            ))
                        # Hentikan pencarian domain lain jika kita sudah mendapatkan hasil yang cukup (top_k)
                        if len(all_results) >= top_k:
                            break
                    except Exception as e:
                        print(f"[MokoRagRetriever] Keyword search error in '{dom}': {e}")
                        continue


        # 5. Deduplicate + sort by score
        seen = set()
        unique: List[RagChunk] = []
        for chunk in sorted(all_results, key=lambda x: x.score, reverse=True):
            key = chunk.text[:30]
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
            if len(unique) >= top_k:
                break

        print(f"[RAG] retrieve({query[:40]!r}) → {len(unique)}/{len(all_results)} chunks")
        return unique


    def format_context(self, chunks: List[RagChunk], max_chars: int = 4000) -> str:
        """
        Format chunks ke dalam string konteks siap pakai untuk prompt LLM.
        """
        if not chunks:
            return ""
        parts = ["=== MOKO KNOWLEDGE BASE ==="]
        total = 0
        for i, chunk in enumerate(chunks, 1):
            snippet = chunk.text[:800].strip()
            line = (
                f"[{i}] Domain:{chunk.domain} | Source:{chunk.source} | Score:{chunk.score:.3f}\n"
                f"{snippet}"
            )
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        parts.append("=== END KNOWLEDGE BASE ===")
        return "\n\n".join(parts)

    def retrieve_formatted(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> str:
        """Convenience: retrieve + format in one call."""
        chunks = self.retrieve(query, top_k=top_k, domain=domain)
        return self.format_context(chunks)


# ─── Singleton ────────────────────────────────────────────────────────────────
_retriever_instance: Optional[MokoRagRetriever] = None
_retriever_lock = __import__("threading").Lock()

def get_rag_retriever() -> MokoRagRetriever:
    """Thread-safe singleton accessor."""
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = MokoRagRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # Quick smoke-test
    r = MokoRagRetriever()
    for q in ["binary search python", "sql injection", "eigenvalue matrix"]:
        chunks = r.retrieve(q, top_k=3)
        print(f"\nQuery: {q!r}")
        for c in chunks:
            print(f"  [{c.score:.3f}] {c.domain}/{c.source}: {c.text[:80]}...")
