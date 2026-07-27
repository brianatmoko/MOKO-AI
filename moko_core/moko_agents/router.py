import json
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from moko_config import settings

# Import Intent-First Router (baru)
try:
    from moko_agents.intent_router import IntentFirstRouter, IntentClass
    _INTENT_ROUTER_AVAILABLE = True
except ImportError:
    _INTENT_ROUTER_AVAILABLE = False


class CognitiveRouter:
    """
    Hakim Penentu Jalur (The Pathfinder) berbasis Ruang Vektor Semantik.
    
    Arsitektur Kelas Industri:
    ─────────────────────────────────────
    1. Tier 0-A (< 1ms)  : Slash commands & Exact fast matches (Bypass).
    2. Tier 0-B (~15ms)  : Semantic Vector Router. Menghitung Cosine Similarity
                           query pengguna terhadap Centroid Domain (averaged embeddings).
    3. Tier 0-C Fallback : Rule-based keyword matching jika API embedding gagal.
    """

    # ── Pola perkenalan / obrolan personal → FAST_PATH ─────────────────────
    _PERSONAL_TRIGGERS = [
        "nama aku", "nama saya", "namaku", "namanya", "kenalin", "kenalkan",
        "perkenalkan", "aku adalah", "saya adalah", "aku ini", "saya ini",
        "panggil aku", "panggil saya", "ingat saya", "ingat aku", "siapa saya",
        "siapa aku", "ingat namaku", "ingat nama aku", "ingat nama saya",
        "siapa namaku", "siapa nama aku", "siapa nama saya",
        "makasih", "terima kasih", "thanks", "thx",
        "mantap", "keren", "bagus", "hebat", "luar biasa", "wow",
        "siap", "noted", "paham", "ngerti", "iya", "betul", "benar",
        "sampai nanti", "dadah", "bye", "sampai jumpa",
    ]

    # ── Trigger FAST_PATH pasti — tidak ambigu ─────────────────────────────
    _FAST_EXACT = {
        "hai", "halo", "hello", "hi", "hey", "p", "oy", "hei", "bro",
        "apa kabar", "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
        "ya", "oke", "ok", "lanjut", "sip", "baik", "terus", "lanjutkan",
        "test", "ping", "coba", "cek",
    }

    # ── Trigger BROWSING_PATH — Pertanyaan yang perlu web search ─────────────
    # Keywords SPESIFIK untuk browsing queries - HARUS non-overlap dengan DEEP keywords
    _BROWSING_KEYWORDS = [
        "bagaimana cara", "bagaimana membuat", "bagaimana sih", "gimana cara",
        "cara membuat", "cara membuat", "resep", "resep makanan",
        "restoran", "café", "tempat makan", "menu", "makanan",
        "berita", "berita terbaru", "headline", "berita hari ini",
        "tempat", "lokasi", "alamat", "peta", "rute",
        "review", "review produk", "rating", "evaluasi",
        "harga", "biaya", "tarif", "berapa harganya",
        "travel", "wisata", "liburan", "destinasi", "perjalanan",
        "hotel", "penginapan", "tiket pesawat", "booking",
        "film", "sinema", "bioskop", "trailer", "sutradara",
        "musik", "lagu", "artis", "konser", "album",
        "acara", "event", "pertunjukan", "pameran",
        "olahraga", "pertandingan", "skor", "pemain",
        "resep dapur", "cara masak", "panduan memasak"
    ]

    # ── Trigger DEEP_PATH pasti ─────────────────────────────────────────────
    _DEEP_KEYWORDS = [
        "buat", "tulis", "analisis", "kode", "code", "program", "debug",
        "hitung", "rumus", "integral", "turunan", "aljabar", "statistik",
        "mengapa", "kenapa", "jelaskan", "definisi", "arti",
        "makna", "bedanya", "perbedaan", "langkah", "tutorial",
        "fakta", "teori", "konsep", "buktikan", "optimasi",
        "implementasi", "arsitektur", "algoritma", "fungsi", "class",
        "enkripsi", "kriptografi", "matematika", "fisika", "kimia",
    ]

    # ── Domain keyword map (heuristik fallback) ─────────────────────────────
    _DOMAIN_MAP = [
        (["arti", "definisi", "makna", "kamus", "apa itu", "artinya", "sinonim"], "lexical", "D", "D0"),
        (["hitung", "rumus", "matematika", "integral", "turunan", "aljabar", "kalkulus", "geometri", "statistik", "rsa", "prima", "modular", "enkripsi", "kriptografi"], "math", "A", "D5"),
        (["fisika", "gaya", "kecepatan", "energi", "newton", "momentum", "gravitasi", "termodinamika"], "physics", "A", "D5"),
        (["kode", "code", "program", "python", "javascript", "java", "rust", "fungsi", "class", "bug", "debug", "docker", "database", "sql", "api"], "code", "G", "D9"),
        (["kenapa", "mengapa", "sebab", "akibat", "logika", "penalaran", "implikasi"], "reasoning", "E", "D9"),
        (["bagaimana cara", "bagaimana membuat", "cara membuat", "resep", "tips", "tempat", "berita", "travel", "review"], "browsing", "B", "D7"),
        (["onion", "darkweb", "darknet", "ahmia", "torch"], "cyberoffensive", "B", "D7"),
    ]

    # Cache untuk centroid domain semantik
    _centroids = None
    _centroids_loaded = False

    @classmethod
    def _ensure_centroids(cls):
        """Memuat atau menghitung centroid domain semantik."""
        if cls._centroids_loaded:
            return

        cache_path = Path(settings.WORKSPACE_DIR) / ".moko_domain_centroids.json"

        # 1. Coba load dari cache disk
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cls._centroids = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
                cls._centroids_loaded = True
                return
            except Exception as e:
                print(f"[Pathfinder Centroid] Gagal memuat cache: {e}")

        # 2. Hitung dinamis jika cache belum terbentuk
        print("[Pathfinder Centroid] Menghitung centroid semantik baru...")
        templates = {
            "lexical": [
                "apa arti kata ini",
                "definisi istilah makna kamus",
                "artinya sinonim antonim kata",
                "penjelasan kosakata frasa kalimat"
            ],
            "math": [
                "hitung rumus matematika kalkulus",
                "integral turunan limit fungsi",
                "kpk fpb bilangan prima faktorisasi",
                "aljabar linear matriks vektor deret",
                "rumus cc silinder mesin motor kendaraan",
                "kapasitas volume diameter bore stroke",
                "fisika teknik mesin rpm torsi daya",
                "berapa hitung hasilnya angka kalkulasi"
            ],
            "physics": [
                "gaya kecepatan momentum newton",
                "gravitasi termodinamika kalor energi",
                "kinetik potensial hukum fisika",
                "relativitas kuantum mekanika gelombang"
            ],
            "code": [
                "fungsi class python javascript program",
                "source code syntax variable array loop structure",
                "coding implementation program script compile debug",
                "docker database sql query json api development"
            ],
            "reasoning": [
                "kenapa mengapa sebab akibat analisa",
                "logika penalaran pembuktian deduktif induktif",
                "bagaimana menjelaskan implikasi logika argumen",
                "analisis kritis sebab mengapa hal ini terjadi"
            ],
            "personal": [
                "hai halo hello apa kabar",
                "siapa kamu namamu penciptamu",
                "terima kasih banyak sama-sama",
                "perkenalkan dirimu panggil saya"
            ],
            "general": [
                "tulis artikel cerita puisi kreatif",
                "informasi umum sejarah geografi dunia",
                "brainstorm ide topik baru",
                "pengetahuan acak fakta populer",
                "cara membuat resep masak makanan",
                "langkah prosedur tutorial panduan",
                "tips trik cara buat diy",
                "rekomendasi saran pilihan info"
            ]
        }

        try:
            from moko_agents.llm_engine import engine
            all_texts = []
            for domain, queries in templates.items():
                all_texts.extend(queries)

            all_embs = engine.get_embeddings_batch(all_texts)
            if len(all_embs) == len(all_texts):
                cls._centroids = {}
                idx = 0
                for domain, queries in templates.items():
                    n = len(queries)
                    domain_embs = all_embs[idx : idx + n]
                    idx += n
                    # Average vector (Centroid)
                    cls._centroids[domain] = np.mean(domain_embs, axis=0)

                # Simpan ke cache
                save_data = {k: v.tolist() for k, v in cls._centroids.items()}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(save_data, f)
                cls._centroids_loaded = True
                print("[Pathfinder Centroid] Centroid semantik sukses dihitung & disimpan.")
            else:
                print("[Pathfinder Centroid] Batch embeddings tidak komplit. Menggunakan rule-based.")
        except Exception as e:
            print(f"[Pathfinder Centroid] Gagal menghitung centroid: {e}. Fallback ke rule-based.")

    @classmethod
    def classify_intent(cls, text: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Klasifikasi Niat Kelas Industri.
        
        Priority:
        1. Intent-First Router (baru, 100% accuracy)
        2. Legacy routing (semantic + rule-based)
        """
        # ── Intent-First Router (Priority 1) ─────────────────────────────
        if _INTENT_ROUTER_AVAILABLE:
            try:
                from moko_agents.intent_router import get_intent_router
                router = get_intent_router()
                result = router.classify(text)
                
                # Convert IntentResult ke format lama
                return result.path, result.reason, {
                    "domain": result.domain,
                    "intent": result.intent.value,
                    "model_key": result.model_key,
                    "confidence": result.confidence,
                    "tier": result.tier,
                    "latency_ms": result.latency_ms,
                }
            except Exception as e:
                print(f"[Router] Intent-First fallback: {e}")

        # ── Legacy Routing (Priority 2) ──────────────────────────────────
        return cls._legacy_classify_intent(text)
    
    @classmethod
    def _legacy_classify_intent(cls, text: str) -> Tuple[str, str, Dict[str, Any]]:
        """Legacy routing untuk backward compatibility."""
        t = text.strip().lower()
        words = t.split()

        # ── Tier 0-A: Slash command ──────────────────────────────────────────
        if t.startswith('/'):
            return "COMMAND_PATH", "Deteksi perintah sistem", {
                "domain": "system_control",
                "logic_type": "G",
                "arousal": "2",
                "depth": "D5"
            }

        # ── Tier 0-C: OMNI Exact Cache Hit Check (Direct Bypass) ─────────────
        # (Crypto bypass removed per Phase 4 Rollback)
        pass

        # ── Tier 1-A: Semantic Vector Router ─────────────────────────────────
        cls._ensure_centroids()
        if cls._centroids_loaded:
            try:
                from moko_agents.llm_engine import engine
                q_emb = engine.get_embedding(text)
                if len(q_emb) == 768:
                    q_v = np.array(q_emb, dtype=np.float32)
                    best_domain = "general"
                    best_sim = -1.0

                    for domain, centroid in cls._centroids.items():
                        # Cosine similarity
                        denom = (np.linalg.norm(q_v) * np.linalg.norm(centroid)) + 1e-9
                        sim = float(np.dot(q_v, centroid) / denom)
                        if sim > best_sim:
                            best_sim = sim
                            best_domain = domain

                    # Hanya route secara semantik jika similarity melewati threshold 0.38
                    if best_sim >= 0.38:
                        domain_params = {
                            "lexical":   ("DEEP_PATH", "D", "D0"),
                            "math":      ("DEEP_PATH", "A", "D5"),
                            "physics":   ("DEEP_PATH", "A", "D5"),
                            "code":      ("DEEP_PATH", "G", "D9"),
                            "reasoning": ("DEEP_PATH", "E", "D9"),
                            "personal":  ("FAST_PATH", "B", "D0"),
                            "general":   ("DEEP_PATH", "A", "D5")
                        }

                        path, logic_type, depth = domain_params.get(best_domain, ("DEEP_PATH", "A", "D5"))
                        if best_domain == "personal":
                            path = "FAST_PATH"

                        return path, f"Semantic vector routing: {best_domain} (sim={best_sim:.3f})", {
                            "domain": best_domain,
                            "logic_type": logic_type,
                            "arousal": "2" if path == "DEEP_PATH" else "1",
                            "depth": depth
                        }
            except Exception as e:
                print(f"[Pathfinder] Kesalahan runtime routing semantik: {e}. Fallback ke rules.")

        # ── Tier 1-B: Rule-Based Fallback (Heuristik) ───────────────────────
        # 0. PRIORITY: Deteksi BROWSING keywords TERLEBIH DAHULU (before domain tech)
        if any(kw in t for kw in cls._BROWSING_KEYWORDS):
            return "BROWSING_PATH", "Browsing intent terdeteksi (web search diperlukan)", {
                "domain": "browsing",
                "logic_type": "B",
                "arousal": "2",
                "depth": "D7",
                "require_web_search": True
            }

        # 1. Query pendek (≤ 3 kata) — cek personal triggers
        if len(words) <= 3:
            if any(trigger in t for trigger in cls._PERSONAL_TRIGGERS):
                return "FAST_PATH", "Obrolan personal / afirmasi sosial (Fallback)", {
                    "domain": "personal",
                    "logic_type": "B",
                    "arousal": "1",
                    "depth": "D0"
                }
            has_deep = any(kw in t for kw in cls._DEEP_KEYWORDS)
            if not has_deep:
                return "FAST_PATH", "Query pendek ringan (Fallback)", {
                    "domain": "personal",
                    "logic_type": "B",
                    "arousal": "1",
                    "depth": "D0"
                }

        # 2. Personal trigger dalam kalimat panjang
        if any(trigger in t for trigger in cls._PERSONAL_TRIGGERS):
            return "FAST_PATH", "Obrolan personal / perkenalan (Fallback)", {
                "domain": "personal",
                "logic_type": "B",
                "arousal": "1",
                "depth": "D0"
            }

        # 3. Deteksi domain teknis
        for keywords, domain, logic, depth in cls._DOMAIN_MAP:
            if domain == "browsing":  # BROWSING sudah di-handle di atas, skip
                continue
            if any(kw in t for kw in keywords):
                return "DEEP_PATH", f"Deteksi domain fallback: {domain}", {
                    "domain": domain,
                    "logic_type": logic,
                    "arousal": "2",
                    "depth": depth
                }

        # 4. Ada keyword DEEP pasti
        if any(kw in t for kw in cls._DEEP_KEYWORDS):
            return "DEEP_PATH", "Keyword analitis terdeteksi (Fallback)", {
                "domain": "general",
                "logic_type": "A",
                "arousal": "2",
                "depth": "D5"
            }

        # 5. Query panjang tanpa keyword khusus → DEEP_PATH default
        if len(words) >= 6:
            return "DEEP_PATH", "Query panjang — analisis mendalam (Fallback)", {
                "domain": "general",
                "logic_type": "A",
                "arousal": "2",
                "depth": "D5"
            }

        # 6. Fallback final
        return "FAST_PATH", "Query ringan tanpa indikasi teknis (Fallback)", {
            "domain": "personal",
            "logic_type": "B",
            "arousal": "1",
            "depth": "D0"
        }

    @staticmethod
    def _fallback_classify_intent(text: str) -> Tuple[str, str]:
        return CognitiveRouter.classify_intent(text)[:2]

    @staticmethod
    def _fallback_metadata(text: str, path: str) -> Dict[str, Any]:
        _, _, meta = CognitiveRouter.classify_intent(text)
        return meta


# Global router instance
router = CognitiveRouter()
