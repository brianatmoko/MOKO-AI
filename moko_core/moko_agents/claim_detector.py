import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from moko_config import settings


class ClaimDetector:
    """
    MOKO OS — Probabilistic Claim Classifier.
    
    Arsitektur Kelas Industri:
    ─────────────────────────────────────
    Mengganti regex heuristik kasar dengan klasifikasi probabilitas jarak semantik.
    Menghitung kedekatan (Cosine Similarity) terhadap centroid 'Klaim/Asersi'
    versus centroid 'Casual/Pertanyaan Faktual Murni'.
    """

    _centroids = None
    _centroids_loaded = False

    @classmethod
    def _ensure_centroids(cls):
        """Memuat atau menghitung centroid untuk klasifikasi klaim."""
        if cls._centroids_loaded:
            return

        cache_path = Path(settings.WORKSPACE_DIR) / ".moko_claim_centroids.json"

        # 1. Coba load dari cache disk
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cls._centroids = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
                cls._centroids_loaded = True
                return
            except Exception as e:
                print(f"[ClaimDetector Centroid] Gagal memuat cache: {e}")

        # 2. Hitung dinamis jika belum ada cache
        print("[ClaimDetector Centroid] Menghitung centroid klaim semantik baru...")
        templates = {
            "claim": [
                "menurut saya rumus ini adalah x=y",
                "saya asumsikan kecepatannya konstan",
                "klaim teoritis ini terbukti benar",
                "hipotesis saya menyatakan bahwa variabel ini",
                "pernyataan saya didukung oleh data berikut",
                "saya pikir hasilnya harus seperti ini"
            ],
            "casual": [
                "bagaimana cara menghitung cc motor",
                "tolong jelaskan materi fisika ini",
                "siapa namamu dan apa tugasmu",
                "selamat pagi ada yang bisa dibantu",
                "bagaimanakah proses terjadinya hujan",
                "rumus matematika dasar untuk kalkulus"
            ]
        }

        try:
            from moko_agents.llm_engine import engine
            all_texts = []
            for category, queries in templates.items():
                all_texts.extend(queries)

            all_embs = engine.get_embeddings_batch(all_texts)
            if len(all_embs) == len(all_texts):
                cls._centroids = {}
                idx = 0
                for category, queries in templates.items():
                    n = len(queries)
                    category_embs = all_embs[idx : idx + n]
                    idx += n
                    # Average vector (Centroid)
                    cls._centroids[category] = np.mean(category_embs, axis=0)

                # Simpan ke cache
                save_data = {k: v.tolist() for k, v in cls._centroids.items()}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(save_data, f)
                cls._centroids_loaded = True
                print("[ClaimDetector Centroid] Centroid klaim berhasil disimpan.")
            else:
                print("[ClaimDetector Centroid] Batch embeddings gagal. Fallback ke threshold default.")
        except Exception as e:
            print(f"[ClaimDetector Centroid] Gagal menghitung centroid: {e}.")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Menganalisis teks untuk menentukan apakah merupakan klaim/asersi deklaratif
        yang diajukan oleh pengguna secara probabilistik.
        """
        text_lower = text.strip().lower()
        
        is_claim = False
        reasons = []
        delta_sim = 0.0

        # Cek inisialisasi centroid
        self._ensure_centroids()

        if self._centroids_loaded:
            try:
                from moko_agents.llm_engine import engine
                q_emb = engine.get_embedding(text)
                if len(q_emb) == 768:
                    q_v = np.array(q_emb, dtype=np.float32)

                    # Hitung similarity ke masing-masing centroid
                    denom_claim = (np.linalg.norm(q_v) * np.linalg.norm(self._centroids["claim"])) + 1e-9
                    denom_casual = (np.linalg.norm(q_v) * np.linalg.norm(self._centroids["casual"])) + 1e-9

                    sim_claim = float(np.dot(q_v, self._centroids["claim"]) / denom_claim)
                    sim_casual = float(np.dot(q_v, self._centroids["casual"]) / denom_casual)

                    # Delta similarity menentukan apakah ini asersi/klaim
                    delta_sim = sim_claim - sim_casual

                    # Threshold delta similarity untuk deteksi klaim
                    if delta_sim >= 0.045:
                        is_claim = True
                        reasons.append(f"Semantic claim confidence delta: {delta_sim:.3f}")
            except Exception as e:
                print(f"[ClaimDetector] Gagal melakukan klasifikasi semantik: {e}")

        # Fallback Heuristik Tambahan (Safety guard jika model offline)
        if not self._centroids_loaded:
            # Jika mengandung kata klaim/hipotesis eksplisit
            explicit_keywords = ["klaim saya", "menurut saya", "saya asumsikan", "saya pikir"]
            if any(kw in text_lower for kw in explicit_keywords):
                is_claim = True
                reasons.append("Mengandung keyword klaim eksplisit (Fallback)")

        critical_instruction = ""
        if is_claim:
            critical_instruction = (
                "\n=== PERINGATAN: DETEKSI KLAIM DARI PENGGUNA ===\n"
                "Pengguna tampaknya mengajukan pernyataan atau klaim teoritis.\n"
                "TUGAS UTAMA: Evaluasi secara kritis dan faktual. Jangan langsung membenarkan (sycophancy).\n"
                "Jika pernyataan tersebut tidak akurat, koreksi dengan sopan menggunakan prinsip yang benar.\n"
            )

        return {
            "is_claim": is_claim,
            "reasons": reasons,
            "delta_similarity": delta_sim,
            "critical_instruction": critical_instruction
        }


# Singleton instance
claim_detector = ClaimDetector()
