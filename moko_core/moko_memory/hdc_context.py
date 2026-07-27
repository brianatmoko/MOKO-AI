"""
MOKO Hyperdimensional Context Compressor (HDC)
==============================================
Implementasi Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC)
berdasarkan riset Kanerva (2009) dan Slaymaker et al. (2024).

Representasi Memori Konteks:
  Konteks panjang (history chat, omni search results) dikompresi menjadi satu
  vektor berdimensi tinggi (D = 2048) menggunakan operasi aljabar VSA:
    1. Binding (⊗)       - Mengaitkan kunci & nilai (e.g. Role ⊗ Content)
    2. Bundling (+)      - Menggabungkan sekumpulan informasi (e.g. Step1 + Step2)
    3. Permutation (Π)   - Menyimpan urutan urutan sekuensial (e.g. Π^k(Turn_k))

Proyeksi:
  Menggunakan Random Projection (RP) yang stabil untuk memetakan embedding float 768-dim
  ke dalam ruang bipolar HDC 2048-dim ({-1, 1}^D). RP mempertahankan kedekatan kosinus
  (JL Lemma - Johnson-Lindenstrauss).
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional, Any


class HDCSpace:
    """
    Representasi Ruang Vektor Simbolik Hyperdimensional.
    Menggunakan representasi bipolar ({-1, 1}^D) untuk robustness dan efisiensi.
    """

    def __init__(self, dimension: int = 2048, seed: int = 42):
        self.D = dimension
        self.rng = np.random.default_rng(seed)
        
        # Generator matriks proyeksi acak yang stabil dari 768 -> D
        # Digunakan untuk memproyeksikan embedding float ke HDC
        self.projection_matrix = self.rng.normal(0.0, 1.0, (self.D, 768))

        # Cache untuk ID/simbol yang sering digunakan (e.g. role, tags)
        self.symbol_cache: Dict[str, np.ndarray] = {}

    def generate_random_vector(self) -> np.ndarray:
        """Menghasilkan satu vektor bipolar acak {-1, 1}^D."""
        raw = self.rng.choice([-1.0, 1.0], size=self.D)
        return raw

    def get_symbol_vector(self, name: str) -> np.ndarray:
        """Mendapatkan atau membuat vektor bipolar stabil untuk suatu simbol/label."""
        if name in self.symbol_cache:
            return self.symbol_cache[name]
        
        # Gunakan hash name sebagai seed untuk determinisme
        h = int(hashlib.md5(name.encode()).hexdigest(), 16) % (2**32)
        local_rng = np.random.default_rng(h)
        vec = local_rng.choice([-1.0, 1.0], size=self.D)
        self.symbol_cache[name] = vec
        return vec

    def project_embedding(self, emb: List[float]) -> np.ndarray:
        """
        Proyeksikan embedding float 768-dim ke ruang bipolar 2048-dim.
        Menggunakan Random Projection dengan threshold sign.
        """
        emb_arr = np.array(emb, dtype=np.float32)
        if emb_arr.shape[0] != 768:
            # Padding atau truncate jika dimensi tidak cocok
            padded = np.zeros(768, dtype=np.float32)
            padded[:min(768, emb_arr.shape[0])] = emb_arr[:min(768, emb_arr.shape[0])]
            emb_arr = padded

        # Perkalian matriks proyeksi
        projected = np.dot(self.projection_matrix, emb_arr)
        
        # Bipolar thresholding: sign(x)
        hdc_vec = np.where(projected >= 0.0, 1.0, -1.0)
        return hdc_vec

    # ── VSA Algebraic Operations ──────────────────────────────────────────

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Operasi BINDING (XOR untuk biner, perkalian elemen untuk bipolar).
        Mengaitkan informasi (e.g., Kunci dengan Nilai).
        Hasilnya adalah vektor yang ortogonal (tidak mirip) dengan a dan b.
        """
        return a * b

    @staticmethod
    def bundle(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Operasi BUNDLING (Penjumlahan elemen + thresholding).
        Menggabungkan informasi (e.g., Set of items).
        Hasilnya adalah vektor rata-rata/superposisi yang mirip dengan anggotanya.
        """
        if not vectors:
            raise ValueError("Tidak dapat mem-bundle list kosong")
        
        total_sum = np.sum(vectors, axis=0)
        
        # Terapkan sign thresholding kembali ke bipolar
        # Jika nilai sum 0, kita tetapkan 1.0 secara deterministik
        bundled = np.where(total_sum >= 0.0, 1.0, -1.0)
        return bundled

    @staticmethod
    def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Operasi PERMUTASI (Circular Shift).
        Menyimpan struktur urutan / posisi sekuensial.
        Permutation bersifat invertible dan orthogonal.
        """
        return np.roll(v, shift)

    # ── Distance / Similarity metrics ─────────────────────────────────────

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Menghitung kemiripan kosinus antara dua vektor bipolar.
        Karena vektor berdimensi D dengan nilai -1 dan 1, ini setara dengan:
        (a . b) / D
        """
        return float(np.dot(a, b) / len(a))


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT COMPRESSOR
# ─────────────────────────────────────────────────────────────────────────────

class ContextCompressor:
    """
    Compressor Konteks berbasis HDC.
    Menerjemahkan riwayat chat/konteks memori kerja menjadi representasi satu vektor HDC.
    """

    def __init__(self, space: Optional[HDCSpace] = None):
        self.space = space if space is not None else HDCSpace()

    def compress_history(self, turns: List[Dict[str, Any]]) -> np.ndarray:
        """
        Mengompresi riwayat chat menjadi satu vektor HDC tunggal.
        
        Format turns:
          [
            {"role": "user", "embedding": [...]},
            {"role": "assistant", "embedding": [...]},
            ...
          ]
          
        Rumus kompresi:
          H = Sum_k [ Π^k (Role_k ⊗ Content_k) ]
        """
        if not turns:
            return self.space.generate_random_vector()

        step_vectors = []
        for idx, turn in enumerate(turns):
            role = turn.get("role", "user")
            emb = turn.get("embedding")
            
            if not emb:
                continue

            # 1. Dapatkan vektor simbolik untuk role
            role_vec = self.space.get_symbol_vector(role)
            
            # 2. Proyeksikan content embedding ke HDC
            content_vec = self.space.project_embedding(emb)
            
            # 3. Bind role dengan content (Role ⊗ Content)
            bound = self.space.bind(role_vec, content_vec)
            
            # 4. Permutasikan berdasarkan urutan index (Π^idx) untuk menyimpan kronologi
            permuted = self.space.permute(bound, shift=idx + 1)
            
            step_vectors.append(permuted)

        if not step_vectors:
            return self.space.generate_random_vector()

        # Bundle semua turn menjadi satu vektor representasi memori
        compressed_vector = self.space.bundle(step_vectors)
        return compressed_vector

    def find_most_similar_turn(self, query_emb: List[float], compressed_history: np.ndarray, turns_count: int) -> int:
        """
        Gunakan unbinding untuk mencari di indeks turn mana query paling cocok.
        """
        query_vec = self.space.project_embedding(query_emb)
        best_idx = -1
        best_sim = -1.0

        for idx in range(turns_count):
            # Lakukan kebalikan permutasi (inverse shift) pada compressed history
            # untuk menganalisis turn ke-idx
            inv_permuted = self.space.permute(compressed_history, shift=-(idx + 1))
            
            # Coba unbind dengan user role vector
            user_role_vec = self.space.get_symbol_vector("user")
            
            # Unbinding: di bipolar, bind bersifat self-inverse! (a * a = 1)
            # Sehingga: (Role ⊗ Content) ⊗ Role = Content
            unbound_content = self.space.bind(inv_permuted, user_role_vec)
            
            sim = self.space.similarity(query_vec, unbound_content)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        return best_idx
