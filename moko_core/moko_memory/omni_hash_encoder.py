"""
MOKO Omni Hash Encoder
========================
Converts text and FP32 vectors into compact OmniAddress objects.
Uses SHA3-256 for content integrity and SimHash for semantic indexing.
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class OmniAddress:
    """
    Representasi alamat unik dalam Omni Index.
    """
    content_hash: str        # SHA3-256 (64 hex chars)
    semantic_bits: int       # SimHash (64-bit uint)
    bucket: int             # 16-bit bucket index
    sub_bucket: int          # 16-bit sub-bucket index
    fp16_vector: bytes       # 768-dim FP16 vector (1536 bytes)
    addr_int: int           # Composite address representation
    folder: str             # string format "{bucket:04x}/{sub_bucket:04x}"

class OmniHashEncoder:
    """
    Encoder untuk memetakan ruang semantik ke dalam struktur bucket 32-bit.
    """
    def __init__(self, dimension: int = 2560):
        # Seed untuk SimHash random projections (stabil)
        self.rng = np.random.default_rng(42)
        self.projections = self.rng.normal(0.0, 1.0, (64, dimension))

    def encode(self, text: str, fp32_vector: List[float]) -> OmniAddress:
        """
        Encode satu pasang (teks, vektor) menjadi OmniAddress.
        """
        # 1. Content Hash (SHA3-256)
        content_hash = hashlib.sha3_256(text.encode('utf-8')).hexdigest()

        # 2. Semantic Bits (SimHash 64-bit)
        vec = np.array(fp32_vector, dtype=np.float32)
        dim = vec.shape[0]
        
        # Penyesuaian dimensi proyeksi secara dinamis jika perlu
        if dim != self.projections.shape[1]:
            # Regenerasi proyeksi untuk dimensi ini (cache-able di prod)
            local_rng = np.random.default_rng(42)
            self.projections = local_rng.normal(0.0, 1.0, (64, dim))
        
        projected = np.dot(self.projections, vec)
        bits = 0
        for i, val in enumerate(projected):
            if val >= 0:
                bits |= (1 << (63 - i))
        
        # 3. Routing (Bucket & Sub-Bucket)
        # Bucket = 16 bit pertama, Sub-Bucket = 16 bit berikutnya
        bucket = (bits >> 48) & 0xFFFF
        sub_bucket = (bits >> 32) & 0xFFFF
        
        # 4. FP16 Vector (D * 2 bytes)
        fp16_vec = vec.astype(np.float16).tobytes()
        
        # 5. Composite Address
        addr_int = bits >> 32  # Ambil 32 bit teratas
        
        return OmniAddress(
            content_hash=content_hash,
            semantic_bits=bits,
            bucket=bucket,
            sub_bucket=sub_bucket,
            fp16_vector=fp16_vec,
            addr_int=addr_int,
            folder=f"{bucket:04x}/{sub_bucket:04x}"
        )

    def encode_batch(self, texts: List[str], vectors: List[List[float]]) -> List[OmniAddress]:
        """Batch encoding untuk efisiensi."""
        return [self.encode(t, v) for t, v in zip(texts, vectors)]

    def compute_hamming(self, bits1: int, bits2: int) -> int:
        """Menghitung Hamming distance antara dua SimHash."""
        xor = bits1 ^ bits2
        return bin(xor).count('1')

    def simhash_to_neighbor_buckets(self, bits: int, flip_bits: int = 1) -> List[Tuple[int, int]]:
        """
        Menemukan bucket tetangga dengan membalik n bit.
        Digunakan untuk memperluas pencarian jika bucket utama sepi.
        """
        neighbors = []
        # Coba balik bit pada 32-bit teratas (routing bits)
        for i in range(32, 64):
            neighbor_bits = bits ^ (1 << i)
            bk = (neighbor_bits >> 48) & 0xFFFF
            sbk = (neighbor_bits >> 32) & 0xFFFF
            neighbors.append((bk, sbk))
        return neighbors

_encoder_instance = None

def get_omni_encoder() -> OmniHashEncoder:
    """Singleton getter."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = OmniHashEncoder()
    return _encoder_instance
