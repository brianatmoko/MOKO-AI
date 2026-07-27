"""
MOKO Omni Storage Engine
==========================
Drop-in replacement for RSAStorage. Manages vector
and text databases using OmniVectorStore and OmniHashEncoder.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from moko_config import settings
from moko_memory.omni_hash_encoder import get_omni_encoder
from moko_memory.omni_vector_store import OmniVectorStore

DEFAULT_DOMAIN = "general"

class OmniStorageEngine:
    """
    Engine utama MOKO-Omni Storage.
    Menyediakan interface drop-in replacement untuk RSAStorage.
    """

    def __init__(self, domain: str = DEFAULT_DOMAIN, root_omni_dir: Optional[Path] = None):
        self.domain = domain
        self.encoder = get_omni_encoder()
        
        # Gunakan path root .moko_omni/
        if root_omni_dir is not None:
            self.root_dir = Path(root_omni_dir)
        else:
            from moko_config import settings
            self.root_dir = Path(settings.OMNI_DIR)

        self.store = OmniVectorStore(domain, base_path=self.root_dir)
        print(f"[OmniStorageEngine] Domain: '{domain}' | Path: {self.store.root}")

    # ─── INGEST ────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        fp32_vector: List[float],
        source_name: str = "manual",
        log_number: int = 1,
        valence: float = 0.0,
        arousal: float = 0.5,
        memory_type: str = "semantic",
        consolidated_count: int = 0
    ) -> Optional[Dict]:
        """
        Menyimpan satu ingatan ke OmniStorage.
        """
        try:
            addr = self.encoder.encode(text, fp32_vector)
            meta = {
                "source": source_name,
                "domain": self.domain,
                "log_number": log_number,
                "valence": valence,
                "arousal": arousal,
                "memory_type": memory_type,
                "consolidated_count": consolidated_count
            }
            return self.store.store(addr, text, meta)
        except Exception as e:
            print(f"[OmniStorage] Error ingest: {e}")
            return None

    def ingest_batch(self, records_list: List[Dict]) -> List[Optional[Dict]]:
        """
        Menyimpan batch ingatan secara efisien menggunakan batch write.
        """
        if not records_list:
            return []

        try:
            texts = [r["text"] for r in records_list]
            vectors = [r["fp32_vector"] for r in records_list]
            metas = []
            for r in records_list:
                metas.append({
                    "source": r.get("source_name", "manual"),
                    "domain": self.domain,
                    "log_number": r.get("log_number", 1),
                    "valence": float(r.get("valence", 0.0)),
                    "arousal": float(r.get("arousal", 0.5)),
                    "memory_type": str(r.get("memory_type", "semantic")),
                    "consolidated_count": int(r.get("consolidated_count", 0))
                })

            # Batch encode
            addrs = self.encoder.encode_batch(texts, vectors)
            
            # Batch store
            return self.store.store_batch(addrs, texts, metas)
        except Exception as e:
            print(f"[OmniStorage] Error ingest_batch: {e}")
            # Fallback jika gagal batch
            results = []
            for r in records_list:
                res = self.ingest(
                    text=r["text"],
                    fp32_vector=r["fp32_vector"],
                    source_name=r.get("source_name", "manual"),
                    log_number=r.get("log_number", 1),
                    valence=float(r.get("valence", 0.0)),
                    arousal=float(r.get("arousal", 0.5)),
                    memory_type=str(r.get("memory_type", "semantic")),
                    consolidated_count=int(r.get("consolidated_count", 0))
                )
                results.append(res)
            return results

    # ─── SEARCH ────────────────────────────────────────────

    def search(
        self,
        fp32_vector: List[float],
        top_k: int = 3,
        n_probe: int = 1,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """
        Mencari ingatan yang paling mirip secara semantik.
        """
        # Encode query vector ke OmniAddress (untuk mendapatkan SimHash bits)
        addr = self.encoder.encode("", fp32_vector)

        # Set default min_score per domain jika tidak dispesifikasi
        if min_score is None:
            # Menggunakan mapping threshold yang sama dengan RSAStorage
            thresholds = {
                "math": 0.55,
                "physics": 0.58,
                "code": 0.60,
                "cybersecurity": 0.60,
                "general": 0.58,
                "lexical": 0.75
            }
            effective_min_score = thresholds.get(self.domain, 0.58)
        else:
            effective_min_score = min_score

        # Gunakan Hamming distance search pada store
        # n_probe disesuaikan untuk neighborhood search (n_probe_extra = n_probe - 1)
        n_probe_extra = max(0, n_probe - 1)
        
        # Max hamming distance diatur ke 24 (cukup dekat)
        raw_results = self.store.search_by_hamming(
            addr=addr,
            fp32_query=fp32_vector,
            top_k=top_k,
            max_hamming=24,
            n_probe_extra=n_probe_extra
        )

        # Filter berdasarkan min_score
        filtered_results = []
        for r in raw_results:
            if r["score"] >= effective_min_score:
                filtered_results.append(r)

        return filtered_results

    # ─── STATS ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """
        Mengembalikan statistik domain.
        """
        stats = self.store.get_stats()
        stats["trained"] = True
        return stats
