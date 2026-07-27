"""
MOKO Search Cache — L1 Hot-Query Cache
=======================================
Cache LRU (Least Recently Used) untuk hasil pencarian vektor yang sering
diulang dalam satu sesi. Menghindari re-encoding dan re-scanning RSA
untuk pertanyaan yang sama persis atau sangat mirip.

Arsitektur 2-level:
  L1 (SearchCache): Cache hasil pencarian berdasarkan hash vektor query
  L2 (DomainMap Cache): Cache domain mapping berbasis nama topik
                        (sudah ada di rsa_storage._DOMAIN_MAP_CACHE)

Manfaat:
  - Percakapan multi-turn: pertanyaan followup sering pakai embedding mirip
  - Pencarian berulang dari RAG context: tidak perlu scan ulang RSA
  - Menghemat hingga 50ms per query (skip C++ mmap scan)

Konfigurasi:
  CACHE_MAX_SIZE  : Maksimum entri cache (default 256)
  CACHE_TTL_SEC   : Time-to-live entri dalam detik (default 300 = 5 menit)
  SIMILARITY_GATE : Cosine similarity minimum agar dianggap "cache hit"
                    berdasarkan partial embedding (32-D pertama).
"""

import time
import threading
import hashlib
import struct
from collections import OrderedDict
from typing import List, Dict, Optional


# ── Konfigurasi ──────────────────────────────────────────────────────────────
CACHE_MAX_SIZE   = 256    # Jumlah maksimum entri
CACHE_TTL_SEC    = 300    # 5 menit time-to-live
SIMILARITY_GATE  = 0.995  # Threshold cosine sim untuk hit (sangat ketat)


def _vec_to_hash(fp32_vector: List[float], precision: int = 4) -> str:
    """
    Hasilkan hash deterministik dari vektor FP32.
    Pembulatan ke `precision` desimal agar variasi noise kecil tidak miss.
    """
    rounded = [round(x, precision) for x in fp32_vector[:64]]  # 64-D pertama
    raw = struct.pack(f"{len(rounded)}f", *rounded)
    return hashlib.md5(raw).hexdigest()


def _cosine_sim_partial(v1: List[float], v2: List[float], dims: int = 32) -> float:
    """Cosine similarity cepat pada dims dimensi pertama."""
    import math
    a = v1[:dims]
    b = v2[:dims]
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class _CacheEntry:
    __slots__ = ("results", "timestamp", "query_vec_partial")

    def __init__(self, results: List[Dict], query_vec: List[float]):
        self.results = results
        self.timestamp = time.monotonic()
        self.query_vec_partial = query_vec[:32]  # simpan hanya 32-D untuk sim check

    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > CACHE_TTL_SEC


class SearchCache:
    """
    Cache LRU thread-safe untuk hasil pencarian vektor.

    Penggunaan:
        cache = SearchCache()
        hit = cache.get(query_vector)
        if hit is None:
            results = mds.search(query_vector, ...)
            cache.put(query_vector, results)
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._max_size = max_size
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, fp32_vector: List[float]) -> Optional[List[Dict]]:
        """
        Cari hasil dari cache. Return None jika miss atau expired.
        Menggunakan hash + similarity gate untuk fleksibilitas minor noise.
        """
        key = _vec_to_hash(fp32_vector)
        with self._lock:
            if key in self._store:
                entry = self._store[key]
                if entry.is_expired():
                    del self._store[key]
                    self._misses += 1
                    return None
                # Move to end (LRU refresh)
                self._store.move_to_end(key)
                self._hits += 1
                return entry.results

            # Hash miss — coba similarity scan pada partial embedding
            # Ini menangkap kasus di mana hash tidak sama karena noise kecil
            for k, entry in reversed(list(self._store.items())):
                if entry.is_expired():
                    continue
                sim = _cosine_sim_partial(fp32_vector, entry.query_vec_partial)
                if sim >= SIMILARITY_GATE:
                    self._store.move_to_end(k)
                    self._hits += 1
                    return entry.results

        self._misses += 1
        return None

    def put(self, fp32_vector: List[float], results: List[Dict]):
        """Simpan hasil ke cache. Evict entri terlama jika penuh."""
        key = _vec_to_hash(fp32_vector)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(results, fp32_vector)
                return

            self._store[key] = _CacheEntry(results, fp32_vector)
            self._store.move_to_end(key)

            # Evict entri terlama jika melebihi batas
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def invalidate(self):
        """Kosongkan seluruh cache (dipanggil setelah ingest besar)."""
        with self._lock:
            self._store.clear()

    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        with self._lock:
            size = len(self._store)
        return {
            "size": size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }


# ── Singleton global ──────────────────────────────────────────────────────────
search_cache = SearchCache()
