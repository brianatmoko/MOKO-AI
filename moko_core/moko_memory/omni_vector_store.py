"""
MOKO Omni Vector Store
========================
Binary storage layer — replaces the old format
(vectors.bin + meta.bin + chain.bin + text_sidecar.jsonl).

Directory structure:
  .moko_omni/
    {domain}/
      _domain_meta.json      ← Domain stats
      {bucket_4hex}/         ← 16-bit bucket
        {sub_bucket_4hex}/   ← 16-bit sub-bucket
          index.bin          ← Header index: (hash32 + bits8 + offset4) × N = 44 bytes/entry
          vectors.f16        ← Stacked FP16 vectors: 1536 bytes × N
          content.bin        ← Packed text entries (zlib compressed blocks)
          meta.jsonl         ← Sidecar metadata (source, valence, arousal, etc.)

Format index.bin (44 bytes per entry):
  [32 bytes] SHA3-256 content hash (hex string, 32 bytes binary representation)
  [8  bytes] semantic_bits uint64 big-endian
  [4  bytes] content_offset uint32 big-endian (byte offset ke content.bin)

Format content.bin:
  Blok berurutan, masing-masing:
  [4 bytes] block_size uint32 big-endian
  [N bytes] zlib.compress(text.encode('utf-8'))

Keunggulan vs format lama:
  1. index.bin: 44 bytes/entry vs 384+64+16=464 bytes → 90% lebih kecil
  2. Hamming search O(bucket_size) vs cosine scan O(all_records)
  3. FP16 vectors: akurasi tinggi, 2x lebih kecil dari FP32
  4. No k-means needed: bucket = simhash bits → instant routing
"""

import json
import os
import struct
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from moko_config import settings
from moko_memory.omni_hash_encoder import (
    OmniAddress, OmniHashEncoder, get_omni_encoder,
)

# ── Konstanta format biner ────────────────────────────────────────────────
INDEX_ENTRY_SIZE   = 44    # bytes per entry di index.bin
HASH_BYTES         = 32    # bytes untuk SHA3-256 (binary repr)
BITS_BYTES         = 8     # bytes untuk semantic_bits uint64
OFFSET_BYTES       = 4     # bytes untuk content_offset uint32
OMNI_DIM           = 2560  # Default dimension (MOKO-AI-4B)
FP16_VECTOR_BYTES  = OMNI_DIM * 2  # D dim × 2 bytes (float16)

# Struct untuk index entry
_INDEX_STRUCT = struct.Struct(">32sQI")   # 32s=hash, Q=bits uint64, I=offset uint32
assert _INDEX_STRUCT.size == INDEX_ENTRY_SIZE

# Struct untuk content block header
_BLOCK_STRUCT = struct.Struct(">I")       # I=block_size uint32


def _get_domain_root(domain: str, base: Optional[Path] = None) -> Path:
    """Mengembalikan root direktori untuk satu domain."""
    if base is None:
        from moko_config import settings
        base = Path(settings.OMNI_DIR)
    else:
        base = Path(base)
    return base / domain


def _get_bucket_path(domain_root: Path, bucket: int, sub_bucket: int) -> Path:
    """Path folder untuk (bucket, sub_bucket)."""
    return domain_root / f"{bucket:04x}" / f"{sub_bucket:04x}"


class BucketStore:
    """
    Penyimpanan satu bucket (bucket, sub_bucket) dalam satu domain.

    Setiap bucket adalah direktori kecil berisi:
      - index.bin     → daftar (hash, bits, offset) semua entry
      - vectors.f16   → stacked FP16 vectors (1536 bytes × N)
      - content.bin   → zlib-compressed text blocks
      - meta.jsonl    → metadata per entry
    """

    def __init__(self, path: Path):
        self.path   = path
        self.path.mkdir(parents=True, exist_ok=True)

        self.index_path   = path / "index.bin"
        self.vector_path  = path / "vectors.f16"
        self.content_path = path / "content.bin"
        self.meta_path    = path / "meta.jsonl"

        # In-memory caches — avoid re-reading same files on repeated queries
        self._idx_cache:  Optional[bytes]           = None  # raw index.bin bytes
        self._vec_cache:  Optional[np.ndarray]      = None  # vectors as float32 array
        self._meta_cache: Optional[Dict[int, Dict]] = None  # meta_map by idx

    def count(self) -> int:
        """Jumlah entri dalam bucket (dari cache jika ada)."""
        if self._idx_cache is not None:
            return len(self._idx_cache) // INDEX_ENTRY_SIZE
        if not self.index_path.exists():
            return 0
        return self.index_path.stat().st_size // INDEX_ENTRY_SIZE

    def _get_index_bytes(self) -> bytes:
        """Baca index.bin sekali, cache di memory untuk panggilan berikutnya."""
        if self._idx_cache is None:
            self._idx_cache = self.index_path.read_bytes() if self.index_path.exists() else b''
        return self._idx_cache

    def _get_vectors(self) -> np.ndarray:
        """Baca vectors.f16 sekali, cache di memory."""
        if self._vec_cache is None:
            if not self.vector_path.exists():
                self._vec_cache = np.empty((0, OMNI_DIM), dtype=np.float32)
            else:
                raw = self.vector_path.read_bytes()
                n = self.count()
                if n == 0 or len(raw) == 0:
                    self._vec_cache = np.empty((0, OMNI_DIM), dtype=np.float32)
                else:
                    dim = (len(raw) // n) // 2
                    if dim > 0:
                        arr = np.frombuffer(raw, dtype=np.float16).reshape(n, dim)
                        self._vec_cache = arr.astype(np.float32)
                    else:
                        self._vec_cache = np.empty((0, OMNI_DIM), dtype=np.float32)
        return self._vec_cache

    def _get_meta_map(self) -> Dict[int, Dict]:
        """Baca meta.jsonl sekali, cache di memory."""
        if self._meta_cache is None:
            self._meta_cache = {}
            if self.meta_path.exists():
                for line in self.meta_path.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self._meta_cache[obj["idx"]] = obj
                    except Exception:
                        pass
        return self._meta_cache


    def append(
        self,
        addr: OmniAddress,
        text: str,
        meta: Dict,
    ) -> int:
        """
        Tambah satu entry ke bucket.

        Returns:
            record_index (0-based) dari entry baru
        """
        # 1. Hitung content_offset (posisi di content.bin sebelum append)
        content_offset = self.content_path.stat().st_size if self.content_path.exists() else 0
        record_index   = self.count()

        # 2. Kompres dan tulis teks ke content.bin
        compressed = zlib.compress(text.encode('utf-8'), level=6)
        with open(self.content_path, 'ab') as f:
            f.write(_BLOCK_STRUCT.pack(len(compressed)))
            f.write(compressed)

        # 3. Tulis hash binary (ambil 32 bytes dari hex SHA3)
        hash_bytes = bytes.fromhex(addr.content_hash)[:HASH_BYTES]

        # 4. Tulis index.bin
        with open(self.index_path, 'ab') as f:
            f.write(_INDEX_STRUCT.pack(hash_bytes, addr.semantic_bits, content_offset))

        # 5. Tulis FP16 vector
        with open(self.vector_path, 'ab') as f:
            f.write(addr.fp16_vector)

        # 6. Tulis metadata
        meta_entry = {
            "idx":    record_index,
            "hash":   addr.content_hash,
            "source": meta.get("source", "unknown"),
            "domain": meta.get("domain", "general"),
            "log":    meta.get("log_number", 1),
            "val":    float(meta.get("valence", 0.0)),
            "ar":     float(meta.get("arousal", 0.5)),
            "mtype":  meta.get("memory_type", "semantic"),
            "cc":     int(meta.get("consolidated_count", 0)),
            "ts":     time.time(),
        }
        with open(self.meta_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(meta_entry, ensure_ascii=False) + '\n')

        return record_index

    def append_batch(
        self,
        addrs: List[OmniAddress],
        texts: List[str],
        metas: List[Dict],
    ) -> List[int]:
        """
        Tambah beberapa entry ke bucket sekaligus untuk meminimalkan I/O.
        """
        n_entries = len(addrs)
        if n_entries == 0:
            return []

        content_offset = self.content_path.stat().st_size if self.content_path.exists() else 0
        record_index   = self.count()

        indices_payload = bytearray()
        vectors_payload = bytearray()
        content_payload = bytearray()
        meta_lines = []
        record_indices = []

        for i in range(n_entries):
            addr = addrs[i]
            text = texts[i]
            meta = metas[i]

            # Compressed text payload
            compressed = zlib.compress(text.encode('utf-8'), level=6)
            content_payload.extend(_BLOCK_STRUCT.pack(len(compressed)))
            content_payload.extend(compressed)

            # Hash bytes
            hash_bytes = bytes.fromhex(addr.content_hash)[:HASH_BYTES]

            # Index payload
            indices_payload.extend(_INDEX_STRUCT.pack(hash_bytes, addr.semantic_bits, content_offset))

            # Vector payload
            vectors_payload.extend(addr.fp16_vector)

            # Meta entry
            meta_entry = {
                "idx":    record_index + i,
                "hash":   addr.content_hash,
                "source": meta.get("source", "unknown"),
                "domain": meta.get("domain", "general"),
                "log":    meta.get("log_number", 1),
                "val":    float(meta.get("valence", 0.0)),
                "ar":     float(meta.get("arousal", 0.5)),
                "mtype":  meta.get("memory_type", "semantic"),
                "cc":     int(meta.get("consolidated_count", 0)),
                "ts":     time.time(),
            }
            meta_lines.append(json.dumps(meta_entry, ensure_ascii=False) + '\n')
            record_indices.append(record_index + i)

            # Offset untuk item berikutnya bertambah sesuai ukuran compressed block
            content_offset += 4 + len(compressed)

        # Tulis semua payload sekaligus ke disk
        with open(self.content_path, 'ab') as f:
            f.write(content_payload)
        with open(self.index_path, 'ab') as f:
            f.write(indices_payload)
        with open(self.vector_path, 'ab') as f:
            f.write(vectors_payload)
        with open(self.meta_path, 'a', encoding='utf-8') as f:
            f.writelines(meta_lines)

        return record_indices

    def read_all(self) -> List[Dict]:
        """
        Baca semua entri dalam bucket.

        Returns:
            List dict dengan keys: hash, semantic_bits, fp16_vector, text, meta, record_index
        """
        n = self.count()
        if n == 0:
            return []

        # Baca index.bin sekaligus
        raw_index = self.index_path.read_bytes()

        # Baca semua FP16 vectors sekaligus
        raw_vecs = self.vector_path.read_bytes() if self.vector_path.exists() else b''
        vector_bytes_per_entry = (len(raw_vecs) // n) if n > 0 else 0
        has_vecs = vector_bytes_per_entry > 0 and (len(raw_vecs) == n * vector_bytes_per_entry)

        # Baca metadata JSONL
        meta_map: Dict[int, Dict] = {}
        if self.meta_path.exists():
            for line in self.meta_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    meta_map[obj["idx"]] = obj
                except Exception:
                    pass

        # Parse setiap index entry
        results = []
        content_cache: Dict[int, str] = {}

        for i in range(n):
            entry_bytes = raw_index[i * INDEX_ENTRY_SIZE: (i + 1) * INDEX_ENTRY_SIZE]
            hash_b, sem_bits, content_offset = _INDEX_STRUCT.unpack(entry_bytes)
            content_hash = hash_b.hex()

            # Baca FP16 vector
            fp16 = raw_vecs[i * vector_bytes_per_entry: (i + 1) * vector_bytes_per_entry] if has_vecs else b''

            # Decode text dari content.bin (lazy, dengan cache per offset)
            if content_offset not in content_cache:
                text = self._read_text_at(content_offset)
                content_cache[content_offset] = text
            text = content_cache[content_offset]

            m = meta_map.get(i, {})
            results.append({
                "record_index":       i,
                "content_hash":       content_hash,
                "semantic_bits":      int(sem_bits),
                "fp16_vector":        fp16,
                "text":               text,
                "source":             m.get("source", "?"),
                "domain":             m.get("domain", "general"),
                "log_number":         m.get("log", 1),
                "valence":            float(m.get("val", 0.0)),
                "arousal":            float(m.get("ar", 0.5)),
                "memory_type":        m.get("mtype", "semantic"),
                "consolidated_count": int(m.get("cc", 0)),
                "timestamp":          float(m.get("ts", 0.0)),
            })

        return results

    def read_vectors_only(self) -> np.ndarray:
        """
        Baca hanya FP16 vectors sebagai numpy array (N, dim) float32.
        Sangat cepat untuk batch cosine similarity.
        """
        if not self.vector_path.exists():
            return np.empty((0, OMNI_DIM), dtype=np.float32)
        raw = self.vector_path.read_bytes()
        n = self.count()
        if n == 0 or len(raw) == 0:
            return np.empty((0, OMNI_DIM), dtype=np.float32)
        
        dim = (len(raw) // n) // 2
        if dim <= 0:
            return np.empty((0, OMNI_DIM), dtype=np.float32)
        arr = np.frombuffer(raw, dtype=np.float16).reshape(n, dim)
        return arr.astype(np.float32)

    def read_index_only(self) -> List[Tuple[str, int]]:
        """
        Baca hanya (content_hash, semantic_bits) tanpa teks/vektor.
        Menggunakan cache in-memory agar disk hanya dibaca sekali.
        """
        raw = self._get_index_bytes()
        n   = len(raw) // INDEX_ENTRY_SIZE
        results = []
        for i in range(n):
            entry = raw[i * INDEX_ENTRY_SIZE: (i + 1) * INDEX_ENTRY_SIZE]
            hash_b, sem_bits, _ = _INDEX_STRUCT.unpack(entry)
            results.append((hash_b.hex(), int(sem_bits)))
        return results

    def exact_lookup(self, content_hash: str) -> Optional[Dict]:
        """
        O(n) exact lookup berdasarkan SHA3 hash.
        Note: Untuk true O(1) lookup, gunakan domain-level hash index.
        """
        entries = self.read_all()
        for e in entries:
            if e["content_hash"] == content_hash:
                return e
        return None

    def _read_text_at(self, offset: int) -> str:
        """Baca dan decompress satu text block dari content.bin pada offset tertentu."""
        if not self.content_path.exists():
            return ""
        try:
            with open(self.content_path, 'rb') as f:
                f.seek(offset)
                size_bytes = f.read(4)
                if len(size_bytes) < 4:
                    return ""
                block_size = _BLOCK_STRUCT.unpack(size_bytes)[0]
                compressed = f.read(block_size)
                return zlib.decompress(compressed).decode('utf-8')
        except Exception:
            return ""


class OmniVectorStore:
    """
    Store seluruh domain: mengatur routing ke BucketStore yang tepat.

    Ini adalah interface level domain — satu instance per domain.
    BucketStore diinstansiasi secara lazy per bucket.
    """

    # ── Process-level stats cache ─────────────────────────────────────────────
    # Key: str(root_path) → Dict stats
    # Menghindari scan 100k+ direktori berulang kali. get_stats() menjadi O(1)
    # setelah pemanggilan pertama.
    _STATS_CACHE: Dict[str, Dict] = {}

    def __init__(self, domain: str, base_path: Optional[Path] = None):
        self.domain      = domain
        self.root        = _get_domain_root(domain, base_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path  = self.root / "_domain_meta.json"

        # Cache BucketStore per (bucket, sub_bucket)
        self._bucket_cache: Dict[Tuple[int, int], BucketStore] = {}

        # Hash index untuk O(1) exact lookup: content_hash → (bucket, sub_bucket, idx)
        self._hash_index: Dict[str, Tuple[int, int, int]] = {}
        self._hash_index_loaded = False

        # Coba muat stats dari _domain_meta.json jika ada (persistent cache)
        self._try_load_stats_from_disk()

    def _try_load_stats_from_disk(self):
        """Muat stats dari _domain_meta.json ke _STATS_CACHE tanpa scan direktori."""
        cache_key = str(self.root)
        if cache_key in OmniVectorStore._STATS_CACHE:
            return  # sudah ada di memory
        if self._meta_path.exists():
            try:
                data = json.loads(self._meta_path.read_text())
                if "total_memories" in data and data["total_memories"] > 0:
                    OmniVectorStore._STATS_CACHE[cache_key] = {
                        "domain":         self.domain,
                        "total_memories": data["total_memories"],
                        "active_buckets": data.get("active_buckets", 0),
                        "root":           str(self.root),
                    }
            except Exception:
                pass

    def has_data(self) -> bool:
        """
        Cek cepat apakah domain ini memiliki data.
        O(1) jika stats sudah di-cache, O(1) filesystem check jika belum.
        Jauh lebih cepat dari get_stats()["total_memories"] > 0.
        """
        cache_key = str(self.root)
        if cache_key in OmniVectorStore._STATS_CACHE:
            return OmniVectorStore._STATS_CACHE[cache_key]["total_memories"] > 0
        # Fast path: cek apakah ada setidaknya 1 index.bin (tanpa scan semua bucket)
        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            for l2 in l1.iterdir():
                if l2.is_dir() and (l2 / "index.bin").exists():
                    return True
            break  # cukup cek 1 level-1 dir saja
        return False


    def get_bucket(self, bucket: int, sub_bucket: int) -> BucketStore:
        """Lazy-load BucketStore, cache instance."""
        key = (bucket, sub_bucket)
        if key not in self._bucket_cache:
            path = _get_bucket_path(self.root, bucket, sub_bucket)
            self._bucket_cache[key] = BucketStore(path)
        return self._bucket_cache[key]

    def store(
        self,
        addr: OmniAddress,
        text: str,
        meta: Dict,
    ) -> Dict:
        """
        Simpan satu entry ke bucket yang tepat.

        Returns:
            Dict dengan folder, domain, record_index, content_hash
        """
        bs  = self.get_bucket(addr.bucket, addr.sub_bucket)
        idx = bs.append(addr, text, meta)

        # Update in-memory hash index
        self._hash_index[addr.content_hash] = (addr.bucket, addr.sub_bucket, idx)

        return {
            "folder":       addr.folder,
            "domain":       self.domain,
            "alpha":        addr.bucket >> 8,
            "digit":        addr.sub_bucket,
            "record_index": idx,
            "content_hash": addr.content_hash,
            "addr_int":     addr.addr_int,
            "error":        0.0,
        }

    def store_batch(
        self,
        addrs: List[OmniAddress],
        texts: List[str],
        metas: List[Dict],
    ) -> List[Dict]:
        """
        Menyimpan batch entri secara efisien dengan mengelompokkannya berdasarkan bucket.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(len(addrs)):
            addr = addrs[i]
            groups[(addr.bucket, addr.sub_bucket)].append(i)

        results = [None] * len(addrs)
        for (bk, sbk), idxs in groups.items():
            bs = self.get_bucket(bk, sbk)
            group_addrs = [addrs[i] for i in idxs]
            group_texts = [texts[i] for i in idxs]
            group_metas = [metas[i] for i in idxs]

            rec_idxs = bs.append_batch(group_addrs, group_texts, group_metas)
            for local_idx, orig_idx in enumerate(idxs):
                addr = addrs[orig_idx]
                results[orig_idx] = {
                    "folder":       addr.folder,
                    "domain":       self.domain,
                    "alpha":        addr.bucket >> 8,
                    "digit":        addr.sub_bucket,
                    "record_index": rec_idxs[local_idx],
                    "content_hash": addr.content_hash,
                    "addr_int":     addr.addr_int,
                    "error":        0.0,
                }
                # Update in-memory hash index
                self._hash_index[addr.content_hash] = (addr.bucket, addr.sub_bucket, rec_idxs[local_idx])

    def read_all(self) -> List[Dict]:
        """
        Membaca semua entri di seluruh bucket domain ini.
        """
        results = []
        if not self.root.exists():
            return results
        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            for l2 in l1.iterdir():
                if not l2.is_dir():
                    continue
                idx_file = l2 / "index.bin"
                if idx_file.exists() and idx_file.stat().st_size > 0:
                    try:
                        bucket = int(l1.name, 16)
                        sub_bucket = int(l2.name, 16)
                        store = self.get_bucket(bucket, sub_bucket)
                        for entry in store.read_all():
                            entry["alpha"] = bucket
                            entry["digit"] = sub_bucket
                            results.append(entry)
                    except Exception:
                        pass
        return results

    def exact_lookup(self, content_hash: str) -> Optional[Dict]:
        """
        O(1) exact lookup menggunakan in-memory hash index.
        Jika hash index belum loaded, scan semua bucket (satu kali).
        """
        if not self._hash_index_loaded:
            self._build_hash_index()

        loc = self._hash_index.get(content_hash)
        if loc is None:
            return None

        bucket, sub_bucket, _ = loc
        bs = self.get_bucket(bucket, sub_bucket)
        return bs.exact_lookup(content_hash)

    def search_by_hamming(
        self,
        addr: OmniAddress,
        fp32_query: List[float],
        top_k: int = 3,
        max_hamming: int = 16,
        n_probe_extra: int = 1,
    ) -> List[Dict]:
        """
        Approximate nearest neighbor search menggunakan Hamming distance + FP16 cosine re-rank.

        Strategy:
          1. Cari di bucket utama (bucket, sub_bucket) → semua entri dalam bucket ini
          2. Cari di neighbor buckets (flip 1 bit) → n_probe_extra bucket tetangga
          3. Pre-filter: Hamming(query_bits, stored_bits) <= max_hamming
          4. Re-rank: cosine similarity FP16 vs FP32 query
          5. Return top-k

        Args:
            addr:          OmniAddress dari query
            fp32_query:    Query vector FP32 untuk re-ranking
            top_k:         Jumlah hasil
            max_hamming:   Batas maksimum Hamming distance untuk pre-filter
            n_probe_extra: Jumlah bucket tetangga yang dicek (0 = hanya bucket utama)
        """
        enc = get_omni_encoder()

        # Kumpulkan semua bucket yang perlu dicek
        # n_probe_extra=1 → 8 tetangga (optimal antara coverage vs kecepatan)
        buckets_to_check = [(addr.bucket, addr.sub_bucket)]
        if n_probe_extra > 0:
            neighbors = enc.simhash_to_neighbor_buckets(addr.semantic_bits, flip_bits=1)
            buckets_to_check.extend(neighbors[:n_probe_extra * 8])

        candidates: List[Dict] = []

        for (bk, sbk) in set(map(tuple, buckets_to_check)):
            bs = self.get_bucket(bk, sbk)
            # Gunakan cache bytes — tidak baca disk jika sudah ter-cache
            idx_bytes = bs._get_index_bytes()
            if len(idx_bytes) == 0:
                continue
            n_entries = len(idx_bytes) // INDEX_ENTRY_SIZE
            if n_entries == 0:
                continue

            # Hamming pre-filter langsung dari bytes (tanpa parse semua tuple dulu)
            candidate_indices = []
            for i in range(n_entries):
                entry = idx_bytes[i * INDEX_ENTRY_SIZE: (i + 1) * INDEX_ENTRY_SIZE]
                _, stored_bits, _ = _INDEX_STRUCT.unpack(entry)
                hamming = enc.compute_hamming(addr.semantic_bits, stored_bits)
                if hamming <= max_hamming:
                    candidate_indices.append((i, stored_bits, hamming))

            if not candidate_indices:
                continue

            # Baca vectors dari cache (bukan disk)
            vecs_matrix = bs._get_vectors()

            if vecs_matrix.shape[0] > 0 and candidate_indices:
                # Gunakan numpy batch multiply
                cand_idxs = [i for i, _, _ in candidate_indices if i < vecs_matrix.shape[0]]
                if cand_idxs:
                    q_norm = np.asarray(fp32_query, dtype=np.float32)
                    qn     = np.linalg.norm(q_norm)
                    if qn > 0:
                        q_norm /= qn
                    # Pad/truncate query ke dimensi vektor tersimpan
                    dim = vecs_matrix.shape[1]
                    if len(q_norm) != dim:
                        if len(q_norm) < dim:
                            q_norm = np.pad(q_norm, (0, dim - len(q_norm)))
                        else:
                            q_norm = q_norm[:dim]
                        qn2 = np.linalg.norm(q_norm)
                        if qn2 > 0:
                            q_norm = q_norm / qn2
                    sub_vecs = vecs_matrix[cand_idxs]     # (M, dim)
                    norms    = np.linalg.norm(sub_vecs, axis=1, keepdims=True)
                    norms    = np.where(norms > 0, norms, 1.0)
                    sub_vecs = sub_vecs / norms
                    scores   = (sub_vecs @ q_norm).astype(np.float32)   # (M,)

                    # Baca meta dari cache (tidak baca disk lagi)
                    meta_map = bs._get_meta_map()

                    for rank, ci in enumerate(cand_idxs):
                        # Baca teks hanya jika score cukup tinggi (lazy decompression)
                        score = float(scores[rank])
                        # Ambil offset dari index bytes
                        entry_bytes = idx_bytes[ci * INDEX_ENTRY_SIZE: (ci + 1) * INDEX_ENTRY_SIZE]
                        hash_b, _, content_offset = _INDEX_STRUCT.unpack(entry_bytes)
                        content_hash = hash_b.hex()
                        m = meta_map.get(ci, {})
                        candidates.append({
                            "text":               "",  # Lazy: teks dibaca setelah top-k dipilih
                            "source":             m.get("source", "?"),
                            "domain":             m.get("domain", self.domain),
                            "log_number":         m.get("log", 1),
                            "score":              score,
                            "folder":             f"{bk:04x}/{sbk:04x}",
                            "idx":                ci,
                            "valence":            float(m.get("val", 0.0)),
                            "arousal":            float(m.get("ar", 0.5)),
                            "memory_type":        m.get("mtype", "semantic"),
                            "consolidated_count": int(m.get("cc", 0)),
                            "content_hash":       content_hash,
                            "_bk":                bk,
                            "_sbk":               sbk,
                            "_offset":            content_offset,
                        })

        # Re-rank semua kandidat berdasarkan cosine score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:top_k]

        # Lazy text loading: hanya dekompresi teks untuk top-k yang terpilih
        for c in top_candidates:
            if c["text"] == "" and "_offset" in c:
                bs = self.get_bucket(c["_bk"], c["_sbk"])
                c["text"] = bs._read_text_at(c["_offset"])

        return top_candidates


    def search_linear(self, fp32_query: List[float], top_k: int = 3) -> List[Dict]:
        """
        Pencarian linier fallback yang sangat cepat dan efisien.
        Hanya memuat dan menghitung vektor FP16, dan hanya men-dekompresi teks untuk top_k hasil akhir.
        """
        import struct

        q_norm = np.asarray(fp32_query, dtype=np.float32)
        qn     = np.linalg.norm(q_norm)
        if qn > 0:
            q_norm /= qn

        all_candidates = []

        if not self.root.exists():
            return []

        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            for l2 in l1.iterdir():
                if not l2.is_dir():
                    continue

                idx_path = l2 / "index.bin"
                vec_path = l2 / "vectors.f16"

                if not (idx_path.exists() and vec_path.exists()):
                    continue

                try:
                    idx_data = idx_path.read_bytes()
                    vec_data = vec_path.read_bytes()
                    entry_count = len(idx_data) // INDEX_ENTRY_SIZE
                    if entry_count == 0:
                        continue

                    dim = (len(vec_data) // entry_count) // 2
                    if dim > 0:
                        vecs = np.frombuffer(vec_data, dtype=np.float16).reshape(entry_count, dim).astype(np.float32)
                        
                        # Pad or truncate query vector to match stored dimension
                        q_vec = q_norm.copy()
                        if len(q_vec) != dim:
                            if len(q_vec) < dim:
                                q_vec = np.pad(q_vec, (0, dim - len(q_vec)))
                            else:
                                q_vec = q_vec[:dim]
                            q_norm_len = np.linalg.norm(q_vec)
                            if q_norm_len > 0:
                                q_vec /= q_norm_len
                            else:
                                q_vec = np.zeros(dim, dtype=np.float32)
                                
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        norms = np.where(norms > 0, norms, 1.0)
                        vecs /= norms

                        scores = vecs @ q_vec
                    else:
                        scores = np.zeros(entry_count, dtype=np.float32)

                    # Simpan kandidat tanpa memuat teks
                    for i in range(entry_count):
                        score = float(scores[i])
                        # Ambil hash dan offset dari index entry
                        entry_bytes = idx_data[i * INDEX_ENTRY_SIZE : (i + 1) * INDEX_ENTRY_SIZE]
                        hash_b, _, content_offset = _INDEX_STRUCT.unpack(entry_bytes)
                        content_hash = hash_b.hex()

                        all_candidates.append({
                            "score":          score,
                            "bucket":         int(l1.name, 16),
                            "sub_bucket":     int(l2.name, 16),
                            "idx":            i,
                            "content_hash":   content_hash,
                            "content_offset": content_offset,
                        })
                except Exception:
                    pass

        # Urutkan kandidat berdasarkan score tertinggi
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = all_candidates[:top_k]

        # Hanya dekompresi teks dan baca meta untuk top_k terpilih!
        results = []
        for c in top_candidates:
            try:
                bs = self.get_bucket(c["bucket"], c["sub_bucket"])

                # Baca text pada offset spesifik
                text = bs._read_text_at(c["content_offset"])

                # Baca meta entry yang sesuai dari meta.jsonl jika ada
                meta = {}
                if bs.meta_path.exists():
                    with open(bs.meta_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if obj.get("idx") == c["idx"]:
                                    meta = obj
                                    break
                            except Exception:
                                pass

                results.append({
                    "text":               text,
                    "source":             meta.get("source", "?"),
                    "domain":             meta.get("domain", self.domain),
                    "log_number":         meta.get("log", 1),
                    "score":              c["score"],
                    "folder":             f"{c['bucket']:04x}/{c['sub_bucket']:04x}",
                    "idx":                c["idx"],
                    "valence":            float(meta.get("val", 0.0)),
                    "arousal":            float(meta.get("ar", 0.5)),
                    "memory_type":        meta.get("mtype", "semantic"),
                    "consolidated_count": int(meta.get("cc", 0)),
                    "content_hash":       c["content_hash"],
                })
            except Exception:
                pass

        return results

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 5,
        max_buckets_scan: int = 2000,
    ) -> List[Dict]:
        """
        Pencarian berbasis kata kunci pada bucket RAG menggunakan C++ native fast-path.

        Fast-path  : C++ std::filesystem traversal (search_keywords_cpp) < 50ms
        Slow-path  : Python iterdir fallback jika kernel tidak tersedia

        C++ mencari file meta.jsonl yang mengandung semua keyword, mengembalikan
        path JSON, lalu Python hanya membaca bucket yang cocok (konten teks).

        Args:
            keywords        : List kata kunci yang dicari (case-insensitive)
            top_k           : Jumlah hasil maksimum
            max_buckets_scan: Batas maksimum folder (hanya untuk Python fallback)
        """
        if not keywords or not self.root.exists():
            return []

        # ── FAST PATH: C++ native filesystem traversal ──────────────────────
        try:
            from moko_memory.disk_manager import MokoKernelBridge
            bridge = MokoKernelBridge()
            if bridge.initialized:
                matched_meta_paths = bridge.search_keywords_native(
                    directory=str(self.root),
                    keywords=keywords,
                    max_results=max_buckets_scan,
                )
                if matched_meta_paths is not None:
                    return self._read_buckets_from_paths(matched_meta_paths, keywords, top_k)
        except Exception:
            pass  # fallback ke Python di bawah

        # ── SLOW PATH: Python iterdir (fallback jika kernel tidak tersedia) ──
        kw_lower  = [k.lower().replace(" ", "_") for k in keywords]
        kw_spaced = [k.lower() for k in keywords]

        matched_buckets: List[tuple] = []
        scanned = 0

        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            for l2 in l1.iterdir():
                if not l2.is_dir() or scanned >= max_buckets_scan:
                    break
                scanned += 1
                meta_path = l2 / "meta.jsonl"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    if not first_line:
                        continue
                    obj    = json.loads(first_line)
                    source = obj.get("source", "").lower()
                    if any(kw in source for kw in kw_lower) or any(kw in source for kw in kw_spaced):
                        try:
                            matched_buckets.append((int(l1.name, 16), int(l2.name, 16)))
                        except ValueError:
                            pass
                except Exception:
                    pass

        return self._collect_results_from_buckets(matched_buckets, top_k)

    # ── Helper: baca hasil dari list path meta.jsonl (C++ fast-path output) ──
    def _read_buckets_from_paths(
        self, meta_paths: List[str], keywords: List[str], top_k: int
    ) -> List[Dict]:
        """
        Membaca konten bucket secara efisien dari path meta.jsonl yang dikembalikan C++ kernel.
        Menggunakan lazy text loading untuk meminimalkan I/O dan dekompresi zlib.
        """
        kw_lower = [k.lower() for k in keywords]
        results : List[Dict] = []
        seen_hashes: set = set()

        for mp in meta_paths:
            if len(results) >= top_k:
                break
            try:
                p = Path(mp)
                l2_dir = p.parent          # e.g. .../0abc/02de/
                l1_dir = l2_dir.parent
                bk     = int(l1_dir.name, 16)
                sbk    = int(l2_dir.name, 16)
                bs     = self.get_bucket(bk, sbk)

                if not p.exists():
                    continue

                # Baca meta.jsonl baris demi baris — sangat cepat dibanding read_all()
                meta_entries = []
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Cek keyword match pada source atau domain
                            # (C++ bridge sudah men-filter di file level, kita tinggal ambil datanya)
                            h = obj.get("hash", "")
                            if h and h not in seen_hashes:
                                meta_entries.append(obj)
                        except Exception:
                            pass

                if not meta_entries:
                    continue

                # Baca index.bin untuk mencari offset konten
                idx_bytes = bs._get_index_bytes()
                n_idx = len(idx_bytes) // INDEX_ENTRY_SIZE

                for m in meta_entries:
                    idx = m.get("idx", 0)
                    h = m.get("hash", "")
                    if h in seen_hashes:
                        continue
                    
                    # Cari offset konten dari index bytes
                    content_offset = 0
                    if idx < n_idx:
                        entry = idx_bytes[idx * INDEX_ENTRY_SIZE: (idx + 1) * INDEX_ENTRY_SIZE]
                        _, _, content_offset = _INDEX_STRUCT.unpack(entry)

                    seen_hashes.add(h)
                    
                    # Teks dibaca secara lazy (hanya jika masuk top_k hasil akhir)
                    results.append({
                        "text":               "",  # Diisi secara lazy di bawah
                        "source":             m.get("source", "?"),
                        "domain":             m.get("domain", self.domain),
                        "log_number":         m.get("log", 1),
                        "score":              0.78,  # Default keyword score
                        "folder":             f"{bk:04x}/{sbk:04x}",
                        "idx":                idx,
                        "valence":            float(m.get("val", 0.0)),
                        "arousal":            float(m.get("ar", 0.5)),
                        "memory_type":        m.get("mtype", "semantic"),
                        "consolidated_count": int(m.get("cc", 0)),
                        "content_hash":       h,
                        "_bk":                bk,
                        "_sbk":               sbk,
                        "_offset":            content_offset,
                    })

                    if len(results) >= top_k:
                        break
            except Exception:
                pass

        # Lazy decompress teks untuk top_k terpilih
        for r in results:
            if r["text"] == "" and "_offset" in r:
                try:
                    bs = self.get_bucket(r["_bk"], r["_sbk"])
                    r["text"] = bs._read_text_at(r["_offset"])
                except Exception:
                    pass

        return results


    def _collect_results_from_buckets(
        self, matched_buckets: List[tuple], top_k: int
    ) -> List[Dict]:
        """Membaca konten dari daftar (bk, sbk) bucket tuple secara efisien."""
        results    : List[Dict] = []
        seen_hashes: set = set()
        n_buckets = max(1, len(matched_buckets))

        for (bk, sbk) in matched_buckets:
            if len(results) >= top_k:
                break
            try:
                bs          = self.get_bucket(bk, sbk)
                # Gunakan lazy reading: ambil meta JSONL dan index offset terlebih dahulu
                idx_bytes = bs._get_index_bytes()
                n_idx = len(idx_bytes) // INDEX_ENTRY_SIZE
                meta_map = bs._get_meta_map()
                
                per_bucket  = max(1, top_k // n_buckets)
                added = 0
                for idx in sorted(meta_map.keys()):
                    if added >= per_bucket or len(results) >= top_k:
                        break
                    m = meta_map[idx]
                    h = m.get("hash", "")
                    if h in seen_hashes:
                        continue
                    
                    content_offset = 0
                    if idx < n_idx:
                        entry = idx_bytes[idx * INDEX_ENTRY_SIZE: (idx + 1) * INDEX_ENTRY_SIZE]
                        _, _, content_offset = _INDEX_STRUCT.unpack(entry)

                    seen_hashes.add(h)
                    results.append({
                        "text":               "",  # Diisi lazy
                        "source":             m.get("source", "?"),
                        "domain":             m.get("domain", self.domain),
                        "log_number":         m.get("log", 1),
                        "score":              0.70,
                        "folder":             f"{bk:04x}/{sbk:04x}",
                        "idx":                idx,
                        "valence":            float(m.get("val", 0.0)),
                        "arousal":            float(m.get("ar", 0.5)),
                        "memory_type":        m.get("mtype", "semantic"),
                        "consolidated_count": int(m.get("cc", 0)),
                        "content_hash":       h,
                        "_bk":                bk,
                        "_sbk":               sbk,
                        "_offset":            content_offset,
                    })
                    added += 1

            except Exception:
                pass

        # Lazy decompress teks untuk top_k terpilih
        for r in results:
            if r["text"] == "" and "_offset" in r:
                try:
                    bs = self.get_bucket(r["_bk"], r["_sbk"])
                    r["text"] = bs._read_text_at(r["_offset"])
                except Exception:
                    pass

        return results


    def get_stats(self, force_rescan: bool = False) -> Dict:
        """
        Statistik domain: jumlah entri, bucket aktif, dll.

        Menggunakan in-memory cache (_STATS_CACHE) agar O(1) setelah scan pertama.
        Scan filesystem hanya dilakukan sekali lalu hasilnya disimpan.

        Args:
            force_rescan: Jika True, paksa scan ulang dan update cache.
        """
        cache_key = str(self.root)
        if not force_rescan and cache_key in OmniVectorStore._STATS_CACHE:
            return OmniVectorStore._STATS_CACHE[cache_key]

        # Scan filesystem (hanya dilakukan sekali per proses)
        total   = 0
        buckets = 0
        t0 = time.time()
        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            for l2 in l1.iterdir():
                if not l2.is_dir():
                    continue
                idx = l2 / "index.bin"
                if idx.exists():
                    n = idx.stat().st_size // INDEX_ENTRY_SIZE
                    if n > 0:
                        total   += n
                        buckets += 1
        elapsed = time.time() - t0

        result = {
            "domain":         self.domain,
            "total_memories": total,
            "active_buckets": buckets,
            "root":           str(self.root),
        }

        # Simpan ke process-level cache
        OmniVectorStore._STATS_CACHE[cache_key] = result

        # Juga tulis ke _domain_meta.json untuk persistent cache
        try:
            self._meta_path.write_text(json.dumps({
                "domain":         self.domain,
                "total_memories": total,
                "active_buckets": buckets,
                "scanned_at":     time.time(),
                "scan_sec":       round(elapsed, 2),
            }))
        except Exception:
            pass

        return result


    def _build_hash_index(self):
        """Scan semua bucket dan bangun in-memory hash index (dijalankan satu kali)."""
        for l1 in self.root.iterdir():
            if not l1.is_dir() or l1.name.startswith('_'):
                continue
            try:
                bk = int(l1.name, 16)
            except ValueError:
                continue
            for l2 in l1.iterdir():
                if not l2.is_dir():
                    continue
                try:
                    sbk = int(l2.name, 16)
                except ValueError:
                    continue
                idx_path = l2 / "index.bin"
                if not idx_path.exists():
                    continue
                raw = idx_path.read_bytes()
                n   = len(raw) // INDEX_ENTRY_SIZE
                for i in range(n):
                    entry = raw[i * INDEX_ENTRY_SIZE: (i + 1) * INDEX_ENTRY_SIZE]
                    hash_b, _, _ = _INDEX_STRUCT.unpack(entry)
                    self._hash_index[hash_b.hex()] = (bk, sbk, i)
        self._hash_index_loaded = True


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import time

    print("💾 OmniVectorStore — Self Test\n")
    enc = get_omni_encoder()

    with tempfile.TemporaryDirectory() as tmpdir:
        store = OmniVectorStore("test", base_path=Path(tmpdir))

        # Test 1: Simpan 20 entry
        rng = __import__('numpy').random.RandomState(42)
        texts = [f"Konsep matematis nomor {i}: teorema ke-{i}" for i in range(20)]
        vecs  = rng.randn(20, OMNI_DIM).tolist()

        t0 = time.perf_counter()
        addrs = enc.encode_batch(texts, vecs)
        for i, (addr, txt) in enumerate(zip(addrs, texts)):
            store.store(addr, txt, {"source": "test", "domain": "test"})
        t1 = time.perf_counter()

        stats = store.get_stats()
        print(f"✅ Simpan 20 entries: {(t1-t0)*1000:.2f}ms")
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Active buckets: {stats['active_buckets']}")

        # Test 2: Exact lookup (O(1))
        target_hash = addrs[5].content_hash
        t0 = time.perf_counter()
        result = store.exact_lookup(target_hash)
        t1 = time.perf_counter()
        found = result is not None and texts[5][:20] in result.get("text", "")
        print(f"\n✅ Exact lookup O(1): {found} | time={( t1-t0)*1000:.3f}ms")

        # Test 3: Hamming search
        q_vec  = vecs[5]
        q_addr = enc.encode("Query mirip entry 5", q_vec)
        t0 = time.perf_counter()
        results = store.search_by_hamming(q_addr, q_vec, top_k=3, max_hamming=32)
        t1 = time.perf_counter()
        print(f"\n✅ Hamming search top-3: {len(results)} results | time={(t1-t0)*1000:.2f}ms")
        if results:
            print(f"   Top score: {results[0]['score']:.4f}")
            print(f"   Top text: {results[0]['text'][:40]}...")

        # Test 4: BucketStore binary format
        bs = list(store._bucket_cache.values())[0]
        idx_entries = bs.read_index_only()
        vecs_mat    = bs.read_vectors_only()
        print(f"\n✅ BucketStore index: {len(idx_entries)} entries")
        print(f"   Vectors matrix shape: {vecs_mat.shape}")
        print(f"   Index.bin size per entry: {bs.index_path.stat().st_size // max(len(idx_entries),1)} bytes")

        # Test 5: Benchmark ingest 1000
        texts_bench = [f"Benchmark entry {i}" for i in range(1000)]
        vecs_bench  = rng.randn(1000, OMNI_DIM).tolist()
        addrs_bench = enc.encode_batch(texts_bench, vecs_bench)

        t0 = time.perf_counter()
        for addr, txt in zip(addrs_bench, texts_bench):
            store.store(addr, txt, {"source": "bench"})
        t1 = time.perf_counter()
        print(f"\n✅ Benchmark ingest 1000: {(t1-t0)*1000:.1f}ms ({(t1-t0)/1000*1000:.3f}ms/entry)")

        # Test 6: Benchmark search
        t0 = time.perf_counter()
        for i in range(100):
            qv = vecs_bench[i]
            qa = enc.encode(f"q{i}", qv)
            store.search_by_hamming(qa, qv, top_k=5, max_hamming=20)
        t1 = time.perf_counter()
        print(f"✅ Benchmark search 100x: {(t1-t0)*1000:.1f}ms ({(t1-t0)/100*1000:.2f}ms/query)")

    print("\n✅ Semua test selesai!")
