"""
MOKO Unified KV Cache Manager
==============================
Manajemen KV Cache multi-tier terinspirasi DeepSeek ESS (Extended Sparse Server).

Mengelola empat tier memori sebagai satu abstraksi tunggal:
  Tier 0: VRAM GPU   → KV hot (token aktif sedang diproses)
  Tier 1: CPU RAM    → KV warm (konteks panjang, sering di-refer)
  Tier 2: SSD NVMe   → KV cold (prefill selesai, jarang diakses)
  Tier 3: Archive    → Frozen checkpoints

Prinsip:
  - LRU eviction dengan prediksi akses temporal
  - Promosi otomatis saat sequence diakses kembali (Tier 2→1→0)
  - Thread-safe dengan RLock per tier
  - Kompatibel dengan llama-cpp-python (tidak memerlukan CUDA)

Integrasi:
  - Digunakan oleh mome_engine.py untuk manajemen KV state panjang
  - Digunakan oleh coding_orchestrator.py untuk state snapshot antar-agent
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("moko_kv_cache")

# ── Konstanta Default ─────────────────────────────────────────────────────────
_DEFAULT_VRAM_MB  = 2500   # GPU VRAM: 2.5GB (sisakan 1.5GB untuk model + OS)
_DEFAULT_RAM_MB   = 8192   # CPU RAM:  8GB
_DEFAULT_SSD_DIR  = Path("/tmp/moko_kv_cache")
_EVICT_BATCH_SIZE = 5      # Evict 5 sekaligus saat tier penuh


class MemoryTier(IntEnum):
    VRAM    = 0  # GPU VRAM — Tercepat, terbatas
    RAM     = 1  # CPU RAM  — Cepat, lebih besar
    SSD     = 2  # SSD NVMe — Lambat, sangat besar
    ARCHIVE = 3  # HDD/Net  — Paling lambat, tak terbatas


@dataclass
class KVEntry:
    """Satu entri KV Cache."""
    key:         str
    size_bytes:  int
    tier:        MemoryTier
    last_access: float = field(default_factory=time.monotonic)
    access_count: int  = 0
    ssd_path:    Optional[str] = None
    # Payload in-memory (hanya ada di VRAM/RAM)
    payload:     Optional[Any] = None

    def touch(self) -> None:
        self.last_access  = time.monotonic()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_access


class MokoKVCacheManager:
    """
    KV Cache Multi-Tier Manager.

    Penggunaan tipikal:
        mgr = MokoKVCacheManager()
        mgr.put("seq_001", kv_tensor, size_bytes=kv_tensor.nbytes)
        payload = mgr.get("seq_001")
        mgr.evict("seq_001")
    """

    def __init__(
        self,
        vram_limit_mb: int  = _DEFAULT_VRAM_MB,
        ram_limit_mb:  int  = _DEFAULT_RAM_MB,
        ssd_cache_dir: Path = _DEFAULT_SSD_DIR,
    ):
        self.vram_limit = vram_limit_mb * 1024 * 1024
        self.ram_limit  = ram_limit_mb  * 1024 * 1024
        self.ssd_dir    = Path(ssd_cache_dir)
        self.ssd_dir.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, KVEntry] = {}
        self._vram_used: int = 0
        self._ram_used:  int = 0
        self._lock = threading.RLock()

        self.stats = {
            "vram_hit":   0,
            "ram_hit":    0,
            "ssd_hit":    0,
            "miss":       0,
            "evictions":  0,
            "promotions": 0,
        }
        logger.info(
            f"[KVCache] Inisialisasi — VRAM {vram_limit_mb}MB | "
            f"RAM {ram_limit_mb}MB | SSD {ssd_cache_dir}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def put(self, key: str, payload: Any, size_bytes: int) -> MemoryTier:
        """
        Simpan payload KV baru. Dimulai dari tier tercepat yang tersedia.
        Mengembalikan tier tempat payload disimpan.
        """
        with self._lock:
            # Jika sudah ada, update
            if key in self._registry:
                self._free_entry(key)

            if self._can_fit_vram(size_bytes):
                tier = MemoryTier.VRAM
                self._vram_used += size_bytes
            elif self._can_fit_ram(size_bytes):
                tier = MemoryTier.RAM
                self._ram_used += size_bytes
                payload = self._maybe_serialize(payload)
            else:
                tier = MemoryTier.SSD
                payload = None  # payload disimpan ke disk

            entry = KVEntry(key=key, size_bytes=size_bytes, tier=tier, payload=payload)

            if tier == MemoryTier.SSD:
                ssd_path = self._write_ssd(key, payload if payload else self._maybe_serialize(payload))
                entry.ssd_path = ssd_path

            self._registry[key] = entry
            logger.debug(f"[KVCache] PUT {key} → Tier.{tier.name} ({size_bytes//1024}KB)")
            return tier

    def get(self, key: str) -> Optional[Any]:
        """
        Ambil payload. Otomatis promosikan ke tier lebih cepat jika diakses.
        Mengembalikan None jika tidak ditemukan.
        """
        with self._lock:
            if key not in self._registry:
                self.stats["miss"] += 1
                logger.debug(f"[KVCache] MISS: {key}")
                return None

            entry = self._registry[key]
            entry.touch()

            if entry.tier == MemoryTier.VRAM:
                self.stats["vram_hit"] += 1
                return entry.payload

            if entry.tier == MemoryTier.RAM:
                self.stats["ram_hit"] += 1
                payload = self._maybe_deserialize(entry.payload)
                # Promosikan ke VRAM jika muat
                self._try_promote_to_vram(key, entry, payload)
                return payload

            if entry.tier == MemoryTier.SSD:
                self.stats["ssd_hit"] += 1
                payload = self._read_ssd(entry.ssd_path)
                if payload is not None:
                    # Promosikan ke RAM → VRAM bertahap
                    self._try_promote_from_ssd(key, entry, payload)
                return payload

            self.stats["miss"] += 1
            return None

    def evict(self, key: str) -> bool:
        """Hapus entri dari semua tier."""
        with self._lock:
            if key not in self._registry:
                return False
            self._free_entry(key)
            del self._registry[key]
            self.stats["evictions"] += 1
            return True

    def evict_lru(self, tier: MemoryTier, n: int = _EVICT_BATCH_SIZE) -> int:
        """Evict N entri LRU dari tier tertentu."""
        with self._lock:
            candidates = [
                (entry.last_access, key)
                for key, entry in self._registry.items()
                if entry.tier == tier
            ]
            candidates.sort()  # Paling lama tidak diakses = paling depan

            evicted = 0
            for _, key in candidates[:n]:
                entry = self._registry[key]
                next_tier = MemoryTier(tier.value + 1)
                self._demote_entry(key, entry, next_tier)
                evicted += 1

            logger.debug(f"[KVCache] LRU evict {evicted} dari Tier.{tier.name}")
            return evicted

    def snapshot_state(self, agent_id: str, state: dict) -> str:
        """
        Simpan state snapshot agen ke SSD (terinspirasi Kimi State Snapshot).
        Digunakan oleh CodingOrchestrator untuk recovery jika agent stuck.
        Returns: path snapshot
        """
        snap_path = self.ssd_dir / f"snap_{agent_id}_{int(time.time())}.pkl"
        with open(snap_path, "wb") as f:
            pickle.dump(state, f, protocol=4)
        logger.info(f"[KVCache] Snapshot agent {agent_id} → {snap_path}")
        return str(snap_path)

    def restore_state(self, snap_path: str) -> Optional[dict]:
        """Pulihkan state snapshot dari SSD."""
        path = Path(snap_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"[KVCache] Gagal restore snapshot {snap_path}: {e}")
            return None

    def get_status(self) -> dict:
        """Status ringkas semua tier."""
        with self._lock:
            by_tier = {t: {"count": 0, "size_mb": 0.0} for t in MemoryTier}
            for entry in self._registry.values():
                by_tier[entry.tier]["count"] += 1
                by_tier[entry.tier]["size_mb"] += entry.size_bytes / 1024**2

            return {
                "vram_used_mb":  round(self._vram_used / 1024**2, 1),
                "vram_limit_mb": round(self.vram_limit  / 1024**2, 1),
                "ram_used_mb":   round(self._ram_used   / 1024**2, 1),
                "ram_limit_mb":  round(self.ram_limit    / 1024**2, 1),
                "total_entries": len(self._registry),
                "tiers": {t.name: by_tier[t] for t in MemoryTier},
                "stats": self.stats,
            }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _can_fit_vram(self, size_bytes: int) -> bool:
        if self._vram_used + size_bytes > self.vram_limit:
            freed = self.evict_lru(MemoryTier.VRAM, n=_EVICT_BATCH_SIZE)
            if freed == 0:
                return False
        return self._vram_used + size_bytes <= self.vram_limit

    def _can_fit_ram(self, size_bytes: int) -> bool:
        if self._ram_used + size_bytes > self.ram_limit:
            freed = self.evict_lru(MemoryTier.RAM, n=_EVICT_BATCH_SIZE)
            if freed == 0:
                return False
        return self._ram_used + size_bytes <= self.ram_limit

    def _free_entry(self, key: str) -> None:
        """Bebaskan memori dari entry yang ada."""
        entry = self._registry.get(key)
        if not entry:
            return
        if entry.tier == MemoryTier.VRAM:
            self._vram_used = max(0, self._vram_used - entry.size_bytes)
        elif entry.tier == MemoryTier.RAM:
            self._ram_used  = max(0, self._ram_used  - entry.size_bytes)
        elif entry.tier == MemoryTier.SSD and entry.ssd_path:
            try:
                Path(entry.ssd_path).unlink(missing_ok=True)
            except Exception:
                pass
        entry.payload = None

    def _demote_entry(self, key: str, entry: KVEntry, target: MemoryTier) -> None:
        """Turunkan entry ke tier yang lebih lambat."""
        if target == MemoryTier.SSD:
            payload = entry.payload or b""
            ssd_path = self._write_ssd(key, self._maybe_serialize(payload))
            entry.ssd_path = ssd_path
        if entry.tier == MemoryTier.VRAM:
            self._vram_used = max(0, self._vram_used - entry.size_bytes)
        elif entry.tier == MemoryTier.RAM:
            self._ram_used  = max(0, self._ram_used  - entry.size_bytes)
        entry.tier    = target
        entry.payload = None

    def _try_promote_to_vram(self, key: str, entry: KVEntry, payload: Any) -> None:
        if self._can_fit_vram(entry.size_bytes):
            entry.tier    = MemoryTier.VRAM
            entry.payload = payload
            self._vram_used += entry.size_bytes
            self._ram_used   = max(0, self._ram_used - entry.size_bytes)
            self.stats["promotions"] += 1
            logger.debug(f"[KVCache] PROMOTE {key}: RAM→VRAM")

    def _try_promote_from_ssd(self, key: str, entry: KVEntry, payload: Any) -> None:
        if self._can_fit_ram(entry.size_bytes):
            entry.tier    = MemoryTier.RAM
            entry.payload = self._maybe_serialize(payload)
            self._ram_used += entry.size_bytes
            # Biarkan SSD file sampai evict eksplisit
            self.stats["promotions"] += 1
            logger.debug(f"[KVCache] PROMOTE {key}: SSD→RAM")
        elif self._can_fit_vram(entry.size_bytes):
            entry.tier    = MemoryTier.VRAM
            entry.payload = payload
            self._vram_used += entry.size_bytes
            self.stats["promotions"] += 1
            logger.debug(f"[KVCache] PROMOTE {key}: SSD→VRAM")

    def _write_ssd(self, key: str, data: bytes) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        path = self.ssd_dir / f"kv_{h}.pkl"
        with open(path, "wb") as f:
            f.write(data if isinstance(data, bytes) else pickle.dumps(data, protocol=4))
        return str(path)

    def _read_ssd(self, path: Optional[str]) -> Optional[Any]:
        if not path or not Path(path).exists():
            return None
        try:
            with open(path, "rb") as f:
                data = f.read()
            return self._maybe_deserialize(data)
        except Exception as e:
            logger.error(f"[KVCache] Gagal baca SSD {path}: {e}")
            return None

    @staticmethod
    def _maybe_serialize(obj: Any) -> bytes:
        if isinstance(obj, bytes):
            return obj
        try:
            return pickle.dumps(obj, protocol=4)
        except Exception:
            return b""

    @staticmethod
    def _maybe_deserialize(data: Any) -> Any:
        if isinstance(data, bytes) and data:
            try:
                return pickle.loads(data)
            except Exception:
                return data
        return data


# ── Singleton ─────────────────────────────────────────────────────────────────
_kv_cache_instance: Optional[MokoKVCacheManager] = None


def get_kv_cache() -> MokoKVCacheManager:
    """Dapatkan singleton KV Cache Manager."""
    global _kv_cache_instance
    if _kv_cache_instance is None:
        _kv_cache_instance = MokoKVCacheManager()
    return _kv_cache_instance
