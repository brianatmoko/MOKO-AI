"""
native_accel — Loader & Binding ctypes untuk MOKO Native Acceleration Core
==========================================================================
Menjembatani Python dengan inti native (C++ atau Rust) jalur panas Anchor-RAG:
- `tokenize(text)`          → tokenisasi cepat (paritas `[a-zA-Z_]{2,}`).
- `build_index(anchors)`    → mesin retrieval anchor (bangun sekali, kueri banyak).

Pemilihan backend otomatis (bisa dipaksa lewat env `MOKO_NATIVE_LIB`):
    Rust (libmoko_native_rs.so)  →  C++ (libmoko_native.so)  →  murni-Python.

Semua jalur menjaga PARITAS PERILAKU yang identik dengan implementasi Python
di `docs/moko_code_knowledge.py`, sehingga hasil tidak pernah berubah — hanya
kecepatannya yang meningkat. Jika library native gagal dimuat, modul ini tetap
berfungsi penuh memakai fallback murni-Python (tidak ada hard-dependency).
"""
from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Pemisah serialisasi — HARUS identik dengan tier C++ (moko_native.cpp) & Rust.
SNIPPET_SEP = "\x1e"  # Record Separator antar-snippet
ANCHOR_SEP = "\n"     # pemisah antar-anchor / antar-token

_THIS_DIR = Path(__file__).resolve().parent
_TOKEN_RE = re.compile(r"[a-zA-Z_]{2,}")


# ── Fallback murni-Python (selalu tersedia) ────────────────────────────────────
def _py_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _candidate_libs() -> List[Path]:
    """Urutan preferensi: env override → Rust → C++."""
    cands: List[Path] = []
    env = os.environ.get("MOKO_NATIVE_LIB")
    if env:
        cands.append(Path(env))
    # Rust lebih "kuat" → diprioritaskan bila tersedia.
    for name in ("libmoko_native_rs.so", "libmoko_native_rs.dylib", "moko_native_rs.dll"):
        cands.append(_THIS_DIR / name)
    for name in ("libmoko_native.so", "libmoko_native.dylib", "moko_native.dll"):
        cands.append(_THIS_DIR / name)
    return cands


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    lib.moko_native_backend.restype = ctypes.c_char_p
    lib.moko_native_backend.argtypes = []

    lib.moko_native_abi_version.restype = ctypes.c_int
    lib.moko_native_abi_version.argtypes = []

    lib.moko_tokenize.restype = ctypes.c_int
    lib.moko_tokenize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]

    lib.moko_index_build.restype = ctypes.c_void_p
    lib.moko_index_build.argtypes = [ctypes.c_char_p, ctypes.c_int]

    lib.moko_index_query.restype = ctypes.c_int
    lib.moko_index_query.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]

    lib.moko_index_query_text.restype = ctypes.c_int
    lib.moko_index_query_text.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]

    lib.moko_index_free.restype = None
    lib.moko_index_free.argtypes = [ctypes.c_void_p]


def _load_native() -> Tuple[Optional[ctypes.CDLL], str]:
    for path in _candidate_libs():
        try:
            if not path.exists():
                continue
            lib = ctypes.CDLL(str(path))
            _configure_prototypes(lib)
            # Validasi ABI + ambil nama backend (kontrak fungsi saat ini = v2).
            if int(lib.moko_native_abi_version()) != 2:
                continue
            backend = lib.moko_native_backend()
            name = backend.decode("utf-8") if backend else "native"
            return lib, name
        except Exception:
            continue
    return None, "python"


_LIB, BACKEND = _load_native()


def is_native() -> bool:
    """True bila backend native (C++/Rust) berhasil dimuat."""
    return _LIB is not None


def backend_name() -> str:
    """Nama backend aktif: 'rust', 'cpp', atau 'python'."""
    return BACKEND


# ── API publik: tokenize ────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """Tokenisasi cepat setara `re.findall(r"[a-zA-Z_]{2,}", text.lower())`.

    Memakai inti native bila tersedia; jika tidak, fallback murni-Python. Hasil
    keduanya dijamin identik (diverifikasi oleh test paritas).
    """
    if _LIB is None:
        return _py_tokenize(text)
    if not text:
        return []
    try:
        raw = text.encode("utf-8")
        # Kapasitas aman: panjang output <= panjang input + jumlah pemisah.
        cap = len(raw) * 2 + 16
        buf = ctypes.create_string_buffer(cap)
        n = _LIB.moko_tokenize(raw, buf, cap)
        if n < 0:
            return _py_tokenize(text)
        out = buf.value.decode("utf-8", errors="replace")
        return out.split(ANCHOR_SEP) if out else []
    except Exception:
        return _py_tokenize(text)


# ── API publik: mesin indeks anchor ──────────────────────────────────────────────
def _serialize_corpus(anchor_sets: Sequence[Iterable[str]]) -> bytes:
    parts = [ANCHOR_SEP.join(sorted(set(a))) for a in anchor_sets]
    return SNIPPET_SEP.join(parts).encode("utf-8")


class _BaseIndex:
    """Antarmuka umum agar pemanggil tak peduli backend."""

    def query(self, focus_tokens: Iterable[str], limit: int = 3) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def query_text(self, text: str, limit: int = 3) -> List[Tuple[int, int]]:
        """Tokenisasi teks mentah + skoring dalam satu langkah (default: Python)."""
        return self.query(_py_tokenize(text), limit)

    def retrieve_indices(self, focus_tokens: Iterable[str], limit: int = 3) -> List[int]:
        return [idx for idx, _score in self.query(focus_tokens, limit)]

    def retrieve_indices_text(self, text: str, limit: int = 3) -> List[int]:
        return [idx for idx, _score in self.query_text(text, limit)]

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class _PyAnchorIndex(_BaseIndex):
    """Fallback murni-Python — paritas dengan CodeKnowledgeBase.retrieve."""

    def __init__(self, anchor_sets: Sequence[Iterable[str]]):
        self._anchors = [frozenset(a) for a in anchor_sets]

    def query(self, focus_tokens: Iterable[str], limit: int = 3) -> List[Tuple[int, int]]:
        focus = set(focus_tokens)
        scored: List[Tuple[int, int, int]] = []
        for index, anchors in enumerate(self._anchors):
            score = len(anchors & focus)
            if score >= 1:
                scored.append((score, -index, index))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(idx, score) for score, _neg, idx in scored[: max(0, limit)]]


class _NativeAnchorIndex(_BaseIndex):
    """Mesin retrieval anchor native (bangun sekali, kueri berkali-kali)."""

    def __init__(self, anchor_sets: Sequence[Iterable[str]]):
        corpus = _serialize_corpus(anchor_sets)
        self._n = len(anchor_sets)
        self._handle = _LIB.moko_index_build(corpus, len(corpus))
        if not self._handle:
            raise RuntimeError("moko_index_build mengembalikan handle kosong")

    def query(self, focus_tokens: Iterable[str], limit: int = 3) -> List[Tuple[int, int]]:
        if self._handle is None or limit <= 0:
            return []
        focus = ANCHOR_SEP.join(sorted(set(focus_tokens))).encode("utf-8")
        cap = int(limit)
        out_idx = (ctypes.c_int * cap)()
        out_score = (ctypes.c_int * cap)()
        n = _LIB.moko_index_query(self._handle, focus, len(focus), cap, out_idx, out_score)
        return [(int(out_idx[i]), int(out_score[i])) for i in range(int(n))]

    def query_text(self, text: str, limit: int = 3) -> List[Tuple[int, int]]:
        """Jalur gabungan native: tokenisasi + skoring dalam satu panggilan FFI."""
        if self._handle is None or limit <= 0:
            return []
        try:
            raw = (text or "").encode("utf-8")
            cap = int(limit)
            out_idx = (ctypes.c_int * cap)()
            out_score = (ctypes.c_int * cap)()
            n = _LIB.moko_index_query_text(self._handle, raw, cap, out_idx, out_score)
            return [(int(out_idx[i]), int(out_score[i])) for i in range(int(n))]
        except Exception:
            return self.query(_py_tokenize(text), limit)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            try:
                _LIB.moko_index_free(self._handle)
            finally:
                self._handle = None

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


def build_index(anchor_sets: Sequence[Iterable[str]]) -> _BaseIndex:
    """Bangun mesin indeks anchor.

    Mengembalikan mesin native bila tersedia; jika tidak (atau gagal), otomatis
    memakai mesin murni-Python dengan perilaku identik.
    """
    if _LIB is not None:
        try:
            return _NativeAnchorIndex(anchor_sets)
        except Exception:
            pass
    return _PyAnchorIndex(anchor_sets)


__all__ = [
    "SNIPPET_SEP",
    "ANCHOR_SEP",
    "BACKEND",
    "is_native",
    "backend_name",
    "tokenize",
    "build_index",
]
