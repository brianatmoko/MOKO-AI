"""
moko_native — MOKO Native Acceleration Core
===========================================
Paket akselerasi native (C++ / Rust) untuk jalur panas Anchor-RAG pada
Dual-System Moko IDE v5. Menyediakan `tokenize` cepat dan mesin indeks anchor
`build_index`, dengan fallback murni-Python yang identik secara perilaku.

Bangun library native dengan:
    bash moko_core/moko_native/build.sh

Backend dipilih otomatis: Rust → C++ → Python.
"""
from __future__ import annotations

from .native_accel import (
    ANCHOR_SEP,
    BACKEND,
    SNIPPET_SEP,
    backend_name,
    build_index,
    is_native,
    tokenize,
)

__all__ = [
    "ANCHOR_SEP",
    "BACKEND",
    "SNIPPET_SEP",
    "backend_name",
    "build_index",
    "is_native",
    "tokenize",
]
