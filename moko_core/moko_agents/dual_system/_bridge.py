"""
Bridge Loader — Integrasi Komponen Referensi Docs (Anchor-RAG & Runtime Guard)
==============================================================================
Modul jembatan yang memuat komponen inti nyata dari folder `docs/`:
- `moko_code_knowledge.py`  → Anchor-based RAG (Sistem 1 / Kimi style)
- `moko_llm_runtime_guard.py` → LLMRuntimeGuard (Sistem 2 Guard / DeepSeek style)

Pemuatan dilakukan secara defensif menggunakan importlib. Jika berkas docs
tidak dapat ditemukan (mis. saat dijalankan di lingkungan terisolasi), modul ini
menyediakan implementasi fallback minimal agar paket `dual_system` tetap dapat
diuji dan dijalankan secara mandiri (self-contained).
"""
from __future__ import annotations

import importlib.util
import re
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Pastikan paket 'moko_native' (di moko_core/) dapat diimpor apa pun cwd-nya.
_MOKO_CORE_DIR = str(Path(__file__).resolve().parents[2])
if _MOKO_CORE_DIR not in sys.path:
    sys.path.insert(0, _MOKO_CORE_DIR)


def _docs_dir() -> Path:
    """Lokasi folder docs relatif terhadap root repositori."""
    # dual_system/_bridge.py → parents[3] == root repositori (MOKO_OS_Project)
    return Path(__file__).resolve().parents[3] / "docs"


def _load_module(filename: str, modname: str):
    """Muat modul python dari path berkas secara aman. Kembalikan None jika gagal."""
    try:
        path = _docs_dir() / filename
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(modname, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Daftarkan ke sys.modules SEBELUM exec agar resolusi anotasi dataclass
        # (dengan `from __future__ import annotations`) menemukan modulnya.
        sys.modules[modname] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(modname, None)
            raise
        return module
    except Exception:
        return None


# ── Muat komponen docs ────────────────────────────────────────────────────────
_ck_module = _load_module("moko_code_knowledge.py", "moko_dual_code_knowledge")
_rg_module = _load_module("moko_llm_runtime_guard.py", "moko_dual_runtime_guard")


# ── Fallback minimal untuk moko_code_knowledge (Anchor-RAG) ────────────────────
def _fallback_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]{2,}", text.lower())


@dataclass(frozen=True)
class _FallbackSnippet:
    snippet_id: str
    domain: str
    summary: str
    anchors: frozenset
    code: str
    requires_imports: tuple = ()
    source: str = "fallback"

    def score(self, focus_tokens: set) -> int:
        return len(self.anchors & focus_tokens)


@dataclass
class _FallbackKnowledgeBase:
    snippets: tuple = field(default_factory=tuple)

    def retrieve(self, focus_tokens: set, *, limit: int = 3) -> list:
        scored = []
        for index, snippet in enumerate(self.snippets):
            score = snippet.score(focus_tokens)
            if score >= 1:
                scored.append((score, -index, snippet))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [snippet for _s, _o, snippet in scored[:limit]]

    def domains(self) -> list:
        return sorted({s.domain for s in self.snippets})


if _ck_module is not None:
    tokenize: Callable[[str], list] = _ck_module.tokenize
    KnowledgeSnippet = _ck_module.KnowledgeSnippet
    CodeKnowledgeBase = _ck_module.CodeKnowledgeBase
    DEFAULT_SNIPPETS = _ck_module.DEFAULT_SNIPPETS
    CODE_KNOWLEDGE_AVAILABLE = True
else:
    tokenize = _fallback_tokenize
    KnowledgeSnippet = _FallbackSnippet
    CodeKnowledgeBase = _FallbackKnowledgeBase
    DEFAULT_SNIPPETS = (
        _FallbackSnippet(
            snippet_id="trigonometri.derajat",
            domain="trigonometri",
            summary="Fungsi trigonometri berbasis derajat",
            anchors=frozenset({"trigonometri", "sin", "cos", "tan", "sudut", "derajat"}),
            requires_imports=("import math",),
            code=(
                "def hitung_cos_derajat(derajat: float) -> float:\n"
                "    return math.cos(math.radians(derajat))"
            ),
        ),
    )
    CODE_KNOWLEDGE_AVAILABLE = False


# ── Fallback minimal untuk moko_llm_runtime_guard (LLMRuntimeGuard) ─────────────
@dataclass(frozen=True)
class _FallbackGuardedGeneration:
    content: str
    source: str
    server_status: str
    message: str
    used_fallback_reason: str | None


class _FallbackLLMRuntimeGuard:
    def __init__(self, status_provider, llm_generate, fallback_generate) -> None:
        self.status_provider = status_provider
        self.llm_generate = llm_generate
        self.fallback_generate = fallback_generate

    def wait_until_ready(self, *, timeout_seconds: int = 90, poll_seconds: float = 2.0) -> str:
        payload = self.status_provider() if self.status_provider else None
        if payload and (payload.get("ready") or payload.get("status") in {"online", "ok"}):
            return "online"
        return "offline"

    def generate(self, prompt: str) -> _FallbackGuardedGeneration:
        status = self.wait_until_ready()
        if status != "online":
            return _FallbackGuardedGeneration(
                content=self.fallback_generate(prompt),
                source="template",
                server_status=status,
                message="MOKO SERVER OFFLINE — memakai fallback template.",
                used_fallback_reason="server_not_ready",
            )
        generated = self.llm_generate(prompt)
        if not generated or not generated.strip():
            return _FallbackGuardedGeneration(
                content=self.fallback_generate(prompt),
                source="template",
                server_status="online",
                message="LLM kosong. Memakai fallback template.",
                used_fallback_reason="empty_llm_output",
            )
        return _FallbackGuardedGeneration(
            content=generated,
            source="llm",
            server_status="online",
            message="LLM generation sukses.",
            used_fallback_reason=None,
        )


def _fallback_normalize_server_status(payload: dict | None) -> str:
    if not payload:
        return "offline"
    if isinstance(payload.get("ready"), bool):
        return "online" if payload["ready"] else "offline"
    value = str(payload.get("status", "")).strip().lower()
    if value in {"online", "ready", "ok", "running", "healthy", "up"}:
        return "online"
    if value in {"loading", "starting", "booting", "warming", "initializing"}:
        return "loading"
    return "offline"


if _rg_module is not None:
    LLMRuntimeGuard = _rg_module.LLMRuntimeGuard
    GuardedGeneration = _rg_module.GuardedGeneration
    normalize_server_status = _rg_module.normalize_server_status
    RUNTIME_GUARD_AVAILABLE = True
else:
    LLMRuntimeGuard = _FallbackLLMRuntimeGuard
    GuardedGeneration = _FallbackGuardedGeneration
    normalize_server_status = _fallback_normalize_server_status
    RUNTIME_GUARD_AVAILABLE = False


# ── Akselerasi native (C++/Rust) untuk jalur panas Anchor-RAG ──────────────────
# Memakai inti native bila library terkompilasi tersedia; jika tidak, seluruh
# jalur otomatis kembali ke murni-Python DENGAN HASIL YANG IDENTIK (paritas).
try:
    from moko_native import native_accel as _na  # type: ignore
    NATIVE_ACCEL_AVAILABLE = bool(_na.is_native())
    NATIVE_BACKEND = _na.backend_name()
except Exception:
    _na = None
    NATIVE_ACCEL_AVAILABLE = False
    NATIVE_BACKEND = "python"


# Gunakan tokenizer native bila tersedia (paritas identik dgn regex Python).
if _na is not None and NATIVE_ACCEL_AVAILABLE:
    _python_tokenize = tokenize

    def tokenize(text):  # type: ignore[no-redef]
        try:
            return _na.tokenize(text)
        except Exception:
            return _python_tokenize(text)


# Cache indeks anchor native per-knowledge-base (bangun sekali, kueri banyak).
_NATIVE_INDEX_CACHE: dict = {}


def _kb_native_index(kb):
    """Kembalikan mesin indeks native ter-cache untuk `kb`, atau None."""
    if _na is None or not NATIVE_ACCEL_AVAILABLE:
        return None
    snippets = getattr(kb, "snippets", ()) or ()
    key = id(kb)
    entry = _NATIVE_INDEX_CACHE.get(key)
    if entry is not None and entry[0] == len(snippets):
        return entry[1]
    anchor_sets = [set(getattr(s, "anchors", ()) or ()) for s in snippets]
    try:
        index = _na.build_index(anchor_sets)
    except Exception:
        return None
    # Bebaskan entri cache saat kb dikoleksi GC (mencegah id reuse & kebocoran).
    try:
        weakref.finalize(kb, _NATIVE_INDEX_CACHE.pop, key, None)
    except TypeError:
        pass
    _NATIVE_INDEX_CACHE[key] = (len(snippets), index)
    return index


def retrieve(kb, focus_tokens, *, limit: int = 3):
    """Anchor-based retrieval berakselerasi native.

    PARITAS: mengembalikan snippet dalam urutan identik dengan
    `CodeKnowledgeBase.retrieve` (skor menurun, seri -> index terkecil dulu).
    Jika native tidak tersedia/gagal, otomatis memakai `kb.retrieve`.
    """
    focus = set(focus_tokens)
    index = _kb_native_index(kb)
    if index is not None:
        try:
            snippets = kb.snippets
            return [snippets[i] for i in index.retrieve_indices(focus, limit)]
        except Exception:
            pass
    try:
        return list(kb.retrieve(focus, limit=limit))
    except Exception:
        return []


def retrieve_text(kb, text: str, *, limit: int = 3):
    """Retrieval gabungan dari teks mentah (tokenisasi + skoring native sekali jalan)."""
    index = _kb_native_index(kb)
    if index is not None:
        try:
            snippets = kb.snippets
            return [snippets[i] for i in index.retrieve_indices_text(text, limit)]
        except Exception:
            pass
    return retrieve(kb, tokenize(text), limit=limit)


__all__ = [
    "tokenize",
    "KnowledgeSnippet",
    "CodeKnowledgeBase",
    "DEFAULT_SNIPPETS",
    "CODE_KNOWLEDGE_AVAILABLE",
    "LLMRuntimeGuard",
    "GuardedGeneration",
    "normalize_server_status",
    "RUNTIME_GUARD_AVAILABLE",
    "retrieve",
    "retrieve_text",
    "NATIVE_ACCEL_AVAILABLE",
    "NATIVE_BACKEND",
]
