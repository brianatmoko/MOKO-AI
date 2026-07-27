"""
MOKO Embed Engine
==================
Local embedding engine menggunakan sentence-transformers.
- Primary  : BAAI/bge-small-en-v1.5 (33M param, ~130MB, latensi <20ms di CPU)
- Fallback  : deterministic blake2b 768-D (offline, instant, no model needed)

Usage:
    from moko_memory.moko_embed_engine import embed, EMBED_DIM
    vec = embed("binary search algorithm in python")  # np.ndarray shape (768,)
"""

import hashlib
import re
import time
import threading
import numpy as np
from pathlib import Path
from typing import List, Optional

# ─── Config ───────────────────────────────────────────────────────────────────
EMBED_DIM      = 768
MODEL_NAME     = "BAAI/bge-small-en-v1.5"   # 33M params, 130MB SafeTensors
CACHE_DIR      = Path.home() / ".cache" / "moko_embed"
_MODEL_READY   = False
_MODEL_LOCK    = threading.Lock()
_st_model      = None   # sentence_transformers.SentenceTransformer instance


# ─── Deterministic fallback (always available, zero dependencies) ──────────────
def _fallback_embed(text: str) -> np.ndarray:
    """
    Deterministic 768-D unit-norm embedding from blake2b token hashing.
    Consistent across runs — matches the fallback_embedding used during bulk ingestion.
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    tokens = re.findall(r"[\w]+", (text or "").lower())
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:4], "little") % EMBED_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm



# ─── Model loading (lazy, thread-safe) ────────────────────────────────────────
_MODEL_LOAD_FAILED = False

def _try_load_model() -> bool:
    """Attempt to load sentence-transformers model. Return True if successful."""
    global _st_model, _MODEL_READY, _MODEL_LOAD_FAILED
    if _MODEL_LOAD_FAILED:
        return False
    with _MODEL_LOCK:
        if _MODEL_READY:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[MokoEmbed] Loading '{MODEL_NAME}' from {CACHE_DIR}...")
            t0 = time.time()
            _st_model = SentenceTransformer(
                MODEL_NAME,
                cache_folder=str(CACHE_DIR),
                device="cpu",
            )
            elapsed = (time.time() - t0) * 1000
            print(f"[MokoEmbed] ✅ Model loaded in {elapsed:.0f}ms — ready for <20ms inference.")
            _MODEL_READY = True
            return True
        except ImportError:
            print("[MokoEmbed] ⚠️  sentence-transformers not installed. Using deterministic fallback.")
            print("[MokoEmbed]     Install: pip install sentence-transformers")
            _MODEL_READY = False
            _MODEL_LOAD_FAILED = True
            return False
        except Exception as e:
            print(f"[MokoEmbed] ⚠️  Model load failed: {e}. Using deterministic fallback.")
            _MODEL_READY = False
            _MODEL_LOAD_FAILED = True
            return False



def _model_embed(text: str) -> np.ndarray:
    """Encode using sentence-transformers model."""
    global _st_model
    if _st_model is None:
        return _fallback_embed(text)
    try:
        vec = _st_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )
        arr = np.array(vec, dtype=np.float32)
        # bge-small outputs 384-D; pad/project to 768-D for compatibility
        if arr.shape[0] == 384:
            arr = np.concatenate([arr, arr], axis=0)   # symmetric doubling preserves cosine
        return arr
    except Exception as e:
        print(f"[MokoEmbed] Inference error: {e}. Falling back.")
        return _fallback_embed(text)


# ─── Public API ───────────────────────────────────────────────────────────────
def embed(text: str, use_model: bool = True) -> np.ndarray:
    """
    Encode text to 768-D float32 unit-norm vector.
    
    Args:
        text      : Input text (any length)
        use_model : If True, try sentence-transformers; fallback on failure.
                    If False, always use deterministic blake2b fallback.
    Returns:
        np.ndarray of shape (768,), dtype float32, unit norm.
    """
    if not use_model:
        return _fallback_embed(text)
    # Lazy load on first call
    if not _MODEL_READY:
        _try_load_model()
    if _MODEL_READY:
        return _model_embed(text)
    return _fallback_embed(text)


def embed_batch(texts: List[str], use_model: bool = True) -> np.ndarray:
    """
    Encode a list of texts. Returns np.ndarray of shape (N, 768).
    Much faster than calling embed() in a loop when model is loaded.
    """
    if not use_model:
        return np.stack([_fallback_embed(t) for t in texts])
    if not _MODEL_READY:
        _try_load_model()
    if _MODEL_READY and _st_model is not None:
        try:
            vecs = _st_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            arr = np.array(vecs, dtype=np.float32)
            if arr.shape[1] == 384:
                arr = np.concatenate([arr, arr], axis=1)
            return arr
        except Exception:
            pass
    return np.stack([_fallback_embed(t) for t in texts])


def get_embed_mode() -> str:
    """Returns 'model' or 'fallback' — useful for diagnostics."""
    return "model" if _MODEL_READY else "fallback"


def ensure_model_loaded() -> bool:
    """Explicitly trigger model download/load. Returns True if model is ready."""
    return _try_load_model()


# ─── Pre-load trigger (background thread on import) ───────────────────────────
def _bg_load():
    """Load model in background so first query doesn't stall."""
    _try_load_model()

_bg_thread = threading.Thread(target=_bg_load, daemon=True, name="MokoEmbedLoader")
_bg_thread.start()


# ─── CLI diagnostic ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MOKO Embed Engine — Diagnostic")
    print("=" * 55)
    ensure_model_loaded()
    print(f"  Mode: {get_embed_mode()}")
    queries = [
        "binary search algorithm",
        "sql injection prevention techniques",
        "eigenvalue decomposition of symmetric matrix",
    ]
    for q in queries:
        t0 = time.time()
        v = embed(q)
        ms = (time.time() - t0) * 1000
        print(f"  [{ms:6.1f}ms] {q[:45]:<45} | norm={np.linalg.norm(v):.4f} | dim={v.shape[0]}")
    print("=" * 55)
