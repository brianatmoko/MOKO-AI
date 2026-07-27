"""
bench.py — Benchmark MOKO Native Acceleration Core vs murni-Python
==================================================================
Mengukur peningkatan kecepatan jalur panas Anchor-RAG (tokenize + retrieval)
antara backend native aktif (C++/Rust) dan implementasi murni-Python.

Jalankan:
    python moko_core/moko_native/bench.py
    MOKO_NATIVE_LIB=.../libmoko_native_rs.so python moko_core/moko_native/bench.py
"""
from __future__ import annotations

import random
import string
import sys
import time
from pathlib import Path

# Pastikan paket 'moko_native' dapat diimpor apa pun cwd-nya.
_HERE = Path(__file__).resolve()
_MOKO_CORE = _HERE.parents[1]
if str(_MOKO_CORE) not in sys.path:
    sys.path.insert(0, str(_MOKO_CORE))

from moko_native import native_accel as na  # noqa: E402


def _rand_word(rng: random.Random, min_len: int = 2, max_len: int = 12) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def build_workload(seed: int = 7):
    rng = random.Random(seed)
    vocab = [_rand_word(rng) for _ in range(2000)]

    # Teks panjang (~konteks besar ala Kimi) untuk uji tokenisasi.
    words = [rng.choice(vocab) for _ in range(200_000)]
    long_text = " ".join(words)

    # Korpus snippet besar untuk uji retrieval.
    anchor_sets = []
    for _ in range(5000):
        k = rng.randint(3, 12)
        anchor_sets.append({rng.choice(vocab) for _ in range(k)})

    # Sekumpulan kueri fokus (himpunan token siap pakai).
    queries = []
    for _ in range(500):
        k = rng.randint(2, 8)
        queries.append({rng.choice(vocab) for _ in range(k)})

    # Sekumpulan dokumen kueri mentah (~konteks panjang) untuk jalur gabungan.
    query_texts = []
    for _ in range(500):
        n = rng.randint(120, 300)
        query_texts.append(" ".join(rng.choice(vocab) for _ in range(n)))

    return long_text, anchor_sets, queries, query_texts


def bench_tokenize(long_text: str, repeats: int = 5):
    # Native (backend aktif)
    t0 = time.perf_counter()
    for _ in range(repeats):
        native_tokens = na.tokenize(long_text)
    t_native = (time.perf_counter() - t0) / repeats

    # Murni-Python
    t0 = time.perf_counter()
    for _ in range(repeats):
        py_tokens = na._py_tokenize(long_text)
    t_py = (time.perf_counter() - t0) / repeats

    assert native_tokens == py_tokens, "PARITAS tokenize GAGAL!"
    return t_py, t_native, len(py_tokens)


def bench_retrieve(anchor_sets, queries, limit: int = 3):
    # Native: bangun indeks sekali, kueri banyak.
    t0 = time.perf_counter()
    idx_native = na.build_index(anchor_sets)
    build_native = time.perf_counter() - t0
    t0 = time.perf_counter()
    res_native = [idx_native.query(q, limit) for q in queries]
    q_native = time.perf_counter() - t0
    idx_native.close()

    # Murni-Python.
    t0 = time.perf_counter()
    idx_py = na._PyAnchorIndex(anchor_sets)
    build_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    res_py = [idx_py.query(q, limit) for q in queries]
    q_py = time.perf_counter() - t0

    assert res_native == res_py, "PARITAS retrieval GAGAL!"
    return (build_py, q_py), (build_native, q_native), len(queries)


def bench_query_text(anchor_sets, query_texts, limit: int = 3):
    """Jalur gabungan end-to-end: dari teks mentah langsung ke top-k snippet.

    Python: set(tokenize(text)) lalu skoring. Native: query_text (satu panggilan).
    """
    # Native: bangun indeks sekali, kueri-teks banyak.
    idx_native = na.build_index(anchor_sets)
    t0 = time.perf_counter()
    res_native = [idx_native.query_text(t, limit) for t in query_texts]
    q_native = time.perf_counter() - t0
    idx_native.close()

    # Murni-Python: tokenize + skoring.
    idx_py = na._PyAnchorIndex(anchor_sets)
    t0 = time.perf_counter()
    res_py = [idx_py.query(na._py_tokenize(t), limit) for t in query_texts]
    q_py = time.perf_counter() - t0

    assert res_native == res_py, "PARITAS query_text GAGAL!"
    return q_py, q_native, len(query_texts)


def main():
    print("=" * 60)
    print("MOKO Native Acceleration Benchmark")
    print(f"Backend aktif : {na.backend_name()}  (native={na.is_native()})")
    print("=" * 60)

    long_text, anchor_sets, queries, query_texts = build_workload()

    t_py, t_native, n_tok = bench_tokenize(long_text)
    speed_tok = (t_py / t_native) if t_native > 0 else float("inf")
    print(f"[tokenize] tokens={n_tok:,}")
    print(f"  python : {t_py * 1e3:8.2f} ms")
    print(f"  native : {t_native * 1e3:8.2f} ms  ->  speedup x{speed_tok:5.2f}")

    (bpy, qpy), (bnat, qnat), nq = bench_retrieve(anchor_sets, queries)
    speed_q = (qpy / qnat) if qnat > 0 else float("inf")
    print(f"[retrieve] snippets={len(anchor_sets):,} queries={nq:,}")
    print(f"  python : build {bpy * 1e3:7.2f} ms | query {qpy * 1e3:8.2f} ms")
    print(f"  native : build {bnat * 1e3:7.2f} ms | query {qnat * 1e3:8.2f} ms  ->  query speedup x{speed_q:5.2f}")

    qtpy, qtnat, nqt = bench_query_text(anchor_sets, query_texts)
    speed_qt = (qtpy / qtnat) if qtnat > 0 else float("inf")
    print(f"[query_text] jalur gabungan tokenize+score, docs={nqt:,}")
    print(f"  python : {qtpy * 1e3:8.2f} ms")
    print(f"  native : {qtnat * 1e3:8.2f} ms  ->  speedup x{speed_qt:5.2f}")
    print("=" * 60)
    print("PARITAS: OK (hasil native == hasil python)")


if __name__ == "__main__":
    main()
