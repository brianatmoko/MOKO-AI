"""
test_native_accel — Paritas & sanity MOKO Native Acceleration Core
==================================================================
Memverifikasi bahwa inti native (C++/Rust) menghasilkan output yang IDENTIK
dengan implementasi murni-Python untuk seluruh jalur panas Anchor-RAG:
  1. tokenize                → sama dengan regex `[a-zA-Z_]{2,}` pada teks.lower().
  2. index.query(focus)      → sama dengan skoring/ranking referensi.
  3. index.query_text(text)  → sama dengan tokenize+score referensi.
  4. _bridge.retrieve(kb,..) → sama dengan CodeKnowledgeBase.retrieve (integrasi).

Tes ini lolos baik saat backend native tersedia MAUPUN saat fallback murni-Python
(paritas dijaga di kedua kasus). Bila library native ADA di folder paket, tes
juga memastikan library tersebut benar-benar termuat.

Jalankan:  python moko_core/test_native_accel.py
"""
from __future__ import annotations

import random
import re
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve()
_MOKO_CORE = _HERE.parent
if str(_MOKO_CORE) not in sys.path:
    sys.path.insert(0, str(_MOKO_CORE))

from moko_native import native_accel as na  # noqa: E402
from moko_agents.dual_system import _bridge  # noqa: E402

_REF_RE = re.compile(r"[a-zA-Z_]{2,}")


def ref_tokenize(text: str):
    return _REF_RE.findall((text or "").lower())


def ref_retrieve(anchor_sets, focus_tokens, limit):
    """Referensi independen: score=|anchors∩focus|; urut (score,-index) desc."""
    focus = set(focus_tokens)
    scored = []
    for index, anchors in enumerate(anchor_sets):
        score = len(set(anchors) & focus)
        if score >= 1:
            scored.append((score, -index, index))
    scored.sort(key=lambda it: (it[0], it[1]), reverse=True)
    return [(idx, score) for score, _neg, idx in scored[:limit]]


# ── Test cases ────────────────────────────────────────────────────────────────
def test_backend_report():
    print(f"[info] backend aktif = {na.backend_name()} | native = {na.is_native()}")
    # Bila salah satu library native ADA di folder paket, ia HARUS termuat.
    pkg_dir = Path(na.__file__).resolve().parent
    has_lib = any(
        (pkg_dir / n).exists()
        for n in ("libmoko_native_rs.so", "libmoko_native.so",
                  "libmoko_native_rs.dylib", "libmoko_native.dylib")
    )
    if has_lib:
        assert na.is_native(), "Library native ada tetapi gagal dimuat!"
        assert na.backend_name() in {"rust", "cpp"}


def test_tokenize_parity():
    samples = [
        "",
        "a",  # 1 char → tidak masuk
        "ab",  # tepat 2 char
        "Halo Dunia, ini TEST_123 kode!!",
        "buatkan fungsi hitung_luas_persegi(panjang, lebar)",
        "SIN cos TAN derajat radian; trigonometri.",
        "under_score dan CamelCase serta MIXED_case_99",
        "tanda baca .,;:!?()[]{} tidak dihitung",
        "angka123 456 hanya huruf_dan_underscore",
        "unicode: café naïve — 日本語 test_ok",
        "   spasi    banyak   \n\t tab dan newline  ",
        "x" * 5000 + " end_marker_token",
    ]
    for s in samples:
        assert na.tokenize(s) == ref_tokenize(s), f"tokenize berbeda utk: {s!r}"


def _random_anchor_sets(rng, n_snip=200, vocab=None):
    vocab = vocab or [f"tok{i}" for i in range(300)]
    sets = []
    for _ in range(n_snip):
        k = rng.randint(1, 10)
        sets.append({rng.choice(vocab) for _ in range(k)})
    return sets, vocab


def test_index_query_parity():
    rng = random.Random(123)
    anchor_sets, vocab = _random_anchor_sets(rng)
    index = na.build_index(anchor_sets)
    try:
        for _ in range(400):
            k = rng.randint(0, 6)
            focus = {rng.choice(vocab) for _ in range(k)} if k else set()
            for limit in (1, 3, 5, 10):
                got = index.query(focus, limit)
                exp = ref_retrieve(anchor_sets, focus, limit)
                assert got == exp, f"query beda (focus={focus}, limit={limit})"
    finally:
        index.close()


def test_index_query_text_parity():
    rng = random.Random(321)
    vocab = ["luas", "persegi", "panjang", "sin", "cos", "bunga", "median", "prima"]
    anchor_sets, _ = _random_anchor_sets(rng, n_snip=120, vocab=vocab)
    index = na.build_index(anchor_sets)
    try:
        for _ in range(300):
            n = rng.randint(0, 40)
            words = [rng.choice(vocab + ["noise", "xx", "zz"]) for _ in range(n)]
            text = " ".join(words) + " punctuation!!! CAPS Mixed_Case"
            for limit in (1, 3, 5):
                got = index.query_text(text, limit)
                exp = ref_retrieve(anchor_sets, ref_tokenize(text), limit)
                assert got == exp, f"query_text beda (text={text!r}, limit={limit})"
    finally:
        index.close()


def test_bridge_retrieve_matches_kb():
    """Integrasi: _bridge.retrieve harus identik dgn CodeKnowledgeBase.retrieve."""
    kb = _bridge.CodeKnowledgeBase()
    anchor_sets = [set(getattr(s, "anchors", ()) or ()) for s in kb.snippets]
    all_anchors = sorted({a for s in anchor_sets for a in s})
    rng = random.Random(7)
    for _ in range(300):
        k = rng.randint(0, 5)
        focus = {rng.choice(all_anchors) for _ in range(k)} if k else set()
        focus |= {f"noise{rng.randint(0, 5)}"}
        for limit in (1, 2, 3, 6):
            got = _bridge.retrieve(kb, focus, limit=limit)
            exp = list(kb.retrieve(set(focus), limit=limit))
            assert [id(x) for x in got] == [id(x) for x in exp], (
                f"_bridge.retrieve beda dari kb.retrieve (focus={focus}, limit={limit})"
            )


def test_bridge_retrieve_text_matches_reference():
    kb = _bridge.CodeKnowledgeBase()
    prompts = [
        "buatkan fungsi hitung luas persegi panjang",
        "hitung sin cos derajat trigonometri",
        "statistik median rata rata dan deviasi",
        "bunga majemuk cicilan pinjaman",
        "konversi suhu celsius ke fahrenheit",
        "algoritma faktorial fibonacci bilangan prima",
        "tidak ada anchor yang cocok sama sekali xyz",
    ]
    for p in prompts:
        got = [getattr(s, "snippet_id", None) for s in _bridge.retrieve_text(kb, p, limit=3)]
        exp = [getattr(s, "snippet_id", None)
               for s in kb.retrieve(set(_bridge.tokenize(p)), limit=3)]
        assert got == exp, f"retrieve_text beda utk prompt: {p!r}\n got={got}\n exp={exp}"


def main():
    tests = [
        test_backend_report,
        test_tokenize_parity,
        test_index_query_parity,
        test_index_query_text_parity,
        test_bridge_retrieve_matches_kb,
        test_bridge_retrieve_text_matches_reference,
    ]
    passed = 0
    print("=" * 60)
    print("MOKO Native Acceleration — Paritas & Integrasi")
    print("=" * 60)
    for t in tests:
        name = t.__name__
        try:
            t0 = time.perf_counter()
            t()
            dt = (time.perf_counter() - t0) * 1e3
            print(f"[PASS] {name}  ({dt:.1f} ms)")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()
    print("-" * 60)
    print(f"HASIL: {passed}/{len(tests)} tes lolos "
          f"(backend={na.backend_name()}, native={na.is_native()})")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
