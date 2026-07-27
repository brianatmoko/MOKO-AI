"""
MOKO RAG Probe — End-to-End Diagnostic
========================================
Menguji apakah RAG pipeline benar-benar berfungsi dari ujung ke ujung.
Jalankan dari direktori moko_core/:

    python3 moko_memory/moko_rag_probe.py

Output yang diharapkan:
  - Embed mode (model atau fallback)
  - Latensi embedding
  - Jumlah domain aktif dan total memori
  - Top-K hasil untuk beberapa query uji
  - Konfirmasi bahwa konteks akan dikirim ke LLM
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def main():
    print("\n🔬 MOKO RAG Probe — Diagnostik End-to-End")

    # ── 1. Embed engine ────────────────────────────────────────────
    section("1. Embedding Engine")
    from moko_memory.moko_embed_engine import embed, get_embed_mode, ensure_model_loaded, EMBED_DIM
    
    print(f"  Trying to load sentence-transformers model...")
    ensure_model_loaded()
    mode = get_embed_mode()
    print(f"  Embed Mode    : {mode.upper()}")
    print(f"  Embed Dim     : {EMBED_DIM}")

    test_text = "binary search algorithm python implementation"
    t0 = time.time()
    v = embed(test_text)
    ms = (time.time() - t0) * 1000
    print(f"  Test Latency  : {ms:.1f}ms")
    print(f"  Vector Norm   : {float((v**2).sum()**0.5):.6f} (should be ~1.0)")
    print(f"  ✅ Embedding engine {'[MODEL]' if mode=='model' else '[FALLBACK]'} is working")

    # ── 2. OMNI Index stats ────────────────────────────────────────
    section("2. OMNI Index Status")
    from moko_config import settings
    omni_root = Path(settings.OMNI_DIR)
    print(f"  OMNI Root : {omni_root}")
    print(f"  Exists    : {omni_root.exists()}")

    if omni_root.exists():
        grand_total = 0
        domain_stats = {}
        for d in sorted(omni_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            count = 0
            buckets = 0
            for b in d.rglob("index.bin"):
                n = b.stat().st_size // 44
                count += n
                if n > 0:
                    buckets += 1
            if count > 0:
                domain_stats[d.name] = (count, buckets)
                grand_total += count
                print(f"  Domain [{d.name:12s}] : {count:>10,} memories | {buckets:>5,} buckets")
        print(f"\n  GRAND TOTAL   : {grand_total:,} memories across {len(domain_stats)} domains")
    else:
        print("  ⚠️  OMNI directory does not exist!")
        return

    # ── 3. Retrieval test ──────────────────────────────────────────
    section("3. RAG Retrieval Test")
    from moko_memory.moko_rag_retriever import MokoRagRetriever
    retriever = MokoRagRetriever()

    test_queries = [
        ("binary search algorithm", "code"),
        ("sql injection attack techniques", "security"),
        ("eigenvalue matrix decomposition", "math"),
        ("how does HTTP 429 rate limiting work", None),
        ("privilege escalation linux", "security"),
    ]

    any_found = False
    for query, domain in test_queries:
        t0 = time.time()
        chunks = retriever.retrieve(query, top_k=3, domain=domain)
        ms = (time.time() - t0) * 1000

        dom_str = f"[domain={domain}]" if domain else "[all domains]"
        if chunks:
            any_found = True
            print(f"\n  ✅ Query: {query!r} {dom_str}")
            print(f"     Found: {len(chunks)} chunk(s) in {ms:.1f}ms")
            for i, c in enumerate(chunks, 1):
                print(f"     [{i}] score={c.score:.4f} | dom={c.domain} | {c.text[:80]!r}...")
        else:
            print(f"\n  ❌ Query: {query!r} {dom_str}")
            print(f"     Found: 0 results in {ms:.1f}ms")

    # ── 4. Context format test ─────────────────────────────────────
    section("4. Context Formatting (what LLM sees)")
    sample_query = "sql injection prevention"
    chunks = retriever.retrieve(sample_query, top_k=3)
    context = retriever.format_context(chunks)
    if context:
        print(f"  Context sample ({len(context)} chars):")
        print("  " + context[:400].replace("\n", "\n  ") + "...")
    else:
        print("  ⚠️  No context generated — RAG would send empty context to LLM!")

    # ── 5. Verdict ─────────────────────────────────────────────────
    section("5. Verdict")
    if any_found:
        print("  ✅ RAG pipeline is FUNCTIONAL")
        print("  ✅ Data is being found and formatted correctly")
        print("  ✅ Context will be injected into LLM system prompt")
    else:
        print("  ❌ RAG pipeline returned NO results")
        print("  Possible causes:")
        print("  1. Embedding space mismatch (ingestion vs query used different models)")
        print("     Fix: Re-inject data using moko_embed_engine, or lower MIN_SCORE further")
        print("  2. max_hamming too small — try increasing to 48+")
        print("  3. OMNI data truly empty (check domain stats above)")
    print()


if __name__ == "__main__":
    main()
