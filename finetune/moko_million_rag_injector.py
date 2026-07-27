import os
import sys
import time
import json
import zlib
import struct
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "moko_core"))

from moko_config import settings
from moko_memory.omni_hash_encoder import get_omni_encoder

# Binary structs used in omni_vector_store
INDEX_ENTRY_SIZE   = 44
HASH_BYTES         = 32
_INDEX_STRUCT = struct.Struct(">32sQI")   # 32s=hash, Q=bits, I=offset
_BLOCK_STRUCT = struct.Struct(">I")

def fallback_embedding(text: str) -> list:
    """Fast deterministic 768-D embedding generator."""
    import hashlib
    import math
    import re

    vec = [0.0] * 768
    tokens = re.findall(r"[\w]+", (text or "").lower())
    if not tokens:
        return vec

    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        idx = int.from_bytes(digest[:4], "little") % 768
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def run_bulk_injection():
    print("=" * 70)
    print("🚀 MOKO OMNI RAG Bulk Injector — 1,000,000 High-Scale Ingestion 🚀")
    print("=" * 70)

    omni_root = Path(settings.OMNI_DIR)
    print(f"OMNI Directory: {omni_root}")
    omni_root.mkdir(parents=True, exist_ok=True)

    encoder = get_omni_encoder()

    # 1. Prepare templates for combinatorial generation
    languages = [
        "python", "cpp", "javascript", "rust", "go", "java", "sql", "bash", 
        "typescript", "csharp", "ruby", "php", "swift", "kotlin", "scala", 
        "haskell", "perl", "lua", "r", "julia"
    ]
    tasks = [
        "binary_search", "quicksort", "dijkstra", "astar", "event_loop", "socket_listener",
        "http_router", "hash_map", "red_black_tree", "matrix_transpose", "b_tree", "trie",
        "k_means", "linear_regression", "gradient_descent", "backpropagation", "fast_fourier",
        "runge_kutta", "rsa_encryption", "aes_cipher", "sha256_hasher", "docker_builder", 
        "git_merger", "markdown_parser", "html_lexer", "json_serializer", "yaml_deserializer", 
        "sqlite_wrapper", "tcp_handshake", "fibonacci", "factorial", "prime_check", "regex_parser",
        "circular_buffer", "thread_pool", "garbage_collector", "lexer", "parser", "codegen", "interpreter"
    ]
    paradigms = ["procedural", "object_oriented", "functional", "declarative", "event_driven"]

    math_topics = [
        "calculus", "linear_algebra", "probability", "statistics", "number_theory", 
        "graph_theory", "differential_equations", "numerical_analysis", "combinatorics", "topology"
    ]
    math_terms = [
        "jacobian_matrix", "eigenvalue_decomposition", "singular_value_decomposition", "taylor_series",
        "fourier_series", "laplace_transform", "bayes_theorem", "normal_distribution", "poisson_distribution",
        "markov_chain", "euclidean_algorithm", "prime_sieves", "euler_totient", "dijkstra_proof",
        "matrix_rank", "vector_projection", "determinant_calculation", "gradient_vector", "hessian_matrix",
        "partial_differential", "lagrange_multipliers", "riemann_integral", "line_integrals",
        "greens_theorem", "stokes_theorem", "divergence_theorem", "fibonacci_closed_form",
        "combinatorics_n_choose_k", "graph_coloring_problem", "adjacency_matrix_power", "taylor_approximation"
    ]

    total_records = 1000000
    batch_size = 100000
    records_per_domain = total_records // 2

    print(f"\nTarget: {total_records:,} entries ({records_per_domain:,} code, {records_per_domain:,} math)")
    print(f"Batch size: {batch_size:,} entries per grouping block.")

    start_time = time.time()

    # We will generate in memory, group by domain, bucket, sub_bucket, and write out to avoid I/O bottlenecks.
    for phase in ["code", "math"]:
        print(f"\n--- Phase: Injecting into {phase} domain ---")
        
        # Buffer for files
        # (bucket, sub_bucket) -> list of (addr, text, meta)
        bucket_buffers = defaultdict(list)
        
        for i in range(records_per_domain):
            # Combinatorial generation
            if phase == "code":
                lang = languages[i % len(languages)]
                task = tasks[(i // len(languages)) % len(tasks)]
                paradigm = paradigms[(i // (len(languages) * len(tasks))) % len(paradigms)]
                variation = i // (len(languages) * len(tasks) * len(paradigms))
                title = f"MOKO OS RAG Code Entry #{i}: Implementing {task} in {lang}"
                content = (
                    f"This document provides the reference implementation of {task} "
                    f"using the {paradigm} paradigm in the {lang} programming language. "
                    f"This variation covers case {variation}. It is designed for maximum "
                    f"performance within the MOKO IDE compile environment, utilizing optimized "
                    f"syntax structures, complete error handling, and type safety constraints."
                )
                filename = f"code_{task}_{lang}_{variation}.md"
            else:
                topic = math_topics[i % len(math_topics)]
                term = math_terms[(i // len(math_topics)) % len(math_terms)]
                variation = i // (len(math_topics) * len(math_terms))
                title = f"MOKO OS RAG Math Entry #{i}: Analysis of {term} in {topic}"
                content = (
                    f"This mathematical reference covers the principles of {term} within "
                    f"the field of {topic}. This is variation #{variation}. We analyze the "
                    f"fundamental proofs, system equations, and convergence metrics. "
                    f"Designed as a deep learning mathematical resource for MOKO OS "
                    f"applied engineering calculations."
                )
                filename = f"math_{term}_{topic}_{variation}.md"

            text = f"{title}\n\n{content}"
            emb = fallback_embedding(text)
            addr = encoder.encode(text, emb)

            meta = {
                "source": filename,
                "domain": phase,
                "log_number": 1,
                "val": 0.1 if phase == "code" else 0.8,
                "ar": 0.5,
                "mtype": "semantic",
                "cc": 0,
                "ts": int(time.time())
            }

            bucket_buffers[(addr.bucket, addr.sub_bucket)].append((addr, text, meta))

            # Periodic progress log during generation
            if (i + 1) % 100000 == 0:
                print(f"  Generated {i+1:,} / {records_per_domain:,} entries...")

        # Write buffered data to disk in bulk operations
        print(f"  Writing buffered buckets to disk for '{phase}' domain...")
        domain_dir = omni_root / phase
        domain_dir.mkdir(parents=True, exist_ok=True)

        written_count = 0
        for (bk, sbk), entries in bucket_buffers.items():
            bucket_dir = domain_dir / f"{bk:04x}" / f"{sbk:04x}"
            bucket_dir.mkdir(parents=True, exist_ok=True)

            index_path   = bucket_dir / "index.bin"
            vector_path  = bucket_dir / "vectors.f16"
            content_path = bucket_dir / "content.bin"
            meta_path    = bucket_dir / "meta.jsonl"

            # Check starting files sizes
            content_offset = content_path.stat().st_size if content_path.exists() else 0
            
            # Read existing meta lines count for index numbering
            record_index = 0
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    record_index = sum(1 for _ in f)

            # Build binary blocks
            idx_bytes = bytearray()
            vec_bytes = bytearray()
            content_bytes = bytearray()
            meta_lines = []

            for addr, text, meta in entries:
                compressed = zlib.compress(text.encode('utf-8'), level=6)
                content_bytes.extend(_BLOCK_STRUCT.pack(len(compressed)))
                content_bytes.extend(compressed)

                hash_bytes = bytes.fromhex(addr.content_hash)[:HASH_BYTES]
                idx_bytes.extend(_INDEX_STRUCT.pack(hash_bytes, addr.semantic_bits, content_offset))

                vec_bytes.extend(addr.fp16_vector)

                meta_entry = {
                    "idx":    record_index,
                    "hash":   addr.content_hash,
                    "source": meta["source"],
                    "domain": meta["domain"],
                    "log":    meta["log_number"],
                    "val":    meta["val"],
                    "ar":     meta["ar"],
                    "mtype":  meta["mtype"],
                    "cc":     meta["cc"],
                    "ts":     meta["ts"]
                }
                meta_lines.append(json.dumps(meta_entry) + '\n')

                # Increment sizes for the next iteration in this bucket
                content_offset += 4 + len(compressed)
                record_index += 1

            # Append to files in single write calls
            with open(content_path, 'ab') as f:
                f.write(content_bytes)
            with open(index_path, 'ab') as f:
                f.write(idx_bytes)
            with open(vector_path, 'ab') as f:
                f.write(vec_bytes)
            with open(meta_path, 'ab') as f:
                f.write("".join(meta_lines).encode('utf-8'))

            written_count += len(entries)

        # Write domain stats metadata file
        meta_file = domain_dir / "_domain_meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({"entry_count": written_count, "trained": True}, f)

        print(f"  Successfully wrote {written_count:,} entries to '{phase}' domain!")

    end_time = time.time()
    elapsed = end_time - start_time
    print("=" * 70)
    print(f"🎉 BULK INGESTION COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print(f"Ingestion speed: {total_records / elapsed:.0f} entries/second")
    print("=" * 70)

if __name__ == "__main__":
    run_bulk_injection()
