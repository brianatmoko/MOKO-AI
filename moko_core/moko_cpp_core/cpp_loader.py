#!/usr/bin/env python3
import ctypes
import os
from pathlib import Path

# Cari dynamic library .so
_so_dir = Path(__file__).parent
_so_path = _so_dir / "libmoko_cpp.so"

# System-wide path fallback
if not _so_path.exists():
    _so_path = Path("moko_core/moko_cpp_core/libmoko_cpp.so")

if not _so_path.exists():
    raise FileNotFoundError(f"Cannot find dynamic library at {_so_path}")

# Load dynamic library via ctypes
_lib = ctypes.CDLL(str(_so_path.resolve()))

# 1. cpp_fuse_build
_lib.cpp_fuse_build.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
_lib.cpp_fuse_build.restype = ctypes.c_bool

# 2. cpp_fuse_might_contain
_lib.cpp_fuse_might_contain.argtypes = [ctypes.c_char_p]
_lib.cpp_fuse_might_contain.restype = ctypes.c_bool

# 3. cpp_compute_simhash
_lib.cpp_compute_simhash.argtypes = [ctypes.c_char_p]
_lib.cpp_compute_simhash.restype = ctypes.c_uint64

# 4. cpp_shannon_entropy
_lib.cpp_shannon_entropy.argtypes = [ctypes.c_char_p]
_lib.cpp_shannon_entropy.restype = ctypes.c_float

# 5. cpp_fhe_mask_text & cpp_fhe_unmask_text
_lib.cpp_fhe_mask_text.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.cpp_fhe_mask_text.restype = None

_lib.cpp_fhe_unmask_text.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.cpp_fhe_unmask_text.restype = None


# ── Python High Level API ─────────────────────────────────────────────────────

def cpp_fuse_build_python(token_ids: list) -> bool:
    """Build Binary Fuse Filter C++."""
    count = len(token_ids)
    if count == 0:
        return True
    
    bytes_list = [tid.encode("utf-8") for tid in token_ids]
    c_arr = (ctypes.c_char_p * count)()
    for i, b_str in enumerate(bytes_list):
        c_arr[i] = b_str
        
    return bool(_lib.cpp_fuse_build(c_arr, count))


def cpp_fuse_might_contain_python(token_id: str) -> bool:
    """Check membership via Binary Fuse Filter C++."""
    return bool(_lib.cpp_fuse_might_contain(token_id.encode("utf-8")))


def cpp_compute_simhash_python(text: str) -> int:
    """Hitung SimHash-64 cepat di C++."""
    return int(_lib.cpp_compute_simhash(text.encode("utf-8")))


def cpp_shannon_entropy_python(text: str) -> float:
    """Hitung Shannon Entropy cepat di C++."""
    return float(_lib.cpp_shannon_entropy(text.encode("utf-8")))


def cpp_fhe_mask_python(text: str, key: str = "MOKO_SECRET_KEY") -> str:
    """Masking/obfuscation text deterministik di C++ menggunakan key."""
    text_bytes = text.encode("utf-8")
    key_bytes = key.encode("utf-8")
    # Alokasikan string buffer kosong sebesar panjang text + 1 (null terminator)
    out_buf = ctypes.create_string_buffer(len(text_bytes) + 1)
    
    _lib.cpp_fhe_mask_text(text_bytes, key_bytes, out_buf)
    return out_buf.value.decode("utf-8", errors="replace")


def cpp_fhe_unmask_python(cipher: str, key: str = "MOKO_SECRET_KEY") -> str:
    """Unmasking/de-obfuscation text di C++ menggunakan key."""
    cipher_bytes = cipher.encode("utf-8")
    key_bytes = key.encode("utf-8")
    out_buf = ctypes.create_string_buffer(len(cipher_bytes) + 1)
    
    _lib.cpp_fhe_unmask_text(cipher_bytes, key_bytes, out_buf)
    return out_buf.value.decode("utf-8", errors="replace")


# ── Self-Test & Benchmark ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    import hashlib

    print("=" * 60)
    print("  MOKO AI — C++ Native Core Loader Self-Test (Phase 10)")
    print("=" * 60)

    # 1. Test FHE Masking
    print("\n[1] Testing Homomorphic Masking (FHE)...")
    plaintext = "Bagaimana cara menghitung rumus CC motor MOKO dengan bore 56mm dan stroke 50mm?"
    secret_key = "MOKO_KEY_2026"
    
    cipher = cpp_fhe_mask_python(plaintext, secret_key)
    recovered = cpp_fhe_unmask_python(cipher, secret_key)
    
    print(f"  Plaintext: '{plaintext}'")
    print(f"  Ciphertext: '{cipher}'")
    print(f"  Recovered: '{recovered}'")
    assert recovered == plaintext, "FHE recovery failed!"
    print("  Status: OK — Masking & Recovery 100% akurat!")

    # 2. Test SimHash & Entropy
    print("\n[2] Testing SimHash & Shannon Entropy...")
    sh = cpp_compute_simhash_python(plaintext)
    ent = cpp_shannon_entropy_python(plaintext)
    print(f"  SimHash: {sh} (Hex: {format(sh, '016x')})")
    print(f"  Entropy: {ent:.4f} bits/char")

    # 3. Test Binary Fuse Filter
    print("\n[3] Testing Binary Fuse Filter (C++)...")
    tokens = [hashlib.sha256(f"tok_{i}".encode()).hexdigest() for i in range(5000)]
    t0 = time.perf_counter()
    ok = cpp_fuse_build_python(tokens)
    build_us = (time.perf_counter() - t0) * 1_000_000
    print(f"  Build C++ Filter: {ok} | Time: {build_us:.1f}μs (for {len(tokens)} items)")

    tp = sum(1 for tok in tokens[:100] if cpp_fuse_might_contain_python(tok))
    print(f"  True Positive: {tp}/100")

    fake = [hashlib.sha256(f"fake_{i}".encode()).hexdigest() for i in range(1000)]
    fp = sum(1 for fk in fake if cpp_fuse_might_contain_python(fk))
    print(f"  False Positive: {fp}/1000 (FPR: {fp/10:.2f}%)")

    # Benchmark Speed
    print("\n[4] Benchmark 100K queries...")
    N = 100_000
    query_list = tokens[:1000] * 100
    t0 = time.perf_counter()
    for q in query_list:
        cpp_fuse_might_contain_python(q)
    elapsed = (time.perf_counter() - t0) * 1_000_000 / N
    print(f"  Query time: {elapsed:.3f}μs per check")
    print(f"  Throughput: {1_000_000 / elapsed / 1e6:.2f} Million checks/sec")

    print("\n✅ C++ Native Core Loader verified!")
