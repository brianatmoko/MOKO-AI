#!/usr/bin/env bash
# =============================================================================
# MOKO NATIVE ACCELERATION CORE — BUILD SCRIPT
# -----------------------------------------------------------------------------
# Mengompilasi inti native jalur panas Anchor-RAG:
#   1. Tier C++  (g++)   -> libmoko_native.so       (tier utama)
#   2. Tier Rust (cargo) -> libmoko_native_rs.so    (tier "lebih kuat", opsional)
#
# Loader Python (native_accel.py) memilih backend secara otomatis:
#   Rust (jika ada) -> C++ (jika ada) -> fallback murni-Python.
#
# Skrip ini idempotent & aman: jika sebuah toolchain tidak tersedia, tier itu
# dilewati (bukan gagal fatal) selama minimal satu tier berhasil dibangun.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_SRC="${SCRIPT_DIR}/cpp/moko_native.cpp"
CPP_OUT="${SCRIPT_DIR}/libmoko_native.so"
RUST_DIR="${SCRIPT_DIR}/rust"
RUST_OUT="${SCRIPT_DIR}/libmoko_native_rs.so"

cpp_ok=0
rust_ok=0

echo "== MOKO Native Build =="
echo "Script dir : ${SCRIPT_DIR}"

# --- Tier 1: C++ -------------------------------------------------------------
if command -v g++ >/dev/null 2>&1; then
    echo "[C++]  Mengompilasi ${CPP_SRC} -> ${CPP_OUT}"
    if g++ -O3 -std=c++17 -fPIC -shared -Wall -Wextra \
        "${CPP_SRC}" -o "${CPP_OUT}"; then
        echo "[C++]  OK -> ${CPP_OUT}"
        cpp_ok=1
    else
        echo "[C++]  GAGAL mengompilasi."
    fi
else
    echo "[C++]  g++ tidak ditemukan — tier C++ dilewati."
fi

# --- Tier 2: Rust (opsional, 'lebih kuat') -----------------------------------
if command -v cargo >/dev/null 2>&1; then
    echo "[Rust] Membangun crate cdylib di ${RUST_DIR}"
    if ( cd "${RUST_DIR}" && cargo build --release --quiet ); then
        # Nama artefak berbeda per-OS; cari yang cocok.
        built=""
        for cand in \
            "${RUST_DIR}/target/release/libmoko_native_rs.so" \
            "${RUST_DIR}/target/release/libmoko_native_rs.dylib" \
            "${RUST_DIR}/target/release/moko_native_rs.dll"; do
            if [ -f "${cand}" ]; then built="${cand}"; break; fi
        done
        if [ -n "${built}" ]; then
            cp -f "${built}" "${RUST_OUT}"
            echo "[Rust] OK -> ${RUST_OUT}"
            rust_ok=1
        else
            echo "[Rust] Build sukses namun artefak cdylib tidak ditemukan."
        fi
    else
        echo "[Rust] GAGAL membangun crate."
    fi
else
    echo "[Rust] cargo tidak ditemukan — tier Rust dilewati."
fi

echo "-----------------------------------------"
echo "Ringkasan: C++=${cpp_ok}  Rust=${rust_ok}"
if [ "${cpp_ok}" -eq 1 ] || [ "${rust_ok}" -eq 1 ]; then
    echo "STATUS: minimal satu tier native berhasil dibangun."
    exit 0
else
    echo "STATUS: tidak ada tier native yang terbangun (Python fallback tetap jalan)."
    exit 1
fi
