#!/bin/bash
# ============================================================
# MOKO OS — C++ Kernel Build Script (Optimized)
# ============================================================
# Mengkompilasi ulang libmoko_core.so dengan flag optimasi
# tertinggi untuk mendapatkan performa maksimal.
#
# Penggunaan:
#   cd moko_core/moko_cpp_kernel
#   bash ../../build_kernel.sh
# atau dari root project:
#   bash build_kernel.sh
# ============================================================

set -e  # Berhenti jika ada error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/moko_core/moko_cpp_kernel"

# Jika dipanggil dari dalam folder cpp_kernel itu sendiri
if [ -f "$SCRIPT_DIR/moko_kernel.cpp" ]; then
    KERNEL_DIR="$SCRIPT_DIR"
fi

echo "============================================"
echo "  MOKO Kernel Build — Optimized Release"
echo "============================================"
echo "📁 Kernel Dir: $KERNEL_DIR"

cd "$KERNEL_DIR"

# Deteksi jumlah CPU untuk kompilasi paralel
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "🖥️  CPU Cores: $NCPU (paralel build)"

# ── Flag Kompilasi ────────────────────────────────────────
# -O3            : Level optimasi tertinggi GCC/Clang
# -march=native  : Gunakan instruksi CPU terbaik mesin ini (AVX2, SSE4 dll)
# -ffast-math    : Izinkan re-ordering floating point (aman untuk dot product)
# -funroll-loops : Unroll loop otomatis (mempercepat inner loop LUT scan)
# -flto          : Link-Time Optimization (optimasi lintas file)
# -fPIC          : Position Independent Code (wajib untuk .so)
# -shared        : Buat shared library
# -std=c++17     : C++17 untuk std::invoke_result
# -pthread       : Enable POSIX thread support
# -Wall -Wextra  : Tampilkan semua warning selama build
CXX="${CXX:-g++}"

CXXFLAGS="-O3 -march=native -ffast-math -funroll-loops -fPIC -std=c++17 -pthread"
LDFLAGS="-shared -flto -pthread"
WARNINGS="-Wall -Wextra -Wno-unused-parameter"

OUTPUT="libmoko_core.so"
SOURCE="moko_kernel.cpp"

echo ""
echo "🔧 Compiler  : $CXX"
echo "🚩 CXXFLAGS  : $CXXFLAGS"
echo "🔗 LDFLAGS   : $LDFLAGS"
echo ""

# Hapus .so lama agar tidak ada state stale
if [ -f "$OUTPUT" ]; then
    echo "🗑️  Menghapus build lama: $OUTPUT"
    rm -f "$OUTPUT"
fi

# Kompilasi
echo "⚙️  Mengkompilasi $SOURCE → $OUTPUT ..."
$CXX $CXXFLAGS $WARNINGS $LDFLAGS \
    -o "$OUTPUT" \
    "$SOURCE" \
    2>&1

if [ $? -eq 0 ] && [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "✅ BUILD SUKSES!"
    echo "   Output : $KERNEL_DIR/$OUTPUT"
    echo "   Size   : $SIZE"
    echo ""

    # Verifikasi simbol yang diekspor
    echo "🔍 Verifikasi simbol yang diekspor:"
    nm -D "$OUTPUT" | grep -E "moko_kernel_init|encode_qev_c|search_mmap_top_k_c" | head -10
    echo ""
    echo "🎉 libmoko_core.so siap digunakan oleh Python!"
else
    echo ""
    echo "❌ BUILD GAGAL! Periksa error di atas."
    exit 1
fi
