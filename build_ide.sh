#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  MOKO IDE — Build Script (Native C++/Qt5)
#  Usage: ./build_ide.sh [--clean]
# ─────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/moko_ide_cpp"
BUILD_DIR="$SRC_DIR/build"
BINARY="$BUILD_DIR/moko_ide"
DEST="$SCRIPT_DIR/moko_ide"

echo -e "\033[1;35m[MOKO BUILD]\033[0m Starting native IDE build..."

# Optional clean
if [[ "$1" == "--clean" ]]; then
    echo -e "\033[1;33m[MOKO BUILD]\033[0m Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Configure
mkdir -p "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1

# Build with all cores
echo -e "\033[1;35m[MOKO BUILD]\033[0m Compiling with $(nproc) cores..."
make -C "$BUILD_DIR" -j$(nproc) 2>&1

# Deploy binary to project root
if cp "$BINARY" "$DEST" 2>/dev/null; then
    chmod +x "$DEST"
    echo ""
    echo -e "\033[1;32m[MOKO BUILD]\033[0m \u2705 Build success! Binary: $DEST"
    echo -e "\033[1;32m[MOKO BUILD]\033[0m Run with: moko  or  $DEST"
else
    cp "$BINARY" "$DEST.new"
    chmod +x "$DEST.new"
    echo ""
    echo -e "\033[1;33m[MOKO BUILD]\033[0m \u26a0\ufe0f Target binary is busy. Deployed as $DEST.new."
    echo -e "\033[1;33m[MOKO BUILD]\033[0m Restart IDE to swap and load the fresh changes."
fi
