#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  MOKO OS — Launcher (Native C++ IDE)
#  Jalankan: ./moko_launcher.sh  atau  moko  (jika shortcut aktif)
#  GUI: 100% Native C++ Qt5 — tidak ada fallback Python PyQt6
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CORE_DIR="$PROJECT_DIR/moko_core"
VENV_DIR="$CORE_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

echo -e "\033[1;35m[MOKO OS]\033[0m Initializing..."

# Set Python path untuk inference server (backend saja, bukan GUI)
if [ -f "$VENV_PYTHON" ]; then
    source "$VENV_DIR/bin/activate"
fi
export PYTHONPATH="$PROJECT_DIR:$CORE_DIR:$PYTHONPATH"

# Pastikan DISPLAY tersedia (untuk X11/Wayland)
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# ── Auto-swap fresh build jika ada moko_ide.new ─────────────────
if [ -f "$PROJECT_DIR/moko_ide.new" ]; then
    echo -e "\033[1;33m[MOKO OS]\033[0m Fresh build tersedia. Mengganti binary..."
    cp "$PROJECT_DIR/moko_ide.new" "$PROJECT_DIR/moko_ide"
    chmod +x "$PROJECT_DIR/moko_ide"
    rm -f "$PROJECT_DIR/moko_ide.new"
    echo -e "\033[1;32m[MOKO OS]\033[0m Binary diperbarui ✅"
fi

# ── Validasi binary C++ ──────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/moko_ide" ]; then
    echo -e "\033[1;31m[ERROR]\033[0m moko_ide binary tidak ditemukan."
    echo -e "\033[1;33m[INFO]\033[0m  Build dengan perintah: ./build_ide.sh"
    echo -e "\033[1;33m[INFO]\033[0m  Atau: cd moko_ide_cpp && cmake -B build && make -C build -j\$(nproc)"
    exit 1
fi

# ── Start inference server (Python backend, bukan GUI) ───────────
echo -e "\033[1;32m[MOKO OS]\033[0m Starting local LLM server..."
if [ -f "$VENV_PYTHON" ]; then
    python -c "from moko_inference.server_manager import MokoLocalInferenceServer; MokoLocalInferenceServer.start_servers()" || true
fi

# ── Launch Native C++ IDE ────────────────────────────────────────
echo -e "\033[1;32m[MOKO OS]\033[0m Launching Native C++ IDE..."
"$PROJECT_DIR/moko_ide" "$@"

# ── Stop inference server ────────────────────────────────────────
echo -e "\033[1;32m[MOKO OS]\033[0m Stopping local servers..."
if [ -f "$VENV_PYTHON" ]; then
    python -c "from moko_inference.server_manager import MokoLocalInferenceServer; MokoLocalInferenceServer.stop_servers()" || true
fi
