#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  MOKO OS — Terminal CLI Launcher
#  Jalankan: ./moko_cli.sh
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CORE_DIR="$PROJECT_DIR/moko_core"
VENV_DIR="$CORE_DIR/venv"

echo -e "\033[1;35m[MOKO OS CLI]\033[0m Initializing..."

# Validasi venv exists
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo -e "\033[1;31m[ERROR]\033[0m Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" --upgrade-deps
fi

# Aktivasi virtual environment
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"

# Set Python path untuk imports yang benar
export PYTHONPATH="$PROJECT_DIR:$CORE_DIR:$PYTHONPATH"

# Jalankan Terminal UI
echo -e "\033[1;35m[MOKO OS]\033[0m Starting MOKO OS..."
python moko_core/moko_os.py "$@"
