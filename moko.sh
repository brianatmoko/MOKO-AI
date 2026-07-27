#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  MOKO OS — Smart Launcher (Auto-selects mode)
#  Jalankan: ./moko.sh  atau  ./moko.sh --cli  atau  ./moko.sh --gui
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CORE_DIR="$PROJECT_DIR/moko_core"
VENV_DIR="$CORE_DIR/venv"

# Parse arguments
MODE="${1:---auto}"

echo -e "\033[1;35m[MOKO OS]\033[0m Initializing..."

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

# Auto-detect mode
if [ "$MODE" = "--auto" ]; then
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
        MODE="--gui"
    else
        MODE="--cli"
    fi
fi

# Jalankan mode yang dipilih
case "$MODE" in
    --gui)
        echo -e "\033[1;35m[MOKO OS]\033[0m Launching GUI Mode..."
        echo -e "\033[1;35m[MOKO OS]\033[0m Starting local LLM server..."
        python -c "from moko_inference.server_manager import MokoLocalInferenceServer; MokoLocalInferenceServer.start_servers()" || true

        # Auto-sync: copy latest compiled binary from build dir if it's newer
        BUILD_IDE="$PROJECT_DIR/moko_ide_cpp/build/moko_ide"
        ROOT_IDE="$PROJECT_DIR/moko_ide"
        if [ -f "$BUILD_IDE" ]; then
            if [ ! -f "$ROOT_IDE" ] || [ "$BUILD_IDE" -nt "$ROOT_IDE" ]; then
                echo -e "\033[1;36m[MOKO OS]\033[0m Syncing latest IDE binary from build..."
                cp "$BUILD_IDE" "$ROOT_IDE"
                chmod +x "$ROOT_IDE"
            fi
        fi

        if [ -f "$PROJECT_DIR/moko_ide" ]; then
            echo -e "\033[1;32m[MOKO OS]\033[0m Running Native C++ IDE..."
            "$PROJECT_DIR/moko_ide" "$@"
        else
            echo -e "\033[1;33m[WARN]\033[0m Native C++ IDE binary not found. Falling back to Python GUI..."
            cd "$CORE_DIR"
            python moko_desktop.py "$@"
        fi

        echo -e "\033[1;35m[MOKO OS]\033[0m Stopping local servers..."
        python -c "from moko_inference.server_manager import MokoLocalInferenceServer; MokoLocalInferenceServer.stop_servers()" || true
        ;;
    --cli)
        echo -e "\033[1;35m[MOKO OS]\033[0m Launching Terminal CLI Mode..."
        python moko_core/moko_os.py "$@"
        ;;
    --daemon)
        echo -e "\033[1;35m[MOKO OS]\033[0m Launching Daemon Mode..."
        cd "$CORE_DIR"
        # Cek apakah daemon sudah berjalan
        if pgrep -f "uvicorn.*moko_daemon" > /dev/null; then
            echo -e "\033[1;33m[WARN]\033[0m Daemon already running"
        else
            echo "Starting daemon on http://127.0.0.1:8000"
            python -m uvicorn moko_daemon:app --reload --host 127.0.0.1 --port 8000 &
            sleep 2
            echo -e "\033[1;32m[OK]\033[0m Daemon started. Visit http://127.0.0.1:8000/gui"
        fi
        ;;
    --help)
        cat << EOF
MOKO OS — Smart Launcher

Usage: $0 [--auto|--cli|--gui|--daemon|--help]

Modes:
  --auto       Auto-detect (GUI if display available, else CLI)
  --cli        Terminal CLI mode (default if no display)
  --gui        Desktop GUI mode (requires X11/Wayland)
  --daemon     Web daemon mode (http://127.0.0.1:8000)
  --help       Show this help message

Examples:
  $0              # Auto mode
  $0 --cli        # Force terminal mode
  $0 --gui        # Force GUI mode
  $0 --daemon     # Start daemon mode
EOF
        ;;
    *)
        echo -e "\033[1;31m[ERROR]\033[0m Unknown mode: $MODE"
        echo "Use: $0 --help for usage information"
        exit 1
        ;;
esac
