#!/usr/bin/env bash
# =============================================================================
# MOKO Coder 1B — Quick Integration Test
# =============================================================================
# Menguji apakah model 1B berhasil di-load dan bisa menjawab pertanyaan
# dasar seputar MOKO OS.
#
# Usage:
#   bash finetune/test_coder_1b.sh
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT_DIR/moko_core/venv/bin/python3"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MOKO Coder 1B — Integration Test                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Jalankan agent status check dan quick test
cd "$PROJECT_DIR/moko_core"
"$PYTHON" moko_agents/moko_coder_1b_agent.py

echo ""
echo "Test selesai."
