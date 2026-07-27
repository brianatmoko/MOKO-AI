#!/usr/bin/env bash
# =============================================================================
# MOKO Coder 1B — Training Pipeline Runner
# =============================================================================
# Script ini menjalankan seluruh pipeline fine-tuning MOKO Coder 1B:
#   Training QLoRA → Merge LoRA → Convert ke GGUF Q4_K_M
#
# Base Model (HuggingFace format / safetensors):
#   huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
#   https://huggingface.co/huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
#
# PRASYARAT — Download manual model terlebih dahulu:
#   1. Buka: https://huggingface.co/huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
#   2. Download file: model.safetensors, config.json, tokenizer.json,
#                     tokenizer_config.json, generation_config.json,
#                     vocab.json, merges.txt
#   3. Simpan semua ke: finetune/base_model_coder_hf/
#
# Usage:
#   chmod +x finetune/run_training.sh
#   cd /home/brianatmokoo/Documents/Linux/MOKO_OS_Project
#   bash finetune/run_training.sh [epochs]   # default epochs=3
#
# =============================================================================

set -e  # Hentikan jika ada error

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FINETUNE_DIR="$PROJECT_DIR/finetune"
PYTHON="$PROJECT_DIR/moko_core/venv/bin/python3"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         MOKO Coder 1B — Training Pipeline                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project Dir : $PROJECT_DIR"
echo "  Python      : $PYTHON"
echo "  Start Time  : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Cek Python ────────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "❌ Python venv tidak ditemukan di: $PYTHON"
    echo "   Pastikan virtual environment sudah dibuat."
    exit 1
fi

# ── Cek base model sudah didownload ────────────────────────────────────────
WEIGHT_FILE="$FINETUNE_DIR/base_model_coder_hf/model.safetensors"
if [ ! -f "$WEIGHT_FILE" ] || [ $(stat -c%s "$WEIGHT_FILE") -lt 2000000000 ]; then
    echo "⚠️  Base model belum lengkap. Menjalankan download terlebih dahulu..."
    echo ""
    "$PYTHON" "$FINETUNE_DIR/moko_trainer_v2.py" --prepare
    echo ""
fi

# ── Cek dataset tersedia ────────────────────────────────────────────────────
DATASET_DIR="$FINETUNE_DIR/moko_datasets"
OS_DATASET="$DATASET_DIR/moko_os_code_dataset.jsonl"
if [ ! -f "$OS_DATASET" ]; then
    echo "⚠️  OS Code dataset tidak ditemukan. Menjalankan extractor..."
    echo ""
    "$PYTHON" "$FINETUNE_DIR/moko_os_code_extractor.py"
    echo ""
fi

# ── STEP 1: TRAINING ───────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1 / 2 — QLoRA Fine-tuning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

EPOCHS=${1:-3}
echo "  Epochs      : $EPOCHS"
echo "  Max Samples : 10000"
echo "  Seq Length  : 1536 tokens"
echo ""

START_TRAIN=$(date +%s)
"$PYTHON" "$FINETUNE_DIR/moko_trainer_v2.py" --train --epochs "$EPOCHS"
END_TRAIN=$(date +%s)

TRAIN_DURATION=$((END_TRAIN - START_TRAIN))
echo ""
echo "  ✅ Training selesai dalam $(($TRAIN_DURATION / 60)) menit $(($TRAIN_DURATION % 60)) detik"
echo ""

# ── STEP 2: GGUF CONVERSION ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2 / 2 — Merge LoRA + Convert ke GGUF Q4_K_M"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_GGUF=$(date +%s)
"$PYTHON" "$FINETUNE_DIR/moko_trainer_v2.py" --gguf
END_GGUF=$(date +%s)

GGUF_DURATION=$((END_GGUF - START_GGUF))
echo ""
echo "  ✅ GGUF conversion selesai dalam $(($GGUF_DURATION / 60)) menit $(($GGUF_DURATION % 60)) detik"
echo ""

# ── HASIL ──────────────────────────────────────────────────────────────────
GGUF_PATH="$PROJECT_DIR/moko-coder-1b.gguf"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  ✅  PIPELINE SELESAI!                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$GGUF_PATH" ]; then
    GGUF_SIZE=$(du -sh "$GGUF_PATH" | cut -f1)
    echo "  📦 Model GGUF  : $GGUF_PATH"
    echo "  📏 Ukuran      : $GGUF_SIZE"
    echo "  🕒 End Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "  Langkah selanjutnya:"
    echo "  1. Restart MOKO OS agar settings.py mendeteksi moko-coder-1b.gguf"
    echo "  2. Cek status: python3 moko_core/moko_agents/moko_coder_1b_agent.py"
    echo "  3. Test inference: bash finetune/test_coder_1b.sh"
else
    echo "  ⚠️  GGUF tidak ditemukan di $GGUF_PATH"
    echo "     Cek log error di atas untuk detail."
    exit 1
fi
