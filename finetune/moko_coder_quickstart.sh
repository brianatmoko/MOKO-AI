#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MOKO Coder v3 — Quick Start
# ═══════════════════════════════════════════════════════════════
#
# Satu script untuk setup lengkap:
#   1. Check dependencies
#   2. Download coding datasets
#   3. Convert ke HEX format
#   4. Train LoRA adapter
#   5. Convert ke GGUF untuk deployment
#
# Usage:
#   bash moko_coder_quickstart.sh              # Full pipeline
#   bash moko_coder_quickstart.sh --download   # Download only
#   bash moko_coder_quickstart.sh --train      # Train only
#   bash moko_coder_quickstart.sh --status     # Status only
#
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FINETUNE_DIR="$SCRIPT_DIR"
DATASET_DIR="$FINETUNE_DIR/moko_datasets"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  MOKO Coder v3 — Quick Start Pipeline${NC}"
    echo -e "${BLUE}  Hex Binary Mode: code → compress → hex → LLM${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "\n${YELLOW}▸ Step $1: $2${NC}"
}

print_ok() {
    echo -e "  ${GREEN}✅ $1${NC}"
}

print_fail() {
    echo -e "  ${RED}❌ $1${NC}"
}

print_warn() {
    echo -e "  ${YELLOW}⚠️  $1${NC}"
}

# ═══════════════════════════════════════════════════════════════
# STEP 1: CHECK DEPENDENCIES
# ═══════════════════════════════════════════════════════════════
check_deps() {
    print_step "1" "Checking dependencies..."

    # Python
    if command -v python3 &>/dev/null; then
        PYVER=$(python3 --version 2>&1)
        print_ok "Python: $PYVER"
    else
        print_fail "Python3 not found"
        exit 1
    fi

    # PyTorch
    if python3 -c "import torch; print(torch.__version__)" &>/dev/null; then
        TORCHVER=$(python3 -c "import torch; print(torch.__version__)")
        print_ok "PyTorch: $TORCHVER"

        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
            VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}')")
            print_ok "GPU: $GPU_NAME ($VRAM GB)"
        else
            print_warn "CUDA not available — training will be slow (CPU only)"
        fi
    else
        print_warn "PyTorch not installed — training will fail"
    fi

    # transformers + peft + trl
    for pkg in transformers peft trl datasets bitsandbytes; do
        if python3 -c "import $pkg" 2>/dev/null; then
            print_ok "$pkg installed"
        else
            print_warn "$pkg not installed — training may fail"
        fi
    done

    # huggingface_hub
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        print_ok "huggingface_hub installed"
    else
        print_warn "huggingface_hub not installed — download will fail"
    fi
}

# ═══════════════════════════════════════════════════════════════
# STEP 2: DOWNLOAD DATASETS
# ═══════════════════════════════════════════════════════════════
download_datasets() {
    print_step "2" "Downloading coding datasets..."

    if [ -f "$DATASET_DIR/moko_coder_dataset.jsonl" ]; then
        COUNT=$(wc -l < "$DATASET_DIR/moko_coder_dataset.jsonl")
        print_ok "Dataset already exists: $COUNT samples"
    else
        python3 "$FINETUNE_DIR/download_coding_datasets.py" --dataset codealpaca --max-samples 5000
        if [ $? -eq 0 ]; then
            print_ok "Download complete"
        else
            print_fail "Download failed"
            return 1
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════
# STEP 3: CONVERT TO HEX FORMAT
# ═══════════════════════════════════════════════════════════════
convert_hex() {
    print_step "3" "Converting to HEX binary format..."

    if [ -f "$DATASET_DIR/moko_coder_hex.jsonl" ]; then
        COUNT=$(wc -l < "$DATASET_DIR/moko_coder_hex.jsonl")
        print_ok "HEX dataset already exists: $COUNT samples"
    else
        python3 "$FINETUNE_DIR/moko_hex_encoder.py" --validate \
            "$DATASET_DIR/moko_coder_dataset.jsonl" \
            "$DATASET_DIR/moko_coder_hex.jsonl"
        if [ $? -eq 0 ]; then
            print_ok "HEX conversion complete"
        else
            print_fail "HEX conversion failed"
            return 1
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════
# STEP 4: TRAIN LORA ADAPTER
# ═══════════════════════════════════════════════════════════════
train_model() {
    print_step "4" "Training MOKO Coder LoRA adapter..."

    python3 "$FINETUNE_DIR/moko_finetune.py" --train --epochs 3 --max-samples 2000
    if [ $? -eq 0 ]; then
        print_ok "Training complete"
    else
        print_fail "Training failed"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════
# STEP 5: SHOW STATUS
# ═══════════════════════════════════════════════════════════════
show_status() {
    print_step "5" "Status..."
    python3 "$FINETUNE_DIR/moko_finetune.py" --status
}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print_header

case "${1:-}" in
    --download)
        check_deps
        download_datasets
        convert_hex
        ;;
    --train)
        check_deps
        train_model
        ;;
    --status)
        show_status
        ;;
    --hex)
        convert_hex
        ;;
    *)
        check_deps
        download_datasets
        convert_hex
        show_status
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  Setup complete! Next steps:${NC}"
        echo -e "${GREEN}    python3 finetune/moko_finetune.py --train --epochs 3${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        ;;
esac
