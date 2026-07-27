"""
MOKO Trainer V3 — 7B Model Deep Fine-Tuning
============================================
Target: Fine-tune Qwen2.5-Coder-7B / DeepSeek-Coder-7B -> MOKO AI 9B INT4

Strategy:
  - Base: Qwen2.5-Coder-7B-Instruct (or deepseek-coder-7b-instruct)
  - Method: QLoRA 4-bit NF4 + PEFT LoRA (r=32)
  - Data: 50K+ samples dari moko_datasets/
  - Output: MOKO-AI-7B-Coder.gguf (Q4_K_M ~4.5GB)

Hardware Profile: RTX 2050 4GB VRAM
  - load_in_4bit + double_quant: ~3.5GB model
  - gradient_checkpointing: hemat 40% activation memory
  - max_seq_len=512: fit di 4GB dengan headroom
  - paged_adamw_8bit: optimizer di CPU jika overflow

Usage:
  python3 moko_trainer_v3_7b.py --check      # Cek hardware & dependencies
  python3 moko_trainer_v3_7b.py --prepare    # Download base model 7B
  python3 moko_trainer_v3_7b.py --train      # Mulai training
  python3 moko_trainer_v3_7b.py --train --epochs 5 --seq 512
  python3 moko_trainer_v3_7b.py --merge      # Merge LoRA -> BF16
  python3 moko_trainer_v3_7b.py --gguf       # Convert -> GGUF INT4
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# ─── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR  = FINETUNE_DIR / "moko_datasets"
OUTPUT_DIR   = FINETUNE_DIR / "moko_adapters"
LOG_DIR      = FINETUNE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Tambahkan venv libs ke path
_site = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site.exists() and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

# ─── Config ───────────────────────────────────────────────────────────────────
# Opsi model base (pilih salah satu):
BASE_MODELS = {
    "qwen_7b":       "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek_7b":   "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "qwen_7b_abl":   "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated",
}
SELECTED_BASE  = "qwen_7b"         # Ganti ke pilihan Anda
BASE_MODEL_DIR = FINETUNE_DIR / "base_model_7b_hf"
ADAPTER_DIR    = OUTPUT_DIR / "moko_coder_7b"
MERGED_DIR     = OUTPUT_DIR / "moko_coder_7b_merged"
FINAL_GGUF     = PROJECT_DIR / "MOKO-AI-7B-Coder-Q4_K_M.gguf"

# LoRA config — dioptimalkan untuk 7B di 4GB VRAM
LORA_CONFIG = {
    "r":              32,    # Rank (lebih besar = lebih ekspresif, lebih banyak VRAM)
    "lora_alpha":     64,    # Alpha = 2x rank standar
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout":   0.05,
    "bias":           "none",
    "task_type":      "CAUSAL_LM",
}

# Training config — hemat VRAM untuk RTX 2050 4GB
TRAINING_CONFIG = {
    "num_train_epochs":              3,
    "per_device_train_batch_size":   1,
    "gradient_accumulation_steps":  16,   # Effective batch = 16
    "gradient_checkpointing":       True, # WAJIB untuk 7B di 4GB
    "learning_rate":                1e-4,
    "max_seq_length":               512,  # Aman untuk 4GB + NF4
    "warmup_ratio":                 0.05,
    "lr_scheduler_type":            "cosine",
    "optim":                        "paged_adamw_8bit",
    "fp16":                         False,
    "bf16":                         True,
    "logging_steps":                10,
    "save_strategy":                "epoch",
    "save_total_limit":             2,
    "report_to":                    "none",
    "dataloader_num_workers":       0,    # Aman untuk single GPU
}

# BnB 4-bit quantization config
BNB_CONFIG = {
    "load_in_4bit":              True,
    "bnb_4bit_quant_type":       "nf4",
    "bnb_4bit_compute_dtype":    "bfloat16",
    "bnb_4bit_use_double_quant": True,    # Hemat ~0.4GB VRAM
}


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    icons = {"INFO": "->", "OK": "[OK]", "WARN": "[!]", "ERROR": "[ERR]"}
    print(f"[{ts}] {icons.get(level,'.')} {msg}")


def check_hardware():
    """Cek GPU dan ketersediaan VRAM."""
    import torch
    if not torch.cuda.is_available():
        log("GPU tidak terdeteksi! Training akan sangat lambat di CPU.", "WARN")
        return False, 0.0
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_gb  = torch.cuda.mem_get_info()[0] / 1e9
    log(f"GPU: {name} | Total: {total_gb:.1f}GB | Free: {free_gb:.1f}GB")
    if total_gb < 4.0:
        log("VRAM < 4GB — training 7B mungkin OOM!", "WARN")
    return True, total_gb


def merge_datasets() -> Path:
    """Gabungkan semua dataset JSONL ke satu file untuk training."""
    import json, random

    DATASET_FILES = [
        DATASET_DIR / "moko_coder_dataset.jsonl",
        DATASET_DIR / "moko_os_code_dataset.jsonl",
        DATASET_DIR / "moko_cpp_qt_dataset.jsonl",
        DATASET_DIR / "moko_algo_dataset.jsonl",
        DATASET_DIR / "moko_security_dataset.jsonl",
        DATASET_DIR / "moko_ide_integration.jsonl",
        DATASET_DIR / "moko_multiturn_dataset.jsonl",
        DATASET_DIR / "moko_reasoning_dataset.jsonl",
        DATASET_DIR / "moko_docs_dataset.jsonl",
    ]

    merged_path = DATASET_DIR / "moko_merged_7b.jsonl"
    all_samples = []

    for f in DATASET_FILES:
        if not f.exists():
            log(f"  Skipping {f.name} (not found)", "WARN")
            continue
        count = 0
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    if 'messages' in d:
                        all_samples.append(d)
                        count += 1
                except:
                    pass
        log(f"  Loaded {count:,} from {f.name}")

    # Shuffle untuk curriculum yang baik
    random.shuffle(all_samples)

    with open(merged_path, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    log(f"Merged {len(all_samples):,} samples -> {merged_path.name}", "OK")
    return merged_path


def prepare_base_model():
    """Download base model 7B dari HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("Install huggingface_hub: pip install huggingface-hub", "ERROR")
        return False

    model_id = BASE_MODELS[SELECTED_BASE]
    log(f"Downloading {model_id} ke {BASE_MODEL_DIR}...")
    log("Ukuran: ~15GB — pastikan storage cukup.", "WARN")

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.bin"],  # Skip .bin, hanya download .safetensors
        )
        log(f"Download selesai: {BASE_MODEL_DIR}", "OK")
        return True
    except Exception as e:
        log(f"Download error: {e}", "ERROR")
        log("Download manual: https://huggingface.co/" + model_id, "INFO")
        return False


def train(epochs: int = None, seq_len: int = None):
    """Jalankan QLoRA training."""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        import json
    except ImportError as e:
        log(f"Dependency missing: {e}", "ERROR")
        log("Run: pip install transformers peft datasets bitsandbytes trl", "INFO")
        return False

    if not BASE_MODEL_DIR.exists() or not any(BASE_MODEL_DIR.glob("*.safetensors")):
        log("Base model tidak ditemukan! Jalankan --prepare dulu.", "ERROR")
        return False

    # Override config jika ada args
    cfg = dict(TRAINING_CONFIG)
    if epochs:   cfg["num_train_epochs"]  = epochs
    if seq_len:  cfg["max_seq_length"]    = seq_len

    log(f"Training config: epochs={cfg['num_train_epochs']}, "
        f"seq_len={cfg['max_seq_length']}, "
        f"eff_batch={cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    log("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Load model in 4-bit ───────────────────────────────────────────────────
    log("Loading model in 4-bit NF4...")
    bnb = BitsAndBytesConfig(**{k: v for k, v in BNB_CONFIG.items()
                                if k != "bnb_4bit_compute_dtype"},
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model,
        use_gradient_checkpointing=cfg["gradient_checkpointing"])

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    log(f"Applying LoRA (r={LORA_CONFIG['r']})...")
    lora_cfg = LoraConfig(**{k: v for k, v in LORA_CONFIG.items()})
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Prepare dataset ───────────────────────────────────────────────────────
    log("Loading & merging datasets...")
    merged = merge_datasets()

    samples = []
    with open(merged, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                # Format menjadi ChatML text
                text = tok.apply_chat_template(
                    d["messages"],
                    tokenize=False,
                    add_generation_prompt=False
                )
                samples.append({"text": text})
            except:
                pass

    log(f"Dataset: {len(samples):,} training samples")
    dataset = Dataset.from_list(samples)

    def tokenize_fn(batch):
        out = tok(batch["text"],
                  max_length=cfg["max_seq_length"],
                  truncation=True,
                  padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, batched=True,
                             remove_columns=["text"],
                             num_proc=1)

    # ── Training args ─────────────────────────────────────────────────────────
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        optim=cfg["optim"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg["report_to"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    from transformers import Trainer, DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(tok, model=model,
                                       label_pad_token_id=-100,
                                       pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    log("Starting training...", "OK")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log(f"Training selesai dalam {elapsed/3600:.1f} jam!", "OK")

    # Save adapter
    model.save_pretrained(str(ADAPTER_DIR / "lora_adapter"))
    tok.save_pretrained(str(ADAPTER_DIR / "lora_adapter"))
    log(f"LoRA adapter saved: {ADAPTER_DIR / 'lora_adapter'}", "OK")

    # Save status
    with open(ADAPTER_DIR / "status.json", 'w') as f:
        json.dump({
            "status":    "complete",
            "base":      BASE_MODELS[SELECTED_BASE],
            "adapter":   str(ADAPTER_DIR / "lora_adapter"),
            "epochs":    cfg["num_train_epochs"],
            "samples":   len(samples),
            "timestamp": int(time.time()),
        }, f, indent=2)

    return True


def merge_lora():
    """Merge LoRA weights ke base model (BF16 full model)."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
    except ImportError as e:
        log(f"Missing: {e}", "ERROR"); return False

    adapter_path = ADAPTER_DIR / "lora_adapter"
    if not adapter_path.exists():
        log("LoRA adapter tidak ada! Train dulu.", "ERROR"); return False

    log("Loading base model untuk merge...")
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="cpu",      # CPU untuk merge (tidak butuh VRAM)
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    log("Merging LoRA weights...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(adapter_path))
    tok.save_pretrained(str(MERGED_DIR))
    log(f"Merged model saved: {MERGED_DIR}", "OK")
    return True


def convert_to_gguf():
    """Convert merged model ke GGUF INT4 (Q4_K_M)."""
    llama_cpp = PROJECT_DIR / "moko_core" / "moko_inference" / "llama_bin"
    convert_script = llama_cpp / "convert-hf-to-gguf.py"
    quantize_bin   = llama_cpp / "llama-quantize"

    if not convert_script.exists():
        log(f"convert script tidak ada: {convert_script}", "ERROR")
        return False

    # Step 1: Convert ke GGUF F16
    f16_path = PROJECT_DIR / "MOKO-AI-7B-Coder-F16.gguf"
    log("Converting ke GGUF F16...")
    r = subprocess.run(
        ["python3", str(convert_script), str(MERGED_DIR), "--outfile", str(f16_path)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        log(f"Convert error: {r.stderr}", "ERROR"); return False

    # Step 2: Quantize F16 -> Q4_K_M
    if quantize_bin.exists():
        log("Quantizing ke Q4_K_M...")
        r = subprocess.run(
            [str(quantize_bin), str(f16_path), str(FINAL_GGUF), "Q4_K_M"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            log(f"GGUF INT4 siap: {FINAL_GGUF}", "OK")
            size_gb = FINAL_GGUF.stat().st_size / 1e9
            log(f"Ukuran: {size_gb:.2f} GB")
        else:
            log(f"Quantize error: {r.stderr}", "WARN")
            log(f"F16 GGUF tersedia: {f16_path}")
    else:
        log("llama-quantize tidak ditemukan, F16 GGUF saja.", "WARN")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOKO Trainer V3 — 7B QLoRA")
    parser.add_argument("--check",   action="store_true", help="Cek hardware & deps")
    parser.add_argument("--prepare", action="store_true", help="Download base model 7B")
    parser.add_argument("--train",   action="store_true", help="Jalankan training")
    parser.add_argument("--merge",   action="store_true", help="Merge LoRA ke base model")
    parser.add_argument("--gguf",    action="store_true", help="Convert ke GGUF INT4")
    parser.add_argument("--epochs",  type=int,            help="Override jumlah epochs")
    parser.add_argument("--seq",     type=int,            help="Override max_seq_length")
    parser.add_argument("--base",    type=str,            choices=list(BASE_MODELS.keys()),
                        help="Pilih base model")
    args = parser.parse_args()

    if args.base:
        SELECTED_BASE = args.base

    if args.check:
        import torch
        has_gpu, vram = check_hardware()
        log(f"CUDA: {torch.version.cuda}")
        log(f"PyTorch: {torch.__version__}")
        for pkg in ["transformers","peft","datasets","bitsandbytes","trl"]:
            try:
                __import__(pkg); log(f"  {pkg}: OK", "OK")
            except ImportError:
                log(f"  {pkg}: MISSING", "WARN")

    elif args.prepare:
        prepare_base_model()

    elif args.train:
        ok, _ = check_hardware()
        if not ok:
            log("Tidak ada GPU — training akan sangat lambat!", "WARN")
        train(epochs=args.epochs, seq_len=args.seq)

    elif args.merge:
        merge_lora()

    elif args.gguf:
        convert_to_gguf()

    else:
        parser.print_help()
        print("\nContoh alur lengkap:")
        print("  python3 moko_trainer_v3_7b.py --check")
        print("  python3 moko_trainer_v3_7b.py --prepare")
        print("  python3 moko_trainer_v3_7b.py --train --epochs 3 --seq 512")
        print("  python3 moko_trainer_v3_7b.py --merge")
        print("  python3 moko_trainer_v3_7b.py --gguf")
