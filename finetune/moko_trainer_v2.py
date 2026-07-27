"""
MOKO Trainer V2 — Advanced Training Pipeline
==============================================
Fine-tuning dari base model uncensored (abliterated) → MOKO Coder 1B.

Arsitektur:
  Base Model : huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
               (Qwen2.5-Coder-1.5B dengan refusal mechanism dihapus)
  Method     : QLoRA 4-bit NF4 + PEFT LoRA adapter
  Output     : moko-coder-1b.gguf (Q4_K_M quantized, ~1GB)

Fitur Utama:
  1. Multi-source Dataset Merge (Distill + OS Code + Public Coder)
  2. Curriculum Learning (Urutkan dari mudah/pendek → kompleks/panjang)
  3. Loss Masking (Hanya latih pada output Assistant, abaikan System/User)
  4. QLoRA 4-bit NF4 optimized untuk GPU RTX 2050 4GB VRAM

Usage:
  python3 moko_trainer_v2.py --prepare        # Download base model uncensored
  python3 moko_trainer_v2.py --train          # Mulai training QLoRA
  python3 moko_trainer_v2.py --epochs 3       # Set epochs (default: 3)
  python3 moko_trainer_v2.py --gguf           # Merge LoRA + Convert ke GGUF"""

import os
import sys
import json
import time
import argparse
import subprocess
import random
from pathlib import Path
from typing import List, Dict, Optional

# Tambah project lib ke python path sebelum import torch
PROJECT_DIR = Path(__file__).parent.parent
_site_packages = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

import torch

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR = FINETUNE_DIR / "moko_datasets"
OUTPUT_DIR = FINETUNE_DIR / "moko_adapters"
LOG_DIR = FINETUNE_DIR / "logs"

# Tambah project lib ke python path jika ada
_site_packages = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

# Base model: Qwen2.5-Coder-1.5B-Instruct ABLITERATED (uncensored, HF format)
# Refusal mechanism sudah dihapus via abliteration oleh huihui-ai
# Kita fine-tune di atas model ini agar MOKO Coder 1B tidak punya sensor bawaan
BASE_MODEL_ID = "huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated"
BASE_MODEL_DIR = FINETUNE_DIR / "base_model_coder_hf"

# Target output adapter path
ADAPTER_SAVE_PATH = OUTPUT_DIR / "moko_coder_1b"

# Target GGUF model output
FINAL_GGUF_PATH = PROJECT_DIR / "moko-coder-1b.gguf"


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ── LoRA Config: Diupgrade dari r=16 → r=64 untuk deep fine-tuning ────────────
# r=64 memberi model kapasitas belajar 4x lebih besar dibanding r=16.
# Untuk 1.5B, ini aman secara VRAM karena model kecil + 4-bit NF4.
LORA_CONFIG = {
    "r": 64,           # Naik dari 16 → 64: learning capacity 4x lebih dalam
    "lora_alpha": 128, # Alpha = 2x rank (standar optimal)
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ── Training Config: Deep training untuk 1.5B model ───────────────────────────
# Dengan 7,600+ sampel dan 5 epoch → ~38K gradient steps.
# max_seq_length=2048 agar model bisa pelajari kode yang lebih panjang.
TRAINING_CONFIG = {
    "learning_rate": 1e-4,        # Lebih stabil untuk training dalam (2e-4 terlalu cepat)
    "max_seq_length": 2048,       # Naik dari 1536 → 2048: handle kode lebih panjang
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,  # Naik dari 8 → 16: effective batch = 16
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": True,           # RTX 2050 dukung BF16
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "save_total_limit": 2,  # Simpan hanya 2 checkpoint terbaru
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU Terdeteksi: {name} ({vram:.1f} GB VRAM)")
        return True, vram
    log("Tidak ada GPU NVIDIA terdeteksi!", "ERROR")
    return False, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPARE BASE MODEL CODER
# ═══════════════════════════════════════════════════════════════════════════

def prepare_base_model() -> bool:
    """
    Download base model uncensored (abliterated) dari HuggingFace Hub.
    
    Model: huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
    Format: HuggingFace safetensors (untuk fine-tuning QLoRA)
    Ukuran: ~3.1 GB
    
    Untuk download manual:
      Buka: https://huggingface.co/huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated
      Download file: model.safetensors + config.json + tokenizer files
      Simpan ke: finetune/base_model_coder_hf/
    """
    log(f"Mempersiapkan base model uncensored: {BASE_MODEL_ID}")
    
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cek apakah model weights sudah ada dan lengkap (> 2GB)
    weight_file = BASE_MODEL_DIR / "model.safetensors"
    if (BASE_MODEL_DIR / "config.json").exists() and weight_file.exists() and weight_file.stat().st_size > 2 * 1024 * 1024 * 1024:
        log(f"✅ Base model sudah ada dan lengkap di {BASE_MODEL_DIR}")
        return True
    
    if (BASE_MODEL_DIR / "config.json").exists() and weight_file.exists():
        size_mb = weight_file.stat().st_size / 1024**2
        log(f"⚠️  model.safetensors ditemukan tapi ukurannya hanya {size_mb:.0f}MB (belum lengkap)", "WARNING")
        log("    Download ulang atau lanjutkan download yang terpotong.", "WARNING")
        
    log(f"Mendownload {BASE_MODEL_ID} ke {BASE_MODEL_DIR}...")
    log("Untuk download manual, kunjungi:")
    log("  https://huggingface.co/huihui-ai/Qwen2.5-Coder-1.5B-Instruct-abliterated")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.bin", "*.onnx", "*.h5", "*.ot"],
        )
        log("✅ Base model berhasil di-download!")
        return True
    except Exception as e:
        log(f"Gagal download model: {e}", "ERROR")
        log("Coba download manual ke finetune/base_model_coder_hf/", "INFO")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD & MERGE DATASETS (CURRICULUM LEARNING)
# ═══════════════════════════════════════════════════════════════════════════

def load_all_datasets(max_samples: int = 7500) -> List[Dict]:
    """Menggabungkan dataset secara dinamis dari folder moko_datasets."""
    samples = []
    
    try:
        from moko_agents.dual_system.dataset_compressor import get_compressor
        comp = get_compressor()
    except Exception:
        comp = None

    # Cari semua file dataset di DATASET_DIR
    dataset_files = []
    for ext in ("*.jsonl", "*.jsonl.zst"):
        for path in DATASET_DIR.glob(ext):
            # Skip file hex (karena versi raw moko_coder_dataset.jsonl sudah ada)
            # Skip file merged sementara
            if "hex" in path.name or "merged" in path.name:
                continue
            dataset_files.append(path)
            
    # Urutkan file agar proses loading deterministik
    dataset_files.sort()
    
    # Track base names untuk menghindari loading file yang sama (misal raw + compressed)
    loaded_bases = set()
    
    for path in dataset_files:
        base_name = path.name
        if base_name.endswith(".zst"):
            base_name = base_name[:-4]
            
        if base_name in loaded_bases:
            continue
        loaded_bases.add(base_name)
        
        count = 0
        log(f"Memuat dataset: {path.name}...")
        try:
            if comp and (".zst" in path.suffixes or ".gz" in path.suffixes):
                for sample in comp.stream_read(path):
                    samples.append(sample)
                    count += 1
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                            count += 1
            log(f"  ✅ Memuat {count} samples")
        except Exception as e:
            log(f"  ❌ Gagal membaca {path.name}: {e}", "ERROR")
            
    if not samples:
        log("Tidak ada training samples ditemukan!", "ERROR")
        log("Silakan jalankan moko_data_factory.py terlebih dahulu.", "INFO")
        return []
        
    # Deduplikasi berdasarkan hash user message
    unique_samples = []
    seen_hashes = set()
    for s in samples:
        msgs = s.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        msg_hash = hashlib_str(user_msg)
        if msg_hash not in seen_hashes:
            seen_hashes.add(msg_hash)
            unique_samples.append(s)
            
    log(f"Total samples unik: {len(unique_samples)} (dari total {len(samples)} sebelum dedup)")
    
    # CURRICULUM LEARNING:
    # Urutkan dataset berdasarkan panjang total konten (user prompt + assistant response).
    # Model belajar kode-kode pendek/konsep dasar dulu sebelum masuk ke kode panjang/kompleks.
    log("Mengurutkan dataset berdasarkan panjang karakter (Curriculum Sorting)...")
    unique_samples.sort(key=get_sample_length)
    
    # Limit to max_samples
    if len(unique_samples) > max_samples:
        log(f"Membatasi dataset ke {max_samples} samples terpendek/terkompresi")
        unique_samples = unique_samples[:max_samples]
        
    return unique_samples


def hashlib_str(text: str) -> str:
    import hashlib
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def get_sample_length(sample: Dict) -> int:
    """Hitung total panjang teks dalam sample untuk curriculum sorting."""
    total_len = 0
    for m in sample.get("messages", []):
        total_len += len(m.get("content", ""))
    return total_len


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def train(epochs: int = 5, max_samples: int = 7500):
    """Jalankan proses training QLoRA."""
    log("===== Memulai Proses Training MOKO Coder 1B =====")
    
    # 1. Check GPU & RAM
    gpu_ok, vram = check_gpu()
    if not gpu_ok:
        return
        
    # 2. Prepare Base Model
    if not prepare_base_model():
        return
        
    # 3. Load & Sort Datasets
    samples = load_all_datasets(max_samples)
    if not samples:
        return
        
    # 4. Initialize Tokenizer & Model
    log("Loading tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    # trl 1.7+ — DataCollatorForCompletionOnlyLM dihapus;
    # gunakan completion_only_loss=True di SFTConfig sebagai gantinya.
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    log("Loading Qwen2.5-Coder-1.5B-Instruct dengan QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if TRAINING_CONFIG["bf16"] else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Convert list ke HF Dataset
    log("Mempersiapkan tokenization dataset...")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FORMAT TRAINING: Prompt-Completion TANPA System Prompt
    # ─────────────────────────────────────────────────────────────────────────
    # KENAPA tidak pakai system prompt?
    #   1. Hemat token — system prompt 200-300 token × 1800 samples = ~500K
    #      token terbuang per epoch hanya untuk konteks boilerplate.
    #   2. Knowledge MOKO OS tertanam LANGSUNG ke bobot model via fine-tuning,
    #      sehingga model "sudah tahu" tentang MOKO OS tanpa perlu diberitahu
    #      setiap saat via prompt.
    #   3. Lebih efisien saat inference — response langsung tanpa overhead
    #      pemrosesan system prompt panjang di setiap request.
    #
    # Format yang dipakai:
    #   <|im_start|>user\n{pertanyaan}<|im_end|>\n<|im_start|>assistant\n{jawaban}<|im_end|>
    #
    # System role di-DROP saat training. Model belajar menjawab pertanyaan
    # coding/MOKO OS secara langsung berdasarkan pola yang dilihat di data.
    # ─────────────────────────────────────────────────────────────────────────
    
    def format_prompt_completion(sample):
        """Format sample ke prompt-completion tanpa system prompt."""
        messages = sample.get("messages", [])
        
        # Filter: ambil hanya user dan assistant, buang system
        filtered = []
        for m in messages:
            if m["role"] == "system":
                continue  # DROP system prompt — knowledge ada di bobot
            
            content = m.get("content", "")
            
            # Decode hex-compressed content jika ada
            if m["role"] == "assistant" and len(content) > 10:
                try:
                    if all(c in "0123456789abcdefABCDEF" for c in content[:20]):
                        import zlib
                        content = zlib.decompress(bytes.fromhex(content)).decode("utf-8")
                        m = {"role": m["role"], "content": content}
                except Exception:
                    pass
            
            filtered.append({"role": m["role"], "content": content})
        
        if not filtered:
            return {"text": ""}
        
        # Build format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        text_parts = []
        for m in filtered:
            text_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        
        # Tambah newline separator antar turn
        text = "\n".join(text_parts) + "\n"
        return {"text": text}
        
    formatted_samples = [format_prompt_completion(s) for s in samples]
    # Filter out empty samples
    formatted_samples = [s for s in formatted_samples if s["text"].strip()]
    
    log(f"Total samples setelah format: {len(formatted_samples)} (tanpa system prompt)")
    dataset = Dataset.from_list(formatted_samples)
    
    # LOSS MASKING — trl 1.7+ menggunakan completion_only_loss=True di SFTConfig
    # Ini memastikan model hanya belajar pada token jawaban assistant,
    # bukan pada token pertanyaan user. Assistant responses dimulai setelah
    # token '<|im_start|>assistant\n'.
    
    # Training Arguments
    log(f"Configuring training: {epochs} epochs, max sequence length {TRAINING_CONFIG['max_seq_length']}...")
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints_coder"),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        bf16=TRAINING_CONFIG["bf16"],
        fp16=TRAINING_CONFIG["fp16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        report_to="none",
        # trl 1.7+: max_seq_length diganti dengan max_length
        max_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=TRAINING_CONFIG["optim"],
        save_total_limit=TRAINING_CONFIG.get("save_total_limit", 2),
        # trl 1.7+ loss masking: hitung loss hanya pada token assistant response
        completion_only_loss=True,
    )

    # ── MokoMuonClip Integration (Riset 25) ────────────────────────────────────
    custom_optimizers = (None, None)
    try:
        sys.path.insert(0, str(PROJECT_DIR / "finetune"))
        from moko_optimizer import create_optimizer_groups, get_muon_lr_schedule
        
        # Hitung total training steps
        num_train_steps = (len(dataset) * epochs) // (TRAINING_CONFIG["per_device_train_batch_size"] * TRAINING_CONFIG["gradient_accumulation_steps"])
        num_warmup_steps = int(num_train_steps * TRAINING_CONFIG["warmup_ratio"])
        
        # Hanya parameter LoRA (2D linear weights) yang ditargetkan oleh Muon
        muon_opt, adamw_opt = create_optimizer_groups(
            model,
            lr_muon=TRAINING_CONFIG["learning_rate"],
            lr_adamw=TRAINING_CONFIG["learning_rate"] * 0.2,
            use_muonclip=True,
            clip_alpha=0.1
        )
        
        # Karena target parameter LoRA bias='none' (1D parameters kosong), adamw_opt bernilai None.
        # Jadi kita cukup mengoperasikan muon_opt sebagai satu-satunya optimizer.
        if muon_opt is not None:
            optimizer = muon_opt
            lr_scheduler = get_muon_lr_schedule(optimizer, num_warmup_steps, num_train_steps)
            custom_optimizers = (optimizer, lr_scheduler)
            log("✅ Custom MuonClip Optimizer & Cosine Warmup Scheduler berhasil terintegrasi!")
    except Exception as e:
        log(f"⚠️ Gagal memuat MuonClip optimizer: {e}. Fallback ke AdamW bawaan.", "WARNING")
    
    # SFT Trainer (trl 1.7+)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        optimizers=custom_optimizers,
    )
    
    log("=" * 60)
    log("🚀 FINE-TUNING MOKO CODER 1B DIMULAI")
    log("=" * 60)
    
    # Cek checkspoints terakhir
    checkpoint_dir = OUTPUT_DIR / "checkpoints_coder"
    latest_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted(list(checkpoint_dir.glob("checkpoint-*")), key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            latest_checkpoint = str(checkpoints[-1])
            log(f"Resuming dari checkpoint terakhir: {latest_checkpoint}")
            
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    # Save adapter
    ADAPTER_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_SAVE_PATH))
    tokenizer.save_pretrained(str(ADAPTER_SAVE_PATH))
    
    log(f"LoRA Adapter berhasil disimpan ke: {ADAPTER_SAVE_PATH}")
    log("=" * 60)
    log("Fine-tuning SELESAI!")
    log("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: GGUF CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_gguf() -> bool:
    """Menggabungkan base model dan LoRA adapter lalu mengonversinya ke GGUF Q4_K_M."""
    log("===== Memulai Konversi LoRA Adapter ke GGUF Q4_K_M =====")
    
    if not ADAPTER_SAVE_PATH.exists():
        log(f"LoRA Adapter tidak ditemukan di {ADAPTER_SAVE_PATH}. Silakan jalankan --train dulu.", "ERROR")
        return False
        
    merged_model_dir = OUTPUT_DIR / "moko_coder_1b_merged"
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Merge LoRA dengan Base Model ke format HuggingFace F16
    log("Menggabungkan LoRA adapter dengan base model (Merge)...")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load base model F16
        log("Loading base model (float16)...")
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="cpu",  # Gabungkan di CPU agar tidak OOM
            trust_remote_code=True,
        )
        
        # Load LoRA
        log("Loading LoRA adapter...")
        lora_model = PeftModel.from_pretrained(base, str(ADAPTER_SAVE_PATH))
        
        # Merge
        log("Merging weights...")
        merged_model = lora_model.merge_and_unload()
        
        # Save merged
        log(f"Menyimpan merged model ke {merged_model_dir}...")
        merged_model.save_pretrained(str(merged_model_dir))
        
        tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_SAVE_PATH))
        tokenizer.save_pretrained(str(merged_model_dir))
        
        # Free memory
        del base
        del lora_model
        del merged_model
        torch.cuda.empty_cache()
        log("Merge sukses!")
    except Exception as e:
        log(f"Gagal merge model: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False
        
    # 2. Mengonversi merged model HF ke GGUF
    # Kita menggunakan scripts conversion dari moko_core/moko_inference/llama_bin/ jika tersedia,
    # atau fallback ke python scripts llama.cpp.
    log("Mengonversi format HuggingFace ke GGUF...")
    
    # Cari convert_hf_to_gguf.py atau convert.py dari llama.cpp
    # LLaMA.cpp clone biasanya terpasang di sistem atau moko_core
    convert_script = None
    
    # Cari di direktori moko_core
    paths_to_search = [
        PROJECT_DIR / "llama.cpp" / "convert_hf_to_gguf.py",
        PROJECT_DIR / "llama.cpp" / "convert.py",
        PROJECT_DIR / "moko_core" / "moko_inference" / "llama_bin" / "convert_hf_to_gguf.py",
        Path("/home/brianatmokoo/llama.cpp/convert_hf_to_gguf.py"),
        Path("/home/brianatmokoo/llama.cpp/convert.py"),
    ]
    
    for p in paths_to_search:
        if p.exists():
            convert_script = p
            break
            
    if not convert_script:
        log("convert_hf_to_gguf.py dari llama.cpp tidak ditemukan!", "WARNING")
        log("Mencoba import/download script otomatis atau jalankan pip install gguf...")
        # Alternative: install gguf package and run command
        convert_script = PROJECT_DIR / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            # Download convert_hf_to_gguf.py
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
                log(f"Mendownload convert script dari GitHub...")
                urllib.request.urlretrieve(url, str(convert_script))
            except Exception as e:
                log(f"Gagal download convert script: {e}", "ERROR")
                return False

    temp_f16_gguf = OUTPUT_DIR / "moko_coder_1b_f16.gguf"
    
    # Jalankan script convert
    log(f"Menjalankan script convert: {convert_script}")
    try:
        # Install dependencies untuk conversion jika belum ada
        subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "numpy", "sentencepiece"], check=True, capture_output=True)
        
        cmd = [
            sys.executable, str(convert_script),
            str(merged_model_dir),
            "--outfile", str(temp_f16_gguf),
            "--outtype", "f16"
        ]
        
        log(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        log(f"F16 GGUF berhasil dibuat di: {temp_f16_gguf}")
    except Exception as e:
        log(f"Gagal konversi ke GGUF F16: {e}", "ERROR")
        return False
        
    # 3. Quantisasi F16 GGUF ke Q4_K_M (Sangat cepat dan ringan untuk RTX 2050 4GB)
    log("Mengonversi F16 GGUF ke Q4_K_M (Quantize)...")
    llama_quantize_bin = None
    
    # Cari file executable llama-quantize
    bin_paths = [
        PROJECT_DIR / "moko_core" / "moko_inference" / "llama_bin" / "llama-quantize",
        PROJECT_DIR / "llama.cpp" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
        Path("/usr/bin/llama-quantize"),
    ]
    
    for p in bin_paths:
        if p.exists():
            llama_quantize_bin = p
            break
            
    if not llama_quantize_bin:
        log("Executable llama-quantize tidak ditemukan di path default!", "WARNING")
        log("Mencoba menggunakan compiler g++ lokal jika source tersedia, atau cari di system PATH...")
        # fallback ke command system path
        llama_quantize_bin = "llama-quantize"

    # Jalankan quantize
    cmd = [
        str(llama_quantize_bin),
        str(temp_f16_gguf),
        str(FINAL_GGUF_PATH),
        "Q4_K_M"
    ]
    
    log(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        log(f"🎯 QUANTIZATION SUKSES! Model final: {FINAL_GGUF_PATH}")
        
        # Clean up temporary F16 GGUF & Merged HF
        if temp_f16_gguf.exists():
            temp_f16_gguf.unlink()
        # Hapus folder merged model HF untuk hemat space SSD (3.4GB)
        import shutil
        if merged_model_dir.exists():
            shutil.rmtree(merged_model_dir)
            log("Temporary merged folder cleared.")
            
        return True
    except Exception as e:
        log(f"Gagal quantize model: {e}", "ERROR")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MOKO Coder 1B Trainer V2")
    parser.add_argument("--prepare", action="store_true", help="Download base model Qwen2.5-Coder-1.5B")
    parser.add_argument("--train", action="store_true", help="Jalankan training")
    parser.add_argument("--gguf", action="store_true", help="Merge LoRA & convert ke GGUF Q4_K_M")
    parser.add_argument("--epochs", type=int, default=5, help="Jumlah epochs untuk training")
    parser.add_argument("--max-samples", type=int, default=7500, help="Maksimal samples training")
    
    args = parser.parse_args()
    
    # Buat directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if not (args.prepare or args.train or args.gguf):
        parser.print_help()
        return
        
    if args.prepare:
        prepare_base_model()
        
    if args.train:
        train(epochs=args.epochs, max_samples=args.max_samples)
        
    if args.gguf:
        convert_to_gguf()


if __name__ == "__main__":
    main()
