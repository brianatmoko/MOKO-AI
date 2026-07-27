"""
MOKO Coder Fine-Tuning Script
===============================
Fine-tuning Qwen2.5-1.5B-Instruct → MOKO Coder (100% coding focus).

Base Model: Qwen2.5-1.5B-Instruct (HF format, 3GB)
Output:     MOKO Coder LoRA adapter → GGUF untuk llama-server

Hardware: RTX 2050 4GB VRAM, 16GB RAM
Usage:
  python3 moko_finetune.py --prepare    # Download base model HF
  python3 moko_finetune.py --build      # Build coding dataset
  python3 moko_finetune.py --train      # Start fine-tuning
  python3 moko_finetune.py --status     # Show status
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
MOKO_CORE_DIR = PROJECT_DIR / "moko_core"
OUTPUT_DIR = FINETUNE_DIR / "moko_adapters"
DATASET_DIR = FINETUNE_DIR / "moko_datasets"
LOG_DIR = FINETUNE_DIR / "logs"

# Add project lib to Python path
_site_packages = PROJECT_DIR / "lib" / "python3.12" / "site-packages"
if _site_packages.exists() and str(_site_packages) not in sys.path:
    sys.path.insert(0, str(_site_packages))

# Base model: Qwen2.5-1.5B-Instruct (HuggingFace format untuk training)
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL_DIR = FINETUNE_DIR / "base_model_hf"

# GGUF model untuk inference (hasil akhir)
GGUF_MODEL = PROJECT_DIR / "Qwen2.5-1.5B-Instruct-uncensored.f16.gguf"


# ═══════════════════════════════════════════════════════════════════════════
# MOKO CODER IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

MOKO_CODER_SYSTEM = """You are MOKO Coder v3 (Hex Mode), an expert AI programming assistant built for MOKO IDE.

IDENTITY:
- Name: MOKO Coder
- Version: 3.0.0 (Hex Binary Mode)
- Built for: MOKO IDE (AI-Powered Development Environment)
- Base model: Qwen2.5-1.5B
- Motto: "Kode yang efisien, solusi yang cerdas."

OUTPUT FORMAT — HEX BINARY MODE:
You output code as hex-encoded compressed binary (lowercase hex string).
Pipeline: source code → UTF-8 bytes → zlib compress → hex string
To decode: bytes.fromhex(hex) → zlib.decompress() → decode('utf-8')

RULES:
1. ALWAYS output COMPLETE hex strings — never truncate
2. NEVER output natural language code — ONLY hex strings
3. Each hex string represents ONE complete code solution
4. The hex string must decode to valid, runnable code
5. Use zlib compression level 9 for maximum compression
6. Output lowercase hex characters only (0-9, a-f)

EXAMPLE:
User: "Write binary search in Python"
Assistant: "789c0bca4cc8b12f2a2d4e2e..." (hex of compressed code)

SUPPORTED LANGUAGES (encoded as hex):
- Python, JavaScript, TypeScript, HTML/CSS, SQL
- Bash/Shell, JSON/YAML, Markdown
- C, C++, Java, Go, Rust (basic)"""


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 256,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "save_steps": 50,
    "logging_steps": 1,
    "fp16": False,
    "bf16": True,
    "optim": "paged_adamw_8bit",
    "group_by_length": False,
    "report_to": "none",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log(f"GPU: {name} ({vram:.1f} GB VRAM)")
            return True, name, vram
        else:
            log("No GPU detected", "WARNING")
            return False, None, 0
    except ImportError:
        log("PyTorch not installed", "ERROR")
        return False, None, 0


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPARE BASE MODEL
# ═══════════════════════════════════════════════════════════════════════════

def prepare_base_model() -> bool:
    """Download Qwen2.5-1.5B-Instruct dari HuggingFace Hub."""
    log("Preparing base model: Qwen2.5-1.5B-Instruct")
    
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cek apakah sudah ada
    if (BASE_MODEL_DIR / "config.json").exists():
        log(f"Base model already exists at {BASE_MODEL_DIR}")
        return True
    
    log(f"Downloading {BASE_MODEL_ID} from HuggingFace Hub...")
    
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.bin", "*.onnx", "*.h5"],
        )
        log("Base model downloaded successfully")
        
        # Verify
        config_file = BASE_MODEL_DIR / "config.json"
        if config_file.exists():
            log("Base model verified: config.json found")
            return True
        else:
            log("Download may be incomplete", "WARNING")
            return False
            
    except ImportError:
        log("huggingface_hub not installed", "ERROR")
        return False
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD CODING DATASET
# ═══════════════════════════════════════════════════════════════════════════

def build_coding_dataset(max_samples: int = 5000) -> Optional[Path]:
    """Build dataset coding-only dalam format HEX (binary compressed)."""
    log("Building HEX-encoded coding dataset for MOKO Coder v3")
    
    # Import hex encoder
    sys.path.insert(0, str(FINETUNE_DIR))
    from moko_hex_encoder import MokoEncoder, MokoDatasetConverter
    
    encoder = MokoEncoder()
    converter = MokoDatasetConverter()
    
    # Load dari hex dataset yang sudah di-convert
    hex_data_file = DATASET_DIR / "moko_coder_hex.jsonl"
    raw_data_file = DATASET_DIR / "moko_coder_dataset.jsonl"
    
    samples = []
    
    # Prioritas: load hex format dulu
    if hex_data_file.exists():
        log(f"Loading HEX dataset: {hex_data_file.name}...")
        with open(hex_data_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        if sample.get("messages"):
                            samples.append(sample)
                            if len(samples) >= max_samples:
                                break
                    except:
                        pass
        log(f"Loaded {len(samples)} hex samples")
    
    # Fallback: convert dari raw dataset
    if not samples and raw_data_file.exists():
        log(f"Converting raw dataset to HEX format...")
        with open(raw_data_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        converted = converter.convert_sample(sample)
                        samples.append(converted)
                        if len(samples) >= max_samples:
                            break
                    except:
                        pass
        log(f"Converted {len(samples)} samples to HEX")
    
    # Limit
    samples = samples[:max_samples]
    
    if not samples:
        log("No samples found", "ERROR")
        return None
    
    # Save
    dataset_path = DATASET_DIR / "moko_coder_hex.jsonl"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    log(f"HEX Dataset saved: {dataset_path} ({len(samples)} samples)")
    return dataset_path


def _generate_coding_samples() -> list:
    """Generate synthetic coding instruction samples."""
    templates = [
        {
            "user": "Buat fungsi Python untuk menghitung factorial",
            "assistant": """```python
def factorial(n: int) -> int:
    \"\"\"Hitung factorial dari n secara rekursif.\"\"\"
    if n < 0:
        raise ValueError("n harus non-negatif")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```"""
        },
        {
            "user": "Buat fungsi binary search dalam Python",
            "assistant": """```python
from typing import List, Optional

def binary_search(arr: List[int], target: int) -> Optional[int]:
    \"\"\"Cari target dalam sorted array. Return index atau None.\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None
```"""
        },
        {
            "user": "Buatin program CRUD sederhana dalam Python dengan SQLite",
            "assistant": """```python
import sqlite3
from typing import List, Dict, Any

class Database:
    def __init__(self, db_name: str = "app.db"):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def create_table(self, table: str, columns: Dict[str, str]):
        cols = ", ".join(f"{k} {v}" for k, v in columns.items())
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
        self.conn.commit()
    
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        placeholders = ", ".join(["?"] * len(data))
        cols = ", ".join(data.keys())
        self.cursor.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", list(data.values()))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def select_all(self, table: str) -> List[Dict]:
        self.cursor.execute(f"SELECT * FROM {table}")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def update(self, table: str, data: Dict[str, Any], where: str, params: tuple):
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        self.cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {where}", list(data.values()) + list(params))
        self.conn.commit()
    
    def delete(self, table: str, where: str, params: tuple):
        self.cursor.execute(f"DELETE FROM {table} WHERE {where}", params)
        self.conn.commit()
    
    def close(self):
        self.conn.close()
```"""
        },
        {
            "user": "Buat fungsi untuk validasi email dalam Python",
            "assistant": """```python
import re

def is_valid_email(email: str) -> bool:
    \"\"\"Validasi format email menggunakan regex.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

# Atau dengan lebih detail:
from dataclasses import dataclass

@dataclass
class EmailValidation:
    valid: bool
    error: str = ""
    
def validate_email(email: str) -> EmailValidation:
    if not email:
        return EmailValidation(False, "Email kosong")
    if "@" not in email:
        return EmailValidation(False, "Tidak ada @")
    parts = email.split("@")
    if len(parts) != 2:
        return EmailValidation(False, "Format invalid")
    if not parts[0]:
        return EmailValidation(False, "Username kosong")
    if "." not in parts[1]:
        return EmailValidation(False, "Domain invalid")
    return EmailValidation(True)
```"""
        },
        {
            "user": "Buat REST API sederhana dengan FastAPI",
            "assistant": """```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="MOKO API", version="1.0.0")

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(Item):
    id: int

# In-memory storage
items_db: List[ItemResponse] = []
counter = 0

@app.get("/items", response_model=List[ItemResponse])
def list_items():
    return items_db

@app.post("/items", response_model=ItemResponse)
def create_item(item: Item):
    global counter
    counter += 1
    new_item = ItemResponse(id=counter, **item.model_dump())
    items_db.append(new_item)
    return new_item

@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    global items_db
    items_db = [i for i in items_db if i.id != item_id]
    return {"status": "deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```"""
        },
    ]
    
    samples = []
    for t in templates:
        samples.append({
            "messages": [
                {"role": "system", "content": MOKO_CODER_SYSTEM},
                {"role": "user", "content": t["user"]},
                {"role": "assistant", "content": t["assistant"]}
            ]
        })
    
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: FINE-TUNE
# ═══════════════════════════════════════════════════════════════════════════

def train_moko_coder(epochs: int = 3, max_samples: int = 5000) -> bool:
    """Fine-tune Qwen2.5-1.5B-Instruct → MOKO Coder."""
    log("Starting MOKO Coder fine-tuning")
    
    # 1. Check base model
    if not BASE_MODEL_DIR.exists() or not (BASE_MODEL_DIR / "config.json").exists():
        log("Base model not found. Run --prepare first.", "ERROR")
        return False
    
    # 2. Build dataset (HEX format)
    dataset_path = DATASET_DIR / "moko_coder_hex.jsonl"
    if not dataset_path.exists():
        log("HEX dataset not found. Building...")
        dataset_path = build_coding_dataset(max_samples)
        if not dataset_path:
            return False
    
    # 3. Check GPU
    gpu_ok, gpu_name, vram = check_gpu()
    if not gpu_ok:
        log("GPU required for training", "ERROR")
        return False
    
    if vram < 4.0:
        log(f"VRAM low ({vram:.1f}GB). Training may be slow.", "WARNING")
    
    # 4. Prepare output
    adapter_dir = OUTPUT_DIR / "moko_coder"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = adapter_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "base_model": BASE_MODEL_ID,
            "lora": LORA_CONFIG,
            "training": TRAINING_CONFIG,
            "epochs": epochs,
        }, f, indent=2)
    
    # 5. Train
    log(f"Training MOKO Coder: {epochs} epochs")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType
        from datasets import Dataset
        from trl import SFTTrainer
        
        # Load tokenizer
        log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(BASE_MODEL_DIR),
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with QLoRA (4-bit) for 4GB VRAM
        log("Loading Qwen2.5-1.5B-Instruct with QLoRA (4-bit)...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_DIR),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Add LoRA
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
        
        # Load dataset (limit to max_samples for VRAM safety)
        log(f"Loading dataset (max {max_samples} samples)...")
        samples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                        if len(samples) >= max_samples:
                            break
                    except:
                        pass
        
        log(f"Dataset: {len(samples)} samples loaded")
        
        # Format to ChatML
        def format_sample(sample):
            messages = sample.get("messages", [])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return text
        
        dataset = Dataset.from_list(samples)
        
        # Training arguments using SFTConfig
        from trl import SFTConfig
        total_steps = (len(dataset) // TRAINING_CONFIG["per_device_train_batch_size"]) // TRAINING_CONFIG["gradient_accumulation_steps"]
        warmup_steps = int(total_steps * TRAINING_CONFIG["warmup_ratio"])
        training_args = SFTConfig(
            output_dir=str(adapter_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            warmup_steps=warmup_steps,
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            bf16=TRAINING_CONFIG["bf16"],
            fp16=TRAINING_CONFIG["fp16"],
            optim=TRAINING_CONFIG["optim"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_strategy="epoch",
            report_to=TRAINING_CONFIG["report_to"],
            max_length=TRAINING_CONFIG["max_seq_length"],
            dataset_text_field="text",
            packing=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=format_sample,
            args=training_args,
        )
        
        # Train
        log("=" * 60)
        log("TRAINING STARTED")
        log("=" * 60)
        
        checkpoint_dir = adapter_dir / "checkpoints"
        latest_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = sorted(list(checkpoint_dir.glob("checkpoint-*")), key=lambda x: int(x.name.split("-")[-1]))
            if checkpoints:
                latest_checkpoint = str(checkpoints[-1])
                log(f"Resuming from checkpoint: {latest_checkpoint}")

        trainer.train(resume_from_checkpoint=latest_checkpoint)
        
        # Save adapter
        adapter_save_path = adapter_dir / "lora_adapter"
        model.save_pretrained(str(adapter_save_path))
        tokenizer.save_pretrained(str(adapter_save_path))
        
        log(f"Adapter saved: {adapter_save_path}")
        
        # Convert to GGUF
        gguf_output = adapter_dir / "moko_coder.gguf"
        convert_lora_to_gguf(adapter_save_path, GGUF_MODEL, gguf_output)
        
        # Save status
        status = {
            "status": "complete",
            "adapter": str(adapter_save_path),
            "gguf": str(gguf_output) if gguf_output.exists() else None,
            "samples": len(samples),
            "epochs": epochs,
            "timestamp": int(time.time()),
        }
        (adapter_dir / "status.json").write_text(json.dumps(status, indent=2))
        
        log("=" * 60)
        log("MOKO Coder fine-tuning COMPLETE!")
        log("=" * 60)
        return True
        
    except ImportError as e:
        log(f"Missing library: {e}", "ERROR")
        log("Install: pip install torch transformers peft datasets trl", "INFO")
        return False
    except Exception as e:
        log(f"Training failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def convert_lora_to_gguf(adapter_dir: Path, base_model: Path, output_gguf: Path) -> bool:
    """Convert LoRA adapter ke GGUF."""
    log("Converting LoRA adapter to GGUF...")
    
    if not base_model.exists():
        log(f"Base GGUF not found: {base_model}", "WARNING")
        log("Adapter saved in HuggingFace format only", "INFO")
        return False
    
    llama_bin = MOKO_CORE_DIR / "moko_inference" / "llama_bin"
    export_lora = llama_bin / "llama-export-lora"
    
    if not export_lora.exists():
        log(f"llama-export-lora not found at {llama_bin}", "WARNING")
        log("Adapter saved in HuggingFace format", "INFO")
        return False
    
    result = subprocess.run(
        [str(export_lora),
         "--model-base", str(base_model),
         "--lora", str(adapter_dir),
         "--output", str(output_gguf)],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode == 0:
        log(f"LoRA GGUF created: {output_gguf}")
        return True
    else:
        log(f"GGUF conversion failed: {result.stderr[:300]}", "WARNING")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════

def show_status():
    """Tampilkan status MOKO Coder fine-tuning."""
    print("\n" + "=" * 50)
    print("  MOKO CODER - Fine-Tuning Status")
    print("=" * 50)
    
    # Base model (HF)
    print("\n📦 Base Model (HuggingFace):")
    if BASE_MODEL_DIR.exists() and (BASE_MODEL_DIR / "config.json").exists():
        config = json.loads((BASE_MODEL_DIR / "config.json").read_text())
        size_mb = sum(f.stat().st_size for f in BASE_MODEL_DIR.glob("*") if f.is_file()) / 1024 / 1024
        print(f"  ✅ {BASE_MODEL_ID}")
        print(f"  Path: {BASE_MODEL_DIR}")
        print(f"  Size: {size_mb:.0f} MB")
    else:
        print(f"  ❌ Not downloaded yet")
        print(f"  Run: python3 moko_finetune.py --prepare")
    
    # Base model (GGUF)
    print("\n📦 Base Model (GGUF):")
    if GGUF_MODEL.exists():
        size_gb = GGUF_MODEL.stat().st_size / 1024**3
        print(f"  ✅ {GGUF_MODEL.name} ({size_gb:.1f} GB)")
    else:
        print(f"  ❌ {GGUF_MODEL.name} not found")
    
    # Dataset
    print("\n📊 Dataset:")
    hex_file = DATASET_DIR / "moko_coder_hex.jsonl"
    raw_file = DATASET_DIR / "moko_coder_dataset.jsonl"
    
    if hex_file.exists():
        count = sum(1 for _ in open(hex_file))
        size_mb = hex_file.stat().st_size / 1024 / 1024
        print(f"  ✅ HEX: {count:,} samples ({size_mb:.1f} MB) — TRAINING FORMAT")
    if raw_file.exists():
        count = sum(1 for _ in open(raw_file))
        size_mb = raw_file.stat().st_size / 1024 / 1024
        print(f"  ✅ RAW: {count:,} samples ({size_mb:.1f} MB) — source")
    if not hex_file.exists() and not raw_file.exists():
        print(f"  ❌ Not built yet")
        print(f"  Run: python3 moko_finetune.py --build")
    
    # Adapter
    print("\n🔧 LoRA Adapter:")
    adapter_dir = OUTPUT_DIR / "moko_coder"
    if adapter_dir.exists() and (adapter_dir / "lora_adapter").exists():
        status_file = adapter_dir / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
            print(f"  ✅ MOKO Coder adapter trained")
            print(f"  Samples: {status.get('samples', '?')}")
            print(f"  Epochs: {status.get('epochs', '?')}")
        
        gguf_file = adapter_dir / "moko_coder.gguf"
        if gguf_file.exists():
            size_mb = gguf_file.stat().st_size / 1024 / 1024
            print(f"  GGUF: {size_mb:.0f} MB")
        else:
            print(f"  GGUF: ❌ Not converted")
    else:
        print(f"  ❌ Not trained yet")
        print(f"  Run: python3 moko_finetune.py --train")
    
    # GPU
    print("\n💻 GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = torch.cuda.mem_get_info(0)[0] / 1024**3
            print(f"  ✅ {name}")
            print(f"  VRAM: {vram:.1f} GB total, {free:.1f} GB free")
        else:
            print(f"  ❌ No GPU")
    except:
        print(f"  ❌ PyTorch not installed")
    
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MOKO Coder Fine-Tuning")
    parser.add_argument("--prepare", action="store_true",
                        help="Download base model dari HuggingFace")
    parser.add_argument("--build", action="store_true",
                        help="Build coding dataset")
    parser.add_argument("--train", action="store_true",
                        help="Start fine-tuning")
    parser.add_argument("--status", action="store_true",
                        help="Show status")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max training samples")
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    print("\n" + "=" * 60)
    print("  MOKO CODER Fine-Tuning Pipeline")
    print("  Base: Qwen2.5-1.5B-Instruct → MOKO Coder")
    print("=" * 60 + "\n")
    
    if args.status:
        show_status()
        return
    
    if args.prepare:
        success = prepare_base_model()
        if success:
            print("\n✅ Base model ready!")
        else:
            print("\n❌ Failed to prepare base model")
        return
    
    if args.build:
        path = build_coding_dataset(args.max_samples)
        if path:
            print(f"\n✅ Dataset ready: {path}")
        else:
            print("\n❌ Failed to build dataset")
        return
    
    if args.train:
        success = train_moko_coder(args.epochs, args.max_samples)
        if success:
            print("\n✅ MOKO Coder training complete!")
        else:
            print("\n❌ Training failed")
        return
    
    # Default: show status
    show_status()


if __name__ == "__main__":
    main()
