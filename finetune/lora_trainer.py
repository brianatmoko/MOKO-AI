"""
MOKO LoRA Trainer
==================
Fine-tuning Qwen3.5-4B Uncensored dengan data MOKO menggunakan LoRA.
Dirancang untuk RTX 2050 4GB VRAM menggunakan QLoRA (4-bit quantization).

Dipanggil otomatis oleh sleep_consolidation.py saat ada data baru.
"""
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path

# ── Path Config ───────────────────────────────────────────────────────────────
FINETUNE_DIR    = Path(__file__).parent
PROJECT_DIR     = FINETUNE_DIR.parent
BASE_MODEL_DIR  = FINETUNE_DIR / "base_model"
ADAPTER_DIR     = FINETUNE_DIR / "adapters"
DATASET_FILE    = FINETUNE_DIR / "moko_datasets" / "moko_coder_hex.jsonl"
LOG_FILE        = FINETUNE_DIR / "training.log"
STATUS_FILE     = FINETUNE_DIR / ".training_status.json"

# ── Training Config ───────────────────────────────────────────────────────────
LORA_CONFIG = {
    "r": 16,                    # LoRA rank — lebih kecil = lebih cepat, lebih besar = lebih pintar
    "lora_alpha": 32,           # LoRA alpha — biasanya 2x rank
    "target_modules": [         # Modul yang di-fine-tune
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

TRAINING_CONFIG = {
    "num_train_epochs": 1,       # 1 epoch per cycle — incremental learning
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "save_steps": 100,
    "logging_steps": 10,
    "fp16": False,
    "bf16": True,                # RTX 2050 mendukung BF16
    "optim": "muonclip",         # Kimi K2 style (stabil); fallback: "paged_adamw_8bit"
    "group_by_length": True,
}

# ── MuonClip Optimizer Config (Kimi K2 style) ─────────────────────────────────
# Menstabilkan fine-tuning dengan RMS matching & QK-Clip untuk mencegah
# logit/gradient explosion. Diteruskan ke train_lora.py via argumen CLI.
MUONCLIP_CONFIG = {
    "muon_lr": 2e-4,
    "muon_wd": 0.01,        # weight decay orto-konsisten
    "muon_alpha": 0.2,      # γ = α · RMS(W)
    "muon_qk_clip": 1.0,    # ambang norma spektral proyeksi q/k
}

VENV_PYTHON = str(PROJECT_DIR / "bin" / "python")
MIN_SAMPLES_TO_TRAIN = 50  # Minimal sampel baru sebelum fine-tuning dipicu


def get_muonclip_optimizer(model, **overrides):
    """Re-export praktis: bangun optimizer MuonClip via implementasi di train_lora.

    Memungkinkan pemakaian MuonClip langsung dari lora_trainer tanpa duplikasi
    logika. Torch diimpor lazy di dalam train_lora, jadi aman diimpor di sini.
    """
    if str(FINETUNE_DIR) not in sys.path:
        sys.path.insert(0, str(FINETUNE_DIR))
    from train_lora import create_muonclip_optimizer
    params = {**MUONCLIP_CONFIG, **overrides}
    return create_muonclip_optimizer(
        model,
        lr=params["muon_lr"],
        weight_decay=params["muon_wd"],
        rms_clip_alpha=params["muon_alpha"],
        qk_clip=params["muon_qk_clip"],
    )


def get_dataset_count() -> int:
    """Hitung total sampel di dataset."""
    if not DATASET_FILE.exists():
        return 0
    count = 0
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def update_status(status: str, details: dict = None):
    """Update file status training."""
    data = {
        "status": status,
        "timestamp": int(time.time()),
        "details": details or {}
    }
    STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[LoRA Trainer] Status: {status}")


def find_base_model() -> Path:
    """Cari file BF16 GGUF yang sudah didownload."""
    bf16_files = list(BASE_MODEL_DIR.glob("*BF16*.gguf"))
    if bf16_files:
        return bf16_files[0]
    return None


def get_latest_adapter() -> Path:
    """Dapatkan adapter LoRA terbaru."""
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    adapters = sorted(ADAPTER_DIR.glob("moko_lora_*.gguf"), reverse=True)
    return adapters[0] if adapters else None


def run_training():
    """
    Jalankan proses fine-tuning LoRA menggunakan script terpisah.
    Menggunakan subprocess untuk isolasi VRAM.
    """
    update_status("PREPARING")
    
    # 1. Build dataset (incremental)
    print("[LoRA Trainer] Membangun dataset dari .moko_crypto...")
    result = subprocess.run(
        [VENV_PYTHON, str(FINETUNE_DIR / "dataset_builder.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[LoRA Trainer] ERROR saat build dataset: {result.stderr}")
        update_status("ERROR", {"error": result.stderr[:500]})
        return False
    
    # 2. Cek jumlah dataset
    count = get_dataset_count()
    if count < MIN_SAMPLES_TO_TRAIN:
        print(f"[LoRA Trainer] Dataset terlalu sedikit ({count} < {MIN_SAMPLES_TO_TRAIN}). Skip.")
        update_status("SKIPPED", {"reason": f"Only {count} samples, need {MIN_SAMPLES_TO_TRAIN}"})
        return False
    
    # 3. Cek base model tersedia
    base_model = find_base_model()
    if not base_model:
        print("[LoRA Trainer] ERROR: Base model BF16 GGUF tidak ditemukan di finetune/base_model/")
        print("[LoRA Trainer] Download sedang berlangsung atau belum selesai.")
        update_status("ERROR", {"error": "Base model not found - download may be in progress"})
        return False
    
    print(f"[LoRA Trainer] Base model: {base_model}")
    print(f"[LoRA Trainer] Dataset: {count} sampel")
    
    update_status("TRAINING", {"samples": count, "base_model": str(base_model)})
    
    # 4. Jalankan training script (dengan optimizer MuonClip jika dikonfigurasi)
    train_script = FINETUNE_DIR / "train_lora.py"
    train_cmd = [
        VENV_PYTHON, str(train_script),
        "--base", str(base_model),
        "--dataset", str(DATASET_FILE),
        "--output", str(ADAPTER_DIR),
        "--optim", str(TRAINING_CONFIG.get("optim", "paged_adamw_8bit")),
    ]
    if TRAINING_CONFIG.get("optim") == "muonclip":
        train_cmd += [
            "--muon-lr", str(MUONCLIP_CONFIG["muon_lr"]),
            "--muon-wd", str(MUONCLIP_CONFIG["muon_wd"]),
            "--muon-alpha", str(MUONCLIP_CONFIG["muon_alpha"]),
            "--muon-qk-clip", str(MUONCLIP_CONFIG["muon_qk_clip"]),
        ]
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\n=== Training Run {int(time.time())} ===\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("STDERR:\n" + result.stderr)
    
    if result.returncode != 0:
        print(f"[LoRA Trainer] Training gagal. Lihat {LOG_FILE}")
        update_status("ERROR", {"error": result.stderr[-500:]})
        return False
    
    # 5. Hot-swap LoRA di llama-server
    latest_adapter = get_latest_adapter()
    if latest_adapter:
        result = hotswap_lora(latest_adapter)
        if result:
            update_status("COMPLETE", {
                "adapter": str(latest_adapter),
                "samples_trained": count
            })
            print(f"[LoRA Trainer] ✅ Fine-tuning selesai! Adapter aktif: {latest_adapter.name}")
            return True
    
    update_status("COMPLETE_NO_SWAP", {"adapter": str(latest_adapter)})
    return True


def hotswap_lora(adapter_path: Path) -> bool:
    """
    Hot-swap LoRA adapter di llama-server yang sedang berjalan.
    llama-server mendukung endpoint /lora-adapters untuk runtime update.
    """
    import urllib.request
    import urllib.error
    
    llama_url = "http://127.0.0.1:11435"
    
    try:
        # Coba gunakan API llama-server untuk update LoRA
        data = json.dumps([{
            "id": 0,
            "path": str(adapter_path),
            "scale": 1.0
        }]).encode("utf-8")
        
        req = urllib.request.Request(
            f"{llama_url}/lora-adapters",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[LoRA Trainer] Hot-swap berhasil! Status: {resp.status}")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"[LoRA Trainer] Hot-swap HTTP error: {e.code} — {e.reason}")
    except Exception as e:
        print(f"[LoRA Trainer] Hot-swap gagal: {e}")
        print("[LoRA Trainer] Restart llama-server manual dengan --lora flag untuk mengaktifkan adapter.")
    
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Paksa training meski dataset kecil")
    args = parser.parse_args()
    
    if args.force:
        MIN_SAMPLES_TO_TRAIN = 1
    
    success = run_training()
    sys.exit(0 if success else 1)
