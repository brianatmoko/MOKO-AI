"""
MOKO LoRA Core Training Script
================================
Script training LoRA yang sesungguhnya menggunakan unsloth + PEFT.
Dipanggil oleh lora_trainer.py sebagai subprocess terpisah.

Mendukung:
  - Loading model dari BF16 GGUF via llama_cpp_python
  - QLoRA (4-bit) untuk 4GB VRAM
  - Incremental training (1 epoch per cycle)
  - Simpan adapter sebagai GGUF via llama-quantize
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# MuonClip Optimizer (Kimi K2 style) — RMS Matching + QK-Clip
# ══════════════════════════════════════════════════════════════════════════════
# Muon standar mengortogonalisasi gradien matriks (via Newton-Schulz) untuk
# konvergensi hemat-token, tetapi pada model besar memicu logit/gradient explosion
# akibat pertumbuhan nilai singular proyeksi Query/Key. MuonClip menstabilkannya
# dengan: (1) weight decay orto-konsisten, (2) RMS matching batas update terhadap
# RMS(W), dan (3) QK-Clip yang membatasi norma spektral proyeksi q_proj/k_proj.
#
# Referensi matematis (lihat docs/riset/21_KIMI_AI_RESEARCH_MASTER.md):
#   ΔW        = orthonormalize(G)
#   γ         = α · RMS(W)
#   ΔW_clip   = clamp(ΔW, −γ, γ)
#   W_{t+1}   = W_t − η · (ΔW_clip + λ_decay · W_t)
# Torch diimpor secara lazy agar berkas ini tetap dapat diimpor tanpa PyTorch.

_MUONCLIP_CLASS = None


def _build_muonclip_class():
    """Bangun kelas `MuonClip` (subclass torch.optim.Optimizer) secara lazy."""
    import torch

    def _zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
        """Ortogonalisasi matriks gradien via iterasi kuintik Newton-Schulz.

        Menghasilkan aproksimasi matriks ortogonal terdekat (semi-orthogonal),
        setara langkah inti Muon namun stabil pada bfloat16/float32.
        """
        assert G.ndim == 2, "Newton-Schulz hanya untuk matriks 2D"
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.to(torch.float32)
        X = X / (X.norm() + eps)
        transposed = False
        if X.size(0) > X.size(1):
            X = X.T
            transposed = True
        for _ in range(max(1, steps)):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X

    def _spectral_norm_approx(W, iters: int = 3, eps: float = 1e-9):
        """Aproksimasi norma spektral (nilai singular terbesar) via power iteration."""
        with torch.no_grad():
            v = torch.randn(W.size(1), device=W.device, dtype=torch.float32)
            v = v / (v.norm() + eps)
            for _ in range(max(1, iters)):
                u = W.float() @ v
                u = u / (u.norm() + eps)
                v = W.float().T @ u
                v = v / (v.norm() + eps)
            return (W.float() @ v).norm()

    class MuonClip(torch.optim.Optimizer):
        """Optimizer MuonClip: Muon + RMS matching + QK-Clip.

        Param groups dapat menandai:
          - ``use_muon`` (bool): pakai jalur ortogonalisasi Muon (untuk bobot 2D).
          - ``is_qk`` (bool): terapkan QK-Clip (untuk q_proj/k_proj).
        Bobot 1D (bias/norm) memakai jalur AdamW ringan.
        """

        def __init__(
            self,
            params,
            lr: float = 2e-4,
            momentum: float = 0.95,
            nesterov: bool = True,
            weight_decay: float = 0.01,
            rms_clip_alpha: float = 0.2,
            qk_clip: float = 1.0,
            ns_steps: int = 5,
            eps: float = 1e-8,
        ) -> None:
            defaults = dict(
                lr=lr, momentum=momentum, nesterov=nesterov,
                weight_decay=weight_decay, rms_clip_alpha=rms_clip_alpha,
                qk_clip=qk_clip, ns_steps=ns_steps, eps=eps,
                use_muon=True, is_qk=False,
            )
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                wd = group["weight_decay"]
                alpha = group["rms_clip_alpha"]
                qk_clip = group["qk_clip"]
                ns_steps = group["ns_steps"]
                eps = group["eps"]
                use_muon = group.get("use_muon", True)
                is_qk = group.get("is_qk", False)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]

                    if use_muon and p.ndim == 2:
                        # Momentum buffer.
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        grad = g.add(buf, alpha=momentum) if nesterov else buf

                        # Ortogonalisasi (inti Muon).
                        ortho = _zeropower_via_newtonschulz5(grad, steps=ns_steps)
                        # Skala RMS Muon standar agar magnitudo update konsisten.
                        rows, cols = p.shape
                        ortho = ortho * (max(1.0, rows / cols) ** 0.5)

                        # RMS matching: batas update terhadap RMS bobot aktual.
                        rms_w = p.data.pow(2).mean().sqrt()
                        gamma = alpha * rms_w + eps
                        update = ortho.clamp(min=-gamma, max=gamma)

                        # Weight decay orto-konsisten lalu update.
                        p.data.mul_(1.0 - lr * wd)
                        p.data.add_(update, alpha=-lr)

                        # QK-Clip: batasi norma spektral proyeksi Query/Key.
                        if is_qk:
                            sigma = _spectral_norm_approx(p.data)
                            if float(sigma) > qk_clip:
                                p.data.mul_(qk_clip / (float(sigma) + eps))
                    else:
                        # Jalur AdamW ringan untuk parameter 1D (bias/norm/embedding).
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(g)
                            state["exp_avg_sq"] = torch.zeros_like(g)
                            state["step"] = 0
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        state["step"] += 1
                        beta1, beta2 = 0.9, 0.999
                        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                        bias1 = 1 - beta1 ** state["step"]
                        bias2 = 1 - beta2 ** state["step"]
                        denom = (exp_avg_sq / bias2).sqrt().add_(eps)
                        p.data.mul_(1.0 - lr * wd)
                        p.data.addcdiv_(exp_avg / bias1, denom, value=-lr)
            return loss

    return MuonClip


def get_muonclip_class():
    """Kembalikan kelas MuonClip (dibangun sekali, lazy)."""
    global _MUONCLIP_CLASS
    if _MUONCLIP_CLASS is None:
        _MUONCLIP_CLASS = _build_muonclip_class()
    return _MUONCLIP_CLASS


def build_muonclip_param_groups(model, lr: float, weight_decay: float,
                                rms_clip_alpha: float, qk_clip: float):
    """Pisahkan parameter model ke grup Muon-2D (dengan tanda QK) dan AdamW-1D."""
    qk_2d, other_2d, one_d = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            if ("q_proj" in name) or ("k_proj" in name):
                qk_2d.append(p)
            else:
                other_2d.append(p)
        else:
            one_d.append(p)
    groups = []
    if qk_2d:
        groups.append(dict(params=qk_2d, use_muon=True, is_qk=True,
                           lr=lr, weight_decay=weight_decay,
                           rms_clip_alpha=rms_clip_alpha, qk_clip=qk_clip))
    if other_2d:
        groups.append(dict(params=other_2d, use_muon=True, is_qk=False,
                           lr=lr, weight_decay=weight_decay,
                           rms_clip_alpha=rms_clip_alpha, qk_clip=qk_clip))
    if one_d:
        groups.append(dict(params=one_d, use_muon=False, is_qk=False,
                           lr=lr, weight_decay=weight_decay))
    return groups


def create_muonclip_optimizer(model, lr: float = 2e-4, weight_decay: float = 0.01,
                              rms_clip_alpha: float = 0.2, qk_clip: float = 1.0,
                              momentum: float = 0.95, ns_steps: int = 5):
    """Factory: buat instance MuonClip yang siap dipakai Trainer/SFTTrainer."""
    cls = get_muonclip_class()
    groups = build_muonclip_param_groups(model, lr, weight_decay, rms_clip_alpha, qk_clip)
    return cls(groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
               rms_clip_alpha=rms_clip_alpha, qk_clip=qk_clip, ns_steps=ns_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path ke base model BF16 GGUF")
    parser.add_argument("--dataset", required=True, help="Path ke training JSONL")
    parser.add_argument("--output", required=True, help="Output dir untuk adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=5000, 
                        help="Batasi jumlah sampel per training run")
    parser.add_argument("--optim", default="paged_adamw_8bit",
                        help="Optimizer: 'muonclip' (Kimi K2 style) atau optim HF standar")
    parser.add_argument("--muon-lr", type=float, default=2e-4, help="Learning rate MuonClip")
    parser.add_argument("--muon-wd", type=float, default=0.01, help="Weight decay MuonClip")
    parser.add_argument("--muon-alpha", type=float, default=0.2,
                        help="Skala RMS clipping (γ = α·RMS(W))")
    parser.add_argument("--muon-qk-clip", type=float, default=1.0,
                        help="Ambang QK-Clip (batas norma spektral q_proj/k_proj)")
    args = parser.parse_args()
    
    base_path = Path(args.base)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[TrainLoRA] Base: {base_path.name}")
    print(f"[TrainLoRA] Dataset: {dataset_path}")
    print(f"[TrainLoRA] Output: {output_dir}")
    
    # ── Import training libraries ──────────────────────────────────────────────
    try:
        import torch
        print(f"[TrainLoRA] PyTorch: {torch.__version__}")
        print(f"[TrainLoRA] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[TrainLoRA] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[TrainLoRA] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("[TrainLoRA] ERROR: PyTorch tidak terinstall!")
        print("[TrainLoRA] Jalankan: finetune/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    # ── Konversi GGUF ke format yang bisa di-load PyTorch ─────────────────────
    # Strategy: gunakan llama_cpp_python untuk embed GGUF, lalu fine-tune
    # dengan LoRA menggunakan teknik "phantom gradient" via hook
    #
    # Alternatif: convert BF16 GGUF → safetensors terlebih dahulu
    
    print("[TrainLoRA] Mengkonversi BF16 GGUF → SafeTensors untuk training...")
    safetensors_dir = output_dir.parent / "base_model_hf"
    
    if not safetensors_dir.exists() or not list(safetensors_dir.glob("*.safetensors")):
        safetensors_dir.mkdir(parents=True, exist_ok=True)
        success = convert_gguf_to_hf(base_path, safetensors_dir)
        if not success:
            print("[TrainLoRA] Konversi gagal. Coba manual dengan: python -m gguf.convert_gguf_to_hf")
            sys.exit(1)
    else:
        print(f"[TrainLoRA] SafeTensors sudah ada di {safetensors_dir}, skip konversi.")
    
    # ── Load model dengan unsloth QLoRA ───────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
        print("[TrainLoRA] Menggunakan Unsloth (optimized)")
    except ImportError:
        USE_UNSLOTH = False
        print("[TrainLoRA] Unsloth tidak tersedia, fallback ke PEFT standard")
    
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(safetensors_dir),
            max_seq_length=2048,
            dtype=None,          # Auto detect
            load_in_4bit=True,   # QLoRA — hemat VRAM
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",  # 30% lebih hemat VRAM
            random_state=42,
        )
    else:
        # Standard PEFT fallback
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(str(safetensors_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(safetensors_dir),
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # ── Load Dataset ──────────────────────────────────────────────────────────
    print(f"[TrainLoRA] Memuat dataset dari {dataset_path}...")
    from datasets import Dataset
    
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    pass
    
    # Batasi jumlah sampel per run
    if len(samples) > args.max_samples:
        import random
        random.shuffle(samples)
        samples = samples[:args.max_samples]
    
    print(f"[TrainLoRA] {len(samples)} sampel akan digunakan untuk training")
    
    # Format ke template ChatML Qwen
    def format_sample(sample):
        messages = sample.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = Dataset.from_list([format_sample(s) for s in samples])
    
    # ── Training ──────────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    timestamp = int(time.time())
    adapter_name = f"moko_lora_{timestamp}"
    adapter_save_path = output_dir / adapter_name
    
    use_muonclip = str(args.optim).lower() == "muonclip"
    # Jika MuonClip dipakai, optim HF di-set 'adamw_torch' (diabaikan Trainer karena
    # optimizer kustom di-inject via `optimizers`), namun tetap valid sebagai default.
    hf_optim = "adamw_torch" if use_muonclip else args.optim

    training_args = TrainingArguments(
        output_dir=str(adapter_save_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.muon_lr if use_muonclip else 2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=hf_optim,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # ── MuonClip: inject optimizer kustom (Kimi K2 style) ─────────────────────
    custom_optimizers = (None, None)
    if use_muonclip:
        print("[TrainLoRA] Optimizer: MuonClip (RMS matching + QK-Clip) diaktifkan.")
        muon_opt = create_muonclip_optimizer(
            model,
            lr=args.muon_lr,
            weight_decay=args.muon_wd,
            rms_clip_alpha=args.muon_alpha,
            qk_clip=args.muon_qk_clip,
        )
        # (optimizer, lr_scheduler) — scheduler dibuat otomatis oleh Trainer.
        custom_optimizers = (muon_opt, None)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
        optimizers=custom_optimizers,
    )
    
    print(f"[TrainLoRA] Mulai training {len(samples)} sampel × {args.epochs} epoch...")
    trainer.train()
    
    # ── Simpan LoRA Adapter ───────────────────────────────────────────────────
    model.save_pretrained(str(adapter_save_path))
    tokenizer.save_pretrained(str(adapter_save_path))
    print(f"[TrainLoRA] LoRA adapter disimpan: {adapter_save_path}")
    
    # ── Konversi adapter ke GGUF untuk llama-server ───────────────────────────
    gguf_adapter = output_dir / f"moko_lora_{timestamp}.gguf"
    convert_lora_to_gguf(adapter_save_path, gguf_adapter)
    
    print(f"[TrainLoRA] ✅ Selesai! GGUF adapter: {gguf_adapter}")


def convert_gguf_to_hf(gguf_path: Path, output_dir: Path) -> bool:
    """Konversi BF16 GGUF ke HuggingFace format (safetensors)."""
    import subprocess
    
    # Coba pakai gguf package
    try:
        result = subprocess.run(
            [sys.executable, "-m", "gguf.convert_gguf_to_hf",
             str(gguf_path), str(output_dir)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"[TrainLoRA] Konversi berhasil via gguf package")
            return True
    except Exception as e:
        print(f"[TrainLoRA] gguf package gagal: {e}")
    
    # Coba pakai llama.cpp convert script
    llama_cpp_convert = Path(__file__).parent.parent / "moko_core" / "moko_inference" / "convert_hf_to_gguf.py"
    if llama_cpp_convert.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(llama_cpp_convert),
                 "--outfile", str(output_dir / "model.safetensors"),
                 str(gguf_path)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
    
    print("[TrainLoRA] Tidak bisa konversi GGUF secara otomatis.")
    print("[TrainLoRA] Install: pip install gguf transformers")
    return False


def convert_lora_to_gguf(adapter_dir: Path, output_gguf: Path) -> bool:
    """Konversi LoRA adapter ke GGUF untuk dipakai llama-server."""
    import subprocess
    
    llama_bin = Path(__file__).parent.parent / "moko_core" / "moko_inference" / "llama_bin"
    export_lora = llama_bin / "llama-export-lora"
    
    if not export_lora.exists():
        print(f"[TrainLoRA] llama-export-lora tidak ditemukan di {llama_bin}")
        print(f"[TrainLoRA] Adapter tersimpan dalam format HF di {adapter_dir}")
        return False
    
    result = subprocess.run(
        [str(export_lora),
         "--model-base", str(list(Path(__file__).parent.glob("base_model/*BF16*.gguf"))[0]),
         "--lora", str(adapter_dir),
         "--output", str(output_gguf)],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode == 0:
        print(f"[TrainLoRA] LoRA GGUF berhasil dibuat: {output_gguf}")
        return True
    else:
        print(f"[TrainLoRA] Export LoRA GGUF gagal: {result.stderr[:300]}")
        return False


if __name__ == "__main__":
    main()
