import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import os

# Path ke base model yang ada di folder finetune
BASE_MODEL_DIR = "../finetune/base_model_hf"

def bedah_saraf():
    print("="*60)
    print("  MOKO NEURAL SURGERY — Powered by PyTorch & Transformers")
    print("="*60)
    
    if not os.path.exists(BASE_MODEL_DIR):
        print(f"Error: Base model tidak ditemukan di {BASE_MODEL_DIR}")
        return

    print(f"[*] Memuat model dari {BASE_MODEL_DIR}...")
    try:
        # Memuat model dengan device_map auto untuk menangani VRAM terbatas (RTX 2050 4GB)
        # Menggunakan torch_dtype=torch.float16 untuk efisiensi
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
        print("[+] Model berhasil dimuat ke memori.")
    except Exception as e:
        print(f"[-] Gagal memuat model: {e}")
        return

    # 1. Inspeksi Arsitektur (Bedah Anatomi)
    print("\n[1] Inspeksi Arsitektur (Anatomi Saraf):")
    print(f"    - Tipe Model: {type(model).__name__}")
    print(f"    - Jumlah Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Menampilkan struktur layer
    print("\n    - Struktur Layer Utama:")
    for name, module in model.named_children():
        print(f"      |-- {name}: {type(module).__name__}")

    # 2. Bedah Bobot (Weight Inspection)
    print("\n[2] Bedah Bobot (Neuro-Analysis):")
    # Ambil contoh layer pertama (embedding) dan layer perhatian (attention)
    with torch.no_grad():
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_weights = model.model.embed_tokens.weight
            print(f"    - Embedding Layer: {embed_weights.shape}")
            print(f"      |-- Mean: {embed_weights.mean().item():.6f}")
            print(f"      |-- Std:  {embed_weights.std().item():.6f}")
        
        # Bedah layer terakhir untuk melihat logit head
        if hasattr(model, 'lm_head'):
            head_weights = model.lm_head.weight
            print(f"    - LM Head (Output Layer): {head_weights.shape}")

    # 3. Neural Manipulation (Simulasi Bedah)
    print("\n[3] Manipulasi Saraf (Neural Manipulation):")
    print("    [*] Simulasi pengubahan identitas pada level bobot...")
    # Contoh: Kita bisa mencari representasi token "Qwen" dan menggantinya ke "MOKO" 
    # di level embedding jika kita ingin melakukan bedah permanen.
    
    qwen_tokens = tokenizer.encode("Qwen", add_special_tokens=False)
    moko_tokens = tokenizer.encode("MOKO", add_special_tokens=False)
    
    print(f"    - Token 'Qwen' IDs: {qwen_tokens}")
    print(f"    - Token 'MOKO' IDs: {moko_tokens}")

    print("\n" + "="*60)
    print("  BEDAH SELESAI. Model siap untuk optimasi tingkat lanjut.")
    print("="*60)

if __name__ == "__main__":
    bedah_saraf()
