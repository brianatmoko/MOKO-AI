"""
Convert GGUF → SafeTensors (HuggingFace format)
=================================================
Konversi Qwen2.5-1.5B-Instruct GGUF ke safetensors agar bisa
digunakan untuk LoRA fine-tuning tanpa download ulang dari HuggingFace.
"""
import sys
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file
from gguf import GGUFReader


def map_tensor_name(gguf_name: str) -> str:
    """Map GGUF tensor name to HuggingFace Qwen2 format."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"

    parts = gguf_name.split(".")
    if len(parts) < 3:
        return gguf_name

    idx = parts[1]  # block index

    if ".attn_norm.weight" in gguf_name:
        return f"model.layers.{idx}.input_layernorm.weight"
    if ".attn_norm.bias" in gguf_name:
        return f"model.layers.{idx}.input_layernorm.bias"
    if ".ffn_norm.weight" in gguf_name:
        return f"model.layers.{idx}.post_attention_layernorm.weight"
    if ".ffn_norm.bias" in gguf_name:
        return f"model.layers.{idx}.post_attention_layernorm.bias"
    if ".attn_q.weight" in gguf_name:
        return f"model.layers.{idx}.self_attn.q_proj.weight"
    if ".attn_k.weight" in gguf_name:
        return f"model.layers.{idx}.self_attn.k_proj.weight"
    if ".attn_v.weight" in gguf_name:
        return f"model.layers.{idx}.self_attn.v_proj.weight"
    if ".attn_output.weight" in gguf_name:
        return f"model.layers.{idx}.self_attn.o_proj.weight"
    if ".ffn_gate.weight" in gguf_name:
        return f"model.layers.{idx}.mlp.gate_proj.weight"
    if ".ffn_up.weight" in gguf_name:
        return f"model.layers.{idx}.mlp.up_proj.weight"
    if ".ffn_down.weight" in gguf_name:
        return f"model.layers.{idx}.mlp.down_proj.weight"

    return gguf_name


def convert_gguf_to_safetensors(gguf_path: str, output_dir: str):
    gguf_path = Path(gguf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Convert] Reading GGUF: {gguf_path}")
    reader = GGUFReader(str(gguf_path))

    tensors = {}
    for t in reader.tensors:
        name = map_tensor_name(t.name)
        data = t.data
        if data.dtype == np.float32:
            data = data.astype(np.float16)
        tensors[name] = data
        print(f"  {t.name} -> {name}  {data.shape} {data.dtype}")

    out_file = output_dir / "model.safetensors"
    print(f"\n[Convert] Saving {len(tensors)} tensors to {out_file}")
    save_file(tensors, str(out_file))

    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"[Convert] Done! Size: {size_mb:.0f} MB")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_gguf_to_safetensors.py <input.gguf> <output_dir>")
        sys.exit(1)
    convert_gguf_to_safetensors(sys.argv[1], sys.argv[2])
