"""
Convert LoRA adapter (safetensors) → GGUF for llama-server
"""
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
import torch
import sys
from gguf import GGUFWriter, GGUFReader


def convert_lora_to_gguf(adapter_dir: str, output_gguf: str):
    adapter_dir = Path(adapter_dir)
    output_gguf = Path(output_gguf)

    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)

    r = config["r"]
    alpha = config["lora_alpha"]
    scaling = alpha / r

    print(f"LoRA: r={r}, alpha={alpha}, scaling={scaling}")

    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    print(f"Loaded {len(weights)} tensors")

    # Group by (layer, module, proj)
    layers = {}
    for name, tensor in weights.items():
        parts = name.split(".")
        layer_idx = int(parts[4])
        module_type = parts[5]
        proj_name = parts[6]
        lora_side = parts[7]

        key = (layer_idx, module_type, proj_name)
        if key not in layers:
            layers[key] = {}
        layers[key][lora_side] = tensor.to(torch.float32).cpu().numpy()

    print(f"Grouped into {len(layers)} LoRA pairs")

    writer = GGUFWriter(str(output_gguf), "adapter")
    writer.add_name("moko_coder_lora")

    count = 0
    for (layer_idx, module_type, proj_name), lora_pair in sorted(layers.items()):
        if "lora_A" not in lora_pair or "lora_B" not in lora_pair:
            continue

        if module_type == "self_attn":
            gguf_proj = {"q_proj": "attn_q", "k_proj": "attn_k",
                         "v_proj": "attn_v", "o_proj": "attn_output"}.get(proj_name)
        elif module_type == "mlp":
            gguf_proj = {"gate_proj": "ffn_gate", "up_proj": "ffn_up",
                         "down_proj": "ffn_down"}.get(proj_name)
        else:
            continue

        if not gguf_proj:
            continue

        lora_a = lora_pair["lora_A"]  # (r, in_dim)
        lora_b = lora_pair["lora_B"]  # (out_dim, r)

        name_a = f"blk.{layer_idx}.{gguf_proj}.loraA.weight"
        name_b = f"blk.{layer_idx}.{gguf_proj}.loraB.weight"

        writer.add_tensor(name_a, lora_a)
        writer.add_tensor(name_b, lora_b * scaling)
        count += 2

        print(f"  Layer {layer_idx:2d} {module_type}.{proj_name}: A={lora_a.shape} B={lora_b.shape}")

    print(f"\nWriting {count} tensors...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf.stat().st_size / 1024 / 1024
    print(f"✅ GGUF LoRA saved: {output_gguf} ({size_mb:.1f} MB)")

    # Verify
    reader = GGUFReader(str(output_gguf))
    print(f"Verification: {len(reader.tensors)} tensors in GGUF")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_lora_to_gguf.py <adapter_dir> <output.gguf>")
        sys.exit(1)
    convert_lora_to_gguf(sys.argv[1], sys.argv[2])
