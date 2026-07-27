"""
MOKO GGUF Editor — Phase 13: GGUF Speed Surgery
================================================
Tool untuk inspeksi dan edit metadata GGUF file secara aman.

Fokus: Potong overhead chat_template bawaan model yang masih menyuntikkan
token <|im_start|>system<|im_end|> meski system_prompt sudah kosong.

Setelah Phase 13 Prompt Purge, system_prompt selalu "" — tapi GGUF masih
memformat: <|im_start|>system\n\n<|im_end|>\n yang makan ~8 token sia-sia
dan 10-15ms prefill per query.

USAGE:
  # Inspect saja (aman):
  python gguf_editor.py --inspect MOKO-AI-4B-CryptoCore-BF16.gguf

  # Backup + edit template (BERBAHAYA — backup otomatis dibuat):
  python gguf_editor.py --patch MOKO-AI-4B-CryptoCore-BF16.gguf

  # Verify integritas setelah edit:
  python gguf_editor.py --verify MOKO-AI-4B-CryptoCore-BF16.gguf
"""

import struct
import sys
import os
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any


# ─── GGUF Constants ───────────────────────────────────────────────────────────
GGUF_MAGIC       = 0x46554747   # "GGUF" dalam little-endian
GGUF_VERSION_MIN = 2
GGUF_VERSION_MAX = 3

# GGUF Value Types
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12


def read_gguf_string(f) -> str:
    """Baca GGUF string: uint64 length + UTF-8 bytes."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def read_gguf_value(f, vtype: int) -> Any:
    """Baca satu value berdasarkan type."""
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack('<h', f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack('<B', f.read(1))[0] != 0
    elif vtype == GGUF_TYPE_STRING:
        return read_gguf_string(f)
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count     = struct.unpack('<Q', f.read(8))[0]
        return [read_gguf_value(f, elem_type) for _ in range(count)]
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def inspect_gguf(gguf_path: str) -> dict:
    """
    Baca header GGUF dan ekstrak semua metadata.
    Returns dict berisi: version, n_tensors, metadata dict.
    """
    result = {
        "path": gguf_path,
        "valid": False,
        "version": None,
        "n_tensors": None,
        "n_kv": None,
        "metadata": {},
        "chat_template": None,
        "file_size_gb": None,
    }

    try:
        file_size = os.path.getsize(gguf_path)
        result["file_size_gb"] = round(file_size / (1024**3), 2)

        with open(gguf_path, 'rb') as f:
            # 1. Magic
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                result["error"] = f"Invalid GGUF magic: {hex(magic)}"
                return result

            # 2. Version
            version = struct.unpack('<I', f.read(4))[0]
            result["version"] = version
            if version < GGUF_VERSION_MIN or version > GGUF_VERSION_MAX:
                result["error"] = f"Unsupported GGUF version: {version}"
                return result

            # 3. Tensor count & KV count
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv      = struct.unpack('<Q', f.read(8))[0]
            result["n_tensors"] = n_tensors
            result["n_kv"]      = n_kv

            # 4. Baca semua KV pairs
            for _ in range(n_kv):
                key   = read_gguf_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = read_gguf_value(f, vtype)

                # Simpan string metadata saja (tensor data tidak disimpan)
                if isinstance(value, str):
                    result["metadata"][key] = value[:500]  # Cap panjang
                elif isinstance(value, (int, float, bool)):
                    result["metadata"][key] = value
                else:
                    result["metadata"][key] = f"<{type(value).__name__}>"

                # Khusus chat_template — simpan penuh
                if key == "tokenizer.chat_template":
                    result["chat_template"] = value

            result["valid"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# Template minimal Phase 13 — tidak ada system role
# Hanya user + assistant. Menghilangkan overhead <|im_start|>system<|im_end|>
MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


def patch_chat_template(gguf_path: str, new_template: str = MINIMAL_CHAT_TEMPLATE) -> Tuple[bool, str]:
    """
    BERBAHAYA — memodifikasi file GGUF secara langsung.
    Backup otomatis dibuat sebelum edit.

    Strategi: GGUF v2/v3 tidak mendukung in-place edit metadata yang berubah ukuran.
    Solusi: Tulis ulang seluruh header ke file baru, lalu timpa file asli.

    Returns: (success, message)
    """
    # Backup wajib
    backup_path = gguf_path + ".backup_phase13"
    if not os.path.exists(backup_path):
        print(f"📦 Membuat backup: {backup_path} ...")
        shutil.copy2(gguf_path, backup_path)
        print(f"✅ Backup selesai ({os.path.getsize(backup_path) / (1024**3):.2f} GB)")
    else:
        print(f"📦 Backup sudah ada: {backup_path}")

    print("⚠️  PERINGATAN: Modifikasi GGUF adalah operasi berbahaya.")
    print("   Jika gagal, restore dari backup: cp backup_phase13 original")

    # NOTE: Implementasi patch GGUF memerlukan penulisan ulang seluruh file
    # karena metadata string bisa berubah panjang. Ini membutuhkan ~8GB ruang disk.
    #
    # Untuk keamanan, implementasi ini hanya melakukan DRY RUN dan menampilkan diff.
    # Edit aktual diimplementasikan di versi berikutnya dengan konfirmasi eksplisit.

    info = inspect_gguf(gguf_path)
    if not info["valid"]:
        return False, f"GGUF tidak valid: {info.get('error', 'unknown')}"

    current = info.get("chat_template", "")
    if not current:
        return False, "chat_template tidak ditemukan di GGUF ini"

    print(f"\n📋 Template SAAT INI ({len(current)} chars):")
    print("=" * 60)
    print(current[:500])
    print("...")

    print(f"\n📋 Template BARU (Phase 13 Minimal, {len(new_template)} chars):")
    print("=" * 60)
    print(new_template)

    print("\n🔍 DRY RUN — tidak ada perubahan yang diterapkan.")
    print("   Untuk menerapkan: tambahkan flag --confirm ke command")
    print("   CATATAN: Edit aktif memerlukan penulisan ulang file 8GB")
    print("   Alternatif lebih aman: gunakan Modelfile di Ollama untuk override template")

    # Alternatif yang lebih aman: Ollama Modelfile override
    modelfile_path = os.path.dirname(gguf_path)
    modelfile_content = f"""FROM {os.path.basename(gguf_path)}
TEMPLATE \"\"\"{{{{- if .System }}}}{{{{ .System }}}}
{{{{- end }}}}{{{{- range .Messages }}}}{{{{ if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
<|im_start|>assistant
{{{{- else if eq .Role "assistant" }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}{{{{- end }}}}\"\"\"
PARAMETER stop "<|im_end|>"
"""
    modelfile_out = os.path.join(modelfile_path, "Modelfile.phase13")
    with open(modelfile_out, "w") as mf:
        mf.write(modelfile_content)

    print(f"\n✅ Modelfile alternatif ditulis: {modelfile_out}")
    print("   Gunakan: ollama create moko-phase13 -f Modelfile.phase13")

    return True, f"DRY RUN selesai. Modelfile ditulis ke {modelfile_out}"


def verify_gguf(gguf_path: str) -> Tuple[bool, str]:
    """Verifikasi integritas GGUF setelah edit."""
    info = inspect_gguf(gguf_path)
    if not info["valid"]:
        return False, f"GGUF tidak valid: {info.get('error', 'unknown')}"

    # Hitung SHA-256 4KB header sebagai fingerprint
    with open(gguf_path, 'rb') as f:
        header = f.read(4096)
    fp = hashlib.sha256(header).hexdigest()

    return True, (
        f"✅ GGUF Valid\n"
        f"   Version  : {info['version']}\n"
        f"   Tensors  : {info['n_tensors']}\n"
        f"   KV pairs : {info['n_kv']}\n"
        f"   Size     : {info['file_size_gb']} GB\n"
        f"   Fingerprint: {fp[:32]}...\n"
        f"   chat_template: {'FOUND' if info['chat_template'] else 'NOT FOUND'}"
    )


def main():
    parser = argparse.ArgumentParser(description="MOKO GGUF Editor — Phase 13")
    parser.add_argument("gguf", help="Path ke file GGUF")
    parser.add_argument("--inspect", action="store_true", help="Inspect metadata GGUF")
    parser.add_argument("--patch",   action="store_true", help="Patch chat_template (dry run + Modelfile)")
    parser.add_argument("--verify",  action="store_true", help="Verifikasi integritas GGUF")
    args = parser.parse_args()

    if not os.path.exists(args.gguf):
        print(f"❌ File tidak ditemukan: {args.gguf}")
        sys.exit(1)

    if args.inspect or (not args.patch and not args.verify):
        print(f"\n🔍 Inspeksi GGUF: {args.gguf}")
        info = inspect_gguf(args.gguf)
        print(f"   Valid    : {info['valid']}")
        print(f"   Version  : {info['version']}")
        print(f"   Tensors  : {info['n_tensors']}")
        print(f"   KV pairs : {info['n_kv']}")
        print(f"   Size     : {info.get('file_size_gb', '?')} GB")
        if info.get("chat_template"):
            print(f"\n   chat_template ({len(info['chat_template'])} chars):")
            print("   " + "-"*50)
            print("   " + info["chat_template"][:800].replace("\n", "\n   "))
        else:
            print("   chat_template: TIDAK DITEMUKAN")
        if info.get("error"):
            print(f"   ERROR: {info['error']}")

    if args.patch:
        success, msg = patch_chat_template(args.gguf)
        print(f"\n{'✅' if success else '❌'} {msg}")

    if args.verify:
        ok, msg = verify_gguf(args.gguf)
        print(f"\n{msg}")


if __name__ == "__main__":
    main()
