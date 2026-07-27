#!/usr/bin/env python3
"""
MOKO AI Rebrander — GGUF Metadata Transformer
==============================================
Transform identity from MOKO to MOKO AI.
Non-cryptographic version.
"""

import hashlib
import json
import os
import struct
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# MOKO AI CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MOKO_MODEL_METADATA = {
    "general.name":        "MOKO-AI",
    "general.basename":    "MOKO-AI-4B",
    "general.size_label":  "4B",
    "general.type":        "model",
    "general.architecture": "qwen35",  # Keep architecture for llama.cpp compatibility
}

MOKO_AI_DESCRIPTION = (
    "MOKO AI — Industrial Intelligence System. "
    "Built on MOKO3.5-4B Hybrid SSM-Attention architecture. "
    "Optimized for MOKO OS."
)

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

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

GGUF_TYPE_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
}


# ═══════════════════════════════════════════════════════════════════════════
# GGUF LOW-LEVEL PARSER
# ═══════════════════════════════════════════════════════════════════════════

class GGUFReader:
    """Minimal GGUF Parser."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, 'rb')
        self.kv_pairs: List[Tuple[str, int, Any]] = []
        self.tensor_header_offset: int = 0
        self.tensor_data_offset: int = 0
        self._parse_header()

    def _ru32(self) -> int: return struct.unpack('<I', self.f.read(4))[0]
    def _ru64(self) -> int: return struct.unpack('<Q', self.f.read(8))[0]
    def _read_str_bytes(self) -> bytes:
        n = self._ru64()
        return self.f.read(n)

    def _skip_value(self, vtype: int):
        pos_before = self.f.tell()
        if vtype == GGUF_TYPE_STRING:
            n = self._ru64()
            self.f.seek(n, 1)
        elif vtype == GGUF_TYPE_ARRAY:
            item_type = self._ru32()
            count = self._ru64()
            if item_type == GGUF_TYPE_STRING:
                for _ in range(count):
                    n = self._ru64()
                    self.f.seek(n, 1)
            elif item_type in GGUF_TYPE_SIZES:
                self.f.seek(GGUF_TYPE_SIZES[item_type] * count, 1)
            else:
                raise ValueError(f"Unknown array item type: {item_type}")
        elif vtype in GGUF_TYPE_SIZES:
            self.f.read(GGUF_TYPE_SIZES[vtype])
        else:
            raise ValueError(f"Unknown GGUF type: {vtype}")
        pos_after = self.f.tell()
        return pos_before, pos_after

    def _parse_header(self):
        magic = self._ru32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a valid GGUF file! Magic: 0x{magic:08X}")

        self.version = self._ru32()
        self.n_tensors = self._ru64()
        self.n_kv = self._ru64()

        self.kv_positions = []
        self.kv_keys = []
        self.kv_vtypes = []

        for _ in range(self.n_kv):
            pos_start = self.f.tell()
            key_bytes = self._read_str_bytes()
            key = key_bytes.decode('utf-8', errors='replace')
            vtype = self._ru32()
            pos_before_val, pos_after_val = self._skip_value(vtype)
            pos_end = self.f.tell()

            self.kv_positions.append((pos_start, pos_end))
            self.kv_keys.append(key)
            self.kv_vtypes.append(vtype)

        self.tensor_header_offset = self.f.tell()

        for _ in range(self.n_tensors):
            name_len = self._ru64()
            self.f.read(name_len)
            n_dims = self._ru32()
            self.f.read(8 * n_dims)
            self._ru32()
            self._ru64()

        current_pos = self.f.tell()
        self.tensor_headers_end = current_pos
        alignment = 32
        aligned_pos = ((current_pos + alignment - 1) // alignment) * alignment
        self.tensor_data_offset = aligned_pos

    def close(self):
        self.f.close()

    def get_kv_raw(self, idx: int) -> bytes:
        pos_start, pos_end = self.kv_positions[idx]
        self.f.seek(pos_start)
        return self.f.read(pos_end - pos_start)


# ═══════════════════════════════════════════════════════════════════════════
# GGUF WRITER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def write_str(f, s: str) -> int:
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)
    return 8 + len(b)

def write_kv_string(f, key: str, value: str) -> int:
    n = write_str(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    n += 4
    n += write_str(f, value)
    return n


# ═══════════════════════════════════════════════════════════════════════════
# FINGERPRINTING
# ═══════════════════════════════════════════════════════════════════════════

def compute_tensor_fingerprint(model_path: str, tensor_data_offset: int,
                                chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 fingerprint of the tensor data block."""
    sha = hashlib.sha256()
    with open(model_path, 'rb') as f:
        f.seek(tensor_data_offset)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN REBRANDER
# ═══════════════════════════════════════════════════════════════════════════

def rebrand(
    input_path: str,
    output_path: str,
    skip_fingerprint: bool = False,
) -> bool:
    """Rebrand GGUF model from MOKO to MOKO AI."""
    print(f"  🤖 MOKO AI Rebrander")

    if not os.path.exists(input_path):
        print(f"  ❌ Input file not found: {input_path}")
        return False

    reader = GGUFReader(input_path)

    if skip_fingerprint:
        tensor_fingerprint = "SKIPPED"
    else:
        tensor_fingerprint = compute_tensor_fingerprint(input_path, reader.tensor_data_offset)

    build_timestamp = int(time.time())

    moko_metadata = {
        "moko.identity":          "MOKO-AI-v1",
        "moko.fingerprint":       tensor_fingerprint,
        "moko.build_timestamp":   str(build_timestamp),
        "moko.base_architecture": "qwen35",
        "moko.description":       MOKO_AI_DESCRIPTION,
    }

    OVERRIDE_KEYS = set(MOKO_MODEL_METADATA.keys())
    written_overrides = set()
    new_moko_keys = set(moko_metadata.keys())
    
    overriding_existing = OVERRIDE_KEYS.intersection(set(reader.kv_keys))
    kept_original = reader.n_kv - len(overriding_existing)
    new_kv_count = kept_original + len(OVERRIDE_KEYS) + len(new_moko_keys)

    with open(output_path, 'wb') as out:
        out.write(struct.pack('<I', GGUF_MAGIC))
        out.write(struct.pack('<I', reader.version))
        out.write(struct.pack('<Q', reader.n_tensors))
        out.write(struct.pack('<Q', new_kv_count))

        for key, value in MOKO_MODEL_METADATA.items():
            write_kv_string(out, key, value)
            written_overrides.add(key)

        for key, value in moko_metadata.items():
            write_kv_string(out, key, value)
            written_overrides.add(key)

        with open(input_path, 'rb') as src:
            for i, key in enumerate(reader.kv_keys):
                if key in written_overrides:
                    continue
                raw_bytes = reader.get_kv_raw(i)
                out.write(raw_bytes)

        with open(input_path, 'rb') as src:
            src.seek(reader.tensor_header_offset)
            tensor_header_size = reader.tensor_headers_end - reader.tensor_header_offset
            out.write(src.read(tensor_header_size))

        current_pos = out.tell()
        alignment = 32
        pad_size = (alignment - (current_pos % alignment)) % alignment
        if pad_size:
            out.write(b'\x00' * pad_size)

        chunk_size = 64 * 1024 * 1024
        with open(input_path, 'rb') as src:
            src.seek(reader.tensor_data_offset)
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)

    reader.close()
    print(f"  ✅ SUCCESS: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="MOKO AI Rebrander")
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--skip-fingerprint', action='store_true')
    args = parser.parse_args()

    rebrand(args.input, args.output, args.skip_fingerprint)


if __name__ == "__main__":
    main()
