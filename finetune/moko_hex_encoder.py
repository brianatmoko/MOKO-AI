"""
MOKO Hex Encoder — Binary Compressed Data Encoding
====================================================
Encode code/text → binary → compress → hex string.
Decode hex string → decompress → binary → code/text.

Pipeline:
  Source code → UTF-8 bytes → zlib compress → hex string

Compression ratio:
  Raw UTF-8:    N bytes
  zlib typical:  N/3 bytes (3x compression on code)
  Hex encoding:  N/3 × 2 = 2N/3 hex chars
  Net result:    ~1.5x smaller than raw text

Usage:
  from moko_hex_encoder import MokoEncoder
  enc = MokoEncoder()
  hex_str = enc.encode("def hello(): print('hi')")
  code = enc.decode(hex_str)
"""

import zlib
import binascii
import json
import hashlib
from pathlib import Path
from typing import Optional


class MokoEncoder:
    """
    Encoder: Teks → Binary → Compress → Hex.
    Decoder: Hex → Decompress → Binary → Teks.
    """

    # Compression levels: 1=fast, 6=default, 9=max
    COMPRESS_LEVEL = 9

    # Header: menandai format ini adalah MOKO encoded
    MAGIC = b"MK"  # 2 bytes magic header
    VERSION = 1

    def encode(self, text: str) -> str:
        """
        Encode teks ke hex string.

        Args:
            text: Source code atau teks

        Returns:
            Hex string (lowercase)
        """
        # UTF-8 bytes
        raw = text.encode("utf-8")

        # Tambah header: magic + version + original length
        header = self.MAGIC + bytes([self.VERSION])
        payload = header + raw

        # Compress
        compressed = zlib.compress(payload, level=self.COMPRESS_LEVEL)

        # Ke hex
        hex_str = binascii.hexlify(compressed).decode("ascii")
        return hex_str

    def decode(self, hex_str: str) -> Optional[str]:
        """
        Decode hex string ke teks.

        Args:
            hex_str: Hex-encoded compressed data

        Returns:
            Decoded teks, atau None jika gagal
        """
        try:
            # Dari hex ke bytes
            compressed = binascii.unhexlify(hex_str)

            # Decompress
            payload = zlib.decompress(compressed)

            # Cek header
            if payload[:2] != self.MAGIC:
                return None

            # Extract raw data (skip header: magic 2 + version 1)
            raw = payload[3:]
            return raw.decode("utf-8")
        except Exception:
            return None

    def encode_to_dict(self, text: str, meta: dict = None) -> dict:
        """
        Encode teks ke dict dengan metadata.

        Returns:
            {
                "hex": "789c0bca...",
                "size_raw": 1234,
                "size_compressed": 456,
                "ratio": 2.7,
                "hash": "abc123...",
                ...meta
            }
        """
        raw_bytes = text.encode("utf-8")
        hex_str = self.encode(text)
        compressed_bytes = binascii.unhexlify(hex_str)

        result = {
            "hex": hex_str,
            "size_raw": len(raw_bytes),
            "size_compressed": len(compressed_bytes),
            "ratio": round(len(raw_bytes) / max(len(compressed_bytes), 1), 2),
            "hash": hashlib.sha256(raw_bytes).hexdigest()[:16],
        }
        if meta:
            result.update(meta)
        return result

    def decode_from_dict(self, data: dict) -> Optional[str]:
        """Decode dari dict yang berisi key 'hex'."""
        return self.decode(data.get("hex", ""))


class MokoDatasetConverter:
    """
    Convert existing ChatML dataset ke hex-encoded format.
    """

    def __init__(self):
        self.encoder = MokoEncoder()

    def convert_sample(self, sample: dict) -> dict:
        """
        Convert 1 ChatML sample ke hex format.

        Input format:
          {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Write binary search"},
            {"role": "assistant", "content": "def binary_search..."}
          ]}

        Output format:
          {"messages": [
            {"role": "system", "content": "MOKO_HEX_V1: ..."},
            {"role": "user", "content": "Write binary search"},
            {"role": "assistant", "content": "hex_encoded_compressed_code"}
          ]}
        """
        msgs = sample.get("messages", [])
        if len(msgs) < 2:
            return sample

        # System prompt: tambahkan instruksi hex
        system_content = msgs[0].get("content", "")
        hex_system = (
            "You are MOKO Coder v3 (Hex Mode). "
            "You generate code as hex-encoded compressed binary. "
            "Format: lowercase hex string (zlib-compressed UTF-8). "
            "To decode: bytes.fromhex(hex_str) → zlib.decompress() → decode('utf-8'). "
            "Always output COMPLETE hex strings, no truncation. "
            "Original system: " + system_content
        )

        # User prompt: tetap sama
        user_msg = msgs[1].get("content", "") if len(msgs) > 1 else ""

        # Assistant response: encode ke hex
        assistant_msg = ""
        for m in msgs:
            if m.get("role") == "assistant":
                assistant_msg = m.get("content", "")
                break

        if assistant_msg:
            hex_encoded = self.encoder.encode(assistant_msg)
        else:
            hex_encoded = ""

        return {
            "messages": [
                {"role": "system", "content": hex_system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": hex_encoded},
            ],
            "_meta": {
                "original_size": len(assistant_msg.encode("utf-8")),
                "hex_size": len(hex_encoded),
                "ratio": round(len(assistant_msg.encode("utf-8")) / max(len(hex_encoded) // 2, 1), 2),
            }
        }

    def convert_file(self, input_path: Path, output_path: Path) -> int:
        """
        Convert seluruh JSONL dataset ke hex format.

        Returns: jumlah sample yang di-convert
        """
        count = 0
        errors = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for i, line in enumerate(fin):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    converted = self.convert_sample(sample)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  ⚠️  Error at line {i+1}: {e}")

        return count

    def convert_and_validate(self, input_path: Path, output_path: Path) -> dict:
        """
        Convert + validate roundtrip (encode → decode = original).
        """
        count = 0
        roundtrip_ok = 0
        roundtrip_fail = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    converted = self.convert_sample(sample)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1

                    # Validate roundtrip
                    hex_str = converted["messages"][2]["content"]
                    decoded = self.encoder.decode(hex_str)

                    # Get original assistant content
                    original = ""
                    for m in sample.get("messages", []):
                        if m.get("role") == "assistant":
                            original = m.get("content", "")
                            break

                    if decoded == original:
                        roundtrip_ok += 1
                    else:
                        roundtrip_fail += 1
                        if roundtrip_fail <= 3:
                            print(f"  ⚠️  Roundtrip mismatch at sample {count}")
                            print(f"     Original: {original[:50]}...")
                            print(f"     Decoded:  {decoded[:50] if decoded else 'None'}...")

                except Exception as e:
                    pass

        return {
            "total": count,
            "roundtrip_ok": roundtrip_ok,
            "roundtrip_fail": roundtrip_fail,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MOKO Hex Encoder")
    parser.add_argument("--encode", type=str, help="Encode text string to hex")
    parser.add_argument("--decode", type=str, help="Decode hex string to text")
    parser.add_argument("--convert", nargs=2, metavar=("INPUT", "OUTPUT"),
                       help="Convert JSONL dataset to hex format")
    parser.add_argument("--validate", nargs=2, metavar=("INPUT", "OUTPUT"),
                       help="Convert + validate roundtrip")
    parser.add_argument("--test", action="store_true", help="Run self-test")

    args = parser.parse_args()
    enc = MokoEncoder()

    if args.encode:
        hex_str = enc.encode(args.encode)
        print(f"Input:  {args.encode}")
        print(f"Output: {hex_str}")
        print(f"Size:   {len(args.encode)} → {len(hex_str)//2} bytes ({len(hex_str)} hex chars)")

    elif args.decode:
        text = enc.decode(args.decode)
        if text:
            print(f"Decoded: {text}")
        else:
            print("❌ Decode failed")

    elif args.convert:
        inp, out = Path(args.convert[0]), Path(args.convert[1])
        print(f"Converting {inp} → {out}")
        converter = MokoDatasetConverter()
        count = converter.convert_file(inp, out)
        print(f"✅ Converted {count} samples")

    elif args.validate:
        inp, out = Path(args.validate[0]), Path(args.validate[1])
        print(f"Converting + validating {inp} → {out}")
        converter = MokoDatasetConverter()
        result = converter.convert_and_validate(inp, out)
        print(f"✅ Total: {result['total']}")
        print(f"   Roundtrip OK:    {result['roundtrip_ok']}")
        print(f"   Roundtrip FAIL:  {result['roundtrip_fail']}")

    elif args.test:
        print("=== MOKO Hex Encoder Self-Test ===\n")

        test_cases = [
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return None",
            "SELECT * FROM users WHERE age > 18 ORDER BY name;",
            "#!/bin/bash\necho 'Hello World'\nfor i in {1..10}; do\n    echo \"Count: $i\"\ndone",
            "import React from 'react';\nconst App = () => <div>Hello</div>;",
        ]

        for i, text in enumerate(test_cases):
            hex_str = enc.encode(text)
            decoded = enc.decode(hex_str)
            original_size = len(text.encode("utf-8"))
            compressed_size = len(hex_str) // 2

            print(f"Test {i+1}: {text[:40]}...")
            print(f"  Original:    {original_size} bytes")
            print(f"  Compressed:  {compressed_size} bytes")
            print(f"  Ratio:       {original_size/compressed_size:.2f}x")
            print(f"  Roundtrip:   {'✅ OK' if decoded == text else '❌ FAIL'}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
