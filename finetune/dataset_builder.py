"""
MOKO Coder — Dataset Builder (v3 — Clean Slate)
=================================================
Build training dataset untuk MOKO Coder dari berbagai sumber.

Sumber data:
  1. finetune/moko_datasets/*.jsonl  (downloaded datasets)
  2. .moko_omni/                     (knowledge base entries)

Output:
  finetune/moko_datasets/moko_coder_dataset.jsonl (merged, deduplicated)

Usage:
  python3 dataset_builder.py --status     # Lihat status
  python3 dataset_builder.py --build      # Build/merge dataset
  python3 dataset_builder.py --validate   # Validate format
"""

import json
import sys
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Generator

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = Path(__file__).parent / "moko_datasets"
OUTPUT_FILE = DATASET_DIR / "moko_coder_dataset.jsonl"
OMNI_DIR = PROJECT_DIR / ".moko_omni"

SYSTEM_PROMPT = (
    "You are MOKO Coder, an expert AI programming assistant built for MOKO IDE.\n"
    "Motto: \"Kode yang efisien, solusi yang cerdas.\"\n"
    "Always output complete, runnable code. Use proper error handling."
)


def stream_jsonl(filepath: Path) -> Generator[dict, None, None]:
    """Stream JSONL file baris per baris."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_valid_chatml(sample: dict) -> bool:
    """Validasi format ChatML."""
    msgs = sample.get("messages", [])
    if len(msgs) < 2:
        return False

    roles = [m.get("role") for m in msgs]
    if "user" not in roles:
        return False
    if "assistant" not in roles and roles[-1] != "assistant":
        return False

    # System prompt harus ada
    if msgs[0].get("role") != "system":
        return False

    return True


def quality_check(sample: dict) -> tuple[bool, str]:
    """Quality check 1 sample. Return (pass, reason)."""
    msgs = sample.get("messages", [])

    for m in msgs:
        content = m.get("content", "")
        if not content or not content.strip():
            return False, "empty content"

        # Terlalu pendek
        if m.get("role") == "assistant" and len(content.strip()) < 10:
            return False, "assistant response too short"

        # Junk detection
        junk = ["I cannot", "I can't", "I'm sorry", "As an AI",
                "I don't have", "I am unable", "I'm not able"]
        if any(j in content for j in junk):
            return False, "junk response"

    return True, "ok"


def deduplicate(samples: List[dict]) -> List[dict]:
    """Deduplicate berdasarkan user message hash."""
    seen = set()
    unique = []
    for s in samples:
        user_msg = ""
        for m in s.get("messages", []):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        key = hashlib.md5(user_msg.strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def build_dataset() -> int:
    """Build merged dataset dari semua sumber."""
    all_samples = []

    # Source 1: Downloaded datasets
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.glob("*.jsonl")):
            if f.name == "moko_coder_dataset.jsonl":
                continue
            count = 0
            for sample in stream_jsonl(f):
                if is_valid_chatml(sample):
                    passed, reason = quality_check(sample)
                    if passed:
                        all_samples.append(sample)
                        count += 1
            if count > 0:
                print(f"  📄 {f.name}: {count:,} samples")

    # Source 2: .moko_omni knowledge base
    for domain_dir in sorted(OMNI_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        count = 0
        for entry_file in sorted(domain_dir.glob("*.jsonl")):
            for sample in stream_jsonl(entry_file):
                # Convert knowledge entry to ChatML if needed
                if "messages" not in sample:
                    # Auto-convert from knowledge format
                    text = sample.get("text", sample.get("content", ""))
                    if len(text) < 50:
                        continue
                    sample = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Explain this concept:\n\n{text[:200]}"},
                            {"role": "assistant", "content": text},
                        ]
                    }
                if is_valid_chatml(sample):
                    passed, reason = quality_check(sample)
                    if passed:
                        all_samples.append(sample)
                        count += 1
        if count > 0:
            print(f"  📚 .moko_omni/{domain_dir.name}: {count:,} samples")

    if not all_samples:
        print("  ❌ No samples found")
        return 0

    print(f"\n  📊 Total raw: {len(all_samples):,}")

    # Deduplicate
    before = len(all_samples)
    all_samples = deduplicate(all_samples)
    after = len(all_samples)
    print(f"  🔀 After dedup: {after:,} ({before - after:,} duplicates)")

    # Save raw
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"  💾 Saved: {OUTPUT_FILE.name} ({after:,} samples, {size_mb:.1f} MB)")

    # Convert to HEX format
    hex_file = OUTPUT_FILE.parent / "moko_coder_hex.jsonl"
    print(f"\n  🔧 Converting to HEX format...")
    try:
        from moko_hex_encoder import MokoDatasetConverter
        converter = MokoDatasetConverter()
        result = converter.convert_and_validate(OUTPUT_FILE, hex_file)
        hex_size = hex_file.stat().st_size / 1024 / 1024
        print(f"  💾 Saved: {hex_file.name} ({result['total']:,} samples, {hex_size:.1f} MB)")
        print(f"  ✅ Roundtrip: {result['roundtrip_ok']}/{result['total']}")
    except ImportError:
        print(f"  ⚠️  moko_hex_encoder not found, skipping HEX conversion")

    return after


def validate_dataset():
    """Validate format dataset."""
    if not OUTPUT_FILE.exists():
        print("  ❌ Dataset not found. Run --build first.")
        return

    total = 0
    valid = 0
    errors = []

    for i, sample in enumerate(stream_jsonl(OUTPUT_FILE)):
        total += 1
        passed, reason = is_valid_chatml(sample), quality_check(sample)

        if passed and quality_check(sample)[0]:
            valid += 1
        else:
            if len(errors) < 5:
                errors.append(f"  Line {i+1}: {reason}")

    print(f"\n  Dataset: {OUTPUT_FILE.name}")
    print(f"  Total:   {total:,}")
    print(f"  Valid:   {valid:,}")
    if errors:
        print(f"  Errors:")
        for e in errors:
            print(f"    {e}")


def show_status():
    """Tampilkan status dataset."""
    print("\n" + "=" * 60)
    print("  MOKO Coder — Dataset Status")
    print("=" * 60)

    # Per-dataset files
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.glob("*.jsonl")):
            if f.name == "moko_coder_dataset.jsonl":
                continue
            count = sum(1 for _ in open(f, encoding="utf-8"))
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:40s} {count:>8,} samples  {size_mb:>6.1f} MB")

    # Merged dataset
    if OUTPUT_FILE.exists():
        count = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
        size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
        print(f"  {'─' * 56}")
        print(f"  {'moko_coder_dataset.jsonl':40s} {count:>8,} samples  {size_mb:>6.1f} MB")
    else:
        print(f"\n  ❌ Dataset not built yet")

    # .moko_omni
    print(f"\n  📚 Knowledge Base (.moko_omni/):")
    if OMNI_DIR.exists():
        for d in sorted(OMNI_DIR.iterdir()):
            if d.is_dir():
                entry_count = sum(1 for _ in d.glob("*.jsonl"))
                print(f"    {d.name:20s} {entry_count:>6,} entries")
    else:
        print(f"    (empty)")

    print()


def main():
    parser = argparse.ArgumentParser(description="MOKO Coder Dataset Builder v3")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--build", action="store_true", help="Build/merge dataset")
    parser.add_argument("--validate", action="store_true", help="Validate format")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.build:
        print("\n🔨 Building MOKO Coder dataset...")
        count = build_dataset()
        if count > 0:
            print(f"\n✅ Dataset ready: {count:,} samples")
        else:
            print("\n❌ No samples found")
    elif args.validate:
        validate_dataset()
    else:
        show_status()


if __name__ == "__main__":
    main()
