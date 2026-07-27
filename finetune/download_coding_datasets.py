"""
MOKO Coder — Coding Dataset Downloader
========================================
Download coding datasets dari HuggingFace, convert ke format ChatML MOKO,
filter kualitas, dan simpan siap untuk fine-tuning.

Dataset yang didukung:
  1. CodeAlpaca-20K      (22K coding instructions, ringan)
  2. OpenHermes-2.5      (1M+ instructions, ada coding section)
  3. Code-Feedback        (instruction + code feedback)
  4. Magicoder-Evol-Instruct (code evolution instructions)
  5. Custom JSONL         (file lokal)

Usage:
  python3 download_coding_datasets.py                    # Download semua
  python3 download_coding_datasets.py --dataset codealpaca  # Download 1 dataset
  python3 download_coding_datasets.py --max-samples 5000    # Limit per dataset
  python3 download_coding_datasets.py --merge             # Merge ke 1 file
  python3 download_coding_datasets.py --stats             # Tampilkan statistik
"""

import os
import sys
import json
import time
import hashlib
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Generator
from urllib.request import urlretrieve
from urllib.error import HTTPError

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).parent.parent
FINETUNE_DIR = Path(__file__).parent
DATASET_DIR = FINETUNE_DIR / "moko_datasets"
RAW_DIR = DATASET_DIR / "raw"
OUTPUT_FILE = DATASET_DIR / "moko_coder_dataset.jsonl"

# ═══════════════════════════════════════════════════════════════════════════
# MOKO CODER IDENTITY (sama dengan moko_finetune.py)
# ═══════════════════════════════════════════════════════════════════════════

MOKO_CODER_SYSTEM = """You are MOKO Coder, an expert AI programming assistant built for MOKO IDE.

IDENTITY:
- Name: MOKO Coder
- Version: 1.0.0
- Built for: MOKO IDE (AI-Powered Development Environment)
- Base model: Qwen2.5-1.5B
- Motto: "Kode yang efisien, solusi yang cerdas."

CORE CAPABILITIES:
1. Code Generation — Write clean, working code from natural language descriptions
2. Code Completion — Predict and complete code as the user types
3. Bug Detection — Find and fix bugs with precise explanations
4. Refactoring — Improve code quality without changing behavior
5. Code Review — Analyze code for issues, security, and best practices
6. Documentation — Generate docstrings, comments, and README content
7. Testing — Write unit tests and test cases
8. Architecture — Design system structure and patterns

SUPPORTED LANGUAGES (prioritized):
- Python, JavaScript, TypeScript, HTML/CSS, SQL
- Bash/Shell, JSON/YAML, Markdown
- C, C++, Java, Go, Rust (basic)

RULES:
1. Always output complete, runnable code
2. Use proper error handling and edge case management
3. Follow language-specific conventions and style guides
4. Keep responses concise — code first, explanation when asked
5. Use meaningful variable and function names
6. Include type hints when applicable (Python)
7. Prefer composition over inheritance
8. Write code that is easy to test and debug
9. Never output code with syntax errors
10. If unsure, ask for clarification rather than guessing

RESPONSE FORMAT:
- For code generation: output the code block directly
- For questions about code: brief explanation + code
- For debugging: identify the bug + fix + explanation
- For refactoring: show before/after with improvements noted"""

# ═══════════════════════════════════════════════════════════════════════════
# DATASET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

DATASETS = {
    "codealpaca": {
        "repo": "sahil2801/CodeAlpaca-20k",
        "filename": "code_alpaca_20k.json",
        "description": "20K coding instructions (Python-focused)",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "max_samples": 5000,
        "priority": 1,
    },
    "openhermes": {
        "repo": "teknium/OpenHermes-2.5",
        "filename": "openhermes2_5.json",
        "description": "1M+ instructions, coding-heavy mix",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "max_samples": 8000,
        "priority": 1,
        "filter_keywords": [
            "code", "function", "class", "python", "javascript", "program",
            "algorithm", "debug", "error", "sql", "api", "git", "compile",
            "implement", "write a", "create a", "build", "def ", "import ",
            "```", "html", "css", "react", "node", "database", "query",
        ],
    },
    "magicoder": {
        "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "filename": "data-evol_instruct-decontaminated.jsonl",
        "description": "110K code evolution instructions",
        "fields": {"instruction": "instruction", "input": "input", "output": "output"},
        "max_samples": 5000,
        "priority": 2,
    },
    "glaive_code": {
        "repo": "glaiveai/glaive-code-assistant",
        "filename": "c9bc9129-eba0-4b10-8292-4ae70fc7fa0d.json",
        "description": "Code assistant conversations",
        "fields": {"instruction": "question", "input": "", "output": "answer"},
        "max_samples": 3000,
        "priority": 3,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DOWNLOAD ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def download_dataset_file(repo_id: str, filename: str, dest: Path) -> bool:
    """Download file dari HuggingFace repos menggunakan HF API."""
    print(f"  📡 Downloading {filename} from {repo_id}...")

    # Setup cache dir agar tidak mengganggu project
    cache_dir = Path(tempfile.mkdtemp(prefix="moko_hf_"))

    # Strategy 1: Try direct HF API (works for most repos)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        downloaded_path = Path(downloaded)
        # Move to destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(downloaded_path), str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✅ Downloaded via HF Hub API ({size_mb:.1f} MB)")
        # Cleanup temp cache
        shutil.rmtree(str(cache_dir), ignore_errors=True)
        return True
    except Exception as e1:
        print(f"  ⚠️  HF Hub failed: {e1}")

    # Strategy 2: Try direct URL download
    api_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    try:
        urlretrieve(api_url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  ✅ Downloaded via direct URL ({size_mb:.1f} MB)")
        return True
    except HTTPError as e:
        print(f"  ⚠️  HTTP error: {e}")
    except Exception as e2:
        print(f"  ⚠️  Direct download failed: {e2}")

    # Strategy 3: Try alternative filenames
    alt_names = [
        filename.replace(".json", ".jsonl"),
        filename.replace(".json", ".parquet"),
        "data/" + filename,
        "data/" + filename.replace(".json", ".jsonl"),
    ]
    for alt in alt_names:
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=alt,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            downloaded_path = Path(downloaded)
            target = dest.with_suffix(downloaded_path.suffix)
            shutil.copy2(str(downloaded_path), str(target))
            print(f"  ✅ Found as: {alt}")
            shutil.rmtree(str(cache_dir), ignore_errors=True)
            return True
        except:
            continue

    shutil.rmtree(str(cache_dir), ignore_errors=True)
    print(f"  ❌ Could not download {filename} from {repo_id}")
    return False


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


def stream_json_array(filepath: Path) -> Generator[dict, None, None]:
    """Stream JSON array file (satu objek besar)."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                # Maybe it's a single entry
                yield data
        except json.JSONDecodeError:
            # Try streaming as JSONL fallback
            f.seek(0)
            yield from stream_jsonl(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def has_code_markers(text: str) -> bool:
    """Cek apakah teks mengandung code markers."""
    markers = ["```", "def ", "class ", "function ", "import ", "from ",
               "const ", "let ", "var ", "SELECT ", "CREATE TABLE",
               "<div", "<html", "if (", "for (", "while ("]
    return any(m in text for m in markers)


def is_coding_related(instruction: str, output: str, keywords: List[str] = None) -> bool:
    """Cek apakah sample terkait coding."""
    if keywords:
        combined = (instruction + " " + output).lower()
        return any(kw.lower() in combined for kw in keywords)

    # Default: check for code markers in output
    return has_code_markers(output)


def quality_filter(instruction: str, output: str) -> bool:
    """Filter kualitas sample."""
    # Terlalu pendek
    if len(instruction.strip()) < 10:
        return False
    if len(output.strip()) < 20:
        return False

    # Output terlalu pendek (mungkin bukan code)
    if len(output.strip().split()) < 3:
        return False

    # Junk detection
    junk_patterns = [
        "I cannot", "I can't", "I'm sorry", "As an AI",
        "I don't have", "I am unable",
    ]
    if any(jp in output for jp in junk_patterns):
        return False

    return True


def deduplicate(samples: List[dict]) -> List[dict]:
    """Deduplicate berdasarkan instruction hash."""
    seen = set()
    unique = []
    for s in samples:
        key = hashlib.md5(
            s.get("instruction", "").strip().lower().encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# ═══════════════════════════════════════════════════════════════════════════
# CONVERT TO CHATML
# ═══════════════════════════════════════════════════════════════════════════

def to_chatml(sample: dict, fields: dict) -> dict:
    """Konversi 1 sample ke format ChatML MOKO."""
    instruction = sample.get(fields["instruction"], "").strip()
    inp = sample.get(fields.get("input", ""), "").strip()
    output = sample.get(fields["output"], "").strip()

    # Gabungkan instruction + input
    user_msg = instruction
    if inp:
        user_msg = f"{instruction}\n\nInput:\n{inp}"

    return {
        "messages": [
            {"role": "system", "content": MOKO_CODER_SYSTEM},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS DATASET
# ═══════════════════════════════════════════════════════════════════════════

def process_dataset(
    name: str,
    config: dict,
    max_samples: int = None,
    filter_keywords: List[str] = None,
) -> List[dict]:
    """Download, filter, dan convert 1 dataset."""
    limit = max_samples or config["max_samples"]
    fields = config["fields"]

    # Cari file lokal
    local_file = RAW_DIR / config["filename"]
    if not local_file.exists():
        # Download
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        success = download_dataset_file(config["repo"], config["filename"], RAW_DIR)
        if not success:
            return []

    # Stream dan process
    samples = []
    total_read = 0

    print(f"  📖 Processing {local_file.name}...")

    try:
        # Try JSONL first, then JSON array
        ext = local_file.suffix.lower()
        if ext == ".jsonl":
            reader = stream_jsonl(local_file)
        else:
            reader = stream_json_array(local_file)

        for item in reader:
            total_read += 1

            # Extract fields
            instruction = item.get(fields["instruction"], "")
            inp = item.get(fields.get("input", ""), "")
            output = item.get(fields["output"], "")

            if not instruction or not output:
                continue

            # Filter: coding related?
            if filter_keywords and not is_coding_related(instruction, output, filter_keywords):
                continue

            # Filter: quality?
            if not quality_filter(instruction, output):
                continue

            # Convert to ChatML
            chatml = to_chatml(item, fields)
            samples.append(chatml)

            if len(samples) >= limit:
                break

            if total_read % 1000 == 0:
                print(f"    ... read {total_read:,} entries, kept {len(samples):,}")

    except Exception as e:
        print(f"  ❌ Error processing {local_file.name}: {e}")

    print(f"  ✅ {name}: {len(samples):,} quality samples from {total_read:,} entries")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# MERGE & SAVE
# ═══════════════════════════════════════════════════════════════════════════

def save_samples(samples: List[dict], filepath: Path):
    """Simpan samples ke JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"  💾 Saved: {filepath} ({len(samples):,} samples, {size_mb:.1f} MB)")


def merge_all_datasets(datasets_used: List[str], max_total: int = 10000):
    """Merge semua dataset yang sudah diproses."""
    all_samples = []

    for name in datasets_used:
        file = DATASET_DIR / f"{name}.jsonl"
        if file.exists():
            for item in stream_jsonl(file):
                all_samples.append(item)

    # Deduplicate
    before = len(all_samples)
    all_samples = deduplicate(all_samples)
    after = len(all_samples)
    print(f"\n🔀 Dedup: {before:,} → {after:,} ({before - after:,} duplicates removed)")

    # Limit
    if len(all_samples) > max_total:
        all_samples = all_samples[:max_total]
        print(f"✂️  Limited to {max_total:,} samples")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_samples)

    # Save merged
    save_samples(all_samples, OUTPUT_FILE)
    
    # Auto-convert to HEX format
    hex_file = OUTPUT_FILE.parent / "moko_coder_hex.jsonl"
    print(f"\n🔧 Converting to HEX format...")
    try:
        from moko_hex_encoder import MokoDatasetConverter
        converter = MokoDatasetConverter()
        result = converter.convert_and_validate(OUTPUT_FILE, hex_file)
        hex_size = hex_file.stat().st_size / 1024 / 1024
        print(f"  💾 Saved: {hex_file.name} ({result['total']:,} samples, {hex_size:.1f} MB)")
        print(f"  ✅ Roundtrip: {result['roundtrip_ok']}/{result['total']}")
    except ImportError:
        print(f"  ⚠️  moko_hex_encoder not found, run: python3 finetune/moko_hex_encoder.py --convert")
    
    return len(all_samples)


def show_stats():
    """Tampilkan statistik dataset."""
    print("\n" + "=" * 60)
    print("  MOKO Coder — Dataset Statistics")
    print("=" * 60)

    # Per-dataset files
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
        print(f"\n  ❌ Merged dataset not built yet")
        print(f"  Run: python3 download_coding_datasets.py --merge")

    # Raw downloads
    if RAW_DIR.exists():
        raw_files = list(RAW_DIR.glob("*"))
        if raw_files:
            raw_size = sum(f.stat().st_size for f in raw_files) / 1024 / 1024
            print(f"\n  📦 Raw downloads: {len(raw_files)} files, {raw_size:.1f} MB")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download & process coding datasets for MOKO Coder fine-tuning"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS.keys()),
        help="Download single dataset (default: all)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int, default=None,
        help="Max samples per dataset (overrides default)",
    )
    parser.add_argument(
        "--max-total",
        type=int, default=10000,
        help="Max total samples in merged dataset (default: 10000)",
    )
    parser.add_argument(
        "--merge", "-m",
        action="store_true",
        help="Merge all per-dataset files into moko_coder_dataset.jsonl",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show dataset statistics",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete raw downloads (keep processed files)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MOKO Coder — Coding Dataset Downloader")
    print("=" * 60)

    if args.stats:
        show_stats()
        return

    if args.clean:
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
            print("🧹 Raw downloads cleaned")
        return

    # Determine which datasets to process
    if args.dataset:
        to_process = {args.dataset: DATASETS[args.dataset]}
    else:
        # Sort by priority
        to_process = dict(
            sorted(DATASETS.items(), key=lambda x: x[1]["priority"])
        )

    # Process each dataset
    processed = []
    for name, config in to_process.items():
        print(f"\n{'─' * 60}")
        print(f"📥 Dataset: {name}")
        print(f"   {config['description']}")
        print(f"{'─' * 60}")

        samples = process_dataset(
            name=name,
            config=config,
            max_samples=args.max_samples,
            filter_keywords=config.get("filter_keywords"),
        )

        if samples:
            # Save per-dataset
            out_file = DATASET_DIR / f"{name}.jsonl"
            save_samples(samples, out_file)
            processed.append(name)

    # Merge if requested or if multiple datasets processed
    if args.merge or len(processed) > 1:
        print(f"\n{'═' * 60}")
        print(f"🔀 Merging {len(processed)} datasets...")
        total = merge_all_datasets(processed, max_total=args.max_total)
        print(f"\n✅ Final dataset: {total:,} samples → {OUTPUT_FILE}")

    # Always show stats at end
    show_stats()


if __name__ == "__main__":
    main()
