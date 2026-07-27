"""
MOKO Distill Trainer — Konverter Dataset & Trigger Fine-tuning Periodik
=======================================================================
Menjembatani loop Guru-Murid ke pipeline fine-tuning LoRA yang sudah ada.

Alur kerja (sesuai riset 23_REVISI_MANDOR_API_MURID_LOKAL.md):
  1. Baca distill_dataset/moko_distill_samples.jsonl (diisi oleh interaction_logger.py)
  2. Filter dan konversi sampel ke format ChatML SFT
  3. Merge dengan dataset utama di finetune/moko_datasets/
  4. Panggil lora_trainer.run_training() untuk fine-tuning inkremental
  5. Setelah selesai, update status sehingga local confidence bisa naik

Mode operasi:
  - One-shot  : python distill_trainer.py --run
  - Daemon     : python distill_trainer.py --watch  (check setiap N menit)
  - Status     : python distill_trainer.py --status

Pipeline distilasi ini adalah inti dari cara MOKO 1.5B bisa setara 600B:
setiap interaksi sukses Guru API → sampel SFT → fine-tuning lokal →
confidence lokal naik → lebih banyak tugas dikerjakan mandiri.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("moko_distill_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parents[4]
FINETUNE_DIR   = PROJECT_DIR / "finetune"
DISTILL_JSONL  = PROJECT_DIR / "distill_dataset" / "moko_distill_samples.jsonl"
SFT_OUTPUT     = FINETUNE_DIR / "moko_datasets" / "moko_distill_sft.jsonl"
PROCESSED_LOG  = PROJECT_DIR / "distill_dataset" / ".moko_distill_processed.json"

# ── Quality filters ─────────────────────────────────────────────────────────
MIN_COMPLEXITY_THRESHOLD = 0.0   # Terima semua kompleksitas
MIN_REASONING_LEN        = 10    # Minimal karakter reasoning trace
MIN_CODE_LEN             = 30    # Minimal karakter kode output
MIN_SAMPLES_TO_TRIGGER   = 20    # Minimal sampel baru sebelum training dipicu

# ── SFT format template (ChatML) ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Kamu adalah MOKO Coder, asisten AI yang ahli dalam pemrograman Python. "
    "Ketika diberikan tugas, pikirkan langkah-langkah penyelesaian secara terstruktur "
    "lalu tuliskan kode Python yang bersih, efisien, dan berhasil diverifikasi."
)


def _load_processed_ids() -> set[str]:
    """Muat set ID sampel yang sudah diproses ke SFT dataset."""
    try:
        if PROCESSED_LOG.exists():
            with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("processed", []))
    except Exception:
        pass
    return set()


def _save_processed_ids(ids: set[str]) -> None:
    """Simpan set ID sampel yang sudah diproses."""
    try:
        PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
            json.dump({"processed": list(ids)}, f)
    except Exception as e:
        logger.warning(f"Gagal menyimpan processed IDs: {e}")


def _sample_to_sft(sample: dict) -> dict | None:
    """
    Konversi satu DistillSample ke format SFT ChatML.

    Format output kompatibel dengan train_lora.py dan TRL SFTTrainer.
    """
    prompt = sample.get("prompt", "").strip()
    thought = sample.get("reasoning_trace", "").strip()
    code = sample.get("code_output", "").strip()
    passed = sample.get("passed_guard", False)
    complexity = float(sample.get("task_complexity", 0.5))
    source = sample.get("source", "guru_api")

    # Quality gate
    if not passed:
        return None
    if len(thought) < MIN_REASONING_LEN:
        return None
    if len(code) < MIN_CODE_LEN:
        return None
    if not prompt:
        return None

    # Bobot kualitas — sampel dari Guru API lebih dipercaya
    quality_weight = 1.0
    if source == "guru_api":
        quality_weight = 1.0 + complexity * 0.5   # kompleks = lebih berharga
    elif source == "lokal":
        quality_weight = 0.8                        # lokal sedikit lebih rendah (self-play)

    # Format ChatML dengan reasoning trace
    assistant_content = (
        f"<think>\n{thought}\n</think>\n\n"
        f"```python\n{code}\n```"
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source":         source,
            "complexity":     complexity,
            "quality_weight": round(quality_weight, 3),
            "category":       sample.get("task_category", "general"),
            "timestamp":      sample.get("timestamp", 0),
        }
    }


def convert_distill_to_sft(force: bool = False) -> int:
    """
    Baca JSONL distill baru, konversi ke format SFT, append ke output file.

    Mengembalikan jumlah sampel baru yang berhasil dikonversi.
    """
    if not DISTILL_JSONL.exists():
        logger.info(f"Distill JSONL belum ada: {DISTILL_JSONL}")
        return 0

    processed_ids = _load_processed_ids() if not force else set()

    # Baca semua sampel distill
    raw_samples = []
    with open(DISTILL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    # Filter sampel baru (belum diproses)
    new_samples = []
    for s in raw_samples:
        # Gunakan kombinasi prompt + timestamp sebagai ID unik
        sample_id = f"{s.get('prompt','')[:40]}_{s.get('timestamp', 0)}"
        if sample_id not in processed_ids:
            new_samples.append((sample_id, s))

    if not new_samples:
        logger.info("Tidak ada sampel baru untuk dikonversi.")
        return 0

    logger.info(f"Menemukan {len(new_samples)} sampel baru. Mengkonversi ke SFT format...")

    # Konversi ke SFT
    SFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    new_ids = set()

    with open(SFT_OUTPUT, "a", encoding="utf-8") as fout:
        for sample_id, sample in new_samples:
            sft_item = _sample_to_sft(sample)
            if sft_item is not None:
                fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
                converted_count += 1
                new_ids.add(sample_id)

    # Simpan IDs baru
    processed_ids.update(new_ids)
    _save_processed_ids(processed_ids)

    logger.info(
        f"✅ Berhasil mengkonversi {converted_count}/{len(new_samples)} sampel baru "
        f"ke {SFT_OUTPUT.name}"
    )
    return converted_count


def get_sft_count() -> int:
    """Hitung total sampel di SFT dataset."""
    if not SFT_OUTPUT.exists():
        return 0
    count = 0
    with open(SFT_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def trigger_training(min_samples: int = MIN_SAMPLES_TO_TRIGGER) -> bool:
    """
    Picu fine-tuning LoRA jika ada cukup sampel SFT baru.

    Memanggil lora_trainer.run_training() dengan dataset distill hasil konversi.
    """
    sys.path.insert(0, str(FINETUNE_DIR))
    try:
        import lora_trainer
    except ImportError as e:
        logger.error(f"Tidak bisa import lora_trainer: {e}")
        return False

    total_sft = get_sft_count()
    logger.info(f"Total sampel SFT tersedia: {total_sft}")

    if total_sft < min_samples:
        logger.info(
            f"Belum cukup sampel ({total_sft} < {min_samples}). "
            f"Training ditunda sampai cukup data terkumpul."
        )
        return False

    logger.info(
        f"🚀 Memulai distilasi LoRA ({total_sft} sampel SFT)...\n"
        "  Ini adalah inti dari mekanisme Guru → Murid:\n"
        "  Guru API menghasilkan reasoning trace, Murid (1.5B) belajar dari contoh nyata."
    )

    # Override dataset file di lora_trainer agar menggunakan SFT distill kita
    original_dataset = lora_trainer.DATASET_FILE
    lora_trainer.DATASET_FILE = SFT_OUTPUT
    lora_trainer.MIN_SAMPLES_TO_TRAIN = min_samples

    try:
        success = lora_trainer.run_training()
        return success
    except Exception as e:
        logger.error(f"Training gagal: {e}")
        return False
    finally:
        # Kembalikan setting ke semula
        lora_trainer.DATASET_FILE = original_dataset


def print_status() -> None:
    """Tampilkan status ringkas distill pipeline."""
    distill_count = 0
    if DISTILL_JSONL.exists():
        with open(DISTILL_JSONL, "r", encoding="utf-8") as f:
            distill_count = sum(1 for line in f if line.strip())

    sft_count = get_sft_count()
    processed_ids = _load_processed_ids()

    # Load confidence DB
    conf_db_path = PROJECT_DIR / ".moko_local_confidence.json"
    confidence_info = {}
    if conf_db_path.exists():
        try:
            with open(conf_db_path, "r", encoding="utf-8") as f:
                confidence_info = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  MOKO Distill Trainer — Status Sistem")
    print("=" * 60)
    print(f"  📊 Distill samples terkumpul : {distill_count}")
    print(f"  ✅ Sampel sudah diproses      : {len(processed_ids)}")
    print(f"  📚 SFT dataset total          : {sft_count}")
    print(f"  🎯 Min. untuk trigger training : {MIN_SAMPLES_TO_TRIGGER}")
    print()

    if confidence_info:
        print("  🧠 Confidence Model Lokal per Kategori:")
        for cat, info in confidence_info.items():
            conf = info.get("confidence", 0.0)
            success = info.get("success", 0)
            total = info.get("total", 0)
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"    {cat:10s}: [{bar}] {conf:.0%}  ({success}/{total})")
    else:
        print("  🧠 Confidence lokal: belum ada data (mulai chat untuk membangun)")

    print("=" * 60 + "\n")


def run_once(force: bool = False, min_samples: int = MIN_SAMPLES_TO_TRIGGER) -> bool:
    """Jalankan satu siklus: convert → trigger training jika cukup."""
    logger.info("🔄 Menjalankan satu siklus distilasi...")
    new_samples = convert_distill_to_sft(force=force)
    if new_samples > 0 or force:
        return trigger_training(min_samples=min_samples)
    return False


def watch_daemon(interval_minutes: int = 30) -> None:
    """
    Jalankan distill trainer sebagai daemon yang mengecek secara periodik.
    Ideal dijalankan via systemd atau cron, atau langsung di terminal sebagai background.
    """
    logger.info(
        f"⏰ Distill Trainer Daemon aktif. "
        f"Checking setiap {interval_minutes} menit..."
    )
    while True:
        try:
            run_once()
        except Exception as e:
            logger.error(f"Error dalam siklus distilasi: {e}")
        time.sleep(interval_minutes * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MOKO Distill Trainer — Guru API → Murid Lokal"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",    action="store_true", help="Jalankan satu siklus distilasi")
    group.add_argument("--watch",  action="store_true", help="Jalankan daemon periodik")
    group.add_argument("--status", action="store_true", help="Tampilkan status pipeline")
    group.add_argument("--convert-only", action="store_true",
                       help="Hanya konversi dataset tanpa trigger training")

    parser.add_argument("--force", action="store_true",
                        help="Re-proses semua sampel meski sudah diproses sebelumnya")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_TO_TRIGGER,
                        help=f"Minimal sampel untuk trigger training (default: {MIN_SAMPLES_TO_TRIGGER})")
    parser.add_argument("--interval", type=int, default=30,
                        help="Interval daemon dalam menit (default: 30)")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.convert_only:
        n = convert_distill_to_sft(force=args.force)
        print(f"Dikonversi: {n} sampel baru.")
    elif args.run:
        ok = run_once(force=args.force, min_samples=args.min_samples)
        sys.exit(0 if ok else 1)
    elif args.watch:
        watch_daemon(interval_minutes=args.interval)
