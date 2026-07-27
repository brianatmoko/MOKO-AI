#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  MOKO RAM WARMUP — Memuat Otak AI ke Dalam Darah (RAM)     ║
# ║  Jalankan dengan: sudo bash moko-warmup.sh                  ║
# ╚══════════════════════════════════════════════════════════════╝

set -e

RAM_DISK="/mnt/moko_ram"
RAM_SIZE="9500M"
MODELS_SRC="/home/brianatmokoo/.ollama/models"

# Cek apakah sudah ter-mount
if mountpoint -q "$RAM_DISK"; then
    echo "⚡ RAM Disk MOKO sudah aktif di $RAM_DISK"
    df -h "$RAM_DISK"
    exit 0
fi

echo ""
echo "🧠 MOKO RAM WARMUP — Memuat otak AI ke dalam memori sistem..."
echo "   Ukuran RAM Disk : $RAM_SIZE"
echo "   Tujuan          : $RAM_DISK"
echo ""

# Buat direktori mount
mkdir -p "$RAM_DISK"

# Mount tmpfs (RAM Disk)
echo "[1/5] Membuat RAM Disk $RAM_SIZE di $RAM_DISK..."
mount -t tmpfs -o size=$RAM_SIZE,mode=0755 tmpfs "$RAM_DISK"

# Buat struktur direktori yang sama dengan Ollama models
echo "[2/5] Menyiapkan struktur direktori..."
mkdir -p "$RAM_DISK/blobs"
mkdir -p "$RAM_DISK/manifests"

# Salin manifests seluruhnya (kecil, penting untuk mengenali model)
echo "[3/5] Menyalin manifest model..."
cp -r "$MODELS_SRC/manifests/" "$RAM_DISK/"

# Salin blobs yang dibutuhkan (Qwen, Embedder)
echo "[4/5] Menyalin model ke RAM (ini butuh 1-2 menit)..."

BLOBS=(
    # Qwen 2.5 7B Abliterated — GGUF utama (~4.7 GB)
    "sha256-99c3bddafb9cc198190203cf96cc4cc43b897f1a94eaa7330eacf29c6c9718e0"
    "sha256-eb4402837c7829a690fa845de4d7f3fd842c2adee476d5341da8a46ea9255175"
    "sha256-66b9ea09bd5b7099cbb4fc820f31b575c0366fa439b08245566692c6784e281e"
    "sha256-832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
    # nomic-embed-text — Embedder (~0.27 GB)
    "sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
    "sha256-c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    "sha256-ce4a164fc04605703b485251fe9f1a181688ba0eb6badb80cc6335c0de17ca0d"
)

TOTAL=${#BLOBS[@]}
IDX=0
for BLOB in "${BLOBS[@]}"; do
    IDX=$((IDX + 1))
    SRC="$MODELS_SRC/blobs/$BLOB"
    DST="$RAM_DISK/blobs/$BLOB"
    
    if [ ! -f "$SRC" ]; then
        echo "  [SKIP] $BLOB (tidak ditemukan di disk)"
        continue
    fi
    
    SIZE_MB=$(du -m "$SRC" | cut -f1)
    echo "  [$IDX/$TOTAL] $BLOB... (${SIZE_MB} MB)"
    cp "$SRC" "$DST"
done

# Set kepemilikan agar user brianatmokoo bisa membaca
chown -R brianatmokoo:brianatmokoo "$RAM_DISK" 2>/dev/null || true
chmod -R 755 "$RAM_DISK"

echo ""
echo "[5/5] Verifikasi..."
df -h "$RAM_DISK"
echo ""
echo "✅ MOKO RAM WARMUP SELESAI!"
echo "   Otak AI sekarang hidup di dalam RAM sistem Anda."
echo "   Gunakan OLLAMA_MODELS=$RAM_DISK untuk mengarahkan Ollama ke RAM Disk."
echo ""
echo "   Untuk mematikan RAM Disk:"
echo "   sudo bash /home/brianatmokoo/moko-research/moko-ramdisk-stop.sh"
