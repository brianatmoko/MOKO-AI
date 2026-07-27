#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MOKO OS — System Limits Fixer
# Jalankan SATU KALI dengan sudo untuk memperbaiki:
#   1. inotify watch limit (mencegah error "[Errno 28] inotify watch limit reached")
#   2. mlock limit (mencegah warning "failed to mlock N-byte buffer")
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  MOKO OS — System Limits Fixer"
echo "================================================"

# 1. inotify: Naikkan jumlah file yang bisa dimonitor secara real-time
echo "[1/3] Menaikkan inotify limit..."
echo 'fs.inotify.max_user_watches=524288'    >> /etc/sysctl.conf
echo 'fs.inotify.max_user_instances=512'     >> /etc/sysctl.conf
echo 'fs.inotify.max_queued_events=131072'   >> /etc/sysctl.conf
sysctl -p
echo "  ✅ inotify limit: 524288 watches, 512 instances"

# 2. mlock: Naikkan batas memori yang bisa di-lock oleh process biasa
# Ini mencegah warning "failed to mlock N-byte buffer: Cannot allocate memory"
echo "[2/3] Menaikkan mlock limit..."
cat >> /etc/security/limits.conf << 'EOF'
# MOKO OS — Allow model mlock for inference
*               soft    memlock         unlimited
*               hard    memlock         unlimited
EOF
echo "  ✅ mlock limit: unlimited"

# 3. Tambah entry bersih di /etc/sysctl.conf (tanpa duplikat)
echo "[3/3] Menghapus duplikat entri sysctl..."
# Deduplicate sysctl.conf
awk '!seen[$0]++' /etc/sysctl.conf > /tmp/sysctl_clean.conf && mv /tmp/sysctl_clean.conf /etc/sysctl.conf
echo "  ✅ sysctl.conf dibersihkan"

echo ""
echo "================================================"
echo "  SELESAI. Nilai sekarang:"
echo "  inotify max_user_watches: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo "  inotify max_user_instances: $(cat /proc/sys/fs/inotify/max_user_instances)"
echo "================================================"
echo ""
echo "⚡ Jalankan 'moko' lagi. Tidak perlu reboot."
