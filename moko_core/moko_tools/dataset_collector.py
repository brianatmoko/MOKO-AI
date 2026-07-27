"""
MOKO Bulk Dataset Collector & Auto-Encryptor
==============================================
Tujuan: Membantu pengguna mengunduh dataset pemrograman, website, dan kripto
        skala besar secara legal, lalu langsung mengenkripsinya ke dalam format
        MOKO Crypto Engine (.moko_crypto/) sebelum menghapus file mentahnya.

Mendukung:
  - StackOverflow QA Subsets (Pemrograman, Kriptografi)
  - Hugging Face Dataset Stream (Python, JS, Security)
  - Wikipedia Computer Science Subsets (Bahasa Indonesia / Inggris)
"""

import os
import sys
import time
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Setup path agar bisa import moko_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moko_memory.crypto_storage import CryptoStorageEngine


# Daftar Seed Dataset Terkurasi & Ekonomis
DATASET_SEEDS: Dict[str, Dict[str, str]] = {
    "cryptography_qa": {
        "url": "https://archive.org/download/stackexchange/cryptography.stackexchange.com.7z",
        "description": "Tanya jawab kriptografi formal dari StackExchange (7z format, ~15MB)",
        "domain": "cybersecurity"
    },
    "reverse_engineering_qa": {
        "url": "https://archive.org/download/stackexchange/reverseengineering.stackexchange.com.7z",
        "description": "Diskusi reverse engineering, exploit, malware analisis (7z format, ~10MB)",
        "domain": "cyberoffensive"
    },
    "security_qa": {
        "url": "https://archive.org/download/stackexchange/security.stackexchange.com.7z",
        "description": "Tanya jawab keamanan informasi & web security (7z format, ~70MB)",
        "domain": "cybersecurity"
    },
    "indonesian_wiki_cs": {
        "url": "https://raw.githubusercontent.com/brianatmokoo/MOKO_OS_Project/main/crawler_data/seeds.txt", # Fallback seeds user
        "description": "Seed web CS & programming Indonesia",
        "domain": "general"
    }
}


class MokoDatasetCollector:
    """
    Kolektor data otomatis untuk MOKO OS.
    Mengunduh, mengekstrak, mengenkripsi ke .moko_crypto, dan menghapus sisa file.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.crypto_engine = CryptoStorageEngine(domain="general") # Default domain
        self.temp_dir = Path(tempfile.mkdtemp(prefix="moko_dataset_"))

    def cleanup_temp(self):
        """Hapus sisa-sisa file temp unduhan secara bersih"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"  🧹 [Collector] File temp dihapus bersih: {self.temp_dir.name}")

    def download_and_encrypt(self, dataset_key: str) -> bool:
        """Unduh dataset spesifik, enkripsi per baris/dokumen, lalu bersihkan SSD"""
        if dataset_key not in DATASET_SEEDS:
            print(f"❌ Key dataset '{dataset_key}' tidak ditemukan.")
            return False

        seed = DATASET_SEEDS[dataset_key]
        url = seed["url"]
        domain = seed["domain"]
        
        # Atur domain enkripsi yang tepat
        self.crypto_engine = CryptoStorageEngine(domain=domain)

        print(f"📡 [Collector] Memulai unduhan/pembacaan: {dataset_key}")
        print(f"   Deskripsi: {seed['description']}")
        print(f"   Target Domain: .moko_crypto/{domain}/")

        temp_file_path = self.temp_dir / f"{dataset_key}_raw.tmp"

        try:
            # 1. Download atau baca lokal
            if dataset_key == "indonesian_wiki_cs":
                local_seed = self.workspace_dir / "crawler_data" / "seeds.txt"
                if local_seed.exists():
                    shutil.copy(local_seed, temp_file_path)
                    print(f"   ✅ Berhasil memuat berkas seeds lokal")
                else:
                    raise FileNotFoundError(f"File seeds lokal tidak ada di {local_seed}")
            else:
                t0 = time.time()
                urllib.request.urlretrieve(url, temp_file_path)
                elapsed_dl = time.time() - t0
                size_mb = temp_file_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ Selesai unduh: {size_mb:.2f} MB dalam {elapsed_dl:.1f} detik")

            # 2. Proses enkripsi & ingest
            print(f"🔒 [Collector] Memulai enkripsi & penyimpanan ke database MOKO...")
            t1 = time.time()
            
            # Membuat dummy vector 768-dimensi untuk menghemat komputasi embedding
            dummy_vector = [0.0] * 768
            
            # Membaca file sebagai teks mentah dan meng-ingest per baris/chunk
            chunk_size = 5 * 1024 # 5 KB chunk
            count = 0
            with open(temp_file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content_str = chunk.decode('utf-8', errors='replace').strip()
                    if len(content_str) > 20:
                        self.crypto_engine.ingest(
                            text=content_str,
                            fp32_vector=dummy_vector,
                            source_name=f"bulk_{dataset_key}",
                            log_number=count
                        )
                        count += 1

            elapsed_enc = time.time() - t1
            print(f"   ✅ Enkripsi Selesai: {count} chunk di-ingest ke .moko_crypto/{domain}/ ({elapsed_enc:.1f} detik)")

            # 3. Clean up raw file
            if temp_file_path.exists():
                temp_file_path.unlink()
                print("   🗑️  File unduhan mentah langsung dihapus (Zero Retention)")

            return True

        except Exception as e:
            print(f"❌ Gagal mengunduh/proses dataset: {e}")
            return False
        finally:
            self.cleanup_temp()


if __name__ == "__main__":
    print("=== MOKO BULK DATASET COLLECTOR & AUTO-ENCRYPTOR ===")
    print("Dataset yang tersedia:")
    for key, val in DATASET_SEEDS.items():
        print(f"  - [{key}]: {val['description']}")
    
    # Menjalankan unduhan default jika dijalankan dari shell
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "indonesian_wiki_cs" # Default lightweight seed

    workspace = "/home/brianatmokoo/Documents/Linux/MOKO_OS_Project"
    collector = MokoDatasetCollector(workspace)
    collector.download_and_encrypt(target)
