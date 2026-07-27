#!/usr/bin/env python3
"""
MOKO Vision: Local Multimodal Test Script
=========================================
Verifies if the local llama-server handles visual (multimodal) inputs.
Usage:
  ./bin/python moko_core/moko_benchmark/test_vision.py --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moko_config import settings
from moko_tools.vision_helper import MokoVision
from moko_inference.server_manager import MokoLocalInferenceServer

def main():
    parser = argparse.ArgumentParser(description="Moko OS local vision verification.")
    parser.add_argument("--image", required=True, help="Path to local image file to analyze.")
    parser.add_argument("--query", default="Jelaskan isi gambar ini secara singkat.", help="Text query to ask.")
    args = parser.parse_args()

    print("🔍 Checking server status...")
    status = MokoLocalInferenceServer.get_server_status(settings.MOKO_LLM_PORT)
    print(f"Status MOKO server: {status.upper()}")

    # Check mmproj config
    mmproj_path = getattr(settings, 'MODEL_MMPROJ_GGUF_PATH', '')
    if not mmproj_path or not Path(mmproj_path).exists():
        print(f"⚠️  MODEL_MMPROJ_GGUF_PATH ({mmproj_path}) tidak ditemukan atau tidak dikonfigurasi.")
        print("Pastikan file projector GGUF diletakkan di workspace dan settings.py telah diperbarui.")
        
    print(f"🧪 Membaca gambar: {args.image}...")
    print(f"💬 Query: {args.query}")
    print("⏳ Menghubungi Sovereign Engine...")
    
    resp = MokoVision.analyze_image(args.image, args.query)
    
    print("\n" + "═"*50)
    print(" 👁️ MOKO VISION RESPONSE:")
    print("═"*50)
    print(resp)
    print("═"*50 + "\n")

if __name__ == "__main__":
    main()
