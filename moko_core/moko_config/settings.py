import os
from pathlib import Path

# ==============================================================================
# MOKO OS: GLOBAL SETTINGS & CONFIGURATION
# ==============================================================================

# --- PATHS ---
# Otomatis mendeteksi folder MOKO_OS_Project di mana pun ia diletakkan (Portable Mode)
PROJECT_DIR     = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR   = PROJECT_DIR

# --- RAM DISK SUPPORT UNTUK OMNI DB ---
# Prioritas: /mnt/moko_ram (sudo warmup) > /dev/shm (tanpa sudo) > SSD
RAM_OMNI_DIR_SUDO = Path("/mnt/moko_ram/.moko_omni")
RAM_OMNI_DIR_SHM  = Path("/dev/shm/moko_omni")

if RAM_OMNI_DIR_SUDO.exists():
    OMNI_DIR = RAM_OMNI_DIR_SUDO          # RAM disk penuh (moko-warmup aktif)
elif RAM_OMNI_DIR_SHM.exists():
    OMNI_DIR = RAM_OMNI_DIR_SHM           # /dev/shm (tanpa sudo, selalu ada)
else:
    OMNI_DIR = WORKSPACE_DIR / ".moko_omni"  # Fallback SSD

SSD_OMNI_DIR = WORKSPACE_DIR / ".moko_omni"  # Selalu menunjuk ke SSD untuk sync
    
CACHE_DIR       = WORKSPACE_DIR / ".moko_cache"

# --- RAM DISK ---
# Lokasi RAM disk (tmpfs) tempat model AI dimuat saat warmup
MOKO_RAM_DISK_PATH   = Path("/mnt/moko_ram")


# MOKO COMPACT ARCHITECTURE (Phase 17):
# Q3_K_M sebagai backbone inference (2.1GB, hemat RAM)
# BF16 hanya disimpan sebagai referensi identity fingerprint (tidak dimuat ke RAM)
MOKO_BF16_MODEL_PATH     = str(PROJECT_DIR / "MOKO-AI-4B-BF16.gguf")   # Referensi identity
MOKO_Q3_MODEL_PATH       = str(PROJECT_DIR / "MOKO-AI-4B-Q3_K_M.gguf")
MODEL_MOKO_GGUF_PATH     = MOKO_Q3_MODEL_PATH    # Inference: Q3_K_M (hemat RAM)
MODEL_IDENTITY_PATH      = MOKO_BF16_MODEL_PATH  # Identity fingerprint (tidak dimuat)
MODEL_EMBEDDER_GGUF_PATH = MOKO_Q3_MODEL_PATH    # Embedding: unified Q3_K_M
# Projector Multimodal (CLIP / mmproj) untuk mendukung input gambar/vision secara lokal
MODEL_MMPROJ_GGUF_PATH  = str(PROJECT_DIR / "mmproj-model-f16.gguf")

# --- CPU GOVERNOR ---
LLM_MAX_THREADS = 6   # Q4_K_M: 6 thread cukup (i7-11800H 8 core HT)

# --- SOVEREIGN DAEMON (LLaMA.cpp Local Servers) ---
MOKO_LLM_HOST = "127.0.0.1"
MOKO_LLM_PORT = 11435
# Phase 17 Dual-Server: Embedder terpisah pada port 11438 (MOKO-0.5B, 340MB RAM)
# Inference server (port 11435) tanpa embedding=True → hemat ~10GB RAM
MOKO_EMBED_PORT = 11438
# Phase 19: RAG-Dedicated server (port 11437) untuk ultra-efficient RAG LLM (200MB VRAM)
MOKO_RAG_PORT = 11437
MOKO_LLM_API_URL   = f"http://{MOKO_LLM_HOST}:{MOKO_LLM_PORT}/v1"
MOKO_EMBED_API_URL = f"http://{MOKO_LLM_HOST}:{MOKO_EMBED_PORT}/v1"
MOKO_RAG_API_URL   = f"http://{MOKO_LLM_HOST}:{MOKO_RAG_PORT}/v1"

# --- MODEL REGISTRY ---
# Model utama
MODEL_MOKO_UNSENSOR  = "moko-ai-4b-uncensored"
# Model Analyst Dual: Gunakan model MOKO yang sama untuk hemat VRAM dan menghindari swapping
MODEL_DOLPHIN        = MODEL_MOKO_UNSENSOR
MODEL_ANALYST        = MODEL_DOLPHIN


# --- AI MODE (bisa diubah runtime oleh user) ---
# "SOLO"  : Hanya MOKO tanpa DeepThink (1x LLM call) — CEPAT
# "DUAL"  : MOKO + DeepThink 3-pass — MENDALAM tapi lebih lambat
AI_MODE = "SOLO"

# Pilih model LLM aktif berdasarkan mode
MODEL_LLM = MODEL_MOKO_UNSENSOR

# --- COGNITIVE ENGINE ---
MAX_CONTEXT_TOKENS = 2048  # Q4_K_M Compact: 2K ctx (minimal RAM, 28 GPU layers)
MIN_SIMILARITY_SCORE = 0.60

# --- TOKEN STREAM (Sistem Token Keran T0) ---
TOKEN_STREAM_ENABLED = True
TOKEN_STREAM_RESERVE_TOKENS = 512
TOKEN_STREAM_RECENT_TURNS = 2

# --- SEMANTIC COMPRESSOR (SimpleMem T0c) ---
SEMANTIC_COMPRESSOR_V2 = True
SEMANTIC_COMPRESS_MAX_TOKENS = 96
SEMANTIC_COMPRESS_LLM_MIN_CHARS = 400
SEMANTIC_COMPRESS_USE_LLM = False          # heuristic default (no block)
SEMANTIC_COMPRESS_USE_LLM_MARATHON = False # set True saat LLM server warm

# --- OMNI (v3 — Clean Slate) ---
CRYPTO_ENABLED = False
OMNI_AUDIT_ENABLED = False
OMNI_DIR_VERSION = "v3"

# --- TOR ONION PROXY SETTINGS ---
TOR_ENABLED     = True
TOR_SOCKS_PORT  = 9050  # Port default daemon Tor (9150 untuk Tor Browser Bundle)

# --- INFERENCE HARDWARE SETTINGS ---
# BF16 model (7.9GB) tidak muat penuh di RTX 2050 4GB VRAM.
# Gunakan CPU+GPU hybrid: sebagian layer di GPU, sisanya di RAM.
# GPU_LAYERS diset dinamis berdasarkan VRAM yang tersedia.
FORCE_CPU = False
GPU_LAYERS = 99   # server_manager akan auto-fallback ke jumlah layer optimal berdasarkan VRAM

# --- MULTI-MODEL DOMAIN EXPERT (v3 — Clean Slate) ---
MULTI_MODEL_ENABLED = True

_moko_coder_path = PROJECT_DIR / "moko-coder-1b.gguf"
_is_moko_coder_active = _moko_coder_path.exists()

CODER_MODEL_PATH = str(_moko_coder_path) if _is_moko_coder_active else str(PROJECT_DIR / "MOKO-Coder-1.5B-Uncensored-F16.gguf")
CODER_MODEL_SIZE = 0.95 if _is_moko_coder_active else 3.4
CODER_MODEL_QUANT = "Q4_K_M" if _is_moko_coder_active else "F16"

DOMAIN_MODEL_REGISTRY = {
    "coding": {
        "path": CODER_MODEL_PATH,
        "base_model": None,
        "size_gb": CODER_MODEL_SIZE,
        "domain": ["code", "programming", "software", "debug", "refactor"],
        "quant": CODER_MODEL_QUANT,
        "params": "1.5B",
        "temperature": 0.0,
        "context_window": 8192,
    },
    "math": {
        "path": CODER_MODEL_PATH,
        "base_model": None,
        "size_gb": CODER_MODEL_SIZE,
        "domain": ["math", "physics", "engineering", "formula"],
        "quant": CODER_MODEL_QUANT,
        "params": "1.5B",
        "temperature": 0.0,
        "context_window": 2048,
    },
    "security": {
        "path": CODER_MODEL_PATH,
        "base_model": None,
        "size_gb": CODER_MODEL_SIZE,
        "domain": ["cybersecurity", "hacking", "cryptography", "penetration"],
        "quant": CODER_MODEL_QUANT,
        "params": "1.5B",
        "temperature": 0.1,
        "context_window": 4096,
    },
    "general": {
        "path": CODER_MODEL_PATH,
        "base_model": None,
        "size_gb": CODER_MODEL_SIZE,
        "domain": ["general", "conversation", "language", "creative"],
        "quant": CODER_MODEL_QUANT,
        "params": "1.5B",
        "temperature": 0.7,
        "context_window": 4096,
    },
}

# --- RAG EXTREME EFFICIENCY (200MB VRAM Target) ---
# Target: RTX 2050 (4GB VRAM) running Main LLM (2.5GB) + RAG LLM (200MB)
# Kita menggunakan Partial Offloading untuk menjaga VRAM tetap rendah.
MODEL_RAG_LLM_PATH = str(PROJECT_DIR / "moko-rag.gguf")
RAG_VRAM_BUDGET    = 200   # MB (Sesuai permintaan user)
RAG_GPU_LAYERS     = 4     # Hanya offload 4 layer ke GPU untuk menghemat VRAM
RAG_CONTEXT_WINDOW = 512   # Dibatasi untuk meminimalkan KV Cache di VRAM
MOKO_RAG_PORT      = 11437

# --- MOKO MODEL IDENTITY ---
MOKO_MODEL_VERSION = "1.0.0"
MOKO_MODEL_CODENAME = "MOKO-AI"
MOKO_MODEL_BUILD = "2026.07.03"

# --- INTENT-FIRST ROUTER (R2 Fase 4) ---
ROUTER_MODE = "intent_first"  # "rule_based" | "semantic" | "intent_first"
ROUTER_CONFIDENCE_THRESHOLD = 0.65
ROUTER_DOMAIN_FALLBACK = "general"

# --- KNOWLEDGE BASE PATHS ---
KNOWLEDGE_BASE_DIR = OMNI_DIR
KNOWLEDGE_DOMAINS = [
    "code",
    "math",
    "security",
    "general",
]
