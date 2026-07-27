"""
MOKO Multi-Model Dispatcher — Model Switching & VRAM Management
===============================================================
Dispatcher untuk mengelola multiple domain-specialized models.

Arsitektur:
────────────
  4 domain models (1-2B each) + 1 active + 1 standby
  VRAM: 4 GB total, ~2 GB active model, ~0.5 GB standby, ~0.5 GB KV cache

Domain Models:
──────────────
  1. coding   (1.5B) → programming, software engineering
  2. math     (1.0B) → mathematics, physics, engineering
  3. security (1.0B) → cybersecurity, hacking, cryptography
  4. general  (2.0B) → general knowledge, personal, lexical

Filosofi:
─────────
  "Satu ahli pada satu waktu, bukan semua ahli sekaligus."
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class ModelState(Enum):
    """Status model dalam VRAM."""
    UNLOADED = "unloaded"      # Tidak di VRAM
    LOADING = "loading"        # Sedang dimuat
    STANDBY = "standby"        # Siap dipakai (di VRAM)
    ACTIVE = "active"          # Sedang dipakai
    UNLOADING = "unloading"    # Sedang dilepas


@dataclass
class ModelInfo:
    """Informasi tentang satu model."""
    key: str                   # Model key (coding, math, security, general)
    path: str                  # Path ke GGUF file
    size_gb: float             # Model size in GB
    domains: List[str]         # Knowledge base domains
    state: ModelState = ModelState.UNLOADED
    load_time_ms: float = 0.0  # Waktu loading terakhir
    last_used: float = 0.0     # Timestamp terakhir dipakai
    ref_count: int = 0         # Berapa kali model ini dipakai
    # Parameter spesialisasi domain (jembatan sebelum ada model terpisah)
    temperature: float = 0.7   # Suhu generasi khas domain
    context_window: int = 4096 # Panjang konteks khas domain
    system_prompt: str = ""    # System prompt tambahan khas domain (opsional)


@dataclass
class VRAMStatus:
    """Status VRAM saat ini."""
    total_gb: float            # Total VRAM
    used_gb: float             # VRAM terpakai
    free_gb: float             # VRAM tersedia
    active_model: Optional[str]  # Model aktif
    standby_model: Optional[str] # Model standby


# ═══════════════════════════════════════════════════════════════════════════
# VRAM MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class VRAMManager:
    """
    Manages VRAM allocation untuk multiple models.
    
    Strategy:
    - 1 model aktif (loaded ke VRAM)
    - 1 model standby (loaded ke VRAM, ready untuk switch)
    - OS overhead: ~0.5 GB
    - KV cache: ~0.5 GB per model
    """
    
    TOTAL_VRAM_GB = 4.0        # RTX 2050
    OS_OVERHEAD_GB = 0.5       # OS + driver
    KV_CACHE_GB = 0.5          # KV cache per model
    STANDBY_RESERVE_GB = 1.0   # Reserve untuk standby model
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._allocated: Dict[str, float] = {}  # model_key → allocated GB
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  💾 [VRAMManager] {msg}")
    
    @property
    def available_gb(self) -> float:
        """VRAM available untuk models (after overhead)."""
        return self.TOTAL_VRAM_GB - self.OS_OVERHEAD_GB
    
    @property
    def usable_gb(self) -> float:
        """VRAM usable untuk active + standby models."""
        return self.available_gb - self.KV_CACHE_GB
    
    def can_fit(self, model_size_gb: float) -> bool:
        """Apakah model muat di VRAM yang tersisa?"""
        with self._lock:
            current_used = sum(self._allocated.values())
            return (current_used + model_size_gb) <= self.usable_gb
    
    def allocate(self, model_key: str, size_gb: float) -> bool:
        """Alokasikan VRAM untuk model."""
        with self._lock:
            current_used = sum(self._allocated.values())
            if (current_used + size_gb) > self.usable_gb:
                self._log(f"Cannot fit {model_key} ({size_gb:.1f} GB)")
                return False
            self._allocated[model_key] = size_gb
            self._log(f"Allocated {size_gb:.1f} GB for {model_key}")
            return True
    
    def deallocate(self, model_key: str):
        """Lepas alokasi VRAM untuk model."""
        with self._lock:
            if model_key in self._allocated:
                del self._allocated[model_key]
                self._log(f"Deallocated {model_key}")
    
    def get_status(self) -> VRAMStatus:
        """Status VRAM saat ini."""
        with self._lock:
            used = sum(self._allocated.values())
            return VRAMStatus(
                total_gb=self.TOTAL_VRAM_GB,
                used_gb=used + self.OS_OVERHEAD_GB,
                free_gb=self.available_gb - used,
                active_model=None,  # Akan diisi oleh dispatcher
                standby_model=None
            )


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-MODEL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

class MultiModelDispatcher:
    """
    Dispatcher untuk multiple domain-specialized models.
    
    Responsibilities:
    1. Manage model registry (4 domain models)
    2. Handle model loading/unloading
    3. VRAM allocation
    4. Standby model preloading
    5. Model switching with minimal latency
    
    Usage:
        dispatcher = MultiModelDispatcher()
        dispatcher.load_model("coding")
        response = dispatcher.generate("tulis fungsi Python")
        dispatcher.unload_model("coding")
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.vram = VRAMManager(verbose=verbose)
        self._lock = threading.Lock()
        
        # Model registry
        self._models: Dict[str, ModelInfo] = {}
        
        # Current state
        self._active_model: Optional[str] = None
        self._standby_model: Optional[str] = None

        # Path model (hasil resolusi) yang sedang benar-benar dimuat ke server.
        # Dipakai untuk menghindari reload berulang saat beberapa domain
        # menunjuk ke file GGUF yang sama.
        self._loaded_path: Optional[str] = None
        
        # Statistics
        self._switch_count = 0
        self._total_switch_time_ms = 0.0
        
        # Initialize registry
        self._init_registry()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  🚀 [Dispatcher] {msg}")
    
    def _init_registry(self):
        """Initialize model registry dari settings."""
        try:
            from moko_config import settings
            registry = getattr(settings, 'DOMAIN_MODEL_REGISTRY', {})
            
            for key, config in registry.items():
                path = config['path']
                # Prefer Byte-Q version if available
                byteq_path = path.replace(".gguf", ".byteq.gguf")
                if os.path.exists(byteq_path):
                    path = byteq_path
                    self._log(f"Using Byte-Q optimized model for {key}: {os.path.basename(path)}")
                
                self._models[key] = ModelInfo(
                    key=key,
                    path=path,
                    size_gb=config['size_gb'] if ".byteq" not in path else config['size_gb'] * 0.6, # Byte-Q saves ~40% VRAM
                    domains=config['domain'],
                    temperature=config.get('temperature', 0.7),
                    context_window=config.get('context_window', 4096),
                    system_prompt=config.get('system_prompt', ''),
                )
            
            self._log(f"Registered {len(self._models)} models")
            
        except Exception as e:
            self._log(f"Registry init error: {e}")
    
    # ─────────────────────────────────────────────────────────────────────
    # MODEL MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def load_model(self, model_key: str) -> bool:
        """
        Load model ke VRAM.
        
        Args:
            model_key: Model key (coding, math, security, general)
        
        Returns:
            True if successful
        """
        if model_key not in self._models:
            self._log(f"Unknown model: {model_key}")
            return False
        
        model = self._models[model_key]
        
        with self._lock:
            # Check if already loaded
            if model.state == ModelState.ACTIVE:
                self._log(f"{model_key} already active")
                return True
            
            if model.state == ModelState.STANDBY:
                self._log(f"{model_key} already standby")
                return True
            
            # Check VRAM (outside lock to avoid deadlock)
            need_unload = not self.vram.can_fit(model.size_gb)
        
        # Fast-path: jika file model (hasil resolusi) yang sama sudah dimuat ke
        # server, tidak perlu reload sama sekali — cukup tandai sebagai standby.
        # Ini menghindari restart server berulang saat beberapa domain menunjuk
        # ke file GGUF yang sama (registry saat ini: coding/math/security/general).
        if self._loaded_path is not None and self._loaded_path == model.path:
            with self._lock:
                model.state = ModelState.STANDBY
                model.last_used = time.time()
                self.vram.allocate(model_key, model.size_gb)
                if self._standby_model is None:
                    self._standby_model = model_key
                self._log(
                    f"{model_key} memakai file model yang sudah dimuat "
                    f"(skip reload): {os.path.basename(model.path)}"
                )
            return True
        
        if need_unload:
            self._log(f"VRAM full, need to unload something")
            self._make_room(model.size_gb)
        
        with self._lock:
            # Load model
            model.state = ModelState.LOADING
            self._log(f"Loading {model_key} ({model.size_gb:.1f} GB)...")
        
        # Simulate loading (in real implementation, this would call llama-server)
        t0 = time.perf_counter()
        
        try:
            from moko_agents.llm_engine import engine
            success = engine.load_model(model.path)
            if not success:
                self._log(f"Failed to load {model_key}")
                with self._lock:
                    model.state = ModelState.UNLOADED
                return False
        except Exception as e:
            self._log(f"Load error: {e}")
            with self._lock:
                model.state = ModelState.UNLOADED
            return False
            
        load_time = (time.perf_counter() - t0) * 1000
        
        with self._lock:
            model.state = ModelState.STANDBY
            model.load_time_ms = load_time
            model.last_used = time.time()
            
            # Catat file yang benar-benar dimuat ke server agar switch domain
            # berikutnya ke file yang sama tidak memicu reload.
            self._loaded_path = model.path
            
            # Allocate VRAM
            self.vram.allocate(model_key, model.size_gb)
            
            # Update standby
            if self._standby_model is None:
                self._standby_model = model_key
            
            self._log(f"Loaded {model_key} in {load_time:.1f}ms")
        
        return True
    
    def unload_model(self, model_key: str) -> bool:
        """
        Unload model dari VRAM.
        
        Args:
            model_key: Model key
        
        Returns:
            True if successful
        """
        if model_key not in self._models:
            return False
        
        model = self._models[model_key]
        
        with self._lock:
            if model.state == ModelState.UNLOADED:
                return True
            
            if model_key == self._active_model:
                self._log(f"Cannot unload active model: {model_key}")
                return False
            
            model.state = ModelState.UNLOADING
        
        # Simulate unloading
        # In real implementation: await llama-server model unload
        
        with self._lock:
            model.state = ModelState.UNLOADED
            self.vram.deallocate(model_key)
            
            if self._standby_model == model_key:
                self._standby_model = None
            
            self._log(f"Unloaded {model_key}")
        
        return True
    
    def _make_room(self, needed_gb: float):
        """Make room di VRAM dengan unload model yang tidak dipakai."""
        # Priority: unload standby dulu
        if self._standby_model and self._standby_model != self._active_model:
            model = self._models[self._standby_model]
            if model.size_gb >= needed_gb:
                self._unload_internal(self._standby_model)
                return
        
        # Jika masih kurang, unload model lain berdasarkan LRU
        candidates = []
        for key, model in self._models.items():
            if key == self._active_model:
                continue
            if model.state in (ModelState.STANDBY, ModelState.LOADING):
                candidates.append((key, model.last_used))
        
        # Sort by last_used (oldest first)
        candidates.sort(key=lambda x: x[1])
        
        for key, _ in candidates:
            model = self._models[key]
            if model.size_gb >= needed_gb:
                self._unload_internal(key)
                return
    
    def _unload_internal(self, model_key: str):
        """Internal unload (called from _make_room, no lock needed)."""
        if model_key not in self._models:
            return
        
        model = self._models[model_key]
        
        with self._lock:
            if model.state == ModelState.UNLOADED:
                return
            if model_key == self._active_model:
                return
            model.state = ModelState.UNLOADING
        
        # Simulate unloading
        
        with self._lock:
            model.state = ModelState.UNLOADED
            self.vram.deallocate(model_key)
            if self._standby_model == model_key:
                self._standby_model = None
            self._log(f"Unloaded {model_key}")
    
    # ─────────────────────────────────────────────────────────────────────
    # MODEL SWITCHING
    # ─────────────────────────────────────────────────────────────────────
    
    def switch_to(self, model_key: str) -> bool:
        """
        Switch ke model tertentu.
        
        Args:
            model_key: Model key
        
        Returns:
            True if successful
        """
        if model_key not in self._models:
            self._log(f"Unknown model: {model_key}")
            return False
        
        model = self._models[model_key]
        
        # Load jika belum loaded
        if model.state == ModelState.UNLOADED:
            if not self.load_model(model_key):
                return False
        
        # Switch
        t0 = time.perf_counter()
        
        with self._lock:
            # Update states
            if self._active_model:
                old_model = self._models[self._active_model]
                old_model.state = ModelState.STANDBY
            
            model.state = ModelState.ACTIVE
            model.last_used = time.time()
            model.ref_count += 1
            
            self._standby_model = self._active_model
            self._active_model = model_key
        
        switch_time = (time.perf_counter() - t0) * 1000
        
        self._switch_count += 1
        self._total_switch_time_ms += switch_time
        
        self._log(f"Switched to {model_key} in {switch_time:.2f}ms")
        return True
    
    # ─────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────────────────────────────
    
    def generate(
        self,
        prompt: str,
        model_key: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response menggunakan model tertentu.
        
        Args:
            prompt: User prompt
            model_key: Model to use (auto-detect if None)
            **kwargs: Additional parameters
        
        Returns:
            Generated text or None
        """
        # Auto-detect model jika belum ditentukan
        if model_key is None:
            model_key = self._detect_model(prompt)
        
        # Switch ke model yang tepat
        if not self.switch_to(model_key):
            self._log(f"Failed to switch to {model_key}")
            return None
        
        # Generate (dalam implementasi nyata, ini akan memanggil LLM server)
        try:
            from moko_agents.llm_engine import engine
            response = engine.generate_text(prompt, **kwargs)
            return response
        except Exception as e:
            self._log(f"Generation error: {e}")
            return None
    
    def _detect_model(self, prompt: str) -> str:
        """Deteksi model mana yang harus dipakai berdasarkan prompt."""
        try:
            from moko_agents.intent_router import get_intent_router
            router = get_intent_router()
            result = router.classify(prompt)
            return result.model_key
        except Exception:
            return "general"
    
    # ─────────────────────────────────────────────────────────────────────
    # DOMAIN GENERATION PARAMS (jembatan spesialisasi)
    # ─────────────────────────────────────────────────────────────────────
    
    def get_params_for(self, model_key: str) -> Dict[str, Any]:
        """
        Parameter generasi khas domain untuk sebuah model.
        
        Selama registry masih menunjuk ke satu file GGUF untuk semua domain,
        inilah jembatan spesialisasi: tiap domain tetap bisa punya suhu,
        panjang konteks, dan system prompt yang berbeda.
        
        Returns:
            dict berisi model_key, temperature, context_window, system_prompt.
            dict kosong jika model_key tidak dikenal.
        """
        model = self._models.get(model_key)
        if model is None:
            return {}
        return {
            "model_key": model_key,
            "temperature": model.temperature,
            "context_window": model.context_window,
            "system_prompt": model.system_prompt,
        }
    
    def get_active_params(self) -> Dict[str, Any]:
        """Parameter generasi khas domain untuk model yang sedang aktif."""
        if self._active_model is None:
            return {}
        return self.get_params_for(self._active_model)
    
    # ─────────────────────────────────────────────────────────────────────
    # PRELOADING
    # ─────────────────────────────────────────────────────────────────────
    
    def preload_standby(self, model_key: str):
        """
        Pre-load model sebagai standby.
        
        Args:
            model_key: Model key
        """
        if model_key not in self._models:
            return
        
        model = self._models[model_key]
        
        if model.state == ModelState.UNLOADED:
            self.load_model(model_key)
    
    def preload_likely_next(self, current_model: str):
        """
        Pre-load model yang kemungkinan besar dipakai selanjutnya.
        
        Args:
            current_model: Current active model
        """
        # Prediction logic: if coding → likely math next
        predictions = {
            "coding": "math",
            "math": "coding",
            "security": "coding",
            "general": "math",
        }
        
        next_model = predictions.get(current_model)
        if next_model and next_model in self._models:
            self.preload_standby(next_model)
    
    # ─────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────
    
    def get_status(self) -> Dict[str, Any]:
        """Status lengkap dispatcher."""
        return {
            "active_model": self._active_model,
            "standby_model": self._standby_model,
            "models": {
                key: {
                    "state": model.state.value,
                    "size_gb": model.size_gb,
                    "load_time_ms": model.load_time_ms,
                    "ref_count": model.ref_count,
                }
                for key, model in self._models.items()
            },
            "vram": {
                "total_gb": self.vram.TOTAL_VRAM_GB,
                "allocated_gb": sum(self.vram._allocated.values()),
                "available_gb": self.vram.usable_gb,
            },
            "switch_count": self._switch_count,
            "avg_switch_time_ms": (
                self._total_switch_time_ms / self._switch_count
                if self._switch_count > 0 else 0
            ),
        }
    
    def print_status(self):
        """Print status dalam format yang mudah dibaca."""
        status = self.get_status()
        
        print("\n" + "━" * 60)
        print("  🚀 MOKO Multi-Model Dispatcher — Status")
        print("━" * 60)
        print(f"  Active Model  : {status['active_model'] or 'None'}")
        print(f"  Standby Model : {status['standby_model'] or 'None'}")
        print(f"  Switch Count  : {status['switch_count']}")
        print(f"  Avg Switch    : {status['avg_switch_time_ms']:.2f}ms")
        print()
        
        print("  Models:")
        for key, info in status['models'].items():
            state_icon = {
                "unloaded": "⬜",
                "loading": "🔄",
                "standby": "🟡",
                "active": "🟢",
                "unloading": "🔄",
            }.get(info['state'], "❓")
            
            print(f"    {state_icon} {key:<10} {info['size_gb']:.1f}GB | "
                  f"Load: {info['load_time_ms']:.1f}ms | "
                  f"Used: {info['ref_count']}x")
        
        print()
        print("  VRAM:")
        print(f"    Total    : {status['vram']['total_gb']:.1f} GB")
        print(f"    Allocated: {status['vram']['allocated_gb']:.1f} GB")
        print(f"    Available: {status['vram']['available_gb']:.1f} GB")
        print("━" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════

_dispatcher_instance: Optional[MultiModelDispatcher] = None

def get_dispatcher(verbose: bool = False) -> MultiModelDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = MultiModelDispatcher(verbose=verbose)
    return _dispatcher_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n🧪 Multi-Model Dispatcher — Self Test\n")
    
    # Create test registry
    test_registry = {
        "coding": {
            "path": "/tmp/test_coding.gguf",
            "size_gb": 0.75,
            "domain": ["code", "programming"],
        },
        "math": {
            "path": "/tmp/test_math.gguf",
            "size_gb": 0.5,
            "domain": ["math", "physics"],
        },
        "security": {
            "path": "/tmp/test_security.gguf",
            "size_gb": 0.5,
            "domain": ["cybersecurity"],
        },
        "general": {
            "path": "/tmp/test_general.gguf",
            "size_gb": 1.0,
            "domain": ["general", "personal"],
        },
    }
    
    # Patch settings
    from moko_config import settings
    settings.DOMAIN_MODEL_REGISTRY = test_registry
    
    # Create dispatcher
    dispatcher = MultiModelDispatcher(verbose=True)
    
    # Test 1: Load models
    print("\n── Test 1: Load Models ──────────────────────────")
    dispatcher.load_model("coding")
    dispatcher.load_model("math")
    
    # Test 2: Switch models
    print("\n── Test 2: Switch Models ───────────────────────")
    dispatcher.switch_to("coding")
    dispatcher.switch_to("math")
    dispatcher.switch_to("coding")
    
    # Test 3: Preload
    print("\n── Test 3: Preload Standby ─────────────────────")
    dispatcher.preload_standby("security")
    dispatcher.preload_likely_next("coding")
    
    # Test 4: Status
    print("\n── Test 4: Status ──────────────────────────────")
    dispatcher.print_status()
    
    # Test 5: VRAM management
    print("\n── Test 5: VRAM Management ─────────────────────")
    print(f"  Can fit 1.0 GB: {dispatcher.vram.can_fit(1.0)}")
    print(f"  Can fit 3.0 GB: {dispatcher.vram.can_fit(3.0)}")
    
    # Test 6: Unload
    print("\n── Test 6: Unload Models ───────────────────────")
    dispatcher.unload_model("security")
    dispatcher.print_status()
    
    print("\n✅ Self test selesai!\n")


if __name__ == "__main__":
    _self_test()
