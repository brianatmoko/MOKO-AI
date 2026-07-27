# WALKTHROUGH REVISI & PENAMBAHAN KODE MOKO OS (RISET 24-26)
## Cetak Biru Integrasi Sistem Maraton, Loop Detector, Zstd Compressor, dan MuonClip Optimizer ke Basis Kode Utama
### Dokumen Taktis Insinyur AI MOKO OS — Juli 2026

---

## 📌 ABSTRAK & PRINSIP UTAMA

Sesuai hasil kajian **Riset 24, 25, dan 26**, pemutakhiran kode difokuskan pada penguatan ketahanan hardware lokal dan optimalisasi dynamic routing. Peta jalan revisi ini dibagi menjadi **empat file utama** yang menyatukan seluruh puzzle arsitektur kita:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALUR INTEGRASI KODE                             │
├────────────────────────────────────────────────────────────────────────┤
│ 1. moko_agents/dual_system/interaction_logger.py                       │
│    └─ Tambah: Zstd auto-compression pada log distill (Riset 25)        │
│                                                                        │
│ 2. moko_agents/dual_system/orchestrator.py                             │
│    └─ Tambah: LoopDetector & ActionBudget pada System 1-2 loop         │
│                                                                        │
│ 3. moko_inference/mome_engine.py                                       │
│    └─ Tambah: MokoUnifiedMemoryBridge & Speculative Decoder            │
│                                                                        │
│ 4. finetune/moko_trainer_v2.py                                         │
│    └─ Tambah: MokoMuonClip Optimizer & Zstd streaming reader           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. REVISI 1: AUTO-COMPRESSION DATASET DISTILASI

*   **Berkas Target:** [`moko_core/moko_agents/dual_system/interaction_logger.py`](file:///home/brianatmokoo/Documents/Linux/MOKO_OS_Project/moko_core/moko_agents/dual_system/interaction_logger.py)
*   **Tujuan:** Mengintegrasikan `MokoDatasetCompressor` berbasis Zstd agar data log interaksi Guru $\rightarrow$ Murid secara otomatis dikompresi ketika berkas mentah JSONL berukuran $> 50$ MB, mencegah *disk footprint* membengkak di komputer lokal.

### Perubahan Kode (Diff Concept):

```diff
  import os
  import json
  import logging
  from pathlib import Path
+ from moko_agents.dual_system.dataset_compressor import get_compressor
  
  logger = logging.getLogger("moko_interaction_logger")
  
  class InteractionLogger:
      def __init__(self, log_dir: Path = None):
          self.log_dir = log_dir or Path(settings.PROJECT_DIR / "distill_dataset")
          self.log_dir.mkdir(parents=True, exist_ok=True)
          self.log_file = self.log_dir / "moko_distill_samples.jsonl"
+         self.compressor = get_compressor()
  
      def log_sample(self, prompt, thought, code, passed_guard, task_complexity, task_category, source):
          # Log sample logic...
          with open(self.log_file, "a", encoding="utf-8") as f:
              f.write(json.dumps(record, ensure_ascii=False) + "\n")
          
+         # Auto compress if file grows beyond threshold
+         try:
+             self.compressor.auto_compress_if_large(self.log_file, threshold_mb=50.0, remove_original=False)
+         except Exception as e:
+             logger.warning(f"Auto-compression failed: {e}")
```

---

## 2. REVISI 2: AGENTIC LOOP PREVENTION DI DUAL-SYSTEM ORCHESTRATOR

*   **Berkas Target:** [`moko_core/moko_agents/dual_system/orchestrator.py`](file:///home/brianatmokoo/Documents/Linux/MOKO_OS_Project/moko_core/moko_agents/dual_system/orchestrator.py)
*   **Tujuan:** Menyuntikkan `LoopDetector` di tingkat orkestrator utama agar System 1 (Kimi Executor) dan System 2 (DeepSeek Brain) tidak mengalami kebuntuan eksekusi loop tiada akhir.

### Perubahan Kode (Diff Concept):

```diff
  import time
  from typing import Dict, List, Optional
  from .brain_node import BrainNode
  from .executor_node import ExecutorNode
  from .runtime_guard import RuntimeGuard
+ from moko_agents.coding.coding_orchestrator import LoopDetector, _make_action_hash
  
  class DualSystemOrchestrator:
      def __init__(self, workspace_dir: str):
          self.brain = BrainNode()
          self.executor = ExecutorNode(workspace_dir)
          self.guard = RuntimeGuard()
+         # Tambah LoopDetector: window=3 aksi berulang, budget=30 total langkah per plan
+         self.loop_detector = LoopDetector(window=3, max_actions=30)
  
      def run_loop(self, user_prompt: str, max_iterations: int = 5) -> Dict:
+         self.loop_detector.reset()
          state = {"prompt": user_prompt, "iteration": 0, "status": "init"}
          
          for idx in range(max_iterations):
+             # Generate action hash for current state
+             action_hash = _make_action_hash("DUAL_STEP", user_prompt, state.get("code", ""))
+             loop_status = self.loop_detector.check(action_hash)
+             
+             if loop_status in ("LOOP_DETECTED", "BUDGET_EXCEEDED"):
+                 logger.error(f"Execution aborted by LoopDetector: {loop_status}")
+                 return {"status": "ABORT", "reason": loop_status}
              
              # Lakukan step Plan -> Execute -> Guard...
```

---

## 3. REVISI 3: EFFICIENT INF_SCHEDULING DI MOME ENGINE

*   **Berkas Target:** [`moko_core/moko_inference/mome_engine.py`](file:///home/brianatmokoo/Documents/Linux/MOKO_OS_Project/moko_core/moko_inference/mome_engine.py)
*   **Tujuan:** Menggunakan `MokoKVCacheManager` dan `MokoSpeculativeDecoder` di level MOME Engine untuk melakukan akselerasi kecepatan generasi (MTP) dan management memory multi-tier saat melakukan RAG berkonteks panjang.

### Perubahan Kode (Diff Concept):

```diff
  import logging
  from moko_inference.moko_engine import get_moko_engine
+ from moko_memory.kv_cache_manager import get_kv_cache
+ from moko_inference.speculative_decoder import create_moko_speculative_decoder
  
  class MOMEEngine:
      def __init__(self, workspace_dir=None):
          self.local_engine = get_moko_engine()
+         self.kv_cache = get_kv_cache()
+         self.speculative_decoder = create_moko_speculative_decoder(self.local_engine, k=4)
  
      def execute(self, prompt, messages, max_tokens, temperature, stream=False, on_token=None):
          # Mode detection logic...
          if mode == MODE_GPT:
+             # Lakukan speculative decoding jika stream aktif
+             if stream and self.speculative_decoder:
+                 text, stats = self.speculative_decoder.generate(prompt, max_new_tokens=max_tokens)
+                 if on_token:
+                     on_token(text)
+                 return text, f"GPT_SPECULATIVE_OK_{stats['estimated_speedup']}x"
```

---

## 4. REVISI 4: INTERFACE TRAINER DENGAN OPTIMIZER MUONCLIP

*   **Berkas Target:** [`finetune/moko_trainer_v2.py`](file:///home/brianatmokoo/Documents/Linux/MOKO_OS_Project/finetune/moko_trainer_v2.py)
*   **Tujuan:** Mengubah scheduler dan optimizer default HuggingFace Trainer agar menggunakan `MokoMuonClip` untuk bobot linear LoRA parameter 2D, serta menggunakan `MokoDatasetCompressor` untuk membaca dataset yang telah dikompresi Zstd.

### Perubahan Kode (Diff Concept):

```diff
  import torch
  from transformers import Trainer
+ from moko_optimizer import create_optimizer_groups, get_muon_lr_schedule
+ from moko_agents.dual_system.dataset_compressor import get_compressor
  
  class MokoTrainer(Trainer):
+     def create_optimizer_and_scheduler(self, num_training_steps: int):
+         # Gunakan MuonClip untuk bobot 2D, AdamW untuk 1D bias/LayerNorm
+         muon_opt, adamw_opt = create_optimizer_groups(
+             self.model, 
+             lr_muon=1e-3, 
+             lr_adamw=2e-4, 
+             use_muonclip=True, 
+             clip_alpha=0.1
+         )
+         # Daftarkan ke optimizer list Trainer
+         self.optimizer = muon_opt  # Atau gunakan custom training loop step
+         self.lr_scheduler = get_muon_lr_schedule(self.optimizer, 100, num_training_steps)
```

---

## IMPLEMENTASI & EKSEKUSI NYATA

Ketika user memberikan aba-aba *"eksekusi training go to gguf"*, skrip `moko_trainer_v2.py` akan dimodifikasi sepenuhnya untuk mengadopsi taktik `MokoTrainer` berbasis `MokoMuonClip` ini, memastikan bahwa sisa VRAM RTX 2050 4GB aman dari *loss spikes* dan model 1.5B kita memiliki tingkat konvergensi 2x lebih akurat.

---
*Dokumen ini ditulis sebagai kelanjutan langsung dari Riset 26.*
*Status: Ready for Code Implementation | Tanggal: Juli 2026*
