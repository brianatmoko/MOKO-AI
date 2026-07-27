# 14 — Reconstruction Roadmap

> **Tujuan:** Peta jalan lengkap untuk rekonstruksi MOKO OS dari state saat ini
> ke arsitektur multi-model domain expert.

---

## Daftar Isi

1. [Current State](#1-current-state)
2. [Phase 1: Cleanup (Done)](#2-phase-1-cleanup-done)
3. [Phase 2: Documentation (Done)](#3-phase-2-documentation-done)
4. [Phase 3: Architecture Upgrade](#4-phase-3-architecture-upgrade)
5. [Phase 4: New Features](#5-phase-4-new-features)
6. [Timeline](#6-timeline)

---

## 1. Current State

### Completed
- ✅ R2 Fase 1: CRYPTO_ENABLED=False
- ✅ R2 Fase 2: OMNI_DIR → .moko_omni
- ✅ R2 Fase 3: Cleanup old directories & unused models (>15GB saved)
- ✅ Documentation: 14 research documents
- ✅ Settings: Updated for new architecture & RAG 200MB
- ✅ Phase 3.1: Intent-First Router implementation & integration
- ✅ Phase 3.2: Byte-Q Quantizer core (Lloyd + Huffman)
- ✅ Phase 3.3: Multi-Model Dispatcher core (VRAM Manager)
- ✅ Phase 3.4: RAG 200MB profile in server_manager.py
- ✅ Phase 4: Final Rebranding (Qwen → MOKO) & Crypto Removal

### Pending
- ✅ Phase 3.2: Byte-Q integration into LLM engine (dequantize-before-load bridge)
- ✅ Phase 3.3: Dispatcher integration into CoreNode (domain-params bridge + same-file skip-reload)
- ✅ Phase 3.4: RAG 200MB model deployment & pipeline integration (dedicated server + context distillation)
- ✅ Phase 3.5: End-to-end integration testing (cross-component E2E chain + full Phase 3 regression gate)
- ✅ Phase 3.6: Performance optimization testing (VRAM budget/eviction, Byte-Q saving, skip-reload, preload, telemetry)
- ✅ Phase 3.7: Marathon/auto-continue fix (SSE finish_reason captured; truncated code auto-continued natively from partial; unclosed ``` fence trigger)
- ✅ Phase 4.1: Domain-specialized models (MOKO-Coder-MOKO2.5-1.5B-Lora.gguf Ready)
- ✅ Phase 5.2: RAG + AI Agent (Basic implemented)
- ✅ Phase 5.3: Onion Search & TorBot Integration (Done)
- 🟡 Phase 6.1: OMNI System Reconstruction (Formula injection pending)
- 🟡 Phase 7.1: AI Coding Revolution (Research phase)

---

## 2. Phase 1: Cleanup (Done)

### R2 Fase 1 (2026-07-02)
- Disabled MokoCryptoCore hot path
- Disabled blockchain audit/chain logging
- Updated settings.py

### R2 Fase 2 (2026-07-03)
- Renamed .moko_crypto → .moko_omni
- Updated OMNI_DIR path
- Verified directory structure

### R2 Fase 3 (2026-07-03)
- Deleted .moko_crypto_backup (704 MB)
- Deleted .moko_crypto_cache (52 KB)
- Deleted .moko_crypto_omni (164 KB)
- Updated crypto module paths
- Total savings: ~704 MB disk

---

## 3. Phase 2: Documentation (Done)

### Research Documents (docs/riset/)
1. 01_KONTRIBUSI_ASLI.md — Byte-Q innovation proof
2. 02_ANALISIS_KERAS.md — Hardware constraints
3. 03_LANDSKAP_KOMPETITOR.md — Competitive landscape
4. 04_MASALAH_DAN_SOLUSI.md — Crypto failure lessons
5. 05_RUMUSAN_MATEMATIKA.md — All verified formulas
6. 06_ARSITEKTUR_INTI.md — System architecture
7. 07_MASA_DEPAN.md — Feasible vs impossible
8. 08_PETA_EFISIENSI.md — 8 efficiency gaps
9. 09_FORMULASI_EFISIENSI.md — 10 complete formulas
10. 10_MULTI_MODEL_DOMAIN_EXPERT.md — Domain expert concept
11. 11_DATA_INVENTORY.md — Data inventory
12. 12_INJECT_SCRIPTS.md — Inject scripts documentation
13. 13_MODEL_FILES.md — Model files documentation
14. 14_RECONSTRUCTION_ROADMAP.md — This document

---

## 4. Phase 3: Architecture Upgrade

### 3.1 Intent-First Router (Status: ✅ DONE)
```
Current: Intent-first with 8-class priority chain
Tasks:
1. Create moko_core/moko_agents/intent_router.py (Done)
2. Implement 8 intent classes (Done)
3. Add confidence scoring (Done)
4. Integrate with existing router.py (Done)
5. Test with sample queries (Done - 100% accuracy in self-test)
```

### 3.2 Byte-Q Quantization (Status: ✅ INTEGRATED)
```
Current: Byte-Q core + inference bridge implemented
Tasks:
1. Create moko_core/moko_tools/byteq_quantizer.py (Done)
2. Implement Lloyd's algorithm for optimal levels (Done)
3. Implement Huffman coding for entropy coding (Done)
4. Benchmark against Q3_K_M (Done - self-test benchmark available)
5. Test accuracy vs compression ratio (Done)
6. Integrate into LLM inference engine (Done):
   - quantize_domain_models.py → reconstructable .byteq.gguf container
     (I8 indices + per-tensor Lloyd levels metadata).
   - moko_tools/byteq_loader.py → dequantize-before-load: rebuild a
     standard F16 GGUF (cached) that llama.cpp/llama-server can load.
   - MokoEngine.load_model auto-resolves .byteq.gguf transparently.
   - MultiModelDispatcher prefers .byteq.gguf sibling when present.
   - Verified via moko_tools/test_byteq_roundtrip.py (NMSE ~0.11).
```

### 3.3 Multi-Model Dispatcher (Status: ✅ INTEGRATED INTO CORENODE)
```
Current: Dispatcher core + VRAM manager + CoreNode integration done
Tasks:
1. Create moko_core/moko_agents/model_dispatcher.py (Done)
2. Implement VRAM manager (Done)
3. Implement model switching (Done)
4. Add standby model preloading (Done)
5. Test switching latency (Done)
6. Integrate into CoreNode pipeline (Done):
   - ModelInfo carries per-domain generation params
     (temperature/context_window/system_prompt) from DOMAIN_MODEL_REGISTRY.
   - Dispatcher exposes get_active_params()/get_params_for() as the domain
     specialization bridge (works even while all domains share one GGUF file).
   - CoreNode.quick_reply applies the active domain temperature to the main
     generation coop_params (switch model → real behavioral effect).
   - Same-file skip-reload guard (_loaded_path): switching between domains that
     point to the same GGUF no longer restarts the server.
   - Verified via moko_agents/test_dispatcher_integration.py (server-free).
Next: real per-domain weights + real llama-server unload (still simulated).
```

### 3.4 Ultra-Efficient RAG (200MB) (Status: ✅ INTEGRATED INTO PIPELINE)
```
Target: MOKO-RAG-1.5B (Partial Offload) = ~200MB VRAM budget.
Model: moko-rag.gguf (Byte-Q IQ2_XXS of Qwen2.5-1.5B, ~980MB on disk) — present.
Tasks:
1. Define RAG_VRAM_BUDGET and MOKO_RAG_PORT in settings.py (Done)
2. Add get_gpu_layers(is_rag=True) in server_manager.py (Done)
3. Implement multi-agent bridge: RAG Agent & Output Agent (Done)
4. Apply 3-layer architecture: Knowledge, Retrieval, Synthesis (Done)
5. Set RAG_GPU_LAYERS=4 and RAG_CONTEXT_WINDOW=512 for 200MB footprint (Done)
6. Deploy & wire the dedicated RAG server into the runtime (Done):
   - start_servers() now AUTO-BOOTS start_rag_server() (port 11437) once the
     main server is online (previously start_rag_server was never called).
   - MokoEngine gains rag_available() + generate_rag(): a lightweight,
     fail-quiet client that targets the dedicated RAG server (port 11437),
     never the main server (11435).
   - RetrievalLayer now DISTILLS raw Omni facts into a compact, query-relevant
     context via the 200MB RAG model when its server is up; if the RAG server
     is down it safely falls back to the raw concatenated context (pipeline
     never breaks). Fixed wrong default RAG port (11435 → 11437).
   - moko_daemon status now reports the RAG server (port 11437) too.
   - Verified via moko_agents/test_rag_integration.py (server-free, 7 asserts).
   - Runtime smoke test with the real moko-rag.gguf on the RTX 2050 PASSED
     (boot 1.11s, /health=ok, VRAM ~225 MiB, HTTP 200, generate_rag non-empty).
```

### 3.5 Integration Testing (Status: ✅ DONE — E2E + REGRESSION GATE)
```
Current: End-to-end integration test that WIRES all Phase 3 components together
         (3.1 Router → 3.2 Byte-Q → 3.3 Dispatcher → 3.4 RAG), not just per-unit.
Tasks:
1. Create moko_agents/test_phase35_integration.py (Done)
2. Cross-component E2E chain (all server-free / monkeypatched) (Done):
   - Router → Dispatcher: IntentFirstRouter intent picks the correct specialist
     model; switch_to activates it and exposes the domain generation params
     (temperature/context_window). (3.1↔3.3)
   - Byte-Q preferred in registry: when a .byteq.gguf sibling exists the
     dispatcher selects it (dequantize-before-load bridge) and corrects the
     VRAM budget (~40% saving); resolve_loadable_model passes plain GGUF
     through unchanged. (3.2↔3.3)
   - Router-driven skip-reload: a realistic multi-intent conversation
     (coding → math → general → coding) that shares one GGUF file triggers
     exactly ONE real load. (3.3)
   - Router → CoreNode: the routed domain temperature really flows into the
     main-generation coop_params of CoreNode.quick_reply. (3.1→3.3→gen)
   - RAG in pipeline: RetrievalLayer distills context via the 200MB RAG model
     when the server is up, and safely falls back to raw context when it is
     down (pipeline never breaks). (3.4)
3. Full Phase 3 regression gate: the runner also re-runs the three component
   suites (Byte-Q roundtrip, Dispatcher, RAG) so one command validates all of
   Phase 3. (Done)
4. Verified: E2E chain 6/6 asserts + all three component suites PASS under
   ./bin/python (single command).
Run: PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_phase35_integration.py
```

### 3.6 Performance Optimization Testing (Status: ✅ DONE — PERF GATE + REGRESSION)
```
Current: Performance gate that verifies the OPTIMIZATION paths making the
         architecture fit & fast on the RTX 2050 (4 GB) work measurably —
         not just functionally (Phase 3.5).
Tasks:
1. Create moko_agents/test_phase36_integration.py (Done)
2. Performance chain (all server-free / monkeypatched) (Done):
   - Router fast-path: keyword-clear queries resolve at Tier 0/1 (pure
     heuristic, NO embedding/LLM) with small, recorded latency. (3.1)
   - VRAM budget enforcement: VRAMManager enforces the usable budget exactly
     (TOTAL−OS−KV = 3.0 GB); over-budget allocations are rejected. (3.3)
   - Byte-Q saving: the ~40% correction makes 3 domains that would NOT fit
     at full size fit together within the same budget. (3.2↔3.3)
   - LRU/standby eviction: when VRAM is full, _make_room evicts the
     standby/oldest model so the new one fits & total stays ≤ budget. (3.3)
   - Skip-reload efficiency: an 8-turn conversation sharing one GGUF file
     triggers exactly ONE real load (reuse ratio ≥ 80%). (3.3)
   - Predictive preload: preload_likely_next warms the next candidate model
     to STANDBY ahead of time (hides the next switch latency). (3.3)
   - Switch telemetry: get_status reports switch_count, ref_count and
     avg_switch_time_ms accurately for performance observability. (3.3)
3. Full regression gate: the runner also re-runs the entire Phase 3.5 gate
   (E2E chain + Byte-Q/Dispatcher/RAG component suites). (Done)
4. Verified: perf chain 7/7 asserts + full Phase 3.5 gate PASS under
   ./bin/python (single command); no leftover 11435/11437 listeners.
Run: PYTHONPATH=moko_core ./bin/python moko_core/moko_agents/test_phase36_integration.py
```

---

## 5. Phase 4: New Features

### 5.1 Domain-Specialized Models (Status: ✅ MODEL READY)
```
Current: MOKO-Coder-MOKO2.5-1.5B-Lora.gguf finished training
Tasks:
1. Quantize new LoRA model via Byte-Q (Pipeline ready: quantize_domain_models.py)
2. Register in settings.py DOMAIN_MODEL_REGISTRY (Pending)
3. Test switching latency in CoreNode (Pending)
4. Verify CODING intent precision (Pending)
```

### 5.2 RAG + AI Agent (Status: ✅ BASIC DONE)
```
Current: RAGAgent core implemented with Onion Search
Tasks:
1. Create moko_core/moko_agents/rag_agent.py (Done)
2. Implement file organization (Done)
3. Implement knowledge ingestion (Done)
4. Implement session management (Done)
5. Integrate OnionSearchTool (Done)
6. Test with sample files (Pending)
```

### 5.3 Onion Search & TorBot Integration (Status: ✅ DONE)
```
Current: OnionSearchTool + TorBotIntegrator implemented
Tasks:
1. Create moko_core/moko_tools/onion_search.py (Done)
2. Add DARKWEB intent to IntentFirstRouter (Done)
3. Create moko_core/moko_tools/torbot_integrator.py (Done)
4. Integrate TorBot into OnionSearch (Done)
5. Implement status verification and email harvesting (Done)
6. Integrate TorBot-enhanced search into RAGAgent and AnalystNode (Done)
```

---

## 6. OMNI System Reconstruction

### 6.1 Memory Cleanup (Status: ✅ DONE)
```
Current: OMNI data cleared for new system implementation
Tasks:
1. Delete .moko_omni/ contents (Done)
2. Delete legacy kbbi.csv and state files (Done)
3. Delete legacy onion_crawler.db (Done)
4. Prepare for 100% empty OMNI state for user reconstruction (Done)
```

### 6.2 Formula Injection (Status: ❌ PENDING)
```
Target: Re-inject core OMNI formulas and knowledge base.
Tasks:
1. Consolidate scattered inject scripts (Pending)
2. Re-run inject_math_curriculum.py (Pending)
3. Re-run inject_motor_cc_formula.py (Pending)
4. Verify knowledge retrieval in CoreNode (Pending)
```

---

## 6. Timeline

### Week 1-2: Architecture Upgrade
- Day 1-3: Intent-First Router
- Day 4-7: Byte-Q Quantization
- Day 8-14: Multi-Model Dispatcher

### Week 3-4: New Features
- Day 15-21: Domain-Specialized Models
- Day 22-28: RAG + AI Agent

### Week 5: Testing & Optimization
- Day 29-35: Integration testing — ✅ Phase 3.5 E2E chain + component regression gate (test_phase35_integration.py)
- Day 36-42: Performance optimization — ✅ Phase 3.6 perf gate (VRAM budget/eviction, Byte-Q saving, skip-reload, preload, telemetry) + full Phase 3.5 regression (test_phase36_integration.py)

---

## 7. Success Metrics

### Performance
- tok/s: 30 → 50 (via Byte-Q)
- Latency: 100ms → 50ms (via VRAM)
- VRAM usage: 87% → 65% (via multi-model)

### Quality
- Domain accuracy: 75% → 90% (via specialization)
- Routing accuracy: 60% → 85% (via intent-first)
- Knowledge retrieval: 70% → 85% (via RAG agent)

### Efficiency
- Disk: 221 GB → 205 GB (via cleanup)
- RAM: 8 GB → 4 GB (via VRAM optimization)
- Latency: 6s → 0.1s (via VRAM streaming)

---

## 8. Phase 7: AI Coding Revolution (Priority: HIGH)

### 7.1 Omni-Project Indexing (OPI)
```
Target: Memahami 100% struktur project lokal secara struktural (AST).
Tasks:
1. Buat moko_tools/project_indexer.py.
2. Integrasi tree-sitter untuk multi-language support.
3. Simpan relasi antar file ke dalam OmniStorageEngine.
```

### 7.2 Self-Healing Loop (SHL)
```
Target: MOKO bisa memperbaiki kodenya sendiri berdasarkan output compiler.
Tasks:
1. Tambahkan terminal context ke AnalystNode.
2. Implementasi mekanisme retry-on-error dalam CoreNode.
3. Verifikasi dengan test case "Fixing a bug in real-time".
```
