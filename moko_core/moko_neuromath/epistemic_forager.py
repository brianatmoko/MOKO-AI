import json
import time
import psutil
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from moko_agents.llm_engine import engine
from moko_config import settings
from moko_neuromath.web_crawler import multi_crawler
from moko_neuromath.thalamus_gate import thalamus_gate
from moko_neuromath.dopamine_scheduler import dopamine_scheduler
from moko_neuromath.basal_ganglia import basal_ganglia
from moko_neuromath.serotonin_system import serotonin_node
from moko_neuromath.ach_system import ach_system
from moko_neuromath.sleep_scheduler import sleep_scheduler
from moko_neuromath.mutation_queue import mutation_queue
from moko_neuromath.fep_engine import FEPEngine
from moko_neuromath.bcm_synapse import BCMSynapse
from moko_memory.disk_manager import DiskManager
from moko_memory.math_omni import math_omni
from moko_memory.rsa_storage import map_topic_to_domain
from moko_cpu.governor import CPUGovernor
from moko_utils.text_utils import chunk_text_code_aware


class EpistemicForager(QThread):
    log_signal    = pyqtSignal(str, str)   # (text, color)
    sleep_trigger = pyqtSignal()           # Memicu SleepWorker di main panel

    def __init__(self, hyperspeed: bool = False):
        super().__init__()
        self.sidecar_path = Path(settings.WORKSPACE_DIR) / ".math_omni" / "formula_sidecar.jsonl"
        self.is_running   = True
        self.sleep_active = False
        self.hyperspeed   = hyperspeed

    def stop(self):
        self.is_running = False

    def run(self):
        self.log_signal.emit("╔══════════════════════════════════════════╗", "#00ffff")
        self.log_signal.emit("║  EPISTEMIC FORAGER v2 — MULTI-DOMAIN     ║", "#00ffff")
        self.log_signal.emit("║  Math | Logic | Code | AI | Technology   ║", "#00ffff")
        self.log_signal.emit("╚══════════════════════════════════════════╝", "#00ffff")
        if self.hyperspeed:
            self.log_signal.emit("⚡ Hyperspeed Mode: AKTIF (Batching + Zero Sleep)", "#00ff88")

        self.disk_mgr = DiskManager(settings.WORKSPACE_DIR)

        # --- RAM Guard Helper ---
        def _check_ram_pressure() -> bool:
            """Return True jika RAM aman, False jika harus pause."""
            mem = psutil.virtual_memory()
            if mem.percent > 85 or mem.available < (2.5 * 1024**3):
                self.log_signal.emit(
                    f"⚠️ RAM GUARD: RAM tersisa {mem.available / 1024**3:.1f} GB "
                    f"({100 - mem.percent:.0f}% free) — Forager Pause...",
                    "#ff6600"
                )
                wait_sec = 10 if self.hyperspeed else 60
                for _ in range(wait_sec):
                    if not self.is_running:
                        break
                    self.sleep(1)
                return False
            return True

        while self.is_running:
            try:
                # 1. Pilih domain & query (BasalGanglia — Fase 3, backward-compatible dengan DopamineScheduler)
                # Modulasi temperatur eksplorasi dari SerotoninNode (cognitive_flexibility)
                temperature = round(0.3 + 0.4 * serotonin_node.cognitive_flexibility, 4)
                domain, query = basal_ganglia.get_next_domain(temperature=temperature)
                
                # Dapatkan monolog batin deterministik (masih dari DopamineScheduler)
                renungan = dopamine_scheduler.get_monologue(domain, query)

                self.log_signal.emit(f"\n💭 MOKO Merenung: \"{renungan}\"", "#ffaa00")
                self.log_signal.emit(f"🔎 MOKO Memutuskan Mencari: \"{query}\" (Fokus: {domain})", "#aaaaff")

                # 2. Fetch artikel dari web
                results = multi_crawler.route_and_fetch(query, topic_hint=domain.lower())
                if not results:
                    self.log_signal.emit("❌ Tidak ada konten ditemukan untuk query ini. Istirahat...", "#ff4444")
                    wait_sec = 2 if self.hyperspeed else 15
                    for _ in range(wait_sec):
                        if not self.is_running: break
                        self.sleep(1)
                    continue

                self.log_signal.emit(f"✅ Ditemukan {len(results)} artikel untuk dipelajari.", "#00ff88")

                # 3. Proses Setiap Artikel (AWAKE Mode — Raw Encoding)
                for article in results:
                    if not self.is_running:
                        break
                    
                    src = article.get("source", "?")
                    url = article.get("url", "")
                    text = article.get("text", "")
                    
                    if not text or len(text) < 100:
                        continue

                    self.log_signal.emit(f"\n📖 Membaca [{src}]: {url[:80]}...", "#00ffff")
                    
                    source_name = f"{src.lower()}_{domain.replace(' ', '_')}.txt"
                    
                    # Split ke dalam logical chunks (paragraf) secara code-aware
                    chunks = chunk_text_code_aware(text)
                        
                    self.log_signal.emit(f"   Memisahkan teks menjadi {len(chunks)} chunk.", "#888")

                    # Batasi chunk agar tidak berlebihan jika tidak hyperspeed
                    max_chunks = 30 if self.hyperspeed else 10
                    target_chunks = chunks[:max_chunks]
                    
                    # Batch embedding
                    self.log_signal.emit(f"   Menghasilkan embedding untuk {len(target_chunks)} chunk secara batch...", "#aaa")
                    embs = engine.get_embeddings_batch(target_chunks)
                    if not embs or len(embs) != len(target_chunks):
                        embs = [engine.get_embedding(c) for c in target_chunks]

                    batch_items = []
                    valid_indices = []
                    
                    for idx, (chunk, emb) in enumerate(zip(target_chunks, embs)):
                        if not emb or len(emb) != 768:
                            continue

                        # Thalamus novelty filter
                        novelty_score = thalamus_gate.get_novelty_score(emb)
                        if not thalamus_gate.is_novel(emb):
                            continue

                        self.log_signal.emit(f"   [Chunk {idx+1}/{len(target_chunks)}] Novelty: {novelty_score:.2f} → Merekam...", "#00ff88")
                        
                        raw_enriched = f"[SOURCE: {source_name}]\n{chunk}"
                        batch_items.append({
                            "file_path": source_name,
                            "text_chunk": raw_enriched,
                            "fp32_vector": emb
                        })
                        valid_indices.append((chunk, emb, novelty_score))

                    if batch_items:
                        # Batch ingest dengan perutean domain dinamis
                        target_dom = map_topic_to_domain(domain)
                        self.disk_mgr.ingest_chunks_batch(batch_items, source_type="factual", domain=target_dom)
                        
                        # Jalankan proses Fast FEP & Dopamine update untuk item yang lolos
                        for chunk, emb, novelty_score in valid_indices:
                            # Fast FEP
                            math_result = math_omni.search(emb)
                            formula_id = math_result.get("formula_id")
                            
                            if formula_id:
                                formula = math_omni._find_entry(formula_id)
                                if formula and "trigger_text" in formula:
                                    formula_emb = engine.get_embedding(formula["trigger_text"])
                                    if len(formula_emb) == 768:
                                        surprisal = FEPEngine.calculate_free_energy_fast(emb, formula_emb)
                                        
                                        if surprisal > 0.60:
                                            self.log_signal.emit(f"      ⚠️ Surprisal tinggi ({surprisal:.2f}) → Synaptic Tagging pada {formula_id}.", "#ff8800")
                                            mutation_queue.enqueue(formula_id, chunk, surprisal, domain)
                                        else:
                                            BCMSynapse.apply_plasticity(formula_id, is_satisfied=True)

                            # Update dopamine / Fase 3: update reward ke BasalGanglia
                            basal_ganglia.update_reward(domain, novelty_score)
                            # Update ACh berdasarkan novelty yang baru ditemukan
                            ach_system.update_ach(novelty_score)
                            # Update sleep scheduler
                            sleep_scheduler.tick_chunk()

                            if not self.hyperspeed:
                                CPUGovernor.breathe("ForagerAwake")
                                self.sleep(1)

                    sleep_scheduler.tick_article()

                    # Cek tidur
                    if sleep_scheduler.should_sleep():
                        reason = sleep_scheduler.get_sleep_reason()
                        self.log_signal.emit(f"\n💤 [SLEEP TRIGGER] Memasuki siklus konsolidasi memori. Alasan: {reason}", "#aaaa99")
                        self.sleep_active = True
                        self.sleep_trigger.emit()
                        
                        while self.sleep_active and self.is_running:
                            self.sleep(1)
                            
                        self.log_signal.emit("🌅 [WAKE UP] Bangun tidur! Energi kembali segar. Melanjutkan belajar...\n", "#ffff00")

                # Jeda sebelum mencari topik baru
                wait_sec = 2 if self.hyperspeed else 30
                self.log_signal.emit(f"💤 Jeda {wait_sec} detik sebelum siklus pencarian berikutnya...", "#555")
                for _ in range(wait_sec):
                    if not self.is_running: break
                    self.sleep(1)

            except Exception as e:
                self.log_signal.emit(f"⚠️ Forager Error: {e}", "#ff0000")
                self.sleep(15)

        # Unload model saat thread berhenti
        try:
            engine.release_model()
        except Exception:
            pass
        self.log_signal.emit("🛑 Epistemic Forager dihentikan.", "#ffaa00")
