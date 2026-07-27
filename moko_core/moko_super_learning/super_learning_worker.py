import time
import json
import re
import gc
import psutil
from PyQt6.QtCore import QThread, pyqtSignal

from moko_agents.llm_engine import engine
from moko_memory.disk_manager import DiskManager
from moko_memory.rsa_storage import map_topic_to_domain
from moko_memory.gc_tuner import gc_tuner
from moko_memory.math_omni import math_omni
from moko_neuromath.web_crawler import multi_crawler
from moko_neuromath.thalamus_gate import thalamus_gate
from moko_inference.tor_manager import ensure_tor_ready
from moko_config import settings

from moko_super_learning.curriculum import CurriculumManager, STATUS_MASTERED
from moko_super_learning.satisfaction_engine import SatisfactionEngine

# Ganti split dengan code_utils
from moko_utils.text_utils import chunk_text_code_aware

# Import prompt dan modul neuro
from moko_cpu.governor import CPUGovernor
from moko_neuromath.fep_engine import FEPEngine
from moko_neuromath.active_inference import ActiveInference
from moko_neuromath.bcm_synapse import BCMSynapse
from moko_neuromath.hebb_linker import hebb_linker
from moko_agents.prompts import PROMPT_DEEP_SYNTHESIS

class SuperLearningWorker(QThread):
    log_signal = pyqtSignal(str, str)             # (text_log, color)
    progress_signal = pyqtSignal(str, float, str)  # (node_id, progress_pct, status_info)
    done_signal = pyqtSignal(str, str)            # (node_id, summary_text)

    def __init__(self, node_id: str, curriculum_mgr: CurriculumManager, hyperspeed: bool = False):
        super().__init__()
        self.node_id = node_id
        self.curriculum_mgr = curriculum_mgr
        self.hyperspeed = hyperspeed
        self.is_running = True
        
        self.node = self.curriculum_mgr.nodes.get(self.node_id)
        self.learned_points = []
        
        # Deteksi otomatis domain offensive / anonymity / darkweb
        self.is_dark_web = self.node.subject == "Offensive Cyber" or "onion" in self.node.node_id or "tor" in self.node.node_id

    def stop(self):
        self.is_running = False

    def run(self):
        self.disk_mgr = DiskManager(settings.WORKSPACE_DIR)
        
        self.log_signal.emit("═══════════════════════════════════════════", "#00ff88")
        self.log_signal.emit(f"🚀 [SISTEM BELAJAR SUPER] Memulai: {self.node.title}", "#00ff88")
        self.log_signal.emit(f"   Subjek: {self.node.subject} | Prasyarat: {self.node.prerequisites}", "#888")
        self.log_signal.emit("═══════════════════════════════════════════\n", "#00ff88")

        if self.hyperspeed:
            self.log_signal.emit("⚡ Mode Hyperspeed Aktif: Zero Sleep + Batching agresif.", "#00ff88")

        # ── RAM Guard ──
        mem = psutil.virtual_memory()
        if mem.percent > 85 or mem.available < (2.0 * 1024**3):
            self.log_signal.emit(f"⚠️ RAM Kritis ({mem.percent}%). Menunda pembelajaran super...", "#ff6600")
            wait_sec = 5 if self.hyperspeed else 45
            for _ in range(wait_sec):
                if not self.is_running: return
                self.msleep(1000)

        # ── Tor check jika domain network_security/onion_anonymity/darkweb ──
        if self.is_dark_web:
            self.log_signal.emit("🧅 [Tor Manager] Memeriksa koneksi Tor untuk materi Cyber/Dark Web...", "#ff88ff")
            tor_ok = ensure_tor_ready(log_cb=lambda msg, col="#ff88ff": self.log_signal.emit(msg, col))
            if not tor_ok:
                self.log_signal.emit("⚠️ Tor daemon tidak aktif. Fallback menggunakan gerbang clearnet.", "#ffaa00")
            else:
                self.log_signal.emit("🟢 Tor terowongan aktif! Proses perayapan .onion berjalan mulus.", "#00ff88")

        cycle = 0
        consecutive_fails = 0
        
        # Mulai loop belajar super
        while self.is_running and self.node.mastery < self.node.min_mastery_satisfied:
            cycle += 1
            
            # Heartbeat RAM
            if cycle % 5 == 0:
                mem = psutil.virtual_memory()
                self.log_signal.emit(
                    f"📊 [Progression] Progress: {self.node.mastery:.1f}% | "
                    f"Synaptic Weight: {self.node.synaptic_weight:.2f} | "
                    f"RAM: {mem.available/1024**3:.1f} GB sisa",
                    "#666"
                )

            # Bersihkan RAM secara agresif setiap 15 siklus
            if cycle % 15 == 0:
                gc.collect(2)
                try:
                    gc_tuner._malloc_trim()
                except:
                    pass

            self.log_signal.emit(f"\n[Siklus {cycle}] Merumuskan sub-materi secara otonom...", "#aaa")
            
            # Rumuskan query kognitif menggunakan LLM
            query = ""
            learned_str = ", ".join(self.learned_points[-5:]) if self.learned_points else "None"
            keywords_str = ", ".join(self.node.keywords)
            
            prompt = (
                f"Curriculum Node: '{self.node.title}'.\n"
                f"Syllabus keywords: {keywords_str}.\n"
                f"Already analyzed aspects: '{learned_str}'.\n"
                f"Write ONE highly specific technical search query (max 4 words) in English to acquire deep knowledge "
                f"relevant to '{self.node.title}' and the syllabus keywords. "
                f"Do not duplicate already learned aspects. Output ONLY the query string, without quotes or explanations."
            )
            
            try:
                coop = {"enable_thinking": False, "num_predict": 32}
                raw_query = engine.generate_text(prompt, "Output search query only.", coop_params=coop).strip()
                # Bersihkan kueri
                from moko_utils.text_utils import _clean_llm_query
                query = _clean_llm_query(raw_query)
            except Exception as e:
                self.log_signal.emit(f"⚠️ Gagal merumuskan kueri via LLM: {e}", "#ffaa00")
                query = ""

            if not query:
                # Fallback ke kata kunci kurikulum acak
                query = random.choice(self.node.keywords) if hasattr(self, "random") else self.node.keywords[0]

            self.log_signal.emit(f"🔎 Perayapan Web: \"{query}\"...", "#00e6ff")
            
            results = []
            try:
                if self.is_dark_web:
                    results = multi_crawler.route_and_fetch_darkweb(query, topic_hint=self.node.title)
                else:
                    results = multi_crawler.route_and_fetch(query, topic_hint=self.node.title)
            except Exception as e:
                self.log_signal.emit(f"❌ Crawling gagal: {e}", "#ff4444")
                results = []

            if not results:
                # Coba fallback ke DuckDuckGo
                try:
                    results = multi_crawler.search_and_crawl_duckduckgo(query, max_results=2)
                except Exception as e:
                    self.log_signal.emit(f"❌ Fallback DuckDuckGo gagal: {e}", "#ff4444")

            if not results:
                self.log_signal.emit("❌ Tidak menemukan data baru pada kueri ini. Melakukan adaptasi kueri...", "#ffaa00")
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    # Adaptive backoff
                    self.log_signal.emit("⏸️ Adaptasi kognitif tertunda, istirahat 10 detik...", "#ff8800")
                    time.sleep(10)
                    consecutive_fails = 0
                continue

            consecutive_fails = 0
            self.log_signal.emit(f"✅ Ditemukan {len(results)} artikel. Melakukan dekonstruksi logika...", "#00ff88")
            
            # Proses artikel
            self._process_curriculum_articles(results)
            
            if not self.hyperspeed:
                time.sleep(4.0)

        # ── KELUAR LOOP / SELESAI ──
        if self.node.mastery >= self.node.min_mastery_satisfied:
            # Sukses mastery
            self.node.status = STATUS_MASTERED
            self.curriculum_mgr._update_prerequisite_states()
            self.curriculum_mgr.save_state()
            
            reward_msg = SatisfactionEngine.get_dopamine_reward()
            self.log_signal.emit("\n🎉 🌟 🏆 <b>MATERI TELAH DIKUASAI (MASTERED)</b> 🏆 🌟 🎉", "#00ff88")
            self.log_signal.emit(f"✨ <i>{reward_msg}</i>", "#00ff88")
            self.progress_signal.emit(self.node_id, self.node.mastery, STATUS_MASTERED)
            
            summary = (
                f"Selamat! Anda telah menguasai materi **'{self.node.title}'** (Progress: {self.node.mastery:.1f}%). "
                f"MOKO telah merayap, menganalisis, dan memperkuat jalur logika di Math-Omni untuk subtopik tersebut. "
                "Materi prasyarat berikutnya yang bergantung pada modul ini sekarang telah terbuka! 🔓"
            )
        else:
            # Dibatalkan pengguna
            self.log_signal.emit(f"\n🛑 Pembelajaran super '{self.node.title}' dihentikan oleh pengguna.", "#ff4444")
            self.progress_signal.emit(self.node_id, self.node.mastery, self.node.status)
            summary = f"Pembelajaran materi **'{self.node.title}'** dihentikan sementara pada progress {self.node.mastery:.1f}%."

        # RAM clean
        try:
            gc_tuner.post_learning_session()
        except:
            pass
            
        self.done_signal.emit(self.node_id, summary)

    def _process_curriculum_articles(self, articles):
        """Dekonstruksi artikel & ekstraksi logika kurikulum."""
        batch_size = 2
        for batch_idx in range(0, len(articles), batch_size):
            if not self.is_running:
                break
            
            batch = articles[batch_idx:batch_idx+batch_size]
            for art in batch:
                if not self.is_running:
                    break
                    
                src = art.get("source", "Web")
                url = art.get("url", "")
                text = art.get("text", "")
                if not text or len(text) < 100:
                    continue

                # Kumpulkan learned points
                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                title = ""
                for ln in lines[:3]:
                    if ln.startswith("JUDUL:"):
                        title = ln.replace("JUDUL:", "").strip()
                        break
                if not title and lines:
                    title = lines[0][:50]
                if title:
                    clean_title = re.sub(r'^[*\-•\s\d\.:]+', '', title).strip()
                    if clean_title and clean_title not in self.learned_points:
                        self.learned_points.append(clean_title)

                chunks = chunk_text_code_aware(text)
                source_name = f"super_learn_{self.node_id}_{src.lower().replace('/', '_')}.txt"
                
                # Hanya proses sedikit chunk per artikel agar tidak overload RAM dan waktu LLM
                max_chunks = 6 if self.hyperspeed else 2
                for idx, chunk in enumerate(chunks[:max_chunks]):
                    if not self.is_running:
                        break
                    
                    # RAM check
                    mem = psutil.virtual_memory()
                    if mem.percent > 88:
                        self.log_signal.emit(f"⚠️ RAM Tinggi ({mem.percent}%). Menunggu pembersihan...", "#ffaa00")
                        gc.collect(2)
                        time.sleep(2)
                        continue

                    # Embedding
                    emb = engine.get_embedding(chunk)
                    if len(emb) != 768:
                        continue

                    # Novelty check via Thalamus Gate
                    if not thalamus_gate.is_novel(emb):
                        continue

                    self.log_signal.emit(f"   📖 Menganalisis logis chunk {idx+1}...", "#aaa")
                    
                    # Deep Synthesis
                    prompt = PROMPT_DEEP_SYNTHESIS.replace("{text}", chunk)
                    light_params = {
                        "num_thread": 2,
                        "num_ctx": 2048,
                        "num_predict": 512,
                        "keep_alive": "5m",
                        "reason": f"🧠 Super Learning Logic Extraction: {self.node_id}"
                    }
                    
                    try:
                        llm_response = engine.generate_text(
                            prompt, "Return pure JSON only.",
                            model_override=settings.MODEL_LLM,
                            coop_params=light_params
                        )
                        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                        if not json_match:
                            continue
                        data = json.loads(json_match.group(0))
                        logic = data.get("logic", "A")
                        arousal = str(data.get("arousal", "2"))
                        depth = data.get("depth", "D5")
                        gagasan = data.get("gagasan_pokok", "")
                        instruction = data.get("instruction", "Pikirkan secara analitis.")
                    except Exception as e:
                        self.log_signal.emit(f"   ⚠️ Parsing JSON logika gagal: {e}", "#ffaa00")
                        continue

                    # Gagasan embedding
                    gagasan_emb = engine.get_embedding(gagasan)
                    if len(gagasan_emb) != 768:
                        gagasan_emb = emb

                    # Math-Omni check
                    math_result = math_omni.search(gagasan_emb, logic=logic, arousal=arousal, depth=depth)
                    formula_id = math_result.get("formula_id")
                    
                    if not formula_id:
                        formula_id = math_omni.save_formula(logic, arousal, depth, gagasan, instruction, gagasan_emb)
                    else:
                        instruction = math_result.get("instruction", instruction)

                    # FEP Engine
                    surprisal, is_satisfied, fep_reason = FEPEngine.calculate_free_energy(chunk, instruction, coop_params=light_params)
                    
                    # Plastisitas BCM (LTP / LTD)
                    BCMSynapse.apply_plasticity(formula_id, is_satisfied)
                    
                    # Reload formula info untuk ambil weight terbaru
                    updated_formula = math_omni.search(gagasan_emb, logic=logic, arousal=arousal, depth=depth)
                    weight = updated_formula.get("synaptic_weight", 1.0)

                    # Jika tidak puas, mutasikan formula kognitif (Active Inference)
                    if not is_satisfied:
                        new_instruction = ActiveInference.mutate_formula(chunk, instruction, fep_reason, coop_params=light_params)
                        formula_id = math_omni.save_formula(logic, arousal, depth, gagasan, new_instruction, gagasan_emb)
                        self.log_signal.emit(f"   🚨 FEP Surprisal {surprisal:.2f} (Gelisah) → Mutasi: {formula_id}", "#ff4444")
                    else:
                        self.log_signal.emit(f"   ✅ FEP Surprisal {surprisal:.2f} (Puas) → LTP Sinapsis", "#00ff00")

                    # Fire Hebbian Linker
                    hebb_linker.fire_together(source_name.split(".")[0], formula_id)

                    # Ingest episodic memory ke Omni-Index
                    raw_enriched = f"[SUPER_LEARNING: {self.node_id}] [FORMULA: {formula_id}] [SOURCE: {url[:80]}]\n{chunk}"
                    batch_to_ingest = [
                        {"file_path": source_name, "text_chunk": raw_enriched, "fp32_vector": gagasan_emb}
                    ]
                    
                    target_dom = map_topic_to_domain(self.node.title)
                    self.disk_mgr.ingest_chunks_batch(batch_to_ingest, domain=target_dom)

                    # Hitung kemajuan belajar (Mastery) via SatisfactionEngine
                    new_mastery, delta, boost = SatisfactionEngine.calculate_mastery_step(
                        self.node.mastery, surprisal, weight, is_satisfied, self.hyperspeed
                    )
                    
                    # Update status kurikulum
                    self.curriculum_mgr.update_progress(self.node_id, new_mastery, weight, surprisal)
                    
                    # Emit visual progress ke GUI
                    self.log_signal.emit(
                        f"   Consolidation: Mastery +{delta}% ({self.node.mastery:.1f}%) | Bobot: {weight:.2f} (Boost: x{boost})",
                        "#00ff88"
                    )
                    self.progress_signal.emit(self.node_id, self.node.mastery, self.node.status)
                    
                    if not self.hyperspeed:
                        time.sleep(2.0)
                        
            # GC setelah batch artikel
            gc.collect()
            time.sleep(1.0)
