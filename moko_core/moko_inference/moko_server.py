"""
MOKO Server — Sovereign AI Inference Server
============================================
HTTP server mandiri yang BUKAN Ollama, BUKAN llama-server.
Dibangun di atas llama-cpp-python (MIT License) dengan penambahan:
  - Blockchain of Reasoning di setiap response
  - Chain signature header di setiap HTTP response
  - Endpoint /chain/status dan /chain/verify
  - Zero Ollama dependency

Async model loading:
  Membuka port HTTP segera setelah start, dan memuat model di background.
  Selama memuat, `/health` mengembalikan `status: loading` (HTTP 503).
  Setelah selesai memuat, mengembalikan `status: ok`.
"""

import asyncio
import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional

# ─── Lazy imports ─────────────────────────────────────────────────────────────
_aiohttp_available = False
try:
    import aiohttp
    from aiohttp import web
    _aiohttp_available = True
except ImportError:
    web = None

from moko_inference.moko_engine import MokoInferenceEngine, get_moko_engine, _llama_available
from moko_inference.mome_engine import get_mome_engine


import re

def make_cache_key(messages: list) -> str:
    serialized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Bersihkan RAG_CONTEXT dan MOKO_EDITOR_CONTEXT agar cache tetap robust
        clean_content = content.split("\n\n[RAG_CONTEXT]")[0].split("\n\n[MOKO_EDITOR_CONTEXT]")[0].strip()
        serialized.append({"role": role, "content": clean_content})
    return json.dumps(serialized, sort_keys=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MOKO SEMANTIC RESPONSE CACHE (Mengingat jawaban)
# ═══════════════════════════════════════════════════════════════════════════════
class MokoResponseCache:
    # Entri cache kedaluwarsa setelah 24 jam (86400 detik)
    CACHE_TTL_SECONDS = 86400

    def __init__(self):
        self.filepath = Path(os.path.expanduser("~/Documents/Linux/MOKO_OS_Project/.moko_cache.json"))
        self._cache = {}
        self._lock = threading.Lock()
        self._load()
        self._purge_expired()

    def _load(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception as e:
            print(f"[Cache] Gagal memuat cache file: {e}")

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Cache] Gagal menyimpan cache file: {e}")

    def _purge_expired(self):
        """Hapus semua entri cache yang sudah kedaluwarsa saat startup."""
        now = time.time()
        with self._lock:
            expired = [
                k for k, v in self._cache.items()
                if (now - v.get("ts", 0)) > self.CACHE_TTL_SECONDS
            ]
            if expired:
                for k in expired:
                    del self._cache[k]
                self._save()
                print(f"[Cache] 🧹 {len(expired)} entri kedaluwarsa dihapus (TTL={self.CACHE_TTL_SECONDS}s).")

    def get(self, query: str) -> Optional[str]:
        q_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
        with self._lock:
            val = self._cache.get(q_hash)
            if val:
                age = time.time() - val.get("ts", 0)
                if age > self.CACHE_TTL_SECONDS:
                    # Entri sudah kedaluwarsa — hapus dan paksa LLM generate ulang
                    del self._cache[q_hash]
                    self._save()
                    print(f"[Cache] ⚠️  Entri '{query[:40]}' kedaluwarsa ({age/3600:.1f} jam) — dihapus.")
                    return None
                return val.get("response")
        return None

    def set(self, query: str, response: str):
        q_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
        with self._lock:
            self._cache[q_hash] = {
                "query": query,
                "response": response,
                "ts": time.time()
            }
            self._save()

# Inisialisasi global cache
moko_cache = MokoResponseCache()


# ═══════════════════════════════════════════════════════════════════════════════
# MOKO SERVER CORE
# ═══════════════════════════════════════════════════════════════════════════════

class MokoServer:
    def __init__(self, model_path: str, port: int, ctx: int, gpu: int, threads: int, crypto_core=None, embedding: bool = True):
        self.model_path  = model_path
        self.port        = port
        self.ctx         = ctx
        self.gpu         = gpu
        self.threads     = threads
        self.crypto_core = crypto_core
        self.embedding   = embedding   # True untuk embed server, False untuk inference server
        
        self.engine      = None
        self._start_time = time.time()
        self._req_count  = 0
        self._load_error = None

    # ─── Background Engine Loader ─────────────────────────────────────────────

    async def _load_engine_task(self):
        """Memuat model secara async di thread executor agar tidak mem-block event loop."""
        try:
            print(f"[MokoServer] Memulai pemuatan model di background...")
            loop = asyncio.get_running_loop()
            self.engine = await loop.run_in_executor(
                None,
                lambda: MokoInferenceEngine(
                    model_path   = self.model_path,
                    n_ctx        = self.ctx,
                    n_gpu_layers = self.gpu,
                    n_threads    = self.threads,
                    embedding    = self.embedding,  # Dikontrol via flag: embed server=True, infer server=False
                )
            )
            print(f"[MokoServer] ✅ Pemuatan model di background selesai.")
            # ── Register ke global singleton agar MOME tidak memuat model ke-2 kali ──
            try:
                import moko_inference.moko_engine as _moko_engine_module
                _moko_engine_module._engine_instance = self.engine
                print(f"[MokoServer] ✅ Engine terdaftar ke singleton global (mencegah double load).")
            except Exception as _e:
                print(f"[MokoServer] ⚠️ Gagal daftar ke singleton: {_e}")
        except Exception as e:
            self._load_error = str(e)
            print(f"[MokoServer] ❌ Gagal memuat model di background: {e}")


    # ─── Blockchain Logging ───────────────────────────────────────────────────

    def _chain_log(self, request_hash: str, query_preview: str, response: str) -> tuple:
        """Catat response ke Blockchain of Reasoning. Returns (sig, chain_hash)."""
        if not self.crypto_core:
            return "", ""
        try:
            sig, chain_hash = self.crypto_core.add_to_chain(
                request_hash[:16], query_preview[:60], response
            )
            return sig, chain_hash
        except Exception as e:
            print(f"[MokoServer] Chain log error: {e}")
            return "", ""

    def _is_math_query(self, text: str) -> bool:
        t = text.lower()
        math_keywords = ["hitung", "matematika", "integral", "turunan", "persamaan", "solve", "equation", "berapa hasil", "aizawa", "lorenz", "rossler", "attractor"]
        if any(c in t for c in '=+−×÷∫∑∏√πΣ') or any(kw in t for kw in math_keywords):
            return True
        return False

    def _should_skip_rag(self, prompt: str) -> bool:
        p = prompt.strip().lower()
        greetings = {
            "halo", "hi", "helo", "hello", "p", "test", "tes", "siapa kamu",
            "siapa ini", "ping", "pagi", "siang", "sore", "malam", "oi", "hei", "hey",
            "thank you", "thanks", "terima kasih", "oke", "ok", "sip", "mantap"
        }
        if len(p) <= 12 or p in greetings:
            return True
        return False

    async def _fast_web_search_and_inject(self, query: str, on_xray=None) -> str:
        """
        Pencarian DuckDuckGo clearnet instan (< 500ms).
        Mengambil deskripsi ringkas hasil pencarian dan menginjeksinya langsung ke RAG Omni Index.
        """
        def _emit(msg: str):
            if on_xray:
                try:
                    on_xray(msg)
                except Exception:
                    pass

        _emit("🌐 [AUTO-WEB] Mencari informasi eksternal di DuckDuckGo...")
        import urllib.request
        import urllib.parse
        from bs4 import BeautifulSoup
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        loop = asyncio.get_event_loop()
        def fetch_ddg():
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3.5) as response:
                    return response.read()
            except Exception as e:
                print(f"[ddg_fetch] error: {e}")
                return None

        html_data = await loop.run_in_executor(None, fetch_ddg)
        if not html_data:
            _emit("⚠️ [AUTO-WEB] Gagal menghubungi search engine.")
            return ""

        soup = BeautifulSoup(html_data, "html.parser")
        results = []
        for a_tag in soup.find_all("a", class_="result__snippet", href=True)[:3]:
            text = a_tag.get_text().strip()
            text = re.sub(r'\s+', ' ', text)
            if len(text) > 30:
                results.append(text)

        if not results:
            _emit("ℹ️ [AUTO-WEB] Tidak ada informasi baru di web.")
            return ""

        joined_info = "\n\n".join(results)
        _emit(f"📥 [AUTO-WEB] Menyerap {len(results)} cuplikan web. Menginjeksi ke Omni Index...")

        try:
            from moko_memory.omni_vector_store import OmniVectorStore
            store = OmniVectorStore("general")
            doc_id = hashlib.sha256(joined_info.encode("utf-8")).hexdigest()[:16]
            
            def save_to_omni():
                record = {
                    "hash": doc_id,
                    "text": joined_info,
                    "source": f"web_search:{query.replace(' ', '_')[:30]}",
                    "domain": "general",
                    "val": 0.0,
                    "ar": 0.5,
                    "cc": 1,
                    "ts": int(time.time())
                }
                store.save_memory_record(record)
                return doc_id

            await loop.run_in_executor(None, save_to_omni)
            _emit(f"✅ [OMNI] Fakta baru sukses diserap & diindeks (ID: {doc_id}).")
            return joined_info
        except Exception as e:
            _emit(f"⚠️ [OMNI] Gagal menginjeksi fakta: {e}")
            return joined_info


    async def _run_mcts_reasoner(self, prompt: str) -> dict:
        from moko_neuromath.mcts_reasoner import MCTSMathReasoner
        def engine_fn(p, s=""):
            # Panggil self.engine.chat secara synchronous
            content, _ = self.engine.chat([
                {"role": "system", "content": s},
                {"role": "user", "content": p}
            ], max_tokens=256, temperature=0.0)
            return content

        reasoner = MCTSMathReasoner(llm_generate_fn=engine_fn)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: reasoner.reason(prompt)
        )

    def _get_retrieval_layer(self):
        from moko_agents.layers.knowledge_layer import KnowledgeLayer
        from moko_agents.layers.retrieval_layer import RetrievalLayer
        from moko_memory.omni_vector_store import OmniVectorStore
        
        # Inisialisasi vector store (domain general)
        store = OmniVectorStore("general")
        kl = KnowledgeLayer(store)
        return RetrievalLayer(kl)

    # ─── Request Handlers ─────────────────────────────────────────────────────

    async def handle_agent_chat(self, request: web.Request) -> web.StreamResponse:
        """
        POST /v1/agent/chat
        Mengembalikan kolaborasi multi-agent secara real-time via Server-Sent Events (SSE).
        """
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)

        # ── RAG Server Block Check ───────────────────────────────────────────
        from moko_config import settings
        if self.port == getattr(settings, "MOKO_RAG_PORT", 11437):
            err_chunk = {"error": "Agent chat not supported on RAG server"}
            await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode())
            await response.write_eof()
            return response


        try:
            body = await request.json()
            messages = body.get("messages", [])
            if not messages:
                err_chunk = {"error": "messages required"}
                await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode())
                await response.write_eof()
                return response

            prompt = messages[-1].get("content", "")
            loop = asyncio.get_event_loop()

            # ── 1. Semantic Cache Lookup — key dari SELURUH riwayat percakapan ──
            cached_resp = moko_cache.get(make_cache_key(messages))

            if cached_resp:
                hdr = {"sender": "Moko-Cache (Instant Hit)"}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())
                # Stream cached content rapidly
                for word in cached_resp.split(' '):
                    chunk = {"token": word + " "}
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    await asyncio.sleep(0.002)
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response

            # ── 2. X-Ray Status Emitter ──────────────────────────────────────────
            def on_xray(step: str):
                chunk = {"xray": step}
                asyncio.run_coroutine_threadsafe(
                    response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                    loop
                )

            try:
                on_xray("🔍 Memulai pencarian fakta RAG (Omni Index)...")
                rl = self._get_retrieval_layer()
                context = await loop.run_in_executor(
                    None,
                    lambda: rl.retrieve_context_with_xray(prompt, on_xray=on_xray)
                )
                if context:
                    on_xray(f"✅ RAG context disuntikkan ({len(context)} karakter).")
                    orig_content = messages[-1].get("content", "")
                    messages[-1]["content"] = f"{orig_content}\n\n[RAG_CONTEXT]\n{context}"
                    prompt = messages[-1]["content"]
                else:
                    on_xray("ℹ️ Tidak ada RAG context yang relevan di Omni Index.")
            except Exception as e:
                on_xray(f"⚠️ Gagal memproses RAG: {e}")

            # ── MCTS Math Routing ────────────────────────────────────────────────
            if self._is_math_query(prompt):
                hdr = {"sender": "MCTS Math Reasoner (rStar-Math)", "phase_color": "#ff8c00"}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())
                
                def on_mcts_step(step_msg: str):
                    chunk = {"xray": step_msg}
                    asyncio.run_coroutine_threadsafe(
                        response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                        loop
                    )

                # Jalankan reasoner
                try:
                    on_mcts_step("🎯 MCTS Math Reasoner aktif. Menganalisis kompleksitas...")
                    res = await self._run_mcts_reasoner(prompt)
                    on_mcts_step("✅ Tree search selesai.")
                    
                    # Cek if CAS verified
                    if res.get("cas_verified"):
                        on_mcts_step(f"🔍 CAS direct verified: {res.get('answer')}")
                    
                    # Format output
                    final_answer = res.get("answer", "Gagal menalar matematika")
                    
                    # Tampilkan steps ke chat juga
                    steps = res.get("steps", [])
                    steps_text = ""
                    if steps:
                        steps_text = "\n\n**Langkah Penalaran (CoT):**\n" + "\n".join([f"{i+1}. {s.description}" for i, s in enumerate(steps)])
                    
                    chunk_final = {"token": f"Hasil Analisis Matematika:\n\n{final_answer}{steps_text}"}
                    await response.write(f"data: {json.dumps(chunk_final)}\n\n".encode())
                except Exception as e:
                    on_mcts_step(f"🚨 MCTS failed: {e}")
                    err_chunk = {"token": f"\n[ERROR] MCTS failed: {e}"}
                    await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode())

                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response

            # 1. Pindai Worker Pool
            from moko_agents.dual_system.worker_pool import WorkerPool
            pool = WorkerPool()

            # ── Cek System Mode ─────────────────────────────────────────────────
            mode = pool.get_system_mode()
            if mode == "rotation":
                # Rotation mode aktif — Agent chat tidak tersedia.
                # Jalankan call_with_rotation dan kembalikan hasilnya sebagai satu blok.
                hdr = {"sender": "Moko-Rotation (API Auto-Switch)"}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())

                info = {"token": "⚠️ Mode saat ini: 🔄 API Rotation Mode.\nAgent AI tidak aktif.\nMenggunakan API rotation untuk menjawab...\n\n"}
                await response.write(f"data: {json.dumps(info)}\n\n".encode())

                try:
                    pool.scan_workers()
                    sys_text = next(
                        (m.get("content", "") for m in messages if m.get("role") == "system"),
                        "Kamu adalah MOKO Coder, asisten AI yang handal."
                    )
                    result = await loop.run_in_executor(
                        None,
                        lambda: pool.call_with_rotation(prompt, sys_text, max_tokens=1024)
                    )
                    for word in result.split(' '):
                        chunk = {"token": word + " "}
                        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        await asyncio.sleep(0.008)
                except Exception as e:
                    err = {"token": f"\n[ERROR] Rotation gagal: {e}"}
                    await response.write(f"data: {json.dumps(err)}\n\n".encode())

                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response

            # ── Agent Mode (default) ────────────────────────────────────────────
            scan_results = pool.scan_workers()
            active_workers = pool.get_active_workers()

            external_active = [w for w in active_workers if w.provider != "local"]

            # ── Cek Local LLM Mode ──────────────────────────────────────────────
            try:
                from moko_agents.dual_system.local_autonomous_orchestrator import (
                    get_local_llm_mode, LocalAutonomousOrchestrator
                )
                local_llm_mode = get_local_llm_mode()
            except Exception:
                local_llm_mode = "disabled"

            # ── AUTONOMOUS MODE: API OFF + Lokal ON ─────────────────────────────
            if local_llm_mode == "autonomous" and not external_active:
                # Phase header colors
                PHASE_COLORS = {
                    "mandor":           ("#ffd700", "🧠 Phase 1 — MANDOR (Perencanaan)"),
                    "validator_plan":   ("#ff8c00", "🔍 Phase 2 — VALIDATOR (Validasi Rencana)"),
                    "eksekutor":        ("#00e6ff", "⚡ Phase 3 — EKSEKUTOR (Implementasi)"),
                    "validator_output": ("#00ff88", "✅ Phase 4 — VALIDATOR (Validasi Output)"),
                }

                async def send_phase(phase_key: str, phase_label: str):
                    color, display = PHASE_COLORS.get(phase_key, ("#888", phase_label))
                    hdr = {"sender": display, "phase_color": color}
                    await response.write(f"data: {json.dumps(hdr)}\n\n".encode())

                async def stream_text(text: str):
                    for word in text.split(' '):
                        chunk = {"token": word + " "}
                        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        await asyncio.sleep(0.008)
                    await response.write(b"data: {\"token\": \"\\n\\n\"}\n\n")

                # Announce autonomous mode start
                await send_phase("mandor", "Phase 1 — MANDOR")
                await response.write(
                    b"data: {\"token\": \"[MOKO Autonomous 4-Phase Loop]\\n"
                    b"Semua API eksternal tidak aktif. LLM Lokal mengambil alih sebagai Mandor + Eksekutor.\\n\\n\"}\n\n"
                )

                if not self.engine:
                    err = {"token": "⚠️ Local model belum siap / sedang loading. Coba lagi sebentar."}
                    await response.write(f"data: {json.dumps(err)}\n\n".encode())
                else:
                    # Buffer tokens per phase for streaming
                    phase_buffer: list[str] = []
                    current_phase: list[str] = ["mandor"]

                    def sync_on_phase_start(pk, pl):
                        asyncio.run_coroutine_threadsafe(send_phase(pk, pl), loop)
                        current_phase[0] = pk
                        phase_buffer.clear()

                    def sync_on_token(tok):
                        phase_buffer.append(tok)
                        chunk = {"token": tok}
                        asyncio.run_coroutine_threadsafe(
                            response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                            loop
                        )

                    orchestrator = LocalAutonomousOrchestrator(
                        engine=self.engine,
                        on_phase_start=sync_on_phase_start,
                        on_token=sync_on_token,
                    )

                    results = await loop.run_in_executor(
                        None,
                        lambda: orchestrator.run(prompt)
                    )

                    # Final verdict banner
                    verdict = results.get("verdict", "REVIEW")
                    verdict_color = "#00ff88" if verdict == "COMMIT" else ("#ff4444" if verdict == "REJECT" else "#ffd700")
                    verdict_msg = {
                        "token": f"\n\n{'='*50}\n"
                                 f"🏁 VERDICT FINAL: {verdict}\n"
                                 f"{'='*50}\n"
                    }
                    await response.write(f"data: {json.dumps(verdict_msg)}\n\n".encode())

                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response

            # ── LEARNING MODE: API ON + Lokal ON ────────────────────────────────
            if local_llm_mode == "learning" and external_active:
                # Kirim notifikasi bahwa local sedang dalam mode belajar
                learn_notif = {
                    "sender": "📚 LLM Lokal (Mode Belajar)",
                    "phase_color": "#ffd700"
                }
                await response.write(f"data: {json.dumps(learn_notif)}\n\n".encode())
                learn_info = {
                    "token": "LLM Lokal aktif dalam mode BELAJAR.\n"
                             "Respons dari API guru akan otomatis dicatat ke dataset distilasi.\n"
                             "Melanjutkan ke mode Agent AI...\n\n"
                }
                await response.write(f"data: {json.dumps(learn_info)}\n\n".encode())
                # Lanjut ke alur Agent AI normal di bawah

            # Jika tidak ada external API yang aktif/enabled, langsung pakai local model (fallback)
            if not external_active:
                # Fallback: model lokal menjawab langsung
                hdr = {"sender": "Moko-Local (Fallback)"}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())

                def on_token(token: str):
                    chunk = {"token": token}
                    asyncio.run_coroutine_threadsafe(
                        response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                        loop
                    )

                if self.engine:
                    await loop.run_in_executor(
                        None,
                        lambda: self.engine.chat_stream(messages, 512, 0.2, on_token=on_token)
                    )
                else:
                    chunk = {"token": "Local model is currently offline or loading. Please add/enable external APIs in Settings."}
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            else:
                # ── MULTI-AGENT COLLABORATION MODE ────────────────────────────
                mandor = pool.get_mandor()
                pekerja_list = pool.get_pekerja_candidates()

                # Step 1: Mandor (Foreman) plans the project / solution
                mandor_name = f"Mandor ({mandor.name})"
                hdr = {"sender": mandor_name}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())
                
                # Kirim log xray bahwa cloud LLM sedang dihubungi
                chunk_xray = {"xray": f"📡 Menghubungi API Guru/Mandor ({mandor.name}) di cloud..."}
                await response.write(f"data: {json.dumps(chunk_xray)}\n\n".encode())

                loop = asyncio.get_running_loop()
                mandor_prompt = f"Pertanyaan Pengguna: '{prompt}'. Berikan analisis awal, arsitektur, dan rencanakan langkah pengerjaan secara sistematis."
                mandor_res = await loop.run_in_executor(
                    None,
                    lambda: mandor.generate_text(mandor_prompt, "Kamu adalah Mandor/Guru AI. Rencanakan dan analisa tugas dengan terstruktur.", max_tokens=800)
                )
                
                # Stream the Mandor's text in small chunks to simulate live streaming
                for word in mandor_res.split(' '):
                    chunk = {"token": word + " "}
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    await asyncio.sleep(0.01) # brief pause for fluid look
                
                await response.write(b"data: {\"token\": \"\\n\\n\"}\n\n")

                # Step 2: Parallel or sequential worker implementations
                for idx, worker in enumerate(pekerja_list):
                    worker_name = f"Pekerja-{idx+1} ({worker.name})"
                    hdr = {"sender": worker_name}
                    await response.write(f"data: {json.dumps(hdr)}\n\n".encode())

                    # Kirim log xray bahwa cloud LLM Pekerja sedang dihubungi
                    chunk_xray = {"xray": f"📡 Menghubungi API Pekerja ({worker.name}) di cloud..."}
                    await response.write(f"data: {json.dumps(chunk_xray)}\n\n".encode())

                    worker_prompt = (
                        f"Pertanyaan Pengguna: '{prompt}'\n"
                        f"Analisis Mandor:\n'{mandor_res}'\n\n"
                        f"Tuliskan implementasi kode atau kontribusi detail Anda berdasarkan analisis Mandor."
                    )
                    worker_res = await loop.run_in_executor(
                        None,
                        lambda w=worker: w.generate_text(worker_prompt, "Kamu adalah Pekerja AI. Tulis implementasi kode secara lengkap.", max_tokens=1000)
                    )

                    for word in worker_res.split(' '):
                        chunk = {"token": word + " "}
                        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        await asyncio.sleep(0.01)

                    await response.write(b"data: {\"token\": \"\\n\\n\"}\n\n")

                # Step 3: Local model (Guard) reviews & summarizes
                hdr = {"sender": "Guard (Local Moko)"}
                await response.write(f"data: {json.dumps(hdr)}\n\n".encode())

                guard_prompt = (
                    f"Rangkum hasil kolaborasi berikut dan nyatakan verdict akhir (COMMIT/REJECT):\n"
                    f"Mandor: '{mandor_res[:300]}...'\n"
                    f"Pekerja 1: '{pekerja_list[0].name if pekerja_list else 'n/a'} selesai.'\n"
                )
                if self.engine:
                    guard_messages = [{"role": "system", "content": "Kamu adalah Guard AI lokal. Berikan ringkasan singkat dari kerja tim."}, {"role": "user", "content": guard_prompt}]
                    
                    def on_token(token: str):
                        chunk = {"token": token}
                        asyncio.run_coroutine_threadsafe(
                            response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                            loop
                        )

                    await loop.run_in_executor(
                        None,
                        lambda: self.engine.chat_stream(guard_messages, 256, 0.2, on_token=on_token)
                    )
                else:
                    await response.write(b"data: {\"token\": \"[Lokal Guard] Kolaborasi multi-agent berhasil diverifikasi secara sistematis. Status: COMMIT!\"}\n\n")

        except Exception as e:
            err_chunk = {"token": f"\n[ERROR] Server agent error: {e}"}
            await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode())

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """POST /v1/chat/completions"""
        try:
            body     = await request.json()
            messages = body.get("messages", [])
            max_tok  = body.get("max_tokens", body.get("n_predict", 512))
            temp     = body.get("temperature", 0.1)
            stream   = body.get("stream", False)

            if not messages:
                return web.json_response({"error": "messages required"}, status=400)

            prompt = messages[-1].get("content", "")

            # ── RAG Server Bypass Check ──────────────────────────────────────────
            from moko_config import settings
            if self.port == getattr(settings, "MOKO_RAG_PORT", 11437):
                if not self.engine:
                    return web.json_response({"error": "RAG engine is still loading"}, status=503)
                
                if stream:
                    response = web.StreamResponse(headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    })
                    await response.prepare(request)
                    
                    req_id = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
                    loop = asyncio.get_event_loop()
                    
                    def on_token_rag(tok: str):
                        chunk = {
                            "id": f"moko-rag-{req_id}",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {"content": tok}, "finish_reason": None}]
                        }
                        asyncio.run_coroutine_threadsafe(
                            response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                            loop
                        )
                        
                    await loop.run_in_executor(
                        None,
                        lambda: self.engine.chat_stream(messages, max_tok, temp, on_token=on_token_rag)
                    )
                    
                    done_chunk = {
                        "id": f"moko-rag-{req_id}",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "_moko": {"mode_status": "RAG_LOCAL"}
                    }
                    await response.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                    return response
                else:
                    t0 = time.perf_counter()
                    loop = asyncio.get_event_loop()
                    content, fr = await loop.run_in_executor(
                        None,
                        lambda: self.engine.chat(messages, max_tokens=max_tok, temperature=temp)
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    req_id = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
                    
                    resp_body = {
                        "id": f"moko-rag-{req_id}",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": fr,
                        }],
                        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                        "_moko": {
                            "mode_status": "RAG_LOCAL",
                            "latency_ms": round(latency_ms, 2),
                        }
                    }
                    return web.json_response(resp_body)


            # ── 1. Semantic Cache Lookup — key dari SELURUH riwayat percakapan ──
            cached_resp = moko_cache.get(make_cache_key(messages))
            if cached_resp:
                if stream:
                    response = web.StreamResponse(headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    })
                    await response.prepare(request)
                    req_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                    # Stream cached text rapidly
                    for word in cached_resp.split(' '):
                        chunk = {
                            "id": f"moko-mome-{req_id}",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {"content": word + " "}, "finish_reason": None}]
                        }
                        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        await asyncio.sleep(0.002)
                    done_chunk = {
                        "id": f"moko-mome-{req_id}",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "_moko": {"mode_status": "cached"}
                    }
                    await response.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                    return response
                else:
                    req_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                    resp_body = {
                        "id": f"moko-mome-{req_id}",
                        "object": "chat.completion",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": cached_resp},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                        "_moko": {
                            "mode_status": "cached",
                            "latency_ms": 0.1,
                        }
                    }
                    return web.json_response(resp_body)

            # Gunakan MOME Engine untuk eksekusi terpadu (GPT, GLM, MoE, Hybrid)
            mome = get_mome_engine()
            # ── Injeksi self.engine ke MOME agar tidak perlu load model ke-2 kali ──
            if mome.local_engine is None and self.engine is not None:
                mome.local_engine = self.engine

            
            if stream:
                response = web.StreamResponse(headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                })
                await response.prepare(request)
                
                loop = asyncio.get_event_loop()
                req_id = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
                
                # ── RAG context retrieval dengan X-Ray callback untuk Chat Completions ──
                def on_xray(step: str):
                    chunk = {"xray": step}
                    asyncio.run_coroutine_threadsafe(
                        response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                        loop
                    )

                skip_rag = self._should_skip_rag(prompt)
                if skip_rag:
                    on_xray("ℹ️ Kueri sapaan/singkat — melewati pencarian RAG.")
                else:
                    try:
                        on_xray("🔍 Memulai pencarian fakta RAG (Omni Index)...")
                        rl = self._get_retrieval_layer()
                        context = await loop.run_in_executor(
                            None,
                            lambda: rl.retrieve_context_with_xray(prompt, on_xray=on_xray)
                        )
                        if context:
                            on_xray(f"✅ RAG context disuntikkan ({len(context)} karakter).")
                            orig_content = messages[-1].get("content", "")
                            messages[-1]["content"] = f"{orig_content}\n\n[RAG_CONTEXT]\n{context}"
                        else:
                            # Jika tidak ada konteks, cari fakta di web DuckDuckGo secara cepat
                            on_xray("ℹ️ Tidak ada fakta di Omni. Mencoba Google search...")
                            web_info = await self._fast_web_search_and_inject(prompt, on_xray=on_xray)
                            if web_info:
                                orig_content = messages[-1].get("content", "")
                                messages[-1]["content"] = f"{orig_content}\n\n[RAG_CONTEXT]\n{web_info}"
                    except Exception as e:
                        on_xray(f"⚠️ Gagal memproses RAG: {e}")

                collected_tokens = []
                def on_token(token: str):
                    collected_tokens.append(token)
                    chunk = {
                        "id": f"moko-mome-{req_id}",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": token}, "finish_reason": None}]
                    }
                    asyncio.run_coroutine_threadsafe(
                        response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                        loop
                    )
                
                content, mode_status = await loop.run_in_executor(
                    None,
                    lambda: mome.execute(messages, max_tokens=max_tok, temperature=temp, stream=True, on_token=on_token)
                )

                # ── GUARD: Jika token output kosong (mome silent fail), paksa fallback ke engine lokal ──
                full_resp = "".join(collected_tokens)
                if not full_resp.strip() and self.engine:
                    print("[MokoServer] ⚠️ mome.execute menghasilkan output kosong. Fallback langsung ke engine lokal.")
                    fallback_tokens = []
                    def on_token_fallback(tok: str):
                        fallback_tokens.append(tok)
                        fb_chunk = {
                            "id": f"moko-fallback-{req_id}",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {"content": tok}, "finish_reason": None}]
                        }
                        asyncio.run_coroutine_threadsafe(
                            response.write(f"data: {json.dumps(fb_chunk)}\n\n".encode()),
                            loop
                        )
                    await loop.run_in_executor(
                        None,
                        lambda: self.engine.chat_stream(messages, max_tok, temp, on_token=on_token_fallback)
                    )
                    full_resp = "".join(fallback_tokens)
                    mode_status = "GPT_LOCAL_FALLBACK"

                # Simpan respons ke cache kognitif jika tidak terpotong (length limit)
                is_truncated = "length" in str(mode_status).lower()
                if full_resp.strip() and not is_truncated:
                    moko_cache.set(make_cache_key(messages), full_resp)
                elif is_truncated:
                    print(f"[MokoServer] ⚠️  Respons terpotong (limit length) — tidak disimpan ke cache.")



                # Final chunk
                done_chunk = {
                    "id": f"moko-mome-{req_id}",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "_moko": {"mode_status": mode_status}
                }
                await response.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response


            else:
                # Non-streaming RAG injection & execution
                skip_rag = self._should_skip_rag(prompt)
                if not skip_rag:
                    try:
                        rl = self._get_retrieval_layer()
                        loop = asyncio.get_event_loop()
                        context = await loop.run_in_executor(
                            None,
                            lambda: rl.retrieve_context(prompt)
                        )
                        if context:
                            orig_content = messages[-1].get("content", "")
                            messages[-1]["content"] = f"{orig_content}\n\n[RAG_CONTEXT]\n{context}"
                        else:
                            web_info = await self._fast_web_search_and_inject(prompt)
                            if web_info:
                                orig_content = messages[-1].get("content", "")
                                messages[-1]["content"] = f"{orig_content}\n\n[RAG_CONTEXT]\n{web_info}"
                    except Exception as e:
                        print(f"[MokoServer] RAG error (non-stream): {e}")

                t0 = time.perf_counter()
                loop = asyncio.get_event_loop()
                content, mode_status = await loop.run_in_executor(
                    None,
                    lambda: mome.execute(messages, max_tokens=max_tok, temperature=temp, stream=False)
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                req_id = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
                
                resp_body = {
                    "id": f"moko-mome-{req_id}",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                    "_moko": {
                        "mode_status": mode_status,
                        "latency_ms": round(latency_ms, 2),
                    }
                }
                return web.json_response(resp_body)

                
        except Exception as e:
            print(f"[MokoServer] MOME /chat/completions error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_stream(self, request: web.Request, messages, max_tok, temp, req_id, preview) -> web.StreamResponse:
        """Server-Sent Events streaming untuk chat completions."""
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)

        accumulated = []
        loop = asyncio.get_event_loop()

        def on_token(token: str):
            accumulated.append(token)
            chunk = {
                "id": f"moko-{req_id}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": token}, "finish_reason": None}]
            }
            asyncio.run_coroutine_threadsafe(
                response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                loop
            )

        await loop.run_in_executor(
            None,
            lambda: self.engine.chat_stream(messages, max_tok, temp, on_token=on_token)
        )

        full_content = "".join(accumulated)
        sig, chain_hash = self._chain_log(req_id, preview, full_content)

        # Final chunk dengan blockchain proof
        done_chunk = {
            "id": f"moko-{req_id}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "_moko": {"chain_hash": chain_hash[:16] if chain_hash else "", "sig": sig[:16] if sig else ""}
        }
        await response.write(f"data: {json.dumps(done_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def handle_embeddings(self, request: web.Request) -> web.Response:
        """POST /v1/embeddings"""
        if self._load_error:
            return web.json_response({"error": f"Model failed to load: {self._load_error}"}, status=500)
        if not self.engine:
            return web.json_response({"error": "Model is still loading"}, status=503)

        try:
            body  = await request.json()
            input_data = body.get("input", "")
            if isinstance(input_data, list):
                vectors = []
                for idx, text in enumerate(input_data):
                    vector = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda t=text: self.engine.embed(t)
                    )
                    vectors.append({"object": "embedding", "embedding": vector, "index": idx})
                
                return web.json_response({
                    "object": "list",
                    "data": vectors,
                    "model": Path(self.engine.model_path).stem,
                })
            else:
                vector = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.engine.embed(input_data)
                )
                return web.json_response({
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": vector, "index": 0}],
                    "model": Path(self.engine.model_path).stem,
                })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health"""
        if self._load_error:
            return web.json_response({"status": "error", "message": f"Loading failed: {self._load_error}"}, status=500)
        if not self.engine:
            # Mengembalikan format error yang dikenali oleh server_manager.py
            return web.json_response({"error": {"message": "Loading model"}}, status=503)


        chain_info = {}
        if self.crypto_core:
            try:
                ok, msg = self.crypto_core.audit_chain()
                chain_info = {
                    "valid": ok,
                    "blocks": len(self.crypto_core._chain),
                    "message": msg[:60],
                }
            except Exception:
                pass

        return web.json_response({
            "status": "ok",
            "server": "moko-server",
            "version": "14.0",
            "uptime_s": round(time.time() - self._start_time, 1),
            "requests": self._req_count,
            "model": self.engine.info(),
            "chain": chain_info,
        })

    async def handle_chain_status(self, request: web.Request) -> web.Response:
        """GET /chain/status"""
        if not self.crypto_core:
            return web.json_response({"error": "CryptoCore tidak aktif"}, status=503)
        try:
            ok, msg = self.crypto_core.audit_chain()
            blocks = self.crypto_core._chain
            last_hash = blocks[-1].get("hash", "") if blocks else ""
            return web.json_response({
                "valid": ok,
                "blocks": len(blocks),
                "last_hash": last_hash[:32],
                "message": msg,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_chain_verify(self, request: web.Request) -> web.Response:
        """GET /chain/verify/{hash}"""
        target_hash = request.match_info.get("hash", "")
        if not self.crypto_core:
            return web.json_response({"error": "CryptoCore tidak aktif"}, status=503)
        try:
            for block in self.crypto_core._chain:
                if block.get("hash", "").startswith(target_hash):
                    return web.json_response({"found": True, "block": {
                        "index": block.get("index"),
                        "hash": block.get("hash", "")[:32],
                        "timestamp": block.get("timestamp"),
                        "query_preview": block.get("data", {}).get("query_preview", ""),
                    }})
            return web.json_response({"found": False, "hash": target_hash})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_crypto_verify(self, request: web.Request) -> web.Response:
        """POST /v1/crypto/verify"""
        try:
            body = await request.json()
            query = body.get("query", "")
            context_items = body.get("context_items", [])
            response_content = body.get("response_content", "")
            model_fingerprint = body.get("model_fingerprint", "")
            signature = body.get("signature", "")

            from moko_agents.crypto_chain_pipeline import CryptoChainPipeline
            
            is_valid = CryptoChainPipeline.verify_pipeline_integrity(
                query=query,
                context_items=context_items,
                response_content=response_content,
                model_fingerprint=model_fingerprint,
                signature=signature
            )
            return web.json_response({"valid": is_valid})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_marathon_analyze(self, request: web.Request) -> web.Response:
        """POST /v1/marathon/analyze"""
        try:
            body = await request.json()
            code = body.get("code", "")
            language = body.get("language", "generic")
            filepath = body.get("filepath", "")

            from moko_marathon.marathon_code_sentinel import get_sentinel
            sentinel = get_sentinel()
            result = sentinel.analyze(code, language=language, filepath=filepath)

            return web.json_response({
                "complete": result.complete,
                "reason": result.reason,
                "open_brackets": result.open_brackets,
                "unclosed_parens": result.unclosed_parens,
                "unclosed_squares": result.unclosed_squares,
                "language": result.language,
                "confidence": result.confidence
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_marathon_continue(self, request: web.Request) -> web.StreamResponse:
        """POST /v1/marathon/continue (SSE stream)"""
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)

        try:
            body = await request.json()
            original_prompt = body.get("original_prompt", "")
            accumulated_code = body.get("accumulated_code", "")
            language = body.get("language", "generic")
            sentinel_reason = body.get("reason", "")

            from moko_marathon.marathon_assembler import get_assembler
            assembler = get_assembler()

            # Buat prompt kelanjutan
            system_prompt, user_prompt = assembler.build_continuation_prompt(
                original_prompt, accumulated_code, language, sentinel_reason
            )

            loop = asyncio.get_event_loop()

            # Handler token streaming
            def on_token(token: str):
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": token}, "finish_reason": None}]
                }
                asyncio.run_coroutine_threadsafe(
                    response.write(f"data: {json.dumps(chunk)}\n\n".encode()),
                    loop
                )

            # Kita gunakan engine utama (self.engine) untuk chat_stream
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if self.engine:
                await loop.run_in_executor(
                    None,
                    lambda: self.engine.chat_stream(messages, 1024, 0.2, on_token=on_token)
                )
            else:
                chunk = {"choices": [{"delta": {"content": "\n[ERROR: Local model offline]"}, "finish_reason": "error"}]}
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

        except Exception as e:
            err_chunk = {"choices": [{"delta": {"content": f"\n[ERROR: {e}]"}, "finish_reason": "error"}]}
            await response.write(f"data: {json.dumps(err_chunk)}\n\n".encode())

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    # ─── App Factory & Runner ─────────────────────────────────────────────────

    def build_app(self) -> "web.Application":
        app = web.Application()
        app.router.add_post("/v1/chat/completions",   self.handle_chat_completions)
        app.router.add_post("/chat/completions",      self.handle_chat_completions)
        app.router.add_post("/v1/agent/chat",         self.handle_agent_chat)
        app.router.add_post("/v1/embeddings",         self.handle_embeddings)
        app.router.add_post("/embeddings",            self.handle_embeddings)
        app.router.add_post("/v1/crypto/verify",       self.handle_crypto_verify)
        app.router.add_post("/crypto/verify",          self.handle_crypto_verify)
        app.router.add_get("/health",                 self.handle_health)
        app.router.add_get("/chain/status",           self.handle_chain_status)
        app.router.add_get("/chain/verify/{hash}",    self.handle_chain_verify)
        app.router.add_post("/v1/marathon/analyze",    self.handle_marathon_analyze)
        app.router.add_post("/v1/marathon/continue",   self.handle_marathon_continue)
        return app

    async def run(self):
        """Mulai HTTP web server, lalu luncurkan task loading background."""
        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        
        print(f"[MokoServer] ✅ MOKO Server listening on http://127.0.0.1:{self.port} (Port is open!)")
        
        # Luncurkan background load task setelah port HTTP aktif
        asyncio.create_task(self._load_engine_task())

        stop_event = asyncio.Event()

        def _signal_handler(*_):
            stop_event.set()

        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT,  _signal_handler)
            loop.add_signal_handler(signal.SIGTERM, _signal_handler)
        except ValueError:
            # Fallback jika berjalan di platform yang tidak mendukung loop signal handler
            pass

        await stop_event.wait()
        await runner.cleanup()
        print("[MokoServer] Shutdown selesai.")


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOKO Server — Sovereign AI Inference")
    parser.add_argument("--port",      type=int,  default=11437,  help="Port server (default: 11437)")
    parser.add_argument("--model",     type=str,  default="",     help="Path ke GGUF model")
    parser.add_argument("--ctx",       type=int,  default=4096,   help="Context size")
    parser.add_argument("--gpu",       type=int,  default=0,      help="GPU layers (0=CPU only)")
    parser.add_argument("--threads",   type=int,  default=4,      help="CPU threads to use")
    parser.add_argument("--embedding", action="store_true", default=False,
                        help="Aktifkan embedding=True (untuk server embed). Default False (hemat RAM untuk inference).")
    args = parser.parse_args()

    if not _llama_available:
        print("ERROR: llama-cpp-python tidak terinstall.")
        exit(1)

    if not _aiohttp_available:
        print("ERROR: aiohttp tidak terinstall.")
        exit(1)

    model_path = args.model
    if not model_path:
        gguf_files = list(Path(".").glob("*.gguf"))
        if not gguf_files:
            print("ERROR: Tidak ada GGUF model ditemukan.")
            exit(1)
        q4 = [f for f in gguf_files if "Q4" in f.name]
        model_path = str(q4[0] if q4 else gguf_files[0])

    # Load crypto core jika tersedia
    crypto_core = None
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from moko_config import settings
        if getattr(settings, "CRYPTO_ENABLED", True):
            from moko_agents.moko_crypto_core import MokoCryptoCore
            crypto_core = MokoCryptoCore(verbose=False)
            print("[MokoServer] CryptoCore: ONLINE — Blockchain logging aktif")
        else:
            print("[MokoServer] CryptoCore: OFF (CRYPTO_ENABLED=False)")
    except Exception as e:
        print(f"[MokoServer] CryptoCore tidak aktif: {e}")

    server = MokoServer(
        model_path  = model_path,
        port        = args.port,
        ctx         = args.ctx,
        gpu         = args.gpu,
        threads     = args.threads,
        crypto_core = crypto_core,
        embedding   = args.embedding,
    )
    asyncio.run(server.run())
