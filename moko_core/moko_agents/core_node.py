from moko_agents.llm_engine import engine
from moko_agents.intent_router import get_intent_router
from moko_agents.model_dispatcher import get_dispatcher
from moko_memory.disk_manager import DiskManager
from moko_cpu.governor import CPUGovernor
from moko_agents.omni_direct_answer import omni_direct_answer
from moko_agents.neural_layer_mixer import neural_layer_mixer
from moko_memory.neural_working_memory import neural_working_memory
from moko_agents.auto_continue_engine import auto_continue_engine
from moko_config import settings

# ── Hybrid Chat Engine (API Gateway Fast-Path) ────────────────────────────────
try:
    from moko_agents.hybrid_chat_engine import HybridChatEngine
    _hybrid_chat_available = True
except Exception as _hce_err:
    HybridChatEngine = None
    _hybrid_chat_available = False
    print(f"[CoreNode] HybridChatEngine tidak tersedia: {_hce_err}")

_CRYPTO_ON = getattr(settings, "CRYPTO_ENABLED", True)

# (Crypto components removed per Phase 4 Rollback)
_crypto_tokenizer = None
_tiered_memory = None
_crypto_context = None

# (Crypto imports deleted)

# ── Realtime Knowledge & Anti-Hallucination Systems ───────────────────────────
try:
    from moko_agents.live_knowledge_engine import get_live_knowledge_engine
    _live_engine = get_live_knowledge_engine()
    _live_engine.start()
except Exception as _lke_err:
    _live_engine = None
    print(f"[CoreNode] LiveKnowledge Engine tidak aktif: {_lke_err}")

try:
    from moko_agents.symbol_verifier import get_symbol_verifier
    _symbol_verifier = get_symbol_verifier()
except Exception:
    _symbol_verifier = None

# ── StepLogicProver: Z3-powered Chain-of-Thought Verifier ─────────────────
try:
    from moko_neuromath.step_logic_prover import StepLogicProver
    _step_prover = StepLogicProver()
    print("  ✅ [CoreNode] StepLogicProver (Z3 CoT Verifier): ONLINE")
except Exception as _slp_err:
    _step_prover = None
    print(f"[CoreNode] StepLogicProver tidak aktif: {_slp_err}")

# ── ARMSEngine: Applied Real-World Mathematics Solver ────────────────────
try:
    from moko_neuromath.arms_engine import get_arms
    _arms_engine = get_arms(verbose=False)
    print("  ✅ [CoreNode] ARMSEngine (Applied Math Solver): ONLINE")
except Exception as _arms_err:
    _arms_engine = None
    print(f"[CoreNode] ARMSEngine tidak aktif: {_arms_err}")

# ── CacheAuditor: Background Cache Integrity Daemon ──────────────
try:
    # (CryptoResponseCache removed)
    _cache_auditor = None
except Exception as _crt_err:
    _cache_auditor = None

# ── ZKML Model Merkle Checkpoint Gate (removed - non-crypto architecture) ──
_zkml_available = False

class CoreNode:
    def __init__(self, disk_mgr: DiskManager):
        self.disk_mgr       = disk_mgr
        print("  ✅ [CoreNode] MokoCore Node initialized (OMNI System)")

        # ── Inference Engine Pre-Warm (Background) ───────────────────────────
        # CATATAN: Prewarm otomatis dinonaktifkan untuk mencegah server auto-start
        # yang memboroskan RAM saat MOKO IDE baru dibuka.
        # Server hanya dijalankan saat user klik "Start Server" di Status Panel,
        # atau saat user pertama kali mengirim pesan (lazy start via _wait_until_ready).
        # Untuk start manual: systemctl --user start moko-qwen.service
        pass

        # ── Hybrid Chat Engine: Background Worker Scan ───────────────────────
        # Scan semua API gateway di background agar siap saat user pertama chat
        if _hybrid_chat_available and HybridChatEngine is not None:
            try:
                HybridChatEngine.start_background_scan()
                print("  ✅ [CoreNode] HybridChat background scan dimulai (API gateway scan)")
            except Exception as _scan_err:
                print(f"  ⚠️  [CoreNode] HybridChat scan gagal: {_scan_err}")



    def _tokenize_query(self, text: str):
        """Tokenisasi query ke CryptoTokenChain. Returns None jika tokenizer tidak aktif."""
        if self.crypto_tokenizer:
            try:
                return self.crypto_tokenizer.tokenize(text)
            except Exception as e:
                print(f"[CoreNode] CryptoTokenizer error: {e}")
        return None

    def _get_crypto_context(self, session_id: str, query_chain) -> str:
        """Ambil unlimited context dari CryptoContextEngine."""
        if self.crypto_context and query_chain and session_id:
            try:
                return self.crypto_context.get_context(session_id, query_chain, max_chars=5000)
            except Exception as e:
                print(f"[CoreNode] CryptoContext get error: {e}")
        return ""

    def _record_crypto_turn(self, session_id: str, query_chain, response: str, domain: str = "general"):
        """Record turn ke CryptoContextEngine setelah response dihasilkan."""
        if self.crypto_context and query_chain and session_id and response:
            try:
                self.crypto_context.record_turn(session_id, query_chain, response, domain=domain)
            except Exception as e:
                print(f"[CoreNode] CryptoContext record error: {e}")

    def _store_response_tiered(self, query_chain, response: str, domain: str = "general"):
        """Simpan response ke TieredMemoryBus (L1+L2+L3) untuk lookup cepat di masa depan."""
        if not self.tiered_memory or not query_chain:
            return
        try:
            # Simpan per token dari response
            if self.crypto_tokenizer:
                resp_chain = self.crypto_tokenizer.tokenize(response)
                for tok in resp_chain.tokens:
                    self.tiered_memory.store(
                        token_id=tok.token_id,
                        content=tok.chunk_text,
                        domain=domain,
                        write_l3=False  # L3 async — jangan block response
                    )
        except Exception as e:
            print(f"[CoreNode] TieredMemory store error: {e}")

    def generate_system_prompt(self, question: str = "", omni_context: str = "", session_context: str = "", domain: str = "general", template_instruction: str = "", **kwargs) -> str:
        """
        Phase 13 — Pure Omni Identity.
        """
        # === IDENTITAS MOKO ===
        identity_text = (
            "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian. "
            "Jawab dengan singkat, natural, dan percaya diri sebagai MOKO.\n"
        )

        # === DATA PENGETAHUAN FAKTUAL DARI OMNI ===
        knowledge_context = ""
        if omni_context:
            knowledge_context = f"\n\n[OMNI KNOWLEDGE]:\n{omni_context}"

        # === LIVE KNOWLEDGE ===
        live_knowledge_context = ""
        if _live_engine and not omni_context:
            try:
                live_ctx = _live_engine.get_fresh_context(query=question, max_chars=1500)
                if live_ctx:
                    live_knowledge_context = f"\n\n[LIVE CONTEXT]:\n{live_ctx}"
            except Exception:
                pass

        # === RIWAYAT PERCAKAPAN ===
        history_context = ""
        if session_context:
            history_context = f"\n\n[SESSION HISTORY]:\n{session_context}"

        return (identity_text + history_context + knowledge_context + live_knowledge_context).strip()

    @staticmethod
    def _get_depth_params(route_meta: dict) -> tuple:
        """
        Phase 13 — Pure Math Control.
        Hanya mengembalikan num_predict (kontrol matematis panjang output).
        Tidak ada instruksi verbal — model memutuskan panjang jawaban sendiri.
        D0 = 80 token | D5 = 350 token | D9 = 700 token
        """
        depth = (route_meta or {}).get("depth", "D5")
        if depth == "D0":
            return "", 80
        elif depth == "D9":
            return "", 700
        else:  # D5 default
            return "", 350

    @staticmethod
    def _strip_inline_latex(text: str) -> str:
        """
        Phase 11 Fix — LaTeX Leakage Filter:
        Hapus notasi LaTeX $...$ yang bocor di luar tag <formula>...</formula>.
        Model kadang menggunakan $\\pi$, $H_2O$, $\\Delta G$ meski dilarang di system prompt.
        Filter ini berjalan sebagai post-processing SETELAH LLM generate output.
        """
        import re
        # Hapus blok <think>...</think> jika ada yang bocor dari LLM (termasuk unclosed think blocks)
        text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
        
        # Pisahkan teks di dalam <formula> agar tidak disentuh
        parts = re.split(r'(<formula>.*?</formula>)', text, flags=re.DOTALL)
        result = []
        for part in parts:
            if part.startswith('<formula>'):
                result.append(part)  # Biarkan blok formula utuh
            else:
                # Strip inline $...$ → ganti dengan plain text
                # Contoh: $\pi$ → pi | $H_2O$ → H2O | $\Delta G$ → delta G
                def replace_latex(m):
                    inner = m.group(1).strip()
                    # Bersihkan backslash commands umum
                    inner = re.sub(r'\\([a-zA-Z]+)', lambda x: x.group(1), inner)
                    inner = re.sub(r'[{}]', '', inner)
                    inner = inner.replace('_', '').replace('^', '')
                    return inner
                cleaned = re.sub(r'\$([^$\n]+?)\$', replace_latex, part)
                result.append(cleaned)
        return ''.join(result)

    @staticmethod
    def _ensure_nonempty_answer(answer: str) -> str:
        """Jangan biarkan pipeline signing mengubah output kosong menjadi terlihat valid."""
        if answer and answer.strip():
            return answer
        return (
            "Maaf, mesin inferensi MOKO belum mengembalikan jawaban. "
            "Kemungkinan server LLM lokal sedang mati, masih memuat model, atau port inferensi bermasalah."
        )


    def quick_reply(
        self,
        question: str,
        session_context: str = "",
        route_meta: dict = None,
        session_messages: list = None,
        on_chunk=None,
        on_token=None,
        disable_timeout: bool = False
    ) -> str:
        """
        Jawaban cepat untuk FAST_PATH (sapaan, pertanyaan ringan, perkenalan).
        Menggunakan Intent-First Router dan Multi-Model Dispatcher.

        FAST-PATH #0: Jika ada API gateway eksternal aktif (OmniRoute/9Router/OpenCode),
        langsung streaming ke sana tanpa melewati AnalystNode yang berat.
        """
        # ── FAST-PATH #0: Hybrid Chat via API Gateway Eksternal ───────────────
        if _hybrid_chat_available and HybridChatEngine is not None:
            try:
                if HybridChatEngine.is_external_available():
                    status = HybridChatEngine.get_status()
                    print(
                        f"  🚀 [CoreNode] HybridChat FAST-PATH aktif → "
                        f"mandor='{status.get('mandor')}' ({status.get('mandor_provider')})"
                    )
                    # Build system prompt ringkas (tidak berat seperti AnalystNode)
                    sys_prompt = (
                        "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian. "
                        "Jawab dengan singkat, natural, dan percaya diri. "
                        "Gunakan bahasa Indonesia kecuali diminta sebaliknya."
                    )
                    # Tambahkan konteks sesi jika ada
                    if session_context:
                        sys_prompt += f"\n\n[RIWAYAT PERCAKAPAN]:\n{session_context[-2000:]}"

                    response = HybridChatEngine.generate(
                        prompt=question,
                        system_prompt=sys_prompt,
                        max_tokens=1024,
                        on_token=on_token,
                        session_messages=session_messages,
                    )
                    if response and response.strip():
                        response = self._strip_inline_latex(response)
                        return response
            except Exception as _hce_err:
                print(f"  ⚠️  [CoreNode] HybridChat fast-path error: {_hce_err}. Fallback ke pipeline lokal.")

        # ── FALLBACK: Pipeline Lokal (AnalystNode + LLM lokal) ───────────────
        # 1. Intent Classification (jika belum ada)
        if route_meta is None:
            manifest = get_intent_router().classify(question)
            route_meta = {
                "domain": manifest.domain,
                "model_key": manifest.model_key,
                "path": manifest.path,
                "complexity": manifest.complexity,
                "depth": "D0" if manifest.complexity == "SIMPLE" else "D5"
            }
        
        # 2. Model Dispatching
        model_key = route_meta.get("model_key", "general")
        dispatcher = get_dispatcher()
        dispatcher.switch_to(model_key)
        # Jembatan spesialisasi domain (Phase 3.3): selama registry masih menunjuk
        # ke satu file GGUF untuk semua domain, parameter generasi khas domain
        # (mis. suhu 0.0 untuk coding/math vs 0.7 untuk percakapan umum) diterapkan
        # pada pipeline generasi utama di bawah agar switch model benar-benar berefek.
        domain_params = dispatcher.get_active_params()

        import time
        t_start = time.perf_counter()

        # ── Phase 11: MOKO Fast-Core Engine (FCE) Bypass ──
        # Untuk query D0 (sederhana/cepat), langsung bypass total arsitektur pipeline yang berat
        # dan panggil raw engine dengan system prompt minimalis untuk speedup ekstrim.
        depth = (route_meta or {}).get("depth", "D5")
        is_personal_query = any(w in question.lower() for w in ["nama", "panggil", "siapa", "saya", "aku", "ingat", "masih ingat"])
        if depth == "D0" and not is_personal_query:
            # Phase 13: D0 bypass — no system prompt, pure model inference
            try:
                ans = engine.generate_text(
                    prompt=f"User: {question}",
                    system_prompt="",
                    coop_params={"num_predict": 350, "enable_thinking": False, "temperature": 0.1}
                )
            except Exception as _d0_err:
                print(f"[CoreNode] D0 generation error: {_d0_err}")
                ans = "..."
            
            ans = self._strip_inline_latex(ans)
            ans = self._ensure_nonempty_answer(ans)
            if on_token:
                on_token(ans)
            return ans

        # ── D0 Personal Query: Muat memori personal dulu, lalu generate ────────
        # Kasus: "masih ingat namaku?", "siapa saya?" — depth D0 tapi butuh memori
        if depth == "D0" and is_personal_query:
            try:
                import time as _t2
                emb = engine.get_embedding(question)
                mem_results = self.disk_mgr.search_memory(emb, top_k=3, domain='personal')
                personal_facts = '\n'.join(
                    f"- {r['text']}" for r in mem_results
                    if r.get('score', 0) >= 0.35
                )
                if personal_facts:
                    sys_personal = (
                        "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian.\n"
                        "Jangan pernah menyebut dirimu MOKO, GPT, atau nama model AI lain.\n"
                        "Jawab singkat dan natural.\n\n"
                        f"[MEMORI PERSISTEN PERSONAL]:\n{personal_facts}"
                    )
                else:
                    sys_personal = (
                        "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian.\n"
                        "Jangan pernah menyebut dirimu MOKO, GPT, atau nama model AI lain.\n"
                        "Jawab singkat dan natural."
                    )
            except Exception as _pe:
                print(f"[CoreNode] Personal memory lookup error: {_pe}")
                sys_personal = "Kamu adalah MOKO — AI sovereign."

            try:
                ans = engine.generate_text(
                    prompt=f"User: {question}",
                    system_prompt=sys_personal,
                    coop_params={"num_predict": 350, "enable_thinking": False, "temperature": 0.3}
                )
            except Exception as _p0_err:
                print(f"[CoreNode] D0 Personal generation error: {_p0_err}")
                ans = "Maaf, saya memerlukan waktu lebih lama dari biasanya."

            ans = self._strip_inline_latex(ans)
            ans = self._ensure_nonempty_answer(ans)
            if on_token:
                on_token(ans)
            return ans

        # (Early-exit check simplified)
        if route_meta and route_meta.get("omni_bypass"):
            output = route_meta["omni_output"]
            if on_token:
                on_token(output)
            return output

        # (CryptoCore bypass removed)
        pass

        _, num_predict = self._get_depth_params(route_meta)
        depth = (route_meta or {}).get("depth", "D5")

        # Phase 13: generate_system_prompt sekarang hanya berisi data faktual Omni + riwayat,
        # tanpa instruksi verbal. Math_omni tidak lagi menyuntikkan template instruction.
        domain_val = (route_meta or {}).get("domain", "general")
        sys_prompt = self.generate_system_prompt(
            question,
            omni_context="",
            session_context=session_context,
            domain=domain_val
        )
        prompt = f"User: {question}"

        # ── Phase 12: Thread-Based Hard Timeout Guard ────────────────────────
        # Root cause fix: requests.iter_lines() blocks selama prefill GPU (model proses
        # context), sehingga stop_check di generate_complete_stream tidak pernah terpicu
        # sampai semua token sudah selesai di-generate (20+ detik).
        #
        # Solusi: jalankan generation di daemon thread, main thread menunggu
        # maksimal time_budget detik. Jika timeout, ambil partial result dari
        # Panggil auto_continue_engine secara sinkron langsung di main thread.
        # Ini memastikan respon di-generate secara penuh dan natural di CPU tanpa interupsi waktu paksa.
        coop_params = {"num_predict": num_predict, "enable_thinking": False}
        domain_temp = domain_params.get("temperature")
        if domain_temp is not None:
            coop_params["temperature"] = domain_temp
        try:
            ans = auto_continue_engine.generate_complete_stream(
                prompt=prompt,
                system_prompt=sys_prompt,
                coop_params=coop_params,
                session_messages=session_messages,
                on_token=on_token,
                on_chunk=on_chunk,
                disable_timeout=True
            )
        except Exception as _gen_err:
            print(f"[CoreNode] Generation error: {_gen_err}")
            ans = "Maaf, terjadi kesalahan saat memproses jawaban."
        ans = self._ensure_nonempty_answer(ans)
        # ── End Thread-Based Hard Timeout Guard (DEACTIVATED) ─────────────────

        # (End-to-End Crypto-Chain Pipeline removed)
        pass

        # Phase 11: LaTeX Leakage Post-Processing Filter
        ans = self._strip_inline_latex(ans)
        return ans

    def deep_reply(self, question: str, omni_context: str = "", session_context: str = "") -> str:
        """
        Jawaban mendalam untuk DEEP_PATH di mode SOLO (fallback).
        """
        sys_prompt = self.generate_system_prompt(question, omni_context, session_context)
        prompt = (
            f"Pertanyaan: {question}\n\n"
            "Jawab dengan LENGKAP dan TUNTAS menggunakan teks biasa.\n"
        )
        ans = engine.generate_text(prompt, sys_prompt)
        ans = self._strip_inline_latex(ans)
        ans = self._ensure_nonempty_answer(ans)
        return ans

    def omni_fused_reply(
        self,
        question: str,
        omni_result: dict,
        session_context: str = "",
        route_meta: dict = None,
        session_messages: list = None,
        on_chunk=None,
        on_token=None
    ) -> tuple:
        """
        MENGGUNAKAN MULTI-AGENT BRIDGE (AGENT 1 & AGENT 2).
        """
        from moko_agents.moko_multi_agent import get_multi_agent
        
        # 1. Dispatch model (Agent 1 Switch)
        if route_meta:
            model_key = route_meta.get("model_key", "general")
            get_dispatcher().switch_to(model_key)

        # 2. Panggil Sistem Multi-Agent (Bridge)
        multi_agent = get_multi_agent(self.disk_mgr)
        ans = multi_agent.handle_query(question, history=session_context)
        
        if on_token:
            on_token(ans)
            
        return ("multi_agent_bridge", ans)

    def amplify_response(self, question: str, analyst_thoughts: str, on_token=None, route_path: str = "DEEP_PATH", route_meta: dict = None) -> str:
        # Initialize route_meta if None
        if route_meta is None:
            route_meta = {}
        
        # Jika hasil analisis berasal dari L0/L0-TCS Bypass (deterministik 100%),
        # kembalikan langsung untuk mempertahankan format eksak dan performa instan.
        if "MOKO OS — Sovereign Exact" in analyst_thoughts or "MOKO OS — Turing-Bombe" in analyst_thoughts:
            if on_token:
                # Kirim instan lewat token stream
                on_token(analyst_thoughts)
            return analyst_thoughts

        # ── FAST-PATH: Jika ada API eksternal, gunakan untuk amplifikasi juga ──
        if _hybrid_chat_available and HybridChatEngine is not None:
            try:
                if HybridChatEngine.is_external_available():
                    combined_prompt = (
                        f"Pertanyaan Pengguna: {question}\n\n"
                        f"Analisis Internal:\n{analyst_thoughts}\n\n"
                        "Sampaikan analisis ini kepada pengguna dengan gaya MOKO yang percaya diri. "
                        "Jawaban harus lengkap dan mendalam."
                    )
                    sys_prompt = self.generate_system_prompt(question)
                    response = HybridChatEngine.generate(
                        prompt=combined_prompt,
                        system_prompt=sys_prompt,
                        max_tokens=1536,
                        on_token=on_token,
                    )
                    if response and response.strip():
                        response = self._strip_inline_latex(response)
                        return self._ensure_nonempty_answer(response)
            except Exception as _amp_err:
                print(f"  ⚠️  [CoreNode] HybridChat amplify error: {_amp_err}. Fallback ke lokal.")

        # Suntikkan Symbol Grounding ke dalam prompt sebelum generasi
        grounding = ""
        if _symbol_verifier and ('kode' in question.lower() or 'fungsi' in question.lower() or 'implementasi' in question.lower()):
            try:
                grounding = _symbol_verifier.build_grounding_context(question)
            except Exception:
                pass

        sys_prompt = self.generate_system_prompt(question)
        if grounding:
            sys_prompt += grounding
        
        # Tambahkan instruction khusus untuk BROWSING queries
        browsing_instruction = ""
        if route_path == "BROWSING_PATH":
            browsing_instruction = (
                "\n\n--- SPECIAL INSTRUCTION FOR BROWSING QUERY ---\n"
                "Pertanyaan ini memerlukan informasi real-time dari web search. "
                "Pastikan jawaban Anda menggunakan informasi terbaru dan detail spesifik dari hasil pencarian. "
                "Gunakan istilah, nama, dan data konkret dari sumber yang ditemukan. "
                "Jangan membuat informasi yang tidak ada di hasil pencarian."
            )

        prompt = (
            f"Pertanyaan Pengguna: {question}\n\n"
            f"Analisis Internal MOKO:\n{analyst_thoughts}\n\n"
            "Tugasmu: Sampaikan analisis ini kepada pengguna dengan gayamu sendiri yang khas. "
            "Jangan sebutkan bahwa ini dari 'analis', anggap ini murni pemikiranmu sendiri. "
            "Jawaban harus lengkap dan mendalam."
            + browsing_instruction
        )
        raw_answer = auto_continue_engine.generate_complete_stream(
            prompt=prompt,
            system_prompt=sys_prompt,
            coop_params={"num_predict": 768, "enable_thinking": False},
            on_token=on_token
        )

        # Anti-Hallucination: Verifikasi simbol dan fakta pada output amplifier
        ans = self._self_correct_if_needed(question, raw_answer)
        return ans

    def _self_correct_if_needed(self, question: str, answer: str) -> str:
        if not answer or not _symbol_verifier:
            return answer
        try:
            warnings = _symbol_verifier.verify_factual_text(answer)
            if warnings:
                print(f"  ⚠️  [FactualAuditor] Kesalahan fakta terdeteksi: {warnings}. Menjalankan Self-Correction Loop...")
                
                correction_instructions = "\n".join(f"- {w}" for w in warnings)
                sys_prompt = self.generate_system_prompt(question)
                
                correction_prompt = (
                    f"Pertanyaan Pengguna: {question}\n\n"
                    f"Draf jawabanmu sebelumnya mengandung kesalahan ilmiah/faktual:\n"
                    f"{answer}\n\n"
                    f"PENTING: Auditor Ilmiah mendeteksi kesalahan berikut:\n"
                    f"{correction_instructions}\n\n"
                    f"Tugasmu: Tulis ulang (rewrite) jawaban tersebut secara keseluruhan untuk mengoreksi kesalahan di atas. "
                    f"Pastikan kamu:\n"
                    f"1. Gunakan rumus geometris yang benar untuk kapasitas mesin/silinder (cc motor) dan BUNGKUS dengan tag formula: <formula>V = (pi/4) * d2 * s * N</formula>.\n"
                    f"2. JANGAN menambahkan volume clearance / volume ruang bakar (V_clearance) ke dalam rumus kapasitas silinder/cc motor.\n"
                    f"3. Hitung secara akurat langkah demi langkah tanpa mengarang angka.\n"
                    f"4. Sampaikan dengan gaya bicara MOKO yang dingin, kompeten, dan lugas, serta patuhi format <formula>...</formula> untuk rumus utama."
                )
                
                corrected = auto_continue_engine.generate_complete_stream(
                    prompt=correction_prompt,
                    system_prompt=sys_prompt,
                    coop_params={"num_predict": 768, "enable_thinking": True},
                )
                
                # Uji kembali apakah hasil koreksi sudah bebas error
                re_warnings = _symbol_verifier.verify_factual_text(corrected)
                if re_warnings:
                    annotated, _ = _symbol_verifier.annotate_response(corrected)
                    return annotated
                return corrected
        except Exception as e:
            print(f"[CoreNode] Self-Correction failed: {e}")
        return answer
