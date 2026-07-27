"""
MOKO Omni-Modeling Engine (MOME)
================================
Unifikasi pemodelan canggih: menggabungkan GPT (Causal), GLM (Fim/Blank-filling),
MoE (Mixture of Experts/Worker Pool), dan arsitektur Mandor-Pekerja (API Guru -> Lokal Murid)
menjadi satu engine pemodelan kognitif yang sangat kompleks berbasis INT4 Byte-Q.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

# Setup path imports
PROJECT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from moko_agents.dual_system.worker_pool import WorkerPool
from moko_agents.dual_system.interaction_logger import InteractionLogger
from moko_agents.dual_system.orchestrator import _estimate_task_complexity, should_try_local_first, _load_confidence_db, _save_confidence_db
from moko_agents.dual_system.gemini_adapter import GeminiAdapter, GeminiRequest, get_gemini_adapter
from moko_agents.coding.coding_orchestrator import CodingOrchestrator
from moko_inference.moko_engine import get_moko_engine

logger = logging.getLogger("moko_mome_engine")

# ── Mode Constants ────────────────────────────────────────────────────────────
MODE_GPT    = "GPT"     # Causal Autoregressive (lokal)
MODE_GLM    = "GLM"     # Fill-In-The-Middle (FIM / Blank-Filling)
MODE_MOE    = "MOE"     # Sparse expert routing via CodingOrchestrator
MODE_HYBRID = "HYBRID"  # Mandor-Pekerja Gemini Guru → Lokal Murid distilasi


class MOMEEngine:
    """
    MOKO Omni-Modeling Engine (MOME)
    ================================
    Penggabungan sistem pemodelan terpadu yang sangat kompleks untuk asisten coding profesional.
    Menghilangkan batasan LLM tunggal melalui unifikasi format komparatif.
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or str(PROJECT_DIR)
        
        # Core components
        self.worker_pool        = WorkerPool()
        self.interaction_logger = InteractionLogger()
        self._local_engine      = None
        
        # Gemini Adapter — Guru/Mandor dengan backoff otomatis
        self.gemini_adapter: Optional[GeminiAdapter] = get_gemini_adapter()
        
        # CodingOrchestrator — MoE 5-thread spesialis
        self.coding_orchestrator = CodingOrchestrator()
        
        # Unified Memory Bridge (Riset 25)
        try:
            from moko_memory.kv_cache_manager import get_kv_cache
            self.kv_cache = get_kv_cache()
        except Exception as e:
            self.kv_cache = None
            logger.debug(f"KV Cache Manager initialization skipped: {e}")
            
        # Speculative Decoder (Riset 26)
        # ── DINONAKTIFKAN: Menyebabkan CPU hang 1280%+ dan kehilangan whitespace/newline
        # ── Untuk mengaktifkan kembali, hapus komentar di bawah ini setelah model draft kecil tersedia
        self.speculative_decoder = None
        # try:
        #     from moko_inference.speculative_decoder import create_moko_speculative_decoder
        #     self.speculative_decoder = create_moko_speculative_decoder(self.local_engine, k=4)
        # except Exception as e:
        #     self.speculative_decoder = None
        #     logger.debug(f"Speculative Decoder initialization skipped: {e}")
        
        logger.info(
            f"[MOME] MOKO Omni-Modeling Engine berhasil diinisialisasi. "
            f"Gemini={'AKTIF' if self.gemini_adapter else 'OFFLINE'} | "
            f"LocalEngine={'AKTIF' if self.local_engine else 'CPU-FALLBACK'} | "
            f"KVCache={'AKTIF' if self.kv_cache else 'OFFLINE'} | "
            f"Speculative={'AKTIF' if self.speculative_decoder else 'OFFLINE'}"
        )

    @property
    def local_engine(self):
        if self._local_engine is None:
            self._local_engine = get_moko_engine()
        return self._local_engine

    @local_engine.setter
    def local_engine(self, value):
        self._local_engine = value


    # Keyword coding yang benar-benar spesifik — hanya dipicu untuk pertanyaan pemrograman nyata
    # JANGAN masukkan kata Indonesia umum: 'fungsi', 'sistem', 'kode', 'error' → terlalu ambigu
    _CODING_KEYWORDS = frozenset([
        # Sinyal pemrograman yang tidak ambigu
        "traceback", "exception", "def ", "import ", "class ",
        "buatkan kode", "tulis kode", "generate code", "write code",
        "fix bug", "debug ", "syntax error", "compile error",
        "```python", "```cpp", "```javascript", "```java",
        "[cursor]", "<|fim",
        # Permintaan coding eksplisit
        "buatkan program", "buatkan script", "buat fungsi",
        "implementasi kode", "tulis program", "refactor",
    ])
    
    def detect_mode(self, prompt: str, messages: List[Dict]) -> str:
        """
        Mendeteksi mode pemodelan secara cerdas.

        Prioritas:
          1. GLM  — jika ada token FIM / [CURSOR]
          2. MOE  — jika query mengandung kata kunci coding spesifik
          3. HYBRID — jika tugas kompleks DAN Gemini tersedia
          4. GPT  — fallback default (lokal causal)
        """
        # 1. Cek token GLM/FIM
        full_content = prompt + " ".join(m.get("content", "") for m in messages)
        if "<|fim_prefix|>" in full_content or "[CURSOR]" in full_content:
            return MODE_GLM

        # 2. Cek apakah tugas adalah coding task spesifik → MoE (5-thread specialist)
        lower = prompt.lower()
        if any(kw in lower for kw in self._CODING_KEYWORDS):
            # MoE routing melalui CodingOrchestrator
            return MODE_MOE

        # 3. Hitung kompleksitas untuk hybrid routing
        _, complexity = _estimate_task_complexity(prompt)

        # Jika ada Gemini adapter dan tugas cukup kompleks → HYBRID
        if self.gemini_adapter and complexity >= 0.55:
            return MODE_HYBRID

        return MODE_GPT

    def process_glm(self, prompt: str, messages: List[Dict]) -> Tuple[str, str]:
        """
        Memproses mode GLM (Fill-In-The-Middle / Blank-Filling).
        Memisahkan prefix dan suffix, memformat ke token standard model Qwen2.5-Coder:
        <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>
        """
        # Konversi format [CURSOR] menjadi FIM tokens jika ada
        prefix = ""
        suffix = ""
        
        content = prompt
        if messages:
            content = messages[-1].get("content", "")
            
        if "[CURSOR]" in content:
            parts = content.split("[CURSOR]")
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
        elif "<|fim_prefix|>" in content:
            # Format sudah FIM, gunakan langsung
            return content, "glm_pass"
        else:
            # Fallback split tengah
            half = len(content) // 2
            prefix = content[:half]
            suffix = content[half:]
            
        # Bentuk prompt FIM standar Qwen2.5-Coder
        formatted_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        return formatted_prompt, "glm_formatted"

    def execute(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, str]:
        """
        Mengeksekusi request chat completions dengan menyatukan seluruh mode modeling.
        """
        prompt = messages[-1].get("content", "") if messages else ""
        
        # 1. Deteksi Mode Pemodelan
        mode = self.detect_mode(prompt, messages)
        logger.info(f"[MOME] Mode terdeteksi: {mode}")
        
        # 2. Eksekusi Berdasarkan Mode

        # ── MODE GLM (Blank-Filling / FIM) ──────────────────────────────────
        if mode == MODE_GLM:
            logger.info("[MOME] Mengeksekusi GLM (FIM) Blank-Filling...")
            fim_prompt, status = self.process_glm(prompt, messages)
            glm_messages = [{"role": "user", "content": fim_prompt}]

            if self.local_engine:
                if stream and on_token:
                    content, fr = self.local_engine.chat_stream(
                        glm_messages, max_tokens=max_tokens,
                        temperature=temperature, on_token=on_token
                    )
                else:
                    content, fr = self.local_engine.chat(
                        glm_messages, max_tokens=max_tokens, temperature=temperature
                    )
                return content, f"GLM_OK_{fr}"
            return "Error: Engine model lokal tidak aktif.", "GLM_ERROR"

        # ── MODE MOE (5-Thread Coding Orchestrator) ──────────────────────────
        elif mode == MODE_MOE:
            logger.info("[MOME] Mengeksekusi MOE → CodingOrchestrator...")
            # Ambil kode (jika ada) dari pesan user sebelumnya
            code_ctx = ""
            if len(messages) >= 2:
                # Coba ambil kode dari konteks sebelumnya
                prev = messages[-2].get("content", "")
                if "```" in prev:
                    import re as _re
                    m = _re.search(r"```(?:\w+)?\n(.*?)```", prev, _re.DOTALL)
                    if m:
                        code_ctx = m.group(1)

            moe_result = self.coding_orchestrator.execute(
                query=prompt,
                code=code_ctx,
                language="python"
            )
            # Format output
            result_data = moe_result.get("result", {})
            agent_used  = moe_result.get("routed_agent", "UNKNOWN")
            content = (
                result_data.get("repaired_code")
                or result_data.get("generated_code")
                or result_data.get("explanation")
                or result_data.get("summary")
                or result_data.get("raw_response")
            )
            # Jika content kosong atau hanya dict representation, fallback ke GPT lokal
            if not content or content.strip() == "" or (isinstance(content, str) and content.startswith("{")):
                logger.warning("[MOME] MOE mengembalikan hasil kosong. Fallback ke GPT lokal.")
                return self._execute_gpt_local(messages, max_tokens, temperature, stream, on_token)
            if stream and on_token and content:
                for word in content.split(" "):
                    on_token(word + " ")
            return content, f"MOE_{agent_used}_OK"

        # ── MODE HYBRID (Gemini Mandor → Distilasi ke Murid Lokal) ───────────
        elif mode == MODE_HYBRID:
            logger.info("[MOME] Mengeksekusi HYBRID Gemini Mandor-Pekerja Loop...")

            system_prompt = next(
                (m.get("content", "") for m in messages if m.get("role") == "system"),
                "Kamu adalah MOKO Coder, asisten pemrogram profesional. "
                "Fokus pada kualitas kode, kejelasan, dan efisiensi."
            )
            # CoT system prompt — Gemini menuliskan reasoning dalam thinking mode
            cot_system = (
                f"{system_prompt}\n\n"
                "Kamu adalah Guru/Mandor yang mengajar MOKO AI muda. "
                "Tuliskan kode yang benar dan lengkap. Jelaskan reasoning-mu di dalam "
                "<thought>...</thought> sebelum jawaban akhir."
            )

            gemini_req = GeminiRequest(
                prompt=prompt,
                system_prompt=cot_system,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_budget=1024,  # Enable Gemini thinking mode (1024 tokens)
            )

            try:
                gemini_resp = self.gemini_adapter.generate(gemini_req)

                if not gemini_resp.success:
                    logger.warning(
                        f"[MOME] Gemini gagal ({gemini_resp.error}). Fallback ke lokal."
                    )
                    return self._execute_gpt_local(
                        messages, max_tokens, temperature, stream, on_token
                    )

                response_text = gemini_resp.text

                # Ekstrak <thought> untuk distilasi
                thought = ""
                code_output = response_text
                t_match = re.search(r'<thought>(.*?)</thought>', response_text, re.DOTALL)
                if t_match:
                    thought = t_match.group(1).strip()
                    code_output = re.sub(
                        r'<thought>.*?</thought>', '', response_text, flags=re.DOTALL
                    ).strip()

                # Log ke distill pipeline (Guru → Murid SFT)
                category, complexity = _estimate_task_complexity(prompt)
                self.interaction_logger.log_sample(
                    prompt=prompt,
                    thought=thought or "Gemini Guru reasoning.",
                    code=code_output,
                    passed_guard=True,
                    task_complexity=complexity,
                    task_category=category,
                    source="guru_api"
                )

                logger.info(
                    f"[MOME] Gemini sukses: model={gemini_resp.model_used}, "
                    f"latency={gemini_resp.total_latency_s}s, "
                    f"retries={gemini_resp.retry_count}, "
                    f"thinking_tokens~={gemini_resp.thinking_tokens}"
                )

                if stream and on_token:
                    for word in response_text.split(" "):
                        on_token(word + " ")
                        time.sleep(0.003)

                return response_text, f"HYBRID_GEMINI_OK_retry{gemini_resp.retry_count}"

            except Exception as e:
                logger.error(f"[MOME] Exception pada HYBRID: {e}. Fallback ke lokal.")
                return self._execute_gpt_local(messages, max_tokens, temperature, stream, on_token)

        # ── MODE GPT (Causal Autoregressive Lokal) ───────────────────────────
        else:
            return self._execute_gpt_local(messages, max_tokens, temperature, stream, on_token)

    def _execute_gpt_local(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        stream: bool,
        on_token: Optional[Callable[[str], None]]
    ) -> Tuple[str, str]:
        """Eksekusi standard causal generation lokal."""
        logger.info("[MOME] Mengeksekusi GPT Causal lokal...")
        
        prompt = messages[-1].get("content", "") if messages else ""
        
        # Gunakan Speculative Decoding jika di-stream (MTP Akselerasi)
        # ── DINONAKTIFKAN: CPU hang + text corruption, selalu pakai chat_stream langsung
        # if stream and self.speculative_decoder and on_token:
        #     text, stats = self.speculative_decoder.generate(prompt, max_new_tokens=max_tokens)
        #     for word in text.split(" "):
        #         on_token(word + " ")
        #     return text, f"GPT_LOCAL_SPECULATIVE_{stats['estimated_speedup']}x"
            
        if self.local_engine:
            if stream and on_token:
                content, fr = self.local_engine.chat_stream(
                    messages, max_tokens=max_tokens, temperature=temperature, on_token=on_token
                )
            else:
                content, fr = self.local_engine.chat(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
            return content, f"GPT_LOCAL_{fr}"
        else:
            return "Error: Engine model lokal tidak aktif.", "GPT_LOCAL_ERROR"


# Global singleton instance
mome_engine: Optional[MOMEEngine] = None

def get_mome_engine() -> MOMEEngine:
    global mome_engine
    if mome_engine is None:
        mome_engine = MOMEEngine()
    return mome_engine
