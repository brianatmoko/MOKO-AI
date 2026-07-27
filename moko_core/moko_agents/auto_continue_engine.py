"""
MOKO Auto-Continue Engine — Pelanjutan Generasi Otomatis
=========================================================
Modul ini menangani deteksi dan pelanjutan otomatis saat LLM
mencapai batas token (finish_reason = "length").

Cara Kerja:
  1. Panggil generate_text_raw() → dapat (content, finish_reason)
  2. Jika finish_reason == "stop" → selesai, kembalikan content
  3. Jika finish_reason == "length":
       - Kirim riwayat + partial response ke LLM
       - Instruksi: "lanjutkan dari titik berhenti"
       - Gabungkan: full = partial + continuation
       - Ulangi sampai "stop" atau MAX_ITERATIONS
  4. Kembalikan respons yang utuh

Keamanan:
  - MAX_ITERATIONS = 5 (mencegah infinite loop)
  - Setiap chunk divalidasi panjangnya (skip jika kosong)
  - on_chunk callback untuk feedback real-time ke UI
"""

from typing import Callable, List, Optional

MAX_ITERATIONS = 5   # Batas maksimum iterasi auto-continue


class AutoContinueEngine:
    """
    Engine untuk melanjutkan generasi LLM yang terpotong secara otomatis.

    Cara pakai:
        engine = AutoContinueEngine()
        full_text = engine.generate_complete(
            prompt       = "jelaskan apa itu neural network",
            system_prompt= sys_prompt,
            coop_params  = {"num_predict": 1024, ...},
            session_messages = session_buffer.get_full_chat_messages(n=8),
            on_chunk     = lambda msg: chat_signal.emit("analyst", msg)
        )
    """

    def __init__(self):
        from moko_agents.llm_engine import engine as _llm_engine
        self._engine = _llm_engine

    def generate_complete(
        self,
        prompt: str,
        system_prompt: str = "",
        coop_params: dict = None,
        session_messages: Optional[List[dict]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        max_continues: int = 5
    ) -> str:
        """
        Generate text dengan auto-continue jika token habis.
        
        Args:
            prompt:           Pertanyaan / perintah user
            system_prompt:    System prompt MOKO
            coop_params:      Parameter CPU governor (num_predict, enable_thinking, dll)
            session_messages: Riwayat percakapan dalam format OpenAI messages array.
                              Digunakan sebagai konteks saat melakukan continue.
                              Jika None, auto-continue tetap berjalan tapi tanpa konteks historis.
            on_chunk:         Callback yang dipanggil setiap kali ada update status.
                              Signature: on_chunk(status_message: str)
            max_continues:    Batas maksimum iterasi auto-continue (default: 5)
                              
        Return: str — Respons lengkap yang sudah digabungkan dari semua iterasi
        """
        if coop_params is None:
            from moko_cpu.governor import CPUGovernor
            coop_params = CPUGovernor.get_cooperative_params()

        # ── Panggilan pertama ─────────────────────────────────────────────────
        content, finish_reason = self._engine.generate_text_raw(
            prompt=prompt,
            system_prompt=system_prompt,
            coop_params=coop_params
        )

        if finish_reason == "stop" or finish_reason == "error":
            # Selesai pada panggilan pertama — tidak perlu continue
            return content

        # ── Generasi terpotong → masuk loop auto-continue ────────────────────
        accumulated = content
        
        # Susun konteks messages untuk continue (system + riwayat + prompt asli)
        base_messages = []
        if system_prompt:
            base_messages.append({"role": "system", "content": system_prompt})
        # Sertakan riwayat percakapan relevan sebagai konteks (saring yang kosong)
        if session_messages:
            clean_session = [m for m in session_messages if m.get("content") and str(m.get("content")).strip()]
            base_messages.extend(clean_session)
        # Tambahkan prompt user saat ini
        base_messages.append({"role": "user", "content": prompt})

        for iteration in range(1, max_continues + 1):
            if on_chunk:
                on_chunk(
                    f"⚙️ [AUTO-CONTINUE] Token habis. "
                    f"Melanjutkan otomatis... (Iterasi {iteration}/{max_continues})"
                )

            # Batasi panjang partial_response agar tidak meledakkan konteks prompt saat continue
            CONTEXT_SAFE_LIMIT = 8000  # Sekitar 2000 token
            if len(accumulated) > CONTEXT_SAFE_LIMIT:
                partial_to_send = (
                    accumulated[:2000] +
                    "\n\n...[MOKO CONTEXT COMPRESSION: TENGAH DIKOMPRES]...\n\n" +
                    accumulated[-6000:]
                )
            else:
                partial_to_send = accumulated

            continuation, new_finish_reason = self._engine.continue_generation(
                messages=base_messages,
                partial_response=partial_to_send,
                coop_params=coop_params
            )

            if not continuation:
                # LLM mengembalikan kosong — hentikan
                if on_chunk:
                    on_chunk("⚠️ [AUTO-CONTINUE] LLM mengembalikan respons kosong. Menghentikan.")
                break

            # Gabungkan: biner langsung karena native continuation menyambung di tingkat karakter/token
            accumulated += continuation

            if new_finish_reason == "stop":
                if on_chunk:
                    on_chunk(
                        f"✅ [AUTO-CONTINUE] Respons lengkap setelah {iteration} iterasi."
                    )
                break
            elif new_finish_reason == "error":
                if on_chunk:
                    on_chunk("❌ [AUTO-CONTINUE] Error saat melanjutkan. Menghentikan.")
                break

        else:
            # Batas iterasi tercapai
            if max_continues > 0 and on_chunk:
                on_chunk(
                    f"⚠️ [AUTO-CONTINUE] Batas {max_continues} iterasi tercapai. "
                    f"Respons mungkin belum sepenuhnya lengkap."
                )

        return accumulated

    def generate_complete_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        coop_params: dict = None,
        session_messages: Optional[List[dict]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        disable_timeout: bool = False,
    ) -> str:
        """
        Generate text secara streaming dengan auto-continue jika token terputus.
        Fase 1: stream secara real-time ke UI via on_token dengan 8-second Hard Timeout Guard.
        Fase 2: jika finish_reason=="length", continue secara non-streaming
                dan append hasilnya (panggil on_token untuk setiap karakter tambahan).
        """
        if coop_params is None:
            from moko_cpu.governor import CPUGovernor
            coop_params = CPUGovernor.get_cooperative_params()

        accumulated = ""

        def local_token_cb(token):
            nonlocal accumulated
            accumulated += token
            if on_token:
                on_token(token)

        # Fase 1: Stream pertama ke UI
        self._engine.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            coop_params=coop_params,
            session_messages=session_messages,
            on_token=local_token_cb,
            stop_check=stop_check
        )

        # Fase 2: Deteksi truncation — Phase 3.7 (Marathon Fix)
        # Sinyal PRIMER: finish_reason dari server ("length" = token habis).
        # Sebelumnya sinyal ini dibuang oleh generate_stream sehingga kode yang
        # terpotong di tengah tidak pernah dilanjutkan (sistem maraton mati).
        finish_reason = getattr(self._engine, "last_stream_finish_reason", None)
        hit_token_limit = (finish_reason == "length")

        # Sinyal SEKUNDER: blok kode ``` belum ditutup (jumlah fence ganjil)
        # → hampir pasti jawaban berkode terpotong di tengah.
        unclosed_code_fence = (accumulated.count("```") % 2 == 1)

        # Sinyal CADANGAN (heuristik lama): panjang mendekati budget & tidak
        # berakhir dengan tanda baca kalimat — dipakai bila finish_reason tak tersedia.
        num_predict = coop_params.get("num_predict", 512)
        char_budget_estimate = num_predict * 3
        last_chars = accumulated.strip()[-5:] if accumulated else ""
        sentence_enders = ['.', '!', '?']
        ends_properly = any(last_chars.rstrip().endswith(c) for c in sentence_enders)
        heuristic_truncated = (
            not ends_properly and
            len(accumulated) >= int(char_budget_estimate * 0.85)  # Phase 11 Fix: Naik ke 85% untuk mencegah false positive
        )

        looks_truncated = (
            len(accumulated) > 50 and
            (hit_token_limit or unclosed_code_fence or heuristic_truncated)
        )

        if looks_truncated:
            if on_chunk:
                on_chunk("⚙️ [AUTO-CONTINUE] Jawaban terpotong. Melanjutkan otomatis...")

            # Phase 3.7 (Marathon Fix): lanjutkan LANGSUNG dari partial via
            # continue_generation (native continuation), BUKAN generate ulang
            # dari nol lalu menebak overlap — cara lama sering gagal merge
            # sehingga kode tetap tersangkut di tengah.
            base_messages = []
            if system_prompt:
                base_messages.append({"role": "system", "content": system_prompt})
            if session_messages:
                clean_session = [m for m in session_messages if m.get("content") and str(m.get("content")).strip()]
                base_messages.extend(clean_session)
            base_messages.append({"role": "user", "content": prompt})

            CONTEXT_SAFE_LIMIT = 8000  # Sekitar 2000 token
            for iteration in range(1, 3 + 1):
                if len(accumulated) > CONTEXT_SAFE_LIMIT:
                    partial_to_send = (
                        accumulated[:2000] +
                        "\n\n...[MOKO CONTEXT COMPRESSION: TENGAH DIKOMPRES]...\n\n" +
                        accumulated[-6000:]
                    )
                else:
                    partial_to_send = accumulated

                continuation, new_finish_reason = self._engine.continue_generation(
                    messages=base_messages,
                    partial_response=partial_to_send,
                    coop_params=coop_params
                )

                if not continuation:
                    if on_chunk:
                        on_chunk("⚠️ [AUTO-CONTINUE] LLM mengembalikan respons kosong. Menghentikan.")
                    break

                accumulated += continuation
                if on_token:
                    on_token(continuation)

                if new_finish_reason == "stop":
                    if on_chunk:
                        on_chunk(f"✅ [AUTO-CONTINUE] Respons lengkap setelah {iteration} iterasi.")
                    break
                elif new_finish_reason == "error":
                    if on_chunk:
                        on_chunk("❌ [AUTO-CONTINUE] Error saat melanjutkan. Menghentikan.")
                    break
            else:
                if on_chunk:
                    on_chunk("⚠️ [AUTO-CONTINUE] Batas 3 iterasi tercapai. Respons mungkin belum sepenuhnya lengkap.")

        return accumulated

    def call_llm_with_continue(
        self,
        system_prompt: str,
        user_prompt: str,
        coop_params: dict = None,
        session_messages: Optional[List[dict]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        max_continues: int = 5
    ) -> str:
        """
        Alias dengan signature yang lebih eksplisit — digunakan oleh analyst_node
        untuk menggantikan _call_llm() di deep_think_loop.
        
        Identik dengan generate_complete() tapi parameter diurutkan berbeda.
        """
        return self.generate_complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            coop_params=coop_params,
            session_messages=session_messages,
            on_chunk=on_chunk,
            max_continues=max_continues
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
auto_continue_engine = AutoContinueEngine()
