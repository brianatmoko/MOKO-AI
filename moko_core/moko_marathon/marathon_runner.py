"""
MOKO Marathon Runner
====================
Koordinator siklus penalaran maraton. Mengontrol jalannya langkah demi langkah,
memanggil LLM, melakukan kompresi semantik, dan memperbarui pager memori.
"""
import re
import time
from moko_agents.llm_engine import engine
from moko_marathon.context_pager import ContextPager
from moko_marathon.semantic_compressor import semantic_compressor
from moko_neuromath.math_normalizer import math_normalizer

class MarathonRunner:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.pager = ContextPager(session_id)

    def run_marathon(
        self,
        question: str,
        sys_prompt: str,
        max_steps: int = 10,
        on_breath = None,
        retrieved_context: str = ""
    ) -> str:
        """
        Menjalankan loop penalaran berantai maraton.
        
        Args:
            question: Soal / goal utama
            sys_prompt: System prompt MOKO
            max_steps: Jumlah maksimal langkah maraton sebelum dipaksa selesai
            on_breath: Callback streaming status
            retrieved_context: Pengetahuan tambahan dari CAS / memori
            
        Returns:
            Jawaban akhir terkompresi/lengkap
        """
        self.pager.clear()
        step = 1
        
        if on_breath:
            on_breath(f"🏃 Memulai MOKO Marathon Engine (Limit: {max_steps} langkah)...")

        # Config LLM per langkah: Matikan thinking di langkah intermediate
        # agar tidak menghabiskan ratusan token untuk <thought> yang tidak perlu.
        # Reasoning tetap berjalan — hanya lewat teks langsung, bukan format CoT.
        step_coop_params = {
            "num_predict": 1000,     # Cukup untuk 1 langkah analisis fokus
            "enable_thinking": False # Hemat token: skip <thought>...</thought>
        }
        # Hanya aktifkan thinking di langkah terakhir (force synthesis)
        final_coop_params = {
            "num_predict": 2048,
            "enable_thinking": True
        }

        while step <= max_steps:
            start_step = time.time()
            if on_breath:
                on_breath(f"👣 [Marathon] Menjalankan langkah {step}/{max_steps}...")

            # 1. Bangun active context dari pager
            active_context = self.pager.build_active_context(question, retrieved_context)

            # 1b. Bangun header notasi matematika (jika soal olimpiade)
            math_header = math_normalizer.build_enriched_prompt(question, "").strip()
            math_header_block = f"{math_header}\n\n" if math_header else ""

            # 2. Susun prompt instruksi langkah — lebih memaksa setelah langkah 2
            if step >= 3:
                # Langkah lanjut: paksa model untuk menyimpulkan jawaban akhir
                prompt_step = (
                    f"{math_header_block}"
                    f"You are solving a hard mathematical problem. You have already completed {step-1} reasoning steps.\n\n"
                    f"{active_context}\n\n"
                    "IMPORTANT: You have done enough analysis. You MUST now write the final answer.\n"
                    "Begin your response with [FINAL ANSWER] and provide the complete mathematical formula and proof.\n"
                    "VERIFY your formula against n=1, n=2, n=3 before writing it.\n"
                    "Do NOT use [CONTINUE]. State the answer as a closed-form formula now."
                )
            else:
                prompt_step = (
                    f"{math_header_block}"
                    f"You are solving a hard mathematical problem step by step.\n\n"
                    f"{active_context}\n\n"
                    "MANDATORY OUTPUT RULES:\n"
                    "- Start with [CONTINUE] if you need more reasoning steps. Write your partial analysis/deduction clearly.\n"
                    "- Start with [FINAL ANSWER] if you have derived the complete, rigorous answer. Include the answer formula.\n"
                    "- Be precise. Do NOT repeat what was already analyzed. Build upon previous steps.\n"
                    "- Write in clear mathematical language. Avoid vagueness.\n"
                    "- When writing formulas, treat single-letter pairs as MULTIPLICATION (e.g., 'ai' = a*i), not exponents."
                )

            # 3. Panggil LLM
            # Gunakan direct call agar tidak kembung oleh multi-pass di setiap langkah maraton
            raw_response = engine.generate_text(
                prompt=prompt_step,
                system_prompt=sys_prompt,
                coop_params=step_coop_params
            )

            response_clean = raw_response.strip()
            
            # Cek status kelanjutan
            is_final = "[FINAL ANSWER]" in response_clean or "FINAL ANSWER:" in response_clean.upper()
            is_continue = "[CONTINUE]" in response_clean or "CONTINUE:" in response_clean.upper()

            # Bersihkan prefix tag untuk ekstraksi konten
            content_clean = re.sub(r'^\[(?:CONTINUE|FINAL ANSWER)\]', '', response_clean, flags=re.IGNORECASE).strip()
            content_clean = re.sub(r'^(?:CONTINUE|FINAL ANSWER):', '', content_clean, flags=re.IGNORECASE).strip()

            step_duration = time.time() - start_step
            if on_breath:
                on_breath(f"✔️ Langkah {step} selesai dalam {step_duration:.2f} detik.")

            # Jika LLM memutuskan selesai atau sudah langkah terakhir
            # HANYA keluar jika ada [FINAL ANSWER] eksplisit ATAU batas langkah tercapai.
            # Jangan keluar hanya karena tidak ada [CONTINUE] — model mungkin sedang membangun argumen.
            if is_final or step == max_steps:
                if on_breath:
                    on_breath(f"🏁 Marathon mencapai finis di langkah {step}. Merumuskan jawaban akhir...")
                
                # Jika ini langkah paksaan (batas maksimum tercapai)
                if step == max_steps and not is_final:
                    prompt_force = (
                        f"{active_context}\n\n"
                        f"Batas langkah maraton tercapai ({max_steps}). "
                        "Rumuskan JAWABAN AKHIR komprehensif Anda dari analisis di atas sekarang secara mutlak!"
                    )
                    final_res = engine.generate_text(
                        prompt=prompt_force,
                        system_prompt=sys_prompt,
                        coop_params=final_coop_params
                    )
                    return final_res.strip()
                
                return content_clean

            # Jika berlanjut
            # 4. Lakukan kompresi semantik untuk CoT langkah ini
            if on_breath:
                on_breath("📝 Mengompresi hasil langkah secara semantik...")
            
            compressed = semantic_compressor.compress_thinking(content_clean, question)
            
            # 5. Simpan ke database pager
            self.pager.append_step(step, content_clean, compressed)
            step += 1

        return "Error: Siklus maraton berakhir tanpa menghasilkan jawaban."
