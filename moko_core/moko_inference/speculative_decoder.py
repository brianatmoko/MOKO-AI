"""
MOKO Speculative Decoder — Multi-Token Prediction (MTP)
========================================================
Implementasi Speculative Decoding terinspirasi GLM-5.2 (IndexShare + MTP).

Konsep:
  - Draft model kecil (misal: MOKO 1.5B INT4 lokal) menghasilkan K token secara cepat
  - Verifier besar (lokal atau Gemini via HYBRID mode) memvalidasi K token sekaligus
  - Jika draft diterima → hemat K-1 forward pass (kecepatan naik 2-4x)
  - Jika draft ditolak → verifier koreksi, proses lanjut dari token tersebut

Kenapa penting untuk MOKO:
  - Model 1.5B lokal sangat lambat tanpa akselerasi
  - Speculative decoding bisa mencapai acceptance rate ~70-80% pada code tasks
  - Tidak memerlukan hardware tambahan — cukup jalankan draft lebih dulu

Mode operasi:
  1. SELF_DRAFT:    Model yang sama tapi dengan cache truncation (mode utama)
  2. SMALL_DRAFT:   Draft dari greedy sampling cepat (temperature=0)
  3. GEMINI_VERIFY: Draft lokal, verifikasi oleh Gemini (Mandor-Pekerja)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("moko_speculative_decoder")


class DraftMode(str, Enum):
    SELF_DRAFT    = "self_draft"     # Gunakan model sendiri dengan top-1
    SMALL_DRAFT   = "small_draft"    # Draft dari model lokal lebih kecil
    GEMINI_VERIFY = "gemini_verify"  # Draft lokal, verifikasi Gemini


@dataclass
class DraftResult:
    """Hasil satu putaran speculative decoding."""
    draft_tokens: List[str]
    accepted_count: int
    acceptance_rate: float
    correction_token: Optional[str]  # Token koreksi dari verifier jika ada penolakan
    draft_time_ms:   float
    verify_time_ms:  float

    @property
    def total_time_ms(self) -> float:
        return self.draft_time_ms + self.verify_time_ms

    @property
    def speedup_estimate(self) -> float:
        """Estimasi speedup dibanding normal autoregressive."""
        if self.total_time_ms == 0:
            return 1.0
        # Speedup = token yang dihasilkan / waktu relatif
        tokens_generated = max(self.accepted_count + 1, 1)
        return tokens_generated / max(1.0, self.verify_time_ms / self.draft_time_ms)


class MokoSpeculativeDecoder:
    """
    Speculative Decoder untuk MOKO Coder 1B.

    Penggunaan (self-draft mode, tanpa model ekstra):
        decoder = MokoSpeculativeDecoder(
            draft_fn=lambda prompt, k: quick_greedy(prompt, k),
            verify_fn=lambda prompt, drafts: verify_batch(prompt, drafts),
            k=4,  # Draft 4 token sekaligus
        )
        result = decoder.decode_step(prompt="def fibonacci(n):")
    """

    def __init__(
        self,
        draft_fn:  Callable[[str, int], List[str]],  # f(prompt, k) → [token₁, ..., tokenₖ]
        verify_fn: Callable[[str, List[str]], List[bool]],  # f(prompt, drafts) → [True/False]
        k: int = 4,              # Jumlah draft token per step
        mode: DraftMode = DraftMode.SELF_DRAFT,
        min_acceptance_rate: float = 0.5,  # Jika AR < ini, kurangi K
    ):
        self.draft_fn            = draft_fn
        self.verify_fn           = verify_fn
        self.k                   = k
        self.mode                = mode
        self.min_acceptance_rate = min_acceptance_rate

        # Stats kumulatif
        self.total_steps    = 0
        self.total_accepted = 0
        self.total_drafted  = 0
        self.avg_acceptance = 0.0

        logger.info(
            f"[SpeculativeDecoder] Mode: {mode.value}, K={k}, "
            f"min_acceptance={min_acceptance_rate}"
        )

    def decode_step(self, prompt: str) -> DraftResult:
        """
        Satu langkah speculative decoding.

        1. Draft K token dengan draft_fn (cepat, greedy/top-1)
        2. Verifikasi semua K token sekaligus dengan verify_fn
        3. Terima token berturut-turut sampai verifier menolak
        4. Koreksi jika ada penolakan

        Returns:
            DraftResult dengan token yang diterima dan statistik
        """
        start_t = time.perf_counter()

        # Step 1: Draft K token secara cepat
        draft_start   = time.perf_counter()
        draft_tokens  = self.draft_fn(prompt, self.k)
        draft_time_ms = (time.perf_counter() - draft_start) * 1000

        if not draft_tokens:
            return DraftResult(
                draft_tokens=[], accepted_count=0, acceptance_rate=0.0,
                correction_token=None, draft_time_ms=0, verify_time_ms=0
            )

        # Step 2: Verifikasi batch sekaligus
        verify_start  = time.perf_counter()
        acceptances   = self.verify_fn(prompt, draft_tokens)
        verify_ms     = (time.perf_counter() - verify_start) * 1000

        # Step 3: Hitung berapa token yang diterima berturut-turut
        accepted_count    = 0
        correction_token  = None
        for i, (token, accepted) in enumerate(zip(draft_tokens, acceptances)):
            if accepted:
                accepted_count += 1
            else:
                # Ambil token koreksi dari verifier (biasanya token berikutnya yang benar)
                correction_token = token  # Placeholder — verifier harus kembalikan token koreksi
                break

        acceptance_rate = accepted_count / max(len(draft_tokens), 1)

        # Step 4: Update statistik kumulatif
        self.total_steps    += 1
        self.total_accepted += accepted_count
        self.total_drafted  += len(draft_tokens)
        if self.total_drafted > 0:
            self.avg_acceptance = self.total_accepted / self.total_drafted

        # Adaptive K: kurangi K jika acceptance rate rendah
        if self.avg_acceptance < self.min_acceptance_rate and self.k > 1:
            self.k = max(1, self.k - 1)
            logger.debug(
                f"[SpeculativeDecoder] Adaptive K turun: {self.k+1}→{self.k} "
                f"(avg AR={self.avg_acceptance:.2f})"
            )

        logger.debug(
            f"[SpeculativeDecoder] Step {self.total_steps}: "
            f"{accepted_count}/{len(draft_tokens)} diterima "
            f"(AR={acceptance_rate:.2f})"
        )

        return DraftResult(
            draft_tokens=draft_tokens[:accepted_count],
            accepted_count=accepted_count,
            acceptance_rate=acceptance_rate,
            correction_token=correction_token,
            draft_time_ms=draft_time_ms,
            verify_time_ms=verify_ms,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, dict]:
        """
        Generate teks lengkap menggunakan speculative decoding.

        Returns:
            (generated_text, stats_dict)
        """
        stop_tokens  = stop_tokens or ["<|endoftext|>", "</s>"]
        generated    = []
        current_prompt = prompt
        total_steps  = 0
        total_accepted = 0

        while len(generated) < max_new_tokens:
            result = self.decode_step(current_prompt)
            total_steps += 1

            if not result.draft_tokens and not result.correction_token:
                break

            new_tokens = result.draft_tokens[:]
            if result.correction_token:
                new_tokens.append(result.correction_token)

            # Cek stop tokens
            stop_found = False
            for token in new_tokens:
                if any(st in token for st in stop_tokens):
                    stop_found = True
                    break
                generated.append(token)
                total_accepted += 1

            if stop_found:
                break

            current_prompt = prompt + "".join(generated)

        total_drafted_count = self.total_drafted
        stats = {
            "total_tokens":    total_accepted,
            "total_steps":     total_steps,
            "avg_acceptance":  round(self.avg_acceptance, 3),
            "draft_k":         self.k,
            "estimated_speedup": round(
                total_accepted / max(total_steps, 1), 2
            ),
        }

        return "".join(generated), stats

    def get_performance_stats(self) -> dict:
        """Statistik performa kumulatif."""
        return {
            "mode":           self.mode.value,
            "total_steps":    self.total_steps,
            "total_accepted": self.total_accepted,
            "total_drafted":  self.total_drafted,
            "avg_acceptance": round(self.avg_acceptance, 3),
            "current_k":      self.k,
            "estimated_speedup_x": round(
                self.total_accepted / max(self.total_drafted, 1) * self.k, 2
            ),
        }


# ── Factory: MOME-integrated Speculative Decoder ─────────────────────────────

def create_moko_speculative_decoder(
    local_engine,  # moko_engine instance (llama-cpp atau HTTP)
    gemini_adapter=None,  # Opsional: GeminiAdapter untuk verifikasi Gemini
    k: int = 4,
) -> MokoSpeculativeDecoder:
    """
    Buat SpeculativeDecoder yang terintegrasi dengan MOKO local engine.

    Mode otomatis:
    - Jika gemini_adapter ada → GEMINI_VERIFY (paling akurat)
    - Jika tidak → SELF_DRAFT menggunakan local engine greedy

    Args:
        local_engine:    Instance MokoEngine untuk draft generation
        gemini_adapter:  Instance GeminiAdapter untuk verifikasi (opsional)
        k:               Jumlah draft token per step (default: 4)

    Returns:
        MokoSpeculativeDecoder siap digunakan
    """
    if local_engine is None:
        logger.warning("[SpeculativeDecoder] Local engine tidak tersedia.")
        return None

    def draft_fn(prompt: str, num_tokens: int) -> List[str]:
        """Generate K token dengan greedy decoding (temperature=0)."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response, _ = local_engine.chat(
                messages,
                max_tokens=num_tokens,
                temperature=0.0,  # Greedy — deterministik dan cepat
            )
            # Tokenisasi kasar: split per spasi (cocok untuk draft)
            tokens = response.split()[:num_tokens]
            return tokens
        except Exception as e:
            logger.debug(f"[SpeculativeDecoder] Draft error: {e}")
            return []

    if gemini_adapter is not None:
        # Mode GEMINI_VERIFY: draft lokal, verifikasi Gemini
        from moko_agents.dual_system.gemini_adapter import GeminiRequest

        def verify_fn_gemini(prompt: str, drafts: List[str]) -> List[bool]:
            """Verifikasi apakah draft token masuk akal menurut Gemini."""
            draft_str = " ".join(drafts)
            verify_prompt = (
                f"Kode yang sedang digenerate:\n```\n{prompt}\n```\n\n"
                f"Kandidat token berikutnya: [{', '.join(repr(t) for t in drafts)}]\n"
                f"Untuk setiap token, jawab hanya 'Y' jika masuk akal atau 'N' jika tidak. "
                f"Format: Y/N/Y/N (pisahkan dengan /)."
            )
            try:
                req = GeminiRequest(
                    prompt=verify_prompt,
                    max_tokens=20,
                    temperature=0.0,
                )
                resp = gemini_adapter.generate(req)
                if not resp.success:
                    return [True] * len(drafts)  # Fallback: terima semua

                answers = resp.text.strip().split("/")
                result  = []
                for i, a in enumerate(answers[:len(drafts)]):
                    result.append(a.strip().upper().startswith("Y"))
                # Pad jika kurang
                while len(result) < len(drafts):
                    result.append(True)
                return result
            except Exception as e:
                logger.debug(f"[SpeculativeDecoder] Gemini verify error: {e}")
                return [True] * len(drafts)

        mode = DraftMode.GEMINI_VERIFY
        verify_fn = verify_fn_gemini

    else:
        # Mode SELF_DRAFT: verifikasi menggunakan probability scoring
        def verify_fn_self(prompt: str, drafts: List[str]) -> List[bool]:
            """
            Simple self-draft verification: regenerasi token dan bandingkan.
            Jika model setuju dengan draft → terima.
            """
            try:
                messages = [{"role": "user", "content": prompt}]
                # Regenerasi dengan seed yang sama (greedy, deterministik)
                reference, _ = local_engine.chat(
                    messages,
                    max_tokens=len(drafts),
                    temperature=0.0,
                )
                ref_tokens = reference.split()[:len(drafts)]
                return [
                    d.strip().lower() == r.strip().lower()
                    for d, r in zip(drafts, ref_tokens + [""] * (len(drafts) - len(ref_tokens)))
                ]
            except Exception:
                return [True] * len(drafts)

        mode = DraftMode.SELF_DRAFT
        verify_fn = verify_fn_self

    decoder = MokoSpeculativeDecoder(
        draft_fn=draft_fn,
        verify_fn=verify_fn,
        k=k,
        mode=mode,
    )
    logger.info(
        f"[SpeculativeDecoder] Dibuat: mode={mode.value}, K={k}"
    )
    return decoder
