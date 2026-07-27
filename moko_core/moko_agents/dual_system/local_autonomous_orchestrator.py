"""
LocalAutonomousOrchestrator — 4-Phase Autonomous Loop
======================================================
Digunakan ketika API eksternal SEMUA OFF dan LLM Lokal diaktifkan.
Model lokal (moko-coder-1.5b) bertindak sebagai:

  Phase 1 — MANDOR   : Analisis permintaan, susun rencana terstruktur.
  Phase 2 — VALIDATOR: Validasi rencana (cek kesenjangan, ambiguitas, risiko).
  Phase 3 — EKSEKUTOR: Tulis implementasi kode berdasarkan rencana yang sudah tervalidasi.
  Phase 4 — VALIDATOR: Validasi output akhir (COMMIT / REJECT dengan alasan).

Ketika API ON + Lokal ON:
  → Model lokal TIDAK menjalankan 4-phase ini.
  → Model lokal berstatus "Belajar" — setiap respons API dicatat ke distill_dataset
    oleh interaction_logger.py untuk fine-tuning inkremental.

Referensi arsitektur: riset 23_REVISI_MANDOR_API_MURID_LOKAL.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("moko_local_autonomous")


# ── Helper: baca system mode dari moko_settings.json ──────────────────────────

def get_local_llm_mode() -> str:
    """
    Membaca status peran LLM Lokal dari moko_settings.json.

    Returns:
        "disabled"   — local_llm_enabled = false
        "learning"   — local_llm_enabled = true AND ada external API aktif
        "autonomous" — local_llm_enabled = true AND TIDAK ada external API aktif
    """
    settings_candidates = [
        Path("../moko_config/moko_settings.json"),
        Path("moko_config/moko_settings.json"),
        Path(__file__).parent.parent.parent / "moko_config" / "moko_settings.json",
    ]
    settings = {}
    for p in settings_candidates:
        if p.exists():
            try:
                settings = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    local_enabled = settings.get("local_llm_enabled", False)
    if not local_enabled:
        return "disabled"

    # Check if external API is active by scanning api_keys.json
    api_candidates = [
        Path("../moko_config/api_keys.json"),
        Path("moko_config/api_keys.json"),
        Path(__file__).parent.parent.parent / "moko_config" / "api_keys.json",
    ]
    for p in api_candidates:
        if p.exists():
            try:
                apis = json.loads(p.read_text(encoding="utf-8"))
                external_active = [
                    a for a in apis
                    if a.get("enabled", True)
                    and a.get("provider", "").lower() not in ("local", "")
                    and a.get("api_keys") and len(a.get("api_keys", [])) > 0
                ]
                if external_active:
                    return "learning"
            except Exception:
                pass

    return "autonomous"


# ── 4-Phase Autonomous Orchestrator ───────────────────────────────────────────

PHASE_PROMPTS = {
    "mandor": (
        "Kamu adalah MANDOR AI — perencana dan arsitek solusi.\n"
        "Analisis permintaan berikut secara mendalam. Buat rencana langkah-langkah yang jelas, "
        "terstruktur, dan dapat dieksekusi. Sertakan: tujuan, pendekatan, file yang perlu dibuat/diubah, "
        "dan potensi risiko.\n\n"
        "Permintaan: {prompt}\n\n"
        "Tulis rencana terstruktur di bawah ini:"
    ),
    "validator_plan": (
        "Kamu adalah VALIDATOR AI — penilai kritis dan quality checker.\n"
        "Validasi rencana berikut dengan kritis:\n"
        "1. Apakah rencana lengkap dan tidak ada langkah yang hilang?\n"
        "2. Apakah ada ambiguitas atau risiko yang belum ditangani?\n"
        "3. Apakah urutan langkah logis?\n\n"
        "Rencana:\n{plan}\n\n"
        "Beri verdict: VALID (lanjutkan) atau REVISI (perbaiki: ...). "
        "Jika REVISI, sertakan versi rencana yang sudah diperbaiki."
    ),
    "eksekutor": (
        "Kamu adalah EKSEKUTOR AI — programmer senior yang handal.\n"
        "Implementasikan rencana berikut menjadi kode yang lengkap, bersih, dan siap pakai.\n\n"
        "Permintaan asli: {prompt}\n\n"
        "Rencana tervalidasi:\n{validated_plan}\n\n"
        "Tulis implementasi kode lengkap. Gunakan tag <code>...</code> untuk membungkus kode:"
    ),
    "validator_output": (
        "Kamu adalah VALIDATOR AI — quality assurance final.\n"
        "Review implementasi kode berikut secara kritis:\n"
        "1. Apakah kode sesuai dengan rencana dan permintaan asli?\n"
        "2. Apakah ada bug logika, edge case yang tidak ditangani?\n"
        "3. Apakah kode bersih dan readable?\n\n"
        "Permintaan asli: {prompt}\n\n"
        "Implementasi kode:\n{code}\n\n"
        "Beri verdict akhir: COMMIT ✅ (siap deploy) atau REJECT ❌ (alasan: ...). "
        "Jika REJECT, sertakan perbaikan kode yang diperlukan."
    ),
}


class LocalAutonomousOrchestrator:
    """
    Menjalankan 4-phase autonomous loop menggunakan hanya model lokal.
    Setiap phase distreaming ke callback on_phase_start dan on_token.
    """

    PHASES = [
        ("mandor",           "Phase 1 — MANDOR (Perencanaan)"),
        ("validator_plan",   "Phase 2 — VALIDATOR (Validasi Rencana)"),
        ("eksekutor",        "Phase 3 — EKSEKUTOR (Implementasi)"),
        ("validator_output", "Phase 4 — VALIDATOR (Validasi Output)"),
    ]

    def __init__(
        self,
        engine,                            # MokoInferenceEngine instance
        on_phase_start: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            engine: instance dengan method chat(messages, max_tokens, temperature)
                    dan opsional chat_stream(messages, max_tokens, temperature, on_token)
            on_phase_start: callback(phase_key, phase_label) — dipanggil di awal setiap phase
            on_token: callback(token) — dipanggil untuk setiap token yang dihasilkan
        """
        self.engine = engine
        self.on_phase_start = on_phase_start
        self.on_token = on_token

    def _emit_token(self, token: str):
        if self.on_token:
            self.on_token(token)

    def _call_local(self, system_prompt: str, user_content: str, max_tokens: int = 800) -> str:
        """Panggil model lokal dan kembalikan teks lengkap."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        try:
            # Coba versi streaming dulu
            if hasattr(self.engine, "chat_stream"):
                collected = []

                def _collect(tok):
                    collected.append(tok)
                    self._emit_token(tok)

                self.engine.chat_stream(messages, max_tokens, 0.3, on_token=_collect)
                return "".join(collected)
            else:
                content, _ = self.engine.chat(messages, max_tokens=max_tokens, temperature=0.3)
                self._emit_token(content)
                return content
        except Exception as exc:
            err = f"[ERROR Phase] {exc}"
            self._emit_token(err)
            return err

    def run(self, prompt: str) -> dict:
        """
        Jalankan semua 4 phase secara berurutan.

        Returns dict dengan key:
            plan, validation_plan, code, validation_output, verdict
        """
        results = {}

        # ── Phase 1: MANDOR ─────────────────────────────────────────────────
        phase_key, phase_label = self.PHASES[0]
        if self.on_phase_start:
            self.on_phase_start(phase_key, phase_label)

        plan_prompt = PHASE_PROMPTS["mandor"].format(prompt=prompt)
        plan = self._call_local(
            "Kamu adalah perencana AI yang terstruktur dan sistematis.",
            plan_prompt,
            max_tokens=700,
        )
        results["plan"] = plan

        # ── Phase 2: VALIDATOR (rencana) ─────────────────────────────────────
        phase_key, phase_label = self.PHASES[1]
        if self.on_phase_start:
            self.on_phase_start(phase_key, phase_label)

        val_plan_prompt = PHASE_PROMPTS["validator_plan"].format(plan=plan)
        validation_plan = self._call_local(
            "Kamu adalah validator kritis yang teliti.",
            val_plan_prompt,
            max_tokens=500,
        )
        results["validation_plan"] = validation_plan

        # Ekstrak rencana akhir (dari REVISI jika ada, atau rencana asli)
        validated_plan = plan
        if "REVISI" in validation_plan.upper():
            # Coba ekstrak rencana yang sudah direvisi dari respons validator
            lines = validation_plan.split("\n")
            revisi_start = next(
                (i for i, l in enumerate(lines)
                 if "revisi" in l.lower() or "perbaikan" in l.lower()),
                None
            )
            if revisi_start is not None:
                validated_plan = "\n".join(lines[revisi_start:]).strip()
            logger.info("[Autonomous] Rencana direvisi oleh Validator.")
        else:
            logger.info("[Autonomous] Rencana VALID. Melanjutkan ke Eksekutor.")

        results["validated_plan"] = validated_plan

        # ── Phase 3: EKSEKUTOR ───────────────────────────────────────────────
        phase_key, phase_label = self.PHASES[2]
        if self.on_phase_start:
            self.on_phase_start(phase_key, phase_label)

        exec_prompt = PHASE_PROMPTS["eksekutor"].format(
            prompt=prompt,
            validated_plan=validated_plan,
        )
        code_response = self._call_local(
            "Kamu adalah programmer senior yang menulis kode bersih dan lengkap.",
            exec_prompt,
            max_tokens=1200,
        )
        results["code"] = code_response

        # ── Phase 4: VALIDATOR (output) ──────────────────────────────────────
        phase_key, phase_label = self.PHASES[3]
        if self.on_phase_start:
            self.on_phase_start(phase_key, phase_label)

        val_out_prompt = PHASE_PROMPTS["validator_output"].format(
            prompt=prompt,
            code=code_response[:1500],  # batas konteks lokal
        )
        validation_output = self._call_local(
            "Kamu adalah quality assurance engineer yang kritis dan teliti.",
            val_out_prompt,
            max_tokens=500,
        )
        results["validation_output"] = validation_output

        # Tentukan verdict akhir
        upper_val = validation_output.upper()
        if "COMMIT" in upper_val or "✅" in validation_output:
            results["verdict"] = "COMMIT"
        elif "REJECT" in upper_val or "❌" in validation_output:
            results["verdict"] = "REJECT"
        else:
            results["verdict"] = "REVIEW"

        logger.info(f"[Autonomous] Siklus 4-Phase selesai. Verdict: {results['verdict']}")
        return results
