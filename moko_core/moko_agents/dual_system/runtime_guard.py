"""
DualRuntimeGuard — Sistem 2 Guard (Gaya DeepSeek): Penjaga & Peninjau Runtime
=============================================================================
Bertanggung jawab pada tahap verifikasi & peninjauan:
- Mengevaluasi log output terminal (stdout/stderr) dari Sistem 1.
- Memparsing traceback/galat untuk mendiagnosis akar masalah.
- Mengintegrasikan `LLMRuntimeGuard` (`moko_llm_runtime_guard.py`) untuk peninjauan
  kepatuhan runtime berbasis model, dengan fallback template yang andal.
- Mengeluarkan verdict: SUKSES (siap commit) atau minta koreksi-diri (repair).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from moko_agents.dual_system._bridge import LLMRuntimeGuard


# Verdict Guard
VERDICT_COMMIT = "SUCCESS_COMMIT_READY"
VERDICT_REPAIR = "REPAIR_INSTRUCTION"


@dataclass
class GuardReport:
    """Laporan hasil peninjauan runtime oleh Sistem 2 Guard."""
    verdict: str
    passed: bool
    summary: str = ""
    error_type: Optional[str] = None
    error_message: str = ""
    failing_line: str = ""
    guard_message: str = ""
    guard_source: str = "template"


class DualRuntimeGuard:
    """System 2 Guard & Reviewer (DeepSeek guard style)."""

    _TRACEBACK_RE = re.compile(
        r"^(?P<etype>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)):\s*(?P<emsg>.*)$",
        re.MULTILINE,
    )
    _FILE_LINE_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')

    def __init__(
        self,
        status_provider: Optional[Callable[[], dict | None]] = None,
        llm_generate: Optional[Callable[[str], str]] = None,
        fallback_generate: Optional[Callable[[str], str]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_status = on_status
        self.status_provider = status_provider or (lambda: {"ready": False})
        self._guard = LLMRuntimeGuard(
            status_provider=self.status_provider,
            llm_generate=llm_generate or (lambda prompt: ""),
            fallback_generate=fallback_generate
            or (lambda prompt: "TEMPLATE GUARD: pemeriksaan dasar runtime lolos."),
        )

    def _emit(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)

    # ── Diagnostik traceback ───────────────────────────────────────────────────
    def parse_traceback(self, log: str) -> dict:
        """Ekstrak tipe galat, pesan, dan lokasi baris pertama dari log."""
        info: dict = {"error_type": None, "error_message": "", "failing_line": ""}
        if not log:
            return info
        match = self._TRACEBACK_RE.search(log)
        if match:
            info["error_type"] = match.group("etype")
            info["error_message"] = match.group("emsg").strip()
        file_matches = self._FILE_LINE_RE.findall(log)
        if file_matches:
            fname, lineno = file_matches[-1]
            info["failing_line"] = f"{fname}:{lineno}"
        return info

    # ── Peninjauan runtime ──────────────────────────────────────────────────────
    def review(self, execution_result) -> GuardReport:
        """Tinjau hasil eksekusi Sistem 1; hasilkan verdict Guard."""
        self._emit("🛡️ Sistem 2 (Guard): mengevaluasi log eksekusi & integritas runtime...")

        # Panggil LLMRuntimeGuard untuk peninjauan kepatuhan (guarded generation).
        try:
            guarded = self._guard.generate(
                "Tinjau log eksekusi unit test berikut dan pastikan tidak ada "
                f"kerentanan atau galat runtime:\n{execution_result.log}"
            )
            guard_message = getattr(guarded, "message", "")
            guard_source = getattr(guarded, "source", "template")
        except Exception as exc:  # noqa: BLE001
            guard_message = f"Guard fallback aktif: {exc}"
            guard_source = "template"

        if execution_result.success:
            self._emit("🛡️ Sistem 2 (Guard): verifikasi LULUS — kode siap di-commit.")
            return GuardReport(
                verdict=VERDICT_COMMIT,
                passed=True,
                summary="Unit test lolos & runtime bersih. Siap commit.",
                guard_message=guard_message,
                guard_source=guard_source,
            )

        diag = self.parse_traceback(
            f"{execution_result.stderr}\n{execution_result.stdout}"
        )
        summary = (
            f"Verifikasi GAGAL (rc={execution_result.return_code}). "
            f"Galat: {diag['error_type'] or 'Assertion/Logic'} "
            f"@ {diag['failing_line'] or 'n/a'}."
        )
        self._emit(f"🛡️ Sistem 2 (Guard): verifikasi GAGAL — memicu koreksi-diri. {summary}")
        return GuardReport(
            verdict=VERDICT_REPAIR,
            passed=False,
            summary=summary,
            error_type=diag["error_type"],
            error_message=diag["error_message"],
            failing_line=diag["failing_line"],
            guard_message=guard_message,
            guard_source=guard_source,
        )
