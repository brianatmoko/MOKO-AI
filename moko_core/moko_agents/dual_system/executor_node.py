"""
ExecutorNode — Sistem 1 (Gaya Kimi K2/K2.5): Tangan Eksekutor Agentik
=====================================================================
Bertanggung jawab pada tahap bertindak cepat & otonom:
- Menelusuri pengetahuan/berkas relevan lewat Anchor-RAG (`moko_code_knowledge.py`).
- Menulis & mengedit berkas kode di dalam workspace secara presisi.
- Menjalankan perintah terminal (unit test) di subproses lokal.

Semua operasi berkas dibatasi di dalam `workspace_dir` untuk keamanan. Node ini
tidak bergantung pada PyQt/torch sehingga dapat diuji langsung.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from moko_agents.dual_system._bridge import (
    tokenize,
    CodeKnowledgeBase,
    retrieve,
    retrieve_text,
)


@dataclass
class ExecutionResult:
    """Hasil satu siklus eksekusi agentik oleh Sistem 1."""
    success: bool
    log: str = ""
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    written_files: list[str] = field(default_factory=list)
    test_file: str = ""
    retrieved_snippets: list[str] = field(default_factory=list)


class ExecutorNode:
    """System 1 Agentic Executor (Kimi style)."""

    def __init__(
        self,
        workspace_dir: str | os.PathLike,
        knowledge_base=None,
        on_status: Optional[Callable[[str], None]] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.kb = knowledge_base if knowledge_base is not None else CodeKnowledgeBase()
        self.on_status = on_status
        self.python_executable = python_executable or sys.executable

    def _emit(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)

    # ── Anchor-based retrieval (long-context navigation, Kimi style) ───────────
    def retrieve_context(self, prompt_or_tokens, *, limit: int = 3) -> list:
        # Memakai inti native (C++/Rust) bila tersedia — hasil identik dgn Python.
        try:
            if isinstance(prompt_or_tokens, str):
                # Jalur gabungan: tokenisasi + skoring dalam satu panggilan native.
                hits = retrieve_text(self.kb, prompt_or_tokens, limit=limit)
            else:
                hits = retrieve(self.kb, set(prompt_or_tokens), limit=limit)
        except Exception:
            hits = []
        if hits:
            ids = ", ".join(getattr(h, "snippet_id", "?") for h in hits)
            self._emit(f"🔧 Sistem 1 (Executor): Anchor-RAG menautkan → {ids}")
        else:
            self._emit("🔧 Sistem 1 (Executor): Anchor-RAG tidak menemukan berkas relevan.")
        return hits

    # ── Operasi berkas presisi (dibatasi di dalam workspace) ───────────────────
    def _safe_path(self, relpath: str | os.PathLike) -> Path:
        target = (self.workspace_dir / relpath).resolve()
        if self.workspace_dir not in target.parents and target != self.workspace_dir:
            raise ValueError(f"Penulisan di luar workspace ditolak: {target}")
        return target

    def write_file(self, relpath: str, content: str) -> str:
        target = self._safe_path(relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._emit(f"🔧 Sistem 1 (Executor): menulis berkas {relpath}")
        return str(target)

    def read_file(self, relpath: str) -> str:
        target = self._safe_path(relpath)
        return target.read_text(encoding="utf-8")

    # ── Eksekusi terminal ──────────────────────────────────────────────────────
    def run_terminal(self, args: list[str], *, timeout: int = 15) -> tuple[int, str, str]:
        self._emit(f"🔧 Sistem 1 (Executor): menjalankan terminal → {' '.join(args)}")
        try:
            res = subprocess.run(
                args,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as exc:
            return 124, exc.stdout or "", f"TimeoutExpired: {exc}"
        except Exception as exc:  # noqa: BLE001
            return 1, "", f"ExecutorTerminalError: {exc}"

    def run_tests(self, test_file: str, *, timeout: int = 15) -> tuple[int, str, str]:
        return self.run_terminal([self.python_executable, test_file], timeout=timeout)

    # ── Penerapan rencana (write code + write test + run test) ─────────────────
    def apply_plan(self, plan, *, timeout: int = 15) -> ExecutionResult:
        """Terapkan `ExecutionPlan`: tulis modul + unit test, lalu jalankan test."""
        self._emit("🔧 Sistem 1 (Executor): menerima rencana, memulai siklus tindakan agen...")
        retrieved = self.retrieve_context(plan.focus_tokens)
        written: list[str] = []

        module_path = self.write_file(plan.target_module, plan.module_code)
        written.append(module_path)
        test_path = self.write_file(plan.test_module, plan.test_code)
        written.append(test_path)

        rc, stdout, stderr = self.run_tests(plan.test_module, timeout=timeout)
        success = rc == 0 and plan.expected_signal in (stdout or "")
        log = (
            f"return_code={rc}\n"
            f"--- STDOUT ---\n{stdout}\n"
            f"--- STDERR ---\n{stderr}"
        )
        return ExecutionResult(
            success=success,
            log=log,
            stdout=stdout or "",
            stderr=stderr or "",
            return_code=rc,
            written_files=written,
            test_file=test_path,
            retrieved_snippets=[getattr(h, "snippet_id", "?") for h in retrieved],
        )
