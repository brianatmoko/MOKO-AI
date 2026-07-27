"""
software_builder_agent.py — SoftwareBuilderAgent (Orchestrator Utama)
======================================================================
Mengorkestrasi seluruh proses Software Builder:
  1. Deteksi trigger (/coding + kata kunci software)
  2. Multi-turn interview via InterviewManager
  3. Prompt Enrichment (InterviewData + RAG → super-prompt)
  4. Plan Generation via LLM + PlanGenerator
  5. Step Execution via StepExecutor

Digunakan dari:
  - main_window_v5.py (on_user_input → handle /coding)
  - chat_panel.py (incoming messages routing)

Pola Qt: Berjalan via QThread workers, komunikasi via pyqtSignal.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from moko_agents.software_builder.models import InterviewData, PlanSession, PlanStep
from moko_agents.software_builder.interview_manager import (
    InterviewManager, InterviewState, get_interview_manager, clear_interview_session
)
from moko_agents.software_builder.plan_generator import parse_plan
from moko_agents.software_builder.prompt_enrichment import enrich_prompt
from moko_agents.software_builder.step_executor import StepExecutor
from moko_agents.software_builder.token_manager import marathon_call_llm


# Kata kunci yang men-trigger Software Builder Mode (bukan coding biasa)
_BUILD_TRIGGER_KEYWORDS = [
    "buat game", "bikin game", "create game", "build game",
    "buat aplikasi", "bikin aplikasi", "buat web", "bikin web",
    "buat software", "bikin software", "buat program", "bikin program",
    "buat tool", "bikin tool", "buat bot", "bikin bot",
    "buat website", "bikin website", "buat api", "bikin api",
    "make game", "make app", "make software",
]


def is_build_trigger(text: str) -> bool:
    """
    Cek apakah teks mengandung kata kunci untuk Software Builder Mode.
    Contoh: "/coding buat game RPG" → True
    """
    text_lower = text.strip().lower()
    return any(kw in text_lower for kw in _BUILD_TRIGGER_KEYWORDS)


class PlanGenerationWorker(QThread):
    """
    Worker untuk generate plan di background (tidak freeze UI).
    
    Orchestrates (UPGRADED v2 — Architect Mode):
      Pass 1 — Architect: LLM membuat rencana dalam prosa + analisis teknis
      Pass 2 — Plan Formatter: LLM mengubah rencana menjadi structured steps
      Fallback ke template jika kedua pass gagal

    Teknik ini meniru "architect mode" dari Cursor dan Aider yang memisahkan
    "planning brain" dari "coding brain" — menghasilkan plan lebih terstruktur.
    """
    log_signal   = pyqtSignal(str, str)          # message, color
    plan_signal  = pyqtSignal(list)              # List[PlanStep]
    done_signal  = pyqtSignal(bool, str)         # success, message

    # Architect pass — LLM diminta berpikir dulu sebelum membuat format
    _ARCHITECT_PROMPT = """You are a senior software architect.
Analyze this project and create a technical plan.

## PROJECT
{requirements}

## YOUR ANALYSIS
Think step by step:
1. What are the core components needed?
2. What's the optimal build order (dependencies first)?
3. What files will each component need?
4. What are the main technical challenges?

Write your analysis and proposed implementation plan in plain text.
Be specific about file names and what each file should contain."""

    # Plan formatter pass — mengubah prosa menjadi format yang bisa di-parse
    _FORMATTER_PROMPT = """Convert the following implementation plan into EXACTLY this machine-parseable format.

## PLAN TO CONVERT
{architect_plan}

## OUTPUT FORMAT (follow precisely)
## Step 1: <Action-oriented title>
<2-3 sentence description of what gets implemented>
Files: <file1.ext>, <file2.ext>

## Step 2: <Title>
<Description>
Files: <files>

... (5 to 12 steps total)

## RULES
- Every step needs a Files: line with real filenames
- Steps must be in logical build order
- No more than 12 steps, no fewer than 5
- Titles must be action-oriented ("Implement X", "Create Y", "Add Z")

Convert now:"""

    def __init__(self, interview_data: InterviewData, workspace_dir: str):
        super().__init__()
        self.interview_data = interview_data
        self.workspace_dir = workspace_dir

    def run(self):
        try:
            self.log_signal.emit("  🏛️ [SBA] Architect Mode: Pass 1 — Analisis proyek...", "#00e6ff")

            # Pass 1: Architect — LLM berpikir dalam prosa
            architect_plan = self._architect_pass()

            if architect_plan:
                self.log_signal.emit(
                    f"  ✅ [SBA] Architect plan: {len(architect_plan)} chars",
                    "rgba(0,230,255,0.7)"
                )
                self.log_signal.emit("  📐 [SBA] Pass 2 — Formatting plan ke structured steps...", "#888")

                # Pass 2: Format ke structured steps
                formatted = self._format_pass(architect_plan)
                steps = parse_plan(formatted, self.interview_data)
            else:
                # Fallback ke single-pass (original behavior)
                self.log_signal.emit("  🔁 [SBA] Architect pass gagal, menggunakan single-pass...", "#ffaa00")
                super_prompt = enrich_prompt(self.interview_data)
                plan_response = self._call_llm(super_prompt, max_tokens=1500)
                steps = parse_plan(plan_response, self.interview_data)

            self.log_signal.emit(f"  ✅ Plan siap: {len(steps)} langkah", "#00ff88")
            self.plan_signal.emit(steps)
            self.done_signal.emit(True, f"Plan siap: {len(steps)} langkah")

        except Exception as e:
            self.log_signal.emit(f"  ❌ Error generate plan: {e}", "#ff4444")
            # Fallback ke template
            try:
                steps = parse_plan("", self.interview_data)
                self.plan_signal.emit(steps)
                self.done_signal.emit(True, f"Plan template: {len(steps)} langkah")
            except Exception as e2:
                self.done_signal.emit(False, f"Gagal total: {e2}")

    def _architect_pass(self) -> str:
        """
        Pass 1 (Architect): LLM menganalisis proyek dan membuat rencana dalam prosa.
        Lebih panjang dan lebih terarah daripada langsung ke format.
        """
        data = self.interview_data
        mechanics_str = ", ".join(data.mechanics[:6]) if data.mechanics else "basic features"
        requirements = (
            f"Type: {data.software_type.upper()} — {data.sub_type}\n"
            f"Language: {data.language} | Platform: {data.platform or 'desktop'}\n"
            f"Features: {mechanics_str}\n"
            f"Complexity: {data.complexity}\n"
            + (f"Notes: {data.extra_notes}" if data.extra_notes else "")
        )
        prompt = self._ARCHITECT_PROMPT.format(requirements=requirements)
        return self._call_llm(prompt, max_tokens=800)

    def _format_pass(self, architect_plan: str) -> str:
        """
        Pass 2 (Formatter): Ubah prosa architect menjadi format ## Step N yang parseable.
        """
        prompt = self._FORMATTER_PROMPT.format(architect_plan=architect_plan[:2000])
        return self._call_llm(prompt, max_tokens=1000)

    def _call_llm(self, prompt: str, max_tokens: int = 1200) -> str:
        """
        Panggil LLM via Marathon System (Step 6).
        
        Untuk plan generation, marathon_call_llm memastikan plan yang panjang
        tidak terpotong di tengah, dan prompt yang terlalu besar di-prune dulu.
        """
        def log_fn(msg: str, color: str):
            self.log_signal.emit(msg, color)

        try:
            result = marathon_call_llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.25,
                log_fn=log_fn,
                max_continues=5,  # Plan tidak perlu 8x — cukup 5x
            )
            return result.strip() if result else ""
        except Exception as e:
            self.log_signal.emit(f"  ⚠️ LLM error: {e}", "#ff8800")
            return ""


class SoftwareBuilderAgent:
    """
    Orkestrator utama Software Builder MOKO.
    
    Satu instance per sesi IDE — mengelola state interview dan plan aktif.
    Dipakai dari main_window_v5.py sebagai middleware antara ChatPanel dan LLM.
    
    Cara penggunaan:
        sba = SoftwareBuilderAgent(chat_panel=self.chat_panel, ...)
        
        # Cek apakah pesan ini adalah trigger atau lanjutan interview
        if sba.is_interview_active:
            sba.feed_answer(user_text)
        elif is_build_trigger(user_text):
            sba.start_session(hint=user_text)
    """

    def __init__(
        self,
        chat_panel=None,           # ChatPanel instance
        editor_panel=None,         # EditorPanel instance
        terminal_panel=None,       # TerminalPanel instance
        workspace_dir: str = "",   # Direktori kerja
        session_id: str = "default",
    ):
        self.chat_panel = chat_panel
        self.editor = editor_panel
        self.terminal = terminal_panel
        self.session_id = session_id

        # Workspace
        if workspace_dir:
            self.workspace_dir = workspace_dir
        else:
            from moko_config import settings
            self.workspace_dir = str(getattr(settings, "WORKSPACE_DIR", Path.home()))

        # State
        self._interview_mgr: InterviewManager = get_interview_manager(session_id)
        self._active_session: Optional[PlanSession] = None
        self._plan_worker: Optional[PlanGenerationWorker] = None
        self._step_executor = StepExecutor(
            terminal_execute_fn=self._run_in_terminal
        )

        print(f"  ✅ [SoftwareBuilderAgent] Initialized (session={session_id})")

    @property
    def is_interview_active(self) -> bool:
        """True jika sedang dalam proses interview multi-turn."""
        return self._interview_mgr.is_active

    @property
    def has_active_session(self) -> bool:
        """True jika ada plan session yang aktif."""
        return self._active_session is not None and self._active_session.is_active

    def start_session(self, hint: str = "") -> str:
        """
        Mulai sesi Software Builder baru.
        Inisialisasi interview dan kembalikan pertanyaan pertama.
        
        Args:
            hint: Teks dari perintah /coding (contoh: "buat game RPG platformer")
        
        Returns:
            Pertanyaan pertama untuk ditampilkan ke user
        """
        # Reset sesi lama jika ada
        if self._active_session:
            self._active_session.is_active = False

        self._interview_mgr.reset()

        # Mulai interview dengan hint dari /coding command
        first_question = self._interview_mgr.start(initial_hint=hint)

        self._emit_to_chat("core", first_question)
        print(f"  🎯 [SBA] Interview dimulai. State: {self._interview_mgr.state.name}")
        return first_question

    def feed_answer(self, user_text: str) -> bool:
        """
        Proses jawaban user ke interview yang sedang aktif.
        
        Returns:
            True jika interview masih berlanjut, False jika sudah selesai
        """
        if not self._interview_mgr.is_active:
            return False

        # Handle cancel
        if user_text.strip().lower() in ("/stop", "stop", "batal", "cancel"):
            self._interview_mgr.reset()
            self._emit_to_chat("system", "Interview dibatalkan. Ketik /coding untuk memulai ulang.")
            return False

        next_question, is_complete = self._interview_mgr.process_answer(user_text)

        if not is_complete and next_question:
            # Tampilkan pertanyaan berikutnya
            self._emit_to_chat("core", next_question)
            return True

        if is_complete:
            # Interview selesai — mulai generate plan
            data = self._interview_mgr.data
            summary = data.to_summary()
            self._emit_to_chat("core",
                f"{summary}\n\n"
                f"🔄 **Oke! Semua kebutuhan sudah terkumpul.**\n"
                f"Sekarang MOKO akan membuat **Implementation Plan** untuk proyek kamu...\n"
                f"*(Ini mungkin butuh beberapa detik)*"
            )

            # Buat workspace directory untuk proyek ini
            project_name = f"{data.software_type}_{data.sub_type}".replace(" ", "_").lower()
            project_dir = os.path.join(self.workspace_dir, f"moko_{project_name}_{int(time.time())}")
            os.makedirs(project_dir, exist_ok=True)

            # Buat session
            self._active_session = PlanSession(
                interview_data=data,
                workspace_dir=project_dir
            )

            # Generate plan di background
            self._start_plan_generation(data, project_dir)
            return False

        return True

    def execute_step(self, step_index: int):
        """
        Eksekusi satu langkah dari plan yang sudah ada.
        Dipanggil dari PlanStepCard.btn_run saat user klik "▶ Jalankan".
        
        Args:
            step_index: Index langkah yang akan dieksekusi (0-based)
        """
        if not self._active_session:
            self._emit_to_chat("system", "Tidak ada plan aktif. Gunakan /coding untuk memulai.")
            return

        steps = self._active_session.steps
        if step_index < 0 or step_index >= len(steps):
            self._emit_to_chat("system", f"Langkah {step_index + 1} tidak ditemukan.")
            return

        step = steps[step_index]

        if step.status == "DONE":
            self._emit_to_chat("system", f"Langkah {step.step_number} sudah selesai ✅")
            return

        self._emit_to_chat("system", f"▶ Menjalankan Step {step.step_number}: {step.title}...")

        self._step_executor.execute_step(
            step=step,
            session=self._active_session,
            on_log=lambda msg, color: self._emit_to_chat("thinking", msg),
            on_code=self._on_code_generated,
            on_done=lambda ok, msg: self._on_step_done(ok, msg, step),
        )

    def reset(self):
        """Reset seluruh state agent."""
        self._interview_mgr.reset()
        clear_interview_session(self.session_id)
        if self._active_session:
            self._active_session.is_active = False
            self._active_session = None
        self._step_executor.cancel()
        print(f"  🔄 [SBA] Agent direset")

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _start_plan_generation(self, data: InterviewData, workspace: str):
        """Start PlanGenerationWorker di background."""
        worker = PlanGenerationWorker(data, workspace)
        worker.log_signal.connect(lambda msg, color: self._emit_to_chat("thinking", msg))
        worker.plan_signal.connect(self._on_plan_generated)
        worker.done_signal.connect(self._on_plan_done)
        self._plan_worker = worker
        worker.start()

    def _on_plan_generated(self, steps: List[PlanStep]):
        """Callback saat plan berhasil di-generate."""
        if not self._active_session:
            return

        self._active_session.steps = steps

        # Tampilkan summary plan di chat
        plan_text = self._format_plan_summary(steps)
        self._emit_to_chat("core", plan_text)

        # Emit signal untuk render PlanStepCards di UI
        if hasattr(self, "_on_plan_ready_callback") and self._on_plan_ready_callback:
            self._on_plan_ready_callback(steps, self._active_session)

    def _on_plan_done(self, success: bool, message: str):
        """Callback saat PlanGenerationWorker selesai."""
        if not success:
            self._emit_to_chat("system", f"⚠️ {message}")

    def _on_code_generated(self, filename: str, code: str):
        """Callback saat kode berhasil di-generate — buka di editor."""
        if self._active_session and self.editor:
            fpath = os.path.join(self._active_session.workspace_dir, filename)
            try:
                self.editor.open_file(fpath)
            except Exception as e:
                print(f"  ⚠️ [SBA] Gagal buka editor: {e}")

    def _on_step_done(self, success: bool, message: str, step: PlanStep):
        """Callback saat satu step selesai dieksekusi."""
        if success:
            self._emit_to_chat("core",
                f"✅ **Step {step.step_number} selesai!**\n"
                f"{step.title}\n"
                + (f"Output: {step.terminal_output[:200]}" if step.terminal_output else "")
            )
        else:
            self._emit_to_chat("core",
                f"❌ **Step {step.step_number} gagal.**\n"
                f"Error: {step.error_message or 'Unknown error'}\n"
                f"Coba klik tombol Run lagi, atau perbaiki manual di editor."
            )

    def _run_in_terminal(self, cmd: str):
        """Jalankan command di TerminalPanel (dari UI thread via timer)."""
        if self.terminal:
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.terminal.execute_command(cmd))
            except Exception as e:
                print(f"  ⚠️ [SBA] Terminal execute error: {e}")

    def _emit_to_chat(self, source: str, text: str):
        """Emit pesan ke ChatPanel (thread-safe via Qt)."""
        if self.chat_panel:
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.chat_panel.append_message(source, text))
            except Exception as e:
                print(f"  ⚠️ [SBA] Chat emit error: {e}")
        else:
            print(f"  [SBA → chat/{source}]: {text[:80]}")

    def _format_plan_summary(self, steps: List[PlanStep]) -> str:
        """Format ringkasan plan untuk ditampilkan di chat."""
        lines = [
            "## 🗺️ Implementation Plan Siap!\n",
            f"Total: **{len(steps)} langkah** untuk menyelesaikan proyek kamu.\n",
            "Klik tombol **▶ Jalankan** di setiap langkah untuk mulai coding.\n\n",
        ]
        for step in steps:
            files_str = ", ".join(step.files_to_create) if step.files_to_create else "—"
            lines.append(
                f"**Step {step.step_number}: {step.title}**\n"
                f"{step.description[:100]}...\n"
                f"*Files: {files_str}*\n"
            )
        return "\n".join(lines)

    def set_plan_ready_callback(self, callback: Callable):
        """
        Set callback yang dipanggil saat plan siap.
        Callback signature: fn(steps: List[PlanStep], session: PlanSession)
        Dipakai oleh main_window_v5.py untuk render PlanStepCards.
        """
        self._on_plan_ready_callback = callback


# ─── Singleton factory ────────────────────────────────────────────────────────
_sba_instance: Optional[SoftwareBuilderAgent] = None


def get_software_builder_agent(
    chat_panel=None,
    editor_panel=None,
    terminal_panel=None,
    workspace_dir: str = "",
) -> SoftwareBuilderAgent:
    """
    Dapatkan atau buat SoftwareBuilderAgent singleton.
    Jika panel berubah (reinit), buat instance baru.
    """
    global _sba_instance
    if _sba_instance is None:
        _sba_instance = SoftwareBuilderAgent(
            chat_panel=chat_panel,
            editor_panel=editor_panel,
            terminal_panel=terminal_panel,
            workspace_dir=workspace_dir,
        )
    elif chat_panel is not None:
        # Update panel references jika berubah
        _sba_instance.chat_panel = chat_panel
        _sba_instance.editor = editor_panel
        _sba_instance.terminal = terminal_panel
    return _sba_instance


if __name__ == "__main__":
    print("=== Unit Test: software_builder_agent.py ===\n")

    # Test is_build_trigger
    assert is_build_trigger("/coding buat game RPG"), "buat game harus trigger"
    assert is_build_trigger("/coding bikin aplikasi web"), "bikin aplikasi web harus trigger"
    assert not is_build_trigger("/coding sort list python"), "sort list bukan trigger"
    assert not is_build_trigger("/coding perbaiki bug ini"), "perbaiki bug bukan trigger"
    print("✅ is_build_trigger() → OK")

    # Test SoftwareBuilderAgent tanpa UI (mode headless)
    agent = SoftwareBuilderAgent(workspace_dir="/tmp")
    assert not agent.is_interview_active
    print("✅ SoftwareBuilderAgent() init → OK")

    # Test start_session
    q1 = agent.start_session("buat game platformer")
    assert agent.is_interview_active
    from moko_agents.software_builder.interview_manager import InterviewState
    assert agent._interview_mgr.state == InterviewState.ASKING_SUBTYPE
    print(f"✅ start_session('buat game platformer'): auto-detect game, state=ASKING_SUBTYPE → OK")

    # Test feed_answer flow
    still_active = agent.feed_answer("platformer 2D")
    assert still_active  # masih di tengah interview
    print(f"✅ feed_answer('platformer 2D'): state={agent._interview_mgr.state.name} → OK")

    still_active = agent.feed_answer("player movement, collision, scoring")
    assert still_active
    print(f"✅ feed_answer mechanics: state={agent._interview_mgr.state.name} → OK")

    still_active = agent.feed_answer("python")
    assert still_active
    print(f"✅ feed_answer language: state={agent._interview_mgr.state.name} → OK")

    still_active = agent.feed_answer("medium")
    assert still_active
    print(f"✅ feed_answer complexity: state={agent._interview_mgr.state.name} → OK")

    # Jawab notes → interview complete → plan generation dimulai
    still_active = agent.feed_answer("skip")
    # Interview selesai → plan generation mulai di background (mungkin masih running)
    assert not agent.is_interview_active
    print(f"✅ feed_answer('skip'): interview complete, plan generation dimulai → OK")

    # Test reset
    agent.reset()
    assert not agent.is_interview_active
    assert not agent.has_active_session
    print(f"✅ reset(): clean state → OK")

    print("\n✅ Semua unit test software_builder_agent berhasil!")
