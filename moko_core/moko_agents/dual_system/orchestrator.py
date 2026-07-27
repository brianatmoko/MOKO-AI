"""
DualSystemOrchestrator — Koordinator Loop Sistem Ganda (Dual-System)
====================================================================
Koordinator pusat yang menyatukan Sistem 1 (Executor/Kimi) dan Sistem 2
(Brain + Guard/DeepSeek) ke dalam kalang iteratif otonom:

    Plan (Brain) → Execute (Executor) → Guard (Review)
        └─ jika gagal → analisis → Re-plan (Brain) → ulangi
        └─ jika lolos → Commit → laporkan sukses

Orkestrator mengekspos callback opsional agar lapisan UI (CognitiveWorker/PyQt6)
dapat menampilkan state kognitif aktif secara real-time:
- `on_state(label)`   → salah satu dari STATE_BRAIN / STATE_EXECUTOR / STATE_GUARD.
- `on_status(msg)`    → pesan progres granular dari tiap node.
- `on_progress(pct, label)` → persentase progres kasar untuk progress bar.

Berjalan mandiri tanpa PyQt/torch; workspace default berupa direktori sementara
agar tidak merusak repositori nyata.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from moko_agents.dual_system._bridge import CodeKnowledgeBase
from moko_agents.dual_system.brain_node import BrainNode
from moko_agents.dual_system.executor_node import ExecutorNode
from moko_agents.dual_system.runtime_guard import DualRuntimeGuard, VERDICT_COMMIT
from moko_agents.dual_system.worker_pool import WorkerPool
from moko_agents.dual_system.interaction_logger import InteractionLogger
from moko_agents.coding.coding_orchestrator import LoopDetector, _make_action_hash


# ── Confidence DB I/O ──────────────────────────────────────────────────────────
# Menyimpan riwayat keberhasilan MOKO lokal per kategori tugas.
# Format: { "coding": {"success": 5, "total": 8, "confidence": 0.625}, ... }
_CONFIDENCE_DB_PATH = Path(__file__).resolve().parents[4] / ".moko_local_confidence.json"


def _load_confidence_db() -> dict:
    """Muat riwayat confidence lokal dari disk."""
    try:
        if _CONFIDENCE_DB_PATH.exists():
            with open(_CONFIDENCE_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_confidence_db(db: dict) -> None:
    """Simpan riwayat confidence lokal ke disk."""
    try:
        _CONFIDENCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIDENCE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _estimate_task_complexity(prompt: str) -> tuple[str, float]:
    """
    Estimasi kompleksitas dan kategori tugas dari prompt pengguna.

    Mengembalikan (category, complexity_score) di mana:
      - category  : string domain (coding, math, general, dll.)
      - complexity : float 0.0 (sangat mudah) hingga 1.0 (sangat sulit)

    Ini adalah heuristik ringan tanpa memanggil LLM — cukup cepat untuk
    dijalankan setiap kali sebelum routing decision.
    """
    p = prompt.lower()
    word_count = len(p.split())

    # Tentukan kategori
    coding_kw = {"python", "javascript", "kode", "code", "fungsi", "function", "class",
                 "debug", "bug", "program", "rust", "api", "sql", "database"}
    math_kw   = {"hitung", "rumus", "integral", "turunan", "kalkulus", "matriks",
                 "statistik", "probabilitas", "prime", "prima", "formula"}

    coding_score = sum(1 for kw in coding_kw if kw in p)
    math_score   = sum(1 for kw in math_kw if kw in p)

    if coding_score > math_score and coding_score > 0:
        category = "coding"
    elif math_score > 0:
        category = "math"
    else:
        category = "general"

    # Estimasi kompleksitas dari heuristik
    complexity = 0.3  # Baseline

    # Panjang prompt → lebih panjang biasanya lebih kompleks
    complexity += min(0.3, word_count / 100.0)

    # Kata kunci tinggi-kompleksitas
    hard_kw = {"optimize", "optimasi", "arsitektur", "architecture", "sistem", "system",
               "multi-agent", "distributed", "concurrency", "paralel", "machine learning",
               "neural network", "transformer", "finetune", "distilasi"}
    complexity += min(0.3, sum(0.08 for kw in hard_kw if kw in p))

    # Kata kunci rendah-kompleksitas
    easy_kw = {"hello world", "ping", "jumlah", "sum", "pangkat", "square",
               "print", "tulis", "sederhana", "simple", "contoh", "example"}
    complexity -= min(0.2, sum(0.06 for kw in easy_kw if kw in p))

    complexity = max(0.0, min(1.0, complexity))
    return category, complexity


def should_try_local_first(
    task_category: str,
    task_complexity: float,
    confidence_db: dict,
    has_external_guru: bool,
) -> bool:
    """
    Menentukan apakah MOKO lokal 1.5B harus dicoba terlebih dahulu.

    Logika (sesuai riset 23_REVISI_MANDOR_API_MURID_LOKAL.md):
    - Jika tidak ada Guru API yang aktif → HARUS lokal (tidak ada pilihan lain).
    - Jika tugas sangat mudah (complexity < 0.3) → coba lokal dulu.
    - Jika confidence lokal untuk kategori ini tinggi (>= 0.70) → coba lokal dulu.
    - Jika complexity tinggi (>= 0.7) dan ada Guru API → delegasikan ke Guru.
    - Default: delegasikan ke Guru API untuk memaksimalkan data latih distilasi.

    Seiring waktu, semakin banyak data latih SFT yang dikumpulkan, confidence
    lokal akan naik dan lebih banyak tugas dikerjakan mandiri.
    """
    # Tidak ada Guru API → WAJIB lokal
    if not has_external_guru:
        return True

    # Tugas sangat mudah → lokal bisa
    if task_complexity < 0.25:
        return True

    # Cek confidence historis untuk kategori ini
    cat_entry = confidence_db.get(task_category, {})
    local_confidence = cat_entry.get("confidence", 0.0)
    total_tries = cat_entry.get("total", 0)

    # Butuh minimal 5 data points sebelum percaya confidence
    if total_tries >= 5 and local_confidence >= 0.70 and task_complexity < 0.6:
        return True

    # Default: delegasikan ke Guru untuk menghasilkan data latih baru
    return False


# ── Label state kognitif (dipakai UI Moko IDE v5) ──────────────────────────────
STATE_BRAIN = "🧠 BRAIN PLANNING"
STATE_EXECUTOR = "🔧 EXECUTOR ACTING"
STATE_GUARD = "🛡️ GUARD VALIDATING"


@dataclass
class IterationTrace:
    """Jejak satu iterasi kalang agentik."""
    attempt: int
    intent: str
    plan_thought: str
    plan_steps: list[str] = field(default_factory=list)
    retrieved_snippets: list[str] = field(default_factory=list)
    execution_success: bool = False
    guard_verdict: str = ""
    guard_summary: str = ""
    repair_hint: str = ""
    log: str = ""


@dataclass
class OrchestratorResult:
    """Hasil akhir eksekusi loop Sistem Ganda."""
    success: bool
    iterations: int
    committed: bool = False
    commit_ref: str = ""
    final_thought: str = ""
    summary: str = ""
    workspace_dir: str = ""
    traces: list[IterationTrace] = field(default_factory=list)


class DualSystemOrchestrator:
    """Koordinator loop iteratif Plan → Execute → Guard → (re-plan) → Commit."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        brain: Optional[BrainNode] = None,
        executor: Optional[ExecutorNode] = None,
        guard: Optional[DualRuntimeGuard] = None,
        knowledge_base=None,
        on_state: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, str], None]] = None,
        max_iterations: int = 3,
        git_commit: bool = False,
        force_bug_first: bool = False,
        worker_pool: Optional[WorkerPool] = None,
        interaction_logger: Optional[InteractionLogger] = None,
    ) -> None:
        self.on_state = on_state
        self.on_status = on_status
        self.on_progress = on_progress
        self.max_iterations = max(1, int(max_iterations))
        self.git_commit = git_commit
        self.force_bug_first = force_bug_first

        # Workspace: pakai temp dir jika tidak diberikan (non-destruktif).
        self._owns_tempdir = workspace_dir is None
        if workspace_dir is None:
            self._tempdir = tempfile.mkdtemp(prefix="moko_dual_ws_")
            self.workspace_dir = self._tempdir
        else:
            self._tempdir = None
            self.workspace_dir = str(Path(workspace_dir).resolve())

        self.kb = knowledge_base if knowledge_base is not None else CodeKnowledgeBase()
        self.brain = brain or BrainNode(knowledge_base=self.kb, on_status=self.on_status)
        self.executor = executor or ExecutorNode(
            self.workspace_dir, knowledge_base=self.kb, on_status=self.on_status
        )
        self.guard = guard or DualRuntimeGuard(on_status=self.on_status)

        # Worker Pool & Interaction Logger
        self.worker_pool = worker_pool
        self.interaction_logger = interaction_logger or InteractionLogger()

        # Loop Detector & Action Budget (Riset 26)
        self.loop_detector = LoopDetector(window=3, max_actions=max(30, self.max_iterations * 3))

    # ── Signaling helpers ──────────────────────────────────────────────────────
    def _set_state(self, label: str) -> None:
        if self.on_state:
            self.on_state(label)

    def _status(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)

    def _progress(self, pct: int, label: str) -> None:
        if self.on_progress:
            self.on_progress(pct, label)

    # ── Commit (aman & opsional) ───────────────────────────────────────────────
    def _finalize_commit(self, plan, execution_result) -> tuple[bool, str]:
        """Catat/lakukan commit. Secara default hanya mencatat commit-ready
        (tidak menyentuh git nyata). Commit git dilakukan hanya jika `git_commit`
        aktif DAN workspace adalah repo git valid."""
        files = ", ".join(Path(f).name for f in execution_result.written_files)
        commit_msg = f"MOKO Dual-System: {plan.intent} ({files})"
        if self.git_commit and (Path(self.workspace_dir) / ".git").exists():
            try:
                subprocess.run(
                    ["git", "add"] + list(execution_result.written_files),
                    cwd=self.workspace_dir, capture_output=True, text=True, timeout=15,
                )
                res = subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=self.workspace_dir, capture_output=True, text=True, timeout=15,
                )
                if res.returncode == 0:
                    return True, commit_msg
                return False, f"git commit gagal: {res.stderr.strip()[:120]}"
            except Exception as exc:  # noqa: BLE001
                return False, f"git commit error: {exc}"
        # Mode aman: catat commit-ready ke log workspace.
        try:
            log_path = Path(self.workspace_dir) / ".moko_commits.log"
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{int(time.time())}\tCOMMIT-READY\t{commit_msg}\n")
        except Exception:
            pass
        return False, f"[commit-ready] {commit_msg}"

    # ── Loop utama ──────────────────────────────────────────────────────────────
    def run_loop(self, user_prompt: str) -> OrchestratorResult:
        traces: list[IterationTrace] = []
        repair_hint = ""
        force_bug = self.force_bug_first
        final_thought = ""

        self._status("🚀 Dual-System Orchestrator: memulai kalang agentik otonom.")
        self.loop_detector.reset()

        # Cek ketersediaan worker API jika pool terpasang
        use_api_orchestration = False
        has_external_guru = False
        confidence_db = _load_confidence_db()

        if self.worker_pool:
            self._status("🔍 Mandor: Memindai koneksi pekerja di pool...")
            scan_results = self.worker_pool.scan_workers()
            active_workers = self.worker_pool.get_active_workers()
            external_active = [w for w in active_workers if w.provider != "local"]
            local_active   = [w for w in active_workers if w.provider == "local"]

            self._status(
                f"🔍 Mandor: Scan selesai → {len(external_active)} API eksternal aktif, "
                f"{len(local_active)} model lokal aktif."
            )

            has_external_guru = bool(external_active)

            # Estimasi kompleksitas tugas & routing decision
            task_category, task_complexity = _estimate_task_complexity(user_prompt)
            try_local_first = should_try_first = should_try_local_first(
                task_category, task_complexity, confidence_db, has_external_guru
            )

            if active_workers:
                use_api_orchestration = True
            else:
                self._status("⚠️ Mandor: Tidak ada pekerja aktif terdeteksi. Menggunakan jalur lokal existing.")
                use_api_orchestration = False

            if try_local_first:
                self._status(
                    f"🧠 Router: Mencoba lokal terlebih dahulu "
                    f"(kategori='{task_category}', complexity={task_complexity:.2f})."
                )
            elif has_external_guru:
                self._status(
                    f"🌐 Router: Delegasi ke Guru API "
                    f"(kategori='{task_category}', complexity={task_complexity:.2f})."
                )
        else:
            task_category = "general"
            task_complexity = 0.5
            try_local_first = True  # Tanpa pool, selalu lokal
            confidence_db = {}

        for attempt in range(self.max_iterations):
            # Cek loop dan budget (Riset 26)
            action_hash = _make_action_hash("DUAL_STEP", user_prompt, repair_hint)
            loop_status = self.loop_detector.check(action_hash)
            if loop_status in ("LOOP_DETECTED", "BUDGET_EXCEEDED"):
                self._status(f"⚠️ Kalang agen dihentikan otomatis oleh LoopDetector: {loop_status}")
                break

            base_pct = int(attempt / self.max_iterations * 100)

            # 1) Sistem 2 — Brain: rencana + unit test
            self._set_state(STATE_BRAIN)
            self._progress(base_pct + 5, STATE_BRAIN)

            if use_api_orchestration and self.worker_pool:
                # Tentukan mode: coba lokal atau delegasi ke Guru API
                run_local_this_attempt = try_local_first and attempt == 0

                if run_local_this_attempt:
                    # ── MODE LOKAL PERTAMA ────────────────────────────────────────────
                    self._status("🏠 Mode Lokal: MOKO 1.5B mencoba mengerjakan sendiri...")
                    plan = self.brain.reason_and_plan(
                        user_prompt, attempt=attempt, force_bug=force_bug, repair_hint=repair_hint
                    )
                    final_thought = plan.thought

                else:
                    # ── MODE GURU API (Mandor-Pekerja) ──────────────────────────────
                    mandor = self.worker_pool.get_mandor()
                    pekerja_list = self.worker_pool.get_pekerja_candidates()
                    worker_names = ", ".join(w.name for w in pekerja_list)

                    self._status(f"🧠 Mandor/Guru ({mandor.name}): Menyusun perencanaan struktur folder & target.")

                    # Prompt Mandor untuk merencanakan struktur & target outlines
                    mandor_prompt = (
                        f"Tugas utama: {user_prompt}\n"
                        f"Ini iterasi ke-{attempt}. Perbaikan sebelumnya (jika ada): {repair_hint}\n"
                        f"Pekerja yang tersedia: {worker_names}\n\n"
                        "Lakukan analisis dan kembalikan struktur perencanaan dalam format JSON valid "
                        "yang mengandung key berikut:\n"
                        "- 'thought': penalaran singkat cara menyelesaikan masalah ini.\n"
                        "- 'target_module': nama berkas kode utama (mis. 'moko_generated_runtime.py').\n"
                        "- 'test_module': nama berkas unit test (mis. 'test_moko_generated.py').\n"
                        "- 'folder_map': list dari subfolder yang perlu dipastikan ada.\n"
                        "- 'steps': daftar langkah pengerjaan (list of strings).\n"
                        "- 'outlines': dictionary berisi target file dan deskripsi spesifik tentang apa yang harus diisi di dalamnya.\n\n"
                        "Pastikan respon Anda hanya berupa JSON valid tanpa teks markdown pembungkus di luar JSON."
                    )

                    try:
                        mandor_response = mandor.generate_text(
                            prompt=mandor_prompt,
                            system_prompt="Kamu adalah Mandor/Guru AI (Koordinator perencana). Respon hanya dengan JSON valid.",
                            max_tokens=1024
                        )
                        
                        # Bersihkan jika ada code block markdown
                        clean_res = mandor_response.strip()
                        if clean_res.startswith("```json"):
                            clean_res = clean_res[7:]
                        if clean_res.endswith("```"):
                            clean_res = clean_res[:-3]
                        clean_res = clean_res.strip()
                        
                        plan_data = json.loads(clean_res)
                        
                        # Parse data plan
                        thought = plan_data.get("thought", "Perencanaan Guru API.")
                        target_module = plan_data.get("target_module", "moko_generated_runtime.py")
                        test_module = plan_data.get("test_module", "test_moko_generated.py")
                        steps = plan_data.get("steps", ["Tulis kode", "Jalankan unit test"])
                        outlines = plan_data.get("outlines", {})
                        
                    except Exception as e:
                        self._status(f"⚠️ Gagal memparsing perencanaan Mandor: {e}. Menggunakan fallback manual.")
                        # Fallback ke template
                        thought = "Gagal memanggil Mandor. Fallback ke deterministik."
                        target_module = "moko_generated_runtime.py"
                        test_module = "test_moko_generated.py"
                        steps = ["Tulis kode", "Jalankan unit test"]
                        outlines = {
                            target_module: "Tulis kode fitur utama",
                            test_module: "Tulis unit test otomatis"
                        }

                    # ── TAHAP DELEGASI KE PEKERJA ────────────────────────────────────
                    self._status("🔧 Orkestrator: Mendelegasikan penulisan kode ke Pekerja...")
                    
                    # ── TAHAP DELEGASI KE PEKERJA SECARA PARALEL ─────────────────────
                    self._status("🔧 Orkestrator: Mendelegasikan target module dan test module ke Pekerja secara paralel...")
                    
                    module_code = ""
                    test_code = ""

                    # Helper to call specific worker with failover
                    def call_pekerja_specific(worker_idx, prompt_text, system_prompt_text):
                        # Try to use the worker at index worker_idx, if it fails, rotate to others
                        workers = pekerja_list[worker_idx:] + pekerja_list[:worker_idx]
                        errors = []
                        for worker in workers:
                            try:
                                self._status(f"🔧 [Pekerja: {worker.name}] Memulai tugas...")
                                return worker.generate_text(prompt_text, system_prompt_text, max_tokens=1500)
                            except Exception as exc:
                                errors.append(f"Worker {worker.name} gagal: {exc}")
                        raise RuntimeError(f"Semua Pekerja gagal: {'; '.join(errors)}")

                    # Target Module Task
                    outline_target = outlines.get(target_module, "Tulis kode fitur utama")
                    buggy = bool(force_bug) and attempt == 0
                    buggy_desc = "Sengaja tambahkan bug logika untuk pengujian." if buggy else ""
                    worker_prompt_target = (
                        f"Tulis kode implementasi Python untuk berkas: '{target_module}'\n"
                        f"Berdasarkan outline: '{outline_target}'\n"
                        f"Permintaan asli: '{user_prompt}'\n"
                        f"Catatan error sebelumnya (jika ada): {repair_hint}\n"
                        f"{buggy_desc}\n\n"
                        "Tuliskan kode program Python lengkap yang valid. Bungkus kode program Anda di dalam tag <code> dan </code>."
                    )

                    # Test Module Task
                    outline_test = outlines.get(test_module, "Tulis unit test otomatis")
                    worker_prompt_test = (
                        f"Tulis kode unit test Python untuk menguji modul '{target_module[:-3]}'\n"
                        f"Berdasarkan outline: '{outline_test}'\n"
                        f"Unit test harus mengembalikan print 'MOKO_DUAL_TEST_PASSED' jika seluruh asersi lulus.\n\n"
                        "Tuliskan kode program unit test Python lengkap yang valid. Bungkus kode program Anda di dalam tag <code> dan </code>."
                    )

                    from concurrent.futures import ThreadPoolExecutor
                    
                    def run_target():
                        try:
                            # Target is assigned to first worker candidate
                            res = call_pekerja_specific(0, worker_prompt_target, "Kamu adalah Pekerja AI (Pemrogram Python). Keluarkan kode di dalam tag <code>...</code>")
                            if "<code>" in res and "</code>" in res:
                                return res.split("<code>")[1].split("</code>")[0].strip()
                            else:
                                clean = res.strip()
                                if clean.startswith("```python"): clean = clean[9:]
                                if clean.endswith("```"): clean = clean[:-3]
                                return clean.strip()
                        except Exception as e:
                            self._status(f"⚠️ Gagal mendapatkan kode target dari Pekerja: {e}")
                            return self.brain._build_module_code(None, buggy=force_bug)

                    def run_test():
                        try:
                            # Test is assigned to second worker candidate (or first if only one available)
                            w_idx = 1 if len(pekerja_list) > 1 else 0
                            res = call_pekerja_specific(w_idx, worker_prompt_test, "Kamu adalah Pekerja AI (Pemrogram Python Unit Test). Keluarkan kode di dalam tag <code>...</code>")
                            if "<code>" in res and "</code>" in res:
                                return res.split("<code>")[1].split("</code>")[0].strip()
                            else:
                                clean = res.strip()
                                if clean.startswith("```python"): clean = clean[9:]
                                if clean.endswith("```"): clean = clean[:-3]
                                return clean.strip()
                        except Exception as e:
                            self._status(f"⚠️ Gagal mendapatkan test code dari Pekerja: {e}")
                            return self.brain._build_test_code(target_module, "MOKO_DUAL_TEST_PASSED")

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        fut_target = executor.submit(run_target)
                        fut_test = executor.submit(run_test)
                        module_code = fut_target.result()
                        test_code = fut_test.result()

                    # Masukkan ke plan object
                    from moko_agents.dual_system.brain_node import ExecutionPlan
                    plan = ExecutionPlan(
                        user_prompt=user_prompt,
                        intent="agentic_code_task",
                        thought=thought,
                        steps=steps,
                        target_module=target_module,
                        test_module=test_module,
                        module_code=module_code,
                        test_code=test_code,
                        attempt=attempt,
                        repair_hint=repair_hint
                    )
                    final_thought = thought

            else:
                # ── ALUR LOKAL EXISTING ──
                plan = self.brain.reason_and_plan(
                    user_prompt, attempt=attempt, force_bug=force_bug, repair_hint=repair_hint
                )
                final_thought = plan.thought

            # 2) Sistem 1 — Executor: tulis kode + jalankan test
            self._set_state(STATE_EXECUTOR)
            self._progress(base_pct + 15, STATE_EXECUTOR)
            execution_result = self.executor.apply_plan(plan)

            # 3) Sistem 2 — Guard: tinjau log
            self._set_state(STATE_GUARD)
            self._progress(base_pct + 25, STATE_GUARD)
            report = self.guard.review(execution_result)

            trace = IterationTrace(
                attempt=attempt,
                intent=plan.intent,
                plan_thought=plan.thought,
                plan_steps=list(plan.steps),
                retrieved_snippets=list(execution_result.retrieved_snippets),
                execution_success=execution_result.success,
                guard_verdict=report.verdict,
                guard_summary=report.summary,
                repair_hint=repair_hint,
                log=execution_result.log,
            )
            traces.append(trace)

            if report.verdict == VERDICT_COMMIT:
                # Catat hasil sukses ke distill log
                is_guru_api_run = use_api_orchestration and not (try_local_first and attempt == 0)
                if self.interaction_logger:
                    self.interaction_logger.log_sample(
                        prompt=user_prompt,
                        thought=plan.thought,
                        code=plan.module_code,
                        passed_guard=True,
                        task_complexity=task_complexity,
                        task_category=task_category,
                        source="guru_api" if is_guru_api_run else "lokal"
                    )

                # Update confidence DB jika run lokal berhasil
                if try_local_first and attempt == 0 and report.verdict == VERDICT_COMMIT:
                    cat_entry = confidence_db.get(task_category, {"success": 0, "total": 0})
                    cat_entry["success"] = cat_entry.get("success", 0) + 1
                    cat_entry["total"]   = cat_entry.get("total",   0) + 1
                    cat_entry["confidence"] = cat_entry["success"] / cat_entry["total"]
                    confidence_db[task_category] = cat_entry
                    _save_confidence_db(confidence_db)
                    self._status(
                        f"📈 Confidence lokal untuk '{task_category}' diperbarui: "
                        f"{cat_entry['confidence']:.2f} ({cat_entry['success']}/{cat_entry['total']})"
                    )

                committed, commit_ref = self._finalize_commit(plan, execution_result)
                self._progress(100, "DONE")
                self._status(f"✅ Sukses pada iterasi {attempt + 1}. {commit_ref}")
                return OrchestratorResult(
                    success=True,
                    iterations=attempt + 1,
                    committed=committed,
                    commit_ref=commit_ref,
                    final_thought=final_thought,
                    summary=f"Sistem Ganda menyelesaikan tugas dalam {attempt + 1} iterasi.",
                    workspace_dir=self.workspace_dir,
                    traces=traces,
                )

            # 4) Gagal → koreksi-diri: Brain menganalisis log
            self._set_state(STATE_BRAIN)
            repair_hint = self.brain.analyze_failure(plan, execution_result.log)
            force_bug = False  # iterasi koreksi harus benar
            self._status(f"🔁 Iterasi {attempt + 1} gagal. Instruksi perbaikan: {repair_hint}")

        self._progress(100, "FAILED")
        self._status("❌ Loop selesai tanpa keberhasilan setelah batas iterasi.")
        return OrchestratorResult(
            success=False,
            iterations=self.max_iterations,
            committed=False,
            commit_ref="",
            final_thought=final_thought,
            summary=f"Gagal setelah {self.max_iterations} iterasi koreksi-diri.",
            workspace_dir=self.workspace_dir,
            traces=traces,
        )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """Hapus workspace sementara jika orkestrator yang membuatnya."""
        if self._owns_tempdir and self._tempdir:
            import shutil
            shutil.rmtree(self._tempdir, ignore_errors=True)

