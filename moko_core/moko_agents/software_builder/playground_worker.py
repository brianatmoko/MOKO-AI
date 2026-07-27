"""
playground_worker.py — Sistem Bermain (Playground Mode) MOKO IDE
=================================================================
Menghasilkan dan menjalankan kode secara sementara di direktori temp
tanpa menyimpan apapun ke folder project permanen.

Fitur utama:
  - Generate kode via LLM menggunakan marathon system
  - Tulis ke tempfile.mkdtemp() — BUKAN folder project
  - Jalankan di terminal terintegrasi MOKO
  - Tampilkan output di Chat Panel
  - Cleanup otomatis saat sesi selesai
  - Tombol "Simpan" opsional: pindah dari tempdir ke folder project aktif
  - Support Python, JavaScript (node), Bash, dan plaintext
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, List

from PyQt6.QtCore import QThread, pyqtSignal


# ─── PlaygroundSession ─────────────────────────────────────────────────────────

class PlaygroundSession:
    """
    Merepresentasikan satu sesi playground (sementara, tidak disimpan).
    Setiap sesi memiliki tempdir tersendiri yang akan dihapus saat cleanup().
    """

    def __init__(self):
        self.tempdir: str = tempfile.mkdtemp(prefix="moko_play_")
        self.files: List[str] = []     # file-file yang dibuat di tempdir
        self.language: str = "python"
        self.main_file: str = ""       # path file utama yang dijalankan
        self.code: str = ""            # kode terakhir yang di-generate
        self.output: str = ""          # output eksekusi
        self.is_active: bool = True

    def cleanup(self):
        """Hapus seluruh tempdir. Dipanggil saat sesi selesai."""
        if self.tempdir and os.path.isdir(self.tempdir):
            try:
                shutil.rmtree(self.tempdir)
            except Exception:
                pass
        self.is_active = False

    def save_to(self, dest_folder: str) -> List[str]:
        """
        Pindahkan semua file playground ke dest_folder.
        Digunakan saat user klik tombol 'Simpan'.
        
        Returns:
            List path file yang berhasil dipindahkan.
        """
        saved = []
        os.makedirs(dest_folder, exist_ok=True)
        for fpath in self.files:
            if os.path.isfile(fpath):
                fname = os.path.basename(fpath)
                dest = os.path.join(dest_folder, fname)
                # Jangan overwrite tanpa versioning
                if os.path.exists(dest):
                    base, ext = os.path.splitext(fname)
                    for i in range(1, 100):
                        alt = os.path.join(dest_folder, f"{base}_{i}{ext}")
                        if not os.path.exists(alt):
                            dest = alt
                            break
                shutil.copy2(fpath, dest)
                saved.append(dest)
        return saved

    def __repr__(self):
        return f"<PlaygroundSession dir={self.tempdir} lang={self.language} files={len(self.files)}>"


# ─── PlaygroundWorker ──────────────────────────────────────────────────────────

class PlaygroundWorker(QThread):
    """
    QThread worker yang:
    1. Generate kode menggunakan LLM (dengan marathon system)
    2. Tulis ke tempdir (ephemeral, bukan folder project)
    3. Jalankan di terminal
    4. Emit hasilnya ke Chat Panel
    
    Signals:
        log_signal(message, color)   — progress log ke chat/XRay
        code_ready(code, filepath)   — kode siap ditampilkan di editor
        run_command(cmd, cwd)        — minta terminal untuk menjalankan command
        done_signal(session)         — sesi selesai, kirim PlaygroundSession
        error_signal(message)        — terjadi error
    """
    log_signal   = pyqtSignal(str, str)
    code_ready   = pyqtSignal(str, str)        # (code_text, file_path)
    run_command  = pyqtSignal(str, str)        # (command, cwd)
    done_signal  = pyqtSignal(object)          # PlaygroundSession
    error_signal = pyqtSignal(str)

    # ── Prompt template untuk playground (bukan software builder) ─────────────
    SYSTEM_PROMPT = (
        "You are MOKO, an expert AI programmer running in PLAYGROUND MODE. "
        "Your task: write COMPLETE, RUNNABLE code based on the user's request. "
        "CRITICAL RULES:\n"
        "- Output ONLY code inside a single fenced block (```lang ... ```)\n"
        "- NEVER use lazy comments like '# rest of code here' or '...' or 'TODO'\n"
        "- Code MUST be self-contained and runnable without any external setup\n"
        "- Prefer short, focused scripts that demonstrate the concept clearly\n"
        "- Add a brief # PLAYGROUND: ... comment at the top explaining what this does\n"
        "- Use standard library whenever possible (no pip install required)\n"
        "- For games: use pygame (import pygame at top) with a simple game loop\n"
        "- For web: use http.server or Flask with minimal setup\n"
        "- For data: use random/math/statistics from stdlib"
    )

    def __init__(
        self,
        instruction: str,
        language: str = "auto",
        context_code: str = "",
        rag_context: str = "",
        parent=None
    ):
        super().__init__(parent)
        self.instruction = instruction
        self.language = language        # "auto", "python", "javascript", "bash"
        self.context_code = context_code
        self.rag_context = rag_context
        self.session: Optional[PlaygroundSession] = None
        self._should_stop = False
        # Storage untuk Template Learning Engine (bisa dioverride saat test).
        self._learning_storage_dir: Optional[str] = None

    def stop(self):
        self._should_stop = True

    def run(self):
        self.session = PlaygroundSession()
        try:
            self._run_playground()
        except Exception as e:
            self.error_signal.emit(f"Playground error: {e}")
            if self.session:
                self.session.cleanup()
        finally:
            if self.session and self.session.is_active:
                self.done_signal.emit(self.session)

    def _log(self, msg: str, color: str = "#00e6ff"):
        self.log_signal.emit(msg, color)

    def _check_server_online(self) -> tuple[bool, str]:
        """Cek status server AI dan kembalikan (is_online, status_label)."""
        try:
            from moko_inference.server_manager import MOKO_PID_FILE, MokoLocalInferenceServer
            from moko_config import settings as _s
            status = MokoLocalInferenceServer.get_server_status(_s.MOKO_LLM_PORT)
            status_label = str(status).lower().strip()
            if status_label in ("ok", "online", "ready", "running", "healthy"):
                return True, status_label
            if status_label in ("loading", "initializing", "starting", "warmup"):
                return False, "loading"
            # Race condition startup: proses server sudah hidup tapi port health belum terbuka.
            # Perlakukan sebagai loading agar playground menunggu, bukan auto-close terlalu cepat.
            if status_label in ("offline", "unknown", ""):
                try:
                    pid = MokoLocalInferenceServer._read_pid_from_file(MOKO_PID_FILE)
                    start_ts = MokoLocalInferenceServer._read_start_time(MOKO_PID_FILE)
                    age_s = time.time() - start_ts
                    if MokoLocalInferenceServer._is_process_running(pid) and age_s < 60:
                        return False, "loading"
                except Exception:
                    pass
            if not status_label:
                return False, "unknown"
            return False, status_label
        except Exception:
            return False, "unknown"

    def _wait_until_server_ready(self, timeout_seconds: int = 240, poll_interval: float = 1.0) -> tuple[bool, str]:
        """
        Tunggu server dari status loading hingga benar-benar online.
        Return:
            (True, status) jika ready
            (False, status_akhir) jika timeout / berubah ke status tidak siap / stop
        """
        start = time.time()
        last_log_second = -1

        while not self._should_stop:
            is_online, state = self._check_server_online()
            if is_online:
                return True, state

            elapsed = int(time.time() - start)
            if state != "loading":
                return False, state

            # Log progress setiap 5 detik agar chat tidak spam
            if elapsed // 5 != last_log_second // 5:
                last_log_second = elapsed
                self._log(
                    f"  ⏳ Server loading ({elapsed}s) — menunggu hingga 100% aktif...",
                    "#ffaa00",
                )

            if elapsed >= timeout_seconds:
                return False, "loading_timeout"

            time.sleep(poll_interval)

        return False, "stopped"

    def _attempt_server_recovery(self) -> tuple[bool, str]:
        """
        Coba recovery server saat status awal offline/unknown.
        Tetap patuh fail-close: jika recovery gagal, caller wajib menutup sesi.
        """
        try:
            from moko_inference.server_manager import MokoLocalInferenceServer

            self._log("  🔄 Server AI tidak siap — mencoba start/recover server...", "#ffaa00")
            started = bool(MokoLocalInferenceServer.start_servers())
            if self._should_stop:
                return False, "stopped"

            is_online, state = self._check_server_online()
            if is_online:
                return True, state

            if started and state == "loading":
                return self._wait_until_server_ready(timeout_seconds=240)

            return False, state
        except Exception as e:
            self._log(f"  ⚠️ Recovery server gagal: {e}", "#ff8800")
            return False, "recover_error"

    def _fail_and_close_playground(self, reason: str):
        """Hard fail playground: kirim error lalu auto close sesi."""
        self._log(f"  ❌ {reason}", "#ff4444")
        self.error_signal.emit(reason)
        self._should_stop = True
        if self.session and self.session.is_active:
            self.session.cleanup()

    def _run_playground(self):
        """Main playground pipeline."""
        self._log("🎮 [Playground] Mode aktif — kode tidak akan disimpan permanen", "#ff00ff")

        # 1. Deteksi bahasa
        lang = self._detect_language(self.instruction, self.language)
        self.session.language = lang
        self._log(f"  📝 Bahasa terdeteksi: {lang.upper()}", "#00e6ff")

        # 2. Cek server AI
        server_online, server_state = self._check_server_online()
        if not server_online:
            if server_state == "loading":
                self._log(
                    "  ⏳ MOKO SERVER loading — playground menunggu server 100% aktif",
                    "#ffaa00"
                )
                server_online, server_state = self._wait_until_server_ready()
            else:
                server_online, server_state = self._attempt_server_recovery()

            if not server_online:
                if server_state == "loading_timeout":
                    self._fail_and_close_playground(
                        "Server masih loading melewati batas waktu. Playground dihentikan otomatis."
                    )
                else:
                    self._fail_and_close_playground(
                        f"Server AI tidak online (status: {server_state}). Playground dihentikan otomatis."
                    )
                return

        # 3. Build prompt
        prompt = self._build_prompt(lang)

        # 4. Generate kode via LLM (marathon)
        code = ""
        if not self._should_stop:
            self._log("  🤖 AI sedang membuat kode untuk: " + self.instruction[:60] + ("..." if len(self.instruction) > 60 else ""), "#ffcc00")
            code = self._call_llm(prompt)
            if code:
                self._log("  ✅ AI berhasil generate kode!", "#00ff88")
            else:
                self._fail_and_close_playground(
                    "AI tidak merespons saat server online. Playground dihentikan otomatis (tanpa template fallback)."
                )
                return

        if self._should_stop:
            return

        # 4. Ekstrak kode dari response
        clean_code = self._extract_code(code, lang)
        if not clean_code:
            clean_code = code  # Pakai raw jika ekstraksi gagal

        self.session.code = clean_code

        # 4b. Auto-learning: catat hasil generasi LLM sebagai data pembelajaran.
        #     Non-fatal — kegagalan capture TIDAK boleh menghentikan playground.
        self._capture_learning(self.instruction, clean_code, lang)

        # 5. Tulis ke tempdir
        filepath = self._write_to_tempdir(clean_code, lang)
        self.session.main_file = filepath
        self._log(f"  📁 File sementara: {filepath}", "rgba(0,230,255,0.5)")
        self._log("     (file ini TIDAK disimpan ke project — hanya di /tmp)", "#888888")

        # 6. Emit kode ke editor (read-only preview)
        self.code_ready.emit(clean_code, filepath)

        # 7. Jalankan di terminal
        cmd = self._build_run_command(filepath, lang)
        self._log(f"  ▶ Menjalankan: {cmd}", "#00ff88")
        self.run_command.emit(cmd, self.session.tempdir)

        self._log("  ✅ [Playground] Siap! Kode berjalan di terminal.", "#00ff88")
        self._log("  💡 Ketik `/play clear` untuk bersihkan, atau klik [Simpan] untuk menyimpan", "#888888")

    def _detect_language(self, instruction: str, hint: str) -> str:
        """Deteksi bahasa dari instruksi atau hint."""
        if hint and hint not in ("auto", ""):
            return hint.lower()

        inst_lower = instruction.lower()

        # JavaScript / Node
        if any(kw in inst_lower for kw in ["javascript", "nodejs", "node.js", "js", "react", "vue"]):
            return "javascript"

        # Bash / shell
        if any(kw in inst_lower for kw in ["bash", "shell", "sh script", "zsh"]):
            return "bash"

        # HTML
        if any(kw in inst_lower for kw in ["html", "webpage", "web page", "website"]):
            return "html"

        # Default: Python
        return "python"

    def _build_prompt(self, lang: str) -> str:
        """Build prompt untuk LLM playground — spesifik dan personal sesuai instruksi."""
        inst_lower = self.instruction.lower()

        # Tentukan contoh spesifik berdasarkan kata kunci instruksi
        example_hint = ""
        if any(k in inst_lower for k in ["kalkulator", "calculator", "hitung", "kalkulasi"]):
            example_hint = (
                "Create an interactive calculator that:\n"
                "- Reads math expressions from user input (like: 2 + 3 * 4)\n"
                "- Shows the result\n"
                "- Loops until user types 'exit' or 'q'\n"
                "- Handles ZeroDivisionError and invalid input gracefully\n"
            )
        elif any(k in inst_lower for k in ["game", "snake", "platformer", "puzzle", "arcade"]):
            example_hint = (
                "Create a playable mini game with:\n"
                "- Clear win/lose condition\n"
                "- User interaction (keyboard input)\n"
                "- Score tracking\n"
                "- Use only Python stdlib (no pygame required for text games)\n"
            )
        elif any(k in inst_lower for k in ["todo", "task", "list", "daftar"]):
            example_hint = (
                "Create a to-do list manager with add/remove/view/complete commands.\n"
            )
        elif any(k in inst_lower for k in ["sort", "urutkan", "sorting", "search", "cari"]):
            example_hint = (
                "Implement the algorithm, demo it on a random sample list, print step-by-step.\n"
            )
        elif any(k in inst_lower for k in ["cuaca", "weather", "api", "fetch"]):
            example_hint = (
                "Use Python's urllib to fetch and display data. Show response parsing clearly.\n"
            )

        parts = [
            f"## USER REQUEST\n{self.instruction}\n",
            f"## TARGET LANGUAGE\n{lang.upper()}\n",
        ]

        if example_hint:
            parts.append(f"## SPECIFIC REQUIREMENTS\n{example_hint}\n")

        if self.context_code:
            parts.append(f"## EXISTING CODE (for reference)\n```{lang}\n{self.context_code[:1500]}\n```\n")

        if self.rag_context:
            parts.append(f"## TECHNICAL CONTEXT\n{self.rag_context[:600]}\n")

        parts.append(
            f"## OUTPUT FORMAT (CRITICAL)\n"
            f"- Output ONLY a single fenced code block starting with ```{lang}\n"
            f"- The code MUST be complete and runnable as-is (no missing pieces)\n"
            f"- Use ONLY Python standard library — NO external packages (no pip install needed)\n"
            f"- Every function must be fully implemented — NEVER write '# rest of code here' or '# TODO'\n"
            f"- Every function body must have real code, not just 'pass'\n"
            f"- Code should be under 150 lines\n"
            f"- Start the code block NOW:\n"
        )

        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Panggil LLM dengan marathon system."""
        try:
            from moko_agents.software_builder.token_manager import marathon_call_llm
            return marathon_call_llm(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.12,
                system_prompt=self.SYSTEM_PROMPT,
                log_fn=self._log,
                max_continues=5,
            )
        except Exception as e:
            self._log(f"  ⚠️ LLM call error: {e}", "#ff8800")
            return ""

    def _detect_task_type(self, instruction: str) -> str:
        """Petakan instruksi ke task-type untuk Template Learning Engine."""
        low = instruction.lower()
        if any(k in low for k in ("kalkulator", "calculator", "hitung", "kalkulasi")):
            return "kalkulator"
        if any(k in low for k in ("game", "snake", "platformer", "puzzle", "arcade")):
            return "game"
        if any(k in low for k in ("todo", "task", "daftar", "list")):
            return "todo"
        if any(k in low for k in ("sort", "urut", "search", "cari", "algorit")):
            return "algoritma"
        return "general"

    def _learning_docs_dir(self) -> Path:
        """Lokasi paket riset (docs) tempat Template Learning Engine berada."""
        return Path(__file__).resolve().parents[3] / "docs"

    def _capture_learning(self, intent: str, code: str, lang: str) -> bool:
        """
        Catat hasil generasi LLM sebagai data pembelajaran (auto-capture dataset).

        Mempromosikan output menjadi template baru (origin="generated") sehingga
        setiap generasi memperkaya library untuk iterasi berikutnya —
        mewujudkan continuous learning loop dari walkthrough Template Learning.

        Sifatnya NON-FATAL: exception apa pun ditelan agar `/play` tidak gagal.
        """
        if lang != "python" or not code.strip() or not intent.strip():
            return False
        try:
            import sys as _sys

            docs_dir = self._learning_docs_dir()
            if docs_dir.is_dir() and str(docs_dir) not in _sys.path:
                _sys.path.insert(0, str(docs_dir))

            from moko_template_learning import TemplateLearningEngine

            storage = self._learning_storage_dir or str(
                docs_dir / "riset" / "data" / "template_learning"
            )
            engine = TemplateLearningEngine(storage_dir=storage)
            keywords = re.findall(r"[a-zA-Z_]{2,}", intent.lower())
            engine.register_template(
                self._detect_task_type(intent),
                code,
                notes=f"LLM playground generation: {intent[:120]}",
                keywords=keywords,
                origin="generated",
            )
            self._log("  🧠 Output dicatat ke Template Learning dataset", "#66ffcc")
            return True
        except Exception as e:  # pragma: no cover - non-fatal capture
            self._log(f"  ⚠️ Gagal mencatat learning (diabaikan): {e}", "#ff8800")
            return False

    def _extract_code(self, text: str, lang: str) -> str:
        """
        Ekstrak kode dari response LLM.
        Strategi: fenced block dengan tag bahasa → fenced block tanpa tag → raw.
        """
        if not text:
            return ""

        # 1. Fenced block dengan tag bahasa spesifik
        pattern = re.compile(
            rf"```{re.escape(lang)}[\s\S]*?\n([\s\S]+?)```",
            re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            return m.group(1).strip()

        # 2. Fenced block tanpa tag (generic)
        m2 = re.search(r"```(?:\w+)?\s*\n([\s\S]+?)```", text)
        if m2:
            return m2.group(1).strip()

        # 3. Raw: kalau sudah ada def / import / class di awal — pakai langsung
        stripped = text.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("#")
            or stripped.startswith("<!DOCTYPE")
            or stripped.startswith("<html")
        ):
            return stripped

        return stripped

    def _write_to_tempdir(self, code: str, lang: str) -> str:
        """Tulis kode ke tempdir dan return filepath."""
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "html": ".html",
            "ruby": ".rb",
            "go": ".go",
        }
        ext = ext_map.get(lang, ".txt")
        filename = f"playground{ext}"
        filepath = os.path.join(self.session.tempdir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        # Bash: jadikan executable
        if lang == "bash":
            os.chmod(filepath, 0o755)

        self.session.files.append(filepath)
        return filepath

    def _build_run_command(self, filepath: str, lang: str) -> str:
        """Build command untuk menjalankan file."""
        cmd_map = {
            "python": f"python3 '{filepath}'",
            "javascript": f"node '{filepath}'",
            "bash": f"bash '{filepath}'",
            "html": f"xdg-open '{filepath}'",
        }
        return cmd_map.get(lang, f"cat '{filepath}'")

    def _fallback_template(self, lang: str) -> str:
        """Template fallback jika LLM tidak tersedia — berbasis keyword instruksi."""
        instruction_preview = self.instruction[:80].replace('"', "'")
        inst_lower = self.instruction.lower()

        if lang == "python":
            # Kalkulator
            if any(k in inst_lower for k in ["kalkulator", "calculator", "hitung", "kalkulasi"]):
                return f"""# PLAYGROUND: Kalkulator Multimode — "{instruction_preview}"
# (Generated by MOKO Offline Synthesizer)

import ast
import math
import operator


ALLOWED_BINARY = {{
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}}

ALLOWED_UNARY = {{
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}}

ALLOWED_FUNCS = {{
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "luas_persegi_panjang": lambda p, l: p * l,
    "luas_persegi": lambda s: s * s,
    "keliling_persegi_panjang": lambda p, l: 2 * (p + l),
    "keliling_persegi": lambda s: 4 * s,
    "volume_balok": lambda p, l, t: p * l * t,
    "volume_kubus": lambda s: s ** 3,
}}


def evaluate_expression(expr: str, last_result: float = 0.0) -> float:
    cleaned = expr.strip()
    if not cleaned:
        raise ValueError("Input kosong.")

    tree = ast.parse(cleaned, mode="eval")
    variables = {{"ans": float(last_result), "pi": math.pi, "e": math.e}}
    return _eval_ast(tree.body, variables)


def _eval_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in variables:
            return float(variables[key])
        raise ValueError(f"Variabel '{{node.id}}' tidak dikenali")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINARY:
            raise ValueError("Operator tidak diizinkan")
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        return float(ALLOWED_BINARY[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY:
            raise ValueError("Operator unary tidak diizinkan")
        value = _eval_ast(node.operand, variables)
        return float(ALLOWED_UNARY[op_type](value))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id.lower()
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Fungsi '{{node.func.id}}' tidak diizinkan")
        args = [_eval_ast(arg, variables) for arg in node.args]
        return float(ALLOWED_FUNCS[func_name](*args))

    raise ValueError("Ekspresi tidak diizinkan")


def kalkulator_multimode() -> None:
    print("=" * 52)
    print("  🔢 MOKO Kalkulator Multimode — Smart & Geometry")
    print("=" * 52)
    print("Mode Kalkulasi:")
    print("  1. Ekspresi Matematika: 2+3*4, sqrt(81), (10-3)**2")
    print("  2. Luas Persegi Panjang: luas_persegi_panjang(panjang, lebar)")
    print("  3. Luas Persegi: luas_persegi(sisi)")
    print("  4. Keliling Persegi Panjang: keliling_persegi_panjang(panjang, lebar)")
    print("  5. Keliling Persegi: keliling_persegi(sisi)")
    print("  6. Volume Balok: volume_balok(panjang, lebar, tinggi)")
    print("  7. Volume Kubus: volume_kubus(sisi)")
    print("Fungsi: sqrt, sin, cos, tan, log, abs, round")
    print("Konstanta: pi, e | Variabel: ans")
    print("Perintah: help, mode, q\\n")

    ans = 0.0
    current_mode = "ekspresi"

    while True:
        if current_mode == "ekspresi":
            expr = input(">>> ").strip()
        else:
            expr = input(f"[{current_mode}] >>> ").strip()

        lower = expr.lower()
        if not expr or lower in ("q", "quit", "exit", "keluar"):
            print("Sampai jumpa!")
            break

        if lower in ("help", "bantuan"):
            print("    Contoh mode ekspresi: 2+3*4, sqrt(81), ans/2")
            print("    Contoh geometri: luas_persegi_panjang(5, 3), luas_persegi(4)")
            print("    Ketik 'mode' untuk ganti mode")
            continue

        if lower in ("mode",):
            print("\\nPilih mode:")
            print("  1. Ekspresi Matematika")
            print("  2. Kalkulator Geometri")
            pilihan = input("Pilihan (1/2): ").strip()
            if pilihan == "2":
                current_mode = "geometri"
                print("Mode geometri aktif. Gunakan: luas_persegi_panjang(p, l), luas_persegi(s)")
            else:
                current_mode = "ekspresi"
                print("Mode ekspresi aktif.")
            continue

        try:
            ans = evaluate_expression(expr, last_result=ans)
            print(f"    = {{ans:g}}")
        except ZeroDivisionError:
            print("    ⚠ Tidak bisa dibagi nol!")
        except Exception as error:
            print(f"    ⚠ {{error}}")


if __name__ == "__main__":
    kalkulator_multimode()
'''

import ast
import math
import operator


ALLOWED_BINARY = {{
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}}

ALLOWED_UNARY = {{
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}}

ALLOWED_FUNCS = {{
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
}}


def evaluate_expression(expr: str, last_result: float = 0.0) -> float:
    cleaned = expr.strip()
    if not cleaned:
        raise ValueError("Input kosong.")

    tree = ast.parse(cleaned, mode="eval")
    variables = {{"ans": float(last_result), "pi": math.pi, "e": math.e}}
    return _eval_ast(tree.body, variables)


def _eval_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in variables:
            return float(variables[key])
        raise ValueError(f"Variabel '{{node.id}}' tidak dikenali")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINARY:
            raise ValueError("Operator tidak diizinkan")
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        return float(ALLOWED_BINARY[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY:
            raise ValueError("Operator unary tidak diizinkan")
        value = _eval_ast(node.operand, variables)
        return float(ALLOWED_UNARY[op_type](value))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id.lower()
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Fungsi '{{node.func.id}}' tidak diizinkan")
        args = [_eval_ast(arg, variables) for arg in node.args]
        return float(ALLOWED_FUNCS[func_name](*args))

    raise ValueError("Ekspresi tidak diizinkan")


def kalkulator_keren() -> None:
    print("=" * 52)
    print("  🔢 MOKO Kalkulator Playground — Smart Mode")
    print("=" * 52)
    print("Operasi: +  -  *  /  //  **  %")
    print("Fungsi: sqrt, sin, cos, tan, log, abs, round")
    print("Konstanta: pi, e | Variabel: ans")
    print("Perintah: help, q\\n")

    ans = 0.0
    while True:
        expr = input(">>> ").strip()
        lower = expr.lower()
        if not expr or lower in ("q", "quit", "exit", "keluar"):
            print("Sampai jumpa!")
            break
        if lower in ("help", "bantuan"):
            print("    Contoh: 2+3*4, sqrt(81), ans/2, (10-3)**2")
            continue
        try:
            ans = evaluate_expression(expr, last_result=ans)
            print(f"    = {{ans:g}}")
        except ZeroDivisionError:
            print("    ⚠ Tidak bisa dibagi nol!")
        except Exception as error:
            print(f"    ⚠ {{error}}")


if __name__ == "__main__":
    kalkulator_keren()
"""
            # Game / snake / platformer / puzzle
            elif any(k in inst_lower for k in ["game", "snake", "platformer", "puzzle", "arcade"]):
                return f"""# PLAYGROUND: Mini Game — "{instruction_preview}"
# (Generated by MOKO Offline Synthesizer)

import random

def game():
    '''Mini tebak-angka game MOKO Playground.'''
    print("=" * 40)
    print("  🎮 MOKO Mini Game Playground")
    print("=" * 40)
    target = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    print(f"Tebak angka 1-100! Kamu punya {{max_attempts}} kesempatan.\\n")

    while attempts < max_attempts:
        try:
            guess = int(input(f"Tebakan {{attempts+1}}/{{max_attempts}}: "))
            attempts += 1
            if guess == target:
                print(f"🎉 BENAR! Angkanya {{target}}. Selesai dalam {{attempts}} percobaan!")
                return
            elif guess < target:
                print("  ↑ Terlalu kecil!")
            else:
                print("  ↓ Terlalu besar!")
        except ValueError:
            print("  ⚠ Masukkan angka!")
    
    print(f"\\n😅 Habis! Angkanya adalah {{target}}.")

if __name__ == "__main__":
    game()
"""
            # Default: demo sederhana
            else:
                return f"""# PLAYGROUND: "{instruction_preview}"
# (Generated by MOKO Offline Synthesizer)

import random
import time

def demo():
    '''Demo playground MOKO IDE.'''
    print("=" * 50)
    print("MOKO Playground Mode")
    print(f"Instruksi: {instruction_preview}")
    print("=" * 50)

    results = []
    for i in range(10):
        val = random.randint(1, 100)
        results.append(val)
        print(f"  Item {{i+1:2d}}: {{val}}")
        time.sleep(0.05)

    print()
    print(f"Total: {{sum(results)}}")
    print(f"Rata-rata: {{sum(results)/len(results):.2f}}")
    print(f"Min: {{min(results)}}, Max: {{max(results)}}")
    print("\\n✅ Demo selesai!")

if __name__ == "__main__":
    demo()
"""

        elif lang == "javascript":
            return f'''// PLAYGROUND: Demo untuk "{instruction_preview}"
// (Generated by MOKO Offline Synthesizer)

console.log("=".repeat(50));
console.log("MOKO Playground Mode (JavaScript)");
console.log("Instruksi: {instruction_preview}");
console.log("=".repeat(50));

const results = Array.from({{length: 10}}, () => Math.floor(Math.random() * 100) + 1);
results.forEach((v, i) => console.log(`  Item ${{String(i+1).padStart(2, " ")}}: ${{v}}`));

const sum = results.reduce((a, b) => a + b, 0);
console.log(`\\nTotal: ${{sum}}`);
console.log(`Rata-rata: ${{(sum / results.length).toFixed(2)}}`);
console.log("\\n✅ Demo selesai!");
'''

        elif lang == "bash":
            return f'''#!/bin/bash
# PLAYGROUND: Demo untuk "{instruction_preview}"
# (Generated by MOKO Offline Synthesizer)

echo "=================================="
echo "MOKO Playground Mode (Bash)"
echo "Instruksi: {instruction_preview}"
echo "=================================="

for i in $(seq 1 5); do
    echo "  Step $i: Processing..."
    sleep 0.1
done

echo ""
echo "System info:"
echo "  OS   : $(uname -s)"
echo "  User : $USER"
echo "  Date : $(date)"
echo ""
echo "✅ Demo selesai!"
'''

        elif lang == "html":
            return f'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>MOKO Playground</title>
    <style>
        body {{ font-family: monospace; background: #0a0d1a; color: #00ff88; padding: 2em; }}
        h1 {{ color: #00e6ff; }}
        .box {{ border: 1px solid #00e6ff; padding: 1em; border-radius: 8px; }}
        button {{ background: #00e6ff; color: #000; border: none; padding: 8px 16px;
                  border-radius: 4px; cursor: pointer; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>🎮 MOKO Playground</h1>
    <div class="box">
        <p>Instruksi: {instruction_preview}</p>
        <p>Status: <span id="status">Siap</span></p>
        <button onclick="runDemo()">▶ Jalankan Demo</button>
    </div>
    <script>
        function runDemo() {{
            document.getElementById("status").textContent = "Berjalan...";
            let count = 0;
            const timer = setInterval(() => {{
                count++;
                document.getElementById("status").textContent = `Step ${{count}}/5`;
                if (count >= 5) {{
                    clearInterval(timer);
                    document.getElementById("status").textContent = "✅ Selesai!";
                }}
            }}, 300);
        }}
    </script>
</body>
</html>
'''

        return f"# Playground Demo\nprint('Hello from MOKO Playground!')\n"


# ─── PlaygroundManager ─────────────────────────────────────────────────────────

class PlaygroundManager:
    """
    Mengelola sesi-sesi playground aktif.
    Bertanggung jawab untuk cleanup tempdir saat sesi ditutup.
    """

    def __init__(self):
        self._sessions: list[PlaygroundSession] = []
        self._current: Optional[PlaygroundSession] = None

    @property
    def current(self) -> Optional[PlaygroundSession]:
        return self._current

    def new_session(self) -> PlaygroundSession:
        """Buat sesi baru, cleanup sesi sebelumnya jika ada."""
        if self._current and self._current.is_active:
            self._current.cleanup()
        session = PlaygroundSession()
        self._sessions.append(session)
        self._current = session
        return session

    def set_current(self, session: PlaygroundSession):
        """Set sesi aktif (dipanggil saat worker selesai)."""
        self._current = session

    def clear(self):
        """Bersihkan semua sesi playground."""
        for s in self._sessions:
            if s.is_active:
                s.cleanup()
        self._sessions.clear()
        self._current = None

    def save_current(self, dest_folder: str) -> List[str]:
        """Simpan sesi saat ini ke folder permanen."""
        if self._current and self._current.is_active:
            return self._current.save_to(dest_folder)
        return []

    def __del__(self):
        """Pastikan semua tempdir dibersihkan saat object dihapus."""
        try:
            self.clear()
        except Exception:
            pass


# ─── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=== Unit Test: playground_worker.py ===\n")

    # Test 1: PlaygroundSession
    sess = PlaygroundSession()
    assert os.path.isdir(sess.tempdir), "tempdir harus ada"
    assert sess.is_active
    print(f"  ✅ PlaygroundSession: tempdir = {sess.tempdir}")

    # Test 2: Tulis file ke session
    test_file = os.path.join(sess.tempdir, "test.py")
    with open(test_file, "w") as f:
        f.write("print('hello')\n")
    sess.files.append(test_file)
    assert os.path.isfile(test_file)
    print(f"  ✅ File ditulis ke tempdir: {test_file}")

    # Test 3: Save ke folder lain
    dest = tempfile.mkdtemp(prefix="moko_play_save_test_")
    saved = sess.save_to(dest)
    assert len(saved) == 1
    assert os.path.isfile(saved[0])
    print(f"  ✅ save_to: {saved[0]}")
    shutil.rmtree(dest)

    # Test 4: Cleanup
    sess.cleanup()
    assert not os.path.isdir(sess.tempdir), "tempdir harus dihapus setelah cleanup"
    assert not sess.is_active
    print("  ✅ cleanup: tempdir dihapus")

    # Test 5: PlaygroundManager
    mgr = PlaygroundManager()
    s1 = mgr.new_session()
    td1 = s1.tempdir
    assert os.path.isdir(td1)
    s2 = mgr.new_session()  # s1 harus di-cleanup
    assert not os.path.isdir(td1), "s1.tempdir harus dihapus saat new_session()"
    print("  ✅ PlaygroundManager: new_session() cleanup sesi sebelumnya")
    mgr.clear()
    print("  ✅ PlaygroundManager.clear() → OK")

    # Test 6: PlaygroundWorker._detect_language
    # (Tanpa LLM — test deteksi bahasa dan template fallback)
    worker = PlaygroundWorker("buat game python sederhana", language="auto")
    lang = worker._detect_language("buat game python sederhana", "auto")
    assert lang == "python", f"Expected python, got {lang}"
    print(f"  ✅ _detect_language('python'): {lang}")

    lang_js = worker._detect_language("buat script javascript untuk sorting", "auto")
    assert lang_js == "javascript"
    print(f"  ✅ _detect_language('javascript'): {lang_js}")

    lang_bash = worker._detect_language("buat bash script untuk monitoring", "auto")
    assert lang_bash == "bash"
    print(f"  ✅ _detect_language('bash'): {lang_bash}")

    # Test 7: Fallback template
    py_tmpl = worker._fallback_template("python")
    assert "import random" in py_tmpl
    assert "PLAYGROUND" in py_tmpl
    print("  ✅ _fallback_template(python): OK")

    js_tmpl = worker._fallback_template("javascript")
    assert "console.log" in js_tmpl
    print("  ✅ _fallback_template(javascript): OK")

    bash_tmpl = worker._fallback_template("bash")
    assert "#!/bin/bash" in bash_tmpl
    print("  ✅ _fallback_template(bash): OK")

    html_tmpl = worker._fallback_template("html")
    assert "<!DOCTYPE html>" in html_tmpl
    print("  ✅ _fallback_template(html): OK")

    # Test 8: Code extraction
    response_with_fence = "Here is the code:\n```python\nprint('hello')\n```\n"
    extracted = worker._extract_code(response_with_fence, "python")
    assert extracted == "print('hello')", f"Got: {extracted!r}"
    print("  ✅ _extract_code (fenced python): OK")

    response_raw = "import os\nprint(os.getcwd())"
    extracted_raw = worker._extract_code(response_raw, "python")
    assert "import os" in extracted_raw
    print("  ✅ _extract_code (raw import): OK")

    # Test 9: Write to tempdir
    worker2 = PlaygroundWorker("test", language="python")
    worker2.session = PlaygroundSession()
    fpath = worker2._write_to_tempdir("print('test')\n", "python")
    assert fpath.endswith(".py")
    assert os.path.isfile(fpath)
    content = open(fpath).read()
    assert "print('test')" in content
    worker2.session.cleanup()
    print("  ✅ _write_to_tempdir(python): OK, cleanup OK")

    # Test 10: Run command
    cmd_py = worker._build_run_command("/tmp/test.py", "python")
    assert "python3" in cmd_py
    cmd_js = worker._build_run_command("/tmp/test.js", "javascript")
    assert "node" in cmd_js
    cmd_sh = worker._build_run_command("/tmp/test.sh", "bash")
    assert "bash" in cmd_sh
    print("  ✅ _build_run_command: python/javascript/bash OK")

    print("\n🎮 ALL PLAYGROUND TESTS PASSED!")
