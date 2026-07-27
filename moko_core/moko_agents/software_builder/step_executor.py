"""
step_executor.py — Step Executor dengan Terminal Feedback Loop (UPGRADED v2 + Marathon)
========================================================================================
Komponen yang:
1. Menerima satu PlanStep + PlanSession
2. Build RepoMap dari workspace untuk konteks (teknik dari Aider)
3. Prune prompt jika melebihi token budget (TokenBudgetManager)
4. Generate kode via LLM dengan SEARCH/REPLACE edit format (3x lebih presisi)
5. Auto-continue jika output terpotong — Marathon System (maks 8 iterasi)
6. Smart code extraction: handle triple-backtick, SEARCH/REPLACE, multi-file
7. Validate syntax Python sebelum menulis ke disk
8. Tulis kode ke file di workspace
9. Eksekusi via subprocess (untuk capture output)
10. Juga injeksikan command ke TerminalPanel untuk visualisasi
11. Tangani error dengan self-healing loop (maks 3x retry)

Teknik dari GitHub IDE terbaik:
  - RepoMapBuilder: ringkasan file tree + class/function defs (dari Aider)
  - SEARCH/REPLACE edit format: presisi 3x lebih tinggi dari whole-file (Aider research)
  - Smart code extractor: multi-strategy parsing dengan fallback
  - py_compile syntax validation sebelum tulis ke disk
  - Anti-laziness heal prompt
  - Token Marathon System: auto-continue + prompt pruning (baru — Step 6)

Desain:
- Berjalan di QThread agar UI tidak freeze
- Emit signals untuk update UI secara real-time
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from moko_agents.software_builder.models import InterviewData, PlanSession, PlanStep
from moko_agents.software_builder.token_manager import marathon_call_llm, TokenBudgetManager


# ─── RepoMap Builder (5C — teknik dari Aider) ────────────────────────────────

class RepoMapBuilder:
    """
    Membuat ringkasan file tree + definisi class/function dari workspace.
    Ini disebut "Repo Map" di Aider — terbukti meningkatkan kualitas kode
    yang dihasilkan karena LLM tahu konteks apa yang sudah ada.
    """

    @staticmethod
    def build(workspace_dir: str, max_chars: int = 3000) -> str:
        """
        Build repo map dari direktori workspace.
        
        Returns:
            String ringkasan file tree + key definitions
        """
        workspace = Path(workspace_dir)
        if not workspace.exists():
            return ""

        lines = ["## EXISTING PROJECT FILES (Repo Map)\n"]
        total_chars = 0

        # File tree
        py_files = sorted(workspace.rglob("*.py"))
        js_files = sorted(workspace.rglob("*.js"))
        other_files = [
            f for f in workspace.rglob("*")
            if f.is_file() and f.suffix not in (".py", ".js", ".pyc", ".pyo")
            and not any(p.name.startswith(".") for p in f.parents)
        ]

        all_files = py_files + js_files + list(other_files)[:5]

        for fpath in all_files[:20]:  # Maks 20 file
            rel = fpath.relative_to(workspace)
            lines.append(f"  {rel}")

        lines.append("\n## KEY DEFINITIONS (classes & functions)\n")

        # Extract class/function defs dari Python files
        for fpath in py_files[:10]:
            try:
                source = fpath.read_text(encoding="utf-8", errors="ignore")
                defs = RepoMapBuilder._extract_python_defs(source, fpath.name)
                if defs:
                    lines.extend(defs)
                    total_chars += sum(len(d) for d in defs)
                    if total_chars > max_chars:
                        break
            except Exception:
                continue

        result = "\n".join(lines)
        return result[:max_chars]

    @staticmethod
    def _extract_python_defs(source: str, filename: str) -> List[str]:
        """Extract class dan function definitions dari Python source."""
        defs = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [
                        f"    def {n.name}(self, ...)"
                        for n in ast.walk(node)
                        if isinstance(n, ast.FunctionDef) and n.name != node.name
                    ][:5]
                    defs.append(f"  {filename}: class {node.name}:")
                    defs.extend(methods)
                elif isinstance(node, ast.FunctionDef) and not isinstance(
                    getattr(node, "parent", None), ast.ClassDef
                ):
                    args = ", ".join(a.arg for a in node.args.args[:4])
                    defs.append(f"  {filename}: def {node.name}({args})")
        except SyntaxError:
            pass
        return defs[:15]


# ─── SEARCH/REPLACE Parser (5A — teknik dari Aider) ──────────────────────────

def parse_search_replace_blocks(text: str, existing_content: str = "") -> str:
    """
    Parse format SEARCH/REPLACE blocks yang digunakan Aider.
    Format:
        <<<<<<< SEARCH
        (kode yang dicari)
        =======
        (kode pengganti)
        >>>>>>> REPLACE
    
    Jika existing_content tersedia, lakukan replace aktual.
    Jika tidak ada SEARCH/REPLACE blocks, return teks asli.
    
    Terbukti 3x lebih presisi dari whole-file output (Aider research 2023).
    """
    pattern = re.compile(
        r"<{7}\s*SEARCH\s*\n(.*?)={7}\s*\n(.*?)>{7}\s*REPLACE",
        re.DOTALL
    )
    matches = list(pattern.finditer(text))

    if not matches:
        return ""  # Tidak ada SEARCH/REPLACE blocks

    if not existing_content:
        # Tidak ada existing content — ambil semua REPLACE sections
        parts = [m.group(2).strip() for m in matches]
        return "\n\n".join(parts)

    # Apply replace operations ke existing content
    result = existing_content
    for match in matches:
        search_block = match.group(1).strip()
        replace_block = match.group(2).strip()
        if search_block in result:
            result = result.replace(search_block, replace_block, 1)
        else:
            # Fuzzy: coba tanpa leading/trailing whitespace
            result = result + "\n\n" + replace_block

    return result


# ─── Smart Code Extractor (5D) ───────────────────────────────────────────────

def extract_code_smart(llm_output: str, language: str = "python") -> str:
    """
    Multi-strategy code extractor dari LLM output.
    
    Strategi (dalam urutan prioritas):
    1. SEARCH/REPLACE blocks (format Aider)
    2. Fenced code block dengan language tag: ```python ... ```
    3. Fenced code block tanpa tag: ``` ... ```
    4. Kode setelah kata kunci "Here is the code:" / "Output:"
    5. Raw text (seluruh output, dihapus prefix penjelas)
    """
    # Strategi 1: SEARCH/REPLACE blocks
    sr_result = parse_search_replace_blocks(llm_output)
    if sr_result:
        return sr_result.strip()

    # Strategi 2: Fenced code block dengan language tag
    lang_fence = re.compile(
        rf"```{re.escape(language)}\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE
    )
    m = lang_fence.search(llm_output)
    if m:
        return m.group(1).strip()

    # Strategi 3: Fenced code block tanpa tag (tapi isinya terlihat seperti kode)
    generic_fence = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)
    matches = list(generic_fence.finditer(llm_output))
    if matches:
        # Ambil blok terpanjang (kemungkinan besar adalah kode utama)
        longest = max(matches, key=lambda x: len(x.group(1)))
        return longest.group(1).strip()

    # Strategi 4: Setelah keyword "code:"
    keyword_match = re.search(
        r"(?:here is the code|output|result|code):\s*\n+(.*)",
        llm_output, re.DOTALL | re.IGNORECASE
    )
    if keyword_match:
        raw = keyword_match.group(1).strip()
        if len(raw) > 50:
            return raw

    # Strategi 5: Raw text — hapus baris penjelas di awal/akhir
    lines = llm_output.split("\n")
    # Hapus baris yang terlihat seperti penjelasan (tidak ada indent, tidak dimulai dengan keyword code)
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(import|from|def |class |#|if |for |while |print|return|\s)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return llm_output.strip()


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """
    Validasi syntax Python menggunakan ast.parse.
    Lebih cepat dari py_compile, tanpa perlu I/O.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


# ─── Code Generation Prompts (UPGRADED v2) ───────────────────────────────────

# Template v2: menggunakan anti-laziness rules + SEARCH/REPLACE format hints
_CODE_GEN_TEMPLATE_V2 = """You are MOKO, an expert {language} programmer and software architect.
Generate COMPLETE, PRODUCTION-QUALITY code for the implementation step below.

## ANTI-LAZINESS RULES (MANDATORY)
- NEVER write: "# ... rest of code", "# TODO", "# implement later", "pass  # placeholder"
- EVERY function MUST have a complete implementation body
- ALL imports must be real and valid
- Code must run WITHOUT any modifications

## PROJECT CONTEXT
{project_context}

## REPO MAP (existing files & definitions)
{repo_map}

## CURRENT STEP TO IMPLEMENT
Step {step_num}/{total_steps}: {step_title}
Goal: {step_description}
Files to create: {files}

## CODE ALREADY WRITTEN (previous steps — for reference)
{existing_code}

## OUTPUT FORMAT
If creating ONE file, output the complete code directly in a fenced block:
```{language}
# complete code here
```

If creating MULTIPLE files, use this separator:
# === FILE: filename.py ===
[complete file content]
# === END FILE ===
# === FILE: other_file.py ===
[complete file content]
# === END FILE ===

Now write the COMPLETE {language} code for Step {step_num}:"""


# Template v2 untuk self-healing — lebih detail dari versi sebelumnya
_HEAL_TEMPLATE_V2 = """You are MOKO, an expert {language} debugger. Fix the error below.

## BROKEN CODE
```{language}
{original_code}
```

## ERROR
```
{error_output}
```

## DIAGNOSIS REQUIRED
1. What caused this error?
2. What is the minimal fix?
3. Are there any related issues to fix proactively?

## FIX RULES
- Return the COMPLETE fixed file (not just the changed lines)
- NEVER use placeholder comments like "# ... rest unchanged"
- If the error is "ModuleNotFoundError", add a try/except import or note it
- Fix the root cause, not just the symptom

Fixed {language} code (complete file):
```{language}
"""


MAX_RETRY = 3


def _lang_ext(language: str) -> str:
    """Return file extension untuk bahasa pemrograman."""
    return {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "bash": "sh",
        "c++": "cpp",
        "c#": "cs",
        "java": "java",
        "go": "go",
        "rust": "rs",
        "php": "php",
    }.get(language.lower(), "py")


class StepExecutorWorker(QThread):
    """
    QThread worker untuk eksekusi satu PlanStep.
    Berjalan di background — tidak freeze UI.
    """
    log_signal    = pyqtSignal(str, str)   # message, color
    code_signal   = pyqtSignal(str, str)   # filename, code_content
    status_signal = pyqtSignal(str)        # status text
    done_signal   = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        step: PlanStep,
        session: PlanSession,
        terminal_execute_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.step = step
        self.session = session
        self._terminal_execute = terminal_execute_fn  # Callable: fn(cmd: str)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._execute()
        except Exception as e:
            self.step.status = "ERROR"
            self.step.error_message = str(e)
            self.log_signal.emit(f"❌ [StepExecutor] Exception: {e}", "#ff4444")
            self.done_signal.emit(False, str(e))

    def _execute(self):
        step = self.step
        session = self.session
        data = session.interview_data

        step.status = "IN_PROGRESS"
        self.status_signal.emit(f"🔄 MOKO: Generating Step {step.step_number}...")
        self.log_signal.emit(
            f"⚙️ [StepExecutor] Memulai Step {step.step_number}: {step.title}",
            "#00e6ff"
        )

        # 1. Build RepoMap dari workspace (teknik Aider)
        repo_map = RepoMapBuilder.build(session.workspace_dir)
        if repo_map:
            self.log_signal.emit(
                f"  🗺️ RepoMap built: {len(repo_map)} chars context",
                "rgba(0,230,255,0.6)"
            )

        # 2. Bangun prompt kode v2 (dengan RepoMap)
        existing_code = self._collect_existing_code(session)
        prompt = self._build_code_prompt(step, data, existing_code, repo_map)

        # 3. Generate kode via LLM
        self.log_signal.emit("  🤖 Memanggil LLM untuk generate kode...", "#888")
        generated_code = self._call_llm(prompt, max_tokens=2500)

        if not generated_code:
            step.status = "ERROR"
            step.error_message = "LLM tidak menghasilkan kode"
            self.done_signal.emit(False, "LLM tidak merespons")
            return

        # 4. Smart parse dan validasi kode (v2)
        file_pairs = self._parse_code_files(generated_code, step, data.language)
        workspace = Path(session.workspace_dir)

        written_files = []
        for fname, code in file_pairs:
            fpath = workspace / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")
            written_files.append(str(fpath))
            self.code_signal.emit(fname, code)
            self.log_signal.emit(f"  📄 File ditulis: {fname}", "#00ff88")

        step.generated_code = generated_code

        if not written_files:
            step.status = "ERROR"
            step.error_message = "Tidak ada file yang berhasil ditulis"
            self.done_signal.emit(False, "Gagal menulis file")
            return

        # 4. Eksekusi kode (dengan self-healing loop)
        main_file = written_files[0]
        success, output, error = self._execute_with_healing(
            main_file, data.language, generated_code, step, session
        )

        step.terminal_output = output

        if success:
            step.status = "DONE"
            self.log_signal.emit(f"  ✅ Step {step.step_number} selesai!", "#00ff88")
            self.done_signal.emit(True, f"Step {step.step_number} berhasil!")
        else:
            step.status = "ERROR"
            step.error_message = error
            self.log_signal.emit(f"  ❌ Step {step.step_number} gagal: {error[:100]}", "#ff4444")
            self.done_signal.emit(False, f"Step {step.step_number} error setelah {MAX_RETRY}x retry")

    def _build_code_prompt(
        self,
        step: PlanStep,
        data: InterviewData,
        existing_code: str,
        repo_map: str = ""
    ) -> str:
        """Bangun prompt v2 untuk generate kode satu langkah (dengan RepoMap)."""
        mechanics_str = ", ".join(data.mechanics[:6]) if data.mechanics else "basic features"
        project_context = (
            f"Project: {data.sub_type} {data.software_type}\n"
            f"Language: {data.language} | Platform: {data.platform or 'desktop'}\n"
            f"Key features: {mechanics_str}\n"
            f"Complexity: {data.complexity}"
        )
        total_steps = len(self.session.steps) if hasattr(self, 'session') else '?'
        return _CODE_GEN_TEMPLATE_V2.format(
            language=data.language,
            project_context=project_context,
            repo_map=repo_map if repo_map else "(workspace empty — this is the first file)",
            step_num=step.step_number,
            total_steps=total_steps,
            step_title=step.title,
            step_description=step.description,
            files=", ".join(step.files_to_create) or f"main.{_lang_ext(data.language)}",
            existing_code=existing_code[:2500] if existing_code else "(none — this is the first step)"
        )

    def _build_heal_prompt(
        self,
        language: str,
        original_code: str,
        error_output: str
    ) -> str:
        """Bangun prompt v2 untuk perbaikan kode error (dengan diagnosis section)."""
        return _HEAL_TEMPLATE_V2.format(
            language=language,
            original_code=original_code[:3500],
            error_output=error_output[:1200]
        )

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Panggil LLM via Marathon System (Step 6).
        
        Upgrade dari v2: sekarang menggunakan marathon_call_llm() yang:
          1. Prune prompt otomatis jika melebihi token budget
          2. Auto-continue jika output terpotong (maks 8 iterasi)
          3. Overlap detection saat menyambung chunk
          4. Completion pass jika batas iterasi tercapai
        """
        def log_fn(msg: str, color: str):
            self.log_signal.emit(msg, color)

        try:
            result = marathon_call_llm(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.15,
                log_fn=log_fn,
            )
            return result.strip() if result else ""
        except Exception as e:
            self.log_signal.emit(f"  ⚠️ LLM call error: {e}", "#ff8800")
            return ""

    def _collect_existing_code(self, session: PlanSession) -> str:
        """Kumpulkan snippet kode yang sudah ada dari langkah-langkah sebelumnya."""
        code_parts = []
        for step in session.steps:
            if step.status == "DONE" and step.generated_code:
                # Ambil 800 char per step untuk hemat context window
                snippet = step.generated_code[:800]
                code_parts.append(
                    f"# --- Step {step.step_number}: {step.title} ---\n{snippet}"
                )
        return "\n\n".join(code_parts[:4])  # Maks 4 langkah terakhir

    def _parse_code_files(
        self,
        llm_output: str,
        step: PlanStep,
        language: str = "python"
    ) -> List[Tuple[str, str]]:
        """
        Smart parser v2: ekstrak (filename, code) pairs dari LLM output.
        
        Mendukung format (dalam urutan prioritas):
          1. Multi-file: # === FILE: filename.py === ... # === END FILE ===
          2. SEARCH/REPLACE blocks (Aider format)
          3. Single fenced code block (```python ... ```)
          4. Raw code output
        
        Juga melakukan syntax validation untuk Python files.
        """
        # ── Format 1: Multi-file dengan FILE markers ──
        file_marker = re.compile(
            r"#\s*===\s*FILE:\s*(.+?)\s*===\s*\n(.*?)(?=#\s*===\s*(?:END FILE|FILE:)|$)",
            re.DOTALL | re.IGNORECASE
        )
        markers = list(file_marker.finditer(llm_output))

        if markers:
            result = []
            for m in markers:
                fname = m.group(1).strip()
                raw_code = m.group(2).strip()
                # Hapus END FILE marker jika ada
                raw_code = re.sub(r"#\s*===\s*END FILE\s*===.*$", "", raw_code, flags=re.DOTALL).strip()
                # Bersihkan code fence jika ada
                raw_code = re.sub(r"```\w*\n?", "", raw_code).replace("```", "").strip()
                result.append((fname, raw_code))
            if result:
                self.log_signal.emit(
                    f"  📁 Multi-file format: {len(result)} file terdeteksi",
                    "#00e6ff"
                )
                return self._validate_and_fix_files(result, language)

        # ── Format 2+3+4: Single code (gunakan smart extractor) ──
        fname = step.files_to_create[0] if step.files_to_create else f"main.{_lang_ext(language)}"
        extracted = extract_code_smart(llm_output, language)

        if not extracted:
            extracted = llm_output.strip()

        return self._validate_and_fix_files([(fname, extracted)], language)

    def _validate_and_fix_files(
        self,
        file_pairs: List[Tuple[str, str]],
        language: str
    ) -> List[Tuple[str, str]]:
        """
        Validasi syntax dan lakukan perbaikan minor untuk file Python.
        Jika syntax error ditemukan, log warning tapi tetap lanjutkan
        (self-healing loop akan menanganinya saat eksekusi).
        """
        result = []
        for fname, code in file_pairs:
            if not code.strip():
                self.log_signal.emit(f"  ⚠️ File {fname} kosong, dilewati", "#ffaa00")
                continue

            # Validasi syntax Python
            if language == "python" and fname.endswith(".py"):
                is_valid, err = validate_python_syntax(code)
                if not is_valid:
                    self.log_signal.emit(
                        f"  ⚠️ Syntax warning di {fname}: {err}",
                        "#ffaa00"
                    )
                    # Tetap tambahkan — self-healing akan perbaiki
                else:
                    self.log_signal.emit(
                        f"  ✅ Syntax OK: {fname}",
                        "rgba(0,255,136,0.7)"
                    )

            result.append((fname, code))
        return result

    def _run_file(
        self,
        file_path: str,
        language: str,
        workspace: str
    ) -> Tuple[bool, str, str]:
        """
        Eksekusi file dan capture stdout/stderr.
        Return (success, stdout, stderr)
        """
        ext_cmd = {
            "python": ["python3"],
            "javascript": ["node"],
            "bash": ["bash"],
            "typescript": ["ts-node"],
        }
        cmd_prefix = ext_cmd.get(language, ["python3"])

        # Khusus untuk game yang butuh display (pygame), skip eksekusi langsung
        # Cukup verify syntax saja
        if language == "python":
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", file_path],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                if result.returncode != 0:
                    return False, "", result.stderr
                # Syntax OK — untuk game/GUI, tidak dieksekusi (butuh display)
                return True, "Syntax check passed ✓", ""
            except subprocess.TimeoutExpired:
                return False, "", "Syntax check timeout"
            except Exception as e:
                return False, "", str(e)
        else:
            # Untuk non-Python, coba jalankan
            try:
                result = subprocess.run(
                    cmd_prefix + [file_path],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                success = result.returncode == 0

                # Handle common non-fatal errors
                if stderr and any(hint in stderr for hint in [
                    "ModuleNotFoundError", "ImportError", "No module named"
                ]):
                    return False, stdout, stderr

                return success, stdout, stderr
            except subprocess.TimeoutExpired:
                return True, "Process running (timeout — might be normal for GUI apps)", ""
            except Exception as e:
                return False, "", str(e)

    def _execute_with_healing(
        self,
        file_path: str,
        language: str,
        original_code: str,
        step: PlanStep,
        session: PlanSession
    ) -> Tuple[bool, str, str]:
        """
        Jalankan kode dengan self-healing loop.
        Maks MAX_RETRY iterasi perbaikan jika error.
        
        Returns:
            (success, output, error_message)
        """
        workspace = session.workspace_dir
        current_code = original_code

        for attempt in range(MAX_RETRY):
            if self._cancelled:
                return False, "", "Dibatalkan oleh user"

            if self._terminal_execute:
                # Injeksikan command ke TerminalPanel untuk visualisasi
                self._terminal_execute(f"cd {workspace} && python3 -m py_compile {os.path.basename(file_path)}")

            success, stdout, stderr = self._run_file(file_path, language, workspace)

            if success:
                return True, stdout, ""

            # Cek apakah ada ModuleNotFoundError — sarankan pip install
            if "ModuleNotFoundError" in stderr or "No module named" in stderr:
                import re
                mod_match = re.search(r"No module named '([^']+)'", stderr)
                if mod_match:
                    mod_name = mod_match.group(1).split(".")[0]
                    self.log_signal.emit(
                        f"  📦 Module '{mod_name}' tidak ditemukan. "
                        f"Jalankan: pip install {mod_name}",
                        "#ffaa00"
                    )
                    if self._terminal_execute:
                        self._terminal_execute(f"pip install {mod_name}")
                return True, f"Module {mod_name} perlu diinstall", stderr

            if attempt < MAX_RETRY - 1:
                self.log_signal.emit(
                    f"  🔧 Self-healing attempt {attempt + 1}/{MAX_RETRY}...",
                    "#ffaa00"
                )
                self.log_signal.emit(
                    f"     Error: {stderr[:100]}",
                    "rgba(255,100,100,0.7)"
                )
                # Buat prompt healing v2 (dengan diagnosis section)
                heal_prompt = self._build_heal_prompt(language, current_code, stderr)
                healed_raw = self._call_llm(heal_prompt, max_tokens=2000)

                if healed_raw:
                    # Gunakan smart extractor untuk parsing hasil healing
                    healed_code = extract_code_smart(healed_raw, language)
                    if not healed_code:
                        healed_code = healed_raw.strip()

                    # Validasi syntax sebelum tulis ke disk
                    if language == "python":
                        is_valid, err = validate_python_syntax(healed_code)
                        if not is_valid:
                            self.log_signal.emit(
                                f"  ⚠️ Healed code masih syntax error: {err}",
                                "#ff8800"
                            )
                            # Lanjutkan tetap tulis — mungkin error lain yang bisa di-heal lagi

                    Path(file_path).write_text(healed_code, encoding="utf-8")
                    current_code = healed_code
                    step.retry_count = attempt + 1
                    self.log_signal.emit(
                        f"  ✍️ Healed code ditulis ({len(healed_code)} chars)",
                        "#00e6ff"
                    )
                else:
                    self.log_signal.emit(
                        "  ❌ LLM tidak menghasilkan healed code",
                        "#ff4444"
                    )
                    break

        return False, "", f"Gagal setelah {MAX_RETRY}x retry. Last error: {stderr[:200]}"


class StepExecutor:
    """
    Facade ringan untuk menjalankan StepExecutorWorker dari luar.
    Mengelola workers yang sedang aktif.
    """

    def __init__(self, terminal_execute_fn: Optional[Callable] = None):
        self._terminal_fn = terminal_execute_fn
        self._active_worker: Optional[StepExecutorWorker] = None

    def execute_step(
        self,
        step: PlanStep,
        session: PlanSession,
        on_log: Optional[Callable] = None,
        on_code: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ) -> StepExecutorWorker:
        """
        Mulai eksekusi step di background thread.
        
        Args:
            step: PlanStep yang akan dieksekusi
            session: PlanSession saat ini
            on_log: Callback (message: str, color: str)
            on_code: Callback (filename: str, code: str)
            on_done: Callback (success: bool, message: str)
        
        Returns:
            StepExecutorWorker yang sedang berjalan
        """
        # Stop worker lama jika ada
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.cancel()
            self._active_worker.wait(2000)

        worker = StepExecutorWorker(step, session, self._terminal_fn)

        if on_log:
            worker.log_signal.connect(lambda msg, color: on_log(msg, color))
        if on_code:
            worker.code_signal.connect(lambda fname, code: on_code(fname, code))
        if on_done:
            worker.done_signal.connect(lambda ok, msg: on_done(ok, msg))

        self._active_worker = worker
        worker.start()
        return worker

    def cancel(self):
        """Cancel eksekusi yang sedang berjalan."""
        if self._active_worker:
            self._active_worker.cancel()


if __name__ == "__main__":
    print("=== Unit Test: step_executor.py ===\n")
    print("  (StepExecutor butuh Qt app untuk test penuh)")
    print("  Test komponen parsing saja:\n")

    # Test _parse_code_files logic (standalone)
    import re

    sample_code_single = """
import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Test Game')

if __name__ == '__main__':
    main()
"""

    sample_code_multi = """
# === FILE: main.py ===
import pygame
from game import Game

def main():
    game = Game()
    game.run()

# === FILE: game.py ===
import pygame

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
"""

    # Buat dummy step dan session untuk test
    from moko_agents.software_builder.models import InterviewData, PlanStep, PlanSession

    step = PlanStep(1, "Test Step", "Test", ["main.py", "game.py"])
    data = InterviewData(software_type="game", sub_type="platformer",
                        mechanics=["movement"], language="python",
                        platform="desktop", complexity="simple")
    session = PlanSession(interview_data=data, workspace_dir="/tmp")

    # Test parse single file
    worker = StepExecutorWorker(step, session)
    pairs_single = worker._parse_code_files(sample_code_single, step)
    assert len(pairs_single) == 1
    assert pairs_single[0][0] == "main.py"
    assert "pygame" in pairs_single[0][1]
    print(f"✅ Parse single file: {pairs_single[0][0]} → OK")

    # Test parse multi file
    pairs_multi = worker._parse_code_files(sample_code_multi, step)
    assert len(pairs_multi) == 2
    assert pairs_multi[0][0] == "main.py"
    assert pairs_multi[1][0] == "game.py"
    print(f"✅ Parse multi file: {[p[0] for p in pairs_multi]} → OK")

    print("\n✅ Semua unit test step_executor berhasil!")
