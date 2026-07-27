"""
BrainNode — Sistem 2 (Gaya DeepSeek-R1): Otak Penalaran & Perencana
===================================================================
Bertanggung jawab pada tahap berpikir lambat & taktis:
- Melakukan penalaran Chain-of-Thought (CoT) atas instruksi pengguna.
- Menyusun rencana kerja langkah-demi-langkah (execution plan).
- Merancang berkas kode target beserta unit test otomatis untuk verifikasi.
- Menganalisis log kegagalan dan menghasilkan instruksi perbaikan (self-correction).

Node ini murni logika (tanpa PyQt/torch) sehingga dapat diuji secara mandiri.
Jika `llm_generate` disediakan (fungsi model nyata), CoT akan diperkaya oleh LLM;
jika tidak, digunakan penalaran deterministik berbasis template agar tetap andal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from moko_agents.dual_system._bridge import tokenize, CodeKnowledgeBase, retrieve


@dataclass
class ExecutionPlan:
    """Rencana kerja terstruktur yang dihasilkan Sistem 2 untuk Sistem 1."""
    user_prompt: str
    intent: str
    thought: str
    steps: list[str]
    focus_tokens: list[str] = field(default_factory=list)
    target_module: str = "moko_generated_runtime.py"
    test_module: str = "test_moko_generated.py"
    module_code: str = ""
    test_code: str = ""
    knowledge_snippet_id: Optional[str] = None
    knowledge_source: str = ""
    expected_signal: str = "MOKO_DUAL_TEST_PASSED"
    attempt: int = 0
    repair_hint: str = ""


class BrainNode:
    """System 2 Reasoning & Planner (DeepSeek style)."""

    FEATURE_NAME = "moko_sum_of_squares"

    def __init__(
        self,
        knowledge_base=None,
        llm_generate: Optional[Callable[[str], str]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.kb = knowledge_base if knowledge_base is not None else CodeKnowledgeBase()
        self.llm_generate = llm_generate
        self.on_status = on_status

    def _emit(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)

    # ── Anchor-signal untuk retrieval (dipakai bersama Sistem 1) ───────────────
    def focus_tokens(self, user_prompt: str) -> list[str]:
        return sorted(set(tokenize(user_prompt)))

    def _retrieve_knowledge(self, focus: set[str]):
        try:
            # `retrieve` memakai akselerasi native (C++/Rust) bila tersedia,
            # dengan hasil identik terhadap CodeKnowledgeBase.retrieve.
            hits = retrieve(self.kb, focus, limit=1)
        except Exception:
            hits = []
        return hits[0] if hits else None

    # ── Chain-of-Thought reasoning ─────────────────────────────────────────────
    def _chain_of_thought(self, user_prompt: str, snippet, attempt: int, repair_hint: str) -> str:
        if self.llm_generate:
            try:
                sys_hint = (
                    "Kamu adalah Sistem 2 (Otak penalaran gaya DeepSeek-R1). "
                    "Uraikan penalaran Chain-of-Thought singkat untuk merancang solusi kode."
                )
                out = self.llm_generate(f"{sys_hint}\n\nInstruksi: {user_prompt}")
                if out and out.strip():
                    return out.strip()
            except Exception:
                pass
        # Penalaran deterministik (fallback andal).
        lines = [
            f"Menganalisis instruksi: {user_prompt!r}.",
            "Langkah penalaran (CoT):",
            "1. Identifikasi intent utama dan variabel yang terlibat.",
            "2. Petakan pengetahuan relevan dari Anchor-RAG untuk mendukung implementasi.",
            f"3. Rancang fungsi murni `{self.FEATURE_NAME}` yang deterministik & mudah diuji.",
            "4. Susun unit test yang memverifikasi keluaran numerik yang diketahui.",
        ]
        if snippet is not None:
            lines.append(
                f"5. Pengetahuan pendukung: {snippet.snippet_id} (domain {snippet.domain})."
            )
        if attempt > 0:
            lines.append(
                f"6. Ini iterasi koreksi-diri ke-{attempt}. Terapkan perbaikan: {repair_hint}"
            )
        return "\n".join(lines)

    # ── Sintesis kode fitur + unit test ────────────────────────────────────────
    def _feature_source(self, buggy: bool) -> str:
        """Kode fitur inti. Jika `buggy`, sengaja menaruh galat logika untuk
        mendemonstrasikan loop verifikasi & koreksi-diri Sistem 2 Guard."""
        operator = "+ nilai" if buggy else "+ nilai * nilai"
        return (
            f"def {self.FEATURE_NAME}(daftar_nilai):\n"
            f"    \"\"\"Menghitung jumlah kuadrat dari daftar bilangan.\"\"\"\n"
            f"    total = 0\n"
            f"    for nilai in daftar_nilai:\n"
            f"        total = total {operator}\n"
            f"    return total\n"
        )

    def _build_module_code(self, snippet, buggy: bool) -> str:
        header = "# BERKAS DIHASILKAN OTOMATIS OLEH MOKO DUAL-SYSTEM (Sistem 1: Executor)\n"
        parts = [header]
        imports = []
        knowledge_block = ""
        if snippet is not None:
            imports = list(getattr(snippet, "requires_imports", ()) or ())
            knowledge_block = (
                "\n# --- Pengetahuan pendukung dari Anchor-RAG "
                f"({snippet.snippet_id}) ---\n{snippet.code}\n"
            )
        if imports:
            parts.append("\n".join(imports) + "\n")
        parts.append("\n" + self._feature_source(buggy))
        if knowledge_block:
            parts.append(knowledge_block)
        return "".join(parts)

    def _build_test_code(self, plan_module: str, expected_signal: str) -> str:
        module_name = plan_module[:-3] if plan_module.endswith(".py") else plan_module
        return (
            f"from {module_name} import {self.FEATURE_NAME}\n\n\n"
            f"def test_{self.FEATURE_NAME}():\n"
            f"    # sum of squares [1, 2, 3] = 1 + 4 + 9 = 14\n"
            f"    hasil = {self.FEATURE_NAME}([1, 2, 3])\n"
            f"    assert hasil == 14, f'diharapkan 14, diperoleh {{hasil}}'\n"
            f"    print('{expected_signal}')\n\n\n"
            f"if __name__ == '__main__':\n"
            f"    test_{self.FEATURE_NAME}()\n"
        )

    def _classify_intent(self, focus: set[str]) -> str:
        coding_markers = {
            "buat", "buatkan", "tulis", "kode", "code", "implementasi",
            "implementasikan", "fungsi", "program", "perbaiki", "fix", "bug",
        }
        if focus & coding_markers:
            return "agentic_code_task"
        return "general_reasoning_task"

    def reason_and_plan(
        self,
        user_prompt: str,
        *,
        attempt: int = 0,
        force_bug: bool = False,
        repair_hint: str = "",
    ) -> ExecutionPlan:
        """Hasilkan `ExecutionPlan` lengkap: CoT, langkah kerja, kode & unit test.

        Pada iterasi pertama dengan `force_bug=True`, kode sengaja mengandung galat
        agar Sistem 2 Guard dapat mendeteksinya. Pada iterasi koreksi (`attempt>=1`),
        kode selalu benar.
        """
        self._emit("🧠 Sistem 2 (Brain): menyusun penalaran & rencana kerja...")
        focus = set(self.focus_tokens(user_prompt))
        snippet = self._retrieve_knowledge(focus)
        intent = self._classify_intent(focus)

        buggy = bool(force_bug) and attempt == 0
        thought = self._chain_of_thought(user_prompt, snippet, attempt, repair_hint)

        steps = [
            "Cari berkas/pengetahuan target melalui Anchor-RAG.",
            "Tulis modul fitur ke workspace secara presisi.",
            "Tulis berkas unit test otomatis.",
            "Jalankan unit test di subproses terminal lokal.",
            "Serahkan log eksekusi ke Sistem 2 Guard untuk verifikasi.",
        ]

        plan = ExecutionPlan(
            user_prompt=user_prompt,
            intent=intent,
            thought=thought,
            steps=steps,
            focus_tokens=sorted(focus),
            knowledge_snippet_id=getattr(snippet, "snippet_id", None) if snippet else None,
            knowledge_source=getattr(snippet, "source", "") if snippet else "",
            attempt=attempt,
            repair_hint=repair_hint,
        )
        plan.module_code = self._build_module_code(snippet, buggy)
        plan.test_code = self._build_test_code(plan.target_module, plan.expected_signal)
        self._emit(
            f"🧠 Sistem 2 (Brain): rencana siap (intent={intent}, "
            f"snippet={plan.knowledge_snippet_id})."
        )
        return plan

    # ── Analisis kegagalan untuk loop koreksi-diri ─────────────────────────────
    def analyze_failure(self, plan: ExecutionPlan, error_log: str) -> str:
        """Menghasilkan instruksi perbaikan ringkas dari log galat."""
        self._emit("🧠 Sistem 2 (Brain): menganalisis log kegagalan untuk koreksi-diri...")
        if self.llm_generate:
            try:
                out = self.llm_generate(
                    "Analisis log error unit test berikut dan berikan instruksi "
                    f"perbaikan singkat:\n{error_log}"
                )
                if out and out.strip():
                    return out.strip()
            except Exception:
                pass
        log_lower = (error_log or "").lower()
        if "assert" in log_lower or "diharapkan" in log_lower:
            return (
                "Unit test gagal pada assertion: logika perhitungan salah. "
                f"Perbaiki `{self.FEATURE_NAME}` agar menjumlahkan kuadrat nilai "
                "(nilai * nilai), bukan nilai mentah."
            )
        if "importerror" in log_lower or "modulenotfound" in log_lower:
            return "Perbaiki nama/lokasi modul agar dapat diimpor oleh unit test."
        if "syntaxerror" in log_lower:
            return "Perbaiki galat sintaksis pada modul yang dihasilkan."
        return "Perbaiki implementasi berdasarkan traceback dan jalankan ulang unit test."
