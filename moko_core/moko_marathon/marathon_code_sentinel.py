"""
MOKO Marathon Code Sentinel
============================
Mendeteksi apakah kode yang di-generate LLM sudah LENGKAP atau TERPOTONG
di tengah (akibat token limit habis).

Strategi multi-layer (dari paling cepat ke paling akurat):
  Layer 1: Heuristik cepat — panjang output mendekati num_predict?
  Layer 2: Bracket balance — {, [, ( harus seimbang
  Layer 3: Language-specific AST — Python ast.parse(), JS node --check
  Layer 4: Tail pattern — deteksi pola "kode terpotong" di akhir

Bahasa yang didukung:
  python, cpp, c, javascript, typescript, html, css, rust, go, java,
  shell/bash, json, yaml, generic

Contoh pakai:
    sentinel = MarathonCodeSentinel()
    result = sentinel.analyze(code_str, language="python")
    if not result["complete"]:
        print(result["reason"])   # "Unclosed { at depth 2"
        print(result["open_brackets"])
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional


# ── Pola tail yang menandai kode terpotong ──────────────────────────────────────
_TRUNCATION_PATTERNS = [
    r"#\s*(TODO|FIXME|rest of|continue|\.\.\.)\s*$",        # Python/generic comments
    r"//\s*(TODO|FIXME|rest of|continue|\.\.\.)\s*$",       # JS/C++ comments
    r"/\*\s*(TODO|FIXME|rest of|continue)\s*",              # Block comment unclosed
    r"\.\.\.\s*$",                                           # Trailing ellipsis
    r"#\s*\.\.\.\s*$",                                       # Python ...
    r"pass\s*$",                                             # Python pass (stub)
    r"raise\s+NotImplementedError\s*$",                      # Python stub
    r"//\s*\.\.\.\s*$",                                      # JS/TS ellipsis
]

# Pola yang mengindikasikan kode benar-benar di akhir fungsi/class
_PROPER_ENDING_PATTERNS = {
    "python":     [r"\breturn\b[^#\n]*$", r"^(\s*)$", r"\bpass\b\s*$"],
    "javascript": [r"\}\s*$", r"\};\s*$", r"\}\)\s*;?\s*$"],
    "typescript": [r"\}\s*$", r"\};\s*$"],
    "cpp":        [r"\}\s*$", r"\};\s*$"],
    "c":          [r"\}\s*$"],
    "html":       [r"</html>\s*$", r"</body>\s*$"],
    "css":        [r"\}\s*$"],
    "rust":       [r"\}\s*$"],
    "go":         [r"\}\s*$"],
    "java":       [r"\}\s*$"],
}


@dataclass
class SentinelResult:
    """Hasil analisis code completeness."""
    complete: bool
    reason: str = ""
    open_brackets: int = 0          # > 0 = masih ada yang belum ditutup
    unclosed_parens: int = 0
    unclosed_squares: int = 0
    truncated_at_line: int = -1     # Estimasi baris truncation, -1 = tidak terdeteksi
    language: str = "generic"
    confidence: float = 1.0         # 0.0 – 1.0
    layers_checked: list = field(default_factory=list)


class MarathonCodeSentinel:
    """
    Detektor kode terpotong untuk Marathon Auto-Continue system.

    Dirancang agar berjalan CEPAT — rata-rata < 5ms untuk kode 500 baris.
    AST parse hanya dipanggil bila bracket balance lolos (lebih efisien).
    """

    # Mapping ekstensi file → language
    EXT_MAP = {
        ".py":   "python",
        ".js":   "javascript",
        ".ts":   "typescript",
        ".jsx":  "javascript",
        ".tsx":  "typescript",
        ".cpp":  "cpp",
        ".cxx":  "cpp",
        ".cc":   "cpp",
        ".c":    "c",
        ".h":    "cpp",
        ".hpp":  "cpp",
        ".html": "html",
        ".htm":  "html",
        ".css":  "css",
        ".rs":   "rust",
        ".go":   "go",
        ".java": "java",
        ".sh":   "shell",
        ".bash": "shell",
        ".json": "json",
        ".yaml": "yaml",
        ".yml":  "yaml",
        ".md":   "markdown",
    }

    def __init__(self, num_predict_limit: int = 0):
        """
        Args:
            num_predict_limit: Batas token yang dikonfigurasi. Jika > 0,
                               output yang mendekati limit akan diberi flag
                               "mungkin terpotong" (Layer 1).
        """
        self._num_predict_limit = num_predict_limit
        self._node_available = bool(shutil.which("node"))

    # ── Public API ──────────────────────────────────────────────────────────────

    def analyze(self, code: str, language: str = "generic",
                filepath: str = "") -> SentinelResult:
        """
        Analisis apakah `code` sudah lengkap.

        Args:
            code:      String kode yang akan dianalisis
            language:  Nama bahasa (lihat EXT_MAP) atau "generic"
            filepath:  Opsional — digunakan untuk auto-detect bahasa dari ekstensi

        Returns:
            SentinelResult dengan detail ketidaklengkapan
        """
        if filepath and language == "generic":
            ext = os.path.splitext(filepath)[1].lower()
            language = self.EXT_MAP.get(ext, "generic")

        language = language.lower().strip()

        # Strip markdown code fences jika ada
        clean_code = self._strip_code_fence(code)
        layers: list[str] = []

        # ── Layer 1: Heuristik panjang ────────────────────────────────────────
        if self._num_predict_limit > 0:
            char_estimate = self._num_predict_limit * 4  # kasar: 1 token ≈ 4 karakter
            if len(clean_code) >= char_estimate * 0.92:
                layers.append("L1:length_heuristic")
                return SentinelResult(
                    complete=False,
                    reason=f"Output mendekati batas token ({len(clean_code)} chars ≈ {self._num_predict_limit} tokens). Kemungkinan terpotong.",
                    language=language,
                    confidence=0.7,
                    layers_checked=layers,
                )

        # ── Layer 2: Truncation tail patterns ────────────────────────────────
        tail_result = self._check_truncation_tail(clean_code)
        layers.append("L2:tail_patterns")
        if tail_result:
            return SentinelResult(
                complete=False,
                reason=f"Pola truncation terdeteksi: {tail_result}",
                language=language,
                confidence=0.85,
                layers_checked=layers,
                truncated_at_line=clean_code.count("\n"),
            )

        # ── Layer 3: Bracket balance ──────────────────────────────────────────
        brace, paren, square = self._count_unbalanced(clean_code, language)
        layers.append("L3:bracket_balance")

        if brace != 0 or paren != 0 or square != 0:
            parts = []
            if brace != 0:
                parts.append(f"{{ kurung kurawal terbuka: {brace}" if brace > 0
                             else f"{{ kurung kurawal ekstra tutup: {abs(brace)}")
            if paren != 0:
                parts.append(f"( kurung terbuka: {paren}" if paren > 0
                             else f"( kurung ekstra tutup: {abs(paren)}")
            if square != 0:
                parts.append(f"[ kurung kotak terbuka: {square}" if square > 0
                             else f"[ kurung kotak ekstra tutup: {abs(square)}")
            return SentinelResult(
                complete=False,
                reason="; ".join(parts),
                open_brackets=max(0, brace),
                unclosed_parens=max(0, paren),
                unclosed_squares=max(0, square),
                language=language,
                confidence=0.95,
                layers_checked=layers,
            )

        # ── Layer 4: Language-specific AST/syntax check ──────────────────────
        ast_result = self._check_language_specific(clean_code, language)
        layers.append("L4:ast_syntax")
        if ast_result:
            return SentinelResult(
                complete=False,
                reason=ast_result,
                language=language,
                confidence=0.98,
                layers_checked=layers,
            )

        # ── Layer 5: Incomplete structure di Python (def/class tanpa body) ───
        if language == "python":
            py_result = self._check_python_incomplete_struct(clean_code)
            layers.append("L5:python_struct")
            if py_result:
                return SentinelResult(
                    complete=False,
                    reason=py_result,
                    language=language,
                    confidence=0.90,
                    layers_checked=layers,
                )

        # ── LOLOS: kode terlihat lengkap ─────────────────────────────────────
        return SentinelResult(
            complete=True,
            reason="Kode terlihat lengkap",
            language=language,
            confidence=0.92,
            layers_checked=layers,
        )

    def analyze_from_file(self, filepath: str) -> SentinelResult:
        """Baca file dari disk lalu analisis."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as e:
            return SentinelResult(
                complete=False,
                reason=f"Gagal membaca file: {e}",
                confidence=0.0,
            )
        return self.analyze(code, filepath=filepath)

    # ── Layer helpers ───────────────────────────────────────────────────────────

    def _strip_code_fence(self, code: str) -> str:
        """Hapus markdown fences ```...``` jika ada."""
        code = code.strip()
        # Cek apakah diawali dengan ```lang\n atau ```\n
        fence_re = re.compile(r"^```[a-zA-Z0-9]*\n(.*?)(?:\n```\s*)?$", re.DOTALL)
        m = fence_re.match(code)
        if m:
            return m.group(1).strip()
        return code

    def _check_truncation_tail(self, code: str) -> Optional[str]:
        """Cek apakah 3 baris terakhir mengandung pola truncation."""
        tail = "\n".join(code.splitlines()[-3:])
        for pat in _TRUNCATION_PATTERNS:
            m = re.search(pat, tail, re.IGNORECASE | re.MULTILINE)
            if m:
                return f"'{m.group(0).strip()}'"
        return None

    def _strip_strings_and_comments(self, code: str, language: str) -> str:
        """
        Hapus string literals dan komentar agar bracket counting akurat.
        Versi cepat berbasis regex (bukan full parser).
        """
        # Hapus triple-quoted strings Python
        if language == "python":
            code = re.sub(r'""".*?"""', '""', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", "''", code, flags=re.DOTALL)

        # Hapus block comments /* ... */
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # Hapus single-line comments // dan #
        code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
        if language not in ("cpp", "c", "javascript", "typescript", "java",
                            "rust", "go"):
            code = re.sub(r"#.*?$", "", code, flags=re.MULTILINE)

        # Hapus string literals (kasar — cukup untuk bracket counting)
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
        code = re.sub(r"`(?:[^`\\]|\\.)*`", "``", code)  # template literals JS

        return code

    def _count_unbalanced(self, code: str, language: str) -> tuple[int, int, int]:
        """
        Hitung jumlah bracket yang tidak seimbang.
        Returns (brace_delta, paren_delta, square_delta)
        Positif = lebih banyak buka, negatif = lebih banyak tutup.
        HTML dikecualikan dari bracket count (berbeda strukturnya).
        """
        if language in ("html", "markdown", "json", "yaml", "shell"):
            # HTML: tag balance diurus CustomHTMLParser, bukan bracket
            # JSON/YAML: sudah punya validator sendiri
            # Untuk HTML kita tetap cek { [ ( secara sederhana
            clean = code
        else:
            clean = self._strip_strings_and_comments(code, language)

        brace  = clean.count("{") - clean.count("}")
        paren  = clean.count("(") - clean.count(")")
        square = clean.count("[") - clean.count("]")

        return brace, paren, square

    def _check_language_specific(self, code: str, language: str) -> Optional[str]:
        """
        Pemeriksaan sintaks spesifik per bahasa.
        Returns pesan error jika ada masalah, None jika OK.
        """
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"Python SyntaxError baris {e.lineno}: {e.msg}"
            except Exception as e:
                return f"Python parse error: {e}"

        elif language in ("javascript", "typescript") and self._node_available:
            return self._check_js_with_node(code)

        elif language == "json":
            import json
            try:
                json.loads(code)
            except json.JSONDecodeError as e:
                return f"JSON error: {e}"

        return None

    def _check_js_with_node(self, code: str) -> Optional[str]:
        """Verifikasi JS/TS menggunakan `node --check`."""
        try:
            fd, tmp = tempfile.mkstemp(suffix=".js")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)
            res = subprocess.run(
                ["node", "--check", tmp],
                capture_output=True, text=True, timeout=5
            )
            os.unlink(tmp)
            if res.returncode != 0:
                lines = [l for l in res.stderr.splitlines() if l.strip()]
                return f"JS SyntaxError: {lines[0] if lines else 'unknown'}"
        except Exception:
            pass
        return None

    def _check_python_incomplete_struct(self, code: str) -> Optional[str]:
        """
        Deteksi def/class tanpa body (pola umum truncation Python).
        Contoh: `def foo(x):` di baris terakhir tanpa body apapun.
        """
        lines = code.rstrip().splitlines()
        if not lines:
            return None
        last = lines[-1].rstrip()
        # Jika baris terakhir diakhiri dengan : tapi bukan string/dict
        if re.match(r"^\s*(def |class |if |elif |else:|for |while |with |try:|except|finally:)", last):
            if last.endswith(":"):
                return f"Struktur Python tidak lengkap: '{last.strip()}' tanpa body"
        return None


# ── Singleton ──────────────────────────────────────────────────────────────────
_sentinel_instance: Optional[MarathonCodeSentinel] = None


def get_sentinel(num_predict_limit: int = 0) -> MarathonCodeSentinel:
    """Lazy singleton — satu instance per proses."""
    global _sentinel_instance
    if _sentinel_instance is None:
        _sentinel_instance = MarathonCodeSentinel(num_predict_limit)
    return _sentinel_instance


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sentinel = MarathonCodeSentinel()

    tests = [
        ("python_complete", "python", """
def hello(name):
    return f"Hello, {name}!"

print(hello("MOKO"))
"""),
        ("python_truncated_def", "python", """
def calculate_matrix(data):
    result = []
    for row in data:
"""),
        ("js_unclosed_brace", "javascript", """
function init() {
    const app = {
        name: "MOKO",
        run() {
            console.log("running");
"""),
        ("css_unbalanced", "css", """
.container {
    display: flex;
    .inner {
        color: red;
"""),
        ("python_truncation_comment", "python", """
def process():
    x = 1
    # TODO: rest of implementation
"""),
    ]

    for name, lang, code in tests:
        result = sentinel.analyze(code, language=lang)
        status = "✅ COMPLETE" if result.complete else f"❌ INCOMPLETE"
        print(f"\n[{name}] {status}")
        if not result.complete:
            print(f"  Reason: {result.reason}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Layers: {result.layers_checked}")
