"""
code_structure.py — Analisis Struktur Kode MOKO IDE (Fase 1)
=============================================================
"Otak" pengecekan struktur ala VS Code, murni lokal (tanpa Node.js):
  - Deteksi open/end tag HTML/XML (stack-based, presisi per-tag)
  - Deteksi bracket ()[]{} tak seimbang (sadar string & komentar)
  - Error sintaks umum via tree-sitter (bila terpasang), fallback pure-Python
Dipakai oleh editor_panel.py untuk pemeriksaan real-time, dan hasilnya
bisa diberikan ke MOKO-Coder sebagai konteks terstruktur (Fase 3).
"""
import re
from bisect import bisect_right
from dataclasses import dataclass, asdict

# ─── tree-sitter opsional ─────────────────────────────────────────────────────
try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover — environment tanpa tree-sitter
    _ts_get_parser = None
    _TS_AVAILABLE = False

_PARSER_CACHE: dict = {}

# Batas aman performa
_MAX_ANALYZE_CHARS = 300_000
_MAX_ISSUES = 50

# Pemetaan ekstensi file → id bahasa (selaras dengan editor_panel._EXT_LANG)
EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".html": "html", ".htm": "html", ".xml": "xml", ".css": "css",
    ".json": "json", ".md": "markdown", ".sh": "shell",
    ".yaml": "yaml", ".yml": "yaml", ".txt": "text",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".rs": "rust",
}

# Bahasa yang aman untuk pengecekan bracket ()[]{}
_BRACKET_LANGS = {"python", "javascript", "typescript", "json", "css", "c", "cpp", "rust"}

# Bahasa yang diparse tree-sitter untuk error sintaks umum
_TS_LANGS = {"python", "javascript", "typescript", "json", "css", "c", "cpp", "rust", "yaml", "bash"}

# Sintaks string/komentar per bahasa: (line_comments, block_comment, string_delims)
_LANG_SYNTAX = {
    "python":     (("#",),  None,         ("'''", '"""', "'", '"')),
    "javascript": (("//",), ("/*", "*/"), ("'", '"', "`")),
    "typescript": (("//",), ("/*", "*/"), ("'", '"', "`")),
    "json":       ((),      None,         ('"',)),
    "css":        ((),      ("/*", "*/"), ("'", '"')),
    "c":          (("//",), ("/*", "*/"), ("'", '"')),
    "cpp":        (("//",), ("/*", "*/"), ("'", '"')),
    "rust":       (("//",), ("/*", "*/"), ('"',)),
}

_OPENERS = {"(", "[", "{"}
_CLOSER_OF = {")": "(", "]": "[", "}": "{"}

# Elemen HTML void (tidak butuh tag penutup)
_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
# Elemen raw-text: isinya bukan HTML
_RAWTEXT_ELEMENTS = {"script", "style"}

# Tokenizer HTML: komentar / CDATA / doctype / PI / tag
_HTML_TOKEN_RE = re.compile(
    r"<!--.*?-->"
    r"|<!\[CDATA\[.*?\]\]>"
    r"|<![^>]*>"
    r"|<\?.*?\?>"
    r"|<(/?)([a-zA-Z][a-zA-Z0-9:._-]*)"
    r"((?:\"[^\"]*\"|'[^']*'|[^<>\"'])*?)"
    r"(/?)\s*>",
    re.DOTALL,
)


# ─── Model hasil ──────────────────────────────────────────────────────────────
@dataclass
class StructureIssue:
    """Satu masalah struktur. Posisi 0-based (line, col)."""
    kind: str          # unclosed_tag | unopened_tag | mismatched_tag |
                       # unclosed_bracket | unmatched_bracket | mismatched_bracket |
                       # syntax_error | missing_token
    message: str       # pesan siap-tampil (Bahasa Indonesia)
    line: int
    col: int
    end_line: int
    end_col: int
    severity: str = "error"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"baris {self.line + 1}: {self.message}"


# ─── Helper posisi ────────────────────────────────────────────────────────────
def _line_starts(source: str):
    starts = [0]
    for m in re.finditer("\n", source):
        starts.append(m.end())
    return starts


def _pos(starts, index: int):
    line = bisect_right(starts, index) - 1
    return line, index - starts[line]


def language_for_path(path: str) -> str:
    """Tebak id bahasa dari ekstensi file. '' bila tidak dikenal."""
    if not path:
        return ""
    dot = path.rfind(".")
    if dot < 0:
        return ""
    return EXT_LANG.get(path[dot:].lower(), "")


def is_supported_language(language: str) -> bool:
    """True bila ada minimal satu checker yang berlaku untuk bahasa ini."""
    return language in ("html", "xml") or language in _BRACKET_LANGS \
        or (language in _TS_LANGS and _TS_AVAILABLE)


def is_treesitter_available() -> bool:
    return _TS_AVAILABLE


# ─── Scanner kode (lewati string & komentar) ─────────────────────────────────
def _iter_code_chars(source: str, language: str):
    """Yield (index, char) hanya untuk karakter di luar string/komentar."""
    syn = _LANG_SYNTAX.get(language)
    if syn is None:
        for i, ch in enumerate(source):
            yield i, ch
        return
    line_comments, block_comment, strings = syn
    i, n = 0, len(source)
    while i < n:
        # Komentar satu baris
        skipped = False
        for lc in line_comments:
            if source.startswith(lc, i):
                nl = source.find("\n", i)
                i = n if nl == -1 else nl + 1
                skipped = True
                break
        if skipped:
            continue
        # Komentar blok
        if block_comment and source.startswith(block_comment[0], i):
            end = source.find(block_comment[1], i + len(block_comment[0]))
            i = n if end == -1 else end + len(block_comment[1])
            continue
        # String literal
        for delim in strings:
            if source.startswith(delim, i):
                j = i + len(delim)
                while j < n:
                    if source[j] == "\\":
                        j += 2
                        continue
                    if source.startswith(delim, j):
                        j += len(delim)
                        break
                    if len(delim) == 1 and source[j] == "\n":
                        break  # string 1-baris tak tertutup
                    j += 1
                i = max(j, i + 1)
                skipped = True
                break
        if skipped:
            continue
        yield i, source[i]
        i += 1


# ─── Checker bracket ──────────────────────────────────────────────────────────
def check_brackets(source: str, language: str):
    """Deteksi bracket ()[]{} tak seimbang di luar string/komentar."""
    if language not in _BRACKET_LANGS:
        return []
    starts = _line_starts(source)
    issues, stack = [], []
    for i, ch in _iter_code_chars(source, language):
        if ch in _OPENERS:
            stack.append((ch, i))
        elif ch in _CLOSER_OF:
            if not stack:
                ln, col = _pos(starts, i)
                issues.append(StructureIssue(
                    "unmatched_bracket",
                    f"Kurung penutup '{ch}' tanpa pembuka",
                    ln, col, ln, col + 1))
            elif stack[-1][0] != _CLOSER_OF[ch]:
                opener, _oi = stack.pop()
                ln, col = _pos(starts, i)
                issues.append(StructureIssue(
                    "mismatched_bracket",
                    f"Kurung penutup '{ch}' tidak cocok dengan pembuka '{opener}'",
                    ln, col, ln, col + 1))
            else:
                stack.pop()
    for opener, oi in stack:
        ln, col = _pos(starts, oi)
        issues.append(StructureIssue(
            "unclosed_bracket",
            f"Kurung pembuka '{opener}' belum ditutup",
            ln, col, ln, col + 1))
    return issues


def find_matching_bracket(source: str, index: int, language: str):
    """
    Cari pasangan bracket untuk karakter di `index`.
    Return (idx_buka, idx_tutup) atau None bila tak ada pasangan.
    """
    if language not in _BRACKET_LANGS or len(source) > _MAX_ANALYZE_CHARS:
        return None
    if index < 0 or index >= len(source) or source[index] not in "()[]{}":
        return None
    pairs, stack = {}, []
    for i, ch in _iter_code_chars(source, language):
        if ch in _OPENERS:
            stack.append((ch, i))
        elif ch in _CLOSER_OF and stack:
            if stack[-1][0] == _CLOSER_OF[ch]:
                _, oi = stack.pop()
                pairs[oi] = i
                pairs[i] = oi
            else:
                stack.pop()  # mismatch — tidak dianggap pasangan
    if index not in pairs:
        return None
    a, b = index, pairs[index]
    return (a, b) if a < b else (b, a)


# ─── Checker tag HTML/XML ─────────────────────────────────────────────────────
def check_tags(source: str, xml_mode: bool = False):
    """
    Deteksi open/end tag HTML (atau XML) yang tidak seimbang:
    tag belum ditutup, tag penutup tanpa pembuka, dan salah urutan nesting.
    """
    starts = _line_starts(source)
    issues, stack = [], []  # stack: (key, nama_asli, index)
    pos = 0
    while True:
        m = _HTML_TOKEN_RE.search(source, pos)
        if not m:
            break
        pos = m.end()
        name = m.group(2)
        if not name:
            continue  # komentar / doctype / CDATA / PI
        closing = m.group(1) == "/"
        self_closing = m.group(4) == "/"
        key = name if xml_mode else name.lower()
        if closing:
            if stack and stack[-1][0] == key:
                stack.pop()
            elif any(entry[0] == key for entry in stack):
                # Tag penutup cocok dengan pembuka lebih luar:
                # semua tag di atasnya berarti belum ditutup.
                while stack and stack[-1][0] != key:
                    _k, disp, idx = stack.pop()
                    ln, col = _pos(starts, idx)
                    issues.append(StructureIssue(
                        "unclosed_tag",
                        f"Tag <{disp}> belum ditutup",
                        ln, col, ln, col + len(disp) + 1))
                stack.pop()
            else:
                ln, col = _pos(starts, m.start())
                issues.append(StructureIssue(
                    "unopened_tag",
                    f"Tag penutup </{name}> tanpa tag pembuka",
                    ln, col, ln, col + len(name) + 3))
        elif self_closing:
            continue
        elif not xml_mode and key in _VOID_ELEMENTS:
            continue
        else:
            stack.append((key, name, m.start()))
            # Elemen raw-text: lompati isinya hingga tag penutupnya
            if not xml_mode and key in _RAWTEXT_ELEMENTS:
                mm = re.search(rf"</{key}\s*>", source[pos:], re.IGNORECASE)
                if mm:
                    stack.pop()
                    pos += mm.end()
                # bila tak ada penutup → tetap di stack → unclosed di EOF
    for _k, disp, idx in stack:
        ln, col = _pos(starts, idx)
        issues.append(StructureIssue(
            "unclosed_tag",
            f"Tag <{disp}> belum ditutup",
            ln, col, ln, col + len(disp) + 1))
    return issues


# ─── Error sintaks via tree-sitter ───────────────────────────────────────────
def _val(x):
    """Binding tree-sitter 0.26 memakai method, bukan property."""
    return x() if callable(x) else x


def _get_parser(language: str):
    if language not in _PARSER_CACHE:
        _PARSER_CACHE[language] = _ts_get_parser(language)
    return _PARSER_CACHE[language]


def treesitter_issues(source: str, language: str, limit: int = 20):
    """Kumpulkan node ERROR/MISSING dari parse tree tree-sitter."""
    if not _TS_AVAILABLE or language not in _TS_LANGS:
        return []
    try:
        tree = _get_parser(language).parse(source)
        root = _val(tree.root_node)
        if not _val(root.has_error):
            return []
    except Exception:
        return []
    issues = []

    def _walk(node):
        if len(issues) >= limit:
            return
        if _val(node.is_missing):
            sp, ep = _val(node.start_position), _val(node.end_position)
            issues.append(StructureIssue(
                "missing_token",
                f"Sintaks: token '{_val(node.kind)}' hilang",
                _val(sp.row), _val(sp.column), _val(ep.row), _val(ep.column)))
            return
        if _val(node.is_error):
            sp, ep = _val(node.start_position), _val(node.end_position)
            issues.append(StructureIssue(
                "syntax_error",
                "Struktur sintaks tidak valid / belum lengkap",
                _val(sp.row), _val(sp.column), _val(ep.row), _val(ep.column)))
        for i in range(_val(node.child_count)):
            _walk(node.child(i))

    _walk(root)
    return issues


# ─── API utama ────────────────────────────────────────────────────────────────
def analyze_source(source: str, language: str, use_treesitter: bool = True):
    """
    Analisis penuh satu berkas: gabungan tag checker, bracket checker,
    dan error sintaks tree-sitter. Return list[StructureIssue] terurut.
    """
    if not source or not language or len(source) > _MAX_ANALYZE_CHARS:
        return []
    issues = []
    if language in ("html", "xml"):
        issues.extend(check_tags(source, xml_mode=(language == "xml")))
    if language in _BRACKET_LANGS:
        issues.extend(check_brackets(source, language))
    if use_treesitter and language not in ("html", "xml"):
        seen = {(it.line, it.col) for it in issues}
        for it in treesitter_issues(source, language):
            if (it.line, it.col) not in seen:
                issues.append(it)
    issues.sort(key=lambda it: (it.line, it.col))
    return issues[:_MAX_ISSUES]


def summarize_issues(issues, max_items: int = 5) -> str:
    """Ringkasan singkat siap-tampil (footer editor / tooltip / prompt AI)."""
    if not issues:
        return "Struktur OK — semua tag & bracket seimbang"
    parts = [str(it) for it in issues[:max_items]]
    if len(issues) > max_items:
        parts.append(f"(+{len(issues) - max_items} masalah lain)")
    return "; ".join(parts)


def structure_report(source: str, language: str) -> dict:
    """
    Laporan terstruktur untuk konteks AI (MOKO-Coder, Fase 3).
    Contoh: {'language': 'html', 'ok': False, 'issue_count': 1,
             'engine': 'tree-sitter', 'issues': [...]}
    """
    issues = analyze_source(source, language)
    return {
        "language": language or "unknown",
        "ok": not issues,
        "issue_count": len(issues),
        "engine": "tree-sitter" if _TS_AVAILABLE else "fallback",
        "issues": [it.to_dict() for it in issues],
    }
