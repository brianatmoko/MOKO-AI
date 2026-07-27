"""
moko_utils/text_utils.py
========================
Utility pemrosesan teks murni — dipindah dari moko_ui.workers.code_utils
dan moko_ui.workers.topic_learning_worker.

TIDAK ada dependency PyQt6 di sini.
Aman diimport dari agent, neuromath, super_learning, dll.
"""
from __future__ import annotations
import re

# ─── Bahasa yang dikenali untuk fenced code block ─────────────────────────────
KNOWN_LANGUAGES = {
    "html", "css", "js", "javascript", "typescript", "ts",
    "python", "py", "java", "kotlin", "kt", "scala",
    "go", "golang", "rust", "rs", "cpp", "c++", "c", "h",
    "csharp", "cs", "dotnet", "fsharp", "vb",
    "ruby", "rb", "php", "swift", "objc", "dart",
    "haskell", "hs", "erlang", "elixir", "clojure", "lisp", "scheme",
    "bash", "sh", "zsh", "fish", "shell", "powershell", "ps1", "cmd",
    "sql", "mysql", "postgresql", "sqlite", "graphql",
    "json", "yaml", "yml", "toml", "ini", "env",
    "xml", "xsl", "svg",
    "dockerfile", "docker", "makefile", "cmake",
    "asm", "assembly", "x86", "x64", "mips", "riscv",
    "r", "matlab", "julia", "fortran", "cobol",
    "latex", "tex", "markdown", "md",
    "text", "txt", "log", "diff", "patch",
    "console", "terminal", "output", "plaintext",
}

_FENCE_PATTERN = r'((?:```|~~~)[^\n]*\n[\s\S]*?(?:```|~~~))'
_TERMINAL_PROMPT_RE = re.compile(
    r'^(?:\$\s|\#\s|\%\s|\>\s|\>\>\>\s|\.\.\.|\.\.\.\ |\w+\>\s|[A-Z]:\\\>)'
)


def _is_terminal_prompt_line(line: str) -> bool:
    return bool(_TERMINAL_PROMPT_RE.match(line.lstrip()))


def chunk_text_code_aware(text: str, max_chunk_len: int = 1500) -> list:
    """
    Membagi teks menjadi chunk sambil menjaga blok kode (``` / ~~~) 100% utuh.
    Tidak merusak indentasi, whitespace, atau simbol di dalam kode.
    """
    if not text:
        return []

    parts = re.split(_FENCE_PATTERN, text)
    chunks: list = []
    current_chunk: list = []
    current_len = 0

    for part in parts:
        if not part:
            continue
        stripped = part.strip()
        is_fenced = stripped.startswith("```") or stripped.startswith("~~~")

        if is_fenced:
            if current_chunk and current_len + len(part) > max_chunk_len:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_len = [], 0
            current_chunk.append(stripped)
            current_len += len(part)
        else:
            for p in [x.strip() for x in part.split("\n\n")]:
                if not p:
                    continue
                if len(p) > max_chunk_len:
                    for sp in [p[i:i+max_chunk_len] for i in range(0, len(p), max_chunk_len)]:
                        if current_chunk and current_len + len(sp) > max_chunk_len:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk, current_len = [], 0
                        current_chunk.append(sp)
                        current_len += len(sp)
                else:
                    if current_chunk and current_len + len(p) > max_chunk_len:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk, current_len = [], 0
                    current_chunk.append(p)
                    current_len += len(p)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def protect_code_blocks(text: str) -> tuple:
    """
    Gantikan blok kode dengan placeholder agar tidak rusak oleh operasi string.
    Returns: (text_with_placeholders, {placeholder: original_block})
    """
    placeholders: dict = {}
    counter = [0]

    def _replace(match):
        ph = f"__MOKO_CODE_BLOCK_{counter[0]}__"
        placeholders[ph] = match.group(0)
        counter[0] += 1
        return ph

    protected = re.sub(_FENCE_PATTERN, _replace, text)
    protected = re.sub(r'(`[^`\n]{4,}`)', _replace, protected)
    return protected, placeholders


def restore_code_blocks(text: str, placeholders: dict) -> str:
    """Kembalikan semua placeholder ke konten kode aslinya."""
    for ph, original in placeholders.items():
        text = text.replace(ph, original)
    return text


def clean_llm_query(raw: str) -> str:
    """
    Bersihkan output LLM dari artefak thinking/chain-of-thought sebelum
    digunakan sebagai search query.
    (Dipindah dari moko_ui.workers.topic_learning_worker._clean_llm_query)
    """
    if not raw:
        return ""
    text = raw.strip()

    # Hapus blok thinking
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r'\[THINKING\].*?\[/THINKING\]', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r'\*\[Thinking\]\*.*?(?=\n[^\n]{0,150}\n|$)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

    skip_prefixes = (
        "*[", "#", "1.", "2.", "3.", "-", "Wait", "Actually", "Let", "Step",
        "Analyze", "Now", "Think", "The", "Since", "Note", "Given", "Based",
        "However", "But", "So,", "So ", "So", "Okay", "Ok,", "First", "Finally",
        "In ", "To ", "For ", "If ", "As ", "Write", "Generate", "Create",
        "Here", "Sure", "Task", "Input", "Output", "Query", "Search",
    )

    lines = []
    for ln in text.split('\n'):
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r'^[*\-\•\s]+', '', ln).strip()
        ln = ln.strip('"').strip("'").strip()
        if not ln:
            continue
        if any(ln.startswith(pfx) for pfx in skip_prefixes):
            continue
        lines.append(ln)

    result = " ".join(lines[:3]).strip()
    result = re.sub(r'\s+', ' ', result)
    result = result[:200]
    return result


# Alias untuk backward-compat (nama lama dengan underscore)
_clean_llm_query = clean_llm_query
