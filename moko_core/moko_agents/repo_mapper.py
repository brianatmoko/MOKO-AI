"""
MOKO Polyglot & Security-Aware Repo Mapper
===========================================
Berdasarkan: Aider "Repo Map" (Paul Gauthier, 2023)
             Static Application Security Testing (SAST) Patterns

Mesin pemetaan repositori tingkat lanjut yang memahami multi-bahasa pemrograman:
Python, JS/TS, C/C++, Rust, Go, HTML/CSS.
Selain memetakan struktur simbol (class, function, struct), ia secara otomatis
mendeteksi pola kerentanan keamanan (vulnerability) di codebase dan melabelinya sebagai
simbol dengan tingkat keparahan tertentu agar MOKO dapat mengaudit kode secara realtime.
"""

import ast
import os
import re
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any


@dataclass
class SymbolDef:
    """Definisi simbol dalam codebase"""
    name: str
    kind: str           # 'function', 'class', 'method', 'struct', 'vulnerability', 'import'
    file: str
    line: int
    signature: str      # Tanda tangan lengkap / deskripsi formal
    docstring: str
    security_risk: str = ""  # 'HIGH', 'MEDIUM', 'LOW' atau kosong jika aman
    references: Set[str] = field(default_factory=set)


@dataclass
class FileMap:
    """Representasi satu file dalam peta repositori"""
    path: str
    rel_path: str
    size_bytes: int
    symbols: List[SymbolDef]
    imports: List[str]
    last_hash: str
    language: str


class ASTSymbolExtractor(ast.NodeVisitor):
    """Mengekstrak simbol dari AST Python satu file"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: List[SymbolDef] = []
        self.imports: List[str] = []
        self._class_stack: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        docstring = ast.get_docstring(node) or ""
        bases = [self._unparse_node(b) for b in node.bases]
        sig = f"class {node.name}({', '.join(bases)})"
        self.symbols.append(SymbolDef(
            name=node.name,
            kind='class',
            file=self.file_path,
            line=node.lineno,
            signature=sig,
            docstring=docstring[:120] if docstring else ""
        ))
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._extract_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._extract_function(node, is_async=True)

    def _extract_function(self, node, is_async=False):
        docstring = ast.get_docstring(node) or ""
        args = self._extract_args(node.args)
        ret = self._unparse_node(node.returns) if node.returns else "Any"
        prefix = "async " if is_async else ""
        parent = f"{self._class_stack[-1]}." if self._class_stack else ""
        kind = 'method' if self._class_stack else 'function'
        sig = f"{prefix}def {parent}{node.name}({args}) -> {ret}"
        self.symbols.append(SymbolDef(
            name=f"{parent}{node.name}",
            kind=kind,
            file=self.file_path,
            line=node.lineno,
            signature=sig,
            docstring=docstring[:120] if docstring else ""
        ))
        self.generic_visit(node)

    def _extract_args(self, args: ast.arguments) -> str:
        parts = []
        defaults = args.defaults
        n_without_defaults = len(args.args) - len(defaults)
        
        for i, arg in enumerate(args.args):
            ann = f": {self._unparse_node(arg.annotation)}" if arg.annotation else ""
            if i >= n_without_defaults:
                default_val = self._unparse_node(defaults[i - n_without_defaults])
                parts.append(f"{arg.arg}{ann}={default_val}")
            else:
                parts.append(f"{arg.arg}{ann}")
        
        if args.vararg:
            ann = f": {self._unparse_node(args.vararg.annotation)}" if args.vararg.annotation else ""
            parts.append(f"*{args.vararg.arg}{ann}")
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
        return ", ".join(parts)

    def _unparse_node(self, node) -> str:
        if node is None:
            return "None"
        try:
            return ast.unparse(node)
        except Exception:
            return "..."


class RepoMapper:
    """
    Membangun peta simbol multi-bahasa dan audit keamanan seluruh repositori MOKO OS.
    """

    SKIP_DIRS = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', '.moko_crypto', 'models', 'dist', 'build'}
    SKIP_FILES = {'setup.py', 'conftest.py', 'package-lock.json', 'yarn.lock'}

    # Pola SAST Keamanan untuk audit statis
    SECURITY_PATTERNS = {
        "python": [
            (r'\beval\([^)]*\)', "eval() usage", "HIGH", "Penggunaan eval dapat menyebabkan Remote Code Execution (RCE)"),
            (r'\bexec\([^)]*\)', "exec() usage", "HIGH", "Penggunaan exec dinamis sangat berbahaya"),
            (r'subprocess\.run\([^)]*shell\s*=\s*True[^)]*\)', "subprocess with shell=True", "HIGH", "Subprocess shell injection risk"),
            (r'(?:api_key|secret|password|private_key)\s*=\s*[\'"][A-Za-z0-9+/]{16,}[\'"]', "Hardcoded credentials", "HIGH", "Kredensial atau secret kunci API hardcoded"),
        ],
        "javascript": [
            (r'\beval\([^)]*\)', "eval() usage", "HIGH", "Penggunaan eval menyebabkan kerentanan XSS/RCE"),
            (r'innerHTML\s*=\s*', "Direct innerHTML modification", "MEDIUM", "Bahaya XSS jika input tidak disanitasi"),
            (r'dangerouslySetInnerHTML', "React dangerouslySetInnerHTML", "MEDIUM", "Potensi kerentanan DOM XSS"),
            (r'child_process\.exec\([^)]*\)', "child_process execution", "HIGH", "Command injection risk"),
        ],
        "cpp": [
            (r'\bstrcpy\([^)]*\)', "strcpy() usage", "HIGH", "strcpy tidak memeriksa ukuran buffer (potensi Buffer Overflow)"),
            (r'\bgets\([^)]*\)', "gets() usage", "HIGH", "gets() sangat berbahaya dan sudah didepresiasi"),
            (r'\bsprintf\([^)]*\%s[^)]*\)', "sprintf raw format specifier", "MEDIUM", "sprintf format string vulnerability"),
        ],
        "rust": [
            (r'\bunsafe\s*\{', "unsafe block", "LOW", "Blok kode unsafe melangkahi pengaman kompilator Rust"),
        ],
        "sql": [
            (r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\s*\+\s*\w+', "Dynamic SQL string concat", "HIGH", "Potensi SQL Injection dari konkatenasi string dinamis"),
        ]
    }

    def __init__(self, workspace_root: str, verbose: bool = False):
        self.workspace_root = Path(workspace_root).resolve()
        self.verbose = verbose
        self._file_maps: Dict[str, FileMap] = {}
        self._symbol_index: Dict[str, List[SymbolDef]] = {}
        self._last_full_scan: float = 0.0
        self._scan_cache_ttl: float = 30.0

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🗺️  [RepoMap] {msg}")

    def scan(self, force: bool = False) -> int:
        """Scan seluruh file pemrograman di workspace dan ekstrak graf simbol & kerentanan"""
        now = time.time()
        if not force and (now - self._last_full_scan) < self._scan_cache_ttl:
            return len(self._symbol_index)

        self._log(f"Memulai polyglot & security scan: {self.workspace_root}")
        t0 = time.time()
        self._file_maps.clear()
        self._symbol_index.clear()

        total_symbols = 0
        for code_file in self._iter_code_files():
            rel_path = str(code_file.relative_to(self.workspace_root))
            try:
                fmap = self._parse_file(code_file, rel_path)
                if fmap:
                    self._file_maps[rel_path] = fmap
                    total_symbols += len(fmap.symbols)
                    for sym in fmap.symbols:
                        self._symbol_index.setdefault(sym.name, []).append(sym)
            except Exception as e:
                self._log(f"Parse error {rel_path}: {e}")

        self._last_full_scan = time.time()
        elapsed = (time.time() - t0) * 1000
        self._log(f"Scan selesai: {len(self._file_maps)} files, {total_symbols} symbols, {elapsed:.1f}ms")
        return total_symbols

    def _iter_code_files(self):
        """Iterasi semua berkas program: py, js, ts, tsx, jsx, c, cpp, h, hpp, rs, go, html, css, sql"""
        extensions = ["*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.c", "*.cpp", "*.h", "*.hpp", "*.rs", "*.go", "*.html", "*.css", "*.sql"]
        for ext in extensions:
            for path in sorted(self.workspace_root.rglob(ext)):
                skip = False
                for part in path.parts:
                    if part in self.SKIP_DIRS:
                        skip = True
                        break
                if not skip and path.name not in self.SKIP_FILES:
                    yield path

    def _parse_file(self, path: Path, rel_path: str) -> Optional[FileMap]:
        """Ekstraktor polyglot berbasis extension"""
        try:
            source = path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return None

        file_hash = hashlib.md5(source.encode()).hexdigest()[:12]
        ext = path.suffix.lower()

        symbols: List[SymbolDef] = []
        imports: List[str] = []
        lang = "unknown"

        # 1. Klasifikasikan bahasa & parser
        if ext == ".py":
            lang = "python"
            try:
                tree = ast.parse(source, filename=str(path))
                extractor = ASTSymbolExtractor(rel_path)
                extractor.visit(tree)
                symbols.extend(extractor.symbols)
                imports.extend(extractor.imports)
            except SyntaxError:
                # Fallback to regex jika ada syntax error python
                symbols.extend(self._regex_extract_python(source, rel_path))
        elif ext in [".js", ".jsx", ".ts", ".tsx"]:
            lang = "javascript"
            symbols.extend(self._regex_extract_js(source, rel_path))
        elif ext in [".c", ".cpp", ".h", ".hpp"]:
            lang = "cpp"
            symbols.extend(self._regex_extract_cpp(source, rel_path))
        elif ext == ".rs":
            lang = "rust"
            symbols.extend(self._regex_extract_rust(source, rel_path))
        elif ext == ".go":
            lang = "go"
            symbols.extend(self._regex_extract_go(source, rel_path))
        elif ext == ".html":
            lang = "html"
            symbols.extend(self._regex_extract_html(source, rel_path))
        elif ext == ".css":
            lang = "css"
            symbols.extend(self._regex_extract_css(source, rel_path))
        elif ext == ".sql":
            lang = "sql"
            symbols.extend(self._regex_extract_sql(source, rel_path))

        # 2. Jalankan Static Application Security Testing (SAST) Audit
        symbols.extend(self._audit_security(source, lang, rel_path))

        return FileMap(
            path=str(path),
            rel_path=rel_path,
            size_bytes=path.stat().st_size,
            symbols=symbols,
            imports=imports,
            last_hash=file_hash,
            language=lang
        )

    # --- REGEX EXTRACTORS ---

    def _regex_extract_python(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            # Class
            m = re.match(r'^\s*class\s+([A-Za-z_]\w*)', line)
            if m:
                symbols.append(SymbolDef(m.group(1), 'class', file_path, i, line.strip(), ""))
            # Def
            m = re.match(r'^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)', line)
            if m:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line.strip(), ""))
        return symbols

    def _regex_extract_js(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # Function declaration: function name(...)
            m = re.search(r'\bfunction\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line_str, ""))
            # Class declaration: class Name
            m = re.search(r'\bclass\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'class', file_path, i, line_str, ""))
            # Arrow function const/let: const name = (...) =>
            m = re.search(r'\b(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line_str, ""))
        return symbols

    def _regex_extract_cpp(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # C++ Class / Struct
            m = re.match(r'\b(?:class|struct)\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'class' if "class" in line_str else "struct", file_path, i, line_str, ""))
            # Function: type name(args) { or ;
            m = re.match(r'^[A-Za-z_]\w*(?:\s*\*+)?\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*(?:\{|;)?', line_str)
            if m and m.group(1) not in ["if", "while", "for", "switch", "return"]:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line_str, ""))
        return symbols

    def _regex_extract_rust(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # Struct / Enum / Trait
            m = re.match(r'^pub\s+(?:struct|enum|trait)\s+([A-Za-z_]\w*)', line_str) or re.match(r'^(?:struct|enum|trait)\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'struct', file_path, i, line_str, ""))
            # Fn: pub fn or fn
            m = re.match(r'^(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line_str, ""))
        return symbols

    def _regex_extract_go(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # Struct / Interface
            m = re.match(r'^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'struct', file_path, i, line_str, ""))
            # Function: func name(...)
            m = re.match(r'^func\s+([A-Za-z_]\w*)\s*\(', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'function', file_path, i, line_str, ""))
            # Method: func (r *Receiver) name(...)
            m = re.match(r'^func\s*\([^)]*\)\s+([A-Za-z_]\w*)\s*\(', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'method', file_path, i, line_str, ""))
        return symbols

    def _regex_extract_html(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # Find IDs or class definitions in templates
            m = re.findall(r'id=["\']([^"\']+)["\']', line_str)
            for match in m:
                symbols.append(SymbolDef(match, 'html-id', file_path, i, f"id=\"{match}\"", ""))
            m = re.findall(r'name=["\']([^"\']+)["\']', line_str)
            for match in m:
                symbols.append(SymbolDef(match, 'html-name', file_path, i, f"name=\"{match}\"", ""))
        return symbols

    def _regex_extract_css(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # Class selectors: .name {
            m = re.match(r'^\.([A-Za-z_][A-Za-z0-9_-]*)\s*\{', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'css-class', file_path, i, line_str, ""))
            # ID selectors: #name {
            m = re.match(r'^#([A-Za-z_][A-Za-z0-9_-]*)\s*\{', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'css-id', file_path, i, line_str, ""))
        return symbols

    def _regex_extract_sql(self, source: str, file_path: str) -> List[SymbolDef]:
        symbols = []
        for i, line in enumerate(source.split('\n'), 1):
            line_str = line.strip()
            # CREATE TABLE / PROCEDURE
            m = re.match(r'(?i)^CREATE\s+(?:TABLE|PROCEDURE|VIEW|INDEX)\s+([A-Za-z_]\w*)', line_str)
            if m:
                symbols.append(SymbolDef(m.group(1), 'sql-schema', file_path, i, line_str, ""))
        return symbols

    # --- SECURITY AUDIT ---

    def _audit_security(self, source: str, lang: str, file_path: str) -> List[SymbolDef]:
        """Scan source code untuk kerentanan keamanan dan jadikan sebagai simbol khusus"""
        vulns = []
        patterns = self.SECURITY_PATTERNS.get(lang, [])
        
        # Tambahkan SQL pattern untuk semua bahasa pemrograman jika ada raw string query
        if lang in ["python", "javascript", "go", "rust", "cpp"]:
            patterns = patterns + self.SECURITY_PATTERNS["sql"]

        for pattern, vuln_name, risk, description in patterns:
            for i, line in enumerate(source.split('\n'), 1):
                if re.search(pattern, line):
                    vulns.append(SymbolDef(
                        name=f"VULN::{vuln_name}",
                        kind="vulnerability",
                        file=file_path,
                        line=i,
                        signature=line.strip()[:100],
                        docstring=description,
                        security_risk=risk
                    ))
        return vulns

    # --- MATCHING & CONTEXT ---

    def find_symbol(self, name: str) -> List[SymbolDef]:
        self.scan()
        exact = self._symbol_index.get(name, [])
        if exact:
            return exact
        name_lower = name.lower()
        results = []
        for sym_name, defs in self._symbol_index.items():
            if name_lower in sym_name.lower():
                results.extend(defs)
        return results[:10]

    def get_context_for_query(self, query: str, max_symbols: int = 25) -> str:
        """Membangun konteks codebase dan ringkasan audit keamanan jika ditemukan"""
        self.scan()
        q_lower = query.lower()
        query_tokens = set(re.findall(r'\b\w+\b', q_lower))

        scored: List[Tuple[float, SymbolDef]] = []
        for sym_name, defs in self._symbol_index.items():
            for sym in defs:
                score = 0.0
                name_tokens = set(re.findall(r'\b\w+\b', sym.name.lower()))
                overlap = len(query_tokens & name_tokens)
                score += overlap * 2.0
                if sym.docstring:
                    doc_tokens = set(re.findall(r'\b\w+\b', sym.docstring.lower()))
                    score += len(query_tokens & doc_tokens) * 0.5
                
                # Boost untuk file yang namanya mirip query
                if Path(sym.file).stem.lower() in q_lower:
                    score += 5.0
                
                # Boost jika query mengandung security/vuln keywords
                if sym.kind == "vulnerability" and any(k in q_lower for k in ["aman", "keamanan", "audit", "security", "vuln", "rce", "xss"]):
                    score += 10.0

                if score > 0:
                    scored.append((score, sym))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_symbols]

        if not top:
            return ""

        lines = []
        lines.append("═" * 66)
        lines.append("  MOKO POLYGLOT & SECURITY-AWARE REPO MAP — Codebase Context")
        lines.append("═" * 66)

        seen_files = set()
        vulnerabilities = []

        for score, sym in top:
            if sym.kind == "vulnerability":
                vulnerabilities.append(sym)
                continue

            if sym.file not in seen_files:
                lines.append(f"\n📄 {sym.file}")
                seen_files.add(sym.file)
            lines.append(f"  [{sym.kind:12s}] L{sym.line:4d} │ {sym.signature}")
            if sym.docstring:
                lines.append(f"                └─ {sym.docstring[:80]}")

        # Inject Security Vulnerability audit jika terdeteksi
        if vulnerabilities:
            lines.append("\n" + "🚨" * 33)
            lines.append("  SECURITY AUDIT ALERT — Kerentanan Potensial Ditemukan:")
            lines.append("  " + "═" * 62)
            for v in vulnerabilities[:5]:
                lines.append(f"  [{v.security_risk:6s}] L{v.line:4d} in {v.file}")
                lines.append(f"  Kode  │ {v.signature}")
                lines.append(f"  Sebab │ {v.docstring}")
                lines.append("  " + "─" * 62)
            lines.append("🚨" * 33)

        lines.append(f"\n[{len(top)} simbol relevan dari {len(self._file_maps)} file dipetakan]")
        lines.append("═" * 66)
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        self.scan()
        total_syms = sum(len(v) for v in self._symbol_index.values())
        kinds = {}
        languages = {}
        for fmap in self._file_maps.values():
            languages[fmap.language] = languages.get(fmap.language, 0) + 1
            for sym in fmap.symbols:
                kinds[sym.kind] = kinds.get(sym.kind, 0) + 1
        return {
            "files": len(self._file_maps),
            "total_symbols": total_syms,
            "unique_names": len(self._symbol_index),
            "by_kind": kinds,
            "by_language": languages,
        }


# Singleton
_repo_mapper_instance: Optional[RepoMapper] = None

def get_repo_mapper(workspace_root: str = None, verbose: bool = False) -> RepoMapper:
    global _repo_mapper_instance
    if _repo_mapper_instance is None:
        if not workspace_root:
            workspace_root = str(Path(__file__).resolve().parent.parent.parent)
        _repo_mapper_instance = RepoMapper(workspace_root, verbose=verbose)
    return _repo_mapper_instance
