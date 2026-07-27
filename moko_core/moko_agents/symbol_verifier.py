"""
MOKO Polyglot Symbol Verifier & Security Auditor
=================================================
Berdasarkan: AlphaCode Symbol Grounding & Copilot Enterprise
             Static Application Security Testing (SAST) Rule Engine

Tugas Utama:
  1. Memverifikasi pemanggilan simbol pada berbagai bahasa pemrograman
     (Python, JS/TS, Rust, Go, C/C++) terhadap RepoMapper AST Symbol Graph.
  2. Melakukan audit keamanan statis (realtime) pada blok kode buatan LLM
     sebelum disajikan kepada user.
  3. Mendeteksi kerentanan seperti RCE, XSS, SQLi, Buffer Overflow, hardcoded secrets,
     dan memberi anotasi warning box yang informatif.
"""

import re
import ast
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from moko_agents.repo_mapper import RepoMapper


@dataclass
class SymbolVerification:
    """Hasil verifikasi satu simbol"""
    name: str
    status: str         # 'verified' | 'unknown' | 'builtin' | 'stdlib'
    confidence: float
    suggestion: Optional[str] = None   # Nama simbol alternatif yang ditemukan


@dataclass
class SecurityVulnerability:
    """Representasi kerentanan keamanan yang terdeteksi di output LLM"""
    vuln_type: str
    severity: str       # 'HIGH' | 'MEDIUM' | 'LOW'
    line: int
    evidence: str
    recommendation: str


@dataclass
class CodeVerificationReport:
    """Laporan verifikasi lengkap untuk satu blok kode"""
    original_code: str
    verified_symbols: List[SymbolVerification]
    security_vulnerabilities: List[SecurityVulnerability]
    hallucination_score: float  # 0.0 = sempurna, 1.0 = semua halusinasi
    security_risk_score: float  # Jumlah poin kerentanan (HIGH = 3, MEDIUM = 2, LOW = 1)
    annotations: List[str]      # Catatan gabungan (halusinasi + keamanan)
    is_safe: bool               # True jika aman dari halusinasi dan kerentanan HIGH


# Kumpulan builtins dan stdlib untuk berbagai bahasa
BUILTINS_BY_LANG: Dict[str, Set[str]] = {
    "python": {
        "print", "len", "range", "int", "str", "float", "list", "dict", "set",
        "tuple", "bool", "type", "isinstance", "issubclass", "hasattr", "getattr",
        "setattr", "delattr", "vars", "dir", "id", "hash", "callable", "repr",
        "abs", "min", "max", "sum", "sorted", "reversed", "enumerate", "zip",
        "map", "filter", "any", "all", "round", "open", "input", "exit", "quit",
        "super", "object", "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "ImportError", "RuntimeError"
    },
    "javascript": {
        "console", "log", "error", "warn", "info", "dir", "JSON", "parse", "stringify",
        "Math", "abs", "floor", "ceil", "round", "min", "max", "random", "sin", "cos",
        "Object", "keys", "values", "entries", "assign", "create", "defineProperty",
        "Array", "isArray", "from", "of", "Promise", "resolve", "reject", "all",
        "setTimeout", "clearTimeout", "setInterval", "clearInterval", "process"
    },
    "cpp": {
        "std", "cout", "cin", "endl", "printf", "scanf", "malloc", "free", "new", "delete",
        "vector", "string", "map", "set", "unordered_map", "unordered_set", "shared_ptr",
        "unique_ptr", "make_shared", "make_unique", "size", "push_back", "begin", "end"
    },
    "rust": {
        "println", "print", "format", "vec", "panic", "assert", "assert_eq", "Option",
        "Some", "None", "Result", "Ok", "Err", "String", "Vec", "HashMap", "HashSet",
        "Box", "Rc", "Arc", "Mutex", "Cell", "RefCell", "unwrap", "expect", "clone"
    },
    "go": {
        "fmt", "Println", "Printf", "Print", "Errorf", "make", "new", "append", "copy",
        "delete", "len", "cap", "panic", "recover", "close", "nil", "true", "false",
        "string", "int", "float64", "bool", "map", "chan", "error"
    }
}


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dlt = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dlt, sub))
        prev_row = curr_row
    return prev_row[-1]


class SymbolVerifier:
    """
    Engine verifikasi simbol polyglot & audit keamanan untuk output LLM.
    """

    def __init__(self, max_edit_distance: int = 3):
        self.max_edit_distance = max_edit_distance
        self._repo_symbol_names: Set[str] = set()
        self._last_repo_scan: float = 0.0

    def _refresh_repo_symbols(self):
        import time
        now = time.time()
        if now - self._last_repo_scan < 60.0 and self._repo_symbol_names:
            return

        try:
            from pathlib import Path
            core_path = Path(__file__).resolve().parent.parent
            mapper = RepoMapper(str(core_path), verbose=False)
            mapper.scan(force=True)
            self._repo_symbol_names = set(mapper._symbol_index.keys())
            self._last_repo_scan = now
        except Exception:
            pass

    def _classify_symbol(self, name: str, lang: str) -> SymbolVerification:
        # Cek builtins bahasa spesifik
        lang_builtins = BUILTINS_BY_LANG.get(lang, set())
        if name in lang_builtins:
            return SymbolVerification(name=name, status="builtin", confidence=1.0)

        # Cek exact match di workspace
        if name in self._repo_symbol_names:
            return SymbolVerification(name=name, status="verified", confidence=1.0)

        # Fuzzy match Levenshtein
        best_match: Optional[str] = None
        best_dist = self.max_edit_distance + 1
        for sym in self._repo_symbol_names:
            dist = _levenshtein(name.lower(), sym.lower())
            if dist < best_dist:
                best_dist = dist
                best_match = sym

        if best_match and best_dist <= self.max_edit_distance:
            confidence = 1.0 - (best_dist / (self.max_edit_distance + 1))
            return SymbolVerification(
                name=name,
                status="unknown",
                confidence=confidence,
                suggestion=best_match
            )

        return SymbolVerification(name=name, status="unknown", confidence=0.0)

    def _extract_called_symbols(self, code: str, lang: str) -> Set[str]:
        """Ekstraksi simbol yang dipanggil/di-import berdasarkan bahasa pemrograman"""
        symbols = set()
        if lang == "python":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            symbols.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            symbols.add(node.func.attr)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            symbols.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            symbols.add(node.module.split('.')[0])
                        for alias in node.names:
                            symbols.add(alias.name)
            except SyntaxError:
                lang = "javascript"  # Fallback to regex extractor

        if lang != "python":
            # Polyglot Regex Extractor
            # Menangkap pola call: name(...)
            for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', code):
                symbols.add(m.group(1))
            # Menangkap pola property access: obj.name
            for m in re.finditer(r'\.[A-Za-z_][A-Za-z0-9_]*\b', code):
                symbols.add(m.group(0)[1:])

        # Clean keywords
        keywords = {
            "if", "else", "for", "while", "do", "return", "switch", "case", "break", "continue",
            "const", "let", "var", "function", "class", "import", "export", "pub", "fn", "struct",
            "type", "func", "package", "go", "select", "unsafe", "impl", "trait", "mod", "use"
        }
        return {s for s in symbols if s not in keywords and len(s) > 2}

    def verify_code(self, code: str, language: str = "python") -> CodeVerificationReport:
        """
        Memverifikasi pemanggilan simbol serta melakukan audit keamanan statis
        pada blok kode buatan LLM.
        """
        self._refresh_repo_symbols()
        symbols_used = self._extract_called_symbols(code, language)
        
        verifications: List[SymbolVerification] = []
        vulns: List[SecurityVulnerability] = []
        annotations: List[str] = []
        unknown_count = 0
        total_meaningful = 0

        # 1. Validasi Simbol
        for sym in sorted(symbols_used):
            if sym.startswith('_'):
                continue
            total_meaningful += 1
            result = self._classify_symbol(sym, language)
            verifications.append(result)

            if result.status == "unknown":
                unknown_count += 1
                if result.suggestion:
                    annotations.append(
                        f"⚠️ Simbol '{sym}' tidak ditemukan di codebase. "
                        f"Mungkin maksudnya: '{result.suggestion}'?"
                    )
                else:
                    annotations.append(
                        f"🔴 Simbol '{sym}' tidak terdaftar di workspace (potensi halusinasi)."
                    )

        # 2. Audit Keamanan Statis (SAST)
        patterns = RepoMapper.SECURITY_PATTERNS.get(language, [])
        for pattern, vuln_name, risk, desc in patterns:
            for i, line in enumerate(code.split('\n'), 1):
                if re.search(pattern, line):
                    vulns.append(SecurityVulnerability(
                        vuln_type=vuln_name,
                        severity=risk,
                        line=i,
                        evidence=line.strip()[:100],
                        recommendation=desc
                    ))

        # 3. Hitung Skor Resiko & Halusinasi
        hallucination_score = (unknown_count / max(1, total_meaningful))
        
        security_risk_score = 0.0
        for v in vulns:
            if v.severity == "HIGH":
                security_risk_score += 3.0
                annotations.append(f"🚨 **KERENTANAN TINGGI (L{v.line})**: {v.vuln_type} — {v.recommendation}")
            elif v.severity == "MEDIUM":
                security_risk_score += 2.0
                annotations.append(f"🔶 **KERENTANAN SEDANG (L{v.line})**: {v.vuln_type} — {v.recommendation}")
            else:
                security_risk_score += 1.0
                annotations.append(f"🔹 **AUDIT KEAMANAN (L{v.line})**: {v.vuln_type} — {v.recommendation}")

        is_safe = (hallucination_score < 0.35) and (not any(v.severity == "HIGH" for v in vulns))

        return CodeVerificationReport(
            original_code=code,
            verified_symbols=verifications,
            security_vulnerabilities=vulns,
            hallucination_score=hallucination_score,
            security_risk_score=security_risk_score,
            annotations=annotations,
            is_safe=is_safe
        )

    def verify_factual_text(self, text: str) -> List[str]:
        """
        Audit heuristik untuk mendeteksi halusinasi faktual, pencampuran sains teoretis
        dengan mekanika klasik sehari-hari, atau persamaan yang tidak valid.
        """
        warnings = []
        text_lower = text.lower()

        # Rule 1: c^2 atau c² dihubungkan dengan CC motor / kapasitas silinder
        if ("c^2" in text_lower or "c²" in text_lower) and any(x in text_lower for x in ["cc motor", "kapasitas silinder", "piston", "stroke", "bore"]):
            warnings.append(
                "Persamaan 'c²' (kecepatan cahaya dikuadratkan) salah dihubungkan dengan perhitungan CC motor (kapasitas silinder). "
                "Perhitungan CC motor murni geometris: V = (π/4) * d² * s."
            )

        # Rule 2: Maxwell tensor atau relativitas dihubungkan dengan mesin pembakaran/CC motor
        if any(x in text_lower for x in ["maxwell", "relativistik", "relativitas"]) and any(x in text_lower for x in ["cc motor", "silinder motor", "kapasitas silinder"]):
            warnings.append(
                "Konsep elektromagnetik Maxwell atau relativitas Einstein salah dihubungkan dengan kapasitas mesin silinder motor klasik."
            )

        # Rule 3: cc = c^2
        if re.search(r"\bcc\s*=\s*c\^?2\b", text_lower) or re.search(r"\bcc\s*=\s*c²\b", text_lower):
            warnings.append(
                "Mendefinisikan CC (displacement silinder) sebagai c² adalah kesalahan konseptual fatal."
            )

        # Rule 4: Volume clearance / ruang bakar ditambahkan ke CC motor
        if any(x in text_lower for x in ["cc motor", "kapasitas silinder", "kapasitas mesin", "displacement"]) and \
           any(x in text_lower for x in ["clearance", "ruang bakar", "tma", "tmb"]) and \
           any(x in text_lower for x in ["tambah", "ditambah", "ditambahkan", "ditambah v_", "+ v_", "+v_", "ditambah volume"]):
            warnings.append(
                "Kapasitas silinder/CC motor (displacement / swept volume) hanya dihitung berdasarkan volume langkah piston (bore dan stroke): "
                "V = (π/4) * d² * s * N. Volume clearance / ruang bakar TIDAK boleh ditambahkan ke dalam perhitungan kapasitas silinder/CC motor."
            )

        return warnings

    def annotate_response(self, response_text: str) -> Tuple[str, float]:
        """
        Scan seluruh teks respons:
        1. Temukan dan audit blok kode multi-bahasa.
        2. Audit fakta dan sains teoretis pada teks respons.
        3. Tambahkan anotasi peringatan yang sesuai.
        """
        # Tangkap blok kode beserta bahasa pemrograman-nya
        code_blocks = re.findall(r'```([A-Za-z0-9+#-]+)?\s*\n(.*?)```', response_text, re.DOTALL)
        
        total_score = 0.0
        annotated = response_text
        block_count = 0

        # 1. Audit Blok Kode
        for i, (lang_tag, block) in enumerate(code_blocks, 1):
            lang = (lang_tag or "python").lower()
            # Normalisasi tag bahasa
            if lang in ["js", "jsx", "ts", "tsx"]:
                lang = "javascript"
            elif lang in ["c", "cpp", "h", "hpp"]:
                lang = "cpp"
            elif lang in ["rs"]:
                lang = "rust"
            elif lang in ["go"]:
                lang = "go"
            elif lang in ["sql"]:
                lang = "sql"
            else:
                lang = "python"

            report = self.verify_code(block, lang)
            total_score += report.hallucination_score
            block_count += 1

            if report.annotations:
                annotation_text = "\n".join(report.annotations)
                warning_box = (
                    f"\n> 🛡️ **MOKO Polyglot Security & Symbol Verifier** — Blok Kode #{i} ({lang})\n"
                    f"> Hallucination Score: {report.hallucination_score:.0%} | Risk Poin: {report.security_risk_score:.1f}\n"
                    f"> {annotation_text.replace(chr(10), chr(10)+'> ')}\n"
                )
                tag_prefix = f"```{lang_tag}" if lang_tag else "```"
                annotated = annotated.replace(
                    f"{tag_prefix}\n{block}```",
                    f"{tag_prefix}\n{block}```{warning_box}",
                    1
                )

        avg_score = total_score / max(1, block_count) if block_count > 0 else 0.0

        # 2. Audit Fakta/Sains
        factual_warnings = self.verify_factual_text(response_text)
        if factual_warnings:
            warning_text = "\n".join(f"- {w}" for w in factual_warnings)
            factual_box = (
                f"\n\n---\n"
                f"🛡️ **[MOKO Factual & Scientific Auditor]** Temuan Inkonsistensi Sains/Fakta:\n"
                f"{warning_text}\n"
                f"Harap verifikasi ulang kalkulasi/konsep di atas."
            )
            annotated += factual_box
            # Tingkatkan score halusinasi karena ada error fakta berat
            avg_score = max(avg_score, 0.50)

        return annotated, avg_score

    def build_grounding_context(self, query: str) -> str:
        self._refresh_repo_symbols()
        if not self._repo_symbol_names:
            return ""

        q_lower = query.lower()
        q_tokens = set(re.findall(r'\b\w+\b', q_lower))
        
        relevant_symbols: List[str] = []
        for sym in self._repo_symbol_names:
            sym_lower = sym.lower()
            sym_tokens = set(re.findall(r'\b\w+\b', sym_lower))
            if q_tokens & sym_tokens:
                relevant_symbols.append(sym)

        if not relevant_symbols:
            return ""

        sym_list = ", ".join(f"`{s}`" for s in sorted(relevant_symbols)[:25])
        return (
            f"\n[SYMBOL GROUNDING — Simbol yang TERSEDIA di workspace ini: {sym_list}. "
            f"GUNAKAN HANYA simbol-simbol ini saat menulis kode. "
            f"JANGAN mengarang nama fungsi/class baru yang tidak ada di daftar ini.]\n"
        )


# Singleton
_verifier_instance: Optional[SymbolVerifier] = None

def get_symbol_verifier() -> SymbolVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = SymbolVerifier()
    return _verifier_instance
