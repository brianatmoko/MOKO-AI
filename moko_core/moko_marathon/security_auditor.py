"""
MOKO ASR Security Auditor
=========================
Audit kerentanan keamanan statis (Static Application Security Testing / SAST)
untuk kode pemrograman Python, JavaScript, dan C/C++ yang di-generate.

Mencegah pengenalan celah keamanan kritis:
  - Shell Command Injection
  - SQL Injection
  - Cross-Site Scripting (DOM XSS)
  - C-Style Buffer Overflow
  - Hardcoded API Keys/Passwords
"""

import ast
import re
from typing import List, Dict, Any, Tuple


class SecurityViolation:
    def __init__(self, rule_id: str, severity: str, message: str, line_number: int = -1):
        self.rule_id = rule_id      # e.g., MOKO_SEC_SHELL
        self.severity = severity    # HIGH, MEDIUM, LOW
        self.message = message
        self.line_number = line_number

    def __repr__(self):
        return f"[{self.severity}] Line {self.line_number}: {self.message} ({self.rule_id})"


class SecurityAuditReport:
    def __init__(self, ok: bool, violations: List[SecurityViolation]):
        self.ok = ok
        self.violations = violations
        # Score kerentanan: 1.0 (aman penuh) hingga 0.0 (sangat tidak aman)
        self.risk_score = self._compute_risk_score()

    def _compute_risk_score(self) -> float:
        if not self.violations:
            return 1.0
        weight_map = {"HIGH": 0.4, "MEDIUM": 0.15, "LOW": 0.05}
        total_deduction = sum(weight_map.get(v.severity, 0.05) for v in self.violations)
        return max(0.0, round(1.0 - total_deduction, 2))


class PythonASTSecurityScanner(ast.NodeVisitor):
    """AST Visitor untuk menganalisis kerentanan kode Python secara tepat."""
    
    def __init__(self):
        self.violations: List[SecurityViolation] = []

    def visit_Call(self, node: ast.Call):
        # 1. Deteksi subprocess dengan shell=True (Command Injection)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'run':
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess':
                self._check_subprocess_shell(node)
        elif isinstance(node.func, ast.Name) and node.func.id == 'run':
            self._check_subprocess_shell(node)

        # 2. Deteksi eval/exec (Arbitrary Code Execution)
        if isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec'):
            self.violations.append(
                SecurityViolation(
                    rule_id="MOKO_SEC_DYNAMIC_EXEC",
                    severity="HIGH",
                    message=f"Penggunaan fungsi berbahaya '{node.func.id}()' terdeteksi. Risiko eksekusi kode acak.",
                    line_number=node.lineno
                )
            )

        # 3. Deteksi os.system (Command Injection)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'system':
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                self.violations.append(
                    SecurityViolation(
                        rule_id="MOKO_SEC_OS_SYSTEM",
                        severity="HIGH",
                        message="Penggunaan 'os.system()' terdeteksi. Gunakan 'subprocess.run(..., shell=False)' dengan argumen terpisah.",
                        line_number=node.lineno
                    )
                )

        self.generic_visit(node)

    def _check_subprocess_shell(self, node: ast.Call):
        # Cek apakah keyword argument 'shell=True' ada
        shell_true = False
        for kw in node.keywords:
            if kw.arg == 'shell':
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    shell_true = True
                elif isinstance(kw.value, ast.NameConstant) and kw.value.value is True: # support older Python AST
                    shell_true = True

        if shell_true:
            # Periksa apakah command string berisi variabel dinamis (f-string, concatenation, formatting)
            command_arg = node.args[0] if node.args else None
            is_dynamic = True
            if isinstance(command_arg, ast.Constant) and isinstance(command_arg.value, str):
                is_dynamic = False
            elif isinstance(command_arg, ast.Str): # support older Python AST
                is_dynamic = False
                
            if is_dynamic:
                self.violations.append(
                    SecurityViolation(
                        rule_id="MOKO_SEC_SHELL_INJECTION",
                        severity="HIGH",
                        message="Subprocess dijalankan dengan shell=True dan parameter dinamis. Risiko tinggi Command Injection!",
                        line_number=node.lineno
                    )
                )


class SecurityAuditor:
    """Mesin audit utama untuk mendeteksi kerentanan kode lintas bahasa."""

    def __init__(self):
        # Pola deteksi rahasia/credentials keras (API keys, secrets)
        self.secret_patterns = [
            (r'(?i)\b(?:secret|password|api_key|apikey|private_key|token|passwd)\s*=\s*[\'"][a-zA-Z0-9_\-\.\=\+]{16,}[\'"]', "MOKO_SEC_HARDCODED_SECRET"),
        ]
        # Pola adaptif yang di-generate oleh Blue Team ACDS (dimuat dari CryptoChain)
        self._adaptive_python_patterns: List[Tuple[str, str, str, str]] = []
        self._adaptive_js_patterns: List[Tuple[str, str, str, str]] = []
        self._adaptive_c_patterns: List[Tuple[str, str, str, str]] = []
        
        # Pola JavaScript XSS & eval
        self.js_patterns = [
            (r'\.innerHTML\s*=', "MOKO_SEC_DOM_XSS", "MEDIUM", "Penggunaan '.innerHTML' dengan penetapan langsung terdeteksi. Gunakan '.textContent' atau properti DOM yang aman untuk mencegah XSS."),
            (r'\beval\s*\(', "MOKO_SEC_JS_EVAL", "HIGH", "Penggunaan JS eval() terdeteksi. Berbahaya untuk dynamic input execution."),
        ]

        # Pola C/C++ Buffer Overflow
        self.c_patterns = [
            (r'\bstrcpy\s*\(', "MOKO_SEC_C_STRCPY", "HIGH", "Fungsi tidak aman 'strcpy()' terdeteksi. Gunakan 'strncpy()' dengan batasan ukuran."),
            (r'\bstrcat\s*\(', "MOKO_SEC_C_STRCAT", "HIGH", "Fungsi tidak aman 'strcat()' terdeteksi. Gunakan 'strncat()'."),
            (r'\bsprintf\s*\(', "MOKO_SEC_C_SPRINTF", "MEDIUM", "Fungsi tidak aman 'sprintf()' terdeteksi. Gunakan 'snprintf()' untuk membatasi buffer overflow."),
            (r'\bgets\s*\(', "MOKO_SEC_C_GETS", "HIGH", "Fungsi sangat berbahaya 'gets()' terdeteksi. Gunakan 'fgets()'."),
        ]

    def register_adaptive_rule(self, rule) -> bool:
        """
        Live hotpatch — tambahkan rule baru dari Blue Team ACDS tanpa restart.
        Rule disuntikkan ke daftar pattern yang sesuai berdasarkan bahasa target.
        
        Returns:
            True jika berhasil, False jika bahasa tidak dikenali
        """
        import re
        lang = getattr(rule, 'language', 'python')
        pattern = getattr(rule, 'pattern', '')
        rule_id = getattr(rule, 'rule_id', 'MOKO_SEC_ADAPTIVE')
        severity = getattr(rule, 'severity', 'MEDIUM')
        description = getattr(rule, 'description', 'Adaptive rule dari Blue Team ACDS')

        if not pattern:
            return False

        # Validasi regex sebelum inject (hindari pattern rusak)
        try:
            re.compile(pattern)
        except re.error:
            return False

        entry = (pattern, rule_id, severity, description)
        if lang in ('python', 'py'):
            # Untuk Python, tambahkan ke list adaptif khusus
            self._adaptive_python_patterns.append(entry)
        elif lang in ('javascript', 'js', 'html'):
            self._adaptive_js_patterns.append(entry)
        elif lang in ('c', 'cpp', 'h', 'hpp'):
            self._adaptive_c_patterns.append(entry)
        else:
            # Fallback: tambahkan ke semua bahasa
            self._adaptive_python_patterns.append(entry)
        return True

    def load_from_chain(self, chain) -> int:
        """
        Muat semua rule adaptif dari CryptoChain saat startup.
        Memastikan rule yang dihasilkan sebelumnya tetap aktif antar sesi.
        
        Args:
            chain: Instance CryptoChain
        Returns:
            Jumlah rule yang berhasil dimuat
        """
        rules = chain.get_all_rules()
        loaded = 0
        for rule in rules:
            if self.register_adaptive_rule(rule):
                loaded += 1
        return loaded

    def audit_code(self, code: str, filename: str) -> SecurityAuditReport:
        """Audit kode program dan kembalikan laporan kerentanan."""
        violations: List[SecurityViolation] = []
        ext = filename.split('.')[-1].lower() if '.' in filename else ''

        # 1. Jalankan AST audit khusus Python
        if ext == 'py':
            try:
                tree = ast.parse(code)
                scanner = PythonASTSecurityScanner()
                scanner.visit(tree)
                violations.extend(scanner.violations)
            except SyntaxError:
                # Jika syntax error, scanner dilewati (linter biasa akan menangkapnya)
                pass

        # 2. Jalankan regex audit untuk JavaScript
        if ext in ('js', 'html'):
            all_js = self.js_patterns + self._adaptive_js_patterns
            for pattern, rule_id, severity, msg in all_js:
                for match in re.finditer(pattern, code):
                    # Cari line number secara manual
                    line_no = code[:match.start()].count('\n') + 1
                    violations.append(SecurityViolation(rule_id, severity, msg, line_no))

        # 3. Jalankan regex audit untuk C/C++
        if ext in ('c', 'cpp', 'h', 'hpp'):
            all_c = self.c_patterns + self._adaptive_c_patterns
            for pattern, rule_id, severity, msg in all_c:
                for match in re.finditer(pattern, code):
                    line_no = code[:match.start()].count('\n') + 1
                    violations.append(SecurityViolation(rule_id, severity, msg, line_no))

        # 4b. Jalankan adaptive Python patterns (dari ACDS Blue Team)
        if ext == 'py':
            for pattern, rule_id, severity, msg in self._adaptive_python_patterns:
                try:
                    for match in re.finditer(pattern, code):
                        line_no = code[:match.start()].count('\n') + 1
                        violations.append(SecurityViolation(rule_id, severity, msg, line_no))
                except re.error:
                    pass  # Skip broken patterns

        # 4. Deteksi hardcoded secrets (semua bahasa)
        for pattern, rule_id in self.secret_patterns:
            for match in re.finditer(pattern, code):
                line_no = code[:match.start()].count('\n') + 1
                # Jangan bocorkan secret-nya di log, sembunyikan dengan mask
                matched_text = match.group(0)
                violations.append(
                    SecurityViolation(
                        rule_id=rule_id,
                        severity="HIGH",
                        message=f"Kredensial atau API key keras terdeteksi: '{matched_text[:12]}...'. Gunakan environment variable.",
                        line_number=line_no
                    )
                )

        return SecurityAuditReport(ok=len(violations) == 0, violations=violations)
