"""
MOKO Red Team Fuzzer
====================
Mesin penyerang — menghasilkan exploit nyata dari 8 kategori kerentanan
berdasarkan OWASP Top 10 dan CWE Top 25 untuk menguji SecurityAuditor.

Konsep: Seperti tim penetration tester otomatis yang tidak pernah berhenti
mencoba celah baru. Setiap exploit yang berhasil lolos menjadi bahan baku
bagi Blue Team untuk menghasilkan aturan pertahanan baru.

Referensi akademis:
  - CWE Top 25 Most Dangerous Software Weaknesses (MITRE, 2024)
  - OWASP Top 10 (2021 edition)
  - "Automated Exploit Generation" (AEG) — David Brumley et al., 2011
  - "Fuzzing: Breaking Things with Random Inputs" — Charlie Miller, 2008
"""

import re
import ast
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════
# KATEGORI EXPLOIT
# ═══════════════════════════════════════════════════════════════════════════

class ExploitCategory(Enum):
    """Kategori kerentanan berdasarkan CWE/OWASP."""
    COMMAND_INJECTION     = "CWE-78"   # OS Command Injection
    SQL_INJECTION         = "CWE-89"   # SQL Injection
    XSS                   = "CWE-79"   # Cross-Site Scripting
    BUFFER_OVERFLOW       = "CWE-120"  # Buffer Copy Without Checking Size
    PATH_TRAVERSAL        = "CWE-22"   # Path Traversal
    DESERIALIZATION       = "CWE-502"  # Deserialization of Untrusted Data
    CODE_INJECTION        = "CWE-94"   # Code Injection via eval/exec
    CREDENTIAL_LEAKAGE    = "CWE-798"  # Hardcoded Credentials


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExploitPayload:
    """Representasi satu payload exploit."""
    category: ExploitCategory
    language: str                          # 'python', 'javascript', 'c', 'cpp'
    code: str                              # Kode exploit aktual
    description: str                       # Deskripsi teknik
    expected_rule_ids: List[str]           # Rule ID yang HARUS mendeteksi ini
    severity: str                          # HIGH / MEDIUM / LOW
    cwe_id: str                            # Nomor CWE
    technique_name: str                    # Nama teknik (misal: "Backtick sub-shell")
    evasion_level: int = 1                 # 1=basic, 2=evasion, 3=advanced evasion


@dataclass
class FuzzBypass:
    """Exploit yang berhasil lolos dari SecurityAuditor (bypass terdeteksi)."""
    payload: ExploitPayload
    auditor_report: Any    # SecurityAuditReport
    bypass_reason: str     # Mengapa lolos? (rule tidak ada, pattern tidak cocok, dsb.)


@dataclass
class FuzzReport:
    """Laporan lengkap satu siklus fuzzing."""
    total_payloads: int = 0
    detected: int = 0
    bypassed: int = 0
    bypasses: List[FuzzBypass] = field(default_factory=list)
    detection_rate: float = 0.0
    category_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def finalize(self):
        """Hitung statistik akhir."""
        self.detection_rate = (self.detected / self.total_payloads * 100) if self.total_payloads > 0 else 0
        return self


# ═══════════════════════════════════════════════════════════════════════════
# RED TEAM FUZZER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RedTeamFuzzer:
    """
    Generator exploit adversarial dengan 8 kategori kerentanan.
    
    Setiap kategori menyertakan:
    - Payload level 1 (basic): Exploit paling umum dan mudah dideteksi
    - Payload level 2 (evasion): Varian dengan obfuscation ringan
    - Payload level 3 (advanced): Teknik evasion canggih (string concat, encoding)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._exploit_library = self._build_exploit_library()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔴 [RED TEAM] {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # EXPLOIT LIBRARY BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def _build_exploit_library(self) -> Dict[ExploitCategory, List[ExploitPayload]]:
        """Bangun perpustakaan exploit untuk semua 8 kategori."""
        library = {}

        # ── 1. COMMAND INJECTION (CWE-78) ─────────────────────────────
        library[ExploitCategory.COMMAND_INJECTION] = [
            ExploitPayload(
                category=ExploitCategory.COMMAND_INJECTION,
                language="python",
                code='''import subprocess
def run_cmd(user_input):
    result = subprocess.run(f"ls {user_input}", shell=True, capture_output=True)
    return result.stdout.decode()
''',
                description="F-string injection ke subprocess dengan shell=True",
                expected_rule_ids=["MOKO_SEC_SHELL_INJECTION"],
                severity="HIGH", cwe_id="CWE-78",
                technique_name="F-string sub-shell", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.COMMAND_INJECTION,
                language="python",
                code='''import os
def execute(cmd):
    os.system("ping -c 1 " + cmd)
''',
                description="String concatenation ke os.system",
                expected_rule_ids=["MOKO_SEC_OS_SYSTEM"],
                severity="HIGH", cwe_id="CWE-78",
                technique_name="os.system concat", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.COMMAND_INJECTION,
                language="python",
                code='''import subprocess
def ping_host(host):
    cmd = ["ping", "-c", "1"]
    cmd_str = " ".join(cmd) + " " + host
    # Menggunakan Popen dengan shell=True (evasion: pisahkan konstruksi command)
    proc = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
    return proc.communicate()
''',
                description="Evasion: Popen dengan shell=True lewat variabel terpisah",
                expected_rule_ids=["MOKO_SEC_SHELL_INJECTION"],
                severity="HIGH", cwe_id="CWE-78",
                technique_name="Popen variable evasion", evasion_level=2
            ),
        ]

        # ── 2. SQL INJECTION (CWE-89) ──────────────────────────────────
        library[ExploitCategory.SQL_INJECTION] = [
            ExploitPayload(
                category=ExploitCategory.SQL_INJECTION,
                language="python",
                code='''import sqlite3
def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchall()
''',
                description="F-string SQL query tanpa parameterized query",
                expected_rule_ids=["MOKO_SEC_SQL_INJECTION"],
                severity="HIGH", cwe_id="CWE-89",
                technique_name="F-string SQL concat", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.SQL_INJECTION,
                language="python",
                code='''import sqlite3
def get_product(product_id):
    conn = sqlite3.connect("store.db")
    query = "SELECT * FROM products WHERE id = " + str(product_id)
    return conn.execute(query).fetchall()
''',
                description="String + operator untuk SQL query",
                expected_rule_ids=["MOKO_SEC_SQL_INJECTION"],
                severity="HIGH", cwe_id="CWE-89",
                technique_name="String concat SQL", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.SQL_INJECTION,
                language="python",
                code='''import sqlite3
def search(keyword):
    conn = sqlite3.connect("db.sqlite")
    # Evasion: menggunakan % formatting
    q = "SELECT id, name FROM items WHERE name LIKE '%s'" % keyword
    return conn.execute(q).fetchall()
''',
                description="Evasion: %-format SQL injection",
                expected_rule_ids=["MOKO_SEC_SQL_INJECTION"],
                severity="HIGH", cwe_id="CWE-89",
                technique_name="%-format SQL", evasion_level=2
            ),
        ]

        # ── 3. XSS (CWE-79) ───────────────────────────────────────────
        library[ExploitCategory.XSS] = [
            ExploitPayload(
                category=ExploitCategory.XSS,
                language="javascript",
                code='''function displayMessage(userInput) {
    document.getElementById("output").innerHTML = userInput;
}
''',
                description="innerHTML langsung dari user input — DOM XSS",
                expected_rule_ids=["MOKO_SEC_DOM_XSS"],
                severity="MEDIUM", cwe_id="CWE-79",
                technique_name="innerHTML direct", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.XSS,
                language="javascript",
                code='''function renderTemplate(data) {
    var template = '<div>' + data.name + '</div>';
    document.body.innerHTML = template;
}
''',
                description="body.innerHTML dengan string concatenation",
                expected_rule_ids=["MOKO_SEC_DOM_XSS"],
                severity="MEDIUM", cwe_id="CWE-79",
                technique_name="body.innerHTML concat", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.XSS,
                language="javascript",
                code='''function dynamicScript(userCode) {
    eval(userCode);
}
''',
                description="eval() dari user input — Arbitrary JS execution",
                expected_rule_ids=["MOKO_SEC_JS_EVAL"],
                severity="HIGH", cwe_id="CWE-79",
                technique_name="eval user code", evasion_level=1
            ),
        ]

        # ── 4. BUFFER OVERFLOW (CWE-120) ──────────────────────────────
        library[ExploitCategory.BUFFER_OVERFLOW] = [
            ExploitPayload(
                category=ExploitCategory.BUFFER_OVERFLOW,
                language="c",
                code='''#include <string.h>
void copy_name(char *name) {
    char buf[64];
    strcpy(buf, name);  // Tidak ada batasan ukuran!
}
''',
                description="strcpy tanpa bounds checking — stack buffer overflow",
                expected_rule_ids=["MOKO_SEC_C_STRCPY"],
                severity="HIGH", cwe_id="CWE-120",
                technique_name="strcpy overflow", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.BUFFER_OVERFLOW,
                language="c",
                code='''#include <stdio.h>
void read_input() {
    char buffer[256];
    gets(buffer);  // Sangat berbahaya!
}
''',
                description="gets() — paling berbahaya karena tidak bisa dibatasi",
                expected_rule_ids=["MOKO_SEC_C_GETS"],
                severity="HIGH", cwe_id="CWE-120",
                technique_name="gets() unbounded", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.BUFFER_OVERFLOW,
                language="c",
                code='''#include <stdio.h>
void format_output(char *user_str) {
    char result[128];
    sprintf(result, "Hello: %s", user_str);  // Tidak ada batasan!
}
''',
                description="sprintf tanpa snprintf — format string overflow",
                expected_rule_ids=["MOKO_SEC_C_SPRINTF"],
                severity="MEDIUM", cwe_id="CWE-120",
                technique_name="sprintf unbounded", evasion_level=1
            ),
        ]

        # ── 5. PATH TRAVERSAL (CWE-22) ────────────────────────────────
        library[ExploitCategory.PATH_TRAVERSAL] = [
            ExploitPayload(
                category=ExploitCategory.PATH_TRAVERSAL,
                language="python",
                code='''import os
def read_file(filename):
    base_dir = "/var/www/uploads/"
    # Tidak ada validasi — user bisa kirim "../../etc/passwd"
    path = base_dir + filename
    with open(path, "r") as f:
        return f.read()
''',
                description="Path concatenation tanpa normalisasi — directory traversal",
                expected_rule_ids=["MOKO_SEC_PATH_TRAVERSAL"],
                severity="HIGH", cwe_id="CWE-22",
                technique_name="Direct path concat", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.PATH_TRAVERSAL,
                language="python",
                code='''import os
def serve_file(user_path):
    # Evasion: menggunakan os.path.join tapi tanpa validasi absolute path
    full_path = os.path.join("/var/www", user_path)
    return open(full_path).read()
''',
                description="Evasion: os.path.join tanpa realpath validation",
                expected_rule_ids=["MOKO_SEC_PATH_TRAVERSAL"],
                severity="HIGH", cwe_id="CWE-22",
                technique_name="os.path.join evasion", evasion_level=2
            ),
        ]

        # ── 6. DESERIALIZATION (CWE-502) ──────────────────────────────
        library[ExploitCategory.DESERIALIZATION] = [
            ExploitPayload(
                category=ExploitCategory.DESERIALIZATION,
                language="python",
                code='''import pickle
def load_data(serialized_bytes):
    # SANGAT berbahaya: pickle.loads dari user input memungkinkan RCE
    return pickle.loads(serialized_bytes)
''',
                description="pickle.loads dari user input — Remote Code Execution",
                expected_rule_ids=["MOKO_SEC_PICKLE_DESERIAL"],
                severity="HIGH", cwe_id="CWE-502",
                technique_name="pickle.loads RCE", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.DESERIALIZATION,
                language="python",
                code='''import yaml
def parse_config(config_str):
    # yaml.load tanpa Loader — memungkinkan arbitrary object instantiation
    return yaml.load(config_str)
''',
                description="yaml.load tanpa Loader aman — YAML deserialization",
                expected_rule_ids=["MOKO_SEC_YAML_DESERIAL"],
                severity="MEDIUM", cwe_id="CWE-502",
                technique_name="yaml.load no Loader", evasion_level=1
            ),
        ]

        # ── 7. CODE INJECTION (CWE-94) ────────────────────────────────
        library[ExploitCategory.CODE_INJECTION] = [
            ExploitPayload(
                category=ExploitCategory.CODE_INJECTION,
                language="python",
                code='''def calculate(expression):
    # eval dari user input — Code Injection
    return eval(expression)
''',
                description="eval() langsung dari user input",
                expected_rule_ids=["MOKO_SEC_DYNAMIC_EXEC"],
                severity="HIGH", cwe_id="CWE-94",
                technique_name="eval() direct", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.CODE_INJECTION,
                language="python",
                code='''def run_plugin(plugin_code):
    # compile + exec dari string — arbitrary code execution
    compiled = compile(plugin_code, "<string>", "exec")
    exec(compiled)
''',
                description="compile() + exec() dari user string",
                expected_rule_ids=["MOKO_SEC_DYNAMIC_EXEC"],
                severity="HIGH", cwe_id="CWE-94",
                technique_name="compile+exec combo", evasion_level=2
            ),
            ExploitPayload(
                category=ExploitCategory.CODE_INJECTION,
                language="python",
                code='''import builtins
def safe_eval(code_str):
    # Evasion: mencoba menggunakan __builtins__ secara eksplisit
    return eval(code_str, {"__builtins__": builtins})
''',
                description="Evasion: eval dengan explicit builtins (masih berbahaya)",
                expected_rule_ids=["MOKO_SEC_DYNAMIC_EXEC"],
                severity="HIGH", cwe_id="CWE-94",
                technique_name="eval builtins bypass attempt", evasion_level=3
            ),
        ]

        # ── 8. CREDENTIAL LEAKAGE (CWE-798) ───────────────────────────
        library[ExploitCategory.CREDENTIAL_LEAKAGE] = [
            ExploitPayload(
                category=ExploitCategory.CREDENTIAL_LEAKAGE,
                language="python",
                code='''import requests
API_KEY = "sk-1234567890abcdef1234567890abcdef"
def call_api(endpoint):
    return requests.get(endpoint, headers={"Authorization": f"Bearer {API_KEY}"})
''',
                description="API key hardcoded langsung dalam kode sumber",
                expected_rule_ids=["MOKO_SEC_HARDCODED_SECRET"],
                severity="HIGH", cwe_id="CWE-798",
                technique_name="Hardcoded API key", evasion_level=1
            ),
            ExploitPayload(
                category=ExploitCategory.CREDENTIAL_LEAKAGE,
                language="python",
                code='''import psycopg2
def get_db():
    password = "super_secret_db_pass_2024!"
    return psycopg2.connect(
        host="localhost", database="mydb",
        user="admin", password=password
    )
''',
                description="Password database hardcoded dalam variabel",
                expected_rule_ids=["MOKO_SEC_HARDCODED_SECRET"],
                severity="HIGH", cwe_id="CWE-798",
                technique_name="Hardcoded DB password", evasion_level=1
            ),
        ]

        return library

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def get_all_payloads(self) -> List[ExploitPayload]:
        """Dapatkan semua payload dari semua kategori."""
        all_payloads = []
        for category, payloads in self._exploit_library.items():
            all_payloads.extend(payloads)
        return all_payloads

    def get_payloads_by_category(self, category: ExploitCategory) -> List[ExploitPayload]:
        """Dapatkan payload untuk kategori tertentu."""
        return self._exploit_library.get(category, [])

    def get_payloads_by_evasion_level(self, level: int) -> List[ExploitPayload]:
        """Dapatkan payload berdasarkan level evasion (1/2/3)."""
        return [p for payloads in self._exploit_library.values()
                for p in payloads if p.evasion_level == level]

    def fuzz_against_auditor(self, auditor) -> FuzzReport:
        """
        Jalankan seluruh perpustakaan exploit terhadap SecurityAuditor.
        Hasilkan laporan lengkap termasuk semua bypass yang terdeteksi.
        """
        report = FuzzReport()
        all_payloads = self.get_all_payloads()
        report.total_payloads = len(all_payloads)

        self._log(f"Memulai fuzzing campaign: {report.total_payloads} payload terhadap auditor...")

        for payload in all_payloads:
            cat_name = payload.category.name
            if cat_name not in report.category_breakdown:
                report.category_breakdown[cat_name] = {"detected": 0, "bypassed": 0}

            # Tentukan filename berdasarkan language
            ext_map = {"python": "test.py", "javascript": "test.js",
                       "c": "test.c", "cpp": "test.cpp"}
            filename = ext_map.get(payload.language, "test.txt")

            # Jalankan audit
            audit_result = auditor.audit_code(payload.code, filename)

            # Periksa apakah minimal satu expected rule terdeteksi
            detected_rule_ids = {v.rule_id for v in audit_result.violations}
            expected_detected = any(rid in detected_rule_ids for rid in payload.expected_rule_ids)

            if not expected_detected and audit_result.ok:
                # BYPASS: exploit lolos dari auditor!
                bypass_reason = (
                    f"Rule {payload.expected_rule_ids} tidak ada atau tidak cocok. "
                    f"Detected rules: {detected_rule_ids if detected_rule_ids else 'NONE'}"
                )
                report.bypasses.append(FuzzBypass(
                    payload=payload,
                    auditor_report=audit_result,
                    bypass_reason=bypass_reason
                ))
                report.bypassed += 1
                report.category_breakdown[cat_name]["bypassed"] += 1
                self._log(f"⚠️  BYPASS [{payload.category.name}] [{payload.technique_name}]: {bypass_reason}")
            else:
                report.detected += 1
                report.category_breakdown[cat_name]["detected"] += 1
                self._log(f"✅ DETECTED [{payload.category.name}] [{payload.technique_name}]")

        report.finalize()
        self._log(f"Campaign selesai: {report.detected}/{report.total_payloads} terdeteksi "
                  f"({report.detection_rate:.1f}%). {report.bypassed} bypass ditemukan.")
        return report

    def generate_summary(self, report: FuzzReport) -> str:
        """Hasilkan laporan teks yang ringkas dan informatif."""
        lines = [
            "=" * 60,
            "  🔴 RED TEAM FUZZING REPORT",
            "=" * 60,
            f"  Total Payloads : {report.total_payloads}",
            f"  Detected       : {report.detected} ({report.detection_rate:.1f}%)",
            f"  Bypassed       : {report.bypassed}",
            "",
            "  Per-Category Breakdown:",
        ]
        for cat, stats in report.category_breakdown.items():
            total = stats['detected'] + stats['bypassed']
            lines.append(f"    [{cat}] {stats['detected']}/{total} detected")

        if report.bypasses:
            lines += ["", "  ⚠️  BYPASS DETAILS:"]
            for bypass in report.bypasses:
                lines.append(f"    → [{bypass.payload.category.name}] "
                             f"Evasion L{bypass.payload.evasion_level}: "
                             f"{bypass.payload.technique_name}")
                lines.append(f"      Reason: {bypass.bypass_reason[:80]}...")
        lines.append("=" * 60)
        return "\n".join(lines)
