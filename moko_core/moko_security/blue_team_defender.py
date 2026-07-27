"""
MOKO Blue Team Adaptive Defender
==================================
Sistem pertahanan yang BELAJAR dari bypass Red Team.

Konsep inti:
  Setiap kali Red Team berhasil lolos dari SecurityAuditor (bypass),
  Blue Team menganalisis exploit tersebut, mengekstrak pola berbahaya,
  menghasilkan aturan keamanan baru, memverifikasi aturan dengan bukti
  matematis, lalu HOTPATCH aturan baru ke SecurityAuditor TANPA restart.

  Ini adalah implementasi "Adversarial Self-Play" dalam domain keamanan —
  konsep yang digunakan AlphaGo untuk menjadi pemain terkuat dengan
  bermain melawan dirinya sendiri, diterapkan ke keamanan kode.

Teknik adaptive rule generation:
  1. Token Extraction     : Identifikasi literal paling khas dari exploit
  2. Pattern Generalization: Buat regex yang menangkap varian exploit
  3. Anti-Overfitting     : Uji terhadap kode bersih (tidak boleh false-positive)
  4. Z3 Proof Verification: Verifikasi formal sebelum commit ke chain
  5. Live Hotpatching     : Inject rule ke SecurityAuditor yang berjalan

Referensi:
  - "Reinforcement Learning from Human Feedback" (RLHF) — Christiano et al., 2017
    (diaplikasikan ke security: reward = exploit terdeteksi)
  - "Program Synthesis for Security" — Gulwani, 2011
  - NIST SP 800-53 Adaptive Baseline Security Controls
"""

import re
import ast
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

# Hindari circular import
if TYPE_CHECKING:
    from ..moko_marathon.security_auditor import SecurityAuditor

from .red_team_fuzzer import FuzzBypass, FuzzReport, ExploitCategory
from .crypto_chain import CryptoChain, SecurityRule, Z3ProofWitness


# ═══════════════════════════════════════════════════════════════════════════
# RULE GENERATION RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DefenseResult:
    """Hasil satu siklus pertahanan Blue Team."""
    bypass: FuzzBypass
    new_rule: Optional[SecurityRule]
    rule_accepted: bool       # True jika proof berhasil dan rule ditambahkan ke chain
    rejection_reason: str = ""
    block_index: int = -1    # Index block di chain jika diterima


@dataclass
class DefenseReport:
    """Laporan lengkap satu siklus pertahanan."""
    total_bypasses_analyzed: int = 0
    rules_generated: int = 0
    rules_accepted: int = 0
    rules_rejected: int = 0
    defense_results: List[DefenseResult] = field(default_factory=list)
    new_rule_ids: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE RULE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveRuleGenerator:
    """
    Menghasilkan SecurityRule dari analisis bypass exploit.
    
    Proses generasi menggunakan kombinasi teknik:
    - AST analysis (untuk Python): Identifikasi node AST yang berbahaya
    - Token extraction: Cari keyword paling khas dari exploit
    - Pattern generalization: Buat regex yang robust (tidak terlalu spesifik)
    """

    # Counter untuk ID rule unik
    _rule_counter: int = 0

    # Template rule untuk tiap kategori exploit yang belum ada coverage-nya
    _CATEGORY_RULE_TEMPLATES = {
        ExploitCategory.SQL_INJECTION: {
            "patterns": [
                # F-string SQL
                (r'f["\'].*SELECT.*\{', "python", "HIGH",
                 "SQL query dibangun dengan f-string (interpolasi) — risiko SQL Injection"),
                # String + concat SQL
                (r'".*SELECT.*"\s*\+', "python", "HIGH",
                 "SQL query dibangun dengan string concatenation — risiko SQL Injection"),
                # %-format SQL
                (r'".*SELECT.*"\s*%', "python", "HIGH",
                 "SQL query dibangun dengan %-formatting — risiko SQL Injection"),
                # .execute() dengan dynamic string
                (r'\.execute\s*\(\s*f["\']', "python", "HIGH",
                 "cursor.execute() dengan f-string langsung — risiko SQL Injection"),
            ],
            "cwe": "CWE-89"
        },
        ExploitCategory.PATH_TRAVERSAL: {
            "patterns": [
                # Direct path concatenation
                (r'open\s*\(\s*\w+\s*\+\s*\w+', "python", "HIGH",
                 "open() dengan path yang dikonstruksi dari concatenation — risiko Path Traversal"),
                # os.path.join tanpa realpath check
                (r'os\.path\.join\s*\(', "python", "MEDIUM",
                 "os.path.join() tanpa validasi realpath — potensi Path Traversal jika input user"),
                # Direct base_dir + filename
                (r'base_dir\s*\+\s*\w+', "python", "HIGH",
                 "Konstruksi path langsung dengan base_dir + variable — risiko Path Traversal"),
            ],
            "cwe": "CWE-22"
        },
        ExploitCategory.DESERIALIZATION: {
            "patterns": [
                # pickle.loads
                (r'pickle\s*\.\s*loads\s*\(', "python", "HIGH",
                 "pickle.loads() dari input eksternal — risiko Remote Code Execution"),
                # yaml.load tanpa Loader
                (r'yaml\s*\.\s*load\s*\([^,)]+\)', "python", "MEDIUM",
                 "yaml.load() tanpa Loader aman — gunakan yaml.safe_load()"),
                # marshal.loads
                (r'marshal\s*\.\s*loads\s*\(', "python", "HIGH",
                 "marshal.loads() dari input eksternal — risiko code execution"),
            ],
            "cwe": "CWE-502"
        },
    }

    @classmethod
    def _next_rule_id(cls) -> str:
        cls._rule_counter += 1
        return f"MOKO_SEC_ADAPTIVE_{cls._rule_counter:04d}"

    def generate_rule_for_bypass(self, bypass: FuzzBypass) -> Optional[SecurityRule]:
        """
        Analisis satu bypass dan hasilkan rule keamanan baru.
        
        Strategi:
        1. Cek apakah kategori exploit sudah punya template pattern baru
        2. Jika ada, ambil pattern dari template
        3. Jika tidak, ekstrak pattern secara heuristik dari kode exploit
        
        Returns:
            SecurityRule baru, atau None jika gagal menghasilkan rule valid
        """
        category = bypass.payload.category
        exploit_code = bypass.payload.code
        language = bypass.payload.language

        # ── Strategi 1: Gunakan template yang sudah disiapkan ──────────
        if category in self._CATEGORY_RULE_TEMPLATES:
            template = self._CATEGORY_RULE_TEMPLATES[category]
            for pattern, lang, severity, description in template["patterns"]:
                if lang == language:
                    # Verifikasi pattern ini benar-benar mendeteksi exploit
                    try:
                        if re.search(pattern, exploit_code):
                            return SecurityRule(
                                rule_id=self._next_rule_id(),
                                pattern=pattern,
                                language=language,
                                severity=severity,
                                description=description,
                                cwe_id=template["cwe"],
                                generated_from_bypass=bypass.payload.technique_name,
                            )
                    except re.error:
                        continue

        # ── Strategi 2: Heuristik — ekstrak token paling khas ──────────
        return self._heuristic_rule_generation(bypass)

    def _heuristic_rule_generation(self, bypass: FuzzBypass) -> Optional[SecurityRule]:
        """
        Hasilkan rule secara heuristik dari analisis token exploit.
        
        Teknik: Temukan identifier dan function call yang paling khas
        dalam exploit, lalu buat regex yang mendeteksinya.
        """
        exploit_code = bypass.payload.code
        language = bypass.payload.language

        # ── Untuk Python: gunakan AST untuk ekstrak function calls berbahaya
        if language == "python":
            return self._python_ast_heuristic(bypass)

        # ── Untuk bahasa lain: ekstrak token berbahaya via regex ─────
        return self._token_extraction_heuristic(bypass)

    def _python_ast_heuristic(self, bypass: FuzzBypass) -> Optional[SecurityRule]:
        """Analisis AST Python untuk menemukan pattern berbahaya yang unik."""
        exploit_code = bypass.payload.code

        try:
            tree = ast.parse(exploit_code)
        except SyntaxError:
            return self._token_extraction_heuristic(bypass)

        # Cari function call yang paling khas
        dangerous_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Dapatkan nama function
                if isinstance(node.func, ast.Attribute):
                    call_name = f"{node.func.value.id if isinstance(node.func.value, ast.Name) else '?'}.{node.func.attr}"
                    dangerous_calls.append((call_name, node.lineno))
                elif isinstance(node.func, ast.Name):
                    dangerous_calls.append((node.func.id, node.lineno))

        if not dangerous_calls:
            return None

        # Ambil function call pertama yang muncul sebagai basis pattern
        primary_call, line_no = dangerous_calls[0]
        primary_func = primary_call.split('.')[-1]  # Ambil method name saja

        # Buat pattern regex yang generik untuk function call tersebut
        escaped_func = re.escape(primary_func)
        pattern = rf'\b{escaped_func}\s*\('

        return SecurityRule(
            rule_id=self._next_rule_id(),
            pattern=pattern,
            language="python",
            severity=bypass.payload.severity,
            description=(
                f"Penggunaan '{primary_func}()' terdeteksi dari analisis bypass "
                f"teknik '{bypass.payload.technique_name}'. "
                f"CWE: {bypass.payload.cwe_id}."
            ),
            cwe_id=bypass.payload.cwe_id,
            generated_from_bypass=bypass.payload.technique_name,
        )

    def _token_extraction_heuristic(self, bypass: FuzzBypass) -> Optional[SecurityRule]:
        """Ekstrak token paling khas dari exploit via regex token analysis."""
        exploit_code = bypass.payload.code

        # Cari identifier berbahaya yang unik (kata terpanjang yang bukan keyword umum)
        common_words = {
            'import', 'from', 'def', 'class', 'return', 'if', 'else',
            'for', 'while', 'with', 'try', 'except', 'pass', 'None',
            'True', 'False', 'and', 'or', 'not', 'in', 'is', 'self',
            'print', 'len', 'str', 'int', 'list', 'dict', 'set', 'open',
            'function', 'var', 'const', 'let', 'document', 'window',
            'include', 'void', 'char', 'int', 'return', 'NULL',
        }

        # Ekstrak semua identifier dari exploit
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', exploit_code)
        unique_ids = [id_ for id_ in identifiers if id_ not in common_words]

        if not unique_ids:
            return None

        # Ambil identifier yang paling sering muncul
        from collections import Counter
        id_counts = Counter(unique_ids)
        most_common = id_counts.most_common(3)

        if not most_common:
            return None

        # Pilih identifier sebagai basis pattern
        target_id = most_common[0][0]
        pattern = rf'\b{re.escape(target_id)}\s*\('

        # Tentukan language extension
        lang_map = {"python": "python", "javascript": "javascript",
                    "c": "c", "cpp": "cpp"}

        return SecurityRule(
            rule_id=self._next_rule_id(),
            pattern=pattern,
            language=lang_map.get(bypass.payload.language, "python"),
            severity=bypass.payload.severity,
            description=(
                f"Token berbahaya '{target_id}' terdeteksi dari analisis "
                f"bypass '{bypass.payload.technique_name}'. CWE: {bypass.payload.cwe_id}."
            ),
            cwe_id=bypass.payload.cwe_id,
            generated_from_bypass=bypass.payload.technique_name,
        )


# ═══════════════════════════════════════════════════════════════════════════
# BLUE TEAM DEFENDER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BlueTeamDefender:
    """
    Mesin pertahanan adaptif — belajar dari serangan Red Team.
    
    Workflow per round:
    1. Terima FuzzReport dari RedTeamFuzzer
    2. Untuk setiap bypass:
       a. AdaptiveRuleGenerator menganalisis exploit → hasilkan SecurityRule baru
       b. CryptoChain.add_rule() → verifikasi proof + commit ke ledger
       c. SecurityAuditor.register_adaptive_rule() → live hotpatch
    3. Return DefenseReport dengan statistik evolusi keamanan
    """

    def __init__(self, crypto_chain: CryptoChain, verbose: bool = True):
        self.chain = crypto_chain
        self.rule_generator = AdaptiveRuleGenerator()
        self.verbose = verbose
        self._hotpatched_rules: List[SecurityRule] = []  # History semua rule yang di-inject

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🔵 [BLUE TEAM] {msg}")

    def run_defense_round(self, fuzz_report: FuzzReport, auditor=None) -> DefenseReport:
        """
        Jalankan satu siklus pertahanan terhadap semua bypass yang ditemukan.
        
        Args:
            fuzz_report: Laporan dari RedTeamFuzzer.fuzz_against_auditor()
            auditor:     SecurityAuditor instance untuk di-hotpatch (opsional)
        
        Returns:
            DefenseReport dengan detail semua rule yang dihasilkan
        """
        report = DefenseReport()
        report.total_bypasses_analyzed = len(fuzz_report.bypasses)

        if not fuzz_report.bypasses:
            self._log("Tidak ada bypass untuk dianalisis. Pertahanan sudah kuat! 💪")
            return report

        self._log(f"Menganalisis {len(fuzz_report.bypasses)} bypass dari Red Team...")

        for bypass in fuzz_report.bypasses:
            self._log(f"Menganalisis bypass: [{bypass.payload.category.name}] "
                      f"'{bypass.payload.technique_name}'...")

            # ── Step 1: Generate rule dari bypass ─────────────────────
            new_rule = self.rule_generator.generate_rule_for_bypass(bypass)
            report.rules_generated += 1

            if new_rule is None:
                result = DefenseResult(
                    bypass=bypass,
                    new_rule=None,
                    rule_accepted=False,
                    rejection_reason="Gagal menghasilkan pattern yang valid dari exploit ini",
                )
                report.rules_rejected += 1
                report.defense_results.append(result)
                self._log(f"  ⚠️  Gagal hasilkan rule untuk '{bypass.payload.technique_name}'")
                continue

            # ── Step 2: Commit rule ke Crypto Chain ─────────────────
            try:
                block = self.chain.add_rule(new_rule, bypass.payload.code)
                rule_accepted = True
                rejection_reason = ""
                block_index = block.index
                self._log(
                    f"  ✅ Rule '{new_rule.rule_id}' diterima → Block #{block_index} "
                    f"[{new_rule.severity}] Pattern: {new_rule.pattern[:40]}..."
                )
            except ValueError as e:
                rule_accepted = False
                rejection_reason = str(e)
                block_index = -1
                self._log(f"  ❌ Rule '{new_rule.rule_id}' ditolak chain: {rejection_reason[:60]}...")

            # ── Step 3: Hotpatch SecurityAuditor jika rule diterima ──
            if rule_accepted and auditor is not None:
                try:
                    self._hotpatch_auditor(auditor, new_rule)
                    self._hotpatched_rules.append(new_rule)
                    self._log(f"  💉 SecurityAuditor di-hotpatch dengan '{new_rule.rule_id}'")
                except Exception as e:
                    self._log(f"  ⚠️  Hotpatch gagal: {e}")

            result = DefenseResult(
                bypass=bypass,
                new_rule=new_rule,
                rule_accepted=rule_accepted,
                rejection_reason=rejection_reason,
                block_index=block_index,
            )
            if rule_accepted:
                report.rules_accepted += 1
                report.new_rule_ids.append(new_rule.rule_id)
            else:
                report.rules_rejected += 1
            report.defense_results.append(result)

        self._log(
            f"Defense round selesai: {report.rules_accepted} rule baru diterima, "
            f"{report.rules_rejected} ditolak."
        )
        return report

    def _hotpatch_auditor(self, auditor, rule: SecurityRule):
        """
        Live injection rule baru ke SecurityAuditor tanpa restart.
        
        Ini adalah "zero-downtime security upgrade" — seperti sistem imun
        tubuh yang menghasilkan antibodi baru tanpa menghentikan fungsi
        sistem lainnya.
        """
        if hasattr(auditor, 'register_adaptive_rule'):
            auditor.register_adaptive_rule(rule)
        else:
            # Fallback: inject langsung ke pattern list yang relevan
            import re
            if rule.language == "python":
                # Tambahkan ke secret_patterns dengan format yang sesuai
                auditor.secret_patterns.append((rule.pattern, rule.rule_id))
            elif rule.language in ("javascript", "js"):
                auditor.js_patterns.append(
                    (rule.pattern, rule.rule_id, rule.severity, rule.description)
                )
            elif rule.language in ("c", "cpp"):
                auditor.c_patterns.append(
                    (rule.pattern, rule.rule_id, rule.severity, rule.description)
                )

    def get_evolution_summary(self) -> str:
        """Hasilkan laporan evolusi pertahanan sejak awal."""
        rules = self.chain.get_all_rules()
        lines = [
            "=" * 60,
            "  🔵 BLUE TEAM — DEFENSE EVOLUTION SUMMARY",
            "=" * 60,
            f"  Total adaptive rules: {len(rules)}",
            f"  Hotpatched rules    : {len(self._hotpatched_rules)}",
            f"  Chain blocks        : {self.chain.get_chain_length()}",
            "",
            "  Adaptive Rules Generated:",
        ]
        for rule in rules:
            lines.append(
                f"    [{rule.severity}] {rule.rule_id} — {rule.description[:50]}..."
            )
        lines.append("=" * 60)
        return "\n".join(lines)
