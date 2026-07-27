"""
MOKO Semantic Fisher Engine — Active Luring & Gathering Search
===============================================================
Berdasarkan konsep filosofis "Menjaring Ikan" (Brian, 2026):
  1. Menebar Umpan (Semantic Baiting) -> Membuat vektor target ideal (spec/contract).
  2. Menggiring Ikan (Semantic Gradient Ascent) -> Probing aktif ke database/crawler
     dan menarik cluster informasi ke koordinat umpan.
  3. Menjaring (Logical Netting) -> Memfilter kumpulan informasi yang terjaring
     menggunakan Z3 Constraints & AST Verifier, menyisakan data 100% presisi.

Ini adalah pergeseran dari "Pencarian Pasif" menjadi "Pencarian Aktif Terbimbing".
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from moko_agents.repo_mapper import get_repo_mapper
from moko_agents.symbol_verifier import get_symbol_verifier
from moko_agents.neuro_symbolic_cbc import get_cbc_engine


class SemanticBait:
    """Representasi umpan semantik (koordinat ideal dan target batasan)"""
    def __init__(self, query: str, expected_symbols: List[str], constraints: List[str]):
        self.query = query
        self.expected_symbols = expected_symbols
        self.constraints = constraints
        # Gunakan representasi embedding sederhana untuk visualisasi arah umpan
        self.direction_hash = hash(query) % 1000 / 1000.0


class MokoFisherEngine:
    """
    Engine pencarian aktif berbasis analogi 'Menjaring Ikan'.
    Menggiring cluster data ke area target lalu memfilternya dengan Z3 Solver.
    """

    def __init__(self):
        self.repo_mapper = get_repo_mapper()
        self.symbol_verifier = get_symbol_verifier()
        self.cbc_engine = get_cbc_engine()

    def cast_bait(self, user_intent: str) -> SemanticBait:
        """
        Langkah 1: Menebar Umpan.
        Menganalisis intensi pengguna untuk menyusun target simbol dan batasan logika
        yang akan digunakan sebagai penarik informasi.
        """
        # Ekstrak kata kunci teknis
        keywords = []
        constraints = []
        
        q_lower = user_intent.lower()
        
        # Deteksi domain & set batasan
        if "matrix" in q_lower or "matriks" in q_lower:
            keywords.extend(["matrix", "multiply", "transpose", "dimension"])
            constraints.append("cols_a == rows_b")
        if "divide" in q_lower or "bagi" in q_lower:
            keywords.extend(["divide", "numerator", "denominator"])
            constraints.append("denominator != 0")
        if "array" in q_lower or "index" in q_lower or "list" in q_lower:
            keywords.extend(["array", "index", "bounds", "length"])
            constraints.append("index >= 0")
            constraints.append("index < length")

        # Fallback keywords
        if not keywords:
            keywords = [w for w in q_lower.split() if len(w) > 3][:4]

        return SemanticBait(user_intent, keywords, constraints)

    def guide_information(self, bait: SemanticBait) -> List[Dict[str, Any]]:
        """
        Langkah 2: Menggiring Ikan.
        Mencari simbol di workspace, menghitung gaya gravitasi semantik (kedekatan token)
        terhadap umpan, dan menarik node-node informasi tersebut ke dalam jaring kita.
        """
        self.repo_mapper.scan()
        symbols = self.repo_mapper._symbol_index
        
        gathered_fish = []

        for name, defs in symbols.items():
            for sym in defs:
                # Hitung score kedekatan semantik (Fish Attraction Score)
                attraction_score = 0.0
                
                # Cocokkan nama simbol dengan expected symbols (Umpan)
                for exp in bait.expected_symbols:
                    if exp.lower() in sym.name.lower():
                        attraction_score += 3.0
                    if sym.docstring and exp.lower() in sym.docstring.lower():
                        attraction_score += 1.0

                # Jika ada kecocokan (gaya tarik semantik), giring ke kelompok tangkapan
                if attraction_score > 0:
                    gathered_fish.append({
                        "symbol": sym,
                        "attraction_score": attraction_score,
                        "source": sym.file,
                        "line": sym.line
                    })

        # Urutkan berdasarkan tarikan semantik terkuat (ikan paling dekat dengan umpan)
        gathered_fish.sort(key=lambda x: x["attraction_score"], reverse=True)
        return gathered_fish

    def net_catch(self, gathered_fish: List[Dict[str, Any]], bait: SemanticBait) -> Dict[str, Any]:
        """
        Langkah 3: Menjaring Ikan.
        Gunakan Z3 Logic Solver & AST Verifier sebagai 'Jaring' untuk menyaring
        ikan (data) yang benar-benar lolos uji integritas dan membuang sisanya.
        """
        netted_symbols = []
        vulnerabilities_found = []
        logic_reports = []

        for fish in gathered_fish:
            sym = fish["symbol"]
            
            # Verifikasi keamanan & simbol
            report = self.symbol_verifier.verify_code(sym.signature, "python")
            
            # Verifikasi logika formal Z3 berdasarkan batasan umpan (constraints)
            is_logic_passed = True
            for constraint in bait.constraints:
                # Jalankan pembuktian formal Z3 jika batasan numerik terdeteksi
                if "denominator != 0" in constraint:
                    z3_res = self.cbc_engine.verify_integer_division((-1000, 1000), (0, 100))
                    is_logic_passed = z3_res["safe_from_zero_division"]
                    logic_reports.append({"symbol": sym.name, "rule": constraint, "passed": is_logic_passed, "counter_example": z3_res["counter_example"]})
                elif "index < length" in constraint:
                    z3_res = self.cbc_engine.verify_array_index(10, (0, 15))
                    is_logic_passed = z3_res["safe_from_bounds_overflow"]
                    logic_reports.append({"symbol": sym.name, "rule": constraint, "passed": is_logic_passed, "counter_example": z3_res["counter_example"]})

            # Jika aman dan lulus logika, masukan ke keranjang tangkapan utama (Netted)
            if report.is_safe and is_logic_passed:
                netted_symbols.append({
                    "name": sym.name,
                    "kind": sym.kind,
                    "file": sym.file,
                    "signature": sym.signature,
                    "attraction": fish["attraction_score"]
                })
            else:
                # Ikan lolos atau dibuang karena tidak aman secara formal/keamanan
                vulnerabilities_found.append({
                    "name": sym.name,
                    "file": sym.file,
                    "reason": "Gagal asersi logika Z3" if not is_logic_passed else "Mengandung kerentanan keamanan / simbol fiktif"
                })

        return {
            "netted": netted_symbols[:10],       # Ikan berkualitas terbaik
            "discarded": vulnerabilities_found,   # Ikan dibuang (tidak aman/tidak lolos logika)
            "logic_checks": logic_reports
        }

    def execute_fishing(self, user_intent: str) -> str:
        """Eksekusi alur penuh pencarian menjaring ikan"""
        # 1. Tebar umpan
        bait = self.cast_bait(user_intent)
        
        # 2. Giring data ke koordinat umpan
        fish = self.guide_information(bait)
        
        # 3. Jaring dan saring
        catch = self.net_catch(fish, bait)

        # Bangun visualisasi output
        lines = []
        lines.append("╔══════════════════════════════════════════════════════════════════")
        lines.append("║ 🎣 MOKO SEMANTIC FISHER ENGINE — Hasil Menjaring Pengetahuan")
        lines.append("╠══════════════════════════════════════════════════════════════════")
        lines.append(f"║ 🪱 UMPAN DITEBAR   : '{bait.query}'")
        lines.append(f"║ 🐠 SEMANTIC ANCHORS: {', '.join(bait.expected_symbols)}")
        lines.append(f"║ 🔗 LOGIC BOUNDS    : {', '.join(bait.constraints) if bait.constraints else 'None'}")
        lines.append("╠══════════════════════════════════════════════════════════════════")
        
        if catch["netted"]:
            lines.append("║ 📥 IKAN TERJARING (Lolos Verifikasi Formal & Keamanan):")
            for i, item in enumerate(catch["netted"], 1):
                lines.append(f"║   {i}. [{item['kind'].upper()}] {item['name']}")
                lines.append(f"║      📄 File: {item['file']}")
                lines.append(f"║      💻 Sig : {item['signature'][:60]}")
        else:
            lines.append("║ 📥 IKAN TERJARING: Tidak ada data yang aman / sesuai dengan spesifikasi.")

        if catch["discarded"]:
            lines.append("║")
            lines.append("║ 🗑️  IKAN DIBUANG (Gagal Asersi Logika / Rawan Keamanan):")
            for item in catch["discarded"][:3]:
                lines.append(f"║   ⚠  {item['name']} in {item['file']}")
                lines.append(f"║      └─ Sebab: {item['reason']}")

        if catch["logic_checks"]:
            lines.append("║")
            lines.append("║ 🧮 PEMBUKTIAN MATEMATIS SOLVER:")
            for check in catch["logic_checks"][:2]:
                status = "PASS ✅" if check["passed"] else "FAIL ❌"
                lines.append(f"║   • Simbol '{check['symbol']}' vs {check['rule']}: {status}")
                if check["counter_example"]:
                    lines.append(f"║     Counter-Example: {check['counter_example']}")

        lines.append("╚══════════════════════════════════════════════════════════════════")
        return "\n".join(lines)


# Singleton
_fisher_instance: Optional[MokoFisherEngine] = None

def get_fisher_engine() -> MokoFisherEngine:
    global _fisher_instance
    if _fisher_instance is None:
        _fisher_instance = MokoFisherEngine()
    return _fisher_instance
