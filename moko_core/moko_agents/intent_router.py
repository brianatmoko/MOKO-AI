"""
MOKO Intent-First Router — 8-Class Priority Chain
===================================================
Router yang menentukan model spesialis mana yang harus menangani query.

Arsitektur:
────────────
Tier 0 (< 1ms)   : Exact match (slash commands, greetings)
Tier 1 (< 5ms)   : Rule-based keyword scoring
Tier 2 (< 20ms)  : Semantic similarity (centroid-based)
Tier 3 (< 50ms)  : Fallback to general model

8 Intent Classes:
─────────────────
1. CODING     → coding model (1.5B)
2. MATH       → math model (1.0B)
3. SECURITY   → security model (1.0B)
4. GENERAL    → general model (2.0B)
5. BROWSING   → web search + general
6. PERSONAL   → general model (fast)
7. REASONING  → general model (deep)
8. SYSTEM     → system control (no model)

Filosofi:
─────────
"Kirim ke ahli yang tepat, bukan ke semua orang."
- Confidence scoring: seberapa yakin kita intent-nya benar
- Priority chain: coba cara cepat dulu, baru cara lambat
- Domain dispatch: setiap intent → 1 model spesifik
"""

import json
import re
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from moko_agents.dispatch_types import IntentClass, DispatchManifest, TokenBudget


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════



@dataclass
class IntentResult(DispatchManifest):
    """Hasil dari intent classification."""
    confidence: float = 0.0     # 0.0 - 1.0
    reason: str = ""            # Human-readable explanation
    tier: int = 0               # Which tier made the decision (0-3)
    latency_ms: float = 0.0     # Classification time

    @property
    def intent(self) -> str:
        """Alias for intent_class.value for backward compatibility."""
        return self.intent_class.value


# ═══════════════════════════════════════════════════════════════════════════
# INTENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Intent → (model_key, domain, path)
INTENT_MAP = {
    IntentClass.CODING:    ("coding",     "code",        "DEEP_PATH"),
    IntentClass.MATH:      ("math",       "math",        "DEEP_PATH"),
    IntentClass.SECURITY:  ("security",   "cybersecurity","DEEP_PATH"),
    IntentClass.DARKWEB:   ("security",   "cyberoffensive","BROWSING_PATH"),
    IntentClass.GENERAL:   ("general",    "general",     "DEEP_PATH"),
    IntentClass.BROWSING:  ("general",    "general",     "BROWSING_PATH"),
    IntentClass.PERSONAL:  ("general",    "personal",    "FAST_PATH"),
    IntentClass.REASONING: ("general",    "general",     "DEEP_PATH"),
    IntentClass.SYSTEM:    ("general",    "system",      "COMMAND_PATH"),
}

# ═══════════════════════════════════════════════════════════════════════════
# KEYWORD DEFINITIONS (Tier 1)
# ═══════════════════════════════════════════════════════════════════════════

# Coding keywords (weight: 1.0 = strong signal)
CODING_KEYWORDS = {
    # Programming languages
    "python", "javascript", "typescript", "rust", "golang", "java", "c++",
    "c#", "ruby", "php", "swift", "kotlin", "scala", "haskell", "elixir",
    # Programming concepts
    "function", "class", "method", "variable", "array", "dictionary",
    "loop", "if else", "switch", "try catch", "import", "export",
    "module", "package", "library", "framework",
    # Development
    "code", "coding", "program", "debug", "compile", "run", "execute",
    "git", "github", "commit", "push", "pull", "merge", "branch",
    "docker", "kubernetes", "ci/cd", "devops",
    # Data structures
    "algorithm", "data structure", "linked list", "binary tree",
    "hash map", "stack", "queue", "graph", "sort", "search",
    # Web
    "api", "rest", "graphql", "database", "sql", "nosql", "mongodb",
    "redis", "frontend", "backend", "fullstack", "html", "css", "react",
    # System
    "kernel", "driver", "syscall", "process", "thread", "memory",
    "file system", "network", "socket", "protocol",
}

# Math keywords
MATH_KEYWORDS = {
    # Basic math
    "hitung", "rumus", "matematika", "matematik", "math", "calculate",
    "formula", "equation", "persamaan",
    # Algebra
    "aljabar", "algebra", "variabel", "polinomial", "kuadrat", "linear",
    "sistem persamaan", "matriks", "determinan",
    # Calculus
    "kalkulus", "calculus", "integral", "turunan", "diferensial",
    "limit", "konvergen", "divergen",
    # Geometry
    "geometri", "geometry", "segitiga", "lingkaran", "bangun ruang",
    "volume", "luas", "keliling", "diameter", "jari-jari",
    # Statistics
    "statistik", "statistics", "probabilitas", "probability",
    "mean", "median", "modus", "standar deviasi", "varians",
    # Advanced
    "teori bilangan", "number theory", "prima", "prime",
    "modular", "kriptografi", "cryptography",
    "vektor", "vector", "tensor", "matrix",
    # Physics math
    "fisika", "physics", "gaya", "kecepatan", "energi", "momentum",
    "newton", "gravitasi", "termodinamika",
}

# Security keywords
SECURITY_KEYWORDS = {
    # Offensive
    "exploit", "vulnerability", "vulnerabilities", "vuln",
    "penetration", "pentest", "penetration testing",
    "hack", "hacking", "hacker", "crack", "cracking",
    "buffer overflow", "stack overflow", "heap overflow",
    "sql injection", "sqli", "xss", "csrf", "rfi", "lfi",
    "remote code execution", "rce", "code injection",
    # Defensive
    "firewall", "ids", "ips", "siem", "soc",
    "encryption", "decryption", "cipher", "aes", "rsa",
    "hash", "sha256", "md5", "hmac",
    "authentication", "authorization", "oauth", "jwt",
    # Analysis
    "malware", "virus", "trojan", "ransomware", "rootkit",
    "reverse engineering", "disassembly", "debugger",
    "forensics", "incident response", "threat",
    "cve", "cve-", "cvss", "exploit-db",
    # Crypto
    "kriptografi", "cryptography", "cryptographic",
    "public key", "private key", "certificate", "ssl", "tls",
    "zero knowledge", "zkp", "zk-snark",
}

# Darkweb keywords
DARKWEB_KEYWORDS = {
    "onion", ".onion", "darkweb", "darknet", "deepweb",
    "tor search", "onion search", "hidden service",
    "ahmia", "torch", "haystak", "onion link",
    "dark web", "deep web", "surface web",
    "hidden wiki", "darknet market", "leak data",
    "data scanning darkweb", "search darkweb",
}

# Personal keywords
PERSONAL_KEYWORDS = {
    "nama aku", "nama saya", "namaku", "siapa saya", "siapa aku",
    "kenalin", "kenalkan", "perkenalkan",
    "aku adalah", "saya adalah", "panggil aku",
    "makasih", "terima kasih", "thanks",
    "mantap", "keren", "bagus", "hebat",
    "siap", "noted", "paham", "ngerti",
    "hai", "halo", "hello", "hi", "hey",
    "apa kabar", "selamat pagi", "selamat malam",
    "ya", "oke", "ok", "lanjut", "sip",
    "dadah", "bye", "sampai jumpa",
}

# Browsing keywords
BROWSING_KEYWORDS = {
    "bagaimana cara", "bagaimana membuat", "gimana cara",
    "cara membuat", "cara memasak", "cara menggunakan",
    "resep", "resep makanan", "resep masakan",
    "restoran", "cafe", "tempat makan", "warung",
    "berita", "berita terbaru", "headline",
    "tempat wisata", "lokasi", "alamat", "peta",
    "harga", "biaya", "tarif", "berapa harga",
    "travel", "wisata", "liburan", "hotel",
    "film", "sinema", "bioskop", "trailer",
    "musik", "lagu", "artis", "konser",
    "olahraga", "pertandingan", "skor",
    "tips", "trik", "panduan",
}

# System keywords
SYSTEM_KEYWORDS = {
    "/exit", "/quit", "/help", "/status", "/config",
    "/model", "/mode", "/debug", "/test", "/clear",
    "shutdown", "restart", "reboot", "logout",
}

# Reasoning keywords (higher weight for "mengapa", "jelaskan", etc.)
REASONING_KEYWORDS = {
    # Strong signals (weight 2.0)
    "mengapa", "kenapa", "why", "jelaskan", "explain",
    "definisi", "definition", "arti", "makna", "meaning",
    "apa itu", "apa arti", "bedanya", "perbedaan",
    # Medium signals (weight 1.0)
    "karena", "sebab", "akibat", "fakta", "fact",
    "teori", "theory", "konsep", "concept",
    "buktikan", "prove", "argumentasi", "logika", "logic",
    "implikasi", "implication", "konklusi", "conclusion",
    "analogi", "analogy", "metafora", "perumpamaan",
}

# General/Lexical keywords (for definitions, meanings)
GENERAL_KEYWORDS = {
    "apa itu", "apa arti", "arti", "makna", "definisi",
    "kamus", "sinonim", "antonim", "artinya",
    "penjelasan", "maksud", "istilah",
}


# ═══════════════════════════════════════════════════════════════════════════
# INTENT-FIRST ROUTER
# ═══════════════════════════════════════════════════════════════════════════

class IntentFirstRouter:
    """
    Intent-First Router: Menentukan model spesialis berdasarkan intent user.
    
    Priority Chain:
    ───────────────
    Tier 0: Exact match (< 1ms)
    Tier 1: Keyword scoring (< 5ms)
    Tier 2: Semantic similarity (< 20ms)
    Tier 3: Fallback (< 1ms)
    
    Filosofi:
    ─────────
    "Kirim ke ahli yang tepat, bukan ke semua orang."
    """
    
    # Confidence threshold: jika di bawah ini, fallback ke general
    CONFIDENCE_THRESHOLD = 0.65
    
    # Semantic centroids (lazy loaded)
    _centroids = None
    _centroids_loaded = False
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  🎯 [IntentRouter] {msg}")
    
    # ─────────────────────────────────────────────────────────────────────
    # MAIN CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────
    
    def classify(self, text: str) -> IntentResult:
        """
        Klasifikasi intent dari query user.
        
        Args:
            text: Query dari user
            
        Returns:
            IntentResult dengan intent, confidence, domain, model
        """
        t0 = time.perf_counter()
        text_lower = text.strip().lower()
        words = text_lower.split()
        
        # ── Tier 0: Exact Match (< 1ms) ──────────────────────────────────
        result = self._tier0_exact(text_lower, words)
        if result:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result
        
        # ── Tier 1: Keyword Scoring (< 5ms) ──────────────────────────────
        result = self._tier1_keywords(text_lower, words)
        if result and result.confidence >= self.CONFIDENCE_THRESHOLD:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result
        
        # ── Tier 2: Semantic Similarity (< 20ms) ─────────────────────────
        result_t2 = self._tier2_semantic(text_lower)
        if result_t2 and result_t2.confidence >= self.CONFIDENCE_THRESHOLD:
            result_t2.latency_ms = (time.perf_counter() - t0) * 1000
            return result_t2
        
        # ── Tier 3: Fallback (< 1ms) ─────────────────────────────────────
        result_t3 = self._tier3_fallback(text_lower, words, result, result_t2)
        result_t3.latency_ms = (time.perf_counter() - t0) * 1000
        return result_t3
    
    # ─────────────────────────────────────────────────────────────────────
    # TIER 0: EXACT MATCH
    # ─────────────────────────────────────────────────────────────────────
    
    def _tier0_exact(self, text: str, words: list) -> Optional[IntentResult]:
        """
        Tier 0: Exact match untuk slash commands dan greetings.
        Latensi: < 1ms
        """
        # System commands
        if text.startswith("/"):
            return self._make_result(
                IntentClass.SYSTEM, confidence=1.0, tier=0,
                reason="Slash command detected"
            )
        
        # Exact greetings (very short queries)
        if len(words) <= 2:
            if text in PERSONAL_KEYWORDS or text in {"hai", "halo", "hello", "hi", "hey"}:
                return self._make_result(
                    IntentClass.PERSONAL, confidence=1.0, tier=0,
                    reason="Exact greeting match"
                )
        
        return None
    
    # ─────────────────────────────────────────────────────────────────────
    # TIER 1: KEYWORD SCORING
    # ─────────────────────────────────────────────────────────────────────
    
    def _tier1_keywords(self, text: str, words: list) -> Optional[IntentResult]:
        """
        Tier 1: Keyword-based scoring dengan weighted intersection.
        Latensi: < 5ms
        """
        scores = {
            IntentClass.CODING: 0.0,
            IntentClass.MATH: 0.0,
            IntentClass.SECURITY: 0.0,
            IntentClass.DARKWEB: 0.0,
            IntentClass.PERSONAL: 0.0,
            IntentClass.BROWSING: 0.0,
            IntentClass.REASONING: 0.0,
            IntentClass.GENERAL: 0.0,
            IntentClass.SYSTEM: 0.0,
        }
        
        # Score each intent
        scores[IntentClass.CODING] = self._score_keywords(text, CODING_KEYWORDS)
        scores[IntentClass.MATH] = self._score_keywords(text, MATH_KEYWORDS)
        scores[IntentClass.SECURITY] = self._score_keywords(text, SECURITY_KEYWORDS)
        scores[IntentClass.DARKWEB] = self._score_keywords(text, DARKWEB_KEYWORDS)
        scores[IntentClass.PERSONAL] = self._score_keywords(text, PERSONAL_KEYWORDS)
        scores[IntentClass.BROWSING] = self._score_keywords(text, BROWSING_KEYWORDS)
        scores[IntentClass.REASONING] = self._score_keywords(text, REASONING_KEYWORDS)
        scores[IntentClass.GENERAL] = self._score_keywords(text, GENERAL_KEYWORDS)
        scores[IntentClass.SYSTEM] = self._score_keywords(text, SYSTEM_KEYWORDS)
        
        # Priority overrides: REASONING > GENERAL > domain-specific
        # If "apa arti", "apa itu", "definisi" present → GENERAL wins
        if any(kw in text for kw in ["apa arti", "apa itu", "definisi", "arti dari", "makna dari"]):
            scores[IntentClass.GENERAL] += 5.0  # Strong boost
            scores[IntentClass.REASONING] *= 0.3
        
        # If "mengapa", "jelaskan", "kenapa" present → REASONING wins
        if any(kw in text for kw in ["mengapa", "kenapa", "jelaskan", "explain", "why"]):
            scores[IntentClass.REASONING] += 5.0  # Strong boost
            scores[IntentClass.MATH] *= 0.3
            scores[IntentClass.CODING] *= 0.3
        
        # If security keywords present → SECURITY wins over BROWSING
        if any(kw in text for kw in ["hack", "exploit", "vulnerability", "penetration", "crack"]):
            scores[IntentClass.SECURITY] += 5.0  # Strong boost
            scores[IntentClass.BROWSING] *= 0.3
            
        # If darkweb keywords present → DARKWEB wins
        if any(kw in text for kw in ["onion", "darkweb", "darknet", "ahmia"]):
            scores[IntentClass.DARKWEB] += 5.0
            scores[IntentClass.BROWSING] *= 0.1
            scores[IntentClass.SECURITY] *= 0.8 # Security is related but darkweb is more specific
        
        # Find best
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Normalize confidence (0-1)
        total = sum(scores.values()) + 1e-9
        confidence = best_score / total if best_score > 0 else 0.0
        
        # Boost confidence for strong signals
        if best_score >= 3.0:
            confidence = min(confidence * 1.5, 1.0)
        elif best_score >= 2.0:
            confidence = min(confidence * 1.2, 1.0)
        
        if best_score > 0:
            return self._make_result(
                best_intent, confidence=confidence, tier=1,
                reason=f"Keyword scoring: {best_intent.value}={best_score:.1f}"
            )
        
        return None
    
    def _score_keywords(self, text: str, keywords: set) -> float:
        """Score berdasarkan berapa banyak keyword yang match."""
        score = 0.0
        for kw in keywords:
            if kw in text:
                # Longer keywords = stronger signal
                weight = 1.0 + (len(kw.split()) - 1) * 0.5
                score += weight
        return score
    
    # ─────────────────────────────────────────────────────────────────────
    # TIER 2: SEMANTIC SIMILARITY
    # ─────────────────────────────────────────────────────────────────────
    
    def _tier2_semantic(self, text: str) -> Optional[IntentResult]:
        """
        Tier 2: Semantic similarity menggunakan centroids.
        Latensi: < 20ms (termasuk embedding time)
        """
        try:
            self._ensure_centroids()
            if not self._centroids_loaded:
                return None
            
            from moko_agents.llm_engine import engine
            q_emb = engine.get_embedding(text)
            if not q_emb or len(q_emb) != 768:
                return None
            
            q_v = np.array(q_emb, dtype=np.float32)
            best_domain = None
            best_sim = -1.0
            
            for domain, centroid in self._centroids.items():
                denom = (np.linalg.norm(q_v) * np.linalg.norm(centroid)) + 1e-9
                sim = float(np.dot(q_v, centroid) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_domain = domain
            
            if best_sim < 0.38:
                return None
            
            # Map domain to intent
            domain_intent_map = {
                "code": IntentClass.CODING,
                "math": IntentClass.MATH,
                "physics": IntentClass.MATH,
                "cybersecurity": IntentClass.SECURITY,
                "security": IntentClass.SECURITY,
                "cyberoffensive": IntentClass.DARKWEB,
                "lexical": IntentClass.GENERAL,
                "general": IntentClass.GENERAL,
                "personal": IntentClass.PERSONAL,
            }
            
            intent = domain_intent_map.get(best_domain, IntentClass.GENERAL)
            confidence = min(best_sim * 1.2, 1.0)
            
            return self._make_result(
                intent, confidence=confidence, tier=2,
                reason=f"Semantic similarity: {best_domain}={best_sim:.3f}"
            )
            
        except Exception as e:
            self._log(f"Semantic fallback error: {e}")
            return None
    
    def _ensure_centroids(self):
        """Load semantic centroids dari cache."""
        if self._centroids_loaded:
            return
        
        cache_path = Path(settings.WORKSPACE_DIR) / ".moko_domain_centroids.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._centroids = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
                self._centroids_loaded = True
            except Exception:
                pass
    
    # ─────────────────────────────────────────────────────────────────────
    # TIER 3: FALLBACK
    # ─────────────────────────────────────────────────────────────────────
    
    def _tier3_fallback(
        self, text: str, words: list,
        tier1_result: Optional[IntentResult],
        tier2_result: Optional[IntentResult]
    ) -> IntentResult:
        """
        Tier 3: Fallback logic berdasarkan query characteristics.
        Latensi: < 1ms
        """
        # Jika ada tier1 atau tier2 result (tapi confidence rendah), gunakan
        if tier1_result and tier1_result.confidence > 0.3:
            return self._make_result(
                tier1_result.intent, confidence=tier1_result.confidence * 0.8, tier=3,
                reason=f"Tier 1 weak signal: {tier1_result.reason}"
            )
        
        if tier2_result and tier2_result.confidence > 0.3:
            return self._make_result(
                tier2_result.intent, confidence=tier2_result.confidence * 0.8, tier=3,
                reason=f"Tier 2 weak signal: {tier2_result.reason}"
            )
        
        # Query characteristics-based fallback
        if len(words) >= 6:
            # Long query → likely needs deep analysis
            return self._make_result(
                IntentClass.GENERAL, confidence=0.5, tier=3,
                reason="Long query → general deep analysis"
            )
        
        if len(words) <= 2:
            # Very short → personal/greeting
            return self._make_result(
                IntentClass.PERSONAL, confidence=0.6, tier=3,
                reason="Short query → personal"
            )
        
        # Default: general
        return self._make_result(
            IntentClass.GENERAL, confidence=0.4, tier=3,
            reason="Fallback: general"
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────────────────────────────
    
    def _make_result(
        self,
        intent: IntentClass,
        confidence: float,
        tier: int,
        reason: str,
        query: str = ""
    ) -> IntentResult:
        """Buat IntentResult dengan mapping otomatis."""
        model_key, domain, path = INTENT_MAP[intent]
        
        # Determine complexity based on text length and intent
        complexity = "SIMPLE"
        if len(query.split()) > 10 or intent in (IntentClass.REASONING, IntentClass.SECURITY):
            complexity = "COMPLEX"
        elif len(query.split()) > 5:
            complexity = "MEDIUM"
            
        import hashlib
        query_id = hashlib.sha256(query.encode()).hexdigest()[:12] if query else "0"*12
        
        return IntentResult(
            query_id=query_id,
            intent_class=intent,
            confidence=confidence,
            domain=domain,
            model_key=model_key,
            path=path,
            complexity=complexity,
            reason=reason,
            tier=tier,
            latency_ms=0.0,
            metadata={},
            token_budget=TokenBudget(),
            thinking_enabled=(complexity == "COMPLEX"),
            governor_mode="quick" if complexity == "SIMPLE" else "full",
            primary_subsystem=domain
        )


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════

_router_instance: Optional[IntentFirstRouter] = None

def get_intent_router(verbose: bool = False) -> IntentFirstRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = IntentFirstRouter(verbose=verbose)
    return _router_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n🧪 IntentFirstRouter — Self Test\n")
    
    router = IntentFirstRouter(verbose=True)
    
    test_cases = [
        # (query, expected_intent)
        ("tulis fungsi Python untuk sorting", IntentClass.CODING),
        ("hitung integral dari x^2", IntentClass.MATH),
        ("bagaimana cara exploit buffer overflow", IntentClass.SECURITY),
        ("cari data bocor di darkweb", IntentClass.DARKWEB),
        ("apa arti kata komputasi", IntentClass.GENERAL),
        ("hai, apa kabar?", IntentClass.PERSONAL),
        ("mengapa gravitasi bisa menarik benda?", IntentClass.REASONING),
        ("/status", IntentClass.SYSTEM),
        ("resep nasi goreng sederhana", IntentClass.BROWSING),
        ("jelaskan algoritma quicksort", IntentClass.REASONING),
        ("bug di function main()", IntentClass.CODING),
        ("ahmia search untuk ransomware leaks", IntentClass.DARKWEB),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        result = router.classify(query)
        status = "✅" if result.intent_class == expected else "❌"
        if result.intent_class == expected:
            correct += 1
        print(f"  {status} '{query[:40]}...' → {result.intent_class.value} "
              f"(conf={result.confidence:.2f}, tier={result.tier}, {result.latency_ms:.1f}ms)")
    
    print(f"\n  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print("\n✅ Self test selesai!\n")


if __name__ == "__main__":
    _self_test()
