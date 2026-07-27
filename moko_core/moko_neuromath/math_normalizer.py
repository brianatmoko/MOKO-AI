"""
MOKO Math Normalizer
====================
Preprocessing dan normalisasi notasi matematika sebelum dikirim ke LLM.

Masalah yang diselesaikan:
  1. Notasi implisit: LLM membaca 'ai + bj' sebagai 'a^i + b^j' (eksponen)
     padahal makna sebenarnya 'a*i + b*j' (perkalian).
  2. Simbol ambigu: LaTeX \\cdot vs ^, implicit multiplication vs exponentiation.
  3. Tidak ada konteks variabel — LLM menebak sendiri.

Arsitektur:
  Query Mentah → Detector → Normalizer → Query Ternormalisasi + Header Klarifikasi
"""
import re
from typing import Tuple


# ── Konstanta ──────────────────────────────────────────────────────────────────

# Pola yang menandakan soal matematika tingkat lanjut
MATH_OLYMPIAD_MARKERS = [
    r"putnam", r"olympiad", r"competition", r"imo\b", r"aime\b", r"amc\b",
    r"determine\s+the", r"find\s+all", r"prove\s+that", r"show\s+that",
    r"buktikan", r"tentukan", r"tunjukkan", r"jumlah\s+semua",
    r"bilangan\s+bulat", r"bilangan\s+prima", r"modulo\b", r"\bdet\b",
    r"matriks\b", r"matrix\b", r"determinant", r"binomial", r"kombinasi",
]

# Variabel satu-huruf yang umum dalam soal matematika
COMMON_MATH_VARS = set("abcdefghijklmnpqrstuvwxyz")

# Pasangan variabel 2-huruf yang SANGAT UMUM menandakan perkalian implisit dalam soal
# Contoh: ai, bj, mn, xy, ab, cd
IMPLICIT_MULT_PAIRS = re.compile(
    r'\b([a-z])\s*([a-z])\s*(?=[=+\-\*/,\.\s\)]|$)',
    re.IGNORECASE
)

# Pola numerik: 2n, 3k, dst
NUMERIC_IMPLICIT = re.compile(
    r'\b(\d+)\s*([a-z])\b',
    re.IGNORECASE
)


def is_math_olympiad_query(question: str) -> bool:
    """Deteksi apakah pertanyaan adalah soal olimpiade matematika tingkat lanjut."""
    q_lower = question.lower()
    return any(re.search(pat, q_lower) for pat in MATH_OLYMPIAD_MARKERS)


def extract_notation_hint(question: str) -> str:
    """
    Ekstrak petunjuk notasi dari pertanyaan.
    Misalnya: deteksi apakah soal mendefinisikan variabel tertentu.
    """
    hints = []

    # Deteksi definisi variabel dalam soal (pola: "let i, j ∈ {1,...,n}")
    var_defs = re.findall(r'(?:let|for)\s+([a-z](?:\s*,\s*[a-z])*)\s+(?:∈|in|be)', question, re.IGNORECASE)
    if var_defs:
        vars_found = []
        for vd in var_defs:
            vars_found.extend([v.strip() for v in vd.split(',')])
        if vars_found:
            hints.append(f"Variabel yang didefinisikan dalam soal: {', '.join(vars_found)}")

    # Deteksi persamaan dengan pola ai + bj = n (implicit multiplication)
    implicit_eq = re.findall(r'\b([a-z])([a-z])\s*[+\-]\s*([a-z])([a-z])\s*=', question, re.IGNORECASE)
    if implicit_eq:
        examples = []
        for m in implicit_eq[:2]:
            examples.append(f"'{m[0]}{m[1]}' berarti '{m[0]}×{m[1]}' (perkalian, BUKAN eksponen)")
        hints.append("NOTASI PERKALIAN IMPLISIT: " + "; ".join(examples))

    return "; ".join(hints) if hints else ""


def build_math_notation_header(question: str) -> str:
    """
    Buat blok header yang menjelaskan notasi matematika secara eksplisit
    untuk mencegah LLM salah membaca notasi.
    """
    header_lines = [
        "═══════════════════════════════════════════════════",
        "PROTOKOL NOTASI MATEMATIKA (WAJIB DIIKUTI):",
        "═══════════════════════════════════════════════════",
        "1. PERKALIAN IMPLISIT: Dalam soal ini, 'xy' atau 'ai' BERARTI x×y atau a×i",
        "   (perkalian biasa), BUKAN x^y atau a^i (eksponen/pangkat).",
        "   Contoh: 'ai + bj = n' dibaca sebagai '(a×i) + (b×j) = n'.",
        "2. EKSPONEN: Gunakan tanda '^' atau 'pangkat'. Jika soal TIDAK menggunakan",
        "   tanda '^', jangan asumsikan eksponen.",
        "3. VARIABEL: Setiap huruf tunggal adalah variabel terpisah kecuali dinyatakan lain.",
        "4. VERIFIKASI: Setelah menurunkan rumus, SELALU verifikasi dengan kasus kecil",
        "   (n=1, n=2, n=3) sebelum menyatakan jawaban final.",
        "5. KASUS KECIL WAJIB: Tunjukkan perhitungan manual untuk setidaknya n=1 dan n=2",
        "   untuk membuktikan rumus yang didapat adalah benar.",
        "═══════════════════════════════════════════════════",
    ]

    # Tambah petunjuk spesifik jika terdeteksi
    notation_hint = extract_notation_hint(question)
    if notation_hint:
        header_lines.insert(-1, f"CATATAN SPESIFIK: {notation_hint}")

    return "\n".join(header_lines)


def normalize_math_question(question: str) -> Tuple[str, str]:
    """
    Normalkan pertanyaan matematika.

    Returns:
        (normalized_question, notation_header)
        - normalized_question: Pertanyaan yang sudah dinormalisasi
        - notation_header: Header protokol notasi untuk disuntikkan ke prompt
    """
    if not is_math_olympiad_query(question):
        return question, ""

    notation_header = build_math_notation_header(question)
    return question, notation_header


def get_cas_verification_prompt(question: str, llm_answer: str, cas_result: str) -> str:
    """
    Buat prompt verifikasi CAS untuk membandingkan jawaban LLM dengan CAS.
    """
    return (
        f"Soal Asli: {question}\n\n"
        f"Jawaban LLM: {llm_answer}\n\n"
        f"Hasil CAS (SymPy): {cas_result}\n\n"
        "Bandingkan kedua jawaban di atas. Jika ada ketidakcocokan, identifikasi "
        "secara spesifik di mana kesalahan LLM terjadi dan berikan koreksi."
    )


class MathNormalizer:
    """
    Singleton wrapper untuk normalisasi matematika yang terintegrasi
    dengan pipeline MOKO Analyst Node.
    """

    def __init__(self):
        self._cache = {}

    def preprocess(self, question: str) -> Tuple[str, str]:
        """
        Preprocessing utama — mengembalikan (question, notation_header).
        notation_header kosong jika bukan soal matematika.
        """
        cache_key = hash(question)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = normalize_math_question(question)
        self._cache[cache_key] = result
        return result

    def is_olympiad(self, question: str) -> bool:
        return is_math_olympiad_query(question)

    def build_enriched_prompt(self, question: str, base_prompt: str) -> str:
        """
        Bangun prompt yang sudah diperkaya dengan header notasi matematika.
        Digunakan di Marathon Runner step prompts.
        """
        _, header = self.preprocess(question)
        if header:
            return f"{header}\n\n{base_prompt}"
        return base_prompt


# Singleton global
math_normalizer = MathNormalizer()
