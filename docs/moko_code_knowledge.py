from __future__ import annotations

from dataclasses import dataclass, field
import re


"""MOKO Code Knowledge Base.

Corpus pengetahuan pemrograman lintas-domain yang dipakai MOKO untuk
mengubah instruksi menjadi program yang lebih kaya (bukan template mentah).

Desain retrieval mengikuti praktik industri agen coding:
- Anchor-based retrieval: pilih pengetahuan berdasarkan sinyal terkuat
  pada intent, bukan menyerap semua token ("retrieve-everything" antipattern).
- Setiap snippet punya anchor eksplisit + domain + sumber referensi, sehingga
  jejak "apa yang dipelajari" bisa diaudit dan selalu relevan dengan permintaan.

Referensi (GitHub / komunitas programmer):
- StarCoder / The Stack (bigcode-project) — korpus kode multi-bahasa.
- CommitPackFT (OctoPack) — commit Git sebagai instruksi berkualitas.
- OpenCodeInstruct & Code-Feedback (m-a-p) — pasangan instruksi + umpan balik eksekusi.
- OSS-Instruct (MagiCoder) — sintesis instruksi dari snippet kode nyata.
"""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]{2,}", text.lower())


@dataclass(frozen=True)
class KnowledgeSnippet:
    snippet_id: str
    domain: str
    summary: str
    anchors: frozenset[str]
    code: str
    requires_imports: tuple[str, ...] = ()
    source: str = "moko-curated"

    def score(self, focus_tokens: set[str]) -> int:
        return len(self.anchors & focus_tokens)


GEOMETRY_SNIPPET = KnowledgeSnippet(
    snippet_id="geometri.luas_dasar",
    domain="geometri",
    summary="Rumus luas & keliling bangun datar dasar",
    anchors=frozenset(
        {
            "rumus",
            "formula",
            "geometri",
            "luas",
            "keliling",
            "persegi",
            "panjang",
            "lebar",
            "area",
            "lingkaran",
            "segitiga",
            "bangun",
        }
    ),
    requires_imports=("import math",),
    source="OSS-Instruct pattern (geometry helpers)",
    code="""def hitung_luas_persegi_panjang(panjang: float, lebar: float) -> float:
    return panjang * lebar


def hitung_luas_persegi(sisi: float) -> float:
    return sisi * sisi


def hitung_luas_lingkaran(jari_jari: float) -> float:
    return math.pi * jari_jari * jari_jari


def hitung_luas_segitiga(alas: float, tinggi: float) -> float:
    return 0.5 * alas * tinggi""",
)


TRIGONOMETRY_SNIPPET = KnowledgeSnippet(
    snippet_id="trigonometri.derajat",
    domain="trigonometri",
    summary="Fungsi trigonometri berbasis derajat",
    anchors=frozenset(
        {
            "trigonometri",
            "trigonometry",
            "sin",
            "cos",
            "tan",
            "sudut",
            "derajat",
            "radian",
        }
    ),
    requires_imports=("import math",),
    source="StarCoder math utilities pattern",
    code="""def hitung_sin_derajat(derajat: float) -> float:
    return math.sin(math.radians(derajat))


def hitung_cos_derajat(derajat: float) -> float:
    return math.cos(math.radians(derajat))


def hitung_tan_derajat(derajat: float) -> float:
    return math.tan(math.radians(derajat))""",
)


STATISTICS_SNIPPET = KnowledgeSnippet(
    snippet_id="statistika.deskriptif",
    domain="statistika",
    summary="Statistika deskriptif dasar (rata-rata, median, deviasi)",
    anchors=frozenset(
        {
            "statistik",
            "statistika",
            "statistics",
            "rata",
            "mean",
            "average",
            "median",
            "modus",
            "deviasi",
            "standar",
            "stdev",
            "variansi",
            "variance",
        }
    ),
    requires_imports=("import statistics",),
    source="CommitPackFT pattern (stats helpers)",
    code="""def hitung_rata_rata(data: list[float]) -> float:
    if not data:
        raise ValueError("Data tidak boleh kosong")
    return statistics.fmean(data)


def hitung_median(data: list[float]) -> float:
    if not data:
        raise ValueError("Data tidak boleh kosong")
    return statistics.median(data)


def hitung_standar_deviasi(data: list[float]) -> float:
    if len(data) < 2:
        raise ValueError("Butuh minimal dua data")
    return statistics.pstdev(data)""",
)


FINANCE_SNIPPET = KnowledgeSnippet(
    snippet_id="finansial.bunga",
    domain="finansial",
    summary="Perhitungan bunga majemuk & cicilan tetap",
    anchors=frozenset(
        {
            "bunga",
            "interest",
            "finansial",
            "keuangan",
            "kredit",
            "cicilan",
            "pinjaman",
            "angsuran",
            "investasi",
            "tabungan",
            "majemuk",
        }
    ),
    requires_imports=(),
    source="OpenCodeInstruct pattern (finance calculators)",
    code="""def hitung_bunga_majemuk(pokok: float, bunga_tahunan: float, tahun: float) -> float:
    return pokok * ((1 + bunga_tahunan) ** tahun)


def hitung_cicilan_bulanan(pokok: float, bunga_tahunan: float, jumlah_bulan: int) -> float:
    if jumlah_bulan <= 0:
        raise ValueError("Jumlah bulan harus positif")
    bunga_bulanan = bunga_tahunan / 12
    if bunga_bulanan == 0:
        return pokok / jumlah_bulan
    faktor = (1 + bunga_bulanan) ** jumlah_bulan
    return pokok * bunga_bulanan * faktor / (faktor - 1)""",
)


CONVERSION_SNIPPET = KnowledgeSnippet(
    snippet_id="konversi.satuan",
    domain="konversi",
    summary="Konversi satuan suhu & panjang umum",
    anchors=frozenset(
        {
            "konversi",
            "convert",
            "suhu",
            "celsius",
            "fahrenheit",
            "kelvin",
            "satuan",
            "temperatur",
        }
    ),
    requires_imports=(),
    source="The Stack pattern (unit conversion)",
    code="""def celsius_ke_fahrenheit(celsius: float) -> float:
    return (celsius * 9 / 5) + 32


def fahrenheit_ke_celsius(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9


def celsius_ke_kelvin(celsius: float) -> float:
    return celsius + 273.15""",
)


ALGORITHM_SNIPPET = KnowledgeSnippet(
    snippet_id="algoritma.dasar",
    domain="algoritma",
    summary="Algoritma numerik dasar (faktorial, fibonacci, fpb, prima)",
    anchors=frozenset(
        {
            "algoritma",
            "algorithm",
            "faktorial",
            "factorial",
            "fibonacci",
            "fpb",
            "gcd",
            "prima",
            "prime",
            "deret",
        }
    ),
    requires_imports=("import math",),
    source="HumanEvalPack pattern (algorithmic tasks)",
    code="""def hitung_faktorial(n: int) -> int:
    if n < 0:
        raise ValueError("Faktorial butuh bilangan non-negatif")
    return math.factorial(n)


def deret_fibonacci(jumlah: int) -> list[int]:
    if jumlah <= 0:
        return []
    urutan = [0, 1]
    while len(urutan) < jumlah:
        urutan.append(urutan[-1] + urutan[-2])
    return urutan[:jumlah]


def hitung_fpb(a: int, b: int) -> int:
    return math.gcd(a, b)


def apakah_prima(n: int) -> bool:
    if n < 2:
        return False
    for pembagi in range(2, int(math.isqrt(n)) + 1):
        if n % pembagi == 0:
            return False
    return True""",
)


DEFAULT_SNIPPETS: tuple[KnowledgeSnippet, ...] = (
    GEOMETRY_SNIPPET,
    TRIGONOMETRY_SNIPPET,
    STATISTICS_SNIPPET,
    FINANCE_SNIPPET,
    CONVERSION_SNIPPET,
    ALGORITHM_SNIPPET,
)


@dataclass
class CodeKnowledgeBase:
    snippets: tuple[KnowledgeSnippet, ...] = field(default_factory=lambda: DEFAULT_SNIPPETS)

    def retrieve(self, focus_tokens: set[str], *, limit: int = 3) -> list[KnowledgeSnippet]:
        """Anchor-based retrieval.

        Hanya mengembalikan snippet yang benar-benar cocok dengan sinyal intent
        (skor >= 1). Ini mencegah "retrieve-everything" sehingga pengetahuan yang
        di-crawl selalu relevan dengan permintaan user.
        """
        scored: list[tuple[int, int, KnowledgeSnippet]] = []
        for index, snippet in enumerate(self.snippets):
            score = snippet.score(focus_tokens)
            if score >= 1:
                scored.append((score, -index, snippet))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [snippet for _score, _order, snippet in scored[:limit]]

    def domains(self) -> list[str]:
        return sorted({snippet.domain for snippet in self.snippets})
