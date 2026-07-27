"""
MOKO Story Math Parser — Layer 0 of ARMS
=========================================
Mengekstrak problem terstruktur dari narasi bahasa manusia.

Transformasi:
  "Sebuah piston diameter 80mm pada tekanan 12 bar. Berapa gaya?"
    →  {
         known:    {D: (0.08, 'm'), P: (1_200_000, 'Pa')},
         unknown:  ['F'],
         domain:   'fluid_mechanics',
         question: 'Hitung gaya F dalam Newton'
       }

Prinsip desain:
  - TIDAK menggunakan LLM — pure algorithmic (deterministik)
  - Regex + rule-based (reproducible, fast, testable)
  - Semua unit dinormalisasi ke SI base units otomatis
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT NORMALIZATION DATABASE
# Setiap unit → (faktor konversi ke SI base, dimensi SI)
# ═══════════════════════════════════════════════════════════════════════════════

# Format: unit_string → (multiplier, si_unit)
UNIT_CONVERSIONS: Dict[str, Tuple[float, str]] = {
    # Length → meter
    "m": (1.0, "m"), "meter": (1.0, "m"), "meters": (1.0, "m"),
    "km": (1e3, "m"), "kilometer": (1e3, "m"),
    "cm": (1e-2, "m"), "centimeter": (1e-2, "m"),
    "mm": (1e-3, "m"), "milimeter": (1e-3, "m"), "millimeter": (1e-3, "m"),
    "μm": (1e-6, "m"), "um": (1e-6, "m"), "mikrometer": (1e-6, "m"),
    "nm": (1e-9, "m"), "nanometer": (1e-9, "m"),
    "inch": (0.0254, "m"), "in": (0.0254, "m"),
    "ft": (0.3048, "m"), "feet": (0.3048, "m"), "foot": (0.3048, "m"),

    # Mass → kilogram
    "kg": (1.0, "kg"), "kilogram": (1.0, "kg"),
    "g": (1e-3, "kg"), "gram": (1e-3, "kg"),
    "mg": (1e-6, "kg"), "miligram": (1e-6, "kg"),
    "ton": (1e3, "kg"), "tonne": (1e3, "kg"),
    "lb": (0.453592, "kg"), "pound": (0.453592, "kg"),

    # Time → second
    "s": (1.0, "s"), "detik": (1.0, "s"), "second": (1.0, "s"), "seconds": (1.0, "s"),
    "ms": (1e-3, "s"), "milisecond": (1e-3, "s"), "millisecond": (1e-3, "s"),
    "μs": (1e-6, "s"), "us": (1e-6, "s"), "microsecond": (1e-6, "s"),
    "ns": (1e-9, "s"), "nanosecond": (1e-9, "s"),
    "menit": (60.0, "s"), "minute": (60.0, "s"), "min": (60.0, "s"),
    "jam": (3600.0, "s"), "hour": (3600.0, "s"), "h": (3600.0, "s"), "hr": (3600.0, "s"),
    "hari": (86400.0, "s"), "day": (86400.0, "s"),
    "tahun": (365.25 * 86400, "s"), "year": (365.25 * 86400, "s"),

    # Temperature → Kelvin (handled separately for offset)
    "k": (1.0, "K"), "kelvin": (1.0, "K"),
    "°c": (1.0, "°C"), "c": (1.0, "°C"), "celsius": (1.0, "°C"), "derajat": (1.0, "°C"),
    "°f": (1.0, "°F"), "f": (1.0, "°F"), "fahrenheit": (1.0, "°F"),

    # Pressure → Pascal
    "pa": (1.0, "Pa"), "pascal": (1.0, "Pa"),
    "kpa": (1e3, "Pa"), "kilopascal": (1e3, "Pa"),
    "mpa": (1e6, "Pa"), "megapascal": (1e6, "Pa"),
    "bar": (1e5, "Pa"),
    "mbar": (1e2, "Pa"), "millibar": (1e2, "Pa"),
    "atm": (101325.0, "Pa"), "atmosfer": (101325.0, "Pa"),
    "psi": (6894.76, "Pa"),
    "mmhg": (133.322, "Pa"), "torr": (133.322, "Pa"),

    # Force → Newton
    "n": (1.0, "N"), "newton": (1.0, "N"),
    "kn": (1e3, "N"), "kilonewton": (1e3, "N"),
    "mn": (1e6, "N"), "meganewton": (1e6, "N"),
    "kgf": (9.80665, "N"), "kgforce": (9.80665, "N"),
    "lbf": (4.44822, "N"),

    # Energy → Joule
    "j": (1.0, "J"), "joule": (1.0, "J"),
    "kj": (1e3, "J"), "kilojoule": (1e3, "J"),
    "mj": (1e6, "J"), "megajoule": (1e6, "J"),
    "cal": (4.184, "J"), "kalori": (4.184, "J"), "calorie": (4.184, "J"),
    "kcal": (4184.0, "J"), "kilokalori": (4184.0, "J"), "kkal": (4184.0, "J"),
    "kwh": (3.6e6, "J"), "kilowatthour": (3.6e6, "J"),
    "wh": (3600.0, "J"),
    "ev": (1.602e-19, "J"), "electronvolt": (1.602e-19, "J"),
    "btu": (1055.06, "J"),

    # Power → Watt
    "w": (1.0, "W"), "watt": (1.0, "W"),
    "kw": (1e3, "W"), "kilowatt": (1e3, "W"),
    "mw": (1e6, "W"), "megawatt": (1e6, "W"),
    "hp": (745.7, "W"), "horsepower": (745.7, "W"), "dk": (735.5, "W"),

    # Frequency → Hertz
    "hz": (1.0, "Hz"), "hertz": (1.0, "Hz"),
    "khz": (1e3, "Hz"), "kilohertz": (1e3, "Hz"),
    "mhz": (1e6, "Hz"), "megahertz": (1e6, "Hz"),
    "ghz": (1e9, "Hz"), "gigahertz": (1e9, "Hz"),
    "rpm": (1/60, "rev/s"),  # rev per second

    # Electric → base SI
    "v": (1.0, "V"), "volt": (1.0, "V"), "volts": (1.0, "V"),
    "mv": (1e-3, "V"), "millivolt": (1e-3, "V"),
    "kv": (1e3, "V"), "kilovolt": (1e3, "V"),
    "a": (1.0, "A"), "ampere": (1.0, "A"), "amp": (1.0, "A"),
    "ma": (1e-3, "A"), "milliampere": (1e-3, "A"), "miliampere": (1e-3, "A"),
    "μa": (1e-6, "A"), "ua": (1e-6, "A"),
    "ω": (1.0, "Ω"), "ohm": (1.0, "Ω"), "ohms": (1.0, "Ω"),
    "kω": (1e3, "Ω"), "kohm": (1e3, "Ω"),
    "mω": (1e6, "Ω"), "mohm": (1e6, "Ω"),
    "f": (1.0, "F"), "farad": (1.0, "F"),
    "μf": (1e-6, "F"), "uf": (1e-6, "F"), "microfarad": (1e-6, "F"),
    "nf": (1e-9, "F"), "nanofarad": (1e-9, "F"),
    "pf": (1e-12, "F"), "picofarad": (1e-12, "F"),
    "h": (1.0, "H"), "henry": (1.0, "H"),
    "mh": (1e-3, "H"), "millihenry": (1e-3, "H"),
    "μh": (1e-6, "H"), "uh": (1e-6, "H"),

    # Volume → m³
    "m3": (1.0, "m³"), "m³": (1.0, "m³"),
    "l": (1e-3, "m³"), "liter": (1e-3, "m³"), "litre": (1e-3, "m³"),
    "ml": (1e-6, "m³"), "mililiter": (1e-6, "m³"), "milliliter": (1e-6, "m³"),
    "cc": (1e-6, "m³"), "cm3": (1e-6, "m³"), "cm³": (1e-6, "m³"),

    # Speed → m/s
    "m/s": (1.0, "m/s"), "mps": (1.0, "m/s"),
    "km/h": (1/3.6, "m/s"), "kmh": (1/3.6, "m/s"), "kph": (1/3.6, "m/s"),
    "m/s²": (1.0, "m/s²"), "m/s2": (1.0, "m/s²"),

    # Area → m²
    "m2": (1.0, "m²"), "m²": (1.0, "m²"),
    "cm2": (1e-4, "m²"), "cm²": (1e-4, "m²"),
    "mm2": (1e-6, "m²"), "mm²": (1e-6, "m²"),
    "km2": (1e6, "m²"), "km²": (1e6, "m²"),

    # Currency (non-SI, but useful)
    "rp": (1.0, "IDR"), "idr": (1.0, "IDR"),
    "usd": (1.0, "USD"), "$": (1.0, "USD"),

    # Dimensionless
    "%": (0.01, "dimensionless"), "persen": (0.01, "dimensionless"), "percent": (0.01, "dimensionless"),
    "db": (1.0, "dB"),
    "mol": (1.0, "mol"),
}

# Temperature conversions (special — offset needed)
TEMP_TO_KELVIN = {
    "°C": lambda c: c + 273.15,
    "°F": lambda f: (f - 32) * 5/9 + 273.15,
    "K": lambda k: k,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN SIGNALS
# Kata kunci yang mengindikasikan domain teknik terapan
# ═══════════════════════════════════════════════════════════════════════════════

APPLIED_DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "fluid_mechanics": [
        "piston", "silinder", "cylinder", "tekanan", "pressure", "bar", "psi",
        "gaya", "force", "bore", "stroke", "hidrolik", "hydraulic", "pneumatik",
        "pneumatic", "fluida", "fluid", "aliran", "flow", "debit", "bernoulli",
        "venturi", "pompa", "pump", "diameter", "torsi piston"
    ],
    "acoustics": [
        "frekuensi", "frequency", "hz", "suara", "sound", "audio", "gelombang",
        "wave", "wavelength", "panjang gelombang", "desibel", "decibel", "db",
        "tabung", "tube", "organ", "senar", "string", "harmonik", "harmonic",
        "nada", "pitch", "resonansi", "resonance", "amplitudo", "amplitude",
        "akustik", "acoustic", "bunyi", "kecepatan suara"
    ],
    "thermodynamics": [
        "suhu", "temperature", "panas", "heat", "kalor", "termal", "thermal",
        "celsius", "kelvin", "fahrenheit", "kalori", "joule", "entropi", "entropy",
        "efisiensi termal", "mesin carnot", "carnot", "konduksi", "konveksi",
        "radiasi", "specific heat", "kapasitas panas", "cp", "cv", "gas ideal",
        "tekanan gas", "volume gas", "ekspansi", "kompresi"
    ],
    "electronics": [
        "resistor", "kapasitor", "capacitor", "induktor", "inductor", "tegangan",
        "voltage", "arus", "current", "daya listrik", "power", "hambatan",
        "resistance", "ohm", "volt", "ampere", "watt", "rangkaian", "circuit",
        "seri", "paralel", "parallel", "frekuensi cutoff", "filter", "rc circuit",
        "rl circuit", "lc circuit", "impedansi", "impedance", "reaktansi",
        "reactance", "baterai", "battery", "trafo", "transformator"
    ],
    "kinematics": [
        "kecepatan", "velocity", "speed", "percepatan", "acceleration", "gerak",
        "motion", "jarak", "distance", "waktu", "time", "lintasan", "trajectory",
        "proyektil", "projectile", "gravitasi", "gravity", "g =", "m/s",
        "jatuh bebas", "free fall", "melingkar", "circular", "rotasi", "rotation"
    ],
    "structural": [
        "tegangan", "stress", "regangan", "strain", "elastisitas", "elastic",
        "modulus young", "young modulus", "momen", "moment", "torsi", "torque",
        "lentur", "bending", "geser", "shear", "deformasi", "deformation",
        "baja", "steel", "beton", "concrete", "beban", "load", "balok", "beam",
        "kolom", "column", "fondasi", "foundation"
    ],
    "optics": [
        "cahaya", "light", "lensa", "lens", "cermin", "mirror", "refraksi",
        "refraction", "indeks bias", "refractive index", "fokus", "focus",
        "titik api", "focal length", "pembesaran", "magnification", "prisma",
        "prism", "difraksi", "diffraction", "interferensi", "interference",
        "laser", "panjang gelombang cahaya", "spektrum"
    ],
    "chemistry": [
        "mol", "molar", "konsentrasi", "concentration", "ph", "asam", "acid",
        "basa", "base", "larutan", "solution", "reaksi", "reaction", "stoikiometri",
        "stoichiometry", "massa molar", "molar mass", "avogadro", "tekanan parsial",
        "partial pressure", "entalpi", "enthalpy", "reaksi eksoterm", "endoterm"
    ],
    "finance": [
        "bunga", "interest", "modal", "capital", "investasi", "investment",
        "roi", "return", "keuntungan", "profit", "npv", "irr", "cagr",
        "inflasi", "inflation", "cicilan", "installment", "kredit", "credit",
        "deposito", "deposit", "tahun", "year", "bunga majemuk", "compound",
        "present value", "future value", "rp", "rupiah", "dolar"
    ],
    "signal_processing": [
        "fft", "fourier", "sinyal", "signal", "frekuensi sampling", "nyquist",
        "bandwidth", "filter", "low pass", "high pass", "band pass", "convolusi",
        "convolution", "modulasi", "modulation", "am", "fm", "sampling rate",
        "aliasing", "spektrum frekuensi"
    ],
    "materials": [
        "kekuatan tarik", "tensile strength", "yield strength", "kekerasan",
        "hardness", "fatigue", "kelelahan material", "ketangguhan", "toughness",
        "koefisien pemuaian", "thermal expansion", "konduktivitas termal",
        "thermal conductivity", "densitas", "density", "massa jenis", "specific gravity"
    ],
    "energy": [
        "efisiensi", "efficiency", "energi", "energy", "daya", "power", "watt",
        "kwh", "konsumsi energi", "energy consumption", "panel surya", "solar",
        "turbin", "turbine", "generator", "konversi energi", "energy conversion"
    ],
    "engine_mechanics": [
        "cc", "bore", "stroke", "langkah piston", "diameter silinder", "sudut pengapian",
        "ignition angle", "rasio kompresi", "compression ratio", "ruang bakar",
        "combustion chamber", "volume clearance", "piston stroke", "crank angle",
        "derajat pengapian", "ignition timing", "timing pengapian", "silinder mesin",
        "kompresi", "pengapian", "rpm", "kapasitas", "seher", "stang", "rambat api",
        "mesin"
    ],
}

# Mapping untuk mengenali pola khusus pada unit yang berdiri sendiri (tidak langsung di belakang angka)
# Pola: (regex_untuk_teks, symbol, value_group, unit_override)
SPECIAL_VALUE_PATTERNS = [
    # "5 tahun" → n=5 (periode, bukan waktu)
    (r'(\d+)\s+tahun\b', 'n', None),
    (r'(\d+)\s+years?\b', 'n', None),
    # "X% per tahun" → r = X/100
    (r'(\d+(?:[.,]\d+)?)\s*%\s*per\s*tahun\b', 'r', 0.01),
    (r'(\d+(?:[.,]\d+)?)\s*%\s*per\s*year\b', 'r', 0.01),
    (r'bunga\s+(\d+(?:[.,]\d+)?)\s*%', 'r', 0.01),
    # "X derajat celsius/kelvin"
    (r'(\d+(?:[.,]\d+)?)\s+derajat\s+celsius\b', 'T_celsius', None),
    (r'(\d+(?:[.,]\d+)?)\s+derajat\s+kelvin\b', 'T_kelvin', None),
    # "X liter" / "X L" (volume, bukan length)
    (r'(\d+(?:[.,]\d+)?)\s+liter\b', 'm_vol', 1e-3),
    (r'(\d+(?:[.,]\d+)?)\s+litre\b', 'm_vol', 1e-3),
    # "sepanjang X m" → L (panjang)
    (r'sepanjang\s+(\d+(?:[.,]\d+)?)\s*m\b', 'L', None),
    # "panjang X m"
    (r'panjang\s+(\d+(?:[.,]\d+)?)\s*m\b', 'L', None),
]

# Kata kunci yang mengindikasikan "apa yang dicari"
UNKNOWN_INTENT_PATTERNS = [
    (r'\bberapa\b', "quantity"),              # berapa X?
    (r'\bhitung(?:lah)?\b', "calculate"),     # hitunglah X
    (r'\bcari(?:lah)?\b', "find"),            # carilah X
    (r'\btentukan\b', "determine"),           # tentukan X
    (r'\bapa(?:kah)?\b.*\b(nilai|besar|ukuran)\b', "value"),
    (r'\bbesar(?:nya)?\b', "magnitude"),      # besarnya X
    (r'\bnilai\b', "value"),                  # nilai X
    (r'\bwhat is\b', "query_en"),
    (r'\bhow much\b', "quantity_en"),
    (r'\bcalculate\b', "calculate_en"),
    (r'\bfind\b', "find_en"),
    (r'\bdetermine\b', "determine_en"),
]

# Pattern untuk mengekstrak variabel yang dicari dari teks
UNKNOWN_VARIABLE_PATTERNS = [
    # "berapa gaya" → gaya
    (r'\bberapa\s+(\w+(?:\s+\w+){0,2})', "id"),
    # "hitung tekanan" → tekanan
    (r'\bhitung(?:lah)?\s+(\w+(?:\s+\w+){0,2})', "id"),
    # "tentukan kecepatan" → kecepatan
    (r'\btentukan\s+(\w+(?:\s+\w+){0,2})', "id"),
    # "cari frekuensi" → frekuensi
    (r'\bcari(?:lah)?\s+(\w+(?:\s+\w+){0,2})', "id"),
    # "what is the pressure" → pressure
    (r'\bwhat is (?:the\s+)?(\w+(?:\s+\w+){0,2})', "en"),
    # "calculate the force" → force
    (r'\bcalculate (?:the\s+)?(\w+(?:\s+\w+){0,2})', "en"),
    # "find the frequency" → frequency
    (r'\bfind (?:the\s+)?(\w+(?:\s+\w+){0,2})', "en"),
]

# Mapping istilah ke simbol variabel standar
QUANTITY_TO_SYMBOL: Dict[str, str] = {
    # Mekanik
    "gaya": "F", "force": "F",
    "tekanan": "P", "pressure": "P",
    "luas": "A", "area": "A",
    "diameter": "D",
    "jari-jari": "r", "radius": "r",
    "kecepatan": "v", "velocity": "v", "speed": "v",
    "percepatan": "a", "acceleration": "a",
    "massa": "m", "mass": "m",
    "berat": "W", "weight": "W",
    "jarak": "s", "distance": "s", "displacement": "s",
    "waktu": "t", "time": "t",
    "panjang": "L", "length": "L",
    "lebar": "w", "width": "w",
    "tinggi": "h", "height": "h",
    "volume": "V",
    "densitas": "ρ", "density": "ρ",
    "torsi": "τ", "torque": "τ",
    "momen inersia": "I", "moment of inertia": "I",
    "tegangan": "σ", "stress": "σ",
    "regangan": "ε", "strain": "ε",
    "modulus young": "E", "elastic modulus": "E",
    # Termodinamika
    "suhu": "T", "temperature": "T",
    "panas": "Q", "heat": "Q", "kalor": "Q",
    "entropi": "S", "entropy": "S",
    "entalpi": "H", "enthalpy": "H",
    "kapasitas panas": "C", "specific heat": "c",
    "efisiensi": "η", "efficiency": "η",
    # Gelombang & Akustik
    "frekuensi": "f", "frequency": "f",
    "panjang gelombang": "λ", "wavelength": "λ",
    "amplitudo": "A_wave", "amplitude": "A_wave",
    "kecepatan suara": "v_s", "sound speed": "v_s",
    # Listrik
    "tegangan listrik": "V", "voltage": "V",
    "arus": "I", "current": "I",
    "hambatan": "R", "resistance": "R",
    "kapasitansi": "C", "capacitance": "C",
    "induktansi": "L", "inductance": "L",
    "daya": "P_e", "power": "P_e",
    "frekuensi cutoff": "f_c",
    # Keuangan
    "modal": "PV", "capital": "PV", "investasi": "PV",
    "bunga": "r", "interest rate": "r",
    "hasil": "FV", "future value": "FV",
    "nilai akhir": "FV",
    "keuntungan": "profit",
    "npv": "NPV",
    # Energi kinetik/potensial
    "energi kinetik": "Ek", "kinetic energy": "Ek",
    "energi potensial": "Ep", "potential energy": "Ep",
    "energi": "Ek",  # default energi → kinetik
    # Mekanika Mesin (Engine Mechanics)
    "rasio kompresi": "CR", "compression ratio": "CR",
    "volume ruang bakar": "V_c", "ruang bakar": "V_c", "combustion chamber": "V_c", "volume clearance": "V_c",
    "langkah piston": "S", "stroke": "S", "langkah": "S",
    "diameter silinder": "B", "bore": "B",
    "jumlah silinder": "N_cyl", "silinder mesin": "N_cyl", "silinder": "N_cyl",
    "sudut pengapian": "θ_ign", "ignition angle": "θ_ign", "ignition timing": "θ_ign", "timing pengapian": "θ_ign",
    "kecepatan api": "v_f", "flame speed": "v_f",
    "sudut crank": "θ_crank", "crank angle": "θ_crank",
    "panjang stang seher": "L_con", "connecting rod": "L_con",
    "jari-jari crank": "r_crank", "crank radius": "r_crank",
    "kapasitas mesin": "V_d", "displacement": "V_d",
}

# Sort by key length descending to prioritize longer phrases (e.g. "panjang gelombang" over "panjang")
QUANTITY_TO_SYMBOL = dict(sorted(QUANTITY_TO_SYMBOL.items(), key=lambda item: len(item[0]), reverse=True))



# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeasuredValue:
    """Nilai terukur dengan satuan."""
    value: float
    original_unit: str         # unit asli dari teks ("bar", "mm", dsb.)
    si_value: float            # nilai setelah konversi ke SI
    si_unit: str               # satuan SI ("Pa", "m", dsb.)
    symbol: Optional[str] = None  # simbol variabel ("P", "D", dsb.)
    uncertainty: Optional[float] = None  # ketidakpastian (±)

    def __repr__(self):
        unc = f" ±{self.uncertainty:.3g}" if self.uncertainty else ""
        return f"{self.value} {self.original_unit} = {self.si_value:.6g} {self.si_unit}{unc}"


@dataclass
class ParsedProblem:
    """Hasil parsing story math problem."""
    original_text: str
    known: Dict[str, MeasuredValue]       # variabel yang diketahui
    unknown: List[str]                     # variabel yang dicari (simbol)
    unknown_descriptions: List[str]        # deskripsi yang dicari ("gaya piston")
    domain: str                            # domain teknik ("fluid_mechanics")
    domain_confidence: float               # 0.0 - 1.0
    constraints: List[str]                 # kondisi ("tidak melebihi", "minimal")
    question_type: str                     # "quantity", "verify", "derive"
    raw_numbers: List[Tuple[float, str]]   # semua angka+satuan yang ditemukan

    def summary(self) -> str:
        lines = [
            f"Problem: {self.original_text[:100]}...",
            f"Domain: {self.domain} (confidence={self.domain_confidence:.2f})",
            f"Known variables:",
        ]
        for sym, mv in self.known.items():
            lines.append(f"  {sym} = {mv}")
        lines.append(f"Unknown: {self.unknown}")
        lines.append(f"Seeking: {self.unknown_descriptions}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# STORY MATH PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class StoryMathParser:
    """
    Mengekstrak struktur matematis dari narasi bahasa manusia.

    Algoritma:
    1. Temukan semua angka + satuan dalam teks
    2. Asosiasikan setiap angka dengan variabel terdekat
    3. Normalisasi satuan ke SI
    4. Deteksi domain teknik dari keyword
    5. Ekstrak apa yang dicari (unknown)
    """

    # Regex untuk menemukan angka (integer, float, scientific notation)
    _NUMBER_PATTERN = re.compile(
        r'(?<![a-zA-Z])(-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*'
        r'(m³|m²|m3|m2|m/s²|m/s2|m/s|km/h|'
        r'μm|μf|μh|μa|μs|mΩ|kΩ|'
        r'mm²|mm2|mm|cm²|cm2|cm|km|nm|'
        r'kg|mg|g\b|'
        r'ms|μs|us|ns|'
        r'°[CcFf]|°[Kk]|'
        r'kpa|mpa|bar|psi|atm|pa\b|mmhg|torr|'
        r'kn|mn|n\b|kgf|lbf|'
        r'mj|kj|kwh|wh|cal\b|kcal\b|kkal|ev|btu|j\b|'
        r'kw|mw|hp|dk|w\b|'
        r'ghz|mhz|khz|hz|rpm|'
        r'kv|mv|v\b|ma\b|mh|uh|nf|pf|μf|uf|mf|'
        r'kohm|mohm|ohm|kΩ|mΩ|Ω\b|'
        r'ml|cc|cm3|cm³|m3|m³|l\b|liter|litre|'
        r'km²|km2|'
        r'lb\b|ft\b|in\b|'
        r'idr|usd|rp\b|'
        r'%\b|persen|'
        r'mol\b|db\b|'
        r'detik\b|menit\b|jam\b|hari\b|tahun\b|'
        r'second\b|minute\b|hour\b|day\b|year\b)',
        re.IGNORECASE
    )

    # Regex untuk menemukan variabel sebelum angka
    _VAR_BEFORE_PATTERN = re.compile(
        r'(\b(?:' + '|'.join(list(QUANTITY_TO_SYMBOL.keys())) + r')\b)\s*(?:=|:|\s+adalah\s+)?',
        re.IGNORECASE
    )

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Pre-compile domain pattern
        self._domain_patterns = {
            domain: re.compile(
                '|'.join(re.escape(sig) for sig in signals),
                re.IGNORECASE
            )
            for domain, signals in APPLIED_DOMAIN_SIGNALS.items()
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [StoryParser] {msg}")

    # ── 1. EXTRACT ALL NUMBERS + UNITS ────────────────────────────────────

    def extract_measurements(self, text: str) -> List[Tuple[float, str, int]]:
        """
        Temukan semua (angka, satuan, posisi) dalam teks.

        Return: list of (value, unit_str, char_position)
        """
        results = []
        for match in self._NUMBER_PATTERN.finditer(text):
            num_str = match.group(1).replace(',', '.')
            unit_str = match.group(2).strip()
            try:
                value = float(num_str)
                results.append((value, unit_str, match.start()))
            except ValueError:
                pass
        self._log(f"Found {len(results)} measurements: {[(v, u) for v, u, _ in results]}")
        return results

    # ── 2. NORMALIZE UNIT TO SI ───────────────────────────────────────────

    def normalize_to_si(self, value: float, unit: str) -> Tuple[float, str]:
        """
        Konversikan nilai dan satuan ke SI base unit.

        Return: (si_value, si_unit)
        """
        unit_lower = unit.lower().strip()

        # Handle special temperature (dengan offset)
        if unit in ("°C", "C", "celsius"):
            return value + 273.15, "K"
        if unit in ("°F", "F", "fahrenheit"):
            return (value - 32) * 5/9 + 273.15, "K"

        # Lookup dalam konversi table
        if unit_lower in UNIT_CONVERSIONS:
            factor, si_unit = UNIT_CONVERSIONS[unit_lower]
            return value * factor, si_unit

        # Coba dengan unit original (case-insensitive variations)
        for key, (factor, si_unit) in UNIT_CONVERSIONS.items():
            if key == unit_lower:
                return value * factor, si_unit

        # Tidak ditemukan — kembalikan as-is
        self._log(f"WARNING: Unit '{unit}' tidak dikenali, tidak dikonversi")
        return value, unit

    # ── 3. MATCH VARIABLE TO MEASUREMENT ─────────────────────────────────

    def _find_variable_for_measurement(
        self, text: str, pos: int, value: float, unit: str
    ) -> Optional[str]:
        unit_lower = unit.lower().strip()
        if unit_lower in ("rpm", "rev/s"):
            return "RPM"

        # Ambil 60 karakter sebelum angka
        window = text[max(0, pos - 70):pos].lower()
        window = re.sub(r'nya\b', '', window)

        # Coba cocokkan dengan QUANTITY_TO_SYMBOL
        best_match = None
        best_distance = 999

        for qty_name, symbol in QUANTITY_TO_SYMBOL.items():
            matches = list(re.finditer(r'\b' + re.escape(qty_name.lower()) + r'\b', window))
            if matches:
                idx = matches[-1].start()
                distance = len(window) - idx
                if distance < best_distance:
                    best_distance = distance
                    best_match = symbol

        # Fallback: coba inferensi dari satuan
        if best_match is None:
            unit_lower = unit.lower()
            if unit_lower in ("pa", "kpa", "mpa", "bar", "psi", "atm"):
                best_match = "P"
            elif unit_lower in ("mm", "cm", "m", "km", "in", "ft"):
                best_match = "L"  # generic length
            elif unit_lower in ("n", "kn", "mn", "kgf"):
                best_match = "F"
            elif unit_lower in ("hz", "khz", "mhz", "ghz"):
                best_match = "f"
            elif unit_lower in ("rpm", "rev/s"):
                best_match = "RPM"
            elif unit_lower in ("v", "kv", "mv"):
                best_match = "V"
            elif unit_lower in ("a", "ma", "μa"):
                best_match = "I"
            elif unit_lower in ("ω", "ohm", "kω", "mω"):
                best_match = "R"
            elif unit_lower in ("w", "kw", "mw", "hp"):
                best_match = "P_e"
            elif unit_lower in ("j", "kj", "mj", "cal", "kcal"):
                best_match = "Q"
            elif unit_lower in ("°c", "k", "°f"):
                best_match = "T"
            elif unit_lower in ("kg", "g", "mg"):
                best_match = "m"
            elif unit_lower in ("m3", "m³", "l", "liter", "litre", "ml", "cc", "cm3", "cm³"):
                best_match = "V"

        return best_match

    # ── 3b. DOMAIN CONTEXT POST-PROCESSING ──────────────────────────────

    def _apply_domain_context(self, known: Dict, domain: str, text: str) -> Dict:
        """
        Sesuaikan simbol variabel berdasarkan domain yang terdeteksi.
        """
        text_lower = text.lower()

        # Fluid mechanics: D → compute A = π(D/2)²
        if domain == 'fluid_mechanics' and 'D' in known and 'A' not in known:
            D = known['D'].si_value
            A = math.pi * (D / 2) ** 2
            known['A'] = MeasuredValue(
                value=A, original_unit='m²',
                si_value=A, si_unit='m²', symbol='A'
            )
            self._log(f"Auto-derive: A = π(D/2)² = {A:.6g} m² from D={D:.4g} m")

        # Engine Mechanics: D -> B (bore) and D -> compute A = π(D/2)²
        if domain == 'engine_mechanics' and 'D' in known and 'B' not in known:
            known['B'] = known['D']
            D = known['D'].si_value
            A = math.pi * (D / 2) ** 2
            known['A'] = MeasuredValue(
                value=A, original_unit='m²',
                si_value=A, si_unit='m²', symbol='A'
            )
            self._log(f"Auto-derive: B = D = {D:.4g} m and A = π(D/2)² = {A:.6g} m² from D={D:.4g} m")

        # Finance: jika ada 'r' (bunga time s), konversi ke rate desimal
        if domain == 'finance':
            # Pastikan 'r' adalah rate, bukan radius
            if 'r' in known:
                r_val = known['r'].si_value
                if r_val > 1.0:  # pasti salah interpretasi (harusnya 0.06 bukan 6)
                    known['r'] = MeasuredValue(
                        value=r_val, original_unit='%',
                        si_value=r_val / 100.0, si_unit='dimensionless', symbol='r'
                    )
                elif r_val > 157_000_000:  # ini tahun dalam detik, hapus
                    del known['r']

        # Thermodynamics: 'm_vol' → gunakan sebagai massa air
        if domain == 'thermodynamics' and 'm_vol' in known and 'm' not in known:
            # Untuk air: 1 liter ≈ 1 kg
            known['m'] = known['m_vol']

        # Kinematics / energy: rename 'm' mass dari volume
        # Pastikan 'm_vol' tidak masuk sebagai mass di domain kinematics
        if domain in ('kinematics', 'energy', 'structural') and 'm_vol' in known:
            del known['m_vol']

        return known

    # ── 4. DETECT DOMAIN ─────────────────────────────────────────────────

    def detect_domain(self, text: str) -> Tuple[str, float]:
        """
        Deteksi domain teknik dari teks.

        Return: (domain_name, confidence_score)
        """
        scores: Dict[str, int] = {}
        for domain, pattern in self._domain_patterns.items():
            matches = pattern.findall(text)
            if matches:
                scores[domain] = len(matches)

        if not scores:
            return "general_math", 0.5

        total = sum(scores.values())
        best = max(scores, key=scores.get)
        confidence = scores[best] / total if total > 0 else 0.5
        self._log(f"Domain scores: {scores}")
        return best, min(1.0, confidence + 0.2)

    # ── 5. EXTRACT UNKNOWNS ───────────────────────────────────────────────

    def extract_unknowns(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Ekstrak apa yang dicari (unknown).

        Return: (symbols, descriptions)
        """
        symbols = []
        descriptions = []

        for pattern_str, lang in UNKNOWN_VARIABLE_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(text)
            if match:
                desc = match.group(1).strip()
                descriptions.append(desc)
                # Map description ke simbol
                desc_lower = desc.lower()
                desc_lower = re.sub(r'nya\b', '', desc_lower).strip()
                for qty_name, symbol in QUANTITY_TO_SYMBOL.items():
                    if re.search(r'\b' + re.escape(qty_name) + r'\b', desc_lower):
                        if symbol not in symbols:
                            symbols.append(symbol)
                        break

        # Fallback: jika tidak ada simbol tapi ada deskripsi
        if descriptions and not symbols:
            symbols.append("?")

        self._log(f"Unknowns: symbols={symbols}, desc={descriptions}")
        return symbols, descriptions

    # ── 6. EXTRACT CONSTRAINTS ────────────────────────────────────────────

    def extract_constraints(self, text: str) -> List[str]:
        """
        Ekstrak kondisi/batasan dari teks.
        """
        constraints = []
        constraint_patterns = [
            r'tidak (?:lebih dari|melebihi|boleh|boleh melebihi)\s+[\d\s\w]+',
            r'minimal\s+[\d\s\w]+',
            r'maksimal\s+[\d\s\w]+',
            r'paling (?:sedikit|banyak)\s+[\d\s\w]+',
            r'asalkan\s+[\w\s]+',
            r'jika\s+[\w\s]+',
            r'dengan syarat\s+[\w\s]+',
            r'at (?:most|least)\s+[\d\s\w]+',
            r'no (?:more|less) than\s+[\d\s\w]+',
            r'maximum\s+[\d\s\w]+',
            r'minimum\s+[\d\s\w]+',
        ]
        for pat in constraint_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            constraints.extend(matches[:2])  # max 2 per pattern
        return constraints

    # ── 7. MAIN PARSE METHOD ──────────────────────────────────────────────

    def _preprocess_text(self, text: str) -> str:
        """
        Normalisasi teks sebelum parsing — tangani pola bahasa alami yang tidak standar.
        """
        # "X derajat celsius" → "X°C"
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s+derajat\s+celsius\b', r'\1°C', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s+derajat\s+kelvin\b', r'\1K', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s+derajat\s+fahrenheit\b', r'\1°F', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s+celsius\b', r'\1°C', text, flags=re.IGNORECASE)
        # "X ohm" → "Xohm" (untuk regex tangkap)
        text = re.sub(r'(\d+(?:[.,]\d+)?)\s+ohm\b', r'\1ohm', text, flags=re.IGNORECASE)
        return text

    def _extract_special_patterns(self, text: str) -> Dict[str, 'MeasuredValue']:
        """
        Ekstrak pola khusus yang tidak ditangkap oleh regex angka+satuan.
        Contoh: "5 tahun" → n=5, "6% per tahun" → r=0.06, "2 liter" → m_vol=0.002 m³
        """
        specials = {}

        # "X tahun" → n (periode keuangan), bukan waktu
        for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s+tahun\b', text, re.IGNORECASE):
            n_val = float(m.group(1).replace(',', '.'))
            specials['n'] = MeasuredValue(
                value=n_val, original_unit='tahun',
                si_value=n_val, si_unit='period', symbol='n'
            )

        # "X% per tahun" → r (interest rate desimal)
        for pat in [r'(\d+(?:[.,]\d+)?)\s*%\s*per\s*tahun', r'bunga\s+(\d+(?:[.,]\d+)?)\s*%',
                    r'(\d+(?:[.,]\d+)?)\s*%\s*per\s*year', r'(\d+(?:[.,]\d+)?)\s*persen\s*per\s*tahun']:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                r_val = float(m.group(1).replace(',', '.')) / 100.0
                specials['r'] = MeasuredValue(
                    value=r_val, original_unit='%/tahun',
                    si_value=r_val, si_unit='dimensionless', symbol='r'
                )
                break

        # "Rp X" / "X rupiah" / "Rp X juta" → PV
        m = re.search(r'(?:rp\.?\s*|rupiah\s*)(\d+(?:[.,]\d+)?(?:\s*(?:juta|miliar|ribu))?)', text, re.IGNORECASE)
        if m:
            val_str = m.group(1).strip()
            mult = 1
            if 'juta' in val_str.lower(): mult = 1_000_000
            elif 'miliar' in val_str.lower(): mult = 1_000_000_000
            elif 'ribu' in val_str.lower(): mult = 1_000
            num = re.search(r'\d+(?:[.,]\d+)?', val_str)
            if num:
                pv_val = float(num.group().replace(',', '.')) * mult
                specials['PV'] = MeasuredValue(
                    value=pv_val, original_unit='IDR',
                    si_value=pv_val, si_unit='IDR', symbol='PV'
                )

        # "X liter" → m_vol (massa/volume air → untuk termodinamika)
        m = re.search(r'(\d+(?:[.,]\d+)?)\s+liter\b', text, re.IGNORECASE)
        if m and 'PV' not in specials:  # jangan override PV keuangan
            l_val = float(m.group(1).replace(',', '.'))
            # Untuk termodinamika: 1 liter air = 1 kg
            specials['m'] = MeasuredValue(
                value=l_val, original_unit='liter',
                si_value=l_val * 1.0, si_unit='kg',  # 1L air ≈ 1 kg
                symbol='m'
            )
            # Juga tambahkan sebagai volume
            specials['V_liq'] = MeasuredValue(
                value=l_val, original_unit='liter',
                si_value=l_val * 1e-3, si_unit='m³', symbol='V_liq'
            )

        # Tangkap T₁ dan T₂ dari pola "dari X°C ke Y°C"
        m_range = re.search(
            r'dari\s+(\d+(?:[.,]\d+)?)\s*°?C\s+ke\s+(\d+(?:[.,]\d+)?)\s*°?C',
            text, re.IGNORECASE
        )
        if m_range:
            t1 = float(m_range.group(1)) + 273.15
            t2 = float(m_range.group(2)) + 273.15
            specials['T_1'] = MeasuredValue(value=float(m_range.group(1)), original_unit='°C', si_value=t1, si_unit='K', symbol='T_1')
            specials['T_2'] = MeasuredValue(value=float(m_range.group(2)), original_unit='°C', si_value=t2, si_unit='K', symbol='T_2')
            specials['ΔT'] = MeasuredValue(value=t2-t1, original_unit='K', si_value=t2-t1, si_unit='K', symbol='ΔT')

        # Tangkap "sepanjang X m" atau "panjang X m" → L
        m_L = re.search(r'(?:sepanjang|panjang(?:\s+tabung)?)\s+(\d+(?:[.,]\d+)?)\s*m\b', text, re.IGNORECASE)
        if m_L:
            L_val = float(m_L.group(1))
            specials['L'] = MeasuredValue(value=L_val, original_unit='m', si_value=L_val, si_unit='m', symbol='L')

        # Tangkap "X silinder" -> N_cyl
        for m in re.finditer(r'(\d+)\s+silinder\b', text, re.IGNORECASE):
            n_val = float(m.group(1))
            specials['N_cyl'] = MeasuredValue(
                value=n_val, original_unit='silinder',
                si_value=n_val, si_unit='dimensionless', symbol='N_cyl'
            )

        # Tangkap "kompresi X" / "kompresi X:1" -> CR
        for m in re.finditer(r'kompresi\s+(\d+(?:[.,]\d+)?)(?:\s*:\s*1)?\b', text, re.IGNORECASE):
            cr_val = float(m.group(1).replace(',', '.'))
            specials['CR'] = MeasuredValue(
                value=cr_val, original_unit='ratio',
                si_value=cr_val, si_unit='dimensionless', symbol='CR'
            )

        return specials

    def parse(self, text: str) -> ParsedProblem:
        """
        Entry point utama: parse cerita → ParsedProblem terstruktur.
        """
        self._log(f"Parsing: '{text[:80]}...'")

        # Step 0: Preprocess text
        text_processed = self._preprocess_text(text)

        # Step 1: Extract semua angka + satuan
        raw_measurements = self.extract_measurements(text_processed)

        # Step 2: Bangun known variables dict
        known: Dict[str, MeasuredValue] = {}
        symbol_counter: Dict[str, int] = {}  # untuk deduplikasi (D1, D2, dsb.)

        for value, unit, pos in raw_measurements:
            si_value, si_unit = self.normalize_to_si(value, unit)
            symbol = self._find_variable_for_measurement(text_processed, pos, value, unit)

            # Skip "tahun" sebagai waktu jika domain kemungkinan finance
            # (akan di-handle oleh _extract_special_patterns)
            if unit.lower() == 'tahun':
                continue

            if symbol:
                # Handle duplikasi simbol
                if symbol in known:
                    count = symbol_counter.get(symbol, 1) + 1
                    symbol_counter[symbol] = count
                    symbol = f"{symbol}_{count}"

                known[symbol] = MeasuredValue(
                    value=value,
                    original_unit=unit,
                    si_value=si_value,
                    si_unit=si_unit,
                    symbol=symbol,
                )

        # Step 2b: Inject special patterns (tahun, %, liter, range, dll.)
        specials = self._extract_special_patterns(text_processed)
        for sym, mv in specials.items():
            if sym not in known:  # jangan override yang sudah ada
                known[sym] = mv

        # Step 3: Detect domain
        domain, confidence = self.detect_domain(text)

        # Step 3b: Apply domain context (auto-derive A from D, fix finance r, etc.)
        known = self._apply_domain_context(known, domain, text)

        # Step 4: Extract unknowns
        unknown_syms, unknown_descs = self.extract_unknowns(text)

        # Step 5: Extract constraints
        constraints = self.extract_constraints(text)

        # Step 6: Determine question type
        q_type = "quantity"
        if any(re.search(p, text, re.IGNORECASE) for p in [
            r'\bverifikasi\b', r'\bbuktikan\b', r'\bapakah benar\b', r'\bprove\b', r'\bverify\b'
        ]):
            q_type = "verify"
        elif any(re.search(p, text, re.IGNORECASE) for p in [
            r'\bturunkan\b', r'\bdeduce\b', r'\bderive\b', r'\btemukan rumus\b'
        ]):
            q_type = "derive"

        problem = ParsedProblem(
            original_text=text,
            known=known,
            unknown=unknown_syms,
            unknown_descriptions=unknown_descs,
            domain=domain,
            domain_confidence=confidence,
            constraints=constraints,
            question_type=q_type,
            raw_numbers=[(v, u) for v, u, _ in raw_measurements],
        )

        if self.verbose:
            print(problem.summary())

        return problem


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_parser_instance: Optional[StoryMathParser] = None

def get_story_parser(verbose: bool = False) -> StoryMathParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = StoryMathParser(verbose=verbose)
    return _parser_instance


def parse_problem(text: str) -> ParsedProblem:
    """Shortcut: parse langsung dari string."""
    return get_story_parser().parse(text)
