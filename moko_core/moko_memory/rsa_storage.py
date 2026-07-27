"""
MOKO RSA Storage (Compatibility Layer)
======================================
This file provides compatibility for older modules that still import RSAStorage.
It maps RSAStorage to the new OmniStorageEngine.
"""

from pathlib import Path
from typing import Optional
from moko_config import settings
from moko_memory.omni_storage import OmniStorageEngine

# Export DEFAULT_DOMAIN for compatibility
DEFAULT_DOMAIN = "general"

# Export ROOT_OMNI_DIR for compatibility
ROOT_OMNI_DIR = Path(settings.OMNI_DIR)

# Keyword mapping: domain keywords -> domain name
_DOMAIN_KEYWORD_MAP = {
    "code": [
        "python", "javascript", "java", "c++", "c#", "golang", "rust", "php", "ruby",
        "typescript", "kotlin", "swift", "scala", "programming", "coding", "software",
        "algorithm", "data structure", "api", "backend", "frontend", "web", "database",
        "sql", "nosql", "git", "docker", "kubernetes", "linux", "bash", "shell",
        "framework", "library", "debug", "compile", "function", "class", "object",
        "async", "thread", "recursion", "sorting", "machine learning", "neural",
        "deep learning", "pytorch", "tensorflow", "numpy", "pandas", "llm", "ai model",
    ],
    "math": [
        "math", "matematika", "calculus", "kalkulus", "algebra", "aljabar", "geometry",
        "geometri", "statistics", "statistik", "probability", "probabilitas",
        "trigonometry", "trigonometri", "linear algebra", "differential", "integral",
        "matrix", "matriks", "vector", "vektor", "eigenvalue", "fourier", "laplace",
        "topology", "number theory", "combinatorics", "graph theory", "optimization",
        "physics", "fisika", "formula", "equation", "persamaan", "theorem",
    ],
    "science": [
        "biology", "biologi", "chemistry", "kimia", "science", "sains", "quantum",
        "astronomy", "astronomi", "geology", "geologi", "ecology", "ekologi",
        "neuroscience", "genetics", "genetika", "molecule", "atom", "dna", "rna",
        "cell", "organism", "evolution", "evolusi",
    ],
    "history": [
        "history", "sejarah", "historical", "ancient", "medieval", "war", "perang",
        "civilization", "peradaban", "empire", "kerajaan", "dynasty", "revolution",
        "revolusi", "colonial", "colonial", "independence", "kemerdekaan",
    ],
    "language": [
        "language", "bahasa", "grammar", "tata bahasa", "vocabulary", "kosakata",
        "translation", "terjemahan", "linguistics", "linguistik", "semantics",
        "syntax", "morphology", "phonology", "literature", "sastra", "novel", "poem",
    ],
    "philosophy": [
        "philosophy", "filsafat", "ethics", "etika", "logic", "logika", "epistemology",
        "ontology", "metaphysics", "consciousness", "existentialism", "stoicism",
        "phenomenology", "rationalism", "empiricism",
    ],
    "finance": [
        "finance", "keuangan", "economics", "ekonomi", "investment", "investasi",
        "stock", "saham", "crypto", "blockchain", "trading", "market", "pasar",
        "banking", "perbankan", "money", "uang", "inflation", "inflasi", "gdp",
        "accounting", "akuntansi", "tax", "pajak", "budget", "anggaran",
    ],
    "health": [
        "health", "kesehatan", "medical", "medis", "medicine", "obat", "doctor",
        "dokter", "hospital", "rumah sakit", "disease", "penyakit", "symptom",
        "gejala", "nutrition", "gizi", "diet", "exercise", "olahraga", "mental health",
        "psychology", "psikologi", "anatomy", "anatomi",
    ],
}


def map_topic_to_domain(topic: str) -> str:
    """
    Map a topic string to the best matching Omni domain name.

    Scans the topic text against keyword lists for each domain.
    Falls back to 'general' if no domain matches.

    Args:
        topic: The topic string (e.g., "Python sorting algorithms").

    Returns:
        A domain string suitable for use with OmniStorageEngine
        (e.g., 'code', 'math', 'general').
    """
    if not topic:
        return DEFAULT_DOMAIN

    topic_lower = topic.lower()
    best_domain = DEFAULT_DOMAIN
    best_score = 0

    for domain, keywords in _DOMAIN_KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in topic_lower)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def get_domain_path(domain: str, root_omni_dir: Optional[Path] = None) -> Path:
    """Helper to get the path for a specific domain."""
    root = Path(root_omni_dir) if root_omni_dir else ROOT_OMNI_DIR
    return root / domain

class RSAStorage(OmniStorageEngine):
    """
    Compatibility class that inherits from OmniStorageEngine.
    """
    def __init__(self, domain: str = DEFAULT_DOMAIN, root_omni_dir: Optional[Path] = None):
        super().__init__(domain=domain, root_omni_dir=root_omni_dir)
