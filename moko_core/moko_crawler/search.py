"""
moko_crawler/search.py
======================
Modul runner/orchestrator pencarian link onion.
Mengeksekusi chain of search engines secara berurutan (fallback/failover).
Jika satu search engine gagal/kembali [], otomatis beralih ke engine berikutnya.
"""

import logging
from typing import List, Optional

from .config import TOR_CONFIG, TorConfig
from .search_providers import TorchSearchEngine, AhmiaSearchEngine, DuckDuckGoHtmlEngine, CustomSearchEngine

log = logging.getLogger("moko_crawler.search")

# ── Chain default dari search engines yang aktif ──────────────────────────────
DEFAULT_ENGINES = [
    TorchSearchEngine(),                   # Torch Onion Directory via Tor
    AhmiaSearchEngine(use_onion=False),    # Ahmia clearnet via SOCKS5
    DuckDuckGoHtmlEngine(),                # DuckDuckGo HTML via SOCKS5
    CustomSearchEngine(),                  # Placeholder Kustom
]

def search_onion(
    query: str,
    limit: int = 5,
    use_onion_engine: bool = False,
    tor_cfg: TorConfig = TOR_CONFIG,
    engine_names: Optional[List[str]] = None
) -> List[str]:
    """
    Mencari keyword menggunakan chain search engines yang tersedia.
    Jika satu search engine mengembalikan hasil kosong [], otomatis berpindah ke yang berikutnya.
    
    Args:
        query: Kata kunci pencarian.
        limit: Batas maksimal hasil pencarian teratas (default: 5).
        use_onion_engine: Khusus untuk Ahmia, apakah menggunakan onion address.
        tor_cfg: Konfigurasi Tor proxy.
        engine_names: List nama engine tertentu untuk dipaksa dijalankan (opsional).
        
    Returns:
        List URL .onion unik teratas yang ditemukan.
    """
    log.info(f"Memulai pencarian berantai untuk query: '{query}' (limit: {limit})")
    
    # Filter engine list jika ditentukan user
    engines = DEFAULT_ENGINES
    if use_onion_engine:
        # Update AhmiaSearchEngine untuk menggunakan onion address
        engines = [
            TorchSearchEngine(),
            AhmiaSearchEngine(use_onion=True),
            DuckDuckGoHtmlEngine(),
            CustomSearchEngine()
        ]

    if engine_names:
        filtered = []
        name_map = {}
        for e in engines:
            name_map[e.name.lower()] = e
        # Tambahan aliasing praktis
        name_map["torch"] = next((e for e in engines if isinstance(e, TorchSearchEngine)), None)
        name_map["ahmia"] = next((e for e in engines if isinstance(e, AhmiaSearchEngine)), None)
        name_map["duckduckgo"] = next((e for e in engines if isinstance(e, DuckDuckGoHtmlEngine)), None)
        name_map["ddg"] = name_map["duckduckgo"]
        name_map["custom"] = next((e for e in engines if isinstance(e, CustomSearchEngine)), None)
        
        for name in engine_names:
            name_clean = name.strip().lower()
            if name_clean in name_map:
                filtered.append(name_map[name_clean])
            else:
                log.warning(f"Search engine '{name}' tidak dikenal. Dilewati.")
        if filtered:
            engines = filtered

    # Jalankan Chain of Responsibility pencarian
    for idx, engine in enumerate(engines):
        log.info(f"Mencoba mencari dengan engine [{idx + 1}/{len(engines)}]: {engine.name}...")
        try:
            results = engine.search(query, limit, tor_cfg)
            if results:
                log.info(f"✅ {engine.name} berhasil menemukan {len(results)} onion links!")
                return results
            else:
                log.warning(f"⚠️  {engine.name} mengembalikan 0 hasil. Beralih ke engine berikutnya...")
        except Exception as e:
            log.error(f"❌ Error saat menggunakan engine {engine.name}: {e}. Beralih ke engine berikutnya...")
            
    log.error("❌ Semua search engine dalam chain gagal mengembalikan hasil pencarian.")
    return []
