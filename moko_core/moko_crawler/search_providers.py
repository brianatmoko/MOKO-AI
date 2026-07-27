"""
moko_crawler/search_providers.py
================================
Daftar provider search engine untuk pencarian situs .onion.
Di sini Anda dapat dengan mudah menambahkan search engine baru dengan mewarisi kelas BaseSearchEngine.
"""

import logging
import urllib.parse
import re
from typing import List
from bs4 import BeautifulSoup
from .config import TorConfig
from .tor_crawler import TorSession

log = logging.getLogger("moko_crawler.search_providers")


# ── Base Class Search Engine ──────────────────────────────────────────────────
class BaseSearchEngine:
    """Kelas abstrak/base untuk seluruh provider search engine."""
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def search(self, query: str, limit: int, tor_cfg: TorConfig) -> List[str]:
        """
        Jalankan pencarian.
        Returns: List URL .onion unik yang ditemukan.
        """
        raise NotImplementedError("Setiap Search Engine harus mengimplementasikan method search().")


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 1: Ahmia Search Engine (Onion / Clearnet via Tor)
# ═══════════════════════════════════════════════════════════════════════════════
class AhmiaSearchEngine(BaseSearchEngine):
    """Pencarian menggunakan Ahmia Index."""

    def __init__(self, use_onion: bool = False):
        self.use_onion = use_onion
        self.onion_url = "http://juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion/search/"
        self.clearnet_url = "https://ahmia.fi/search/"

    def search(self, query: str, limit: int, tor_cfg: TorConfig) -> List[str]:
        base_url = self.onion_url if self.use_onion else self.clearnet_url
        # Ahmia memerlukan parameter token ac2922 agar memproses search request
        params = {"q": query, "ac2922": "2008d2"}
        search_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        session = TorSession(tor_cfg)
        try:
            resp = session.get(search_url)
            if not resp or resp.status_code != 200:
                log.warning(f"[{self.name}] HTTP Error/No Response. Status: {resp.status_code if resp else 'None'}")
                return []
                
            soup = BeautifulSoup(resp.text, "html.parser")
            onion_urls = []
            
            # Parsing standar ahmia
            for li in soup.find_all("li", class_="result"):
                a_tag = li.find("a", href=True)
                if not a_tag:
                    continue
                href = a_tag["href"]
                # Parse redirect url
                parsed_href = urllib.parse.urlparse(href)
                if parsed_href.path == "/redirect" or "redirect" in parsed_href.path:
                    query_params = urllib.parse.parse_qs(parsed_href.query)
                    redirect_url = query_params.get("search_result")
                    if redirect_url:
                        href = redirect_url[0]
                
                from .url_manager import normalize_url
                norm = normalize_url(href)
                if norm and norm not in onion_urls:
                    onion_urls.append(norm)
                    if len(onion_urls) >= limit:
                        break
            
            # Fallback regex
            if not onion_urls:
                raw_onions = re.findall(r'(https?://[a-z2-7]{16,56}\.onion[^\s"\'>]*)', resp.text)
                for url in raw_onions:
                    from .url_manager import normalize_url
                    norm = normalize_url(url)
                    if norm and "juhanurmi" not in norm and norm not in onion_urls:
                        onion_urls.append(norm)
                        if len(onion_urls) >= limit:
                            break
                            
            return onion_urls
        except Exception as e:
            log.error(f"[{self.name}] Error: {e}")
            return []
        finally:
            session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 2: DuckDuckGo HTML Engine (Mengekstrak onion link dari hasil pencarian web)
# ═══════════════════════════════════════════════════════════════════════════════
class DuckDuckGoHtmlEngine(BaseSearchEngine):
    """Pencarian menggunakan DuckDuckGo HTML version (tanpa JavaScript, ramah Tor SOCKS5)."""

    def __init__(self):
        # Versi HTML ramah bot/Tor
        self.search_url = "https://html.duckduckgo.com/html/"

    def search(self, query: str, limit: int, tor_cfg: TorConfig) -> List[str]:
        # Tambahkan filter "site:onion" atau ".onion" agar DDG memprioritaskan link onion
        full_query = f"{query} site:onion"
        
        session = TorSession(tor_cfg)
        try:
            # DuckDuckGo HTML menerima pencarian via POST
            data = {"q": full_query}
            
            # Request ke DDG
            resp = session._session.post(
                self.search_url,
                data=data,
                timeout=(tor_cfg.connect_timeout, tor_cfg.read_timeout)
            )
            
            if resp.status_code != 200:
                log.warning(f"[{self.name}] HTTP Error/No Response. Status: {resp.status_code}")
                return []
                
            soup = BeautifulSoup(resp.text, "html.parser")
            onion_urls = []
            
            # Parsing link dari hasil pencarian DuckDuckGo HTML
            for a_tag in soup.find_all("a", class_="result__snippet", href=True):
                href = a_tag["href"]
                # Parse redirect url jika ada
                parsed = urllib.parse.urlparse(href)
                if parsed.netloc == "duckduckgo.com" and parsed.path == "/l/":
                    qs = urllib.parse.parse_qs(parsed.query)
                    uddg = qs.get("uddg")
                    if uddg:
                        href = uddg[0]
                
                # Ekstrak onion link dari teks deskripsi atau url rujukan
                from .url_manager import normalize_url
                norm = normalize_url(href)
                if norm and norm not in onion_urls:
                    onion_urls.append(norm)
                    if len(onion_urls) >= limit:
                        break
                        
            # Fallback regex untuk mencari .onion mentah di seluruh halaman
            if len(onion_urls) < limit:
                raw_onions = re.findall(r'(https?://[a-z2-7]{16,56}\.onion[^\s"\'>]*)', resp.text)
                for url in raw_onions:
                    from .url_manager import normalize_url
                    norm = normalize_url(url)
                    if norm and norm not in onion_urls:
                        onion_urls.append(norm)
                        if len(onion_urls) >= limit:
                            break
                            
            return onion_urls
        except Exception as e:
            log.error(f"[{self.name}] Error: {e}")
            return []
        finally:
            session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 3: Torch Search Engine (WordPress Onion Directory via Tor SOCKS5)
# ═══════════════════════════════════════════════════════════════════════════════
class TorchSearchEngine(BaseSearchEngine):
    """Pencarian menggunakan Torch Onion Search Engine (WordPress-based)."""

    def __init__(self):
        self.search_url = "http://rz6wxogwwbqdadlncnp2q26kbgcbbaqnitzueohj73fzmlx3mt467wqd.onion/"

    def search(self, query: str, limit: int, tor_cfg: TorConfig) -> List[str]:
        # Torch menggunakan standar search query parameter 's'
        params = {"s": query}
        search_query_url = f"{self.search_url}?{urllib.parse.urlencode(params)}"
        
        session = TorSession(tor_cfg)
        try:
            log.info(f"[{self.name}] Mengirim request pencarian ke Torch: {search_query_url}")
            resp = session.get(search_query_url)
            if not resp or resp.status_code != 200:
                log.warning(f"[{self.name}] HTTP Error/No Response. Status: {resp.status_code if resp else 'None'}")
                return []
                
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract page_id links (links pointing to internal Torch pages that contain details of search results)
            page_links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                # Cek apakah link ini mengarah ke page_id internal Torch
                if "page_id=" in href and "rz6wxogwwbqdadln" in href:
                    if href not in page_links:
                        page_links.append(href)
                        
            log.info(f"[{self.name}] Menemukan {len(page_links)} sub-halaman hasil pencarian.")
            
            onion_urls = []
            # Untuk setiap sub-halaman (page_id), kunjungi dan ambil link onion eksternalnya
            # Batasi kunjungan sub-halaman hanya sampai kita mencapai 'limit'
            for page_url in page_links:
                if len(onion_urls) >= limit:
                    break
                    
                log.debug(f"[{self.name}] Memproses sub-halaman hasil: {page_url}")
                page_resp = session.get(page_url)
                if not page_resp or page_resp.status_code != 200:
                    continue
                    
                page_soup = BeautifulSoup(page_resp.text, "html.parser")
                
                # Ekstrak link onion eksternal yang bukan domain Torch sendiri
                for a_tag in page_soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if ".onion" in href and "rz6wxogwwbqdadln" not in href:
                        from .url_manager import normalize_url
                        norm = normalize_url(href)
                        if norm and norm not in onion_urls:
                            onion_urls.append(norm)
                            log.info(f"[{self.name}] Berhasil mengekstrak onion: {norm}")
                            break # Tiap sub-halaman biasanya hanya memuat satu link target utama
                            
            return onion_urls
        except Exception as e:
            log.error(f"[{self.name}] Error: {e}")
            return []
        finally:
            session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 4: Custom Search Engine Placeholder (Bisa Anda isi di masa mendatang!)
# ═══════════════════════════════════════════════════════════════════════════════
class CustomSearchEngine(BaseSearchEngine):
    """
    Template kelas untuk menambahkan search engine ke-2 atau ke-3 buatan Anda (misal: Torch, Haystak, dll).
    Cukup sesuaikan url, method request, dan BeautifulSoup parser-nya.
    """

    def __init__(self):
        # Contoh: letakkan onion URL search engine di sini
        self.search_url = "http://torchdeedp3i2jigzjdmfpn5ttjhthh5wbmda2rr3jvqjg5p77c54dqd.onion/search"

    def search(self, query: str, limit: int, tor_cfg: TorConfig) -> List[str]:
        # IMPLEMENTASI PENCARIAN ANDA DI SINI
        log.info(f"[{self.name}] Memulai pencarian kustom (sementara dilewati/mock)...")
        
        # Contoh mock data / parsing kerangka:
        # session = TorSession(tor_cfg)
        # resp = session.get(f"{self.search_url}?query={query}")
        # ... parse logic ...
        
        # Kembalikan list kosong sebagai default agar beralih ke engine berikutnya jika belum diimplementasikan
        return []
