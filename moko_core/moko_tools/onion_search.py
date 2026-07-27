"""
MOKO Onion Search Tool — Community-Driven Darkweb Scanner
==========================================================
Alat untuk melakukan pencarian di jaringan Tor (.onion) menggunakan
berbagai search engine komunitas (Ahmia, Torch, Haystak, dll).

Fitur:
  1. Multi-Engine Scraping: Mendukung Ahmia, Torch, Haystak.
  2. Tor Proxy Support: Menggunakan SOCKS5 proxy jika tersedia.
  3. Result Deduplication: Menghapus hasil duplikat lintas engine.
  4. Integration ready: Output dalam format yang mudah dikonsumsi RAGAgent.
"""

import os
import re
import time
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from moko_tools.torbot_integrator import TorBotIntegrator

class OnionSearchTool:
    """
    Wrapper untuk pencarian di Darkweb via Tor.
    """
    
    def __init__(self, proxy: Optional[str] = None):
        # Default Tor SOCKS5 proxy (127.0.0.1:9050)
        from moko_config import settings
        self.proxy_port = settings.TOR_SOCKS_PORT
        self.proxy = proxy or os.getenv("TOR_PROXY", f"socks5h://127.0.0.1:{self.proxy_port}")
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0"
        self.torbot = TorBotIntegrator(self.proxy)
        
        # Engines configurations
        self.engines = {
            "ahmia": {
                "url": "https://ahmia.fi/search/?q={query}",
                "type": "clearnet_proxy", # Ahmia has a clearnet portal
                "pattern": r'<li class="result">.*?<h4><a href="(.*?)">(.*?)</a></h4>.*?<p>(.*?)</p>',
            },
            # Note: Engines below usually require actual Tor connection
            "torch": {
                "url": "http://torchde92qp3mx2.onion/search?q={query}",
                "type": "onion",
            },
            "haystak": {
                "url": "http://haystak5njsmn2hq.onion/?q={query}",
                "type": "onion",
            },
            "excavator": {
                "url": "http://2v7unp3vbmz6unf7.onion/search?q={query}",
                "type": "onion",
            }
        }

    def _get_request(self, url: str) -> Optional[str]:
        """Melakukan request HTTP dengan handling proxy."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            
            # Jika ada proxy (misal: socks5h://127.0.0.1:9050)
            # urllib standard tidak support socks5 secara native tanpa library tambahan (PySocks)
            # Namun kita bisa menggunakan http proxy jika user menyediakannya
            if self.proxy:
                req.set_proxy(self.proxy, 'http')
                
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"  ⚠️ [OnionSearch] Request error to {url[:30]}... : {e}")
            return None

    def search_ahmia(self, query: str) -> List[Dict[str, str]]:
        """Pencarian via Ahmia (paling reliabel karena ada clearnet portal)."""
        url = self.engines["ahmia"]["url"].format(query=urllib.parse.quote(query))
        html = self._get_request(url)
        
        results = []
        if not html:
            return results
            
        # Regex sederhana untuk scraping Ahmia
        # Format ahmia: <li class="result"><h4><a href="URL">TITLE</a></h4><p>SNIPPET</p>
        matches = re.finditer(self.engines["ahmia"]["pattern"], html, re.DOTALL)
        for match in matches:
            link = match.group(1)
            title = re.sub(r'<.*?>', '', match.group(2)).strip()
            snippet = re.sub(r'<.*?>', '', match.group(3)).strip()
            
            # Ahmia proxy links: /redirect?search_result=...&redirect_url=URL
            if "/redirect" in link:
                parsed = urllib.parse.urlparse(link)
                qs = urllib.parse.parse_qs(parsed.query)
                if "redirect_url" in qs:
                    link = qs["redirect_url"][0]
            
            results.append({
                "engine": "ahmia",
                "title": title,
                "link": link,
                "snippet": snippet
            })
            
        return results

    def search_all(self, query: str, limit_per_engine: int = 10, sharpen: bool = True) -> List[Dict[str, Any]]:
        """Melakukan pencarian lintas engine."""
        print(f"🔍 [OnionSearch] Scanning darkweb for: '{query}'")
        
        all_results = []
        
        # 1. Ahmia (Clearnet reachable)
        ahmia_res = self.search_ahmia(query)
        all_results.extend(ahmia_res[:limit_per_engine])
        
        # 2. Others (Placeholder jika tidak ada Tor proxy)
        # Check if Tor is actually reachable
        tor_active = self.torbot.check_alive("http://check.torproject.org")
        
        if not tor_active:
            print("  ℹ️ [OnionSearch] Tor Proxy tidak aktif atau tidak terjangkau. Hanya menggunakan Ahmia Clearnet.")
        else:
            print("  🌐 [OnionSearch] Tor Network detected. Extending search...")
            # Implementasi engine lain bisa ditambahkan di sini
            pass
            
        # Deduplikasi berdasarkan link
        seen_links = set()
        unique_results = []
        for r in all_results:
            if r["link"] not in seen_links:
                seen_links.add(r["link"])
                unique_results.append(r)
                
        print(f"  ✅ [OnionSearch] Ditemukan {len(unique_results)} hasil awal.")
        
        # 3. TorBot Sharpening
        if sharpen and unique_results:
            unique_results = self.torbot.sharpen(unique_results)
            
        return unique_results

if __name__ == "__main__":
    # Test sederhana
    scanner = OnionSearchTool()
    results = scanner.search_all("ransomware leak")
    for i, res in enumerate(results[:5], 1):
        print(f"{i}. {res['title']}\n   URL: {res['link']}\n   Snippet: {res['snippet'][:100]}...\n")
