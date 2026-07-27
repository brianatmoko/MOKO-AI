"""
MOKO TorBot Integrator — Darkweb Intel Sharpener
================================================
Alat untuk "mempertajam" informasi dari link onion yang ditemukan.
Terinspirasi dari TorBot komunitas untuk crawling dan intelijen.

Fitur:
  1. Link Verification: Memastikan link .onion masih aktif.
  2. Metadata Extraction: Mengambil title dan meta-description.
  3. Intel Harvesting: Mencari email atau link tambahan (placeholder).
  4. Proxy Support: Terintegrasi dengan daemon Tor via SOCKS5.
"""

import re
import requests
from typing import List, Dict, Any, Optional

class TorBotIntegrator:
    """
    Sistem intelijen untuk memperdalam analisis link onion.
    """
    
    def __init__(self, socks_proxy: str = "socks5h://127.0.0.1:9050"):
        self.proxies = {
            'http': socks_proxy,
            'https': socks_proxy
        }
        self.timeout = 20
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0"
        self.headers = {"User-Agent": self.user_agent}

    def check_alive(self, url: str) -> bool:
        """Mengecek apakah link onion aktif."""
        try:
            # Gunakan HEAD request agar cepat
            response = requests.head(url, proxies=self.proxies, headers=self.headers, timeout=self.timeout)
            return response.status_code < 400
        except:
            return False

    def get_metadata(self, url: str) -> Dict[str, str]:
        """Mengambil metadata dasar dari halaman onion."""
        intel = {"url": url, "status": "down", "title": "", "emails": []}
        try:
            response = requests.get(url, proxies=self.proxies, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                intel["status"] = "up"
                # Extract Title
                title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE)
                if title_match:
                    intel["title"] = title_match.group(1).strip()
                
                # Simple Email Harvester (seperti TorBot)
                emails = re.findall(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', response.text, re.IGNORECASE)
                intel["emails"] = list(set(emails))
        except Exception as e:
            intel["error"] = str(e)
            
        return intel

    def sharpen(self, search_results: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Mempertajam hasil pencarian dengan melakukan verifikasi dan ekstraksi intel.
        Hanya memproses N link teratas untuk efisiensi.
        """
        print(f"🗡️ [TorBot] Sharpening {min(len(search_results), limit)} results...")
        sharpened = []
        
        for res in search_results[:limit]:
            url = res.get('link')
            if not url or not url.endswith('.onion'):
                sharpened.append(res)
                continue
                
            print(f"  🔍 [TorBot] Analyzing: {url[:30]}...")
            intel = self.get_metadata(url)
            
            if intel["status"] == "up":
                res["status"] = "ACTIVE"
                if intel["title"]:
                    res["title"] = intel["title"] # Update title jika lebih akurat
                if intel["emails"]:
                    res["emails"] = intel["emails"]
                sharpened.append(res)
            else:
                # Jika link mati, kita tandai atau tetap masukkan tapi dengan status down
                res["status"] = "INACTIVE"
                sharpened.append(res)
                
        # Sisanya tetap dimasukkan tanpa dipertajam (hemat waktu)
        if len(search_results) > limit:
            sharpened.extend(search_results[limit:])
            
        return sharpened

if __name__ == "__main__":
    # Test (hanya jalan jika Tor aktif)
    bot = TorBotIntegrator()
    print("TorBot Integrator Ready.")
    # test_url = "http://check.torproject.org"
    # print(bot.get_metadata(test_url))
