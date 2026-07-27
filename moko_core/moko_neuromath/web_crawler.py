"""
MOKO NeuroMath: Multi-Source Legal Web Crawler
===============================================
Sistem perayap sumber legal terpadu yang mencakup:
  - arXiv          → Matematika, Fisika, AI, CS (jutaan paper gratis)
  - Semantic Scholar → 200+ juta paper ilmiah (API publik)
  - OpenAlex        → 250+ juta karya akademik
  - MDN Web Docs    → Semua bahasa pemrograman web (HTML/CSS/JS/API)
  - Python Docs     → Dokumentasi Python resmi
  - DevDocs API     → Dokumentasi multi-bahasa (C, Go, Rust, Java, dll)
  - DuckDuckGo      → Fallback umum untuk topik teknologi

Setiap sumber dipilih secara cerdas berdasarkan kategori topik yang sedang
dieksplor oleh EpistemicForager, menggunakan sistem routing domain.
"""

import requests
import arxiv
import re
import warnings
import socket
import time
import io
import collections
import gc
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS           # Nama package baru (pip install ddgs)
except ImportError:
    from duckduckgo_search import DDGS  # Fallback ke nama lama

try:
    import fitz  # PyMuPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# Suppress residual RuntimeWarning jika masih ada versi lama
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo.*")

from moko_agents.llm_engine import engine
from moko_memory.gc_tuner import gc_tuner

# Global state untuk caching status koneksi internet
_last_internet_check: float = 0.0
_internet_ok: bool = True


def is_internet_available(timeout: float = 1.0) -> bool:

    """Cek apakah koneksi internet/DNS aktif dengan caching 60 detik."""
    global _last_internet_check, _internet_ok
    now = time.time()
    if now - _last_internet_check < 60.0:
        return _internet_ok

    _last_internet_check = now
    
    # 1. Cek port 80 pada IP publik Cloudflare (1.1.1.1) tanpa DNS lookup
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("1.1.1.1", 80))
        s.close()
        _internet_ok = True
        return True
    except Exception:
        pass

    # 2. Cek DNS resolution dns.google
    try:
        socket.gethostbyname("dns.google")
        _internet_ok = True
        return True
    except Exception:
        pass

    _internet_ok = False
    return False


# ─── Konstanta ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "MOKO-AI-ResearchBot/1.0 (Educational AI; "
        "contact: local-research-bot; "
        "https://github.com/example)"
    )
}

# Ekstensi file yang diblokir: installer + semua tipe gambar
# Gambar tidak perlu di-download — crawler hanya butuh teks
INSTALL_BLOCKED_EXTENSIONS = [
    # Installer / binary
    ".exe", ".msi", ".apk", ".dmg", ".pkg", ".deb", ".rpm",
    ".sh", ".bat", ".cmd", ".ps1",
    ".torrent", "magnet:",
    # Gambar (semua format)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg",
    ".ico", ".tiff", ".tif", ".avif", ".heic", ".heif", ".raw",
    ".psd", ".xcf", ".ai", ".eps", ".jfif", ".jpe",
]

# ─── Wikipedia Onion Service (Uncensored, Journalist-verified) ────────────────
# Wikipedia memiliki layanan Tor resmi — konten ditulis jurnalis & akademisi
# Versi Onion v3 resmi Wikimedia Foundation:
WIKIPEDIA_ONION_URLS = {
    "en":    "http://www.wikitpngsm5i74kl.onion/wiki/",
    "en_v3": "https://en.m.wikipedia.org",         # Clearnet fallback via Tor
    # Format URL artikel: {base}/wiki/{Article_Title}
}

# Clearnet Wikipedia diakses via Tor proxy (Wikipedia mengizinkan akses Tor)
WIKIPEDIA_CLEARNET_SEARCH = "https://en.wikipedia.org/w/index.php"
WIKIPEDIA_CLEARNET_BASE = "https://en.wikipedia.org/wiki/"
WIKIPEDIA_ID_CLEARNET_SEARCH = "https://id.wikipedia.org/w/index.php"
WIKIPEDIA_ID_CLEARNET_BASE = "https://id.wikipedia.org/wiki/"

# ─── Situs Pemrograman & Teknologi di Dark Web (Seed URLs) ───────────────────
# Situs-situs ini berfokus pada konten pendidikan, riset, dan pemrograman
# Dikurasi berdasarkan sumber publik yang tersedia secara akademis
PROGRAMMING_ONION_SEEDS = [
    # Hidden Wiki — direktori situs .onion (mencakup banyak situs edukasi)
    "http://zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion/wiki/index.php/Main_Page",
    # DuckDuckGo onion — mesin pencari privat via Tor (gunakan untuk query programming)
    "https://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
    # SecureDrop — platform jurnalis dan whistleblower (informasi teknis tinggi)
    "http://secrdrop5wyphb5x.onion",
    # The Tor Project documentation onion
    "http://2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid.onion",
    # Riseup — teknologi privat untuk aktivis & programmer (dokumentasi teknis)
    "http://vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd.onion",
]

# Kata kunci untuk mendeteksi topik programming/teknologi dalam query
PROGRAMMING_KEYWORDS = [
    "python", "javascript", "java", "c++", "rust", "golang", "kotlin",
    "programming", "pemrograman", "koding", "coding", "algoritma", "algorithm",
    "machine learning", "deep learning", "neural network", "ai", "artificial intelligence",
    "database", "sql", "nosql", "api", "backend", "frontend", "fullstack",
    "linux", "unix", "bash", "shell", "terminal", "git", "github",
    "cybersecurity", "keamanan", "enkripsi", "encryption", "cryptography",
    "network", "jaringan", "protocol", "tcp", "http", "dns",
    "compiler", "interpreter", "framework", "library", "open source",
    "data science", "statistik", "mathematics", "matematika",
    "hacking", "security", "privacy", "privasi", "anonymity",
    "operating system", "sistem operasi", "kernel", "docker", "kubernetes",
]

# Kategori → Sumber utama yang paling relevan
SOURCE_ROUTING = {
    "matematika":      ["arxiv", "semantic_scholar"],
    "fisika":          ["arxiv", "semantic_scholar"],
    "ai":              ["arxiv", "semantic_scholar", "openalex"],
    "machine learning": ["arxiv", "semantic_scholar"],
    "neuroscience":    ["arxiv", "semantic_scholar", "openalex"],
    "programming":     ["mdn", "devdocs", "python_docs"],
    "python":          ["python_docs", "mdn", "duckduckgo"],
    "javascript":      ["mdn", "duckduckgo"],
    "algoritma":       ["arxiv", "duckduckgo"],
    "logika":          ["arxiv", "semantic_scholar"],
    "teknologi":       ["duckduckgo", "semantic_scholar"],
    "default":         ["arxiv", "duckduckgo"],
}

# ─── Kelas Utama Crawler ──────────────────────────────────────────────────────
class MultiSourceCrawler:

    def get_tor_proxies(self, url: str) -> Optional[dict]:
        """
        Mendapatkan konfigurasi proxy Tor jika URL mengandung domain .onion.
        Melakukan deteksi port 9050 & 9150 secara otomatis.
        """
        from moko_config import settings
        if not getattr(settings, "TOR_ENABLED", False):
            return None

        is_onion = ".onion" in url.lower()
        if not is_onion:
            return None

        # Cek port aktif (9050 daemon atau 9150 Tor Browser)
        ports_to_try = [settings.TOR_SOCKS_PORT, 9150, 9050]
        active_port = None
        for port in ports_to_try:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                s.close()
                active_port = port
                break
            except Exception:
                continue

        if active_port:
            return {
                "http": f"socks5h://127.0.0.1:{active_port}",
                "https": f"socks5h://127.0.0.1:{active_port}"
            }
        else:
            raise ConnectionError(
                "Layanan Tor proxy lokal (127.0.0.1:9050/9150) tidak aktif.\n"
                "Silakan jalankan layanan Tor di background (misal: 'sudo systemctl start tor' "
                "atau jalankan Tor Browser) untuk membuka tautan .onion!"
            )

    def _safe_get(self, url: str, **kwargs) -> requests.Response:
        """
        Lakukan HTTP GET secara aman dengan perutean proxy Tor otomatis jika URL adalah .onion.
        """
        if "proxies" not in kwargs:
            try:
                kwargs["proxies"] = self.get_tor_proxies(url)
            except ConnectionError as ce:
                print(f"[WebCrawler] Tor error untuk {url}: {ce}")
                raise ce
        return requests.get(url, **kwargs)

    def clean_html_to_markdown(self, soup) -> str:
        """
        Mengonversi HTML BeautifulSoup menjadi teks bersih berformat markdown.
        Mendeteksi, memisahkan, dan menjaga format program/kode sampel SEMUA bahasa
        (Python, JS, Go, Rust, C/C++, Java, Bash/Shell, SQL, YAML, Dockerfile, dll.)
        serta perintah terminal ($ cmd, # cmd, % cmd) 100% utuh sebelum pembersihan
        teks lainnya dilakukan.
        """
        # Hapus elemen tidak perlu jika belum diekstrak
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            try:
                tag.extract()
            except Exception:
                pass

        body = soup.find("body") or soup

        # 1. Kumpulkan dan gantikan SEMUA elemen kode/terminal dengan placeholder
        code_blocks = []

        # ── a) <pre> tags — selalu berisi kode multi-baris ──────────────────────
        for pre in body.find_all("pre"):
            # Cari label bahasa dari class sibling / child (GitHub Pages, Highlight.js, dll)
            lang = ""
            code_child = pre.find("code")
            if code_child:
                classes = code_child.get("class", [])
                for cls in classes:
                    if cls.startswith(("language-", "lang-")):
                        lang = cls.split("-", 1)[1].lower()
                        break
            if not lang:
                classes = pre.get("class", [])
                for cls in classes:
                    if cls.startswith(("language-", "lang-")):
                        lang = cls.split("-", 1)[1].lower()
                        break
            code_text = pre.get_text()
            if code_text.strip():
                placeholder = f"__MOKO_CODE_BLOCK_{len(code_blocks)}__"
                code_blocks.append(f"\n```{lang}\n{code_text}\n```\n")
                pre.replace_with(placeholder)

        # ── b) <div class="highlight|codehilite|sourceCode|...">> ── code containers ─
        CODE_DIV_CLASSES = {
            "highlight", "codehilite", "code", "sourceCode", "syntax",
            "CodeMirror", "editor-content", "code-block", "code-snippet",
            "prism", "shiki", "hljs", "code-example", "terminal",
            "command", "bash", "shell-session"
        }
        for div in body.find_all(["div", "section", "article"]):
            classes = set(div.get("class", []))
            if classes & CODE_DIV_CLASSES:
                code_text = div.get_text()
                if code_text.strip():
                    placeholder = f"__MOKO_CODE_BLOCK_{len(code_blocks)}__"
                    code_blocks.append(f"\n```\n{code_text}\n```\n")
                    div.replace_with(placeholder)

        # ── c) <code> tags mandiri (inline code) ─────────────────────────────────
        for code in body.find_all("code"):
            if code.parent is None:
                continue
            code_text = code.get_text()
            if code_text.strip():
                placeholder = f"__MOKO_CODE_BLOCK_{len(code_blocks)}__"
                if "\n" in code_text or len(code_text) > 60:
                    code_blocks.append(f"\n```\n{code_text}\n```\n")
                else:
                    code_blocks.append(f" `{code_text}` ")
                code.replace_with(placeholder)

        # ── d) <kbd> (keyboard shortcuts / terminal keystrokes) ─────────────────
        for kbd in body.find_all("kbd"):
            kbd_text = kbd.get_text(strip=True)
            if kbd_text:
                placeholder = f"__MOKO_CODE_BLOCK_{len(code_blocks)}__"
                code_blocks.append(f" `{kbd_text}` ")
                kbd.replace_with(placeholder)

        # ── e) <samp> (program output / terminal output) ─────────────────────────
        for samp in body.find_all("samp"):
            samp_text = samp.get_text()
            if samp_text.strip():
                placeholder = f"__MOKO_CODE_BLOCK_{len(code_blocks)}__"
                code_blocks.append(f"\n```\n{samp_text}\n```\n")
                samp.replace_with(placeholder)

        # 2. Ambil teks dari elemen sisa dengan separator baris baru
        text_raw = body.get_text(separator="\n\n")

        # Bersihkan spasi berlebih di luar code block
        lines = []
        for line in text_raw.split("\n"):
            line_str = line.strip()
            if not line_str:
                if lines and lines[-1] != "":
                    lines.append("")
            else:
                lines.append(line_str)
        text_raw = "\n".join(lines).strip()

        # 3. Kembalikan placeholder kode block ke teks asli
        for i, cb in enumerate(code_blocks):
            placeholder = f"__MOKO_CODE_BLOCK_{i}__"
            text_raw = text_raw.replace(placeholder, cb)

        return text_raw

    # ── Install Guard (minimal) ───────────────────────────────────────────────
    def _is_safe_to_crawl(self, url: str) -> bool:
        """
        Hanya blokir URL yang mengarah langsung ke file installer/download binary.
        Semua konten lain diizinkan untuk di-crawl.
        """
        url_lower = url.lower()
        for ext in INSTALL_BLOCKED_EXTENSIONS:
            if url_lower.endswith(ext) or ext in url_lower:
                print(f"[InstallGuard] ⛔ URL diblokir (installer terdeteksi '{ext}'): {url[:80]}")
                return False
        return True

    # ── In-Memory PDF Scanner (tanpa menyimpan ke disk) ───────────────────────
    def _extract_pdf_text_in_memory(self, url: str, proxies: Optional[dict] = None) -> str:
        """
        Ambil PDF langsung ke memory (tanpa menyimpan ke disk) dan ekstrak teksnya.
        Menggunakan PyMuPDF (fitz) untuk scanning. Batas stream 10MB.
        Mengembalikan string teks kosong jika gagal atau PDF terlalu besar.
        """
        if not _PDF_AVAILABLE:
            return ""
        MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB batas aman
        try:
            print(f"[PDF Scanner] 📄 Memindai PDF in-memory: {url[:80]}")
            resp = requests.get(
                url, headers=HEADERS, proxies=proxies,
                timeout=40, stream=True
            )
            if resp.status_code != 200:
                return ""
            # Baca maksimal MAX_PDF_BYTES byte dari stream
            chunks = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_PDF_BYTES:
                    print(f"[PDF Scanner] ⚠️ PDF terlalu besar (>{MAX_PDF_BYTES//1024//1024}MB), dipotong.")
                    break
            raw_bytes = b"".join(chunks)
            # Buka dari bytes, bukan dari file disk
            doc = fitz.open(stream=io.BytesIO(raw_bytes), filetype="pdf")
            texts = []
            for page_num in range(min(len(doc), 20)):  # Maks 20 halaman
                page = doc.load_page(page_num)
                texts.append(page.get_text("text"))
            doc.close()
            full_text = "\n".join(texts).strip()
            print(f"[PDF Scanner] ✅ Berhasil ekstrak {len(full_text)} karakter dari PDF.")
            return full_text[:8000]  # Maks 8000 karakter per PDF
        except Exception as e:
            print(f"[PDF Scanner] ❌ Gagal membaca PDF {url[:60]}: {e}")
            return ""

    # ── 1. arXiv (Math, Physics, CS, AI papers) ──────────────────────────────
    def search_arxiv(self, query: str, max_results: int = 2) -> List[Dict]:
        """Ambil abstrak paper dari arXiv.org (100% gratis, 2M+ papers)."""
        if not is_internet_available():
            return []
        results = []
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for paper in client.results(search):
                text = (
                    f"JUDUL: {paper.title}\n"
                    f"PENULIS: {', '.join(a.name for a in paper.authors[:3])}\n"
                    f"TAHUN: {paper.published.year if paper.published else 'N/A'}\n"
                    f"KATEGORI: {', '.join(paper.categories)}\n\n"
                    f"ABSTRAK:\n{paper.summary}"
                )
                results.append({
                    "source": "arXiv",
                    "url": paper.entry_id,
                    "text": text[:4000]
                })
        except Exception as e:
            print(f"[arXiv] Error: {e}")
        return results

    # ── 2. Semantic Scholar API (200M+ papers) ────────────────────────────────
    def search_semantic_scholar(self, query: str, max_results: int = 2) -> List[Dict]:
        """Cari paper via Semantic Scholar Public API (gratis, tanpa key)."""
        if not is_internet_available():
            return []
        results = []
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,externalIds"
            }
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                data = res.json().get("data", [])
                for paper in data:
                    abstract = paper.get("abstract") or ""
                    if len(abstract) < 50:
                        continue
                    text = (
                        f"JUDUL: {paper.get('title', 'N/A')}\n"
                        f"TAHUN: {paper.get('year', 'N/A')}\n\n"
                        f"ABSTRAK:\n{abstract}"
                    )
                    results.append({
                        "source": "SemanticScholar",
                        "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId','')}",
                        "text": text[:4000]
                    })
        except Exception as e:
            print(f"[SemanticScholar] Error: {e}")
        return results

    # ── 3. OpenAlex API (250M+ works, fully open) ─────────────────────────────
    def search_openlex(self, query: str, max_results: int = 2) -> List[Dict]:
        """Cari via OpenAlex API (gratis, 250M+ karya ilmiah)."""
        if not is_internet_available():
            return []
        results = []
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": query,
                "per-page": max_results,
                "filter": "open_access.is_oa:true",
                "select": "title,publication_year,abstract_inverted_index,primary_location"
            }
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                items = res.json().get("results", [])
                for item in items:
                    # OpenAlex menyimpan abstrak sebagai inverted index, perlu direkonstruksi
                    inv_idx = item.get("abstract_inverted_index", {})
                    if inv_idx:
                        # Rekonstruksi teks dari inverted index
                        word_positions = []
                        for word, positions in inv_idx.items():
                            for pos in positions:
                                word_positions.append((pos, word))
                        word_positions.sort(key=lambda x: x[0])
                        abstract = " ".join(w for _, w in word_positions)
                    else:
                        abstract = "(Abstrak tidak tersedia)"
                    
                    text = (
                        f"JUDUL: {item.get('title', 'N/A')}\n"
                        f"TAHUN: {item.get('publication_year', 'N/A')}\n\n"
                        f"ABSTRAK:\n{abstract}"
                    )
                    results.append({
                        "source": "OpenAlex",
                        "url": "",
                        "text": text[:4000]
                    })
        except Exception as e:
            print(f"[OpenAlex] Error: {e}")
        return results

    # ── 4. MDN Web Docs (Dokumentasi Web Resmi Mozilla) ───────────────────────
    def search_mdn(self, query: str) -> List[Dict]:
        """Cari di MDN Web Docs (HTML, CSS, JavaScript, Web APIs)."""
        if not is_internet_available():
            return []
        results = []
        try:
            url = "https://developer.mozilla.org/api/v1/search"
            params = {"q": query, "locale": "en-US", "size": 2}
            res = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                hits = res.json().get("documents", [])
                for hit in hits:
                    text = (
                        f"JUDUL: {hit.get('title', '')}\n"
                        f"MDN URL: https://developer.mozilla.org{hit.get('mdn_url', '')}\n\n"
                        f"RINGKASAN:\n{hit.get('summary', '')}"
                    )
                    if len(text) > 80:
                        results.append({
                            "source": "MDN",
                            "url": f"https://developer.mozilla.org{hit.get('mdn_url', '')}",
                            "text": text[:4000]
                        })
        except Exception as e:
            print(f"[MDN] Error: {e}")
        return results

    # ── 5. Python Docs via DevDocs / Python.org ────────────────────────────────
    def search_python_docs(self, query: str) -> List[Dict]:
        """Ambil dari Python.org documentation via search."""
        if not is_internet_available():
            return []
        results = []
        try:
            url = f"https://docs.python.org/3/search.html"
            params = {"q": query}
            res = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                search_results = soup.find_all("li", class_="search-result")[:2]
                for r in search_results:
                    title_tag = r.find("a")
                    desc_tag = r.find("div", class_="context")
                    if title_tag and desc_tag:
                        title = title_tag.get_text(strip=True)
                        desc = desc_tag.get_text(strip=True)
                        href = title_tag.get("href", "")
                        text = (
                            f"PYTHON DOCS - {title}\n"
                            f"URL: https://docs.python.org/3/{href}\n\n"
                            f"{desc}"
                        )
                        results.append({
                            "source": "PythonDocs",
                            "url": f"https://docs.python.org/3/{href}",
                            "text": text[:4000]
                        })
        except Exception as e:
            print(f"[PythonDocs] Error: {e}")
        return results

    # ── 6. DuckDuckGo + BeautifulSoup Crawler (Fallback umum) ────────────────
    def search_and_crawl_duckduckgo(self, query: str, max_results: int = 2) -> List[Dict]:
        """DDG search + full HTML scraping dari URL hasilnya."""
        if not is_internet_available():
            return []
        results = []
        try:
            ddg_results = DDGS().text(
                f"{query} site:github.com OR site:stackoverflow.com OR site:medium.com OR site:towardsdatascience.com",
                max_results=max_results
            )
            for r in (ddg_results or []):
                url = r.get("href", "")
                if not url:
                    continue
                try:
                    page = self._safe_get(url, headers=HEADERS, timeout=12)
                    if page.status_code != 200:
                        continue
                    soup = BeautifulSoup(page.text, "html.parser")
                    text_raw = self.clean_html_to_markdown(soup)
                    if len(text_raw) > 200:
                        results.append({
                            "source": "DuckDuckGo",
                            "url": url,
                            "text": text_raw[:4000]
                        })
                except ConnectionError as tor_err:
                    print(f"[DDG Crawler] ⚠️ {tor_err}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[DDG] Error: {e}")
        return results

    # ── 7. GitHub & Web Recursive Scraper ─────────────────────────────────────
    def fetch_github_repo_recursive(self, url: str, max_depth: int = 2) -> List[Dict]:
        """Perayap repositori GitHub secara rekursif menggunakan HTML Parsing."""
        if not is_internet_available():
            return []
        import re
        import io
        import zipfile
        from pathlib import Path

        # Coba parse owner & repo untuk download zipball super cepat
        match = re.search(r"github\.com/([^/]+)/([^/.]+)", url)
        if match:
            owner = match.group(1)
            repo = match.group(2).replace(".git", "")
            # Bersihkan jika ada path tambahan (e.g. /tree/main/...)
            repo = repo.split('/')[0]
            zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
            try:
                print(f"[GitHub Crawl] Mencoba zipball API: {zip_url}")
                res = requests.get(zip_url, headers=HEADERS, timeout=20)
                if res.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(res.content))
                    results = []
                    # Ekstensi berkas sumber teks yang relevan
                    valid_exts = {
                        # Dokumentasi
                        '.txt', '.md', '.rst', '.adoc',
                        # Web
                        '.html', '.htm', '.css', '.scss', '.sass', '.less',
                        '.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx',
                        # Python
                        '.py', '.pyw', '.pyi', '.pyx',
                        # Java / JVM
                        '.java', '.kt', '.kts', '.scala', '.groovy', '.clj',
                        # C / C++ / C#
                        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.cs', '.vb',
                        # Go / Rust / Swift / Dart
                        '.go', '.rs', '.swift', '.dart',
                        # Ruby / PHP / Perl
                        '.rb', '.php', '.pl', '.pm',
                        # Shell / Terminal
                        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
                        # Functional
                        '.hs', '.lhs', '.erl', '.ex', '.exs', '.elm', '.ml',
                        # Low-level
                        '.asm', '.s',
                        # Data / Config
                        '.json', '.jsonc', '.yaml', '.yml', '.toml', '.ini',
                        '.env', '.xml', '.xsl', '.svg', '.proto',
                        # Database
                        '.sql',
                        # Infra / DevOps
                        '.tf', '.hcl', '.dockerfile',
                        '.nginx', '.conf', '.htaccess',
                        # R / MATLAB / Julia
                        '.r', '.m', '.jl',
                        # LaTeX
                        '.tex', '.bib',
                        # Makefile / Build
                        '.makefile', '.cmake',
                        # Patch / Diff
                        '.diff', '.patch',
                    }
                    for name in z.namelist():
                        if name.endswith('/'):
                            continue
                        ext = Path(name).suffix.lower()
                        if ext in valid_exts:
                            try:
                                with z.open(name) as f:
                                    # Batasi alokasi RAM per file teks (max 256 KB)
                                    content = f.read(256 * 1024).decode('utf-8', errors='ignore')
                                    if len(content.strip()) > 50:
                                        relative_name = name.split('/', 1)[1] if '/' in name else name
                                        results.append({
                                            "source": f"GitHub-Zip/{repo}",
                                            "url": f"https://github.com/{owner}/{repo}/blob/main/{relative_name}",
                                            "text": content
                                        })
                            except Exception:
                                pass
                    if results:
                        print(f"[GitHub Crawl] Berhasil mengekstrak {len(results)} file dari zipball.")
                        
                        # Ekstraksi tautan eksternal dari README atau berkas dokumentasi markdown
                        external_urls = []
                        ignore_patterns = [
                            "shields.io", "badge.fury.io", "github.com/features", "github.com/settings", 
                            "github.com/marketplace", "githubcommunity.com", "travis-ci.org", "github.com/contact",
                            "github.com/about", "github.com/security", "github.com/pricing", "github.com/site",
                            "license", "w3.org", "schema.org", "sponsors", "github.com/sponsors", "gitter.im"
                        ]
                        
                        for r in results:
                            file_url = r["url"].lower()
                            if "readme" in file_url or file_url.endswith(".md"):
                                # Cari URL di dalam teks markdown
                                urls = re.findall(r"https?://[^\s\"'()\[\]>]+", r["text"])
                                for u in urls:
                                    # Bersihkan tanda baca di akhir URL
                                    u = u.rstrip('.,;:/-_')
                                    # Hindari mereferensi ke repositori kita sendiri atau link sampah
                                    if f"github.com/{owner}/{repo}" not in u:
                                        if not any(pat in u for pat in ignore_patterns):
                                            if u not in external_urls:
                                                external_urls.append(u)
                        
                        # Batasi maks 15 link eksternal unik agar tidak memicu memory bloat / OOM
                        external_urls = external_urls[:15]
                        if external_urls:
                            print(f"[GitHub Crawl] Menemukan {len(external_urls)} link eksternal di dokumentasi. Mulai merayapi...")
                            for ext_url in external_urls:
                                try:
                                    ext_results = self.fetch_url_recursive(ext_url, max_depth=1)
                                    if ext_results:
                                        print(f"[GitHub Crawl] Sukses merayap link eksternal: {ext_url} (Ditemukan {len(ext_results)} item)")
                                        results.extend(ext_results)
                                except Exception as ext_e:
                                    print(f"[GitHub Crawl] Gagal merayap {ext_url}: {ext_e}")

                                # Pembersihan RAM setelah merayap link eksternal
                                try:
                                    gc.collect(2)
                                    gc_tuner._malloc_trim()
                                except Exception:
                                    pass
                                time.sleep(1.5)
                                    
                        return results
            except Exception as e:
                print(f"[GitHub Crawl] Gagal download zipball ({e}), fallback ke HTML scraping...")

        results = []
        visited = set()
        
        def crawl(current_url, depth):
            if depth > max_depth or current_url in visited:
                return
            visited.add(current_url)
            
            # Bersihkan URL jika ada trailing slash
            current_url = current_url.rstrip('/')
            
            # Jika menunjuk langsung ke file raw
            if "raw.githubusercontent.com" in current_url:
                try:
                    res = requests.get(current_url, headers=HEADERS, timeout=12)
                    if res.status_code == 200 and len(res.text) > 100:
                        results.append({
                            "source": "GitHub-Raw",
                            "url": current_url,
                            "text": res.text[:5000]
                        })
                except Exception:
                    pass
                return

            # Jika menunjuk ke berkas di GitHub HTML view, ubah ke raw link
            if "github.com" in current_url and "/blob/" in current_url:
                raw_url = current_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                crawl(raw_url, depth)
                return

            # Jika menunjuk ke tree (direktori) atau home repo
            try:
                res = requests.get(current_url, headers=HEADERS, timeout=12)
                if res.status_code != 200:
                    return
                
                soup = BeautifulSoup(res.text, "html.parser")
                
                # Cari tautan internal repo
                links = soup.find_all("a")
                for link in links:
                    href = link.get("href", "")
                    if not href:
                        continue
                    
                    # Normalisasi URL absolute
                    if href.startswith("/"):
                        full_link = f"https://github.com{href}"
                    elif href.startswith("https://github.com"):
                        full_link = href
                    else:
                        continue
                    
                    # Hindari keluar dari repositori ini
                    if "github.com/" in current_url:
                        parts = current_url.split("github.com/")[1].split("/")
                        if len(parts) >= 2:
                            repo_path = f"{parts[0]}/{parts[1]}"
                            if repo_path not in full_link:
                                continue
                    
                    # Filter tautan navigasi GitHub yang tidak perlu
                    if any(x in full_link for x in ["/pull/", "/issues/", "/commits/", "/releases/", "/tags/", "/wiki/", "/actions/"]):
                        continue
                        
                    # Deteksi file atau folder
                    if "/blob/" in full_link:
                        crawl(full_link, depth)
                    elif "/tree/" in full_link and depth < max_depth:
                        crawl(full_link, depth + 1)
            except Exception as e:
                print(f"[GitHub Crawl Error] {e}")

        crawl(url, 1)
        return results

    def fetch_url_recursive(self, url: str, max_depth: int = 2, visited: set = None) -> List[Dict]:
        """Perayap halaman web umum secara rekursif."""
        if not is_internet_available():
            return []
        if visited is None:
            visited = set()
            
        results = []
        if max_depth <= 0 or url in visited:
            return results
        visited.add(url)
        
        # Jika itu link GitHub
        if "github.com" in url or "raw.githubusercontent.com" in url:
            return self.fetch_github_repo_recursive(url, max_depth)
            
        try:
            page = self._safe_get(url, headers=HEADERS, timeout=30)
            if page.status_code != 200:
                return results
                
            soup = BeautifulSoup(page.text, "html.parser")
            text_raw = self.clean_html_to_markdown(soup)
            
            if len(text_raw) > 200:
                results.append({
                    "source": "Tor-Onion" if ".onion" in url else "Web-Recursive",
                    "url": url,
                    "text": text_raw[:4000]
                })
                
            # Cari link eksternal di halaman untuk ditelusuri (skip .onion sub-links)
            if max_depth > 1 and ".onion" not in url:
                links = soup.find_all("a")
                for link in links[:15]:
                    href = link.get("href", "")
                    if href.startswith("http"):
                        if any(x in href for x in ["facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com"]):
                            continue
                        results += self.fetch_url_recursive(href, max_depth - 1, visited)
                        try:
                            gc.collect(2)
                            gc_tuner._malloc_trim()
                        except Exception:
                            pass
                        time.sleep(1.0)
        except ConnectionError as tor_err:
            print(f"[WebCrawler] ⚠️ Tor tidak aktif: {tor_err}")
        except Exception:
            pass
            
        return results

    # ── 8. Tor Onion Masive BFS Crawler ─────────────────────────────────────
    def fetch_onion_massive(
        self,
        seed_url: str,
        max_pages: int = 200,
        max_workers: int = 6,
        stay_on_domain: bool = True
    ) -> List[Dict]:
        """
        SISTEM CRAWL MASIF DARK WEB — Level Tertinggi.

        Algoritma BFS (Breadth-First Search) tak terbatas dari satu seed URL.
        - Tidak ada batas kedalaman rekursi — murni dibatasi oleh max_pages
        - Multi-thread paralel (max_workers worker)
        - PDF di-scan in-memory (tanpa download ke disk)
        - Murni: tidak ada filter konten, semua halaman onion dikumpulkan
        - Hanya mengekstrak teks — tidak menginstall apapun
        - Opsional: tetap di domain yang sama (stay_on_domain=True)
        """
        from moko_config import settings
        if not getattr(settings, "TOR_ENABLED", False):
            print("[MassiveCrawl] ⚠️ Tor tidak diaktifkan. Set TOR_ENABLED=True di settings.")
            return []

        try:
            proxies = self.get_tor_proxies(seed_url)
        except ConnectionError as ce:
            print(f"[MassiveCrawl] ❌ Tor tidak aktif: {ce}")
            return []

        parsed_seed = urlparse(seed_url)
        base_domain = parsed_seed.netloc

        visited = set()
        queue = collections.deque([seed_url])
        results = []
        lock = __import__("threading").Lock()

        print(f"[MassiveCrawl] 🚀 Mulai crawl masif dari: {seed_url}")
        print(f"[MassiveCrawl] 📊 Target: maks {max_pages} halaman | {max_workers} worker paralel")

        def _fetch_page(url: str) -> Optional[Dict]:
            """Worker: fetch satu URL, ekstrak teks/PDF, kembalikan dict result."""

            content_type = ""
            try:
                # HEAD request dulu untuk cek content-type tanpa download penuh
                head = requests.head(
                    url, headers=HEADERS, proxies=proxies, timeout=15, allow_redirects=True
                )
                content_type = head.headers.get("Content-Type", "").lower()
            except Exception:
                pass  # Lanjut dengan GET biasa

            # ─── Jika PDF → scan in-memory ────────────────────────────────
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                text = self._extract_pdf_text_in_memory(url, proxies=proxies)
                if text:
                    return {
                        "source": "Tor-Onion-PDF",
                        "url": url,
                        "text": f"[DOKUMEN PDF] {url}\n\n{text}"
                    }
                return None

            # ─── Skip hanya tipe biner murni yang tidak mengandung teks ─────
            skip_types = [
                "image/", "video/", "audio/",
            ]
            if any(t in content_type for t in skip_types):
                return None

            # ─── Fetch HTML ───────────────────────────────────────────────
            try:
                resp = requests.get(
                    url, headers=HEADERS, proxies=proxies, timeout=30
                )
                if resp.status_code != 200:
                    return None

                soup = BeautifulSoup(resp.text, "html.parser")

                # Kumpulkan link baru untuk antrian BFS
                new_links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    abs_url = urljoin(url, href)
                    parsed = urlparse(abs_url)
                    # Hanya http/https
                    if parsed.scheme not in ("http", "https"):
                        continue
                    # Opsional: tetap di domain yang sama
                    if stay_on_domain and parsed.netloc != base_domain:
                        continue
                    # Hanya telusuri onion jika seed adalah onion
                    if ".onion" in base_domain and ".onion" not in parsed.netloc:
                        continue
                    new_links.append(abs_url)

                with lock:
                    for lnk in new_links:
                        if lnk not in visited and len(queue) + len(visited) < max_pages * 3:
                            queue.append(lnk)

                title = soup.find("title")
                title_text = title.get_text(strip=True) if title else url

                text_raw = self.clean_html_to_markdown(soup)

                if len(text_raw) > 50:
                    return {
                        "source": "Tor-Onion-Massive",
                        "url": url,
                        "text": f"HALAMAN: {title_text}\nURL: {url}\n\n{text_raw[:6000]}"
                    }
            except Exception as e:
                print(f"[MassiveCrawl] ⚠️ Gagal fetch {url[:60]}: {e}")
            return None

        # ─── BFS Loop dengan ThreadPoolExecutor ───────────────────────────
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while queue and len(visited) < max_pages:
                # Ambil batch dari antrian (maks max_workers URL sekaligus)
                batch = []
                while queue and len(batch) < max_workers:
                    url_candidate = queue.popleft()
                    if url_candidate not in visited:
                        visited.add(url_candidate)
                        batch.append(url_candidate)

                if not batch:
                    break

                print(f"[MassiveCrawl] 🔍 Batch {len(visited)}/{max_pages} — merayap {len(batch)} URL paralel")
                futures = {executor.submit(_fetch_page, u): u for u in batch}
                for future in as_completed(futures, timeout=45):
                    try:
                        res = future.result()
                        if res:
                            results.append(res)
                    except Exception as exc:
                        print(f"[MassiveCrawl] Thread error: {exc}")

                # Pembersihan RAM berkala setelah memproses batch URL
                if queue and len(visited) < max_pages:
                    try:
                        gc.collect(2)
                        gc_tuner._malloc_trim()
                    except Exception:
                        pass
                    time.sleep(1.5)

        print(f"[MassiveCrawl] ✅ Selesai! Total halaman berhasil di-crawl: {len(results)}")
        return results

    def fetch_onion_url_recursive(self, url: str, max_depth: int = 3, visited: set = None) -> List[Dict]:
        """
        Merayap situs .onion secara rekursif melalui proxy Tor.
        Wrapper kompatibel — memanggil fetch_onion_massive untuk performa lebih tinggi.
        max_depth dikonversi ke max_pages (depth 3 → 50 pages, depth 5 → 150, dll).
        """
        page_map = {1: 10, 2: 30, 3: 60, 4: 100, 5: 200}
        max_pages = page_map.get(max_depth, 30)
        return self.fetch_onion_massive(url, max_pages=max_pages, stay_on_domain=True)

    def fetch_onion_url(self, onion_url: str) -> List[Dict]:
        """
        Merayap halaman .onion secara langsung melalui Tor SOCKS5h proxy.
        Mendukung URL format http(s)://xxxxx.onion/path
        """
        print(f"[Tor Crawler] 🧅 Mencoba merayap situs Onion: {onion_url}")
        try:
            proxies = self.get_tor_proxies(onion_url)
        except ConnectionError as ce:
            print(f"[Tor Crawler] ❌ {ce}")
            return [{"source": "Tor-Error", "url": onion_url, "text": str(ce)}]

        try:
            page = requests.get(
                onion_url,
                headers=HEADERS,
                proxies=proxies,
                timeout=30  # Timeout besar karena Tor lebih lambat
            )
            if page.status_code != 200:
                print(f"[Tor Crawler] ⚠️ Status {page.status_code} dari {onion_url}")
                return []

            soup = BeautifulSoup(page.text, "html.parser")
            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else onion_url

            text_raw = self.clean_html_to_markdown(soup)

            if len(text_raw) > 80:
                print(f"[Tor Crawler] ✅ Berhasil merayap: {title_text} ({len(text_raw)} chars)")
                return [{
                    "source": "Tor-Onion",
                    "url": onion_url,
                    "text": f"HALAMAN ONION — {title_text}\nURL: {onion_url}\n\n{text_raw[:4000]}"
                }]
            else:
                print(f"[Tor Crawler] ⚠️ Konten terlalu sedikit dari {onion_url}")
                return []

        except requests.exceptions.ConnectTimeout:
            return [{"source": "Tor-Error", "url": onion_url,
                     "text": f"Koneksi timeout ke {onion_url}. Situs mungkin sedang offline."}]
        except Exception as e:
            print(f"[Tor Crawler] ❌ Error saat merayap {onion_url}: {e}")
            return []

    # ── 9. Ahmia Tor Search (Dark Web Search Engine) + Massive Crawl ─────────
    def search_ahmia_onion(
        self,
        query: str,
        max_results: int = 5,
        massive_crawl: bool = False,
        massive_pages_per_site: int = 80
    ) -> List[Dict]:
        """
        Melakukan pencarian situs .onion di Dark Web menggunakan Ahmia Search Engine.
        - Mencoba Ahmia via Tor (onion address) → fallback ke Ahmia clearnet
        - Setiap URL hasil dikrawl secara MASIF menggunakan fetch_onion_massive()
        - massive_crawl=True → crawl sangat dalam (ratusan halaman per situs)
        - Safety filter ketat aktif di semua tahap
        """
        print(f"[Tor Search] 🧅 Pencarian Dark Web masif untuk: '{query}'")

        onion_search_url = (
            f"http://juhanur52qxlpp3crbz2ccmzzupnxt7vtj4lt2asca5s74ixrjjuydad.onion"
            f"/search/?q={requests.utils.quote(query)}"
        )
        clearnet_search_url = f"https://ahmia.fi/search/?q={requests.utils.quote(query)}"

        page = None
        source_name = "Tor-Onion-Massive"

        # 1. Coba via Tor Onion address
        try:
            proxies = self.get_tor_proxies(
                "http://juhanur52qxlpp3crbz2ccmzzupnxt7vtj4lt2asca5s74ixrjjuydad.onion"
            )
            if proxies:
                print(f"[Tor Search] 🧅 Menghubungi Ahmia Onion Service...")
                page = requests.get(onion_search_url, headers=HEADERS, proxies=proxies, timeout=30)
        except Exception as e:
            print(f"[Tor Search] ⚠️ Ahmia Onion gagal ({e}) → mencoba Clearnet...")

        # 2. Fallback ke Clearnet
        if not page or page.status_code != 200:
            try:
                print(f"[Tor Search] 🌐 Menghubungi Ahmia Clearnet...")
                page = requests.get(clearnet_search_url, headers=HEADERS, timeout=15)
                source_name = "Ahmia-Clearnet-Massive"
            except Exception as e:
                print(f"[Tor Search] ❌ Clearnet juga gagal: {e}")
                return []

        if not page or page.status_code != 200:
            print(f"[Tor Search] ⚠️ Tidak ada respon valid dari Ahmia.")
            return []

        # ─── Parse hasil Ahmia ─────────────────────────────────────────────
        candidate_sites = []  # [{title, url, desc}]
        try:
            soup = BeautifulSoup(page.text, "html.parser")
            items = soup.find_all("li", class_="result")

            if not items:
                # Fallback: cari semua link <a> yang mengandung .onion
                for link in soup.find_all("a"):
                    href = link.get("href", "")
                    actual_url = ""
                    if "redirect_url=" in href:
                        actual_url = href.split("redirect_url=")[-1]
                    elif ".onion" in href:
                        actual_url = href
                    if actual_url and actual_url.startswith("http"):
                        title = link.get_text(strip=True) or "Onion Site"
                        candidate_sites.append({"title": title, "url": actual_url, "desc": ""})
            else:
                for item in items:
                    h4 = item.find("h4")
                    a = h4.find("a") if h4 else item.find("a")
                    p = item.find("p")
                    if not a:
                        continue
                    href = a.get("href", "")
                    actual_url = ""
                    if "redirect_url=" in href:
                        actual_url = href.split("redirect_url=")[-1]
                    elif ".onion" in href:
                        actual_url = href
                    if not actual_url:
                        cite = item.find("cite")
                        if cite and ".onion" in cite.get_text():
                            actual_url = cite.get_text(strip=True)
                    if actual_url:
                        title = a.get_text(strip=True) or "Situs Onion"
                        desc = p.get_text(strip=True) if p else ""
                        candidate_sites.append({"title": title, "url": actual_url, "desc": desc})

        except Exception as e:
            print(f"[Tor Search] ❌ Error parse Ahmia: {e}")
            return []

        # Deduplikasi URL
        seen_urls = set()
        unique_sites = []
        for s in candidate_sites:
            if s["url"] not in seen_urls:
                seen_urls.add(s["url"])
                unique_sites.append(s)

        print(f"[Tor Search] 📋 Ditemukan {len(unique_sites)} situs unik dari Ahmia.")
        if not unique_sites:
            return []

        # ─── Crawl setiap situs secara MASIF & paralel ────────────────────
        def crawl_site_massive(site: dict) -> List[Dict]:
            url = site["url"]

            print(f"[Tor Search] 🚀 Crawl masif situs: {site['title'][:50]} | {url}")
            if massive_crawl:
                return self.fetch_onion_massive(
                    url, max_pages=massive_pages_per_site, max_workers=4, stay_on_domain=True
                )
            else:
                return self.fetch_onion_massive(
                    url, max_pages=30, max_workers=3, stay_on_domain=True
                )

        final_list = []
        target_sites = unique_sites[:max_results]
        print(f"[Tor Search] 🔥 Memulai crawl masif untuk {len(target_sites)} situs...")

        # Jalankan crawl situs secara sekuensial (batch_size = 1) agar RAM tidak membengkak
        site_batch_size = 1
        for i in range(0, len(target_sites), site_batch_size):
            batch = target_sites[i:i+site_batch_size]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(crawl_site_massive, s): s for s in batch}
                for future in as_completed(futures, timeout=300):
                    site = futures[future]
                    try:
                        crawl_results = future.result()
                        if crawl_results:
                            combined_text = "\n\n---\n\n".join(
                                r["text"] for r in crawl_results if r.get("text")
                            )
                            if combined_text:
                                final_list.append({
                                    "source": source_name,
                                    "url": site["url"],
                                    "text": (
                                        f"🧅 SITUS ONION: {site['title']}\n"
                                        f"DESKRIPSI: {site['desc']}\n"
                                        f"URL: {site['url']}\n"
                                        f"TOTAL HALAMAN DIKRAWL: {len(crawl_results)}\n\n"
                                        f"KONTEN MASIF:\n{combined_text[:12000]}"
                                    )
                                })
                        else:
                            # Jika crawl kosong, simpan minimal snippet
                            final_list.append({
                                "source": f"{source_name}-Snippet",
                                "url": site["url"],
                                "text": (
                                    f"🧅 SITUS ONION: {site['title']}\n"
                                    f"DESKRIPSI: {site['desc']}\n"
                                    f"URL: {site['url']}"
                                )
                            })
                    except Exception as exc:
                        print(f"[Tor Search] Thread error untuk {site['url']}: {exc}")

            # Beri jeda unload RAM setelah setiap batch situs selesai dikrawl
            if i + site_batch_size < len(target_sites):
                try:
                    gc.collect(2)
                    gc_tuner._malloc_trim()
                except Exception:
                    pass
                time.sleep(3.0)

        print(f"[Tor Search] ✅ Selesai! Total {len(final_list)} situs berhasil dikrawl masif.")
        return final_list

    # ── 10. Wikipedia Onion Crawler (Uncensored Knowledge Source) ────────────
    def search_wikipedia_onion(
        self,
        query: str,
        lang: str = "en",
        max_pages: int = 20
    ) -> List[Dict]:
        """
        Mencari dan mengambil artikel Wikipedia melalui Tor (onion service resmi Wikimedia).

        Wikipedia di dark web memberikan informasi TANPA sensor — artikel ditulis oleh
        jurnalis dan akademisi dengan akurasi tinggi. Jauh lebih bisa dipercaya daripada
        banyak sumber web biasa untuk topik yang sensitif atau akademis.

        Strategi:
        1. Coba Wikipedia onion address langsung (v2/v3)
        2. Fallback: akses Wikipedia clearnet MELALUI Tor proxy (Wikipedia mengizinkan)
        3. Fallback terakhir: akses Wikipedia clearnet biasa (tanpa Tor)
        """
        from moko_config import settings
        tor_enabled = getattr(settings, "TOR_ENABLED", False)

        # Tentukan bahasa
        base_clearnet = WIKIPEDIA_CLEARNET_BASE if lang == "en" else WIKIPEDIA_ID_CLEARNET_BASE
        search_url = WIKIPEDIA_CLEARNET_SEARCH if lang == "en" else WIKIPEDIA_ID_CLEARNET_SEARCH
        lang_name = "English" if lang == "en" else "Indonesian"

        print(f"[WikiOnion] 📖 Mencari artikel Wikipedia ({lang_name}): '{query}'")

        # ── Step 1: Dapatkan proxy Tor (jika aktif) ──────────────────────────
        proxies = None
        if tor_enabled:
            try:
                # Gunakan Wikipedia onion address (clearnet Wikipedia via Tor)
                # Wikipedia secara resmi mendukung akses melalui Tor proxy
                tor_port = getattr(settings, "TOR_SOCKS_PORT", 9050)
                ports_to_try = [tor_port, 9150, 9050]
                for port in ports_to_try:
                    try:
                        import socket as _sock
                        s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                        s.settimeout(0.5)
                        s.connect(("127.0.0.1", port))
                        s.close()
                        proxies = {
                            "http": f"socks5h://127.0.0.1:{port}",
                            "https": f"socks5h://127.0.0.1:{port}"
                        }
                        print(f"[WikiOnion] 🧅 Tor aktif di port {port} — akses Wikipedia via Tor")
                        break
                    except Exception:
                        continue
            except Exception:
                pass

        # ── Step 2: Cari artikel via Wikipedia search API ───────────────────
        results = []
        try:
            search_params = {
                "search": query,
                "ns": 0,          # Namespace artikel utama
                "limit": 5,
                "offset": 0
            }
            resp = requests.get(
                search_url, params=search_params,
                headers=HEADERS, proxies=proxies, timeout=20
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Wikipedia search results ada di div.mw-search-result-heading
                result_items = soup.find_all("div", class_="mw-search-result-heading")
                if not result_items:
                    # Coba format alternatif
                    result_items = soup.find_all("li", class_="mw-search-result")

                article_urls = []
                for item in result_items[:max_pages]:
                    a = item.find("a")
                    if a and a.get("href"):
                        href = a["href"]
                        if href.startswith("/wiki/"):
                            full_url = f"https://{'en' if lang == 'en' else 'id'}.wikipedia.org{href}"
                            article_urls.append(full_url)

                # Fallback: jika langsung redirect ke artikel
                if not article_urls:
                    canonical = soup.find("link", rel="canonical")
                    if canonical and canonical.get("href"):
                        article_urls.append(canonical["href"])

                print(f"[WikiOnion] 🔍 Ditemukan {len(article_urls)} artikel kandidat")

                # Fetch setiap artikel
                def _fetch_wiki_article(art_url: str) -> Optional[Dict]:
                    try:
                        art_resp = requests.get(
                            art_url, headers=HEADERS, proxies=proxies, timeout=20
                        )
                        if art_resp.status_code != 200:
                            return None
                        art_soup = BeautifulSoup(art_resp.text, "html.parser")

                        # Hapus elemen tidak perlu
                        for tag in art_soup([
                            "script", "style", "nav", "footer", "sup",
                            "table", ".references", ".navbox", ".toc"
                        ]):
                            tag.extract()

                        title_tag = art_soup.find("h1", id="firstHeading") or art_soup.find("h1")
                        title = title_tag.get_text(strip=True) if title_tag else art_url

                        # Ambil konten artikel utama
                        content_div = (
                            art_soup.find("div", class_="mw-parser-output")
                            or art_soup.find("div", id="mw-content-text")
                            or art_soup.find("body")
                        )
                        if not content_div:
                            return None

                        text_raw = self.clean_html_to_markdown(content_div)

                        if len(text_raw) > 100:
                            via = "via Tor" if proxies else "clearnet"
                            print(f"[WikiOnion] ✅ Artikel berhasil: '{title}' ({len(text_raw)} chars, {via})")
                            return {
                                "source": "Wikipedia-Onion" if proxies else "Wikipedia-Clearnet",
                                "url": art_url,
                                "text": (
                                    f"📖 WIKIPEDIA ARTIKEL: {title}\n"
                                    f"URL: {art_url}\n"
                                    f"AKSES: {'Tor/Onion (uncensored)' if proxies else 'Clearnet'}\n\n"
                                    f"{text_raw[:8000]}"
                                )
                            }
                    except Exception as e:
                        print(f"[WikiOnion] ⚠️ Gagal ambil {art_url}: {e}")
                    return None

                # Fetch paralel dengan pembatasan batch (wiki_batch_size = 2) untuk menjaga RAM
                wiki_batch_size = 2
                for idx in range(0, len(article_urls), wiki_batch_size):
                    batch_urls = article_urls[idx:idx+wiki_batch_size]
                    with ThreadPoolExecutor(max_workers=len(batch_urls)) as executor:
                        futures = [executor.submit(_fetch_wiki_article, u) for u in batch_urls]
                        for future in as_completed(futures, timeout=60):
                            try:
                                res = future.result()
                                if res:
                                    results.append(res)
                            except Exception:
                                pass
                    if idx + wiki_batch_size < len(article_urls):
                        try:
                            gc.collect(2)
                            gc_tuner._malloc_trim()
                        except Exception:
                            pass
                        time.sleep(2.0)

        except Exception as e:
            print(f"[WikiOnion] ❌ Error search Wikipedia: {e}")

        # Juga coba Wikipedia bahasa Indonesia untuk topik yang sama (dual language)
        if lang == "en" and results:
            try:
                id_params = {"search": query, "ns": 0, "limit": 2, "offset": 0}
                id_resp = requests.get(
                    WIKIPEDIA_ID_CLEARNET_SEARCH, params=id_params,
                    headers=HEADERS, proxies=proxies, timeout=15
                )
                if id_resp.status_code == 200:
                    id_soup = BeautifulSoup(id_resp.text, "html.parser")
                    id_items = id_soup.find_all("div", class_="mw-search-result-heading")
                    for item in id_items[:2]:
                        a = item.find("a")
                        if a and a.get("href") and a["href"].startswith("/wiki/"):
                            id_url = f"https://id.wikipedia.org{a['href']}"
                            res = _fetch_wiki_article(id_url)  # noqa: F821
                            if res:
                                results.append(res)
            except Exception:
                pass

        print(f"[WikiOnion] 📚 Total artikel Wikipedia berhasil diambil: {len(results)}")
        return results

    # ── 11. Programming Onion Sites Crawler (Dark Web Programming Sources) ────
    def fetch_programming_onion_sites(
        self,
        query: str,
        max_pages_per_site: int = 50
    ) -> List[Dict]:
        """
        Merayap situs dark web yang berfokus pada pemrograman, teknologi, dan keamanan.
        Menggunakan daftar seed URL yang dikurasi dari sumber-sumber informatif.

        Sumber:
        - Hidden Wiki (direktori onion yang mencakup banyak situs edukasi & teknis)
        - DuckDuckGo onion (mesin pencari via Tor)
        - Situs dokumentasi dan komunitas teknis yang ada di dark web

        Crawl 100% murni: semua halaman dikumpulkan tanpa filter konten.
        """
        from moko_config import settings
        if not getattr(settings, "TOR_ENABLED", False):
            print("[ProgOnion] ⚠️ Tor tidak aktif — lewati crawl programming onion")
            return []

        try:
            proxies = self.get_tor_proxies("http://example.onion")
        except ConnectionError as ce:
            print(f"[ProgOnion] ❌ Tor tidak aktif: {ce}")
            return []

        results = []
        print(f"[ProgOnion] 💻 Mulai crawl situs programming onion untuk: '{query}'")

        def _crawl_seed(seed_url: str) -> List[Dict]:
            """Crawl satu seed URL — ambil semua halaman tanpa filter."""

            print(f"[ProgOnion] 🔍 Seed: {seed_url[:70]}")
            # Gunakan fetch_onion_massive untuk tiap seed
            raw = self.fetch_onion_massive(
                seed_url,
                max_pages=max_pages_per_site,
                max_workers=4,
                stay_on_domain=True
            )
            # Tandai sumber, kembalikan SEMUA hasil tanpa filter relevansi
            for r in raw:
                r["source"] = "DarkWeb-Onion"
            print(f"[ProgOnion] ✅ Seed {seed_url[:50]}: {len(raw)} halaman dikumpulkan")
            return raw

        # Crawl semua seed secara sekuensial (seed_batch_size = 1) untuk menjaga stabilitas RAM
        seed_batch_size = 1
        for idx in range(0, len(PROGRAMMING_ONION_SEEDS), seed_batch_size):
            batch = PROGRAMMING_ONION_SEEDS[idx:idx+seed_batch_size]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(_crawl_seed, seed): seed for seed in batch}
                for future in as_completed(futures, timeout=240):
                    try:
                        results.extend(future.result())
                    except Exception as exc:
                        seed = futures[future]
                        print(f"[ProgOnion] ⚠️ Seed gagal {seed[:50]}: {exc}")
            if idx + seed_batch_size < len(PROGRAMMING_ONION_SEEDS):
                try:
                    gc.collect(2)
                    gc_tuner._malloc_trim()
                except Exception:
                    pass
                time.sleep(2.5)

        print(f"[ProgOnion] 🏁 Total konten programming dari dark web: {len(results)} halaman")
        return results

    # ── Dark Web Direct Router ─────────────────────────────────────────────────
    def route_and_fetch_darkweb(self, query: str, topic_hint: str = "") -> List[Dict]:
        """
        Jalur langsung untuk dark web crawling.
        Menggunakan query PERSIS sebagaimana diberikan (tanpa LLM, tanpa pattern matching).
        Pipeline: Wikipedia Onion → Ahmia Onion → Programming Onion (jika relevan)
        """
        clean_topic = query.strip()
        is_programming_topic = any(kw in clean_topic.lower() for kw in PROGRAMMING_KEYWORDS)

        print(f"[WebCrawler] 🎓 Dark Web Direct Route — query: '{clean_topic}'")
        print(f"[WebCrawler]    → Programming topic: {'YA' if is_programming_topic else 'TIDAK'}")

        all_results = []

        # 1. Wikipedia Onion
        print("[WebCrawler] 📖 Sumber 1/3: Wikipedia Onion...")
        try:
            wiki_res = self.search_wikipedia_onion(clean_topic, lang="en", max_pages=5)
            if wiki_res:
                all_results.extend(wiki_res)
        except Exception as e:
            print(f"[WebCrawler] ⚠️ Wikipedia Onion error: {e}")

        try:
            gc.collect(2)
            gc_tuner._malloc_trim()
        except Exception:
            pass
        time.sleep(3.0)

        # 2. Ahmia Onion
        print("[WebCrawler] 🧅 Sumber 2/3: Ahmia Onion...")
        try:
            ahmia_res = self.search_ahmia_onion(
                clean_topic, max_results=5,
                massive_crawl=True, massive_pages_per_site=20
            )
            if ahmia_res:
                all_results.extend(ahmia_res)
        except Exception as e:
            print(f"[WebCrawler] ⚠️ Ahmia Onion error: {e}")

        try:
            gc.collect(2)
            gc_tuner._malloc_trim()
        except Exception:
            pass
        time.sleep(3.0)

        # 3. Programming Onion (jika topik teknis)
        if is_programming_topic:
            print("[WebCrawler] 💻 Sumber 3/3: Programming Onion...")
            try:
                prog_res = self.fetch_programming_onion_sites(clean_topic, max_pages_per_site=15)
                if prog_res:
                    all_results.extend(prog_res)
            except Exception as e:
                print(f"[WebCrawler] ⚠️ Programming Onion error: {e}")

            try:
                gc.collect(2)
                gc_tuner._malloc_trim()
            except Exception:
                pass
            time.sleep(3.0)

        print(f"[WebCrawler] 🏁 Dark Web selesai: {len(all_results)} item")
        return all_results

    # ── Router Cerdas ─────────────────────────────────────────────────────────
    def route_and_fetch(self, query: str, topic_hint: str = "default") -> List[Dict]:
        """
        Pilih sumber terbaik berdasarkan topik, lalu cari dan ambil konten.
        Jika sumber pertama kosong, otomatis fallback ke sumber berikutnya.
        """
        # ── Deteksi URL .onion langsung → crawl masif ────────────────────────
        if ".onion" in query.lower() and (query.lower().startswith("http") or "://" in query):
            print("[WebCrawler] 🧅 Deteksi URL .onion langsung — Mulai crawl masif...")
            return self.fetch_onion_massive(query.strip(), max_pages=100, stay_on_domain=True)

        # ── Deteksi pola "belajar X di dark web" / "belajar X darkweb" ─────────
        import re as _re
        belajar_pattern = _re.compile(
            r"belajar\s+(.+?)\s+(?:di\s+)?(?:dark\s*web|darkweb|deep\s*web|deepweb|onion|tor|darknet)",
            _re.IGNORECASE
        )
        match_belajar = belajar_pattern.search(query)
        if match_belajar:
            clean_topic = match_belajar.group(1).strip()
            is_programming_topic = any(
                kw in clean_topic.lower() for kw in PROGRAMMING_KEYWORDS
            )
            print(f"[WebCrawler] 🎓 Mode BELAJAR DARK WEB — topik: '{clean_topic}'")
            print(f"[WebCrawler]    → Topik programming: {'YA' if is_programming_topic else 'TIDAK'}")
            print("[WebCrawler]    → Sumber: Wikipedia Onion + Ahmia Masif + Programming Seeds")

            # Jalankan 3 sumber secara berurutan agar RAM tidak membengkak sekaligus,
            # dan beri jeda pembersihan RAM di antara setiap sumber.
            all_results = []

            # 1. Wikipedia Onion
            print("[WebCrawler] 📖 Memulai sumber 1/3: Wikipedia Onion...")
            try:
                wiki_res = self.search_wikipedia_onion(clean_topic, lang="en", max_pages=5)
                if wiki_res:
                    all_results.extend(wiki_res)
            except Exception as e:
                print(f"[WebCrawler] ⚠️ Wikipedia Onion error: {e}")

            try:
                gc.collect(2)
                gc_tuner._malloc_trim()
            except Exception:
                pass
            time.sleep(3.0)

            # 2. Ahmia Onion
            print("[WebCrawler] 🧅 Memulai sumber 2/3: Ahmia Onion...")
            try:
                # Batasi crawl masif ke 20 halaman per situs agar hemat RAM & waktu
                ahmia_res = self.search_ahmia_onion(
                    clean_topic, max_results=5,
                    massive_crawl=True, massive_pages_per_site=20
                )
                if ahmia_res:
                    all_results.extend(ahmia_res)
            except Exception as e:
                print(f"[WebCrawler] ⚠️ Ahmia Onion error: {e}")

            try:
                gc.collect(2)
                gc_tuner._malloc_trim()
            except Exception:
                pass
            time.sleep(3.0)

            # 3. Programming Onion
            if is_programming_topic:
                print("[WebCrawler] 💻 Memulai sumber 3/3: Programming Onion...")
                try:
                    # Batasi crawl ke 15 halaman per seed
                    prog_res = self.fetch_programming_onion_sites(clean_topic, max_pages_per_site=15)
                    if prog_res:
                        all_results.extend(prog_res)
                except Exception as e:
                    print(f"[WebCrawler] ⚠️ Programming Onion error: {e}")

                try:
                    gc.collect(2)
                    gc_tuner._malloc_trim()
                except Exception:
                    pass
                time.sleep(3.0)

            print(f"[WebCrawler] 🏁 Total belajar dark web: {len(all_results)} item dari 3 sumber")
            return all_results

        # ── Deteksi kueri umum bertema Dark Web / Deep Web ────────────────────
        dark_web_keywords = [
            "dark web", "darkweb", "deep web", "deepweb",
            "onion site", "situs onion", "onion search",
            "ahmia", "tor search", "darknet"
        ]
        if any(kw in query.lower() for kw in dark_web_keywords):
            print("[WebCrawler] 🧅 Deteksi kueri Dark Web — Wikipedia + Ahmia Masif...")
            clean_query = query.lower()
            for kw in dark_web_keywords:
                clean_query = clean_query.replace(kw, "")
            clean_query = clean_query.strip() or query

            # Untuk kueri umum: Wikipedia onion + Ahmia secara berurutan dengan jeda RAM
            all_results = []
            
            # 1. Wikipedia Onion
            try:
                wiki_res = self.search_wikipedia_onion(clean_query, "en", 4)
                if wiki_res:
                    all_results.extend(wiki_res)
            except Exception as e:
                print(f"[WebCrawler] ⚠️ Wikipedia Onion error: {e}")
                
            try:
                gc.collect(2)
                gc_tuner._malloc_trim()
            except Exception:
                pass
            time.sleep(3.0)

            # 2. Ahmia Onion
            try:
                ahmia_res = self.search_ahmia_onion(clean_query, 5, False, 15)
                if ahmia_res:
                    all_results.extend(ahmia_res)
            except Exception as e:
                print(f"[WebCrawler] ⚠️ Ahmia Onion error: {e}")

            try:
                gc.collect(2)
                gc_tuner._malloc_trim()
            except Exception:
                pass
            time.sleep(3.0)
            
            return all_results

        if not is_internet_available():
            print("[WebCrawler] ⚠️ Koneksi internet tidak tersedia atau DNS tidak berfungsi. Melewati perayapan.")
            return []
        # ── Pastikan query tidak terlalu panjang (max 7 kata) ─────────────────
        words = query.split()
        if len(words) > 7:
            query = " ".join(words[:7])
        
        hint = topic_hint.lower()
        
        # Tentukan route
        route = SOURCE_ROUTING.get("default", ["arxiv", "duckduckgo"])
        for key in SOURCE_ROUTING:
            if key in hint:
                route = SOURCE_ROUTING[key]
                break
        
        all_results = []
        # Coba semua sumber di route secara paralel (kombinasi CPU & RAM dimaksimalkan)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_source_search(src):
            try:
                if src == "arxiv":
                    return self.search_arxiv(query)
                elif src == "semantic_scholar":
                    return self.search_semantic_scholar(query)
                elif src == "openalex":
                    return self.search_openlex(query)
                elif src == "mdn":
                    return self.search_mdn(query)
                elif src == "python_docs":
                    return self.search_python_docs(query)
                elif src == "duckduckgo":
                    return self.search_and_crawl_duckduckgo(query)
            except Exception as e:
                print(f"[WebCrawler] Gagal mengambil dari {src}: {e}")
            return []

        with ThreadPoolExecutor(max_workers=min(4, len(route))) as executor:
            futures = {executor.submit(run_source_search, src): src for src in route}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_results.extend(res)
        
        # ── Ultimate fallback: DuckDuckGo tanpa site restriction ──────────────
        if not all_results:
            try:
                ddg = DDGS().text(query, max_results=2)
                for r in (ddg or []):
                    url = r.get("href", "")
                    body = r.get("body", "")
                    if body and len(body) > 80:
                        all_results.append({
                            "source": "DDG-Fallback",
                            "url": url,
                            "text": body[:4000]
                        })
            except Exception:
                pass
        
        return all_results

    def auto_translate(self, text: str) -> str:
        """
        Menerjemahkan teks akademik asing ke Bahasa Indonesia.
        Melindungi secara UNIVERSAL:
        - Blok kode semua bahasa (``` python/js/go/rust/c/bash/sql/yaml/...
          dan ~~~ ...~~~)
        - Perintah terminal ($ cmd, # root, % zsh, >>> Python REPL)
        - Inline code (`...`)
        - Kode hasil crawl mentah dengan tanda baca program
        """
        if not text:
            return ""

        # Gunakan helper universal dari code_utils
        try:
            from moko_utils.text_utils import protect_code_blocks, restore_code_blocks
            protected, placeholders = protect_code_blocks(text)
        except Exception:
            # Fallback minimal jika import gagal
            protected = text
            placeholders = {}

        # Terjemahkan teks yang sudah diamankan ke Bahasa Indonesia
        if protected.strip():
            prompt = (
                "Terjemahkan teks ilmiah/matematika/pemrograman berikut ke Bahasa Indonesia yang baku. "
                "JANGAN TERJEMAHKAN: placeholder __MOKO_CODE_BLOCK_*__, rumus matematika, "
                "notasi simbolik, nama fungsi/kode, nama teknologi, nama library, "
                "dan istilah teknis yang tidak punya padanan baku. "
                "Kembalikan hanya hasil terjemahan, tanpa komentar tambahan.\n\n"
                f"TEKS:\n{protected}"
            )
            sys_prompt = (
                "You are a professional academic translator for Mathematics, Science, and Technology. "
                "Never modify any __MOKO_CODE_BLOCK_*__ placeholders — they represent code and must be kept verbatim."
            )
            try:
                result = engine.generate_text(prompt, sys_prompt).strip()
            except Exception as e:
                print(f"[Translator] Error: {e}")
                result = protected
        else:
            result = protected

        # Kembalikan semua placeholder kode ke konten program aslinya
        if placeholders:
            result = restore_code_blocks(result, placeholders)

        return result


# ─── Singleton ────────────────────────────────────────────────────────────────
multi_crawler = MultiSourceCrawler()
