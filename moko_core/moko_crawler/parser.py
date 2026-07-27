"""
moko_crawler/parser.py
=======================
HTML parser, link extractor, teks cleaner, dan metadata extractor.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict
from urllib.parse import urlparse

log = logging.getLogger("moko_crawler.parser")

# ── Lazy import BeautifulSoup ────────────────────────────────────────────────
try:
    from bs4 import BeautifulSoup, FeatureNotFound
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    log.warning("beautifulsoup4 tidak terinstall, parsing terbatas.")


# ── Language Detection ────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Deteksi bahasa teks secara simpel (fallback: 'unknown')."""
    try:
        from langdetect import detect
        if text and len(text.strip()) > 20:
            return detect(text[:2000])
    except Exception:
        pass
    return "unknown"


# ── HTML Content Parser ───────────────────────────────────────────────────────
class ContentParser:
    """Parse HTML halaman: ekstrak link, teks, metadata."""

    def __init__(self, detect_lang: bool = True):
        self._detect_lang = detect_lang

    def parse(self, html: str, base_url: str) -> Dict:
        """
        Parse HTML dan kembalikan dict berisi:
        - title, description, language
        - text_content (teks bersih)
        - links: [(href, anchor_text), ...]
        - onion_links: [(absolute_url, anchor_text), ...]
        """
        if not BS4_AVAILABLE:
            return self._parse_regex_fallback(html, base_url)

        result = {
            "title": "",
            "description": "",
            "language": "unknown",
            "text_content": "",
            "links": [],
            "onion_links": [],
        }

        try:
            # Coba lxml dulu (lebih cepat), fallback ke html.parser
            try:
                soup = BeautifulSoup(html, "lxml")
            except FeatureNotFound:
                soup = BeautifulSoup(html, "html.parser")

            # ── Title ──────────────────────────────────────────────────────
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text(strip=True)[:512]

            # ── Meta description ───────────────────────────────────────────
            for meta in soup.find_all("meta", attrs={"name": True, "content": True}):
                name = meta.get("name", "").lower()
                if name in ("description", "og:description"):
                    result["description"] = meta.get("content", "")[:1024]
                    break

            # ── Language dari HTML lang attribute ──────────────────────────
            html_tag = soup.find("html")
            if html_tag and html_tag.get("lang"):
                result["language"] = html_tag["lang"][:8]

            # ── Teks bersih ────────────────────────────────────────────────
            # Hapus script, style, nav, footer, header
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "noscript", "iframe", "svg", "img"]):
                tag.decompose()

            raw_text = soup.get_text(separator=" ", strip=True)
            # Hapus whitespace berlebih
            clean_text = re.sub(r"\s{2,}", " ", raw_text).strip()
            result["text_content"] = clean_text[:500_000]  # cap 500KB teks

            # Deteksi bahasa dari teks jika belum ada dari HTML
            if result["language"] == "unknown" and self._detect_lang:
                result["language"] = detect_language(clean_text[:3000])

            # ── Link extraction ────────────────────────────────────────────
            from .url_manager import resolve_url, extract_domain
            all_links = []
            onion_links = []

            for tag in soup.find_all("a", href=True):
                href = tag.get("href", "").strip()
                if not href:
                    continue
                anchor = tag.get_text(strip=True)[:256]
                all_links.append((href, anchor))

                # Resolve ke absolute URL
                resolved = resolve_url(base_url, href)
                if resolved:
                    domain = extract_domain(resolved)
                    if domain and domain.endswith(".onion"):
                        onion_links.append((resolved, anchor))

            result["links"] = all_links
            result["onion_links"] = onion_links

        except Exception as e:
            log.error(f"Parse error ({base_url}): {e}")

        return result

    def _parse_regex_fallback(self, html: str, base_url: str) -> Dict:
        """Fallback parsing pakai regex jika bs4 tidak tersedia."""
        from .url_manager import resolve_url, extract_domain

        result = {
            "title": "",
            "description": "",
            "language": "unknown",
            "text_content": "",
            "links": [],
            "onion_links": [],
        }

        # Title
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            result["title"] = re.sub(r"<[^>]+>", "", m.group(1)).strip()[:512]

        # Links
        for href, anchor in re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            html, re.IGNORECASE | re.DOTALL
        ):
            resolved = resolve_url(base_url, href)
            if resolved:
                domain = extract_domain(resolved)
                clean_anchor = re.sub(r"<[^>]+>", "", anchor).strip()[:256]
                result["onion_links"].append((resolved, clean_anchor))
                result["links"].append((href, clean_anchor))

        # Teks bersih (hapus semua tag)
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        result["text_content"] = clean[:500_000]

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
_PARSER = None

def get_parser(detect_lang: bool = True) -> ContentParser:
    global _PARSER
    if _PARSER is None:
        _PARSER = ContentParser(detect_lang=detect_lang)
    return _PARSER
