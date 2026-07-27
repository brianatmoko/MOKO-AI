"""
MOKO Multi-Domain Storage
==========================
Orkestrasi pencarian di beberapa domain SECARA PARALEL.

RSAStorage.search() adalah pure Python (tidak memanggil C++ kernel dengan
global lock), sehingga aman menggunakan ThreadPoolExecutor untuk parallelism
penuh. Sebelumnya search dilakukan serial — 6 domain × N detik = sangat lambat.

Penggunaan:
    from moko_memory.multi_domain_storage import MultiDomainStorage

    mds = MultiDomainStorage()
    results = mds.search(vector, domains=['math', 'physics'], top_k=5)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from moko_memory.rsa_storage import RSAStorage, ROOT_OMNI_DIR, get_domain_path


class MultiDomainStorage:
    """
    Pencarian multi-domain paralel yang menggabungkan hasil dari beberapa
    domain RSAStorage sekaligus dan melakukan re-ranking terpusat.

    Args:
        root_omni_dir: Override root direktori omni. Default: ROOT_OMNI_DIR.
    """

    def __init__(self, root_omni_dir: Optional[Path] = None):
        self.root_dir = Path(root_omni_dir) if root_omni_dir else ROOT_OMNI_DIR
        self._storage_cache: Dict[str, RSAStorage] = {}

    def _get_storage(self, domain: str) -> RSAStorage:
        """Mengembalikan instance RSAStorage untuk domain (di-cache)."""
        if domain not in self._storage_cache:
            self._storage_cache[domain] = RSAStorage(
                domain=domain,
                root_omni_dir=self.root_dir
            )
        return self._storage_cache[domain]

    def list_active_domains(self) -> List[str]:
        """
        Mengembalikan daftar domain yang memiliki data aktif
        (direktori ada dan berisi file index.bin dengan data atau _domain_meta.json dengan count > 0).
        """
        import json
        active = []
        if not self.root_dir.exists():
            return active
        for candidate in sorted(self.root_dir.iterdir()):
            if candidate.is_dir() and not candidate.name.startswith("_"):
                # Cek _domain_meta.json
                meta_file = candidate / "_domain_meta.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        if meta.get("entry_count", 0) > 0:
                            active.append(candidate.name)
                            continue
                    except Exception:
                        pass
                # Fallback: rglob index.bin
                for idx_file in candidate.rglob("index.bin"):
                    if idx_file.stat().st_size > 0:
                        active.append(candidate.name)
                        break
        return active

    def search(
        self,
        fp32_vector: List[float],
        domains: Optional[List[str]] = None,
        top_k: int = 5,
        n_probe: int = 16,
        domain_boost: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Mencari di beberapa domain SECARA PARALEL dan menggabungkan hasilnya.

        PERUBAHAN KRITIS v2.0:
        - Serial loop diganti ThreadPoolExecutor (max 6 worker)
        - Early-exit jika domain pertama sudah menemukan hasil sangat relevan (>= 0.90)
        - n_probe default diturunkan dari 64 → 16 untuk kecepatan

        Args:
            fp32_vector:  Vektor query FP32 (768-D).
            domains:      List nama domain yang akan dicari.
                          Jika None, akan otomatis mencari di semua domain aktif.
            top_k:        Jumlah hasil teratas yang dikembalikan (setelah digabung).
            n_probe:      Jumlah folder yang di-probe per domain (default turun ke 16).
            domain_boost: Optional dict untuk menaikkan skor domain tertentu.

        Returns:
            List of result dict, diurutkan berdasarkan score (tertinggi dulu).
            Setiap result berisi: text, source, domain, score, folder, log_number.
        """
        if domains is None:
            domains = self.list_active_domains()
            if not domains:
                print("[MultiDomain] Tidak ada domain aktif ditemukan.")
                return []

        boost = domain_boost or {}

        # ── Helper: search satu domain ──────────────────────────────────────
        def _search_domain(domain: str) -> List[Dict]:
            domain_path = get_domain_path(domain, self.root_dir)
            if not domain_path.exists():
                return []
            try:
                storage = self._get_storage(domain)
                results = storage.search(fp32_vector, top_k=top_k * 2, n_probe=n_probe)
                multiplier = boost.get(domain, 1.0)
                for r in results:
                    r["score"] *= multiplier
                    r["domain_boosted"] = multiplier != 1.0
                return results
            except Exception as e:
                print(f"[MultiDomain] Error searching domain '{domain}': {e}")
                return []

        # ── Parallel execution di semua domain sekaligus ────────────────────
        all_results: List[Dict] = []
        max_workers = min(len(domains), 6)
        
        # Deduplikasi domain list (mencegah scan ganda saat domain aktif)
        domains = list(dict.fromkeys(domains))

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="omni_search"
        ) as executor:
            future_map = {executor.submit(_search_domain, d): d for d in domains}
            for future in as_completed(future_map):
                domain_results = future.result()
                all_results.extend(domain_results)

                # Early-exit: jika sudah ada hasil sangat relevan (>= 0.92),
                # batalkan sisa future untuk menghemat waktu CPU.
                # 0.92 lebih konservatif dari sebelumnya (0.90) agar tidak
                # keluar terlalu dini dari domain yang bisa memberi konteks lebih baik.
                if any(r.get("score", 0.0) >= 0.92 for r in domain_results):
                    for pending in future_map:
                        pending.cancel()
                    break

        # Re-rank gabungan semua domain
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def ingest(
        self,
        text: str,
        fp32_vector: List[float],
        domain: str,
        source_name: str = "manual",
        log_number: int = 1,
        valence: float = 0.0,
        arousal: float = 0.5,
        memory_type: str = "semantic",
        consolidated_count: int = 0
    ) -> Optional[Dict]:
        """
        Menyimpan satu ingatan ke domain tertentu.
        """
        storage = self._get_storage(domain)
        return storage.ingest(
            text, fp32_vector,
            source_name=source_name,
            log_number=log_number,
            valence=valence,
            arousal=arousal,
            memory_type=memory_type,
            consolidated_count=consolidated_count
        )

    def get_all_stats(self, domains: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Mengembalikan statistik semua domain."""
        if domains is None:
            domains = self.list_active_domains()

        stats = {}
        for domain in domains:
            domain_path = get_domain_path(domain, self.root_dir)
            if domain_path.exists():
                storage = self._get_storage(domain)
                stats[domain] = storage.get_stats()
        return stats
