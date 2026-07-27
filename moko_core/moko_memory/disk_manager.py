import ctypes
import os
import gc
import json
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Path ke library C++ menggunakan lokasi file ini sebagai acuan (Portable Mode)
_BASE_DIR = Path(__file__).resolve().parent.parent  # → moko_core/
KERNEL_LIB_PATH = _BASE_DIR / "moko_cpp_kernel" / "libmoko_core.so"

class MokoKernelBridge:
    """
    Kelas Singleton yang mengatur komunikasi murni antara Python dan C++ Kernel.
    
    ⚠️ THREAD SAFETY: C++ kernel menggunakan global state (g_thread_pool, QEV_LUT).
    Semua panggilan ke .so HARUS diproteksi oleh _cpp_lock untuk mencegah data race
    dari EpistemicForager + DeepSynthesisWorker yang berjalan bersamaan.
    Data race ini yang menyebabkan stack smashing / kernel panic.
    """
    _instance = None
    _cpp_lock = threading.Lock()   # ← KUNCI GLOBAL: serialisasi semua panggilan C++
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MokoKernelBridge, cls).__new__(cls)
            cls._instance._init_library()
        return cls._instance

    def _init_library(self):
        try:
            self.lib = ctypes.CDLL(str(KERNEL_LIB_PATH))
            
            # Setup fungsi C++: void moko_kernel_init(int num_threads)
            self.lib.moko_kernel_init.argtypes = [ctypes.c_int]
            self.lib.moko_kernel_init.restype = None
            
            # Setup fungsi C++: void encode_qev_c(float* vector_fp32, uint8_t* out_qev_192)
            self.lib.encode_qev_c.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)]
            self.lib.encode_qev_c.restype = None
            
            # Setup fungsi C++: int search_mmap_top_k_c(...)
            self.lib.search_mmap_top_k_c.argtypes = [
                ctypes.POINTER(ctypes.c_uint8), 
                ctypes.c_char_p, 
                ctypes.c_int, 
                ctypes.POINTER(ctypes.c_int), 
                ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.search_mmap_top_k_c.restype = ctypes.c_int

            # Setup fungsi C++: int search_keywords_cpp(dir, keywords, out_paths_json, max_results)
            # dir         → const char* (UTF-8 null-terminated)
            # keywords    → const char* (kata dipisahkan spasi, UTF-8)
            # out_buf     → char* buffer output JSON array
            # buf_size    → int ukuran buffer
            # max_results → int jumlah maksimum file yang dikembalikan
            # return      → int jumlah file ditemukan (≤ max_results)
            self.lib.search_keywords_cpp.argtypes = [
                ctypes.c_char_p,          # directory
                ctypes.c_char_p,          # keywords (space-separated)
                ctypes.c_char_p,          # output JSON buffer (mutable)
                ctypes.c_int,             # buffer size
                ctypes.c_int,             # max_results
            ]
            self.lib.search_keywords_cpp.restype = ctypes.c_int

            # Inisialisasi kernel dengan 0 thread → auto-detect hardware_concurrency/2
            self.lib.moko_kernel_init(0)
            self.initialized = True
            
        except Exception as e:
            print(f"\033[1;31m[KERNEL ERROR] Gagal memuat {KERNEL_LIB_PATH}: {e}\033[0m")
            self.initialized = False

    def encode_vector(self, fp32_vector: List[float]) -> bytes:
        if not self.initialized or len(fp32_vector) != 768:
            return b""
        
        with MokoKernelBridge._cpp_lock:
            vec_c = (ctypes.c_float * 768)(*fp32_vector)
            out_c = (ctypes.c_uint8 * 192)()
            self.lib.encode_qev_c(vec_c, out_c)
            return bytes(out_c)

    def search_mmap_file(self, query_bytes: bytes, filepath: Path, top_k: int = 5) -> Tuple[List[int], List[int]]:
        if not self.initialized or len(query_bytes) != 192 or not filepath.exists():
            return [], []
            
        with MokoKernelBridge._cpp_lock:
            q_c = (ctypes.c_uint8 * 192).from_buffer_copy(query_bytes)
            file_c = str(filepath).encode('utf-8')
            out_indices_c = (ctypes.c_int * top_k)()
            out_scores_c = (ctypes.c_int * top_k)()
            
            actual_k = self.lib.search_mmap_top_k_c(q_c, file_c, top_k, out_indices_c, out_scores_c)
            
            if actual_k <= 0:
                return [], []
                
            return list(out_indices_c)[:actual_k], list(out_scores_c)[:actual_k]

    def search_keywords_native(self, directory: str, keywords: list, max_results: int = 50) -> List[str]:
        """
        Pencarian keyword di filesystem menggunakan C++ native (std::filesystem).
        Jauh lebih cepat dari Python glob/open fallback (< 50ms vs ~3800ms).

        Args:
            directory:   Path direktori akar untuk dicari secara rekursif.
            keywords:    List kata kunci; sebuah file cocok jika SEMUA keyword ada di dalamnya.
            max_results: Batas jumlah file hasil (default 50).

        Returns:
            List[str] path file yang cocok, atau [] jika kernel tidak tersedia.
        """
        if not self.initialized or not keywords:
            return []

        kw_str = " ".join(k.strip() for k in keywords if k.strip())
        if not kw_str:
            return []

        BUF_SIZE = 65536  # 64KB cukup untuk 50 path panjang
        out_buf = ctypes.create_string_buffer(BUF_SIZE)

        with MokoKernelBridge._cpp_lock:
            n = self.lib.search_keywords_cpp(
                directory.encode("utf-8"),
                kw_str.encode("utf-8"),
                out_buf,
                ctypes.c_int(BUF_SIZE),
                ctypes.c_int(max_results),
            )

        if n <= 0:
            return []

        try:
            import json as _json
            raw = out_buf.value.decode("utf-8", errors="replace")
            paths = _json.loads(raw)
            return paths[:max_results]
        except Exception:
            return []



import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from moko_memory.rsa_storage import RSAStorage
from moko_memory.multi_domain_storage import MultiDomainStorage
from moko_memory.wal_manager import WALManager
from moko_memory.search_cache import search_cache

class DiskManager:
    """
    Compatibility layer: Mengatur pembacaan dan penulisan OMNI-INDEX menggunakan
    arsitektur baru MOKO-RSA (Recursive Semantic Array).

    Dilengkapi 3 lapisan "rasa puas":
    1. Semantic Deduplication   — Tidak menyimpan memori yang terlalu mirip (>0.85)
    2. Confidence Lock          — Topik yang Omni sudah "kuasai" tidak perlu disimpan lagi
    3. Source Quality Gate      — Jawaban AI (conversation) tidak mencemari data faktual

    Update v2:
    - WALManager menggantikan WAL langsung (rotation + checkpoint otomatis)
    - SearchCache di omni_first_search untuk menghindari scan ulang
    - post_ingest GC cleanup via gc_tuner
    """
    # Similarity threshold di atas ini = duplikat → skip ingestion
    DEDUP_THRESHOLD = 0.85

    # Jika Omni sudah punya cukup entri relevan dengan kepercayaan tinggi → "sudah puas"
    CONFIDENCE_LOCK_SCORE   = 0.72   # Top result score minimum untuk dianggap "sudah tahu"
    CONFIDENCE_LOCK_MIN_K   = 3      # Minimal 3 entri relevan sebelum lock berlaku

    # Checkpoint WAL setiap N ingest berhasil
    WAL_CHECKPOINT_INTERVAL = 200

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self._rsa_instances = {}
        self.mds = MultiDomainStorage()
        self._wal = WALManager(self.workspace_path)
        self._ingest_count = 0  # counter untuk checkpoint WAL berkala
        self._load_wal()

    def get_rsa(self, domain: Optional[str] = None) -> RSAStorage:
        """Mengambil instance RSAStorage berdasarkan domain, dengan lazy loading."""
        from moko_memory.rsa_storage import DEFAULT_DOMAIN
        dom = domain if domain else DEFAULT_DOMAIN
        if dom not in self._rsa_instances:
            self._rsa_instances[dom] = RSAStorage(domain=dom)
        return self._rsa_instances[dom]

    def _load_wal(self):
        """Memuat kembali memori dari WAL ke domain yang benar saat restart."""
        records = self._wal.replay()
        if not records:
            return

        print(f"[DiskManager] Membaca Write-Ahead Log ({len(records)} entri)...")
        count = 0
        try:
            for data in records:
                try:
                    domain = data.get("domain", "general")
                    rsa_inst = self.get_rsa(domain)
                    rsa_inst.ingest(
                        text=data["text"],
                        fp32_vector=data["vector"],
                        source_name=data["source_name"],
                        log_number=1,
                        valence=float(data.get("valence", 0.0)),
                        arousal=float(data.get("arousal", 0.5)),
                        memory_type=str(data.get("memory_type", "semantic")),
                        consolidated_count=int(data.get("consolidated_count", 0))
                    )
                    count += 1
                except Exception:
                    pass
            print(f"[DiskManager] ✅ Berhasil memulihkan {count} memori dari WAL.")
            # Setelah replay berhasil, checkpoint WAL (hapus file lama)
            if count > 0:
                self._wal.checkpoint()
        except Exception as e:
            print(f"[DiskManager] ⚠️ Gagal membaca WAL: {e}")

    def _append_to_wal(
        self, 
        file_path: str, 
        text_chunk: str, 
        fp32_vector: List[float],
        domain: str = "general",
        valence: float = 0.0,
        arousal: float = 0.5,
        memory_type: str = "semantic",
        consolidated_count: int = 0
    ):
        """Delegasikan penulisan WAL ke WALManager (thread-safe, dengan rotasi)."""
        self._wal.append(
            source_name=file_path,
            text=text_chunk,
            vector=fp32_vector,
            domain=domain,
            valence=valence,
            arousal=arousal,
            memory_type=memory_type,
            consolidated_count=consolidated_count,
        )

    def _maybe_checkpoint_wal(self):
        """Checkpoint WAL secara berkala setelah sejumlah ingest berhasil."""
        self._ingest_count += 1
        if self._ingest_count % self.WAL_CHECKPOINT_INTERVAL == 0:
            # Hanya checkpoint jika semua data sudah berhasil masuk RSA
            # WAL masih dibutuhkan sampai setelah restart — jadi ini opsional
            # dan hanya membersihkan file .bak yang sudah lama
            try:
                bak = self._wal.bak_path
                if bak.exists() and bak.stat().st_size > 0:
                    # .bak sudah lama (>10 menit), aman untuk dihapus
                    import time
                    age = time.time() - bak.stat().st_mtime
                    if age > 600:  # 10 menit
                        import os
                        os.remove(str(bak))
                        print("[DiskManager] 🗑️ WAL .bak lama dihapus (sudah diingest).")
            except Exception:
                pass

    def save_memory(
        self,
        text: str,
        embedding: List[float],
        domain: str = "general",
        metadata: Optional[Dict] = None
    ):
        """Wrapper kompatibilitas untuk menyimpan memori ke OMNI-INDEX."""
        file_path = "manual"
        source_type = "factual"
        if metadata:
            file_path = metadata.get("source", metadata.get("path", "manual"))
            if metadata.get("source_type") == "conversation":
                source_type = "conversation"
        return self.ingest_chunk(
            file_path=file_path,
            text_chunk=text,
            fp32_vector=embedding,
            source_type=source_type,
            domain=domain
        )

    def ingest_chunk(
        self,
        file_path: str,
        text_chunk: str,
        fp32_vector: List[float],
        source_type: str = "factual",   # "factual" | "conversation"
        domain: Optional[str] = None,
        valence: float = 0.0,
        arousal: float = 0.5,
        memory_type: str = "semantic",
        consolidated_count: int = 0
    ):
        """
        Menyimpan chunk ke dalam arsitektur RSA.
        
        source_type:
          "factual"      → Data KBBI, buku, Wikipedia. Selalu disimpan.
          "conversation" → Jawaban AI saat chatting. Disimpan dengan aturan ketat.
        """
        rsa_inst = self.get_rsa(domain)

        # ── GATE 1: Semantic Deduplication ────────────────────────────
        similar = rsa_inst.search(fp32_vector, top_k=1, n_probe=5)
        if similar:
            top_sim = similar[0].get("score", 0.0)
            if top_sim >= self.DEDUP_THRESHOLD:
                return ("DEDUP_SKIP", b"")

        # ── GATE 2: Confidence Lock (hanya untuk sumber conversation) ─
        if source_type == "conversation":
            lock_check = rsa_inst.search(fp32_vector, top_k=self.CONFIDENCE_LOCK_MIN_K, n_probe=10)
            if len(lock_check) >= self.CONFIDENCE_LOCK_MIN_K:
                top_score = lock_check[0].get("score", 0.0)
                if top_score >= self.CONFIDENCE_LOCK_SCORE:
                    return ("CONFIDENCE_LOCKED", b"")

        # ── GATE 3: Panjang Minimum ────────────────────────────────────
        if len(text_chunk.strip()) < 50:
            return ("TOO_SHORT", b"")

        # ── Lolos semua gate → Simpan ke RSA ─────────────────────────
        res = rsa_inst.ingest(
            text=text_chunk,
            fp32_vector=fp32_vector,
            source_name=file_path,
            log_number=1,
            valence=valence,
            arousal=arousal,
            memory_type=memory_type,
            consolidated_count=consolidated_count
        )
        
        # ── GATE 4: Write-Ahead Log (WAL) ───────────────────────────────
        if source_type == "conversation":
            self._append_to_wal(
                file_path, text_chunk, fp32_vector,
                domain=domain or "general",
                valence=valence, arousal=arousal,
                memory_type=memory_type, consolidated_count=consolidated_count
            )
        
        # Invalidate search cache karena ada data baru
        search_cache.invalidate()

        self._maybe_checkpoint_wal()
            
        if not res:
            return None

        return (res["folder"], b"")

    def ingest_chunks_batch(
        self,
        items: List[Dict],
        source_type: str = "factual",   # "factual" | "conversation"
        domain: Optional[str] = None
    ) -> List[Optional[Tuple[str, bytes]]]:
        """
        Menyimpan batch chunk ke dalam arsitektur RSA secara efisien.
        Setiap item: {"file_path": str, "text_chunk": str, "fp32_vector": List[float]}
        """
        import concurrent.futures
        rsa_inst = self.get_rsa(domain)
        
        valid_items = []
        skipped_results = {} # index -> (status, bytes)
        
        def process_item_gates(index, item):
            file_path = item["file_path"]
            text_chunk = item["text_chunk"]
            fp32_vector = item["fp32_vector"]
            
            # ── GATE 3: Panjang Minimum ────────────────────────────────────
            if len(text_chunk.strip()) < 50:
                return ("TOO_SHORT", index, item)
                
            # ── GATE 1: Semantic Deduplication ────────────────────────────
            similar = rsa_inst.search(fp32_vector, top_k=1, n_probe=5)
            if similar:
                top_sim = similar[0].get("score", 0.0)
                if top_sim >= self.DEDUP_THRESHOLD:
                    return ("DEDUP_SKIP", index, item)
                    
            # ── GATE 2: Confidence Lock ────────────────────────────
            if source_type == "conversation":
                lock_check = rsa_inst.search(fp32_vector, top_k=self.CONFIDENCE_LOCK_MIN_K, n_probe=10)
                if len(lock_check) >= self.CONFIDENCE_LOCK_MIN_K:
                    top_score = lock_check[0].get("score", 0.0)
                    if top_score >= self.CONFIDENCE_LOCK_SCORE:
                        return ("CONFIDENCE_LOCKED", index, item)
                        
            return ("OK", index, item)

        # Jalankan gate checking secara paralel menggunakan thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_item_gates, i, item) for i, item in enumerate(items)]
            for future in concurrent.futures.as_completed(futures):
                status, idx, item = future.result()
                if status == "OK":
                    valid_items.append((idx, item))
                else:
                    skipped_results[idx] = (status, b"")

        # Urutkan valid items agar urutan masukan tetap konsisten
        valid_items.sort(key=lambda x: x[0])
        
        results = [None] * len(items)
        for idx, val in skipped_results.items():
            results[idx] = val
            
        if valid_items:
            rsa_batch_args = []
            for idx, item in valid_items:
                rsa_batch_args.append({
                    "text": item["text_chunk"],
                    "fp32_vector": item["fp32_vector"],
                    "source_name": item["file_path"],
                    "log_number": 1,
                    "valence": float(item.get("valence", 0.0)),
                    "arousal": float(item.get("arousal", 0.5)),
                    "memory_type": str(item.get("memory_type", "semantic")),
                    "consolidated_count": int(item.get("consolidated_count", 0))
                })
                
            # Jalankan batch ingestion
            rsa_results = rsa_inst.ingest_batch(rsa_batch_args)
            
            # Letakkan hasil kembali ke array results
            for (idx, item), res in zip(valid_items, rsa_results):
                if res:
                    results[idx] = (res["folder"], b"")
                    # WAL append (jika source_type == "conversation")
                    if source_type == "conversation":
                        self._append_to_wal(
                            item["file_path"], item["text_chunk"], item["fp32_vector"],
                            domain=domain or "general",
                            valence=float(item.get("valence", 0.0)),
                            arousal=float(item.get("arousal", 0.5)),
                            memory_type=str(item.get("memory_type", "semantic")),
                            consolidated_count=int(item.get("consolidated_count", 0))
                        )
                else:
                    results[idx] = None

            # Invalidate cache karena ada data baru di RSA
            search_cache.invalidate()
            self._ingest_count += len([r for r in results if r and isinstance(r, tuple) and r[0] not in ("DEDUP_SKIP", "CONFIDENCE_LOCKED", "TOO_SHORT")])
            self._maybe_checkpoint_wal()

        return results

    def search_memory(self, fp32_vector: List[float], top_k: int = 3, domain: Optional[str] = None) -> List[Dict]:
        """Mencari ingatan di arsitektur RSA.
        
        Jika domain ditentukan, hanya cari pada domain tersebut (jauh lebih cepat).
        Jika domain=None, cari lintas semua domain menggunakan MultiDomainStorage.
        """
        if domain:
            # Cari hanya pada RSA instance domain tertentu (cepat)
            rsa_inst = self.get_rsa(domain)
            results = rsa_inst.search(fp32_vector, top_k=top_k, n_probe=64)
        else:
            results = self.mds.search(fp32_vector, top_k=top_k, n_probe=64)

        compat_results = []
        for r in results:
            compat_results.append({
                "file":  r["source"],
                "text":  r["text"],
                "score": r["score"],
                "folder": r["folder"],
                "log":   r["log_number"],
                "valence": float(r.get("valence", 0.0)),
                "arousal": float(r.get("arousal", 0.5)),
                "memory_type": r.get("memory_type", "semantic"),
                "consolidated_count": int(r.get("consolidated_count", 0))
            })

        return compat_results

    def omni_first_search(self, fp32_vector: List[float], top_k: int = 5, target_domain: Optional[str] = None) -> dict:
        """
        Omni-First RAG: Cari di Omni DULU sebelum LLM berpikir.
        Dilengkapi SearchCache L1 untuk menghindari re-scan berulang.
        """
        RELEVANCE_THRESHOLD = 0.40

        # ── L1 Cache Check ────────────────────────────────────────────
        cached = search_cache.get(fp32_vector)
        if cached is not None:
            top_score = cached[0].get("score", 0.0) if cached else 0.0
            if top_score < RELEVANCE_THRESHOLD:
                return {"found": False, "confidence": top_score, "context": "", "results": cached}
            context_lines = [
                f"[Sumber: {r['source']} | Relevansi: {r['score']:.3f}]\n{r['text']}"
                for r in cached if r.get("score", 0.0) >= RELEVANCE_THRESHOLD
            ]
            return {
                "found": True,
                "confidence": top_score,
                "context": "\n\n".join(context_lines),
                "results": cached,
                "_cache_hit": True,
            }

        boost = {}
        if target_domain:
            boost[target_domain] = 1.3

        results = self.mds.search(fp32_vector, top_k=10, n_probe=8, domain_boost=boost)

        if not results:
            search_cache.put(fp32_vector, [])
            return {"found": False, "confidence": 0.0, "context": "", "results": []}

        top_score = results[0].get("score", 0.0)

        # Simpan ke cache (baik hit maupun miss)
        search_cache.put(fp32_vector, results)

        if top_score < RELEVANCE_THRESHOLD:
            return {"found": False, "confidence": top_score, "context": "", "results": results}

        context_lines = [
            f"[Sumber: {r['source']} | Relevansi: {r['score']:.3f}]\n{r['text']}"
            for r in results if r.get("score", 0.0) >= RELEVANCE_THRESHOLD
        ]
        context_text = "\n\n".join(context_lines)

        return {
            "found":      True,
            "confidence": top_score,
            "context":    context_text,
            "results":    results
        }
