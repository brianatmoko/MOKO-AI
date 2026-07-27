"""
MOKO Autonomous Data Forager (ADF)
====================================
Filosofi:
  Model harus BEKERJA KERAS mendapatkan data sendiri — bukan menunggu user.
  Seperti lebah yang keluar sarang setiap hari untuk mengumpulkan nektar,
  MOKO harus keluar ke internet setiap saat ada idle time.

Cara Kerja:
  1. Ketika MOKO OS idle (tidak ada pertanyaan user), ADF aktif
  2. ADF memiliki "Priority Knowledge Queue" — daftar topik yang harus dikuasai
  3. Setiap siklus: pilih topik → generate subtopik via LLM → crawl web → enkripsi → simpan
  4. ADF juga belajar dari kegagalan: jika MOKO tidak bisa jawab → topik itu masuk queue teratas
  5. ADF melacak "coverage map" — topik apa yang sudah dipelajari, berapa banyak

DOMAIN PRIORITAS (berdasarkan fokus proyek):
  - Programming (Python, C, Rust, Assembly, JS)
  - Cryptography & Security
  - Mathematics (formal proofs, algorithms)
  - AI/ML internals
  - Hardware & Low-level systems
  - Blockchain & Crypto tech

Cara aktivasi:
  Otomatis berjalan saat MOKO OS idle > 30 detik
  Atau panggil: /forager start|stop|status|priority <topik>
"""

import json
import os
import re
import threading
import time
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# PRIORITY KNOWLEDGE QUEUE — Topik yang HARUS dikuasai MOKO
# ════════════════════════════════════════════════════════════════════════

DEFAULT_PRIORITY_TOPICS = [
    # Programming & Algorithms (TERTINGGI — ini core MOKO)
    {"topic": "Python internals CPython bytecode", "domain": "programming", "priority": 10},
    {"topic": "C memory management malloc free", "domain": "programming", "priority": 10},
    {"topic": "Rust ownership borrowing lifetimes", "domain": "programming", "priority": 9},
    {"topic": "assembly x86-64 instruction set", "domain": "programming", "priority": 9},
    {"topic": "data structures algorithms complexity", "domain": "programming", "priority": 10},
    {"topic": "dynamic programming memoization", "domain": "programming", "priority": 9},
    {"topic": "graph algorithms BFS DFS Dijkstra", "domain": "programming", "priority": 9},
    {"topic": "compiler design lexer parser AST", "domain": "programming", "priority": 8},
    {"topic": "operating system process thread scheduler", "domain": "programming", "priority": 8},
    {"topic": "network protocol TCP IP socket", "domain": "programming", "priority": 8},
    {"topic": "database SQL query optimization index", "domain": "programming", "priority": 8},
    {"topic": "concurrent programming mutex semaphore", "domain": "programming", "priority": 8},
    
    # Cryptography & Security
    {"topic": "AES encryption symmetric key block cipher", "domain": "cybersecurity", "priority": 9},
    {"topic": "RSA public key cryptography", "domain": "cybersecurity", "priority": 9},
    {"topic": "elliptic curve cryptography ECC", "domain": "cybersecurity", "priority": 9},
    {"topic": "hash function SHA256 collision resistance", "domain": "cybersecurity", "priority": 8},
    {"topic": "zero knowledge proof ZKP", "domain": "cybersecurity", "priority": 8},
    {"topic": "buffer overflow exploitation stack smashing", "domain": "cybersecurity", "priority": 7},
    {"topic": "reverse engineering binary analysis", "domain": "cybersecurity", "priority": 7},
    
    # Mathematics & Formal Methods
    {"topic": "linear algebra matrix operations eigenvalue", "domain": "mathematics", "priority": 8},
    {"topic": "calculus derivatives integrals", "domain": "mathematics", "priority": 8},
    {"topic": "discrete mathematics set theory proof", "domain": "mathematics", "priority": 8},
    {"topic": "probability statistics bayesian inference", "domain": "mathematics", "priority": 8},
    {"topic": "number theory prime factorization modular arithmetic", "domain": "mathematics", "priority": 7},
    {"topic": "formal verification SMT solver Z3", "domain": "mathematics", "priority": 7},
    
    # AI & Machine Learning
    {"topic": "transformer attention mechanism BERT GPT", "domain": "ai_ml", "priority": 9},
    {"topic": "neural network backpropagation gradient descent", "domain": "ai_ml", "priority": 9},
    {"topic": "quantization LLM inference optimization", "domain": "ai_ml", "priority": 9},
    {"topic": "RAG retrieval augmented generation embedding", "domain": "ai_ml", "priority": 8},
    {"topic": "RLHF reinforcement learning human feedback", "domain": "ai_ml", "priority": 8},
    {"topic": "vector database FAISS similarity search", "domain": "ai_ml", "priority": 7},
    
    # Web & API
    {"topic": "REST API design HTTP methods status codes", "domain": "programming", "priority": 7},
    {"topic": "JavaScript async await Promise", "domain": "programming", "priority": 7},
    {"topic": "React hooks state management", "domain": "programming", "priority": 6},
    {"topic": "websocket real-time communication", "domain": "programming", "priority": 7},
    
    # Blockchain & Crypto
    {"topic": "blockchain consensus mechanism proof of work", "domain": "blockchain", "priority": 7},
    {"topic": "smart contract Solidity EVM", "domain": "blockchain", "priority": 7},
    {"topic": "DeFi protocol tokenomics", "domain": "blockchain", "priority": 6},
]


@dataclass
class ForagerTopicItem:
    """Satu item dalam antrian pengetahuan forager."""
    topic: str
    domain: str
    priority: int                    # 1-10, makin tinggi makin penting
    cycles_done: int = 0             # Berapa siklus crawling sudah dilakukan
    chunks_stored: int = 0           # Berapa chunk data berhasil disimpan
    last_crawled: float = 0.0        # Unix timestamp terakhir dicrawl
    source: str = "default_queue"    # 'default_queue' | 'user_request' | 'failure_feedback'
    
    def score(self) -> float:
        """Skor prioritas dinamis: prioritas tinggi + lama tidak dicrawl = dipilih duluan."""
        idle_hours = (time.time() - self.last_crawled) / 3600 if self.last_crawled > 0 else 999
        coverage_penalty = min(self.chunks_stored / 50, 1.0)  # Makin banyak data → penalty
        return self.priority + min(idle_hours / 24, 5) - (coverage_penalty * 3)


class MokoAutonomousForager:
    """
    Forager otonom MOKO OS.
    
    Berjalan sebagai background thread — mengumpulkan data dari internet
    secara terus menerus saat sistem idle, tanpa perlu intervensi user.
    
    Filosofi: "Lebah tidak menunggu petani menyuruhnya keluar. Lebah tahu tugasnya."
    """
    
    STATE_FILE = ".moko_forager_state.json"
    
    def __init__(self, workspace_dir: str, verbose: bool = False):
        self.workspace_dir = Path(workspace_dir)
        self.verbose = verbose
        
        self._state_file = self.workspace_dir / self.STATE_FILE
        self._topics: List[ForagerTopicItem] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_idle = True          # UI set ini ke False saat user aktif
        self._current_topic: Optional[str] = None
        self._total_chunks_today = 0
        self._session_start = time.time()
        # Tunda mulai foraging 120 detik dari startup MOKO
        # agar server inferensi punya waktu muat sebelum dibebani crawl
        self._foraging_allowed_after = time.time() + 120.0
        
        # Load state
        self._load_state()
        
        # Inisialisasi topik default jika queue kosong
        if not self._topics:
            self._init_default_topics()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  🐝 [Forager] {msg}")
    
    def _load_state(self):
        """Load state forager dari disk (bertahan antar restart)."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                self._topics = [ForagerTopicItem(**t) for t in data.get("topics", [])]
                self._total_chunks_today = data.get("total_chunks_today", 0)
                self._log(f"State dimuat: {len(self._topics)} topik, {self._total_chunks_today} chunk hari ini")
            except Exception as e:
                self._log(f"Gagal load state: {e} — reinisialisasi")
                self._topics = []
    
    def _save_state(self):
        """Simpan state ke disk agar bertahan saat restart."""
        try:
            data = {
                "topics": [asdict(t) for t in self._topics],
                "total_chunks_today": self._total_chunks_today,
                "last_saved": time.time()
            }
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log(f"Gagal simpan state: {e}")
    
    def _init_default_topics(self):
        """Inisialisasi daftar topik default."""
        for t in DEFAULT_PRIORITY_TOPICS:
            self._topics.append(ForagerTopicItem(**t))
        self._log(f"Inisialisasi {len(self._topics)} topik default")
        self._save_state()
    
    def add_topic(self, topic: str, domain: str = "general", priority: int = 8,
                  source: str = "user_request"):
        """
        Tambah topik baru ke antrian forager.
        Dipanggil dari UI (/forager priority <topik>) atau dari sistem saat deteksi kegagalan.
        """
        with self._lock:
            # Cek duplikat
            existing = next((t for t in self._topics if t.topic.lower() == topic.lower()), None)
            if existing:
                existing.priority = max(existing.priority, priority)
                self._log(f"Topik sudah ada, update prioritas: '{topic}' → {existing.priority}")
            else:
                self._topics.append(ForagerTopicItem(
                    topic=topic, domain=domain, priority=priority, source=source
                ))
                self._log(f"Topik baru ditambahkan: '{topic}' (priority={priority})")
            self._save_state()
    
    def add_failure_topic(self, failed_query: str, domain: str = "general"):
        """
        Dipanggil otomatis saat MOKO tidak bisa menjawab pertanyaan.
        Topik dari kegagalan mendapat prioritas TERTINGGI (10).
        'Belajar dari kegagalan' — ini feedback loop terpenting.
        """
        self.add_topic(
            topic=failed_query[:80],
            domain=domain,
            priority=10,
            source="failure_feedback"
        )
        self._log(f"🔴 Topik kegagalan ditambahkan ke queue: '{failed_query[:50]}...'")
    
    def _select_next_topic(self) -> Optional[ForagerTopicItem]:
        """Pilih topik berikutnya berdasarkan skor dinamis."""
        with self._lock:
            if not self._topics:
                return None
            
            # Prioritaskan topik dari kegagalan dan user request
            failure_topics = [t for t in self._topics if t.source == "failure_feedback"]
            if failure_topics:
                return max(failure_topics, key=lambda t: t.score())
            
            # Pilih dari semua topik berdasarkan skor
            return max(self._topics, key=lambda t: t.score())
    
    def _do_one_foraging_cycle(self):
        """
        Satu siklus foraging:
        1. Pilih topik terbaik
        2. Generate subtopik via LLM (tanpa thinking, cepat)
        3. Crawl web
        4. Enkripsi & simpan ke Omni
        """
        topic_item = self._select_next_topic()
        if not topic_item:
            return
        
        self._current_topic = topic_item.topic
        self._log(f"Foraging: '{topic_item.topic}' (prio={topic_item.priority}, score={topic_item.score():.1f})")
        
        try:
            from moko_agents.llm_engine import engine as llm_engine
            from moko_neuromath.web_crawler import multi_crawler
            from moko_memory.disk_manager import DiskManager
            from moko_memory.rsa_storage import map_topic_to_domain
            from moko_neuromath.thalamus_gate import thalamus_gate
            from moko_config import settings
            from moko_utils.text_utils import chunk_text_code_aware
            
            disk_mgr = DiskManager(settings.WORKSPACE_DIR)
            
            # ── Step 1: Generate query spesifik via LLM (ultra cepat, no thinking) ──
            if topic_item.cycles_done == 0:
                # Siklus pertama: gunakan topik langsung
                query = topic_item.topic
            else:
                # Siklus berikutnya: minta LLM untuk subtopik yang belum dipelajari
                prompt = (
                    f"Topic: '{topic_item.topic}'. Cycles done: {topic_item.cycles_done}. "
                    f"Write ONE short English search query (max 5 words) about a specific, "
                    f"unexplored technical sub-aspect of this topic. "
                    f"Output the query ONLY. No explanations."
                )
                try:
                    raw = llm_engine.generate_text(
                        prompt, 
                        "Output search query only.",
                        coop_params={"num_predict": 20, "enable_thinking": False}
                    ).strip()
                    # Bersihkan output LLM
                    query = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                    query = re.sub(r'^["\'\d\.\-\*\s]+', '', query).strip()
                    if not query or len(query) > 100:
                        query = topic_item.topic
                except Exception:
                    query = topic_item.topic
            
            self._log(f"  Query: '{query}'")
            
            # ── Step 2: Crawl web ──
            results = multi_crawler.route_and_fetch(query, topic_hint=topic_item.topic)
            
            if not results:
                self._log(f"  ❌ Tidak ada hasil untuk: '{query}'")
                topic_item.last_crawled = time.time()
                topic_item.cycles_done += 1
                self._save_state()
                return
            
            self._log(f"  ✅ {len(results)} artikel ditemukan")
            
            # ── Step 3: Proses & enkripsi ──
            chunks_stored = 0
            target_dom = map_topic_to_domain(topic_item.topic)

            for art in results[:3]:  # Max 3 artikel per siklus (hemat RAM)
                text = art.get("text", "")
                if not text or len(text) < 200:  # Batas minimum 200 karakter
                    continue
                
                chunks = chunk_text_code_aware(text)
                source_name = f"forager_{topic_item.topic.replace(' ', '_')[:30]}.txt"
                
                # Batch embedding
                embs = llm_engine.get_embeddings_batch(chunks[:8])
                if not embs:
                    embs = [llm_engine.get_embedding(c) for c in chunks[:8]]
                
                batch_items = []
                for chunk, emb in zip(chunks[:8], embs):
                    if not emb or len(emb) != 768:
                        continue
                    
                    # Quality Gate: Filter spam/noise tanda baca tinggi (kecuali block kode terformat)
                    chunk_strip = chunk.strip()
                    if len(chunk_strip) < 200:
                        continue
                    
                    punct_ratio = sum(1 for c in chunk_strip if c in '.,!?;:()[]{}#$&*-+/="\'') / len(chunk_strip)
                    # Saring layout garbage/daftar isi tanpa teks normal, tapi abaikan jika ada format code block
                    if punct_ratio > 0.25 and "```" not in chunk_strip:
                        continue

                    if not thalamus_gate.is_novel(emb):
                        continue  # Skip duplikat
                    
                    # Simpan dengan domain tag terstruktur agar Scope Guard dapat bekerja secara presisi
                    batch_items.append({
                        "file_path": source_name,
                        "text_chunk": f"[DOMAIN: {target_dom}] [SOURCE: {source_name}]\n{chunk_strip}",
                        "fp32_vector": emb
                    })
                
                if batch_items:
                    results_ingest = disk_mgr.ingest_chunks_batch(
                        batch_items, source_type="autonomous_forager", domain=target_dom
                    )
                    stored = sum(1 for r in results_ingest 
                                if r and r[0] not in ("DEDUP_SKIP", "CONFIDENCE_LOCKED", "TOO_SHORT"))
                    chunks_stored += stored
            
            # ── Step 4: Update state ──
            with self._lock:
                topic_item.cycles_done += 1
                topic_item.chunks_stored += chunks_stored
                topic_item.last_crawled = time.time()
                self._total_chunks_today += chunks_stored
            
            self._save_state()
            self._log(f"  💾 {chunks_stored} chunk baru disimpan (total topik ini: {topic_item.chunks_stored})")
            
        except Exception as e:
            self._log(f"  ⚠️ Error pada siklus foraging: {e}")
        finally:
            self._current_topic = None
    
    def _forager_loop(self):
        """Loop utama forager — berjalan di background thread."""
        self._log("🐝 Forager loop dimulai")

        while not self._stop_event.is_set():
            # Fase startup: tunggu dulu 120 detik baru mulai forage
            now = time.time()
            if now < self._foraging_allowed_after:
                sisa = int(self._foraging_allowed_after - now)
                self._log(f"⏳ Startup cooldown: {sisa}s tersisa sebelum foraging aktif...")
                self._stop_event.wait(timeout=min(30, sisa + 1))
                continue

            # Cek apakah sistem idle
            if not self._is_idle:
                # User sedang aktif — pause forager, cek lagi 5 detik
                self._stop_event.wait(timeout=5)
                continue

            # Cek apakah inference server sudah online (jangan paksa load)
            try:
                import requests as _req
                from moko_config import settings as _s
                r = _req.get(f"http://127.0.0.1:{_s.MOKO_LLM_PORT}/health", timeout=1)
                if r.status_code != 200:
                    # Server belum siap — tunggu 30 detik lagi
                    self._stop_event.wait(timeout=30)
                    continue
            except Exception:
                # Server offline — jangan trigger startup, tunggu 60 detik
                self._stop_event.wait(timeout=60)
                continue
            
            # Cek RAM — jangan foraging jika RAM hampir penuh
            try:
                import psutil
                mem = psutil.virtual_memory()
                if mem.percent > 85:
                    self._log("⚠️ RAM > 85% — pause foraging 60 detik")
                    self._stop_event.wait(timeout=60)
                    continue
                
                # Cek disk
                disk = psutil.disk_usage("/")
                if disk.free < (3 * 1024**3):  # < 3 GB tersisa
                    self._log("⚠️ Disk < 3GB — pause foraging 300 detik")
                    self._stop_event.wait(timeout=300)
                    continue
            except Exception:
                pass
            
            # Lakukan satu siklus foraging
            self._do_one_foraging_cycle()
            
            # Jeda antar siklus (agar tidak terlalu agresif)
            jeda = random.randint(8, 15)  # 8-15 detik antar artikel
            self._stop_event.wait(timeout=jeda)
        
        self._log("🐝 Forager loop berhenti")
    
    # ── PUBLIC API ──────────────────────────────────────────────────────
    
    def start(self):
        """Mulai forager di background thread."""
        if self._thread and self._thread.is_alive():
            self._log("Forager sudah berjalan")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._forager_loop, 
            daemon=True, 
            name="MokoAutoForager"
        )
        self._thread.start()
        self._log("✅ Forager dimulai (background daemon)")
    
    def stop(self):
        """Hentikan forager."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._save_state()
        self._log("🛑 Forager dihentikan")
    
    def set_idle(self, is_idle: bool):
        """
        Dipanggil oleh UI: 
          set_idle(True)  — user diam, forager boleh aktif
          set_idle(False) — user sedang ketik/nanya, forager pause
        """
        self._is_idle = is_idle
    
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
    
    def get_status(self) -> Dict:
        """Status forager untuk ditampilkan ke user."""
        with self._lock:
            topics_by_priority = sorted(self._topics, key=lambda t: t.score(), reverse=True)
            top5 = topics_by_priority[:5]
        
        return {
            "running": self.is_running(),
            "is_idle": self._is_idle,
            "current_topic": self._current_topic,
            "total_topics": len(self._topics),
            "total_chunks_today": self._total_chunks_today,
            "uptime_hours": (time.time() - self._session_start) / 3600,
            "top_topics": [
                {
                    "topic": t.topic,
                    "domain": t.domain,
                    "priority": t.priority,
                    "cycles": t.cycles_done,
                    "chunks": t.chunks_stored,
                    "source": t.source
                }
                for t in top5
            ]
        }
    
    def get_status_text(self) -> str:
        """Format status sebagai teks untuk chat panel."""
        s = self.get_status()
        lines = [
            "🐝 **MOKO Autonomous Forager Status**",
            f"Status: {'🟢 AKTIF' if s['running'] else '🔴 BERHENTI'}",
            f"Mode: {'😴 PAUSED (user aktif)' if not s['is_idle'] else '🔄 FORAGING'}",
            f"Topik dalam antrian: {s['total_topics']}",
            f"Chunk dikumpulkan hari ini: {s['total_chunks_today']}",
            f"Uptime: {s['uptime_hours']:.1f} jam",
            "",
            "**Topik Prioritas Berikutnya:**",
        ]
        for i, t in enumerate(s['top_topics'], 1):
            src_icon = "🔴" if t['source'] == "failure_feedback" else ("👤" if t['source'] == "user_request" else "📋")
            lines.append(
                f"  {i}. {src_icon} [{t['domain']}] '{t['topic'][:45]}' "
                f"(prio={t['priority']}, {t['chunks']} chunk)"
            )
        if s.get('current_topic'):
            lines.append(f"\n🔄 Sedang foraging: **{s['current_topic']}**")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# SINGLETON
# ════════════════════════════════════════════════════════════════════════

_forager_instance: Optional[MokoAutonomousForager] = None

def get_forager(workspace_dir: str = None) -> MokoAutonomousForager:
    global _forager_instance
    if _forager_instance is None:
        if workspace_dir is None:
            from moko_config import settings
            workspace_dir = str(settings.WORKSPACE_DIR)
        _forager_instance = MokoAutonomousForager(workspace_dir, verbose=True)
    return _forager_instance
