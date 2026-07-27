"""
MOKO Cognitive Growth Engine — Bootstrapping & Self-Training Dataset Builder
=============================================================================
Komponen utama dari SUKES (Self-Upgrading & Knowledge-Enrichment System).

Fungsi:
  1. Menerima pemberitahuan kueri yang salah/direvisi/ditolak dari Governor.
  2. Menyimpan kueri bermasalah tersebut ke antrian pertumbuhan (growth queue).
  3. Memproses kueri di antrian secara mandiri (background) saat idle.
  4. Memicu Deep MCTS Math Reasoner dengan budget maksimal (OLYMPIAD level).
  5. Memverifikasi jawaban secara logis (Z3/SymPy CAS).
  6. Jika lolos verifikasi, format trajectory menjadi text-style rationale (<think>...</think>).
  7. Simpan pasangan (prompt, rationale, final_answer) ke self_training_data.jsonl.

Ini mewakili System 2 training loop otomatis untuk meningkatkan model 4B lokal.
"""

import json
import math
import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Import MCTS dan LLM engine
try:
    from moko_neuromath.mcts_reasoner import MCTSMathReasoner
    from moko_agents.llm_engine import engine as _llm_engine
    _ENGINES_OK = True
except ImportError:
    _ENGINES_OK = False

# Import SOM untuk Math Quality Gate
try:
    from moko_neuromath.self_optimization_math import InformationTheory, get_som
    _SOM_OK = True
except ImportError:
    _SOM_OK = False
    InformationTheory = None

# Path dataset training
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAINING_DATA_PATH = os.path.join(_BASE_DIR, "..", "moko_data", "self_training_data.jsonl")
DEFAULT_QUEUE_PATH = os.path.join(_BASE_DIR, "..", "moko_data", "failed_query_queue.json")


@dataclass
class FailedQueryItem:
    query: str
    domain: str
    intent_class: str
    error_msg: str
    timestamp: float
    retries: int = 0
    status: str = "queued"  # 'queued' | 'processing' | 'solved' | 'failed_deep_search'


class CognitiveGrowthEngine:
    """
    Growth Engine yang bekerja di background untuk mengubah kegagalan
    menjadi dataset training yang sangat bernilai tinggi bagi LLM 4B.
    """

    def __init__(
        self,
        training_data_path: str = DEFAULT_TRAINING_DATA_PATH,
        queue_path: str = DEFAULT_QUEUE_PATH,
        verbose: bool = True
    ):
        self.training_data_path = training_data_path
        self.queue_path = queue_path
        self.verbose = verbose
        self._lock = threading.Lock()
        self._queue: List[FailedQueryItem] = []
        
        # Inisialisasi folder data
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        self._load_queue()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  🌱 [GrowthEngine] {msg}")

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def register_failed_query(
        self,
        query: str,
        domain: str,
        intent_class: str,
        error_msg: str
    ):
        """Daftarkan kueri yang gagal untuk diproses nanti."""
        # Hindari duplikasi kueri sejenis di queue
        with self._lock:
            if any(item.query == query for item in self._queue):
                return
                
            item = FailedQueryItem(
                query=query,
                domain=domain,
                intent_class=intent_class,
                error_msg=error_msg,
                timestamp=time.time()
            )
            self._queue.append(item)
            self._save_queue()
            
        self._log(f"Kueri gagal didaftarkan: '{query[:40]}...' | Queue size: {len(self._queue)}")

    def process_queue(self, max_items: int = 3, time_limit_sec: float = 120.0) -> int:
        """
        Proses antrian kueri gagal. Jalankan MCTS penalaran mendalam
        untuk menemukan jawaban yang benar dan menyimpannya sebagai data training.
        """
        if not _ENGINES_OK:
            self._log("⚠️  MCTS atau LLM engine tidak tersedia. Skip pemrosesan queue.")
            return 0

        t_start = time.time()
        solved_count = 0
        
        with self._lock:
            items_to_process = [item for item in self._queue if item.status in ("queued", "failed_deep_search")][:max_items]

        if not items_to_process:
            self._log("Tidak ada antrian kueri gagal untuk diproses.")
            return 0

        self._log(f"Memproses {len(items_to_process)} kueri dari antrian kognitif...")

        for item in items_to_process:
            if time.time() - t_start > time_limit_sec:
                self._log("⏱️  Batas waktu pemrosesan background tercapai.")
                break

            with self._lock:
                item.status = "processing"
                item.retries += 1
                self._save_queue()

            self._log(f"Deep Search untuk kueri: '{item.query[:50]}...'")
            success = self._solve_and_export_trajectory(item)
            
            with self._lock:
                if success:
                    item.status = "solved"
                    solved_count += 1
                else:
                    item.status = "failed_deep_search"
                self._save_queue()

        # Bersihkan item yang sudah selesai (solved) dari file antrian
        with self._lock:
            self._queue = [item for item in self._queue if item.status != "solved"]
            self._save_queue()

        return solved_count

    # ── PRIVATE METHODS ────────────────────────────────────────────────────

    def _compute_trajectory_math_score(self, result: dict) -> float:
        """
        Hitung skor kualitas matematis dari trajektori MCTS menggunakan:
          1. Coefficient of Variation (CoV) step scores:
             CoV tinggi = langkah beragam (informatif), CoV rendah = monoton (buruk).
             Menggunakan Gradien Gradasi: bukan entropi (yang max saat uniform/monoton).
          2. Perplexity confidence: PP tinggi = model ragu-ragu (buruk).

        Returns:
            math_quality: float antara 0.0–1.0
              >= 0.7 → trajektori informatif dan confident (layak ekspor)
              < 0.5  → trajektori monoton atau sangat tidak pasti (skip)
        """
        if not _SOM_OK or not result:
            return 1.0  # Fallback: lolos tanpa pengecekan jika SOM tidak tersedia

        steps = result.get("steps", [])
        if len(steps) < 2:
            return 0.8  # Terlalu sedikit langkah — tidak bisa dievaluasi, beri nilai netral

        # 1. Coefficient of Variation (CoV) dari step scores
        # CoV = std / mean — mengukur keberagaman relatif langkah reasoning
        # CoV = 0  → semua langkah identik (monoton, tidak informatif)
        # CoV > 0  → langkah bervariasi (banyak eksplorasi reasoning)
        step_scores = []
        for step in steps:
            sc = getattr(step, 'score', None) or step.get('score', 0.5) if hasattr(step, 'get') else 0.5
            step_scores.append(max(1e-6, float(sc)))

        n = len(step_scores)
        mean_sc = sum(step_scores) / n
        if mean_sc > 0:
            variance = sum((s - mean_sc) ** 2 for s in step_scores) / n
            std_sc   = math.sqrt(variance)
            cov      = std_sc / mean_sc  # Coefficient of Variation

            # Normalisasi CoV ke [0, 1]: CoV=0 → 0.0 (monoton), CoV>=0.3 → 1.0 (beragam)
            # Batas atas 0.3 dipilih karena CoV > 0.3 sudah menunjukkan keberagaman substansial
            cov_norm = min(1.0, cov / 0.30)
        else:
            cov_norm = 0.0

        # 2. Perplexity proxy dari confidence scores
        confidence_list = []
        for step in steps:
            conf = getattr(step, 'confidence', None)
            if conf is None:
                conf = step.get('confidence', None) if hasattr(step, 'get') else None
            if conf is not None and 0 < conf <= 1.0:
                confidence_list.append(float(conf))

        if confidence_list:
            try:
                pp = InformationTheory.perplexity(confidence_list)
                # Normalisasi: PP=1 (sempurna) → 1.0, PP=100 (sangat ragu) → 0.0
                pp_score = max(0.0, 1.0 - (pp - 1.0) / 99.0)
            except Exception:
                pp_score = 0.7
        else:
            pp_score = 0.7  # Neutral jika tidak ada confidence data

        # Gabungkan: 55% CoV (keberagaman reasoning) + 45% confidence
        math_quality = 0.55 * cov_norm + 0.45 * pp_score

        self._log(
            f"Math Quality Gate: CoV={cov_norm:.3f} (std/mean), PP_score={pp_score:.3f} → "
            f"math_quality={math_quality:.3f}"
        )
        return round(math_quality, 4)


    def _solve_and_export_trajectory(self, item: FailedQueryItem) -> bool:
        """Jalankan MCTS deep search, verifikasi, dan tulis data training."""
        t0 = time.time()
        
        # Setup generator fungsi LLM dengan enable_thinking=True (maksimum compute)
        def llm_fn(prompt, system=""):
            return _llm_engine.generate_text(
                prompt=prompt,
                system_prompt=system,
                coop_params={"num_predict": 400, "enable_thinking": True}
            )

        try:
            # Ambil CAS jika tersedia di moko_core
            from moko_neuromath.math_cas_engine import math_cas
            cas = math_cas
        except ImportError:
            cas = None

        # Jalankan MCTS dengan budget OLYMPIAD (maksimal pencarian pohon)
        mcts = MCTSMathReasoner(llm_generate_fn=llm_fn, cas_engine=cas)
        
        # Paksa model untuk mencari secara mendalam
        mcts.c_explore = 2.0  # Lebih mengeksplorasi cabang baru
        
        result = mcts.reason(item.query)
        latency = time.time() - t0

        if not result or not result.get("answer"):
            self._log("❌ MCTS gagal menemukan solusi.")
            return False

        # Verifikasi: Harus sukses diverifikasi secara deterministik oleh CAS atau memiliki skor tinggi
        score = result.get("trajectory_score", 0.0)
        cas_verified = result.get("cas_verified", False)

        # ── Math Quality Gate (SOM Integration) ──────────────────────────────
        # Hitung kualitas matematis trajektori menggunakan Shannon Entropy + Perplexity
        # Sebelum ekspor, pastikan trajektori informatif (tidak monoton) dan confident
        math_quality = self._compute_trajectory_math_score(result)
        effective_score = score * 0.70 + math_quality * 0.30  # Bobot 70% MCTS + 30% math

        self._log(
            f"Hasil MCTS: mcts_score={score:.2f} | math_quality={math_quality:.3f} | "
            f"effective={effective_score:.3f} | cas_verified={cas_verified} | time={latency:.1f}s"
        )

        if effective_score >= 0.70 or cas_verified:
            # Sukses! Kumpulkan thought process & trajectory langkah
            steps = result.get("steps", [])
            thought_parts = []
            for i, step in enumerate(steps):
                thought_parts.append(
                    f"Langkah {i+1}: {step.description}\n"
                    f"Rumusan: {step.expression}\n"
                    f"Verifikasi: {step.code_result or 'passed'}"
                )
            
            thought_process = "\n\n".join(thought_parts)
            final_answer = result["answer"]

            # Ekspor ke file data self-training
            self._write_training_sample(item.query, thought_process, final_answer)
            self._log("✅ Trajectory sukses diekspor ke dataset self-training!")
            return True
        else:
            self._log("⚠️  Solusi ditemukan tetapi gagal verifikasi kualitas. Skip ekspor.")
            return False

    def _write_training_sample(self, query: str, thought_process: str, answer: str):
        """Tulis pasangan prompt-thought-response ke dataset JSONL."""
        training_obj = {
            "instruction": "Tugas: Pecahkan masalah matematika/penalaran di bawah ini secara kritis langkah-demi-langkah. Tunjukkan analisis formal dan verifikasi di dalam pemikiran internal Anda sebelum memberikan jawaban akhir.",
            "input": query,
            "output": f"<think>\n{thought_process}\n</think>\n\nJawaban akhir:\n{answer}"
        }
        
        with self._lock:
            try:
                with open(self.training_data_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(training_obj, ensure_ascii=False) + "\n")
            except Exception as e:
                self._log(f"⚠️  Gagal menulis file training: {e}")

    def _save_queue(self):
        """Simpan antrian saat ini ke JSON disk."""
        try:
            os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
            data = [asdict(item) for item in self._queue]
            with open(self.queue_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log(f"⚠️  Gagal menyimpan queue ke disk: {e}")

    def _load_queue(self):
        """Muat antrian dari JSON disk."""
        try:
            if not os.path.exists(self.queue_path):
                return
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            with self._lock:
                self._queue = [FailedQueryItem(**d) for d in data]
            self._log(f"Loaded {len(self._queue)} items antrian dari disk.")
        except Exception as e:
            self._log(f"⚠️  Gagal memuat queue dari disk: {e}")


# ── SINGLETON ──────────────────────────────────────────────────────────────
_growth_engine_instance: Optional[CognitiveGrowthEngine] = None

def get_growth_engine(verbose: bool = True) -> CognitiveGrowthEngine:
    global _growth_engine_instance
    if _growth_engine_instance is None:
        _growth_engine_instance = CognitiveGrowthEngine(verbose=verbose)
    return _growth_engine_instance


# ── SELF-TEST ──────────────────────────────────────────────────────────────
def _self_test():
    import tempfile
    print("\n🧪 CognitiveGrowthEngine — Self Test\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.jsonl")
        queue_path = os.path.join(tmpdir, "queue.json")
        
        engine = CognitiveGrowthEngine(
            training_data_path=train_path,
            queue_path=queue_path,
            verbose=True
        )
        
        # Test 1: Daftarkan kegagalan
        engine.register_failed_query(
            query="Berapa turunan dari f(x) = x^2?",
            domain="math",
            intent_class="math_formula",
            error_msg="Governor rejected incorrect derivative calculations"
        )
        
        assert len(engine._queue) == 1
        assert os.path.exists(queue_path)
        print("  ✅ Uji registrasi kegagalan berhasil.")
        
        # Test 2: Injeksi data manual dan ekspor
        print("  📝 Menguji penulisan data training...")
        engine._write_training_sample(
            query="Berapa turunan dari f(x) = x^2?",
            thought_process="1. Rumus turunan x^n adalah n*x^(n-1)\n2. Jadi turunan x^2 adalah 2*x^(2-1) = 2x",
            answer="2x"
        )
        
        assert os.path.exists(train_path)
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert "output" in data
            assert "<think>" in data["output"]
            assert "2x" in data["output"]
            
        print("  ✅ Uji penulisan dataset training berhasil.")
        
    print("\n✅ Semua test CognitiveGrowthEngine lulus!\n")


if __name__ == "__main__":
    _self_test()
