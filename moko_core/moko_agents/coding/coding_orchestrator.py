"""
MOKO Coding Orchestrator
========================
Mengkoordinasikan 5 thread agen spesialis pemrograman.
Menerima request, merouting secara cerdas menggunakan CodingRouter,
dan mendelegasikan pemrosesan ke agen yang tepat.

Versi 2.0 — Tambahan:
  - LoopDetector: Mencegah infinite agent loops (terinspirasi Kimi K2 Tool Stabilizer)
  - ActionBudget: Batas keras jumlah aksi per session
  - StateSnapshot: Snapshot state setiap N aksi (recovery jika stuck)
  - SessionTracker: Lacak histori routing per session
"""

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple

from .coding_router import CodingRouter
from .syntax_agent import SyntaxAgent
from .error_agent import ErrorAgent
from .generator_agent import GeneratorAgent
from .evaluator_agent import EvaluatorAgent
from .system_agent import SystemAgent

logger = logging.getLogger("moko_coding_orchestrator")

# Import singleton Coder Agent jika ada
try:
    from moko_core.moko_agents.moko_coder_1b_agent import coder_agent
except ImportError:
    coder_agent = None

# Import KV Cache Manager untuk state snapshot
try:
    from moko_memory.kv_cache_manager import get_kv_cache
    _kv_cache = get_kv_cache()
except Exception:
    _kv_cache = None


# ── Loop Detector ─────────────────────────────────────────────────────────────

class LoopDetector:
    """
    Detektor infinite loop untuk agent action sequences.
    Terinspirasi dari Kimi K2 Tool Stabilizer.

    Jika N aksi terakhir identik → LOOP_DETECTED.
    Jika total aksi melebihi budget → BUDGET_EXCEEDED.
    """

    STATUS_OK              = "OK"
    STATUS_LOOP_DETECTED   = "LOOP_DETECTED"
    STATUS_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    def __init__(
        self,
        window: int    = 3,    # Jumlah aksi terakhir yang dicek untuk loop
        max_actions: int = 500, # Batas keras total aksi per session
    ):
        self.window      = window
        self.max_actions = max_actions
        self._history:   Deque[str] = deque(maxlen=window * 2)
        self._total:     int = 0

    def check(self, action_hash: str) -> str:
        """
        Tambahkan satu aksi dan periksa statusnya.

        Args:
            action_hash: Hash/fingerprint dari aksi saat ini
                         (misalnya: hash(routed_agent + query[:50]))

        Returns:
            STATUS_OK | STATUS_LOOP_DETECTED | STATUS_BUDGET_EXCEEDED
        """
        self._total += 1
        self._history.append(action_hash)

        if self._total >= self.max_actions:
            logger.warning(
                f"[LoopDetector] ⚠️ Budget habis: {self._total}/{self.max_actions} aksi."
            )
            return self.STATUS_BUDGET_EXCEEDED

        if len(self._history) >= self.window:
            recent = list(self._history)[-self.window:]
            if len(set(recent)) == 1:
                logger.warning(
                    f"[LoopDetector] 🔁 Loop terdeteksi! "
                    f"Aksi '{action_hash[:16]}...' berulang {self.window}x berturut."
                )
                return self.STATUS_LOOP_DETECTED

        return self.STATUS_OK

    def reset(self) -> None:
        """Reset state detector untuk session baru."""
        self._history.clear()
        self._total = 0

    @property
    def total_actions(self) -> int:
        return self._total


# ── Action Hash Helper ────────────────────────────────────────────────────────

def _make_action_hash(agent_name: str, query: str, code: str = "") -> str:
    """Buat fingerprint unik dari satu aksi agent."""
    raw = f"{agent_name}|{query[:80]}|{code[:40]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Orchestrator ──────────────────────────────────────────────────────────────

@dataclass
class SessionStats:
    """Statistik satu session orchestrator."""
    session_id:   str
    total_calls:  int = 0
    loop_aborts:  int = 0
    budget_aborts: int = 0
    agent_counts: Dict[str, int] = field(default_factory=dict)
    history:      List[Dict] = field(default_factory=list)


class CodingOrchestrator:
    """
    Orchestrator pusat untuk semua agen coding spesialis.

    Menerapkan paradigma 'tidak semua agen perlu mengetahui semua hal'.
    Dilengkapi LoopDetector dan ActionBudget (Kimi K2 Tool Stabilizer pattern).
    """

    # Snapshot otomatis setiap N aksi sukses
    SNAPSHOT_EVERY_N = 50

    def __init__(
        self,
        main_coder_agent  = None,
        loop_window: int  = 3,
        action_budget: int = 500,
    ):
        # Inisialisasi coder client
        self.coder = main_coder_agent or coder_agent

        # Router dan 5 sub-agents spesialis
        self.router = CodingRouter()
        self.agents = {
            "SYNTAX":   SyntaxAgent(self.coder),
            "ERROR":    ErrorAgent(self.coder),
            "GENERATE": GeneratorAgent(self.coder),
            "EVALUATE": EvaluatorAgent(self.coder),
            "SYSTEM":   SystemAgent(self.coder),
        }

        # Loop Detector + Action Budget
        self._loop_detector = LoopDetector(window=loop_window, max_actions=action_budget)

        # Session tracking
        self._session = SessionStats(session_id=f"sess_{int(time.time())}")
        self._action_count_since_snapshot = 0

        logger.info(
            f"[CodingOrchestrator] Inisialisasi — "
            f"5 agen aktif | Budget {action_budget} aksi | Loop window {loop_window}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(
        self,
        query: str,
        code: str       = "",
        error_msg: str  = "",
        language: str   = "python",
        force_agent: Optional[str] = None,
    ) -> Dict:
        """
        Mengeksekusi pemrosesan query dengan mengarahkan ke agen spesialis.

        Args:
            query:       Pertanyaan atau instruksi natural language.
            code:        Source code yang ingin dianalisis (opsional).
            error_msg:   Pesan error/log crash (opsional).
            language:    Bahasa pemrograman (default: "python").
            force_agent: Paksa penggunaan agen tertentu secara eksplisit.

        Returns:
            Dict berisi hasil eksekusi terstruktur lengkap.
        """
        start_time = time.time()

        # 1. Loop Detection sebelum routing
        action_hash = _make_action_hash(force_agent or query[:30], query, code)
        loop_status = self._loop_detector.check(action_hash)

        if loop_status == LoopDetector.STATUS_LOOP_DETECTED:
            self._session.loop_aborts += 1
            return self._abort_result(
                query, "LOOP_ABORT",
                "Agen terdeteksi dalam loop berulang. "
                "Coba reformulasi pertanyaan atau berikan konteks berbeda.",
                start_time,
            )

        if loop_status == LoopDetector.STATUS_BUDGET_EXCEEDED:
            self._session.budget_aborts += 1
            return self._abort_result(
                query, "BUDGET_ABORT",
                f"Action budget habis ({self._loop_detector.max_actions} aksi). "
                "Mulai session baru untuk melanjutkan.",
                start_time,
            )

        # 2. Routing
        if force_agent and force_agent in self.agents:
            routed_agent = force_agent
            confidence   = 1.0
        else:
            routing_input = f"{query} {error_msg}"
            routed_agent, confidence = self.router.route(routing_input)

        routing_time = (time.time() - start_time) * 1000  # ms

        # 3. Eksekusi Agen Spesifik
        agent_start  = time.time()
        agent_result = {}

        try:
            if routed_agent == "SYNTAX":
                agent_result = self.agents["SYNTAX"].process(code or query, language)
            elif routed_agent == "ERROR":
                agent_result = self.agents["ERROR"].process(
                    code or query, error_msg or query, language
                )
            elif routed_agent == "GENERATE":
                agent_result = self.agents["GENERATE"].process(query, language, context=code)
            elif routed_agent == "EVALUATE":
                agent_result = self.agents["EVALUATE"].process(code or query, language)
            elif routed_agent == "SYSTEM":
                agent_result = self.agents["SYSTEM"].process(code or query, language)
        except Exception as e:
            logger.error(f"[CodingOrchestrator] Error pada agen {routed_agent}: {e}")
            agent_result = {
                "error":        str(e),
                "repaired_code": code,
                "explanation":  "Terjadi kegagalan internal pada sub-agent.",
            }

        agent_time = (time.time() - agent_start) * 1000   # ms
        total_time = (time.time() - start_time)  * 1000   # ms

        # 4. Update session stats
        self._session.total_calls += 1
        self._session.agent_counts[routed_agent] = (
            self._session.agent_counts.get(routed_agent, 0) + 1
        )
        self._session.history.append({
            "ts":    time.time(),
            "agent": routed_agent,
            "query": query[:60],
        })

        # 5. Snapshot state secara berkala (Kimi State Snapshot pattern)
        self._action_count_since_snapshot += 1
        if (
            self._action_count_since_snapshot >= self.SNAPSHOT_EVERY_N
            and _kv_cache is not None
        ):
            self._take_snapshot()
            self._action_count_since_snapshot = 0

        # 6. Format Output Gabungan
        return {
            "routed_agent":       routed_agent,
            "routing_confidence": round(confidence, 2),
            "routing_time_ms":    round(routing_time, 2),
            "execution_time_ms":  round(agent_time, 2),
            "total_time_ms":      round(total_time, 2),
            "loop_status":        loop_status,
            "session_actions":    self._loop_detector.total_actions,
            "result":             agent_result,
        }

    def new_session(self) -> str:
        """
        Mulai session baru — reset LoopDetector dan SessionStats.
        Mengembalikan session ID baru.
        """
        self._loop_detector.reset()
        new_id = f"sess_{int(time.time())}"
        self._session = SessionStats(session_id=new_id)
        self._action_count_since_snapshot = 0
        logger.info(f"[CodingOrchestrator] Session baru dimulai: {new_id}")
        return new_id

    def get_session_stats(self) -> dict:
        """Kembalikan statistik session saat ini."""
        return {
            "session_id":     self._session.session_id,
            "total_calls":    self._session.total_calls,
            "loop_aborts":    self._session.loop_aborts,
            "budget_aborts":  self._session.budget_aborts,
            "actions_left":   (
                self._loop_detector.max_actions
                - self._loop_detector.total_actions
            ),
            "agent_distribution": self._session.agent_counts,
            "recent_history": self._session.history[-10:],
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _abort_result(
        self,
        query: str,
        abort_type: str,
        message: str,
        start_time: float,
    ) -> Dict:
        total_time = (time.time() - start_time) * 1000
        logger.warning(f"[CodingOrchestrator] {abort_type}: {message[:80]}")
        return {
            "routed_agent":       "ABORT",
            "routing_confidence": 0.0,
            "routing_time_ms":    round(total_time, 2),
            "execution_time_ms":  0.0,
            "total_time_ms":      round(total_time, 2),
            "loop_status":        abort_type,
            "session_actions":    self._loop_detector.total_actions,
            "result": {
                "error":       abort_type,
                "explanation": message,
            },
        }

    def _take_snapshot(self) -> None:
        """Snapshot state orchestrator ke KV Cache Manager (SSD tier)."""
        try:
            state = {
                "session_id":  self._session.session_id,
                "total_calls": self._session.total_calls,
                "agent_counts": self._session.agent_counts,
                "history_last_10": self._session.history[-10:],
            }
            snap_path = _kv_cache.snapshot_state(
                f"orchestrator_{self._session.session_id}", state
            )
            logger.debug(f"[CodingOrchestrator] Snapshot disimpan: {snap_path}")
        except Exception as e:
            logger.debug(f"[CodingOrchestrator] Snapshot gagal (non-critical): {e}")
