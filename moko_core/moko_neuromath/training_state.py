"""
MOKO Applied Math Training State — Progress Tracker
====================================================
Persistent storage dan tracking performa AI per domain, per difficulty,
dan per topik untuk sistem pelatihan matematika terapan.

Data disimpan ke: .math_omni/applied_training_state.json
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


# ── PATH ───────────────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MATH_OMNI_DIR = os.path.join(_BASE_DIR, ".math_omni")
_STATE_FILE = os.path.join(_MATH_OMNI_DIR, "applied_training_state.json")


@dataclass
class TrainingRecord:
    """Rekaman satu sesi menjawab satu soal."""
    problem_id: str
    domain: str
    difficulty: str
    story_text: str                  # cuplikan soal
    target_symbol: str
    expected_answer: float
    arms_answer: Optional[float]     # jawaban dari ARMS (None jika gagal)
    formula_source: str              # LOOKUP / DERIVED / SYNTHESIZED / FAIL
    is_correct: bool
    percent_error: float
    score: float                     # 0.0 - 1.0
    elapsed_ms: float
    timestamp: str = ""
    tags: List[str] = field(default_factory=list)
    solve_status: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DomainStats:
    """Statistik performa AI per domain."""
    domain: str
    total_attempted: int = 0
    total_correct: int = 0
    total_score: float = 0.0
    avg_time_ms: float = 0.0
    last_updated: str = ""

    # Per difficulty
    easy_correct: int = 0
    easy_total: int = 0
    medium_correct: int = 0
    medium_total: int = 0
    hard_correct: int = 0
    hard_total: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_correct / self.total_attempted

    @property
    def avg_score(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_score / self.total_attempted

    @property
    def mastery_level(self) -> str:
        acc = self.accuracy
        if self.total_attempted < 3:
            return "untested"
        elif acc >= 0.90:
            return "mastered"
        elif acc >= 0.70:
            return "proficient"
        elif acc >= 0.50:
            return "learning"
        else:
            return "struggling"

    def update_difficulty_stats(self, difficulty: str, is_correct: bool):
        if difficulty == "easy":
            self.easy_total += 1
            if is_correct: self.easy_correct += 1
        elif difficulty == "medium":
            self.medium_total += 1
            if is_correct: self.medium_correct += 1
        elif difficulty == "hard":
            self.hard_total += 1
            if is_correct: self.hard_correct += 1


@dataclass
class TrainingSession:
    """Satu sesi training (bisa multi-soal)."""
    session_id: str
    start_time: str
    end_time: str = ""
    n_problems: int = 0
    n_correct: int = 0
    total_score: float = 0.0
    domains_covered: List[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.n_problems == 0:
            return 0.0
        return self.n_correct / self.n_problems


class TrainingStateManager:
    """
    Mengelola state training secara persisten.
    Simpan & load dari JSON file.
    """

    def __init__(self):
        self.records: List[TrainingRecord] = []
        self.domain_stats: Dict[str, DomainStats] = {}
        self.sessions: List[TrainingSession] = []
        self.total_trained: int = 0
        self.created_at: str = datetime.now().isoformat()
        self._load()

    def _load(self):
        """Load state dari file JSON."""
        if not os.path.exists(_STATE_FILE):
            os.makedirs(_MATH_OMNI_DIR, exist_ok=True)
            return

        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.total_trained = data.get("total_trained", 0)
            self.created_at = data.get("created_at", self.created_at)

            # Load records
            for r in data.get("records", []):
                self.records.append(TrainingRecord(**r))

            # Load domain stats
            for domain, ds in data.get("domain_stats", {}).items():
                self.domain_stats[domain] = DomainStats(**ds)

            # Load sessions
            for s in data.get("sessions", []):
                self.sessions.append(TrainingSession(**s))

        except Exception as e:
            print(f"  [TrainingState] Warning: gagal load state: {e}")

    def save(self):
        """Simpan state ke file JSON."""
        os.makedirs(_MATH_OMNI_DIR, exist_ok=True)
        data = {
            "total_trained": self.total_trained,
            "created_at": self.created_at,
            "last_saved": datetime.now().isoformat(),
            "records": [asdict(r) for r in self.records[-1000:]],  # keep last 1000
            "domain_stats": {d: asdict(s) for d, s in self.domain_stats.items()},
            "sessions": [asdict(s) for s in self.sessions[-100:]],  # keep last 100
        }
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record(self, rec: TrainingRecord):
        """Tambahkan satu record training."""
        self.records.append(rec)
        self.total_trained += 1

        # Update domain stats
        if rec.domain not in self.domain_stats:
            self.domain_stats[rec.domain] = DomainStats(domain=rec.domain)

        ds = self.domain_stats[rec.domain]
        ds.total_attempted += 1
        if rec.is_correct:
            ds.total_correct += 1
        ds.total_score += rec.score
        ds.avg_time_ms = (ds.avg_time_ms * (ds.total_attempted - 1) + rec.elapsed_ms) / ds.total_attempted
        ds.update_difficulty_stats(rec.difficulty, rec.is_correct)
        ds.last_updated = datetime.now().isoformat()

    def get_weaknesses(self) -> List[str]:
        """Return domain yang perlu lebih banyak latihan."""
        weak = []
        for domain, ds in self.domain_stats.items():
            if ds.total_attempted >= 3 and ds.accuracy < 0.60:
                weak.append(domain)
        return sorted(weak, key=lambda d: self.domain_stats[d].accuracy)

    def get_strengths(self) -> List[str]:
        """Return domain yang sudah dikuasai."""
        strong = []
        for domain, ds in self.domain_stats.items():
            if ds.total_attempted >= 3 and ds.accuracy >= 0.85:
                strong.append(domain)
        return strong

    def get_untested_domains(self, all_domains: List[str]) -> List[str]:
        """Return domain yang belum pernah dilatih."""
        tested = set(self.domain_stats.keys())
        return [d for d in all_domains if d not in tested]

    def generate_report(self) -> str:
        """Buat laporan progress yang terformat."""
        lines = []
        lines.append("\n" + "═" * 65)
        lines.append(f"  📊 MOKO APPLIED MATH — TRAINING PROGRESS REPORT")
        lines.append("═" * 65)
        lines.append(f"  Total soal dikerjakan : {self.total_trained}")
        lines.append(f"  Sesi training         : {len(self.sessions)}")

        if self.domain_stats:
            total_correct = sum(ds.total_correct for ds in self.domain_stats.values())
            total_all = sum(ds.total_attempted for ds in self.domain_stats.values())
            overall_acc = total_correct / total_all if total_all > 0 else 0.0
            lines.append(f"  Akurasi keseluruhan   : {overall_acc:.1%}")

        lines.append("")
        lines.append("  PERFORMA PER DOMAIN:")
        lines.append(f"  {'Domain':<22} {'Acc':>6} {'Soal':>6} {'Score':>6} {'Level':<12}")
        lines.append("  " + "-" * 58)

        for domain, ds in sorted(self.domain_stats.items(), key=lambda x: x[1].accuracy, reverse=True):
            bar_fill = int(ds.accuracy * 10)
            bar = "█" * bar_fill + "░" * (10 - bar_fill)
            lines.append(
                f"  {domain:<22} {ds.accuracy:>5.1%} {ds.total_attempted:>6} "
                f"{ds.avg_score:>6.2f} {ds.mastery_level:<12}"
            )

        weaknesses = self.get_weaknesses()
        if weaknesses:
            lines.append("")
            lines.append(f"  ⚠️  AREA YANG PERLU DIPERKUAT: {', '.join(weaknesses)}")

        strengths = self.get_strengths()
        if strengths:
            lines.append(f"  ✅ SUDAH DIKUASAI: {', '.join(strengths)}")

        lines.append("═" * 65)
        return "\n".join(lines)


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_state_manager: Optional[TrainingStateManager] = None

def get_state_manager() -> TrainingStateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = TrainingStateManager()
    return _state_manager
