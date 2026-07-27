"""
interview_manager.py — State Machine Multi-Turn Interview
==========================================================
Mengelola proses tanya-jawab antara MOKO dan user untuk
mengumpulkan semua kebutuhan software sebelum generate kode.

State machine:
  IDLE
    → ASKING_TYPE        (tanya: tipe software)
    → ASKING_SUBTYPE     (tanya: sub-tipe / genre)
    → ASKING_MECHANICS   (tanya: fitur / mechanic utama)
    → ASKING_LANGUAGE    (tanya: bahasa pemrograman)
    → ASKING_COMPLEXITY  (tanya: level kompleksitas)
    → ASKING_NOTES       (opsional: catatan tambahan)
    → COMPLETE           (semua data terkumpul)
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

from moko_agents.software_builder.models import InterviewData


class InterviewState(Enum):
    IDLE            = auto()
    ASKING_TYPE     = auto()
    ASKING_SUBTYPE  = auto()
    ASKING_MECHANICS = auto()
    ASKING_LANGUAGE  = auto()
    ASKING_COMPLEXITY = auto()
    ASKING_NOTES    = auto()
    COMPLETE        = auto()


# Pertanyaan untuk setiap state
_QUESTIONS = {
    InterviewState.ASKING_TYPE: (
        "🎯 **MOKO Software Builder — Interview Mode**\n\n"
        "Saya akan membantu membangun software kamu dari nol! "
        "Pertama, apa **tipe software** yang ingin dibuat?\n\n"
        "Contoh:\n"
        "  • `game` — game 2D/3D\n"
        "  • `web app` — aplikasi web\n"
        "  • `tool` — utilitas/tools\n"
        "  • `automation` — script automasi\n\n"
        "Ketik tipe software yang kamu inginkan:"
    ),
    InterviewState.ASKING_SUBTYPE: (
        "🎮 Oke! Sekarang, lebih spesifik lagi — apa **genre/sub-tipe**-nya?\n\n"
        "Contoh untuk game: `RPG`, `platformer`, `puzzle`, `top-down shooter`, `idle game`\n"
        "Contoh untuk web: `e-commerce`, `dashboard`, `blog`, `chat app`\n\n"
        "Ketik sub-tipe:"
    ),
    InterviewState.ASKING_MECHANICS: (
        "⚙️ Bagus! Sekarang ceritakan **fitur/mechanic utama** yang harus ada.\n\n"
        "Pisahkan dengan koma jika lebih dari satu.\n"
        "Contoh game: `player movement, collision, scoring, enemy AI, power-up`\n"
        "Contoh web: `login/auth, CRUD data, real-time update, dashboard chart`\n\n"
        "Ketik fitur-fitur yang diinginkan:"
    ),
    InterviewState.ASKING_LANGUAGE: (
        "💻 Pilih **bahasa pemrograman** yang akan digunakan.\n\n"
        "Rekomendasi:\n"
        "  • Game 2D: `python` (pygame) atau `javascript` (Phaser)\n"
        "  • Web: `python` (Flask/FastAPI) atau `javascript` (Node.js)\n"
        "  • Tool: `python` atau `bash`\n\n"
        "Ketik bahasa pilihan kamu:"
    ),
    InterviewState.ASKING_COMPLEXITY: (
        "📊 Terakhir, seberapa **kompleks** proyek ini?\n\n"
        "  • `simple` — fitur minimal, selesai dalam 1-2 jam\n"
        "  • `medium` — fitur lengkap, butuh beberapa jam\n"
        "  • `advanced` — proyek penuh dengan arsitektur yang matang\n\n"
        "Ketik level kompleksitas:"
    ),
    InterviewState.ASKING_NOTES: (
        "📝 Ada **catatan tambahan** yang ingin disampaikan? "
        "(teknologi spesifik, style, referensi, dll.)\n\n"
        "Ketik catatan, atau ketik `skip` jika tidak ada:"
    ),
}

# Kata kunci untuk deteksi tipe software
_SOFTWARE_KEYWORDS = {
    "game": ["game", "games", "gaming", "buat game", "bikin game", "game 2d", "game 3d"],
    "web app": ["web", "website", "webapp", "aplikasi web", "web app", "api", "rest api"],
    "tool": ["tool", "tools", "utilitas", "utility", "script", "program", "aplikasi"],
    "automation": ["automation", "automasi", "auto", "bot", "otomatis"],
}

# Kompleksitas yang valid
_COMPLEXITY_MAP = {
    "simple": "simple", "sederhana": "simple", "mudah": "simple", "1": "simple",
    "medium": "medium", "menengah": "medium", "sedang": "medium", "2": "medium",
    "advanced": "advanced", "kompleks": "advanced", "advance": "advanced",
    "sulit": "advanced", "hard": "advanced", "3": "advanced",
}


class InterviewManager:
    """
    State machine yang mengelola multi-turn interview untuk Software Builder.
    
    Setiap instance mewakili satu sesi interview.
    Gunakan process_answer() untuk memproses jawaban user dan mendapatkan
    pertanyaan berikutnya, atau None jika interview selesai.
    """

    def __init__(self):
        self._state: InterviewState = InterviewState.IDLE
        self._data: InterviewData = InterviewData()

    @property
    def state(self) -> InterviewState:
        return self._state

    @property
    def data(self) -> InterviewData:
        return self._data

    @property
    def is_active(self) -> bool:
        return self._state not in (InterviewState.IDLE, InterviewState.COMPLETE)

    @property
    def is_complete(self) -> bool:
        return self._state == InterviewState.COMPLETE

    def start(self, initial_hint: str = "") -> str:
        """
        Mulai interview. Terima hints dari perintah /coding awal user.
        Return pertanyaan pertama.
        """
        self._state = InterviewState.ASKING_TYPE
        self._data = InterviewData()

        # Jika hint sudah mengandung tipe software, skip langkah pertama
        if initial_hint:
            detected_type = self._detect_software_type(initial_hint)
            if detected_type:
                self._data.software_type = detected_type
                self._state = InterviewState.ASKING_SUBTYPE
                return (
                    f"🔍 Aku deteksi kamu ingin membuat **{detected_type}**!\n\n"
                    + _QUESTIONS[InterviewState.ASKING_SUBTYPE]
                )

        return _QUESTIONS[InterviewState.ASKING_TYPE]

    def reset(self):
        """Reset state machine ke IDLE."""
        self._state = InterviewState.IDLE
        self._data = InterviewData()

    def process_answer(self, user_text: str) -> Tuple[Optional[str], bool]:
        """
        Proses jawaban user dan advance ke state berikutnya.
        
        Returns:
            (next_question, is_complete)
            - next_question: str pertanyaan berikutnya, atau None jika selesai
            - is_complete: True jika interview selesai
        """
        if self._state == InterviewState.IDLE:
            return None, False

        if self._state == InterviewState.COMPLETE:
            return None, True

        text = user_text.strip()

        # Parsing berdasarkan state saat ini
        if self._state == InterviewState.ASKING_TYPE:
            parsed = self._parse_software_type(text)
            self._data.software_type = parsed
            self._state = InterviewState.ASKING_SUBTYPE
            return _QUESTIONS[InterviewState.ASKING_SUBTYPE], False

        elif self._state == InterviewState.ASKING_SUBTYPE:
            self._data.sub_type = text[:80]  # Batasi panjang
            self._state = InterviewState.ASKING_MECHANICS
            return _QUESTIONS[InterviewState.ASKING_MECHANICS], False

        elif self._state == InterviewState.ASKING_MECHANICS:
            mechanics = [m.strip() for m in text.replace("，", ",").split(",") if m.strip()]
            if not mechanics:
                mechanics = [text[:50]]
            self._data.mechanics = mechanics[:10]  # Maks 10 mechanic
            self._state = InterviewState.ASKING_LANGUAGE
            return _QUESTIONS[InterviewState.ASKING_LANGUAGE], False

        elif self._state == InterviewState.ASKING_LANGUAGE:
            lang = self._parse_language(text)
            self._data.language = lang
            # Infer platform dari language
            self._data.platform = self._infer_platform(lang, self._data.software_type)
            self._state = InterviewState.ASKING_COMPLEXITY
            return _QUESTIONS[InterviewState.ASKING_COMPLEXITY], False

        elif self._state == InterviewState.ASKING_COMPLEXITY:
            complexity = self._parse_complexity(text)
            self._data.complexity = complexity
            self._state = InterviewState.ASKING_NOTES
            return _QUESTIONS[InterviewState.ASKING_NOTES], False

        elif self._state == InterviewState.ASKING_NOTES:
            if text.lower() not in ("skip", "tidak", "no", "n", "-", ""):
                self._data.extra_notes = text[:200]
            self._state = InterviewState.COMPLETE
            return None, True

        return None, False

    def get_current_question(self) -> Optional[str]:
        """Dapatkan pertanyaan untuk state saat ini."""
        return _QUESTIONS.get(self._state)

    # ─── Helper parsers ──────────────────────────────────────────────────────

    def _detect_software_type(self, text: str) -> str:
        """Deteksi tipe software dari teks hints (tidak case-sensitive)."""
        text_lower = text.lower()
        for stype, keywords in _SOFTWARE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return stype
        return ""

    def _parse_software_type(self, text: str) -> str:
        """Parse tipe software dari jawaban user."""
        detected = self._detect_software_type(text)
        if detected:
            return detected
        # Fallback: ambil kata pertama
        return text.strip().lower().split()[0][:30] if text.strip() else "software"

    def _parse_language(self, text: str) -> str:
        """Normalize bahasa pemrograman dari jawaban user."""
        text_lower = text.strip().lower()
        lang_map = {
            "python": "python", "py": "python",
            "javascript": "javascript", "js": "javascript", "node": "javascript", "nodejs": "javascript",
            "c++": "c++", "cpp": "c++", "c plus plus": "c++",
            "c#": "c#", "csharp": "c#",
            "java": "java",
            "go": "go", "golang": "go",
            "rust": "rust",
            "typescript": "typescript", "ts": "typescript",
            "bash": "bash", "shell": "bash",
            "html": "html/js", "html/css": "html/js",
            "php": "php",
        }
        for key, val in lang_map.items():
            if key in text_lower:
                return val
        return text_lower.split()[0][:20] if text_lower else "python"

    def _parse_complexity(self, text: str) -> str:
        """Parse kompleksitas dari jawaban user."""
        text_lower = text.strip().lower()
        return _COMPLEXITY_MAP.get(text_lower, "medium")

    def _infer_platform(self, language: str, software_type: str) -> str:
        """Infer platform dari bahasa dan tipe software."""
        if "web" in software_type or language in ("javascript", "html/js", "php", "typescript"):
            return "web"
        if language in ("python", "c++", "c#", "java", "go", "rust"):
            return "desktop"
        return "desktop"


# Singleton factory per session_id
_interview_sessions: dict = {}


def get_interview_manager(session_id: str) -> InterviewManager:
    """Dapatkan atau buat InterviewManager untuk session_id tertentu."""
    if session_id not in _interview_sessions:
        _interview_sessions[session_id] = InterviewManager()
    return _interview_sessions[session_id]


def clear_interview_session(session_id: str):
    """Hapus sesi interview dari registry."""
    _interview_sessions.pop(session_id, None)


if __name__ == "__main__":
    # Unit test untuk validasi state machine transitions
    print("=== Unit Test: interview_manager.py ===\n")

    mgr = InterviewManager()
    assert mgr.state == InterviewState.IDLE
    assert not mgr.is_active
    print("✅ Initial state = IDLE → OK")

    # Start interview tanpa hint
    q1 = mgr.start()
    assert mgr.state == InterviewState.ASKING_TYPE
    assert mgr.is_active
    print(f"✅ After start(): state = ASKING_TYPE → OK")
    print(f"   Q: {q1[:60]}...")

    # Jawab tipe software
    q2, done = mgr.process_answer("game")
    assert not done
    assert mgr.state == InterviewState.ASKING_SUBTYPE
    assert mgr.data.software_type == "game"
    print(f"✅ Jawab 'game': state = ASKING_SUBTYPE, software_type='game' → OK")

    # Jawab sub-tipe
    q3, done = mgr.process_answer("platformer 2D")
    assert not done
    assert mgr.state == InterviewState.ASKING_MECHANICS
    assert mgr.data.sub_type == "platformer 2D"
    print(f"✅ Jawab sub-tipe: state = ASKING_MECHANICS → OK")

    # Jawab mechanics
    q4, done = mgr.process_answer("player movement, collision detection, scoring system, enemy AI")
    assert not done
    assert mgr.state == InterviewState.ASKING_LANGUAGE
    assert len(mgr.data.mechanics) == 4
    print(f"✅ Jawab mechanics: {mgr.data.mechanics} → OK")

    # Jawab language
    q5, done = mgr.process_answer("python")
    assert not done
    assert mgr.state == InterviewState.ASKING_COMPLEXITY
    assert mgr.data.language == "python"
    assert mgr.data.platform == "desktop"
    print(f"✅ Jawab language: language='python', platform='desktop' → OK")

    # Jawab complexity
    q6, done = mgr.process_answer("medium")
    assert not done
    assert mgr.state == InterviewState.ASKING_NOTES
    assert mgr.data.complexity == "medium"
    print(f"✅ Jawab complexity: 'medium' → OK")

    # Skip notes
    final_q, done = mgr.process_answer("skip")
    assert done
    assert mgr.state == InterviewState.COMPLETE
    assert mgr.is_complete
    assert mgr.data.is_complete()
    print(f"✅ Skip notes: state = COMPLETE, data.is_complete() = True → OK")

    # Test start dengan hint
    mgr2 = InterviewManager()
    q_hint = mgr2.start("buat game RPG")
    assert mgr2.state == InterviewState.ASKING_SUBTYPE
    assert mgr2.data.software_type == "game"
    print(f"✅ Start dengan hint 'buat game RPG': auto-detect tipe = 'game' → OK")

    # Test reset
    mgr.reset()
    assert mgr.state == InterviewState.IDLE
    assert not mgr.is_active
    print(f"✅ Reset: state = IDLE → OK")

    print("\n✅ Semua unit test state machine berhasil!")
