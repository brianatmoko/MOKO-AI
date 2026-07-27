"""
MOKO Neural Surgery: NeuralWorkingMemory
=========================================
"Memori Kerja Neural" — buffer aktif yang menyimpan konteks sesi saat ini
secara persisten, sehingga LLM tidak perlu 'berpikir dari awal' setiap kali
user mengirim pesan baru dalam sesi yang sama.

Analogi biologis: Ini seperti Working Memory manusia yang menyimpan
"apa yang baru saja kita bicarakan" agar bisa dijadikan konteks langsung.

Fitur Utama:
  1. Sliding window 8 topik terpanas saat ini (bukan seluruh riwayat)
  2. Semantic importance weighting — topik yang lebih penting (score tinggi) tetap
  3. Cross-turn concept linking — mendeteksi konsep yang berulang antar pesan
  4. LLM Priming Context — output siap pakai untuk inject ke system prompt LLM
"""

import re
import time
import json
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ── Konfigurasi ───────────────────────────────────────────────────────────────
MAX_WORKING_MEMORY_SLOTS = 8     # Maksimal 8 topik aktif dalam WM
CONCEPT_SIMILARITY_THRESHOLD = 0.75  # Di atas ini = konsep sama, gabungkan
CONCEPT_DECAY_SECONDS = 1800    # 30 menit → konsep mulai melemah
CONTEXT_MAX_CHARS = 1200        # Batas total karakter untuk LLM priming


class WorkingMemorySlot:
    """Satu slot memori kerja — satu topik/konsep aktif."""

    def __init__(
        self,
        concept_key: str,
        text: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5
    ):
        self.concept_key = concept_key
        self.text = text
        self.embedding = embedding or []
        self.importance = importance     # 0.0 - 1.0
        self.access_count = 1
        self.created_at = time.time()
        self.last_accessed = time.time()

    def touch(self, new_text: str = "", boost: float = 0.1):
        """Perbarui slot saat konsep ini diakses lagi."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.importance = min(1.0, self.importance + boost)
        if new_text:
            # Gabungkan teks lama + baru secara singkat
            combined = f"{self.text}\n[Update]: {new_text}"
            self.text = combined[:600]  # Batasi agar tidak membesar

    def decay_score(self) -> float:
        """Skor kepentingan setelah decay temporal."""
        age_seconds = time.time() - self.last_accessed
        decay_factor = max(0.1, 1.0 - (age_seconds / CONCEPT_DECAY_SECONDS))
        return self.importance * decay_factor * (1 + 0.05 * self.access_count)

    def to_dict(self) -> Dict:
        return {
            "concept_key": self.concept_key,
            "text": self.text,
            "embedding": self.embedding[:20] if self.embedding else [],  # Simpan sebagian kecil
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed
        }


class NeuralWorkingMemory:
    """
    Memori Kerja Neural yang persisten per sesi.

    Berbeda dari conv_buffer (yang menyimpan riwayat percakapan mentah),
    NeuralWorkingMemory menyimpan KONSEP-KONSEP AKTIF yang sedang dibicarakan,
    terstruktur dan siap di-inject ke LLM sebagai priming context.
    """

    def __init__(self):
        self.slots: Dict[str, WorkingMemorySlot] = {}  # concept_key → slot
        self._session_start = time.time()
        self._turn_count = 0

        # Lacak konsep yang berulang antar sesi
        self._repeated_concepts: Dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def absorb(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        source: str = "user"  # "user" | "moko" | "omni"
    ):
        """
        Serap teks/konsep baru ke dalam working memory.
        Jika konsep serupa sudah ada → update. Jika baru → tambahkan.
        Jika slot penuh → singkirkan yang paling tidak penting.
        """
        self._turn_count += 1

        # Ekstrak key konsep dari teks
        concept_key = self._extract_concept_key(text)

        # Cek apakah sudah ada slot dengan konsep serupa
        existing_key = self._find_similar_slot(embedding) if embedding else None

        if existing_key:
            # Update slot yang sudah ada
            self.slots[existing_key].touch(new_text=text[:200], boost=0.15)
            return

        # Slot baru
        new_slot = WorkingMemorySlot(
            concept_key=concept_key,
            text=text[:500],
            embedding=embedding[:50] if embedding and len(embedding) >= 50 else [],
            importance=importance
        )

        self.slots[concept_key] = new_slot

        # Lacak pengulangan konsep
        self._repeated_concepts[concept_key] = self._repeated_concepts.get(concept_key, 0) + 1

        # Jika slot melebihi MAX → buang yang paling lemah
        self._prune_if_needed()

    def get_priming_context(self, max_chars: int = CONTEXT_MAX_CHARS) -> str:
        """
        Hasilkan teks priming untuk LLM — snapshot konsep-konsep aktif saat ini.

        Digunakan sebagai tambahan pada system prompt agar LLM 'ingat'
        apa yang sedang dibicarakan tanpa perlu membaca ulang seluruh riwayat.
        """
        if not self.slots:
            return ""

        # Urutkan slot berdasarkan decay score (paling relevan duluan)
        sorted_slots = sorted(
            self.slots.values(),
            key=lambda s: s.decay_score(),
            reverse=True
        )

        lines = ["=== MEMORI KERJA AKTIF (Topik-Topik yang Baru Dibahas) ==="]
        total_chars = len(lines[0])

        for slot in sorted_slots:
            if total_chars >= max_chars:
                break

            # Format singkat per konsep
            age_min = int((time.time() - slot.last_accessed) / 60)
            age_str = f"{age_min} menit lalu" if age_min > 0 else "baru saja"
            line = f"• [{slot.concept_key}] ({age_str}, diakses {slot.access_count}x): {slot.text[:200]}"

            if total_chars + len(line) <= max_chars:
                lines.append(line)
                total_chars += len(line)

        if len(lines) <= 1:
            return ""

        lines.append(
            "\nINSTRUKSI: Gunakan konteks di atas sebagai 'ingatan jangka pendek' sesi ini. "
            "Jika user menyebut konsep yang sama, kamu sudah tahu konteksnya."
        )

        return "\n".join(lines)

    def get_active_concepts(self) -> List[str]:
        """Daftar konsep yang sedang aktif (sorted by importance)."""
        return [
            s.concept_key for s in sorted(
                self.slots.values(),
                key=lambda x: x.decay_score(),
                reverse=True
            )
        ]

    def boost_concept(self, concept_key: str, boost: float = 0.2):
        """Tingkatkan importance dari konsep tertentu (saat user menyebut lagi)."""
        if concept_key in self.slots:
            self.slots[concept_key].touch(boost=boost)

    def clear_session(self):
        """Bersihkan working memory saat sesi baru dimulai."""
        self.slots.clear()
        self._turn_count = 0
        self._session_start = time.time()

    def get_stats(self) -> Dict:
        """Statistik working memory untuk monitoring."""
        return {
            "active_slots": len(self.slots),
            "turn_count": self._turn_count,
            "session_age_min": round((time.time() - self._session_start) / 60, 1),
            "top_concept": self.get_active_concepts()[0] if self.slots else "none",
            "repeated_concepts": len([k for k, v in self._repeated_concepts.items() if v > 1])
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _extract_concept_key(self, text: str) -> str:
        """
        Ekstrak kunci konsep ringkas dari teks.
        Digunakan sebagai identifer slot di dictionary.
        """
        # Ambil 3-5 kata pertama yang bermakna
        words = re.findall(r'\b[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]{3,}\b', text.lower())
        stopwords = {
            "yang", "dan", "untuk", "dengan", "ini", "itu", "dari", "pada",
            "adalah", "ada", "jadi", "saya", "aku", "kamu", "moko", "bisa",
            "the", "and", "for", "with", "this", "that", "from"
        }
        meaningful_words = [w for w in words if w not in stopwords]

        if meaningful_words:
            key = "_".join(meaningful_words[:3])
        else:
            key = text[:30].lower().replace(" ", "_")

        return key[:40]  # Batasi panjang key

    def _find_similar_slot(self, embedding: List[float]) -> Optional[str]:
        """
        Cari slot yang memiliki embedding mirip (cosine similarity).
        Menggunakan numpy batch dot product pada partial embedding (50-D) untuk kecepatan.
        """
        if not embedding or not self.slots:
            return None

        import numpy as np
        # Ambil 50-D pertama sebagai "fingerprint" cepat
        query_partial = np.array(embedding[:50], dtype=np.float32)
        q_norm = query_partial / (np.linalg.norm(query_partial) + 1e-9)

        # Kumpulkan semua slot yang punya embedding
        keys = []
        slot_vecs = []
        for key, slot in self.slots.items():
            if slot.embedding and len(slot.embedding) >= 50:
                keys.append(key)
                slot_vecs.append(slot.embedding[:50])

        if not keys:
            return None

        # Hitung cosine similarity batch (satu operasi numpy)
        mat = np.array(slot_vecs, dtype=np.float32)          # (N, 50)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat_norm = mat / norms                                # normalize rows
        sims = mat_norm @ q_norm                              # (N,) dot product

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= CONCEPT_SIMILARITY_THRESHOLD:
            return keys[best_idx]

        return None

    def _prune_if_needed(self):
        """Buang slot paling lemah jika melebihi batas MAX_WORKING_MEMORY_SLOTS."""
        if len(self.slots) <= MAX_WORKING_MEMORY_SLOTS:
            return

        # Urutkan dari yang paling lemah
        sorted_slots = sorted(
            self.slots.items(),
            key=lambda kv: kv[1].decay_score()
        )

        # Hapus slot yang paling lemah
        slots_to_remove = len(self.slots) - MAX_WORKING_MEMORY_SLOTS
        for key, _ in sorted_slots[:slots_to_remove]:
            del self.slots[key]

    @staticmethod
    def _cosine_sim(v1: List[float], v2: List[float]) -> float:
        """
        Cosine similarity via numpy (dipertahankan untuk backward compat).
        Lebih cepat dari pure-Python karena operasi BLAS.
        """
        import numpy as np
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ── Singleton (satu per proses, reset otomatis saat restart) ─────────────────
neural_working_memory = NeuralWorkingMemory()
