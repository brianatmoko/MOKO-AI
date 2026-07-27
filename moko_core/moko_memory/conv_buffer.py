"""
MOKO Conversation Buffer — Riwayat Percakapan dengan Vector Cache
=================================================================
Setiap giliran percakapan disimpan dalam dua lapisan:

1. TEKS (tampilan): Disimpan sebagai string biasa untuk ditampilkan ke user
2. VEKTOR QEV (biner): Setiap giliran juga dikodekan sebagai vektor 768-D
   lalu dikompres ke QEV 384-byte menggunakan RSA Encoder yang sama.

Saat user mengirim pesan baru:
  - Buffer mencari giliran percakapan PALING RELEVAN (bukan hanya yang terbaru)
    menggunakan dot-product pada QEV, bukan string matching.
  - Hasilnya diinjeksikan ke system prompt sebagai "memori sesi".
  - TokenStreamPager (T0) membatasi injeksi ke active window budget.

Keunggulan:
  - Pencarian konteks relevan = O(N) dot product 384-byte, bukan NLP parsing
  - Percakapan "lanjutkan ini" akan menemukan topik yang benar walau diutarakan berbeda
  - RAM footprint kecil: setiap giliran hanya 384 byte + teks aslinya
  - Disk stream unlimited via TokenStreamPager
"""

import struct
import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from moko_config import settings

DEFAULT_CHAT_SESSION = "moko_chat"

_CONTEXT_FOOTER = (
    "INSTRUKSI: Gunakan riwayat di atas untuk menjawab dengan konsisten. "
    "Jika user meminta 'lanjutkan' atau 'jelaskan lebih', itu merujuk ke topik terakhir di atas. "
    "Jika user menyebut nama atau fakta pribadi yang pernah diucapkan, kamu WAJIB ingat."
)


@dataclass
class ConvTurn:
    """Satu giliran percakapan (user + moko)."""
    user_text:  str
    moko_text:  str
    timestamp:  float
    user_qev:   bytes   # 384 byte — vektor biner user message
    moko_qev:   bytes   # 384 byte — vektor biner moko response
    user_fp32:  Optional[List[float]] = field(default=None)  # Raw embedding (untuk HDC)
    moko_fp32:  Optional[List[float]] = field(default=None)  # Raw embedding (untuk HDC)


class ConversationBuffer:
    """
    Ring buffer percakapan sesi ini dengan pencarian semantik via QEV.
    
    Cara pakai:
        buffer = ConversationBuffer(max_turns=20)
        buffer.add_turn(user_text, moko_text, user_vec, moko_vec)
        
        context = buffer.get_relevant_context(query_vec, top_k=3)
        # → Mengembalikan 3 giliran percakapan yang paling relevan dengan query saat ini
    """
    
    def __init__(self, max_turns: int = 20, session_id: str = DEFAULT_CHAT_SESSION):
        self.max_turns = max_turns
        self.session_id = session_id
        self.turns: List[ConvTurn] = []
        self._encoder = None  # Lazy init
        self._token_pager = self._make_token_pager()

    def _token_stream_enabled(self) -> bool:
        return getattr(settings, "TOKEN_STREAM_ENABLED", True)

    def _make_token_pager(self):
        from moko_marathon.token_stream_pager import TokenStreamPager

        return TokenStreamPager(
            session_id=self.session_id,
            stream_kind="chat",
            reserve_tokens=getattr(settings, "TOKEN_STREAM_RESERVE_TOKENS", 512),
            recent_detail_count=getattr(settings, "TOKEN_STREAM_RECENT_TURNS", 2),
        )

    def _sync_turn_to_pager(self, user_text: str, moko_text: str) -> None:
        if not self._token_stream_enabled():
            return
        user_clean = user_text.strip()
        moko_clean = moko_text.strip()
        if getattr(settings, "SEMANTIC_COMPRESSOR_V2", True):
            from moko_marathon.semantic_compressor import semantic_compressor

            pair_summary = semantic_compressor.compress_turn(user_clean, moko_clean)
        else:
            pair_summary = f"User: {user_clean[:80]} | MOKO: {moko_clean[:120]}"
        self._token_pager.append("user", user_clean)
        self._token_pager.append("assistant", moko_clean, compressed=pair_summary)

    def _hydrate_pager_from_turns(self) -> None:
        if not self._token_stream_enabled():
            return
        self._token_pager.clear()
        for turn in self.turns:
            self._sync_turn_to_pager(turn.user_text, turn.moko_text)

    def _get_encoder(self):
        """Lazy load CryptoHashEncoder untuk kompresi FP16."""
        if self._encoder is None:
            try:
                from moko_memory.crypto_hash_encoder import get_crypto_encoder
                self._encoder = get_crypto_encoder()
            except Exception:
                self._encoder = None
        return self._encoder

    def _encode_to_qev(self, fp32_vec: List[float]) -> bytes:
        """Kompresi vektor 768-D float32 → 1536 byte FP16 biner (menggantikan QEV)."""
        enc = self._get_encoder()
        if enc is None:
            return b""
        try:
            addr = enc.encode("", fp32_vec)
            return addr.fp16_vector
        except Exception:
            return b""

    def _qev_dot_product(self, q: bytes, db: bytes) -> float:
        """
        Hitung kemiripan antara dua FP16 vectors.
        Hasil: Cosine similarity [-1.0, 1.0]
        """
        if not q or not db:
            return 0.0
        try:
            return self._get_encoder().cosine_similarity_fp16(q, db)
        except Exception:
            return 0.0

    def add_turn(
        self,
        user_text: str,
        moko_text: str,
        user_fp32: Optional[List[float]] = None,
        moko_fp32: Optional[List[float]] = None
    ):
        """
        Tambahkan satu giliran percakapan ke buffer.
        
        user_fp32/moko_fp32: Vektor embedding 768-D. Jika None, akan dikomputasi di sini
                             (butuh Ollama embedder aktif).
        """
        # Kompresi ke QEV biner
        user_qev = self._encode_to_qev(user_fp32) if user_fp32 else b""
        moko_qev = self._encode_to_qev(moko_fp32) if moko_fp32 else b""

        turn = ConvTurn(
            user_text=user_text,
            moko_text=moko_text,
            timestamp=time.time(),
            user_qev=user_qev,
            moko_qev=moko_qev,
            user_fp32=user_fp32,   # Simpan raw embedding untuk HDC
            moko_fp32=moko_fp32,   # Simpan raw embedding untuk HDC
        )
        
        self.turns.append(turn)
        
        # Ring buffer: buang giliran tertua jika melebihi batas
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        self._sync_turn_to_pager(user_text, moko_text)

    def _select_relevant_turn_indices(
        self,
        query_fp32: List[float],
        top_k: int = 3,
    ) -> List[int]:
        if not self.turns:
            return []

        query_qev = self._encode_to_qev(query_fp32)
        scored: List[Tuple[float, int]] = []

        for idx, turn in enumerate(self.turns):
            score_u = self._qev_dot_product(query_qev, turn.user_qev)
            score_m = self._qev_dot_product(query_qev, turn.moko_qev)
            combined = 0.6 * score_u + 0.4 * score_m
            scored.append((combined, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:top_k]

        if not selected or selected[0][0] < 0.05:
            selected_indices = [len(self.turns) - 2, len(self.turns) - 1]
            selected_indices = [i for i in selected_indices if i >= 0]
        else:
            selected_indices = [idx for _, idx in selected]

        intro_markers = ["nama saya", "nama aku", "namaku", "kenalkan", "kenalin"]
        for idx, turn in enumerate(self.turns):
            if any(marker in turn.user_text.lower() for marker in intro_markers):
                if idx not in selected_indices:
                    selected_indices.append(idx)

        selected_indices.sort()
        return selected_indices

    def get_relevant_context(
        self,
        query_fp32: List[float],
        top_k: int = 3
    ) -> str:
        """
        Cari giliran percakapan yang paling relevan dengan query saat ini.
        
        Menggunakan QEV dot-product — bukan string matching.
        Mengembalikan teks context yang siap diinjeksikan ke system prompt.
        """
        if not self.turns:
            return ""

        selected_indices = self._select_relevant_turn_indices(query_fp32, top_k)

        if not self._token_stream_enabled():
            selected_turns = [self.turns[idx] for idx in selected_indices]
            return self._format_context(selected_turns)

        recent_n = max(1, self._token_pager.recent_detail_count)
        recent_indices = set(range(max(0, len(self.turns) - recent_n), len(self.turns)))
        extra_indices = [i for i in selected_indices if i not in recent_indices]
        extra_turns = [self.turns[i] for i in extra_indices]

        semantic_block = ""
        if extra_turns:
            lines = []
            for t in extra_turns:
                lines.append(f"User: {t.user_text[:200]}")
                lines.append(f"MOKO: {t.moko_text[:400]}")
                lines.append("")
            semantic_block = "\n".join(lines).strip()

        paged = self._token_pager.build_active_context(
            goal="=== RIWAYAT PERCAKAPAN SESI INI ===",
            retrieved_context=semantic_block,
        )
        if not paged:
            selected_turns = [self.turns[idx] for idx in selected_indices]
            return self._format_context(selected_turns)

        return f"{paged}\n\n{_CONTEXT_FOOTER}"

    def _format_context(self, turns: List[ConvTurn]) -> str:
        """Format giliran percakapan sebagai blok konteks untuk system prompt."""
        if not turns:
            return ""
        
        lines = ["=== RIWAYAT PERCAKAPAN SESI INI ==="]
        for t in turns:
            user_preview = t.user_text[:200]
            moko_preview = t.moko_text[:400]
            lines.append(f"User: {user_preview}")
            lines.append(f"MOKO: {moko_preview}")
            lines.append("")
        lines.append(_CONTEXT_FOOTER)
        
        return "\n".join(lines)

    def get_last_turns_text(self, n: int = 3) -> str:
        """Ambil N giliran terakhir sebagai teks untuk fallback."""
        if not self.turns:
            return ""
        last = self.turns[-n:]
        return self._format_context(last)

    def load_from_store(self, session_store, n: int = 20):
        """
        Isi buffer RAM dari PersistentSessionStore saat startup.
        Memungkinkan MOKO mengingat percakapan dari sesi sebelumnya.
        
        Args:
            session_store: Instance dari moko_memory.session_store.SessionStore
            n: Jumlah giliran terakhir yang dimuat ke RAM buffer
        """
        recent = session_store.load_recent(n)
        if not recent:
            return
        
        self.turns.clear()
        for record in recent:
            # Buat ConvTurn dari record disk (tanpa QEV karena tidak disimpan)
            turn = ConvTurn(
                user_text=record.get("user", ""),
                moko_text=record.get("moko", ""),
                timestamp=record.get("ts", 0.0),
                user_qev=b"",  # QEV tidak disimpan ke disk, akan di-encode saat dibutuhkan
                moko_qev=b""
            )
            self.turns.append(turn)
        
        # Trim ke max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        self._hydrate_pager_from_turns()
        print(f"[ConvBuffer] Dimuat {len(self.turns)} giliran dari sesi persisten.")

    def get_full_chat_messages(self, n: int = 10) -> list:
        """
        Kembalikan N giliran terakhir dalam format OpenAI messages array.
        Digunakan oleh Auto-Continue Engine untuk mengirim konteks penuh ke LLM.
        
        Return: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
        """
        if self._token_stream_enabled():
            history = self._token_pager.load_history()
            if history:
                messages = []
                for rec in history[-(n * 2):]:
                    role = rec.get("role")
                    content = rec.get("content", "")
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
                if messages:
                    return messages

        if not self.turns:
            return []
        last = self.turns[-n:]
        messages = []
        for t in last:
            messages.append({"role": "user",      "content": t.user_text})
            messages.append({"role": "assistant", "content": t.moko_text})
        return messages

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def get_token_stream_stats(self) -> dict:
        if not self._token_stream_enabled():
            return {"enabled": False}
        stats = self._token_pager.get_stats()
        stats["enabled"] = True
        return stats

    def get_hdc_context_vector(self) -> Optional[np.ndarray]:
        """
        Kompresi seluruh riwayat sesi menjadi satu vektor HDC 2048-dim.
        
        Menggunakan Vector Symbolic Architecture:
          H = Σ_k [ Π^k (Role_k ⊗ Content_k) ]
        
        Vektor ini merepresentasikan "kesan keseluruhan" sesi saat ini dalam
        ruang HDC yang dapat dibandingkan dengan query baru menggunakan
        cosine similarity.
        
        Return: numpy array 2048-dim, atau None jika belum ada riwayat.
        """
        if not self.turns:
            return None
        
        # Build turn dicts untuk HDC compressor
        hdc_turns = []
        for t in self.turns:
            if t.user_fp32:
                hdc_turns.append({"role": "user",      "embedding": t.user_fp32})
            if t.moko_fp32:
                hdc_turns.append({"role": "assistant", "embedding": t.moko_fp32})
        
        if not hdc_turns:
            return None
        
        try:
            from moko_memory.hdc_context import ContextCompressor
            compressor = ContextCompressor()
            return compressor.compress_history(hdc_turns)
        except Exception as e:
            print(f"[ConvBuffer] HDC compression error: {e}")
            return None

    def clear(self):
        """Reset buffer saat sesi baru dimulai."""
        self.turns.clear()
        if self._token_stream_enabled():
            self._token_pager.clear()

    def reload_from_store(self, session_store, n: int = 20):
        """
        Segarkan ulang buffer RAM dari disk — tanpa perlu restart MOKO.
        Berguna setelah session cleanup manual dilakukan.

        Berbeda dengan load_from_store() yang hanya append,
        reload_from_store() membersihkan buffer terlebih dulu lalu re-load.
        """
        self.clear()
        self.load_from_store(session_store, n=n)
        print(f"[ConvBuffer] ✅ Buffer di-reload: {len(self.turns)} giliran dimuat.")


# ── Singleton Session Buffer ──────────────────────────────────────────────────
# Satu buffer per sesi aplikasi. Direset saat aplikasi ditutup.
session_buffer = ConversationBuffer(max_turns=20)
