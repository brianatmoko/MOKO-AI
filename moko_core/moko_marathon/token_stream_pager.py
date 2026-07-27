"""
MOKO Token Stream Pager (Sistem Token Keran — T0)
==================================================
Generalisasi ContextPager untuk semua jalur inferensi (chat, marathon, episodic).

Paradigma MemGPT + SimpleMem:
  - stream_storage  → unlimited jsonl di disk (.moko_cache)
  - active_window   → bounded oleh budget token (VRAM-calibrated via settings)
  - katup (valve)   → older turns compressed, recent turns full detail
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from moko_config import settings


class TokenStreamPager:
    """Pager token stream: simpan unlimited di disk, injeksi bounded ke prompt."""

    CHARS_PER_TOKEN_ESTIMATE = 3.5
    DEFAULT_RESERVE_TOKENS = 512
    DEFAULT_RECENT_DETAIL = 2

    def __init__(
        self,
        session_id: str,
        stream_kind: str = "chat",
        budget_tokens: int | None = None,
        reserve_tokens: int | None = None,
        recent_detail_count: int | None = None,
    ):
        self.session_id = session_id
        self.stream_kind = stream_kind
        self.budget_tokens = budget_tokens or getattr(
            settings, "MAX_CONTEXT_TOKENS", 2048
        )
        self.reserve_tokens = (
            reserve_tokens if reserve_tokens is not None else self.DEFAULT_RESERVE_TOKENS
        )
        self.recent_detail_count = (
            recent_detail_count
            if recent_detail_count is not None
            else self.DEFAULT_RECENT_DETAIL
        )
        self._inject_budget = max(256, self.budget_tokens - self.reserve_tokens)

        cache = Path(settings.CACHE_DIR)
        cache.mkdir(parents=True, exist_ok=True)
        self.stream_path = cache / f"token_stream_{stream_kind}_{session_id}.jsonl"
        self._last_active_tokens = 0

    # ------------------------------------------------------------------ utils

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / cls.CHARS_PER_TOKEN_ESTIMATE))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        est = self.estimate_tokens(text)
        if est <= max_tokens:
            return text
        max_chars = int(max_tokens * self.CHARS_PER_TOKEN_ESTIMATE)
        if max_chars <= 4:
            return text[:max_chars]
        return "...\n" + text[-(max_chars - 4) :]

    def _write_record(self, record: dict[str, Any]) -> None:
        try:
            with open(self.stream_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[TOKEN PAGER] Gagal menulis stream: {e}")

    # ------------------------------------------------------------------ I/O

    def clear(self) -> None:
        if self.stream_path.exists():
            try:
                self.stream_path.unlink()
            except OSError as e:
                print(f"[TOKEN PAGER] Gagal menghapus stream: {e}")
        self._last_active_tokens = 0

    def append(
        self,
        role: str,
        content: str,
        compressed: str | None = None,
        **meta: Any,
    ) -> int:
        """Append satu turn ke stream. Return seq number."""
        history = self.load_history()
        seq = len(history) + 1
        record = {
            "seq": seq,
            "role": role,
            "content": content,
            "compressed": compressed or "",
            "ts": time.time(),
            "meta": meta,
        }
        self._write_record(record)
        return seq

    def append_step(self, step_num: int, raw_thought: str, compressed_state: str) -> None:
        """API kompatibel marathon (ContextPager legacy)."""
        self.append(
            role="step",
            content=raw_thought[:4000],
            compressed=compressed_state,
            step=step_num,
        )

    def load_history(self) -> list[dict[str, Any]]:
        if not self.stream_path.exists():
            return []
        history: list[dict[str, Any]] = []
        try:
            with open(self.stream_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[TOKEN PAGER] Gagal membaca stream: {e}")
        return history

    # ------------------------------------------------------------------ katup

    def _record_summary(self, record: dict[str, Any]) -> str:
        compressed = (record.get("compressed") or "").strip()
        if compressed:
            return compressed
        content = (record.get("content") or "").strip()
        return self._truncate_to_tokens(content, 120)

    def _record_detail(self, record: dict[str, Any], max_detail_tokens: int) -> str:
        role = record.get("role", "unknown")
        step = record.get("meta", {}).get("step")
        header = f"Langkah {step}" if step is not None else role.upper()
        content = (record.get("content") or "").strip()
        compressed = (record.get("compressed") or "").strip()

        body = self._truncate_to_tokens(content, max(64, max_detail_tokens - 40))
        block = f"{header} (DETAIL):\n{body}"
        if compressed:
            summary = self._truncate_to_tokens(compressed, 80)
            block += f"\n\nRingkasan:\n{summary}"
        return block

    def _assemble_within_budget(
        self,
        parts: list[tuple[str, int]],
        budget: int,
    ) -> tuple[str, int]:
        """Gabung parts (text, priority) — priority lebih kecil = masuk dulu."""
        ordered = sorted(parts, key=lambda p: p[1])
        selected: list[str] = []
        used = 0
        for text, _prio in ordered:
            cost = self.estimate_tokens(text)
            if used + cost > budget:
                trimmed = self._truncate_to_tokens(text, budget - used)
                if trimmed:
                    selected.append(trimmed)
                    used += self.estimate_tokens(trimmed)
                break
            selected.append(text)
            used += cost
        return "\n\n".join(selected), used

    def build_active_context(
        self,
        goal: str = "",
        retrieved_context: str = "",
        *,
        profile: str = "default",
    ) -> str:
        """
        Susun active window dalam budget token.

        profile:
          - default / chat: goal + compressed older + recent detail
          - marathon: format goal/riwayat/momentum seperti ContextPager
        """
        history = self.load_history()
        if profile == "marathon":
            return self._build_marathon_context(goal, retrieved_context, history)

        return self._build_chat_context(goal, retrieved_context, history)

    def _build_chat_context(
        self,
        goal: str,
        retrieved_context: str,
        history: list[dict[str, Any]],
    ) -> str:
        if not history and not goal and not retrieved_context:
            return ""

        parts: list[tuple[str, int]] = []
        if goal:
            parts.append((f"=== GOAL / SYSTEM ===\n{goal}", 0))

        recent_n = max(1, self.recent_detail_count)
        older = history[:-recent_n] if len(history) > recent_n else []
        recent = history[-recent_n:] if history else []

        if older:
            summaries = []
            for rec in older:
                summaries.append(
                    f"[{rec.get('role', '?')}] {self._record_summary(rec)}"
                )
            parts.append(
                ("=== RIWAYAT (TERKOMPRESI) ===\n" + "\n".join(summaries), 1)
            )

        if recent:
            detail_budget = max(256, self._inject_budget // 2)
            per_turn = detail_budget // len(recent)
            detail_lines = [
                self._record_detail(rec, per_turn) for rec in recent
            ]
            parts.append(
                ("=== TURN TERKINI ===\n" + "\n\n".join(detail_lines), 2)
            )

        if retrieved_context:
            parts.append(
                (
                    f"=== KONTEKS RETRIEVAL ===\n"
                    f"{self._truncate_to_tokens(retrieved_context, 400)}",
                    3,
                )
            )

        text, used = self._assemble_within_budget(parts, self._inject_budget)
        self._last_active_tokens = used
        return text

    def _build_marathon_context(
        self,
        goal: str,
        retrieved_context: str,
        history: list[dict[str, Any]],
    ) -> str:
        if not history:
            self._last_active_tokens = self.estimate_tokens(goal)
            return (
                f"Goal Utama: {goal}\n\n"
                "Status: Belum ada langkah yang diambil. Ini adalah langkah pertama Anda."
            )

        recent_n = max(2, self.recent_detail_count)
        older = history[:-recent_n] if len(history) > recent_n else []
        recent = history[-recent_n:]

        parts: list[tuple[str, int]] = [
            (f"=== TARGET GOAL ===\nGoal Utama: {goal}", 0),
        ]

        if older:
            summaries = []
            for rec in older:
                step = rec.get("meta", {}).get("step", rec.get("seq", "?"))
                summaries.append(
                    f"Langkah {step} (Terkompresi):\n{self._record_summary(rec)}"
                )
            parts.append(
                ("=== RIWAYAT PROSES (RINGKASAN) ===\n" + "\n\n".join(summaries), 1)
            )

        if recent:
            detail_budget = max(300, self._inject_budget // 2)
            per_step = detail_budget // len(recent)
            detail = "\n\n".join(
                self._record_detail(rec, per_step) for rec in recent
            )
            parts.append((f"=== MOMENTUM TERKINI ===\n{detail}", 2))

        if retrieved_context:
            parts.append(
                (
                    f"=== PENGETAHUAN PENDUKUNG ===\n"
                    f"{self._truncate_to_tokens(retrieved_context, 400)}",
                    3,
                )
            )

        text, used = self._assemble_within_budget(parts, self._inject_budget)
        self._last_active_tokens = used
        return text

    def get_stats(self) -> dict[str, Any]:
        history = self.load_history()
        total_content = sum(len(r.get("content") or "") for r in history)
        total_est = sum(
            self.estimate_tokens(r.get("content") or "")
            + self.estimate_tokens(r.get("compressed") or "")
            for r in history
        )
        stream_bytes = self.stream_path.stat().st_size if self.stream_path.exists() else 0
        return {
            "session_id": self.session_id,
            "stream_kind": self.stream_kind,
            "record_count": len(history),
            "stream_bytes": stream_bytes,
            "estimated_stream_tokens": total_est,
            "estimated_content_chars": total_content,
            "active_window_tokens": self._last_active_tokens,
            "inject_budget_tokens": self._inject_budget,
            "budget_tokens": self.budget_tokens,
            "stream_path": str(self.stream_path),
        }
