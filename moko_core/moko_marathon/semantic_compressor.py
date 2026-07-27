"""
MOKO Semantic Compressor v2 (SimpleMem 3-Stage)
================================================
Stage 1 — Semantic Structured Compression: atomize + density gate
Stage 2 — Online Semantic Synthesis: dedupe + structured memory unit
Stage 3 — Intent-Aware Recall Pack: token-efficient string for pager injection

Ref: SimpleMem (2601.02553) — write-time compression, read-time via QEV/pager.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from moko_config import settings
from moko_agents.llm_engine import engine


class SemanticCompressor:
    """SimpleMem-inspired 3-stage compressor for marathon CoT and chat turns."""

    _LOW_INFO_RE = re.compile(
        r"^(hi|halo|hai|hello|ok|oke|thanks|terima kasih|makasih|ya|tidak)[\s!.?]*$",
        re.IGNORECASE,
    )
    _ENTITY_RE = re.compile(
        r"\b(?:nama(?:ku| saya)?(?:\s+(?:adalah|aku))?\s+[A-Z][a-z]+"
        r"|(?:Brian|Bryan)\b"
        r"|\b\d{1,2}\s+(?:tahun|thn)\b"
        r"|\b(?:Jakarta|Bandung|Surabaya|Indonesia)\b)",
        re.IGNORECASE,
    )

    @classmethod
    def _max_pack_tokens(cls) -> int:
        return getattr(settings, "SEMANTIC_COMPRESS_MAX_TOKENS", 96)

    @classmethod
    def _llm_min_chars(cls) -> int:
        return getattr(settings, "SEMANTIC_COMPRESS_LLM_MIN_CHARS", 400)

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 3.5))

    @classmethod
    def _truncate_to_tokens(cls, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        if cls.estimate_tokens(text) <= max_tokens:
            return text
        max_chars = int(max_tokens * 3.5)
        return text[:max_chars].rsplit(" ", 1)[0] + "…"

    # ── Stage 1: atomize + gate ───────────────────────────────────────────

    @classmethod
    def _passes_density_gate(cls, sentence: str) -> bool:
        s = sentence.strip()
        if len(s) < 12:
            return False
        if cls._LOW_INFO_RE.match(s):
            return False
        if len(set(s.lower().split())) <= 2 and len(s) < 40:
            return False
        return True

    @classmethod
    def atomize(cls, raw_text: str) -> list[str]:
        """Stage 1: pecah teks jadi atomic facts yang lolos density gate."""
        clean = cls._clean_raw(raw_text)
        if not clean:
            return []

        chunks = re.split(r"(?<=[.!?\n])\s+", clean)
        atoms: list[str] = []
        for chunk in chunks:
            line = re.sub(r"\s+", " ", chunk).strip(" -•")
            if not line or not cls._passes_density_gate(line):
                continue
            if line not in atoms:
                atoms.append(line)
            if len(atoms) >= 12:
                break
        return atoms

    @staticmethod
    def _clean_raw(raw_text: str) -> str:
        text = raw_text or ""
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
        if not text:
            thoughts = re.findall(r"<thought>(.*?)</thought>", raw_text or "", flags=re.DOTALL)
            if thoughts:
                text = thoughts[-1].strip()
        return text.strip()

    # ── Stage 2: synthesize ───────────────────────────────────────────────

    @classmethod
    def _extract_entities(cls, text: str) -> list[str]:
        found = cls._ENTITY_RE.findall(text)
        entities: list[str] = []
        for item in found:
            norm = item.strip()
            if norm and norm not in entities:
                entities.append(norm[:60])
        return entities[:6]

    @classmethod
    def synthesize(
        cls,
        atoms: list[str],
        goal: str,
        profile: str = "chat",
    ) -> dict[str, Any]:
        """Stage 2: gabung atoms jadi memory unit terstruktur."""
        unique: list[str] = []
        for atom in atoms:
            lower = atom.lower()
            if any(lower in u.lower() or u.lower() in lower for u in unique if len(u) > 20):
                continue
            unique.append(atom)

        combined = " ".join(unique)
        topic = (goal or unique[0] if unique else "general")[:80]
        unit: dict[str, Any] = {
            "topic": topic,
            "facts": unique[:5],
            "entities": cls._extract_entities(combined),
            "profile": profile,
            "ts": time.time(),
        }
        if profile == "reasoning":
            unit["roadblocks"] = [
                a for a in unique if re.search(r"error|gagal|belum|tidak bisa|hambat", a, re.I)
            ][:2]
            unit["next_steps"] = unique[-1][:120] if unique else ""
        return unit

    def _structure_with_llm(self, raw_text: str, goal: str) -> dict[str, Any] | None:
        """Stage 2 LLM path untuk teks panjang (marathon / reasoning)."""
        clean = self._clean_raw(raw_text)[:6000]
        sys_prompt = (
            "Kamu MOKO Memory Builder (SimpleMem Stage 2). "
            "Ekstrak memory unit padat dari teks. Output HANYA JSON valid:\n"
            '{"topic":"...","facts":["..."],"entities":["..."],'
            '"roadblocks":["..."],"next_steps":"..."}'
        )
        user_prompt = (
            f"Goal: {goal}\n\nTeks:\n{clean}\n\n"
            "Maks 5 facts, masing-masing <= 25 kata. entities = nama/tempat/angka penting."
        )
        try:
            raw = engine.generate_text(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                coop_params={"num_predict": 350, "enable_thinking": False},
            )
            if not raw:
                return None
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            if isinstance(data.get("facts"), list):
                return {
                    "topic": str(data.get("topic") or goal)[:80],
                    "facts": [str(f)[:200] for f in data["facts"][:5]],
                    "entities": [str(e)[:60] for e in (data.get("entities") or [])[:6]],
                    "roadblocks": [str(r)[:120] for r in (data.get("roadblocks") or [])[:2]],
                    "next_steps": str(data.get("next_steps") or "")[:120],
                    "profile": "reasoning",
                    "ts": time.time(),
                }
        except Exception as e:
            print(f"[COMPRESSOR v2] LLM stage-2 error: {e}")
        return None

    # ── Stage 3: recall pack ──────────────────────────────────────────────

    @classmethod
    def pack_for_recall(
        cls,
        unit: dict[str, Any],
        max_tokens: int | None = None,
        query_hint: str = "",
    ) -> str:
        """Stage 3: format memory unit → string minimal untuk pager compressed."""
        cap = max_tokens or cls._max_pack_tokens()
        topic = unit.get("topic", "general")[:60]
        lines = [f"[MEM] T:{topic}"]

        entities = unit.get("entities") or []
        if entities:
            lines.append(f"E:{', '.join(entities[:4])}")

        facts = list(unit.get("facts") or [])
        if query_hint:
            q = query_hint.lower()
            facts.sort(key=lambda f: (q not in f.lower(), len(f)))

        for fact in facts[:3]:
            lines.append(f"• {fact[:140]}")

        if unit.get("profile") == "reasoning":
            for rb in (unit.get("roadblocks") or [])[:1]:
                lines.append(f"! {rb[:100]}")
            nxt = unit.get("next_steps") or ""
            if nxt:
                lines.append(f"→ {nxt[:100]}")

        packed = "\n".join(lines)
        return cls._truncate_to_tokens(packed, cap)

    # ── Public API ────────────────────────────────────────────────────────

    def compress_episode(
        self,
        raw_text: str,
        goal: str,
        profile: str = "chat",
        query_hint: str = "",
        use_llm: bool | None = None,
    ) -> str:
        """Full 3-stage pipeline."""
        clean = self._clean_raw(raw_text)
        if not clean:
            return ""

        if use_llm is None:
            if profile == "reasoning":
                use_llm = getattr(settings, "SEMANTIC_COMPRESS_USE_LLM_MARATHON", False)
            else:
                use_llm = getattr(settings, "SEMANTIC_COMPRESS_USE_LLM", False)

        if len(clean) < 80:
            unit = self.synthesize(self.atomize(clean), goal, profile)
            return self.pack_for_recall(unit, query_hint=query_hint)

        unit = None
        if profile == "reasoning" and use_llm and len(clean) >= self._llm_min_chars():
            unit = self._structure_with_llm(clean, goal)

        if unit is None:
            unit = self.synthesize(self.atomize(clean), goal, profile)

        return self.pack_for_recall(unit, query_hint=query_hint)

    def compress_turn(self, user_text: str, moko_text: str, query_hint: str = "") -> str:
        """Compress satu giliran chat untuk TokenStreamPager."""
        raw = f"User: {user_text.strip()}\nMOKO: {moko_text.strip()}"
        goal = user_text.strip()[:80] or "chat"
        return self.compress_episode(raw, goal, profile="chat", query_hint=query_hint)

    def compress_thinking(self, raw_thinking: str, goal: str) -> str:
        """Marathon CoT compression — backward-compatible API."""
        if not raw_thinking or len(raw_thinking.strip()) < 150:
            return raw_thinking.strip()

        if not getattr(settings, "SEMANTIC_COMPRESSOR_V2", True):
            return self._legacy_compress_thinking(raw_thinking, goal)

        packed = self.compress_episode(raw_thinking, goal, profile="reasoning")
        if packed:
            return packed

        clean = self._clean_raw(raw_thinking)
        return f"=== STATE COMPRESSED (FALLBACK) ===\nGoal: {goal}\nSnippet: {clean[:500]}…"

    def _legacy_compress_thinking(self, raw_thinking: str, goal: str) -> str:
        """Fallback ke prompt lama jika v2 dimatikan."""
        clean_text = self._clean_raw(raw_thinking)
        sys_prompt = (
            "Kamu adalah MOKO State Compressor. Ringkas pemikiran panjang menjadi status padat.\n"
            "Format:\n=== STATE REASONING COMPRESSED ===\n"
            "- Goal:\n- Steps Done:\n- Key Facts:\n- Roadblocks:\n- Next Steps:"
        )
        user_prompt = f"Original Goal: {goal}\n\nRaw Text:\n{clean_text[:6000]}\n\nRingkas (max 150 kata)."
        try:
            compressed = engine.generate_text(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                coop_params={"num_predict": 400, "enable_thinking": False},
            )
            if compressed and len(compressed.strip()) > 30:
                return compressed.strip()
        except Exception as e:
            print(f"[MARATHON COMPRESSOR] Error: {e}")
        return f"=== STATE COMPRESSED (FALLBACK) ===\nGoal: {goal}\nSnippet: {clean_text[:500]}…"


semantic_compressor = SemanticCompressor()
