from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


ONLINE_STATES = {"online", "ready", "running", "ok", "healthy", "up"}
LOADING_STATES = {"loading", "starting", "booting", "warming", "initializing"}
OFFLINE_STATES = {"offline", "down", "stopped", "error", "failed"}


def normalize_server_status(payload: dict | None) -> str:
    """Normalisasi payload status server ke: online/loading/offline."""
    if not payload:
        return "offline"

    if isinstance(payload.get("ready"), bool):
        return "online" if payload["ready"] else "offline"
    if isinstance(payload.get("online"), bool):
        return "online" if payload["online"] else "offline"
    if isinstance(payload.get("is_ready"), bool):
        return "online" if payload["is_ready"] else "offline"
    if isinstance(payload.get("running"), bool):
        return "online" if payload["running"] else "offline"

    candidates = []
    for key in ("status", "state", "health"):
        value = payload.get(key)
        if isinstance(value, str):
            candidates.append(value.strip().lower())

    for value in candidates:
        if value in ONLINE_STATES:
            return "online"
        if value in LOADING_STATES:
            return "loading"
        if value in OFFLINE_STATES:
            return "offline"

    return "offline"


@dataclass(frozen=True)
class GuardedGeneration:
    content: str
    source: str
    server_status: str
    message: str
    used_fallback_reason: str | None


class LLMRuntimeGuard:
    def __init__(
        self,
        status_provider: Callable[[], dict | None],
        llm_generate: Callable[[str], str],
        fallback_generate: Callable[[str], str],
    ) -> None:
        self.status_provider = status_provider
        self.llm_generate = llm_generate
        self.fallback_generate = fallback_generate

    def wait_until_ready(self, *, timeout_seconds: int = 90, poll_seconds: float = 2.0) -> str:
        """Jika offline, return cepat tanpa auto-start. Jika loading, tunggu terbatas."""
        status = normalize_server_status(self.status_provider())
        if status in {"online", "offline"}:
            return status

        deadline = time.monotonic() + max(timeout_seconds, 0)
        while time.monotonic() < deadline:
            time.sleep(max(poll_seconds, 0.0))
            status = normalize_server_status(self.status_provider())
            if status in {"online", "offline"}:
                return status

        return "offline"

    def generate(self, prompt: str) -> GuardedGeneration:
        server_status = self.wait_until_ready()
        if server_status != "online":
            return GuardedGeneration(
                content=self.fallback_generate(prompt),
                source="template",
                server_status=server_status,
                message="MOKO SERVER OFFLINE — Klik ▶ START di Status Panel, lalu coba lagi.",
                used_fallback_reason="server_not_ready",
            )

        generated = self.llm_generate(prompt)
        if not generated or not generated.strip():
            return GuardedGeneration(
                content=self.fallback_generate(prompt),
                source="template",
                server_status="online",
                message="LLM tidak mengembalikan output. Menggunakan fallback template.",
                used_fallback_reason="empty_llm_output",
            )

        return GuardedGeneration(
            content=generated,
            source="llm",
            server_status="online",
            message="LLM generation sukses.",
            used_fallback_reason=None,
        )
