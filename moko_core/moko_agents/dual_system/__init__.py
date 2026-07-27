"""
MOKO Dual-System — Paket Sistem Ganda (DeepSeek Sistem 2 + Kimi AI Sistem 1)
============================================================================
Mengimplementasikan kalang agentik kolaboratif & self-correcting untuk Moko IDE v5:

- `BrainNode`             → Sistem 2 (DeepSeek): penalaran CoT, perencanaan, unit test.
- `ExecutorNode`          → Sistem 1 (Kimi): Anchor-RAG, operasi berkas, eksekusi terminal.
- `DualRuntimeGuard`      → Sistem 2 Guard: peninjauan log, diagnosis, koreksi-diri.
- `DualSystemOrchestrator`→ koordinator loop Plan → Execute → Guard → (re-plan) → Commit.

Lihat `docs/WALKTHROUGH_MOKO_DUAL_SYSTEM_OVERHAUL.md` untuk arsitektur lengkap.
"""
from __future__ import annotations

from moko_agents.dual_system.brain_node import BrainNode, ExecutionPlan
from moko_agents.dual_system.executor_node import ExecutorNode, ExecutionResult
from moko_agents.dual_system.runtime_guard import (
    DualRuntimeGuard,
    GuardReport,
    VERDICT_COMMIT,
    VERDICT_REPAIR,
)
from moko_agents.dual_system.orchestrator import (
    DualSystemOrchestrator,
    OrchestratorResult,
    IterationTrace,
    STATE_BRAIN,
    STATE_EXECUTOR,
    STATE_GUARD,
)
from moko_agents.dual_system.worker_pool import WorkerPool
from moko_agents.dual_system.interaction_logger import InteractionLogger

__all__ = [
    "BrainNode",
    "ExecutionPlan",
    "ExecutorNode",
    "ExecutionResult",
    "DualRuntimeGuard",
    "GuardReport",
    "VERDICT_COMMIT",
    "VERDICT_REPAIR",
    "DualSystemOrchestrator",
    "OrchestratorResult",
    "IterationTrace",
    "STATE_BRAIN",
    "STATE_EXECUTOR",
    "STATE_GUARD",
    "WorkerPool",
    "InteractionLogger",
]
