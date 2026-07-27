"""
MOKO Coding Multi-Agent Subsystem
=================================
Membagi intelligence MOKO Coder menjadi 5 thread spesialis:
1. SyntaxAgent     - Analisis & perbaikan sintaksis
2. ErrorAgent      - Analisis bug & logs
3. GeneratorAgent  - Pembuat kode (generation)
4. EvaluatorAgent  - Code review & scoring kualitas
5. SystemAgent     - Optimasi performa & arsitektur sistem
"""

from .coding_router import CodingRouter
from .syntax_agent import SyntaxAgent
from .error_agent import ErrorAgent
from .generator_agent import GeneratorAgent
from .evaluator_agent import EvaluatorAgent
from .system_agent import SystemAgent
from .coding_orchestrator import CodingOrchestrator

__all__ = [
    "CodingRouter",
    "SyntaxAgent",
    "ErrorAgent",
    "GeneratorAgent",
    "EvaluatorAgent",
    "SystemAgent",
    "CodingOrchestrator",
]
