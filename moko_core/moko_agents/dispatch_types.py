"""
MOKO Dispatch Types — Shared Data Structures
============================================
Data structures used by IntentRouter, ModelDispatcher, and CognitiveExecutive.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class IntentClass(Enum):
    """Kelas Intent untuk MOKO Multi-Model."""
    CODING = "coding"
    MATH = "math"
    SECURITY = "security"
    DARKWEB = "darkweb"
    GENERAL = "general"
    BROWSING = "browsing"
    PERSONAL = "personal"
    REASONING = "reasoning"
    SYSTEM = "system"

@dataclass
class TokenBudget:
    """Budget token untuk setiap fase eksekusi."""
    dispatch: int = 1000
    execution: int = 4000
    governor: int = 6000
    assembly: int = 2000

@dataclass
class DispatchManifest:
    """Hasil dari Phase 1: Dispatching."""
    query_id: str
    intent_class: IntentClass
    domain: str
    model_key: str
    path: str
    complexity: str  # SIMPLE, MEDIUM, COMPLEX
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    thinking_enabled: bool = False
    governor_mode: str = "full"  # full, quick, none
    primary_subsystem: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
