"""
software_builder — MOKO Software Builder Agent Package
=======================================================
Modul AI yang mengorkestrasi proses:
  Interview → Prompt Enrichment → Plan Generation → Step Execution

Komponen utama:
  - models.py          : Dataclasses InterviewData, PlanStep, PlanSession
  - interview_manager.py : State machine multi-turn interview
  - prompt_enrichment.py : Prompt Enrichment Engine (RAG + user data)
  - plan_generator.py    : Parse LLM output → List[PlanStep]
  - step_executor.py     : Generate kode + terminal feedback loop
"""
from moko_agents.software_builder.models import InterviewData, PlanStep, PlanSession
from moko_agents.software_builder.interview_manager import InterviewManager

__all__ = [
    "InterviewData",
    "PlanStep",
    "PlanSession",
    "InterviewManager",
]
