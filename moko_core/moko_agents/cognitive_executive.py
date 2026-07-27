"""
MOKO Omni Executive — Simplified Orchestrator
=============================================
Sistem eksekutif yang disederhanakan untuk MOKO OS.
Menggantikan Cognitive Executive System (CES) yang terlalu kompleks.

Alur:
1. Intent Detection (IntentFirstRouter)
2. Tool/Agent Dispatching
3. Context Gathering (Omni RAG / Onion Search)
4. Final Assembly (LLM)
"""

import time
from typing import Any, Dict, List, Optional
from moko_agents.intent_router import get_intent_router
from moko_agents.dispatch_types import IntentClass
from moko_agents.rag_agent import RAGAgent

class OmniExecutive:
    """
    Orchestrator utama untuk sistem OMNI baru.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.router = get_intent_router()
        self.rag_agent = None # Lazy load

    def _log(self, msg: str):
        if self.verbose:
            print(f"🧠 [OmniExecutive] {msg}")

    def process(self, query: str) -> Dict[str, Any]:
        """
        Proses query user melalui alur OMNI yang disederhanakan.
        """
        t_start = time.perf_counter()
        self._log(f"Processing query: '{query[:50]}...'")

        # 1. Intent Routing
        intent_res = self.router.classify(query)
        self._log(f"Intent detected: {intent_res.intent_class.value} (conf={intent_res.confidence:.2f})")

        # 2. Agent/Tool Execution
        answer = ""
        source = "llm"

        if intent_res.intent_class == IntentClass.DARKWEB:
            from moko_tools.onion_search import OnionSearchTool
            search = OnionSearchTool()
            results = search.search_all(query)
            if results:
                answer = self._format_darkweb_results(results)
                source = "onion_search"
        
        # 3. Fallback to RAG Agent for Knowledge/General
        if not answer:
            if self.rag_agent is None:
                from moko_memory.disk_manager import DiskManager
                self.rag_agent = RAGAgent(DiskManager())
            
            answer = self.rag_agent.answer(query)
            source = "rag_agent"

        t_total_ms = (time.perf_counter() - t_start) * 1000
        self._log(f"Finished in {t_total_ms:.1f}ms")

        return {
            "answer": answer,
            "intent": intent_res.intent_class.value,
            "source": source,
            "latency_ms": t_total_ms,
            "verdict": "accept"
        }

    def _format_darkweb_results(self, results: List[Dict]) -> str:
        lines = ["## 🔍 MOKO Darkweb Scan Results", ""]
        for res in results[:5]:
            status = res.get('status', 'ACTIVE')
            lines.append(f"### [{status}] {res['title']}")
            lines.append(f"- **URL**: {res['link']}")
            lines.append(f"- **Snippet**: {res['snippet']}")
            if res.get('emails'):
                lines.append(f"- **Emails**: {', '.join(res['emails'])}")
            lines.append("")
        return "\n".join(lines)

_executive_instance = None

def get_executive(verbose: bool = True) -> OmniExecutive:
    global _executive_instance
    if _executive_instance is None:
        _executive_instance = OmniExecutive(verbose=verbose)
    return _executive_instance
