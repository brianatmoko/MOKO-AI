from moko_agents.layers.knowledge_layer import KnowledgeLayer
from moko_agents.layers.retrieval_layer import RetrievalLayer, RAGAgent
from moko_agents.layers.synthesis_layer import SynthesisLayer, OutputAgent
from moko_memory.disk_manager import DiskManager

class MokoMultiAgentSystem:
    """
    Sistem Multi-Agent MOKO OS.
    Mengimplementasikan 2 Agent yang bekerja bersama dalam struktur 3 Layer.
    """
    def __init__(self, disk_mgr: DiskManager):
        # 1. Initialize Layers
        self.knowledge_layer = KnowledgeLayer(disk_mgr)
        self.retrieval_layer = RetrievalLayer(self.knowledge_layer)
        self.synthesis_layer = SynthesisLayer()

        # 2. Initialize Agents
        self.rag_agent    = RAGAgent(self.retrieval_layer)      # Agent 2
        self.output_agent = OutputAgent(self.synthesis_layer)   # Agent 1

    def handle_query(self, query: str, history: str = "") -> str:
        """
        Alur kerja multi-agent:
        User -> RAGAgent (Extract Context) -> OutputAgent (Generate Answer) -> User
        """
        print(f"\n[MOKO BRIDGE] Memulai pemrosesan query: {query[:40]}...")
        
        # Langkah 1: RAG Agent (Agent 2) mengelola data dari Omni
        omni_context = self.rag_agent.process_query_for_context(query)
        
        # Langkah 2: Output Agent (Agent 1) mengolah data dan memberi output ke user
        final_answer = self.output_agent.respond_to_user(query, omni_context, history)
        
        print("[MOKO BRIDGE] Pemrosesan selesai.\n")
        return final_answer

_multi_agent_system = None

def get_multi_agent(disk_mgr: DiskManager = None) -> MokoMultiAgentSystem:
    global _multi_agent_system
    if _multi_agent_system is None:
        if disk_mgr is None:
            # Fallback jika disk_mgr tidak diberikan (biasanya saat testing)
            disk_mgr = DiskManager()
        _multi_agent_system = MokoMultiAgentSystem(disk_mgr)
    return _multi_agent_system
