import sys
from moko_agents.core_node import CoreNode
from moko_memory.disk_manager import DiskManager
from moko_config import settings

disk_mgr = DiskManager(settings.WORKSPACE_DIR)
core = CoreNode(disk_mgr)

# 1. Simulasikan pertanyaan user
q = "ingat untuk selalu gunakan bahasa indonesia"
ans = "Baik, saya akan selalu menggunakan bahasa Indonesia."

# 2. Ingest
from moko_agents.llm_engine import engine
emb = engine.get_embedding(f"User: '{q}'. MOKO: '{ans}'")
disk_mgr.ingest_chunk("memory_conversation.txt", f"User: '{q}'. MOKO: '{ans}'", emb)

# 3. Generate system prompt untuk pertanyaan selanjutnya
sys_prompt = core.generate_system_prompt("hai")
print(sys_prompt)
