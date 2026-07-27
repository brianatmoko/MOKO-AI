import sys
from moko_memory.disk_manager import DiskManager
from moko_agents.llm_engine import engine
from moko_config import settings

disk_mgr = DiskManager(settings.WORKSPACE_DIR)

query = "Aturan penting, preferensi bahasa, identitas. Pertanyaan: hai"
emb = engine.get_embedding(query)
print(f"Embedding length: {len(emb)}")
if len(emb) == 768:
    results = disk_mgr.search_memory(emb, top_k=5)
    print(f"Results found: {len(results)}")
    for r in results:
        print(r)
else:
    print("Failed to get embedding")
