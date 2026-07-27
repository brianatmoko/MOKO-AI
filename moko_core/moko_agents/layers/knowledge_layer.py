from moko_memory.disk_manager import DiskManager
from moko_config import settings
from typing import List, Dict, Any

class KnowledgeLayer:
    """
    Layer 1: Knowledge Layer (Omni)
    Bertanggung jawab atas penyimpanan dan pencarian data faktual di lokal disk.
    """
    def __init__(self, disk_mgr: DiskManager):
        self.disk_mgr = disk_mgr

    def search_facts(self, query_embedding: List[float], top_k: int = 5, domain: str = "general") -> List[Dict[str, Any]]:
        """Mencari fakta relevan dari Omni index."""
        try:
            results = self.disk_mgr.search_memory(query_embedding, top_k=top_k, domain=domain)
            return results
        except Exception as e:
            print(f"[KnowledgeLayer] Search error: {e}")
            return []

    def get_omni_stats(self) -> Dict[str, Any]:
        """Mendapatkan statistik index Omni."""
        return {
            "total_items": self.disk_mgr.get_total_count() if hasattr(self.disk_mgr, "get_total_count") else 0,
            "index_path": str(settings.PROJECT_DIR / ".moko_omni")
        }
