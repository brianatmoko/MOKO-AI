"""
MOKO RAG Agent — Knowledge Ingestion & Directory Management
=============================================================
Agen yang bertanggung jawab atas pengelolaan basis pengetahuan (RAG)
dan organisasi file otomatis berbasis AI.

Fitur:
  1. Ingesti Pengetahuan: Menambahkan file/teks ke .moko_omni/
  2. Pencarian Pengetahuan: Mencari informasi relevan lintas domain
  3. Organisasi File: Memindahkan file ke folder yang tepat berdasarkan konten
  4. Sesi Memori: Mengelola context window untuk percakapan panjang
"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from moko_config import settings
from moko_memory.disk_manager import DiskManager
from moko_agents.llm_engine import engine
from moko_tools.onion_search import OnionSearchTool
from moko_tools.project_indexer import ProjectIndexer

class RAGAgent:
    """
    Hippocampus MOKO — Mengelola ingatan jangka panjang dan struktur file.
    """
    
    def __init__(self, disk_mgr: Optional[DiskManager] = None):
        self.workspace = settings.WORKSPACE_DIR
        self.omni_dir = Path(settings.OMNI_DIR)
        self.disk_mgr = disk_mgr or DiskManager(self.workspace)
        self.onion_tool = OnionSearchTool()
        self.indexer = ProjectIndexer(self.workspace)
        
    def ingest_document(self, file_path: str, domain: str = "general") -> bool:
        """
        Membaca file dan memasukkannya ke dalam basis pengetahuan OMNI.
        """
        p = Path(file_path)
        if not p.exists():
            print(f"  ❌ [RAGAgent] File tidak ditemukan: {file_path}")
            return False
            
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            if not content.strip():
                return False
                
            # Gunakan DiskManager untuk menyimpan ke domain yang ditentukan
            # Metadata menyertakan sumber file
            metadata = {
                "source": p.name,
                "path": str(p.absolute()),
                "timestamp": time.time()
            }
            
            # Mendapatkan embedding dari teks
            embedding = engine.get_embedding(content[:2000]) # Ambil 2000 karakter pertama untuk context
            
            self.disk_mgr.save_memory(
                text=content,
                embedding=embedding,
                domain=domain,
                metadata=metadata
            )
            
            print(f"  ✅ [RAGAgent] Berhasil ingest: {p.name} -> domain:{domain}")
            return True
        except Exception as e:
            print(f"  ❌ [RAGAgent] Gagal ingest {p.name}: {e}")
            return False

    def auto_organize_files(self, source_dir: str):
        """
        Scans direktori dan memindahkan file ke folder domain yang sesuai
        berdasarkan analisis konten sederhana atau router.
        """
        src = Path(source_dir)
        if not src.exists() or not src.is_dir():
            return
            
        from moko_agents.router import CognitiveRouter
        
        for item in src.iterdir():
            if item.is_file() and not item.name.startswith("."):
                try:
                    # Baca sedikit konten untuk identifikasi
                    with open(item, "r", encoding="utf-8", errors="ignore") as f:
                        snippet = f.read(500)
                    
                    if not snippet: continue
                    
                    # Gunakan router untuk menentukan domain
                    _, _, meta = CognitiveRouter.classify_intent(snippet)
                    domain = meta.get("domain", "general")
                    
                    # Pindahkan ke folder domain di workspace
                    target_dir = Path(self.workspace) / "organized" / domain
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    shutil.move(str(item), str(target_dir / item.name))
                    print(f"  📂 [RAGAgent] Organized: {item.name} -> {domain}")
                    
                except Exception as e:
                    print(f"  ⚠️ [RAGAgent] Gagal memproses {item.name}: {e}")

    def search_context(self, query: str, top_k: int = 5) -> str:
        """
        Mencari potongan teks relevan dari basis pengetahuan untuk augmentasi prompt.
        
        Menggunakan MokoRagRetriever dengan moko_embed_engine (lokal, offline-capable).
        Tidak bergantung pada Ollama untuk embedding — selalu tersedia.
        """
        try:
            from moko_memory.moko_rag_retriever import get_rag_retriever
            from moko_agents.router import CognitiveRouter

            context_parts = []

            # Deteksi domain dari intent
            try:
                _, _, meta = CognitiveRouter.classify_intent(query)
                intent = meta.get("intent", "general")
                domain = meta.get("domain", None)
            except Exception:
                intent = "general"
                domain = None

            # ── 1. Cari di OMNI Memory (semua domain, atau domain spesifik) ──
            retriever = get_rag_retriever()
            chunks = retriever.retrieve(query, top_k=top_k, domain=domain)
            if chunks:
                context_parts.append(
                    "=== INTERNAL KNOWLEDGE ===\n" +
                    retriever.format_context(chunks, max_chars=3000)
                )

            # ── 2. Darkweb jika intent == darkweb ──────────────────────────
            if intent == "darkweb":
                dark_results = self.search_darkweb(query)
                if dark_results:
                    context_parts.append("=== DARKWEB SCAN RESULTS ===\n" + dark_results)

            # ── 3. Project structure jika intent coding ─────────────────────
            is_coding = (
                intent == "coding" or
                any(w in query.lower() for w in ["code", "program", "fungsi", "class", "file", "struktur"])
            )
            if is_coding:
                project_map = self.search_project_structure()
                if project_map:
                    context_parts.append("=== PROJECT ARCHITECTURE MAP ===\n" + project_map)

            return "\n\n".join(context_parts)

        except Exception as e:
            print(f"  ⚠️ [RAGAgent] Search error: {e}")
            return ""


    def search_darkweb(self, query: str) -> str:
        """
        Melakukan scanning ke darkweb untuk mendapatkan informasi terbaru.
        """
        try:
            results = self.onion_tool.search_all(query)
            if not results:
                return ""
                
            formatted = []
            for res in results[:5]:
                status = res.get('status', 'UNKNOWN')
                emails = res.get('emails', [])
                email_str = f"\nEmails Found: {', '.join(emails)}" if emails else ""
                formatted.append(f"[{status}] Title: {res['title']}\nURL: {res['link']}\nSnippet: {res['snippet']}{email_str}")
                
            return "\n---\n".join(formatted)
        except Exception as e:
            print(f"  ⚠️ [RAGAgent] Darkweb search error: {e}")
            return ""

    def search_project_structure(self) -> str:
        """
        Menggunakan ProjectIndexer untuk memetakan arsitektur proyek saat ini.
        """
        try:
            self.indexer.scan()
            return self.indexer.generate_summary()
        except Exception as e:
            print(f"  ⚠️ [RAGAgent] Project indexing error: {e}")
            return ""

# Singleton helper
_rag_agent_instance = None

def get_rag_agent(disk_mgr=None) -> RAGAgent:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent(disk_mgr)
    return _rag_agent_instance

if __name__ == "__main__":
    # Test sederhana
    agent = RAGAgent()
    print("MOKO RAG Agent Initialized.")
