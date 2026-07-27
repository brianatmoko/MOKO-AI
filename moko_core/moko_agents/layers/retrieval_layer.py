from typing import List, Dict, Any, Callable
from moko_agents.layers.knowledge_layer import KnowledgeLayer
from moko_agents.llm_engine import engine
from moko_config import settings
from moko_memory.moko_rag_retriever import get_rag_retriever, RagChunk

class RetrievalLayer:
    """
    Layer 2: Retrieval Layer (RAG Agent Context)
    Bertanggung jawab untuk mengambil fakta dan memprosesnya menjadi konteks yang siap pakai.
    """
    def __init__(self, knowledge_layer: KnowledgeLayer):
        self.knowledge_layer = knowledge_layer
        self.rag_port = getattr(settings, "MOKO_RAG_PORT", 11437)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Mengambil fakta relevan menggunakan MokoRagRetriever terpadu.
        """
        retriever = get_rag_retriever()
        chunks = retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return ""
        raw_context = retriever.format_context(chunks)
        return self._distill_with_rag(query, raw_context)

    def retrieve_context_with_xray(self, query: str, top_k: int = 3,
                                    on_xray: Callable[[str], None] = None) -> str:
        """
        Mengambil fakta relevan dengan output kognitif detail untuk panel X-Ray.
        """
        def _emit(msg: str):
            if on_xray:
                try:
                    on_xray(msg)
                except Exception:
                    pass

        _emit("🧠 [X-RAY RAG] Mengaktifkan Pipeline Retrieval Kognitif...")
        
        # 1. Local Embedding
        _emit("🔌 Menghubungkan ke embedding engine lokal (bge-small)...")
        try:
            retriever = get_rag_retriever()
            t0 = __import__("time").time()
            query_emb = retriever.embed_query(query)
            dt = (__import__("time").time() - t0) * 1000
            _emit(f"✅ Embedding sukses: {len(query_emb)}-dimensi dalam {dt:.2f}ms")
        except Exception as e:
            _emit(f"🚨 Gagal generating embedding: {e}")
            return ""

        # 2. SimHash & Hamming space retrieval
        _emit("🗺️ Routing kueri via SimHash & Hamming search...")
        try:
            chunks = retriever.retrieve(query, top_k=top_k)
        except Exception as e:
            _emit(f"🚨 Gagal retrieve dari OMNI: {e}")
            return ""

        if not chunks:
            _emit("⚠️ [OMNI] Tidak ada fakta relevan yang cocok di indeks.")
            return ""

        _emit(f"📚 Ditemukan {len(chunks)} potongan fakta yang cocok dari OMNI Index:")
        for idx, chunk in enumerate(chunks, 1):
            _emit(f"  ↳ [{idx}] [{chunk.domain.upper()}] {chunk.source} | Skor: {chunk.score:.3f}")
            # Tampilkan sedikit kutipan konten di log xray
            snippet = chunk.text[:90].replace('\n', ' ').strip()
            _emit(f"    📝 Kutipan: \"{snippet}...\"")

        # 3. Format Context
        raw_context = retriever.format_context(chunks)

        # 4. RAG Distillation
        _emit("⚗️ Melakukan kompresi/distilasi konteks via model RAG 200MB...")
        result = self._distill_with_rag(query, raw_context, on_xray=_emit)
        if result and len(result) < len(raw_context):
            _emit(f"✅ Distilasi sukses: {len(raw_context)} karakter → {len(result)} karakter.")
        else:
            _emit("⚠️ Menggunakan konteks mentah (RAG model dilewati / tidak aktif).")

        return result

    def _distill_with_rag(self, query: str, raw_context: str,
                           on_xray=None) -> str:
        """
        Meringkas konteks mentah memakai server RAG khusus (port 11437).

        Gerbang aman: hanya jalan bila server RAG aktif. Jika tidak aktif,
        distilasi menghasilkan teks kosong, atau terjadi error → kembalikan
        konteks mentah apa adanya (pipeline tidak boleh putus).
        """
        if not raw_context:
            return raw_context
        try:
            if not engine.rag_available():
                if on_xray:
                    on_xray("⚠️ [RAG Server] Port 11437 tidak aktif — skip distilasi")
                return raw_context

            system_prompt = (
                "Kamu adalah MOKO RAG Bridge — model ringkas untuk retrieval.\n"
                "Ringkas FAKTA berikut menjadi konteks padat yang relevan dengan "
                "PERTANYAAN. Jangan menambah informasi baru; buang yang tidak relevan."
            )
            prompt = (
                f"PERTANYAAN: {query}\n\n"
                f"FAKTA:\n{raw_context}\n\n"
                f"KONTEKS RINGKAS:"
            )
            distilled = engine.generate_rag(
                prompt=prompt,
                system_prompt=system_prompt,
                coop_params={"num_predict": 256, "temperature": 0.0, "enable_thinking": False},
            )
            if distilled and distilled.strip():
                print("  ✅ [RetrievalLayer] Konteks didistilasi via RAG 200MB.")
                return distilled.strip()
            return raw_context
        except Exception as e:
            print(f"  ⚠️ [RetrievalLayer] Distilasi RAG gagal ({e}); pakai konteks mentah.")
            return raw_context

class RAGAgent:
    """
    Agen RAG (Agent 2)
    Bekerja secara independen untuk mengelola data dari Omni.
    """
    def __init__(self, retrieval_layer: RetrievalLayer):
        self.retrieval_layer = retrieval_layer
        self.role = "RAG Specialist"

    def process_query_for_context(self, query: str, on_xray=None) -> str:
        """Mengambil dan memvalidasi konteks untuk query."""
        print(f"[*] [RAGAgent] Menganalisis Omni untuk: {query[:50]}...")
        if on_xray:
            context = self.retrieval_layer.retrieve_context_with_xray(
                query, on_xray=on_xray
            )
        else:
            context = self.retrieval_layer.retrieve_context(query)
        if context:
            print(f"  ✅ [RAGAgent] Berhasil mengekstrak {len(context)} karakter konteks.")
        else:
            print("  ⚠️ [RAGAgent] Tidak ditemukan data relevan di Omni.")
        return context
