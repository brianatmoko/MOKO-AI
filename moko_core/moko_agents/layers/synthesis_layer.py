from moko_agents.llm_engine import engine
from moko_config import settings

class SynthesisLayer:
    """
    Layer 3: Synthesis Layer (Output Agent Response)
    Bertanggung jawab untuk merangkai jawaban akhir yang natural dan akurat.
    """
    def __init__(self):
        self.main_port = getattr(settings, "MOKO_LLM_PORT", 11434)

    def synthesize_response(self, query: str, context: str, history: str = "") -> str:
        """
        Menghasilkan jawaban menggunakan model utama (Agent 1).
        """
        system_prompt = (
            "Kamu adalah MOKO — AI sovereign yang dibuat khusus untuk Brian.\n"
            "Tugasmu adalah memberikan jawaban terbaik berdasarkan data yang tersedia.\n"
        )
        
        if context:
            system_prompt += f"\n[DATA DARI OMNI (RAG AGENT)]:\n{context}"
        
        if history:
            system_prompt += f"\n\n[KONTEKS PERCAKAPAN]:\n{history}"

        prompt = f"User: {query}"
        
        try:
            # Menggunakan engine utama (Agent 1)
            response = engine.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                coop_params={"temperature": 0.3, "num_predict": 500}
            )
            return response
        except Exception as e:
            return f"Error saat sintesis: {e}"

class OutputAgent:
    """
    Agen Output (Agent 1)
    Bertanggung jawab atas interaksi user dan hasil akhir pengolahan data.
    """
    def __init__(self, synthesis_layer: SynthesisLayer):
        self.synthesis_layer = synthesis_layer
        self.role = "User Interaction Specialist"

    def respond_to_user(self, query: str, context: str, history: str = "") -> str:
        """Memberikan respon akhir kepada user."""
        print(f"[*] [OutputAgent] Mensintesis jawaban untuk user...")
        return self.synthesis_layer.synthesize_response(query, context, history)
