class BasePuzzle:
    """
    Interface Dasar untuk Kepingan Modular Puzzle (MOKO OS).
    Setiap modul kognitif / tools harus mewarisi class ini.
    """
    name: str = "base_puzzle"
    description: str = "Base description of the puzzle."
    version: str = "1.0.0"

    def evaluate_suitability(self, query: str) -> float:
        """
        Mengevaluasi seberapa cocok puzzle ini untuk memproses query user.
        Mengembalikan skor kecocokan antara 0.0 (tidak cocok) sampai 1.0 (sangat cocok).
        """
        return 0.0

    def execute(self, query: str, context: dict) -> dict:
        """
        Menjalankan logika puzzle kognitif ini.
        
        Args:
            query:   Pertanyaan asli dari user.
            context: Kamus data tambahan (vitals, session_context, dsb).

        Returns:
            dict: Data fakta/konteks yang dihasilkan untuk diinjeksi ke LLM.
                  Disarankan mengembalikan format:
                  {
                      "facts": "Teks fakta terstruktur",
                      "confidence": float,
                      "metadata": dict
                  }
        """
        return {
            "facts": "",
            "confidence": 0.0,
            "metadata": {}
        }
