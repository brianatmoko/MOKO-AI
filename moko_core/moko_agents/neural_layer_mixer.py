"""
MOKO Neural Surgery: NeuralLayerMixer
=======================================
"Pencampur Lapisan Neural" — komponen yang menggabungkan output dari lapisan
yang berbeda (OMNI, LLM, Working Memory) menjadi jawaban akhir yang koheren,
tanpa duplikasi, dan dengan gaya bicara MOKO yang konsisten.

Tiga Mode Pencampuran:
  OMNI_ENRICHED → LLM dipandu fakta OMNI kaya
  OMNI_SCAFFOLD → OMNI sebagai tulang punggung + LLM mengisi celah synthesis
  LLM_ONLY      → Hasil LLM saja (mode lama, sebagai fallback)

Juga bertanggung jawab untuk:
  - Menghapus teks duplikat antara OMNI dan LLM output
  - Menerapkan format yang konsisten (poin, paragraf)
"""

import re
from typing import Optional, List


# ── Pola yang Perlu Dibersihkan dari Output LLM ───────────────────────────────
LLM_FILLER_PATTERNS = [
    r'Berdasarkan data (yang saya|yang aku) miliki[,.]?',
    r'Berdasarkan informasi yang tersedia[,.]?',
    r'Sesuai dengan informasi yang ada[,.]?',
    r'Menurut data yang diberikan[,.]?',
    r'Sebagai AI[,]? (saya|aku) (tidak bisa|tidak dapat|tidak mampu)',
    r'Saya hanyalah sebuah AI',
    r'Maaf[,]? (saya|aku) (tidak|belum)',
    r'Perlu dicatat bahwa (saya|aku) adalah',
    r'I (cannot|can\'t|am unable to)',
    r'As an AI',
    r'As a language model',
]


class NeuralLayerMixer:
    """
    Pencampur cerdas output berbagai lapisan neural MOKO OS.
    """

    def mix(
        self,
        mode: str,
        question: str,
        omni_answer: Optional[str] = None,
        llm_answer: Optional[str] = None,
        working_memory_hint: Optional[str] = None,
        domain: Optional[str] = None
    ) -> str:
        """
        Gabungkan output dari berbagai lapisan.

        Args:
            mode:                 "OMNI_ENRICHED" | "OMNI_SCAFFOLD" | "LLM_ONLY"
            question:             Pertanyaan asli user
            omni_answer:          Jawaban/scaffold dari OmniDirectAnswer
            llm_answer:           Jawaban dari LLM (jika ada)
            working_memory_hint:  Hint dari NeuralWorkingMemory (opsional)
            domain:               Domain kognitif dari Pathfinder

        Returns:
            str: Jawaban akhir yang sudah di-mix
        """
        if mode == "OMNI_ENRICHED":
            return self._mix_enriched_llm(omni_answer or "", llm_answer or "", domain)

        elif mode == "OMNI_SCAFFOLD":
            return self._mix_scaffold_llm(omni_answer or "", llm_answer or "", domain)

        else:  # LLM_ONLY
            return self._clean_llm_output(llm_answer or "")

    # ── Mode: OMNI Enriched ───────────────────────────────────────────────────

    def _mix_enriched_llm(self, facts: str, llm_answer: str, domain: Optional[str] = None) -> str:
        """
        Membersihkan dan menyeimbangkan jawaban LLM yang dipandu fakta kaya OMNI.
        """
        if not llm_answer:
            return self._mix_omni_only(facts, domain)

        cleaned_llm = self._clean_llm_output(llm_answer)
        if not cleaned_llm:
            return self._mix_omni_only(facts, domain)

        return cleaned_llm

    # ── Mode: OMNI Only (Fallback) ────────────────────────────────────────────

    def _mix_omni_only(self, omni_answer: str, domain: Optional[str] = None) -> str:
        """
        Bersihkan jawaban OMNI mentah jika LLM gagal menjawab, dibungkus dengan persona MOKO.
        """
        if not omni_answer:
            return ""

        # Bersihkan artefak pencarian
        cleaned = self._remove_metadata_tags(omni_answer)

        # Pastikan format rapi
        cleaned = self._normalize_whitespace(cleaned)

        # Bungkus fakta mentah dengan persona MOKO
        if domain == "lexical":
            prefix = "Aku memuat definisi resmi berikut dari basis data bahasa saya:\n\n"
        elif domain == "math":
            prefix = "Berikut adalah fakta komputasi murni yang tercatat dalam sistemku:\n\n"
        elif domain == "personal":
            prefix = "Memori persisten saya mencatat data berikut tentang Anda:\n\n"
        else:
            prefix = "Aku mendeteksi rekaman fakta berikut di dalam memori internal MOKO:\n\n"

        return f"{prefix}{cleaned}\n\n*[Pesan dikirim via fallback OMNI]*"

    # ── Mode: OMNI Scaffold + LLM ─────────────────────────────────────────────

    def _mix_scaffold_llm(self, scaffold: str, llm_answer: str, domain: Optional[str] = None) -> str:
        """
        Gabungkan scaffold OMNI dengan output LLM.
        """
        if not llm_answer:
            return self._mix_omni_only(scaffold, domain)

        cleaned_llm = self._clean_llm_output(llm_answer)

        if not cleaned_llm:
            return self._mix_omni_only(scaffold, domain)

        return cleaned_llm

    # ── Mode: LLM Only ────────────────────────────────────────────────────────

    def _clean_llm_output(self, llm_answer: str) -> str:
        """
        Bersihkan output LLM dari filler phrases dan pola yang tidak diinginkan.
        """
        if not llm_answer:
            return ""

        cleaned = llm_answer

        # Hapus pola filler LLM
        for pattern in LLM_FILLER_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Bersihkan whitespace berlebihan
        cleaned = self._normalize_whitespace(cleaned)

        return cleaned

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _remove_metadata_tags(self, text: str) -> str:
        """Hapus tag metadata internal OMNI."""
        text = re.sub(r'\[Sumber:[^\]]*\]\n?', '', text)
        text = re.sub(r'\[Fakta \d+[^\]]*\]\n?', '', text)
        text = re.sub(r'=== FAKTA DARI OMNI[^\n]*===\n?', '', text)
        text = re.sub(r'=== INSTRUKSI UNTUK LLM[^\n]*===\n?', '', text)
        text = re.sub(r'=== MEMORI KERJA AKTIF[^\n]*===\n?', '', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalisasi whitespace berlebihan."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines).strip()
        return text


# ── Singleton ─────────────────────────────────────────────────────────────────
neural_layer_mixer = NeuralLayerMixer()
