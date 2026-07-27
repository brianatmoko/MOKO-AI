"""
MOKO Neural Surgery: OmniDirectAnswer Engine
============================================
"Saraf OMNI yang memandu LLM secara instan" — mesin yang mengevaluasi tingkat
kekayaan data OMNI untuk memandu generasi LLM.

Tiga Mode Output:
  OMNI_ENRICHED — Confidence > 0.72: Fakta sangat lengkap, LLM menjawab instan & padat
  OMNI_SCAFFOLD — Confidence 0.40-0.72: Fakta parsial, LLM mensintesis kerangka kerja
  LLM_ONLY      — Confidence < 0.40: LLM penuh (berpikir bebas dari awal)
  OMNI_ENRICHED — Confidence > 0.78: Fakta sangat lengkap, LLM menjawab instan & padat
  OMNI_SCAFFOLD — Confidence 0.52-0.78: Fakta parsial, LLM mensintesis kerangka kerja
  LLM_ONLY      — Confidence < 0.52: LLM penuh (berpikir bebas dari awal)
"""

import re
from typing import List, Dict, Optional, Tuple


# ── Threshold Kepercayaan ─────────────────────────────────────────────────────
CONFIDENCE_OMNI_ENRICHED = 0.78   # Sebelumnya 0.72 — dinaikkan untuk akurasi
CONFIDENCE_OMNI_SCAFFOLD = 0.52   # Sebelumnya 0.40 — dinaikkan untuk kurangi cross-domain noise
MIN_CHUNKS_FOR_ENRICHED  = 1      # Min 1 chunk sudah cukup jika confidence tinggi
                                   # (Puzzle KBBI definitif = 1 chunk, boosted 0.80)
MAX_ENRICHED_CONTEXT_CHARS = 3000   # Batas panjang teks fakta OMNI


class OmniDirectAnswer:
    """
    Mesin evaluasi fakta berbasis OMNI Index untuk memandu LLM.
    """

    def __init__(self):
        pass

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        omni_results: List[Dict],
        confidence: float
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluasi tingkat kekayaan data OMNI untuk memandu LLM.

        Args:
            question:      Pertanyaan user
            omni_results:  Hasil pencarian dari DiskManager.omni_first_search()
            confidence:    Score top-1 dari OMNI search

        Returns:
            Tuple[mode, facts_or_scaffold]:
              - ("OMNI_ENRICHED", str)  → fakta lengkap siap disuapkan ke LLM
              - ("OMNI_SCAFFOLD", str)  → kerangka untuk disintesis LLM
              - ("LLM_ONLY",     None) → serahkan ke LLM sepenuhnya
        """
        if not omni_results or confidence < CONFIDENCE_OMNI_SCAFFOLD:
            return ("LLM_ONLY", None)

        # Filter hanya chunk yang relevan (di atas threshold scaffold)
        relevant = [
            r for r in omni_results
            if r.get("score", 0.0) >= CONFIDENCE_OMNI_SCAFFOLD
        ]

        if not relevant:
            return ("LLM_ONLY", None)

        # ── Mode OMNI_ENRICHED: Data sangat kaya, LLM merespons instan ──
        if confidence >= CONFIDENCE_OMNI_ENRICHED and len(relevant) >= MIN_CHUNKS_FOR_ENRICHED:
            facts = self._assemble_facts_context(question, relevant)
            if facts:
                return ("OMNI_ENRICHED", facts)

        # ── Mode OMNI_SCAFFOLD: Buat kerangka untuk LLM ───────────────
        scaffold = self._build_scaffold(question, relevant)
        return ("OMNI_SCAFFOLD", scaffold)

    # ── Private: Rakit Fakta Bersih ───────────────────────────────────────────

    def _assemble_facts_context(self, question: str, results: List[Dict]) -> str:
        """
        Merakit teks fakta bersih tanpa tambahan sapaan template MOKO.
        Fakta disiapkan agar LLM tinggal memformulasikan jawaban akhir.
        """
        # Pisahkan chunks berdasarkan sumber agar fakta terstruktur
        chunks_by_source: Dict[str, List[str]] = {}
        for r in results:
            source = r.get("file", r.get("source", "general"))
            text = r.get("text", "").strip()
            if text and len(text) > 30:
                src_key = self._clean_source_name(source)
                if src_key not in chunks_by_source:
                    chunks_by_source[src_key] = []
                chunks_by_source[src_key].append(text)

        if not chunks_by_source:
            return ""

        fact_parts = []

        # Jika single source dengan banyak chunk (misal: KBBI query)
        if len(chunks_by_source) == 1:
            src = list(chunks_by_source.keys())[0]
            texts = chunks_by_source[src]
            combined = self._merge_chunks(texts)
            fact_parts.append(combined)
        else:
            # Multi-source: urutkan chunk terbaik dari setiap sumber
            for src, texts in chunks_by_source.items():
                best_chunk = texts[0]  # Sudah terurut berdasarkan skor relevansi
                if len(best_chunk) > 50:
                    fact_parts.append(f"Fakta dari {src}:\n{best_chunk}")

        if not fact_parts:
            return ""

        full_facts = "\n\n".join(fact_parts)

        # Batasi panjang
        if len(full_facts) > MAX_ENRICHED_CONTEXT_CHARS:
            full_facts = full_facts[:MAX_ENRICHED_CONTEXT_CHARS] + "..."

        return full_facts

    def _build_scaffold(self, question: str, results: List[Dict]) -> str:
        """
        Membangun 'kerangka jawaban' dari fakta parsial OMNI untuk diisi LLM.
        """
        scaffold_parts = []

        scaffold_parts.append("=== FAKTA INTI DARI MEMORI OMNI ===")

        for i, r in enumerate(results[:5], 1):  # Maksimal 5 fakta terbaik
            text = r.get("text", "").strip()
            score = r.get("score", 0.0)
            source = self._clean_source_name(r.get("file", r.get("source", "unknown")))

            if text and len(text) > 30:
                preview = text[:400] + ("..." if len(text) > 400 else "")
                scaffold_parts.append(f"[Fakta {i} | Sumber: {source} | Relevansi: {score:.2f}]\n{preview}")

        return "\n\n".join(scaffold_parts)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _clean_source_name(self, source: str) -> str:
        """Bersihkan nama sumber agar terbaca natural (hapus path dan ekstensi)."""
        name = source.split("/")[-1]
        name = re.sub(r'\.(txt|jsonl|csv|json)$', '', name, flags=re.IGNORECASE)
        name = name.replace("_", " ").replace("-", " ").title()
        return name or "Pengetahuan Internal"

    def _merge_chunks(self, chunks: List[str]) -> str:
        """Gabungkan beberapa chunk dari sumber yang sama, hilangkan duplikasi."""
        seen_sentences = set()
        merged_lines = []

        for chunk in chunks:
            # Pisahkan per kalimat atau per baris
            lines = chunk.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Deduplication dasar
                key = re.sub(r'\s+', ' ', line.lower())[:60]
                if key not in seen_sentences:
                    seen_sentences.add(key)
                    merged_lines.append(line)

        return "\n".join(merged_lines)


# ── Singleton ─────────────────────────────────────────────────────────────────
omni_direct_answer = OmniDirectAnswer()
