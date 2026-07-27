"""
MOKO NeuroMath: Episodic Buffer
================================
Berdasarkan: Baddeley's Working Memory Model — Episodic Buffer Component (2000)

Jembatan integratif yang menyatukan:
  - Fakta dari Omni-Index (Long-Term Semantic Memory)
  - Kerangka Logika dari Math-Omni (Procedural Memory)
  - Konteks Percakapan dari ConvBuffer (Phonological Loop)

menjadi satu "Episode" yang kohesif sebelum dikirim ke LLM.

Tanpa komponen ini, LLM menerima fakta dan logika yang tidak sinkron —
menyebabkan "Cognitive Dissonance" dalam output AI.

Theta-Gamma Analogy:
  - Theta (7 Hz) = Satu "jendela episode" (125ms di otak, 1 query di MOKO)
  - Gamma (30Hz) = Slot dalam jendela tersebut (4-7 item)
  - Setiap slot memiliki prioritas berdasarkan "relevansi + bobot sinapsis"
"""

from dataclasses import dataclass, field
from typing import List, Optional
from moko_neuromath.bcm_synapse import BCMSynapse


GAMMA_SLOTS = 6  # Baddeley's "7±2" — kita batasi ke 6 untuk keamanan


@dataclass
class EpisodicSlot:
    """Satu slot Gamma dalam jendela Theta."""
    content: str
    source: str          # "omni", "math_omni", "conv_buffer", "synergy"
    priority: float      # 0.0 - 1.0 (lebih tinggi = lebih penting)
    formula_id: str = ""


@dataclass
class EpisodicEpisode:
    """
    Satu "Episode" lengkap — representasi MOKO dari satu momen kognitif.
    Setara dengan satu siklus Theta di otak.
    """
    query: str
    slots: List[EpisodicSlot] = field(default_factory=list)

    def add_slot(self, content: str, source: str, priority: float, formula_id: str = ""):
        """Tambahkan satu slot. Jika penuh, ganti slot dengan prioritas terendah."""
        slot = EpisodicSlot(content=content, source=source,
                            priority=priority, formula_id=formula_id)
        if len(self.slots) < GAMMA_SLOTS:
            self.slots.append(slot)
        else:
            # Gamma Competition: ganti slot dengan prioritas terendah
            min_slot = min(self.slots, key=lambda s: s.priority)
            if slot.priority > min_slot.priority:
                self.slots.remove(min_slot)
                self.slots.append(slot)

    def compile_to_context(self) -> str:
        """
        Kompilasi semua slot menjadi satu string konteks terstruktur untuk LLM.
        Slot diurutkan dari prioritas tertinggi ke terendah (Theta sequencing).
        """
        sorted_slots = sorted(self.slots, key=lambda s: s.priority, reverse=True)

        parts = []
        for i, slot in enumerate(sorted_slots, 1):
            label = {
                "omni": "📚 FAKTA",
                "math_omni": "🧠 KERANGKA LOGIKA",
                "conv_buffer": "💬 KONTEKS PERCAKAPAN",
                "synergy": "⚡ SYNERGY REKOMBINASI",
            }.get(slot.source, "📎 KONTEKS")

            parts.append(f"[{label} — Slot {i}]\n{slot.content}")

        return "\n\n".join(parts)

    def get_dominant_formula(self) -> str:
        """Kembalikan formula_id dari slot dengan prioritas tertinggi."""
        math_slots = [s for s in self.slots if s.formula_id]
        if not math_slots:
            return ""
        return max(math_slots, key=lambda s: s.priority).formula_id


class EpisodicBuffer:
    """
    Membangun satu Episode kognitif dari berbagai sumber memori.
    Ini adalah "Otak Tengah" MOKO — koordinator antara semua sistem.
    """

    @staticmethod
    def build_episode(query: str, omni_results: list, math_result: dict,
                      conv_history: str = "", synergy_content: str = "") -> EpisodicEpisode:
        """
        Merakit Episode dari seluruh sumber memori yang tersedia.

        Args:
            query: Pertanyaan user asli
            omni_results: List hasil pencarian dari Omni-Index [{text, score, route}]
            math_result: Hasil dari Math-Omni {formula_id, instruction, score}
            conv_history: Ringkasan percakapan sebelumnya dari ConvBuffer
            synergy_content: Konten Synergy dari SynergyRouter (opsional)
        """
        episode = EpisodicEpisode(query=query)

        # 1. Slot Konteks Percakapan (Phonological Loop — prioritas tinggi jika relevan)
        if conv_history.strip():
            episode.add_slot(
                content=conv_history,
                source="conv_buffer",
                priority=0.70
            )

        # 2. Slot Fakta dari Omni-Index (prioritas berdasarkan skor pencarian)
        for result in omni_results[:3]:  # Ambil 3 hasil teratas
            text = result.get("text", "")
            score = float(result.get("score", 0.5))
            if text:
                episode.add_slot(
                    content=text,
                    source="omni",
                    priority=round(score * 0.9, 3)  # Sedikit diturunkan agar logika bisa bersaing
                )

        # 3. Slot Kerangka Logika dari Math-Omni
        formula_id = math_result.get("formula_id", "")
        instruction = math_result.get("instruction", "")
        if instruction:
            # Bobot sinapsis BCM memengaruhi prioritas logika di episode
            weights = BCMSynapse.get_synaptic_weights()
            syn_weight = weights.get(formula_id, 1.0)
            # Normalisasi: bobot 1.0 = prioritas 0.80; bobot 5.0 = prioritas 0.98
            logic_priority = min(0.98, 0.70 + (syn_weight / 5.0) * 0.28)

            episode.add_slot(
                content=f"Gunakan kerangka berpikir berikut:\n{instruction}",
                source="math_omni",
                priority=round(logic_priority, 3),
                formula_id=formula_id
            )

        # 4. Slot Synergy Rekombinasi (prioritas tertinggi jika tersedia)
        if synergy_content.strip():
            episode.add_slot(
                content=f"[SYNERGY BARU — Rekombinasi Rumus Aktif]\n{synergy_content}",
                source="synergy",
                priority=0.95
            )

        return episode


# Singleton
episodic_buffer = EpisodicBuffer()
