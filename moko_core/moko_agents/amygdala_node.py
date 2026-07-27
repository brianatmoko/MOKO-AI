"""
MOKO Agent: AmygdalaNode — Emotional Valence & Threat Detection
==============================================================
Tugas:
1. Valence Tagging: menilai sentimen teks (-1.0 negatif/takut hingga +1.0 positif/senang).
2. Arousal Scoring: menilai tingkat kegelisahan/urgensi (0.0 hingga 1.0).
3. Threat Detection: mendeteksi ancaman kognitif atau kesalahan fatal.
4. Amygdala Hijack: jika arousal > 0.9, intervensi respons tanpa deliberasi LLM.
"""

from typing import Dict, Tuple, Optional

class AmygdalaNode:
    def __init__(self):
        # Kata kunci yang menunjukkan emosi negatif / ancaman
        self.negative_keywords = {
            "salah", "gagal", "rusak", "error", "fatal", "bahaya", 
            "ancaman", "krisis", "buruk", "jelek", "sedih", "marah", 
            "kecewa", "takut", "benci", "rugi", "meledak", "hancur"
        }
        
        # Kata kunci yang menunjukkan emosi positif
        self.positive_keywords = {
            "berhasil", "sukses", "hebat", "bagus", "baik", "senang", 
            "cinta", "gembira", "puas", "untung", "solusi", "membantu",
            "kreatif", "pintar", "cerdas", "mantap", "keren"
        }

        # Kata kunci untuk mengukur arousal (kegelisahan/urgensi)
        self.arousal_keywords = {
            "cepat", "segera", "darurat", "panik", "tolong", "awas",
            "kritis", "fatal", "sekarang", "mendesak", "instan", "penting"
        }

    def analyze_input(self, text: str) -> Dict:
        """
        Menganalisis teks input dan mengembalikan metrik emosional.
        """
        text_lower = text.strip().lower()
        words = text_lower.split()
        
        if not words:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "threat_flag": False,
                "ltp_boost": 1.0,
                "hijack_triggered": False,
                "hijack_response": None
            }

        # 1. Hitung Valence Score heuristik
        neg_count = sum(1 for w in words if any(kw in w for kw in self.negative_keywords))
        pos_count = sum(1 for w in words if any(kw in w for kw in self.positive_keywords))
        
        total_valence_words = neg_count + pos_count
        if total_valence_words > 0:
            valence = (pos_count - neg_count) / total_valence_words
        else:
            valence = 0.0

        # 2. Hitung Arousal Score heuristik
        arousal_count = sum(1 for w in words if any(kw in w for kw in self.arousal_keywords))
        # Porsi kata-kata arousal terhadap panjang kalimat
        arousal = min(1.0, arousal_count / max(1, len(words) // 2) + (0.3 if neg_count > 0 else 0.0))
        
        # 3. Deteksi Ancaman (Threat Detection)
        # Jika ada kata kunci fatal/bahaya atau valence sangat negatif
        has_fatal_kw = any(w in text_lower for w in ["fatal", "kritis", "meledak", "hancur", "bahaya"])
        threat_flag = has_fatal_kw or (valence < -0.7 and arousal > 0.6)

        # 4. Hitung LTP Boost (penguat plastisitas berdasarkan tingkat ketegangan)
        ltp_boost = 1.0 + (arousal * 0.5)

        # 5. Amygdala Hijack (Arousal > 0.9)
        hijack_triggered = False
        hijack_response = None
        
        if arousal > 0.9 or (threat_flag and arousal > 0.85):
            hijack_triggered = True
            if "meledak" in text_lower or "hancur" in text_lower or "fatal" in text_lower:
                hijack_response = "🚨 [AMYGDALA HIJACK] Ancaman kritis terdeteksi! Mengaktifkan protokol darurat kognitif untuk melindungi integritas memori."
            else:
                hijack_response = "⚠️ [AMYGDALA HIJACK] Tingkat ketegangan terlalu tinggi! Proses kognitif prefrontal dialihkan demi kestabilan emosi sistem."

        return {
            "valence": round(valence, 2),
            "arousal": round(arousal, 2),
            "threat_flag": threat_flag,
            "ltp_boost": round(ltp_boost, 2),
            "hijack_triggered": hijack_triggered,
            "hijack_response": hijack_response
        }
