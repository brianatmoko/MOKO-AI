import random

DOPAMINE_QUOTES = [
    "Jalur sinaptik baru berhasil terbentuk dan distabilkan! 🌟",
    "Tingkat keheranan kognitif (Surprisal) mendekati nol. Pemahaman matang! 🧠",
    "Asosiasi Hebbian terkonsolidasi dengan bobot sinaptik prima! 🧮",
    "Model generatif berhasil mengasimilasi fakta tanpa kontradiksi! 🏛️",
    "Energi Bebas (Free Energy) menurun tajam! Sistem beradaptasi sukses. 🟢",
    "Kombinatorial logika terintegrasi penuh ke dalam memori jangka panjang! 💾"
]

class SatisfactionEngine:
    """
    SatisfactionEngine
    Menghitung kemajuan belajar (Mastery) materi berdasarkan:
    1. Surprisal (dari FEP Engine) — Kejutan/Error kognitif
    2. Synaptic Weight (dari BCM Plasticity di Math-Omni) — Stabilitas memori jangka panjang
    """
    
    @staticmethod
    def calculate_mastery_step(
        current_mastery: float,
        surprisal: float,
        synaptic_weight: float,
        is_satisfied: bool,
        hyperspeed: bool = False
    ) -> tuple:
        """
        Menghitung langkah kemajuan mastery berikutnya.
        Returns: (new_mastery, delta, boost_factor)
        """
        # Kecepatan dasar
        base_rate = 12.0 if hyperspeed else 4.0
        
        # Boost dari berat sinaptik (Oja's rule: bobot stabil mempercepat pemahaman)
        # Jika bobot > 1.0 (LTP), memberikan faktor perkalian. Jika < 1.0 (LTD), melambatkan.
        boost_factor = max(0.4, min(2.5, synaptic_weight))
        
        # Pengaruh kejutan kognitif (FEP)
        # Jika is_satisfied (surprisal rendah), pemahaman meningkat pesat.
        # Jika tidak satisfied (surprisal tinggi), pemahaman hanya naik sedikit atau diam.
        if is_satisfied:
            fep_factor = (1.0 - surprisal)
            delta = base_rate * fep_factor * boost_factor
        else:
            # Mengalami kebingungan (surprisal tinggi)
            delta = base_rate * 0.15 * boost_factor
            # Beri pinalti kecil jika surprisal ekstrem
            if surprisal > 0.75:
                delta = -2.0  # Lupa/distorsi kognitif
                
        new_mastery = current_mastery + delta
        new_mastery = max(0.0, min(100.0, new_mastery))
        
        return round(new_mastery, 2), round(delta, 2), round(boost_factor, 2)

    @staticmethod
    def get_dopamine_reward() -> str:
        """Mengambil pesan kepuasan kognitif acak ketika materi dikuasai."""
        return random.choice(DOPAMINE_QUOTES)
