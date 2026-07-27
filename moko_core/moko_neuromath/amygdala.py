"""
MOKO NeuroMath: Amygdala (System 1 / Reflex Engine)
=====================================================
Berdasarkan: Kahneman's System 1 (Fast, Instinctive, Emotional)
             Koneksi langsung Amygdala -> HPA Axis (Stress Response)

Tugas Amygdala adalah mengevaluasi input user dalam waktu < 10ms.
Ia bisa "Membajak" (Hijack) sistem, memotong akses ke AnalystNode (LLM/Neocortex)
dan langsung memberikan output refleks yang BERVARIASI berdasarkan konteks.

PERUBAHAN KRITIS:
- Tidak ada lagi jawaban hardcoded yang identik per query.
- Setiap sapaan menghasilkan respons yang dipilih secara dinamis
  berdasarkan waktu hari, sesi sebelumnya, dan random weighted selection.
- Ini membuat MOKO terasa hidup, bukan mesin template.
"""

from typing import Tuple, Optional
import time
import random
from moko_cpu.governor import CPUGovernor


class Amygdala:
    def __init__(self):
        # ── Pola sapaan DENGAN VARIASI — tidak ada satu pun respons identik ──
        # Format: keyword -> list of possible responses
        self.fast_greetings = {
            "hai": [
                "Hai! Ada yang bisa saya bantu?",
                "Hai. Mau ngerjain apa hari ini?",
                "Hadir. Ada keperluan?",
                "Hai — sistem aktif penuh.",
            ],
            "halo": [
                "Halo. Mau mulai dari mana?",
                "Halo — ada yang perlu dipecahkan?",
                "Halo, saya di sini.",
                "Halo. Apa yang ada di pikiran kamu?",
                "Sistem aktif. Halo.",
            ],
            "hello": [
                "Hello. Siap.",
                "Hello — let's work.",
                "Hello, ada keperluan?",
            ],
            "p": [
                "Ya, kenapa?",
                "Iya, ada apa?",
                "Hadir.",
            ],
            "oy": [
                "Hadir!",
                "Ya?",
                "Apa?",
            ],
            "selamat pagi": [
                "Pagi! Hari baru, siap produktif.",
                "Selamat pagi — ada yang ingin dikerjakan?",
                "Pagi. Sistem sudah hangat dari tadi.",
            ],
            "selamat siang": [
                "Siang. Masih semangat?",
                "Siang — ada yang perlu dibantu?",
            ],
            "selamat sore": [
                "Sore. Masih ada yang belum selesai?",
                "Sore — apa yang tersisa untuk diselesaikan?",
            ],
            "selamat malam": [
                "Malam. Sistem tetap aktif selama kamu butuh.",
                "Malam — masih kerja malam ini?",
                "Malam. Ada apa?",
            ],
            "test": [
                "Test diterima. Semua sistem normal.",
                "Roger. Ping balik.",
                "Test OK — respons waktu nyata.",
            ],
            "ping": [
                "Pong!",
                "Pong — latency minimal.",
                "Pong. Saya di sini.",
            ],
            "ok": [
                "Siap.",
                "OK.",
                "Mengerti.",
            ],
            "oke": [
                "Siap.",
                "Baik.",
                "Oke.",
            ],
            "ya": [
                "Ya?",
                "Iya, ada apa?",
                "Silakan.",
            ],
            "thanks": [
                "Sama-sama.",
                "Siap.",
                "Oke.",
            ],
            "makasih": [
                "Sama-sama.",
                "Tentu.",
                "Siap.",
            ],
            "terima kasih": [
                "Sama-sama.",
                "Dengan senang hati.",
                "Tentu — ada lagi yang perlu?",
            ],
        }

        # ── Pola panik / darurat ──────────────────────────────────────────
        self.panic_triggers = {
            "tolong":  ["Ada apa?! Saya di sini.", "Tenang — ceritakan.", "Ada apa?"],
            "panik":   ["Jangan panik. Tarik napas. Ada apa?", "Tenang. Ceritakan masalahnya."],
            "batal":   ["Proses sebelumnya dibatalkan.", "Oke, saya stop.", "Dibatalkan."],
            "awas":    ["Waspada. Mengaktifkan mode aman.", "Siap — ada ancaman apa?"],
        }

        # Tracking query terakhir untuk deteksi pengulangan
        self._last_query: str = ""
        self._last_response: str = ""
        self._repeat_count: int = 0

    def _pick(self, options: list) -> str:
        """
        Pilih respons dari list dengan weighted random.
        Jika query sama dengan sebelumnya (pengulangan), paksa pilih yang berbeda.
        """
        if not options:
            return ""
        if len(options) == 1:
            return options[0]

        # Hindari mengulangi respons yang persis sama
        filtered = [o for o in options if o != self._last_response]
        if not filtered:
            filtered = options
        return random.choice(filtered)

    def _enrich_with_context(self, base_response: str) -> str:
        """
        Tambahkan variasi kontekstual berdasarkan waktu dan kondisi sistem.
        Sesekali tambahkan sentuhan personal tanpa template kaku.
        """
        hour = time.localtime().tm_hour

        # Sesekali (30% chance) tambahkan konteks waktu yang natural
        if random.random() < 0.30:
            if 5 <= hour < 12:
                suffixes = [" Pagi yang bagus untuk mulai.", " Masih pagi — banyak yang bisa dikerjakan."]
            elif 12 <= hour < 17:
                suffixes = [" Semangat siang ini.", ""]
            elif 17 <= hour < 21:
                suffixes = [" Sore sudah — masih ada target?", ""]
            else:
                suffixes = [" Malam ini kita kerja apa?", " Malam — sistem tidak tidur."]
            base_response += random.choice(suffixes)

        return base_response.strip()

    def evaluate(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Evaluasi input secara instan.
        Mengembalikan tuple: (is_hijacked, fast_response_text, state_flag)
        """
        text = user_input.strip().lower()

        # 1. Cek Vitals (Intervensi Fisik Murni) — dengan Thermal Retry
        # Daripada langsung tolak, tunggu hingga 6 detik (3x poll × 2 detik) agar
        # suhu bisa turun natural dan proses dilanjut tanpa user harus input ulang.
        vitals = CPUGovernor.read_vitals()
        if vitals.get('cpu_temp', 0) > 90 or vitals.get('cpu_pct', 0) > 95:
            for _retry in range(3):
                time.sleep(2)
                vitals = CPUGovernor.read_vitals()
                if vitals.get('cpu_temp', 0) <= 90 and vitals.get('cpu_pct', 0) <= 95:
                    break  # Suhu sudah turun — lanjut proses seperti normal
            else:
                # Masih panas setelah 6 detik — tolak dengan pesan
                return True, "Suhu atau beban CPU kritis! Menolak proses berat. Tunggu sistem dingin.", "PANIC_TEMP"

        # 2. Cek Panic Triggers
        for p_word, p_replies in self.panic_triggers.items():
            if p_word in text.split():
                reply = self._pick(p_replies)
                return True, reply, "PANIC_USER"

        # 3. Cek Fast Greetings — hanya untuk input sangat pendek (≤ 3 kata)
        words = text.split()
        if len(words) <= 3:
            for g_word, g_replies in self.fast_greetings.items():
                if g_word == text or (len(words) == 1 and g_word in words):
                    # Deteksi pengulangan query identik
                    if text == self._last_query:
                        self._repeat_count += 1
                    else:
                        self._repeat_count = 0

                    self._last_query = text
                    reply = self._pick(g_replies)
                    reply = self._enrich_with_context(reply)
                    self._last_response = reply
                    return True, reply, "FAST_GREETING"

        # 4. Tidak ada trigger — biarkan NeoCortex (LLM) mengambil alih
        return False, None, None


# Singleton
amygdala = Amygdala()
