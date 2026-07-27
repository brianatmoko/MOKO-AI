"""
moko_agents/prompts.py
=======================
Konstanta prompt yang digunakan oleh agen-agen MOKO.
Dipindah dari moko_ui.workers.deep_synthesis_worker (tidak ada PyQt6 dependency).
"""

PROMPT_DEEP_SYNTHESIS = """
Kamu adalah MOKO Dual Absorption Engine — mesin penyerap pengetahuan ganda.
Tugasmu: dari 1 teks, hasilkan 2 lapisan pengetahuan sekaligus.

TEKS:
"{text}"

=== LAPISAN 1: MATH-OMNI (Kerangka Logika / Formula Berpikir) ===
Logic: A=Faktual, B=Empatik, C=Sintesis, D=Definisional, E=Kausal, F=Filosofis,
G=Instruktif, H=Metakognitif
Arousal: 1=Tenang, 2=Sedang, 3=Mendalam
Depth: D0=Singkat, D5=Sedang, D9=Sangat Mendalam
Instruksi: 1-2 kalimat CARA BERPIKIR AI untuk jenis teks ini di masa depan.
Gagasan Pokok: Intisari 2 kalimat.

=== LAPISAN 2: OMNI-SEMANTIC (Kamus Pengetahuan Semantik) ===
Entitas: Nama konsep, tokoh, teknologi, persamaan, atau istilah kunci yang muncul (list).
Kombinasi Baru: Gabungan dua atau lebih konsep/kata yang menciptakan makna baru
atau hubungan unik (list, contoh: 'Kalkulus + Saraf = Neurokalkulus').
Fakta Penting: 2-4 fakta spesifik yang bisa langsung dijawabkan tanpa membaca ulang keseluruhan teks.
Paket Semantik: 1 paragraf ringkasan padat yang berisi semua entitas dan fakta di atas dalam 1 teks utuh,
untuk langsung disimpan sebagai memori percakapan.

KEMBALIKAN HANYA OBJEK JSON (tanpa markdown), persis format ini:
{
  "gagasan_pokok": "...",
  "logic": "E",
  "arousal": "2",
  "depth": "D9",
  "instruction": "...",
  "entitas": ["...", "..."],
  "kombinasi_baru": ["konsep A + konsep B = makna C"],
  "fakta_penting": ["...", "..."],
  "paket_semantik": "..."
}
"""
