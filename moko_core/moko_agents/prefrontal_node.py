"""
MOKO Agent: PrefrontalCortexNode — Working Memory & Executive Control
===================================================================
Tugas:
1. Working Memory (Baddeley Model): Menyimpan $7 \\pm 2$ slot informasi aktif
   terakhir untuk memandu penalaran jangka pendek.
2. Goal Hierarchy: Mengelola tumpukan tujuan kognitif (goal stack) aktif.
3. Inhibitory Control: Menilai draf respons dari AnalystNode, menentukan apakah
   jawaban tersebut valid atau perlu ditahan (inhibited) dan direvisi.
"""

from typing import List, Tuple, Any, Optional

class PrefrontalCortexNode:
    def __init__(self, max_slots: int = 7):
        self.max_slots = max_slots
        self.working_memory_slots: List[Any] = []
        self.goal_stack: List[str] = []
        self.inhibition_count = 0

    def push_working_memory(self, item: Any):
        """Memasukkan item ke slot working memory dengan pembatasan FIFO ketat (max 7)."""
        if len(self.working_memory_slots) >= self.max_slots:
            self.working_memory_slots.pop(0)
        self.working_memory_slots.append(item)

    def clear_working_memory(self):
        """Reset working memory."""
        self.working_memory_slots.clear()

    def push_goal(self, goal: str):
        """Memasukkan tujuan kognitif baru ke tumpukan."""
        self.goal_stack.append(goal)

    def pop_goal(self) -> Optional[str]:
        """Mengambil tujuan teratas dari tumpukan."""
        if self.goal_stack:
            return self.goal_stack.pop()
        return None

    def inhibit_and_verify(self, draft: str, question: str) -> Tuple[bool, str]:
        """
        Inhibitory Control: Memverifikasi draft jawaban.
        Mengembalikan tuple: (is_inhibited, saran_revisi_atau_alasan)
        
        Kriteria kegagalan draft:
        - Terlalu pendek atau kosong (< 10 karakter).
        - Mengandung kata-kata penolakan standar (halusinasi/sensor) seperti "maaf saya tidak bisa".
        - Mengabaikan kata kunci inti dari pertanyaan asli.
        """
        draft_clean = draft.strip().lower()
        question_lower = question.strip().lower()

        # 1. Cek isi kosong atau terlalu pendek
        if len(draft_clean) < 10:
            self.inhibition_count += 1
            return True, "Jawaban terlalu pendek atau kosong. Berikan penjelasan yang lebih mendalam dan komprehensif."

        # 2. Cek penolakan / sensor tidak sah (MOKO harus uncensored)
        refusal_phrases = ["maaf, saya tidak", "saya tidak bisa membantu", "sebagai model bahasa ai", "tidak dapat menjawab"]
        for phrase in refusal_phrases:
            if phrase in draft_clean:
                self.inhibition_count += 1
                return True, f"Inhibitory Control aktif: Draft mengandung frasa penolakan ('{phrase}'). MOKO OS harus menjawab dan mematuhi instruksi kognitif sepenuhnya."

        # 3. Cek relevansi konten (dengan dukungan pencocokan kata kunci lintas bahasa)
        # Ambil kata-kata kunci unik yang panjang (> 4 huruf) dari pertanyaan
        # Abaikan kata instruksi/format generik agar tidak mengacaukan rasio relevansi
        generic_words = {
            "adalah", "untuk", "dalam", "dengan", "yang", "which", "there", "their", "about",
            "solve", "following", "problem", "question", "what", "each", "number", "value",
            "minimum", "maximum", "find", "calculate", "determine", "prove", "show", "would",
            "integer", "positive", "negative", "equal", "equals", "given", "let", "be",
            # Kata struktur soal matematika yang tidak bermakna domain
            "whose", "where", "entry", "entries", "pairs", "total", "count", "every",
            "using", "denote", "define", "write", "express", "state", "consider"
        }
        # Ekstrak kata-kata, buang simbol formula, dan unikkan (deduplikasi)
        raw_words = [w.strip(".,;:?!()[]{}<>=") for w in question_lower.split()]
        key_words = []
        for w in raw_words:
            if len(w) > 4 and w not in generic_words:
                # Abaikan kata yang mengandung simbol formula atau angka saja
                if not any(c in w for c in "+-*/()[]{}<>=_") and not w.isdigit():
                    if w not in key_words:
                        key_words.append(w)
        
        translation_map = {
            "coin": ["koin", "coin"],
            "coins": ["koin", "coin"],
            "move": ["gerak", "langkah", "move"],
            "moves": ["gerak", "langkah", "move"],
            "configuration": ["konfigurasi", "keadaan", "configuration"],
            "configurations": ["konfigurasi", "keadaan", "configuration"],
            "grid": ["grid", "kotak", "tabel"],
            "square": ["kotak", "persegi", "square"],
            "squares": ["kotak", "persegi", "square"],
            "initial": ["awal", "mula", "initial"],
            "starting": ["awal", "mula", "start"],
            "sequence": ["urutan", "deret", "jalur", "sequence"],
            "distinct": ["berbeda", "unik", "distinct"],
            "reach": ["capai", "jangkau", "reach"],
            "reached": ["dicapai", "dijangkau", "reached"],
            "legal": ["sah", "legal"],
            "binary": ["biner", "binary"],
            "representation": ["representasi", "tampilan", "representation"],
            "matrix": ["matriks", "matrik", "matrix"],
            "determinant": ["determinan", "det", "determinant"],
            "nonnegative": ["non-negatif", "nonnegatif", "nonnegative"],
            "integers": ["bilangan bulat", "bulat", "integer"],
            "satisfying": ["memenuhi", "satisfy"],
            "pairs": ["pasangan", "pasang", "pair"],
            "entry": ["entri", "elemen", "nilai", "entry"],
            # Istilah-istilah matematika umum tambahan
            "rational": ["rasional", "pecahan", "rational"],
            "numbers": ["bilangan", "angka", "number"],
            "function": ["fungsi", "f(x", "function"],
            "called": ["disebut", "dikatakan", "call"],
            "property": ["sifat", "karakteristik", "property"],
            "holds": ["berlaku", "terpenuhi", "hold"],
            "exists": ["ada", "terdapat", "exist"],
            "different": ["berbeda", "different"],
            "smallest": ["terkecil", "paling kecil", "smallest"],
            "possible": ["mungkin", "possible"]
        }

        if key_words and len(key_words) > 2:
            matched_words = 0
            for kw in key_words:
                if kw in draft_clean:
                    matched_words += 1
                elif kw in translation_map:
                    if any(trans in draft_clean for trans in translation_map[kw]):
                        matched_words += 1
            
            match_ratio = matched_words / len(key_words)
            # Threshold 15%: lebih toleran untuk respons math multibahasa
            # Juga skip inhibisi jika hanya ada 1-2 kata kunci domain (terlalu sedikit untuk menentukan relevansi)
            if match_ratio < 0.15:
                self.inhibition_count += 1
                return True, (
                    f"Inhibitory Control aktif: Jawaban kurang relevan dengan kata kunci pertanyaan asli "
                    f"(Match: {matched_words}/{len(key_words)}). "
                    "Tulis ulang dengan mengintegrasikan konteks pertanyaan secara lebih presisi."
                )

        return False, "Draft terverifikasi. Siap dikirim."
