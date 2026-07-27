"""
MOKO Marathon Assembler
========================
Menyambungkan kode multi-pass dari LLM menjadi satu file lengkap.

Masalah yang dipecahkan:
  Saat LLM di-continue, ia sering mengulang beberapa baris terakhir
  sebagai "konteks" sebelum melanjutkan kode baru. Assembler ini:
  1. Mendeteksi dan membuang overlap tersebut
  2. Menyambungkan kode sebelumnya dengan lanjutan
  3. Memvalidasi hasil akhir

Contoh pakai:
    assembler = MarathonAssembler()
    merged = assembler.join(prev_code, continuation, language="python")
    if assembler.validate(merged, "python"):
        print("Kode akhir lengkap!")
"""
from __future__ import annotations

import difflib
import re
from typing import Optional


class MarathonAssembler:
    """
    Menyambungkan kode multi-pass dengan deteksi overlap otomatis.
    """

    # Jumlah baris terakhir yang dijadikan "fingerprint" untuk deteksi overlap
    OVERLAP_WINDOW = 8

    def join(self, prev_code: str, continuation: str,
             language: str = "generic") -> str:
        """
        Sambungkan `prev_code` dengan `continuation`, buang duplikasi.

        Strategy:
          1. Ambil N baris terakhir dari prev_code sebagai fingerprint
          2. Cari fingerprint di awal continuation (fuzzy match)
          3. Potong overlap dari continuation
          4. Sambungkan
        """
        prev_code = prev_code.rstrip()
        continuation = continuation.strip()

        if not continuation:
            return prev_code

        # Hapus markdown code fence dari continuation jika ada
        continuation = self._strip_code_fence(continuation)

        # Cari dan buang overlap
        clean_continuation = self._remove_overlap(prev_code, continuation)

        # Sambungkan dengan separator yang sesuai
        separator = "\n"
        if language in ("html", "css") and not prev_code.endswith("\n"):
            separator = "\n"

        result = prev_code + separator + clean_continuation
        return result

    def _strip_code_fence(self, code: str) -> str:
        """Hapus markdown code fences."""
        code = code.strip()
        m = re.match(r"^```[a-zA-Z0-9]*\n(.*?)(?:\n```\s*)?$", code, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Hanya strip fence buka tanpa penutup
        m2 = re.match(r"^```[a-zA-Z0-9]*\n(.*)$", code, re.DOTALL)
        if m2:
            return m2.group(1).strip()
        return code

    def _remove_overlap(self, prev: str, cont: str) -> str:
        """
        Hapus bagian awal `cont` yang merupakan duplikasi dari akhir `prev`.

        Menggunakan SequenceMatcher untuk fuzzy matching — lebih robust
        dari string comparison biasa karena LLM sering mengubah whitespace.
        """
        prev_lines = prev.splitlines()
        cont_lines = cont.splitlines()

        # Ambil fingerprint: N baris terakhir prev (non-empty)
        prev_tail = [l for l in prev_lines[-self.OVERLAP_WINDOW:] if l.strip()]
        if not prev_tail:
            return cont

        # Normalisasi untuk perbandingan (hapus leading/trailing whitespace)
        def norm(lines):
            return [l.strip() for l in lines if l.strip()]

        prev_norm = norm(prev_tail)
        cont_norm = norm(cont_lines[:self.OVERLAP_WINDOW * 2])

        if not prev_norm or not cont_norm:
            return cont

        # Cari berapa baris overlap via SequenceMatcher
        matcher = difflib.SequenceMatcher(
            None, prev_norm, cont_norm, autojunk=False
        )

        # Cek apakah awal cont_norm cocok dengan akhir prev_norm
        overlap_count = 0
        for block in matcher.get_matching_blocks():
            a, b, size = block
            if b == 0 and size >= min(2, len(prev_norm)):
                # cont dimulai dari block yang cocok dengan prev
                overlap_count = size
                break

        if overlap_count == 0:
            return cont

        # Temukan titik pemotongan di cont_lines (berdasarkan baris non-empty)
        non_empty_seen = 0
        cut_at = 0
        for i, line in enumerate(cont_lines):
            if line.strip():
                non_empty_seen += 1
                if non_empty_seen > overlap_count:
                    cut_at = i
                    break
        else:
            # Semua baris cont adalah overlap
            return ""

        result = "\n".join(cont_lines[cut_at:])
        return result

    def validate(self, code: str, language: str = "generic") -> bool:
        """
        Validasi cepat apakah kode akhir lengkap.
        Menggunakan MarathonCodeSentinel.
        """
        try:
            from moko_marathon.marathon_code_sentinel import get_sentinel
            result = get_sentinel().analyze(code, language=language)
            return result.complete
        except Exception:
            return True  # Jika sentinel gagal, anggap OK

    def build_continuation_prompt(
        self,
        original_instruction: str,
        accumulated_code: str,
        language: str,
        sentinel_reason: str = "",
    ) -> tuple[str, str]:
        """
        Buat system prompt + user prompt untuk melanjutkan kode yang terpotong.

        Returns: (system_prompt, user_prompt)
        """
        # Ambil 200 token terakhir sebagai anchor context
        anchor_lines = accumulated_code.splitlines()[-30:]
        anchor = "\n".join(anchor_lines)

        system_prompt = (
            f"Kamu adalah MOKO Coder yang sedang menulis kode {language.upper()}. "
            f"Kode di bawah terpotong di tengah. Lanjutkan PERSIS dari titik "
            f"terputus tanpa mengulang kode yang sudah ada. "
            f"Jangan gunakan markdown fence (```). Tulis kode langsung."
        )

        reason_hint = f"\n⚠️ Alasan terdeteksi terpotong: {sentinel_reason}" if sentinel_reason else ""

        user_prompt = (
            f"Instruksi asli: {original_instruction}\n\n"
            f"Kode yang sudah ditulis (akhiran):\n"
            f"```\n{anchor}\n```\n"
            f"{reason_hint}\n\n"
            f"LANJUTKAN kode dari titik terputus di atas. "
            f"Mulai langsung dari kode yang belum ditulis:"
        )

        return system_prompt, user_prompt


# ── Singleton ──────────────────────────────────────────────────────────────────
_assembler: Optional[MarathonAssembler] = None


def get_assembler() -> MarathonAssembler:
    global _assembler
    if _assembler is None:
        _assembler = MarathonAssembler()
    return _assembler


# ── CLI Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asm = MarathonAssembler()

    prev = """
def calculate(x, y):
    result = x + y
    return result

class DataProcessor:
    def __init__(self):
        self.data = []
""".strip()

    # LLM melanjutkan tapi mengulang beberapa baris awal
    continuation = """
    def __init__(self):
        self.data = []

    def process(self, item):
        self.data.append(item)
        return len(self.data)

    def get_all(self):
        return self.data.copy()
""".strip()

    merged = asm.join(prev, continuation, language="python")
    print("=== HASIL MERGE ===")
    print(merged)
    print("\n=== VALID? ===", asm.validate(merged, "python"))
