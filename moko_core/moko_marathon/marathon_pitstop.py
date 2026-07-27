"""
MOKO Marathon Pit Stop Engine (Hardened)
========================================
Engine penghasil kode segment-by-segment dengan verifikasi AST/Linter (Pit Stop),
eksekusi unit test (TDD Loop), dan checkpoint version control (Git Sandbox).
"""
import re
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from moko_agents.llm_engine import engine
from moko_marathon.code_verifier import run_all_verifications, VerifyResult
from moko_marathon.code_assembler import CodeAssembler
from moko_marathon.git_manager import GitSandboxManager
from moko_marathon.test_runner import TestRunner

@dataclass
class CodeSegment:
    name: str              # Nama segmen, e.g. "HTML_META", "CSS_STYLES"
    prompt: str            # Prompt instruksi khusus segmen ini
    verify_types: List[str]# Jenis verifikasi: ["html_structure", "css_braces", "js_syntax", "math_lorenz", "dom_references"]
    num_predict: int = 1500# Token budget untuk segmen ini


class MarathonPitStop:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.assembler = CodeAssembler()
        self.test_runner = TestRunner()
        self.assembled_parts: Dict[str, str] = {}

    def run_code_marathon(
        self,
        spec: str,
        segments: List[CodeSegment],
        output_path: str,
        max_pit_retries: int = 2,
        on_breath = None
    ) -> str:
        """
        Menjalankan maraton penulisan kode dengan siklus Pit Stop, Git Sandbox, & TDD.
        """
        project_dir = os.path.dirname(output_path)
        git_mgr = GitSandboxManager(project_dir)
        git_active = git_mgr.initialize_repo()

        if on_breath:
            on_breath(f"🏎️ Memulai Marathon Pit Stop Engine untuk file: {os.path.basename(output_path)}")
            if git_active:
                on_breath(f"📁 Git Sandbox aktif di direktori proyek: {project_dir}")
            on_breath(f"📊 Rencana: {len(segments)} segmen kode akan dirakit berurutan.")

        self.assembled_parts.clear()

        for idx, seg in enumerate(segments):
            seg_start_time = time.time()
            if on_breath:
                on_breath(f"\n🟢 [LEG {idx+1}/{len(segments)}] Menulis segmen: {seg.name}...")

            # 1. Panggil LLM untuk menulis segmen ini
            code_part = self._generate_segment_code(spec, seg, on_breath)
            
            # 2. Siklus Pit Stop (Verifikasi, Test, & Self-Correction)
            attempt = 1
            segment_passed = False
            while attempt <= max_pit_retries + 1:
                context_code = ""
                if "dom_references" in seg.verify_types:
                    if "html" in seg.name.lower():
                        context_code = self.assembled_parts.get("JS_MATH", "") + "\n" + self.assembled_parts.get("JS_RENDER", "")
                    else:
                        context_code = self.assembled_parts.get("HTML_BODY", "") + "\n" + self.assembled_parts.get("HTML_META", "")

                # 2a. Jalankan Verifikasi Statik (AST / Syntax Check)
                result = run_all_verifications(seg.name, code_part, seg.verify_types, context_code)

                # 2b. Jalankan TDD (Unit Test Check jika ada test file terkait)
                # Cari file test (misal test_math.py untuk segmen JS_MATH)
                test_errors = []
                test_file = self._find_matching_test(project_dir, seg.name)
                if result.ok and test_file:
                    if on_breath:
                        on_breath(f"🧪 [TDD] Ditemukan file unit test: {os.path.basename(test_file)}. Menjalankan sandbox test...")
                    
                    # Tulis draft sementara agar runner bisa mengetes berkas nyata
                    self.assembled_parts[seg.name] = code_part
                    temp_content = self._assemble_output(output_path)
                    self.assembler.write_file(temp_content, output_path)

                    test_res = self.test_runner.execute_test(test_file)
                    if not test_res["ok"]:
                        test_errors = test_res["errors"]
                        if on_breath:
                            on_breath(f"❌ [TDD FAILED] Eror asersi/runtime tertangkap: {test_errors}")

                # Gabungkan error statis & test runner
                all_errors = result.errors + test_errors

                if not all_errors:
                    if on_breath:
                        on_breath(f"✅ [PIT STOP {seg.name}] Verifikasi & Unit Test Lolos pada percobaan {attempt}!")
                    segment_passed = True
                    break
                else:
                    if on_breath:
                        on_breath(f"⚠️ [PIT STOP {seg.name}] Gagal verifikasi/test (percobaan {attempt}): {all_errors}")
                    
                    if attempt > max_pit_retries:
                        if on_breath:
                            on_breath(f"❌ [PIT STOP {seg.name}] Batas retry tercapai. Rollback perubahan segmen...")
                        if git_active:
                            git_mgr.rollback_to_last_commit()
                        break

                    # Coba perbaiki (Self-Correction)
                    if on_breath:
                        on_breath(f"🛠️ [SELF-HEALING] Memulai perbaikan otomatis untuk segmen {seg.name}...")
                    code_part = self._fix_segment_code(spec, seg, code_part, all_errors, on_breath)
                    attempt += 1

            # Simpan hasil segmen
            self.assembled_parts[seg.name] = code_part
            duration = time.time() - seg_start_time
            
            # Commit segmen stabil jika lolos verifikasi
            if segment_passed and git_active:
                # Tulis file final sementara untuk dicommit
                temp_content = self._assemble_output(output_path)
                self.assembler.write_file(temp_content, output_path)
                git_mgr.commit_segment(seg.name)
                if on_breath:
                    on_breath(f"💾 Git: Commit berhasil untuk segmen {seg.name}.")

            if on_breath:
                on_breath(f"⏱️ Segmen {seg.name} selesai dalam {duration:.1f} detik.")

        # 3. Assembling Final
        if on_breath:
            on_breath("\n⚙️ Merakit seluruh segmen menjadi file final...")
        final_content = self._assemble_output(output_path)

        # 4. Penulisan File Final
        write_res = self.assembler.write_file(final_content, output_path)
        if write_res.ok:
            size_kb = os.path.getsize(output_path) / 1024.0
            msg = f"🏆 [FINISH] Berhasil menulis {output_path} ({size_kb:.1f} KB) dengan {len(segments)} segmen lolos verifikasi!"
            if on_breath:
                on_breath(msg)
            return msg
        else:
            err_msg = f"❌ [ERROR] Gagal menulis file final: {write_res.errors}"
            if on_breath:
                on_breath(err_msg)
            return err_msg

    def _assemble_output(self, output_path: str) -> str:
        """
        Smart assembly routing berdasarkan ekstensi file.
        - .html  → assemble_html() (full HTML template + JS inject)
        - .css / .js / .py / other → concatenate segment code langsung
        """
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".html":
            return self.assembler.assemble_html(self.assembled_parts)
        else:
            # Untuk CSS/JS/Python/plain text — gabungkan segmen secara langsung
            parts_ordered = list(self.assembled_parts.values())
            return "\n\n".join(p for p in parts_ordered if p.strip())

    def _find_matching_test(self, project_dir: str, segment_name: str) -> Optional[str]:
        """Mencari file test yang cocok untuk segmen tertentu di folder proyek."""
        # Cari file test_*.py atau test_*.js
        for f in os.listdir(project_dir):
            if f.startswith("test_") and (f.endswith(".py") or f.endswith(".js")):
                # Asosiasi sederhana: jika nama segmen (misal JS_MATH -> math) ada di nama file test
                seg_keyword = segment_name.split("_")[-1].lower()
                if seg_keyword in f.lower() or "code" in f.lower() or "app" in f.lower():
                    return os.path.join(project_dir, f)
        return None

    def _generate_segment_code(self, spec: str, seg: CodeSegment, on_breath) -> str:
        """Panggil LLM untuk membuat kode segmen awal."""
        history_context = "\n".join([f"/* Segmen {k} */\n{v}" for k, v in self.assembled_parts.items()])
        
        # Tambah instruksi spesifik tipe file berdasarkan nama segmen
        type_hint = ""
        if seg.name.startswith("JS_") or seg.name == "JS_CODE":
            type_hint = "CRITICAL: You are writing a standalone .js file. Do NOT include <script> or </script> HTML tags. Write pure JavaScript only."
        elif seg.name.startswith("CSS_") or seg.name == "CSS_STYLES":
            type_hint = "CRITICAL: Make sure every opening brace { has a matching closing brace }. Count them manually before submitting."
        
        sys_prompt = (
            "You are a Senior Software and Web UI Engineer. "
            "You write highly structured, production-ready, clean, and bug-free code. "
            "Only return raw code matching the request. No explanations, no markdown wrap (do NOT wrap code in ```)."
        )
        
        prompt = (
            f"Spesifikasi Program Utama:\n{spec}\n\n"
            f"Kode yang sudah dibuat sebelumnya:\n{history_context}\n\n"
            f"TUGAS KAMU SEKARANG: Tulis kode untuk segmen '{seg.name}'.\n"
            f"Instruksi Khusus Segmen:\n{seg.prompt}\n\n"
            + (f"{type_hint}\n\n" if type_hint else "")
            + "PENTING: JANGAN tulis markdown wrap seperti ```html atau ```javascript. "
            "Tulis langsung kodenya saja secara lengkap tanpa placeholder."
        )

        raw = engine.generate_text(
            prompt=prompt,
            system_prompt=sys_prompt,
            coop_params={"num_predict": seg.num_predict, "enable_thinking": False}
        )
        return self._clean_segment_output(raw, seg.name)

    def _fix_segment_code(self, spec: str, seg: CodeSegment, bad_code: str, errors: List[str], on_breath) -> str:
        """Siklus perbaikan otomatis dengan menyuapkan feedback error ke LLM."""
        sys_prompt = (
            "You are a Senior Software and Web UI Engineer. "
            "You correct bugs, syntax errors, and test failures instantly. "
            "Only return corrected raw code. No explanations, no markdown wrapper (do NOT wrap code in ```)."
        )
        
        error_list = "\n".join([f"- {e}" for e in errors])
        
        # Tambah instruksi khusus untuk CSS brace balancing
        specific_fix = ""
        if seg.name.startswith("CSS_") or seg.name == "CSS_STYLES":
            opens = bad_code.count('{')
            closes = bad_code.count('}')
            diff = opens - closes
            if diff > 0:
                specific_fix = f"\nSPECIFIC FIX: Your CSS has {opens} opening braces '{{' but only {closes} closing braces '}}'. You are missing exactly {diff} closing brace(s). Find the unclosed rule and add the missing '}}' at the end."
            elif diff < 0:
                specific_fix = f"\nSPECIFIC FIX: Your CSS has {opens} opening braces '{{' but {closes} closing braces '}}'. Remove {abs(diff)} extra closing brace(s)."
        elif seg.name.startswith("JS_") or seg.name == "JS_CODE":
            specific_fix = "\nSPECIFIC FIX: Do NOT include <script> or </script> HTML tags. Write pure JavaScript only — this is a standalone .js file."
        
        prompt = (
            f"Spesifikasi Program Utama:\n{spec}\n\n"
            f"KODE KAMU SEBELUMNYA yang memiliki error:\n{bad_code}\n\n"
            f"DAFTAR ERROR/TEST FAILURES YANG DIHASILKAN:\n{error_list}\n"
            + specific_fix + "\n\n"
            + f"TUGAS: Perbaiki kode di atas agar bebas dari error/kegagalan test di atas. "
            f"Tulis seluruh kode segmen '{seg.name}' yang sudah diperbaiki secara lengkap. "
            f"JANGAN gunakan markdown wrap seperti ```. Tulis langsung kodenya saja."
        )

        raw = engine.generate_text(
            prompt=prompt,
            system_prompt=sys_prompt,
            coop_params={"num_predict": seg.num_predict, "enable_thinking": False}
        )
        return self._clean_segment_output(raw, seg.name)

    def _clean_markdown_wrappers(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r'^```[a-zA-Z]*\n', '', cleaned)
        cleaned = re.sub(r'\n```$', '', cleaned)
        return cleaned.strip()

    def _clean_segment_output(self, text: str, seg_name: str) -> str:
        """
        Bersihkan output LLM berdasarkan tipe segmen:
        - Hapus markdown wrappers (``` ... ```)
        - Untuk JS segmen: hapus <script> dan </script> tags
        - Untuk CSS segmen: pastikan tidak ada <style> wrapper
        - Untuk Python segmen: hapus jika ada HTML tag di awal (kesalahan LLM)
        """
        cleaned = self._clean_markdown_wrappers(text)
        
        # Untuk file JavaScript standalone — hapus <script> wrapper jika ada
        if seg_name.startswith("JS_") or seg_name == "JS_CODE":
            cleaned = re.sub(r'^\s*<script[^>]*>\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*</script>\s*$', '', cleaned, flags=re.IGNORECASE)
        
        # Untuk CSS standalone — hapus <style> wrapper jika ada
        if seg_name.startswith("CSS_") or seg_name == "CSS_STYLES":
            cleaned = re.sub(r'^\s*<style[^>]*>\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*</style>\s*$', '', cleaned, flags=re.IGNORECASE)
        
        # Untuk Python script — jika LLM salah menulis HTML, ambil hanya kode Python
        if seg_name == "PYTHON_SCRIPT":
            # Jika ada <!DOCTYPE> atau <html>, artinya LLM salah tulis HTML
            if cleaned.lstrip().startswith('<!DOCTYPE') or cleaned.lstrip().startswith('<html'):
                # Coba ekstrak kode Python dari dalam teks (cari 'import' atau '#!')
                py_match = re.search(r'(#!/usr/bin/env python|import |def |class )', cleaned)
                if py_match:
                    cleaned = cleaned[py_match.start():]
                else:
                    # Fallback: kembalikan pesan error agar self-healing aktif
                    cleaned = "# ERROR: LLM generated HTML instead of Python. Please rewrite as Python."
        
        return cleaned.strip()



def build_html_segments(spec: str) -> List[CodeSegment]:
    """Otomatis membuat daftar segmen rencana kerja berdasarkan spesifikasi program."""
    return [
        CodeSegment(
            name="HTML_META",
            prompt="Tulis bagian kepala HTML (<!DOCTYPE html>, <html>, <head>, meta viewport, link Google Fonts 'Inter' dan 'JetBrains Mono', dsb). JANGAN tutup tag head atau body.",
            verify_types=["html_structure"]
        ),
        CodeSegment(
            name="CSS_STYLES",
            prompt="Tulis kode CSS lengkap untuk style glassmorphism dark space. Gunakan neon glows, harmony colors, layout rapi untuk control panel sidebar kanan (width: 320px), dan canvas full screen. Pastikan semua kurung kurawal seimbang.",
            verify_types=["css_braces"]
        ),
        CodeSegment(
            name="HTML_BODY",
            prompt="Tulis elemen HTML di dalam <body>. Harus mencakup <canvas id=\"cv\"></canvas>, panel kontrol <div id=\"sidebar\">, slider-slider parameter attractor (sigma, rho, beta, speed, pts, tail, zoom, vol), selektor waveform, tombol audio-btn, fps-badge, eq-info, dan theory-box. Tutup elemen dengan rapi.",
            verify_types=["html_structure"]
        ),
        CodeSegment(
            name="JS_MATH",
            prompt="Tulis fungsi matematika langkah Euler untuk Lorenz Attractor (stepLorenz), Rossler Attractor (stepRossler), dan Aizawa Attractor (stepAizawa). Gunakan rumus fisika asli dan kembalikan objek {x, y, z}. Pastikan rumus terverifikasi benar.",
            verify_types=["js_syntax", "math_lorenz", "math_rossler", "math_aizawa"]
        ),
        CodeSegment(
            name="JS_RENDER",
            prompt="Tulis logika render loop: inisialisasi partikel, fungsi resetPoints(), draw() loop dengan requestAnimationFrame, proyeksi 3D ke 2D menggunakan rotasi (rotX, rotY) dan zoom, penggambaran trail partikel dengan warna gradasi cyan ke magenta berdasarkan koordinat Z.",
            verify_types=["js_syntax"]
        ),
        CodeSegment(
            name="JS_AUDIO",
            prompt="Tulis Web Audio API sonifikasi: setup AudioContext, OscillatorNode (menggunakan waveform dari panel selektor), GainNode untuk kontrol volume dinamis. Modulasi frekuensi oscillator di dalam render loop berdasarkan nilai z rata-rata partikel (avgZ).",
            verify_types=["js_syntax"]
        ),
        CodeSegment(
            name="JS_CONTROLS",
            prompt="Tulis event listener untuk interaktivitas: mouse drag pada canvas untuk memutar kamera (mengubah rotX, rotY), scroll wheel untuk mengubah zoom, listener input slider untuk meng-update parameter secara dinamis dan memperbarui tulisan angka di UI.",
            verify_types=["js_syntax", "dom_references"]
        ),
        CodeSegment(
            name="JS_INIT",
            prompt="Tulis fungsi inisialisasi akhir: resize listener untuk canvas, pemanggilan resetPoints(), jalankan draw() loop pertama kali, dan pasang event listener tombol reset & toggle audio.",
            verify_types=["js_syntax", "dom_references"]
        )
    ]
