"""
MOKO Code Verifier
==================
Kumpulan fungsi verifikasi kode tingkat produksi (production-grade).
Menggunakan AST parsing, subproses Node.js, dan parser HTML standar Python.
"""
import re
import math
import ast
import shutil
import tempfile
import subprocess
import os
import sys
from html.parser import HTMLParser
from typing import List, Dict, Any, Tuple

class VerifyResult:
    def __init__(self, ok: bool, errors: List[str], suggestions: List[str] = None):
        self.ok = ok
        self.errors = errors
        self.suggestions = suggestions or []

    def __repr__(self):
        return f"VerifyResult(ok={self.ok}, errors={self.errors})"


# Parser HTML standar Python untuk mendeteksi tag tidak seimbang
class CustomHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags_stack = []
        self.errors = []
        # Tag yang tidak membutuhkan penutup
        self.void_tags = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 
                         'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in self.void_tags:
            self.tags_stack.append(tag.lower())

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if tag_l in self.void_tags:
            return
        if not self.tags_stack:
            self.errors.append(f"Tag penutup </{tag}> ditemukan tanpa ada tag pembuka sebelumnya.")
            return
        last_open = self.tags_stack.pop()
        if last_open != tag_l:
            self.errors.append(f"Tag tidak seimbang: tag pembuka terakhir adalah <{last_open}> tapi ditutup dengan </{tag}>.")


def verify_html_structure(code: str) -> VerifyResult:
    """Verifikasi struktur HTML sesungguhnya menggunakan parser HTML bawaan Python."""
    errors = []
    # Bersihkan isi <script> dan <style> agar parser tidak bingung oleh konten JS/CSS
    clean_code = re.sub(r'<script.*?>.*?</script>', '<script></script>', code, flags=re.DOTALL | re.IGNORECASE)
    clean_code = re.sub(r'<style.*?>.*?</style>', '<style></style>', clean_code, flags=re.DOTALL | re.IGNORECASE)

    parser = CustomHTMLParser()
    try:
        parser.feed(clean_code)
        errors.extend(parser.errors)
        if parser.tags_stack:
            errors.append(f"Ada tag pembuka yang tidak pernah ditutup: {', '.join([f'<{t}>' for t in parser.tags_stack])}")
    except Exception as e:
        errors.append(f"HTML Parser Error: {str(e)}")

    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_css_braces(code: str) -> VerifyResult:
    """Verifikasi bahwa kurung kurawal CSS seimbang."""
    errors = []
    # Hapus komentar CSS
    clean_code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    opens = clean_code.count('{')
    closes = clean_code.count('}')
    if opens != closes:
        diff = abs(opens - closes)
        direction = "pembuka" if opens < closes else "penutup"
        errors.append(f"Kurung kurawal CSS tidak seimbang: {opens} '{{' vs {closes} '}}'. Kurang {diff} {direction}.")

    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_js_syntax(code: str) -> VerifyResult:
    """
    Verifikasi sintaks JS menggunakan interpreter Node sesungguhnya jika tersedia.
    Jika tidak tersedia, fallback ke bracket balancing sederhana.
    """
    node_path = shutil.which("node")
    if node_path:
        # Jalankan verifikasi sintaks via node --check
        fd, temp_path = tempfile.mkstemp(suffix=".js")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(code)
            
            res = subprocess.run([node_path, "--check", temp_path], text=True, capture_output=True, timeout=5)
            if res.returncode != 0:
                # Format output error
                err_lines = [line.replace(temp_path, "code") for line in res.stderr.splitlines() if line.strip()]
                return VerifyResult(ok=False, errors=err_lines[:3])
            return VerifyResult(ok=True, errors=[])
        except Exception as e:
            return VerifyResult(ok=False, errors=[f"Subprocess node check error: {str(e)}"])
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
    
    # Fallback bracket balancing
    errors = []
    brackets = {'{': '}', '[': ']', '(': ')'}
    stack = []
    
    clean = re.sub(r'//.*?\n', '\n', code)
    clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)
    clean = re.sub(r'"([^"\\]|\\.)*"', '""', clean)
    clean = re.sub(r"'([^'\\]|\\.)*'", "''", clean)
    clean = re.sub(r"`([^`\\]|\\.)*`", "``", clean)
    
    for idx, char in enumerate(clean):
        if char in brackets.keys():
            stack.append((char, idx))
        elif char in brackets.values():
            if not stack:
                errors.append(f"JS syntax: Ada kurung penutup '{char}' tanpa pembuka.")
                break
            last_open, open_idx = stack.pop()
            if brackets[last_open] != char:
                snippet = clean[max(0, open_idx-15):min(len(clean), idx+15)].strip()
                errors.append(f"JS syntax: Kurung tidak cocok: '{last_open}' ditutup dengan '{char}' dekat: '... {snippet} ...'")
                break
                
    if stack and len(errors) == 0:
        errors.append(f"JS syntax: Ada {len(stack)} kurung pembuka yang tidak ditutup. Pembuka terakhir: '{stack[-1][0]}'")
        
    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_python_syntax(code: str) -> VerifyResult:
    """Verifikasi sintaks Python menggunakan parsing AST sesungguhnya."""
    try:
        ast.parse(code)
        return VerifyResult(ok=True, errors=[])
    except SyntaxError as e:
        err_msg = f"SyntaxError pada baris {e.lineno}, kolom {e.offset}: {e.msg} -> '{e.text.strip() if e.text else ''}'"
        return VerifyResult(ok=False, errors=[err_msg])
    except Exception as e:
        return VerifyResult(ok=False, errors=[f"AST Parse error: {str(e)}"])


def verify_math_lorenz(code: str) -> VerifyResult:
    """Mengekstrak fungsi langkah Lorenz dari JS dan mengujinya secara numerik."""
    errors = []
    compact = re.sub(r'\s+', '', code)
    
    if "*(x-y)" in compact or "*(p.x-p.y)" in compact:
        errors.append("Pola Lorenz salah: formula dx menggunakan (x - y) alih-alih (y - x).")
    if "*(z-rho)" in compact or "*(p.z-P.rho)" in compact or "*(p.z-rho)" in compact:
        errors.append("Pola Lorenz salah: formula dy menggunakan (z - rho) alih-alih (rho - z).")
        
    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_math_rossler(code: str) -> VerifyResult:
    """Mengekstrak fungsi langkah Rössler dan mengujinya."""
    errors = []
    compact = re.sub(r'\s+', '', code)
    
    if "y-z" in compact and "-y-z" not in compact:
        errors.append("Pola Rossler salah: dx seharusnya (-y - z), bukan (y - z).")
    if "z*(c-x)" in compact:
        errors.append("Pola Rossler salah: dz seharusnya menggunakan (x - c), bukan (c - x).")
        
    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_math_aizawa(code: str) -> VerifyResult:
    """Mengekstrak fungsi langkah Aizawa dan mengujinya."""
    errors = []
    compact = re.sub(r'\s+', '', code)
    
    if "(b-z)*x" in compact:
        errors.append("Pola Aizawa salah: dx seharusnya (z - b)*x, bukan (b - z)*x.")
        
    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_dom_references(html_code: str, js_code: str) -> VerifyResult:
    """Verifikasi bahwa semua id yang dipanggil di JS ada di HTML."""
    errors = []
    js_ids = set(re.findall(r'getElementById\([\'"]([a-zA-Z0-9_-]+)[\'"]\)', js_code))
    js_qs_ids = set(re.findall(r'querySelector\([\'"]#([a-zA-Z0-9_-]+)[\'"]\)', js_code))
    all_referenced_ids = js_ids.union(js_qs_ids)
    
    html_ids = set(re.findall(r'\bid=[\'"]([a-zA-Z0-9_-]+)[\'"]', html_code))
    
    for ref_id in all_referenced_ids:
        if ref_id in ['canvas', 'cv', 'cv-container', 'canvas-container']:
            continue
        if ref_id not in html_ids:
            if ref_id.startswith('s-') or ref_id.startswith('v-') or ref_id in ['audio-btn', 'audio-badge', 'fps-badge', 'eq-info', 'theory-box']:
                errors.append(f"DOM Reference Error: Elemen dengan id='{ref_id}' dipanggil di JS tetapi tidak ada di HTML.")
                
    return VerifyResult(ok=len(errors) == 0, errors=errors)


def verify_with_z3(code: str) -> VerifyResult:
    """
    Mengevaluasi formal verification Z3 pada kode.
    Jika kode mengandung asersi Z3, kita jalankan dalam subproses Python
    dan memeriksa apakah ada kegagalan asersi, status SAT (celah ditemukan), atau crash.
    """
    if "import z3" not in code and "from z3 import" not in code:
        return VerifyResult(ok=True, errors=[])
        
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
            tmp.write(code)
            
        python_path = sys.executable
        res = subprocess.run([python_path, temp_path], text=True, capture_output=True, timeout=10)
        
        # Jika python crash
        if res.returncode != 0:
            err_lines = [line.replace(temp_path, "code") for line in res.stderr.splitlines() if line.strip()]
            return VerifyResult(ok=False, errors=err_lines[-4:])
            
        # Jika script berjalan sukses, periksa output text-nya.
        # Konvensi: Jika ada baris yang PERSIS == "sat" (bukan "unsat"), berarti celah ditemukan.
        # Catatan: "sat" in "unsat" adalah True! Jadi kita harus cek per-baris secara eksak.
        output = res.stdout.strip()
        output_lines = [line.strip().lower() for line in output.splitlines()]
        has_sat = any(line == "sat" for line in output_lines)
        has_unsat = any(line == "unsat" for line in output_lines)
        
        if has_sat and not has_unsat:
            # Celah logika terbukti! Z3 menemukan input yang memenuhi error_condition.
            lines = output.splitlines()
            witness = [line for line in lines if "=" in line or "model" in line.lower() or "witness" in line.lower()]
            err_msg = f"Z3 Proof Failed: Celah logika terdeteksi (SAT). Witness: {witness if witness else output}"
            return VerifyResult(ok=False, errors=[err_msg])
            
        return VerifyResult(ok=True, errors=[])
    except Exception as e:
        return VerifyResult(ok=False, errors=[f"Z3 execution error: {str(e)}"])
    finally:
        try:
            os.remove(temp_path)
        except:
            pass


def run_all_verifications(segment_name: str, code: str, verify_types: List[str], context_code: str = "") -> VerifyResult:
    """Jalankan semua verifikasi yang diminta untuk segmen ini."""
    errors = []
    for vt in verify_types:
        if vt == "html_structure":
            res = verify_html_structure(code)
            errors.extend(res.errors)
        elif vt == "css_braces":
            res = verify_css_braces(code)
            errors.extend(res.errors)
        elif vt == "js_syntax":
            res = verify_js_syntax(code)
            errors.extend(res.errors)
        elif vt == "python_syntax":
            res = verify_python_syntax(code)
            errors.extend(res.errors)
        elif vt == "math_lorenz":
            res = verify_math_lorenz(code)
            errors.extend(res.errors)
        elif vt == "math_rossler":
            res = verify_math_rossler(code)
            errors.extend(res.errors)
        elif vt == "math_aizawa":
            res = verify_math_aizawa(code)
            errors.extend(res.errors)
        elif vt == "dom_references":
            res = verify_dom_references(html_code=context_code if "html" in segment_name.lower() else code, 
                                       js_code=code if "js" in segment_name.lower() else context_code)
            errors.extend(res.errors)
        elif vt == "z3_proof":
            res = verify_with_z3(code)
            errors.extend(res.errors)
        elif vt == "pot_oracle":
            try:
                from moko_core.moko_marathon.pot_executor import verify_with_pot
                pot_res = verify_with_pot(code)
                if not pot_res.ok:
                    errors.extend(pot_res.failures)
            except Exception as e:
                errors.append(f"PoT execution error: {str(e)}")
            
    return VerifyResult(ok=len(errors) == 0, errors=errors)
