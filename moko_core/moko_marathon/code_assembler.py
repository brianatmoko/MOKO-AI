"""
MOKO Code Assembler
===================
Modul untuk merakit/menggabungkan segmen-segmen kode yang dihasilkan
oleh Marathon Pit Stop Engine menjadi satu file utuh dan menulisnya ke disk.
"""
import os
from typing import Dict, List
from moko_marathon.code_verifier import VerifyResult, run_all_verifications

class CodeAssembler:
    def __init__(self):
        pass

    def assemble_html(self, parts: Dict[str, str]) -> str:
        """
        Menggabungkan segmen HTML, CSS, dan JS menjadi satu file HTML mandiri.
        Struktur segmen umum:
        - HTML_META: Bagian DOCTYPE, head, meta
        - CSS_STYLES: Kode CSS (dalam tag style)
        - HTML_BODY: Elemen DOM (canvas, sidebar, overlay, dsb)
        - JS_MATH: Fungsi matematika (stepLorenz, stepRossler, dll)
        - JS_RENDER: Render loop dan logika visualisasi
        - JS_AUDIO: Web Audio API sonifikasi
        - JS_CONTROLS: Kontrol UI dan interaksi kamera
        - JS_INIT: Inisialisasi utama (main, window listeners)
        """
        # Susun HTML secara terstruktur
        meta = parts.get("HTML_META", "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n<title>MOKO App</title>")
        css = parts.get("CSS_STYLES", "")
        body = parts.get("HTML_BODY", "<body>\n  <canvas id=\"cv\"></canvas>")
        
        # Gabungkan semua JavaScript ke dalam satu tag script
        js_math = parts.get("JS_MATH", "")
        js_render = parts.get("JS_RENDER", "")
        js_audio = parts.get("JS_AUDIO", "")
        js_controls = parts.get("JS_CONTROLS", "")
        js_init = parts.get("JS_INIT", "")
        
        js_combined = f"""
<script>
// ── JS_MATH ──
{js_math}

// ── JS_RENDER ──
{js_render}

// ── JS_AUDIO ──
{js_audio}

// ── JS_CONTROLS ──
{js_controls}

// ── JS_INIT ──
{js_init}
</script>
"""
        
        # Buat wrapper jika tag style belum ada
        style_block = ""
        if css:
            if "<style>" not in css:
                style_block = f"\n<style>\n{css}\n</style>\n"
            else:
                style_block = f"\n{css}\n"

        # Gabungkan segmen-segmen
        html_out = ""
        
        # Pastikan tag penutup head ada
        if "</head>" not in meta:
            html_out += meta + style_block + "\n</head>\n"
        else:
            # Sisipkan CSS sebelum </head>
            html_out += meta.replace("</head>", f"{style_block}</head>")
            
        # Gabungkan body dan script
        if "</body>" not in body:
            html_out += body + "\n" + js_combined + "\n</body>\n</html>"
        else:
            html_out += body.replace("</body>", f"{js_combined}\n</body>")
            
        return html_out

    def write_file(self, content: str, path: str) -> VerifyResult:
        """Menulis konten ke path tujuan di disk dan melakukan verifikasi dasar."""
        try:
            # Pastikan direktori tujuan ada
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            if not os.path.exists(path):
                return VerifyResult(ok=False, errors=[f"File tidak berhasil dibuat di path: {path}"])
                
            size = os.path.getsize(path)
            if size < 5:
                return VerifyResult(ok=False, errors=[f"File terlalu kecil ({size} bytes). Penulisan mungkin gagal."])
                
            return VerifyResult(ok=True, errors=[])
        except Exception as e:
            return VerifyResult(ok=False, errors=[f"Gagal menulis file: {str(e)}"])
