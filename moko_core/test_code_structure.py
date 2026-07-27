"""
test_code_structure.py — Uji Fase 1 AI-IDE: checker struktur kode
==================================================================
Jalankan:  moko_core/venv/bin/python moko_core/test_code_structure.py
Menguji moko_ui.code_structure (tag/bracket checker + tree-sitter)
dan integrasinya di EditorPanel (offscreen, tanpa display).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moko_ui.code_structure import (
    analyze_source, check_brackets, check_tags, find_matching_bracket,
    is_supported_language, is_treesitter_available, language_for_path,
    structure_report, summarize_issues, treesitter_issues,
)

PASSED = 0


def ok(cond, label):
    global PASSED
    assert cond, f"GAGAL: {label}"
    PASSED += 1
    print(f"  ✔ {label}")


# ─── 1. Deteksi bahasa dari path ──────────────────────────────────────────────
print("\n[1] language_for_path")
ok(language_for_path("/a/b/main.py") == "python", "ekstensi .py → python")
ok(language_for_path("index.HTML") == "html", "ekstensi .HTML (kapital) → html")
ok(language_for_path("tanpa_ekstensi") == "", "tanpa ekstensi → ''")
ok(language_for_path("") == "", "path kosong → ''")

# ─── 2. Tag checker HTML ──────────────────────────────────────────────────────
print("\n[2] check_tags — HTML")
issues = check_tags("<div>\n  <p>halo\n</div>")
ok(len(issues) == 1 and issues[0].kind == "unclosed_tag", "tag <p> belum ditutup terdeteksi")
ok(issues[0].line == 1 and issues[0].col == 2, "posisi <p> tepat (baris 2, kolom 3)")

ok(check_tags("<div><p>halo</p></div>") == [], "HTML valid → tanpa masalah")
ok(check_tags("<div><br><img src='x'><hr></div>") == [], "void elements tidak dianggap unclosed")
ok(check_tags("<a/><b></b>") == [], "self-closing tag valid")

issues = check_tags("</span>")
ok(len(issues) == 1 and issues[0].kind == "unopened_tag", "tag penutup tanpa pembuka terdeteksi")

ok(check_tags("<!-- <div> --><p></p>") == [], "tag dalam komentar diabaikan")
ok(check_tags("<!DOCTYPE html><html></html>") == [], "doctype diabaikan")
ok(check_tags('<script>if (a<b) { x("<div>"); }</script>') == [], "isi <script> dilewati (rawtext)")

issues = check_tags("<script>var a = 1;")
ok(len(issues) == 1 and "script" in issues[0].message, "<script> tanpa penutup terdeteksi")

issues = check_tags("<UL><li>x</li></ul>")
ok(issues == [], "nama tag HTML case-insensitive")

issues = check_tags("<b><i>tebal</b>")
ok(len(issues) == 1 and "<i>" in issues[0].message, "nesting salah: <i> dilaporkan belum ditutup")

# ─── 3. Tag checker XML (case-sensitive) ─────────────────────────────────────
print("\n[3] check_tags — XML")
issues = check_tags("<Data></data>", xml_mode=True)
ok(len(issues) == 2, "XML case-sensitive: <Data> vs </data> = 2 masalah")
ok(check_tags("<root><item/></root>", xml_mode=True) == [], "XML valid → tanpa masalah")

# ─── 4. Bracket checker ──────────────────────────────────────────────────────
print("\n[4] check_brackets")
issues = check_brackets("x = (1 + 2", "python")
ok(len(issues) == 1 and issues[0].kind == "unclosed_bracket", "kurung '(' belum ditutup")
ok(issues[0].line == 0 and issues[0].col == 4, "posisi kurung pembuka tepat")

issues = check_brackets("x = [1)", "python")
ok(len(issues) >= 1 and issues[0].kind == "mismatched_bracket", "'[' ditutup ')' → mismatch")

issues = check_brackets("x = 1)", "python")
ok(len(issues) == 1 and issues[0].kind == "unmatched_bracket", "penutup tanpa pembuka")

ok(check_brackets("s = '(' + \"[\"  # }", "python") == [], "bracket dalam string & komentar diabaikan")
ok(check_brackets('doc = """\n( [ {\n"""', "python") == [], "bracket dalam triple-quote diabaikan")
ok(check_brackets("def f(x):\n    return [x, {1: (2,)}]\n", "python") == [], "Python valid → tanpa masalah")
ok(check_brackets("// (((\nlet a = `{{{`;\nlet b = [1];", "javascript") == [],
   "JS: komentar // dan template literal diabaikan")
ok(check_brackets("<p>(halo</p>", "html") == [], "bahasa non-bracket (html) dilewati")

# ─── 5. find_matching_bracket ────────────────────────────────────────────────
print("\n[5] find_matching_bracket")
src = "a = (b[1] + {2: 3})"
ok(find_matching_bracket(src, 4, "python") == (4, 18), "pasangan ( luar ditemukan")
ok(find_matching_bracket(src, 6, "python") == (6, 8), "pasangan [ dalam ditemukan")
ok(find_matching_bracket(src, 18, "python") == (4, 18), "dari penutup kembali ke pembuka")
ok(find_matching_bracket("x = (1", 4, "python") is None, "bracket tak berpasangan → None")
ok(find_matching_bracket("abc", 1, "python") is None, "bukan bracket → None")

# ─── 6. tree-sitter ──────────────────────────────────────────────────────────
print("\n[6] treesitter_issues")
if is_treesitter_available():
    issues = treesitter_issues("def foo(:\n    return 1\n", "python")
    ok(len(issues) >= 1, "error sintaks Python terdeteksi tree-sitter")
    ok(treesitter_issues("def foo():\n    return 1\n", "python") == [],
       "Python valid → tanpa error tree-sitter")
    issues = treesitter_issues('{"a": 1,}', "json")
    ok(isinstance(issues, list), "parser json berjalan tanpa crash")
else:
    print("  ⚠ tree-sitter tidak terpasang — dilewati (fallback aktif)")

# ─── 7. analyze_source (gabungan) ────────────────────────────────────────────
print("\n[7] analyze_source")
issues = analyze_source("<div>\n  <p>halo\n</div>", "html")
ok(len(issues) == 1 and issues[0].kind == "unclosed_tag", "HTML: gabungan mendeteksi unclosed tag")

issues = analyze_source("x = (1 + 2\n", "python")
kinds = {it.kind for it in issues}
ok("unclosed_bracket" in kinds, "Python: bracket checker aktif")

ok(analyze_source("", "python") == [], "source kosong → []")
ok(analyze_source("x = 1", "") == [], "bahasa kosong → []")
ok(analyze_source("halo dunia", "text") == [], "plain text → tanpa masalah")

issues = analyze_source("def f():\n    return {'a': [1, 2]}\n", "python")
ok(issues == [], "Python valid → tanpa masalah")

# ─── 8. summarize & report (hook Fase 3) ─────────────────────────────────────
print("\n[8] summarize_issues & structure_report")
issues = analyze_source("<div>\n  <p>halo\n</div>", "html")
summary = summarize_issues(issues)
ok("baris 2" in summary and "<p>" in summary, "ringkasan menyebut baris & tag")
ok("OK" in summarize_issues([]), "tanpa masalah → 'Struktur OK'")

report = structure_report("<div>", "html")
ok(report["ok"] is False and report["issue_count"] == 1, "structure_report: laporan benar")
ok(report["issues"][0]["kind"] == "unclosed_tag", "structure_report: issue ter-serialisasi")
report_ok = structure_report("print(1)\n", "python")
ok(report_ok["ok"] is True, "structure_report: kode valid → ok=True")

ok(is_supported_language("html") and is_supported_language("python"), "html & python didukung")
ok(not is_supported_language("markdown") and not is_supported_language(""), "markdown/kosong tidak dicek")

# ─── 9. Smoke test integrasi EditorPanel (offscreen) ─────────────────────────
print("\n[9] Integrasi EditorPanel (offscreen)")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PyQt6.QtWidgets import QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("  ⚠ PyQt6 tidak tersedia di venv ini — smoke test GUI dilewati")

if HAS_QT:
    app = QApplication.instance() or QApplication(sys.argv)
    from moko_ui.panels.editor_panel import EditorPanel

    panel = EditorPanel()
    panel.set_content("<div>\n  <p>halo\n</div>", "/tmp/contoh.html")
    panel._run_structure_check()
    ok(len(panel.get_structure_issues()) == 1, "editor mendeteksi 1 masalah pada HTML rusak")
    ok("⚠" in panel._lbl_struct.text() and "issue" in panel._lbl_struct.text().lower(),
       "footer menampilkan indikator ⚠")
    ok(len(panel._editor.extraSelections()) >= 1, "underline error terpasang di editor")

    panel.set_content("<div><p>halo</p></div>", "/tmp/contoh.html")
    panel._run_structure_check()
    ok(panel.get_structure_issues() == [], "setelah diperbaiki → tanpa masalah")
    ok("OK" in panel._lbl_struct.text(), "footer menampilkan status OK")

    panel.set_content("x = (1 + 2\n", "/tmp/contoh.py")
    panel._run_structure_check()
    ok(any(it.kind == "unclosed_bracket" for it in panel.get_structure_issues()),
       "editor mendeteksi bracket Python belum ditutup")
    ok("baris 1" in panel.get_structure_summary(), "ringkasan untuk AI menyebut lokasi")

    # Highlight pasangan bracket saat kursor di sebelah bracket
    panel.set_content("a = (b + c)\n", "/tmp/contoh.py")
    panel._run_structure_check()
    cursor = panel._editor.textCursor()
    cursor.setPosition(5)  # tepat setelah '('
    panel._editor.setTextCursor(cursor)
    ok(len(panel._bracket_selections) == 2, "pasangan bracket tersorot (2 seleksi)")

print(f"\n{'='*60}\nSEMUA TEST LULUS ✔  ({PASSED} assertion)\n{'='*60}")
