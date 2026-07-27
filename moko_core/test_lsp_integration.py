"""
test_lsp_integration.py — Uji Fase 2/3 AI-IDE (LSP + context bridge)
=====================================================================
Jalankan: moko_core/venv/bin/python moko_core/test_lsp_integration.py

Catatan:
- Test ini memakai fake LSP server (python -c) agar tidak bergantung
  pada binary eksternal seperti pylsp atau vscode-html-language-server.
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moko_ui.lsp_client import LocalLspClient, summarize_diagnostics
from moko_ui.editor_ai_bridge import (
    is_code_or_ide_query,
    clamp_editor_context,
    build_cognitive_prompt_with_editor_context,
    CausalVisualFlowCompressor,
)

PASSED = 0


def ok(cond, label):
    global PASSED
    assert cond, f"FAILED: {label}"
    PASSED += 1
    print(f"  ✔ {label}")


def wait_until(predicate, timeout=2.0, step=0.05, tick=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        if tick is not None:
            try:
                tick()
            except Exception:
                pass
        time.sleep(step)
    return predicate()


FAKE_LSP_SERVER_CODE = r'''
import json
import sys


def send(msg):
    data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            if headers:
                break
            continue
        txt = line.decode("ascii", errors="replace")
        if ":" in txt:
            k, v = txt.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    length = int(headers.get("content-length", "0") or "0")
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8", errors="replace"))


def _line_col_for(text, token):
    idx = text.find(token)
    if idx < 0:
        return (0, 0)
    line = text.count("\n", 0, idx)
    line_start = text.rfind("\n", 0, idx)
    if line_start < 0:
        line_start = 0
    else:
        line_start += 1
    col = idx - line_start
    return (line, col)


def publish(uri, text):
    diagnostics = []
    line, col = _line_col_for(text, "ERROR")
    if "ERROR" in text:
        diagnostics.append({
            "range": {
                "start": {"line": line, "character": col},
                "end": {"line": line, "character": col + 5},
            },
            "severity": 1,
            "source": "fake-lsp",
            "message": "Found ERROR token",
        })
    send({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics,
        },
    })


docs = {}
while True:
    msg = read_message()
    if msg is None:
        break

    method = msg.get("method")
    req_id = msg.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "capabilities": {
                    "textDocumentSync": 1,
                    "completionProvider": {"resolveProvider": False},
                    "definitionProvider": True,
                }
            },
        })
        continue

    if method == "initialized":
        continue

    if method == "textDocument/didOpen":
        td = (msg.get("params") or {}).get("textDocument") or {}
        uri = td.get("uri") or ""
        text = td.get("text") or ""
        docs[uri] = text
        publish(uri, text)
        continue

    if method == "textDocument/didChange":
        params = msg.get("params") or {}
        td = params.get("textDocument") or {}
        uri = td.get("uri") or ""
        changes = params.get("contentChanges") or []
        text = (changes[0].get("text") if changes else "") or ""
        docs[uri] = text
        publish(uri, text)
        continue

    if method == "textDocument/didClose":
        continue

    if method == "textDocument/completion":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "isIncomplete": False,
                "items": [
                    {"label": "alpha_completion", "detail": "fake completion"},
                    {"label": "beta_completion", "detail": "fake completion"},
                ],
            },
        })
        continue

    if method == "textDocument/definition":
        uri = ((msg.get("params") or {}).get("textDocument") or {}).get("uri") or ""
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": [
                {
                    "uri": uri,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1},
                    },
                }
            ],
        })
        continue

    if method == "shutdown":
        send({"jsonrpc": "2.0", "id": req_id, "result": None})
        continue

    if method == "exit":
        break

    # fallback agar request lain tidak menggantung
    if req_id is not None:
        send({"jsonrpc": "2.0", "id": req_id, "result": None})
'''


def fake_lsp_command():
    return [sys.executable, "-u", "-c", FAKE_LSP_SERVER_CODE]


print("\n[1] LocalLspClient with fake server")
tmp = tempfile.NamedTemporaryFile(prefix="moko_lsp_", suffix=".py", delete=False)
tmp_path = tmp.name
tmp.close()

client = LocalLspClient(
    workspace_dir=os.path.dirname(tmp_path),
    server_commands={"python": fake_lsp_command()},
)

ok(client.sync_document(tmp_path, "x = 1\n"), "sync_document starts fake LSP and sends didOpen")
ok(wait_until(lambda: client.get_diagnostics(tmp_path) == []), "clean code -> no diagnostics")

ok(client.sync_document(tmp_path, "value = ERROR\n"), "didChange sent to LSP")
ok(wait_until(lambda: len(client.get_diagnostics(tmp_path)) == 1), "diagnostic arrives from publishDiagnostics")
diag = client.get_diagnostics(tmp_path)[0]
ok(diag.source == "fake-lsp" and "ERROR" in diag.message, "diagnostic content parsed correctly")
ok("issue" in summarize_diagnostics([diag]).lower(), "diagnostic summary generated")

completions = client.request_completion(tmp_path, "a", 0, 1)
ok(any(it.get("label") == "alpha_completion" for it in completions), "completion request works")

definition = client.request_definition(tmp_path, "x = 1\n", 0, 0)
ok(isinstance(definition, dict) and definition.get("line") == 0, "definition request works")

client.stop()


print("\n[2] Negative path (server missing)")
dead = LocalLspClient(server_commands={"python": ["__moko_missing_lsp_binary__"]})
ok(dead.sync_document(tmp_path, "print(1)\n") is False, "missing server binary -> safe fallback")
ok(dead.request_completion(tmp_path, "x", 0, 0) == [], "completion fallback empty")
ok(dead.request_definition(tmp_path, "x", 0, 0) is None, "definition fallback None")
dead.stop()


print("\n[3] EditorPanel integration (offscreen)")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PyQt6.QtWidgets import QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("  ⚠ PyQt6 not available in this venv — GUI smoke test skipped")

if HAS_QT:
    app = QApplication.instance() or QApplication(sys.argv)
    from moko_ui.panels.editor_panel import EditorPanel

    panel = EditorPanel(lsp_server_overrides={"python": fake_lsp_command()})
    panel.set_content("value = ERROR\n", tmp_path)
    panel._sync_lsp_document()

    def _pump_qt():
        app.processEvents()
        panel._pull_lsp_diagnostics()
        app.processEvents()

    ok(wait_until(lambda: len(panel.get_lsp_issues()) == 1, tick=_pump_qt),
       "EditorPanel receives LSP diagnostics")
    panel._run_structure_check()
    ok("L:1" in panel._lbl_struct.text() or "issue" in panel._lbl_struct.text().lower(),
       "footer shows merged issue indicator")

    ctx = panel.get_ai_context_summary()
    ok("[LSP]" in ctx and "[Structure]" in ctx, "AI context contains structure + diagnostics blocks")

    completions = panel.request_completion_items()
    ok(any(it.get("label") == "alpha_completion" for it in completions), "EditorPanel completion API works")

    location = panel.request_definition_location()
    ok(location is not None and location.get("path") == tmp_path, "EditorPanel definition API works")

    panel.deleteLater()
    app.processEvents()


print("\n[4] Editor context bridge helpers")
ok(is_code_or_ide_query("tolong perbaiki bug di fungsi login"),
   "code query detector: positive case")
ok(not is_code_or_ide_query("halo moko, apa kabar hari ini"),
   "code query detector: negative case")

clamped = clamp_editor_context("A" * 2400, max_chars=120)
ok("truncated" in clamped.lower() and len(clamped) < 200,
   "editor context is clamped with truncation marker")

plain_prompt = build_cognitive_prompt_with_editor_context(
    "halo moko",
    "File: app.py\n[Structure]\n- issue",
)
ok(plain_prompt == "halo moko", "non-coding chat keeps original prompt")

bridged_prompt = build_cognitive_prompt_with_editor_context(
    "tolong fix error syntax di file ini",
    "File: app.py\n[Structure]\n- Unclosed bracket\n[LSP]\n- Line 3: error",
    max_context_chars=300,
)
ok("[MOKO_EDITOR_CONTEXT]" in bridged_prompt and "File: app.py" in bridged_prompt,
   "coding chat includes editor context block")

empty_ctx_prompt = build_cognitive_prompt_with_editor_context(
    "tolong refactor fungsi ini",
    "",
)
ok(empty_ctx_prompt == "tolong refactor fungsi ini",
   "empty editor context safely falls back to original prompt")


print("\n[5] CausalVisualFlowCompressor (DeepSeek-OCR 2 Style)")
# Test 1: Empty or no issues code compression
comp_empty = CausalVisualFlowCompressor.compress(
    code="def hello():\n    print('world')\n",
    structure_issues=[],
    lsp_issues=[],
    active_path="hello.py",
    language="python"
)
ok("DEEPSEEK-OCR 2 CAUSAL VISUAL FLOW MAP" in comp_empty, "compressor maps empty-issue code successfully")
ok("hello.py" in comp_empty and "python" in comp_empty, "compressor metadata matches")

# Test 2: Code with structure & lsp issues
from moko_ui.code_structure import StructureIssue
from moko_ui.lsp_client import LspDiagnostic

struct_issue = StructureIssue(
    kind="unclosed_bracket",
    message="Bracket ')' tidak ditutup",
    line=1,
    col=10,
    end_line=1,
    end_col=11
)
lsp_diagnostic = LspDiagnostic(
    message="Undefined variable ERROR",
    line=2,
    col=4,
    end_line=2,
    end_col=9,
    severity="error",
    source="fake-lsp"
)

comp_issues = CausalVisualFlowCompressor.compress(
    code="def test():\n    val = (\n    val = ERROR\n    return val\n",
    structure_issues=[struct_issue],
    lsp_issues=[lsp_diagnostic],
    active_path="test_func.py",
    language="python"
)

ok("COORD_2D" in comp_issues, "compressor places 2D Bounding Boxes around focal zones")
ok("OPTICAL_ANCHOR" in comp_issues, "compressor charts issue locations with 2D bounds")
ok("Bracket ')' tidak ditutup" in comp_issues and "ERROR" in comp_issues, "issue descriptions successfully encoded")
ok("DECODER_DIRECTIVE" in comp_issues, "compressor emits decoder directives")


if os.path.exists(tmp_path):
    os.remove(tmp_path)

print(f"\n{'=' * 60}\nALL TESTS PASSED ✔ ({PASSED} assertions)\n{'=' * 60}")
