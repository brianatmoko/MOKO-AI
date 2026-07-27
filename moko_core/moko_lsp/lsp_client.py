"""
lsp_client.py — Lightweight Local LSP Bridge untuk MOKO IDE (Fase 2)
====================================================================
Implementasi klien LSP sederhana berbasis JSON-RPC + stdio, tanpa
dependency eksternal tambahan.

Fitur utama:
  - Start/stop language server per bahasa (subprocess, local-only)
  - Sinkronisasi dokumen aktif (didOpen/didChange/didClose)
  - Terima publishDiagnostics
  - Request completion dan go-to-definition

Catatan:
  - Jika binary language server tidak tersedia, client fallback aman
    (tidak crash, hanya mengembalikan hasil kosong).
"""
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


# ─── Mapping ekstensi → language id LSP ──────────────────────────────────────
_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
}


# Kandidat server command (diuji berurutan sampai ada yang tersedia)
_DEFAULT_SERVER_COMMANDS = {
    "python": [
        ["pylsp"],
        ["pyright-langserver", "--stdio"],
    ],
    "javascript": [
        ["typescript-language-server", "--stdio"],
    ],
    "typescript": [
        ["typescript-language-server", "--stdio"],
    ],
    "json": [
        ["vscode-json-language-server", "--stdio"],
    ],
    "css": [
        ["vscode-css-language-server", "--stdio"],
    ],
    "html": [
        ["vscode-html-language-server", "--stdio"],
    ],
}

_SEVERITY_MAP = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}


def language_id_for_path(path: str) -> str:
    """Tebak language id LSP dari ekstensi file."""
    if not path:
        return ""
    ext = os.path.splitext(path)[1].lower()
    return _EXT_TO_LANGUAGE.get(ext, "")


def _path_to_uri(path: str) -> str:
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


def uri_to_path(uri: str) -> str:
    """Konversi file:// URI ke path lokal."""
    if not uri:
        return ""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return ""
    path = url2pathname(unquote(parsed.path or ""))
    # Windows: /C:/x -> C:/x
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


@dataclass
class LspDiagnostic:
    """Satu diagnostic LSP. Posisi 0-based (line, col)."""
    message: str
    line: int
    col: int
    end_line: int
    end_col: int
    severity: str = "error"
    source: str = ""
    code: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        src = f"{self.source}: " if self.source else ""
        return f"line {self.line + 1}: {src}{self.message}"


class LocalLspClient:
    """Klien LSP ringan untuk satu server aktif (sesuai bahasa dokumen aktif)."""

    def __init__(self, workspace_dir: str | None = None, server_commands: dict | None = None):
        self.workspace_dir = workspace_dir or os.getcwd()
        self.server_commands = server_commands or {}

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()

        self._io_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._diag_lock = threading.Lock()

        self._next_id = 1
        self._pending: dict[str, tuple[threading.Event, dict]] = {}
        self._diagnostics_by_uri: dict[str, list[LspDiagnostic]] = {}

        self._active_language = ""
        self._opened_uri = ""
        self._doc_version = 0
        self._last_text = ""

    # ─── Server / command helpers ────────────────────────────────────────────
    def _normalize_commands(self, value) -> list[list[str]]:
        if value is None:
            return []
        # Bentuk 1: ["pylsp"]
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return [value]
        # Bentuk 2: [["pylsp"], ["pyright-langserver", "--stdio"]]
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, list) and item and all(isinstance(x, str) for x in item):
                    out.append(item)
            return out
        return []

    def _command_exists(self, command: list[str]) -> bool:
        if not command:
            return False
        exe = command[0]
        if os.path.isabs(exe):
            return os.path.exists(exe) and os.access(exe, os.X_OK)
        return shutil.which(exe) is not None

    def _resolve_server_command(self, language: str) -> list[str] | None:
        candidates = self._normalize_commands(self.server_commands.get(language))
        if not candidates:
            candidates = self._normalize_commands(_DEFAULT_SERVER_COMMANDS.get(language))
        for cmd in candidates:
            if self._command_exists(cmd):
                return cmd
        return None

    def server_hint_for_language(self, language: str) -> str:
        """Hint command pertama untuk membantu user install server yang sesuai."""
        candidates = self._normalize_commands(self.server_commands.get(language))
        if not candidates:
            candidates = self._normalize_commands(_DEFAULT_SERVER_COMMANDS.get(language))
        if not candidates:
            return ""
        return " ".join(candidates[0])

    def is_active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ─── JSON-RPC core ────────────────────────────────────────────────────────
    def _next_request_id(self) -> str:
        with self._id_lock:
            rid = str(self._next_id)
            self._next_id += 1
        return rid

    def _send_payload(self, payload: dict) -> bool:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        try:
            with self._io_lock:
                proc.stdin.write(header)
                proc.stdin.write(data)
                proc.stdin.flush()
            return True
        except Exception:
            return False

    def _notify(self, method: str, params: dict | None = None) -> bool:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        return self._send_payload(payload)

    def _request(self, method: str, params: dict | None = None, timeout: float = 1.5):
        req_id = self._next_request_id()
        event = threading.Event()
        bucket = {"result": None, "error": None}
        with self._pending_lock:
            self._pending[req_id] = (event, bucket)

        ok = self._send_payload({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        })
        if not ok:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return None

        if not event.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return None

        if bucket["error"] is not None:
            return None
        return bucket["result"]

    def _read_message(self, stdout) -> dict | None:
        headers = {}
        while True:
            line = stdout.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                if headers:
                    break
                continue
            decoded = line.decode("ascii", errors="replace")
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        length_text = headers.get("content-length")
        if not length_text:
            return None
        try:
            length = int(length_text)
        except Exception:
            return None
        body = stdout.read(length)
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            return None

    def _reader_loop(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stdout = proc.stdout
        while not self._stop_reader.is_set():
            msg = self._read_message(stdout)
            if msg is None:
                if proc.poll() is not None:
                    break
                continue
            self._handle_message(msg)

    def _handle_message(self, message: dict):
        if "id" in message and ("result" in message or "error" in message):
            req_id = str(message.get("id"))
            with self._pending_lock:
                pending = self._pending.pop(req_id, None)
            if pending:
                event, bucket = pending
                bucket["result"] = message.get("result")
                bucket["error"] = message.get("error")
                event.set()
            return

        method = message.get("method")
        if method == "textDocument/publishDiagnostics":
            params = message.get("params") or {}
            uri = params.get("uri") or ""
            raw_diags = params.get("diagnostics") or []
            parsed = []
            for d in raw_diags:
                rng = d.get("range") or {}
                start = rng.get("start") or {}
                end = rng.get("end") or {}
                severity = _SEVERITY_MAP.get(d.get("severity"), "error")
                code = d.get("code")
                parsed.append(LspDiagnostic(
                    message=str(d.get("message") or "LSP diagnostic"),
                    line=int(start.get("line", 0)),
                    col=int(start.get("character", 0)),
                    end_line=int(end.get("line", start.get("line", 0))),
                    end_col=int(end.get("character", start.get("character", 0) + 1)),
                    severity=severity,
                    source=str(d.get("source") or ""),
                    code=str(code) if code is not None else "",
                ))
            with self._diag_lock:
                self._diagnostics_by_uri[uri] = parsed

    # ─── Lifecycle & document sync ───────────────────────────────────────────
    def _spawn_server(self, language: str) -> bool:
        cmd = self._resolve_server_command(language)
        if not cmd:
            return False

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.workspace_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception:
            self._proc = None
            return False

        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        init_result = self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "clientInfo": {"name": "moko-ide", "version": "v5"},
                "rootUri": _path_to_uri(self.workspace_dir),
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {"relatedInformation": False},
                        "completion": {"completionItem": {"snippetSupport": False}},
                        "definition": {},
                    }
                },
                "trace": "off",
            },
            timeout=1.8,
        )
        if init_result is None:
            self.stop()
            return False

        self._notify("initialized", {})
        self._active_language = language
        return True

    def _ensure_started(self, language: str) -> bool:
        if self.is_active() and self._active_language == language:
            return True
        self.stop()
        return self._spawn_server(language)

    def _open_document(self, path: str, language: str, text: str) -> bool:
        uri = _path_to_uri(path)
        if not uri:
            return False
        self._opened_uri = uri
        self._doc_version = 1
        self._last_text = text
        return self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language,
                "version": self._doc_version,
                "text": text,
            }
        })

    def _change_document(self, text: str) -> bool:
        if not self._opened_uri:
            return False
        if text == self._last_text:
            return True
        self._doc_version += 1
        self._last_text = text
        return self._notify("textDocument/didChange", {
            "textDocument": {
                "uri": self._opened_uri,
                "version": self._doc_version,
            },
            "contentChanges": [{"text": text}],
        })

    def close_document(self):
        if self._opened_uri:
            self._notify("textDocument/didClose", {
                "textDocument": {"uri": self._opened_uri}
            })
        with self._diag_lock:
            self._diagnostics_by_uri.pop(self._opened_uri, None)
        self._opened_uri = ""
        self._doc_version = 0
        self._last_text = ""

    def sync_document(self, path: str, text: str) -> bool:
        """
        Pastikan dokumen aktif tersinkron ke server.
        Return True jika server aktif dan sinkronisasi dikirim.
        """
        if not path:
            self.close_document()
            return False

        language = language_id_for_path(path)
        if not language:
            self.close_document()
            return False

        if not self._ensure_started(language):
            return False

        uri = _path_to_uri(path)
        if not uri:
            return False

        if self._opened_uri != uri:
            self.close_document()
            return self._open_document(path, language, text)
        return self._change_document(text)

    def get_diagnostics(self, path: str) -> list[LspDiagnostic]:
        uri = _path_to_uri(path)
        if not uri:
            return []
        with self._diag_lock:
            return list(self._diagnostics_by_uri.get(uri, []))

    # ─── Language features ───────────────────────────────────────────────────
    def request_completion(
        self,
        path: str,
        text: str,
        line: int,
        character: int,
        max_items: int = 20,
    ) -> list[dict]:
        if not self.sync_document(path, text):
            return []
        if not self._opened_uri:
            return []

        result = self._request(
            "textDocument/completion",
            {
                "textDocument": {"uri": self._opened_uri},
                "position": {"line": max(0, int(line)), "character": max(0, int(character))},
            },
            timeout=1.2,
        )
        if result is None:
            return []

        items = []
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict):
            items = result.get("items") or []

        out = []
        for item in items[:max_items]:
            out.append({
                "label": str(item.get("label") or ""),
                "detail": str(item.get("detail") or ""),
                "kind": item.get("kind"),
                "insert_text": item.get("insertText") or item.get("label") or "",
            })
        return out

    def request_definition(self, path: str, text: str, line: int, character: int) -> dict | None:
        if not self.sync_document(path, text):
            return None
        if not self._opened_uri:
            return None

        result = self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": self._opened_uri},
                "position": {"line": max(0, int(line)), "character": max(0, int(character))},
            },
            timeout=1.2,
        )
        if not result:
            return None

        location = None
        if isinstance(result, list):
            location = result[0] if result else None
        elif isinstance(result, dict):
            location = result
        if not isinstance(location, dict):
            return None

        # LocationLink
        if "targetUri" in location:
            uri = location.get("targetUri") or ""
            rng = location.get("targetSelectionRange") or location.get("targetRange") or {}
        else:
            uri = location.get("uri") or ""
            rng = location.get("range") or {}

        start = (rng.get("start") or {}) if isinstance(rng, dict) else {}
        end = (rng.get("end") or {}) if isinstance(rng, dict) else {}
        return {
            "uri": uri,
            "path": uri_to_path(uri),
            "line": int(start.get("line", 0)),
            "col": int(start.get("character", 0)),
            "end_line": int(end.get("line", start.get("line", 0))),
            "end_col": int(end.get("character", start.get("character", 0) + 1)),
        }

    # ─── Stop / cleanup ──────────────────────────────────────────────────────
    def stop(self):
        proc = self._proc
        if proc is None:
            return

        try:
            self.close_document()
            self._request("shutdown", {}, timeout=0.6)
            self._notify("exit", {})
        except Exception:
            pass

        self._stop_reader.set()

        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=0.8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        with self._pending_lock:
            pendings = list(self._pending.values())
            self._pending.clear()
        for event, bucket in pendings:
            bucket["error"] = {"message": "stopped"}
            event.set()

        self._proc = None
        self._active_language = ""
        self._opened_uri = ""
        self._doc_version = 0
        self._last_text = ""
        with self._diag_lock:
            self._diagnostics_by_uri.clear()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def summarize_diagnostics(issues: list[LspDiagnostic], max_items: int = 8) -> str:
    """Ringkasan diagnostics siap ditampilkan di UI/prompt."""
    if not issues:
        return "LSP diagnostics: OK"

    lines = [f"LSP diagnostics: {len(issues)} issue(s)"]
    for i, it in enumerate(issues[:max_items], start=1):
        src = f"{it.source}: " if it.source else ""
        lines.append(
            f"{i}. line {it.line + 1}, col {it.col + 1} [{it.severity}] {src}{it.message}"
        )
    hidden = len(issues) - max_items
    if hidden > 0:
        lines.append(f"... +{hidden} more issue(s)")
    return "\n".join(lines)
