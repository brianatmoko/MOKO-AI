"""
MOKO IDE Backend — FastAPI Router
Endpoint:
  GET  /health            → Status daemon
  GET  /ide/files?path=   → List isi direktori
  GET  /ide/read?path=    → Baca isi file
  POST /ide/write         → Tulis/simpan file
  POST /ide/run           → Jalankan file (Python/Bash/JS)
  POST /ide/ai_generate   → Generate kode via MOKO AI
"""

import os
import re
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/ide", tags=["MOKO IDE"])

# ── Config ────────────────────────────────────────────────────────────────────
# Batasi akses file hanya ke dalam HOME (keamanan dasar)
HOME_DIR   = Path.home()
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

# File/folder yang diblacklist dari explorer
HIDDEN_PATTERNS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                   ".mypy_cache", ".pytest_cache", "*.gguf", "*.bin", "*.pyc"}

# ── Models ────────────────────────────────────────────────────────────────────
class WriteRequest(BaseModel):
    path: str
    content: str

class RunRequest(BaseModel):
    path: str
    lang: str = "python"  # python | bash | javascript | text

class AiGenerateRequest(BaseModel):
    prompt: str
    lang: str = "python"
    current_code: Optional[str] = ""

# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_path(raw: str) -> Path:
    """Validasi path agar tidak keluar dari HOME."""
    p = Path(raw).expanduser().resolve()
    try:
        p.relative_to(HOME_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="Akses di luar direktori home dilarang.")
    return p

def _is_hidden(name: str) -> bool:
    return name.startswith('.') or name in HIDDEN_PATTERNS

def _extract_code_block(text: str, lang: str) -> Optional[str]:
    """Ekstrak kode dari blok markdown ```lang ... ```"""
    # Coba ekstrak code fence
    patterns = [
        rf"```{lang}\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    # Jika tidak ada fence, kembalikan seluruh teks (mungkin sudah bersih)
    if len(text.strip()) > 10 and '\n' in text:
        return text.strip()
    return None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/files")
async def list_files(path: str = Query(default=str(Path.home()))):
    """List isi direktori."""
    dir_path = _safe_path(path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Direktori tidak ditemukan.")

    entries = []
    try:
        items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for item in items:
            if _is_hidden(item.name):
                continue
            entries.append({
                "name":   item.name,
                "path":   str(item.resolve()),
                "is_dir": item.is_dir(),
                "size":   item.stat().st_size if item.is_file() else None
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail="Akses ditolak ke direktori ini.")

    return {"path": str(dir_path), "entries": entries}


@router.get("/read")
async def read_file(path: str = Query(...)):
    """Baca isi sebuah file teks."""
    file_path = _safe_path(path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")

    size = file_path.stat().st_size
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File terlalu besar ({size // 1024} KB). Maks 2 MB.")

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca file: {e}")

    return {"path": str(file_path), "content": content, "size": size}


@router.post("/write")
async def write_file(req: WriteRequest):
    """Tulis/simpan konten ke file."""
    file_path = _safe_path(req.path)
    # Buat direktori parent jika belum ada
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(req.content, encoding="utf-8")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Tidak ada izin menulis ke lokasi ini.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan: {e}")

    return {"ok": True, "path": str(file_path), "bytes": len(req.content.encode())}


@router.post("/run")
async def run_code(req: RunRequest):
    """Jalankan file dan kembalikan stdout/stderr."""
    file_path = _safe_path(req.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan. Simpan terlebih dahulu.")

    lang_cmd = {
        "python":     ["python3", str(file_path)],
        "bash":       ["bash", str(file_path)],
        "javascript": ["node", str(file_path)],
        "text":       None
    }

    cmd = lang_cmd.get(req.lang)
    if cmd is None:
        return {"stdout": "(File teks — tidak ada yang dijalankan)", "stderr": "", "returncode": 0}

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(file_path.parent),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return {
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "cmd":        " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "⏱ Timeout: proses melebihi 30 detik.", "returncode": -1}
    except FileNotFoundError as e:
        return {"stdout": "", "stderr": f"Interpreter tidak ditemukan: {e}", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": f"Error menjalankan: {e}", "returncode": -1}


@router.post("/ai_generate")
async def ai_generate(req: AiGenerateRequest):
    """
    Generate kode menggunakan MOKO AI pipeline.
    Memanggil CoreNode + AnalystNode agar tetap menggunakan MOKO logic.
    """
    try:
        # Import pipeline MOKO (import di sini agar tidak circular saat startup)
        from moko_agents.core_node import CoreNode
        from moko_agents.analyst_node import AnalystNode
        from moko_memory.disk_manager import DiskManager
        from moko_config import settings

        disk_mgr = DiskManager(settings.WORKSPACE_DIR)
        analyst  = AnalystNode(disk_mgr)
        core     = CoreNode(disk_mgr)

        # Susun prompt IDE
        lang_names = {
            "python": "Python", "bash": "Bash/Shell",
            "javascript": "JavaScript (Node.js)", "text": "Plain Text"
        }
        lang_label = lang_names.get(req.lang, req.lang)

        ide_prompt = (
            f"[MOKO IDE MODE — Code Generator]\n"
            f"Bahasa: {lang_label}\n"
            f"Permintaan: {req.prompt}\n\n"
            f"Tulis kode {lang_label} yang:\n"
            f"1. Lengkap dan langsung bisa dijalankan\n"
            f"2. Diberi komentar singkat di setiap bagian penting\n"
            f"3. Hanya output kode dalam satu blok kode\n"
            f"Jangan tambahkan penjelasan di luar blok kode."
        )

        if req.current_code and len(req.current_code.strip()) > 10:
            ide_prompt += f"\n\n[Kode yang sudah ada untuk dimodifikasi/diperbaiki]:\n```\n{req.current_code[:3000]}\n```"

        # Jalankan pipeline
        analyst_thoughts = await asyncio.to_thread(analyst.deep_think_loop, ide_prompt)
        raw_response     = await asyncio.to_thread(core.amplify_response, ide_prompt, analyst_thoughts)

        # Ekstrak kode
        code = _extract_code_block(raw_response, req.lang)

        if code:
            # Tentukan nama file default berdasarkan bahasa
            ext_map = {"python": "py", "bash": "sh", "javascript": "js", "text": "txt"}
            ext = ext_map.get(req.lang, "txt")
            filename = f"moko_generated.{ext}"
            return {"ok": True, "code": code, "filename": filename, "raw": raw_response}
        else:
            return {"ok": False, "code": None, "raw": raw_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generate error: {e}")
