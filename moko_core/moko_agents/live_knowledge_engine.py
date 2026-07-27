"""
MOKO Live Knowledge Engine — Realtime Context Hot-Reload
=========================================================
Solution: MOKO AI gets fresh context from the knowledge base (.moko_omni/).
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
from collections import deque
from dataclasses import dataclass
from moko_config import settings

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


@dataclass
class KnowledgeDelta:
    """Delta change in knowledge base"""
    domain: str
    file_path: str
    change_type: str   # 'created' | 'modified' | 'deleted'
    timestamp: float
    size_bytes: int = 0


class LiveContextBuffer:
    """
    Knowledge context buffer updated in realtime.
    """

    def __init__(self, max_context_chars: int = 4000, max_entries: int = 30):
        self._lock = threading.RLock()
        self._entries: deque = deque(maxlen=max_entries)
        self._domain_snapshots: Dict[str, str] = {}
        self._last_rebuild: float = 0.0
        self._max_chars = max_context_chars
        self._delta_queue: deque = deque(maxlen=100)
        self._total_updates: int = 0

    def push_delta(self, delta: KnowledgeDelta):
        with self._lock:
            self._delta_queue.append(delta)
            self._total_updates += 1

    def update_domain_snapshot(self, domain: str, content: str):
        with self._lock:
            self._domain_snapshots[domain] = content[:self._max_chars // 4]
            self._last_rebuild = time.time()

    def get_live_context(self, query: str = "", max_chars: int = 2000) -> str:
        with self._lock:
            if not self._domain_snapshots:
                return ""

            q_lower = query.lower()
            domain_scores: Dict[str, float] = {}
            
            domain_keywords = {
                "math": ["matematika", "rumus", "integral", "kalkulus", "turunan", "aljabar"],
                "code": ["kode", "fungsi", "class", "python", "implementasi", "algoritma"],
                "physics": ["fisika", "gaya", "energi", "kinematika", "gerak", "momentum"],
                "security": ["keamanan", "enkripsi", "cipher", "serangan", "hacking"],
                "general": ["apa", "bagaimana", "kenapa", "siapa", "dimana", "kapan"],
            }
            
            for domain, snapshot in self._domain_snapshots.items():
                score = 0.1
                domain_base = domain.split('/')[0].lower()
                
                for kw_domain, keywords in domain_keywords.items():
                    if kw_domain in domain_base:
                        matches = sum(1 for kw in keywords if kw in q_lower)
                        score += matches * 0.3
                        break
                        
                domain_scores[domain] = score

            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            
            parts = []
            total_chars = 0
            
            for domain, _ in sorted_domains:
                snapshot = self._domain_snapshots.get(domain, "")
                if not snapshot:
                    continue
                    
                chunk = f"[Knowledge: {domain}]\n{snapshot[:600]}"
                if total_chars + len(chunk) > max_chars:
                    break
                parts.append(chunk)
                total_chars += len(chunk)

            if not parts:
                return ""
                
            header = f"=== LIVE KNOWLEDGE CONTEXT (Update #{self._total_updates}, {time.strftime('%H:%M:%S')}) ==="
            return header + "\n" + "\n\n".join(parts)

    @property
    def total_updates(self) -> int:
        with self._lock:
            return self._total_updates


class OmniWatcher(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    FileSystem event handler for OMNI knowledge base.
    """

    def __init__(self, buffer: LiveContextBuffer, root_path: Path):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        self.buffer = buffer
        self.root_path = root_path
        self._cooldown: Dict[str, float] = {}

    def _should_process(self, path: str) -> bool:
        now = time.time()
        last = self._cooldown.get(path, 0.0)
        if now - last < 2.0:
            return False
        self._cooldown[path] = now
        return True

    def _get_domain(self, file_path: str) -> str:
        path = Path(file_path)
        try:
            rel = path.relative_to(self.root_path)
            return str(rel.parts[0]) if rel.parts else "general"
        except ValueError:
            return "general"

    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            domain = self._get_domain(event.src_path)
            delta = KnowledgeDelta(
                domain=domain,
                file_path=event.src_path,
                change_type="created",
                timestamp=time.time(),
                size_bytes=os.path.getsize(event.src_path) if os.path.exists(event.src_path) else 0
            )
            self.buffer.push_delta(delta)
            threading.Thread(target=self._rebuild_domain_snapshot, args=(domain, event.src_path), daemon=True).start()

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            domain = self._get_domain(event.src_path)
            delta = KnowledgeDelta(
                domain=domain,
                file_path=event.src_path,
                change_type="modified",
                timestamp=time.time()
            )
            self.buffer.push_delta(delta)
            threading.Thread(target=self._rebuild_domain_snapshot, args=(domain, event.src_path), daemon=True).start()

    def _rebuild_domain_snapshot(self, domain: str, trigger_file: str):
        domain_dir = self.root_path / domain
        if not domain_dir.exists():
            domain_dir = Path(trigger_file).parent
            
        try:
            texts = []
            for txt_file in sorted(domain_dir.rglob("*.txt"), key=os.path.getmtime, reverse=True)[:5]:
                try:
                    content = txt_file.read_text(encoding='utf-8', errors='replace')
                    texts.append(f"[{txt_file.name}]\n{content[:600]}")
                except Exception:
                    pass
                    
            if texts:
                combined = "\n\n".join(texts[:3])
                self.buffer.update_domain_snapshot(domain, combined)
                print(f"  🔴 [LiveKnowledge] Hot-reload domain '{domain}' — {len(texts)} files")
        except Exception as e:
            print(f"  ⚠️ [LiveKnowledge] Rebuild error domain '{domain}': {e}")


class LiveKnowledgeEngine:
    """
    Main engine for realtime knowledge injection.
    """

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.buffer = LiveContextBuffer(max_context_chars=8000, max_entries=50)
        self._observer: Optional[object] = None
        self._is_running = False

    def start(self):
        if self._is_running:
            return

        omni_path = settings.OMNI_DIR
        crawler_path = self.workspace_root / "crawler_data"

        if not WATCHDOG_AVAILABLE:
            print("  ⚠️ [LiveKnowledge] watchdog not available — using polling")
            self._start_polling_fallback(omni_path, crawler_path)
            return

        self._observer = Observer()
        
        watch_added = False
        for watch_root in [omni_path, crawler_path]:
            if not watch_root.exists():
                continue
            
            try:
                handler = OmniWatcher(self.buffer, watch_root)
                self._observer.schedule(handler, str(watch_root), recursive=True if watch_root == crawler_path else False)
                watch_added = True
            except Exception:
                pass

        if watch_added:
            try:
                self._observer.start()
                self._is_running = True
                threading.Thread(target=self._initial_load, daemon=True).start()
                print(f"  ✅ [LiveKnowledge] Realtime Hot-Reload ACTIVE")
            except Exception as e:
                print(f"  ⚠️ [LiveKnowledge] Watchdog failed ({e}) — using polling")
                self._start_polling_fallback(omni_path, crawler_path)
        else:
            print("  ⚠️ [LiveKnowledge] No knowledge directories found")

    def _initial_load(self):
        omni_path = settings.OMNI_DIR
        if not omni_path.exists():
            return
            
        for domain_dir in sorted(omni_path.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            texts = []
            for txt_file in sorted(domain_dir.rglob("*.txt"), key=os.path.getmtime, reverse=True)[:3]:
                try:
                    content = txt_file.read_text(encoding='utf-8', errors='replace')
                    texts.append(content[:500])
                except Exception:
                    pass
            if texts:
                self.buffer.update_domain_snapshot(domain, "\n\n".join(texts))

    def _start_polling_fallback(self, *watch_paths):
        def _poll_loop():
            file_mtimes: Dict[str, float] = {}
            while True:
                for watch_path in watch_paths:
                    if not watch_path.exists():
                        continue
                    for f in watch_path.rglob("*.txt"):
                        try:
                            mtime = f.stat().st_mtime
                            key = str(f)
                            if key not in file_mtimes or file_mtimes[key] != mtime:
                                file_mtimes[key] = mtime
                                domain = f.parent.name
                                content = f.read_text(encoding='utf-8', errors='replace')[:500]
                                self.buffer.update_domain_snapshot(domain, content)
                        except Exception:
                            pass
                time.sleep(10)

        threading.Thread(target=_poll_loop, daemon=True).start()
        self._is_running = True

    def stop(self):
        if self._observer and WATCHDOG_AVAILABLE:
            try:
                self._observer.stop()
                self._observer.join()
            except Exception:
                pass
        self._is_running = False

    def get_fresh_context(self, query: str, max_chars: int = 2000) -> str:
        return self.buffer.get_live_context(query=query, max_chars=max_chars)


_live_engine: Optional[LiveKnowledgeEngine] = None

def get_live_knowledge_engine(workspace_root: str = None) -> LiveKnowledgeEngine:
    global _live_engine
    if _live_engine is None:
        if not workspace_root:
            workspace_root = str(settings.WORKSPACE_DIR)
        _live_engine = LiveKnowledgeEngine(workspace_root)
    return _live_engine
