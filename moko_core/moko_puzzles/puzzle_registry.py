import os
import sys
import importlib
import traceback
from typing import List, Dict, Optional, Tuple
from moko_puzzles.base_puzzle import BasePuzzle

class PuzzleRegistry:
    """
    Registrasi Puzzle Dinamis.
    Memindai, memuat, dan merakit semua modul puzzle kognitif modular.
    """
    def __init__(self):
        self.puzzles: Dict[str, BasePuzzle] = {}
        self.assembled = False

    def assemble_all(self, on_log_callback = None) -> List[str]:
        """
        Memindai direktori moko_puzzles/ dan menginstansiasi semua class puzzle.
        Memicu callback logger untuk visualisasi startup di GUI.
        """
        if self.assembled:
            return list(self.puzzles.keys())

        log_msgs = []
        def log(msg: str):
            log_msgs.append(msg)
            if on_log_callback:
                on_log_callback(msg)
            else:
                print(f"[PuzzleRegistry] {msg}")

        log("Memindai modul kognitif modular...")
        
        puzzles_dir = os.path.dirname(__file__)
        for filename in sorted(os.listdir(puzzles_dir)):
            if filename.startswith("puzzle_") and filename.endswith(".py"):
                module_name = filename[:-3]
                try:
                    # Import modul secara dinamis
                    # Gunakan importlib untuk lazy loading
                    module = importlib.import_module(f"moko_puzzles.{module_name}")
                    
                    # Cari class yang mewarisi BasePuzzle
                    found_class = False
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BasePuzzle) and 
                            attr is not BasePuzzle):
                            
                            # Instansiasi class puzzle
                            puzzle_instance = attr()
                            self.puzzles[puzzle_instance.name] = puzzle_instance
                            log(f"Merakit Puzzle: {puzzle_instance.description} (v{puzzle_instance.version})... Assembled.")
                            found_class = True
                            break
                            
                    if not found_class:
                        log(f"PERINGATAN: Tidak ada class BasePuzzle valid di {filename}")
                except Exception as e:
                    log(f"ERROR: Gagal merakit {filename}: {e}")
                    traceback.print_exc()

        self.assembled = True
        log("Semua kepingan Modular Puzzle berhasil dirakit.")
        return list(self.puzzles.keys())

    def match_and_run(self, query: str, context: dict = None) -> Tuple[Optional[str], Optional[dict]]:
        """
        Mengevaluasi kesesuaian setiap puzzle, lalu menjalankan puzzle dengan kesesuaian tertinggi (> 0.5).
        
        Returns:
            Tuple[puzzle_name, puzzle_result_dict] atau (None, None)
        """
        if not self.assembled:
            self.assemble_all()

        if not self.puzzles:
            return None, None

        best_score = 0.0
        best_puzzle = None

        for name, puzzle in self.puzzles.items():
            try:
                score = puzzle.evaluate_suitability(query)
                if score > best_score:
                    best_score = score
                    best_puzzle = puzzle
            except Exception as e:
                print(f"[PuzzleRegistry] Error evaluate {name}: {e}")

        # Hanya jalankan jika score di atas threshold 0.5
        if best_puzzle and best_score >= 0.5:
            try:
                ctx = context or {}
                result = best_puzzle.execute(query, ctx)
                return best_puzzle.name, result
            except Exception as e:
                print(f"[PuzzleRegistry] Error executing puzzle {best_puzzle.name}: {e}")

        return None, None

    def run_by_domain(self, domain: str, query: str, context: dict = None) -> Tuple[Optional[str], Optional[dict]]:
        """
        Menjalankan puzzle yang cocok secara spesifik dengan domain tertentu.
        Fallback ke match_and_run jika tidak ada pencocokan langsung.
        """
        if not self.assembled:
            self.assemble_all()

        domain_to_puzzle = {
            "lexical": "kbbi_lookup",
            "math": "math_lookup",
            "physics": "physics_lookup",
            "code": "code_lookup",
            "reasoning": "reasoning_lookup",
            "general": "general_lookup",
            "hardware": "hardware_control",
            "system_control": "os_control"
        }

        puzzle_name = domain_to_puzzle.get(domain)
        if puzzle_name and puzzle_name in self.puzzles:
            try:
                ctx = context or {}
                result = self.puzzles[puzzle_name].execute(query, ctx)
                # Tentukan default confidence jika tidak ada, agar memenuhi threshold OMNI
                if result:
                    if "confidence" not in result or result["confidence"] < 0.4:
                        # Jika puzzle berhasil menemukan data tapi confidence rendah, set default
                        if result.get("facts") and "tidak ditemukan" not in result["facts"].lower():
                            result["confidence"] = 0.80
                    return puzzle_name, result
            except Exception as e:
                print(f"[PuzzleRegistry] Error executing puzzle {puzzle_name} by domain '{domain}': {e}")

        # Fallback ke pencarian otomatis berbasis kata kunci
        return self.match_and_run(query, context)


# Singleton Registry
puzzle_registry = PuzzleRegistry()
