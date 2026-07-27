"""
MOKO NeuroMath: CognitiveMapBuilder — Spatial Topology & Concept Maps
=====================================================================
Berdasarkan:
  - Lisman et al.: Hippocampal cognitive maps.
  - Concept topology in concept space representation.
  - Dijkstra pathfinding on concepts as biological reasoning chains.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from moko_config import settings

class CognitiveMapBuilder:
    """
    CognitiveMapBuilder — Membangun dan mengelola Peta Konseptual MOKO OS.
    Menyediakan visualisasi topologi konsep, navigasi jalur (pathfinding),
    serta deteksi skema konseptual.
    """
    def __init__(self, workspace_dir: Optional[str] = None):
        workspace = Path(workspace_dir or settings.WORKSPACE_DIR)
        self.map_path = workspace / ".math_omni" / "cognitive_map.json"
        
        # Graph: {node_a: {node_b: weight_ab, node_c: weight_ac}}
        self.graph: Dict[str, Dict[str, float]] = self._load_map()

    def _load_map(self) -> Dict[str, Dict[str, float]]:
        if self.map_path.exists():
            try:
                return json.loads(self.map_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_map(self):
        try:
            self.map_path.parent.mkdir(parents=True, exist_ok=True)
            self.map_path.write_text(json.dumps(self.graph, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def add_concept_link(self, concept_a: str, concept_b: str, weight: float):
        """Menambahkan hubungan konseptual ke peta (undirected graph)."""
        if not concept_a or not concept_b or concept_a == concept_b:
            return

        # Pastikan node A terdaftar
        if concept_a not in self.graph:
            self.graph[concept_a] = {}
        # Pastikan node B terdaftar
        if concept_b not in self.graph:
            self.graph[concept_b] = {}

        # Update bobot (ambil max jika link sudah ada)
        self.graph[concept_a][concept_b] = round(max(self.graph[concept_a].get(concept_b, 0.0), weight), 4)
        self.graph[concept_b][concept_a] = round(max(self.graph[concept_b].get(concept_a, 0.0), weight), 4)
        self.save_map()

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Dijkstra's Pathfinding Algorithm.
        Mencari rantai hubungan konseptual terpendek antara dua konsep.
        Bobot jarak = 1.0 / weight.
        """
        if start not in self.graph or end not in self.graph:
            return None

        # Dijkstra
        distances: Dict[str, float] = {node: float('inf') for node in self.graph}
        distances[start] = 0.0
        previous: Dict[str, Optional[str]] = {node: None for node in self.graph}
        unvisited: Set[str] = set(self.graph.keys())

        while unvisited:
            # Cari node dengan distance terkecil di unvisited
            current = min(unvisited, key=lambda node: distances[node])
            if distances[current] == float('inf'):
                break

            if current == end:
                break

            unvisited.remove(current)

            for neighbor, weight in self.graph[current].items():
                if neighbor not in unvisited:
                    continue
                # Jarak inversi terhadap bobot
                dist_factor = 1.0 / max(0.01, weight)
                new_dist = distances[current] + dist_factor
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current

        # Reconstruct path
        path: List[str] = []
        curr = end
        while curr is not None:
            path.insert(0, curr)
            curr = previous.get(curr)

        if path[0] == start:
            return path
        return None

    def detect_schemas(self, threshold: float = 0.5) -> List[List[str]]:
        """
        Mendeteksi klaster/subgraf konseptual yang saling terhubung erat
        (diasosiasikan sebagai 'Skema Kognitif').
        """
        visited: Set[str] = set()
        schemas: List[List[str]] = []

        for node in self.graph:
            if node in visited:
                continue

            # BFS untuk mencari komponen terhubung
            schema: List[str] = []
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                schema.append(current)

                for neighbor, weight in self.graph[current].items():
                    if neighbor not in visited and weight >= threshold:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(schema) >= 2:
                schemas.append(schema)

        return schemas

# Singleton instance
cognitive_map_builder = CognitiveMapBuilder()
