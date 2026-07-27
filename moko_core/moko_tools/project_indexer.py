import os
import ast
from pathlib import Path

class ProjectIndexer:
    """
    MOKO Project Indexer (Prototipe OPI)
    Menganalisis struktur proyek secara struktural menggunakan AST.
    """
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.index = {}

    def scan(self, exclude_dirs=None):
        if exclude_dirs is None:
            exclude_dirs = {'.git', '__pycache__', '.moko_omni', 'venv', 'node_modules', 'lib', 'include', 'bin'}
        
        print(f"[*] Scanning project structure in: {self.root_path}")
        
        for path in self.root_path.rglob('*.py'):
            # Cek apakah path mengandung direktori yang di-exclude
            if any(part in exclude_dirs for part in path.parts):
                continue
                
            relative_path = path.relative_to(self.root_path)
            self.index[str(relative_path)] = self._analyze_file(path)
            
        return self.index

    def _analyze_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            return {"error": str(e)}

        file_data = {
            "classes": [],
            "functions": [],
            "imports": []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                file_data["classes"].append({
                    "name": node.name,
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "lineno": node.lineno
                })
            elif isinstance(node, ast.FunctionDef):
                # Hanya ambil fungsi top-level (bukan method dalam class)
                # Sederhananya, cek apakah parent-nya adalah Module
                # Untuk prototipe ini kita ambil semua tapi bisa difilter nanti
                file_data["functions"].append({
                    "name": node.name,
                    "lineno": node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_data["imports"].append(alias.name)
                else:
                    file_data["imports"].append(f"{node.module}.{node.names[0].name}" if node.module else node.names[0].name)

        return file_data

    def generate_summary(self):
        summary = []
        summary.append("# MOKO Project Structure Summary\n")
        
        for file_path, data in self.index.items():
            summary.append(f"## File: `{file_path}`")
            if data.get("classes"):
                summary.append("  - **Classes**:")
                for cls in data["classes"]:
                    summary.append(f"    - `{cls['name']}` (Methods: {', '.join(cls['methods'])})")
            if data.get("functions"):
                summary.append("  - **Top-level Functions**:")
                # Filter out functions that are already listed as methods (simple check)
                methods = set()
                for cls in data["classes"]:
                    methods.update(cls["methods"])
                
                funcs = [f['name'] for f in data["functions"] if f['name'] not in methods]
                if funcs:
                    summary.append(f"    - {', '.join(funcs)}")
            summary.append("")
            
        return "\n".join(summary)

if __name__ == "__main__":
    # Test run pada project root
    indexer = ProjectIndexer("/home/brianatmokoo/Documents/Linux/MOKO_OS_Project")
    indexer.scan()
    print(indexer.generate_summary())
