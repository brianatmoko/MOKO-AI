from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from hashlib import sha1
import argparse
import json
from pathlib import Path
import re

from moko_code_knowledge import CodeKnowledgeBase, KnowledgeSnippet


DEFAULT_CALCULATOR_TEMPLATE = """import ast
import math
import operator


ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def evaluate_expression(expr: str, ans: float = 0.0) -> float:
    expr = expr.strip()
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body, {"ans": ans, "pi": math.pi})


def _eval(node, variables):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id.lower() in variables:
        return float(variables[node.id.lower()])
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        left = _eval(node.left, variables)
        right = _eval(node.right, variables)
        return float(ALLOWED_OPS[type(node.op)](left, right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() == "sqrt":
        args = [_eval(arg, variables) for arg in node.args]
        return float(math.sqrt(*args))
    raise ValueError("Ekspresi tidak diizinkan")


def kalkulator():
    # Kalkulator sederhana MOKO Playground.
    print("=" * 40)
    print("  🔢 MOKO Kalkulator Playground")
    print("=" * 40)
    print("Operasi: +  -  *  /  **  %")
    print("Ketik 'q' untuk keluar\\n")

    ans = 0.0
    while True:
        try:
            expr = input(">>> " ).strip()
            if not expr or expr.lower() in ("q", "quit", "exit", "keluar"):
                print("Sampai jumpa!")
                break
            ans = evaluate_expression(expr, ans=ans)
            print(f"    = {ans:g}")
        except ZeroDivisionError:
            print("    ⚠ Tidak bisa dibagi nol!")
        except Exception as e:
            print(f"    ⚠ Error: {e}")


if __name__ == "__main__":
    kalkulator()
"""


STOPWORDS = {
    "buat",
    "bikin",
    "yang",
    "dengan",
    "untuk",
    "dan",
    "atau",
    "serta",
    "agar",
    "dari",
    "ke",
    "di",
    "tanpa",
    "lebih",
    "sistem",
    "template",
    "kode",
}


@dataclass(frozen=True)
class TemplateSample:
    template_id: str
    task_type: str
    code: str
    notes: str
    keywords: list[str]
    origin: str
    parent_template_id: str | None
    created_at: str


@dataclass(frozen=True)
class LearningRecord:
    record_id: str
    task_type: str
    intent: str
    source_template_id: str
    generated_template_id: str | None
    mutations: list[str]
    generated_code: str
    created_at: str
    retrieval_focus: list[str] = field(default_factory=list)


class TemplateLearningEngine:
    def __init__(
        self,
        storage_dir: str | Path = "riset/data/template_learning",
        *,
        knowledge_base: CodeKnowledgeBase | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.template_library_path = self.storage_dir / "template_library.jsonl"
        self.learning_dataset_path = self.storage_dir / "template_learning_dataset.jsonl"
        self.knowledge_base = knowledge_base or CodeKnowledgeBase()

    def register_template(
        self,
        task_type: str,
        code: str,
        *,
        notes: str = "",
        keywords: list[str] | None = None,
        origin: str = "human",
        parent_template_id: str | None = None,
    ) -> TemplateSample:
        normalized_code = code.strip()
        if not normalized_code:
            raise ValueError("Template code tidak boleh kosong.")

        normalized_task = task_type.strip().lower() or "general"
        created_at = self._now_iso()
        template_hash = sha1(f"{normalized_task}:{normalized_code}".encode("utf-8")).hexdigest()[:12]
        template_id = f"{normalized_task}:{template_hash}"

        sample = TemplateSample(
            template_id=template_id,
            task_type=normalized_task,
            code=normalized_code,
            notes=notes.strip(),
            keywords=sorted(set((keywords or []) + self._tokenize(code))),
            origin=origin,
            parent_template_id=parent_template_id,
            created_at=created_at,
        )

        if not self._template_exists(template_id):
            self._append_jsonl(self.template_library_path, asdict(sample))

        return sample

    def list_templates(self, task_type: str | None = None) -> list[TemplateSample]:
        items = [TemplateSample(**payload) for payload in self._read_jsonl(self.template_library_path)]
        if task_type is None:
            return items
        normalized_task = task_type.strip().lower()
        return [item for item in items if item.task_type == normalized_task]

    def list_learning_records(self) -> list[LearningRecord]:
        return [LearningRecord(**payload) for payload in self._read_jsonl(self.learning_dataset_path)]

    def generate_code(
        self,
        intent: str,
        task_type: str,
        *,
        promote_generated_template: bool = True,
    ) -> tuple[str, LearningRecord]:
        normalized_intent = intent.strip()
        if not normalized_intent:
            raise ValueError("Intent tidak boleh kosong.")

        normalized_task = task_type.strip().lower() or "general"
        source_template = self._select_template(normalized_task, normalized_intent)
        retrieval_focus = self._build_retrieval_focus(normalized_intent, normalized_task)
        generated_code, mutations = self._personalize_template(
            source_template.code,
            normalized_intent,
            normalized_task,
            retrieval_focus,
        )

        generated_template_id = None
        if promote_generated_template:
            promoted = self.register_template(
                normalized_task,
                generated_code,
                notes=f"Auto-learned dari intent: {normalized_intent}",
                keywords=self._tokenize(normalized_intent),
                origin="generated",
                parent_template_id=source_template.template_id,
            )
            generated_template_id = promoted.template_id

        record = LearningRecord(
            record_id=sha1(f"{normalized_task}:{normalized_intent}:{self._now_iso()}".encode("utf-8")).hexdigest()[:16],
            task_type=normalized_task,
            intent=normalized_intent,
            source_template_id=source_template.template_id,
            generated_template_id=generated_template_id,
            mutations=mutations,
            generated_code=generated_code,
            created_at=self._now_iso(),
            retrieval_focus=retrieval_focus,
        )
        self._append_jsonl(self.learning_dataset_path, asdict(record))
        return generated_code, record

    def _select_template(self, task_type: str, intent: str) -> TemplateSample:
        candidates = self.list_templates(task_type)
        if not candidates:
            return self._builtin_template(task_type)

        intent_tokens = set(self._tokenize(intent))
        best = max(candidates, key=lambda candidate: self._score_template(candidate, intent_tokens))
        return best

    def _score_template(self, sample: TemplateSample, intent_tokens: set[str]) -> int:
        keyword_tokens = set(sample.keywords)
        code_tokens = set(self._tokenize(sample.code))
        return (3 * len(intent_tokens & keyword_tokens)) + len(intent_tokens & code_tokens)

    def _personalize_template(
        self,
        code: str,
        intent: str,
        task_type: str,
        retrieval_focus: list[str],
    ) -> tuple[str, list[str]]:
        working_code = code.strip()
        mutations: list[str] = []

        descriptor = self._build_descriptor(intent)
        safe_descriptor = descriptor.replace('"', "'")
        if "MOKO Kalkulator Playground" in working_code:
            working_code = working_code.replace("MOKO Kalkulator Playground", f"MOKO {safe_descriptor}", 1)
            mutations.append("update_title")

        target_name = None
        if re.search(r"def\s+kalkulator\s*\(", working_code):
            target_name = "kalkulator"
        else:
            function_match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", working_code)
            if function_match:
                target_name = function_match.group(1)

        if target_name:
            new_name = self._build_function_name(task_type, intent)
            if target_name != new_name:
                working_code = re.sub(rf"\b{target_name}\b", new_name, working_code)
                mutations.append("rename_function")

        operations = self._extract_operations(intent)
        if operations:
            operation_text = "  ".join(operations)
            replaced = re.sub(
                r'print\("Operasi:\s*[^\"]*"\)',
                f'print("Operasi: {operation_text}")',
                working_code,
                count=1,
            )
            if replaced != working_code:
                working_code = replaced
                mutations.append("update_operations")

        knowledge_snippets = self._select_knowledge_snippets(task_type, retrieval_focus)
        knowledge_sources: list[str] = []
        if knowledge_snippets:
            helper_codes = [snippet.code.strip() for snippet in knowledge_snippets]
            enriched = self._inject_helper_snippets(working_code, helper_codes)
            if enriched != working_code:
                required_imports = [
                    statement
                    for snippet in knowledge_snippets
                    for statement in snippet.requires_imports
                ]
                working_code = self._ensure_imports(enriched, required_imports)
                knowledge_sources = [snippet.domain for snippet in knowledge_snippets]
                mutations.append("inject_formula_knowledge")

        focus_text = ", ".join(retrieval_focus) if retrieval_focus else "none"
        sources_text = ", ".join(knowledge_sources) if knowledge_sources else "none"

        metadata = (
            "# Auto-generated by MOKO Template Learning Engine\n"
            f"# intent: {intent}\n"
            f"# retrieval_focus: {focus_text}\n"
            f"# knowledge_sources: {sources_text}\n"
            "# strategy: transform-template-not-copy\n\n"
        )
        if not working_code.startswith("# Auto-generated by MOKO Template Learning Engine"):
            working_code = metadata + working_code
            mutations.append("inject_metadata")

        if working_code.strip() == code.strip():
            working_code = f"{working_code}\n\n# mutation_guard: personalized_output"
            mutations.append("mutation_guard")

        return working_code, mutations

    def _build_descriptor(self, intent: str) -> str:
        tokens = [token for token in self._tokenize(intent) if token not in STOPWORDS]
        if not tokens:
            return "Playground Personal"
        summary = " ".join(tokens[:3])
        return f"Playground {summary.title()}"

    def _build_function_name(self, task_type: str, intent: str) -> str:
        tokens = [
            token
            for token in self._tokenize(intent)
            if token not in STOPWORDS and token != task_type
        ]
        suffix = "_".join(tokens[:2]) if tokens else "custom"
        candidate = f"{task_type}_{suffix}"
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", candidate)
        candidate = re.sub(r"_+", "_", candidate).strip("_")
        if not candidate:
            return "moko_generated"
        if candidate[0].isdigit():
            candidate = f"moko_{candidate}"
        return candidate

    def _build_retrieval_focus(self, intent: str, task_type: str) -> list[str]:
        focus: list[str] = []
        for token in self._tokenize(intent):
            if token in STOPWORDS or token == task_type:
                continue
            if token not in focus:
                focus.append(token)

        if not focus:
            return [task_type]
        return focus[:8]

    def _extract_operations(self, intent: str) -> list[str]:
        lower_intent = intent.lower()
        requested: list[str] = []

        mapping = [
            ("tambah", "+"),
            ("kurang", "-"),
            ("kali", "*"),
            ("bagi", "/"),
            ("pangkat", "**"),
            ("mod", "%"),
            ("modulo", "%"),
        ]
        for keyword, symbol in mapping:
            if keyword in lower_intent and symbol not in requested:
                requested.append(symbol)

        if not requested:
            requested = ["+", "-", "*", "/"]

        return requested

    def _select_knowledge_snippets(
        self,
        task_type: str,
        retrieval_focus: list[str],
    ) -> list[KnowledgeSnippet]:
        focus_tokens = set(retrieval_focus)
        focus_tokens.add(task_type)
        return self.knowledge_base.retrieve(focus_tokens)

    def _inject_helper_snippets(self, code: str, snippets: list[str]) -> str:
        helper_block = "\n\n".join(snippets).strip()
        if not helper_block:
            return code

        entrypoint = '\n\nif __name__ == "__main__":'
        if entrypoint in code:
            return code.replace(entrypoint, f"\n\n{helper_block}{entrypoint}", 1)
        return f"{code}\n\n{helper_block}\n"

    def _ensure_imports(self, code: str, imports: list[str]) -> str:
        missing: list[str] = []
        for statement in imports:
            if not statement:
                continue
            if re.search(rf"(?m)^{re.escape(statement)}\b", code):
                continue
            if statement not in missing:
                missing.append(statement)

        if not missing:
            return code
        return "\n".join(missing) + "\n" + code

    def _builtin_template(self, task_type: str) -> TemplateSample:
        template_code = DEFAULT_CALCULATOR_TEMPLATE if task_type == "kalkulator" else "def run():\n    return 'TODO'\n"
        return TemplateSample(
            template_id=f"builtin:{task_type}",
            task_type=task_type,
            code=template_code.strip(),
            notes="Builtin fallback template",
            keywords=self._tokenize(template_code),
            origin="builtin",
            parent_template_id=None,
            created_at=self._now_iso(),
        )

    def _template_exists(self, template_id: str) -> bool:
        return any(item.get("template_id") == template_id for item in self._read_jsonl(self.template_library_path))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]{2,}", text.lower())

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            rows.append(json.loads(cleaned))
        return rows

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Belajar dari template lalu personalisasi output, bukan menyalin mentah."
    )
    parser.add_argument("--intent", required=True, help="Instruksi user, misalnya: buat kalkulator pecahan")
    parser.add_argument("--task-type", default="kalkulator", help="Jenis task, contoh: kalkulator")
    parser.add_argument("--storage", default="riset/data/template_learning", help="Folder penyimpanan JSONL")
    parser.add_argument("--output", default="", help="Path file output kode hasil generasi")
    args = parser.parse_args()

    engine = TemplateLearningEngine(args.storage)
    generated_code, record = engine.generate_code(args.intent, args.task_type)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated_code + "\n", encoding="utf-8")
        print(f"[+] Kode tersimpan di: {output_path}")

    print("\n=== GENERATED CODE ===\n")
    print(generated_code)
    print("\n=== LEARNING RECORD ===")
    print(f"record_id={record.record_id}")
    print(f"source_template_id={record.source_template_id}")
    print(f"generated_template_id={record.generated_template_id}")
    print(f"retrieval_focus={record.retrieval_focus}")
    print(f"mutations={record.mutations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
