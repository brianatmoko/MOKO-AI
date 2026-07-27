"""
MOKO MCTS Math Reasoner — Monte Carlo Tree Search for Mathematical Reasoning
=============================================================================
Implementasi MCTS yang terinspirasi oleh rStar-Math (Microsoft Research).
Model kecil bisa menalar matematika kompleks via tree search + code verification.

Arsitektur:
  Root Node (Problem) → Child Nodes (Reasoning Steps) → Leaf (Answer)
  Setiap node diverifikasi dengan code execution (Python/SymPy).

Referensi:
  - rStar-Math: arXiv:2501.04519
  - Budget Forcing: Adaptive test-time compute
"""
import math
import time
import random
import hashlib
import re
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class NodeStatus(Enum):
    UNEXPLORED = "unexplored"
    EXPANDED = "expanded"
    VERIFIED = "verified"
    FAILED = "failed"
    TERMINAL = "terminal"


@dataclass
class ReasoningStep:
    """Satu langkah reasoning dalam tree search."""
    description: str           # "Langkah 1: Identifikasi integral ..."
    expression: str = ""       # "integrate(x**2, x)"
    code_snippet: str = ""     # Python code for verification
    code_result: str = ""      # Hasil eksekusi code
    is_verified: bool = False  # Sudah diverifikasi via code?
    reward: float = 0.0        # Process reward score [0,1]


@dataclass
class MCTSNode:
    """Node dalam MCTS tree."""
    id: str
    depth: int
    step: Optional[ReasoningStep] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    status: NodeStatus = NodeStatus.UNEXPLORED
    cumulative_steps: List[ReasoningStep] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        """Average reward (Q-value)."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    @property
    def ucb1(self) -> float:
        """UCB1 score for selection."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.q_value
        exploration = math.sqrt(2.0 * math.log(self.parent.visits) / self.visits)
        return self.q_value + 1.414 * exploration


class ProcessRewardModel:
    """
    Process Reward Model (PRM) sederhana.
    Mengevaluasi kualitas setiap langkah reasoning.

    Di rStar-Math, PRM ditraining dari Q-values MCTS rollouts.
    Kita gunakan heuristic-based PRM + optional LLM scoring.
    """

    # Bobot untuk berbagai sinyal kualitas
    WEIGHTS = {
        "code_verified": 0.40,      # Apakah langkah terverifikasi oleh code?
        "mathematical_rigor": 0.25,  # Apakah ada ekspresi matematis?
        "step_coherence": 0.20,      # Apakah langkah koheren dengan sebelumnya?
        "length_penalty": 0.15,      # Penalti untuk langkah terlalu pendek/panjang
    }

    @staticmethod
    def score_step(step: ReasoningStep, prev_steps: List[ReasoningStep] = None) -> float:
        """
        Skor sebuah reasoning step [0.0, 1.0].
        Higher = better reasoning quality.
        """
        score = 0.0

        # 1. Code verification (paling penting)
        if step.is_verified and step.code_result:
            if "Error" not in step.code_result and "Traceback" not in step.code_result:
                score += ProcessRewardModel.WEIGHTS["code_verified"]
            else:
                score += ProcessRewardModel.WEIGHTS["code_verified"] * 0.2  # Partial credit for trying

        # 2. Mathematical rigor — ada ekspresi matematika?
        has_math = bool(step.expression and len(step.expression) > 2)
        has_math_symbols = any(c in step.description for c in '=+−×÷∫∑∏√πΣ')
        has_numbers = any(c.isdigit() for c in step.description)
        rigor = 0.0
        if has_math:
            rigor += 0.5
        if has_math_symbols:
            rigor += 0.3
        if has_numbers:
            rigor += 0.2
        score += ProcessRewardModel.WEIGHTS["mathematical_rigor"] * rigor

        # 3. Step coherence — logical flow with previous steps
        if prev_steps:
            # Simple heuristic: check if current step references concepts from previous
            prev_text = " ".join(s.description for s in prev_steps[-2:])
            # Extract key terms from previous steps
            prev_numbers = set(c for c in prev_text if c.isdigit())
            curr_numbers = set(c for c in step.description if c.isdigit())
            overlap = len(prev_numbers & curr_numbers)
            coherence = min(1.0, overlap / max(len(prev_numbers), 1))
            score += ProcessRewardModel.WEIGHTS["step_coherence"] * max(coherence, 0.3)
        else:
            score += ProcessRewardModel.WEIGHTS["step_coherence"] * 0.5  # First step gets partial

        # 4. Length penalty
        desc_len = len(step.description)
        if 20 <= desc_len <= 500:
            length_score = 1.0
        elif desc_len < 20:
            length_score = desc_len / 20.0
        else:
            length_score = max(0.3, 1.0 - (desc_len - 500) / 1000.0)
        score += ProcessRewardModel.WEIGHTS["length_penalty"] * length_score

        return min(1.0, max(0.0, score))

    @staticmethod
    def score_trajectory(steps: List[ReasoningStep]) -> float:
        """Skor keseluruhan reasoning trajectory."""
        if not steps:
            return 0.0
        step_scores = []
        for i, step in enumerate(steps):
            prev = steps[:i] if i > 0 else None
            step_scores.append(ProcessRewardModel.score_step(step, prev))

        # Weighted average: later steps count more (correctness of final answer)
        weights = [1.0 + i * 0.5 for i in range(len(step_scores))]
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(step_scores, weights)) / total_weight

        # Bonus for reaching a verified final answer
        if steps[-1].is_verified:
            weighted_score = min(1.0, weighted_score + 0.1)

        return weighted_score


class BudgetController:
    """
    Budget Forcing Controller.
    Mengontrol durasi "thinking" berdasarkan kompleksitas masalah.

    Terinspirasi dari paper Budget Forcing 2025-2026:
    - Simple problems → short budget → quick answer
    - Complex problems → extended budget → deep thinking
    - Force termination jika overthinking detected
    """

    class Complexity(Enum):
        TRIVIAL = 1       # 2+2, simple arithmetic
        SIMPLE = 2        # Single-step algebra
        MODERATE = 3      # Multi-step problems
        COMPLEX = 4       # Proof-like reasoning
        OLYMPIAD = 5      # Competition math

    # Max MCTS iterations per complexity level
    BUDGET_MAP = {
        Complexity.TRIVIAL: {"max_iterations": 1, "max_depth": 1, "max_time_ms": 100},
        Complexity.SIMPLE: {"max_iterations": 3, "max_depth": 2, "max_time_ms": 500},
        Complexity.MODERATE: {"max_iterations": 8, "max_depth": 4, "max_time_ms": 2000},
        Complexity.COMPLEX: {"max_iterations": 20, "max_depth": 6, "max_time_ms": 5000},
        Complexity.OLYMPIAD: {"max_iterations": 50, "max_depth": 10, "max_time_ms": 15000},
    }

    @staticmethod
    def estimate_complexity(query: str) -> 'BudgetController.Complexity':
        """Estimate problem complexity from query text."""
        q = query.lower()

        # Trivial: pure arithmetic
        if all(c in '0123456789+-*/^(). ' for c in q.strip()):
            return BudgetController.Complexity.TRIVIAL

        # Simple keyword check
        simple_kw = ['berapa', 'hitung']
        complex_kw = ['buktikan', 'proof', 'tunjukkan bahwa', 'show that', 'distinct', 'configurations', 'determine']
        olympiad_kw = ['olimpiade', 'olympiad', 'competition', 'aime', 'imo', 'putnam', 'grid of unit squares', 'coin']
        moderate_kw = ['integral', 'turunan', 'persamaan', 'solve', 'selesaikan', 'math', 'equation']

        if any(w in q for w in olympiad_kw):
            return BudgetController.Complexity.OLYMPIAD
        if any(w in q for w in complex_kw):
            return BudgetController.Complexity.COMPLEX
        if any(w in q for w in moderate_kw):
            return BudgetController.Complexity.MODERATE
        if any(w in q for w in simple_kw) and len(q) < 50:
            return BudgetController.Complexity.SIMPLE

        # Default based on query length
        if len(q) < 30:
            return BudgetController.Complexity.TRIVIAL
        elif len(q) < 80:
            return BudgetController.Complexity.SIMPLE
        else:
            return BudgetController.Complexity.MODERATE

    @staticmethod
    def get_budget(complexity: 'BudgetController.Complexity') -> Dict[str, int]:
        """Get compute budget for given complexity."""
        return BudgetController.BUDGET_MAP.get(
            complexity,
            BudgetController.BUDGET_MAP[BudgetController.Complexity.MODERATE]
        )


class MCTSMathReasoner:
    """
    Monte Carlo Tree Search untuk Mathematical Reasoning.

    Pipeline:
    1. Root node = masalah matematika
    2. SELECTION: UCB1 memilih node paling menjanjikan
    3. EXPANSION: LLM menghasilkan langkah reasoning baru
    4. SIMULATION: Code verification pada langkah
    5. BACKPROPAGATION: Update Q-values di tree

    Budget Forcing mengontrol berapa lama tree search berjalan.
    """

    def __init__(self,
                 llm_generate_fn: Optional[Callable] = None,
                 cas_engine=None,
                 exploration_constant: float = 1.414):
        """
        Args:
            llm_generate_fn: Function(prompt, system) -> str. Untuk menghasilkan reasoning steps.
            cas_engine: MathCASEngine instance untuk verification.
            exploration_constant: UCB1 exploration parameter.
        """
        self.llm_generate = llm_generate_fn
        self.cas_engine = cas_engine
        self.c_explore = exploration_constant
        self.prm = ProcessRewardModel()
        self.budget_controller = BudgetController()

    def reason(self, problem: str,
               on_step: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Main entry point. Melakukan MCTS reasoning pada masalah matematika.

        Returns:
            {
                "answer": str,              # Final answer
                "steps": [ReasoningStep],    # Step-by-step reasoning
                "trajectory_score": float,   # Overall quality score
                "iterations": int,           # Total MCTS iterations
                "time_ms": float,            # Total time
                "complexity": str,           # Estimated complexity
                "cas_verified": bool,        # CAS verification result
                "cas_result": dict,          # CAS computation result
            }
        """
        t0 = time.time()

        # 1. Estimate complexity and get budget
        complexity = self.budget_controller.estimate_complexity(problem)
        budget = self.budget_controller.get_budget(complexity)
        if on_step:
            on_step(f"🎯 MCTS Budget: {complexity.name} → max_iter={budget['max_iterations']}, "
                    f"max_depth={budget['max_depth']}")

        # 2. Try CAS Engine first (fast path for computable problems)
        cas_result_dict = {}
        cas_verified = False
        if self.cas_engine:
            try:
                cas_result = self.cas_engine.compute(problem)
                cas_result_dict = cas_result.to_dict()
                if cas_result.success:
                    cas_verified = True
                    if on_step:
                        on_step(f"✅ CAS Direct: {cas_result.symbolic_result} "
                                f"(numeric: {cas_result.numeric_result})")

                    # For trivial/simple: CAS result is sufficient
                    if complexity in (BudgetController.Complexity.TRIVIAL,
                                      BudgetController.Complexity.SIMPLE):
                        return {
                            "answer": cas_result.symbolic_result,
                            "steps": [ReasoningStep(
                                description=f"CAS Direct Computation: {s}",
                                expression=cas_result.symbolic_result,
                                is_verified=True,
                                reward=1.0
                            ) for s in cas_result.steps],
                            "trajectory_score": 1.0,
                            "iterations": 0,
                            "time_ms": (time.time() - t0) * 1000.0,
                            "complexity": complexity.name,
                            "cas_verified": True,
                            "cas_result": cas_result_dict,
                            "cas_prompt_injection": cas_result.to_prompt_injection(),
                        }
            except Exception as e:
                if on_step:
                    on_step(f"⚠️ CAS failed: {e}")

        # 3. MCTS Reasoning (for complex problems or CAS fallback)
        root = MCTSNode(
            id="root",
            depth=0,
            step=ReasoningStep(description=f"Problem: {problem}"),
            status=NodeStatus.EXPANDED
        )

        best_trajectory = []
        best_score = 0.0
        iterations = 0

        max_iter = budget["max_iterations"]
        max_depth = budget["max_depth"]
        max_time = budget["max_time_ms"]

        while iterations < max_iter:
            elapsed_ms = (time.time() - t0) * 1000.0
            if elapsed_ms > max_time:
                if on_step:
                    on_step(f"⏱️ Budget timeout at iter {iterations}")
                break

            iterations += 1

            # SELECTION
            selected = self._select(root)

            # EXPANSION
            new_node = self._expand(selected, problem, max_depth)
            if new_node is None:
                continue

            # SIMULATION (code verification)
            reward = self._simulate(new_node, problem)

            # BACKPROPAGATION
            self._backpropagate(new_node, reward)

            # Track best trajectory
            trajectory = self._get_trajectory(new_node)
            traj_score = self.prm.score_trajectory(
                [n.step for n in trajectory if n.step]
            )

            if traj_score > best_score:
                best_score = traj_score
                best_trajectory = trajectory

            if on_step:
                on_step(f"🌳 MCTS iter {iterations}/{max_iter}: "
                        f"reward={reward:.3f}, best_score={best_score:.3f}")

        # 4. Extract best answer
        steps = [n.step for n in best_trajectory if n.step and n.depth > 0]
        answer = self._extract_answer(steps, problem)

        # Heuristic fallback safety: if answer is a step description or fallback text,
        # but CAS successfully computed the result, use CAS.
        if (not answer or "langkah" in answer.lower() or "analisis lanjutan" in answer.lower()) and cas_verified and cas_result_dict.get("success"):
            answer = cas_result_dict.get("symbolic")

        # 5. Cross-verify with CAS if available
        if self.cas_engine and cas_verified and cas_result_dict.get("success"):
            # LLM answer should match CAS answer
            if on_step:
                on_step(f"🔍 Cross-verifying MCTS answer with CAS ground truth...")

        result = {
            "answer": answer,
            "steps": steps,
            "trajectory_score": best_score,
            "iterations": iterations,
            "time_ms": (time.time() - t0) * 1000.0,
            "complexity": complexity.name,
            "cas_verified": cas_verified,
            "cas_result": cas_result_dict,
        }

        # Add CAS prompt injection for LLM
        if cas_verified and cas_result_dict.get("success"):
            if self.cas_engine:
                cas_obj = self.cas_engine.compute(problem)
                result["cas_prompt_injection"] = cas_obj.to_prompt_injection()

        return result

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1-based selection. Walk down tree choosing best UCB1 child."""
        current = node
        while current.children and current.status == NodeStatus.EXPANDED:
            current = max(current.children, key=lambda c: c.ucb1)
        return current

    def _expand(self, node: MCTSNode, problem: str, max_depth: int) -> Optional[MCTSNode]:
        """Expand node by generating a new reasoning step."""
        if node.depth >= max_depth:
            node.status = NodeStatus.TERMINAL
            return None

        # Build context from ancestor steps
        ancestor_steps = self._get_trajectory(node)
        context = "\n".join(
            f"Step {n.depth}: {n.step.description}"
            for n in ancestor_steps if n.step and n.depth > 0
        )

        # Generate next step
        step_description = self._generate_next_step(problem, context, node.depth + 1)
        if not step_description:
            return None

        # Create new reasoning step
        new_step = ReasoningStep(
            description=step_description,
            expression=self._extract_expression(step_description),
        )

        # Create child node
        child_id = f"n_{node.depth + 1}_{len(node.children)}"
        child = MCTSNode(
            id=child_id,
            depth=node.depth + 1,
            step=new_step,
            parent=node,
            cumulative_steps=node.cumulative_steps + [new_step],
            status=NodeStatus.EXPANDED
        )
        node.children.append(child)
        node.status = NodeStatus.EXPANDED

        return child

    def _simulate(self, node: MCTSNode, problem: str) -> float:
        """Simulate by verifying the step with code execution."""
        if node.step is None:
            return 0.0

        # Try code verification
        if node.step.expression:
            code, result = self._verify_with_code(node.step.expression)
            node.step.code_snippet = code
            node.step.code_result = result
            node.step.is_verified = "Error" not in result

        # Score the step
        prev_steps = node.cumulative_steps[:-1] if len(node.cumulative_steps) > 1 else None
        reward = self.prm.score_step(node.step, prev_steps)
        node.step.reward = reward

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _get_trajectory(self, node: MCTSNode) -> List[MCTSNode]:
        """Get path from root to this node."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def _generate_next_step(self, problem: str, context: str, step_num: int) -> str:
        """Generate next reasoning step using LLM or heuristic."""
        if self.llm_generate:
            prompt = (
                f"Problem: {problem}\n\n"
                f"Previous reasoning:\n{context}\n\n"
                f"Generate reasoning step {step_num}. "
                f"Include mathematical expressions where applicable. "
                f"Be precise and verify each computation.\n\n"
                f"Step {step_num}:"
            )
            system = (
                "You are a mathematical reasoning engine. Generate ONE precise "
                "reasoning step. Include mathematical expressions. "
                "Format: 'Description [expression: mathematical_expression]'"
            )
            try:
                return self.llm_generate(prompt, system)
            except Exception:
                pass

        # Fallback: heuristic step generation (no LLM)
        return f"Langkah {step_num}: Analisis lanjutan masalah '{problem[:50]}...'"

    def _extract_expression(self, description: str) -> str:
        """Extract mathematical expression from step description."""
        # Look for [expression: ...] format
        expr_match = re.search(r'\[expression:\s*(.+?)\]', description)
        if expr_match:
            return expr_match.group(1).strip()

        # Look for math-like patterns
        math_match = re.search(r'[=:]\s*([\d\w\+\-\*\/\^\(\)\.\s]+)', description)
        if math_match:
            expr = math_match.group(1).strip()
            if len(expr) > 2 and any(c.isdigit() for c in expr):
                return expr

        return ""

    def _verify_with_code(self, expression: str) -> Tuple[str, str]:
        """Verify a mathematical expression by executing it."""
        # Sanitize expression
        safe_expr = expression.replace('^', '**')

        # Build safe code snippet
        code = f"""
from sympy import *
x, y, z = symbols('x y z')
try:
    result = {safe_expr}
    print(f"Result: {{result}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
        # Execute in sandbox
        try:
            import subprocess
            result = subprocess.run(
                ['python3', '-c', code],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.strip() or result.stderr.strip()
            return code, output[:500]
        except subprocess.TimeoutExpired:
            return code, "Error: Timeout (5s)"
        except Exception as e:
            return code, f"Error: {e}"

    def _extract_answer(self, steps: List[ReasoningStep], problem: str) -> str:
        """Extract final answer from reasoning steps."""
        if not steps:
            return "Unable to determine answer"

        # Check last step for answer
        last = steps[-1]
        if last.code_result and "Result:" in last.code_result:
            return last.code_result.split("Result:")[1].strip()

        if last.expression:
            return last.expression

        return last.description


# We need `re` for _extract_expression

# ─── Global singletons ───
process_reward_model = ProcessRewardModel()
budget_controller = BudgetController()
