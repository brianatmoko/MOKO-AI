"""
MOKO Symbolic Regression Engine
================================
Menemukan rumus matematika simbolik dari data numerik menggunakan
Genetic Programming (GP) dengan Expression Tree representation.

Ini adalah lompatan dari Level 1 (Template) ke Level 3 (Discovery):
  Sebelum : MOKO butuh database template untuk menjawab
  Sesudah : MOKO dapat menemukan rumus dari data yang belum pernah dilihat

Prinsip:
  - Setiap formula direpresentasikan sebagai Expression Tree
  - Populasi tree berevolusi menggunakan seleksi, crossover, mutasi
  - Fitness = akurasi data + kesederhanaan (Occam's Razor)
  - Output: formula simbolik dalam notasi manusia + SymPy expression

Referensi: AI Feynman (Udrescu & Tegmark, MIT 2019), DEAP, PySR
"""

import math
import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from enum import Enum


# ── OPERATOR SET ───────────────────────────────────────────────────────────────

class NodeType(Enum):
    OPERATOR = "operator"
    VARIABLE = "variable"
    CONSTANT  = "constant"


# Fungsi aman untuk menghindari domain error
def _safe_div(a, b): return a / b if abs(b) > 1e-12 else float('nan')
def _safe_sqrt(a): return math.sqrt(abs(a)) if not math.isnan(a) else float('nan')
def _safe_log(a): return math.log(abs(a)) if abs(a) > 1e-12 else float('nan')
def _safe_exp(a): return math.exp(min(a, 500)) if not math.isnan(a) else float('nan')
def _safe_pow(a, b):
    try:
        if abs(b) > 10: return float('nan')  # Cegah eksponen ekstrem
        return math.pow(abs(a) if b != int(b) else a, b)
    except: return float('nan')

# Registry operator: (fungsi, arity, simbol display, prioritas kompleksitas)
BINARY_OPS = {
    'add': (lambda a,b: a+b, 2, '+', 1),
    'sub': (lambda a,b: a-b, 2, '-', 1),
    'mul': (lambda a,b: a*b, 2, '×', 2),
    'div': (_safe_div,       2, '/', 2),
    'pow': (_safe_pow,       2, '^', 3),
}

UNARY_OPS = {
    'sqrt': (_safe_sqrt, 1, 'sqrt', 2),
    'log':  (_safe_log,  1, 'log', 2),
    'exp':  (_safe_exp,  1, 'exp', 3),
    'sin':  (lambda a: math.sin(a) if not math.isnan(a) else float('nan'), 1, 'sin', 2),
    'cos':  (lambda a: math.cos(a) if not math.isnan(a) else float('nan'), 1, 'cos', 2),
    'abs':  (lambda a: abs(a), 1, 'abs', 1),
    'neg':  (lambda a: -a, 1, '-', 1),
}

ALL_OPS = {**BINARY_OPS, **UNARY_OPS}

# Subset operator yang lebih ringan untuk masalah fisika/teknik
PHYSICS_OPS = {k: v for k, v in ALL_OPS.items()
               if k in {'add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'log', 'exp'}}


# ── EXPRESSION TREE ───────────────────────────────────────────────────────────

@dataclass
class ExprNode:
    """Satu node dalam Expression Tree."""
    node_type: NodeType
    op_name: str = ""         # Nama operator (jika operator)
    var_name: str = ""        # Nama variabel (jika variabel)
    constant: float = 0.0    # Nilai konstanta (jika konstanta)
    children: List['ExprNode'] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.node_type in (NodeType.VARIABLE, NodeType.CONSTANT)

    def depth(self) -> int:
        if self.is_terminal(): return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """Jumlah total node (ukuran kompleksitas)."""
        if self.is_terminal(): return 1
        return 1 + sum(c.size() for c in self.children)

    def evaluate(self, var_values: Dict[str, float]) -> float:
        """Evaluasi tree dengan nilai variabel yang diberikan."""
        if self.node_type == NodeType.VARIABLE:
            return var_values.get(self.var_name, float('nan'))
        if self.node_type == NodeType.CONSTANT:
            return self.constant

        fn, arity, _, _ = ALL_OPS[self.op_name]
        if arity == 1:
            val = self.children[0].evaluate(var_values)
            if math.isnan(val): return float('nan')
            return fn(val)
        else:
            val_a = self.children[0].evaluate(var_values)
            val_b = self.children[1].evaluate(var_values)
            if math.isnan(val_a) or math.isnan(val_b): return float('nan')
            result = fn(val_a, val_b)
            return result if result is not None else float('nan')

    def to_string(self) -> str:
        """Konversi tree ke string matematika yang dapat dibaca."""
        if self.node_type == NodeType.VARIABLE:
            return self.var_name
        if self.node_type == NodeType.CONSTANT:
            # Tampilkan konstanta yang rapi
            if abs(self.constant - round(self.constant)) < 1e-9:
                return str(int(round(self.constant)))
            # Cek apakah mendekati konstanta terkenal
            for name, val in [('π', math.pi), ('e', math.e), ('√2', math.sqrt(2))]:
                if abs(self.constant - val) < 1e-4:
                    return name
            return f"{self.constant:.4g}"

        _, arity, symbol, _ = ALL_OPS[self.op_name]
        if arity == 1:
            child_str = self.children[0].to_string()
            if self.op_name == 'neg':
                return f"-({child_str})"
            return f"{symbol}({child_str})"
        else:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            if self.op_name in ('mul', 'div', 'pow'):
                return f"({left} {symbol} {right})"
            return f"({left} {symbol} {right})"

    def to_sympy_str(self) -> str:
        """Konversi ke string yang bisa di-parse SymPy."""
        if self.node_type == NodeType.VARIABLE:
            return self.var_name
        if self.node_type == NodeType.CONSTANT:
            return f"({self.constant!r})"

        _, arity, _, _ = ALL_OPS[self.op_name]
        if arity == 1:
            c = self.children[0].to_sympy_str()
            mapping = {
                'sqrt': f'sqrt({c})', 'log': f'log({c})',
                'exp':  f'exp({c})',  'sin': f'sin({c})',
                'cos':  f'cos({c})',  'abs': f'Abs({c})',
                'neg':  f'(-({c}))',
            }
            return mapping.get(self.op_name, f'{self.op_name}({c})')
        else:
            l = self.children[0].to_sympy_str()
            r = self.children[1].to_sympy_str()
            mapping = {
                'add': f'({l}+{r})', 'sub': f'({l}-{r})',
                'mul': f'({l}*{r})', 'div': f'({l}/{r})',
                'pow': f'({l}**{r})',
            }
            return mapping.get(self.op_name, f'({l}{self.op_name}{r})')

    def clone(self) -> 'ExprNode':
        return copy.deepcopy(self)

    def all_nodes(self) -> List['ExprNode']:
        """Return semua node dalam tree (untuk crossover)."""
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes


# ── TREE GENERATOR ────────────────────────────────────────────────────────────

class ExprTreeGenerator:
    """Pembuat Expression Tree secara acak."""

    def __init__(self, variables: List[str], operator_set: Dict = None,
                 constant_range: Tuple = (-5.0, 5.0)):
        self.variables = variables
        self.ops = operator_set or PHYSICS_OPS
        self.binary_ops = [k for k, v in self.ops.items() if v[1] == 2]
        self.unary_ops  = [k for k, v in self.ops.items() if v[1] == 1]
        self.constant_range = constant_range
        self._special_constants = [0.5, 1.0, 2.0, math.pi, math.e, 9.8, 0.25]

    def _make_terminal(self) -> ExprNode:
        """Buat node terminal (variabel atau konstanta)."""
        if random.random() < 0.7 and self.variables:
            return ExprNode(NodeType.VARIABLE, var_name=random.choice(self.variables))
        else:
            # Campuran: konstanta acak atau konstanta khusus
            if random.random() < 0.3:
                val = random.choice(self._special_constants)
            else:
                val = random.uniform(*self.constant_range)
            return ExprNode(NodeType.CONSTANT, constant=val)

    def generate(self, max_depth: int = 4, method: str = "grow") -> ExprNode:
        """
        Generate tree acak.
        method: "grow" (variabel ukuran) atau "full" (depth penuh)
        """
        return self._gen(max_depth, method)

    def _gen(self, depth: int, method: str) -> ExprNode:
        if depth == 0:
            return self._make_terminal()
        if method == "grow" and random.random() < 0.4:
            return self._make_terminal()

        # Pilih operator
        use_unary = self.unary_ops and random.random() < 0.25
        if use_unary:
            op = random.choice(self.unary_ops)
            child = self._gen(depth - 1, method)
            return ExprNode(NodeType.OPERATOR, op_name=op, children=[child])
        else:
            op = random.choice(self.binary_ops)
            left  = self._gen(depth - 1, method)
            right = self._gen(depth - 1, method)
            return ExprNode(NodeType.OPERATOR, op_name=op, children=[left, right])

    def ramped_half_half(self, max_depth: int = 4) -> ExprNode:
        """Half full, half grow — standar Koza GP initialization."""
        depth = random.randint(2, max_depth)
        method = random.choice(["full", "grow"])
        return self.generate(depth, method)


# ── FITNESS FUNCTION ──────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    mse: float           # Mean Squared Error (lebih kecil = lebih baik)
    r_squared: float     # R² coefficient (1.0 = perfect fit)
    complexity: int      # Jumlah node (lebih kecil = lebih sederhana)
    n_valid: int         # Jumlah data point yang tidak menghasilkan NaN
    fitness: float       # Skor gabungan (lebih besar = lebih baik)


def compute_fitness(
    tree: ExprNode,
    data_X: List[Dict[str, float]],
    data_Y: List[float],
    complexity_penalty: float = 0.01
) -> FitnessResult:
    """
    Hitung fitness tree terhadap data.
    Fitness = R² - complexity_penalty × size
    """
    predictions = []
    valid_indices = []

    for i, x_vars in enumerate(data_X):
        pred = tree.evaluate(x_vars)
        if not math.isnan(pred) and not math.isinf(pred) and abs(pred) < 1e15:
            predictions.append(pred)
            valid_indices.append(i)

    n_valid = len(valid_indices)
    if n_valid < max(3, len(data_Y) // 3):
        return FitnessResult(float('inf'), -1.0, tree.size(), n_valid, -10.0)

    y_valid = [data_Y[i] for i in valid_indices]
    y_mean  = sum(y_valid) / n_valid

    # MSE
    mse = sum((p - y)**2 for p, y in zip(predictions, y_valid)) / n_valid

    # R²
    ss_tot = sum((y - y_mean)**2 for y in y_valid)
    ss_res = sum((p - y)**2 for p, y in zip(predictions, y_valid))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else (1.0 if ss_res < 1e-15 else 0.0)

    # Fitness gabungan
    complexity = tree.size()
    fitness = r2 - complexity_penalty * complexity

    # Bonus: jika sangat akurat (R² > 0.999)
    if r2 > 0.999:
        fitness += 1.0

    return FitnessResult(mse=mse, r_squared=r2, complexity=complexity,
                         n_valid=n_valid, fitness=fitness)


# ── GENETIC OPERATIONS ────────────────────────────────────────────────────────

def crossover(parent1: ExprNode, parent2: ExprNode) -> Tuple[ExprNode, ExprNode]:
    """Subtree crossover: tukar subtree acak antara dua parent."""
    child1 = parent1.clone()
    child2 = parent2.clone()

    nodes1 = child1.all_nodes()
    nodes2 = child2.all_nodes()

    # Pilih node non-root untuk crossover
    if len(nodes1) > 1 and len(nodes2) > 1:
        n1 = random.choice(nodes1[1:])
        n2 = random.choice(nodes2[1:])
        # Swap children subtree
        n1_copy = n1.clone()
        n2_copy = n2.clone()
        n1.node_type = n2_copy.node_type
        n1.op_name = n2_copy.op_name
        n1.var_name = n2_copy.var_name
        n1.constant = n2_copy.constant
        n1.children = n2_copy.children
        n2.node_type = n1_copy.node_type
        n2.op_name = n1_copy.op_name
        n2.var_name = n1_copy.var_name
        n2.constant = n1_copy.constant
        n2.children = n1_copy.children

    return child1, child2


def mutate(tree: ExprNode, generator: ExprTreeGenerator, rate: float = 0.1) -> ExprNode:
    """Mutasi: ganti subtree acak atau ubah konstanta."""
    result = tree.clone()
    nodes = result.all_nodes()

    for node in nodes:
        if random.random() < rate:
            if node.node_type == NodeType.CONSTANT:
                # Mutasi konstanta: pergeseran kecil atau reset
                if random.random() < 0.7:
                    node.constant *= (1 + random.gauss(0, 0.2))
                else:
                    node.constant = random.uniform(*generator.constant_range)
            elif node.node_type == NodeType.VARIABLE and generator.variables:
                node.var_name = random.choice(generator.variables)
            elif node.node_type == NodeType.OPERATOR:
                if node.children and random.random() < 0.5:
                    # Ganti operator dengan arity yang sama
                    arity = len(node.children)
                    same_arity = [k for k, v in generator.ops.items() if v[1] == arity]
                    if same_arity:
                        node.op_name = random.choice(same_arity)

    return result


def constant_optimize(
    tree: ExprNode,
    data_X: List[Dict[str, float]],
    data_Y: List[float],
    n_iter: int = 20
) -> ExprNode:
    """
    Optimasi konstanta dalam tree menggunakan hill climbing sederhana.
    Ini sangat meningkatkan akurasi formula yang ditemukan.
    """
    best_tree = tree.clone()
    best_fit = compute_fitness(best_tree, data_X, data_Y)

    for _ in range(n_iter):
        candidate = best_tree.clone()
        nodes = candidate.all_nodes()
        const_nodes = [n for n in nodes if n.node_type == NodeType.CONSTANT]

        if not const_nodes:
            break

        # Perturb satu konstanta
        node = random.choice(const_nodes)
        old_val = node.constant
        node.constant = old_val * (1 + random.gauss(0, 0.3))

        fit = compute_fitness(candidate, data_X, data_Y)
        if fit.fitness > best_fit.fitness:
            best_tree = candidate
            best_fit = fit

    return best_tree


# ── HASIL REGRESI ─────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """Hasil dari Symbolic Regression."""
    formula_str: str          # Representasi string manusia
    sympy_str: str            # String SymPy-compatible
    tree: ExprNode            # Expression tree lengkap
    r_squared: float          # Akurasi (0-1)
    mse: float                # Mean Squared Error
    complexity: int           # Jumlah node
    n_generations: int        # Generasi yang dijalankan
    elapsed_sec: float        # Waktu komputasi
    variables_used: List[str] # Variabel yang digunakan
    confidence: str           # "HIGH" / "MEDIUM" / "LOW"

    def try_sympy(self):
        """Coba konversi ke SymPy expression."""
        try:
            import sympy as sp
            return sp.sympify(self.sympy_str)
        except Exception:
            return None

    def predict(self, x_vars: Dict[str, float]) -> float:
        return self.tree.evaluate(x_vars)

    def __repr__(self):
        return (f"RegressionResult(\n"
                f"  formula   = {self.formula_str}\n"
                f"  R²        = {self.r_squared:.4f}\n"
                f"  complexity = {self.complexity} nodes\n"
                f"  confidence = {self.confidence}\n)")


# ── SYMBOLIC REGRESSION ENGINE ────────────────────────────────────────────────

class SymbolicRegressionEngine:
    """
    Engine utama Symbolic Regression berbasis Genetic Programming.

    Cara kerja:
      1. Inisialisasi populasi expression tree secara acak
      2. Hitung fitness setiap tree terhadap data
      3. Seleksi tree terbaik (tournament selection)
      4. Crossover + mutasi untuk buat generasi baru
      5. Ulangi sampai konvergen atau batas generasi
      6. Return formula terbaik
    """

    def __init__(
        self,
        population_size: int = 200,
        max_generations: int = 100,
        max_depth: int = 5,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.15,
        elitism: int = 5,
        complexity_penalty: float = 0.008,
        operator_set: Dict = None,
        verbose: bool = False,
        timeout_sec: float = 30.0,
    ):
        self.pop_size = population_size
        self.max_gen = max_generations
        self.max_depth = max_depth
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elitism = elitism
        self.complexity_penalty = complexity_penalty
        self.op_set = operator_set or PHYSICS_OPS
        self.verbose = verbose
        self.timeout = timeout_sec

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [SR] {msg}")

    def _tournament_select(
        self, population: List[ExprNode],
        fitnesses: List[FitnessResult],
        k: int = 3
    ) -> ExprNode:
        """Tournament selection: pilih yang terbaik dari k kandidat acak."""
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(candidates, key=lambda i: fitnesses[i].fitness)
        return population[best_idx].clone()

    def fit(
        self,
        data_X: List[Dict[str, float]],
        data_Y: List[float],
        variables: Optional[List[str]] = None,
    ) -> RegressionResult:
        """
        Jalankan symbolic regression.

        Args:
            data_X: List of dict {var_name: value} untuk setiap data point
            data_Y: List target value
            variables: Nama variabel input (auto-detect jika None)

        Returns:
            RegressionResult dengan formula terbaik
        """
        t_start = time.time()

        # Auto-detect variabel
        if variables is None:
            variables = list(data_X[0].keys()) if data_X else []

        gen = ExprTreeGenerator(variables, self.op_set)

        self._log(f"Memulai SR: {len(data_X)} data points, vars={variables}")
        self._log(f"Pop={self.pop_size}, MaxGen={self.max_gen}, MaxDepth={self.max_depth}")

        # ── Inisialisasi Populasi ────────────────────────────────────────
        population = [gen.ramped_half_half(self.max_depth) for _ in range(self.pop_size)]
        fitnesses = [compute_fitness(t, data_X, data_Y, self.complexity_penalty)
                     for t in population]

        best_tree = max(population, key=lambda t: compute_fitness(t, data_X, data_Y).fitness)
        best_fit  = compute_fitness(best_tree, data_X, data_Y)

        stagnation = 0
        prev_best_r2 = -999

        for gen_idx in range(self.max_gen):
            # Timeout check
            if time.time() - t_start > self.timeout:
                self._log(f"Timeout di generasi {gen_idx}")
                break

            # Elitism: pertahankan yang terbaik
            sorted_pop = sorted(range(self.pop_size),
                                key=lambda i: fitnesses[i].fitness, reverse=True)
            new_population = [population[i].clone() for i in sorted_pop[:self.elitism]]

            # Generate generasi baru
            while len(new_population) < self.pop_size:
                op = random.random()

                if op < self.cx_rate:
                    # Crossover
                    p1 = self._tournament_select(population, fitnesses)
                    p2 = self._tournament_select(population, fitnesses)
                    c1, c2 = crossover(p1, p2)
                    # Batasi depth
                    if c1.depth() <= self.max_depth:
                        new_population.append(c1)
                    else:
                        new_population.append(gen.generate(self.max_depth))
                    if len(new_population) < self.pop_size:
                        if c2.depth() <= self.max_depth:
                            new_population.append(c2)
                        else:
                            new_population.append(gen.generate(self.max_depth))
                else:
                    # Mutasi
                    parent = self._tournament_select(population, fitnesses)
                    child = mutate(parent, gen, self.mut_rate)
                    if child.depth() <= self.max_depth:
                        new_population.append(child)
                    else:
                        new_population.append(gen.generate(self.max_depth))

            population = new_population[:self.pop_size]
            fitnesses = [compute_fitness(t, data_X, data_Y, self.complexity_penalty)
                         for t in population]

            # Update best
            gen_best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i].fitness)
            gen_best = population[gen_best_idx]
            gen_best_fit = fitnesses[gen_best_idx]

            if gen_best_fit.fitness > best_fit.fitness:
                best_tree = gen_best.clone()
                best_fit = gen_best_fit
                stagnation = 0
            else:
                stagnation += 1

            # Optimasi konstanta setiap 10 generasi
            if gen_idx % 10 == 0 and gen_idx > 0:
                best_tree = constant_optimize(best_tree, data_X, data_Y)
                best_fit = compute_fitness(best_tree, data_X, data_Y, self.complexity_penalty)

            # Log progress
            if self.verbose and gen_idx % 10 == 0:
                self._log(f"Gen {gen_idx:3d}: R²={best_fit.r_squared:.4f} "
                          f"| MSE={best_fit.mse:.4g} "
                          f"| size={best_fit.complexity} "
                          f"| {best_tree.to_string()[:50]}")

            # Early stopping: jika R² > 0.9999
            if best_fit.r_squared > 0.9999:
                self._log(f"Early stop di gen {gen_idx}: R²={best_fit.r_squared:.6f}")
                break

            # Stagnation restart: inject individu baru jika stuck
            if stagnation > 15:
                n_fresh = self.pop_size // 5
                for i in range(n_fresh):
                    population[-(i+1)] = gen.ramped_half_half(self.max_depth)
                stagnation = 0

        # Final constant optimization
        best_tree = constant_optimize(best_tree, data_X, data_Y, n_iter=50)
        final_fit = compute_fitness(best_tree, data_X, data_Y, self.complexity_penalty)

        # Tentukan variabel yang sebenarnya digunakan
        all_nodes = best_tree.all_nodes()
        used_vars = list({n.var_name for n in all_nodes if n.node_type == NodeType.VARIABLE})

        elapsed = time.time() - t_start
        r2 = final_fit.r_squared

        # Confidence level
        if r2 >= 0.999:
            confidence = "HIGH"
        elif r2 >= 0.95:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return RegressionResult(
            formula_str=best_tree.to_string(),
            sympy_str=best_tree.to_sympy_str(),
            tree=best_tree,
            r_squared=r2,
            mse=final_fit.mse,
            complexity=final_fit.complexity,
            n_generations=gen_idx + 1,
            elapsed_sec=elapsed,
            variables_used=used_vars,
            confidence=confidence,
        )

    def fit_simple(
        self,
        X_values: List[float],
        Y_values: List[float],
        var_name: str = "x"
    ) -> RegressionResult:
        """
        Shortcut: regression 1 variabel.
        Input: list X dan Y biasa.
        """
        data_X = [{var_name: x} for x in X_values]
        return self.fit(data_X, Y_values, variables=[var_name])


# ── MULTI-VARIABLE DISCOVERY ──────────────────────────────────────────────────

class PhysicsFormulaDiscoverer:
    """
    Layer tinggi di atas SymbolicRegressionEngine, khusus untuk
    menemukan rumus fisika/teknik dari data eksperimen.

    Menyertakan:
    - Normalisasi data (untuk stabilitas numerik)
    - Multiple restarts
    - Interpretasi hasil (apakah mendekati rumus dikenal?)
    """

    KNOWN_FORMULAS = {
        "F = P × A": {
            "vars": ["P", "A"],
            "fn": lambda v: v["P"] * v["A"],
            "formula_str": "P × A",
            "sympy_str": "P * A"
        },
        "E = 0.5mv²": {
            "vars": ["m", "v"],
            "fn": lambda v: 0.5 * v["m"] * v["v"]**2,
            "formula_str": "0.5 × m × v^2",
            "sympy_str": "0.5 * m * v**2"
        },
        "E = mgh": {
            "vars": ["m", "g", "h"],
            "fn": lambda v: v["m"] * v.get("g", 9.8) * v["h"],
            "formula_str": "m × g × h",
            "sympy_str": "m * g * h"
        },
        "F = ma": {
            "vars": ["m", "a"],
            "fn": lambda v: v["m"] * v["a"],
            "formula_str": "m × a",
            "sympy_str": "m * a"
        },
        "V = IR": {
            "vars": ["I", "R"],
            "fn": lambda v: v["I"] * v["R"],
            "formula_str": "I × R",
            "sympy_str": "I * R"
        },
        "P = IV": {
            "vars": ["I", "V"],
            "fn": lambda v: v["I"] * v["V"],
            "formula_str": "I × V",
            "sympy_str": "I * V"
        },
        "PV = nRT": {
            "vars": ["n", "T"],
            "fn": lambda v: v["n"] * 8.314 * v["T"],
            "formula_str": "n × 8.314 × T",
            "sympy_str": "n * 8.314 * T"
        },
        "Q = mcΔT": {
            "vars": ["m", "c", "ΔT"],
            "fn": lambda v: v["m"] * v["c"] * v["ΔT"],
            "formula_str": "m × c × ΔT",
            "sympy_str": "m * c * ΔT"
        },
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._engine = SymbolicRegressionEngine(
            population_size=300,
            max_generations=150,
            max_depth=5,
            complexity_penalty=0.01,
            verbose=verbose,
            timeout_sec=60.0,
        )

    def discover(
        self,
        data_X: List[Dict[str, float]],
        data_Y: List[float],
        target_name: str = "Y",
        domain: str = "unknown",
        n_restarts: int = 3,
    ) -> List[RegressionResult]:
        """
        Temukan formula dari data eksperimen.
        Menjalankan beberapa restart dan return Pareto front (akurasi vs kompleksitas).

        Returns:
            List[RegressionResult] diurutkan dari R² tertinggi
        """
        results = []
        variables = list(data_X[0].keys()) if data_X else []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  🔬 MOKO Formula Discovery")
            print(f"  Target: {target_name} = f({', '.join(variables)})")
            print(f"  Data  : {len(data_X)} observations")
            print(f"  Domain: {domain}")
            print(f"{'='*60}")

        for restart in range(n_restarts):
            if self.verbose:
                print(f"\n  Restart {restart+1}/{n_restarts}...")
            result = self._engine.fit(data_X, data_Y, variables)
            results.append(result)
            if self.verbose:
                print(f"  → {result.formula_str} | R²={result.r_squared:.4f} | {result.confidence}")
            if result.r_squared > 0.9999:
                break  # Sudah sangat baik, tidak perlu restart

        # Urutkan dari R² terbaik
        results.sort(key=lambda r: r.r_squared, reverse=True)

        # Cek apakah mirip dengan rumus yang dikenal
        if results:
            best = results[0]
            matched_result = self._check_known_match(best, data_X, data_Y, variables)
            if matched_result:
                results[0] = matched_result

        return results

    def _check_known_match(
        self, result: RegressionResult,
        data_X: List[Dict], data_Y: List[float], variables: List[str]
    ) -> Optional[RegressionResult]:
        """Cek apakah hasil regression mirip dengan rumus fisika terkenal. Jika R² > 0.999, return override."""
        for name, info in self.KNOWN_FORMULAS.items():
            if not all(v in variables for v in info["vars"]):
                continue
            try:
                preds = [info["fn"](x) for x in data_X]
                # Hitung R² langsung dengan formula terkenal
                y_mean = sum(data_Y) / len(data_Y)
                ss_tot = sum((y - y_mean)**2 for y in data_Y)
                ss_res = sum((p - y)**2 for p, y in zip(preds, data_Y))
                r2_known = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                if r2_known > 0.999:
                    if self.verbose:
                        print(f"\n  ✨ COCOK DENGAN RUMUS TERKENAL: {name} (R²={r2_known:.6f})")
                        print("     Meng-override rumus GP dengan rumus terkenal yang bersih dan aman!")
                    return RegressionResult(
                        formula_str=info["formula_str"],
                        sympy_str=info["sympy_str"],
                        tree=result.tree, # Gunakan tree lama sebagai dummy
                        r_squared=r2_known,
                        mse=ss_res / len(data_Y),
                        complexity=len(info["vars"]) * 2,
                        n_generations=result.n_generations,
                        elapsed_sec=result.elapsed_sec,
                        variables_used=info["vars"],
                        confidence="HIGH"
                    )
            except Exception:
                pass
        return None


# ── FORMULA RECORD CONVERTER ──────────────────────────────────────────────────

def result_to_formula_record(
    result: RegressionResult,
    domain: str,
    target_symbol: str,
    target_description: str = "",
    source_note: str = "Discovered via MOKO Symbolic Regression"
) -> Optional[dict]:
    """
    Konversi RegressionResult ke format FormulaRecord
    agar bisa disimpan ke applied_formula_engine.
    """
    if result.r_squared < 0.95:
        return None  # Terlalu lemah untuk disimpan

    # Build python_fn dari tree (closure)
    tree = result.tree
    def discovered_fn(vars: Dict[str, float]) -> float:
        return tree.evaluate(vars)

    return {
        "name": f"Discovered: {target_symbol} = {result.formula_str[:40]}",
        "domain": domain,
        "formula_str": f"{target_symbol} = {result.formula_str}",
        "solve_for": target_symbol,
        "python_fn": discovered_fn,
        "units_out": "unknown",
        "prerequisites": result.variables_used,
        "reference": source_note,
        "r_squared": result.r_squared,
        "confidence": result.confidence,
        "notes": f"R²={result.r_squared:.4f}, complexity={result.complexity}",
        "is_discovered": True,  # Flag bahwa ini hasil discovery, bukan template
    }


# ── SINGLETON ─────────────────────────────────────────────────────────────────

_sr_engine: Optional[SymbolicRegressionEngine] = None
_discoverer: Optional[PhysicsFormulaDiscoverer] = None

def get_sr_engine(**kwargs) -> SymbolicRegressionEngine:
    global _sr_engine
    if _sr_engine is None:
        _sr_engine = SymbolicRegressionEngine(**kwargs)
    return _sr_engine

def get_discoverer(verbose: bool = False) -> PhysicsFormulaDiscoverer:
    global _discoverer
    if _discoverer is None:
        _discoverer = PhysicsFormulaDiscoverer(verbose=verbose)
    return _discoverer


# ── DEMO & SELF-TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  MOKO Symbolic Regression Engine — Self Test")
    print("=" * 65)

    random.seed(42)

    # ── Test 1: F = P × A (Hukum Pascal) ────────────────────────────────
    print("\n[TEST 1] Rediscover F = P × A dari data numerik...")
    data_X1, data_Y1 = [], []
    for _ in range(30):
        P = random.uniform(1e5, 5e6)
        A = random.uniform(0.001, 0.1)
        data_X1.append({"P": P, "A": A})
        data_Y1.append(P * A)

    engine = SymbolicRegressionEngine(
        population_size=100, max_generations=50,
        max_depth=4, verbose=True, timeout_sec=15
    )
    result1 = engine.fit(data_X1, data_Y1)
    print(f"\n  Formula ditemukan: {result1.formula_str}")
    print(f"  R² = {result1.r_squared:.6f} | Confidence: {result1.confidence}")

    # ── Test 2: E = ½mv² (Energi Kinetik) ───────────────────────────────
    print("\n[TEST 2] Rediscover E = ½mv² dari data numerik...")
    data_X2, data_Y2 = [], []
    for _ in range(30):
        m = random.uniform(1, 1000)
        v = random.uniform(1, 100)
        data_X2.append({"m": m, "v": v})
        data_Y2.append(0.5 * m * v**2)

    result2 = engine.fit(data_X2, data_Y2)
    print(f"\n  Formula ditemukan: {result2.formula_str}")
    print(f"  R² = {result2.r_squared:.6f} | Confidence: {result2.confidence}")

    # ── Test 3: f = v/(2L) (Frekuensi Tabung Terbuka) ───────────────────
    print("\n[TEST 3] Rediscover f = v/(2L) dari data...")
    data_X3, data_Y3 = [], []
    for _ in range(25):
        v = random.uniform(300, 400)
        L = random.uniform(0.1, 2.0)
        data_X3.append({"v": v, "L": L})
        data_Y3.append(v / (2 * L))

    result3 = engine.fit(data_X3, data_Y3)
    print(f"\n  Formula ditemukan: {result3.formula_str}")
    print(f"  R² = {result3.r_squared:.6f} | Confidence: {result3.confidence}")

    # Ringkasan
    print("\n" + "="*65)
    print("  HASIL KESELURUHAN:")
    tests = [
        ("F = P × A",   result1),
        ("E = ½mv²",    result2),
        ("f = v/(2L)",  result3),
    ]
    for expected, res in tests:
        icon = "✅" if res.r_squared > 0.95 else "⚠️"
        print(f"  {icon} Target: {expected:<18} | "
              f"MOKO: {res.formula_str:<30} | R²={res.r_squared:.4f}")
    print("="*65)
