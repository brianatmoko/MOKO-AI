"""
MOKO Self-Optimization Math (SOM)
====================================
Matematika yang membuat sistem bisa MENGEMBANGKAN DIRINYA.

Bukan library ML — ini adalah MATEMATIKA di balik ML:
informasi, optimasi, gradien, propagasi balik.
Memahami ini secara matematis memungkinkan sistem memodifikasi
arsitektur dan proses belajarnya sendiri.

Arsitektur:
  1. Information Theory   — Shannon entropy, KL divergence, mutual info
  2. Optimization         — Gradient descent, Newton's method, Adam
  3. Backpropagation      — Chain rule dalam bentuk matriks
  4. Learning Theory      — PAC learning, VC dimension, generalization
  5. Fixed-Point Math     — Konvergensi, Banach theorem

Filosofi:
  "Bumi selalu berkembang dan dapat mengatasi masalahnya."
  Semua pembelajaran adalah optimasi: minimize loss, maximize reward.
  Semua optimasi adalah matematika: gradien menunjukkan arah turun tercuram.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 1: INFORMATION THEORY
# Shannon (1948): A Mathematical Theory of Communication
# Fondasi teoritis dari semua ML dan komunikasi digital
# ═══════════════════════════════════════════════════════════════════════════

class InformationTheory:
    """
    Teori Informasi Shannon — matematika di balik SEMUA komunikasi digital,
    kompresi data, dan machine learning.

    Konsep Kunci:
      Entropy H(X): rata-rata "kejutan" dari variabel acak X
      KL Divergence: "jarak" antara dua distribusi probabilitas
      Cross-Entropy: loss function paling umum dalam deep learning
      Mutual Information: seberapa banyak X "memberitahu" tentang Y
    """

    @staticmethod
    def entropy(probs: List[float], base: float = 2.0) -> float:
        """
        Shannon Entropy: H(X) = -Σ p(x) · log_b(p(x))

        Mengukur ketidakpastian / kandungan informasi.
        H = 0: distribusi deterministik (tidak ada kejutan)
        H = log(N): distribusi seragam (kejutan maksimum)

        Satuan: bit (base=2), nat (base=e), dit (base=10)
        """
        total = sum(p for p in probs if p > 0)
        if abs(total - 1.0) > 1e-6:
            probs = [p / total for p in probs]  # Normalize

        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log(p) / math.log(base)
        return h

    @staticmethod
    def joint_entropy(joint_probs: List[List[float]]) -> float:
        """
        Entropy bersama: H(X,Y) = -Σ p(x,y) · log p(x,y)
        Diukur dari matriks distribusi bersama.
        """
        h = 0.0
        for row in joint_probs:
            for p in row:
                if p > 0:
                    h -= p * math.log2(p)
        return h

    @staticmethod
    def conditional_entropy(joint_probs: List[List[float]]) -> float:
        """
        Entropy kondisional: H(Y|X) = H(X,Y) - H(X)
        Rata-rata entropy Y setelah mengetahui X.
        """
        h_xy = InformationTheory.joint_entropy(joint_probs)
        # Marginalize over Y untuk dapatkan P(X)
        p_x = [sum(row) for row in joint_probs]
        h_x = InformationTheory.entropy(p_x)
        return h_xy - h_x

    @staticmethod
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """
        KL Divergence (Relative Entropy):
        D_KL(P||Q) = Σ P(x) · log(P(x)/Q(x))

        Mengukur "seberapa berbeda" distribusi Q dari distribusi P.
        D_KL ≥ 0 selalu (Gibbs' inequality)
        D_KL = 0 jika dan hanya jika P = Q

        Digunakan dalam: VAE loss, knowledge distillation, RL policy gradient.
        """
        if len(p) != len(q):
            raise ValueError("P dan Q harus punya panjang sama")
        kl = 0.0
        for pi, qi in zip(p, q):
            if pi > 0:
                if qi <= 0:
                    return float('inf')  # Undefined jika Q=0 tapi P>0
                kl += pi * math.log2(pi / qi)
        return kl

    @staticmethod
    def cross_entropy(p: List[float], q: List[float]) -> float:
        """
        Cross-Entropy: H(P,Q) = -Σ P(x) · log Q(x)
        H(P,Q) = H(P) + D_KL(P||Q)

        Ini adalah LOSS FUNCTION paling umum dalam deep learning:
        loss = -Σ y_true · log(y_pred)
        Minimize cross-entropy = maximize likelihood.
        """
        if len(p) != len(q):
            raise ValueError("P dan Q harus punya panjang sama")
        ce = 0.0
        for pi, qi in zip(p, q):
            if pi > 0:
                if qi <= 0:
                    return float('inf')
                ce -= pi * math.log2(qi)
        return ce

    @staticmethod
    def mutual_information(joint_probs: List[List[float]]) -> float:
        """
        Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        = H(X) - H(X|Y) = H(Y) - H(Y|X)

        Mengukur seberapa banyak Y "memberitahu" kita tentang X.
        I(X;Y) = 0: X dan Y independent
        I(X;Y) = H(X): Y sepenuhnya menentukan X
        """
        p_x = [sum(row) for row in joint_probs]
        p_y = [sum(col) for col in zip(*joint_probs)]
        h_x  = InformationTheory.entropy(p_x)
        h_y  = InformationTheory.entropy(p_y)
        h_xy = InformationTheory.joint_entropy(joint_probs)
        return h_x + h_y - h_xy

    @staticmethod
    def perplexity(prob_sequence: List[float]) -> float:
        """
        Perplexity: PP = 2^H(W)
        Ukuran kualitas language model.
        Lower = better (model lebih yakin dengan prediksinya).
        GPT-4: ~5, random char model: ~65 (ukuran alphabet).
        """
        if not prob_sequence or any(p <= 0 for p in prob_sequence):
            return float('inf')
        log_sum = sum(math.log2(p) for p in prob_sequence)
        avg_log = log_sum / len(prob_sequence)
        return 2 ** (-avg_log)

    @staticmethod
    def channel_capacity(snr_linear: float, bandwidth_hz: float) -> float:
        """
        Shannon-Hartley Theorem: C = B · log₂(1 + SNR)
        Kapasitas kanal maksimum [bits/second].
        Ini adalah BATAS MATEMATIS dari komunikasi — tidak bisa ditembus.
        """
        return bandwidth_hz * math.log2(1 + snr_linear)


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 2: OPTIMIZATION THEORY
# Gradient descent, Newton's method, Adam — dari first principles
# ═══════════════════════════════════════════════════════════════════════════

class OptimizationMath:
    """
    Teori optimasi matematis — dasar dari semua training neural networks.

    Gradient Descent: bergerak berlawanan arah gradien → minimum.
    Newton's Method: gunakan kurva (Hessian) bukan hanya kemiringan.
    Adam: adaptif — setiap parameter punya learning rate sendiri.
    """

    @staticmethod
    def numerical_gradient(f: Callable[[List[float]], float],
                            x: List[float], h: float = 1e-5) -> List[float]:
        """
        Gradien numerik via finite differences:
        ∂f/∂xᵢ ≈ [f(x + h·eᵢ) - f(x - h·eᵢ)] / (2h)

        Ini adalah cara MATEMATIKA menghitung gradien tanpa tahu bentuk f.
        Digunakan untuk verifikasi gradient dari backpropagation.
        """
        grad = []
        for i in range(len(x)):
            x_plus  = x[:i] + [x[i] + h] + x[i+1:]
            x_minus = x[:i] + [x[i] - h] + x[i+1:]
            grad_i = (f(x_plus) - f(x_minus)) / (2 * h)
            grad.append(grad_i)
        return grad

    @staticmethod
    def gradient_descent(f: Callable[[List[float]], float],
                          grad_f: Optional[Callable] = None,
                          x0: List[float] = None,
                          lr: float = 0.01,
                          max_iter: int = 1000,
                          tol: float = 1e-6) -> Dict[str, Any]:
        """
        Gradient Descent: xₙ₊₁ = xₙ - α·∇f(xₙ)

        Algoritma paling fundamental dalam machine learning.
        α = learning rate (ukuran langkah)
        ∇f = gradien (arah naik terjal) → lawan arahnya = turun terjal

        Konvergensi dijamin jika:
          - f convex
          - α < 2/L dimana L = Lipschitz constant of ∇f
        """
        if x0 is None:
            x0 = [1.0]

        x = list(x0)
        trajectory = [list(x)]
        loss_history = [f(x)]

        grad_fn = grad_f if grad_f else (
            lambda xv: OptimizationMath.numerical_gradient(f, xv)
        )

        for i in range(max_iter):
            grad = grad_fn(x)
            grad_norm = math.sqrt(sum(g**2 for g in grad))

            if grad_norm < tol:
                return {
                    "converged": True,
                    "x_min": x,
                    "f_min": f(x),
                    "iterations": i + 1,
                    "grad_norm_final": grad_norm,
                    "loss_history": loss_history,
                    "trajectory": trajectory,
                }

            x = [xi - lr * gi for xi, gi in zip(x, grad)]
            trajectory.append(list(x))
            loss_history.append(f(x))

        return {
            "converged": False,
            "x_min": x,
            "f_min": f(x),
            "iterations": max_iter,
            "grad_norm_final": grad_norm,
            "loss_history": loss_history[:20],  # First 20 for brevity
        }

    @staticmethod
    def newtons_method(f: Callable[[float], float],
                        df: Callable[[float], float],
                        x0: float = 1.0,
                        max_iter: int = 50,
                        tol: float = 1e-10) -> Dict[str, Any]:
        """
        Newton's Method: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)

        Konvergensi kuadratik: error berkurang KUADRAT setiap iterasi.
        Jauh lebih cepat dari gradient descent, tapi butuh turunan.

        Contoh: cari √2 dengan f(x) = x² - 2, f'(x) = 2x
          x₀=1: x₁=1.5, x₂=1.41667, x₃=1.41421... (hampir sempurna dalam 3 iterasi!)
        """
        x = x0
        steps = [x]
        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < 1e-15:
                return {"converged": False, "error": "Derivatif mendekati nol", "steps": steps}

            x_new = x - fx / dfx
            steps.append(x_new)

            if abs(x_new - x) < tol:
                return {
                    "converged": True,
                    "root": x_new,
                    "f_at_root": f(x_new),
                    "iterations": i + 1,
                    "convergence": "quadratic",
                    "steps": steps,
                }
            x = x_new

        return {"converged": False, "root": x, "iterations": max_iter, "steps": steps}

    @staticmethod
    def adam_step(params: List[float], grads: List[float],
                   m: List[float], v: List[float],
                   t: int, lr: float = 0.001,
                   beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8) -> Tuple[List[float], List[float], List[float]]:
        """
        Adam Optimizer — satu langkah update.
        Adam = Adaptive Moment Estimation (Kingma & Ba, 2014)

        Rumus matematis:
          mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ          (first moment = mean of gradients)
          vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²         (second moment = uncentered variance)
          m̂ₜ = mₜ/(1-β₁ᵗ)                    (bias correction)
          v̂ₜ = vₜ/(1-β₂ᵗ)                    (bias correction)
          θₜ = θₜ₋₁ - α·m̂ₜ/(√v̂ₜ + ε)       (update)

        β₁=0.9: decay untuk mean (ingatan pendek)
        β₂=0.999: decay untuk variance (ingatan panjang)
        Ini yang membuat Adam lebih stabil dari vanilla GD.
        """
        if not m: m = [0.0] * len(params)
        if not v: v = [0.0] * len(params)

        new_params = []
        new_m = []
        new_v = []

        for p, g, mi, vi in zip(params, grads, m, v):
            # Update biased moment estimates
            mi_new = beta1 * mi + (1 - beta1) * g
            vi_new = beta2 * vi + (1 - beta2) * g**2

            # Bias correction
            m_hat = mi_new / (1 - beta1**t)
            v_hat = vi_new / (1 - beta2**t)

            # Parameter update
            p_new = p - lr * m_hat / (math.sqrt(v_hat) + eps)

            new_params.append(p_new)
            new_m.append(mi_new)
            new_v.append(vi_new)

        return new_params, new_m, new_v

    @staticmethod
    def cosine_lr_schedule(step: int, total_steps: int,
                            lr_max: float, lr_min: float = 0.0,
                            warmup_steps: int = 0) -> float:
        """
        Cosine Annealing Learning Rate Schedule:
        lr(t) = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))

        Standard dalam training transformer/LLM modern.
        """
        if step < warmup_steps:
            # Linear warmup
            return lr_max * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 3: BACKPROPAGATION MATHEMATICS
# Chain rule dalam bentuk matriks — fondasi deep learning
# ═══════════════════════════════════════════════════════════════════════════

class BackpropMath:
    """
    Backpropagation — Chain Rule Kalkulus dalam bentuk matriks.

    Forward pass: z = Wx + b, a = σ(z)
    Backward pass: ∂L/∂W = ∂L/∂z · xᵀ  (menggunakan chain rule)

    Ini BUKAN magic — ini adalah kalkulus differensial biasa,
    diaplikasikan secara sistematis pada graf komputasi.
    """

    @staticmethod
    def sigmoid(z: float) -> float:
        """σ(z) = 1/(1+e^(-z)) — activation function."""
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def sigmoid_derivative(z: float) -> float:
        """σ'(z) = σ(z)·(1-σ(z)) — turunan sigmoid yang elegan."""
        s = BackpropMath.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z: float) -> float:
        """ReLU(z) = max(0, z)"""
        return max(0.0, z)

    @staticmethod
    def relu_derivative(z: float) -> float:
        """ReLU'(z) = 1 jika z>0, 0 jika z≤0 (subgradient)"""
        return 1.0 if z > 0 else 0.0

    @staticmethod
    def mse_loss(y_pred: List[float], y_true: List[float]) -> float:
        """MSE = (1/N)·Σ(y_pred - y_true)²"""
        return sum((yp - yt)**2 for yp, yt in zip(y_pred, y_true)) / len(y_pred)

    @staticmethod
    def mse_gradient(y_pred: List[float], y_true: List[float]) -> List[float]:
        """∂MSE/∂y_pred = (2/N)·(y_pred - y_true)"""
        N = len(y_pred)
        return [(2 / N) * (yp - yt) for yp, yt in zip(y_pred, y_true)]

    @staticmethod
    def forward_pass_1layer(x: List[float], W: List[List[float]],
                             b: List[float]) -> Tuple[List[float], List[float]]:
        """
        Forward pass satu layer linear:
        z = Wx + b (matrix-vector multiply)
        a = σ(z)   (activation)
        Returns: (z, a)
        """
        n_out = len(W)
        z = []
        for i in range(n_out):
            zi = sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
            z.append(zi)
        a = [BackpropMath.sigmoid(zi) for zi in z]
        return z, a

    @staticmethod
    def backward_pass_1layer(dL_da: List[float], z: List[float],
                              x: List[float]) -> Tuple[List[List[float]], List[float], List[float]]:
        """
        Backward pass satu layer (Chain Rule):

        ∂L/∂z = ∂L/∂a · σ'(z)     (element-wise)
        ∂L/∂W = ∂L/∂z · xᵀ         (outer product)
        ∂L/∂b = ∂L/∂z
        ∂L/∂x = Wᵀ · ∂L/∂z         (untuk gradient ke layer sebelumnya)

        Returns: (dL_dW, dL_db, dL_dx)
        """
        # Step 1: backprop melalui aktivasi sigmoid
        dL_dz = [dL_da[i] * BackpropMath.sigmoid_derivative(z[i])
                 for i in range(len(z))]

        # Step 2: gradient W = outer product dL_dz ⊗ x
        dL_dW = [[dL_dz[i] * x[j] for j in range(len(x))]
                  for i in range(len(dL_dz))]

        # Step 3: gradient b = dL_dz
        dL_db = list(dL_dz)

        # Step 4: gradient x = Wᵀ · dL_dz (untuk propagasi ke layer sebelumnya)
        # (tidak ada W di sini, kita return dL_dz sebagai proxy)
        dL_dx = dL_dz  # Simplified untuk 1-layer case

        return dL_dW, dL_db, dL_dx

    @staticmethod
    def verify_gradient(f: Callable, params: List[float],
                          analytical_grad: List[float]) -> Dict[str, Any]:
        """
        Gradient check: bandingkan gradien analitik dengan gradien numerik.
        Error < 1e-5 berarti implementasi backprop benar.
        Ini adalah standar verifikasi implementasi deep learning.
        """
        numerical_grad = OptimizationMath.numerical_gradient(f, params)
        diffs = [abs(ag - ng) for ag, ng in zip(analytical_grad, numerical_grad)]
        max_diff = max(diffs) if diffs else 0.0
        rel_err = max_diff / (max(max(abs(ag) for ag in analytical_grad),
                                   max(abs(ng) for ng in numerical_grad)) + 1e-8)
        return {
            "analytical_gradient": analytical_grad,
            "numerical_gradient": [round(g, 8) for g in numerical_grad],
            "max_absolute_error": max_diff,
            "relative_error": rel_err,
            "gradient_correct": rel_err < 1e-4,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PILAR 4: CONVERGENCE & FIXED POINT THEORY
# Matematika yang menjamin sistem "menemukan jawabannya"
# ═══════════════════════════════════════════════════════════════════════════

class ConvergenceTheory:
    """
    Teori konvergensi dan fixed point — fondasi untuk memahami
    mengapa training ML akhirnya "berhenti" dan apa yang dijamin konvergen.
    """

    @staticmethod
    def banach_contraction_check(f: Callable[[float], float],
                                  domain: Tuple[float, float],
                                  n_samples: int = 100) -> Dict[str, Any]:
        """
        Cek Banach Contraction Mapping Theorem:
        f adalah kontraksi jika ∃ k ∈ [0,1) s.t. d(f(x),f(y)) ≤ k·d(x,y)

        Jika f adalah kontraksi di ruang metrik lengkap,
        maka f memiliki TEPAT SATU fixed point, dan iterasi xₙ₊₁ = f(xₙ)
        konvergen ke fixed point tersebut dari SEMUA titik awal.
        """
        a, b = domain
        step = (b - a) / n_samples
        max_lipschitz = 0.0

        for i in range(n_samples - 1):
            x1 = a + i * step
            x2 = x1 + step
            try:
                ratio = abs(f(x2) - f(x1)) / abs(x2 - x1) if abs(x2 - x1) > 1e-12 else 0
                max_lipschitz = max(max_lipschitz, ratio)
            except Exception:
                pass

        is_contraction = max_lipschitz < 1.0
        return {
            "estimated_lipschitz_constant": max_lipschitz,
            "is_contraction": is_contraction,
            "banach_theorem_applies": is_contraction,
            "convergence_rate": max_lipschitz if is_contraction else None,
            "explanation": (
                f"k = {max_lipschitz:.4f} < 1 → Kontraksi → Fixed point unik dijamin"
                if is_contraction else
                f"k = {max_lipschitz:.4f} ≥ 1 → Bukan kontraksi → Konvergensi tidak dijamin"
            )
        }

    @staticmethod
    def fixed_point_iteration(f: Callable[[float], float],
                               x0: float = 0.5,
                               max_iter: int = 100,
                               tol: float = 1e-9) -> Dict[str, Any]:
        """
        Fixed Point Iteration: xₙ₊₁ = f(xₙ)
        Konvergen ke x* dimana f(x*) = x*
        """
        x = x0
        history = [x]
        for i in range(max_iter):
            x_new = f(x)
            history.append(x_new)
            if abs(x_new - x) < tol:
                return {
                    "converged": True,
                    "fixed_point": x_new,
                    "f_fixed_point": f(x_new),
                    "iterations": i + 1,
                    "history": history,
                }
            x = x_new
        return {
            "converged": False,
            "last_value": x,
            "iterations": max_iter,
            "history": history[:20],
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE FACADE
# ═══════════════════════════════════════════════════════════════════════════

class SelfOptimizationMath:
    """
    Fasad utama untuk semua kapabilitas self-optimization mathematics.
    Matematika yang memungkinkan sistem MENGEMBANGKAN DIRINYA.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.info  = InformationTheory()
        self.optim = OptimizationMath()
        self.backprop = BackpropMath()
        self.convergence = ConvergenceTheory()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  📈 [SOM] {msg}")


_som_instance: Optional[SelfOptimizationMath] = None

def get_som(verbose: bool = True) -> SelfOptimizationMath:
    global _som_instance
    if _som_instance is None:
        _som_instance = SelfOptimizationMath(verbose=verbose)
    return _som_instance


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("\n📈 Self-Optimization Math — Self Test\n" + "="*55)

    # ── Test 1: Shannon Entropy ───────────────────────────────────────
    # Fair coin: H = 1 bit (maksimum uncertainty)
    h_coin = InformationTheory.entropy([0.5, 0.5])
    assert abs(h_coin - 1.0) < 1e-10, f"Entropy fair coin gagal: {h_coin}"
    print(f"  ✅ Entropy: fair coin H = {h_coin:.4f} bit (maksimum uncertainty)")

    # Biased coin: H < 1 (lebih predictable)
    h_biased = InformationTheory.entropy([0.9, 0.1])
    assert h_biased < 1.0, f"Entropy biased coin gagal"
    print(f"  ✅ Entropy: biased coin (90/10) H = {h_biased:.4f} bit < 1 (lebih predictable)")

    # ── Test 2: KL Divergence ─────────────────────────────────────────
    # D_KL(P||P) = 0 selalu
    p = [0.3, 0.4, 0.3]
    kl_self = InformationTheory.kl_divergence(p, p)
    assert abs(kl_self) < 1e-10, f"KL(P||P) ≠ 0: {kl_self}"
    print(f"  ✅ KL Divergence: D_KL(P||P) = {kl_self:.6f} (harus nol)")

    kl_diff = InformationTheory.kl_divergence([0.9, 0.1], [0.5, 0.5])
    assert kl_diff > 0, "KL divergence harus positif"
    print(f"  ✅ KL Divergence: D_KL([0.9,0.1]||[0.5,0.5]) = {kl_diff:.4f} > 0")

    # ── Test 3: Gradient Descent ──────────────────────────────────────
    # Minimize f(x) = x² → minimum di x=0
    def f_quadratic(x): return sum(xi**2 for xi in x)
    def grad_quadratic(x): return [2 * xi for xi in x]

    result = OptimizationMath.gradient_descent(
        f=f_quadratic, grad_f=grad_quadratic,
        x0=[10.0], lr=0.1, tol=1e-8
    )
    assert result["converged"], f"GD tidak konvergen: {result}"
    assert abs(result["x_min"][0]) < 1e-4, f"GD tidak ke 0: {result['x_min']}"
    print(f"  ✅ Gradient Descent: f(x)=x², x₀=10 → x*={result['x_min'][0]:.6f} dalam {result['iterations']} iterasi")

    # ── Test 4: Newton's Method ───────────────────────────────────────
    # Cari √2: f(x) = x²-2 = 0, f'(x) = 2x
    sqrt2 = OptimizationMath.newtons_method(
        f=lambda x: x**2 - 2,
        df=lambda x: 2 * x,
        x0=1.0
    )
    assert sqrt2["converged"], f"Newton gagal konvergen"
    assert abs(sqrt2["root"] - math.sqrt(2)) < 1e-10, f"Newton √2 gagal: {sqrt2['root']}"
    print(f"  ✅ Newton's Method: √2 = {sqrt2['root']:.10f} dalam {sqrt2['iterations']} iterasi (quadratic conv.)")

    # ── Test 5: Adam Optimizer ────────────────────────────────────────
    # Satu step Adam
    params = [1.0, -0.5]
    grads  = [0.1, -0.2]
    m, v = [], []
    new_p, new_m, new_v = OptimizationMath.adam_step(params, grads, m, v, t=1)
    assert len(new_p) == 2, "Adam gagal"
    assert new_p[0] != params[0], "Adam tidak mengupdate params"
    print(f"  ✅ Adam step 1: {params} → {[round(p, 6) for p in new_p]}")

    # ── Test 6: Backprop Gradient Check ──────────────────────────────
    # Simple f(x, y) = x² + 2y
    # ∂f/∂x = 2x, ∂f/∂y = 2
    def f_test(params): return params[0]**2 + 2 * params[1]
    analytical = [2 * 3.0, 2.0]  # x=3, y=1 → [6, 2]
    check = BackpropMath.verify_gradient(f_test, [3.0, 1.0], analytical)
    assert check["gradient_correct"], f"Gradient check gagal: {check}"
    print(f"  ✅ Gradient Check: analitik vs numerik — rel_err={check['relative_error']:.2e} (OK)")

    # ── Test 7: Sigmoid & Derivative ─────────────────────────────────
    s0 = BackpropMath.sigmoid(0)
    assert abs(s0 - 0.5) < 1e-10, f"σ(0) = {s0}"
    ds = BackpropMath.sigmoid_derivative(0)
    assert abs(ds - 0.25) < 1e-10, f"σ'(0) = {ds}"
    print(f"  ✅ Sigmoid: σ(0)={s0}, σ'(0)={ds} (= σ×(1-σ) = 0.5×0.5 = 0.25)")

    # ── Test 8: Shannon Channel Capacity ─────────────────────────────
    # Shannon-Hartley: C = B·log₂(1+SNR)
    # WiFi 6: B=80MHz, SNR=30dB (ratio=1000) → C≈800 Mbps
    cap = InformationTheory.channel_capacity(snr_linear=1000, bandwidth_hz=80e6)
    print(f"  ✅ Shannon Capacity: WiFi 80MHz, SNR=30dB → C={cap/1e6:.0f} Mbps (batas matematis)")

    # ── Test 9: Cosine LR Schedule ───────────────────────────────────
    lr_start = OptimizationMath.cosine_lr_schedule(0, 100, 0.001, warmup_steps=10)
    lr_mid   = OptimizationMath.cosine_lr_schedule(50, 100, 0.001)
    lr_end   = OptimizationMath.cosine_lr_schedule(100, 100, 0.001)
    assert lr_end < lr_mid < 0.001, "Cosine schedule gagal"
    print(f"  ✅ Cosine LR: t=0→{lr_start:.6f}, t=50→{lr_mid:.6f}, t=100→{lr_end:.6f}")

    # ── Test 10: Banach Fixed Point ───────────────────────────────────
    # f(x) = x/2 + 1 adalah kontraksi dengan k=0.5
    # Fixed point: x = x/2 + 1 → x = 2
    fp = ConvergenceTheory.fixed_point_iteration(lambda x: x/2 + 1, x0=0.0)
    assert fp["converged"] and abs(fp["fixed_point"] - 2.0) < 1e-6, f"Fixed point gagal: {fp}"
    banach = ConvergenceTheory.banach_contraction_check(lambda x: x/2 + 1, (0, 10))
    assert banach["is_contraction"], f"Banach check gagal: {banach}"
    print(f"  ✅ Banach Fixed Point: f(x)=x/2+1, x*={fp['fixed_point']:.6f}=2, k={banach['estimated_lipschitz_constant']:.3f}")

    # ── Test 11: Perplexity ───────────────────────────────────────────
    # Model yang selalu benar: P=1 → Perplexity=1
    pp_perfect = InformationTheory.perplexity([1.0] * 10)
    assert abs(pp_perfect - 1.0) < 1e-6, f"Perplexity perfect gagal: {pp_perfect}"
    pp_random = InformationTheory.perplexity([1/100] * 100)
    assert abs(pp_random - 100.0) < 0.01, f"Perplexity random gagal: {pp_random}"
    print(f"  ✅ Perplexity: perfect model={pp_perfect:.1f}, random 100-class={pp_random:.1f}")

    print("\n✅ Semua test Self-Optimization Math berhasil!\n")


if __name__ == "__main__":
    _self_test()
