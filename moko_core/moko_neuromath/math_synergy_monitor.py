"""
MOKO Math Synergy Monitor (MSM)
================================
Dashboard kesehatan matematis seluruh subsystem MOKO OS.

Menggunakan pilar-pilar dari SelfOptimizationMath (SOM):
  - Shannon Entropy   → mengukur diversitas distribusi bobot Hebb dan domain
  - Perplexity        → mengukur kualitas dataset self-training
  - Banach Fixed Point → memverifikasi apakah learning rate konvergen dengan aman

Cara pakai:
    from moko_neuromath.math_synergy_monitor import get_monitor
    monitor = get_monitor()
    report  = monitor.generate_health_report()
    print(report)

Dirancang sebagai komponen pasif (read-only) — tidak mengubah state apapun.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from moko_neuromath.self_optimization_math import (
    InformationTheory,
    ConvergenceTheory,
    get_som,
)

try:
    from moko_config import settings
    _WORKSPACE_DIR = settings.WORKSPACE_DIR
except Exception:
    _WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Konstanta ─────────────────────────────────────────────────────────────────

HEBB_ENTROPY_HEALTHY_MIN   = 1.0   # bit — di bawah ini bobot terlalu terpusat (saturasi)
HEBB_ENTROPY_HEALTHY_MAX   = 5.5   # bit — di atas ini terlalu uniform (tidak belajar)
PERPLEXITY_GOOD_MAX        = 20.0  # PP dataset training — di atas ini model terlalu ragu
BANACH_SAFE_LR             = 0.20  # Learning rate maksimum agar sistem converge dengan aman
BANACH_K_THRESHOLD         = 0.99  # Lipschitz constant — di bawah 1.0 dijamin converge (Hebb normal: k=1-eta)


class MathSynergyMonitor:
    """
    Monitor kesehatan matematis seluruh subsystem MOKO OS.

    Menganalisis:
      1. Hebb Assembly Health    — entropi distribusi bobot sinapsis
      2. Training Data Quality   — perplexity dari confidence scores dataset
      3. Learning Rate Safety    — apakah Hebb LR masuk zona konvergensi Banach?
      4. Domain Balance          — entropi distribusi domain di self-model DMN
    """

    def __init__(self, workspace_dir: str = _WORKSPACE_DIR, verbose: bool = False):
        workspace = Path(workspace_dir)
        self.hebb_path          = workspace / ".math_omni" / "hebb_assemblies.jsonl"
        self.training_data_path = workspace.parent / "moko_data" / "self_training_data.jsonl"
        self.dmn_state_path     = workspace / ".math_omni" / "dmn_state.json"
        self.consolidation_log  = workspace / ".math_omni" / "sleep_consolidation_log.jsonl"
        self.verbose            = verbose
        self._som               = get_som(verbose=False)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  📊 [MathMonitor] {msg}")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. HEBB ASSEMBLY HEALTH
    # ══════════════════════════════════════════════════════════════════════════

    def compute_hebb_entropy(self) -> Dict[str, Any]:
        """
        Hitung Shannon Entropy dari distribusi bobot Hebbian.

        Interpretasi:
          H rendah (< 1.0 bit) → bobot sangat terkonsentrasi → saturasi / sedikit link aktif
          H tinggi (> 5.5 bit) → bobot terdistribusi merata → sistem belum belajar pola
          H ideal (1–4 bit)    → distribusi sehat: beberapa link dominan + banyak lemah
        """
        if not self.hebb_path.exists():
            return {
                "status": "no_data",
                "entropy_bits": None,
                "assembly_count": 0,
                "health": "UNKNOWN",
                "detail": "File hebb_assemblies.jsonl belum ada."
            }

        weights = []
        try:
            with open(self.hebb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            w = obj.get("weight", 0.0)
                            if w > 0:
                                weights.append(w)
                        except Exception:
                            pass
        except Exception as e:
            return {"status": "error", "error": str(e), "health": "ERROR"}

        if not weights:
            return {
                "status": "empty",
                "entropy_bits": 0.0,
                "assembly_count": 0,
                "health": "EMPTY",
                "detail": "Tidak ada Hebbian link aktif."
            }

        # Normalisasi bobot menjadi distribusi probabilitas
        total_w = sum(weights)
        probs   = [w / total_w for w in weights]

        entropy = InformationTheory.entropy(probs, base=2.0)

        # Klasifikasi kesehatan
        if entropy < HEBB_ENTROPY_HEALTHY_MIN:
            health  = "SATURATED"
            detail  = f"Distribusi bobot terpusat (H={entropy:.2f}b). Hanya sedikit link aktif dominan — potensi saturasi Hebb."
        elif entropy > HEBB_ENTROPY_HEALTHY_MAX:
            health  = "UNDIFFERENTIATED"
            detail  = f"Distribusi bobot terlalu merata (H={entropy:.2f}b). Sistem belum membentuk pola belajar."
        else:
            health  = "HEALTHY"
            detail  = f"Distribusi bobot sehat (H={entropy:.2f}b). Campuran link kuat dan lemah yang seimbang."

        # Statistik tambahan
        w_arr = sorted(weights, reverse=True)
        top10 = sum(w_arr[:10]) / total_w if len(w_arr) >= 10 else sum(w_arr) / total_w

        return {
            "status": "ok",
            "entropy_bits": round(entropy, 4),
            "assembly_count": len(weights),
            "weight_mean": round(sum(weights) / len(weights), 4),
            "weight_max": round(max(weights), 4),
            "weight_min": round(min(weights), 4),
            "top10_concentration": round(top10, 4),
            "health": health,
            "detail": detail,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 2. TRAINING DATA QUALITY (PERPLEXITY PROXY)
    # ══════════════════════════════════════════════════════════════════════════

    def compute_training_perplexity(self) -> Dict[str, Any]:
        """
        Estimasi kualitas dataset self-training menggunakan Perplexity proxy.

        Proxy: rasio panjang <think> vs panjang answer akhir sebagai "confidence signal".
        Trajectory panjang + jawaban pendek → uncertainty tinggi (PP tinggi).
        Trajectory ringkas + jawaban jelas → confidence tinggi (PP rendah, bagus).
        """
        path = self.training_data_path
        if not path.exists():
            return {
                "status": "no_data",
                "sample_count": 0,
                "perplexity_proxy": None,
                "health": "UNKNOWN",
                "detail": "File self_training_data.jsonl belum ada."
            }

        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception:
                            pass
        except Exception as e:
            return {"status": "error", "error": str(e), "health": "ERROR"}

        if not samples:
            return {
                "status": "empty",
                "sample_count": 0,
                "perplexity_proxy": None,
                "health": "EMPTY",
                "detail": "Dataset training kosong."
            }

        # Hitung rasio confidence per sampel
        # Proxy: P(conf_i) = len(answer) / (len(think) + len(answer) + 1)
        # Semakin besar P → semakin tinggi confidence relatif
        conf_probs = []
        for s in samples:
            output = s.get("output", "")
            think_start = output.find("<think>")
            think_end   = output.find("</think>")
            if think_start >= 0 and think_end > think_start:
                think_len  = think_end - think_start
                answer_len = len(output) - think_end
            else:
                think_len  = 0
                answer_len = len(output)
            total_len = think_len + answer_len + 1
            conf       = answer_len / total_len
            conf_probs.append(max(1e-6, conf))

        perplexity = InformationTheory.perplexity(conf_probs)

        if perplexity <= PERPLEXITY_GOOD_MAX:
            health = "GOOD"
            detail = f"Kualitas dataset baik (PP={perplexity:.1f}). Jawaban relatif ringkas dan percaya diri."
        elif perplexity <= 50.0:
            health = "MODERATE"
            detail = f"Kualitas dataset sedang (PP={perplexity:.1f}). Beberapa trajektori terlalu panjang."
        else:
            health = "POOR"
            detail = f"Kualitas dataset buruk (PP={perplexity:.1f}). Terlalu banyak chain-of-thought panjang tanpa jawaban jelas."

        return {
            "status": "ok",
            "sample_count": len(samples),
            "perplexity_proxy": round(perplexity, 2),
            "mean_confidence": round(sum(conf_probs) / len(conf_probs), 4),
            "health": health,
            "detail": detail,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. LEARNING RATE CONVERGENCE (BANACH FIXED POINT)
    # ══════════════════════════════════════════════════════════════════════════

    def check_banach_convergence(self, eta: float = 0.08) -> Dict[str, Any]:
        """
        Verifikasi apakah Hebb learning rate `eta` memenuhi syarat konvergensi Banach.

        Banach Fixed-Point Theorem (kontraksi):
          Mapping T konvergen jika ‖T(x) - T(y)‖ ≤ k·‖x - y‖ untuk k < 1.
          Untuk Hebb update: w_{t+1} = w_t + eta*(1 - w_t)
          Lipschitz constant k = |1 - eta| (perlu k < 1)

        Jika eta terlalu besar → sistem tidak konvergen (oscillasi/diverge).
        Jika eta terlalu kecil → belajar terlalu lambat.
        """
        # Mapping Hebb: T(w) = w + eta*(1 - w) = (1-eta)*w + eta
        # Ini adalah kontraksi dengan k = (1-eta) asalkan 0 < eta <= 1
        # k = |1 - eta|: valid only when eta ∈ (0, 1].
        # If eta > 1, the Hebb weight mapping overshoots → oscillation / divergence
        k_estimated    = abs(1.0 - eta)
        is_valid_range = 0 < eta <= 1.0   # Physical requirement for Hebb update
        is_contraction = is_valid_range and (k_estimated < BANACH_K_THRESHOLD)
        is_safe        = eta <= BANACH_SAFE_LR

        # Estimasi kecepatan konvergensi
        if is_contraction:
            # Jumlah iterasi untuk error < 1e-6 dari titik awal |w_0 - w*| = 1
            # Error_n <= k^n * |w_0 - w*| < tol → n > log(tol) / log(k)
            if k_estimated > 0 and k_estimated < 1:
                n_iters = math.ceil(math.log(1e-6) / math.log(k_estimated))
            else:
                n_iters = 0
        else:
            n_iters = None

        if is_contraction and is_safe:
            health = "SAFE_CONVERGING"
            detail = f"eta={eta:.4f} aman. k={k_estimated:.4f} < 0.9. Konvergen dalam ~{n_iters} langkah."
        elif is_contraction and not is_safe:
            health = "CONVERGING_BUT_AGGRESSIVE"
            detail = f"eta={eta:.4f} terlalu besar (>{BANACH_SAFE_LR}). Masih konvergen (k={k_estimated:.4f}) tapi rentan oscillasi."
        else:
            health = "DIVERGING_RISK"
            detail = f"eta={eta:.4f} sangat besar! k={k_estimated:.4f} >= 0.9. Risiko divergensi Hebb!"

        return {
            "eta": eta,
            "lipschitz_k": round(k_estimated, 4),
            "is_contraction": is_contraction,
            "convergence_iterations_est": n_iters,
            "health": health,
            "detail": detail,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 4. DOMAIN BALANCE (DMN SELF-MODEL ENTROPY)
    # ══════════════════════════════════════════════════════════════════════════

    def compute_domain_balance(self) -> Dict[str, Any]:
        """
        Hitung Shannon Entropy dari distribusi domain di DMN self-model.

        Entropi tinggi → MOKO belajar dari banyak domain berbeda (sehat, generalist).
        Entropi rendah → MOKO terlalu terfokus pada satu domain (specialist, kurang fleksibel).
        """
        if not self.dmn_state_path.exists():
            return {
                "status": "no_data",
                "entropy_bits": None,
                "domain_count": 0,
                "health": "UNKNOWN",
                "detail": "File dmn_state.json belum ada."
            }

        try:
            state = json.loads(self.dmn_state_path.read_text(encoding='utf-8'))
            domain_dist = state.get("self_model", {}).get("domain_distribution", {})
        except Exception as e:
            return {"status": "error", "error": str(e), "health": "ERROR"}

        if not domain_dist:
            return {
                "status": "empty",
                "entropy_bits": 0.0,
                "domain_count": 0,
                "health": "UNDIFFERENTIATED",
                "detail": "Belum ada riwayat domain dalam DMN self-model."
            }

        total_queries = sum(domain_dist.values())
        probs = [cnt / total_queries for cnt in domain_dist.values()]
        entropy = InformationTheory.entropy(probs, base=2.0)

        max_entropy = math.log2(len(domain_dist)) if len(domain_dist) > 1 else 1.0
        balance_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

        if balance_ratio >= 0.8:
            health = "BALANCED"
            detail = f"Domain seimbang (H={entropy:.2f}b, balance={balance_ratio:.0%}). MOKO belajar luas."
        elif balance_ratio >= 0.5:
            health = "MODERATE"
            detail = f"Domain cukup beragam (H={entropy:.2f}b, balance={balance_ratio:.0%}). Bisa lebih luas."
        else:
            health = "SPECIALIZED"
            detail = f"MOKO sangat terfokus (H={entropy:.2f}b, balance={balance_ratio:.0%}). Generalisasi rendah."

        # Temukan domain teratas
        top_domain = max(domain_dist, key=domain_dist.get) if domain_dist else "N/A"

        return {
            "status": "ok",
            "entropy_bits": round(entropy, 4),
            "domain_count": len(domain_dist),
            "total_queries": total_queries,
            "balance_ratio": round(balance_ratio, 4),
            "top_domain": top_domain,
            "distribution": domain_dist,
            "health": health,
            "detail": detail,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 5. SLEEP CONSOLIDATION STATS
    # ══════════════════════════════════════════════════════════════════════════

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Baca statistik konsolidasi tidur terakhir."""
        if not self.consolidation_log.exists():
            return {"status": "no_data", "last_session": None}

        last = None
        try:
            with open(self.consolidation_log, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            last = json.loads(line)
                        except Exception:
                            pass
        except Exception:
            pass

        if not last:
            return {"status": "empty", "last_session": None}

        age_hours = (time.time() - last.get("timestamp", 0)) / 3600.0
        return {
            "status": "ok",
            "last_session": last,
            "age_hours": round(age_hours, 2),
            "health": "STALE" if age_hours > 24 else "RECENT",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN REPORT
    # ══════════════════════════════════════════════════════════════════════════

    def generate_health_report(self, hebb_eta: float = 0.08) -> Dict[str, Any]:
        """
        Generate laporan kesehatan matematis lengkap seluruh subsystem MOKO OS.

        Returns:
            Dict dengan 5 section: hebb, training, convergence, domain, sleep
            + overall_score (0–100) dan overall_health string
        """
        self._log("Menjalankan analisis kesehatan matematis penuh...")

        hebb_report   = self.compute_hebb_entropy()
        train_report  = self.compute_training_perplexity()
        banach_report = self.check_banach_convergence(eta=hebb_eta)
        domain_report = self.compute_domain_balance()
        sleep_report  = self.get_consolidation_stats()

        # Hitung overall score
        score = 100.0
        issues = []

        # Hebb health
        hebb_health = hebb_report.get("health", "UNKNOWN")
        if hebb_health == "SATURATED":
            score -= 20
            issues.append("⚠️  Hebb weights mendekati saturasi — perlu pruning atau LTD lebih agresif")
        elif hebb_health in ("UNDIFFERENTIATED", "EMPTY", "UNKNOWN"):
            score -= 10
            issues.append("ℹ️  Hebb assemblies belum terbentuk atau terlalu uniform")

        # Training quality
        train_health = train_report.get("health", "UNKNOWN")
        if train_health == "POOR":
            score -= 20
            issues.append("⚠️  Kualitas dataset training buruk — trajektori terlalu panjang/tidak jelas")
        elif train_health == "MODERATE":
            score -= 10
            issues.append("ℹ️  Kualitas dataset training sedang — bisa dioptimalkan")

        # Convergence
        banach_health = banach_report.get("health", "")
        if banach_health == "DIVERGING_RISK":
            score -= 30
            issues.append("🚨 KRITIS: Learning rate Hebb terlalu besar — risiko divergensi sistem!")
        elif banach_health == "CONVERGING_BUT_AGGRESSIVE":
            score -= 10
            issues.append("⚠️  Learning rate Hebb agresif — monitor stabilitas")

        # Domain balance
        domain_health = domain_report.get("health", "UNKNOWN")
        if domain_health == "SPECIALIZED":
            score -= 10
            issues.append("ℹ️  Domain terlalu terspesialisasi — pertimbangkan cross-domain training")

        # Sleep recency
        sleep_health = sleep_report.get("health", "UNKNOWN")
        if sleep_health == "STALE":
            score -= 10
            issues.append("ℹ️  Konsolidasi tidur sudah > 24 jam — jalankan sleep cycle")

        score = max(0.0, min(100.0, score))

        if score >= 85:
            overall = "EXCELLENT"
            summary = "✅ Sistem matematika MOKO dalam kondisi prima."
        elif score >= 65:
            overall = "GOOD"
            summary = "🟡 Sistem matematika baik, ada beberapa area yang bisa dioptimalkan."
        elif score >= 45:
            overall = "DEGRADED"
            summary = "⚠️  Sistem matematika terdegradasi. Perbaikan disarankan."
        else:
            overall = "CRITICAL"
            summary = "🚨 Kondisi kritis! Intervensi segera diperlukan."

        report = {
            "timestamp": time.time(),
            "overall_score": round(score, 1),
            "overall_health": overall,
            "summary": summary,
            "issues": issues,
            "sections": {
                "hebb_assembly": hebb_report,
                "training_quality": train_report,
                "banach_convergence": banach_report,
                "domain_balance": domain_report,
                "sleep_consolidation": sleep_report,
            }
        }

        if self.verbose:
            self._print_report(report)

        return report

    def _print_report(self, report: Dict):
        """Print laporan dalam format tabel yang terbaca."""
        print("\n" + "═"*60)
        print("  📊 MOKO MATH SYNERGY HEALTH REPORT")
        print("═"*60)
        print(f"  Overall Score : {report['overall_score']}/100")
        print(f"  Overall Health: {report['overall_health']}")
        print(f"  Summary       : {report['summary']}")

        if report["issues"]:
            print("\n  Issues Terdeteksi:")
            for issue in report["issues"]:
                print(f"    {issue}")

        s = report["sections"]
        print("\n  ─── Detil Subsystem ───────────────────────────────────")
        print(f"  Hebb Assembly : {s['hebb_assembly'].get('health','?')} "
              f"| H={s['hebb_assembly'].get('entropy_bits','N/A')}b "
              f"| {s['hebb_assembly'].get('assembly_count',0)} links")
        print(f"  Training Data : {s['training_quality'].get('health','?')} "
              f"| PP={s['training_quality'].get('perplexity_proxy','N/A')} "
              f"| {s['training_quality'].get('sample_count',0)} samples")
        print(f"  LR Convergence: {s['banach_convergence'].get('health','?')} "
              f"| eta={s['banach_convergence'].get('eta','N/A')} "
              f"| k={s['banach_convergence'].get('lipschitz_k','N/A')}")
        print(f"  Domain Balance: {s['domain_balance'].get('health','?')} "
              f"| H={s['domain_balance'].get('entropy_bits','N/A')}b "
              f"| {s['domain_balance'].get('domain_count',0)} domains")
        print(f"  Sleep Cycle   : {s['sleep_consolidation'].get('health','?')} "
              f"| Last: {s['sleep_consolidation'].get('age_hours','N/A')}h ago")
        print("═"*60 + "\n")


# ── Singleton ──────────────────────────────────────────────────────────────────

_monitor_instance: Optional[MathSynergyMonitor] = None

def get_monitor(verbose: bool = True) -> MathSynergyMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = MathSynergyMonitor(verbose=verbose)
    return _monitor_instance


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 MathSynergyMonitor — Self Test")
    monitor = MathSynergyMonitor(verbose=True)
    report  = monitor.generate_health_report(hebb_eta=0.08)

    # Validasi struktur laporan
    assert "overall_score" in report
    assert "sections" in report
    assert "hebb_assembly" in report["sections"]
    assert "training_quality" in report["sections"]
    assert "banach_convergence" in report["sections"]
    assert "domain_balance" in report["sections"]

    # Banach check selalu bisa dijalankan
    b = monitor.check_banach_convergence(eta=0.08)
    assert b["is_contraction"] == True, f"eta=0.08 seharusnya kontraksi: {b}"
    assert b["health"] == "SAFE_CONVERGING", f"Health tidak sesuai: {b}"

    b_danger = monitor.check_banach_convergence(eta=1.5)
    assert b_danger["is_contraction"] == False, f"eta=1.5 seharusnya diverge: {b_danger}"

    print("\n✅ MathSynergyMonitor — semua test lulus!")
