"""
MOKO Applied Math Trainer — Training Loop
==========================================
Sistem training loop untuk melatih ARMS menjawab pertanyaan
matematika terapan nyata. Menggabungkan:

  1. Dataset soal (applied_math_dataset.py)
  2. Solver ARMS (arms_orchestrator.py)
  3. Evaluasi jawaban (AnswerEvaluator)
  4. Adaptive curriculum (berdasarkan weakness)
  5. Progress tracking (training_state.py)

Mode:
  AUTO        — Batch otomatis, cetak ringkasan
  INTERACTIVE — Satu soal sekali, tampilkan penjelasan detail
  EVALUATION  — Evaluasi saja tanpa update state (dry run)
"""

import time
import sys
import os
import uuid
from typing import List, Optional, Dict, Tuple
from datetime import datetime

# Tambahkan path agar import relatif berfungsi
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.dirname(_HERE)
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)
if os.path.dirname(_CORE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CORE))


class AppliedMathTrainer:
    """
    Training loop utama untuk melatih ARMS menjawab soal matematika terapan.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Import komponen
        from moko_neuromath.applied_math_dataset import get_dataset, Difficulty, ProblemVariationGenerator
        from moko_neuromath.arms_orchestrator import get_orchestrator, AnswerEvaluator
        from moko_neuromath.training_state import get_state_manager, TrainingRecord, TrainingSession

        self.dataset = get_dataset()
        self.orchestrator = get_orchestrator(verbose=verbose)
        self.evaluator = AnswerEvaluator()
        self.state = get_state_manager()
        self.Difficulty = Difficulty
        self.TrainingRecord = TrainingRecord
        self.TrainingSession = TrainingSession
        self.ProblemVariationGenerator = ProblemVariationGenerator

        self._current_session_id = str(uuid.uuid4())[:8]
        self._session_records = []

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Trainer] {msg}")

    # ── ADAPTIVE CURRICULUM ────────────────────────────────────────────────

    def _pick_next_problem(self, n: int = 1):
        """
        Pilih soal berikutnya berdasarkan adaptive curriculum:
        - Jika ada domain yang lemah → fokus ke sana
        - Jika belum ada data → mulai dari EASY semua domain
        - Jika mastered semua EASY → mulai MEDIUM
        """
        from moko_neuromath.applied_math_dataset import Difficulty

        weaknesses = self.state.get_weaknesses()
        untested_domains = self.state.get_untested_domains(
            list({p.domain for p in self.dataset.problems})
        )

        problems = []

        # Prioritas 1: domain yang belum pernah dicoba → EASY
        if untested_domains:
            for domain in untested_domains[:3]:
                domain_easy = [p for p in self.dataset.problems
                               if p.domain == domain and p.difficulty == Difficulty.EASY]
                problems.extend(domain_easy[:2])

        # Prioritas 2: domain yang lemah → EASY dulu
        if weaknesses:
            for domain in weaknesses[:2]:
                weak_easy = [p for p in self.dataset.problems
                             if p.domain == domain and p.difficulty == Difficulty.EASY]
                problems.extend(weak_easy[:2])

        # Fallback: campuran semua domain
        if not problems:
            # Prioritas ke MEDIUM jika EASY sudah dikuasai
            all_domains = list({p.domain for p in self.dataset.problems})
            strengths = self.state.get_strengths()

            for domain in all_domains:
                ds = self.state.domain_stats.get(domain)
                if ds is None or ds.easy_total < 2:
                    pool = [p for p in self.dataset.problems
                            if p.domain == domain and p.difficulty == Difficulty.EASY]
                else:
                    pool = [p for p in self.dataset.problems
                            if p.domain == domain and p.difficulty == Difficulty.MEDIUM]
                if pool:
                    import random
                    problems.append(random.choice(pool))

        # Remove duplikat berdasarkan ID
        seen = set()
        unique_problems = []
        for p in problems:
            if p.id not in seen:
                seen.add(p.id)
                unique_problems.append(p)

        import random
        random.shuffle(unique_problems)
        return unique_problems[:n]

    # ── SINGLE PROBLEM SOLVE ──────────────────────────────────────────────

    def solve_problem(self, problem, dry_run: bool = False) -> Dict:
        """
        Solve satu soal dan return hasil evaluasi.

        Returns dict dengan:
          - problem_id, domain, difficulty
          - arms_answer, expected_answer
          - is_correct, percent_error, score
          - elapsed_ms, solution_steps, formula_source
          - verdict
        """
        # Solve via ARMS (mode terstruktur, bukan NLP karena kita punya variabel eksak)
        sol = self.orchestrator.solve_structured(
            domain=problem.domain,
            known=problem.known_vars,
            target=problem.target_symbol,
            story_text=problem.story_text,
        )

        # Evaluasi
        is_correct, pct_err, verdict = self.evaluator.evaluate(
            sol, problem.expected_answer, problem.tolerance_pct
        )
        score = self.evaluator.score_solution(sol, problem.expected_answer, problem.tolerance_pct)

        result = {
            "problem_id": problem.id,
            "domain": problem.domain,
            "difficulty": problem.difficulty.value,
            "tags": problem.tags,
            "story_text": problem.story_text[:120],
            "target_symbol": problem.target_symbol,
            "expected_answer": problem.expected_answer,
            "expected_unit": problem.expected_unit,
            "arms_answer": sol.result_value,
            "arms_unit": sol.result_unit,
            "formula_used": sol.formula_used,
            "formula_source": sol.formula_source,
            "solve_status": sol.status.value,
            "is_correct": is_correct,
            "percent_error": pct_err,
            "score": score,
            "elapsed_ms": sol.elapsed_ms,
            "solution_steps": sol.solution_steps,
            "verdict": verdict,
            "warnings": sol.warnings,
        }

        # Simpan ke state (kecuali dry run)
        if not dry_run:
            rec = self.TrainingRecord(
                problem_id=problem.id,
                domain=problem.domain,
                difficulty=problem.difficulty.value,
                story_text=problem.story_text[:120],
                target_symbol=problem.target_symbol,
                expected_answer=problem.expected_answer,
                arms_answer=sol.result_value,
                formula_source=sol.formula_source,
                is_correct=is_correct,
                percent_error=pct_err,
                score=score,
                elapsed_ms=sol.elapsed_ms,
                tags=problem.tags,
                solve_status=sol.status.value,
            )
            self.state.record(rec)
            self._session_records.append(rec)

        return result

    # ── AUTO MODE ─────────────────────────────────────────────────────────

    def run_auto(self, n_problems: int = 20, save_interval: int = 5) -> Dict:
        """
        Mode otomatis: solve batch soal, cetak ringkasan.

        Args:
            n_problems: Jumlah soal yang dikerjakan
            save_interval: Simpan state setiap N soal

        Returns: dict ringkasan sesi
        """
        session_start = datetime.now().isoformat()
        print(f"\n{'═'*65}")
        print(f"  🚀 MOKO Applied Math Training — AUTO MODE")
        print(f"     Soal yang dikerjakan: {n_problems}")
        print(f"     Dataset: {len(self.dataset.problems)} soal tersedia")
        print(f"     Sesi: #{self._current_session_id}")
        print(f"{'═'*65}\n")

        # Pilih soal
        problems = self._pick_next_problem(n_problems)

        # Jika kurang dari yang diminta, ambil random
        if len(problems) < n_problems:
            import random
            all_p = list(self.dataset.problems)
            random.shuffle(all_p)
            extra = [p for p in all_p if p not in problems]
            problems.extend(extra[:n_problems - len(problems)])

        problems = problems[:n_problems]

        results = []
        n_correct = 0
        total_score = 0.0
        domains_covered = set()

        for i, problem in enumerate(problems, 1):
            result = self.solve_problem(problem)
            results.append(result)

            if result["is_correct"]:
                n_correct += 1
            total_score += result["score"]
            domains_covered.add(result["domain"])

            # Print per-soal progress
            icon = "✅" if result["is_correct"] else "❌"
            arms_ans = result["arms_answer"]
            expected = result["expected_answer"]

            arms_ans_str = f"{arms_ans:.4g}" if arms_ans is not None else "FAIL"
            exp_str = f"{expected:.4g}"
            print(f"  [{i:>2}/{n_problems}] {icon} {result['domain']:<20} "
                  f"[{result['difficulty']:<6}] "
                  f"| {result['target_symbol']:<10} "
                  f"| Jawaban: {arms_ans_str:<12} "
                  f"| Harapan: {exp_str:<12} "
                  f"| {result['verdict'][:30]}")

            # Simpan secara periodik
            if i % save_interval == 0:
                self.state.save()
                self._log(f"State disimpan (checkpoint {i}/{n_problems})")

        # Simpan state final
        session = self.TrainingSession(
            session_id=self._current_session_id,
            start_time=session_start,
            end_time=datetime.now().isoformat(),
            n_problems=len(problems),
            n_correct=n_correct,
            total_score=total_score,
            domains_covered=list(domains_covered),
        )
        self.state.sessions.append(session)
        self.state.save()

        # Ringkasan sesi
        acc = n_correct / len(problems) if problems else 0
        avg_score = total_score / len(problems) if problems else 0

        print(f"\n{'─'*65}")
        print(f"  📋 RINGKASAN SESI #{self._current_session_id}")
        print(f"  ✅ Benar    : {n_correct}/{len(problems)} ({acc:.1%})")
        print(f"  📊 Avg Score: {avg_score:.2f}/1.00")
        print(f"  🎯 Domain   : {', '.join(sorted(domains_covered))}")
        print(f"{'─'*65}")

        # Tampilkan areas for improvement
        weaknesses = self.state.get_weaknesses()
        if weaknesses:
            print(f"\n  ⚠️  Perlu diperkuat: {', '.join(weaknesses)}")
        strengths = self.state.get_strengths()
        if strengths:
            print(f"  🏆 Sudah dikuasai : {', '.join(strengths)}")

        print(f"\n  💾 State disimpan ke: {_HERE.replace('moko_neuromath', '...')}/.math_omni/applied_training_state.json")

        return {
            "session_id": self._current_session_id,
            "n_problems": len(problems),
            "n_correct": n_correct,
            "accuracy": acc,
            "avg_score": avg_score,
            "domains": list(domains_covered),
        }

    # ── INTERACTIVE MODE ──────────────────────────────────────────────────

    def run_interactive(self, n_problems: int = 5):
        """
        Mode interaktif: tampilkan soal, solver menjawab, tampilkan penjelasan.
        """
        from moko_neuromath.applied_math_dataset import Difficulty

        print(f"\n{'═'*65}")
        print(f"  🎓 MOKO Applied Math Training — INTERACTIVE MODE")
        print(f"{'═'*65}")
        print("  Tekan ENTER untuk lanjut ke soal berikutnya, 'q' untuk berhenti.\n")

        problems = self._pick_next_problem(n_problems)
        if not problems:
            problems = self.dataset.sample(n=n_problems)

        for i, problem in enumerate(problems[:n_problems], 1):
            print(f"\n{'─'*65}")
            print(f"  SOAL {i}/{n_problems} | Domain: {problem.domain} | "
                  f"Difficulty: {problem.difficulty.value.upper()}")
            print(f"{'─'*65}")
            print(f"  📝 {problem.story_text}")
            print(f"\n  ❓ Dicari: {problem.target_symbol} = ?")
            print(f"  📊 Variabel yang diketahui:")
            for sym, val in problem.known_vars.items():
                print(f"     {sym} = {val:.6g}")

            input("\n  [ENTER untuk lihat jawaban ARMS...] ")

            result = self.solve_problem(problem)

            # Tampilkan jawaban ARMS
            print(f"\n  🤖 JAWABAN MOKO ARMS:")
            if result["arms_answer"] is not None:
                print(f"     {result['target_symbol']} = {result['arms_answer']:.6g} {result['arms_unit']}")
                print(f"     Rumus: {result['formula_used']}")
                print(f"     Sumber: {result['formula_source'].upper()}")
            else:
                print(f"     ❌ Gagal menjawab (status: {result['solve_status']})")

            print(f"\n  {result['verdict']}")
            print(f"\n  ✅ JAWABAN BENAR: {problem.expected_answer:.6g} {problem.expected_unit}")
            print(f"\n  📖 LANGKAH PENYELESAIAN:")
            for step in problem.solution_steps:
                print(f"     {step}")

            if result["warnings"]:
                print(f"\n  ⚠️  Peringatan: {result['warnings']}")

            choice = input("\n  [ENTER lanjut | 'q' keluar]: ")
            if choice.lower() == 'q':
                break

        # Simpan state
        self.state.save()
        print(f"\n  ✅ Sesi interaktif selesai. Progress tersimpan.")
        print(self.state.generate_report())

    # ── EVALUATION MODE ───────────────────────────────────────────────────

    def run_evaluation(self, n_problems: int = 30) -> Dict:
        """
        Mode evaluasi: test seluruh dataset tanpa menyimpan ke state (dry run).
        Menghasilkan snapshot akurasi saat ini.
        """
        print(f"\n{'═'*65}")
        print(f"  🔬 MOKO Applied Math — EVALUATION MODE (Dry Run)")
        print(f"{'═'*65}\n")

        import random
        problems = random.sample(self.dataset.problems, min(n_problems, len(self.dataset.problems)))

        results_by_domain: Dict[str, List] = {}
        total_correct = 0

        for problem in problems:
            result = self.solve_problem(problem, dry_run=True)

            dom = result["domain"]
            if dom not in results_by_domain:
                results_by_domain[dom] = []
            results_by_domain[dom].append(result)

            if result["is_correct"]:
                total_correct += 1

        # Cetak per domain
        print(f"  {'Domain':<22} {'Acc':>6} {'Benar':>6} {'Total':>6} {'Avg Score':>9}")
        print("  " + "-" * 58)

        for domain in sorted(results_by_domain.keys()):
            rlist = results_by_domain[domain]
            correct = sum(1 for r in rlist if r["is_correct"])
            acc = correct / len(rlist)
            avg_score = sum(r["score"] for r in rlist) / len(rlist)
            print(f"  {domain:<22} {acc:>5.1%} {correct:>6} {len(rlist):>6} {avg_score:>9.2f}")

        overall_acc = total_correct / len(problems) if problems else 0
        avg_overall_score = sum(r["score"] for p in results_by_domain.values() for r in p) / len(problems) if problems else 0

        print("  " + "=" * 58)
        print(f"  {'OVERALL':<22} {overall_acc:>5.1%} {total_correct:>6} {len(problems):>6} {avg_overall_score:>9.2f}")

        # Per difficulty breakdown
        print(f"\n  PER DIFFICULTY:")
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for p in results_by_domain.values() for r in p
                           if r["difficulty"] == diff]
            if diff_results:
                d_correct = sum(1 for r in diff_results if r["is_correct"])
                d_acc = d_correct / len(diff_results)
                print(f"    {diff.upper():<8}: {d_acc:.1%} ({d_correct}/{len(diff_results)})")

        return {
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_evaluated": len(problems),
            "by_domain": {
                d: {
                    "accuracy": sum(1 for r in rlist if r["is_correct"]) / len(rlist),
                    "n": len(rlist),
                }
                for d, rlist in results_by_domain.items()
            }
        }

    # ── NLP SOLVE MODE ────────────────────────────────────────────────────

    def solve_from_text(self, text: str, show_detail: bool = True) -> None:
        """
        Solve soal langsung dari teks narasi (menggunakan StoryMathParser).
        Mode demo/interactive.
        """
        print(f"\n  🔢 Memproses soal...")
        print(f"  📝 Input: {text[:100]}...")

        sol = self.orchestrator.solve(text)
        print(sol.pretty_print())


# ── SINGLETON ──────────────────────────────────────────────────────────────────

_trainer_instance: Optional[AppliedMathTrainer] = None

def get_trainer(verbose: bool = False) -> AppliedMathTrainer:
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = AppliedMathTrainer(verbose=verbose)
    return _trainer_instance
