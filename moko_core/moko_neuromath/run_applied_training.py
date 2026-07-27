"""
MOKO Applied Math Training Runner — CLI
========================================
Script CLI utama untuk menjalankan sistem pelatihan AI
menjawab pertanyaan matematika terapan nyata.

Penggunaan:
  python run_applied_training.py --auto          # Sesi otomatis (20 soal)
  python run_applied_training.py --auto --n 50   # Batch 50 soal
  python run_applied_training.py --interactive    # Mode interaktif
  python run_applied_training.py --eval           # Evaluasi saja (dry run)
  python run_applied_training.py --solve "teks soal..."  # Solve satu soal
  python run_applied_training.py --report         # Tampilkan laporan progress
  python run_applied_training.py --demo           # Demo cepat semua domain
"""

import sys
import os
import argparse
import time

# Pastikan path benar
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CORE)

for p in [_CORE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         MOKO APPLIED MATH AI TRAINING SYSTEM                ║
║         Melatih AI Menjawab Matematika Terapan Nyata        ║
╠══════════════════════════════════════════════════════════════╣
║  Domain: Fluida · Termal · Elektronik · Mekanika · Optik    ║
║          Akustik · Kimia · Struktur · Keuangan · Energi     ║
╚══════════════════════════════════════════════════════════════╝
""")


def cmd_auto(args):
    """Mode otomatis: batch training."""
    from moko_neuromath.applied_math_trainer import get_trainer
    trainer = get_trainer(verbose=args.verbose)
    result = trainer.run_auto(n_problems=args.n, save_interval=args.save_interval)
    print(f"\n  Sesi selesai. Akurasi: {result['accuracy']:.1%}")


def cmd_interactive(args):
    """Mode interaktif."""
    from moko_neuromath.applied_math_trainer import get_trainer
    trainer = get_trainer(verbose=args.verbose)
    trainer.run_interactive(n_problems=args.n)


def cmd_eval(args):
    """Mode evaluasi."""
    from moko_neuromath.applied_math_trainer import get_trainer
    trainer = get_trainer(verbose=args.verbose)
    trainer.run_evaluation(n_problems=args.n)


def cmd_solve(args):
    """Solve satu soal dari teks."""
    if not args.solve:
        print("  ❌ Berikan teks soal dengan --solve 'teks soal...'")
        return
    from moko_neuromath.applied_math_trainer import get_trainer
    trainer = get_trainer(verbose=args.verbose)
    trainer.solve_from_text(args.solve)


def cmd_report(args):
    """Tampilkan laporan progress."""
    from moko_neuromath.training_state import get_state_manager
    state = get_state_manager()
    print(state.generate_report())
    print(f"\n  File state: {os.path.join(os.path.dirname(_ROOT), '.math_omni', 'applied_training_state.json')}")


def cmd_dataset_info(args):
    """Tampilkan info dataset."""
    from moko_neuromath.applied_math_dataset import get_dataset
    ds = get_dataset()
    stats = ds.stats()
    print(f"\n  📚 MOKO Applied Math Dataset")
    print(f"  Total soal    : {stats['total']}")
    print(f"\n  Per Domain:")
    for domain, count in sorted(stats['by_domain'].items()):
        print(f"    {domain:<25}: {count} soal")
    print(f"\n  Per Difficulty:")
    for diff, count in sorted(stats['by_difficulty'].items()):
        print(f"    {diff:<10}: {count} soal")

    # Tampilkan sample soal
    print(f"\n  Contoh Soal:")
    for p in ds.problems[:5]:
        print(f"    [{p.id}] [{p.difficulty.value.upper()}] {p.story_text[:70]}...")
        print(f"           → {p.target_symbol} = {p.expected_answer:.4g} {p.expected_unit}")


def cmd_demo(args):
    """Demo cepat: solve satu soal dari setiap domain."""
    from moko_neuromath.applied_math_dataset import get_dataset
    from moko_neuromath.arms_orchestrator import get_orchestrator, AnswerEvaluator

    ds = get_dataset()
    orch = get_orchestrator(verbose=False)
    evaluator = AnswerEvaluator()

    # Ambil satu soal EASY per domain
    domains_seen = set()
    demo_problems = []
    for p in ds.problems:
        if p.domain not in domains_seen and p.difficulty.value == "easy":
            demo_problems.append(p)
            domains_seen.add(p.domain)

    print(f"\n  🎬 DEMO: Menyelesaikan soal dari {len(demo_problems)} domain berbeda\n")

    n_correct = 0
    t_total = time.time()

    for p in demo_problems:
        t0 = time.time()
        sol = orch.solve_structured(
            domain=p.domain,
            known=p.known_vars,
            target=p.target_symbol,
            story_text=p.story_text,
        )
        elapsed = (time.time() - t0) * 1000

        is_correct, pct_err, verdict = evaluator.evaluate(sol, p.expected_answer, p.tolerance_pct)
        if is_correct:
            n_correct += 1

        icon = "✅" if is_correct else "❌"
        ans_str = f"{sol.result_value:.4g}" if sol.result_value is not None else "FAIL"

        exp_str = f"{p.expected_answer:.4g}"
        print(f"  {icon} [{p.domain:<22}] {p.target_symbol:<8} "
              f"| ARMS={ans_str:<12} | Benar={exp_str:<12} "
              f"| {verdict[:35]}")
        if args.verbose and sol.solution_steps:
            for step in sol.solution_steps[:3]:
                print(f"       {step}")

    elapsed_total = time.time() - t_total
    print(f"\n  📊 Demo selesai: {n_correct}/{len(demo_problems)} benar "
          f"({n_correct/len(demo_problems):.1%}) dalam {elapsed_total:.2f}s")


def cmd_domain_test(args):
    """Test mendalam satu domain tertentu."""
    if not args.domain:
        print("  ❌ Tentukan domain dengan --domain [nama_domain]")
        print("  Domain yang tersedia: fluid_mechanics, acoustics, thermodynamics,")
        print("    electronics, kinematics, engine_mechanics, energy, finance,")
        print("    structural, optics, chemistry, materials, signal_processing")
        return

    from moko_neuromath.applied_math_dataset import get_dataset
    from moko_neuromath.arms_orchestrator import get_orchestrator, AnswerEvaluator

    ds = get_dataset()
    orch = get_orchestrator(verbose=args.verbose)
    evaluator = AnswerEvaluator()

    domain_problems = ds.get_by_domain(args.domain)
    if not domain_problems:
        print(f"  ❌ Domain '{args.domain}' tidak ditemukan dalam dataset.")
        return

    print(f"\n  🔬 Testing domain: {args.domain} ({len(domain_problems)} soal)\n")

    n_correct = 0
    for p in domain_problems:
        sol = orch.solve_structured(
            domain=p.domain,
            known=p.known_vars,
            target=p.target_symbol,
        )
        is_correct, pct_err, verdict = evaluator.evaluate(sol, p.expected_answer, p.tolerance_pct)
        if is_correct:
            n_correct += 1

        icon = "✅" if is_correct else "❌"
        ans_str = f"{sol.result_value:.6g}" if sol.result_value is not None else "FAIL"
        print(f"  {icon} [{p.id}] [{p.difficulty.value.upper():<6}] "
              f"Target: {p.target_symbol:<8} ARMS={ans_str:<14} Benar={p.expected_answer:.6g}")

        if not is_correct and args.verbose:
            print(f"       Soal: {p.story_text[:80]}...")
            print(f"       Steps: {sol.solution_steps[:2]}")
            if sol.warnings:
                print(f"       Warnings: {sol.warnings[:2]}")

    acc = n_correct / len(domain_problems)
    print(f"\n  Hasil: {n_correct}/{len(domain_problems)} benar ({acc:.1%})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MOKO Applied Math AI Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python run_applied_training.py --demo
  python run_applied_training.py --auto --n 30
  python run_applied_training.py --interactive --n 5
  python run_applied_training.py --eval --n 50
  python run_applied_training.py --solve "Piston diameter 80mm tekanan 12 bar, berapa gaya?"
  python run_applied_training.py --report
  python run_applied_training.py --domain-test --domain fluid_mechanics
  python run_applied_training.py --dataset-info
        """
    )

    # Mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--auto", action="store_true",
                            help="Sesi training otomatis (batch)")
    mode_group.add_argument("--interactive", action="store_true",
                            help="Sesi training interaktif")
    mode_group.add_argument("--eval", action="store_true",
                            help="Evaluasi saja (dry run, tidak mengubah state)")
    mode_group.add_argument("--demo", action="store_true",
                            help="Demo cepat semua domain")
    mode_group.add_argument("--report", action="store_true",
                            help="Tampilkan laporan progress")
    mode_group.add_argument("--dataset-info", action="store_true",
                            help="Tampilkan info dataset")
    mode_group.add_argument("--domain-test", action="store_true",
                            help="Test mendalam satu domain")
    mode_group.add_argument("--solve", type=str, default=None,
                            help="Solve satu soal dari teks")

    # Options
    parser.add_argument("--n", type=int, default=20,
                        help="Jumlah soal (default: 20)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain untuk --domain-test")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Simpan state setiap N soal (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    print_banner()

    # Dispatch
    if args.auto:
        cmd_auto(args)
    elif args.interactive:
        cmd_interactive(args)
    elif args.eval:
        cmd_eval(args)
    elif args.demo:
        cmd_demo(args)
    elif args.report:
        cmd_report(args)
    elif args.dataset_info:
        cmd_dataset_info(args)
    elif args.domain_test:
        cmd_domain_test(args)
    elif args.solve:
        cmd_solve(args)
    else:
        # Default: demo
        print("  Tidak ada mode dipilih. Menjalankan --demo ...\n")
        cmd_demo(args)
        print("\n  Gunakan --help untuk melihat semua opsi.")


if __name__ == "__main__":
    main()
