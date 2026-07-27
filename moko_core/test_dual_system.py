"""
Integration Test — MOKO Dual-System (Sistem 1 Kimi + Sistem 2 DeepSeek)
=======================================================================
Menguji kalang agentik end-to-end tanpa PyQt6/torch:
- Alur sukses (Plan → Execute → Guard → Commit-ready) satu iterasi.
- Alur koreksi-diri (bug pertama → repair → sukses iterasi kedua).
- Emisi state kognitif berurutan (Brain → Executor → Guard).
- Anchor-RAG retrieval (Sistem 1) mengambil snippet relevan.
- Diagnosis traceback oleh Sistem 2 Guard.

Jalankan langsung:  python test_dual_system.py   (dari folder moko_core)
"""
import sys
from pathlib import Path

# Pastikan paket moko_agents dapat diimpor apa pun cwd-nya.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moko_agents.dual_system import (  # noqa: E402
    DualSystemOrchestrator,
    ExecutorNode,
    DualRuntimeGuard,
    ExecutionResult,
    VERDICT_COMMIT,
    VERDICT_REPAIR,
    STATE_BRAIN,
    STATE_EXECUTOR,
    STATE_GUARD,
)
from moko_agents.dual_system._bridge import (  # noqa: E402
    CODE_KNOWLEDGE_AVAILABLE,
    RUNTIME_GUARD_AVAILABLE,
)


def test_success_path():
    """Loop harus sukses dalam 1 iterasi dan mengeluarkan state yang benar."""
    states = []
    orch = DualSystemOrchestrator(on_state=states.append, max_iterations=3)
    try:
        result = orch.run_loop("buatkan fungsi kode untuk menghitung jumlah kuadrat")
    finally:
        orch.cleanup()

    assert result.success is True, "Loop seharusnya sukses"
    assert result.iterations == 1, f"Diharapkan 1 iterasi, dapat {result.iterations}"
    assert result.traces[-1].guard_verdict == VERDICT_COMMIT
    assert "commit" in result.commit_ref.lower()
    # State kognitif muncul berurutan Brain → Executor → Guard.
    assert states[0] == STATE_BRAIN
    assert states[1] == STATE_EXECUTOR
    assert states[2] == STATE_GUARD
    assert {STATE_BRAIN, STATE_EXECUTOR, STATE_GUARD}.issubset(set(states))


def test_self_correction_path():
    """Bug pada iterasi pertama harus terdeteksi Guard lalu diperbaiki."""
    orch = DualSystemOrchestrator(max_iterations=3, force_bug_first=True)
    try:
        result = orch.run_loop("perbaiki bug fungsi jumlah kuadrat")
    finally:
        orch.cleanup()

    assert result.success is True, "Loop seharusnya sukses setelah koreksi-diri"
    assert result.iterations == 2, f"Diharapkan 2 iterasi, dapat {result.iterations}"
    assert result.traces[0].execution_success is False
    assert result.traces[0].guard_verdict == VERDICT_REPAIR
    assert result.traces[1].execution_success is True
    assert result.traces[1].guard_verdict == VERDICT_COMMIT
    # Instruksi perbaikan harus terisi pada iterasi kedua.
    assert result.traces[1].repair_hint != ""


def test_progress_signal_emission():
    """Callback progress harus terpanggil & mencapai 100%."""
    progress = []
    orch = DualSystemOrchestrator(
        on_progress=lambda pct, label: progress.append((pct, label)),
        max_iterations=3,
    )
    try:
        orch.run_loop("buat fungsi kode jumlah kuadrat")
    finally:
        orch.cleanup()
    assert progress, "progress callback tidak pernah terpanggil"
    assert progress[-1][0] == 100, "progres akhir harus 100%"


def test_anchor_rag_retrieval():
    """Sistem 1 harus menautkan snippet trigonometri untuk prompt terkait."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ex = ExecutorNode(tmp)
        hits = ex.retrieve_context("hitung sin cos tan sudut derajat trigonometri")
        assert hits, "Anchor-RAG tidak mengembalikan snippet apa pun"
        ids = [getattr(h, "snippet_id", "") for h in hits]
        assert any("trigonometri" in i for i in ids), f"snippet tak sesuai: {ids}"


def test_guard_traceback_parsing():
    """Guard harus memparsing tipe galat & meminta perbaikan saat gagal."""
    guard = DualRuntimeGuard()
    fake = ExecutionResult(
        success=False,
        return_code=1,
        stdout="",
        stderr=(
            'Traceback (most recent call last):\n'
            '  File "test_moko_generated.py", line 5, in <module>\n'
            "    test_moko_sum_of_squares()\n"
            "AssertionError: diharapkan 14, diperoleh 6\n"
        ),
        log="return_code=1",
    )
    report = guard.review(fake)
    assert report.verdict == VERDICT_REPAIR
    assert report.passed is False
    assert report.error_type == "AssertionError"
    assert "test_moko_generated.py:5" == report.failing_line


def test_written_files_executed():
    """Executor menulis modul + unit test dan menjalankannya nyata."""
    import tempfile
    from moko_agents.dual_system import BrainNode
    with tempfile.TemporaryDirectory() as tmp:
        brain = BrainNode()
        plan = brain.reason_and_plan("buat fungsi kode jumlah kuadrat")
        ex = ExecutorNode(tmp)
        res = ex.apply_plan(plan)
        assert res.success is True
        assert len(res.written_files) == 2
        assert (Path(tmp) / "moko_generated_runtime.py").exists()
        assert (Path(tmp) / "test_moko_generated.py").exists()
        assert plan.expected_signal in res.stdout


ALL_TESTS = [
    test_success_path,
    test_self_correction_path,
    test_progress_signal_emission,
    test_anchor_rag_retrieval,
    test_guard_traceback_parsing,
    test_written_files_executed,
]


def main() -> int:
    print("=== MOKO DUAL-SYSTEM INTEGRATION TESTS ===")
    print(f"CODE_KNOWLEDGE_AVAILABLE={CODE_KNOWLEDGE_AVAILABLE} | "
          f"RUNTIME_GUARD_AVAILABLE={RUNTIME_GUARD_AVAILABLE}")
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"  [ERROR] {test.__name__}: {exc}")
            traceback.print_exc()
    total = len(ALL_TESTS)
    print(f"=== HASIL: {total - failures}/{total} lolos ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
