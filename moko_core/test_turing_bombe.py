import sys
import os
import hashlib
sys.path.insert(0, os.path.dirname(__file__))

from moko_neuromath.turing_bombe_solver import DigitalRotor, DiagonalBoard, TuringBombeSolver
from moko_agents.cognitive_executive import get_executive


def test_digital_rotor():
    print("🧪 Testing DigitalRotor...")
    # Rotor I standard
    rotor = DigitalRotor("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Rotor I")
    
    # Test simple forward mapping (offset 0)
    # A (0) -> E (4)
    assert rotor.forward(0, offset=0) == 4, f"A forward expected 4, got {rotor.forward(0)}"
    # E (4) -> A (0) backward
    assert rotor.backward(4, offset=0) == 0
    
    # Test mapping with offset 1
    # Input A (0) + offset 1 = B (1). wiring[1] = K (10). output = 10 - offset 1 = J (9).
    assert rotor.forward(0, offset=1) == 9
    assert rotor.backward(9, offset=1) == 0
    
    print("  ✅ DigitalRotor mapping OK")


def test_diagonal_board():
    print("🧪 Testing DiagonalBoard...")
    board = DiagonalBoard(26)
    
    # Assert initially empty
    assert not board.grid[0][1]
    
    # Apply connection A <-> B (0 <-> 1)
    flows = board.apply_voltage(0, 1)
    assert (0, 1) in flows
    assert (1, 0) in flows
    
    # Check board state
    assert board.grid[0][1]
    assert board.grid[1][0]
    
    # Re-apply should not trigger new flows
    flows2 = board.apply_voltage(0, 1)
    assert len(flows2) == 0
    
    print("  ✅ DiagonalBoard reciprocity OK")


def test_bombe_crib_solving():
    print("🧪 Testing TuringBombeSolver solve_crib...")
    solver = TuringBombeSolver()
    
    # Let's generate a simple short ciphertext with known rotor settings
    # We will use rotors I, II, III. Let's see if we can find a matching rotor state.
    # Plaintext:  WETTER
    # Ciphertext: KQXLMZ (Simulated)
    # Let's run a test crib search
    res = solver.solve_crib("WETTER", "KQXLMZ")
    print(f"  Crib search success: {res['success']}")
    print(f"  Start injection character: {res['start_injection_char']}")
    print(f"  Valid position candidates: {res['candidates']}")
    
    # Even if no candidates match (since we randomly typed ciphertext),
    # the process must complete successfully without crashing
    assert "success" in res
    print("  ✅ TuringBombeSolver solve_crib OK")


def test_bombe_logic_solving():
    print("🧪 Testing TuringBombeSolver solve_logic_constraint...")
    solver = TuringBombeSolver()
    
    # Logic puzzle:
    # 3 Houses: Merah, Hijau, Biru.
    # Occupants: 1, 2, 3.
    # Constraints:
    #   - Merah is house 1.
    #   - Hijau is NOT the same house as Merah.
    #   - Biru is NOT the same house as Hijau.
    #   - Biru is NOT the same house as Merah.
    variables = ["Merah", "Hijau", "Biru"]
    domains = {
        "Merah": [1, 2, 3],
        "Hijau": [1, 2, 3],
        "Biru": [1, 2, 3]
    }
    constraints = [
        {"type": "equal", "var1": "Merah", "value": 1},
        {"type": "different", "var1": "Merah", "var2": "Hijau"},
        {"type": "different", "var1": "Hijau", "var2": "Biru"},
        {"type": "different", "var1": "Merah", "var2": "Biru"},
    ]
    
    res = solver.solve_logic_constraint(variables, domains, constraints)
    print(f"  Logic solve success: {res['success']}")
    print(f"  Logic solved solution: {res['solution']}")
    print(f"  Logic steps: {res['steps']}")
    
    assert res["success"]
    assert res["solution"]["Merah"] == 1
    assert set(res["solution"]["Hijau"]) == {2, 3}
    assert set(res["solution"]["Biru"]) == {2, 3}
    print("  ✅ TuringBombeSolver solve_logic_constraint OK")


def test_cognitive_executive_bypass():
    print("🧪 Testing CognitiveExecutive L0-TCS Bypass integration...")
    exec_system = get_executive(verbose=False)
    import time
    ts = int(time.time())
    
    # 1. Test Enigma Crib Query Routing
    query_bombe = f"pecahkan enigma plaintext=WETTER ciphertext=KQXLMZ ts={ts}"
    res_bombe = exec_system.process(query_bombe)
    print(f"  Enigma query processed: query_id={res_bombe['query_id']}")
    print(f"  Complexity: {res_bombe['complexity']}")
    print(f"  Primary Subsystem: {res_bombe['primary_subsystem']}")
    
    assert res_bombe["primary_subsystem"] == "L0_TCS_Bypass"
    assert "Turing-Bombe" in res_bombe["answer"]
    print("  ✅ CognitiveExecutive Enigma Routing OK")
    
    # 2. Test Logic Constraint Query Routing
    query_logic = f"kendala logic variables=[A, B, C] constraints=[equal(A, 1), different(A, B), different(B, C), different(A, C)] ts={ts}"
    res_logic = exec_system.process(query_logic)
    print(f"  Logic query processed: query_id={res_logic['query_id']}")
    print(f"  Complexity: {res_logic['complexity']}")
    print(f"  Primary Subsystem: {res_logic['primary_subsystem']}")
    
    assert res_logic["primary_subsystem"] == "L0_TCS_Bypass"
    assert "A" in res_logic["answer"] and "B" in res_logic["answer"]
    print("  ✅ CognitiveExecutive Logic Routing OK")


if __name__ == "__main__":
    print("=== Running MOKO Turing-Bombe TCS Test Suite ===")
    test_digital_rotor()
    test_diagonal_board()
    test_bombe_crib_solving()
    test_bombe_logic_solving()
    test_cognitive_executive_bypass()
    print("\n🎉 ALL TURING-BOMBE TCS TESTS PASSED!")
