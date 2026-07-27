import ast
import math
import operator


ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "ln": math.log,
}


def evaluate_expression(expression: str, last_result: float | None = None) -> float:
    """Evaluasi ekspresi matematika dengan AST aman (tanpa eval)."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Input kosong.")

    variables = {
        "ans": 0.0 if last_result is None else last_result,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        parsed = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Ekspresi tidak valid.") from exc

    return _evaluate_ast(parsed.body, variables)


def _evaluate_ast(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Hanya angka yang diizinkan.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINARY_OPERATORS:
            raise ValueError("Operator tidak diizinkan.")
        left = _evaluate_ast(node.left, variables)
        right = _evaluate_ast(node.right, variables)
        return float(ALLOWED_BINARY_OPERATORS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPERATORS:
            raise ValueError("Operator unary tidak diizinkan.")
        value = _evaluate_ast(node.operand, variables)
        return float(ALLOWED_UNARY_OPERATORS[op_type](value))

    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in variables:
            return float(variables[key])
        raise ValueError(f"Variabel '{node.id}' tidak dikenali.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Bentuk fungsi tidak valid.")
        func_name = node.func.id.lower()
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Fungsi '{node.func.id}' tidak diizinkan.")
        args = [_evaluate_ast(arg, variables) for arg in node.args]
        return float(ALLOWED_FUNCTIONS[func_name](*args))

    raise ValueError("Ekspresi mengandung sintaks yang tidak diizinkan.")


def _print_history(history: list[tuple[str, float]]) -> None:
    if not history:
        print("    (riwayat masih kosong)")
        return

    print("    --- Riwayat ---")
    for index, (expr, result) in enumerate(history[-10:], start=max(1, len(history) - 9)):
        print(f"    {index}. {expr} = {result:g}")


def hitung_luas_persegi_panjang(panjang: float, lebar: float) -> float:
    """Hitung luas persegi panjang."""
    return panjang * lebar

def hitung_luas_persegi(sisi: float) -> float:
    """Hitung luas persegi (kotak)."""
    return sisi * sisi

def kalkulator_geometri() -> None:
    """Mode kalkulator geometri untuk MOKO."""
    print("=" * 52)
    print("  📐 MOKO Kalkulator Geometri")
    print("=" * 52)
    print("Pilih mode:")
    print("  1. Luas Persegi Panjang (panjang × lebar)")
    print("  2. Luas Persegi/Kotak (sisi × sisi)")
    print("  3. Kembali ke kalkulator utama")
    print("=" * 52)

    while True:
        pilihan = input("\nPilih mode (1/2/3): ").strip()

        if pilihan == "3":
            print("Kembali ke kalkulator utama...")
            break

        if pilihan == "1":
            print("\n--- Luas Persegi Panjang ---")
            try:
                panjang = float(input("Panjang: "))
                lebar = float(input("Lebar: "))
                luas = hitung_luas_persegi_panjang(panjang, lebar)
                print(f"  Luas Persegi Panjang = {panjang} × {lebar} = {luas:g}")
            except ValueError:
                print("  ⚠ Input harus berupa angka!")
            except Exception as e:
                print(f"  ⚠ Error: {e}")

        elif pilihan == "2":
            print("\n--- Luas Persegi/Kotak ---")
            try:
                sisi = float(input("Sisi: "))
                luas = hitung_luas_persegi(sisi)
                print(f"  Luas Persegi = {sisi} × {sisi} = {luas:g}")
            except ValueError:
                print("  ⚠ Input harus berupa angka!")
            except Exception as e:
                print(f"  ⚠ Error: {e}")

        else:
            print("  ⚠ Pilihan tidak valid! Masukkan 1, 2, atau 3.")


def kalkulator() -> None:
    """Kalkulator MOKO versi aman dan lebih interaktif."""
    print("=" * 52)
    print("  🔢 MOKO Kalkulator Playground — Keren & Aman")
    print("=" * 52)
    print("Operasi: +  -  *  /  //  **  %")
    print("Fungsi: sqrt(x), sin(x), cos(x), tan(x), log(x), abs(x), round(x,n)")
    print("Konstanta: pi, e | Variabel: ans")
    print("Mode Khusus: geom (kalkulator geometri)")
    print("Perintah: help, riwayat, clear, q\n")

    history: list[tuple[str, float]] = []
    last_result: float | None = None

    while True:
        expr = input(">>> ").strip()
        lower = expr.lower()

        if not expr or lower in ("q", "quit", "exit", "keluar"):
            print("Sampai jumpa!")
            break

        if lower in ("help", "bantuan"):
            print("    Contoh: 2+3*4, (10-3)**2, sqrt(81), ans/2")
            print("    Ketik 'geom' untuk mode geometri (luas persegi panjang/kotak)")
            continue

        if lower in ("riwayat", "history"):
            _print_history(history)
            continue

        if lower in ("clear", "c"):
            history.clear()
            print("    Riwayat dibersihkan.")
            continue

        if lower in ("geom", "geometri"):
            kalkulator_geometri()
            print("\nKembali ke mode kalkulator biasa...")
            continue

        try:
            result = evaluate_expression(expr, last_result=last_result)
            last_result = result
            history.append((expr, result))
            print(f"    = {result:g}")
        except ZeroDivisionError:
            print("    ⚠ Tidak bisa dibagi nol!")
        except ValueError as error:
            print(f"    ⚠ {error}")
        except Exception as error:
            print(f"    ⚠ Error tak terduga: {error}")


if __name__ == "__main__":
    kalkulator()
