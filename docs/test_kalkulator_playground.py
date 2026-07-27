import unittest

from kalkulator_playground import evaluate_expression


class KalkulatorPlaygroundTest(unittest.TestCase):
    def test_aritmetika_dasar(self) -> None:
        self.assertEqual(evaluate_expression("2+3*4"), 14)

    def test_ans_variable(self) -> None:
        self.assertEqual(evaluate_expression("ans*2", last_result=7), 14)

    def test_blokir_eksekusi_berbahaya(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_expression("__import__('os').system('echo hacked')")


if __name__ == "__main__":
    unittest.main()
