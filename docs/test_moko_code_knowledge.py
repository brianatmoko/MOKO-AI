import unittest

from moko_code_knowledge import CodeKnowledgeBase


class CodeKnowledgeBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = CodeKnowledgeBase()

    def test_retrieve_returns_matching_domain_only(self) -> None:
        results = self.kb.retrieve({"rumus", "luas", "persegi"})
        domains = {snippet.domain for snippet in results}

        self.assertIn("geometri", domains)
        self.assertNotIn("finansial", domains)

    def test_retrieve_avoids_retrieve_everything_on_no_anchor(self) -> None:
        results = self.kb.retrieve({"siswa", "sekolah", "warna"})

        self.assertEqual(results, [])

    def test_retrieve_ranks_by_anchor_overlap(self) -> None:
        results = self.kb.retrieve(
            {"statistika", "rata", "median", "deviasi", "bunga"},
            limit=2,
        )

        self.assertTrue(results)
        self.assertEqual(results[0].domain, "statistika")

    def test_retrieve_respects_limit(self) -> None:
        results = self.kb.retrieve(
            {"rumus", "sin", "statistika", "bunga", "konversi", "algoritma"},
            limit=2,
        )

        self.assertLessEqual(len(results), 2)

    def test_all_snippets_expose_anchors_and_source(self) -> None:
        for snippet in self.kb.snippets:
            self.assertTrue(snippet.anchors)
            self.assertTrue(snippet.source)
            self.assertIn("def ", snippet.code)


if __name__ == "__main__":
    unittest.main()
