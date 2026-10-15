import unittest

from llmez.documents import _strip_repeated_lines
from llmez.retrieval import Index, compact_labels, split_chunks

DOC = "\n\n".join([
    "--- página 1 ---\nÍndice general del diseño curricular de la carrera.",
    "--- página 2 ---\nFirma el Rector, Ing. Rubén Soro, y el Secretario del Consejo Superior.",
    "--- página 3 ---\nTítulo intermedio: Analista Desarrollador Universitario de Sistemas.",
    "--- página 4 ---\nDuración de la carrera en horas reloj: 3992. Modalidad presencial.",
    "--- página 5 ---\nPlan de estudio. Bases de Datos: 4 horas semanales, 96 horas reloj anuales.",
])


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.idx = Index(DOC)

    def best_page(self, q):
        s = self.idx.scores(q)
        return self.idx.chunks[s.index(max(s))][0]

    def test_finds_the_right_page(self):
        self.assertEqual(self.best_page("¿Quién firma como rector?"), "página 2")
        self.assertEqual(self.best_page("¿Qué título intermedio otorga?"), "página 3")
        self.assertEqual(self.best_page("¿cuantas HORAS reloj dura la carrera?"), "página 4")  # sin tildes ni mayúsculas
        self.assertEqual(self.best_page("carga horaria de la asignatura bases de datos"), "página 5")

    def test_select_respects_budget_and_document_order(self):
        text, labels = self.idx.select("horas reloj bases de datos", budget_tokens=60)
        self.assertIn("página 5", labels)
        self.assertEqual(labels, sorted(labels, key=lambda l: int(l.split()[1])))
        self.assertLessEqual(len(text) / 3.9, 60 + 1)

    def test_irrelevant_pages_are_left_out(self):
        _, labels = self.idx.select("Rubén Soro rector", budget_tokens=10_000)
        self.assertEqual(labels, ["página 2"])

    def test_no_match_falls_back_to_start(self):
        _, labels = self.idx.select("zzzz qqqq", budget_tokens=40)
        self.assertEqual(labels[0], "página 1")

    def test_text_without_pages_is_split_in_fragments(self):
        chunks = split_chunks("\n\n".join(f"Párrafo {i} " + "x" * 900 for i in range(6)))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0][0].startswith("fragmento"))

    def test_never_sends_nothing_when_budget_is_tiny(self):
        text, labels = Index("--- página 1 ---\n" + "palabra " * 500).select("palabra", budget_tokens=50)
        self.assertEqual(labels, ["página 1"])
        self.assertTrue(0 < len(text) <= 50 * 2.8)

    def test_text_without_blank_lines_is_still_split(self):
        text = "Contraseña del wifi: pepino42.\n" + "".join(f"Línea {i}: dato irrelevante.\n" for i in range(3000))
        idx = Index(text)
        self.assertGreater(len(idx.chunks), 10)
        self.assertTrue(all(len(t) <= 2400 for _, t in idx.chunks))
        found, labels = idx.select("¿Cuál es la contraseña del wifi?", budget_tokens=1000)
        self.assertIn("pepino42", found)

    def test_huge_single_line_is_split(self):
        self.assertGreater(len(split_chunks("x" * 10_000)), 3)

    def test_compact_labels(self):
        self.assertEqual(compact_labels(["página 2", "página 3", "página 4", "página 9"]), "páginas 2–4, 9")
        self.assertEqual(compact_labels(["página 7"]), "página 7")
        self.assertEqual(compact_labels(["fragmento 1", "fragmento 3"]), "fragmentos 1, 3")
        self.assertEqual(compact_labels([]), "")
        self.assertEqual(compact_labels(["página 19", "página 19", "página 20"]), "páginas 19–20")

    def test_long_pages_are_split_keeping_the_page_mark(self):
        long_page = "\n\n".join(f"Párrafo {i}: " + "texto " * 60 for i in range(8))
        idx = Index(f"--- página 1 ---\nCorta.\n\n--- página 19 ---\n{long_page}")
        pieces = [t for label, t in idx.chunks if label == "página 19"]
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(t.startswith("--- página 19 ---") and len(t) < 1000 for t in pieces))
        self.assertEqual(idx.units, 2)

    def test_relevant_paragraph_of_a_long_page_fits_a_small_budget(self):
        filler = "\n\n".join("Relleno administrativo sin relación con la pregunta. " * 4 for _ in range(10))
        page19 = filler + "\n\nDuración de la carrera en años: 5 (cinco)\n\n" + filler
        doc = "--- página 1 ---\nÍndice.\n\n--- página 19 ---\n" + page19
        text, labels = Index(doc).select("duración de la carrera en años", budget_tokens=400)
        self.assertIn("5 (cinco)", text)
        self.assertEqual(labels[0], "página 19")


class HeaderStripTest(unittest.TestCase):
    def test_repeated_headers_and_page_numbers_are_removed(self):
        pages = [f"Ministerio de Educación\nPABLO A. HUEL\nContenido propio {i}\n{i}" for i in range(1, 6)]
        out, removed = _strip_repeated_lines(pages)
        self.assertEqual(removed, 2)
        self.assertEqual(out[2], "Contenido propio 3")

    def test_header_repeated_twice_in_one_line_is_removed(self):
        pages = [f"PABLO A. HUEL\nTexto {i}" for i in range(1, 5)]
        pages[0] = "PABLO A. HUEL PABLO A. HUEL\nFirma: Rubén Soro"
        out, _ = _strip_repeated_lines(pages)
        self.assertEqual(out[0], "Firma: Rubén Soro")

    def test_short_documents_are_untouched(self):
        pages = ["Encabezado\nA", "Encabezado\nB"]
        self.assertEqual(_strip_repeated_lines(pages), (pages, 0))


if __name__ == "__main__":
    unittest.main()
