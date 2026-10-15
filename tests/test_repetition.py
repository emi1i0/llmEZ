import unittest

from llmez.repetition import RepetitionDetector, periodic_tail

# Salida real de smollm2:135m con temperatura 0 ("Escribí un poema sobre el mar").
SMOLLM_LOOP = ("Aquí va un poema.\n\n" + (
    "En este artículo, se pueden explicar un poema que se puede explicar en la forma "
    "de la traducción, que es un poema que se puede explicar en la forma de la traducción.\n\n"
) * 6 + "En este artículo, se pueden explicar un poema que se pued")

# Patrón de qwen3:0.6b: párrafo repetido con secciones nuevas entre medio.
PARA = ("El primer día que Nexa aprendió a cocinar fue en un lugar donde la cocina era "
        "un espacio de vida, y los científicos trabajaban sin descanso.")
QWEN_LOOP = f"{PARA}\n\n---\n\n**Capítulo 2**\n\n{PARA}\n\nNexa no era humano.\n\n{PARA}\n"


def feed_all(text, step=7):
    det = RepetitionDetector()
    for i in range(0, len(text), step):
        if det.feed(text[i:i + step]):
            return det
    return None


class RepetitionTest(unittest.TestCase):
    def test_paragraph_loop_smollm(self):
        det = feed_all(SMOLLM_LOOP)
        self.assertIsNotNone(det)
        # Se conserva una sola copia del párrafo repetido.
        self.assertEqual(det.trimmed().count("En este artículo"), 1)
        self.assertTrue(det.trimmed().startswith("Aquí va un poema."))

    def test_repeated_paragraph_with_sections_qwen(self):
        det = feed_all(QWEN_LOOP)
        self.assertIsNotNone(det)
        self.assertIn("apareció 3 veces", det.reason)
        self.assertEqual(det.trimmed().count("El primer día"), 1)

    def test_trim_keeps_first_whole_copy_of_sentence(self):
        # Caso real de smollm2: el bucle no está alineado con el final del texto.
        sentence = "En este artículo, se puede explicar un poema que se escribe en español. "
        text = "Intro distinta. " + sentence * 7 + sentence[:23]
        det = feed_all(text)
        self.assertIsNotNone(det)
        self.assertEqual(det.trimmed(), "Intro distinta. " + sentence.rstrip())

    def test_short_word_loop(self):
        det = feed_all("Respuesta: " + "no " * 200)
        self.assertIsNotNone(det)
        self.assertIn("se repitió", det.reason)

    def test_normal_text_is_not_a_loop(self):
        normal = "\n".join(f"{i}. El animal número {i} es distinto de los anteriores y vive "
                           f"en un hábitat propio del continente {i % 5}." for i in range(1, 40))
        self.assertIsNone(feed_all(normal))

    def test_code_with_repeated_short_lines_is_not_a_loop(self):
        code = "```python\n" + "for x in range(3):\n    print(x)\n" * 8 + "```\nListo."
        self.assertIsNone(feed_all(code))

    def test_short_laugh_is_not_a_loop(self):
        self.assertIsNone(feed_all("Jaja " * 5 + "qué bueno."))

    def test_periodic_tail_finds_period(self):
        self.assertEqual(periodic_tail("x" * 10 + "abcdefghij" * 50)[0], 10)
        self.assertIsNone(periodic_tail("texto común y corriente " * 2))


if __name__ == "__main__":
    unittest.main()
