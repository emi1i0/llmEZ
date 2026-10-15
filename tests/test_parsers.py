import unittest

from llmez import markdown
from llmez.think_parser import ThinkTagSplitter


def split(chunks):
    s = ThinkTagSplitter()
    out = []
    for c in chunks:
        out += s.feed(c)
    out += s.flush()
    merged = []
    for kind, text in out:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


class ThinkSplitterTest(unittest.TestCase):
    def test_plain_content(self):
        self.assertEqual(split(["Hola ", "mundo"]), [("content", "Hola mundo")])

    def test_tags_split_across_chunks(self):
        self.assertEqual(split(["<thi", "nk>razono</thi", "nk>\n\nRespuesta"]),
                         [("thinking", "razono"), ("content", "Respuesta")])

    def test_one_char_per_chunk(self):
        self.assertEqual(split(list("<think>a</think>b")),
                         [("thinking", "a"), ("content", "b")])

    def test_orphan_close_like_qwen3_4b(self):
        # qwen3:4b con think:false: razonamiento sin apertura y </think> suelto.
        self.assertEqual(split(["Okay, the user", " asks.\n</th", "ink>\n\n5"]),
                         [("content", "Okay, the user asks.\n"), ("orphan_close", ""),
                          ("content", "5")])

    def test_html_like_text_is_not_swallowed(self):
        self.assertEqual(split(["usa <b>negrita</b> y a<b"]),
                         [("content", "usa <b>negrita</b> y a<b")])


def plain(runs):
    return "".join(t for t, _ in runs)


def tags_of(runs, text):
    return next(tags for t, tags in runs if t == text)


class MarkdownTest(unittest.TestCase):
    def test_inline_styles(self):
        runs = markdown.parse("a **b** *c* `d` ***e***")
        self.assertEqual(plain(runs), "a b c d e")
        self.assertIn("md_bold", tags_of(runs, "b"))
        self.assertIn("md_italic", tags_of(runs, "c"))
        self.assertIn("md_code", tags_of(runs, "d"))
        self.assertIn("md_bold_italic", tags_of(runs, "e"))

    def test_snake_case_and_math_untouched(self):
        self.assertEqual(plain(markdown.parse("mi_variable_larga y 2 * 3 * 4")),
                         "mi_variable_larga y 2 * 3 * 4")

    def test_heading_list_code_block(self):
        runs = markdown.parse("# Título\n- uno\n2. dos\n```py\nx = 1\n```")
        self.assertIn("md_h1", tags_of(runs, "Título"))
        self.assertEqual(plain(runs), "Título\n• uno\n2. dos\nx = 1")
        self.assertIn("md_codeblock", tags_of(runs, "x = 1"))  # el último \n se recorta

    def test_code_block_ignores_inline_markup(self):
        runs = markdown.parse("```\na **no** b\n```")
        self.assertEqual(runs, [("a **no** b", ("md_codeblock",))])

    def test_unterminated_code_block(self):
        self.assertEqual(plain(markdown.parse("```\ncortado")), "cortado")

    def test_table_is_monospaced(self):
        runs = markdown.parse("| a | b |\n|---|---|")
        self.assertTrue(all("md_table" in tags for t, tags in runs if t.strip()))


if __name__ == "__main__":
    unittest.main()
