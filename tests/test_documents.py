import os
import shutil
import tempfile
import unittest

from llmez.documents import read_document


def make_pdf(pages):
    """PDF mínimo válido; cada página es una lista de líneas (vacía = página sin texto)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    kids = []
    for lines in pages:
        ops = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(
            "(" + l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") '"
            for l in lines) + " ET"
        stream = ops.encode("cp1252")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objs, 1):
        offsets.append(len(out))
        body = obj if isinstance(obj, bytes) else obj.encode()
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@unittest.skipUnless(shutil.which("pdftotext"), "hace falta pdftotext (poppler-utils)")
class PdfTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def write(self, name, data):
        path = os.path.join(self.dir.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_pages_are_marked_and_text_extracted(self):
        path = self.write("informe.pdf", make_pdf([
            ["Informe anual de la cooperativa El Ceibo", "Ventas totales: 1250 unidades"],
            ["Conclusion: la contrasena del deposito es pepino42"]]))
        text, info = read_document(path)
        self.assertEqual(info, "PDF de 2 páginas")
        self.assertIn("--- página 1 ---\nInforme anual", text)
        self.assertIn("--- página 2 ---\nConclusion", text)
        self.assertLess(text.index("1250"), text.index("--- página 2"))

    def test_hyphenated_words_are_joined(self):
        path = self.write("a.pdf", make_pdf([["La comunica-", "cion fue excelente en todo el proceso"]]))
        self.assertIn("comunicacion fue", read_document(path)[0])

    def test_scanned_pdf_is_rejected_with_clear_message(self):
        path = self.write("escaneado.pdf", make_pdf([[], []]))
        with self.assertRaises(ValueError) as cm:
            read_document(path)
        self.assertIn("OCR", str(cm.exception))

    def test_partially_scanned_pdf_reports_empty_pages(self):
        path = self.write("mixto.pdf", make_pdf([["Pagina con texto suficiente para contar"], [], []]))
        text, info = read_document(path)
        self.assertIn("sin texto en 2", info)
        self.assertIn("2, 3", info)

    def test_pdf_detected_by_content_without_extension(self):
        path = self.write("sin_extension", make_pdf([["Detectado por la cabecera del archivo"]]))
        self.assertTrue(read_document(path)[1].startswith("PDF"))

    def test_broken_pdf_gives_readable_error(self):
        path = self.write("roto.pdf", b"%PDF-1.4\nesto no es un pdf de verdad")
        with self.assertRaises(ValueError) as cm:
            read_document(path)
        self.assertTrue(str(cm.exception))


class TextTest(unittest.TestCase):
    def test_binary_file_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02")
        try:
            with self.assertRaises(ValueError) as cm:
                read_document(f.name)
            self.assertIn("PDF", str(cm.exception))  # sugiere los formatos aceptados
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
