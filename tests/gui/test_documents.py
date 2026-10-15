"""Documentos: PDF, PDF escaneado, archivo largo, búsqueda de páginas y por significado.

Usa la Ordenanza 1877 si está (variable LLMEZ_TEST_PDF para otra ruta); si no, esas
pruebas se saltean.
"""
import os
import re
import sys
import time

from common import ORDENANZA, Harness, plain

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_documents import make_pdf  # noqa: E402  (PDF mínimo generado sin dependencias)

h = Harness("llmEZ prueba de documentos")
app, chat = h.app, h.chat


def meta():
    return h.tagged("meta")


def pages_sent():
    """Páginas que se mandaron en la última pregunta (cada pregunta empieza charla nueva)."""
    found = re.findall(r"van las (.*?) \(", meta())
    return found[-1] if found else "archivo entero"


def includes(pages, n):
    return str(n) in re.findall(r"\d+", pages) or any(int(a) <= n <= int(b) for a, b in re.findall(r"(\d+)–(\d+)", pages))


def generated_pdf():
    path = f"{h.out}/informe.pdf"
    with open(path, "wb") as f:
        f.write(make_pdf([["Informe anual de la cooperativa El Ceibo", "Produccion total: 48300 kilogramos"],
                          ["Socios activos: 37"], ["La asamblea sera el 14 de abril, la presenta Marta Gutierrez"]]))
    h.setup("qwen3:1.7b", ctx="4096", temp="0", seed="1")
    h.load_file(path, lambda: (
        h.check("PDF: se carga con las páginas marcadas",
                "--- página 3 ---" in (chat.file_text or "") and "PDF de 3 páginas" in meta()),
        h.send("¿Cuándo es la asamblea y quién la presenta? Indicá la página.", after_generated)))


def after_generated():
    a = plain(h.last_answer())
    h.check("PDF: el modelo encuentra el dato", "14 de abril" in a and "marta" in a, repr(h.last_answer()[:80]))
    h.next()


def scanned_pdf():
    from tkinter import messagebox
    errors = []
    messagebox.showerror = lambda t, m, **k: errors.append(m)
    path = f"{h.out}/escaneado.pdf"
    with open(path, "wb") as f:
        f.write(make_pdf([[], []]))
    before = chat.file_name
    h.load_file(path, lambda: (
        h.check("PDF escaneado: error claro que menciona OCR y no reemplaza el archivo",
                bool(errors) and "OCR" in errors[0] and chat.file_name == before),
        h.next()))


def hidden_at_start():
    path = f"{h.out}/clave_al_principio.txt"
    with open(path, "w") as f:
        f.write("DATO CLAVE: la contraseña del wifi es pepino42.\n"
                + "".join(f"Línea {i}: dato irrelevante {i * 7 % 1000}.\n" for i in range(3000)))
    h.setup("qwen3:1.7b", ctx="2048")
    h.load_file(path, lambda: h.send("¿Cuál es la contraseña del wifi?", after_hidden))


def after_hidden():
    h.check("archivo que no entra: se mandan solo fragmentos y se avisa",
            "no entra entero" in meta())
    h.check("encuentra el dato escondido al principio (Ollama lo hubiera descartado)",
            "pepino42" in h.last_answer().lower(), repr(h.last_answer()[:60]))
    h.check("como no se manda entero, Ollama no lo recorta", "lo recortó en silencio" not in h.tagged("warn"))
    chat.remove_file()
    h.next()


ORD_QS = [("¿Quién firma la ordenanza como Rector?", ["soro"], 2),
          ("¿Qué título intermedio otorga la carrera?", ["analista"], 11),
          ("¿Cuántas horas reloj dura la carrera?", ["3992", "3.992"], 19),
          ("¿Cuál es la carga horaria de la asignatura Bases de Datos?", ["96", "4 horas"], 37),
          ("¿Cuánto cuesta la matrícula de la carrera?", None, None)]
NOT_FOUND = ["no lo encuentro", "no se encuentra", "no se menciona", "no menciona", "no aparece", "no esta",
             "no se especifica", "no se indica", "no hay informacion", "no contiene", "no incluye", "no proporciona"]


def ordenanza():
    if not os.path.exists(ORDENANZA):
        h.info(f"(sin {ORDENANZA}: se saltean las pruebas con la ordenanza)")
        h.next()
        return
    h.setup("qwen3:1.7b", ctx="8192", temp="0", seed="1")
    for m in app.client.running():  # VRAM libre: se prueba la indexación en GPU
        app.client.unload(m["name"])
    time.sleep(2)
    ordenanza.t0 = time.time()
    h.load_file(ORDENANZA, lambda: h.wait(
        lambda: "Búsqueda por significado lista" in meta() or "no se pudo indexar" in meta(), after_index, timeout=900))


def after_index():
    note = meta().split("▸")[-1].strip()
    h.check("la ordenanza se indexa por significado en segundo plano", "Búsqueda por significado lista" in note, note[:90])
    ask_ordenanza(list(ORD_QS))


def ask_ordenanza(qs):
    if not qs:
        synonym()
        return
    q, must, page = qs.pop(0)

    def got():
        a, pages = plain(h.last_answer()), pages_sent()
        if must is None:
            h.check(f"{q} → reconoce que no está", any(w in a for w in NOT_FOUND), repr(h.last_answer()[:80]))
        else:
            h.check(f"{q} → correcta y con la página {page}", any(w in a for w in must) and includes(pages, page),
                    f"{pages[:30]} · {h.last_answer()[:70]!r}")
        ask_ordenanza(qs)
    h.send(q, got)


def synonym():
    h.setup("qwen3:4b", think="auto", ctx="2048")
    h.send("¿Cuánto tiempo lleva recibirse?", after_synonym)


def after_synonym():
    pages = pages_sent()
    h.check("pregunta con sinónimos: la búsqueda combinada manda la página 19", includes(pages, 19), pages)
    a = plain(h.last_answer())
    h.check("…y qwen3:4b responde la duración", "5 anos" in a or "cinco" in a or "3992" in a, repr(h.last_answer()[:70]))
    chat.remove_file()
    h.load_file(ORDENANZA, lambda: h.wait(lambda: "índice guardado" in meta(), lambda: (
        h.check("al volver a cargarla usa el índice guardado", "índice guardado" in meta()),
        chat.remove_file(), h.next()), timeout=10))


sys.exit(h.run([generated_pdf, scanned_pdf, hidden_at_start, ordenanza]))
