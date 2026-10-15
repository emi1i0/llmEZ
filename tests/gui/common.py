"""Base de las pruebas de interfaz: abren la app real y la manejan contra el Ollama real.

No corren con `unittest discover` (abren ventanas y usan la GPU): se corren con
tests/gui/run_all.sh. Cada prueba usa carpetas de configuración, registro y caché
temporales, así no toca las del usuario.
"""
import os
import sys
import tempfile
import time
import tkinter as tk
import unicodedata
from tkinter import filedialog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TMP = tempfile.mkdtemp(prefix="llmez-gui-")
for _var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
    os.environ[_var] = os.path.join(TMP, _var.lower())

from llmez.app import App  # noqa: E402

ORDENANZA = os.environ.get("LLMEZ_TEST_PDF", os.path.expanduser("~/Descargas/Ordenanza 1877.pdf"))


def plain(text):
    """Minúsculas y sin tildes: "Rubén Soró" y "RUBÉN SORO" valen lo mismo."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class Harness:
    def __init__(self, title, geometry="1400x880"):
        self.root = tk.Tk(className="llmEZ")
        self.root.title(title)
        self.root.geometry(geometry)
        self.app = App(self.root)
        self.chat = self.app.chat
        self.results = []
        self.steps = []
        self.out = os.path.join(TMP, "out")
        os.makedirs(self.out, exist_ok=True)

    # ------------------------------------------------------------ resultados
    def check(self, name, ok, detail=""):
        self.results.append(bool(ok))
        print(("OK    " if ok else "FALLA ") + name + (f" — {detail}" if detail else ""), flush=True)

    def info(self, text):
        print("      " + text, flush=True)

    def tagged(self, tag):
        r = self.chat.chat.tag_ranges(tag)
        return "".join(self.chat.chat.get(r[i], r[i + 1]) for i in range(0, len(r), 2))

    def last_answer(self):
        m = self.chat.messages
        return m[-1]["content"] if m and m[-1]["role"] == "assistant" else ""

    # ------------------------------------------------------------ flujo
    def run(self, steps):
        self.steps = list(steps) + [self._finish]
        self.root.after(2500, self.next)
        self.root.mainloop()
        total, ok = len(self.results), sum(self.results)
        print(f"\n{ok}/{total} OK", flush=True)
        return 0 if total and ok == total else 1

    def next(self, ms=100):
        def go():
            if self.steps:
                step = self.steps.pop(0)
                try:
                    step()
                except Exception as e:  # una excepción en un paso es una falla, no un cuelgue
                    self.check(f"paso {getattr(step, '__name__', '?')} sin excepciones", False, repr(e))
                    self.next()
        self.root.after(ms, go)

    def _finish(self):
        self.app.close()

    def wait(self, cond, then, timeout=300, every=150):
        t0 = time.time()

        def poll():
            if cond() or time.time() - t0 > timeout:
                then()
            else:
                self.root.after(every, poll)
        self.root.after(every, poll)

    def wait_gen(self, then, timeout=600):
        self.wait(lambda: self.chat.gen is None, then, timeout)

    # ------------------------------------------------------------ acciones
    def setup(self, model, think="no", ctx="2048", temp="", seed=""):
        c = self.chat
        c.model_var.set(model)
        c._on_model_changed()
        c.think_var.set(think)
        c.ctx_var.set(str(ctx))
        c.temp_var.set(str(temp))
        c.seed_var.set(str(seed))

    def send(self, text, then, keep_history=False):
        if not keep_history:
            self.chat.new_chat()
        self.chat.input.delete("1.0", "end")
        self.chat.input.insert("1.0", text)
        self.chat.send()
        self.wait_gen(then)

    def load_file(self, path, then):
        """Carga un archivo como si se eligiera en el diálogo y espera la lectura."""
        filedialog.askopenfilename = lambda **k: path
        self.chat.load_file()
        self.wait(lambda: not self.chat.status.cget("text").startswith("Leyendo"), then, timeout=60, every=50)
