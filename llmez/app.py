"""Ventana principal: une chat y monitor, y despacha mensajes de los hilos a la UI."""
import queue
import traceback
import tkinter as tk
from tkinter import ttk

from . import config as C
from . import theme
from .chat_panel import ChatPanel
from .metrics import MetricsSampler
from .monitor_panel import MonitorPanel
from .ollama_client import OllamaClient


class App:
    def __init__(self, root):
        self.root = root
        self.client = OllamaClient(C.OLLAMA_URL)
        self._queue = queue.Queue()
        theme.apply(root)

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        # El monitor se crea antes porque el chat lo usa al terminar respuestas.
        self.monitor = MonitorPanel(paned)
        self.chat = ChatPanel(paned, self)
        paned.add(self.chat, weight=3)
        paned.add(self.monitor, weight=2)
        self._place_sash(paned)

        self.sampler = MetricsSampler(self.client, lambda s: self.post(self._on_sample, s),
                                      interval=C.METRICS_INTERVAL_S)
        self.sampler.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll()

    def _place_sash(self, paned, tries=0):
        """Divisor chat|monitor al 60 %. Espera a que la ventana tenga ancho real:
        antes de mapearse mide 1 px y el chat quedaría colapsado."""
        width = paned.winfo_width()
        if width > 200:
            paned.sashpos(0, int(width * 0.6))
        elif tries < 100:
            self.root.after(50, self._place_sash, paned, tries + 1)

    def post(self, fn, *args):
        """Thread-safe: encola una llamada para ejecutarla en el hilo de Tk."""
        self._queue.put((fn, args))

    def _poll(self):
        try:
            while True:
                fn, args = self._queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def _on_sample(self, s):
        s["tok_s"] = self.chat.live_tok_s()
        self.monitor.update_sample(s)

    def close(self):
        self.sampler.stop()
        self.chat.stop()
        self.root.destroy()


def main():
    root = tk.Tk(className="llmEZ")
    root.title("llmEZ · LLMs locales con Ollama")
    root.geometry("1400x880")
    root.minsize(1000, 640)
    App(root)
    root.mainloop()
