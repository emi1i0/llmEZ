"""Ventana principal: une chat y monitor, menús, atajos y configuración."""
import queue
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

from . import __version__
from . import config as C
from . import settings as S
from . import theme
from .chat_panel import ChatPanel
from .dialogs.benchmark import BenchmarkDialog
from .dialogs.models import ModelsDialog
from .metrics import MetricsSampler
from .monitor_panel import MonitorPanel
from .ollama_client import OllamaClient, OllamaError

SHORTCUTS = (
    ("Enter", "Enviar mensaje"),
    ("Shift+Enter", "Salto de línea en el mensaje"),
    ("Esc", "Detener la generación"),
    ("Ctrl+N", "Nueva charla"),
    ("Ctrl+O / Ctrl+S", "Abrir / guardar charla"),
    ("Ctrl+E", "Exportar charla a Markdown"),
    ("Ctrl+R", "Regenerar la última respuesta"),
    ("Ctrl+Shift+C", "Copiar la última respuesta"),
    ("Ctrl+L", "Cargar archivo de contexto"),
    ("F5", "Refrescar lista de modelos"),
    ("Ctrl+B", "Benchmark"),
    ("Ctrl+Q", "Salir"),
)


class App:
    def __init__(self, root):
        self.root = root
        self.client = OllamaClient(C.OLLAMA_URL)
        self._queue = queue.Queue()
        self.settings = S.load()
        self.benchmark_running = False
        self.vram_total = None  # bytes, del último dato de nvidia-smi
        self.exclusive_var = tk.BooleanVar(value=self.settings["exclusive_vram"])
        self.unload_on_exit_var = tk.BooleanVar(value=self.settings["unload_on_exit"])
        self.loop_guard_var = tk.BooleanVar(value=self.settings["loop_guard"])
        theme.apply(root)
        root.geometry(self.settings["geometry"])

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        # El monitor se crea antes porque el chat lo usa al terminar respuestas.
        self.monitor = MonitorPanel(paned)
        self.chat = ChatPanel(paned, self, self.settings)
        paned.add(self.chat, weight=3)
        paned.add(self.monitor, weight=2)
        self._place_sash(paned)
        self._build_menu()
        self._bind_shortcuts()

        self.sampler = MetricsSampler(self.client, lambda s: self.post(self._on_sample, s),
                                      interval=C.METRICS_INTERVAL_S)
        self.sampler.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll()

    # ------------------------------------------------------------- menús
    def _menu(self, parent):
        return tk.Menu(parent, tearoff=0, bg=C.FIELD, fg=C.FG, activebackground=C.ACCENT_DIM,
                       activeforeground=C.FG, selectcolor=C.FG, bd=0, relief="flat")

    def _build_menu(self):
        chat = self.chat
        bar = tk.Menu(self.root, bg=C.PANEL, fg=C.FG, activebackground=C.ACCENT_DIM,
                      activeforeground=C.FG, bd=0, relief="flat")

        m = self._menu(bar)
        m.add_command(label="Nueva charla", accelerator="Ctrl+N", command=chat.new_chat)
        m.add_command(label="Abrir charla…", accelerator="Ctrl+O", command=chat.open_chat)
        m.add_command(label="Guardar charla…", accelerator="Ctrl+S", command=chat.save_chat)
        m.add_command(label="Exportar a Markdown…", accelerator="Ctrl+E",
                      command=chat.export_markdown)
        m.add_separator()
        m.add_command(label="Cargar archivo de contexto…", accelerator="Ctrl+L",
                      command=chat.load_file)
        m.add_separator()
        m.add_command(label="Salir", accelerator="Ctrl+Q", command=self.close)
        bar.add_cascade(label="Archivo", menu=m)

        m = self._menu(bar)
        m.add_command(label="Regenerar última respuesta", accelerator="Ctrl+R",
                      command=chat.regenerate)
        m.add_command(label="Copiar última respuesta", accelerator="Ctrl+Shift+C",
                      command=chat.copy_last_response)
        m.add_command(label="Detener", accelerator="Esc", command=chat.stop)
        m.add_separator()
        m.add_command(label="Prompt de sistema…", command=chat.edit_system_prompt)
        bar.add_cascade(label="Charla", menu=m)

        m = self._menu(bar)
        m.add_command(label="Refrescar lista", accelerator="F5", command=chat.refresh_models)
        m.add_command(label="Benchmark…", accelerator="Ctrl+B", command=self.open_benchmark)
        m.add_command(label="Bajar / borrar modelos…", command=lambda: ModelsDialog(self))
        m.add_separator()
        m.add_command(label="Descargar modelos de VRAM", command=chat.unload_models)
        bar.add_cascade(label="Modelos", menu=m)

        m = self._menu(bar)
        m.add_checkbutton(label="Solo un modelo en VRAM (descargar los demás al cargar)",
                          variable=self.exclusive_var)
        m.add_checkbutton(label="Descargar modelos de VRAM al cerrar la app",
                          variable=self.unload_on_exit_var)
        m.add_checkbutton(label="Cortar la respuesta si el modelo entra en bucle",
                          variable=self.loop_guard_var)
        bar.add_cascade(label="Opciones", menu=m)

        m = self._menu(bar)
        m.add_command(label="Atajos de teclado", command=self.show_shortcuts)
        m.add_command(label="Acerca de llmEZ", command=self.show_about)
        bar.add_cascade(label="Ayuda", menu=m)
        self.root.configure(menu=bar)

    def _bind_shortcuts(self):
        chat = self.chat
        keys = {
            "<Control-n>": chat.new_chat, "<Control-o>": chat.open_chat,
            "<Control-s>": chat.save_chat, "<Control-e>": chat.export_markdown,
            "<Control-r>": chat.regenerate, "<Control-C>": chat.copy_last_response,
            "<Control-l>": chat.load_file, "<F5>": chat.refresh_models,
            "<Control-b>": self.open_benchmark, "<Control-q>": self.close,
        }
        for seq, fn in keys.items():
            self.root.bind_all(seq, lambda e, fn=fn: (fn(), "break")[1])

    def open_benchmark(self):
        if self.benchmark_running:
            return
        if not self.chat.model_names:
            messagebox.showinfo("Sin modelos", "No hay modelos disponibles para comparar.")
            return
        BenchmarkDialog(self)

    def show_shortcuts(self):
        width = max(len(k) for k, _ in SHORTCUTS)
        messagebox.showinfo("Atajos de teclado",
                            "\n".join(f"{k.ljust(width)}   {v}" for k, v in SHORTCUTS))

    def show_about(self):
        try:
            ollama = self.client._json("/api/version", timeout=2).get("version", "?")
        except OllamaError:
            ollama = "no disponible"
        messagebox.showinfo(
            "Acerca de llmEZ",
            f"llmEZ {__version__}\n\nApp para experimentar con LLMs locales en Ollama.\n\n"
            f"Ollama: {ollama} ({C.OLLAMA_URL})\n"
            f"Configuración: {S.settings_path()}")

    # -------------------------------------------------------------------
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
        gpu = s.get("gpu")
        total = int(gpu["mem_total"] * 1024 * 1024) if gpu and gpu.get("mem_total") else None
        if total != self.vram_total:
            self.vram_total = total
            self.chat.refresh_estimate()

    def close(self):
        self.sampler.stop()
        self.chat.stop()
        data = dict(self.settings)
        data.update(self.chat.current_settings())
        data.update(exclusive_vram=self.exclusive_var.get(),
                    unload_on_exit=self.unload_on_exit_var.get(),
                    loop_guard=self.loop_guard_var.get(),
                    system_prompt=self.chat.system_prompt,
                    geometry=self.root.geometry())
        err = S.save(data)
        if err:
            print(f"llmEZ: no se pudo guardar la configuración: {err}")
        if self.unload_on_exit_var.get():
            t = threading.Thread(target=self._unload_all, daemon=True)
            t.start()
            t.join(timeout=3)  # que no trabe el cierre si Ollama no responde
        self.root.destroy()

    def _unload_all(self):
        try:
            for m in self.client.running():
                self.client.unload(m["name"])
        except OllamaError:
            pass


def main():
    root = tk.Tk(className="llmEZ")
    root.title("llmEZ · LLMs locales con Ollama")
    root.minsize(1000, 640)
    App(root)
    root.mainloop()
