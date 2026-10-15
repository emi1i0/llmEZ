"""Gestión de modelos: bajar (con progreso) y borrar del disco."""
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .. import config as C
from ..model_info import fmt_bytes
from ..ollama_client import OllamaError


class ModelsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.client = app.client
        self.stop_event = None
        self.resp = None
        self.title("Gestionar modelos")
        self.configure(bg=C.BG)
        self.transient(app.root)
        self.resizable(True, False)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Bajar modelo", style="Section.TLabel").pack(anchor="w")
        ttk.Label(frm, style="Muted.TLabel", wraplength=480, justify="left", text=(
            "Nombre como en ollama.com/library (p. ej. gemma3:1b). Para 4 GB de VRAM "
            "conviene quedarse en modelos de hasta ~4B parámetros.")).pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(frm)
        row.pack(fill="x")
        installed = set(self.app.chat.model_names)
        self.name_var = tk.StringVar()
        ttk.Combobox(row, textvariable=self.name_var, width=28,
                     values=[m for m in C.SUGGESTED_MODELS if m not in installed]).pack(side="left")
        self.pull_btn = ttk.Button(row, text="Bajar", style="Accent.TButton", command=self.pull)
        self.pull_btn.pack(side="left", padx=6)
        self.cancel_btn = ttk.Button(row, text="Cancelar", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left")
        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", pady=(8, 2))
        self.pull_status = ttk.Label(frm, text="", style="Muted.TLabel")
        self.pull_status.pack(anchor="w")

        ttk.Separator(frm).pack(fill="x", pady=12)
        ttk.Label(frm, text="Borrar modelo del disco", style="Section.TLabel").pack(anchor="w")
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=(4, 0))
        self.del_var = tk.StringVar()
        self.del_cb = ttk.Combobox(row2, textvariable=self.del_var, width=28, state="readonly")
        self.del_cb.pack(side="left")
        ttk.Button(row2, text="Borrar…", command=self.delete).pack(side="left", padx=6)
        self._refresh_delete_list()

        ttk.Button(frm, text="Cerrar", command=self.close).pack(anchor="e", pady=(14, 0))

    def _refresh_delete_list(self):
        sizes = self.app.chat.model_sizes
        self.del_cb.configure(values=[f"{n}  ({fmt_bytes(s)})" for n, s in sizes.items()])
        self.del_var.set("")

    # ---------------------------------------------------------------- pull
    def pull(self):
        name = self.name_var.get().strip()
        if not name:
            return
        self.stop_event = threading.Event()
        self.pull_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.pull_status.configure(text=f"Conectando para bajar {name}…", style="Muted.TLabel")
        threading.Thread(target=self._pull_worker, args=(name, self.stop_event),
                         daemon=True).start()

    def _post(self, fn, *args):
        self.app.post(lambda: fn(*args) if self.winfo_exists() else None)

    def _pull_worker(self, name, stop_event):
        last = (time.monotonic(), 0)
        speed = 0.0
        try:
            for ev in self.client.pull_stream(name, stop_event,
                                              on_open=lambda r: setattr(self, "resp", r)):
                total, done = ev.get("total"), ev.get("completed")
                status = ev.get("status", "")
                if total and done is not None:
                    now = time.monotonic()
                    if now - last[0] >= 0.5:
                        speed = (done - last[1]) / (now - last[0]) if done >= last[1] else 0
                        last = (now, done)
                    text = (f"{status}: {fmt_bytes(done)} / {fmt_bytes(total)}"
                            + (f" · {speed / 1e6:.1f} MB/s" if speed > 0 else ""))
                    self._post(self._on_progress, 1000 * done / total, text)
                else:
                    self._post(self._on_progress, None, status)
            if stop_event.is_set():
                self._post(self._on_pull_end, f"Cancelado. Lo bajado queda en caché y la "
                           f"próxima vez continúa.", False)
            else:
                self._post(self._on_pull_end, f"{name} listo.", True)
        except OllamaError as e:
            msg = str(e)
            if "file does not exist" in msg or "not found" in msg:
                msg += " (¿está bien escrito el nombre? Fijate en ollama.com/library)"
            self._post(self._on_pull_end, f"Error: {msg}", False)

    def _on_progress(self, value, text):
        if value is not None:
            self.progress.configure(value=value)
        self.pull_status.configure(text=text)

    def _on_pull_end(self, text, ok):
        self.pull_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.pull_status.configure(text=text, style="Ok.TLabel" if ok else "Warn.TLabel")
        if ok:
            self.progress.configure(value=1000)
            self.app.chat.refresh_models()
            self.after(800, self._refresh_delete_list)

    def cancel(self):
        if self.stop_event:
            self.stop_event.set()
            if self.resp is not None:
                self.client.abort(self.resp)

    # -------------------------------------------------------------- delete
    def delete(self):
        choice = self.del_var.get()
        if not choice:
            return
        name = choice.split("  (")[0]
        if not messagebox.askyesno(
                "Borrar modelo", f"¿Borrar {name} del disco?\n\nSe puede volver a bajar, "
                "pero hay que descargarlo de nuevo.", parent=self, icon="warning"):
            return
        try:
            self.client.delete(name)
        except OllamaError as e:
            messagebox.showerror("No se pudo borrar", str(e), parent=self)
            return
        self.pull_status.configure(text=f"{name} borrado.", style="Muted.TLabel")
        self.app.chat.refresh_models()
        self.after(800, self._refresh_delete_list)

    def close(self):
        self.cancel()
        self.destroy()
