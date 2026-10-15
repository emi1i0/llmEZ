"""Frame con barra de desplazamiento vertical que aparece solo si hace falta."""
import tkinter as tk
from tkinter import ttk

from .. import config as C


class ScrollFrame(ttk.Frame):
    """Los hijos van en `self.body`. Si no entran a lo alto, aparece la barra
    y la rueda del mouse desplaza el contenido (solo con el puntero encima)."""

    def __init__(self, master, **body_options):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=C.BG, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.body = ttk.Frame(self.canvas, **body_options)
        self._window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self._bar_visible = False
        self.body.bind("<Configure>", lambda e: self._update())
        self.canvas.bind("<Configure>", lambda e: self._update())
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.bind_all(seq, self._on_wheel, add="+")

    def _update(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.itemconfigure(self._window, width=width)
        needed = self.body.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, width, needed))
        overflow = needed > height + 1
        if overflow and not self._bar_visible:
            self.scrollbar.pack(side="right", fill="y", before=self.canvas)
        elif not overflow and self._bar_visible:
            self.scrollbar.pack_forget()
            self.canvas.yview_moveto(0)
        self._bar_visible = overflow

    def _on_wheel(self, event):
        if not self._bar_visible:
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None or not str(widget).startswith(str(self)):
            return  # la rueda es para otro widget (p. ej. el chat)
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-2, "units")
        else:
            self.canvas.yview_scroll(2, "units")
