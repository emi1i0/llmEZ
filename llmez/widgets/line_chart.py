"""Gráfico de línea liviano sobre tk.Canvas (sin matplotlib)."""
import tkinter as tk
from collections import deque

from .. import config as C


class LineChart(tk.Canvas):
    """Ventana deslizante de `points` muestras para una o más series.

    y_max=None escala el eje automáticamente al máximo visible.
    Los valores None se dibujan como huecos (dato no disponible).
    """

    def __init__(self, master, title, series, y_max=100.0, fmt="{:.0f}%",
                 height=70, points=C.HISTORY_POINTS, small=False):
        super().__init__(master, height=height, width=60, bg=C.PANEL,
                         highlightthickness=0, bd=0)
        self.title = title
        self.series = list(series)  # [(nombre, color)]
        self.y_max = y_max
        self.fmt = fmt
        self.small = small
        self.subtitle = ""
        self.data = [deque([None] * points, maxlen=points) for _ in self.series]
        self.bind("<Configure>", lambda e: self.redraw())

    def push(self, *values):
        for d, v in zip(self.data, values):
            d.append(v)
        self.redraw()

    def set_subtitle(self, text):
        self.subtitle = text

    def _scale(self):
        if self.y_max is not None:
            return self.y_max
        peak = max((v for d in self.data for v in d if v is not None), default=0)
        return max(peak * 1.15, 1.0)

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        font = ("TkDefaultFont", 7 if self.small else 9)
        pad = 3 if self.small else 5
        top = 13 if self.small else 18
        x0, y0, x1, y1 = pad, top, w - pad, h - pad
        ymax = self._scale()

        for frac in (0.0, 0.5, 1.0):
            y = y1 - (y1 - y0) * frac
            self.create_line(x0, y, x1, y, fill=C.GRID,
                             dash=() if frac == 0.0 else (2, 3))
        if self.y_max is None and not self.small:
            self.create_text(x0 + 2, y0 + 1, anchor="nw", fill=C.MUTED,
                             font=("TkDefaultFont", 7), text=f"máx {ymax:.1f}")

        title = f"{self.title}   {self.subtitle}" if self.subtitle else self.title
        self.create_text(x0, 1, anchor="nw", text=title, fill=C.MUTED, font=font)

        # Valor actual de cada serie, alineado a la derecha.
        x = x1
        for (name, color), d in reversed(list(zip(self.series, self.data))):
            v = d[-1]
            txt = self.fmt.format(v) if v is not None else "n/d"
            if len(self.series) > 1:
                txt = f"{name} {txt}"
            item = self.create_text(x, 1, anchor="ne", text=txt, fill=color, font=font)
            x = self.bbox(item)[0] - 8

        n = len(self.data[0])
        step = (x1 - x0) / max(n - 1, 1)
        width = 1 if self.small else 2
        for (_, color), d in zip(self.series, self.data):
            pts = []
            for i, v in enumerate(d):
                if v is None:
                    self._polyline(pts, color, width)
                    pts = []
                    continue
                y = y1 - (y1 - y0) * min(max(v / ymax, 0.0), 1.0)
                pts += (x0 + i * step, y)
            self._polyline(pts, color, width)

    def _polyline(self, pts, color, width):
        if len(pts) >= 4:
            self.create_line(*pts, fill=color, width=width)
        elif len(pts) == 2:
            self.create_oval(pts[0] - 1, pts[1] - 1, pts[0] + 1, pts[1] + 1,
                             fill=color, outline=color)
