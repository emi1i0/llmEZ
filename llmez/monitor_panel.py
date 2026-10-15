"""Panel de monitoreo: CPU, RAM, GPU, modelos de Ollama y velocidad de generación."""
import re
from datetime import datetime
from tkinter import ttk

import psutil

from . import config as C
from .chat_panel import fmt_seconds
from .widgets.line_chart import LineChart
from .widgets.scroll_frame import ScrollFrame

GIB = 1024 ** 3
PSTATES = {"P0": "máximo rendimiento", "P2": "alto", "P5": "medio", "P8": "reposo",
           "P12": "ahorro"}


def _expires_in(iso):
    """Segundos hasta expires_at de /api/ps (trae nanosegundos, Python acepta hasta µs)."""
    try:
        iso = re.sub(r"(\.\d{6})\d+", r"\1", iso)
        return (datetime.fromisoformat(iso) - datetime.now().astimezone()).total_seconds()
    except (TypeError, ValueError):
        return None


class MonitorPanel(ScrollFrame):
    """Los widgets van en self.body: con la ventana chica aparece una barra de desplazamiento."""

    def __init__(self, master):
        super().__init__(master, padding=(4, 8, 8, 8))

        self._section("CPU")
        self.cpu_chart = LineChart(self.body, "Total", [("total", C.CPU_COLOR)], height=64)
        self.cpu_chart.pack(fill="x")
        cores = ttk.Frame(self.body)
        cores.pack(fill="x", pady=(2, 0))
        cols = 4
        self.core_charts = []
        for i in range(psutil.cpu_count() or 1):
            ch = LineChart(cores, f"núcleo {i}", [("", C.CORE_COLOR)], height=40, small=True)
            ch.grid(row=i // cols, column=i % cols, sticky="ew", padx=1, pady=1)
            cores.columnconfigure(i % cols, weight=1)
            self.core_charts.append(ch)

        self._section("RAM")
        self.ram_chart = LineChart(self.body, "Usada", [("ram", C.RAM_COLOR)], height=56)
        self.ram_chart.pack(fill="x")

        self._section("GPU (nvidia-smi)")
        self.gpu_chart = LineChart(self.body, "", [("uso", C.GPU_COLOR), ("VRAM", C.VRAM_COLOR)],
                                   height=64)
        self.gpu_chart.pack(fill="x")
        self.gpu_info = ttk.Label(self.body, text="esperando datos…", style="Muted.TLabel")
        self.gpu_info.pack(fill="x", pady=(2, 0))

        self._section("Ollama: modelos cargados")
        cols = ("modelo", "tamaño", "reparto", "ctx", "expira")
        self.tree = ttk.Treeview(self.body, columns=cols, show="headings", height=3,
                                 selectmode="none")
        widths = {"modelo": 120, "tamaño": 64, "reparto": 110, "ctx": 56, "expira": 56}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], stretch=(c == "modelo"),
                             anchor="w" if c == "modelo" else "e")
        self.tree.tag_configure("gpu", foreground=C.GPU_COLOR)
        self.tree.tag_configure("split", foreground=C.WARN_COLOR)
        self.tree.tag_configure("muted", foreground=C.MUTED)
        self.tree.pack(fill="x")

        self._section("Generación")
        self.tok_chart = LineChart(self.body, "tok/s en vivo (aprox.)", [("tok/s", C.TOKS_COLOR)],
                                   y_max=None, fmt="{:.1f}", height=56)
        self.tok_chart.pack(fill="x")
        grid = ttk.Frame(self.body)
        grid.pack(fill="x", pady=(4, 0))
        self.stat_labels = {}
        keys = ["Modelo", "Tokens generados", "Velocidad", "Primer token",
                "Tokens del prompt", "Velocidad prompt", "Carga del modelo",
                "Tiempo total", "Contexto usado"]
        for i, key in enumerate(keys):
            r, c = divmod(i, 2)
            ttk.Label(grid, text=key, style="Muted.TLabel").grid(
                row=r, column=c * 2, sticky="w", padx=(0 if c == 0 else 14, 6))
            val = ttk.Label(grid, text="–", style="Value.TLabel")
            val.grid(row=r, column=c * 2 + 1, sticky="w")
            self.stat_labels[key] = val

    def _section(self, title):
        ttk.Label(self.body, text=title, style="Section.TLabel").pack(fill="x", pady=(8, 2))

    # ---------------------------------------------------------------
    def update_sample(self, s):
        self.cpu_chart.push(s["cpu_total"])
        for ch, v in zip(self.core_charts, s["cpu_cores"]):
            ch.push(v)

        used, total = s["ram_used"], s["ram_total"]
        self.ram_chart.set_subtitle(f"{used / GIB:.1f} / {total / GIB:.1f} GiB")
        self.ram_chart.push(100 * used / total)

        gpu = s["gpu"]
        if gpu:
            mem_pct = (100 * gpu["mem_used"] / gpu["mem_total"]
                       if gpu["mem_used"] is not None and gpu["mem_total"] else None)
            self.gpu_chart.push(gpu["util"], mem_pct)
            self.gpu_chart.title = gpu["name"]
            temp = f"{gpu['temp']:.0f} °C" if gpu["temp"] is not None else "n/d °C"
            mem = (f"{gpu['mem_used']:.0f} / {gpu['mem_total']:.0f} MiB"
                   if mem_pct is not None else "n/d")
            pdesc = PSTATES.get(gpu["pstate"], "")
            self.gpu_info.configure(text=f"VRAM {mem} · {temp} · {gpu['pstate']}"
                                         + (f" ({pdesc})" if pdesc else ""),
                                    style="Muted.TLabel")
        else:
            self.gpu_chart.push(None, None)
            self.gpu_info.configure(text=f"GPU: n/d ({s['gpu_error']})", style="Warn.TLabel")

        self._update_ps(s["ollama_ps"], s["ollama_error"])
        self.tok_chart.push(s.get("tok_s"))

    def _update_ps(self, models, err):
        self.tree.delete(*self.tree.get_children())
        if models is None:
            self.tree.insert("", "end", values=("Ollama n/d", "", "", "", ""), tags=("split",))
            return
        if not models:
            self.tree.insert("", "end", values=("(ninguno)", "", "", "", ""), tags=("muted",))
            return
        for m in models:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            gpu_pct = round(100 * vram / size) if size else 0
            if gpu_pct >= 100:
                split, tag = "100% GPU", "gpu"
            elif gpu_pct <= 0:
                split, tag = "100% CPU", "split"
            else:
                split, tag = f"{100 - gpu_pct}%/{gpu_pct}% CPU/GPU", "split"
            exp = _expires_in(m.get("expires_at"))
            exp_txt = f"{int(exp // 60)}:{int(exp % 60):02d}" if exp is not None and exp > 0 else "–"
            self.tree.insert("", "end", tags=(tag,), values=(
                m.get("name", "?"), f"{size / 1e9:.2f} GB", split,
                m.get("context_length", "n/d"), exp_txt))
        self.tree.configure(height=max(1, min(len(models), 4)))

    def set_last_stats(self, st):
        L = self.stat_labels

        def put(key, text, warn=False):
            L[key].configure(text=text, foreground=C.WARN_COLOR if warn else C.FG)

        approx = " ≈" if st["approx"] else ""
        put("Modelo", st["model"])
        put("Tokens generados", f"{st['eval_count']}{approx}")
        put("Velocidad", f"{st['tok_s']:.1f} tok/s{approx}" if st["tok_s"] else "n/d")
        put("Primer token", fmt_seconds(st["ttft"]))
        prompt = "n/d"
        if st["prompt_count"] is not None:
            prompt = str(st["prompt_count"])
            if st["prompt_cached"]:
                prompt += f" ({st['prompt_cached']} en caché)"
        put("Tokens del prompt", prompt)
        put("Velocidad prompt", f"{st['prompt_tok_s']:.0f} tok/s" if st["prompt_tok_s"] else "n/d")
        put("Carga del modelo", fmt_seconds(st["load_s"]), warn=(st["load_s"] or 0) > 2)
        put("Tiempo total", fmt_seconds(st["total_s"]))
        if st["prompt_count"] is not None:
            used = st["prompt_count"] + st["eval_count"]
            put("Contexto usado", f"{used} / {st['num_ctx']}",
                warn=used >= 0.9 * st["num_ctx"])
        else:
            put("Contexto usado", "n/d")
