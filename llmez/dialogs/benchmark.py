"""Benchmark: el mismo prompt en varios modelos, en condiciones comparables."""
import csv
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .. import config as C
from ..ollama_client import OllamaError

COLUMNS = (
    ("modelo", "modelo", 130, "w"),
    ("load_s", "carga", 70, "e"),
    ("ttft_s", "1er token", 75, "e"),
    ("prompt_tok_s", "prompt tok/s", 90, "e"),
    ("tok_s", "gen. tok/s", 80, "e"),
    ("tokens", "tokens", 60, "e"),
    ("gpu_pct", "en GPU", 65, "e"),
    ("mem_gb", "memoria", 75, "e"),
    ("total_s", "total", 70, "e"),
)


def _fmt(key, v):
    if v is None:
        return "–"
    if key.endswith("tok_s"):  # antes que "_s": prompt_tok_s no es un tiempo
        return f"{v:.1f}"
    if key.endswith("_s"):
        return f"{v:.2f} s"
    if key == "gpu_pct":
        return f"{v:.0f} %"
    if key == "mem_gb":
        return f"{v:.2f} GB"
    return str(v)


class BenchmarkDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.client = app.client
        self.stop_event = None
        self.current_resp = None
        self.results = []
        self.title("Benchmark de modelos")
        self.configure(bg=C.BG)
        self.transient(app.root)
        self.geometry("820x600")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build()

    def _build(self):
        pad = {"padx": 10}
        ttk.Label(self, text="Modelos a comparar", style="Section.TLabel").pack(
            anchor="w", pady=(10, 2), **pad)
        boxes = ttk.Frame(self)
        boxes.pack(fill="x", **pad)
        self.model_vars = {}
        for i, name in enumerate(self.app.chat.model_names):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(boxes, text=name, variable=var).grid(
                row=i // 4, column=i % 4, sticky="w", padx=(0, 16))
            self.model_vars[name] = var

        ttk.Label(self, text="Prompt", style="Section.TLabel").pack(
            anchor="w", pady=(10, 2), **pad)
        self.prompt = tk.Text(self, height=3, wrap="word", bg=C.FIELD, fg=C.FG, font="TkDefaultFont",
                              insertbackground=C.FG, relief="flat", padx=8, pady=6)
        self.prompt.pack(fill="x", **pad)
        self.prompt.insert("1.0", C.BENCH_PROMPT)

        opts = ttk.Frame(self)
        opts.pack(fill="x", pady=(8, 0), **pad)
        self.ctx_var = tk.StringVar(value=str(self.app.chat._num_ctx() or C.DEFAULT_NUM_CTX))
        self.max_var = tk.StringVar(value=str(C.BENCH_MAX_TOKENS))
        self.think_var = tk.StringVar(value="no")
        self.cold_var = tk.BooleanVar(value=True)
        for label, widget in (
                ("Contexto", ttk.Spinbox(opts, textvariable=self.ctx_var, width=7,
                                         values=C.NUM_CTX_CHOICES)),
                ("Máx. tokens", ttk.Spinbox(opts, textvariable=self.max_var, width=6,
                                            from_=16, to=2048, increment=16)),
                ("Pensar", ttk.Combobox(opts, textvariable=self.think_var, state="readonly",
                                        values=("no", "auto"), width=5))):
            ttk.Label(opts, text=label).pack(side="left")
            widget.pack(side="left", padx=(4, 14))
        ttk.Checkbutton(opts, text="Carga en frío (descargar todo antes de cada modelo)",
                        variable=self.cold_var).pack(side="left")
        ttk.Label(self, style="Muted.TLabel", wraplength=780, justify="left", text=(
            "Temperatura 0 y semilla fija para que cada modelo responda siempre igual. "
            "'Máx. tokens' corta la respuesta para comparar velocidades con el mismo "
            "largo (el razonamiento también cuenta).")).pack(anchor="w", pady=(4, 0), **pad)

        self.tree = ttk.Treeview(self, columns=[c[0] for c in COLUMNS], show="headings",
                                 height=8, selectmode="browse")
        for key, title, width, anchor in COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor=anchor, stretch=(key == "modelo"))
        self.tree.tag_configure("error", foreground=C.ERROR_COLOR)
        self.tree.tag_configure("best", foreground=C.GPU_COLOR)
        self.tree.pack(fill="both", expand=True, pady=(10, 0), **pad)

        self.summary = ttk.Label(self, text="", style="Muted.TLabel", anchor="w")
        self.summary.pack(fill="x", pady=(6, 0), **pad)
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=10, **pad)
        self.status = ttk.Label(bar, text="", style="Muted.TLabel")
        self.status.pack(side="left")
        ttk.Button(bar, text="Cerrar", command=self.close).pack(side="right")
        self.export_btn = ttk.Button(bar, text="Exportar CSV…", command=self.export,
                                     state="disabled")
        self.export_btn.pack(side="right", padx=6)
        self.stop_btn = ttk.Button(bar, text="Detener", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.start_btn = ttk.Button(bar, text="Iniciar", style="Accent.TButton",
                                    command=self.start)
        self.start_btn.pack(side="right", padx=6)

    # -----------------------------------------------------------------
    def start(self):
        if self.app.chat.gen:
            messagebox.showinfo("Chat ocupado", "Esperá a que termine la respuesta del chat.",
                                parent=self)
            return
        models = [m for m, v in self.model_vars.items() if v.get()]
        prompt = self.prompt.get("1.0", "end-1c").strip()
        try:
            num_ctx, max_tokens = int(self.ctx_var.get()), int(self.max_var.get())
        except ValueError:
            messagebox.showerror("Valores inválidos", "Contexto y Máx. tokens tienen que ser enteros.",
                                 parent=self)
            return
        if not models or not prompt:
            messagebox.showerror("Faltan datos", "Elegí al menos un modelo y escribí un prompt.",
                                 parent=self)
            return
        self.tree.delete(*self.tree.get_children())
        self.results = []
        self.summary.configure(text="")
        self.stop_event = threading.Event()
        self.app.benchmark_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.export_btn.configure(state="disabled")
        options = {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": 0, "seed": 42}
        think = None if self.think_var.get() == "auto" else False
        threading.Thread(target=self._run, daemon=True,
                         args=(models, prompt, options, think, self.cold_var.get(),
                               self.stop_event)).start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            if self.current_resp is not None:
                self.client.abort(self.current_resp)

    def close(self):
        self.stop()
        self.app.benchmark_running = False
        self.destroy()

    def _post(self, fn, *args):
        self.app.post(lambda: fn(*args) if self.winfo_exists() else None)

    def _run(self, models, prompt, options, think, cold, stop_event):
        """Hilo de trabajo: corre cada modelo en serie y publica una fila por modelo."""
        for i, model in enumerate(models, 1):
            if stop_event.is_set():
                break
            prefix = f"[{i}/{len(models)}] {model}"
            try:
                if cold:
                    self._post(self.status.configure, {"text": f"{prefix}: descargando modelos…"})
                    self._unload_all(stop_event)
                payload = {"model": model, "stream": True, "options": options,
                           "messages": [{"role": "user", "content": prompt}]}
                # think:false solo a los que razonan (al resto le sobra el parámetro).
                if think is False and "thinking" in (self.client.show(model).get("capabilities") or []):
                    payload["think"] = False
                self._post(self.status.configure, {"text": f"{prefix}: cargando y generando…"})
                t0 = time.monotonic()
                ttft = final = None
                for chunk in self.client.chat_stream(
                        payload, stop_event,
                        on_open=lambda r: setattr(self, "current_resp", r)):
                    msg = chunk.get("message") or {}
                    if ttft is None and (msg.get("content") or msg.get("thinking")):
                        ttft = time.monotonic() - t0
                    if chunk.get("done"):
                        final = chunk
                self.current_resp = None
                if stop_event.is_set() or final is None:
                    break
                ps = next((m for m in self.client.running() if m["name"] == model), {})
                row = self._row(model, final, ttft, ps)
            except OllamaError as e:
                row = {"modelo": model, "error": str(e)}
            self._post(self._add_row, row)
        self._post(self._done, stop_event.is_set())

    def _unload_all(self, stop_event):
        names = [m["name"] for m in self.client.running()]
        for name in names:
            self.client.unload(name)
        deadline = time.monotonic() + 10
        while names and time.monotonic() < deadline and not stop_event.is_set():
            if not self.client.running():
                break
            time.sleep(0.25)

    @staticmethod
    def _row(model, f, ttft, ps):
        ns = 1e9
        tokens = f.get("eval_count", 0)
        eval_s = f.get("eval_duration", 0) / ns
        p_count, p_s = f.get("prompt_eval_count", 0), f.get("prompt_eval_duration", 0) / ns
        size, vram = ps.get("size"), ps.get("size_vram")
        return {
            "modelo": model, "load_s": f.get("load_duration", 0) / ns, "ttft_s": ttft,
            "prompt_tok_s": p_count / p_s if p_s else None,
            "tok_s": tokens / eval_s if eval_s else None, "tokens": tokens,
            "gpu_pct": 100 * vram / size if size else None,
            "mem_gb": size / 1e9 if size else None,
            "total_s": f.get("total_duration", 0) / ns,
        }

    def _add_row(self, row):
        self.results.append(row)
        if "error" in row:
            self.tree.insert("", "end", tags=("error",),
                             values=[row["modelo"], "error: " + row["error"][:60]]
                             + [""] * (len(COLUMNS) - 2))
        else:
            self.tree.insert("", "end", iid=row["modelo"],
                             values=[_fmt(k, row.get(k)) for k, *_ in COLUMNS])

    def _done(self, stopped):
        self.app.benchmark_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        ok = [r for r in self.results if "error" not in r and r.get("tok_s")]
        self.export_btn.configure(state="normal" if self.results else "disabled")
        self.status.configure(text="Detenido" if stopped else "Terminado")
        if ok:
            best = max(ok, key=lambda r: r["tok_s"])
            self.tree.item(best["modelo"], tags=("best",))
            slow = min(ok, key=lambda r: r["tok_s"])
            text = f"Más rápido: {best['modelo']} ({best['tok_s']:.1f} tok/s)"
            if slow is not best:
                text += (f" · {best['tok_s'] / slow['tok_s']:.1f}× más que "
                         f"{slow['modelo']} ({slow['tok_s']:.1f} tok/s)")
            partial = [r["modelo"] for r in ok if (r.get("gpu_pct") or 100) < 99.5]
            if partial:
                text += " · No entraron enteros en GPU: " + ", ".join(partial)
            self.summary.configure(text=text)

    def export(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Exportar resultados", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile=time.strftime("benchmark-%Y%m%d-%H%M.csv"))
        if not path:
            return
        keys = [c[0] for c in COLUMNS] + ["error"]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in self.results:
                    w.writerow({k: r.get(k, "") for k in keys})
            self.status.configure(text=f"Exportado a {path}")
        except OSError as e:
            messagebox.showerror("No se pudo exportar", str(e), parent=self)
