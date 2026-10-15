"""Panel de chat: selector de modelo, opciones, streaming y estadísticas."""
import collections
import os
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from . import config as C
from . import conversation, markdown
from .documents import read_document
from .log import log
from .model_info import ModelInfo, fmt_bytes
from .ollama_client import OllamaError
from .repetition import RepetitionDetector
from .retrieval import CHARS_PER_TOKEN, Index, compact_labels, estimate_tokens
from .semantic import EMBED_MODEL, SemanticIndex, is_embedding_only
from .think_parser import ThinkTagSplitter

THINK_MODES = {"auto": None, "sí": True, "no": False}
RENDER_MIN_MS = 80  # intervalo mínimo entre redibujos con formato
CHAT_FILETYPES = [("Charla llmEZ", "*.json"), ("Todos", "*")]


def fmt_seconds(v):
    if v is None:
        return "n/d"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


def stats_line(st):
    rate = f"{st['tok_s']:.1f} tok/s" if st.get("tok_s") else "n/d tok/s"
    approx = " (aprox., cortada)" if st.get("approx") else ""
    return (f"▸ {st['eval_count']} tokens · {rate} · primer token "
            f"{fmt_seconds(st.get('ttft'))} · total {fmt_seconds(st.get('total_s'))}{approx}")


class Generation:
    """Estado de una respuesta en curso (vive en el hilo de la UI)."""

    def __init__(self, model, think, num_ctx):
        self.model = model
        self.think = think
        self.num_ctx = num_ctx
        self.stop_event = threading.Event()
        self.resp = None
        self.t_start = time.monotonic()
        self.t_first = None
        self.t_last = None
        self.n_chunks = 0
        self.recent = collections.deque()  # tiempos de chunks para tok/s en vivo
        self.thinking_text = ""
        self.content_text = ""
        self.content_marked = False  # hay una zona de respuesta (content_start..content_end)
        self.content_offset = 0  # desde dónde de content_text se dibuja esa zona
        self.render_pending = False  # hay un redibujo con formato programado
        self.render_ms = RENDER_MIN_MS
        self.splitter = ThinkTagSplitter()
        self.last_kind = None
        self.pending_ws = ""
        self.thinking_in_field = False  # vino message.thinking
        self.thinking_in_content = False  # vino mezclado en content
        self.final = None
        self.exclusive = False
        self.freed = None  # (modelos descargados, segundos) si se liberó VRAM
        self.loop_guard = None  # {'thinking': detector, 'content': detector} si está activo
        self.loop = None  # (tipo de texto, descripción) si se cortó por bucle
        self.temperature = None  # la pedida; None = la del modelo
        self.seed = None
        self.prompt_chars = 0  # tamaño de lo enviado, para detectar si Ollama lo recortó


class ChatPanel(ttk.Frame):
    def __init__(self, master, app, settings):
        super().__init__(master, padding=(8, 8, 4, 8))
        self.app = app
        self.client = app.client
        self.messages = []  # historial; role/content van al modelo, el resto es metadata
        self.system_prompt = settings["system_prompt"]
        self.file_name = None
        self.file_text = None
        self.gen = None
        self.model_sizes = {}  # nombre → bytes del archivo (de /api/tags)
        self.model_infos = {}  # nombre → ModelInfo (caché de /api/show)
        self._preferred_model = settings["model"]
        self._think_enabled = True
        self._index = None
        self._sem = None  # SemanticIndex del archivo actual (búsqueda por significado)
        self._sem_stop = None
        self._search_kind = ""
        self._chars_per_token = None  # medido en la última respuesta (ver _finish)
        self._build(settings)
        self._note("Elegí un modelo y escribí tu mensaje. Enter envía, Shift+Enter "
                   "agrega un salto de línea, Esc detiene. Clic derecho para más opciones.")
        self.refresh_models()

    @property
    def model_names(self):
        """Modelos para chatear: los de embeddings no generan texto."""
        return [n for n in self.model_sizes if not is_embedding_only(n)]

    # ------------------------------------------------------------------ UI
    def _build(self, s):
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")
        text_font = (family, 11)

        row1 = ttk.Frame(self)
        row1.pack(fill="x")
        ttk.Label(row1, text="Modelo").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(row1, textvariable=self.model_var,
                                     state="readonly", width=18)
        self.model_cb.pack(side="left", padx=(4, 2))
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_changed())
        ttk.Button(row1, text="↻", width=3, command=self.refresh_models).pack(
            side="left", padx=(0, 10))

        ttk.Label(row1, text="Pensar").pack(side="left")
        self.think_var = tk.StringVar(value=s["think"] if s["think"] in THINK_MODES else "auto")
        self.think_cb = ttk.Combobox(row1, textvariable=self.think_var, state="readonly",
                                     values=list(THINK_MODES), width=5)
        self.think_cb.pack(side="left", padx=(4, 10))

        ttk.Label(row1, text="Contexto").pack(side="left")
        self.ctx_var = tk.StringVar(value=str(s["num_ctx"]))
        self.ctx_spin = ttk.Spinbox(row1, textvariable=self.ctx_var, width=7,
                                    values=C.NUM_CTX_CHOICES)
        self.ctx_spin.pack(side="left", padx=(4, 0))
        self.ctx_var.trace_add("write", lambda *a: self.refresh_estimate())

        # Tres filas cortas en vez de dos largas: entran en la ventana mínima.
        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Temp.").pack(side="left")
        self.temp_var = tk.StringVar(value=s["temperature"])
        self.temp_spin = ttk.Spinbox(row2, textvariable=self.temp_var, width=5,
                                     from_=0.0, to=2.0, increment=0.1, format="%.1f")
        self.temp_spin.pack(side="left", padx=(4, 10))
        ttk.Label(row2, text="Semilla").pack(side="left")
        self.seed_var = tk.StringVar(value=s["seed"])
        self.seed_entry = ttk.Entry(row2, textvariable=self.seed_var, width=7)
        self.seed_entry.pack(side="left", padx=(4, 0))
        ttk.Button(row2, text="Prompt de sistema…",
                   command=self.edit_system_prompt).pack(side="right")

        row3 = ttk.Frame(self)
        row3.pack(fill="x", pady=(6, 2))
        ttk.Button(row3, text="Cargar archivo…", command=self.load_file).pack(side="left")
        self.remove_file_btn = ttk.Button(row3, text="Quitar", command=self.remove_file,
                                          state="disabled")
        self.remove_file_btn.pack(side="left", padx=(4, 6))
        self.file_label = ttk.Label(row3, text="sin archivo", style="Muted.TLabel")
        self.file_label.pack(side="left")

        # Info del modelo y estimación de VRAM
        self.info_label = ttk.Label(self, text="", style="Muted.TLabel", anchor="w")
        self.info_label.pack(fill="x", pady=(4, 0))
        self.vram_label = ttk.Label(self, text="", style="Muted.TLabel", anchor="w")
        self.vram_label.pack(fill="x", pady=(0, 6))
        for label in (self.info_label, self.vram_label):
            label.bind("<Configure>", lambda e: e.widget.configure(wraplength=max(e.width - 4, 100)))

        # Área de conversación
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True)
        self.chat = tk.Text(box, wrap="word", bg=C.PANEL, fg=C.FG, font=text_font,
                            insertbackground=C.FG, relief="flat", bd=0,
                            padx=10, pady=8, highlightthickness=0,
                            selectbackground=C.ACCENT_DIM, state="disabled")
        sb = ttk.Scrollbar(box, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)

        bold = (family, 11, "bold")
        self.chat.tag_configure("user_h", foreground=C.USER_COLOR, font=bold, spacing1=10)
        self.chat.tag_configure("asst_h", foreground=C.ASSISTANT_COLOR, font=bold, spacing1=10)
        self.chat.tag_configure("content", foreground=C.FG)
        self.chat.tag_configure("thinking", foreground=C.THINKING_COLOR,
                                font=(family, 10, "italic"))
        self.chat.tag_configure("meta", foreground=C.MUTED, font=(family, 9))
        self.chat.tag_configure("warn", foreground=C.WARN_COLOR, font=(family, 9))
        self.chat.tag_configure("error", foreground=C.ERROR_COLOR)
        markdown.configure_tags(self.chat, family, mono, 11, C.FG, C.MUTED, C.FIELD, C.ACCENT)

        self.menu = tk.Menu(self, tearoff=0, bg=C.FIELD, fg=C.FG, activebackground=C.ACCENT_DIM,
                            activeforeground=C.FG, bd=0)
        self.menu.add_command(label="Copiar", command=self.copy_selection)
        self.menu.add_command(label="Copiar última respuesta", command=self.copy_last_response)
        self.menu.add_command(label="Seleccionar todo",
                              command=lambda: self.chat.tag_add("sel", "1.0", "end"))
        self.menu.add_separator()
        self.menu.add_command(label="Regenerar última respuesta", command=self.regenerate)
        self.chat.bind("<Button-3>", lambda e: self.menu.tk_popup(e.x_root, e.y_root))

        # Entrada y botones
        # La entrada y el estado se anclan abajo *antes* que el área de la charla:
        # si falta alto, se achica la charla y no desaparecen los botones.
        self.status = ttk.Label(self, text="", style="Muted.TLabel", anchor="w")
        self.status.pack(side="bottom", fill="x", pady=(4, 0), before=box)
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", pady=(6, 0), before=box)
        self.input = tk.Text(bottom, height=3, wrap="word", bg=C.FIELD, fg=C.FG,
                             font=text_font, insertbackground=C.FG, relief="flat",
                             padx=8, pady=6, highlightthickness=1,
                             highlightbackground=C.BORDER, highlightcolor=C.ACCENT)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Escape>", lambda e: self.stop())
        self.chat.bind("<Escape>", lambda e: self.stop())

        btns = ttk.Frame(bottom)
        btns.pack(side="right", padx=(6, 0), fill="y")  # antes que la entrada, que se expande
        self.input.pack(side="left", fill="both", expand=True)
        self.send_btn = ttk.Button(btns, text="Enviar", style="Accent.TButton",
                                   command=self.send)
        self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(btns, text="Detener", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        ttk.Button(btns, text="Nueva charla", command=self.new_chat).pack(fill="x")
        ttk.Button(btns, text="Descargar de VRAM", command=self.unload_models).pack(
            fill="x", pady=(2, 0))
        self.input.focus_set()

    def _append(self, text, *tags):
        at_bottom = self.chat.yview()[1] >= 0.999
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tags)
        self.chat.configure(state="disabled")
        if at_bottom:
            self.chat.see("end")

    def _append_markdown(self, text):
        at_bottom = self.chat.yview()[1] >= 0.999
        self.chat.configure(state="normal")
        for txt, tags in markdown.parse(text):
            self.chat.insert("end", txt, ("content",) + tags)
        self.chat.configure(state="disabled")
        if at_bottom:
            self.chat.see("end")

    def _note(self, text, tag="meta"):
        self._append(text + "\n", tag)

    def _set_status(self, text):
        self.status.configure(text=text)

    def _set_busy(self, busy):
        self.send_btn.configure(state="disabled" if busy else "normal")
        self.stop_btn.configure(state="normal" if busy else "disabled")
        self.model_cb.configure(state="disabled" if busy else "readonly")
        self.think_cb.configure(
            state="readonly" if not busy and self._think_enabled else "disabled")
        for w in (self.ctx_spin, self.temp_spin, self.seed_entry):
            w.configure(state="disabled" if busy else "normal")

    def _on_enter(self, event):
        if event.state & 0x1:  # Shift: salto de línea normal
            return None
        self.send()
        return "break"

    # ------------------------------------------------------------ modelos
    def refresh_models(self):
        self._set_status("Consultando modelos…")

        def work():
            try:
                self.app.post(self._on_models, self.client.list_models(), None)
            except OllamaError as e:
                self.app.post(self._on_models, [], e)
        threading.Thread(target=work, daemon=True).start()

    def _on_models(self, models, err):
        self.model_sizes = {m["name"]: m.get("size", 0) for m in models}
        self.model_infos.clear()  # pudieron cambiar (pull/borrado)
        names = self.model_names
        self.model_cb.configure(values=names)
        if err:
            self._set_status(f"Ollama no disponible: {err}")
            self._note(f"Aviso: Ollama no responde ({err}). Revisá el servicio "
                       "(systemctl status ollama) y tocá ↻.", "warn")
            return
        current = self.model_var.get()
        if current not in names:
            if self._preferred_model in names:
                current = self._preferred_model
            else:
                current = names[0] if names else ""
            self.model_var.set(current)
        self._set_status(f"{len(names)} modelos disponibles" if names
                         else "Ollama responde pero no hay modelos (Modelos → Bajar modelo…)")
        self._on_model_changed()

    def _on_model_changed(self):
        model = self.model_var.get()
        if not model:
            self.info_label.configure(text="")
            self.vram_label.configure(text="")
            return
        if model in self.model_infos:
            self._show_model_info()
            return
        self.info_label.configure(text=f"{model}: consultando información…")

        def work():
            try:
                info = ModelInfo(model, self.client.show(model), self.model_sizes.get(model))
                self.app.post(self._on_model_info, model, info, None)
            except OllamaError as e:
                self.app.post(self._on_model_info, model, None, e)
        threading.Thread(target=work, daemon=True).start()

    def _on_model_info(self, model, info, err):
        if err:
            self.info_label.configure(text=f"{model}: no se pudo leer la info ({err})")
            return
        self.model_infos[model] = info
        if model == self.model_var.get():
            self._show_model_info()

    def _show_model_info(self):
        info = self.model_infos.get(self.model_var.get())
        if not info:
            return
        parts = [info.family, f"{info.parameter_size} parámetros", info.quantization]
        if info.context_max:
            parts.append(f"contexto máx. {info.context_max:,}".replace(",", "."))
        parts.append("razona" if info.thinks else "no razona")
        if "temperature" in info.defaults:
            parts.append(f"temp. por defecto {info.defaults['temperature']}")
        self.info_label.configure(text=" · ".join(parts))

        # Un modelo sin capacidad de razonar da error 400 con think:true.
        self._think_enabled = info.thinks
        if not info.thinks:
            self.think_var.set("auto")
        if not self.gen:
            self.think_cb.configure(state="readonly" if info.thinks else "disabled")
        self.refresh_estimate()

    def refresh_estimate(self):
        """Estimación de VRAM del modelo elegido con el contexto actual."""
        info = self.model_infos.get(self.model_var.get())
        num_ctx = self._num_ctx()
        if not info or not num_ctx:
            self.vram_label.configure(text="")
            return
        est = info.estimate_bytes(num_ctx)
        if est is None:
            self.vram_label.configure(text="VRAM: no se pudo estimar para esta arquitectura",
                                      style="Muted.TLabel")
            return
        kv = info.kv_cache_bytes(num_ctx)
        base = (f"VRAM estimada con contexto {num_ctx}: pesos {fmt_bytes(info.file_size)} "
                f"+ caché KV {fmt_bytes(kv)} + grafo ≈ {fmt_bytes(est)}")
        total = self.app.vram_total
        state, frac = info.fit(num_ctx, total)
        if state is None:
            self.vram_label.configure(text=base + " (sin datos de GPU)", style="Muted.TLabel")
            return
        best = self._max_fitting_ctx(info, total)
        tail = f" · contexto máx. que entra ≈ {best}" if best else ""
        if state == "entra":
            text, style = f"{base} → entra 100 % en GPU{tail}", "Ok.TLabel"
        elif state == "justo":
            text, style = f"{base} → entra justo{tail}", "Warn.TLabel"
        else:
            text = (f"{base} → no entra: ~{frac:.0%} en GPU, el resto en CPU (más lento)."
                    + (f" Bajá el contexto a ≤ {best}." if best else " No entra ni con contexto mínimo."))
            style = "Bad.TLabel"
        self.vram_label.configure(text=text, style=style)

    @staticmethod
    def _max_fitting_ctx(info, total):
        """Mayor num_ctx (múltiplo de 512) que entra entero en la GPU."""
        lo, hi = 0, (info.context_max or 131072) // 512
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if info.fit(mid * 512, total)[0] in ("entra", "justo"):
                lo = mid
            else:
                hi = mid - 1
        return lo * 512 or None

    def unload_models(self):
        if self.gen:
            messagebox.showinfo("Generando", "Detené la generación antes de descargar modelos.",
                                parent=self)
            return
        self._set_status("Descargando modelos de memoria…")

        def work():
            try:
                names = [m["name"] for m in self.client.running()]
                for name in names:
                    self.client.unload(name)
                self.app.post(self._set_status, "Descargados: " + ", ".join(names)
                              if names else "No había modelos cargados")
            except OllamaError as e:
                self.app.post(self._set_status, f"Error al descargar: {e}")
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------- archivo / sistema
    def load_file(self):
        path = filedialog.askopenfilename(
            parent=self, title="Elegí un archivo de texto o PDF",
            filetypes=[("Texto y PDF", "*.txt *.md *.pdf *.csv *.json *.py *.log *.html *.xml "
                        "*.yaml *.yml"), ("PDF", "*.pdf"), ("Todos", "*")])
        if not path:
            return
        name = os.path.basename(path)
        self._set_status(f"Leyendo {name}…")

        def work():  # un PDF grande puede tardar: fuera del hilo de la UI
            try:
                text, kind = read_document(path)
                self.app.post(self._on_document, name, text, kind, None)
            except (OSError, ValueError) as e:
                self.app.post(self._on_document, name, None, None, e)
        threading.Thread(target=work, daemon=True).start()

    def _on_document(self, name, text, kind, err):
        if err:
            self._set_status("")
            messagebox.showerror("No se pudo cargar", f"{name}: {err}", parent=self)
            return
        self._set_file(name, text)
        approx_tokens = len(text) // 4
        self._set_status(f"{name} cargado")
        self._note(f"Archivo cargado: {name} ({kind}), {len(text)} caracteres, "
                   f"~{approx_tokens} tokens (estimado).")
        if "sin texto" in kind:
            self._note("Aviso: algunas páginas no tienen texto (probablemente son imágenes "
                       "escaneadas); el modelo no va a ver su contenido.", "warn")
        num_ctx = self._num_ctx() or C.DEFAULT_NUM_CTX
        if approx_tokens > num_ctx * 0.75:
            self._note(f"Aviso: el archivo probablemente no entra en un contexto de {num_ctx} "
                       "tokens. Cuando se pasa, Ollama se queda solo con el final (hasta la mitad del "
                       "contexto) y descarta el resto: el modelo pierde el prompt de sistema y el "
                       "principio del archivo. "
                       "Subí 'Contexto' (ojo: más contexto = más VRAM).", "warn")
        if estimate_tokens(text) > C.SEMANTIC_MIN_TOKENS:
            self._start_semantic()

    def _start_semantic(self):
        """Indexa el archivo por significado en segundo plano (ver semantic.py)."""
        if not self.app.semantic_var.get():
            return
        if EMBED_MODEL not in self.model_sizes:
            self._note(f"Sugerencia: para que la búsqueda en el archivo entienda sinónimos, bajá "
                       f"{EMBED_MODEL} (Modelos → Bajar / borrar modelos…). Mientras tanto se busca "
                       "por palabras clave.", "meta")
            return
        if self._index is None:
            self._index = Index(self.file_text)
        sem = SemanticIndex(self.client, [t for _, t in self._index.chunks])
        self._sem, self._sem_stop = sem, threading.Event()
        if sem.load_cached():
            self._note(f"▸ Búsqueda por significado lista (índice guardado de {len(sem.texts)} "
                       f"{self._index.chunks[0][0].split()[0]}s).")
            threading.Thread(target=sem.warm_up, daemon=True).start()
            return
        stop, t0 = self._sem_stop, time.monotonic()
        total = len(sem.texts)

        def progress(done, n, device):
            self.app.post(self._sem_progress, sem, f"Indexando el archivo por significado: {done}/{n} ({device})…")

        def work():
            try:
                if sem.build(stop, progress):
                    sem.warm_up()
                    self.app.post(self._sem_done, sem, None, time.monotonic() - t0)
            except Exception as e:  # sin embeddings se sigue con BM25
                self.app.post(self._sem_done, sem, e, 0)
        self._set_status(f"Indexando el archivo por significado: 0/{total}…")
        threading.Thread(target=work, daemon=True).start()

    def _sem_progress(self, sem, text):
        if sem is self._sem and not self.gen:
            self._set_status(text)

    def _sem_done(self, sem, err, secs):
        if sem is not self._sem:
            return
        if err:
            self._sem = None
            self._note(f"Aviso: no se pudo indexar por significado ({err}); se busca solo por "
                       "palabras clave.", "warn")
        else:
            self._note(f"▸ Búsqueda por significado lista: {len(sem.texts)} fragmentos en "
                       f"{secs:.0f} s ({sem.device}). Queda guardada para la próxima vez.")
        if not self.gen:
            self._set_status("Listo")

    def _set_file(self, name, text):
        self.file_name, self.file_text = name, text
        self._index = None  # se arma al primer uso si el archivo no entra en el contexto
        if self._sem_stop:
            self._sem_stop.set()  # deja de indexar el archivo anterior
        self._sem = self._sem_stop = None
        if text is None:
            self.file_label.configure(text="sin archivo", style="Muted.TLabel")
            self.remove_file_btn.configure(state="disabled")
        else:
            self.file_label.configure(text=f"{name} (~{len(text) // 4} tokens)", style="TLabel")
            self.remove_file_btn.configure(state="normal")

    def remove_file(self):
        self._set_file(None, None)
        self._note("Archivo quitado del contexto.")

    def edit_system_prompt(self):
        win = tk.Toplevel(self)
        win.title("Prompt de sistema")
        win.configure(bg=C.BG)
        win.transient(self.winfo_toplevel())
        txt = tk.Text(win, width=70, height=12, wrap="word", bg=C.FIELD, fg=C.FG, font="TkDefaultFont",
                      insertbackground=C.FG, relief="flat", padx=8, pady=6)
        txt.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        txt.insert("1.0", self.system_prompt)
        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=10, pady=(0, 10))

        def save():
            self.system_prompt = txt.get("1.0", "end-1c").strip()
            self._note("Prompt de sistema actualizado (aplica desde el próximo mensaje).")
            win.destroy()

        def reset():
            txt.delete("1.0", "end")
            txt.insert("1.0", C.DEFAULT_SYSTEM_PROMPT)

        ttk.Button(bar, text="Guardar", style="Accent.TButton", command=save).pack(side="right")
        ttk.Button(bar, text="Cancelar", command=win.destroy).pack(side="right", padx=6)
        ttk.Button(bar, text="Restaurar por defecto", command=reset).pack(side="left")

    def _system_message(self, query="", num_ctx=None):
        """Mensaje de sistema con el archivo. Si el archivo no entra en el contexto, solo
        las partes más relevantes para `query` (ver retrieval.py). Devuelve (mensaje, partes)."""
        text = self.system_prompt
        parts = None
        if self.file_text is not None:
            body, note = self.file_text, ""
            num_ctx = num_ctx or self._num_ctx() or C.DEFAULT_NUM_CTX
            cpt = self._cpt()
            used = estimate_tokens(text, cpt) + sum(estimate_tokens(m["content"], cpt) for m in self.messages)
            budget = min(int(num_ctx * C.FILE_CONTEXT_SHARE), num_ctx - used - 1024)
            if estimate_tokens(body, cpt) > budget:
                if self._index is None:
                    self._index = Index(self.file_text)
                dense = None
                if self._sem is not None and self._sem.ready and self.app.semantic_var.get():
                    dense = self._sem.scores(query)
                self._search_kind = "palabras clave y significado" if dense else "palabras clave"
                body, parts = self._index.select(query, max(budget, 300), chars_per_token=cpt,
                                                 dense=dense)
                total = f"{len(self._index.chunks)} {self._index.chunks[0][0].split()[0]}s"
                note = C.FILE_EXCERPT_NOTE.format(parts=compact_labels(parts) if parts else "nada",
                                                  total=total)
            text += C.FILE_PROMPT_TEMPLATE.format(name=self.file_name, text=body, note=note)
        return ({"role": "system", "content": text} if text else None), parts

    def _cpt(self):
        """Caracteres por token para estimar: el medido si hay, si no uno prudente."""
        return self._chars_per_token or CHARS_PER_TOKEN

    def _num_ctx(self):
        try:
            n = int(self.ctx_var.get())
        except ValueError:
            return None
        return n if n >= 128 else None

    def current_settings(self):
        return {"model": self.model_var.get(), "think": self.think_var.get(),
                "num_ctx": self._num_ctx() or C.DEFAULT_NUM_CTX,
                "temperature": self.temp_var.get().strip(), "seed": self.seed_var.get().strip()}

    # ---------------------------------------------------------- generación
    def _read_options(self):
        """Valida los controles. Devuelve (model, think, num_ctx, options) o None."""
        model = self.model_var.get()
        if not model:
            self._note("Error: no hay modelo seleccionado (¿Ollama está corriendo?).", "error")
            return None
        num_ctx = self._num_ctx()
        if num_ctx is None:
            self._note("Error: 'Contexto' tiene que ser un número entero ≥ 128.", "error")
            return None
        options = {"num_ctx": num_ctx}
        temp = self.temp_var.get().strip().replace(",", ".")
        if temp:
            try:
                options["temperature"] = float(temp)
                if not 0 <= options["temperature"] <= 2:
                    raise ValueError
            except ValueError:
                self._note("Error: 'Temp.' tiene que ser un número entre 0 y 2 "
                           "(o vacío para usar la del modelo).", "error")
                return None
        seed = self.seed_var.get().strip()
        if seed:
            try:
                options["seed"] = int(seed)
            except ValueError:
                self._note("Error: 'Semilla' tiene que ser un número entero "
                           "(o vacío para que sea aleatoria).", "error")
                return None
        return model, THINK_MODES[self.think_var.get()], num_ctx, options

    def send(self):
        if self.gen:
            return
        if self.app.benchmark_running:
            self._note("Hay un benchmark corriendo; esperá a que termine.", "warn")
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return
        opts = self._read_options()
        if opts is None:
            return
        self.input.delete("1.0", "end")
        self._start(text, *opts)

    def _start(self, text, model, think, num_ctx, options):
        self.messages.append({"role": "user", "content": text})
        self._render_user(text)

        history = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        # La pregunta anterior ayuda a encontrar páginas en preguntas de seguimiento ("¿y eso?").
        prev_q = [m["content"] for m in self.messages[:-1] if m["role"] == "user"][-1:]
        system, parts = self._system_message(" ".join(prev_q + [text]), num_ctx)
        if parts is not None:
            self._note(f"▸ El archivo no entra entero en el contexto: van las "
                       f"{compact_labels(parts) if parts else '(ninguna)'} ({len(parts)} de "
                       f"{len(self._index.chunks)}), elegidas por {self._search_kind}.")
        messages = [m for m in [system] if m] + history
        # Si la respuesta llena el contexto, Ollama no corta: sigue descartando el principio
        # y el modelo delira. Se le pide parar donde se llena (done_reason "length").
        prompt_est = sum(estimate_tokens(m["content"], self._cpt()) for m in messages)
        options = dict(options, num_predict=max(256, num_ctx - prompt_est - 16))
        payload = {"model": model, "stream": True, "options": options, "messages": messages}
        self._append(f"{model}\n", "asst_h")
        self.chat.mark_set("resp_start", "end-1c")
        self.chat.mark_gravity("resp_start", "left")
        if think is not None:
            payload["think"] = think

        gen = Generation(model, think, num_ctx)
        gen.prompt_chars = sum(len(m["content"]) for m in payload["messages"])
        gen.exclusive = self.app.exclusive_var.get()
        gen.temperature = options.get("temperature")
        gen.seed = options.get("seed")
        if self.app.loop_guard_var.get():
            gen.loop_guard = {"thinking": RepetitionDetector(), "content": RepetitionDetector()}
        self.gen = gen
        self._set_busy(True)
        self._set_status("Esperando primer token (si el modelo no estaba cargado, tarda más)…")
        threading.Thread(target=self._worker, args=(gen, payload), daemon=True).start()

    def _worker(self, gen, payload):
        """Corre en un hilo aparte: solo pasa mensajes a la UI vía app.post."""
        def on_open(resp):
            gen.resp = resp
        try:
            sem = self._sem
            if sem is not None and not sem.ready and sem.device == "GPU":
                # Indexando en GPU: dos modelos en 4 GB hacen caer a Ollama (CUDA out of
                # memory). La indexación sigue en CPU y el chat espera a que se libere.
                self.app.post(self._set_status, "Pasando la indexación a CPU para liberar la GPU…")
                sem.release_gpu()
            if gen.exclusive:
                freed = free_vram(self.client, gen.model, gen.stop_event,
                                  lambda msg: self.app.post(self._set_status, msg))
                if freed:
                    gen.freed = freed
                    gen.t_start = time.monotonic()  # el primer token se mide sin la descarga
                if gen.stop_event.is_set():
                    return
            for chunk in self.client.chat_stream(payload, gen.stop_event, on_open=on_open):
                self.app.post(self._on_chunk, gen, chunk)
            self.app.post(self._on_done, gen, None)
        except OllamaError as e:
            self.app.post(self._on_done, gen, e)
        except Exception as e:  # inesperado: mostrarlo, no esconderlo
            self.app.post(self._on_done, gen, OllamaError(f"{type(e).__name__}: {e}"))

    def _on_chunk(self, gen, chunk):
        if gen is not self.gen:  # respuesta ya detenida
            return
        msg = chunk.get("message") or {}
        thinking = msg.get("thinking") or ""
        content = msg.get("content") or ""
        if thinking or content:
            now = time.monotonic()
            if gen.t_first is None:
                gen.t_first = now
                self._set_status("Generando…  (Esc o 'Detener' para cortar)")
            gen.t_last = now
            gen.n_chunks += 1
            gen.recent.append(now)
        if thinking:
            gen.thinking_in_field = True
            self._emit(gen, "thinking", thinking)
        if content:
            for kind, text in gen.splitter.feed(content):
                if kind == "orphan_close":
                    self._retag_as_thinking(gen)
                else:
                    if kind == "thinking":
                        gen.thinking_in_content = True
                    self._emit(gen, kind, text)
        if chunk.get("done"):
            gen.final = chunk

    def _emit(self, gen, kind, text):
        # Los espacios del final se retienen: si después cambia el tipo de
        # texto (razonamiento → respuesta) se descartan y no quedan huecos.
        if gen is not self.gen:  # se cortó (p. ej. por bucle) a mitad del chunk
            return
        if gen.last_kind and gen.last_kind != kind:
            if gen.last_kind == "content" and gen.content_marked:
                # Vuelve a razonar después de responder (raro): cerrar esa zona.
                self._render_content(gen)
                gen.content_marked = False
                gen.content_offset = len(gen.content_text)
            gen.pending_ws = ""
            self._append("\n\n")
        text = gen.pending_ws + text
        body = text.rstrip()
        gen.pending_ws = text[len(body):]
        gen.last_kind = kind
        if not body:
            return
        if kind == "thinking":
            self._append(body, kind)
            gen.thinking_text += body
        else:
            if not gen.content_marked:
                self.chat.mark_set("content_start", "end-1c")
                self.chat.mark_gravity("content_start", "left")
                self.chat.mark_set("content_end", "end-1c")
                self.chat.mark_gravity("content_end", "right")
                gen.content_marked = True
            gen.content_text += body
            self._schedule_render(gen)
        if gen.loop_guard and gen.loop_guard[kind].feed(body) and gen is self.gen:
            self._stop_loop(gen, kind)

    def _schedule_render(self, gen):
        if gen.render_pending:
            return
        gen.render_pending = True

        def run():
            if gen is self.gen and gen.render_pending:
                self._render_content(gen)
        self.after(gen.render_ms, run)

    def _render_content(self, gen):
        """Redibuja con formato markdown la respuesta en curso (zona content_start..end)."""
        gen.render_pending = False
        if not gen.content_marked:
            return
        t0 = time.perf_counter()
        at_bottom = self.chat.yview()[1] >= 0.999
        self.chat.configure(state="normal")
        self.chat.delete("content_start", "content_end")
        for txt, tags in markdown.parse(gen.content_text[gen.content_offset:]):
            self.chat.insert("content_end", txt, ("content",) + tags)
        self.chat.configure(state="disabled")
        if at_bottom:
            self.chat.see("end")
        # Respuestas muy largas: espaciar los redibujos para no trabar la UI.
        cost_ms = (time.perf_counter() - t0) * 1000
        gen.render_ms = max(RENDER_MIN_MS, int(cost_ms * 5))

    def _stop_loop(self, gen, kind):
        """El modelo entró en bucle: cortar y quedarse sin la parte repetida."""
        det = gen.loop_guard[kind]
        gen.loop = (kind, det.reason)
        if kind == "content":
            gen.content_text = det.trimmed()
        self.stop()

    def _retag_as_thinking(self, gen):
        """Llegó un </think> sin apertura: todo lo mostrado como respuesta era razonamiento."""
        gen.thinking_in_content = True
        if gen.content_marked:
            self.chat.configure(state="normal")
            self.chat.delete("content_start", "content_end")
            self.chat.insert("content_end", gen.content_text[gen.content_offset:], ("thinking",))
            self.chat.configure(state="disabled")
        self.chat.tag_remove("content", "resp_start", "end")
        self.chat.tag_add("thinking", "resp_start", "end-1c")
        gen.thinking_text += gen.content_text
        gen.content_text = ""
        gen.content_marked = False
        gen.content_offset = 0
        gen.render_pending = False
        if gen.loop_guard:  # lo que venía como respuesta pasa a contar como razonamiento
            gen.loop_guard["thinking"].feed(gen.loop_guard["content"].text)
            gen.loop_guard["content"] = RepetitionDetector()
        gen.last_kind = "thinking"
        gen.pending_ws = ""
        self._set_status("El modelo razonó dentro de la respuesta; se separó como razonamiento.")

    def _on_done(self, gen, err):
        if gen is not self.gen:
            return
        for kind, text in gen.splitter.flush():
            if kind != "orphan_close":
                self._emit(gen, kind, text)
        if gen is self.gen:  # el último pedazo pudo disparar el corte por bucle
            self._finish(gen, err=err)

    def stop(self):
        gen = self.gen
        if not gen:
            return
        gen.stop_event.set()
        if gen.resp is not None:
            self.client.abort(gen.resp)
        self._finish(gen, stopped=True)

    def _finish(self, gen, err=None, stopped=False):
        self.gen = None
        self._set_busy(False)
        # Último redibujo: incluye lo que llegó después del último programado
        # y, si se cortó por bucle, deja la respuesta ya recortada.
        self._render_content(gen)
        self._append("\n")
        self._log_outcome(gen, err, stopped)
        if err:
            self._note(f"Error: {err}", "error")
            if "does not support thinking" in str(err):
                self._note("Este modelo no soporta razonamiento: poné 'Pensar' en auto o no.", "warn")
            self.messages.pop()
            self._set_status("Error en la generación")
            return

        stats = self._stats(gen, stopped)
        reply = {"role": "assistant", "content": gen.content_text, "model": gen.model,
                 "thinking": gen.thinking_text, "stats": stats}
        if gen.loop:
            kind, reason = gen.loop
            where = "el razonamiento" if kind == "thinking" else "la respuesta"
            self._note(f"[detenido: {where} entró en bucle — {reason}]"
                       + (" Se quitó la parte repetida." if kind == "content" else ""), "warn")
            if gen.n_chunks >= gen.num_ctx * 0.8:
                self._note(f"La respuesta ya iba por ~{gen.n_chunks} tokens con un contexto de "
                           f"{gen.num_ctx}: al pasarse, Ollama descarta el principio de la charla y el "
                           "modelo pierde el hilo. Subí 'Contexto' o pedí una respuesta más corta.",
                           "meta")
            elif gen.temperature == 0:
                self._note("La temperatura 0 es la causa más común: siempre elige la palabra más "
                           "probable y se encierra. Dejá 'Temp.' vacío (la del modelo) o usá 0.6–0.8.",
                           "meta")
            else:
                self._note("Probá regenerar (Ctrl+R) o un modelo más grande: los muy chicos "
                           "se repiten más.", "meta")
        elif stopped:
            self._note("[detenido]", "meta")
        if stopped and not gen.content_text:
            self.messages.pop()  # cortada antes de responder: no queda la pregunta colgada
        else:
            self.messages.append(reply)

        pc = stats.get("prompt_count")
        if pc and gen.prompt_chars and gen.prompt_chars // 4 <= 1.5 * pc:  # sin recorte de Ollama
            self._chars_per_token = min(max(gen.prompt_chars / pc, 2.2), 4.5)
        length_cut = bool(gen.final and gen.final.get("done_reason") == "length")
        if length_cut:
            used = (stats.get("prompt_count") or 0) + stats["eval_count"]
            self._note(f"Aviso: la respuesta se cortó porque se llenó el contexto (~{used} de "
                       f"{gen.num_ctx} tokens"
                       + (", gran parte en razonamiento" if len(gen.thinking_text) > len(gen.content_text) else "")
                       + "). Pasado ese punto Ollama descarta el principio y el modelo pierde el hilo. "
                       "Subí 'Contexto', pedí una respuesta más corta"
                       + (" o desactivá el razonamiento si el modelo lo respeta." if gen.thinking_text else "."),
                       "warn")
        if gen.think is False and (gen.thinking_in_field or gen.thinking_in_content):
            self._note("Aviso: pediste think:false pero el modelo razonó igual. La app lo "
                       "separó (gris) y no lo guarda en el historial.", "warn")
        elif gen.thinking_in_content:
            self._note("Nota: el razonamiento vino mezclado en 'content' (etiquetas <think>); "
                       "se separó automáticamente.", "meta")
        if not stopped and not gen.content_text.strip():
            self._note("Aviso: el modelo no produjo respuesta final (solo razonamiento). "
                       "Puede que se haya quedado sin contexto.", "warn")

        self._note(stats_line(stats))
        if gen.freed:
            names, secs = gen.freed
            self._note(f"▸ VRAM liberada antes de cargar: {', '.join(names)} ({fmt_seconds(secs)})")
        sent = gen.prompt_chars // 4  # cota baja: en castellano suele haber más tokens que esto
        if stats.get("prompt_count") and sent > 1.5 * stats["prompt_count"] and sent > gen.num_ctx // 2:
            self._note(f"Aviso: el prompt no entraba en el contexto ({gen.num_ctx}) y Ollama lo recortó "
                       f"en silencio: procesó {stats['prompt_count']} tokens de ~{sent} o más. Se queda "
                       "con el final y descarta el resto, incluido el prompt de sistema y el principio "
                       "del archivo. Subí 'Contexto' o usá un archivo más chico.", "warn")
        elif not length_cut and stats.get("prompt_count") and stats["prompt_count"] + stats["eval_count"] >= 0.9 * gen.num_ctx:
            self._note(f"Aviso: la charla ya usa ~{stats['prompt_count'] + stats['eval_count']} de "
                       f"{gen.num_ctx} tokens de contexto. Lo próximo puede recortar el principio "
                       "(incluido el prompt de sistema). Subí 'Contexto' o empezá una charla nueva.",
                       "warn")
        self.app.monitor.set_last_stats(stats)
        self._set_status("Listo")

    def _log_outcome(self, gen, err, stopped):
        if err:
            outcome = f"error: {err}"
        elif gen.loop:
            outcome = f"bucle en {gen.loop[0]}: {gen.loop[1]}"
        elif stopped:
            outcome = "detenida por el usuario"
        else:
            outcome = "completa"
        log.info("respuesta %s | modelo=%s think=%s ctx=%s temp=%s seed=%s | %d chunks, "
                 "%d chars de respuesta, %d de razonamiento",
                 outcome, gen.model, gen.think, gen.num_ctx, gen.temperature, gen.seed,
                 gen.n_chunks, len(gen.content_text), len(gen.thinking_text))
        if gen.loop:
            det = gen.loop_guard[gen.loop[0]]
            start = det.loop_start or 0
            log.info("bucle: texto desde el corte (para revisar falsos positivos):\n%s",
                     det.text[max(0, start - 300):start + 1200])

    def _stats(self, gen, stopped):
        f = gen.final or {}
        ns = 1e9
        st = {
            "model": gen.model, "num_ctx": gen.num_ctx, "approx": not f,
            "ttft": (gen.t_first - gen.t_start) if gen.t_first else None,
            "prompt_count": f.get("prompt_eval_count"),
            "prompt_cached": f.get("prompt_eval_cached_count"),
            "prompt_s": f["prompt_eval_duration"] / ns if "prompt_eval_duration" in f else None,
            "load_s": f["load_duration"] / ns if "load_duration" in f else None,
        }
        if f:
            st["eval_count"] = f.get("eval_count", 0)
            st["eval_s"] = f.get("eval_duration", 0) / ns
            st["total_s"] = f.get("total_duration", 0) / ns
        else:  # detenida: estimación del lado del cliente (1 chunk ≈ 1 token)
            st["eval_count"] = gen.n_chunks
            st["eval_s"] = (gen.t_last - gen.t_first) if gen.t_first and gen.t_last else 0
            st["total_s"] = time.monotonic() - gen.t_start
        st["tok_s"] = st["eval_count"] / st["eval_s"] if st["eval_s"] else None
        st["prompt_tok_s"] = (st["prompt_count"] / st["prompt_s"]
                              if st["prompt_count"] and st["prompt_s"] else None)
        return st

    def live_tok_s(self):
        """tok/s de los últimos 2 s (aprox.: cada chunk del stream ≈ 1 token)."""
        gen = self.gen
        if not gen or gen.t_first is None:
            return 0.0
        now = time.monotonic()
        while gen.recent and gen.recent[0] < now - 2.0:
            gen.recent.popleft()
        window = min(2.0, max(now - gen.t_first, 0.5))
        return len(gen.recent) / window

    # ------------------------------------------------------ renderizado
    def _render_user(self, text):
        self._append("Vos\n", "user_h")
        self._append(text + "\n", "content")

    def _render_assistant(self, msg):
        self._append(f"{msg.get('model', 'asistente')}\n", "asst_h")
        if msg.get("thinking"):
            self._append(msg["thinking"] + "\n\n", "thinking")
        self._append_markdown(msg["content"])
        self._append("\n")
        if msg.get("stats"):
            self._note(stats_line(msg["stats"]))

    def _render_all(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        for m in self.messages:
            if m["role"] == "user":
                self._render_user(m["content"])
            else:
                self._render_assistant(m)

    # ---------------------------------------------------- acciones de charla
    def regenerate(self):
        """Vuelve a pedir la última respuesta (útil con temperatura > 0 o sin semilla)."""
        if self.gen or self.app.benchmark_running:
            return
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        if not self.messages or self.messages[-1]["role"] != "user":
            self._set_status("No hay una respuesta para regenerar")
            return
        opts = self._read_options()
        if opts is None:
            return
        user = self.messages.pop()
        self._render_all()
        self._start(user["content"], *opts)

    def copy_selection(self):
        try:
            text = self.chat.get("sel.first", "sel.last")
        except tk.TclError:
            return
        self.clipboard_clear()
        self.clipboard_append(text)

    def copy_last_response(self):
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                self.clipboard_clear()
                self.clipboard_append(m["content"])
                self._set_status("Última respuesta copiada al portapapeles")
                return
        self._set_status("Todavía no hay respuestas para copiar")

    def new_chat(self):
        self.stop()
        self.messages.clear()
        self._render_all()
        self._note("Nueva charla." + (f" Archivo en contexto: {self.file_name}."
                                      if self.file_name else ""))
        self._set_status("")

    def _conversation_data(self):
        return conversation.to_json(self.messages, self.current_settings(), self.system_prompt,
                                    self.file_name, self.file_text)

    def save_chat(self):
        if not self.messages:
            self._set_status("La charla está vacía: no hay nada para guardar")
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Guardar charla", defaultextension=".json",
            filetypes=CHAT_FILETYPES, initialfile=time.strftime("charla-%Y%m%d-%H%M.json"))
        if not path:
            return
        try:
            conversation.save(path, self._conversation_data())
            self._set_status(f"Charla guardada en {path}")
        except OSError as e:
            messagebox.showerror("No se pudo guardar", str(e), parent=self)

    def export_markdown(self):
        if not self.messages:
            self._set_status("La charla está vacía: no hay nada para exportar")
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Exportar a Markdown", defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Todos", "*")],
            initialfile=time.strftime("charla-%Y%m%d-%H%M.md"))
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(conversation.to_markdown(self._conversation_data()))
            self._set_status(f"Exportada a {path}")
        except OSError as e:
            messagebox.showerror("No se pudo exportar", str(e), parent=self)

    def open_chat(self):
        if self.gen:
            return
        path = filedialog.askopenfilename(parent=self, title="Abrir charla",
                                          filetypes=CHAT_FILETYPES)
        if not path:
            return
        try:
            data = conversation.load(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("No se pudo abrir", f"{os.path.basename(path)}: {e}", parent=self)
            return
        self.messages = data["messages"]
        self.system_prompt = data.get("system_prompt") or self.system_prompt
        f = data.get("file")
        if f:
            self._set_file(f["name"], f["text"])
            if estimate_tokens(f["text"]) > C.SEMANTIC_MIN_TOKENS:
                self._start_semantic()
        else:
            self._set_file(None, None)
        s = data.get("settings") or {}
        if s.get("model") in self.model_names:
            self.model_var.set(s["model"])
            self._on_model_changed()
        if s.get("think") in THINK_MODES and self._think_enabled:
            self.think_var.set(s["think"])
        for var, key in ((self.ctx_var, "num_ctx"), (self.temp_var, "temperature"),
                         (self.seed_var, "seed")):
            if key in s:
                var.set(str(s[key]))
        self._render_all()
        missing = s.get("model") and s["model"] not in self.model_names
        self._note(f"Charla abierta: {os.path.basename(path)} ({len(self.messages)} mensajes)."
                   + (f" Ojo: el modelo {s['model']} ya no está instalado." if missing else ""))


def free_vram(client, model, stop_event, report):
    """(Hilo de trabajo) Descarga los otros modelos antes de cargar `model`.

    Con 4 GB de VRAM, un modelo chico ya cargado alcanza para que el 4b quede
    repartido entre CPU y GPU. Si `model` ya está cargado pero no entero en
    GPU, también se descarga para que se recargue al 100 %.
    Devuelve (modelos descargados, segundos) o None si no hizo falta.
    """
    running = client.running()
    # Los que corren en CPU (p. ej. el de embeddings para las preguntas) no ocupan VRAM.
    targets = [m["name"] for m in running if m["name"] != model and (m.get("size_vram") or 0) > 0]
    if not targets:
        return None
    mine = next((m for m in running if m["name"] == model), None)
    if mine and (mine.get("size_vram") or 0) < (mine.get("size") or 0):
        targets.append(model)
    t0 = time.monotonic()
    report("Liberando VRAM: descargando " + ", ".join(targets) + "…")
    for name in targets:
        client.unload(name)
    # La descarga es asíncrona en Ollama: esperar a que /api/ps lo confirme.
    while time.monotonic() < t0 + 10 and not stop_event.is_set():
        if not any(m["name"] in targets for m in client.running()):
            break
        time.sleep(0.25)
    report("VRAM liberada; cargando modelo…")
    return targets, time.monotonic() - t0
