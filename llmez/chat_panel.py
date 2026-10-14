"""Panel de chat: selector de modelo, opciones, streaming y estadísticas."""
import collections
import os
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from . import config as C
from .ollama_client import OllamaError
from .think_parser import ThinkTagSplitter

THINK_MODES = {"auto": None, "sí": True, "no": False}


def fmt_seconds(v):
    if v is None:
        return "n/d"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s"


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
        self.splitter = ThinkTagSplitter()
        self.last_kind = None
        self.pending_ws = ""
        self.thinking_in_field = False  # vino message.thinking
        self.thinking_in_content = False  # vino mezclado en content
        self.final = None
        self.exclusive = False
        self.freed = None  # (modelos descargados, segundos) si se liberó VRAM


class ChatPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=(8, 8, 4, 8))
        self.app = app
        self.client = app.client
        self.messages = []  # historial sin el mensaje de sistema
        self.system_prompt = C.DEFAULT_SYSTEM_PROMPT
        self.file_name = None
        self.file_text = None
        self.gen = None
        self._build()
        self._note("Elegí un modelo y escribí tu mensaje. Enter envía, Shift+Enter "
                   "agrega un salto de línea, Esc detiene la generación.")
        self.refresh_models()

    # ------------------------------------------------------------------ UI
    def _build(self):
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        text_font = (family, 11)

        row1 = ttk.Frame(self)
        row1.pack(fill="x")
        ttk.Label(row1, text="Modelo").pack(side="left")
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(row1, textvariable=self.model_var,
                                     state="readonly", width=20)
        self.model_cb.pack(side="left", padx=(4, 2))
        self.refresh_btn = ttk.Button(row1, text="↻", width=3, command=self.refresh_models)
        self.refresh_btn.pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="Pensar").pack(side="left")
        self.think_var = tk.StringVar(value="auto")
        self.think_cb = ttk.Combobox(row1, textvariable=self.think_var, state="readonly",
                                     values=list(THINK_MODES), width=5)
        self.think_cb.pack(side="left", padx=(4, 12))

        ttk.Label(row1, text="Contexto").pack(side="left")
        self.ctx_var = tk.StringVar(value=str(C.DEFAULT_NUM_CTX))
        self.ctx_spin = ttk.Spinbox(row1, textvariable=self.ctx_var, width=7,
                                    values=C.NUM_CTX_CHOICES)
        self.ctx_spin.pack(side="left", padx=(4, 12))

        self.exclusive_var = tk.BooleanVar(value=C.EXCLUSIVE_VRAM)
        self.exclusive_chk = ttk.Checkbutton(row1, text="Solo un modelo en VRAM",
                                             variable=self.exclusive_var)
        self.exclusive_chk.pack(side="left")

        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=(6, 6))
        ttk.Button(row2, text="Cargar archivo…", command=self.load_file).pack(side="left")
        self.remove_file_btn = ttk.Button(row2, text="Quitar", command=self.remove_file,
                                          state="disabled")
        self.remove_file_btn.pack(side="left", padx=(4, 6))
        self.file_label = ttk.Label(row2, text="sin archivo", style="Muted.TLabel")
        self.file_label.pack(side="left")
        self.unload_btn = ttk.Button(row2, text="Descargar modelos de VRAM",
                                     command=self.unload_models)
        self.unload_btn.pack(side="right")
        ttk.Button(row2, text="Prompt de sistema…",
                   command=self.edit_system_prompt).pack(side="right", padx=6)

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

        # Entrada y botones
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", pady=(6, 0))
        self.input = tk.Text(bottom, height=3, wrap="word", bg=C.FIELD, fg=C.FG,
                             font=text_font, insertbackground=C.FG, relief="flat",
                             padx=8, pady=6, highlightthickness=1,
                             highlightbackground=C.BORDER, highlightcolor=C.ACCENT)
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Escape>", lambda e: self.stop())
        self.chat.bind("<Escape>", lambda e: self.stop())

        btns = ttk.Frame(bottom)
        btns.pack(side="left", padx=(6, 0), fill="y")
        self.send_btn = ttk.Button(btns, text="Enviar", style="Accent.TButton",
                                   command=self.send)
        self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(btns, text="Detener", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        ttk.Button(btns, text="Nueva charla", command=self.new_chat).pack(fill="x")

        self.status = ttk.Label(self, text="", style="Muted.TLabel", anchor="w")
        self.status.pack(fill="x", pady=(4, 0))
        self.input.focus_set()

    def _append(self, text, *tags):
        at_bottom = self.chat.yview()[1] >= 0.999
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tags)
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
        for w in (self.model_cb, self.think_cb):
            w.configure(state="disabled" if busy else "readonly")
        self.ctx_spin.configure(state="disabled" if busy else "normal")
        self.exclusive_chk.configure(state="disabled" if busy else "normal")

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

    def _on_models(self, names, err):
        self.model_cb.configure(values=names)
        if err:
            self._set_status(f"Ollama no disponible: {err}")
            self._note(f"Aviso: Ollama no responde ({err}). Revisá el servicio "
                       "(systemctl status ollama) y tocá ↻.", "warn")
            return
        if self.model_var.get() not in names:
            self.model_var.set(names[0] if names else "")
        self._set_status(f"{len(names)} modelos disponibles" if names
                         else "Ollama responde pero no hay modelos (ollama pull …)")

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
            parent=self, title="Elegí un archivo de texto",
            filetypes=[("Texto", "*.txt *.md *.csv *.json *.py *.log *.html *.xml *.yaml *.yml"),
                       ("Todos", "*")])
        if not path:
            return
        try:
            if os.path.getsize(path) > C.MAX_FILE_BYTES:
                raise ValueError("pesa más de 2 MB; es demasiado para el contexto de un modelo local")
            with open(path, "rb") as f:
                raw = f.read()
            if b"\x00" in raw:
                raise ValueError("parece un archivo binario, no texto")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
        except (OSError, ValueError) as e:
            messagebox.showerror("No se pudo cargar", f"{os.path.basename(path)}: {e}", parent=self)
            return
        self.file_name, self.file_text = os.path.basename(path), text
        approx_tokens = len(text) // 4
        self.file_label.configure(text=f"{self.file_name} (~{approx_tokens} tokens)",
                                  style="TLabel")
        self.remove_file_btn.configure(state="normal")
        self._note(f"Archivo cargado: {self.file_name}, {len(text)} caracteres, "
                   f"~{approx_tokens} tokens (estimado).")
        num_ctx = self._num_ctx() or C.DEFAULT_NUM_CTX
        if approx_tokens > num_ctx * 0.75:
            self._note(f"Aviso: el archivo probablemente no entra en un contexto de {num_ctx} "
                       "tokens. Ollama recorta el principio de la conversación cuando se pasa, "
                       "así que el modelo puede perder el prompt de sistema o parte del archivo. "
                       "Subí 'Contexto' (ojo: más contexto = más VRAM).", "warn")

    def remove_file(self):
        self.file_name = self.file_text = None
        self.file_label.configure(text="sin archivo", style="Muted.TLabel")
        self.remove_file_btn.configure(state="disabled")
        self._note("Archivo quitado del contexto.")

    def edit_system_prompt(self):
        win = tk.Toplevel(self)
        win.title("Prompt de sistema")
        win.configure(bg=C.BG)
        win.transient(self.winfo_toplevel())
        txt = tk.Text(win, width=70, height=12, wrap="word", bg=C.FIELD, fg=C.FG,
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

    def _system_message(self):
        text = self.system_prompt
        if self.file_text is not None:
            text += C.FILE_PROMPT_TEMPLATE.format(name=self.file_name, text=self.file_text)
        return {"role": "system", "content": text} if text else None

    def _num_ctx(self):
        try:
            n = int(self.ctx_var.get())
        except ValueError:
            return None
        return n if n >= 128 else None

    # ---------------------------------------------------------- generación
    def send(self):
        if self.gen:
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return
        model = self.model_var.get()
        if not model:
            self._note("Error: no hay modelo seleccionado (¿Ollama está corriendo?).", "error")
            return
        num_ctx = self._num_ctx()
        if num_ctx is None:
            self._note("Error: 'Contexto' tiene que ser un número entero ≥ 128.", "error")
            return
        think = THINK_MODES[self.think_var.get()]

        self.input.delete("1.0", "end")
        self.messages.append({"role": "user", "content": text})
        self._append("Vos\n", "user_h")
        self._append(text + "\n", "content")
        self._append(f"{model}\n", "asst_h")
        self.chat.mark_set("resp_start", "end-1c")
        self.chat.mark_gravity("resp_start", "left")

        payload = {"model": model, "stream": True, "options": {"num_ctx": num_ctx},
                   "messages": [m for m in [self._system_message()] if m] + self.messages}
        if think is not None:
            payload["think"] = think

        gen = Generation(model, think, num_ctx)
        gen.exclusive = self.exclusive_var.get()
        self.gen = gen
        self._set_busy(True)
        self._set_status("Esperando primer token (si el modelo no estaba cargado, tarda más)…")
        threading.Thread(target=self._worker, args=(gen, payload), daemon=True).start()

    def _worker(self, gen, payload):
        """Corre en un hilo aparte: solo pasa mensajes a la UI vía app.post."""
        def on_open(resp):
            gen.resp = resp
        try:
            if gen.exclusive:
                self._free_vram(gen)
                if gen.stop_event.is_set():
                    return
            for chunk in self.client.chat_stream(payload, gen.stop_event, on_open=on_open):
                self.app.post(self._on_chunk, gen, chunk)
            self.app.post(self._on_done, gen, None)
        except OllamaError as e:
            self.app.post(self._on_done, gen, e)
        except Exception as e:  # inesperado: mostrarlo, no esconderlo
            self.app.post(self._on_done, gen, OllamaError(f"{type(e).__name__}: {e}"))

    def _free_vram(self, gen):
        """(Hilo de trabajo) Descarga los otros modelos antes de cargar gen.model.

        Con 4 GB de VRAM, un modelo chico ya cargado alcanza para que el 4b
        quede repartido entre CPU y GPU. Si el modelo pedido ya está cargado
        pero no entero en GPU, también se descarga para que se recargue al 100 %.
        """
        running = self.client.running()
        targets = [m["name"] for m in running if m["name"] != gen.model]
        if not targets:
            return
        mine = next((m for m in running if m["name"] == gen.model), None)
        if mine and (mine.get("size_vram") or 0) < (mine.get("size") or 0):
            targets.append(gen.model)
        t0 = time.monotonic()
        self.app.post(self._set_status, "Liberando VRAM: descargando " + ", ".join(targets) + "…")
        for name in targets:
            self.client.unload(name)
        # La descarga es asíncrona en Ollama: esperar a que /api/ps lo confirme.
        deadline = t0 + 10
        while time.monotonic() < deadline and not gen.stop_event.is_set():
            if not any(m["name"] in targets for m in self.client.running()):
                break
            time.sleep(0.25)
        gen.freed = (targets, time.monotonic() - t0)
        gen.t_start = time.monotonic()  # el primer token se mide sin contar la descarga
        self.app.post(self._set_status, "VRAM liberada; esperando primer token…")

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
        if gen.last_kind and gen.last_kind != kind:
            gen.pending_ws = ""
            self._append("\n\n")
        text = gen.pending_ws + text
        body = text.rstrip()
        gen.pending_ws = text[len(body):]
        gen.last_kind = kind
        if not body:
            return
        self._append(body, kind)
        if kind == "thinking":
            gen.thinking_text += body
        else:
            gen.content_text += body

    def _retag_as_thinking(self, gen):
        """Llegó un </think> sin apertura: todo lo mostrado como respuesta era razonamiento."""
        gen.thinking_in_content = True
        self.chat.tag_remove("content", "resp_start", "end")
        self.chat.tag_add("thinking", "resp_start", "end-1c")
        gen.thinking_text += gen.content_text
        gen.content_text = ""
        gen.last_kind = "thinking"
        gen.pending_ws = ""
        self._set_status("El modelo razonó dentro de la respuesta; se separó como razonamiento.")

    def _on_done(self, gen, err):
        if gen is not self.gen:
            return
        for kind, text in gen.splitter.flush():
            if kind != "orphan_close":
                self._emit(gen, kind, text)
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
        self._append("\n")
        if err:
            self._note(f"Error: {err}", "error")
            if "does not support thinking" in str(err):
                self._note("Este modelo no soporta razonamiento: poné 'Pensar' en auto o no.", "warn")
            self.messages.pop()
            self._set_status("Error en la generación")
            return

        if stopped:
            self._note("[detenido]", "meta")
            if gen.content_text:
                self.messages.append({"role": "assistant", "content": gen.content_text})
            else:
                self.messages.pop()
        else:
            self.messages.append({"role": "assistant", "content": gen.content_text})

        if gen.think is False and (gen.thinking_in_field or gen.thinking_in_content):
            self._note("Aviso: pediste think:false pero el modelo razonó igual. La app lo "
                       "separó (gris) y no lo guarda en el historial.", "warn")
        elif gen.thinking_in_content:
            self._note("Nota: el razonamiento vino mezclado en 'content' (etiquetas <think>); "
                       "se separó automáticamente.", "meta")
        if not stopped and not gen.content_text.strip():
            self._note("Aviso: el modelo no produjo respuesta final (solo razonamiento). "
                       "Puede que se haya quedado sin contexto.", "warn")

        stats = self._stats(gen, stopped)
        rate = f"{stats['tok_s']:.1f} tok/s" if stats["tok_s"] else "n/d tok/s"
        approx = " (aprox., cortada)" if stats["approx"] else ""
        self._note(f"▸ {stats['eval_count']} tokens · {rate} · primer token "
                   f"{fmt_seconds(stats['ttft'])} · total {fmt_seconds(stats['total_s'])}{approx}")
        if gen.freed:
            names, secs = gen.freed
            self._note(f"▸ VRAM liberada antes de cargar: {', '.join(names)} ({fmt_seconds(secs)})")
        self.app.monitor.set_last_stats(stats)
        self._set_status("Listo")

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

    def new_chat(self):
        self.stop()
        self.messages.clear()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._note("Nueva charla." + (f" Archivo en contexto: {self.file_name}."
                                      if self.file_name else ""))
        self._set_status("")
