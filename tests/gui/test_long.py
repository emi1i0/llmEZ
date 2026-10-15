"""Sesión larga: charla de 200 turnos, memoria estable, sin nvidia-smi, archivo enorme."""
import os
import random
import sys
import time
from tkinter import filedialog

import psutil

from common import Harness

import llmez.metrics as metrics

MEM_MINUTES = float(os.environ.get("MEM_MINUTES", "2"))
h = Harness("llmEZ prueba de sesión larga")
app, chat = h.app, h.chat
app.exclusive_var.set(False)
real_stream = app.client.chat_stream


def long_chat():
    random.seed(1)
    msgs = []
    for i in range(200):
        msgs.append({"role": "user", "content": f"Pregunta {i}: ¿qué es el concepto número {i}?"})
        body = (f"## Respuesta {i}\n\nEl **concepto {i}** se explica así: "
                + " ".join(random.choice(["modelo", "token", "*capa*", "`código`", "atención"]) for _ in range(60))
                + f"\n\n- punto {i}.a\n- punto {i}.b\n\n```python\nprint({i})\n```")
        msgs.append({"role": "assistant", "content": body, "model": "qwen3:1.7b", "thinking": "pienso " * 30,
                     "stats": {"eval_count": 120, "tok_s": 18.5, "ttft": 0.4, "total_s": 7.0}})
    chat.messages = msgs
    t0 = time.perf_counter()
    chat._render_all()
    h.root.update()
    t_render = time.perf_counter() - t0
    h.check("charla de 200 turnos: se dibuja en menos de 3 s", t_render < 3, f"{t_render:.2f} s")
    path = f"{h.out}/larga.json"
    filedialog.asksaveasfilename = lambda **k: path
    filedialog.askopenfilename = lambda **k: path
    chat.save_chat()
    chat.new_chat()
    t0 = time.perf_counter()
    chat.open_chat()
    h.root.update()
    h.check("charla de 200 turnos: se abre en menos de 3 s", len(chat.messages) == 400 and time.perf_counter() - t0 < 3,
            f"{time.perf_counter() - t0:.2f} s")
    chat.new_chat()
    h.next()


def no_gpu():
    orig = metrics.shutil.which
    metrics.shutil.which = lambda name: None if name == "nvidia-smi" else orig(name)
    h.setup("qwen3:0.6b")

    def later():
        h.check("sin nvidia-smi: el monitor lo dice", "nvidia-smi no encontrado" in app.monitor.gpu_info.cget("text"))
        h.check("sin nvidia-smi: el estimador no inventa", "sin datos de GPU" in chat.vram_label.cget("text"))
        metrics.shutil.which = orig
        h.root.after(2500, lambda: (
            h.check("vuelve nvidia-smi: el estimador se recupera", "GPU" in chat.vram_label.cget("text")
                    and "sin datos" not in chat.vram_label.cget("text")), h.next()))
    h.root.after(2500, later)


def huge_file():
    path = f"{h.out}/enorme.txt"
    with open(path, "w") as f:
        for i in range(40000):
            f.write(f"Línea {i}: dato irrelevante número {i * 7 % 1000}.\n")
    t0 = time.perf_counter()
    h.load_file(path, lambda: (
        h.check("archivo de ~1,7 MB: se lee rápido y avisa que no entra",
                time.perf_counter() - t0 < 3 and "no entra" in h.tagged("warn")),
        chat.remove_file(), h.next()))


def memory():
    proc = psutil.Process()
    samples, t0 = [], time.time()

    def sample():
        samples.append(proc.memory_info().rss / 1e6)
        if time.time() - t0 < MEM_MINUTES * 60:
            h.root.after(5000, sample)
        else:
            first, last = sum(samples[2:6]) / 4, sum(samples[-4:]) / 4
            h.check(f"memoria estable en {MEM_MINUTES:.0f} min con el monitor (crece menos de 5 MB)",
                    last - first < 5, f"{first:.1f} → {last:.1f} MB")
            h.next()
    sample()


sys.exit(h.run([long_chat, no_gpu, huge_file, memory]))
