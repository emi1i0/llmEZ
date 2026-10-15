"""Integración contra el Ollama real: estimador, markdown, charlas, benchmark, modelos, fallas, estrés."""
import csv
import json
import os
import re
import sys
import threading
import time
import tkinter as tk
import urllib.request
from tkinter import filedialog, messagebox

from common import Harness

h = Harness("llmEZ prueba con Ollama")
app, chat = h.app, h.chat
payloads = []
real_stream = app.client.chat_stream


def spy(payload, *a, **k):
    payloads.append(json.loads(json.dumps(payload)))
    return real_stream(payload, *a, **k)


app.client.chat_stream = spy


def api(path, payload=None):
    req = urllib.request.Request("http://localhost:11434" + path,
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    return json.loads(body) if body.strip() else {}


def unload_all():
    for m in api("/api/ps")["models"]:
        api("/api/generate", {"model": m["name"], "keep_alive": 0})
    time.sleep(1.5)


# ---------------------------------------------------------------- modelo y estimador
def model_info():
    h.check("los modelos de embeddings no aparecen para chatear",
            not any("embed" in n for n in chat.model_names), str(chat.model_names))
    h.setup("qwen3:4b", ctx="2048")
    h.wait(lambda: "qwen3:4b" in chat.model_infos and app.vram_total, check_estimates, timeout=30)


def check_estimates():
    info, vram = chat.info_label.cget("text"), chat.vram_label.cget("text")
    h.check("info de qwen3:4b (razona, Q4_K_M)", "razona" in info and "Q4_K_M" in info)
    h.check("estimador: con contexto 2048 entra", "entra" in vram and "no entra" not in vram, vram[-60:])
    chat.ctx_var.set("8192")
    h.root.update()
    h.check("estimador: con 8192 no entra y sugiere bajar el contexto",
            "no entra" in chat.vram_label.cget("text") and "Bajá el contexto" in chat.vram_label.cget("text"))
    chat.ctx_var.set("2048")
    h.setup("smollm2:135m")
    h.wait(lambda: "smollm2:135m" in chat.model_infos, lambda: (
        h.check("un modelo que no razona bloquea 'Pensar'",
                str(chat.think_cb.cget("state")) == "disabled" and chat.think_var.get() == "auto"),
        h.next()), timeout=30)


def estimate_vs_real():
    h.setup("qwen3:4b", ctx="8192")
    h.root.update()
    est = chat.vram_label.cget("text")
    unload_all()
    api("/api/generate", {"model": "qwen3:4b", "options": {"num_ctx": 8192}})
    time.sleep(2.5)
    h.root.update()
    ps = api("/api/ps")["models"][0]
    real = 100 * ps["size_vram"] / ps["size"]
    m = re.search(r"~(\d+)% en GPU", est)
    h.check("estimador vs real (qwen3:4b, contexto 8192): diferencia ≤ 10 puntos",
            m is not None and abs(int(m.group(1)) - real) <= 10, f"estimado {m.group(1) if m else '?'}% · real {real:.0f}%")
    rows = [app.monitor.tree.item(i, "values") for i in app.monitor.tree.get_children()]
    h.check("el monitor marca el reparto CPU/GPU", any("CPU/GPU" in r[2] for r in rows), str(rows))
    chat.ctx_var.set("2048")
    unload_all()
    h.next()


# ---------------------------------------------------------------- respuestas
def markdown_and_regenerate():
    h.setup("qwen3:1.7b", ctx="2048", temp="0", seed="42")
    h.send("Dame una lista de 3 frutas, con el nombre de cada una en **negrita**, y después una "
           "línea de código Python en un bloque ```.", after_markdown)


def after_markdown():
    h.check("markdown: negrita, lista y bloque de código con formato",
            all(chat.chat.tag_ranges(t) for t in ("md_bold", "md_list0", "md_codeblock")))
    h.check("markdown: sin asteriscos crudos en pantalla", "**" not in chat.chat.get("content_start", "end"))
    chat.regenerate()
    h.wait_gen(lambda: (setattr(after_markdown, "second", h.last_answer()), chat.regenerate(),
                        h.wait_gen(after_regenerate)))


def after_regenerate():
    h.check("regenerar reemplaza la respuesta (no la duplica)", len(chat.messages) == 2)
    h.check("semilla fija + temperatura 0: las regeneradas son idénticas", h.last_answer() == after_markdown.second)
    path_json, path_md = f"{h.out}/charla.json", f"{h.out}/charla.md"
    filedialog.asksaveasfilename = lambda **k: path_json if k.get("defaultextension") == ".json" else path_md
    filedialog.askopenfilename = lambda **k: path_json
    chat.save_chat()
    chat.export_markdown()
    saved = json.load(open(path_json))
    chat.new_chat()
    chat.open_chat()
    h.check("guardar y abrir restaura los mensajes con formato",
            chat.messages == saved["messages"] and bool(chat.chat.tag_ranges("md_bold")))
    md = open(path_md, encoding="utf-8").read()
    h.check("exportar a Markdown", "## Vos" in md and "## qwen3:1.7b" in md)
    h.next()


def multiturn():
    payloads.clear()
    h.setup("qwen3:1.7b", temp="", seed="")
    h.send("Mi nombre es Ramiro. Respondé solo: anotado.", lambda: (
        h.setup("qwen3:0.6b"),
        h.send("¿Cómo me llamo? Respondé solo el nombre.", after_multiturn, keep_history=True)))


def after_multiturn():
    p = payloads[-1]
    h.check("varios turnos: se manda system, user, assistant, user",
            [m["role"] for m in p["messages"]] == ["system", "user", "assistant", "user"])
    h.check("varios turnos: sin metadata (razonamiento, estadísticas) en lo que se manda",
            {k for m in p["messages"] for k in m} == {"role", "content"})
    h.check("cambio de modelo a mitad de charla: el historial llega al modelo nuevo",
            p["model"] == "qwen3:0.6b" and "Ramiro" in p["messages"][1]["content"])
    h.info(f"(respuesta de qwen3:0.6b: {h.last_answer()[:60]!r})")
    h.next()


def think_yes():
    h.setup("qwen3:4b", think="sí")
    h.send("¿Cuánto es 17 × 3? Respondé solo el número.", lambda: (
        h.check("qwen3:4b con pensar en sí: razonamiento separado y respuesta correcta",
                len(chat.messages[-1]["thinking"]) > 30 and "51" in h.last_answer(), repr(h.last_answer()[:30])),
        h.next()))


def think_no_qwen3_4b():
    h.setup("qwen3:4b", think="no")
    h.send("¿Cuánto es 2+3? Respuesta corta.", lambda: (
        h.check("qwen3:4b con pensar en no: el razonamiento colado se separa y se avisa",
                "pediste think:false" in h.tagged("warn") and len(h.last_answer()) < 40, repr(h.last_answer()[:40])),
        h.next()))


# ---------------------------------------------------------------- archivos
def file_roundtrip():
    path = f"{h.out}/datos_latin1.txt"
    with open(path, "wb") as f:
        f.write("Código secreto del depósito: ÑANDÚ-4471. Año de fundación: 1987.\n".encode("latin-1"))
    h.setup("qwen3:1.7b")
    chat.system_prompt = "Respondé en español, muy breve."
    h.load_file(path, lambda: (
        h.check("archivo en latin-1 se lee bien", "ÑANDÚ-4471" in (chat.file_text or "")),
        h.send("¿Cuál es el código secreto del depósito?", after_file)))


def after_file():
    system = payloads[-1]["messages"][0]["content"]
    h.check("prompt de sistema editado + archivo llegan al modelo",
            system.startswith("Respondé en español, muy breve.") and "ÑANDÚ-4471" in system)
    h.check("el modelo usa el archivo", "4471" in h.last_answer(), repr(h.last_answer()[:60]))
    path = f"{h.out}/con_archivo.json"
    filedialog.asksaveasfilename = lambda **k: path
    filedialog.askopenfilename = lambda **k: path
    chat.save_chat()
    chat.remove_file()
    chat.new_chat()
    chat.open_chat()
    h.check("abrir charla restaura archivo y prompt de sistema",
            chat.file_name == "datos_latin1.txt" and chat.system_prompt == "Respondé en español, muy breve.")
    chat.remove_file()
    chat.system_prompt = app.settings["system_prompt"]
    h.next()


# ---------------------------------------------------------------- atajos y fallas
def shortcuts():
    chat.messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "respuesta para copiar"}]
    chat.chat.focus_set()
    h.root.update()
    h.root.event_generate("<Control-C>", when="tail")
    h.root.update()
    try:
        clip = h.root.clipboard_get()
    except tk.TclError:
        clip = ""
    h.check("Ctrl+Shift+C copia la última respuesta", clip == "respuesta para copiar", repr(clip))
    chat.chat.event_generate("<Control-n>")
    h.root.update()
    h.check("Ctrl+N empieza una charla nueva", chat.messages == [])
    h.next()


def ollama_down():
    good = app.client.base_url
    app.client.base_url = "http://127.0.0.1:9"
    h.setup("qwen3:0.6b")
    h.send("hola", lambda: (
        h.check("Ollama caído: error claro y sin pregunta colgada",
                "no se pudo conectar" in h.tagged("error") and chat.messages == []),
        h.root.after(2500, lambda: ollama_back(good))))


def ollama_back(good):
    rows = [app.monitor.tree.item(i, "values") for i in app.monitor.tree.get_children()]
    h.check("Ollama caído: el monitor muestra n/d", bool(rows) and rows[0][0] == "Ollama n/d", str(rows))
    app.client.base_url = good
    chat.refresh_models()
    h.root.after(2000, lambda: (
        h.check("Ollama vuelve: se recupera la lista de modelos", len(chat.model_names) >= 4),
        h.next()))


# ---------------------------------------------------------------- benchmark y modelos
def benchmark():
    from llmez.dialogs.benchmark import BenchmarkDialog
    d = BenchmarkDialog(app)
    for name, var in d.model_vars.items():
        var.set(name in ("smollm2:135m", "qwen3:0.6b"))
    d.max_var.set("32")
    d.start()
    h.wait(lambda: not app.benchmark_running, lambda: after_benchmark(d), timeout=300)


def after_benchmark(d):
    path = f"{h.out}/bench.csv"
    filedialog.asksaveasfilename = lambda **k: path
    d.export()
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    h.check("benchmark: 2 filas exportadas a CSV con tok/s numérico",
            len(rows) == 2 and all(float(r["tok_s"]) > 0 for r in rows),
            str([(r["modelo"], round(float(r["tok_s"]), 1)) for r in rows]))
    d.close()
    h.next()


def models_dialog():
    from llmez.dialogs.models import ModelsDialog
    d = ModelsDialog(app)
    d.name_var.set("modelo-que-no-existe:1b")
    d.pull()
    h.wait(lambda: str(d.pull_btn.cget("state")) == "normal", lambda: (
        h.check("bajar un nombre inexistente: error claro", "Error" in d.pull_status.cget("text"),
                d.pull_status.cget("text")[:80]),
        d.close(), delete_copy()), timeout=60)


def delete_copy():
    api("/api/copy", {"source": "smollm2:135m", "destination": "llmez-borrar:test"})
    chat.refresh_models()
    h.root.after(1500, delete_copy2)


def delete_copy2():
    from llmez.dialogs.models import ModelsDialog
    d = ModelsDialog(app)
    choice = next((v for v in d.del_cb.cget("values") if v.startswith("llmez-borrar:test")), "")
    d.del_var.set(choice)
    messagebox.askyesno = lambda *a, **k: True
    d.delete()
    names = [m["name"] for m in api("/api/tags")["models"]]
    h.check("borrar un modelo (copia descartable) sin tocar los demás",
            "llmez-borrar:test" not in names and all(n in names for n in ("qwen3:0.6b", "qwen3:1.7b", "qwen3:4b")))
    d.close()
    h.next(1000)


# ---------------------------------------------------------------- estrés
def stress():
    h.setup("qwen3:0.6b")
    chat.new_chat()
    stress.base, stress.n = threading.active_count(), 0
    cycle()


def cycle():
    if stress.n == 20:
        h.root.after(3000, after_stress)
        return
    stress.n += 1
    chat.input.insert("1.0", f"Contá hasta 100. Intento {stress.n}")
    chat.send()
    h.root.after(150 + 40 * (stress.n % 5), lambda: (chat.stop(), h.root.after(60, cycle)))


def after_stress():
    h.check("20 envíos y detenciones rápidas: interfaz libre y sin hilos colgados",
            chat.gen is None and threading.active_count() <= stress.base + 1,
            f"hilos {stress.base} → {threading.active_count()}")
    h.check("historial coherente tras el estrés (roles alternados)",
            all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(chat.messages)))
    n = len(payloads)
    chat.new_chat()
    chat.input.insert("1.0", "   \n  ")
    chat.input.event_generate("<Return>")
    h.root.update()
    h.check("Enter con la entrada vacía no manda nada", len(payloads) == n)
    errors = []
    messagebox.showerror = lambda t, m, **k: errors.append(m)
    for name, body in {"roto.json": "{no es json", "ajeno.json": json.dumps({"hola": 1}),
                       "futuro.json": json.dumps({"format": "llmez-chat", "version": 99, "messages": []})}.items():
        path = f"{h.out}/{name}"
        open(path, "w").write(body)
        filedialog.askopenfilename = lambda p=path, **k: p
        chat.open_chat()
    h.check("charlas dañadas: un aviso por cada una, sin romper", len(errors) == 3)
    h.root.geometry("1000x640")
    h.root.after(1500, small_window)


def small_window():
    h.check("ventana mínima: entrada y botones visibles",
            all(w.winfo_ismapped() and w.winfo_width() > 20 for w in (chat.send_btn, chat.stop_btn, chat.input)))
    h.root.geometry("1400x880")
    h.next()


def close_while_generating():
    h.setup("qwen3:1.7b")
    chat.new_chat()
    chat.input.insert("1.0", "Escribí un cuento largo.")
    chat.send()
    h.root.after(4000, lambda: (setattr(close_while_generating, "t", time.time()), app.close()))


code = h.run([model_info, estimate_vs_real, markdown_and_regenerate, multiturn, think_yes, think_no_qwen3_4b,
              file_roundtrip, shortcuts, ollama_down, benchmark, models_dialog, stress, close_while_generating])
closed = time.time() - close_while_generating.t
cfg = json.load(open(os.path.join(os.environ["XDG_CONFIG_HOME"], "llmez", "settings.json")))
ok = closed < 2 and cfg["model"] == "qwen3:1.7b"
print(("OK    " if ok else "FALLA ") + f"cerrar mientras genera: cierra en {closed:.2f} s y guarda la configuración", flush=True)
sys.exit(code if ok else 1)
