"""Casos borde con respuestas guionadas: no dependen de lo que genere el modelo."""
import json
import sys
import time
from tkinter import filedialog

from common import Harness

h = Harness("llmEZ prueba guionada")
app, chat = h.app, h.chat
app.exclusive_var.set(False)  # los guiones no deben tocar los modelos reales
real_stream = app.client.chat_stream
script = {"chunks": [], "delay": 0.01, "error": None}


def fake_stream(payload, stop_event, on_open=None, timeout=600):
    for ch in script["chunks"]:
        if stop_event.is_set():
            return
        time.sleep(script["delay"])
        yield ch
    if script["error"]:
        from llmez.ollama_client import OllamaError
        raise OllamaError(script["error"])
    yield {"done": True, "eval_count": len(script["chunks"]), "eval_duration": 10**9,
           "prompt_eval_count": 10, "prompt_eval_duration": 10**8, "total_duration": 2 * 10**9}


def C(text):
    return {"message": {"role": "assistant", "content": text}, "done": False}


def T(text):
    return {"message": {"role": "assistant", "content": "", "thinking": text}, "done": False}


def scripted(text, chunks, then, delay=0.01, error=None):
    app.client.chat_stream = fake_stream
    script.update(chunks=chunks, delay=delay, error=error)
    chat.model_var.set("qwen3:1.7b")
    h.send(text, then)


def emojis():
    scripted("emoji", [C("Hola 😀 qué tal 🚀🔥 "), C("**negrita 🎉** y `código 🐍`")], lambda: (
        h.check("emojis no cuelgan la app", "Hola" in h.tagged("content")), h.next()))


def interleaved():
    scripted("x", [T("pienso A"), C("**Parte 1**"), T("pienso B"), C("Parte *2*")], lambda: (
        h.check("razonamiento intercalado: historial con las dos partes",
                h.last_answer() == "**Parte 1**Parte *2*", repr(h.last_answer())),
        h.check("razonamiento intercalado: todo el razonamiento en gris",
                "pienso A" in h.tagged("thinking") and "pienso B" in h.tagged("thinking")),
        h.check("razonamiento intercalado: sin asteriscos ni texto duplicado",
                "**" not in chat.chat.get("resp_start", "end")
                and chat.chat.get("1.0", "end").count("Parte 1") == 1),
        h.next()))


def orphan_close_after_markdown():
    chunks = [C("Okay, **el usuario** pide algo.\n"), C("- punto uno\n"), C("</think>\n\n"),
              C("**Respuesta:** 5")]
    scripted("x", chunks, lambda: (
        h.check("</think> suelto tras markdown: el razonamiento vuelve a gris en crudo",
                "**el usuario**" in h.tagged("thinking")),
        h.check("</think> suelto tras markdown: la respuesta queda con formato",
                h.last_answer() == "**Respuesta:** 5" and "Respuesta" in h.tagged("md_bold")),
        h.next()), delay=0.15)  # lento: hay redibujos con formato antes del </think>


def loop_in_thinking():
    chunks = [T("Wait, let me reconsider the whole question carefully once more. ") for _ in range(30)]
    scripted("x", chunks + [C("no debería llegar")], lambda: (
        h.check("bucle en el razonamiento: se corta", "razonamiento entró en bucle" in h.tagged("warn")),
        h.check("bucle en el razonamiento: no queda la pregunta colgada", chat.messages == []),
        h.next()))


def error_midstream():
    scripted("x", [C("**Antes** del error")], lambda: (
        h.check("error a mitad: se muestra", "sin memoria" in h.tagged("error")),
        h.check("error a mitad: lo recibido queda con formato", "Antes" in h.tagged("md_bold")),
        h.check("error a mitad: el historial no queda con la pregunta colgada", chat.messages == []),
        h.next()), error="se quedó sin memoria")


def huge_response():
    text = "".join(f"## Sección {n}\n\nTexto {n} con **negrita {n}** y `código_{n}`.\n\n"
                   f"- punto {n}.1\n- punto {n}.2\n\n" for n in range(600))
    gaps, last = [], [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now
        if chat.gen is not None or len(gaps) < 3:
            h.root.after(20, tick)
    h.root.after(20, tick)
    scripted("x", [C(text[i:i + 8]) for i in range(0, len(text), 8)], lambda: (
        h.check("respuesta enorme (~50 kB): llega completa", len(h.last_answer()) > 30000,
                f"{len(h.last_answer())} caracteres"),
        h.check("respuesta enorme: la interfaz nunca se traba más de 250 ms", max(gaps) < 0.25,
                f"peor pausa {max(gaps) * 1000:.0f} ms"),
        h.next()), delay=0.001)


def invalid_values():
    app.client.chat_stream = fake_stream
    chat.new_chat()
    for var, bad, label in ((chat.temp_var, "3", "Temp."), (chat.seed_var, "abc", "Semilla"),
                            (chat.ctx_var, "12", "Contexto")):
        old = var.get()
        var.set(bad)
        chat.input.insert("1.0", "mensaje importante")
        chat.send()
        h.check(f"valor inválido en {label}: avisa y conserva el mensaje",
                chat.input.get("1.0", "end-1c") == "mensaje importante" and label in h.tagged("error")
                and chat.gen is None)
        chat.input.delete("1.0", "end")
        var.set(old)
    h.next()


def old_chat_with_missing_model():
    path = f"{h.out}/charla_vieja.json"
    with open(path, "w") as f:
        json.dump({"format": "llmez-chat", "version": 1, "settings": {"model": "llama9:70b", "num_ctx": 4096},
                   "system_prompt": "Sos un pirata.", "file": None,
                   "messages": [{"role": "user", "content": "hola"},
                                {"role": "assistant", "content": "**Arr!**", "model": "llama9:70b"}]}, f)
    filedialog.askopenfilename = lambda **k: path
    before = chat.model_var.get()
    chat.open_chat()
    h.check("charla de un modelo que no está: abre y avisa", "ya no está instalado" in h.tagged("meta"))
    h.check("charla vieja: conserva el modelo elegido y restaura prompt y contexto",
            chat.model_var.get() == before and chat.system_prompt == "Sos un pirata."
            and chat.ctx_var.get() == "4096")
    chat.system_prompt = app.settings["system_prompt"]
    chat.ctx_var.set("2048")
    h.next()


def benchmark_blocks_chat():
    app.benchmark_running = True
    chat.new_chat()
    chat.input.insert("1.0", "hola")
    chat.send()
    h.check("con un benchmark corriendo no se puede enviar", chat.gen is None and "benchmark" in h.tagged("warn"))
    app.benchmark_running = False
    chat.input.delete("1.0", "end")
    h.next()


def stop_while_loading():  # contra Ollama real
    app.client.chat_stream = real_stream
    app.exclusive_var.set(True)
    h.setup("qwen3:4b", think="auto")
    app.client.unload("qwen3:4b")
    time.sleep(1)
    chat.new_chat()
    chat.input.insert("1.0", "Contame un chiste.")
    chat.send()
    h.root.after(700, lambda: (chat.stop(), h.root.after(4000, after_stop)))


def after_stop():
    h.check("detener mientras carga el modelo: la interfaz queda libre",
            chat.gen is None and str(chat.send_btn.cget("state")) == "normal")
    h.check("detener mientras carga: no queda la pregunta colgada", chat.messages == [])
    chat.think_var.set("no")
    h.send("Decí solo: listo", lambda: (
        h.check("después de detener se puede volver a enviar", bool(h.last_answer()), repr(h.last_answer()[-40:])),
        h.next()), keep_history=True)


sys.exit(h.run([emojis, interleaved, orphan_close_after_markdown, loop_in_thinking, error_midstream,
                huge_response, invalid_values, old_chat_with_missing_model, benchmark_blocks_chat,
                stop_while_loading]))
