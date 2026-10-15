"""Guardar, abrir y exportar charlas.

Formato .json propio (versionado) con la configuración usada y cada mensaje,
incluido el razonamiento y las estadísticas, que no se mandan al modelo pero
sirven para revisar después. Exportación a Markdown para leer o compartir.
"""
import json
from datetime import datetime

FORMAT = "llmez-chat"
VERSION = 1


def to_json(messages, settings, system_prompt, file_name=None, file_text=None):
    return {
        "format": FORMAT, "version": VERSION,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": settings, "system_prompt": system_prompt,
        "file": {"name": file_name, "text": file_text} if file_text is not None else None,
        "messages": messages,
    }


def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("no es una charla guardada por llmEZ")
    if data.get("version", 0) > VERSION:
        raise ValueError("la charla es de una versión más nueva de llmEZ")
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not all(
            isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) for m in msgs):
        raise ValueError("la lista de mensajes está dañada")
    return data


def to_markdown(data):
    s = data.get("settings") or {}
    lines = [f"# Charla llmEZ — {data.get('saved_at', '')}", ""]
    conf = ", ".join(f"{k}: {v}" for k, v in s.items() if v not in ("", None))
    if conf:
        lines += [f"*{conf}*", ""]
    if data.get("system_prompt"):
        lines += ["**Prompt de sistema:**", "", "> " + data["system_prompt"].replace("\n", "\n> "), ""]
    if data.get("file"):
        lines += [f"**Archivo en contexto:** `{data['file']['name']}`", ""]
    for m in data["messages"]:
        if m["role"] == "user":
            lines += ["## Vos", "", m["content"], ""]
            continue
        lines += [f"## {m.get('model', 'Asistente')}", ""]
        if m.get("thinking"):
            lines += ["<details><summary>Razonamiento</summary>", "",
                      m["thinking"], "", "</details>", ""]
        lines += [m["content"], ""]
        st = m.get("stats")
        if st and st.get("eval_count"):
            rate = f"{st['tok_s']:.1f} tok/s" if st.get("tok_s") else "n/d"
            lines += [f"*{st['eval_count']} tokens · {rate}*", ""]
    return "\n".join(lines)
