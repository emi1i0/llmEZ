"""Preferencias persistentes en ~/.config/llmez/settings.json."""
import json
import os

from . import config as C

DEFAULTS = {
    "model": "",
    "think": "auto",
    "num_ctx": C.DEFAULT_NUM_CTX,
    "temperature": "",  # vacío = la del modelo
    "seed": "",  # vacío = aleatoria
    "exclusive_vram": C.EXCLUSIVE_VRAM,
    "unload_on_exit": False,
    "loop_guard": True,
    "system_prompt": C.DEFAULT_SYSTEM_PROMPT,
    "geometry": "1400x880",
}


def settings_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "llmez", "settings.json")


def load(path=None):
    """Devuelve DEFAULTS combinados con lo guardado. Un archivo roto no rompe la app."""
    data = dict(DEFAULTS)
    try:
        with open(path or settings_path(), encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            data.update({k: v for k, v in saved.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return data


def save(data, path=None):
    path = path or settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # escritura atómica: nunca queda a medias
        return None
    except OSError as e:
        return str(e)
