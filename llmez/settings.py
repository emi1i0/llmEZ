"""Preferencias persistentes en ~/.config/llmez/settings.json."""
import json
import os
import re

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


_GEOMETRY = re.compile(r"^\d{2,5}x\d{2,5}([+-]-?\d{1,5}[+-]-?\d{1,5})?$")


def _valid(key, value):
    """El valor guardado tiene el mismo tipo que el de por defecto (y es razonable)."""
    if type(value) is not type(DEFAULTS[key]):  # type(): True no cuenta como int
        return False
    if key == "geometry":
        return bool(_GEOMETRY.match(value))
    if key == "num_ctx":
        return 128 <= value <= 1_048_576
    return True


def load(path=None):
    """Devuelve DEFAULTS combinados con lo guardado.

    Un archivo roto o editado a mano no rompe la app: cada valor con tipo o
    formato inválido se reemplaza por el de por defecto.
    """
    data = dict(DEFAULTS)
    try:
        with open(path or settings_path(), encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            data.update({k: v for k, v in saved.items() if k in DEFAULTS and _valid(k, v)})
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
