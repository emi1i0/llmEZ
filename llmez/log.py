"""Registro en ~/.local/state/llmez/llmez.log (rota a 1 MB, guarda 3 archivos).

Sirve para diagnosticar después: cómo terminó cada respuesta y por qué se
cortó, y los errores internos que de otro modo solo verían la terminal.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

log = logging.getLogger("llmez")


def log_path():
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, "llmez", "llmez.log")


def setup():
    """Configura el archivo de registro. Si no se puede escribir, la app sigue igual."""
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    try:
        path = log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
