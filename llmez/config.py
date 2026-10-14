"""Configuración general: URL de Ollama, prompts y colores del tema."""
import os


def _ollama_url():
    """Respeta OLLAMA_HOST si está definida (igual que el cliente oficial)."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return "http://localhost:11434"
    if "://" not in host:
        host = "http://" + host
    if host.count(":") < 2:  # sin puerto explícito
        host += ":11434"
    return host.replace("0.0.0.0", "localhost").rstrip("/")


OLLAMA_URL = _ollama_url()

DEFAULT_NUM_CTX = 2048
NUM_CTX_CHOICES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
METRICS_INTERVAL_S = 1.0
HISTORY_POINTS = 60  # último minuto a 1 muestra/s
# Descargar los demás modelos antes de cargar uno nuevo (4 GB de VRAM no
# alcanzan para qwen3:4b + otro modelo; terminaría repartido con la CPU).
EXCLUSIVE_VRAM = True

DEFAULT_SYSTEM_PROMPT = (
    "Sos un asistente útil y claro. Respondé SIEMPRE en español, aunque la "
    "pregunta o el material adjunto estén en otro idioma. Si razonás antes de "
    "responder, razoná también en español."
)

FILE_PROMPT_TEMPLATE = (
    "\n\nEl usuario adjuntó el archivo «{name}». Usalo como fuente principal "
    "para responder sus preguntas; si la respuesta no está en el archivo, decilo."
    "\n\n<archivo nombre=\"{name}\">\n{text}\n</archivo>"
)

MAX_FILE_BYTES = 2 * 1024 * 1024

# --- Tema oscuro ---
BG = "#1d1f24"
PANEL = "#25282f"
FIELD = "#2e323b"
BORDER = "#3a3f4a"
GRID = "#363a44"
FG = "#e4e6eb"
MUTED = "#8b919c"
ACCENT = "#4c8dff"
ACCENT_DIM = "#2f5aa8"

USER_COLOR = "#7cc4ff"
ASSISTANT_COLOR = "#8fdc8f"
THINKING_COLOR = "#8b919c"
WARN_COLOR = "#e8b452"
ERROR_COLOR = "#ff6b6b"

# Series de los gráficos
CPU_COLOR = "#4c8dff"
CORE_COLOR = "#6fa3ff"
RAM_COLOR = "#b48cff"
GPU_COLOR = "#5fd38a"
VRAM_COLOR = "#f0a35e"
TOKS_COLOR = "#f2d15c"
