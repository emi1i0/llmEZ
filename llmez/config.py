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

# VRAM que Ollama no llega a usar para el modelo (contexto CUDA, reservas).
# Calibrado en una GTX 960M de 4 GB: qwen3:4b entra al 100 % ocupando
# 2,97 GB y con 3,23 GB ya queda al 96 % → el techo práctico ronda 3,1 GB.
VRAM_RESERVE_BYTES = 1_150_000_000

# Benchmark
BENCH_PROMPT = "Explicá en tres oraciones qué es una GPU y por qué sirve para IA."
BENCH_MAX_TOKENS = 128

# Sugerencias para "Bajar modelo…" (chicos, pensados para 4 GB de VRAM).
SUGGESTED_MODELS = ("qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "gemma3:1b", "gemma3:4b",
                    "llama3.2:1b", "llama3.2:3b", "deepseek-r1:1.5b", "smollm2:360m",
                    "qwen3-embedding:0.6b")

DEFAULT_SYSTEM_PROMPT = (
    "Sos un asistente útil y claro. Respondé siempre en español, aunque la "
    "pregunta o el material adjunto estén en otro idioma. Si razonás antes de "
    "responder, razoná también en español."
)

FILE_PROMPT_TEMPLATE = (
    "\n\nEl usuario adjuntó el archivo «{name}». Usalo como fuente principal "
    "para responder sus preguntas; si la respuesta no está en el archivo, decilo. "
    "Si el texto tiene marcas \"--- página N ---\", indicá de qué página sacás cada dato."
    "{note}\n\n<archivo nombre=\"{name}\">\n{text}\n</archivo>"
)
# Si el archivo no entra en el contexto se mandan solo las partes relevantes.
FILE_EXCERPT_NOTE = (" Del archivo (que es largo) se incluyen solo {parts} de {total}, "
                     "elegidas por relevancia para la pregunta; si el dato no aparece, puede "
                     "estar en otra parte del archivo.")
FILE_CONTEXT_SHARE = 0.6  # parte del contexto que puede ocupar el archivo
SEMANTIC_MIN_TOKENS = 1500  # archivos más chicos entran enteros: no hace falta indexarlos

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
