"""Información de un modelo (/api/show + /api/tags) y estimación de memoria.

La memoria que ocupa un modelo cargado es aproximadamente:

    pesos (tamaño del archivo GGUF)
  + caché KV = 2 (K y V) × capas × num_ctx × cabezales_kv × dim_cabezal × 2 bytes (f16)
  + grafo de cómputo (buffers temporales; crece con el modelo)

La fórmula de la caché KV coincide al MB con lo que reporta /api/ps; el grafo
se aproxima como un 10 % de los pesos (medido entre -85 y +350 MB en qwen3).
"""
from . import config as C


class ModelInfo:
    def __init__(self, name, show, file_size):
        self.name = name
        self.file_size = file_size or 0
        details = show.get("details") or {}
        self.family = details.get("family", "?")
        self.parameter_size = details.get("parameter_size", "?")
        self.quantization = details.get("quantization_level", "?")
        self.capabilities = show.get("capabilities") or []
        self.defaults = _parse_parameters(show.get("parameters", ""))

        mi = show.get("model_info") or {}
        arch = mi.get("general.architecture", "")

        def get(key, default=None):
            return mi.get(f"{arch}.{key}", default)

        self.context_max = get("context_length")
        self.layers = get("block_count")
        self.heads = get("attention.head_count")
        self.kv_heads = get("attention.head_count_kv", self.heads)
        emb = get("embedding_length")
        head_dim = emb // self.heads if emb and self.heads else None
        self.key_len = get("attention.key_length", head_dim)
        self.value_len = get("attention.value_length", head_dim)

    @property
    def thinks(self):
        return "thinking" in self.capabilities

    def kv_cache_bytes(self, num_ctx):
        if not all((self.layers, self.kv_heads, self.key_len, self.value_len)):
            return None
        return self.layers * num_ctx * self.kv_heads * (self.key_len + self.value_len) * 2

    def estimate_bytes(self, num_ctx):
        kv = self.kv_cache_bytes(num_ctx)
        if kv is None or not self.file_size:
            return None
        return self.file_size + kv + int(self.file_size * 0.10)

    def fit(self, num_ctx, vram_total_bytes):
        """('entra' | 'justo' | 'no entra' | None, fracción estimada en GPU)."""
        est = self.estimate_bytes(num_ctx)
        if est is None or not vram_total_bytes:
            return None, None
        usable = vram_total_bytes - C.VRAM_RESERVE_BYTES
        if usable <= 0:
            return "no entra", 0.0
        if est <= usable * 0.95:
            return "entra", 1.0
        if est <= usable:
            return "justo", 1.0
        return "no entra", usable / est


def _parse_parameters(text):
    """'temperature  0.6\\ntop_k  20' → {'temperature': '0.6', 'top_k': '20'}."""
    out = {}
    for line in (text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] not in out:
            out[parts[0]] = parts[1].strip().strip('"')
    return out


def fmt_bytes(n):
    if n is None:
        return "n/d"
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
