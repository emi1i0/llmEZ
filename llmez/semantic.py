"""Búsqueda por significado (embeddings) para combinar con BM25.

BM25 solo encuentra páginas que comparten palabras con la pregunta: "¿Cuánto
tiempo lleva recibirse?" no encuentra la página que dice "Duración de la
carrera: 5 años". Los embeddings comparan significados. Medido con la
Ordenanza 1877: la página correcta pasó del puesto 22 al 1.

Detalles que salieron de probar en una GPU de 4 GB:
- nomic-embed-text (entrenado en inglés) funcionó peor que BM25 en castellano;
  qwen3-embedding:0.6b (multilingüe) funciona bien.
- Indexar en GPU es ~4,5 veces más rápido (60 s contra 271 s para 75 páginas),
  pero si hay otro modelo en la VRAM, Ollama reserva igual 1,4 GB de memoria de
  trabajo, se queda sin memoria y el proceso se cae. Por eso se usa la GPU solo
  si está libre y, si falla, se sigue en CPU.
- Las preguntas se convierten en CPU (~110 ms): no desplazan al modelo de chat.
"""
import hashlib
import json
import math
import os
import time

from .ollama_client import OllamaError

EMBED_MODEL = "qwen3-embedding:0.6b"
QUERY_PREFIX = ("Instruct: Dada una pregunta, encontrá los pasajes de un documento que "
                "la responden\nQuery: ")
BATCH = 8
CPU = {"num_gpu": 0}
QUERY_KEEP_ALIVE = "30m"


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "llmez", "embeddings")


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def is_embedding_only(name):
    """Los modelos de embeddings no sirven para chatear (no generan texto)."""
    return "embed" in name.lower()


class SemanticIndex:
    """Vectores de cada fragmento del documento, calculados una vez y guardados en disco."""

    def __init__(self, client, texts, model=EMBED_MODEL):
        self.client = client
        self.texts = list(texts)
        self.model = model
        digest = hashlib.sha256((model + "\0" + "\x1e".join(self.texts)).encode("utf-8")).hexdigest()
        self.path = os.path.join(cache_dir(), digest[:32] + ".json")
        self.vectors = None
        self.done = 0
        self.device = None  # "GPU" / "CPU" / "caché"
        self.prefer_cpu = False  # lo pide el chat para cargar su modelo en la GPU

    @property
    def ready(self):
        return self.vectors is not None

    def load_cached(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") == self.model and len(data.get("vectors", [])) == len(self.texts):
                self.vectors = data["vectors"]
                self.done, self.device = len(self.texts), "caché"
                return True
        except (OSError, ValueError):
            pass
        return False

    def build(self, stop_event, progress=None):
        """(Hilo de trabajo) Calcula los vectores. Lanza OllamaError si no puede."""
        try:
            busy = any((m.get("size_vram") or 0) > 0 for m in self.client.running())
        except OllamaError:
            busy = True
        options = dict(CPU) if busy else {}
        self.device = "CPU" if busy else "GPU"
        vectors = []
        while len(vectors) < len(self.texts):
            if stop_event.is_set():
                return False
            if self.prefer_cpu and self.device == "GPU":
                options, self.device = dict(CPU), "CPU"
            batch = self.texts[len(vectors):len(vectors) + BATCH]
            last = len(vectors) + len(batch) >= len(self.texts)
            try:
                # num_ctx 2048 alcanza para una página y achica la memoria de trabajo.
                out = self.client.embed(self.model, batch, options=dict(options, num_ctx=2048),
                                        keep_alive=0 if last and self.device == "GPU" else None)
            except OllamaError:
                if self.device == "GPU":  # p. ej. se cargó un modelo de chat: seguir en CPU
                    options, self.device = dict(CPU), "CPU"
                    continue
                raise
            vectors += [[round(x, 5) for x in _normalize(v)] for v in out]
            self.done = len(vectors)
            if progress:
                progress(self.done, len(self.texts), self.device)
        self.vectors = vectors
        try:
            os.makedirs(cache_dir(), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"model": self.model, "vectors": vectors}, f)
            os.replace(tmp, self.path)
        except OSError:
            pass  # sin caché: se recalcula la próxima vez
        return True

    def release_gpu(self, wait_s=20):
        """(Hilo de trabajo) Pasa la indexación a CPU y espera a que la VRAM quede libre.

        Si el modelo de chat se carga mientras éste ocupa la GPU, en 4 GB no entran
        los dos: Ollama se queda sin memoria y el proceso de embeddings se cae."""
        self.prefer_cpu = True
        try:
            self.client.unload(self.model)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:  # el lote en curso termina antes de descargar
                if not any(m["name"] == self.model and (m.get("size_vram") or 0) > 0
                           for m in self.client.running()):
                    return
                time.sleep(0.25)
        except OllamaError:
            pass

    def scores(self, query, timeout=8):
        """Similitud de la pregunta con cada fragmento (o None si no se puede)."""
        if not self.ready:
            return None
        try:
            q = self.client.embed(self.model, [QUERY_PREFIX + query], options=dict(CPU, num_ctx=512),
                                  keep_alive=QUERY_KEEP_ALIVE, timeout=timeout)[0]
        except (OllamaError, IndexError):
            return None
        q = _normalize(q)
        return [sum(a * b for a, b in zip(q, v)) for v in self.vectors]

    def warm_up(self):
        """Carga el modelo en CPU para que la primera pregunta no espere."""
        self.scores("hola", timeout=60)
