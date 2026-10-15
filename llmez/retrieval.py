"""Búsqueda de las partes relevantes de un documento que no entra en el contexto.

Si el documento completo no entra, Ollama se queda en silencio con el final y
descarta el resto (incluido el prompt de sistema). En ese caso se mandan solo
las páginas (o fragmentos) más relevantes para la pregunta, elegidas con BM25:
un buscador por palabras clave clásico. Si hay embeddings (semantic.py), las dos
puntuaciones se combinan: cada una tapa los huecos de la otra.
"""
import math
import re
import unicodedata

PAGE_MARK = re.compile(r"^--- página (\d+) ---$", re.M)
# Caracteres por token: ~3,9 en prosa en castellano, pero ~2,8 en páginas con tablas
# y números (medido con qwen3). Se arranca prudente; la app la ajusta con el conteo
# real que informa Ollama después de cada respuesta.
CHARS_PER_TOKEN = 2.8

_STOP = set("""a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con contra cual
cuales cuando de del desde donde dos el ella ellas ellos en entre era es esa ese eso esta estas este esto
estos fue fueron ha han hasta hay la las le les lo los mas me mi muy no nos o otra otro para pero poco por
que quien se segun ser si sin sobre son su sus tambien tan tiene todo todos tu un una unas uno unos y ya
cuanto cuanta cuantos cuantas cual dice decir hace tiene tienen""".split())


def estimate_tokens(text, chars_per_token=CHARS_PER_TOKEN):
    return int(len(text) / chars_per_token) + 1


def _terms(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    out = []
    for w in re.findall(r"[a-z0-9]+", text):
        if w in _STOP or (len(w) < 3 and not w.isdigit()):
            continue
        out.append(w if w.isdigit() else w[:6])  # "asignaturas" y "asignatura" → "asigna"
    return out


def _pack(text, max_chars):
    """Parte un texto en trozos de hasta max_chars sin cortar párrafos ni líneas si se puede.

    Unidades: párrafos; los muy largos (texto sin líneas en blanco) se parten por líneas
    y, si una línea sola es enorme, por caracteres."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= max_chars:
            units.append(para)
            continue
        for line in para.split("\n"):
            units += [line[i:i + max_chars] for i in range(0, max(len(line), 1), max_chars)]
    pieces, buf = [], ""
    for unit in units:
        if buf and len(buf) + len(unit) > max_chars:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}\n{unit}" if buf else unit
    if buf.strip():
        pieces.append(buf)
    return [p.strip() for p in pieces if p.strip()]


def split_chunks(text, max_chars=800):
    """[(etiqueta, texto)]: fragmentos de hasta ~max_chars (800: entran aun con el presupuesto
    mínimo de la app, 300 tokens ≈ 840 caracteres).

    Con marcas de página, cada página se parte en fragmentos que conservan la marca
    ("--- página 19 ---") y la etiqueta "página 19". Fragmentos chicos importan con
    poco contexto: con páginas enteras, una página relevante pero larga no entraba
    y se mandaba otra más chica y menos relevante (pasó con qwen3:4b y contexto 2048)."""
    marks = list(PAGE_MARK.finditer(text))
    if marks:
        chunks = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()
            for piece in _pack(body, max_chars) or [""]:
                chunks.append((f"página {m.group(1)}", f"{m.group(0)}\n{piece}".strip()))
        return chunks
    return [(f"fragmento {i}", c) for i, c in enumerate(_pack(text, max_chars), 1)]


class Index:
    def __init__(self, text, k1=1.5, b=0.75):
        self.chunks = split_chunks(text)
        self.units = len({label for label, _ in self.chunks})  # páginas (o fragmentos) distintos
        self.k1, self.b = k1, b
        self.docs = [_terms(t) for _, t in self.chunks]
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = _terms(query)
        out = []
        for d in self.docs:
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out

    def select(self, query, budget_tokens, min_ratio=0.25, chars_per_token=CHARS_PER_TOKEN,
               dense=None):
        """Los fragmentos más relevantes que entran en budget_tokens, en orden del documento.

        Solo entran los que tienen al menos min_ratio del puntaje del mejor: llenar el
        presupuesto con páginas poco relevantes confunde a los modelos chicos.
        `dense`: similitudes por embeddings (una por fragmento) para combinar con BM25.
        Devuelve (texto, etiquetas elegidas)."""
        scores = combine(self.scores(query), dense)
        best = max(scores, default=0)
        if best == 0:
            order = list(range(len(self.chunks)))  # nada coincide: el principio del documento
        else:
            order = sorted((i for i in range(len(self.chunks)) if scores[i] >= best * min_ratio),
                           key=lambda i: -scores[i])
        chosen, used = [], 0
        for i in order:
            cost = estimate_tokens(self.chunks[i][1], chars_per_token)
            if used + cost > budget_tokens:
                continue
            chosen.append(i)
            used += cost
        if not chosen and order:  # ni el mejor entra entero: va recortado, nunca nada
            i = order[0]
            return self.chunks[i][1][:int(budget_tokens * chars_per_token)], [self.chunks[i][0]]
        chosen.sort()
        return "\n\n".join(self.chunks[i][1] for i in chosen), [self.chunks[i][0] for i in chosen]


def _rescale(values):
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values] if hi > lo else [0.0] * len(values)


def combine(bm25, dense, weight=0.7):
    """Mezcla BM25 y embeddings, cada uno llevado a 0..1 (escalas muy distintas).

    Medido con la Ordenanza 1877 (12 preguntas, fragmentos de 800 caracteres): con el
    70 % para embeddings la respuesta llegó al modelo en 12/12 con 600 a 4000 tokens
    de presupuesto; con 50/50, 11/12 hasta 1000 tokens (la pregunta con sinónimos:
    embeddings la ponen 1.ª y BM25 59.ª); RRF k=60, 9/12; BM25 solo, 9/12."""
    if not dense or len(dense) != len(bm25):
        return bm25
    if max(bm25) == 0:
        return _rescale(dense)
    return [(1 - weight) * b + weight * d for b, d in zip(_rescale(bm25), _rescale(dense))]


def compact_labels(labels):
    """['página 2', 'página 3', 'página 4', 'página 9'] → 'páginas 2–4, 9'."""
    if not labels:
        return ""
    kind = labels[0].split()[0]
    nums = sorted({int(l.split()[1]) for l in labels})  # una página puede venir en varios fragmentos
    ranges = []
    start = prev = nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}–{prev}")
        if n is not None:
            start = prev = n
    return f"{kind}{'s' if len(nums) > 1 else ''} {', '.join(ranges)}"
