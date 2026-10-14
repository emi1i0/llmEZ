"""Separa razonamiento <think>…</think> que llega mezclado dentro de `content`.

Algunos modelos (p. ej. qwen3:4b con think:false) ignoran el pedido y
escriben el razonamiento en `content`, a veces sin la etiqueta de apertura y
terminando con un `</think>` "huérfano". El splitter trabaja en streaming:
las etiquetas pueden venir partidas entre chunks.
"""

OPEN = "<think>"
CLOSE = "</think>"


def _partial_tag_len(buf):
    """Largo del sufijo de buf que podría ser el comienzo de una etiqueta."""
    for n in range(min(len(buf), len(CLOSE) - 1), 0, -1):
        tail = buf[-n:]
        if OPEN.startswith(tail) or CLOSE.startswith(tail):
            return n
    return 0


class ThinkTagSplitter:
    """feed() devuelve una lista de eventos:
    ("thinking", texto), ("content", texto) u ("orphan_close", "") cuando
    aparece </think> sin apertura (todo lo anterior era razonamiento).
    """

    def __init__(self):
        self.buf = ""
        self.in_think = False
        self.strip_next = True  # saca los saltos de línea tras una etiqueta

    def _emit(self, out, text):
        kind = "thinking" if self.in_think else "content"
        if self.strip_next:
            text = text.lstrip()
            if not text:
                return
            self.strip_next = False
        out.append((kind, text))

    def feed(self, text):
        self.buf += text
        out = []
        while True:
            i_open = -1 if self.in_think else self.buf.find(OPEN)
            i_close = self.buf.find(CLOSE)
            candidates = [(i, t) for i, t in ((i_open, OPEN), (i_close, CLOSE)) if i >= 0]
            if not candidates:
                keep = _partial_tag_len(self.buf)
                ready = self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                if ready:
                    self._emit(out, ready)
                return out
            i, tag = min(candidates)
            if i > 0:
                self._emit(out, self.buf[:i])
            self.buf = self.buf[i + len(tag):]
            if tag == OPEN:
                self.in_think = True
            elif self.in_think:
                self.in_think = False
            else:
                out.append(("orphan_close", ""))
            self.strip_next = True

    def flush(self):
        out = []
        if self.buf:
            self._emit(out, self.buf)
            self.buf = ""
        return out
