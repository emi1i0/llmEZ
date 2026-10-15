"""Detecta cuando un modelo entra en bucle y repite lo mismo sin parar.

Dos señales, pensadas para que el texto normal (listas, código, estribillos
cortos) no dispare falsos positivos:

1. Cola periódica: el final del texto es el mismo fragmento repetido muchas
   veces seguidas ("no no no no…", o un párrafo entero una y otra vez).
2. Línea larga repetida: una línea de 60+ caracteres aparece por tercera vez
   (qwen3:0.6b repite párrafos enteros con secciones nuevas en el medio).
"""

MIN_LINE = 60  # largo mínimo de una línea para contarla como repetida
LINE_REPEATS = 3  # a la tercera aparición se considera bucle
MIN_LOOP_CHARS = 400  # largo mínimo de la parte periódica
MAX_PERIOD = 1500  # fragmento repetido más largo que se busca
CHECK_EVERY = 40  # caracteres nuevos entre chequeos de la cola


def periodic_tail(text, min_chars=MIN_LOOP_CHARS, max_period=MAX_PERIOD):
    """(período, repeticiones) si el final de text es un fragmento repetido; si no None."""
    text = text[-max_period * 4:]  # acota el costo en respuestas largas
    n = len(text)
    for p in range(4, min(max_period, n // 2) + 1):
        # Repeticiones necesarias: más para fragmentos cortos ("ja ja ja" es normal).
        need = max(3 if p >= 40 else 6, -(-min_chars // p))
        if p * need > n:
            continue
        tail = text[-p:]
        if all(text[n - p * (i + 1):n - p * i] == tail for i in range(1, need)):
            k = need
            while p * (k + 1) <= n and text[n - p * (k + 1):n - p * k] == tail:
                k += 1
            return p, k
    return None


def _clean_cut(text, start, end):
    """Índice de corte dentro de la primera copia [start, end]: tras el último fin de
    oración o salto de línea; si no hay, tras el último espacio; si no, en end."""
    window = text[start:end + 2]
    best = -1
    for sep, keep in ((". ", 1), ("! ", 1), ("? ", 1), (".\n", 1), ("\n", 0)):
        i = window.rfind(sep)
        if i >= 0:
            best = max(best, i + keep)
    if best <= 0:
        best = window.rfind(" ", 0, end - start)
    return start + best if best > 0 else end


class RepetitionDetector:
    def __init__(self):
        self.text = ""
        self.reason = None  # descripción del bucle detectado
        self.loop_start = None  # índice donde empieza la parte repetida
        self._line_counts = {}
        self._line_start = 0  # índice en text donde empieza la línea en curso
        self._since_check = 0

    def feed(self, chunk):
        """Agrega texto; devuelve True cuando detecta un bucle (y deja reason)."""
        if self.reason:
            return True
        self.text += chunk
        while True:
            nl = self.text.find("\n", self._line_start)
            if nl < 0:
                break
            if self._count_line(self._line_start, nl):
                return True
            self._line_start = nl + 1
        self._since_check += len(chunk)
        if self._since_check >= CHECK_EVERY:
            self._since_check = 0
            found = periodic_tail(self.text)
            if found:
                p, k = found
                # El corte cae en cualquier punto del ciclo: retroceder hasta donde
                # empezó de verdad la repetición y conservar esa primera copia entera.
                start = len(self.text) - p * k
                while start > 0 and self.text[start - 1] == self.text[start - 1 + p]:
                    start -= 1
                self.loop_start = _clean_cut(self.text, start, start + p)
                self.reason = f"un fragmento de {p} caracteres se repitió {k} veces seguidas"
                return True
        return False

    def _count_line(self, start, end):
        key = " ".join(self.text[start:end].split())
        if len(key) < MIN_LINE:
            return False
        seen = self._line_counts.setdefault(key, [])
        seen.append(start)
        if len(seen) < LINE_REPEATS:
            return False
        self.loop_start = seen[1]  # se conserva la primera aparición
        self.reason = f"la línea «{key[:50]}…» apareció {len(seen)} veces"
        return True

    def trimmed(self):
        """Texto sin la parte repetida (para no meter el bucle en el historial)."""
        if self.loop_start is None:
            return self.text
        return self.text[:self.loop_start].rstrip()
