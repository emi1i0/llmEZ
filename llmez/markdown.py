"""Markdown mínimo → lista de (texto, tags) para insertar en un tk.Text.

Cubre lo que suelen escribir los LLM: títulos, negrita, cursiva, código en
línea, bloques de código, listas, citas, separadores y tablas (en monoespaciado
para que las columnas se alineen). No pretende ser CommonMark.
"""
import re

_INLINE = re.compile(
    r"(?P<esc>\\[\\`*_{}\[\]()#+\-.!~|>])"                       # \* literal
    r"|(?P<code>`[^`\n]+`)"                                         # `código`
    r"|(?P<link>\[(?P<ltext>[^\]\n]+)\]\((?P<lurl>[^)\s]+)\))"       # [texto](url)
    r"|(?P<bi>\*\*\*(?!\s)[^*\n]+?(?<!\s)\*\*\*)"                    # ***negrita cursiva***
    r"|(?P<b>\*\*(?!\s)(?:[^*\n]|\*(?!\*))+?(?<!\s)\*\*)"             # **negrita (*anidable*)**
    r"|(?P<b2>(?<!\w)__(?!\s)[^_\n]+?(?<!\s)__(?!\w))"                # __negrita__
    r"|(?P<s>~~(?!\s)[^~\n]+?(?<!\s)~~)"                            # ~~tachado~~
    r"|(?P<i>(?<![\w*])\*(?![\s*])(?:[^*\n]|\*\*[^*\n]+?\*\*)+?(?<![\s*\\])\*(?![\w*]))"  # *cursiva (**anidable**)*
    r"|(?P<i2>(?<![\w_])_(?![\s_])[^_\n]+?(?<![\s_])_(?![\w_]))"      # _cursiva_ (no snake_case)
)
_IDENT = re.compile(r"\w+")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP = re.compile(r"^:?-{2,}:?$")


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(text):
    return "".join(t for t, _ in inline(text))


def table(lines):
    """Tabla markdown → filas alineadas en monoespaciado (encabezado en negrita)."""
    rows = [_cells(l) for l in lines]
    header = len(rows) > 1 and all(_TABLE_SEP.match(c) for c in rows[1] if c)
    if header:
        rows.pop(1)
    rows = [[_plain(c) for c in r] for r in rows if any(r)]
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(ncols)]
    runs = []
    for n, r in enumerate(rows):
        text = "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
        bold = header and n == 0
        runs.append((text + "\n", ("md_table", "md_table_head") if bold else ("md_table",)))
        if bold:
            runs.append(("─" * (sum(widths) + 2 * (ncols - 1)) + "\n", ("md_table", "md_hr")))
    return runs


def inline(text, base=(), bold=False, italic=False):
    """Formato dentro de una línea. Negrita y cursiva se pueden anidar."""
    emph = ("md_bold_italic",) if bold and italic else ("md_bold",) if bold else \
        ("md_italic",) if italic else ()
    runs = []
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], base + emph))
        kind = m.lastgroup if m.lastgroup not in ("ltext", "lurl") else "link"
        tok = m.group(0)
        if kind == "esc":
            runs.append((tok[1], base + emph))
        elif kind == "code":
            runs.append((tok[1:-1], base + ("md_code",)))
        elif kind == "link":
            runs += inline(m.group("ltext"), base + ("md_link",), bold, italic)
            runs.append((f" ({m.group('lurl')})", base + ("md_url",)))
        elif kind == "bi":
            runs += inline(tok[3:-3], base, True, True)
        elif kind == "b2" and _IDENT.fullmatch(tok[2:-2]):
            runs.append((tok, base + emph))  # __init__ es un nombre, no negrita
        elif kind in ("b", "b2"):
            runs += inline(tok[2:-2], base, True, italic)
        elif kind == "s":
            runs += inline(tok[2:-2], base + ("md_strike",), bold, italic)
        else:
            runs += inline(tok[1:-1], base, bold, True)
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], base + emph))
    return runs


def parse(text):
    runs = []
    code = None  # líneas del bloque de código abierto
    fence_indent = 0
    rows = []  # líneas de la tabla en curso
    for line in text.split("\n"):
        if code is None and line.lstrip().startswith("|"):
            rows.append(line)
            continue
        if rows:
            runs += table(rows)
            rows = []
        if _FENCE.match(line):
            if code is None:
                code = []
                fence_indent = len(line) - len(line.lstrip(" "))
            else:
                runs.append(("\n".join(code) + "\n", ("md_codeblock",)))
                code = None
            continue
        if code is not None:
            # Un bloque con sangría (p. ej. dentro de una lista) la saca de cada línea.
            strip = min(fence_indent, len(line) - len(line.lstrip(" ")))
            code.append(line[strip:])
            continue

        m = _HEADING.match(line)
        if m:
            level = min(len(m.group(1)), 3)
            runs += inline(m.group(2), (f"md_h{level}",))
            runs.append(("\n", (f"md_h{level}",)))
            continue
        if _HR.match(line):
            runs.append(("─" * 40 + "\n", ("md_hr",)))
            continue
        m = _LIST.match(line)
        if m:
            depth = min(len(m.group(1).expandtabs(4)) // 2, 3)
            marker = "•" if m.group(2) in "-*+" else m.group(2)
            tag = (f"md_list{depth}",)
            runs.append((f"{marker} ", tag))
            runs += inline(m.group(3), tag)
            runs.append(("\n", tag))
            continue
        if line.lstrip().startswith(">"):
            body = line.lstrip()[1:].lstrip()
            runs += inline(body, ("md_quote",))
            runs.append(("\n", ("md_quote",)))
            continue
        runs += inline(line)
        runs.append(("\n", ()))

    if rows:
        runs += table(rows)
    if code is not None:  # bloque sin cerrar (respuesta cortada)
        runs.append(("\n".join(code) + "\n", ("md_codeblock",)))
    # Sacar el salto de línea final que agrega el split.
    while runs and runs[-1][0].endswith("\n"):
        txt, tags = runs.pop()
        txt = txt[:-1]
        if txt:
            runs.append((txt, tags))
            break
    return runs


def configure_tags(widget, family, mono_family, size, fg, muted, code_bg, accent):
    """Define en el tk.Text los estilos que usa parse()."""
    widget.tag_configure("md_bold", font=(family, size, "bold"))
    widget.tag_configure("md_italic", font=(family, size, "italic"))
    widget.tag_configure("md_bold_italic", font=(family, size, "bold italic"))
    widget.tag_configure("md_code", font=(mono_family, size - 1), background=code_bg)
    widget.tag_configure("md_codeblock", font=(mono_family, size - 1), background=code_bg,
                         lmargin1=12, lmargin2=12, spacing1=1, spacing3=1)
    widget.tag_configure("md_table", font=(mono_family, size - 1))
    widget.tag_configure("md_table_head", font=(mono_family, size - 1, "bold"))
    widget.tag_configure("md_h1", font=(family, size + 4, "bold"), spacing1=8, spacing3=4)
    widget.tag_configure("md_h2", font=(family, size + 2, "bold"), spacing1=6, spacing3=3)
    widget.tag_configure("md_h3", font=(family, size + 1, "bold"), spacing1=4, spacing3=2)
    widget.tag_configure("md_hr", foreground=muted)
    widget.tag_configure("md_strike", overstrike=True)
    widget.tag_configure("md_link", foreground=accent, underline=True)
    widget.tag_configure("md_url", foreground=muted, font=(family, size - 2))
    widget.tag_configure("md_quote", foreground=muted, lmargin1=14, lmargin2=14)
    for depth in range(4):
        indent = 8 + depth * 18
        widget.tag_configure(f"md_list{depth}", lmargin1=indent, lmargin2=indent + 14)
    # Los estilos de código y títulos tienen prioridad sobre negrita/cursiva.
    for tag in ("md_code", "md_h1", "md_h2", "md_h3", "md_table_head"):
        widget.tag_raise(tag)
