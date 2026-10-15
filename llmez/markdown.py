"""Markdown mínimo → lista de (texto, tags) para insertar en un tk.Text.

Cubre lo que suelen escribir los LLM: títulos, negrita, cursiva, código en
línea, bloques de código, listas, citas, separadores y tablas (en monoespaciado
para que las columnas se alineen). No pretende ser CommonMark.
"""
import re

_INLINE = re.compile(
    r"(`[^`\n]+`"                                   # `código`
    r"|\*\*\*(?!\s)[^*\n]+?(?<!\s)\*\*\*"           # ***negrita cursiva***
    r"|\*\*(?!\s)[^*\n]+?(?<!\s)\*\*"               # **negrita**
    r"|__(?!\s)[^_\n]+?(?<!\s)__"                   # __negrita__
    r"|(?<![\w*])\*(?![\s*])[^*\n]+?(?<![\s*])\*(?![\w*])"  # *cursiva*
    r"|(?<![\w_])_(?![\s_])[^_\n]+?(?<![\s_])_(?![\w_]))"   # _cursiva_ (no snake_case)
)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_HR = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def inline(text, base=()):
    runs = []
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], base))
        tok = m.group(0)
        if tok.startswith("`"):
            runs.append((tok[1:-1], base + ("md_code",)))
        elif tok.startswith("***"):
            runs.append((tok[3:-3], base + ("md_bold_italic",)))
        elif tok.startswith(("**", "__")):
            runs.append((tok[2:-2], base + ("md_bold",)))
        else:
            runs.append((tok[1:-1], base + ("md_italic",)))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], base))
    return runs


def parse(text):
    runs = []
    code = None  # líneas del bloque de código abierto
    for line in text.split("\n"):
        if _FENCE.match(line):
            if code is None:
                code = []
            else:
                runs.append(("\n".join(code) + "\n", ("md_codeblock",)))
                code = None
            continue
        if code is not None:
            code.append(line)
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
        if line.lstrip().startswith("|"):
            runs.append((line.strip() + "\n", ("md_table",)))
            continue
        if line.lstrip().startswith(">"):
            body = line.lstrip()[1:].lstrip()
            runs += inline(body, ("md_quote",))
            runs.append(("\n", ("md_quote",)))
            continue
        runs += inline(line)
        runs.append(("\n", ()))

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
    widget.tag_configure("md_h1", font=(family, size + 4, "bold"), spacing1=8, spacing3=4)
    widget.tag_configure("md_h2", font=(family, size + 2, "bold"), spacing1=6, spacing3=3)
    widget.tag_configure("md_h3", font=(family, size + 1, "bold"), spacing1=4, spacing3=2)
    widget.tag_configure("md_hr", foreground=muted)
    widget.tag_configure("md_quote", foreground=muted, lmargin1=14, lmargin2=14)
    for depth in range(4):
        indent = 8 + depth * 18
        widget.tag_configure(f"md_list{depth}", lmargin1=indent, lmargin2=indent + 14)
    # Los estilos de código y títulos tienen prioridad sobre negrita/cursiva.
    for tag in ("md_code", "md_h1", "md_h2", "md_h3"):
        widget.tag_raise(tag)
