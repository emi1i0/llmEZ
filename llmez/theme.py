"""Tema oscuro para ttk."""
import tkinter.font as tkfont
from tkinter import ttk

from . import config as C


def apply(root):
    base = tkfont.nametofont("TkDefaultFont")
    base.configure(size=10)
    family = base.actual("family")

    root.configure(bg=C.BG)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=C.BG, foreground=C.FG, fieldbackground=C.FIELD,
                    bordercolor=C.BORDER, darkcolor=C.BG, lightcolor=C.BG,
                    troughcolor=C.PANEL, focuscolor=C.ACCENT,
                    selectbackground=C.ACCENT_DIM, selectforeground=C.FG,
                    insertcolor=C.FG, arrowcolor=C.FG)
    style.configure("TFrame", background=C.BG)
    style.configure("TLabel", background=C.BG, foreground=C.FG)
    style.configure("Muted.TLabel", foreground=C.MUTED)
    style.configure("Warn.TLabel", foreground=C.WARN_COLOR)
    style.configure("Ok.TLabel", foreground=C.GPU_COLOR)
    style.configure("Bad.TLabel", foreground=C.ERROR_COLOR)
    style.configure("Section.TLabel", foreground=C.FG, font=(family, 10, "bold"))
    style.configure("Value.TLabel", foreground=C.FG, font=(family, 10, "bold"))

    style.configure("TButton", background=C.PANEL, foreground=C.FG, padding=(10, 4),
                    bordercolor=C.BORDER)
    style.map("TButton",
              background=[("disabled", C.BG), ("pressed", C.FIELD), ("active", C.FIELD)],
              foreground=[("disabled", C.MUTED)])
    style.configure("Accent.TButton", background=C.ACCENT_DIM)
    style.map("Accent.TButton",
              background=[("disabled", C.BG), ("pressed", C.ACCENT), ("active", C.ACCENT)])

    for widget in ("TCombobox", "TSpinbox", "TEntry"):
        style.configure(widget, fieldbackground=C.FIELD, background=C.PANEL,
                        foreground=C.FG, arrowcolor=C.FG, bordercolor=C.BORDER)
        style.map(widget,
                  fieldbackground=[("readonly", C.FIELD), ("disabled", C.BG)],
                  foreground=[("disabled", C.MUTED)],
                  selectbackground=[("readonly", C.FIELD)],
                  selectforeground=[("readonly", C.FG)])
    root.option_add("*TCombobox*Listbox.background", C.FIELD)
    root.option_add("*TCombobox*Listbox.foreground", C.FG)
    root.option_add("*TCombobox*Listbox.selectBackground", C.ACCENT_DIM)
    root.option_add("*TCombobox*Listbox.selectForeground", C.FG)

    style.configure("TCheckbutton", background=C.BG, foreground=C.FG,
                    indicatorbackground=C.FIELD, indicatorforeground=C.FG)
    style.map("TCheckbutton",
              background=[("active", C.BG)],
              foreground=[("disabled", C.MUTED)],
              indicatorbackground=[("selected", C.ACCENT_DIM), ("disabled", C.BG)])

    style.configure("Treeview",background=C.PANEL, fieldbackground=C.PANEL,
                    foreground=C.FG, borderwidth=0, rowheight=22)
    style.configure("Treeview.Heading", background=C.FIELD, foreground=C.MUTED,
                    relief="flat", borderwidth=0)
    style.map("Treeview.Heading", background=[("active", C.FIELD)])
    style.map("Treeview", background=[("selected", C.ACCENT_DIM)])

    style.configure("TScrollbar", background=C.PANEL, troughcolor=C.BG,
                    bordercolor=C.BG, arrowcolor=C.MUTED)
    style.map("TScrollbar", background=[("active", C.FIELD)])
    style.configure("TProgressbar", background=C.ACCENT, troughcolor=C.FIELD,
                    bordercolor=C.BORDER, lightcolor=C.ACCENT, darkcolor=C.ACCENT)
    style.configure("TSeparator", background=C.BORDER)
    style.configure("TPanedwindow", background=C.BG)
    style.configure("Sash", sashthickness=6, gripcount=0, background=C.BORDER)
    return family
