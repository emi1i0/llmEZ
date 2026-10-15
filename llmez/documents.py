"""Lectura de archivos para inyectar en el prompt: texto plano y PDF.

El PDF se convierte con `pdftotext` (poppler-utils, viene con la mayoría de
las distros) para no sumar dependencias de pip. Cada página queda marcada
("--- página N ---") para que el modelo pueda citar dónde está cada dato.
"""
import os
import re
import shutil
import subprocess

from . import config as C

MAX_PDF_BYTES = 100 * 1024 * 1024
MIN_CHARS_PER_PAGE = 20  # menos que esto: página sin texto (imagen escaneada)


def is_pdf(path):
    if path.lower().endswith(".pdf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def read_document(path):
    """Devuelve (texto, descripción). Lanza ValueError con un mensaje para el usuario."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise ValueError(str(e)) from None
    if is_pdf(path):
        if size > MAX_PDF_BYTES:
            raise ValueError("el PDF pesa más de 100 MB")
        return read_pdf(path)
    if size > C.MAX_FILE_BYTES:
        raise ValueError("pesa más de 2 MB; es demasiado para el contexto de un modelo local")
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw:
        raise ValueError("parece un archivo binario, no texto (se aceptan texto plano y PDF)")
    try:
        return raw.decode("utf-8"), "texto"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "texto (latin-1)"


def read_pdf(path):
    exe = shutil.which("pdftotext")
    if not exe:
        raise ValueError("para leer PDF hace falta pdftotext: sudo apt install poppler-utils")
    try:
        proc = subprocess.run([exe, "-enc", "UTF-8", path, "-"], capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise ValueError("pdftotext tardó más de 2 minutos") from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        if "password" in err.lower():
            raise ValueError("el PDF está protegido con contraseña")
        raise ValueError("pdftotext no pudo leerlo: " + (err.strip().splitlines() or ["?"])[-1])
    raw = proc.stdout.decode("utf-8", "replace")
    pages = raw.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # pdftotext termina con un salto de página
    if not pages:
        raise ValueError("el PDF no tiene páginas")
    empty = [i for i, p in enumerate(pages, 1) if len("".join(p.split())) < MIN_CHARS_PER_PAGE]
    if len(empty) == len(pages):
        raise ValueError("no tiene texto extraíble: parece escaneado (son imágenes) y haría falta OCR")
    text = "\n\n".join(f"--- página {i} ---\n{_clean(p)}" for i, p in enumerate(pages, 1))
    if len(text) > C.MAX_FILE_BYTES:
        raise ValueError(f"el texto extraído ({len(text) // 1000} kB) es demasiado para el contexto")
    info = f"PDF de {len(pages)} página{'s' if len(pages) != 1 else ''}"
    if empty:
        listed = ", ".join(map(str, empty[:8])) + ("…" if len(empty) > 8 else "")
        info += f"; sin texto en {len(empty)} (¿escaneadas?): {listed}"
    return text, info


def _clean(page):
    page = re.sub(r"(\w)-\n(\w)", r"\1\2", page)  # palabras cortadas con guion al final de línea
    page = re.sub(r"[ \t]+\n", "\n", page)
    page = re.sub(r"\n{3,}", "\n\n", page)
    return page.strip()
