# llmEZ — contexto para Claude

App de escritorio en Python/Tkinter para experimentar con LLMs locales en Ollama: chat con streaming a la izquierda, monitor de CPU/RAM/GPU a la derecha. Proyecto personal de aprendizaje de Emilio. Repo público: github.com/emi1i0/llmEZ (rama `master`, licencia MIT).

## Cómo trabajar acá

- **Idioma:** castellano rioplatense (voseo) en las respuestas, en los textos de la interfaz, en los comentarios y en los commits.
- **Evidencia antes que opinión:** medir contra el Ollama real y dar números. Si algo falla, mostrar el error tal cual; nunca esconderlo ni dar algo por bueno sin verificarlo.
- **Separar siempre tres causas de falla:** la app, el modelo (un modelo chico que se equivoca) o la prueba (un chequeo mal hecho). Antes de culpar a la app, revisar la respuesta completa y el log de Ollama.
- Cuando Emilio dice "seguí testeando", buscar huecos nuevos (casos borde, robustez, documentos reales); no repetir lo que ya está cubierto.
- **Pedir OK antes de:** bajar modelos grandes, borrar modelos o archivos suyos e instalar cosas en el sistema. Los experimentos que bajan algo se cuentan, y si después no sirve se ofrece borrarlo.

## Git

- Identidad del repo (ya configurada): `Emilio Rehwald <140626466+emi1i0@users.noreply.github.com>`. Nunca usar su Gmail en commits: el repo es público.
- **Sin línea `Co-Authored-By`** (lo pidió explícitamente).
- Commitear en local cuando el trabajo está verificado. **Hacer push solo cuando Emilio dice "subilo"**; al terminar, preguntar "¿Lo subo a GitHub?".
- Mensajes de commit: título corto + cuerpo que cuente qué se encontró y cómo se midió.

## Entorno

- Dell Inspiron 7559: i7 de 4 núcleos / 8 hilos, 15,5 GiB de RAM, **GTX 960M de 4 GB** (driver 580, PRIME on-demand), Linux Mint 22.3 (Xfce).
- Python 3.12 con venv en `.venv/`. Ollama 0.40.4 como servicio systemd en `localhost:11434`; log con `journalctl -u ollama`.
- Modelos: qwen3:0.6b, qwen3:1.7b, qwen3:4b, smollm2:135m (chat); qwen3-embedding:0.6b (búsqueda por significado). nomic-embed-text se probó, se descartó y se borró.
- Node v24 está instalado con nvm, solo para su usuario. `pdftotext`, `soffice` y `xdotool` están disponibles. ImageMagick no está.
- **`/tmp` se borra al reiniciar.** Nada que haya que conservar (pruebas, entornos, scripts) puede vivir solo ahí.

## Arquitectura

```
main.py                    punto de entrada
llmez/app.py               ventana, menús, atajos, cola app.post() entre hilos y UI, cierre
llmez/chat_panel.py        chat, streaming, estadísticas, archivos, búsqueda en documentos, free_vram()
llmez/monitor_panel.py     gráficos (ScrollFrame), Ollama ps, estadísticas de la última respuesta
llmez/ollama_client.py     HTTP mínimo con urllib: tags, show, ps, chat/pull (streams), embed, delete, abort
llmez/metrics.py           hilo de muestreo cada 1 s (psutil, nvidia-smi, /api/ps)
llmez/model_info.py        /api/show y estimación de VRAM (pesos + caché KV + grafo)
llmez/think_parser.py      separa <think>…</think> mezclado en content (incluido </think> huérfano)
llmez/repetition.py        detector de bucles (cola periódica / línea larga ×3) y recorte limpio
llmez/markdown.py          markdown → tags de tk.Text (recursivo, tablas alineadas, enlaces)
llmez/documents.py         texto plano y PDF (pdftotext, marcas de página, quita encabezados repetidos)
llmez/retrieval.py         fragmentos de ~800 caracteres con marca de página, BM25, combine() (70 % embeddings)
llmez/semantic.py          SemanticIndex: embeddings con caché en ~/.cache/llmez, GPU/CPU
llmez/conversation.py      guardar/abrir (JSON llmez-chat v1) y exportar a Markdown
llmez/settings.py          ~/.config/llmez/settings.json, cada valor validado por tipo
llmez/log.py               ~/.local/state/llmez/llmez.log (desenlace de cada respuesta, cortes, errores)
llmez/dialogs/             benchmark.py, models.py (bajar/borrar)
llmez/widgets/             line_chart.py (Canvas), scroll_frame.py
```

Reglas que el código ya sigue y hay que mantener:
- **Tk no es thread-safe.** Los hilos nunca tocan widgets: encolan con `app.post(fn, *args)`.
- **Dependencias:** solo `psutil` en pip. Todo lo demás con la biblioteca estándar; `pdftotext` es opcional.
- Una respuesta en curso es un objeto `Generation`. `_on_chunk`/`_emit`/`_finish` comprueban `gen is self.gen` porque una respuesta puede cortarse a mitad de un chunk (Detener, bucle, contexto lleno).
- Lo que se manda a Ollama es solo `{"role", "content"}`. El historial guarda además `thinking`, `model` y `stats`.
- `num_predict` = contexto libre (los tokens se estiman con `_cpt()`, que se ajusta con el conteo real de Ollama).
- Documento que no entra en el contexto: `Index.select()` con BM25, más embeddings si el índice está listo. La app avisa qué páginas mandó.

## Comandos

```bash
.venv/bin/python main.py                                   # correr la app
.venv/bin/python -m unittest discover -s tests -t .        # tests unitarios (~1 s, sin Ollama)
tests/gui/run_all.sh [test_scripted …]                     # pruebas de interfaz (~40 min, Ollama real, GPU)
.venv/bin/pyinstaller --onefile --windowed --name llmEZ \
    --distpath dist --workpath build --specpath build main.py --noconfirm   # ejecutable (13 MB)
```

## Hechos ya medidos (no volver a medir salvo que cambie algo)

- **Velocidad** (contexto 2048, todo en GPU): smollm2 ~70 tok/s · qwen3:0.6b ~38 · qwen3:1.7b ~19 · qwen3:4b ~10.
- **Techo de VRAM:** Ollama no ubica más de **~3,1 GB** (`VRAM_RESERVE_BYTES` = 1,15 GB).
  - qwen3:4b entra al 100 % hasta ~2.560 de contexto; con 4096 (el defecto de Ollama) queda al 90 %, con 8192 al 76 %.
  - La fórmula de la caché KV coincide al MB con `/api/ps`. El estimador predijo 80 % donde el real fue 76 %.
- **qwen3:4b ignora `think:false`:** razona dentro de `content`, sin `<think>` de apertura y con un `</think>` suelto. Con "Pensar: auto" anda bien. 0.6b y 1.7b sí lo respetan. smollm2 da error 400 con `think:true`.
- **Bucles:** temperatura 0 → 3 de 5 pedidos. Con la temperatura del modelo → 1 de 15. `repeat_penalty` 1.15 empeora (3 de 15). Pasarse del contexto también provoca bucles.
- **Semilla 42 + temperatura 0:** la primera respuesta (prompt sin caché) puede diferir; las siguientes son idénticas.
- **Recorte silencioso:** si el prompt supera `num_ctx`, Ollama se queda con los primeros 4 tokens y el final (≤ mitad del contexto) y descarta el resto, incluido el prompt de sistema. En el log: `truncating input prompt limit=… prompt=…`.
- **Tokens:** ~3,9 caracteres por token en prosa en castellano; ~2,8 en páginas con tablas y números.
- **Ordenanza 1877** (`~/Descargas/Ordenanza 1877.pdf`, 75 páginas, ~26.000 tokens): entera → 0/4; con búsqueda → 4/4 en los tres qwen3. Su sello repetido ("PABLO A. HUEL") confundía al modelo con el firmante.
- **Embeddings:**
  - nomic-embed-text es peor que BM25 en castellano.
  - qwen3-embedding:0.6b: el sinónimo "recibirse" pasa del puesto 22 al 1.
- **Unidad de búsqueda:** páginas enteras no sirven con poco contexto, porque una página relevante pero larga no entra y se cuela otra. Con fragmentos de 800 caracteres y 70 % de peso a los embeddings, la respuesta llega al modelo en 12 de 12 preguntas con 600 a 4000 tokens. Con 50/50 eran 11 de 12; con RRF o con BM25 solo, 9 de 12.
  - Indexar 75 páginas: 60–170 s en GPU, 200–320 s en CPU. Cada pregunta en CPU: ~110 ms.

## Trampas conocidas

- **Dos modelos en la GPU de 4 GB:** Ollama reserva igual ~1,4 GB de trabajo y el proceso se cae (`CUDA error: out of memory`, `llama-server terminated`). Por eso `free_vram()` y `SemanticIndex.release_gpu()`.
- Un modelo cargado a medias en GPU se reusa así: hay que descargarlo para que vuelva a entrar entero.
- `free_vram()` no debe descargar modelos con `size_vram == 0` (el de embeddings corre en CPU).
- qwen3:1.7b entiende mal algunas paráfrasis ("recibirse") aunque reciba la página correcta; qwen3:4b no.
- La firma digital de la última página de la ordenanza tiene otra fecha (27/09/2022) que la del documento (15/06/2022). Las dos respuestas son válidas.

## Cómo probar sin repetir errores

- **Las pruebas viven en el repo:** unitarias en `tests/`, de interfaz en `tests/gui/` sobre `common.Harness`. Nunca dejar scripts de prueba solo en el scratchpad.
- Las pruebas de interfaz usan carpetas XDG temporales: nunca tocan la configuración, el registro ni la caché de Emilio.
- **Chequear lo que depende de la app**, no la puntería del modelo: que llegue la página correcta, que el historial sea correcto. Comparar texto con `common.plain()` (sin tildes ni mayúsculas): "Soró" = "SORO".
- **No correr dos pruebas de interfaz a la vez** (compiten por la GPU). Las largas, en segundo plano.
- **Una prueba que no imprime su resumen es una falla.** Nunca filtrar la salida de forma que se escondan errores: guardar la salida completa de cada corrida en un archivo y después filtrar. Ya se perdieron dos fallas intermitentes de `test_ollama` por mirar solo `tail`.
- **Contar caídas de Ollama** después de cada tanda: `journalctl -u ollama --since <hora> | grep -c 'CUDA error: out of memory'`.
- Al medir códigos de salida, no hacerlo a través de un pipe (`$?` sería el de `tail`). Usar `${PIPESTATUS[0]}` o redirigir a un archivo.
- **Revisar visualmente:** captura con `xdotool windowactivate` + `xfce4-screenshooter -w -s archivo.png` y mirarla. Varias fallas de diseño solo se vieron así.
- Antes de commitear, la regresión completa: tests unitarios + `run_all.sh` + regenerar el ejecutable y comprobar que arranca.

## Cierre de cada integración o tanda de pruebas: el informe

Toda integración nueva o tanda de pruebas se cierra con **un único informe** que es a la vez descriptivo (qué se hizo y por qué), recopilatorio (todo lo probado y medido) y técnico (números, causas, commits). Formato por defecto: **`.docx` en `~/Descargas/`**, con el nombre `Informe de <tema> llmEZ - AAAA-MM-DD.docx`, salvo que Emilio pida otro formato o lugar. Hechos hasta ahora: `Informe de pruebas llmEZ.docx` y `Informe de búsqueda por significado llmEZ - 2026-10-15.docx`, los dos en `~/Descargas/`. El venv con python-docx está en `~/.cache/llmez-tools/docxenv`.

Estructura (adaptar al tema; omitir lo que no aplique):
1. **Portada:** tema, período, commit probado, equipo.
2. **Resumen:** qué se logró y los hallazgos principales, en 5–8 puntos con números.
2b. **Guía didáctica** cuando el informe trate un tema técnico nuevo (por ejemplo embeddings, caché KV, cuantización). Va antes de la parte técnica, para alguien que aprende: qué es, para qué sirve y cómo funciona, con analogías y un ejemplo concreto con datos del propio proyecto (p. ej. dos preguntas y la página que encuentra cada búsqueda), qué significan los números que aparecen después (puesto, dimensión, similitud) y un glosario corto. Lenguaje llano; las fórmulas, si hacen falta, después de la intuición.
3. **Contexto y objetivo:** qué problema se atacó y por qué.
4. **Qué se hizo:** cambios en la app, decisiones de diseño y alternativas descartadas, con el motivo medido.
5. **Metodología:** qué pruebas (unitarias, de interfaz, contra Ollama real, documentos reales) y con qué configuración.
6. **Resultados:** tablas con lo medido (antes/después cuando aplique) y capturas de la app.
7. **Comportamiento de los modelos:** lo observado, separado de las fallas de la app.
8. **Fallas encontradas y corregidas:** tabla con #, falla, cómo se detectó y commit.
9. **Limitaciones y pendientes:** incluir lo que no se pudo explicar, sin maquillarlo.
10. **Recomendaciones de uso.**
11. **Anexo:** commits del período y cómo reproducir las pruebas.

Cómo generarlo:
- **python-docx** en un venv fuera de `/tmp` (por ejemplo `~/.cache/llmez-tools/docxenv`), o la librería `docx` de npm.
- Tablas con ancho fijo en la grilla (`tblGrid` + `tblLayout fixed`), porque si no LibreOffice las dibuja con columnas iguales.
- Pie de figura pegado a la imagen (`keep_with_next`).
- **Verificar antes de entregar:** `soffice --headless --convert-to pdf` + `pdftoppm -jpeg -r 60` y mirar cada página.
- Los números salen de las mediciones de la sesión, no de memoria. Si un dato que se informó antes era incorrecto, corregirlo explícitamente.
