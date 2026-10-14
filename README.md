# llmEZ

App de escritorio en Python para experimentar con modelos de lenguaje locales que corren en [Ollama](https://ollama.com): un chat con streaming a la izquierda y un monitor en tiempo real de CPU, RAM, GPU y velocidad de generación a la derecha.

Es un proyecto personal de aprendizaje, pensado para entender qué pasa adentro de la máquina mientras un LLM genera texto en hardware modesto (una notebook con una GTX 960M de 4 GB).

![Captura de llmEZ: chat con razonamiento separado y monitor de recursos](docs/captura.png)

## Qué hace

**Chat**
- Selector de modelo (lee `/api/tags`, con botón para refrescar).
- Respuestas en streaming vía `/api/chat`.
- Razonamiento (`message.thinking`) separado de la respuesta, en gris y cursiva.
- Opción *Pensar*: `auto` (no manda el parámetro), `sí` (`think: true`) o `no` (`think: false`).
- Tamaño de contexto configurable (`options.num_ctx`, 2048 por defecto).
- Cargar un archivo de texto que se inyecta en el prompt de sistema para hacerle preguntas, con aviso si probablemente no entra en el contexto.
- Prompt de sistema editable; el de por defecto pide responder en español.
- Detener la generación a mitad de camino (corta también del lado de Ollama, no solo en la interfaz).
- *Solo un modelo en VRAM*: antes de cargar un modelo, descarga los demás (ver [por qué](#lo-que-aprendí)).
- Botón para descargar todos los modelos de la VRAM (`keep_alive: 0`).
- Al terminar cada respuesta: tokens generados, tok/s, tiempo al primer token y tiempo total.

**Monitor** (gráficos del último minuto, actualizados cada 1 s)
- CPU total y por núcleo.
- RAM usada / total.
- GPU vía `nvidia-smi`: uso, VRAM, temperatura y estado de energía (P0, P8, …).
- Modelos cargados en Ollama (`/api/ps`): tamaño, reparto CPU/GPU (en naranja si no está 100 % en GPU), contexto y cuánto falta para que se descargue solo.
- tok/s en vivo mientras genera y detalle de la última respuesta: tokens del prompt (y cuántos venían de caché), velocidad del prompt, tiempo de carga del modelo y contexto usado.

Si Ollama o `nvidia-smi` no están disponibles, la app no se rompe: muestra "n/d" o un aviso.

## Requisitos

- Python 3.10 o superior con Tkinter (en Debian/Ubuntu/Mint: `sudo apt install python3-tk`).
- Ollama corriendo (por defecto en `http://localhost:11434`; respeta la variable `OLLAMA_HOST`).
- Opcional: GPU NVIDIA con `nvidia-smi` para el monitor de GPU.

La única dependencia de pip es [psutil](https://pypi.org/project/psutil/), para leer CPU y RAM de forma portable. La interfaz es Tkinter, las llamadas HTTP usan `urllib` y los gráficos se dibujan a mano sobre `tk.Canvas` (sin matplotlib).

## Instalación y uso

```bash
git clone https://github.com/emi1i0/llmEZ.git
cd llmEZ
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Si todavía no tenés modelos: `ollama pull qwen3:1.7b` (u otro).

Atajos: **Enter** envía, **Shift+Enter** agrega un salto de línea, **Esc** detiene la generación.

## Ejecutable con PyInstaller

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pyinstaller --onefile --windowed --name llmEZ \
    --distpath dist --workpath build --specpath build main.py --noconfirm
./dist/llmEZ
```

- `--onefile` genera un único binario (~13 MB). Arranca un poco más lento porque se descomprime en `/tmp` al abrirse; sin esa opción queda una carpeta `dist/llmEZ/` que arranca más rápido.
- El binario funciona en sistemas Linux con una glibc igual o más nueva que la de la máquina donde se compiló. Para Windows o macOS hay que compilarlo en ese sistema.

## Estructura

```
main.py                     punto de entrada
llmez/
├── app.py                  ventana principal; cola de mensajes entre hilos y la UI
├── chat_panel.py           chat, opciones, streaming, estadísticas, liberar VRAM
├── monitor_panel.py        gráficos y datos de CPU, RAM, GPU, Ollama y generación
├── ollama_client.py        cliente HTTP mínimo de Ollama (tags, ps, chat, unload)
├── metrics.py              hilo que muestrea psutil, nvidia-smi y /api/ps cada 1 s
├── think_parser.py         separa <think>…</think> que llega mezclado en content
├── theme.py                tema oscuro de ttk
├── config.py               URL, prompts, valores por defecto y colores
└── widgets/line_chart.py   gráfico de línea sobre tk.Canvas
```

**Hilos.** Las llamadas a Ollama y el muestreo de métricas corren en hilos aparte. Tkinter no es thread-safe, así que esos hilos nunca tocan widgets: encolan funciones con `app.post(...)` y la UI las ejecuta cada 30 ms.

**Detener.** Se activa un `threading.Event` y se cierra el socket del stream; Ollama detecta la desconexión y cancela la generación (en su log aparece `srv stop: cancel task`).

## Lo que aprendí

Medido en una Dell Inspiron 7559 (i7 de 4 núcleos / 8 hilos, GTX 960M de 4 GB, PRIME on-demand), con `num_ctx` 2048:

| Modelo | Tamaño en memoria | Velocidad aprox. |
|---|---|---|
| qwen3:0.6b | 0,76 GB | ~38 tok/s |
| qwen3:1.7b | 1,51 GB | ~17 tok/s |
| qwen3:4b | 2,97 GB | ~8 tok/s |

- **El contexto también ocupa VRAM.** qwen3:4b con el contexto por defecto no entra entero en 4 GB; con `num_ctx` 2048 y la GPU libre (PRIME on-demand, el escritorio en la Intel) entra al 100 %.
- **Dos modelos cargados se pisan.** Con qwen3:0.6b cargado, qwen3:4b quedó solo al 76 % en GPU (el resto en CPU). Y si el modelo ya estaba cargado a medias, Ollama lo reusa así: hay que descargarlo para que vuelva a entrar entero. Por eso existe la opción *Solo un modelo en VRAM*.
- **qwen3:4b ignora `think: false`.** El razonamiento llega dentro de `content`, sin etiqueta de apertura y cerrado con un `</think>` suelto. La app lo detecta en pleno streaming, recolorea como razonamiento lo que ya había mostrado, avisa, y guarda solo la respuesta en el historial.
- **Los modelos chicos razonan en inglés.** Aun con el prompt de sistema pidiendo español, los qwen3 suelen pensar en inglés; la respuesta final sí sale en español.
- **No siempre aciertan.** En la captura, qwen3:1.7b explica vagamente por qué un modelo de 4B entra en 4 GB. La razón concreta es la cuantización: Q4_K_M usa unos 4,5 bits por parámetro, así que 4.000 millones de parámetros ocupan ~2,5 GB en vez de 8 GB en FP16.
- **El servidor de Ollama casi no consume.** `ollama serve` usa ~36 MB de RAM en reposo. Lo que ocupa VRAM es el proceso `llama-server` de cada modelo cargado, que desaparece al descargarlo. No hace falta apagar el servicio.
