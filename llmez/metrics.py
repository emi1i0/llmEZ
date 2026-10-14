"""Muestreo periódico de CPU, RAM, GPU (nvidia-smi) y modelos cargados en Ollama."""
import shutil
import subprocess
import threading
import time

import psutil

from .ollama_client import OllamaError

# El nombre va último porque es el único campo que podría tener comas.
_GPU_FIELDS = ("utilization.gpu", "memory.used", "memory.total",
               "temperature.gpu", "pstate", "name")


def _num(value):
    try:
        return float(value)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


def query_gpu():
    """Devuelve (datos, error). Nunca lanza excepciones."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, "nvidia-smi no encontrado"
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=" + ",".join(_GPU_FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi no respondió"
    except OSError as e:
        return None, str(e)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        return None, msg[0] if msg else f"nvidia-smi salió con código {proc.returncode}"
    lines = proc.stdout.strip().splitlines()
    parts = [p.strip() for p in lines[0].split(",", len(_GPU_FIELDS) - 1)] if lines else []
    if len(parts) != len(_GPU_FIELDS):
        return None, "salida inesperada de nvidia-smi"
    util, used, total, temp, pstate, name = parts
    if all(_num(v) is None for v in (util, used, total, temp)):
        return None, "nvidia-smi no devolvió datos numéricos"
    return {"util": _num(util), "mem_used": _num(used), "mem_total": _num(total),
            "temp": _num(temp), "pstate": pstate, "name": name}, None


class MetricsSampler(threading.Thread):
    """Hilo que toma una muestra por intervalo y se la pasa a on_sample.

    on_sample se llama desde este hilo: no debe tocar widgets de Tk.
    """

    def __init__(self, client, on_sample, interval=1.0):
        super().__init__(daemon=True, name="metrics")
        self.client = client
        self.on_sample = on_sample
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        # La primera llamada de cpu_percent solo fija la referencia.
        psutil.cpu_percent()
        psutil.cpu_percent(percpu=True)
        deadline = time.monotonic()
        while True:
            deadline += self.interval
            if self._stop_event.wait(max(0.0, deadline - time.monotonic())):
                return
            self.on_sample(self.sample())

    def sample(self):
        vm = psutil.virtual_memory()
        s = {
            "cpu_total": psutil.cpu_percent(),
            "cpu_cores": psutil.cpu_percent(percpu=True),
            "ram_used": vm.total - vm.available,
            "ram_total": vm.total,
        }
        s["gpu"], s["gpu_error"] = query_gpu()
        try:
            s["ollama_ps"], s["ollama_error"] = self.client.running(timeout=1.5), None
        except OllamaError as e:
            s["ollama_ps"], s["ollama_error"] = None, str(e)
        return s
