"""Cliente mínimo de la API HTTP de Ollama, solo con la biblioteca estándar."""
import json
import socket
import urllib.error
import urllib.request


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _open(self, path, payload=None, timeout=5.0):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=headers,
            method="POST" if data is not None else "GET")
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(body).get("error") or body
            except ValueError:
                msg = body
            raise OllamaError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise OllamaError(
                f"no se pudo conectar con Ollama en {self.base_url} ({reason})") from None

    def _json(self, path, payload=None, timeout=5.0):
        with self._open(path, payload, timeout) as resp:
            return json.load(resp)

    def list_models(self):
        data = self._json("/api/tags")
        return sorted(m["name"] for m in data.get("models", []))

    def running(self, timeout=2.0):
        return self._json("/api/ps", timeout=timeout).get("models", [])

    def unload(self, model):
        """Saca el modelo de memoria (RAM/VRAM) inmediatamente."""
        self._json("/api/generate", {"model": model, "keep_alive": 0}, timeout=30)

    def chat_stream(self, payload, stop_event, on_open=None, timeout=600):
        """Genera los objetos JSON del stream de /api/chat.

        Corta en silencio si se activa stop_event. on_open recibe la respuesta
        HTTP para poder abortarla desde otro hilo con abort().
        """
        resp = self._open("/api/chat", payload, timeout)
        try:
            if on_open:
                on_open(resp)
            for line in resp:
                if stop_event.is_set():
                    return
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                yield chunk
                if chunk.get("done"):
                    return
        except (OSError, ValueError) as e:
            if stop_event.is_set():
                return
            raise OllamaError(f"se cortó el stream: {e}") from None
        finally:
            resp.close()

    @staticmethod
    def abort(resp):
        """Cierra el socket de un stream en curso: Ollama deja de generar."""
        try:
            resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
