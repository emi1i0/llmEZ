"""Tests del cliente contra un servidor HTTP falso que imita a Ollama."""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llmez.ollama_client import OllamaClient, OllamaError


class FakeOllama(BaseHTTPRequestHandler):
    slow_chunks = 200
    aborted = threading.Event()

    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send_json(200, {"models": [{"name": "b:1", "size": 2}, {"name": "a:1", "size": 1}]})
        else:
            self._send_json(404, {"error": "no existe"})

    def do_DELETE(self):
        self._send_json(200, {})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/chat":
            self._send_json(200, {})
            return
        if payload.get("think") and payload["model"] == "sin-think":
            self._send_json(400, {"error": '"sin-think" does not support thinking'})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        n = self.slow_chunks if payload["model"] == "lento" else 3
        try:
            for i in range(n):
                chunk = {"message": {"role": "assistant", "content": f"t{i} "}, "done": False}
                self.wfile.write((json.dumps(chunk) + "\n").encode())
                self.wfile.flush()
                if payload["model"] == "lento":
                    time.sleep(0.02)
            if payload["model"] == "roto":
                self.wfile.write(b'{"error": "se quedo sin memoria"}\n')
                return
            self.wfile.write((json.dumps({"done": True, "eval_count": n}) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            FakeOllama.aborted.set()


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = OllamaClient(f"http://127.0.0.1:{cls.server.server_port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def chat(self, model, **extra):
        payload = {"model": model, "messages": [], **extra}
        return list(self.client.chat_stream(payload, threading.Event()))

    def test_list_models_sorted(self):
        self.assertEqual([m["name"] for m in self.client.list_models()], ["a:1", "b:1"])

    def test_stream_until_done(self):
        chunks = self.chat("rapido")
        self.assertEqual(len(chunks), 4)
        self.assertTrue(chunks[-1]["done"])

    def test_http_error_message_is_kept(self):
        with self.assertRaises(OllamaError) as cm:
            self.chat("sin-think", think=True)
        self.assertIn("does not support thinking", str(cm.exception))

    def test_error_inside_stream(self):
        with self.assertRaises(OllamaError) as cm:
            self.chat("roto")
        self.assertIn("sin memoria", str(cm.exception))

    def test_abort_stops_quickly_and_server_notices(self):
        stop = threading.Event()
        box = {}
        got = []

        def run():
            for c in self.client.chat_stream({"model": "lento", "messages": []}, stop,
                                             on_open=lambda r: box.setdefault("r", r)):
                got.append(c)
                if len(got) == 5:
                    stop.set()
                    OllamaClient.abort(box["r"])
        t = threading.Thread(target=run)
        t0 = time.monotonic()
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - t0, 2)  # 200 chunks tardarían ~4 s
        self.assertEqual(len(got), 5)
        self.assertTrue(FakeOllama.aborted.wait(3))

    def test_connection_refused(self):
        client = OllamaClient("http://127.0.0.1:9")
        with self.assertRaises(OllamaError) as cm:
            client.list_models()
        self.assertIn("no se pudo conectar", str(cm.exception))

    def test_delete_uses_delete_method(self):
        self.client.delete("a:1")  # no lanza


if __name__ == "__main__":
    unittest.main()
