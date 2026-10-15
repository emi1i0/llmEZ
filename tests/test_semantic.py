import os
import tempfile
import threading
import unittest
from unittest import mock

from llmez.ollama_client import OllamaError
from llmez.retrieval import Index, combine
from llmez.semantic import SemanticIndex, is_embedding_only


class FakeClient:
    """Imita /api/embed: un vector por texto según las palabras que contiene."""
    VOCAB = ["duración", "años", "rector", "firma", "título", "tiempo", "recibirse"]
    SYNONYMS = {"tiempo": "duración", "recibirse": "años"}  # lo que un embedding "entiende"

    def __init__(self, vram_busy=False, fail_on_gpu=False):
        self.vram_busy, self.fail_on_gpu = vram_busy, fail_on_gpu
        self.calls = []

    def running(self):
        return [{"name": "qwen3:1.7b", "size_vram": 2_000_000_000}] if self.vram_busy else []

    def embed(self, model, inputs, options=None, keep_alive=None, timeout=600):
        on_cpu = (options or {}).get("num_gpu") == 0
        self.calls.append(("CPU" if on_cpu else "GPU", len(inputs)))
        if self.fail_on_gpu and not on_cpu:
            raise OllamaError("do embedding request: EOF (CUDA out of memory)")
        out = []
        for text in inputs:
            words = [self.SYNONYMS.get(w, w) for w in text.lower().replace("?", " ").split()]
            out.append([1.0 + words.count(v) for v in self.VOCAB])
        return out


PAGES = ["Duración de la carrera: 5 años", "Firma el Rector Rubén Soro", "Título intermedio: Analista"]


class SemanticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"XDG_CACHE_HOME": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_uses_gpu_when_vram_is_free_and_caches(self):
        client = FakeClient()
        idx = SemanticIndex(client, PAGES)
        self.assertFalse(idx.load_cached())
        self.assertTrue(idx.build(threading.Event()))
        self.assertEqual(idx.device, "GPU")
        again = SemanticIndex(FakeClient(), PAGES)
        self.assertTrue(again.load_cached())
        self.assertEqual(again.device, "caché")
        self.assertEqual(again.vectors, idx.vectors)

    def test_uses_cpu_when_another_model_is_in_vram(self):
        idx = SemanticIndex(FakeClient(vram_busy=True), PAGES)
        idx.build(threading.Event())
        self.assertEqual(idx.device, "CPU")

    def test_gpu_failure_falls_back_to_cpu(self):
        client = FakeClient(fail_on_gpu=True)
        idx = SemanticIndex(client, PAGES)
        self.assertTrue(idx.build(threading.Event()))
        self.assertEqual(idx.device, "CPU")
        self.assertEqual(len(idx.vectors), 3)

    def test_release_gpu_moves_remaining_batches_to_cpu(self):
        client = FakeClient()
        client.unload = lambda model: None
        idx = SemanticIndex(client, PAGES * 6)  # 18 textos: 3 lotes
        orig = client.embed

        def embed(*a, **k):
            if len(client.calls) == 1:   # después del primer lote, el chat pide la GPU
                idx.release_gpu(wait_s=0)
            return orig(*a, **k)
        client.embed = embed
        idx.build(threading.Event())
        self.assertEqual([d for d, _ in client.calls], ["GPU", "GPU", "CPU"])
        self.assertEqual(idx.device, "CPU")

    def test_query_runs_on_cpu_and_finds_synonym(self):
        client = FakeClient()
        idx = SemanticIndex(client, PAGES)
        idx.build(threading.Event())
        scores = idx.scores("¿Cuánto tiempo lleva recibirse?")
        self.assertEqual(client.calls[-1][0], "CPU")
        self.assertEqual(scores.index(max(scores)), 0)

    def test_not_ready_gives_none(self):
        self.assertIsNone(SemanticIndex(FakeClient(), PAGES).scores("x"))

    def test_stop_interrupts_build(self):
        stop = threading.Event()
        stop.set()
        idx = SemanticIndex(FakeClient(), PAGES)
        self.assertFalse(idx.build(stop))
        self.assertFalse(idx.ready)

    def test_embedding_models_are_not_chat_models(self):
        self.assertTrue(is_embedding_only("qwen3-embedding:0.6b"))
        self.assertTrue(is_embedding_only("nomic-embed-text:latest"))
        self.assertFalse(is_embedding_only("qwen3:1.7b"))


class CombineTest(unittest.TestCase):
    def test_without_dense_is_plain_bm25(self):
        self.assertEqual(combine([1.0, 3.0], None), [1.0, 3.0])

    def test_dense_rescues_page_that_bm25_misses(self):
        bm25 = [0.0, 0.0, 5.0]       # la palabra clave cae en otra página
        dense = [0.9, 0.1, 0.2]      # el significado apunta a la primera
        mixed = combine(bm25, dense)
        self.assertGreater(mixed[0], mixed[1])

    def test_only_dense_when_no_keyword_matches(self):
        for got, want in zip(combine([0, 0, 0], [0.2, 0.8, 0.5]), [0.0, 1.0, 0.5]):
            self.assertAlmostEqual(got, want)

    def test_select_uses_dense_scores(self):
        idx = Index("\n\n".join(f"--- página {i} ---\n{t}" for i, t in enumerate(PAGES, 1)))
        _, labels = idx.select("¿Cuánto tiempo lleva recibirse?", 10_000, dense=[0.95, 0.1, 0.1])
        self.assertEqual(labels, ["página 1"])


if __name__ == "__main__":
    unittest.main()
