import unittest

from llmez.model_info import ModelInfo

# Extracto real de /api/show de qwen3:4b (Ollama 0.40).
QWEN3_4B_SHOW = {
    "capabilities": ["completion", "tools", "thinking"],
    "details": {"family": "qwen3", "parameter_size": "4.0B", "quantization_level": "Q4_K_M"},
    "parameters": 'repeat_penalty 1\nstop "<|im_start|>"\nstop "<|im_end|>"\ntemperature 0.6',
    "model_info": {
        "general.architecture": "qwen3",
        "qwen3.block_count": 36,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.attention.value_length": 128,
        "qwen3.embedding_length": 2560,
        "qwen3.context_length": 262144,
    },
}
FILE_SIZE = 2_497_293_931
VRAM_960M = 4096 * 1024 * 1024


class ModelInfoTest(unittest.TestCase):
    def setUp(self):
        self.info = ModelInfo("qwen3:4b", QWEN3_4B_SHOW, FILE_SIZE)

    def test_metadata(self):
        self.assertTrue(self.info.thinks)
        self.assertEqual(self.info.defaults["temperature"], "0.6")
        self.assertEqual(self.info.defaults["stop"], "<|im_start|>")  # se queda con el primero
        self.assertEqual(self.info.context_max, 262144)

    def test_kv_cache_matches_ollama(self):
        # /api/ps reportó 2974 MB con num_ctx 2048 → pesos 2497 + KV 302 + grafo.
        self.assertEqual(self.info.kv_cache_bytes(2048), 301_989_888)
        self.assertEqual(self.info.kv_cache_bytes(8192), 4 * 301_989_888)

    def test_fit_matches_measurements_on_gtx_960m(self):
        # Medido: 2048 → 100 % GPU; 3072 → 96 %; 8192 → 76 %.
        self.assertIn(self.info.fit(2048, VRAM_960M)[0], ("entra", "justo"))
        self.assertEqual(self.info.fit(3072, VRAM_960M)[0], "no entra")
        state, frac = self.info.fit(8192, VRAM_960M)
        self.assertEqual(state, "no entra")
        self.assertAlmostEqual(frac, 0.76, delta=0.08)

    def test_no_gpu(self):
        self.assertEqual(self.info.fit(2048, None), (None, None))

    def test_head_dim_fallback_without_key_length(self):
        show = {"model_info": {"general.architecture": "llama", "llama.block_count": 30,
                               "llama.attention.head_count": 9,
                               "llama.attention.head_count_kv": 3,
                               "llama.embedding_length": 576}}
        info = ModelInfo("smollm2:135m", show, 271_000_000)
        self.assertFalse(info.thinks)
        # 30 capas × 2048 × 3 cabezales × (64+64) × 2 bytes
        self.assertEqual(info.kv_cache_bytes(2048), 30 * 2048 * 3 * 128 * 2)

    def test_unknown_architecture(self):
        info = ModelInfo("raro", {}, 1_000_000)
        self.assertIsNone(info.estimate_bytes(2048))
        self.assertEqual(info.fit(2048, VRAM_960M), (None, None))


if __name__ == "__main__":
    unittest.main()
