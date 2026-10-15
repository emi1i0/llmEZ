import json
import os
import tempfile
import unittest

from llmez import conversation, settings


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sub", "settings.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_gives_defaults(self):
        self.assertEqual(settings.load(self.path), settings.DEFAULTS)

    def test_roundtrip_and_unknown_keys_ignored(self):
        data = dict(settings.DEFAULTS, model="qwen3:4b", num_ctx=4096)
        self.assertIsNone(settings.save(dict(data, basura=1), self.path))
        loaded = settings.load(self.path)
        self.assertEqual(loaded["model"], "qwen3:4b")
        self.assertEqual(loaded["num_ctx"], 4096)
        self.assertNotIn("basura", loaded)

    def test_wrong_types_fall_back_to_defaults(self):
        os.makedirs(os.path.dirname(self.path))
        bad = {"geometry": "basura", "exclusive_vram": "sí", "unload_on_exit": None,
               "num_ctx": "abc", "seed": [1, 2], "system_prompt": None, "loop_guard": 1,
               "temperature": 5, "model": 42}
        with open(self.path, "w") as f:
            json.dump(bad, f)
        self.assertEqual(settings.load(self.path), settings.DEFAULTS)

    def test_valid_values_are_kept(self):
        good = {"geometry": "1200x700+10+-5", "num_ctx": 4096, "loop_guard": False, "seed": "7"}
        self.assertIsNone(settings.save(dict(settings.DEFAULTS, **good), self.path))
        loaded = settings.load(self.path)
        self.assertEqual({k: loaded[k] for k in good}, good)

    def test_absurd_context_is_rejected(self):
        self.assertIsNone(settings.save(dict(settings.DEFAULTS, num_ctx=12), self.path))
        self.assertEqual(settings.load(self.path)["num_ctx"], settings.DEFAULTS["num_ctx"])

    def test_corrupt_file_does_not_break(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write("{roto")
        self.assertEqual(settings.load(self.path), settings.DEFAULTS)


class ConversationTest(unittest.TestCase):
    MESSAGES = [
        {"role": "user", "content": "¿2+3?"},
        {"role": "assistant", "content": "5", "model": "qwen3:4b",
         "thinking": "sumo", "stats": {"eval_count": 10, "tok_s": 8.0}},
    ]

    def test_roundtrip(self):
        data = conversation.to_json(self.MESSAGES, {"model": "qwen3:4b"}, "Respondé en español",
                                    "notas.txt", "contenido")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            conversation.save(path, data)
            loaded = conversation.load(path)
        finally:
            os.unlink(path)
        self.assertEqual(loaded["messages"], self.MESSAGES)
        self.assertEqual(loaded["file"], {"name": "notas.txt", "text": "contenido"})

    def test_rejects_foreign_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"messages": []}, f)
        try:
            with self.assertRaises(ValueError):
                conversation.load(f.name)
        finally:
            os.unlink(f.name)

    def test_markdown_export(self):
        md = conversation.to_markdown(conversation.to_json(self.MESSAGES, {}, "", None, None))
        self.assertIn("## Vos", md)
        self.assertIn("## qwen3:4b", md)
        self.assertIn("<summary>Razonamiento</summary>", md)
        self.assertIn("10 tokens · 8.0 tok/s", md)


if __name__ == "__main__":
    unittest.main()
