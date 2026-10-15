"""Ajusta el peso BM25/embeddings y el umbral de selección con tests/eval/embeddings_eval.json.

Uso:  .venv/bin/python tests/eval/tune_weight.py [modelo]   (por defecto qwen3-embedding:0.6b)
Los vectores se calculan una vez (GPU si está libre) y se guardan en
~/.cache/llmez-tools/eval/vectors-<modelo>.json; las corridas siguientes son instantáneas.
Para no sobreajustar, además de la mejor combinación global se valida dejando un
documento afuera: se elige con dos documentos y se mide en el tercero.
"""
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from llmez.documents import read_document  # noqa: E402
from llmez.ollama_client import OllamaClient  # noqa: E402
from llmez.retrieval import Index, combine  # noqa: E402
from llmez.semantic import QUERY_PREFIX  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-embedding:0.6b"
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
RATIOS = (0.1, 0.25, 0.4)
BUDGETS = (600, 1000, 2000, 4000)
client = OllamaClient("http://localhost:11434")
spec = json.load(open(os.path.join(os.path.dirname(__file__), "embeddings_eval.json"), encoding="utf-8"))
cache = os.path.expanduser(f"~/.cache/llmez-tools/eval/vectors-{MODEL.replace(':', '_')}.json")


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


docs = {k: Index(read_document(os.path.expanduser(d["path"]))[0]) for k, d in spec["documents"].items()}
questions = [q for q in spec["questions"] if q["needle"]]

vectors = json.load(open(cache)) if os.path.exists(cache) else {}
if set(vectors) != {"docs", "queries"} or any(len(vectors["docs"].get(k, [])) != len(i.chunks) for k, i in docs.items()):
    for m in client.running():
        client.unload(m["name"])
    time.sleep(2)
    t0 = time.time()
    vectors = {"docs": {}, "queries": {}}
    for key, idx in docs.items():
        texts = [t for _, t in idx.chunks]
        vectors["docs"][key] = [unit(v) for i in range(0, len(texts), 8)
                                for v in client.embed(MODEL, texts[i:i + 8], options={"num_ctx": 2048})]
    for q in questions:
        vectors["queries"][q["q"]] = unit(client.embed(MODEL, [QUERY_PREFIX + q["q"]], options={"num_gpu": 0, "num_ctx": 512})[0])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(vectors, open(cache, "w"))
    print(f"vectores calculados en {time.time() - t0:.0f} s y guardados en {cache}")

# Puntajes precalculados por pregunta: BM25 y similitud con cada fragmento.
pre = []
for q in questions:
    idx = docs[q["doc"]]
    qv = vectors["queries"][q["q"]]
    dense = [sum(a * b for a, b in zip(qv, v)) for v in vectors["docs"][q["doc"]]]
    pre.append((q, idx, idx.scores(q["q"]), dense))


def evaluate(weight, ratio, subset):
    reach = 0
    rr = []
    worst = 0
    for q, idx, bm, dense in subset:
        mixed = combine(bm, dense, weight=weight) if weight > 0 else bm
        order = sorted(range(len(mixed)), key=lambda i: -mixed[i])
        pos = next(p for p, i in enumerate(order, 1) if q["needle"] in idx.chunks[i][1])
        rr.append(1 / pos)
        worst = max(worst, pos)
        for b in BUDGETS:
            # misma selección que la app (con el peso y el umbral a evaluar)
            import llmez.retrieval as R
            orig = R.combine
            R.combine = lambda b_, d_, weight=weight: orig(b_, d_, weight=weight) if weight > 0 else b_
            text, _ = idx.select(q["q"], b, min_ratio=ratio, chars_per_token=2.8, dense=dense)
            R.combine = orig
            reach += q["needle"] in text
    return reach, sum(rr) / len(rr), worst


def best(subset):
    scored = [((w, r), evaluate(w, r, subset)) for w in WEIGHTS for r in RATIOS]
    # primero que la respuesta llegue; después MRR; después el peor puesto (menor es mejor)
    return max(scored, key=lambda x: (x[1][0], x[1][1], -x[1][2])), scored


top, scored = best(pre)
n = len(pre) * len(BUDGETS)
print(f"\n{len(pre)} preguntas × {len(BUDGETS)} presupuestos = {n} casos. 'llega' = respuesta dentro de lo que se manda.\n")
print("peso emb. | " + " | ".join(f"umbral {r}: llega · MRR · peor" for r in RATIOS))
for w in WEIGHTS:
    row = [next(s for (ww, rr), s in scored if ww == w and rr == r) for r in RATIOS]
    print(f"   {w:>4}   | " + " | ".join(f"{s[0]:>3}/{n} · {s[1]:.3f} · {s[2]:>3}" for s in row))
(w, r), s = top
cur = evaluate(0.7, 0.25, pre)
print(f"\nactual (0.7, umbral 0.25): llega {cur[0]}/{n} · MRR {cur[1]:.3f} · peor {cur[2]}")
print(f"mejor  ({w}, umbral {r}): llega {s[0]}/{n} · MRR {s[1]:.3f} · peor {s[2]}")

print("\nValidación dejando un documento afuera (se elige con 2, se mide en el 3.º):")
for left in docs:
    train = [p for p in pre if p[0]["doc"] != left]
    test = [p for p in pre if p[0]["doc"] == left]
    ((tw, tr), _), _ = best(train)
    got = evaluate(tw, tr, test)
    ref = evaluate(0.7, 0.25, test)
    m = len(test) * len(BUDGETS)
    print(f"  afuera {left:13}: elegido con los otros = ({tw}, {tr}) → llega {got[0]}/{m} · MRR {got[1]:.3f}"
          f"   | actual (0.7, 0.25) → llega {ref[0]}/{m} · MRR {ref[1]:.3f}")
