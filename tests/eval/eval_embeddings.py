"""Compara modelos de embeddings sobre el set de tests/eval/embeddings_eval.json.

Uso:  .venv/bin/python tests/eval/eval_embeddings.py [modelo ...]
Por defecto compara qwen3-embedding:0.6b y qwen3-embedding:4b. Usa el Ollama real y
la GPU (indexa con la VRAM libre). Guarda los resultados completos en
~/.cache/llmez-tools/eval/results-<fecha>.json y los resume por pantalla.
"""
import json
import math
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from llmez.documents import read_document  # noqa: E402
from llmez.ollama_client import OllamaClient  # noqa: E402
from llmez.retrieval import Index, combine, estimate_tokens  # noqa: E402
from llmez.semantic import QUERY_PREFIX  # noqa: E402

MODELS = sys.argv[1:] or ["qwen3-embedding:0.6b", "qwen3-embedding:4b"]
BUDGETS = (600, 1000, 2000, 4000)
CPU_SAMPLE = 16  # fragmentos para estimar la velocidad de indexar en CPU
BATCH = 8
client = OllamaClient("http://localhost:11434")
spec = json.load(open(os.path.join(os.path.dirname(__file__), "embeddings_eval.json"), encoding="utf-8"))


def log(*a):
    print(*a, flush=True)


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def free_vram():
    for m in client.running():
        client.unload(m["name"])
    for _ in range(40):
        if not client.running():
            return
        time.sleep(0.25)


def embed(model, texts, cpu):
    opts = {"num_ctx": 2048, **({"num_gpu": 0} if cpu else {})}
    return [unit(v) for v in client.embed(model, texts, options=opts, timeout=1800)]


def model_memory(model):
    for m in client.running():
        if m["name"] == model:
            return m.get("size", 0), m.get("size_vram", 0)
    return 0, 0


# ------------------------------------------------------------------ documentos
docs = {}
for key, d in spec["documents"].items():
    text, info = read_document(os.path.expanduser(d["path"]))
    idx = Index(text)
    docs[key] = {"index": idx, "texts": [t for _, t in idx.chunks], "info": info,
                 "tokens": estimate_tokens(text, 3.9)}
    log(f"{key}: {info} · ~{docs[key]['tokens']:,} tokens · {len(idx.chunks)} fragmentos")

questions = [q for q in spec["questions"] if q["needle"]]
for q in questions:  # una pregunta mal armada no debe contar como falla del modelo
    if not any(q["needle"] in t for t in docs[q["doc"]]["texts"]):
        sys.exit(f"needle no encontrado en {q['doc']}: {q['needle']!r}")
log(f"{len(questions)} preguntas evaluables ({sum(q['type'] == 'parafrasis' for q in questions)} paráfrasis)")


def rank(scores, texts, needle):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return next(pos for pos, i in enumerate(order, 1) if needle in texts[i])


results = {"date": datetime.now().isoformat(timespec="seconds"), "models": {}, "questions": []}
bm25_ranks = {}
for q in questions:
    d = docs[q["doc"]]
    bm25_ranks[q["q"]] = rank(d["index"].scores(q["q"]), d["texts"], q["needle"])

for model in MODELS:
    log(f"\n=== {model}")
    info = {"index": {}, "cpu_chunk_s": None, "query_ms": None, "ram_cpu_gb": None, "gpu": {}}
    vectors = {}
    for key, d in docs.items():
        free_vram()
        t0, vecs, err = time.time(), [], None
        try:
            for i in range(0, len(d["texts"]), BATCH):
                vecs += embed(model, d["texts"][i:i + BATCH], cpu=False)
                if i == 0:
                    size, vram = model_memory(model)
                    info["gpu"][key] = {"size_gb": round(size / 1e9, 2), "vram_gb": round(vram / 1e9, 2),
                                        "on_gpu_pct": round(100 * vram / size) if size else None}
        except Exception as e:  # p. ej. sin memoria en la GPU: se registra, no se esconde
            err = str(e)[:200]
        secs = time.time() - t0
        info["index"][key] = {"seconds": round(secs, 1), "chunks": len(d["texts"]), "error": err}
        log(f"  indexar {key} en GPU: {secs:.0f} s · {len(vecs)}/{len(d['texts'])} fragmentos"
            + (f" · en GPU {info['gpu'][key]['on_gpu_pct']}%" if key in info["gpu"] else "")
            + (f" · ERROR {err}" if err else ""))
        if err:  # completar en CPU para poder evaluar la calidad igual
            vecs += embed(model, d["texts"][len(vecs):], cpu=True)
        vectors[key] = vecs
    free_vram()
    sample = docs["ordenanza"]["texts"][:CPU_SAMPLE]
    embed(model, sample[:1], cpu=True)  # carga
    t0 = time.time()
    embed(model, sample, cpu=True)
    info["cpu_chunk_s"] = round((time.time() - t0) / len(sample), 3)
    size, _ = model_memory(model)
    info["ram_cpu_gb"] = round(size / 1e9, 2)
    t0 = time.time()
    qvecs = {q["q"]: embed(model, [QUERY_PREFIX + q["q"]], cpu=True)[0] for q in questions}
    info["query_ms"] = round(1000 * (time.time() - t0) / len(questions))
    total_chunks = sum(len(d["texts"]) for d in docs.values())
    log(f"  CPU: {info['cpu_chunk_s']} s por fragmento (→ ~{info['cpu_chunk_s'] * total_chunks / 60:.0f} min para los "
        f"{total_chunks} fragmentos) · pregunta {info['query_ms']} ms · RAM {info['ram_cpu_gb']} GB")
    for q in questions:
        d = docs[q["doc"]]
        dense = [sum(a * b for a, b in zip(qvecs[q["q"]], v)) for v in vectors[q["doc"]]]
        row = next((r for r in results["questions"] if r["q"] == q["q"]), None)
        if row is None:
            row = {"q": q["q"], "doc": q["doc"], "type": q["type"], "bm25_rank": bm25_ranks[q["q"]], "models": {}}
            results["questions"].append(row)
        reach = {}
        for budget in BUDGETS:
            text, _ = d["index"].select(q["q"], budget, chars_per_token=2.8, dense=dense)
            reach[budget] = q["needle"] in text
        row["models"][model] = {"emb_rank": rank(dense, d["texts"], q["needle"]),
                                "hybrid_rank": rank(combine(d["index"].scores(q["q"]), dense), d["texts"], q["needle"]),
                                "reach": reach}
    results["models"][model] = info
    free_vram()


# ------------------------------------------------------------------ resumen
def summary(rows, key):
    ranks = [r[key] for r in rows]
    return {"top1": sum(x == 1 for x in ranks), "top3": sum(x <= 3 for x in ranks),
            "mrr": round(sum(1 / x for x in ranks) / len(ranks), 3), "worst": max(ranks)}


log("\n=== Resumen (puesto del fragmento con la respuesta; MRR = promedio de 1/puesto, 1 es perfecto)")
for subset_name, cond in (("todas", lambda r: True), ("paráfrasis", lambda r: r["type"] == "parafrasis"),
                          ("directas", lambda r: r["type"] == "directa")):
    rows = [r for r in results["questions"] if cond(r)]
    bm = summary([{"x": r["bm25_rank"]} for r in rows], "x")
    log(f"\n[{subset_name}: {len(rows)} preguntas]  BM25: top1 {bm['top1']} · top3 {bm['top3']} · MRR {bm['mrr']} · peor {bm['worst']}")
    for model in MODELS:
        for key, label in (("emb_rank", "solo embeddings"), ("hybrid_rank", "combinada")):
            s = summary([{"x": r["models"][model][key]} for r in rows], "x")
            log(f"  {model:22} {label:16} top1 {s['top1']:>2} · top3 {s['top3']:>2} · MRR {s['mrr']:.3f} · peor {s['worst']}")
        reach = {b: sum(r["models"][model]["reach"][b] for r in rows) for b in BUDGETS}
        log(f"  {model:22} {'llega al modelo':16} " + " · ".join(f"{b} tok: {reach[b]}/{len(rows)}" for b in BUDGETS))
    results.setdefault("summary", {})[subset_name] = {"n": len(rows), "bm25": bm}

out = os.path.expanduser(f"~/.cache/llmez-tools/eval/results-{datetime.now():%Y%m%d-%H%M}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
log(f"\nresultados completos: {out}")
