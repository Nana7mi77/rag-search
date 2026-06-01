import csv
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="mps")
model.encode(["warmup"], batch_size=1, show_progress_bar=False)

texts = []
with open("data/local_subtitles.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        t = (row.get("subtitle") or "").strip()
        if t:
            texts.append(t[:200])

print(f"文档数: {len(texts)}", flush=True)
print(flush=True)

for bs in [16, 32, 64, 128, 256, 512]:
    try:
        t0 = time.time()
        vecs = model.encode(texts[:200], batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.time() - t0
        speed = len(texts[:200]) / elapsed
        total_est = len(texts) / speed
        print(f"batch_size={bs:>4}: {elapsed:.1f}s (200条), speed={speed:.0f} docs/s, 全量估算={total_est:.0f}s", flush=True)
    except Exception as e:
        print(f"batch_size={bs:>4}: FAILED - {type(e).__name__}: {e}", flush=True)
        break
