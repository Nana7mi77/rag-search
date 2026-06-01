"""
本地 bge-m3 Embedding API 服务
启动: python serve_embedding.py
调用: curl -X POST http://localhost:8099/embed -H "Content-Type: application/json" -d '{"texts": ["你好世界"]}'
"""
import os
import time
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from sentence_transformers import SentenceTransformer

app = FastAPI(title="bge-m3 Embedding Server")
model = None


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int
    took_ms: float


@app.on_event("startup")
def load_model():
    global model
    print("Loading BAAI/bge-m3 on MPS...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3", device="mps")
    model.encode(["warmup"], batch_size=1, show_progress_bar=False)
    print(f"Model ready in {time.time() - t0:.1f}s, dim={model.get_sentence_embedding_dimension()}")


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    t0 = time.time()
    vecs = model.encode(req.texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    return EmbedResponse(
        embeddings=vecs.tolist(),
        model="BAAI/bge-m3",
        dim=vecs.shape[1],
        took_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "BAAI/bge-m3" if model else "not loaded"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099)
