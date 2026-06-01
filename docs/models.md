# 本地模型登记

## BAAI/bge-m3

| 项目 | 信息 |
|------|------|
| 模型 | `BAAI/bge-m3` |
| 架构 | XLM-RoBERTa (568M 参数) |
| 向量维度 | 1024 |
| 最大序列长度 | 8192 tokens |
| License | MIT |
| 缓存路径 | `~/.cache/huggingface/hub/models--BAAI--bge-m3/` |
| 缓存大小 | ~4.3 GB |

### 运行环境

| 项目 | 信息 |
|------|------|
| 设备 | MacBook Air M1 8GB |
| Python | 3.9 (x86 via Rosetta) |
| PyTorch | 2.2.2 |
| sentence-transformers | 5.1.2 |
| transformers | 4.49.0（需 <4.50.0，受限于 torch 2.2.2） |
| 加速 | MPS (Apple Metal) |
| 虚拟环境 | `.venv/` |

### 性能

| 指标 | 数值 |
|------|------|
| 模型加载 | ~70s（含网络检查，离线更快） |
| MPS warmup 首次推理 | ~8.5s |
| batch_size=16 吞吐 | 0.03 s/sample |
| 全量 6000 文档估算 | ~3 分钟 |

### 调用方式

**方式 1：Python 直接调用**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="mps")
vectors = model.encode(["文本内容"], batch_size=16, normalize_embeddings=True)
```

**方式 2：本地 Embedding API 服务**

```bash
# 启动服务
python serve_embedding.py
# 服务地址: http://localhost:8099

# 调用
curl -X POST http://localhost:8099/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["浮力在潜水艇中的应用", "光的本质是什么"]}'
```

返回格式：
```json
{
  "embeddings": [[0.01, -0.03, ...], [0.02, 0.01, ...]],
  "model": "BAAI/bge-m3",
  "dim": 1024,
  "took_ms": 3588.6
}
```

- 健康检查：`GET http://localhost:8099/health`
- 服务脚本：`serve_embedding.py`（FastAPI + uvicorn，端口 8099）
- 模型加载约 12s（离线），之后每次请求 batch_size=16

### 注意事项

- torch >= 2.6 不支持 Python 3.9，当前 pin 在 torch 2.2.2 + transformers < 4.50.0
- 若升级 Python 到 3.10+，可直接 `pip install torch>=2.6` 获取最新 transformers
- 模型文件由 HuggingFace Hub 管理，`HF_HUB_OFFLINE=1` 可强制离线加载
- `data/local_subtitles.csv` 被 gitignore，需手动放置
