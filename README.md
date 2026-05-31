# 视频字幕知识增强 RAG Agent

这是从研究生时期字幕检索实验整理出来的轻量重构版。项目有两种运行方式：

1. `Python 本地 fallback`：不用装 Elasticsearch / Neo4j，也能先跑通演示。
2. `轻量 ES + Neo4j`：用 Docker 单机部署 Elasticsearch 和 Neo4j，尽量贴近你原来的研究项目结构。

默认 Python 本地版只用标准库完成：

- 本地 BM25/中文 n-gram 检索，替代 ES 的第一版可运行搜索后端
- CSV 知识图谱补强，替代 Neo4j 的概念扩展能力
- CLI + 本地 Web 页面，方便面试展示
- 证据型回答：返回片名、时间戳、字幕片段和图谱扩展词

## 快速开始：Python 本地 fallback

在本目录运行：

```powershell
python -m rag_search build --data data\local_subtitles.csv --graph data\sample_kg.csv
python -m rag_search ask "浮力在潜水艇中的应用"
python -m rag_search.web
```

然后打开：

```text
http://127.0.0.1:7860
```

如果还没有 `data\local_subtitles.csv`，先运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_legacy_data.ps1
python -m rag_search build --data data\local_subtitles.csv --graph data\sample_kg.csv
```

如果系统 `python` 启动异常，直接运行下面这个脚本；它会自动寻找可用的 Python：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_local.ps1
```

也可以双击或运行：

```powershell
scripts\start_demo.bat
```

## 项目结构

```text
rag_search/
  __main__.py      CLI：build/search/ask/stats
  es_backend.py    ES/Neo4j 轻量服务后端
  index.py         本地倒排索引与 BM25
  graph.py         CSV 知识图谱补强
  rag.py           检索、证据组织、抽取式回答
  web.py           无依赖 Web demo
data/
  sample_kg.csv    可编辑的轻量知识图谱
scripts/
  export_legacy_data.ps1  从旧 notebook 目录导出本地语料
  install_local.ps1       创建本地环境并构建索引
  services_up.ps1         启动轻量 ES/Neo4j
  import_services.ps1     导入字幕到 ES，导入图谱到 Neo4j
  start_demo.bat          一键启动 demo
```

## 轻量 ES + Neo4j 模式

这套模式使用 Docker Compose 单机启动：

- Elasticsearch: `http://127.0.0.1:9200`
- Neo4j Browser: `http://127.0.0.1:7474`
- Neo4j 登录：`neo4j / ragsearch123`

当前机器如果没有 Docker Desktop，先运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_docker_desktop.ps1
```

安装完成后重启 PowerShell，再运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\services_up.ps1
powershell -ExecutionPolicy Bypass -File scripts\import_services.ps1
powershell -ExecutionPolicy Bypass -File scripts\search_services.ps1 -Query "自然光源"
```

也可以直接用 CLI：

```powershell
python -m rag_search import-es --data data\local_subtitles.csv
python -m rag_search search-es "自然光源"
```

停止服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\services_down.ps1
```

这不是生产配置：ES 关闭了安全认证，Neo4j 使用固定本地密码，并且端口只绑定 `127.0.0.1`。它的定位是“本机可复现实验环境”。

## 旧项目对应关系

- `section.ipynb`：字幕按时间分段，对应后续可整理成 `chunk` 模块
- `字幕搜索.ipynb`：BERT + HNSW + KG 补强 + Dash，对应现在的 `rag_search`
- `AP.ipynb` / `P@N.ipynb`：P@N、AP、MAP 评估，后续可整理成 `eval`
- Neo4j 图谱：`data\sample_kg.csv` 可直接导入轻量 Neo4j
- ES：`python -m rag_search import-es` 可直接导入轻量 Elasticsearch

## 简历表述草稿

> 构建面向科普纪录片字幕的知识增强 RAG Agent：将 6k+ 分段字幕构建为本地检索索引，结合概念知识图谱进行 query expansion，返回带片名、时间戳和字幕证据的答案；设计 P@N/MAP 评估，对比纯检索与知识增强检索效果。

## 后续升级路线

1. 把旧 notebook 中的数据清洗和分段逻辑整理成可复现脚本。
2. 增加 embedding 后端：`sentence-transformers` + FAISS 或 Chroma。
3. 增加 LLM 生成层：把 `rag_search.rag.answer` 的抽取式回答替换为可引用证据的生成回答。
4. 增加评测模块：复刻旧项目里的 P@N、AP、MAP 曲线。
5. 需要服务化时，再把 ES/Neo4j 接成可选后端，而不是作为默认依赖。
