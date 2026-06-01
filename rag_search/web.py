import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .llm import LLMConfig
from .rag import HybridRagSearchEngine, RagSearchEngine, default_paths


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>字幕知识检索</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f2;
      --panel: #ffffff;
      --text: #202124;
      --muted: #687076;
      --line: #d8ddd3;
      --accent: #315f72;
      --accent-2: #7a5b2e;
      --good: #2f6b4f;
      --danger: #9b3d33;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      width: min(1280px, calc(100vw - 32px));
      margin: 16px auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .side {
      padding: 18px;
      position: sticky;
      top: 16px;
      height: calc(100vh - 32px);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 22px;
      line-height: 1.25;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    textarea, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fbfcf8;
      color: var(--text);
    }
    textarea {
      min-height: 122px;
      resize: vertical;
      line-height: 1.55;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 112px;
      gap: 10px;
      align-items: end;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      color: var(--text);
      font-size: 14px;
    }
    .toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 4px 14px;
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      border: 1px solid transparent;
      border-radius: 6px;
      min-height: 40px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
      border-color: var(--accent);
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .main {
      min-height: calc(100vh - 32px);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .answer {
      padding: 16px 18px;
      white-space: pre-wrap;
      line-height: 1.65;
      border-left: 4px solid var(--good);
    }
    .status {
      color: var(--muted);
      padding: 14px 16px;
      font-size: 14px;
    }
    .expansion {
      padding: 12px 16px;
      color: var(--accent-2);
      font-size: 14px;
      border-left: 4px solid var(--accent-2);
    }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      cursor: pointer;
      transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s;
      display: flex;
      flex-direction: column;
    }
    .card:hover {
      box-shadow: 0 6px 20px rgba(0,0,0,0.13);
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .card-thumb {
      position: relative;
      aspect-ratio: 16 / 9;
      background: linear-gradient(135deg, #1a3a4a 0%, #315f72 50%, #4a2c5e 100%);
      overflow: hidden;
    }
    .card-thumb svg {
      width: 100%;
      height: 100%;
      display: block;
    }
    .card-score {
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(0,0,0,0.55);
      color: #fff;
      font-size: 12px;
      padding: 3px 9px;
      border-radius: 10px;
      font-weight: 600;
      line-height: 1.4;
    }
    .card-body {
      padding: 12px 14px 14px;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      margin: 0 0 4px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
    }
    .card-time {
      font-size: 12px;
      color: var(--accent);
      margin: 0 0 8px;
    }
    .card-snippet {
      font-size: 14px;
      line-height: 1.6;
      color: var(--muted);
      margin: 0;
      flex: 1;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .error {
      color: var(--danger);
      border-left-color: var(--danger);
    }
    @media (max-width: 1020px) {
      .results {
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      }
    }
    @media (max-width: 820px) {
      .shell {
        grid-template-columns: 1fr;
        width: min(100vw - 20px, 720px);
        margin: 10px auto;
      }
      .side {
        position: static;
        height: auto;
      }
      .controls {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 520px) {
      .results {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel side">
      <div>
        <h1>字幕知识检索</h1>
        <div class="meta" id="stats">本地索引</div>
      </div>
      <div>
        <label for="query">查询</label>
        <textarea id="query">浮力在潜水艇中的应用</textarea>
      </div>
      <div class="controls">
        <div class="toggles">
          <label class="toggle"><input id="kg" type="checkbox" checked> 图谱补强</label>
          <label class="toggle"><input id="hybrid" type="checkbox"> 混合检索</label>
          <label class="toggle"><input id="llm" type="checkbox"> LLM 生成</label>
        </div>
        <div>
          <label for="topk">Top K</label>
          <input id="topk" type="number" min="1" max="20" value="8">
        </div>
      </div>
      <div class="actions">
        <button id="search">检索</button>
        <button class="secondary" id="ask">回答</button>
      </div>
    </aside>
    <main class="main">
      <section class="panel status" id="message">输入问题后检索。</section>
      <section class="panel answer" id="answer" hidden></section>
      <section class="panel expansion" id="expansion" hidden></section>
      <section class="results" id="results"></section>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = {
      query: $("query"),
      kg: $("kg"),
      hybrid: $("hybrid"),
      llm: $("llm"),
      topk: $("topk"),
      search: $("search"),
      ask: $("ask"),
      message: $("message"),
      answer: $("answer"),
      expansion: $("expansion"),
      results: $("results"),
      stats: $("stats")
    };

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    function formatTime(seconds) {
      if (seconds == null || isNaN(seconds)) return "";
      const s = Number(seconds);
      const m = Math.floor(s / 60);
      const sec = Math.floor(s % 60);
      return m + ":" + String(sec).padStart(2, "0");
    }

    function setBusy(busy) {
      state.search.disabled = busy;
      state.ask.disabled = busy;
      state.message.textContent = busy ? "处理中..." : "";
    }

    function renderResult(payload) {
      const result = payload.result || payload;
      const hits = result.hits || [];
      const mode = payload.mode || "";
      const searchMode = result.search_mode || "bm25+kg";
      state.answer.hidden = !payload.answer;
      let answerText = payload.answer || "";
      if (mode === "llm") {
        answerText = "🤖 MiMo 生成：\n\n" + answerText;
      } else if (mode === "llm_fallback") {
        answerText = "⚠️ LLM 失败，回退抽取式：\n\n" + answerText;
      }
      state.answer.textContent = answerText;
      state.expansion.hidden = !(result.matched_terms && result.matched_terms.length);
      const modeLabel = searchMode === "hybrid" ? "混合检索" : "BM25+KG";
      state.expansion.textContent = result.matched_terms && result.matched_terms.length
        ? `图谱补强：${result.matched_terms.join("、")} | 模式：${modeLabel}`
        : `模式：${modeLabel}`;
      state.message.textContent = hits.length ? `找到 ${hits.length} 条证据` : "没有找到匹配证据";
      state.results.innerHTML = hits.map((hit, index) => {
        const docName = escapeHtml(hit.doc_name || hit.title || `doc-${hit.doc_id}`);
        const timeLabel = escapeHtml(hit.time || "");
        const startTime = formatTime(hit.start_seconds);
        const snippet = escapeHtml(hit.snippet || hit.text || "");
        const score = Number(hit.score || 0).toFixed(3);
        const bilibiliUrl = (hit.bilibili_url || "").replace(/'/g, "\\'");
        const thumbText = docName.length > 14 ? docName.slice(0, 14) + "..." : docName;

        return `
        <article class="card" onclick="${bilibiliUrl ? `window.open('${bilibiliUrl}', '_blank')` : ""}" title="${bilibiliUrl ? "点击跳转B站观看" : ""}">
          <div class="card-thumb">
            <svg viewBox="0 0 16 9" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="tg${index}" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#1a3a4a"/>
                  <stop offset="50%" stop-color="#315f72"/>
                  <stop offset="100%" stop-color="#4a2c5e"/>
                </linearGradient>
              </defs>
              <rect width="16" height="9" fill="url(#tg${index})"/>
              <text x="8" y="3.8" text-anchor="middle" fill="rgba(255,255,255,0.88)" font-size="1.2" font-weight="bold" font-family="Microsoft YaHei, sans-serif">${thumbText}</text>
              <text x="8" y="5.8" text-anchor="middle" fill="rgba(255,255,255,0.6)" font-size="0.75" font-family="Microsoft YaHei, sans-serif">${timeLabel}</text>
              <polygon points="7,3.6 7,5.4 9,4.5" fill="rgba(255,255,255,0.45)"/>
            </svg>
            <span class="card-score">${score}</span>
          </div>
          <div class="card-body">
            <h3 class="card-title" title="${escapeHtml(hit.doc_name || hit.title || "")}">${docName}</h3>
            <p class="card-time">${startTime ? "🕐 " + startTime : ""}</p>
            <p class="card-snippet">${snippet}</p>
          </div>
        </article>`;
      }).join("");
    }

    async function run(mode) {
      const q = state.query.value.trim();
      if (!q) {
        state.message.textContent = "请输入查询。";
        return;
      }
      setBusy(true);
      state.answer.hidden = true;
      try {
        const params = new URLSearchParams({
          q,
          k: state.topk.value || "8",
          kg: state.kg.checked ? "1" : "0",
          hybrid: state.hybrid.checked ? "1" : "0",
          llm: state.llm.checked ? "1" : "0"
        });
        const res = await fetch(`/api/${mode}?${params.toString()}`);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.error || "请求失败");
        renderResult(payload);
      } catch (err) {
        state.message.textContent = err.message;
        state.message.classList.add("error");
      } finally {
        setBusy(false);
      }
    }

    state.search.addEventListener("click", () => run("search"));
    state.ask.addEventListener("click", () => run("ask"));
    fetch("/api/stats").then((res) => res.json()).then((data) => {
      state.stats.textContent = `${data.documents} 条字幕段 · ${data.titles} 个片名 · ${data.graph_terms} 个图谱词条`;
    }).catch(() => {});
    run("search");
  </script>
</body>
</html>
"""


class SearchHandler(BaseHTTPRequestHandler):
    engine: RagSearchEngine

    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path.startswith("/api/"):
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            top_k = int(params.get("k", ["8"])[0] or 8)
            use_graph = params.get("kg", ["1"])[0] != "0"
            use_hybrid = params.get("hybrid", ["0"])[0] != "0"
            use_llm = params.get("llm", ["0"])[0] != "0"
            try:
                if parsed.path == "/api/search":
                    if use_hybrid and isinstance(self.engine, HybridRagSearchEngine):
                        result = self.engine.search(query, top_k=top_k, use_graph=use_graph, use_hybrid=True)
                    else:
                        result = self.engine.search(query, top_k=top_k, use_graph=use_graph)
                    self._json(result.to_dict())
                elif parsed.path == "/api/ask":
                    if use_hybrid and isinstance(self.engine, HybridRagSearchEngine):
                        payload = self.engine.answer(query, top_k=top_k, use_graph=use_graph, use_llm=use_llm, use_hybrid=True)
                    else:
                        payload = self.engine.answer(query, top_k=top_k, use_graph=use_graph, use_llm=use_llm)
                    self._json(payload)
                elif parsed.path == "/api/stats":
                    titles = {doc.title for doc in self.engine.index.documents if doc.title}
                    self._json(
                        {
                            "documents": len(self.engine.index.documents),
                            "titles": len(titles),
                            "graph_terms": len(self.engine.graph.entries),
                        }
                    )
                else:
                    self._json({"error": "Not found"}, status=404)
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)
            return

        self._json({"error": "Not found"}, status=404)


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--index", default=paths["index"])
    parser.add_argument("--graph", default=paths["graph"])
    parser.add_argument("--hybrid", action="store_true", help="启用混合检索 (BM25+KG+Vector)")
    parser.add_argument("--hf-api-key", default="", help="HuggingFace API Key")
    parser.add_argument("--vector-cache", default="", help="向量缓存路径")
    parser.add_argument("--proxy", default="http://127.0.0.1:7890", help="代理地址")
    args = parser.parse_args()

    if not Path(args.index).exists():
        raise SystemExit(f"索引不存在：{args.index}。请先运行 python -m rag_search build --data data\\local_subtitles.csv")

    if args.hybrid:
        engine = HybridRagSearchEngine.load(args.index, args.graph)
        hf_key = args.hf_api_key or os.environ.get("HF_API_KEY", "")
        if hf_key:
            cache_path = args.vector_cache or str(Path(args.index).parent / "vector_cache.json")
            engine.configure_vector(
                hf_key,
                proxy=args.proxy if args.proxy != "none" else None,
                cache_path=cache_path,
            )
            print(f"混合检索已启用，向量维度: {len(engine.vector_index.embeddings[0]) if engine.vector_index.embeddings else 0}")
            print(f"融合策略: score (BM25:0.3 + Vector:0.7)")
        else:
            print("警告: 未提供 HF_API_KEY，混合检索不可用，回退到 BM25+KG")
    else:
        engine = RagSearchEngine.load(args.index, args.graph)

    SearchHandler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    print(f"Web demo: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
