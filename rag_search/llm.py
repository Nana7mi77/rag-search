import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, request


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "mimo-v2.5-pro"
    temperature: float = 0.6
    max_tokens: int = 2048
    timeout: int = 60

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("MIMO_API_KEY", "")
        env_base = os.environ.get("MIMO_BASE_URL", "")
        if env_base:
            self.base_url = env_base
        if not self.base_url:
            if self.api_key.startswith("tp-"):
                self.base_url = "https://token-plan-cn.xiaomimimo.com/v1"
            else:
                self.base_url = "https://api.xiaomimimo.com/v1"
        env_model = os.environ.get("MIMO_MODEL", "")
        if env_model:
            self.model = env_model


@dataclass
class ChatMessage:
    role: str
    content: str


class MiMoLLM:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _make_opener(self) -> request.OpenerDirector:
        proxy_handler = request.ProxyHandler({})
        https_handler = request.HTTPSHandler(context=self._ssl_ctx)
        return request.build_opener(proxy_handler, https_handler)

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps({
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }).encode("utf-8")

        req = request.Request(url, data=body, headers=headers, method="POST")
        opener = self._make_opener()
        try:
            with opener.open(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"MiMo API error {e.code}: {detail}") from e

    def generate_answer(
        self,
        query: str,
        evidence_hits: List[Dict[str, object]],
        matched_terms: List[str],
    ) -> str:
        evidence_lines = []
        for i, hit in enumerate(evidence_hits[:5], start=1):
            source = hit.get("title") or f"doc-{hit.get('doc_id', '?')}"
            time = hit.get("time", "")
            snippet = hit.get("snippet") or hit.get("text", "")
            header = f"[{i}] {source}"
            if time:
                header += f" ({time})"
            evidence_lines.append(f"{header}\n{snippet}")

        evidence_block = "\n\n".join(evidence_lines)
        kg_hint = ""
        if matched_terms:
            kg_hint = f"\n\n知识图谱补强命中了以下概念：{'、'.join(matched_terms)}。这些概念与查询相关，回答时可参考。"

        system_prompt = (
            "你是一个面向科普纪录片字幕的知识检索助手。"
            "用户会提出问题，系统会检索相关字幕片段作为证据。"
            "你的任务是根据这些证据，生成一段简洁、准确、有条理的回答。\n"
            "要求：\n"
            "1. 严格基于提供的证据回答，不要编造证据中没有的信息\n"
            "2. 引用证据时注明来源编号，如 [1] [2]\n"
            "3. 如果证据不足以回答问题，如实说明\n"
            "4. 回答使用中文，语言自然流畅\n"
            "5. 保持在 200 字以内，突出关键信息"
        )

        user_prompt = (
            f"用户问题：{query}{kg_hint}\n\n"
            f"检索到的字幕证据：\n\n{evidence_block}\n\n"
            "请根据以上证据回答用户的问题。"
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        return self.chat(messages)


def llm_answer(
    query: str,
    evidence_hits: List[Dict[str, object]],
    matched_terms: List[str],
    *,
    config: Optional[LLMConfig] = None,
) -> str:
    llm = MiMoLLM(config)
    return llm.generate_answer(query, evidence_hits, matched_terms)
