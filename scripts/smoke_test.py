import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_search.rag import RagSearchEngine


def main() -> None:
    index = Path("data/index.json")
    graph = Path("data/sample_kg.csv")
    if not index.exists():
        raise SystemExit("data/index.json does not exist. Run build first.")
    engine = RagSearchEngine.load(str(index), str(graph))
    result = engine.search("浮力在潜水艇中的应用", top_k=3)
    assert result.hits, "expected at least one hit"
    answer = engine.answer("浮力在潜水艇中的应用", top_k=3)
    assert "证据" in answer["answer"] or "字幕证据" in answer["answer"]
    print("smoke ok")


if __name__ == "__main__":
    main()
