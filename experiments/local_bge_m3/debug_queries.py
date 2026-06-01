import csv
from pathlib import Path

DATA_DIR = Path("/Users/ouakira/rag-search/data")
docs = []
with (DATA_DIR / "local_subtitles.csv").open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        text = str(row.get("subtitle", "")).strip()
        name = str(row.get("name", "")).strip()
        if text:
            docs.append({"name": name, "time": row.get("time", ""), "text": text})

print(f"Total docs: {len(docs)}")

queries = [
    ("人工光源的发展历史", ["灯", "LED", "电灯", "人造光", "人工光源", "光源"]),
    ("潜水艇如何上浮和下潜", ["潜水艇", "压载水舱", "浮力", "密度"]),
    ("闪电是怎么形成的", ["闪电", "放电", "云层", "雷"]),
]

for q, keywords in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print(f"Keywords: {keywords}")
    print(f"{'='*60}")
    found = []
    for i, doc in enumerate(docs):
        combined = (doc["name"] + " " + doc["text"]).lower()
        matched_kws = [kw for kw in keywords if kw.lower() in combined]
        if matched_kws:
            found.append((i, doc, matched_kws))
    print(f"Found {len(found)} relevant docs:")
    for idx, doc, matched in found[:8]:
        print(f"  [{idx}] {doc['name']} | {doc['time']}")
        print(f"    Matched: {matched}")
        print(f"    Text: {doc['text'][:100]}")
