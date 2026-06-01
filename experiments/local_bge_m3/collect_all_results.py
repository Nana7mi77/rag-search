import json

files = {
    'V3': 'experiments/local_bge_m3/optimize_v3_results.json',
    'V4': 'experiments/local_bge_m3/optimize_v4_results.json',
    'V5': 'experiments/local_bge_m3/optimize_v5_results.json',
    'V6': 'experiments/local_bge_m3/optimize_v6_results.json',
    'V7': 'experiments/local_bge_m3/optimize_v7_results.json',
    'V8': 'experiments/local_bge_m3/optimize_v8_results.json',
}

queries = [
    '浮力在潜水艇中的应用', '光的本质是什么', '牛顿对光学的贡献', '自然光源有哪些',
    '激光的特点和应用', '欧几里德在光学方面做了什么', '眼睛如何看见物体', '光速是多少',
    '密度计的工作原理', '闪电是怎么形成的', '郑和宝船与航海技术', '反射和折射的区别',
    '人工光源的发展历史', '潜水艇如何上浮和下潜', '光的颜色是怎么产生的',
]

for name, path in files.items():
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', [])
    pq = data.get('per_query', {})
    print(f'=== {name} ({len(results)} methods) ===')
    for r in results:
        print(f'  {r["name"]:45s} MRR={r["mrr"]:.3f} H3={r["hit3"]:.3f} H5={r["hit5"]:.3f}')
    if pq:
        top_method = list(pq.keys())[0]
        pqr = pq[top_method]
        print(f'  Top method [{top_method}] per-query:')
        for i, (q, v) in enumerate(zip(queries, pqr)):
            marker = ' <<< ' if v < 1.0 else ''
            print(f'    Q{i:2d} {q:25s} MRR={v:.3f}{marker}')
    print()
