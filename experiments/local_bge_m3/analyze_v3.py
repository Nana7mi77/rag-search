import json

with open('experiments/local_bge_m3/optimize_v3_results.json') as f:
    data = json.load(f)

pq = data['per_query']
methods = ['BM25', 'BM25+KG(文本扩展)', 'Vec', 'RRF-V2最优', 'RRF+KG_boost(w=0.2)']
queries = [
    '浮力在潜水艇中的应用', '光的本质是什么', '牛顿对光学的贡献', '自然光源有哪些',
    '激光的特点和应用', '欧几里德在光学方面做了什么', '眼睛如何看见物体', '光速是多少',
    '密度计的工作原理', '闪电是怎么形成的', '郑和宝船与航海技术', '反射和折射的区别',
    '人工光源的发展历史', '潜水艇如何上浮和下潜', '光的颜色是怎么产生的',
]

print("Per-Query MRR 对比:")
print("-" * 90)
header = f"{'Query':<25s} | {'BM25':>5s} | {'BM25+KG':>7s} | {'Vec':>5s} | {'RRF':>5s} | {'RRF+KG':>6s}"
print(header)
print("-" * 90)

weak_queries = []
for i, q in enumerate(queries):
    vals = [pq[m][i] for m in methods]
    line = f"{q[:25]:<25s} | {vals[0]:5.3f} | {vals[1]:7.3f} | {vals[2]:5.3f} | {vals[3]:5.3f} | {vals[4]:6.3f}"
    if any(v < 1.0 for v in vals):
        line += "  <<<"
        weak_queries.append((i, q, vals))
    print(line)

print("-" * 90)
print(f"\n弱查询数: {len(weak_queries)}")
for i, q, vals in weak_queries:
    print(f"  Q{i}: {q}")
    for j, m in enumerate(methods):
        if vals[j] < 1.0:
            print(f"    {m}: MRR={vals[j]:.3f}")
