#!/bin/bash
# create-pr.sh - 创建PR并自动关联需求
# 用法: scripts/create-pr.sh <branch> [req-id] [title]
# 示例: scripts/create-pr.sh feat/algo-dev REQ-002 "优化WSF融合权重"

set -e

BRANCH="${1:?用法: scripts/create-pr.sh <branch> [req-id] [title]}"
REQ_ID="${2:-}"
TITLE="${3:-feat: $BRANCH auto PR}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Creating PR: $BRANCH -> main ==="

BODY="## 变更说明
- 分支: \`$BRANCH\`
- 关联需求: ${REQ_ID:-无}
- 时间: $(date '+%Y-%m-%d %H:%M')

## 审核检查清单
- [ ] 任务类型是否跑偏
- [ ] 是否超出授权范围
- [ ] 是否新增平行实现
- [ ] 是否越层
- [ ] 是否引入第二真相源
- [ ] 是否命中审核触发器

## 测试证据
(请补充验证证据)
"

if command -v gh &>/dev/null; then
    gh pr create --base main --head "$BRANCH" --title "$TITLE" --body "$BODY"
    echo "=== PR created ==="
else
    echo "gh CLI not found. Please create PR manually on GitHub."
    echo "URL: https://github.com/Nana7mi77/rag-search/compare/main...$BRANCH"
fi
