#!/bin/bash
# push.sh - 成员分支提交脚本
# 用法: scripts/push.sh <branch-name> [commit-message]
# 示例: scripts/push.sh feat/algo-dev "优化WSF融合权重"

set -e

BRANCH="${1:?用法: scripts/push.sh <branch-name> [commit-message]}"
MSG="${2:-auto: $(date '+%Y-%m-%d %H:%M') commit}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== rag-search push ==="
echo "Branch: $BRANCH"
echo "Message: $MSG"

cd "$REPO_ROOT"

CURRENT=$(git branch --show-current)
if [ "$CURRENT" != "$BRANCH" ]; then
    echo "Switching to $BRANCH ..."
    git checkout "$BRANCH"
fi

git add -A
if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
echo "=== Pushed to origin/$BRANCH ==="
echo ""
echo "下一步: 在GitHub上创建PR合并到main"
echo "  gh pr create --base main --head $BRANCH --title '$MSG'"
