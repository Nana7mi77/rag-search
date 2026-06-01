#!/bin/bash
# merge-pr.sh - 合并PR到main并触发后序同步
# 用法: scripts/merge-pr.sh <pr-number>
# 示例: scripts/merge-pr.sh 3

set -e

PR_NUM="${1:?用法: scripts/merge-pr.sh <pr-number>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Merging PR #$PR_NUM ==="

if command -v gh &>/dev/null; then
    gh pr merge "$PR_NUM" --merge --delete-branch
    echo "=== PR #$PR_NUM merged ==="
else
    echo "gh CLI not found. Please merge manually on GitHub."
    exit 1
fi

echo ""
echo "=== Syncing worktrees ==="
"$REPO_ROOT/scripts/pull.sh" --all

echo ""
echo "=== Done. PR #$PR_NUM merged and worktrees synced ==="
