#!/bin/bash
# launch-workspace.sh - 启动指定成员的Claude Code工作区
# 用法: scripts/launch-workspace.sh <member-name>
# 示例: scripts/launch-workspace.sh algo-dev
#        scripts/launch-workspace.sh be-dev

set -e

MEMBER="${1:?用法: scripts/launch-workspace.sh <member-name>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

declare -A WORKSPACE_MAP=(
    ["owner"]="$REPO_ROOT"
    ["pm-governance"]="$REPO_ROOT"
    ["algo-dev"]="$REPO_ROOT/workspaces/algo-dev"
    ["be-dev"]="$REPO_ROOT/workspaces/be-dev"
    ["fe-dev"]="$REPO_ROOT/workspaces/fe-dev"
    ["verify-scan"]="$REPO_ROOT/workspaces/verify"
    ["triage"]="$REPO_ROOT"
    ["ai-review"]="$REPO_ROOT/workspaces/ai-review"
    ["arch-review"]="$REPO_ROOT"
    ["release-manager"]="$REPO_ROOT"
)

WS_PATH="${WORKSPACE_MAP[$MEMBER]}"
if [ -z "$WS_PATH" ] || [ ! -d "$WS_PATH" ]; then
    echo "Error: 工作区 $MEMBER 不存在"
    echo "可用成员: ${!WORKSPACE_MAP[*]}"
    exit 1
fi

echo "=== 启动 $MEMBER 工作区 ==="
echo "路径: $WS_PATH"
echo "分支: $(cd "$WS_PATH" && git branch --show-current)"
echo ""

cd "$WS_PATH"
exec claude
