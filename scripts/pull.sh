#!/bin/bash
# pull.sh - 拉取最新代码并同步所有工作区
# 用法: scripts/pull.sh [branch]
# 示例: scripts/pull.sh main
#        scripts/pull.sh --all

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

pull_branch() {
    local branch="$1"
    echo "--- Pulling $branch ---"
    git fetch origin "$branch"
    CURRENT=$(git branch --show-current)
    if [ "$CURRENT" = "$branch" ]; then
        git merge "origin/$branch" --no-edit
    else
        git branch -f "$branch" "origin/$branch"
    fi
    echo "  ✓ $branch updated"
}

sync_worktree() {
    local worktree_path="$1"
    local branch="$2"
    if [ -d "$worktree_path" ]; then
        echo "--- Syncing worktree: $worktree_path ($branch) ---"
        cd "$worktree_path"
        git merge "origin/$branch" --no-edit 2>/dev/null || {
            echo "  ⚠ Merge conflict in $worktree_path, trying rebase..."
            git merge --abort 2>/dev/null || true
            git rebase "origin/$branch" 2>/dev/null || echo "  ⚠ Rebase failed, manual resolution needed"
        }
        cd "$REPO_ROOT"
        echo "  ✓ $worktree_path synced"
    fi
}

if [ "$1" = "--all" ]; then
    echo "=== Pulling all branches ==="
    git fetch origin --prune
    pull_branch main

    for ws in workspaces/*/; do
        if [ -d "$ws" ]; then
            branch=$(cd "$ws" && git branch --show-current 2>/dev/null || echo "")
            if [ -n "$branch" ]; then
                pull_branch "$branch"
                sync_worktree "$ws" "$branch"
            fi
        fi
    done
    echo ""
    echo "=== All branches synced ==="
elif [ -n "$1" ]; then
    pull_branch "$1"
    echo "=== Done ==="
else
    echo "=== Pulling main ==="
    pull_branch main
    echo ""
    echo "=== Syncing worktrees ==="
    for ws in workspaces/*/; do
        if [ -d "$ws" ]; then
            branch=$(cd "$ws" && git branch --show-current 2>/dev/null || echo "")
            if [ -n "$branch" ]; then
                sync_worktree "$ws" "$branch"
            fi
        fi
    done
    echo ""
    echo "=== Done ==="
fi
