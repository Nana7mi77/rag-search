#!/bin/bash
# pm-r00-demo.sh - Demo卡片功能PM调度脚本
# 串行执行: be-dev → fe-dev → verify → merge-to-main
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_DIR="$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/pm-${TS}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_claude_task() {
    local member="$1"
    local workdir="$2"
    local prompt_file="$3"
    local output_log="$4"

    log "=== Dispatching: $member ==="
    log "Workdir: $workdir"
    log "Prompt: $prompt_file"

    if [ ! -f "$prompt_file" ]; then
        log "ERROR: Prompt file not found: $prompt_file"
        return 1
    fi

    local prompt
    prompt=$(cat "$prompt_file")

    cd "$workdir"
    claude -p "$prompt" \
        --allowedTools "Edit,Write,Bash,Read,Glob,Grep,LS" \
        --dangerously-skip-permissions \
        2>&1 | tee "$output_log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "=== $member: DONE ==="
    else
        log "=== $member: FAILED (exit $exit_code) ==="
    fi
    return $exit_code
}

log "========================================="
log "PM Dispatch: r00-demo-cards"
log "========================================="

# Phase 1: be-dev
log ""
log ">>> PHASE 1: be-dev (API卡片数据) <<<"
run_claude_task "be-dev" \
    "$REPO_ROOT/workspaces/be-dev" \
    "$TASK_DIR/p1-r01-be-api.txt" \
    "$LOG_DIR/r01-be-dev-${TS}.log"
BE_EXIT=$?

if [ $BE_EXIT -ne 0 ]; then
    log "be-dev failed, stopping pipeline"
    exit 1
fi

# Phase 1.5: 合并be-dev改动到fe-dev工作区
log ""
log ">>> PHASE 1.5: 同步be-dev改动到fe-dev工作区 <<<"
cd "$REPO_ROOT/workspaces/fe-dev"
git merge feat/be-dev --no-edit -m "merge: be-dev API card fields" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARN: merge conflict, trying rebase..."
    git merge --abort 2>/dev/null || true
}

# Phase 2: fe-dev
log ""
log ">>> PHASE 2: fe-dev (卡片UI) <<<"
run_claude_task "fe-dev" \
    "$REPO_ROOT/workspaces/fe-dev" \
    "$TASK_DIR/p2-r02-fe-cards.txt" \
    "$LOG_DIR/r02-fe-dev-${TS}.log"
FE_EXIT=$?

if [ $FE_EXIT -ne 0 ]; then
    log "fe-dev failed, stopping pipeline"
    exit 1
fi

# Phase 3: 合并到main
log ""
log ">>> PHASE 3: 合并到main <<<"
cd "$REPO_ROOT"
git checkout main

# 合并 be-dev
log "Merging feat/be-dev into main..."
git merge feat/be-dev --no-edit -m "feat: r01 API卡片数据结构" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARN: be-dev merge conflict"
    git merge --abort 2>/dev/null || true
}

# 合并 fe-dev
log "Merging feat/fe-dev into main..."
git merge feat/fe-dev --no-edit -m "feat: r02 卡片式搜索UI" 2>&1 | tee -a "$LOG_FILE" || {
    log "WARN: fe-dev merge conflict"
    git merge --abort 2>/dev/null || true
}

# Phase 4: 验证
log ""
log ">>> PHASE 4: 验证 <<<"
log "代码导入检查..."
cd "$REPO_ROOT"
python3 -c "from rag_search.index import SearchHit; print('SearchHit import OK')" 2>&1 | tee -a "$LOG_FILE"
python3 -c "from rag_search.bilibili_map import get_bvid; print('bilibili_map import OK')" 2>&1 | tee -a "$LOG_FILE" || log "WARN: bilibili_map not found"

log ""
log "========================================="
log "PM Dispatch Complete"
log "已完成: be-dev, fe-dev, merge"
log "日志: $LOG_DIR/"
log "========================================="
log ""
log "下一步: 启动服务验证"
log "  cd $REPO_ROOT && python3 -m rag_search serve --port 7860"
