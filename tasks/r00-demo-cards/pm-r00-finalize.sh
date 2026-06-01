#!/bin/bash
# pm-r00-finalize.sh - PM收尾调度：验证+提交
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_DIR="$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/pm-finalize-${TS}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "========================================="
log "PM Finalize: r00-demo-cards"
log "========================================="

PROMPT=$(cat "$TASK_DIR/p4-r04-verify-commit.txt")

cd "$REPO_ROOT"
claude -p "$PROMPT" \
    --allowedTools "Edit,Write,Bash,Read,Glob,Grep,LS" \
    --dangerously-skip-permissions \
    2>&1 | tee "$LOG_FILE"

log "=== PM Finalize Complete ==="
