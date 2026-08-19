#!/usr/bin/env bash
# Show remaining experiment time.
#
# Usage (from workspace):
#   bash /workspace/AI4AI/experiments/claude-code-humaneval/timer.sh
#
# Reads .timer file written by start.sh at experiment launch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

parse_yaml() {
    python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get(sys.argv[2], ''))
" "$1" "$2"
}

CONFIG="${SCRIPT_DIR}/config.yaml"
WORKSPACE=$(parse_yaml "$CONFIG" workspace)
TIMER_FILE="${WORKSPACE}/.timer"

if [[ ! -f "${TIMER_FILE}" ]]; then
    echo "ERROR: Timer not started (no .timer file). Is the experiment running?" >&2
    exit 1
fi

START_EPOCH=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    t = json.load(f)
print(t['start_epoch'])
" "${TIMER_FILE}")

DEADLINE_EPOCH=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    t = json.load(f)
print(t['deadline_epoch'])
" "${TIMER_FILE}")

BUDGET_HOURS=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    t = json.load(f)
print(t['budget_hours'])
" "${TIMER_FILE}")

NOW=$(date +%s)
ELAPSED=$((NOW - START_EPOCH))
REMAINING=$((DEADLINE_EPOCH - NOW))

if [[ ${REMAINING} -le 0 ]]; then
    echo "TIME IS UP. Budget: ${BUDGET_HOURS}h. Elapsed: $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60 ))m."
    exit 0
fi

REM_H=$((REMAINING / 3600))
REM_M=$(( (REMAINING % 3600) / 60 ))
REM_S=$((REMAINING % 60))
ELA_H=$((ELAPSED / 3600))
ELA_M=$(( (ELAPSED % 3600) / 60 ))

echo "Elapsed: ${ELA_H}h ${ELA_M}m | Remaining: ${REM_H}h ${REM_M}m ${REM_S}s | Budget: ${BUDGET_HOURS}h"
