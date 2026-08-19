#!/usr/bin/env bash
# Show remaining autonomous experiment time for claude-code-human.
#
# Usage:
#   bash /workspace/AI4AI/experiments/claude-code-human/timer.sh
#
# The .timer file is written only after the human returns {"option":"keep"}.

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
    echo 'Autonomous timer has not started yet. It starts after the human returns {"option":"keep","reason":"..."}.' >&2
    exit 1
fi

read_timer_field() {
    local field="$1"
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    t = json.load(f)
print(t[sys.argv[2]])
" "${TIMER_FILE}" "${field}"
}

START_EPOCH=$(read_timer_field start_epoch)
DEADLINE_EPOCH=$(read_timer_field deadline_epoch)
BUDGET_HOURS=$(read_timer_field budget_hours)
CLOCK=$(read_timer_field clock)

NOW=$(date +%s)
ELAPSED=$((NOW - START_EPOCH))
REMAINING=$((DEADLINE_EPOCH - NOW))

if [[ ${REMAINING} -le 0 ]]; then
    echo "TIME IS UP. Clock: ${CLOCK}. Budget: ${BUDGET_HOURS}h. Autonomous elapsed: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m."
    exit 0
fi

REM_H=$((REMAINING / 3600))
REM_M=$(((REMAINING % 3600) / 60))
REM_S=$((REMAINING % 60))
ELA_H=$((ELAPSED / 3600))
ELA_M=$(((ELAPSED % 3600) / 60))
ELA_S=$((ELAPSED % 60))

echo "Autonomous elapsed: ${ELA_H}h ${ELA_M}m ${ELA_S}s | Remaining: ${REM_H}h ${REM_M}m ${REM_S}s | Budget: ${BUDGET_HOURS}h | Clock: ${CLOCK}"
