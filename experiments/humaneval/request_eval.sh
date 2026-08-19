#!/usr/bin/env bash
# ============================================================
# request_eval.sh — Request eval agent to evaluate a checkpoint
# ============================================================
# Non-blocking: queues the request and returns immediately.
# The eval agent picks it up, evaluates, and writes results
# to experiment.jsonl.
#
# Usage:
#   bash request_eval.sh <checkpoint_path>
# ============================================================

set -euo pipefail

CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
    echo "Usage: bash request_eval.sh <checkpoint_path>" >&2
    exit 1
fi

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
QUEUE_FILE="${WORKSPACE}/.eval_queue"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

python3 -c "
import json, sys
entry = {'timestamp': sys.argv[1], 'checkpoint': sys.argv[2]}
with open(sys.argv[3], 'a') as f:
    f.write(json.dumps(entry) + '\n')
" "${TIMESTAMP}" "${CHECKPOINT}" "${QUEUE_FILE}"

echo "[request_eval] Queued for evaluation: ${CHECKPOINT}"
echo "[request_eval] The eval agent will write eval_result and eval_analysis to experiment.jsonl."
echo "[request_eval] Check experiment.jsonl for results before planning your next step."
