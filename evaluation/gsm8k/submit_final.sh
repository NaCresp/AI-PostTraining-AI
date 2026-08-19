#!/usr/bin/env bash
# Submit a final checkpoint for GSM8K evaluation and declare the experiment finished.
#
# Usage:
#   bash experiments/evaluator/gsm8k/submit_final.sh <checkpoint_path> "<reason>" [--tp_size N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

CHECKPOINT_PATH="${1:?Usage: submit_final.sh <checkpoint_path> \"<reason>\" [--tp_size N]}"
REASON="${2:?Usage: submit_final.sh <checkpoint_path> \"<reason>\" [--tp_size N]}"
shift 2
TP_SIZE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

EVAL_JSON=$(bash "${SCRIPT_DIR}/submit_checkpoint.sh" "${CHECKPOINT_PATH}" --tp_size "${TP_SIZE}")

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

python3 -c "
import json, sys
results = json.loads(sys.argv[1])
final = {
    'checkpoint_path': sys.argv[2],
    'reason': sys.argv[3],
    'timestamp': sys.argv[4],
    'results': results,
    'is_final': True,
}
path = sys.argv[5]
with open(path, 'w') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
print(f'[submit_final] Final submission recorded to {path}', file=sys.stderr)
print(json.dumps(results, indent=2, ensure_ascii=False))
" "${EVAL_JSON}" "${CHECKPOINT_PATH}" "${REASON}" "${TIMESTAMP}" "${LOG_DIR}/final_submission.json"
