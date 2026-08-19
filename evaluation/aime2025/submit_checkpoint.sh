#!/usr/bin/env bash
# Submit a checkpoint for AIME evaluation.
#
# Usage:
#   bash experiments/evaluator/aime/submit_checkpoint.sh <checkpoint_path> [--tp_size N]
#
# Returns JSON with AIME 2025 accuracy to stdout.
# All diagnostic messages go to stderr.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT_PATH="${1:?Usage: submit_checkpoint.sh <checkpoint_path> [--tp_size N]}"
shift
TP_SIZE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

unset CUDA_VISIBLE_DEVICES

exec python3 "${SCRIPT_DIR}/evaluate.py" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --tp_size "${TP_SIZE}"
