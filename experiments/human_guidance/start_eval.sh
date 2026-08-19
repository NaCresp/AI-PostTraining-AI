#!/usr/bin/env bash
# ============================================================
# Eval Agent — Request-Driven Loop
# ============================================================
# Watches .eval_queue for evaluation requests from the main
# agent. Each request triggers a Claude eval session that runs
# predict → evaluate → analyze and writes results to
# experiment.jsonl.
#
# Session management:
#   - A UUID session ID is generated on first launch and
#     persisted to .eval_session_id.
#   - The first checkpoint starts a new Claude session (-p).
#   - Subsequent checkpoints resume the same session (--resume)
#     to preserve prompt cache and cross-checkpoint context.
#   - On failure (timeout / crash), the session is discarded
#     and a fresh one is created for the next attempt.
#
# Usage (inside docker container):
#   bash experiments/claude-code-human/start_eval.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Parse config.yaml ──
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
REPO=$(parse_yaml "$CONFIG" repo_root)
VERL_PATH=$(parse_yaml "$CONFIG" verl_path)
EVALUATOR_SCRIPT=$(parse_yaml "$CONFIG" evaluator_script)
CLAUDE_MODEL=$(parse_yaml "$CONFIG" claude_model)
EVAL_GPU_IDS=$(parse_yaml "$CONFIG" eval_agent_gpu_ids)
EVAL_GPU_IDS="${EVAL_GPU_IDS:-6,7}"

QUEUE_FILE="${WORKSPACE}/.eval_queue"
PROCESSED_FILE="${WORKSPACE}/.eval_processed"
JOURNAL="${WORKSPACE}/experiment.jsonl"
EVAL_PROGRAM="${SCRIPT_DIR}/eval_program.md"
EVAL_LOG_DIR="${WORKSPACE}/eval_agent_logs"
EVAL_TRAJECTORY="${WORKSPACE}/eval_trajectory.jsonl"
SESSION_FILE="${WORKSPACE}/.eval_session_id"
POLL_SECONDS=5
EVAL_TIMEOUT=10800          # 3 hours per eval cycle
MAX_RETRIES=2               # retry failed evals up to this many times

# ── Load API credentials ──
ENV_FILE="${REPO}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export ANTHROPIC_API_KEY="${API_KEY}"
export ANTHROPIC_BASE_URL="${BASE_URL}"

# ── Create eval runner user if needed ──
# EVAL_AGENT_USER can be set to root so the trusted evaluator can read
# root-only benchmark files while the main training agent remains sandboxed.
AGENT_USER="${EVAL_AGENT_USER:-agent}"
if ! id "$AGENT_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$AGENT_USER"
fi
chmod a+x /root
if [[ -d /root/.nvm/versions/node ]]; then
    chmod -R a+rX /root/.nvm
    NODE_BIN="/root/.nvm/versions/node/$(ls /root/.nvm/versions/node/ | head -1)/bin"
    export PATH="${NODE_BIN}:${PATH}"
fi
if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: claude CLI not found. Install Claude Code in the container or mount /root/.nvm before launching." >&2
    echo "Expected one of: /root/.nvm/versions/node/*/bin/claude or claude on PATH." >&2
    exit 1
fi

# ── Setup directories ──
EVAL_DIR="$(dirname "${EVALUATOR_SCRIPT}")"
mkdir -p "${EVAL_DIR}/logs" "${EVAL_DIR}/.merge_cache" "${EVAL_LOG_DIR}"
chown -R "${AGENT_USER}:${AGENT_USER}" "${EVAL_DIR}/logs" "${EVAL_DIR}/.merge_cache"

# ── Environment for evaluation ──
export CUDA_VISIBLE_DEVICES="${EVAL_GPU_IDS}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS}"
export HF_ENDPOINT="https://hf-mirror.com"
VERL_ROOT="$(dirname "${VERL_PATH}")"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ── Initialize state ──
touch "${PROCESSED_FILE}"
chown "${AGENT_USER}:${AGENT_USER}" "${PROCESSED_FILE}"

new_session() {
    SESSION_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
    echo "$SESSION_ID" > "$SESSION_FILE"
    SESSION_STARTED=false
    echo "[eval-agent] $(date): new session ${SESSION_ID}"
}

if [[ -f "$SESSION_FILE" ]]; then
    SESSION_ID=$(cat "$SESSION_FILE")
    SESSION_STARTED=true
    echo "[eval-agent] $(date): resuming existing session ${SESSION_ID}"
else
    new_session
fi

ALLOWED_TOOLS=(
    "Bash(${EVALUATOR_SCRIPT}*)"
    "Bash(nvidia-smi*)"
    "Bash(python3 *)"
    "Bash(ps *)"
    "Bash(kill *)"
    "Read(${WORKSPACE}/*)"
    "Read(${EVAL_DIR}/*)"
    "Write(${WORKSPACE}/experiment.jsonl)"
)
ALLOWED_TOOLS_CSV=$(IFS=,; echo "${ALLOWED_TOOLS[*]}")

# ── Kill residual vLLM / evaluation processes on eval GPUs ──
cleanup_eval_processes() {
    echo "[eval-agent] $(date): cleaning up residual eval processes..."
    pkill -f "vllm.*serve" 2>/dev/null || true
    pkill -f "submit_checkpoint" 2>/dev/null || true
    pkill -f "evaluate.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm.*serve" 2>/dev/null || true
    sleep 1
}

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Eval Agent — Request-Driven (blocking)              ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Workspace:     ${WORKSPACE}"
echo "║  Queue file:    ${QUEUE_FILE}"
echo "║  GPUs:          ${EVAL_GPU_IDS}"
echo "║  Poll interval: ${POLL_SECONDS}s"
echo "║  Timeout:       ${EVAL_TIMEOUT}s"
echo "║  Claude model:  ${CLAUDE_MODEL}"
echo "║  Session ID:    ${SESSION_ID}"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ── Parse a queue line → checkpoint path ──
parse_request() {
    python3 -c "
import json, sys
try:
    entry = json.loads(sys.argv[1])
    print(entry.get(sys.argv[2], ''))
except:
    print('')
" "$1" "$2"
}

# ── Run one eval cycle, returns 0 on success ──
run_eval() {
    local CKPT_ABS="$1"
    local AGENT_LOG="$2"
    local AGENT_EXIT=0

    cleanup_eval_processes

    JOURNAL_CONTEXT=$(tail -n 50 "$JOURNAL" 2>/dev/null || echo "(no journal yet)")

    if ! $SESSION_STARTED; then
        EVAL_INSTRUCTIONS=$(cat "$EVAL_PROGRAM")

        PROMPT=$(cat <<PROMPT_EOF
${EVAL_INSTRUCTIONS}

---

## First Checkpoint

**Checkpoint to evaluate:** \`${CKPT_ABS}\`

**Recent experiment journal (last 50 entries):**

\`\`\`jsonl
${JOURNAL_CONTEXT}
\`\`\`

Begin. Predict, evaluate, analyze this checkpoint.
PROMPT_EOF
)
        echo "[eval-agent] $(date): starting NEW session ${SESSION_ID} ..."

        timeout "${EVAL_TIMEOUT}" \
            runuser -u "${AGENT_USER}" -- \
                claude -p "$PROMPT" \
                    --model "${CLAUDE_MODEL}" \
                    --verbose \
                    --output-format stream-json \
                    --session-id "${SESSION_ID}" \
                    --allowedTools "${ALLOWED_TOOLS_CSV}" \
                    --dangerously-skip-permissions \
            > >(tee "$AGENT_LOG" >> "${EVAL_TRAJECTORY}") 2>&1 \
            || AGENT_EXIT=$?

        if [[ $AGENT_EXIT -eq 0 ]]; then
            SESSION_STARTED=true
        fi
    else
        PROMPT=$(cat <<PROMPT_EOF
New checkpoint to evaluate: \`${CKPT_ABS}\`

Recent experiment journal (last 50 entries):

\`\`\`jsonl
${JOURNAL_CONTEXT}
\`\`\`

Predict, evaluate, analyze this checkpoint.
PROMPT_EOF
)
        echo "[eval-agent] $(date): RESUMING session ${SESSION_ID} ..."

        timeout "${EVAL_TIMEOUT}" \
            runuser -u "${AGENT_USER}" -- \
                claude --resume "${SESSION_ID}" \
                    -p "$PROMPT" \
                    --model "${CLAUDE_MODEL}" \
                    --verbose \
                    --output-format stream-json \
                    --allowedTools "${ALLOWED_TOOLS_CSV}" \
                    --dangerously-skip-permissions \
            > >(tee "$AGENT_LOG" >> "${EVAL_TRAJECTORY}") 2>&1 \
            || AGENT_EXIT=$?
    fi

    cleanup_eval_processes
    return $AGENT_EXIT
}

# ── Retry tracker: CKPT -> attempt count ──
declare -A RETRY_COUNT

# ── Watching loop ──
while true; do
    if [[ ! -f "$QUEUE_FILE" ]]; then
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Find unprocessed requests
    NEW_REQUESTS=""
    while IFS= read -r line; do
        CKPT=$(parse_request "$line" checkpoint)
        if [[ -z "$CKPT" ]]; then
            continue
        fi
        if ! grep -qxF "$CKPT" "$PROCESSED_FILE" 2>/dev/null; then
            NEW_REQUESTS="${NEW_REQUESTS}${line}"$'\n'
        fi
    done < "$QUEUE_FILE"
    NEW_REQUESTS=$(echo -n "$NEW_REQUESTS" | sed '/^$/d')

    if [[ -z "$NEW_REQUESTS" ]]; then
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Process each request
    while IFS= read -r REQUEST_LINE; do
        CKPT=$(parse_request "$REQUEST_LINE" checkpoint)
        if [[ -z "$CKPT" ]]; then
            continue
        fi

        # Resolve absolute path
        if [[ "$CKPT" = /* ]]; then
            CKPT_ABS="$CKPT"
        else
            CKPT_ABS="${WORKSPACE}/${CKPT}"
        fi

        if [[ ! -d "$CKPT_ABS" ]]; then
            echo "[eval-agent] $(date): checkpoint ${CKPT} not found, skipping."
            echo "$CKPT" >> "$PROCESSED_FILE"
            continue
        fi

        ATTEMPT=${RETRY_COUNT["$CKPT"]:-0}
        if [[ $ATTEMPT -ge $MAX_RETRIES ]]; then
            echo "[eval-agent] $(date): ${CKPT} failed ${ATTEMPT} times, giving up."
            echo "$CKPT" >> "$PROCESSED_FILE"
            continue
        fi
        RETRY_COUNT["$CKPT"]=$((ATTEMPT + 1))

        echo "[eval-agent] $(date): evaluating ${CKPT} (attempt $((ATTEMPT + 1))/${MAX_RETRIES})"

        TS=$(date -u +"%Y%m%d_%H%M%S")
        AGENT_LOG="${EVAL_LOG_DIR}/eval_${CKPT//\//_}_${TS}.log"

        if run_eval "$CKPT_ABS" "$AGENT_LOG"; then
            echo "[eval-agent] $(date): ✓ evaluation COMPLETE for ${CKPT}"
            echo "$CKPT" >> "$PROCESSED_FILE"
        else
            EXIT_CODE=$?
            echo "[eval-agent] $(date): ✗ evaluation FAILED for ${CKPT} (exit=${EXIT_CODE})"
            if [[ $EXIT_CODE -eq 124 ]]; then
                echo "[eval-agent] $(date):   cause: timeout after ${EVAL_TIMEOUT}s"
            fi
            echo "[eval-agent] $(date):   resetting session for clean retry..."
            new_session
        fi

    done <<< "$NEW_REQUESTS"

done

echo "[eval-agent] $(date): loop ended."
