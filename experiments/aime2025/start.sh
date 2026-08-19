#!/usr/bin/env bash
# ============================================================
# claude-code-aime — Launch Script
# ============================================================
# Reads config.yaml, sets up workspace, launches Claude Code
# with timing enforcement and scoped tool permissions.
#
# Agent trajectory is saved to workspace/trajectory.jsonl for
# post-hoc analysis. After the run, all workspace artifacts
# are archived to run/claude-code-aime/run_N/.
#
# Usage (inside docker container):
#   bash experiments/claude-code-aime/start.sh
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

BASE_MODEL_PATH=$(parse_yaml "$CONFIG" base_model_path)
WORKSPACE=$(parse_yaml "$CONFIG" workspace)
REPO=$(parse_yaml "$CONFIG" repo_root)
VERL_PATH=$(parse_yaml "$CONFIG" verl_path)
GPU_IDS=$(parse_yaml "$CONFIG" gpu_ids)
NUM_GPUS=$(parse_yaml "$CONFIG" num_gpus)
TIME_BUDGET_HOURS=$(parse_yaml "$CONFIG" time_budget_hours)
EVALUATOR_SCRIPT=$(parse_yaml "$CONFIG" evaluator_script)
REQUEST_EVAL_SCRIPT=$(parse_yaml "$CONFIG" request_eval_script)
FINAL_SCRIPT=$(parse_yaml "$CONFIG" final_script)
EVAL_TP_SIZE=$(parse_yaml "$CONFIG" eval_tp_size)
CLAUDE_MODEL=$(parse_yaml "$CONFIG" claude_model)
INITIAL_PROMPT=$(parse_yaml "$CONFIG" initial_prompt)

TIME_BUDGET_SECONDS=$((TIME_BUDGET_HOURS * 3600))

# ── Run archive directory ──
RUN_ARCHIVE_BASE="${REPO}/run/claude-code-aime"
mkdir -p "${RUN_ARCHIVE_BASE}"

# ── Load API credentials ──
ENV_FILE="${REPO}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export ANTHROPIC_API_KEY="${API_KEY}"
export ANTHROPIC_BASE_URL="${BASE_URL}"

# ── Create non-root user (claude CLI refuses --dangerously-skip-permissions as root) ──
AGENT_USER="agent"
if ! id "$AGENT_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$AGENT_USER"
fi
chmod a+x /root
chmod -R a+rX /root/.nvm
NODE_BIN="/root/.nvm/versions/node/$(ls /root/.nvm/versions/node/ | head -1)/bin"

# ── Setup workspace and evaluator logs ──
mkdir -p "${WORKSPACE}"
cp "${SCRIPT_DIR}/program.md" "${WORKSPACE}/program.md"
SKILLS_SEED="${SCRIPT_DIR}/.claude/skills"
if [[ -d "${SKILLS_SEED}" ]]; then
    mkdir -p "${WORKSPACE}/.claude"
    cp -r "${SKILLS_SEED}" "${WORKSPACE}/.claude/skills"
    chmod -R a+rx "${WORKSPACE}/.claude/skills"
    find "${WORKSPACE}/.claude/skills" -type f -name '*.sh' -exec chmod a+x {} \;
    find "${WORKSPACE}/.claude/skills" -type f -name '*.py' -exec chmod a+rx {} \;
fi
chown -R "${AGENT_USER}:${AGENT_USER}" "${WORKSPACE}"

EVAL_DIR="$(dirname "${EVALUATOR_SCRIPT}")"
mkdir -p "${EVAL_DIR}/logs" "${EVAL_DIR}/.merge_cache"
chown -R "${AGENT_USER}:${AGENT_USER}" "${EVAL_DIR}/logs" "${EVAL_DIR}/.merge_cache"

# ── Set GPU visibility, HuggingFace mirror, and Python path for verl ──
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_ENDPOINT="https://hf-mirror.com"
export ENABLE_PROMPT_CACHING_1H=1

VERL_ROOT="$(dirname "${VERL_PATH}")"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

# ── Trajectory file — captures full agent interaction stream ──
TRAJECTORY_FILE="${WORKSPACE}/trajectory.jsonl"

# ── Print banner ──
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  AI4AI: claude-code-aime                              ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Base model:    ${BASE_MODEL_PATH}"
echo "║  GPUs:          ${NUM_GPUS} (${GPU_IDS})"
echo "║  Time budget:   ${TIME_BUDGET_HOURS}h"
echo "║  Workspace:     ${WORKSPACE}"
echo "║  Trajectory:    ${TRAJECTORY_FILE}"
echo "║  verl source:   ${VERL_PATH} (read-only)"
echo "║  Skills:        ${WORKSPACE}/.claude/skills/ (read/write)"
echo "║  Claude model:  ${CLAUDE_MODEL}"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# ── Record start time ──
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[$(date)] Experiment started. Time budget: ${TIME_BUDGET_HOURS}h (${TIME_BUDGET_SECONDS}s)"

# ── Launch Claude Code as non-root user ──
# --output-format stream-json emits one JSON object per line (tool calls,
# responses, thinking, results). We tee it to trajectory.jsonl for analysis
# while still printing to stderr for live monitoring.
cd "${WORKSPACE}"

EXIT_CODE=0
export PATH="${NODE_BIN}:${PATH}"

timeout "${TIME_BUDGET_SECONDS}" \
    runuser -u "${AGENT_USER}" -- \
        claude -p "${INITIAL_PROMPT}" \
            --model "${CLAUDE_MODEL}" \
            --verbose \
            --output-format stream-json \
            --allowedTools \
                "Skill" \
                "Bash(${WORKSPACE}/*)" \
                "Bash(${WORKSPACE}/.claude/skills/*/scripts/*)" \
                "Bash(${REQUEST_EVAL_SCRIPT}*)" \
                "Bash(${FINAL_SCRIPT}*)" \
                "Bash(nvidia-smi*)" \
                "Bash(pip *)" \
                "Bash(python3 *)" \
                "Read(${WORKSPACE}/*)" \
                "Read(${VERL_PATH}/trainer/*)" \
                "Read(${VERL_PATH}/utils/reward_score/*)" \
                "Read(${BASE_MODEL_PATH}/*)" \
                "Write(${WORKSPACE}/*)" \
            --dangerously-skip-permissions \
    > "${TRAJECTORY_FILE}" 2>&1 \
    || EXIT_CODE=$?

# ── Record end time ──
END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_H=$((ELAPSED / 3600))
ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
ELAPSED_S=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Experiment finished."
echo "  Start:    ${START_ISO}"
echo "  End:      ${END_ISO}"
echo "  Elapsed:  ${ELAPSED_H}h ${ELAPSED_M}m ${ELAPSED_S}s"
echo "  Trajectory: ${TRAJECTORY_FILE}"
if [[ $EXIT_CODE -eq 124 ]]; then
    echo "  Status:   TIMED OUT (${TIME_BUDGET_HOURS}h budget reached)"
elif [[ $EXIT_CODE -eq 0 ]]; then
    echo "  Status:   COMPLETED (agent exited normally)"
else
    echo "  Status:   EXITED (code=${EXIT_CODE})"
fi
echo "════════════════════════════════════════════════════════"

# ── Save run metadata ──
python3 -c "
import json, sys
meta = {
    'baseline': 'claude-code-aime',
    'start': sys.argv[1],
    'end': sys.argv[2],
    'elapsed_seconds': int(sys.argv[3]),
    'exit_code': int(sys.argv[4]),
    'timed_out': int(sys.argv[4]) == 124,
    'config': {
        'base_model_path': sys.argv[5],
        'num_gpus': int(sys.argv[6]),
        'time_budget_hours': int(sys.argv[7]),
        'claude_model': sys.argv[8],
        'verl_path': sys.argv[9],
    }
}
path = sys.argv[10] + '/run_metadata.json'
with open(path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Run metadata saved to {path}')
" "${START_ISO}" "${END_ISO}" "${ELAPSED}" "${EXIT_CODE}" \
  "${BASE_MODEL_PATH}" "${NUM_GPUS}" "${TIME_BUDGET_HOURS}" \
  "${CLAUDE_MODEL}" "${VERL_PATH}" "${WORKSPACE}"

TRAJ_LINES=$(wc -l < "${TRAJECTORY_FILE}" 2>/dev/null || echo 0)
echo ""
echo "  Trajectory saved: ${TRAJECTORY_FILE} (${TRAJ_LINES} lines)"
echo "  Run metadata:     ${WORKSPACE}/run_metadata.json"
echo ""
echo "  To archive this run:"
echo "    rsync -a --exclude='checkpoints/' --exclude='data/' --exclude='__pycache__/' --exclude='outputs/' ${WORKSPACE}/ ${RUN_ARCHIVE_BASE}/run_N/"

exit ${EXIT_CODE}
