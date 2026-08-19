#!/usr/bin/env bash
# ============================================================
# claude-code-human - Launch Script
# ============================================================
# RQ3 human-guidance condition for AIME.
#
# Flow:
#   1. Set up workspace and run a planning-only Claude session.
#   2. Wait for a human JSON decision: {"option":"keep|revert","reason":"..."}.
#   3. Repeat planning on revert.
#   4. Start the 10h autonomous timer only after keep.
#
# Usage (inside docker container):
#   bash experiments/claude-code-human/start.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
CLAUDE_MODEL=$(parse_yaml "$CONFIG" claude_model)
INITIAL_PROMPT=$(parse_yaml "$CONFIG" initial_prompt)
PLANNING_MAX_ITERATIONS=$(parse_yaml "$CONFIG" planning_max_iterations)
PLANNING_MAX_ITERATIONS="${PLANNING_MAX_ITERATIONS:-5}"

TIME_BUDGET_SECONDS=$((TIME_BUDGET_HOURS * 3600))
RUN_ARCHIVE_BASE="${REPO}/run/claude-code-human"
mkdir -p "${RUN_ARCHIVE_BASE}"

ENV_FILE="${REPO}/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

export ANTHROPIC_API_KEY="${API_KEY}"
export ANTHROPIC_BASE_URL="${BASE_URL}"

AGENT_USER="agent"
if ! id "$AGENT_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$AGENT_USER"
fi
chmod a+x /root
if [[ -d /root/.nvm/versions/node ]]; then
    chmod -R a+rX /root/.nvm
    NODE_BIN=$(python3 - <<'PY_NODE'
from pathlib import Path
nodes = sorted(Path('/root/.nvm/versions/node').iterdir())
print(nodes[0] / 'bin')
PY_NODE
)
    export PATH="${NODE_BIN}:${PATH}"
fi
if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: claude CLI not found. Install Claude Code in the container or mount /root/.nvm before launching." >&2
    echo "Expected one of: /root/.nvm/versions/node/*/bin/claude or claude on PATH." >&2
    exit 1
fi

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

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_ENDPOINT="https://hf-mirror.com"
export ENABLE_PROMPT_CACHING_1H=1
VERL_ROOT="$(dirname "${VERL_PATH}")"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"


ACCESS_GUARD_FILE="${WORKSPACE}/.access_guard_modes.jsonl"

restore_history_paths() {
    if [[ -f "${ACCESS_GUARD_FILE}" ]]; then
        python3 - "${ACCESS_GUARD_FILE}" <<'PY_RESTORE_ACCESS' || true
import json, os, sys
from pathlib import Path
entries = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.strip():
        entries.append(json.loads(line))
# Restore deepest paths first, then parents.
for entry in sorted(entries, key=lambda e: len(e["path"].split("/")), reverse=True):
    path = Path(entry["path"])
    if not path.exists():
        continue
    try:
        os.chown(path, int(entry["uid"]), int(entry["gid"]))
    except PermissionError:
        pass
    os.chmod(path, int(entry["mode"], 8))
PY_RESTORE_ACCESS
    fi
}

protect_history_paths() {
    python3 - "${REPO}" "${WORKSPACE}" "${ACCESS_GUARD_FILE}" <<'PY_PROTECT_ACCESS'
import json, os, stat, sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
workspace = Path(sys.argv[2]).resolve()
record_path = Path(sys.argv[3])
current_experiment = workspace.parent.resolve()
allowed_experiment_names = {current_experiment.name, "evaluator"}

candidates = []

def add(path):
    try:
        p = Path(path).resolve()
    except FileNotFoundError:
        return
    if not p.exists():
        return
    # Never lock the active experiment workspace or its parents.
    try:
        if p == workspace or workspace.is_relative_to(p):
            return
    except AttributeError:
        if str(workspace).startswith(str(p) + os.sep) or str(workspace) == str(p):
            return
    candidates.append(p)

# Deny top-level repo content except the current experiment and evaluator entrypoints.
for child in repo.iterdir():
    if child.name == "experiments":
        continue
    # The agent should not read project docs, env files, run archives, benchmarks, training recipes, wiki, etc.
    add(child)

experiments_dir = repo / "experiments"
if experiments_dir.exists():
    for child in experiments_dir.iterdir():
        if child.name in allowed_experiment_names:
            continue
        add(child)
    # Deny stray files under experiments/, e.g. paper.md.
    for child in experiments_dir.iterdir():
        if child.is_file() or child.is_symlink():
            add(child)

# Deny pass@8 artifacts if any reappear in the active experiment directory.
for pattern in ("pass8*", "*pass@8*", "run_pass8_eval.py"):
    for child in current_experiment.glob(pattern):
        add(child)

# Make /root traversable but not listable; exact /root/models/... remains accessible.
root = Path("/root")
if root.exists():
    candidates.append(root)
for extra in (Path("/root/data"), Path("/data"), Path("/workspace/data")):
    add(extra)

seen = set()
record_path.parent.mkdir(parents=True, exist_ok=True)
with record_path.open("w", encoding="utf-8") as f:
    for path in candidates:
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        st = path.lstat()
        mode = stat.S_IMODE(st.st_mode)
        f.write(json.dumps({"path": resolved, "uid": st.st_uid, "gid": st.st_gid, "mode": format(mode, "04o")}) + "\n")
        if path == root:
            os.chmod(path, 0o711)
            continue
        os.chown(path, 0, 0)
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
PY_PROTECT_ACCESS
    chown "${AGENT_USER}:${AGENT_USER}" "${ACCESS_GUARD_FILE}" 2>/dev/null || true
    trap restore_history_paths EXIT
}

verify_agent_isolation() {
    local failed=0
    local denied_paths=(
        "${REPO}/run"
        "${REPO}/benchmark"
        "${REPO}/training"
        "${REPO}/wiki"
        "${REPO}/.env"
        "${REPO}/experiments/paper.md"
        "${REPO}/experiments/claude-code-gsm8k"
        "${REPO}/experiments/claude-code-humaneval"
        "${REPO}/experiments/claude-code-aime"
    )
    for path in "${denied_paths[@]}"; do
        if [[ -e "$path" ]]; then
            if runuser -u "${AGENT_USER}" -- test -r "$path" 2>/dev/null; then
                echo "ERROR: isolation failed; agent can read forbidden path: $path" >&2
                failed=1
            fi
        fi
    done
    if [[ $failed -ne 0 ]]; then
        exit 3
    fi
}

protect_history_paths
verify_agent_isolation

TRAJECTORY_FILE="${WORKSPACE}/trajectory.jsonl"
PLANNING_TRAJECTORY_FILE="${WORKSPACE}/planning_trajectory.jsonl"
HUMAN_GUIDANCE_FILE="${WORKSPACE}/human_guidance.jsonl"
JOURNAL_FILE="${WORKSPACE}/experiment.jsonl"
TIMER_SCRIPT="${SCRIPT_DIR}/timer.sh"
VALIDATOR_SCRIPT="${SCRIPT_DIR}/validate_human_decision.py"

append_event() {
    local target_file="$1"
    local event_type="$2"
    local payload_json="$3"
    python3 - "$target_file" "$event_type" "$payload_json" <<'PY_EVENT'
import json, sys, time
path, event_type, payload_raw = sys.argv[1:4]
payload = json.loads(payload_raw)
entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "type": event_type}
entry.update(payload)
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY_EVENT
}

append_decision_event() {
    local target_file="$1"
    local event_type="$2"
    local iteration="$3"
    local plan_file="$4"
    local decision_file="$5"
    python3 - "$target_file" "$event_type" "$iteration" "$plan_file" "$decision_file" <<'PY_DECISION'
import json, sys, time
target_file, event_type, iteration, plan_file, decision_file = sys.argv[1:6]
with open(decision_file, encoding="utf-8") as f:
    decision = json.load(f)
entry = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "type": event_type,
    "iteration": int(iteration),
    "plan_path": plan_file,
    "option": decision["option"],
    "reason": decision["reason"],
}
with open(target_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY_DECISION
}

read_valid_human_decision() {
    local output_file="$1"
    local raw_reply
    while true; do
        echo "Human decision required. Reply with exactly one JSON object:"
        echo '  {"option":"keep","reason":"..."}'
        echo '  {"option":"revert","reason":"..."}'
        printf "> "
        if ! IFS= read -r raw_reply; then
            echo "ERROR: stdin closed before a human decision was received." >&2
            return 1
        fi
        if printf '%s' "$raw_reply" | python3 "${VALIDATOR_SCRIPT}" > "${output_file}"; then
            return 0
        fi
        echo "Invalid decision format. Please retry." >&2
    done
}

run_planning_iteration() {
    local iteration="$1"
    local rejection_context="$2"
    local plan_file="${WORKSPACE}/human_plan_iteration_${iteration}.md"
    local prompt
    prompt=$(cat <<PROMPT_EOF
You are preparing the RQ3 Human Guidance AIME experiment.

Read program.md. This is a PLANNING-ONLY phase before the 10-hour autonomous timer starts.

Hard constraints for this phase:
- Do not train, prepare datasets, request evals, run Python training scripts, call the evaluator, or inspect external directories.
- Produce one high-level research plan for improving Qwen3-1.7B-Base on AIME 2025.
- The plan should focus on strategy, hypotheses, cheap evidence to inspect, success criteria, abort criteria, and final submission criteria.
- Write the plan to: ${plan_file}
- Append a JSONL entry to experiment.jsonl with type "human_plan_proposal", iteration ${iteration}, and plan_path "${plan_file}".
- End after the plan is written.

Previous human rejection context, if any:
${rejection_context}
PROMPT_EOF
)

    local planning_exit=0
    runuser -u "${AGENT_USER}" -- \
        claude -p "$prompt" \
            --model "${CLAUDE_MODEL}" \
            --verbose \
            --output-format stream-json \
            --dangerously-skip-permissions \
            --allowedTools \
                "Read(${WORKSPACE}/*)" \
                "Write(${WORKSPACE}/*)" \
        >> "${PLANNING_TRAJECTORY_FILE}" 2>&1 \
        || planning_exit=$?

    if [[ $planning_exit -ne 0 ]]; then
        echo "ERROR: planning agent exited with code ${planning_exit}." >&2
        return $planning_exit
    fi
    if [[ ! -s "${plan_file}" ]]; then
        echo "ERROR: planning agent did not create ${plan_file}." >&2
        return 1
    fi
    echo "${plan_file}"
}

print_plan_for_review() {
    local plan_file="$1"
    echo ""
    echo "================ Human Plan Proposal ================"
    echo "Plan file: ${plan_file}"
    echo "-----------------------------------------------------"
    python3 - "$plan_file" <<'PY_PRINT'
from pathlib import Path
import sys
print(Path(sys.argv[1]).read_text(encoding='utf-8'))
PY_PRINT
    echo "====================================================="
    echo ""
}

write_timer_file() {
    local start_epoch="$1"
    local deadline_epoch="$2"
    local start_iso="$3"
    python3 - "$start_epoch" "$deadline_epoch" "$TIME_BUDGET_HOURS" "$start_iso" "${WORKSPACE}/.timer" <<'PY_TIMER'
import json, sys
start_epoch, deadline_epoch, budget_hours, start_iso, path = sys.argv[1:6]
timer = {
    "start_epoch": int(start_epoch),
    "deadline_epoch": int(deadline_epoch),
    "budget_hours": int(budget_hours),
    "start_iso": start_iso,
    "clock": "autonomous_only",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(timer, f)
PY_TIMER
}

echo "======================================================="
echo "AI4AI: claude-code-human"
echo "Base model:    ${BASE_MODEL_PATH}"
echo "GPUs:          ${NUM_GPUS} (${GPU_IDS})"
echo "Budget:        ${TIME_BUDGET_HOURS}h autonomous time"
echo "Workspace:     ${WORKSPACE}"
echo "Planning traj: ${PLANNING_TRAJECTORY_FILE}"
echo "Run traj:      ${TRAJECTORY_FILE}"
echo "Claude model:  ${CLAUDE_MODEL}"
echo "======================================================="
echo ""

TOTAL_START_TIME=$(date +%s)
TOTAL_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PLANNING_AGENT_SECONDS=0
HUMAN_WAIT_SECONDS=0
PLANNING_ITERATIONS=0
ACCEPTED_PLAN=""
ACCEPTED_DECISION=""
REJECTION_CONTEXT="(none)"

cd "${WORKSPACE}"

touch "${JOURNAL_FILE}" "${HUMAN_GUIDANCE_FILE}" "${PLANNING_TRAJECTORY_FILE}"
chown "${AGENT_USER}:${AGENT_USER}" "${JOURNAL_FILE}" "${HUMAN_GUIDANCE_FILE}" "${PLANNING_TRAJECTORY_FILE}"

for iteration in $(seq 1 "${PLANNING_MAX_ITERATIONS}"); do
    PLANNING_ITERATIONS=$iteration
    echo "[$(date)] Planning iteration ${iteration}/${PLANNING_MAX_ITERATIONS}"
    ITER_START=$(date +%s)
    PLAN_FILE=$(run_planning_iteration "$iteration" "$REJECTION_CONTEXT")
    ITER_END=$(date +%s)
    PLANNING_AGENT_SECONDS=$((PLANNING_AGENT_SECONDS + ITER_END - ITER_START))

    append_event "${HUMAN_GUIDANCE_FILE}" "human_plan_proposal" "{\"iteration\": ${iteration}, \"plan_path\": \"${PLAN_FILE}\"}"
    print_plan_for_review "${PLAN_FILE}"

    DECISION_TMP="${WORKSPACE}/.human_decision_${iteration}.json"
    WAIT_START=$(date +%s)
    read_valid_human_decision "${DECISION_TMP}"
    WAIT_END=$(date +%s)
    HUMAN_WAIT_SECONDS=$((HUMAN_WAIT_SECONDS + WAIT_END - WAIT_START))
    DECISION_JSON=$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).read_text().strip())' "${DECISION_TMP}")
    OPTION=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["option"])' "${DECISION_JSON}")
    REASON=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["reason"])' "${DECISION_JSON}")

    append_decision_event "${HUMAN_GUIDANCE_FILE}" "human_decision" "${iteration}" "${PLAN_FILE}" "${DECISION_TMP}"
    append_decision_event "${JOURNAL_FILE}" "human_decision" "${iteration}" "${PLAN_FILE}" "${DECISION_TMP}"

    if [[ "${OPTION}" == "keep" ]]; then
        ACCEPTED_PLAN="${PLAN_FILE}"
        ACCEPTED_DECISION="${DECISION_JSON}"
        append_decision_event "${JOURNAL_FILE}" "human_plan_accepted" "${iteration}" "${PLAN_FILE}" "${DECISION_TMP}"
        break
    fi

    REJECTION_CONTEXT="Iteration ${iteration} was rejected by the human. Reason: ${REASON}. Revise the next plan accordingly."
done

if [[ -z "${ACCEPTED_PLAN}" ]]; then
    echo "ERROR: no plan accepted after ${PLANNING_MAX_ITERATIONS} planning iterations." >&2
    exit 2
fi

AUTONOMOUS_START_TIME=$(date +%s)
AUTONOMOUS_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DEADLINE_EPOCH=$((AUTONOMOUS_START_TIME + TIME_BUDGET_SECONDS))
write_timer_file "${AUTONOMOUS_START_TIME}" "${DEADLINE_EPOCH}" "${AUTONOMOUS_START_ISO}"
chown "${AGENT_USER}:${AGENT_USER}" "${WORKSPACE}/.timer"

echo "[$(date)] Human accepted ${ACCEPTED_PLAN}. Starting autonomous ${TIME_BUDGET_HOURS}h timer."

AUTONOMOUS_PROMPT=$(cat <<PROMPT_EOF
${INITIAL_PROMPT}

RQ3 Human Guidance setup:
- The human has accepted the initial research plan at: ${ACCEPTED_PLAN}
- Human decision JSON: ${ACCEPTED_DECISION}
- The 10-hour autonomous timer starts now. Waiting for the human is over and no further human guidance is available.
- Follow the accepted plan unless new evidence meets its abort criteria. If you pivot, log the evidence and rationale in experiment.jsonl.
- You may check remaining autonomous time with: bash ${TIMER_SCRIPT}

Begin autonomous training now.
PROMPT_EOF
)

EXIT_CODE=0

timeout "${TIME_BUDGET_SECONDS}" \
    runuser -u "${AGENT_USER}" -- \
        claude -p "${AUTONOMOUS_PROMPT}" \
            --model "${CLAUDE_MODEL}" \
            --verbose \
            --output-format stream-json \
            --dangerously-skip-permissions \
            --allowedTools \
                "Skill" \
                "Bash(${WORKSPACE}/*)" \
                "Bash(${WORKSPACE}/.claude/skills/*/scripts/*)" \
                "Bash(${REQUEST_EVAL_SCRIPT}*)" \
                "Bash(${FINAL_SCRIPT}*)" \
                "Bash(${TIMER_SCRIPT}*)" \
                "Bash(nvidia-smi*)" \
                "Read(${WORKSPACE}/*)" \
                "Read(${BASE_MODEL_PATH}/*)" \
                "Write(${WORKSPACE}/*)" \
    > "${TRAJECTORY_FILE}" 2>&1 \
    || EXIT_CODE=$?

END_TIME=$(date +%s)
END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
AUTONOMOUS_ELAPSED=$((END_TIME - AUTONOMOUS_START_TIME))
TOTAL_WALL_ELAPSED=$((END_TIME - TOTAL_START_TIME))

format_duration() {
    local seconds="$1"
    echo "$((seconds / 3600))h $(((seconds % 3600) / 60))m $((seconds % 60))s"
}

echo ""
echo "========================================================"
echo "Experiment finished."
echo "Total start:       ${TOTAL_START_ISO}"
echo "Autonomous start:  ${AUTONOMOUS_START_ISO}"
echo "End:               ${END_ISO}"
echo "Planning agent:    $(format_duration ${PLANNING_AGENT_SECONDS})"
echo "Human wait:        $(format_duration ${HUMAN_WAIT_SECONDS})"
echo "Autonomous elapsed:$(format_duration ${AUTONOMOUS_ELAPSED})"
echo "Total wall:        $(format_duration ${TOTAL_WALL_ELAPSED})"
echo "Trajectory:        ${TRAJECTORY_FILE}"
if [[ $EXIT_CODE -eq 124 ]]; then
    echo "Status:            TIMED OUT (${TIME_BUDGET_HOURS}h autonomous budget reached)"
elif [[ $EXIT_CODE -eq 0 ]]; then
    echo "Status:            COMPLETED (agent exited normally)"
else
    echo "Status:            EXITED (code=${EXIT_CODE})"
fi
echo "========================================================"

python3 - "$TOTAL_START_ISO" "$AUTONOMOUS_START_ISO" "$END_ISO" "$TOTAL_WALL_ELAPSED" "$AUTONOMOUS_ELAPSED" "$PLANNING_AGENT_SECONDS" "$HUMAN_WAIT_SECONDS" "$EXIT_CODE" "$BASE_MODEL_PATH" "$NUM_GPUS" "$TIME_BUDGET_HOURS" "$CLAUDE_MODEL" "$VERL_PATH" "$WORKSPACE" "$PLANNING_ITERATIONS" "$ACCEPTED_PLAN" "$PLANNING_TRAJECTORY_FILE" "$TRAJECTORY_FILE" <<'PY_META'
import json, sys
(
    total_start,
    autonomous_start,
    end,
    total_wall_elapsed,
    autonomous_elapsed,
    planning_agent_seconds,
    human_wait_seconds,
    exit_code,
    base_model_path,
    num_gpus,
    time_budget_hours,
    claude_model,
    verl_path,
    workspace,
    planning_iterations,
    accepted_plan_path,
    planning_trajectory,
    trajectory,
) = sys.argv[1:19]
meta = {
    "baseline": "claude-code-human",
    "condition": "human_guidance_initial_plan",
    "start": total_start,
    "autonomous_start": autonomous_start,
    "end": end,
    "elapsed_seconds": int(autonomous_elapsed),
    "autonomous_elapsed_seconds": int(autonomous_elapsed),
    "total_wall_seconds": int(total_wall_elapsed),
    "planning_agent_seconds": int(planning_agent_seconds),
    "human_wait_seconds": int(human_wait_seconds),
    "planning_iterations": int(planning_iterations),
    "accepted_plan_path": accepted_plan_path,
    "planning_trajectory": planning_trajectory,
    "trajectory": trajectory,
    "exit_code": int(exit_code),
    "timed_out": int(exit_code) == 124,
    "timer_semantics": "10h timeout starts after human keep decision; human wait is excluded",
    "config": {
        "base_model_path": base_model_path,
        "num_gpus": int(num_gpus),
        "time_budget_hours": int(time_budget_hours),
        "claude_model": claude_model,
        "verl_path": verl_path,
    },
}
path = workspace + "/run_metadata.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"Run metadata saved to {path}")
PY_META

TRAJ_LINES=$(wc -l < "${TRAJECTORY_FILE}" 2>/dev/null || echo 0)
PLANNING_TRAJ_LINES=$(wc -l < "${PLANNING_TRAJECTORY_FILE}" 2>/dev/null || echo 0)
echo ""
echo "  Planning trajectory: ${PLANNING_TRAJECTORY_FILE} (${PLANNING_TRAJ_LINES} lines)"
echo "  Autonomous trajectory: ${TRAJECTORY_FILE} (${TRAJ_LINES} lines)"
echo "  Human guidance log: ${HUMAN_GUIDANCE_FILE}"
echo "  Run metadata: ${WORKSPACE}/run_metadata.json"
echo ""
echo "  To archive this run:"
echo "    rsync -a --exclude='checkpoints/' --exclude='data/' --exclude='__pycache__/' --exclude='outputs/' ${WORKSPACE}/ ${RUN_ARCHIVE_BASE}/run_N/"

exit ${EXIT_CODE}
