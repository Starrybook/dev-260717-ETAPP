#!/usr/bin/env bash

if [ -n "${ETAPP_COMMON_SH_LOADED:-}" ]; then
    return 0
fi
ETAPP_COMMON_SH_LOADED=1

set -o pipefail

ETAPP_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ETAPP_ROOT="$(cd "$ETAPP_SCRIPTS_DIR/.." && pwd -P)"

if [ -n "${ETAPP_RESOURCE_ROOT:-}" ]; then
    : "${QWEN_MODEL_DIR:=$ETAPP_RESOURCE_ROOT/models/Qwen2.5-72B-Instruct}"
    : "${MINILM_MODEL_DIR:=$ETAPP_RESOURCE_ROOT/models/paraphrase-MiniLM-L3-v2}"
    : "${WIKIPEDIA_INDEX_PATH:=$ETAPP_RESOURCE_ROOT/indexes/wikipedia-kilt-doc}"
fi

: "${QWEN_MODEL_DIR:=}"
: "${MINILM_MODEL_DIR:=}"
: "${WIKIPEDIA_INDEX_PATH:=}"
: "${PYTHON_BIN:=python}"
: "${VLLM_BIN:=vllm}"
: "${CURL_BIN:=curl}"
: "${ETAPP_KILL_BIN:=kill}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${VLLM_HOST:=127.0.0.1}"
: "${VLLM_PORT:=8000}"
: "${QWEN_SERVED_MODEL_NAME:=qwen2.5-72b-etapp}"
: "${QWEN_MODEL_REVISION:=unknown}"
: "${VLLM_TENSOR_PARALLEL_SIZE:=4}"
: "${VLLM_DTYPE:=bfloat16}"
: "${VLLM_MAX_MODEL_LEN:=32768}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_STARTUP_TIMEOUT:=1800}"
: "${VLLM_SHUTDOWN_TIMEOUT:=120}"
: "${REQUEST_TIMEOUT:=600}"
: "${FC_MAX_TURN:=20}"
: "${FC_MAX_NEW_TOKENS:=1024}"
: "${FC_GENERATION_TEMPERATURE:=0.0}"
: "${FC_MAX_OBSERVATION_LENGTH:=8192}"
: "${EVALUATION_MAX_TOKENS:=8192}"
: "${EVALUATION_TEMPERATURE:=0.0}"
: "${EVALUATION_REQUEST_TIMEOUT:=600}"

: "${INSTRUCTION_FILE:=$ETAPP_ROOT/data/instruction/instruction.json}"
: "${PROFILE_FILE:=$ETAPP_ROOT/profile/profiles.json}"
: "${CONCRETE_PROFILE_DIR:=$ETAPP_ROOT/profile/concrete_profile}"
: "${FC_OUTPUT_DIR:=$ETAPP_ROOT/output/fc_qwen72b_tool_retrieval}"
: "${FC_EVALUATION_OUTPUT_DIR:=$FC_OUTPUT_DIR/evaluation}"
: "${VLLM_RUNTIME_DIR:=$ETAPP_ROOT/output/runtime/qwen_vllm}"
: "${SMOKE_INPUT_DIR:=$ETAPP_ROOT/output/smoke_inputs/fc}"

VLLM_SERVER_URL="http://$VLLM_HOST:$VLLM_PORT"
VLLM_BASE_URL="$VLLM_SERVER_URL/v1"
VLLM_PID_FILE="$VLLM_RUNTIME_DIR/server.pid"
VLLM_MODEL_PATH_FILE="$VLLM_RUNTIME_DIR/model_path"
VLLM_LOG_FILE="$VLLM_RUNTIME_DIR/server.log"

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

info() {
    printf '[%s] INFO: %s\n' "$(timestamp)" "$*"
}

warn() {
    printf '[%s] WARNING: %s\n' "$(timestamp)" "$*" >&2
}

die() {
    printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_directory() {
    [ -n "$2" ] || die "$1 is not set"
    [ -d "$2" ] || die "$1 is not a directory: $2"
}

require_file() {
    [ -f "$2" ] || die "$1 is not a file: $2"
}

require_nonnegative_number() {
    case "$2" in
        ''|*[!0-9.]*|*.*.*) die "$1 must be a non-negative number: $2" ;;
    esac
    case "$2" in
        *[0-9]*) ;;
        *) die "$1 must contain at least one digit: $2" ;;
    esac
}

require_positive_integer() {
    case "$2" in
        ''|*[!0-9]*|0) die "$1 must be a positive integer: $2" ;;
    esac
}

require_common_inputs() {
    require_command "$PYTHON_BIN"
    require_file "INSTRUCTION_FILE" "$INSTRUCTION_FILE"
    require_file "PROFILE_FILE" "$PROFILE_FILE"
    require_directory "CONCRETE_PROFILE_DIR" "$CONCRETE_PROFILE_DIR"
}

require_generation_resources() {
    require_directory "QWEN_MODEL_DIR" "$QWEN_MODEL_DIR"
    require_directory "MINILM_MODEL_DIR" "$MINILM_MODEL_DIR"
    require_directory "WIKIPEDIA_INDEX_PATH" "$WIKIPEDIA_INDEX_PATH"
    find "$WIKIPEDIA_INDEX_PATH" -maxdepth 1 -name 'segments_*' -print -quit | grep -q . \
        || die "WIKIPEDIA_INDEX_PATH does not directly contain a Lucene segments_* file: $WIKIPEDIA_INDEX_PATH"
}

service_health_ok() {
    "$CURL_BIN" -fsS --max-time 5 "$VLLM_SERVER_URL/health" >/dev/null 2>&1
}

service_model_ok() {
    "$CURL_BIN" -fsS --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null \
        | grep -F "$QWEN_SERVED_MODEL_NAME" >/dev/null 2>&1
}

require_qwen_service() {
    require_command "$CURL_BIN"
    service_health_ok || die "vLLM health check failed: $VLLM_SERVER_URL/health"
    service_model_ok || die "Served model '$QWEN_SERVED_MODEL_NAME' not found at $VLLM_BASE_URL/models"
}

pid_is_running() {
    kill -0 "$1" >/dev/null 2>&1
}

send_process_signal() {
    "$ETAPP_KILL_BIN" "-$1" "$2"
}

process_command() {
    ps -ww -p "$1" -o command= 2>/dev/null || true
}

pid_is_managed_vllm() {
    _managed_pid="$1"
    _managed_model_path="$2"
    _managed_command="$(process_command "$_managed_pid")"
    case "$_managed_command" in
        *vllm*serve*"$_managed_model_path"*) return 0 ;;
        *) return 1 ;;
    esac
}

shell_quote_command() {
    for _quote_arg in "$@"; do
        printf '%q ' "$_quote_arg"
    done
    printf '\n'
}

write_run_metadata() {
    _metadata_file="$1"
    _metadata_stage="$2"
    shift 2
    mkdir -p "$(dirname "$_metadata_file")"
    {
        printf 'stage=%s\n' "$_metadata_stage"
        printf 'timestamp=%s\n' "$(timestamp)"
        printf 'etapp_root=%s\n' "$ETAPP_ROOT"
        printf 'git_commit=%s\n' "$(git -C "$ETAPP_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'python='; "$PYTHON_BIN" --version 2>&1 || true
        printf 'command='; shell_quote_command "$@"
        printf 'qwen_model_dir=%s\n' "$QWEN_MODEL_DIR"
        printf 'qwen_model_revision=%s\n' "$QWEN_MODEL_REVISION"
        printf 'served_model_name=%s\n' "$QWEN_SERVED_MODEL_NAME"
        printf 'vllm_base_url=%s\n' "$VLLM_BASE_URL"
        printf 'vllm_max_model_len=%s\n' "$VLLM_MAX_MODEL_LEN"
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
        printf '\npython_packages:\n'
        "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); import vllm; print("vllm=" + vllm.__version__)' 2>&1 || true
        printf '\ngpu_summary:\n'
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 || true
        else
            printf 'nvidia-smi unavailable\n'
        fi
    } >"$_metadata_file"
}

run_logged_command() {
    _run_stage="$1"
    _run_metadata="$2"
    _run_stdout_log="$3"
    shift 3
    mkdir -p "$(dirname "$_run_stdout_log")"
    write_run_metadata "$_run_metadata" "$_run_stage" "$@"
    info "Running $_run_stage; output log: $_run_stdout_log"
    (
        cd "$ETAPP_ROOT"
        "$@"
    ) 2>&1 | tee "$_run_stdout_log"
}

validate_fc_outputs() {
    "$PYTHON_BIN" - "$PROFILE_FILE" "$INSTRUCTION_FILE" "$FC_OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

profile_file, instruction_file, output_dir = map(Path, sys.argv[1:])
profiles = json.loads(profile_file.read_text(encoding="utf-8"))
instructions = json.loads(instruction_file.read_text(encoding="utf-8"))
errors = []
total = 0
required = {"query", "timestamp", "output", "tools"}
expected_files = {person.replace(" ", "_") + "_instruction.json" for person in profiles}
actual_files = {path.name for path in output_dir.glob("*_instruction.json")}
if actual_files != expected_files:
    errors.append(
        "output user files differ: "
        f"missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}"
    )
for person in profiles:
    filename = person.replace(" ", "_") + "_instruction.json"
    path = output_dir / filename
    if not path.is_file():
        errors.append(f"missing output file: {path}")
        continue
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        continue
    if len(rows) != len(instructions):
        errors.append(f"{path}: expected {len(instructions)} rows, got {len(rows)}")
    for idx, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            errors.append(f"{path} row {idx}: missing {sorted(missing)}")
        if idx < len(instructions) and row.get("query") != instructions[idx].get("query"):
            errors.append(f"{path} row {idx}: query does not match instruction")
    total += len(rows)
expected = len(profiles) * len(instructions)
if total != expected:
    errors.append(f"expected {expected} total trajectories, got {total}")
if errors:
    print("FC output validation failed:", file=sys.stderr)
    print("\n".join(f"- {item}" for item in errors), file=sys.stderr)
    raise SystemExit(1)
print(f"FC output validation passed: users={len(profiles)}, instructions={len(instructions)}, trajectories={total}")
PY
}

validate_evaluation_outputs() {
    "$PYTHON_BIN" - "$PROFILE_FILE" "$INSTRUCTION_FILE" "$FC_EVALUATION_OUTPUT_DIR/summary.json" "$FC_EVALUATION_OUTPUT_DIR/evaluate_result.json" "$QWEN_SERVED_MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

profile_file, instruction_file, summary_file, result_file, model = sys.argv[1:]
profiles = json.loads(Path(profile_file).read_text(encoding="utf-8"))
instructions = json.loads(Path(instruction_file).read_text(encoding="utf-8"))
expected = len(profiles) * len(instructions)
summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))
results = json.loads(Path(result_file).read_text(encoding="utf-8"))
errors = []
if summary.get("completed_samples") != expected:
    errors.append(f"completed_samples: expected {expected}, got {summary.get('completed_samples')}")
if summary.get("failed_samples") != 0:
    errors.append(f"failed_samples: expected 0, got {summary.get('failed_samples')}")
if len(results) != expected:
    errors.append(f"evaluate_result rows: expected {expected}, got {len(results)}")
metrics = summary.get("metrics", {}).get(model, {})
for metric in ("PRC", "PSN", "PTV"):
    if metric not in metrics:
        errors.append(f"missing metric {metric} for model {model}")
if errors:
    print("Evaluation output validation failed:", file=sys.stderr)
    print("\n".join(f"- {item}" for item in errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Evaluation output validation passed: samples={expected}, model={model}, metrics={metrics}")
PY
}
