#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_command "$PYTHON_BIN"
require_command "$VLLM_BIN"
require_command "$CURL_BIN"
require_command "$ETAPP_KILL_BIN"
require_command nohup
require_directory "QWEN_MODEL_DIR" "$QWEN_MODEL_DIR"
require_positive_integer "VLLM_PORT" "$VLLM_PORT"
require_positive_integer "VLLM_TENSOR_PARALLEL_SIZE" "$VLLM_TENSOR_PARALLEL_SIZE"
require_positive_integer "VLLM_MAX_MODEL_LEN" "$VLLM_MAX_MODEL_LEN"
require_positive_integer "VLLM_STARTUP_TIMEOUT" "$VLLM_STARTUP_TIMEOUT"
require_nonnegative_number "VLLM_GPU_MEMORY_UTILIZATION" "$VLLM_GPU_MEMORY_UTILIZATION"

mkdir -p "$VLLM_RUNTIME_DIR"

if [ -f "$VLLM_PID_FILE" ]; then
    existing_pid="$(sed -n '1p' "$VLLM_PID_FILE")"
    if [ -n "$existing_pid" ] && pid_is_running "$existing_pid"; then
        expected_model="$QWEN_MODEL_DIR"
        if [ -f "$VLLM_MODEL_PATH_FILE" ]; then
            expected_model="$(sed -n '1p' "$VLLM_MODEL_PATH_FILE")"
        fi
        if pid_is_managed_vllm "$existing_pid" "$expected_model" && service_health_ok && service_model_ok; then
            info "Managed Qwen vLLM service is already ready (PID $existing_pid)"
            exit 0
        fi
        die "PID file points to a live process that is not a healthy matching managed service: $existing_pid"
    fi
    warn "Removing stale vLLM PID file"
    rm -f "$VLLM_PID_FILE" "$VLLM_MODEL_PATH_FILE"
fi

if service_health_ok; then
    die "A service already responds at $VLLM_SERVER_URL but is not owned by $VLLM_PID_FILE"
fi

export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

VLLM_COMMAND=(
    "$VLLM_BIN" serve "$QWEN_MODEL_DIR"
    --served-model-name "$QWEN_SERVED_MODEL_NAME"
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
)

write_run_metadata "$VLLM_RUNTIME_DIR/start_metadata.txt" "start_qwen_vllm" "${VLLM_COMMAND[@]}"
info "Starting Qwen vLLM; service log: $VLLM_LOG_FILE"
nohup "${VLLM_COMMAND[@]}" >"$VLLM_LOG_FILE" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" >"$VLLM_PID_FILE"
printf '%s\n' "$QWEN_MODEL_DIR" >"$VLLM_MODEL_PATH_FILE"

start_seconds=$SECONDS
while true; do
    if ! pid_is_running "$server_pid"; then
        warn "vLLM exited before becoming ready; last log lines follow"
        tail -n 80 "$VLLM_LOG_FILE" >&2 || true
        rm -f "$VLLM_PID_FILE" "$VLLM_MODEL_PATH_FILE"
        die "Qwen vLLM process exited during startup"
    fi
    if service_health_ok && service_model_ok; then
        info "Qwen vLLM is ready: PID=$server_pid, model=$QWEN_SERVED_MODEL_NAME, base_url=$VLLM_BASE_URL"
        exit 0
    fi
    if [ $((SECONDS - start_seconds)) -ge "$VLLM_STARTUP_TIMEOUT" ]; then
        warn "vLLM startup timed out; terminating PID $server_pid"
        send_process_signal TERM "$server_pid" >/dev/null 2>&1 || true
        sleep 2
        if pid_is_running "$server_pid"; then
            send_process_signal KILL "$server_pid" >/dev/null 2>&1 || true
        fi
        tail -n 80 "$VLLM_LOG_FILE" >&2 || true
        rm -f "$VLLM_PID_FILE" "$VLLM_MODEL_PATH_FILE"
        die "Qwen vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT}s"
    fi
    sleep 5
done
