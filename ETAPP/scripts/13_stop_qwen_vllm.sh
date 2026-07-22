#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_command "$ETAPP_KILL_BIN"
require_positive_integer "VLLM_SHUTDOWN_TIMEOUT" "$VLLM_SHUTDOWN_TIMEOUT"

if [ ! -f "$VLLM_PID_FILE" ]; then
    if service_health_ok; then
        die "A service responds at $VLLM_SERVER_URL, but no managed PID file exists; refusing to stop it"
    fi
    info "No managed Qwen vLLM PID file exists; nothing to stop"
    exit 0
fi

server_pid="$(sed -n '1p' "$VLLM_PID_FILE")"
[ -n "$server_pid" ] || die "Managed PID file is empty: $VLLM_PID_FILE"

if ! pid_is_running "$server_pid"; then
    warn "Managed PID $server_pid is no longer running; removing stale runtime files"
    rm -f "$VLLM_PID_FILE" "$VLLM_MODEL_PATH_FILE"
    exit 0
fi

[ -f "$VLLM_MODEL_PATH_FILE" ] || die "Missing managed model-path file: $VLLM_MODEL_PATH_FILE"
expected_model="$(sed -n '1p' "$VLLM_MODEL_PATH_FILE")"
pid_is_managed_vllm "$server_pid" "$expected_model" \
    || die "PID $server_pid does not match the managed 'vllm serve' command; refusing to kill it"

info "Sending SIGTERM to managed Qwen vLLM PID $server_pid"
send_process_signal TERM "$server_pid"
shutdown_start=$SECONDS
while pid_is_running "$server_pid"; do
    if [ $((SECONDS - shutdown_start)) -ge "$VLLM_SHUTDOWN_TIMEOUT" ]; then
        warn "PID $server_pid did not stop within ${VLLM_SHUTDOWN_TIMEOUT}s; sending SIGKILL"
        send_process_signal KILL "$server_pid" >/dev/null 2>&1 || true
        break
    fi
    sleep 2
done

rm -f "$VLLM_PID_FILE" "$VLLM_MODEL_PATH_FILE"

health_wait_start=$SECONDS
while service_health_ok; do
    if [ $((SECONDS - health_wait_start)) -ge 30 ]; then
        die "The managed PID stopped, but a service still responds at $VLLM_SERVER_URL"
    fi
    sleep 1
done

info "Managed Qwen vLLM service stopped"
