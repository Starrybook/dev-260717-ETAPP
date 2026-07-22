#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

pipeline_owns_service=0

cleanup_pipeline() {
    cleanup_status=$?
    trap - EXIT
    if [ "$pipeline_owns_service" -eq 1 ]; then
        info "Pipeline cleanup: stopping owned Qwen vLLM service"
        if ! "$SCRIPT_DIR/13_stop_qwen_vllm.sh"; then
            warn "Pipeline could not cleanly stop its owned Qwen service"
            if [ "$cleanup_status" -eq 0 ]; then
                cleanup_status=1
            fi
        fi
    fi
    exit "$cleanup_status"
}

trap cleanup_pipeline EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -f "$VLLM_PID_FILE" ] || service_health_ok; then
    die "The end-to-end pipeline requires an unused endpoint and no managed PID file; use stage scripts 11 and 12 to reuse an existing service"
fi

"$SCRIPT_DIR/10_start_qwen_vllm.sh"
pipeline_owns_service=1
"$SCRIPT_DIR/11_run_fc_retrieval.sh"
"$SCRIPT_DIR/12_evaluate_fc_local_qwen.sh"

info "FC generation and local Qwen evaluation completed"
