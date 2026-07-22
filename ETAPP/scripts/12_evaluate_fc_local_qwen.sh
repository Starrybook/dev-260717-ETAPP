#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_common_inputs
require_directory "FC_OUTPUT_DIR" "$FC_OUTPUT_DIR"
require_qwen_service
require_positive_integer "EVALUATION_MAX_TOKENS" "$EVALUATION_MAX_TOKENS"
require_nonnegative_number "EVALUATION_TEMPERATURE" "$EVALUATION_TEMPERATURE"
require_positive_integer "EVALUATION_REQUEST_TIMEOUT" "$EVALUATION_REQUEST_TIMEOUT"

validate_fc_outputs
mkdir -p "$FC_EVALUATION_OUTPUT_DIR/run_metadata"

EVALUATION_COMMAND=(
    "$PYTHON_BIN" -m evaluation.evaluate
    --result-file "$FC_OUTPUT_DIR"
    --instruction-file "$INSTRUCTION_FILE"
    --profile-file "$PROFILE_FILE"
    --concrete-profile-dir "$CONCRETE_PROFILE_DIR"
    --evaluation-model "$QWEN_SERVED_MODEL_NAME"
    --evaluation-base-url "$VLLM_BASE_URL"
    --evaluation-api-key EMPTY
    --evaluation-max-tokens "$EVALUATION_MAX_TOKENS"
    --evaluation-temperature "$EVALUATION_TEMPERATURE"
    --evaluation-request-timeout "$EVALUATION_REQUEST_TIMEOUT"
    --evaluate-output-dir "$FC_EVALUATION_OUTPUT_DIR"
    --logging-dir "$FC_EVALUATION_OUTPUT_DIR/evaluation.log"
    --setting fc_qwen72b_tool_retrieval
)

run_logged_command \
    "evaluate_fc_local_qwen" \
    "$FC_EVALUATION_OUTPUT_DIR/run_metadata/evaluation.txt" \
    "$FC_EVALUATION_OUTPUT_DIR/evaluator.stdout.log" \
    "${EVALUATION_COMMAND[@]}"

validate_evaluation_outputs
info "FC evaluation completed successfully: $FC_EVALUATION_OUTPUT_DIR"
