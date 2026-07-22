#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_common_inputs
require_generation_resources
require_qwen_service
require_positive_integer "VLLM_MAX_MODEL_LEN" "$VLLM_MAX_MODEL_LEN"
require_positive_integer "FC_MAX_TURN" "$FC_MAX_TURN"
require_positive_integer "FC_MAX_NEW_TOKENS" "$FC_MAX_NEW_TOKENS"
require_positive_integer "FC_MAX_OBSERVATION_LENGTH" "$FC_MAX_OBSERVATION_LENGTH"
require_nonnegative_number "FC_GENERATION_TEMPERATURE" "$FC_GENERATION_TEMPERATURE"

mkdir -p "$FC_OUTPUT_DIR/run_metadata"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

FC_COMMAND=(
    "$PYTHON_BIN" -m Inference.evaluate_prompted_agent
    --model-type OpenModel
    --model-name Qwen2.5-72B-Instruct
    --model-name-or-path "$QWEN_MODEL_DIR"
    --max-sequence-length "$VLLM_MAX_MODEL_LEN"
    --method fine-tuned
    --mode function_calling
    --use-retrieval
    --use-vllm
    --vllm-base-url "$VLLM_BASE_URL"
    --served-model-name "$QWEN_SERVED_MODEL_NAME"
    --vllm-api-key EMPTY
    --request-timeout "$REQUEST_TIMEOUT"
    --max-turn "$FC_MAX_TURN"
    --max-new-tokens "$FC_MAX_NEW_TOKENS"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --generation-temperature "$FC_GENERATION_TEMPERATURE"
    --max-observation-length "$FC_MAX_OBSERVATION_LENGTH"
    --tool-retriever-model-path "$MINILM_MODEL_DIR"
    --wikipedia-index-path "$WIKIPEDIA_INDEX_PATH"
    --instruction-file "$INSTRUCTION_FILE"
    --profile-file "$PROFILE_FILE"
    --concrete-profile-dir "$CONCRETE_PROFILE_DIR"
    --output-dir "$FC_OUTPUT_DIR"
    --logging-dir "$FC_OUTPUT_DIR/inference.log"
)

run_logged_command \
    "fc_qwen72b_tool_retrieval" \
    "$FC_OUTPUT_DIR/run_metadata/generation.txt" \
    "$FC_OUTPUT_DIR/controller.stdout.log" \
    "${FC_COMMAND[@]}"

validate_fc_outputs
info "FC generation completed successfully: $FC_OUTPUT_DIR"
