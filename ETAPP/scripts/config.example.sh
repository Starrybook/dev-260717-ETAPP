#!/usr/bin/env bash

# Source this file, or export the same variables in your shell. The run scripts
# never download models, indexes, or Python packages.

# Required resource locations. ETAPP_RESOURCE_ROOT can provide all three
# conventional defaults, or each path can be exported independently.
# export ETAPP_RESOURCE_ROOT=/path/to/etapp-resources
# export QWEN_MODEL_DIR=/path/to/Qwen2.5-72B-Instruct
# export MINILM_MODEL_DIR=/path/to/paraphrase-MiniLM-L3-v2
# export WIKIPEDIA_INDEX_PATH=/path/to/wikipedia-kilt-doc

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen2.5-72b-etapp}"
# Set this to the Hugging Face commit SHA recorded during resource preparation.
export QWEN_MODEL_REVISION="${QWEN_MODEL_REVISION:-unknown}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
export VLLM_SHUTDOWN_TIMEOUT="${VLLM_SHUTDOWN_TIMEOUT:-120}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

export FC_MAX_TURN="${FC_MAX_TURN:-20}"
export FC_MAX_NEW_TOKENS="${FC_MAX_NEW_TOKENS:-1024}"
export FC_GENERATION_TEMPERATURE="${FC_GENERATION_TEMPERATURE:-0.0}"
export FC_MAX_OBSERVATION_LENGTH="${FC_MAX_OBSERVATION_LENGTH:-8192}"

export EVALUATION_MAX_TOKENS="${EVALUATION_MAX_TOKENS:-8192}"
export EVALUATION_TEMPERATURE="${EVALUATION_TEMPERATURE:-0.0}"
export EVALUATION_REQUEST_TIMEOUT="${EVALUATION_REQUEST_TIMEOUT:-600}"

# Optional executable overrides.
export PYTHON_BIN="${PYTHON_BIN:-python}"
export VLLM_BIN="${VLLM_BIN:-vllm}"
export CURL_BIN="${CURL_BIN:-curl}"

# Optional input/output overrides. Defaults are anchored to the ETAPP root.
# export INSTRUCTION_FILE=/path/to/instruction.json
# export PROFILE_FILE=/path/to/profiles.json
# export CONCRETE_PROFILE_DIR=/path/to/concrete_profile
# export FC_OUTPUT_DIR=/path/to/fc_output
# export FC_EVALUATION_OUTPUT_DIR=/path/to/fc_output/evaluation
