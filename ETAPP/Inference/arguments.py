"""Shared command-line arguments for ETAPP inference entry points."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", "--model-type", type=str, help="Type of the model to evaluate")
    parser.add_argument("--model_name", "--model-name", default="")
    parser.add_argument("--base_name_or_path", "--base-name-or-path", default="")
    parser.add_argument("--model_name_or_path", "--model-name-or-path", default="")
    parser.add_argument("--max_sequence_length", "--max-sequence-length", type=int, default=8192)
    parser.add_argument("--setting", choices=["base", "enhanced"])
    parser.add_argument("--prompt_type", "--prompt-type", choices=["InjecAgent", "hwchase17_react"])
    parser.add_argument("--logging_dir", "--logging-dir", default=None)
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--instruction_file", "--instruction-file", default=str(REPO_ROOT / "data/instruction/instruction.json"))
    parser.add_argument("--profile_file", "--profile-file", default=str(REPO_ROOT / "profile/profiles.json"))
    parser.add_argument("--concrete_profile_dir", "--concrete-profile-dir", default=str(REPO_ROOT / "profile/concrete_profile"))
    parser.add_argument("--method", default="general")
    parser.add_argument("--add_example", "--add-example", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--mode", type=str, default="function_calling")
    parser.add_argument("--use_retrieval", "--use-retrieval", action="store_true")
    parser.add_argument("--add_reminder", "--add-reminder", action="store_true")
    parser.add_argument("--use_vllm", "--use-vllm", action="store_true")
    parser.add_argument("--peft_path", "--peft-path", default=None)
    parser.add_argument("--max_turn", "--max-turn", type=int, default=3)
    parser.add_argument("--max_observation_length", "--max-observation-length", type=int, default=8192)
    parser.add_argument("--total_max_tokens", "--total-max-tokens", type=int, default=16384)
    parser.add_argument("--max_parallel_calls", "--max-parallel-calls", type=int, default=10)
    parser.add_argument("--max_tool_calls", "--max-tool-calls", type=int, default=50)
    parser.add_argument("--vllm_base_url", "--vllm-base-url", default=None)
    parser.add_argument("--served_model_name", "--served-model-name", default=None)
    parser.add_argument("--vllm_api_key", "--vllm-api-key", default=None)
    parser.add_argument("--request_timeout", "--request-timeout", type=float, default=None)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=None)
    parser.add_argument("--max_model_len", "--max-model-len", type=int, default=None)
    parser.add_argument("--tool_retriever_model_path", "--tool-retriever-model-path", default=None)
    parser.add_argument("--wikipedia_index_path", "--wikipedia-index-path", default=None)
    return parser


def finalize_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    suffix = "_retrieve" if args.use_retrieval else ""
    if not args.output_dir:
        args.output_dir = str(
            REPO_ROOT / "output" / f"prompted_{args.model_type}_{args.model_name}_{args.method}{suffix}"
        )
    if not args.logging_dir:
        args.logging_dir = str(Path(args.output_dir) / "inference.log")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).parent.mkdir(parents=True, exist_ok=True)
    return args
