"""Shared local/cloud judge helpers and resumable metric aggregation."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("Procedure", "Personalization", "Proactivity")


def build_evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_file", "--profile-file", default=str(REPO_ROOT / "profile/profiles.json"))
    parser.add_argument("--instruction_file", "--instruction-file", default=str(REPO_ROOT / "data/instruction/instruction.json"))
    parser.add_argument("--concrete_profile_dir", "--concrete-profile-dir", default=str(REPO_ROOT / "profile/concrete_profile"))
    parser.add_argument(
        "--result_file", "--result-file", "--output_dir", "--output-dir",
        dest="result_file", default=str(REPO_ROOT / "output/prompted_CHATGPT_gpt-4o_react"),
    )
    parser.add_argument("--logging_dir", "--logging-dir", default=None)
    parser.add_argument("--evaluation_model", "--evaluation-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--evaluation_base_url", "--evaluation-base-url", default=None)
    parser.add_argument("--evaluation_api_key", "--evaluation-api-key", default=None)
    parser.add_argument("--evaluation_max_tokens", "--evaluation-max-tokens", type=int, default=8192)
    parser.add_argument("--evaluation_temperature", "--evaluation-temperature", type=float, default=0.0)
    parser.add_argument("--evaluation_request_timeout", "--evaluation-request-timeout", type=float, default=None)
    parser.add_argument("--evaluate_output_dir", "--evaluate-output-dir", default=None)
    parser.add_argument("--setting", default=None)
    return parser


def finalize_evaluation_args(args: argparse.Namespace) -> argparse.Namespace:
    requested = Path(args.evaluate_output_dir) if args.evaluate_output_dir else Path(args.result_file)
    if requested.suffix.lower() == ".json":
        args.evaluate_result_file = str(requested)
        output_dir = requested.parent
    else:
        output_dir = requested
        args.evaluate_result_file = str(output_dir / "evaluate_result.json")
    args.evaluate_output_dir = str(output_dir)
    args.summary_file = str(output_dir / "summary.json")
    if not args.logging_dir:
        args.logging_dir = str(output_dir / f"evaluate_result_logging_{time.time()}.log")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).parent.mkdir(parents=True, exist_ok=True)
    return args


def create_evaluation_client(
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout: Optional[float] = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The 'openai' package is required to run evaluation") from exc
    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs.update(base_url=base_url, api_key=api_key or "EMPTY")
    elif model_name == "deepseek-chat":
        kwargs.update(base_url="https://api.deepseek.com", api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"))
    else:
        kwargs.update(api_key=api_key or os.environ.get("API_KEY"))
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    return OpenAI(**kwargs)


def parse_evaluation_json(raw_response: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    payload = match.group(1).strip() if match else raw_response.strip()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Evaluation response must be a JSON object")
    return parsed


def request_evaluation(
    model_name: str,
    messages: list,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    request_timeout: Optional[float] = None,
) -> Tuple[Dict[str, Any], str]:
    client = create_evaluation_client(model_name, base_url, api_key, request_timeout)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_response = response.choices[0].message.content or ""
    try:
        return parse_evaluation_json(raw_response), raw_response
    except Exception as exc:
        exc.raw_response = raw_response
        raise


def score_with_retries(model_name: str, messages: list, retries: int = 20, **kwargs) -> Dict[str, Any]:
    last_error = None
    last_response = None
    for _ in range(retries):
        try:
            evaluation, last_response = request_evaluation(model_name, messages, **kwargs)
            result = {
                "status": "success",
                "model": model_name,
                "evaluation_result": evaluation,
            }
            for metric in METRICS:
                result[metric] = int(evaluation[metric]["Final Assessment"]["score"])
            return result
        except Exception as exc:
            last_error = str(exc)
            candidate = getattr(exc, "raw_response", None)
            if candidate is not None:
                last_response = candidate
    return {
        "status": "failed",
        "model": model_name,
        "error": last_error or "unknown evaluation error",
        "raw_response": last_response,
    }


def make_sample_key(setting: str, model_name: str, person_name: str, instruction_id: int) -> str:
    return f"{setting}|{model_name}|{person_name}|{instruction_id}"


def is_successful(entry: Dict[str, Any]) -> bool:
    scores = entry.get("evaluation_result", {})
    if not isinstance(scores, dict):
        return False
    return any(
        isinstance(value, dict)
        and value.get("status", "success") == "success"
        and all(metric in value for metric in METRICS)
        for value in scores.values()
    )


def summarize(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    failed = 0
    completed = 0
    for entry in entries:
        if not is_successful(entry):
            failed += 1
            continue
        completed += 1
        for model_name, score in entry["evaluation_result"].items():
            if not isinstance(score, dict) or score.get("status", "success") != "success":
                continue
            totals.setdefault(model_name, {metric: 0 for metric in METRICS})
            counts[model_name] = counts.get(model_name, 0) + 1
            for metric in METRICS:
                totals[model_name][metric] += int(score[metric])
    averages = {
        model: {metric: totals[model][metric] / counts[model] for metric in METRICS}
        for model in totals
        if counts[model]
    }
    metric_aliases = {
        model: {
            "PRC": values["Procedure"],
            "PSN": values["Personalization"],
            "PTV": values["Proactivity"],
        }
        for model, values in averages.items()
    }
    return {
        "completed_samples": completed,
        "failed_samples": failed,
        "scores": averages,
        "metrics": metric_aliases,
    }


def load_results(path: str) -> list:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def write_results(entries: list, result_file: str, summary_file: str) -> Dict[str, Any]:
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=4)
    summary = summarize(entries)
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=4)
    return summary
