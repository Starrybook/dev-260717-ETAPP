import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ETAPP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ETAPP_ROOT))


class FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "one two three"

    def encode(self, prompt, add_special_tokens=False):
        return prompt.split()


class CliAndPathTests(unittest.TestCase):
    def test_inference_cli_accepts_old_and_kebab_names(self):
        from Inference.arguments import REPO_ROOT, build_inference_parser, finalize_inference_args

        self.assertEqual(REPO_ROOT, ETAPP_ROOT)
        parser = build_inference_parser()
        args = parser.parse_args([
            "--profile-file", "/profiles.json",
            "--instruction_file", "/instructions.json",
            "--output-dir", "/tmp/explicit-output",
            "--max_new_tokens", "1024",
        ])
        self.assertEqual(args.profile_file, "/profiles.json")
        self.assertEqual(args.instruction_file, "/instructions.json")
        self.assertEqual(args.max_new_tokens, 1024)
        with mock.patch.object(Path, "mkdir"):
            finalized = finalize_inference_args(args)
        self.assertEqual(finalized.output_dir, "/tmp/explicit-output")

    def test_all_entrypoint_help_works_from_another_cwd(self):
        modules = [
            "Inference.evaluate_prompted_agent",
            "Inference.evaluate_prompted_agent_reason_inference",
            "evaluation.evaluate",
            "evaluation.evaluate_reason_inference",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ETAPP_ROOT)
        with tempfile.TemporaryDirectory() as cwd:
            for module in modules:
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=cwd,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_evaluation_cli_defaults_and_local_overrides(self):
        from evaluation.common import build_evaluation_parser, finalize_evaluation_args

        parser = build_evaluation_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.evaluation_model, "gpt-4o-2024-11-20")
        args = parser.parse_args([
            "--output-dir", "/tmp/generated",
            "--evaluate_output_dir", "/tmp/evaluated",
            "--evaluation-base-url", "http://judge/v1",
            "--evaluation_model", "qwen-judge",
        ])
        with mock.patch.object(Path, "mkdir"):
            finalized = finalize_evaluation_args(args)
        self.assertEqual(finalized.result_file, "/tmp/generated")
        self.assertEqual(finalized.evaluate_result_file, "/tmp/evaluated/evaluate_result.json")
        self.assertEqual(finalized.evaluation_model, "qwen-judge")


class OpenModelTests(unittest.TestCase):
    def _make_model(self, **overrides):
        from Inference import models

        client = mock.Mock()
        completion = SimpleNamespace(choices=[SimpleNamespace(text="ok")])
        client.completions.create.return_value = completion
        constructor = mock.Mock(return_value=client)
        kwargs = dict(
            model_name="qwen-local",
            model_name_or_path="/models/qwen",
            use_vllm=True,
            vllm_base_url="http://explicit:8000/v1",
            served_model_name="qwen-served",
            vllm_api_key="local-key",
            max_new_tokens=123,
        )
        kwargs.update(overrides)
        with mock.patch.object(models.OpenModel, "_load_tokenizer"), mock.patch.object(models, "OpenAI", constructor):
            model = models.OpenModel(**kwargs)
        model.tokenizer = FakeTokenizer()
        return model, client, constructor

    def test_explicit_endpoint_and_model_override_legacy_environment(self):
        with mock.patch.dict(os.environ, {"url": "http://legacy/v1", "qwen_model_name": "legacy-model"}):
            model, _client, constructor = self._make_model()
        self.assertEqual(model.vllmurl_model_name, "qwen-served")
        self.assertEqual(constructor.call_args.kwargs["base_url"], "http://explicit:8000/v1")
        self.assertEqual(constructor.call_args.kwargs["api_key"], "local-key")

    def test_vllm_receives_configured_generation_budget(self):
        model, client, _constructor = self._make_model()
        model.change_messages([{"role": "user", "content": "hello"}])
        self.assertEqual(model.prediction("react", timestamp="2024-01-01 00:00:00"), "ok")
        self.assertEqual(client.completions.create.call_args.kwargs["max_tokens"], 123)

    def test_context_overflow_is_reported_before_request(self):
        model, client, _constructor = self._make_model(max_new_tokens=4, max_model_len=6)
        model.change_messages([{"role": "user", "content": "hello"}])
        with self.assertRaisesRegex(ValueError, "exceed max_model_len"):
            model.prediction("react", timestamp="2024-01-01 00:00:00")
        client.completions.create.assert_not_called()

    def test_legacy_generation_defaults_are_preserved(self):
        model, _client, _constructor = self._make_model(max_new_tokens=None)
        self.assertEqual(model._generation_budget(), 16384)
        model.use_vllm = False
        self.assertEqual(model._generation_budget(), 1024)


class EvaluationTests(unittest.TestCase):
    SCORE = {
        "Procedure": {"Final Assessment": {"score": "4"}},
        "Personalization": {"Final Assessment": {"score": "3"}},
        "Proactivity": {"Final Assessment": {"score": "5"}},
    }

    def test_direct_and_fenced_json_are_supported(self):
        from evaluation.common import parse_evaluation_json

        raw = json.dumps(self.SCORE)
        self.assertEqual(parse_evaluation_json(raw), self.SCORE)
        self.assertEqual(parse_evaluation_json(f"```json\n{raw}\n```"), self.SCORE)

    def test_local_endpoint_does_not_read_cloud_key(self):
        from evaluation import common

        fake_openai = SimpleNamespace(OpenAI=mock.Mock(return_value=mock.Mock()))
        with mock.patch.dict(sys.modules, {"openai": fake_openai}), mock.patch.dict(os.environ, {"API_KEY": "cloud-secret"}):
            common.create_evaluation_client("Qwen2.5-72B-Instruct", "http://judge/v1", None)
        kwargs = fake_openai.OpenAI.call_args.kwargs
        self.assertEqual(kwargs, {"base_url": "http://judge/v1", "api_key": "EMPTY"})

    def test_default_cloud_branch_keeps_original_api_key(self):
        from evaluation import common

        fake_openai = SimpleNamespace(OpenAI=mock.Mock(return_value=mock.Mock()))
        with mock.patch.dict(sys.modules, {"openai": fake_openai}), mock.patch.dict(os.environ, {"API_KEY": "cloud-key"}):
            common.create_evaluation_client("gpt-4o-2024-11-20")
        self.assertEqual(fake_openai.OpenAI.call_args.kwargs["api_key"], "cloud-key")

    def test_local_judge_request_uses_configured_model_and_limits(self):
        from evaluation import common

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.SCORE)))]
        )
        client = mock.Mock()
        client.chat.completions.create.return_value = response
        with mock.patch.object(common, "create_evaluation_client", return_value=client):
            parsed, raw = common.request_evaluation(
                "Qwen2.5-72B-Instruct",
                [{"role": "user", "content": "judge"}],
                base_url="http://judge/v1",
                max_tokens=777,
                temperature=0.2,
            )
        self.assertEqual(parsed, self.SCORE)
        self.assertEqual(json.loads(raw), self.SCORE)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "Qwen2.5-72B-Instruct")
        self.assertEqual(request["max_tokens"], 777)
        self.assertEqual(request["temperature"], 0.2)

    def test_retry_failure_keeps_last_raw_response(self):
        from evaluation import common

        error = ValueError("invalid JSON")
        error.raw_response = "not-json"
        with mock.patch.object(common, "request_evaluation", side_effect=error):
            result = common.score_with_retries("qwen-judge", [], retries=2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["raw_response"], "not-json")

    def test_stable_keys_subset_ids_and_resume_summary(self):
        from evaluation.common import make_sample_key, summarize
        from evaluation.evaluate import get_instruction_id as fc_instruction_id
        from evaluation.evaluate_reason_inference import get_instruction_id as react_instruction_id

        self.assertEqual(fc_instruction_id(3, 7), 7)
        self.assertEqual(react_instruction_id(0, 0), 6)
        key = make_sample_key("tool-retrieval", "qwen-judge", "Alice", 6)
        score = {"status": "success", "model": "qwen-judge", **{k: int(v["Final Assessment"]["score"]) for k, v in self.SCORE.items()}}
        entries = [
            {"sample_key": key, "evaluation_result": {"qwen-judge": score}},
            {"sample_key": "failed", "evaluation_result": {"qwen-judge": {"status": "failed"}}},
        ]
        summary = summarize(entries)
        self.assertEqual(summary["completed_samples"], 1)
        self.assertEqual(summary["failed_samples"], 1)
        self.assertEqual(summary["scores"]["qwen-judge"]["Procedure"], 4)
        self.assertEqual(summary["metrics"]["qwen-judge"], {"PRC": 4, "PSN": 3, "PTV": 5})


class ToolResourceTests(unittest.TestCase):
    def test_tool_manager_maps_local_resource_paths(self):
        from toolkit.tool_manager import tool_constructor_kwargs

        self.assertEqual(
            tool_constructor_kwargs("Tool_And_History_Searcher", "/models/minilm", "/indexes/wiki"),
            {"model_name_or_path": "/models/minilm"},
        )
        self.assertEqual(
            tool_constructor_kwargs("Browser", "/models/minilm", "/indexes/wiki"),
            {"wikipedia_index_path": "/indexes/wiki"},
        )


if __name__ == "__main__":
    unittest.main()
