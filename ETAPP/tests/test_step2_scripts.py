import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ETAPP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ETAPP_ROOT / "scripts"


class Step2ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp = Path(self.tempdir.name)
        self.bin_dir = self.temp / "bin"
        self.bin_dir.mkdir()
        self.ready_file = self.temp / "service.ready"
        self.command_log = self.temp / "python_commands.jsonl"

        self.qwen_dir = self.temp / "Qwen2.5-72B-Instruct"
        self.minilm_dir = self.temp / "paraphrase-MiniLM-L3-v2"
        self.wikipedia_dir = self.temp / "wikipedia-kilt-doc"
        self.concrete_dir = self.temp / "concrete_profile"
        for directory in (self.qwen_dir, self.minilm_dir, self.wikipedia_dir, self.concrete_dir):
            directory.mkdir()
        (self.wikipedia_dir / "segments_1").write_text("mock", encoding="utf-8")

        self.profile_file = self.temp / "profiles.json"
        self.instruction_file = self.temp / "instruction.json"
        self.profile_file.write_text(json.dumps({"Test User": {"profile": "test"}}), encoding="utf-8")
        self.instruction_file.write_text(json.dumps([{
            "query": "test query",
            "timestamp": "2024-01-01 00:00:00",
            "location": "test location",
            "available_tools_name": [],
            "keypoint for personal": [],
            "keypoint for proactive": [],
        }]), encoding="utf-8")
        (self.concrete_dir / "profile_Test_User.json").write_text("{}", encoding="utf-8")

        self.fake_python = self.bin_dir / "fake-python"
        self.fake_curl = self.bin_dir / "fake-curl"
        self.fake_vllm = self.bin_dir / "fake-vllm"
        self.fake_nvidia_smi = self.bin_dir / "nvidia-smi"
        self.fake_ps = self.bin_dir / "ps"
        self.fake_kill = self.bin_dir / "fake-kill"
        self._write_fake_python()
        self._write_fake_curl()
        self._write_fake_vllm()
        self.fake_nvidia_smi.write_text(
            "#!/usr/bin/env bash\necho '0, Mock A100, 81920 MiB, 000.00'\n",
            encoding="utf-8",
        )
        self.fake_ps.write_text(
            '''#!/usr/bin/env bash
if [ "${FAKE_PS_UNRELATED:-0}" = "1" ]; then
    echo "sleep 30"
else
    echo "$VLLM_BIN serve $QWEN_MODEL_DIR"
fi
''',
            encoding="utf-8",
        )
        self.fake_kill.write_text(
            '''#!/usr/bin/env bash
touch "$FAKE_STOP_FILE"
exit 0
''',
            encoding="utf-8",
        )
        for executable in self.bin_dir.iterdir():
            executable.chmod(0o755)

        self.runtime_dir = self.temp / "runtime"
        self.output_dir = self.temp / "fc_output"
        self.eval_dir = self.output_dir / "evaluation"
        self.env = os.environ.copy()
        self.env.update({
            "PATH": f"{self.bin_dir}{os.pathsep}{self.env.get('PATH', '')}",
            "PYTHON_BIN": str(self.fake_python),
            "VLLM_BIN": str(self.fake_vllm),
            "CURL_BIN": str(self.fake_curl),
            "FAKE_REAL_PYTHON": sys.executable,
            "FAKE_READY_FILE": str(self.ready_file),
            "FAKE_COMMAND_LOG": str(self.command_log),
            "FAKE_STOP_FILE": str(self.temp / "service.stop"),
            "ETAPP_KILL_BIN": str(self.fake_kill),
            "QWEN_MODEL_DIR": str(self.qwen_dir),
            "MINILM_MODEL_DIR": str(self.minilm_dir),
            "WIKIPEDIA_INDEX_PATH": str(self.wikipedia_dir),
            "PROFILE_FILE": str(self.profile_file),
            "INSTRUCTION_FILE": str(self.instruction_file),
            "CONCRETE_PROFILE_DIR": str(self.concrete_dir),
            "FC_OUTPUT_DIR": str(self.output_dir),
            "FC_EVALUATION_OUTPUT_DIR": str(self.eval_dir),
            "VLLM_RUNTIME_DIR": str(self.runtime_dir),
            "VLLM_STARTUP_TIMEOUT": "10",
            "VLLM_SHUTDOWN_TIMEOUT": "5",
            "QWEN_SERVED_MODEL_NAME": "qwen-test-served",
        })

    def tearDown(self):
        pid_file = self.runtime_dir / "server.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        self.tempdir.cleanup()

    def _write_fake_python(self):
        code = f'''#!{sys.executable}
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("FAKE_COMMAND_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(args) + "\\n")

def value(flag):
    return args[args.index(flag) + 1]

if args[:2] == ["-m", "Inference.evaluate_prompted_agent"]:
    if os.environ.get("FAKE_FAIL_STAGE") == "inference":
        raise SystemExit(23)
    profiles = json.loads(Path(value("--profile-file")).read_text())
    instructions = json.loads(Path(value("--instruction-file")).read_text())
    output = Path(value("--output-dir"))
    output.mkdir(parents=True, exist_ok=True)
    for person in profiles:
        rows = []
        for source in instructions:
            row = dict(source)
            row["output"] = []
            row["tools"] = []
            rows.append(row)
        filename = person.replace(" ", "_") + "_instruction.json"
        (output / filename).write_text(json.dumps(rows), encoding="utf-8")
    raise SystemExit(0)

if args[:2] == ["-m", "evaluation.evaluate"]:
    if os.environ.get("FAKE_FAIL_STAGE") == "evaluation":
        raise SystemExit(24)
    profiles = json.loads(Path(value("--profile-file")).read_text())
    instructions = json.loads(Path(value("--instruction-file")).read_text())
    output = Path(value("--evaluate-output-dir"))
    model = value("--evaluation-model")
    output.mkdir(parents=True, exist_ok=True)
    count = len(profiles) * len(instructions)
    score = {{"status": "success", "model": model, "Procedure": 4, "Personalization": 3, "Proactivity": 5}}
    results = [{{"sample_key": str(i), "evaluation_result": {{model: score}}}} for i in range(count)]
    summary = {{
        "completed_samples": count,
        "failed_samples": 0,
        "scores": {{model: {{"Procedure": 4, "Personalization": 3, "Proactivity": 5}}}},
        "metrics": {{model: {{"PRC": 4, "PSN": 3, "PTV": 5}}}},
    }}
    (output / "evaluate_result.json").write_text(json.dumps(results), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    raise SystemExit(0)

os.execv(os.environ["FAKE_REAL_PYTHON"], [os.environ["FAKE_REAL_PYTHON"]] + args)
'''
        self.fake_python.write_text(code, encoding="utf-8")

    def _write_fake_curl(self):
        self.fake_curl.write_text(
            '''#!/usr/bin/env bash
if [ "${FAKE_CURL_NEVER_READY:-0}" = "1" ] || [ ! -f "$FAKE_READY_FILE" ]; then
    exit 7
fi
case "${!#}" in
    */models) printf '{"data":[{"id":"%s"}]}\\n' "$QWEN_SERVED_MODEL_NAME" ;;
esac
exit 0
''',
            encoding="utf-8",
        )

    def _write_fake_vllm(self):
        self.fake_vllm.write_text(
            '''#!/usr/bin/env bash
if [ "${FAKE_VLLM_EXIT_EARLY:-0}" = "1" ]; then
    exit 9
fi
touch "$FAKE_READY_FILE"
cleanup() {
    rm -f "$FAKE_READY_FILE"
    exit 0
}
trap cleanup TERM INT
while true; do
    if [ -f "$FAKE_STOP_FILE" ]; then
        cleanup
    fi
    sleep 1
done
''',
            encoding="utf-8",
        )

    def run_script(self, name, env=None, timeout=30):
        return subprocess.run(
            ["bash", str(SCRIPTS_DIR / name)],
            cwd=self.temp,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def logged_commands(self):
        if not self.command_log.exists():
            return []
        return [json.loads(line) for line in self.command_log.read_text().splitlines()]

    def test_all_shell_scripts_pass_bash_syntax(self):
        scripts = list(SCRIPTS_DIR.glob("*.sh")) + list((SCRIPTS_DIR / "lib").glob("*.sh"))
        for script in scripts:
            result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_smoke_input_preparation_is_source_anchored(self):
        smoke_dir = self.temp / "smoke"
        env = self.env | {"SMOKE_INPUT_DIR": str(smoke_dir), "PYTHON_BIN": sys.executable}
        result = self.run_script("09_prepare_fc_smoke_inputs.sh", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((smoke_dir / "profiles.json").read_text())), 1)
        self.assertEqual(len(json.loads((smoke_dir / "instruction.json").read_text())), 1)

    def test_start_and_stop_manage_only_recorded_service(self):
        started = self.run_script("10_start_qwen_vllm.sh")
        self.assertEqual(started.returncode, 0, started.stderr)
        pid = int((self.runtime_dir / "server.pid").read_text().strip())
        os.kill(pid, 0)
        stopped = self.run_script("13_stop_qwen_vllm.sh")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse((self.runtime_dir / "server.pid").exists())
        self.assertFalse(self.ready_file.exists())

    def test_start_refuses_unknown_healthy_endpoint(self):
        self.ready_file.touch()
        result = self.run_script("10_start_qwen_vllm.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not owned", result.stderr)

    def test_start_reports_service_process_early_exit(self):
        env = self.env | {"FAKE_VLLM_EXIT_EARLY": "1"}
        result = self.run_script("10_start_qwen_vllm.sh", env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exited during startup", result.stderr)
        self.assertFalse((self.runtime_dir / "server.pid").exists())

    def test_start_timeout_cleans_runtime_files(self):
        env = self.env | {"FAKE_CURL_NEVER_READY": "1", "VLLM_STARTUP_TIMEOUT": "1"}
        result = self.run_script("10_start_qwen_vllm.sh", env, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not become ready", result.stderr)
        self.assertFalse((self.runtime_dir / "server.pid").exists())

    def test_stop_refuses_unrelated_live_pid(self):
        self.runtime_dir.mkdir(parents=True)
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            (self.runtime_dir / "server.pid").write_text(str(sleeper.pid), encoding="utf-8")
            (self.runtime_dir / "model_path").write_text(str(self.qwen_dir), encoding="utf-8")
            env = self.env | {"FAKE_PS_UNRELATED": "1"}
            result = self.run_script("13_stop_qwen_vllm.sh", env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(sleeper.poll())
        finally:
            sleeper.terminate()
            sleeper.wait(timeout=5)

    def test_generation_and_evaluation_forward_required_arguments(self):
        self.ready_file.touch()
        generated = self.run_script("11_run_fc_retrieval.sh")
        self.assertEqual(generated.returncode, 0, generated.stderr)
        evaluated = self.run_script("12_evaluate_fc_local_qwen.sh")
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        commands = self.logged_commands()
        inference = next(args for args in commands if args[:2] == ["-m", "Inference.evaluate_prompted_agent"])
        evaluation = next(args for args in commands if args[:2] == ["-m", "evaluation.evaluate"])
        self.assertEqual(inference[inference.index("--generation-temperature") + 1], "0.0")
        self.assertEqual(inference[inference.index("--max-model-len") + 1], "32768")
        self.assertIn("--use-retrieval", inference)
        self.assertEqual(evaluation[evaluation.index("--evaluation-model") + 1], "qwen-test-served")
        self.assertEqual(evaluation[evaluation.index("--evaluation-api-key") + 1], "EMPTY")

    def test_pipeline_cleans_up_owned_service_on_success(self):
        result = self.run_script("19_run_fc_pipeline.sh", timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.runtime_dir / "server.pid").exists())
        self.assertFalse(self.ready_file.exists())
        self.assertTrue((self.eval_dir / "summary.json").exists())

    def test_pipeline_cleans_up_owned_service_on_generation_failure(self):
        env = self.env | {"FAKE_FAIL_STAGE": "inference"}
        result = self.run_script("19_run_fc_pipeline.sh", env, timeout=40)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.runtime_dir / "server.pid").exists())
        self.assertFalse(self.ready_file.exists())

    def test_pipeline_cleans_up_owned_service_on_evaluation_failure(self):
        env = self.env | {"FAKE_FAIL_STAGE": "evaluation"}
        result = self.run_script("19_run_fc_pipeline.sh", env, timeout=40)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.runtime_dir / "server.pid").exists())
        self.assertFalse(self.ready_file.exists())


if __name__ == "__main__":
    unittest.main()
