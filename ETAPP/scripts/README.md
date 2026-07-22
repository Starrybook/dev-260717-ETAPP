# ETAPP FC 本地运行脚本

本目录只编排 FC / Tool Retrieval 的 Qwen2.5-72B-Instruct 生成和本地评测，不安装环境、不下载模型或索引，也不包含 ReAct。资源准备见 [`../reproduce_prepare_download.md`](../reproduce_prepare_download.md)。

每个脚本的输入要求、内部处理、输出文件和失败行为详见 [`script_reference.md`](script_reference.md)。

## 1. 配置

所有脚本都从自身位置定位 ETAPP 根目录，可以从任意当前目录执行。先导出资源路径；也可以先 source `config.example.sh` 查看和采用推荐默认值。

```bash
export ETAPP_RESOURCE_ROOT=/data/etapp-resources
export QWEN_MODEL_REVISION=recorded-hugging-face-commit-sha

# 若资源没有使用 reproduce_prepare_download.md 中的推荐目录结构，分别覆盖：
# export QWEN_MODEL_DIR=/models/Qwen2.5-72B-Instruct
# export MINILM_MODEL_DIR=/models/paraphrase-MiniLM-L3-v2
# export WIKIPEDIA_INDEX_PATH=/indexes/wikipedia-kilt-doc
```

`ETAPP_RESOURCE_ROOT` 默认推导：

```text
models/Qwen2.5-72B-Instruct
models/paraphrase-MiniLM-L3-v2
indexes/wikipedia-kilt-doc
```

常用可选覆盖：

```bash
export VLLM_PORT=8000
export VLLM_MAX_MODEL_LEN=32768
export FC_MAX_NEW_TOKENS=1024
export FC_GENERATION_TEMPERATURE=0.0
export FC_OUTPUT_DIR=/data/etapp-output/fc_qwen72b_tool_retrieval
export FC_EVALUATION_OUTPUT_DIR="$FC_OUTPUT_DIR/evaluation"
```

脚本默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。本地 endpoint 固定使用真实 served model name `qwen2.5-72b-etapp`，不会读取云端 `API_KEY`。

## 2. 单样本 smoke

先生成一个用户、一条指令的输入副本：

```bash
bash scripts/09_prepare_fc_smoke_inputs.sh
```

复制该命令最后输出的三条 `export` 命令，或按相同含义设置以下变量：

```bash
export PROFILE_FILE=/absolute/path/to/ETAPP/output/smoke_inputs/fc/profiles.json
export INSTRUCTION_FILE=/absolute/path/to/ETAPP/output/smoke_inputs/fc/instruction.json
export FC_OUTPUT_DIR=/absolute/path/to/ETAPP/output/fc_qwen72b_tool_retrieval_smoke
export FC_EVALUATION_OUTPUT_DIR="$FC_OUTPUT_DIR/evaluation"
```

然后使用下述分阶段或端到端方式运行。切换到正式实验前应取消这些变量，或重新导出完整数据路径和正式输出目录。

## 3. 分阶段运行

```bash
bash scripts/10_start_qwen_vllm.sh
bash scripts/11_run_fc_retrieval.sh
bash scripts/12_evaluate_fc_local_qwen.sh
bash scripts/13_stop_qwen_vllm.sh
```

这种方式允许在生成和评测之间检查轨迹，也允许 11、12 脚本复用已经由 10 启动的服务。停止脚本只处理其 PID 文件记录且命令行匹配的 vLLM 进程；若 endpoint 属于其他服务，它会拒绝终止。

## 4. 端到端运行

```bash
bash scripts/19_run_fc_pipeline.sh
```

端到端入口要求目标端口当前未被占用。它拥有自己启动的服务，并在成功、阶段失败、SIGINT 或 SIGTERM 时执行清理。若需要使用已经运行的 Qwen 服务，应改用分阶段的 11 和 12。

## 5. 输出和续跑

正式默认输出：

```text
output/fc_qwen72b_tool_retrieval/
├── <user>_instruction.json
├── inference.log
├── controller.stdout.log
├── run_metadata/generation.txt
└── evaluation/
    ├── evaluate_result.json
    ├── summary.json
    ├── evaluation.log
    ├── evaluator.stdout.log
    └── run_metadata/evaluation.txt
```

vLLM 的 PID、服务日志和启动元数据位于 `output/runtime/qwen_vllm/`。

- FC 输出已经包含某用户完整的全部指令时，生成入口跳过该用户。
- 某用户文件缺失、为空或不完整时，现有控制器重新生成该用户，并在完成后覆盖该用户文件。
- 评测仅跳过具有成功状态和稳定样本键的结果；失败样本在下一次运行中重试。
- 阶段脚本结束前会按当前 profile 和 instruction 的实际数量验证轨迹及评测结果。正式数据应为 16×50=800；smoke 数据则自动按 1×1 验证。
- 评测验证要求 `failed_samples=0`，并确认本地 Qwen 的 PRC、PSN、PTV 均存在。若验证失败，结果文件仍会保留供排查和续跑。

## 6. 常见失败

- `WIKIPEDIA_INDEX_PATH` 必须直接包含 Lucene `segments_*` 文件。
- 启动脚本若发现没有对应 PID 文件但 `/health` 已可访问，会把它视为未知服务并退出。
- 若 `prompt_tokens + FC_MAX_NEW_TOKENS` 超过 `VLLM_MAX_MODEL_LEN`，Python 会给出明确的上下文超限错误；不要只提高生成上限而不提高服务上下文。
- 评测输出 JSON 被截断时，优先确认上下文预算，再按需调整 `EVALUATION_MAX_TOKENS`。
