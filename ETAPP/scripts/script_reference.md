# ETAPP FC 脚本输入、处理与输出说明

本文档逐一说明 `scripts/` 中 FC / Tool Retrieval 本地运行脚本的输入要求、处理过程和输出内容。所有可执行脚本都会根据自身路径定位 ETAPP 根目录，因此可以从任意当前工作目录调用；这些脚本不安装环境，也不下载模型、数据或索引。

## 1. 脚本关系

```text
config.example.sh（可选配置模板）
              │
              └── lib/common.sh（所有阶段共用）

09_prepare_fc_smoke_inputs.sh（可选：制作 1×1 smoke 输入）

10_start_qwen_vllm.sh
          ↓
11_run_fc_retrieval.sh
          ↓
12_evaluate_fc_local_qwen.sh
          ↓
13_stop_qwen_vllm.sh

19_run_fc_pipeline.sh = 按顺序编排 10 → 11 → 12，并在退出时执行 13
```

环境变量的取值优先级为：调用脚本前已显式设置的值 → 根据 `ETAPP_RESOURCE_ROOT` 推导的资源路径 → `lib/common.sh` 中的默认值。路径若未特别说明，均为相对于 ETAPP 根目录的默认路径。

## 2. `config.example.sh`

### 输入要求

该文件应使用 `source scripts/config.example.sh` 加载，而不是作为独立实验阶段执行。资源路径可以采用以下两种方式之一：

- 设置 `ETAPP_RESOURCE_ROOT`，由脚本推导 Qwen、MiniLM 和 Wikipedia 索引路径。
- 分别设置 `QWEN_MODEL_DIR`、`MINILM_MODEL_DIR` 和 `WIKIPEDIA_INDEX_PATH`。

可选配置包括：

- 服务：`VLLM_HOST`、`VLLM_PORT`、`QWEN_SERVED_MODEL_NAME`、`QWEN_MODEL_REVISION`。
- vLLM：`CUDA_VISIBLE_DEVICES`、`VLLM_TENSOR_PARALLEL_SIZE`、`VLLM_DTYPE`、`VLLM_MAX_MODEL_LEN`、`VLLM_GPU_MEMORY_UTILIZATION`、启动及停止超时。
- FC：最大轮数、生成长度、生成温度、最大 observation 长度和请求超时。
- 评测：最大生成长度、温度和请求超时。
- 命令：`PYTHON_BIN`、`VLLM_BIN`、`CURL_BIN`。
- 数据与输出：`INSTRUCTION_FILE`、`PROFILE_FILE`、`CONCRETE_PROFILE_DIR`、`FC_OUTPUT_DIR`、`FC_EVALUATION_OUTPUT_DIR`。

### 处理过程

该文件只为尚未设置的变量赋推荐默认值。默认配置为 4 张 GPU、TP=4、BF16、32,768 token 上下文、FC 最大生成 1,024 token、评测最大生成 8,192 token，服务绑定 `127.0.0.1:8000`。

### 输出内容

加载后只改变当前 shell 及其子进程的环境变量，不创建文件、不启动服务，也不执行下载。

## 3. `lib/common.sh`

### 输入要求

该文件是公共函数库，由其他脚本自动 `source`，通常不应直接执行。它读取 `config.example.sh` 所列环境变量，并提供以下默认路径：

- 指令：`data/instruction/instruction.json`
- 用户画像：`profile/profiles.json`
- 具体画像目录：`profile/concrete_profile/`
- FC 输出：`output/fc_qwen72b_tool_retrieval/`
- 评测输出：`$FC_OUTPUT_DIR/evaluation/`
- vLLM 运行目录：`output/runtime/qwen_vllm/`
- smoke 输入目录：`output/smoke_inputs/fc/`

### 处理过程

公共库承担以下工作：

- 根据 `BASH_SOURCE` 计算 `ETAPP_SCRIPTS_DIR` 和 `ETAPP_ROOT`，消除当前工作目录依赖。
- 推导资源、数据、输出、endpoint、PID 和日志路径。
- 检查命令、文件、目录以及整数和非负数配置。
- 检查 Wikipedia Lucene 索引目录是否直接包含 `segments_*` 文件。
- 通过 `/health` 和 `/v1/models` 检查服务状态及真实 served model name。
- 结合 PID、进程命令和模型路径识别本脚本管理的 vLLM 进程。
- 记录 shell-escaped 完整命令、时间、Git commit、Python/PyTorch/vLLM 版本、GPU 摘要和关键配置。
- 将阶段 stdout/stderr 同时写入日志，并从 ETAPP 根目录执行 Python 命令。
- 动态校验 FC 轨迹数量、字段和 query 对齐关系。
- 动态校验评测完成数、失败数、结果数以及 PRC、PSN、PTV 是否存在。

### 输出内容

仅加载公共库不会生成实验文件。各阶段调用其函数后，会生成对应的元数据和 stdout/stderr 日志；校验结果会打印到终端，校验失败时返回非零状态。

## 4. `09_prepare_fc_smoke_inputs.sh`

### 输入要求

- ETAPP 原始画像文件 `profile/profiles.json` 必须存在且非空。
- ETAPP 原始指令文件 `data/instruction/instruction.json` 必须存在且非空。
- `PYTHON_BIN` 必须可用。
- 可通过 `SMOKE_INPUT_DIR` 覆盖默认输出目录。

此脚本固定从仓库的完整原始数据选择样本，不读取已覆盖的 `PROFILE_FILE` 或 `INSTRUCTION_FILE`。

### 处理过程

脚本读取完整画像和指令 JSON，按原文件顺序选择第一个用户和第一条指令，分别写成仅包含一个元素的副本。原始数据不会被修改。完成后，它打印后续 smoke 运行所需的 `PROFILE_FILE`、`INSTRUCTION_FILE` 和 `FC_OUTPUT_DIR` 导出命令。

### 输出内容

默认生成：

```text
output/smoke_inputs/fc/
├── profiles.json       # 仅含第一个用户
└── instruction.json    # 仅含第一条指令
```

终端还会输出被选中的用户、指令，以及三条可复制执行的 `export` 命令。该脚本不启动模型，也不执行 FC 生成或评测。

## 5. `10_start_qwen_vllm.sh`

### 输入要求

- 命令：`PYTHON_BIN`、`VLLM_BIN`、`CURL_BIN`、`nohup` 和 `kill` 必须可用。
- 模型：`QWEN_MODEL_DIR` 必须指向本地 Qwen2.5-72B-Instruct 目录。
- 服务：host、port、served model name、启动超时。
- 推理资源：可见 GPU、tensor parallel 数、dtype、最大上下文和 GPU 显存利用率。
- 目标 endpoint 在首次启动时必须未被未知服务占用。

### 处理过程

脚本首先校验命令、目录及数值配置，然后检查 PID 文件和 endpoint：

- 若 PID 指向一个健康、模型名匹配且由脚本管理的服务，则幂等成功退出。
- 若 PID 指向不匹配的存活进程，或 endpoint 可访问但没有受管 PID，则拒绝覆盖。
- 若 PID 已失效，则删除过期的 PID 和模型路径记录。

随后脚本设置 Hugging Face/Transformers 离线模式，以 `nohup vllm serve` 后台启动 Qwen，并显式传入 served model name、TP、dtype、上下文长度、显存利用率、host 和 port。启动期间轮询 `/health` 与 `/v1/models`；进程提前退出或超过 `VLLM_STARTUP_TIMEOUT` 时，脚本会终止自有进程、保留服务日志并返回失败。

### 输出内容

默认生成：

```text
output/runtime/qwen_vllm/
├── server.pid          # 后台服务 PID
├── model_path          # 启动时使用的模型路径
├── server.log          # vLLM stdout/stderr
└── start_metadata.txt  # 启动命令、版本、GPU 和配置快照
```

成功后提供 `http://$VLLM_HOST:$VLLM_PORT` 服务，OpenAI-compatible base URL 为其 `/v1` 子路径。

## 6. `11_run_fc_retrieval.sh`

### 输入要求

- 数据：`INSTRUCTION_FILE`、`PROFILE_FILE`、`CONCRETE_PROFILE_DIR`。
- 本地资源：`QWEN_MODEL_DIR`、`MINILM_MODEL_DIR`、`WIKIPEDIA_INDEX_PATH`；索引根目录必须直接包含 `segments_*`。
- 服务：Qwen vLLM 的 `/health` 可访问，且 `/v1/models` 中包含 `QWEN_SERVED_MODEL_NAME`。
- 生成参数：`REQUEST_TIMEOUT`、`FC_MAX_TURN`、`FC_MAX_NEW_TOKENS`、`FC_GENERATION_TEMPERATURE`、`FC_MAX_OBSERVATION_LENGTH`、`VLLM_MAX_MODEL_LEN`。
- 输出：可选 `FC_OUTPUT_DIR`；目录会在需要时创建，但既有结果不会预先删除。

### 处理过程

脚本启用离线模式，并从 ETAPP 根目录调用：

```text
python -m Inference.evaluate_prompted_agent
```

调用固定为 `OpenModel + Qwen2.5-72B-Instruct + fine-tuned + function_calling + Tool Retrieval + vLLM`，同时显式传入 endpoint、served model、`EMPTY` 协议占位 key、数据路径、本地 MiniLM、本地 Wikipedia 索引、生成预算、输出目录和日志路径。

Python 入口会跳过已经具有完整指令数的用户文件；文件缺失、为空或不完整的用户按现有逻辑整用户重跑。生成结束后，脚本检查：

- 输出用户文件集合与 profile 完全一致。
- 每个用户的轨迹数与 instruction 数完全一致。
- 每条轨迹含 `query`、`timestamp`、`output`、`tools` 字段。
- 输出 query 与原始 instruction 顺序一致。

### 输出内容

默认生成：

```text
output/fc_qwen72b_tool_retrieval/
├── <user>_instruction.json       # 每位用户的 FC 轨迹
├── inference.log                 # Python 推理日志
├── controller.stdout.log         # 阶段 stdout/stderr
└── run_metadata/
    └── generation.txt            # 完整命令和运行环境快照
```

默认正式数据应产生 16 个用户文件、共 800 条轨迹；smoke 输入则按实际的 1×1 规模校验。

## 7. `12_evaluate_fc_local_qwen.sh`

### 输入要求

- 与生成阶段一致的 `INSTRUCTION_FILE`、`PROFILE_FILE` 和 `CONCRETE_PROFILE_DIR`。
- `FC_OUTPUT_DIR` 必须存在，并先通过完整 FC 输出校验。
- Qwen vLLM 必须健康，且 served model name 匹配。
- 评测参数：`EVALUATION_MAX_TOKENS`、`EVALUATION_TEMPERATURE`、`EVALUATION_REQUEST_TIMEOUT`。
- 可通过 `FC_EVALUATION_OUTPUT_DIR` 覆盖评测输出目录。

### 处理过程

脚本先校验全部 FC 轨迹，再从 ETAPP 根目录调用：

```text
python -m evaluation.evaluate
```

评测命令显式使用同一 Qwen 的本地 `/v1` endpoint、真实 served model name 和 `EMPTY` 协议占位 key，不依赖真实云端 API key。评测设置名固定为 `fc_qwen72b_tool_retrieval`。

续跑时，评测器根据稳定样本键只跳过成功结果，失败样本会再次尝试；每次都从当前全部结果重算汇总。脚本随后检查完成样本数是否等于 `profile 数 × instruction 数`、失败数是否为 0、结果行数是否匹配，以及该 served model 下是否存在 PRC、PSN、PTV。

### 输出内容

默认生成：

```text
output/fc_qwen72b_tool_retrieval/evaluation/
├── evaluate_result.json          # 逐样本评测结果，包括失败记录
├── summary.json                  # 完成数、失败数和 PRC/PSN/PTV 汇总
├── evaluation.log                # Python 评测日志
├── evaluator.stdout.log          # 阶段 stdout/stderr
└── run_metadata/
    └── evaluation.txt            # 完整命令和运行环境快照
```

若仍有失败样本，结果文件会保留以便续跑，但阶段最终校验返回非零状态。

## 8. `13_stop_qwen_vllm.sh`

### 输入要求

- `VLLM_RUNTIME_DIR` 中由启动脚本写入的 `server.pid` 和 `model_path`。
- `VLLM_SHUTDOWN_TIMEOUT` 必须为正整数。
- `VLLM_HOST`、`VLLM_PORT` 应与启动时一致，以检查停止后的 endpoint。

### 处理过程

脚本只停止能够被严格确认属于本套脚本的进程：PID 必须存活，进程命令必须包含 `vllm serve`，并且模型路径必须与记录一致。

- 无 PID 且 endpoint 不可访问时，视为无需停止并成功退出。
- 无 PID 但 endpoint 可访问时，视为未知服务并拒绝停止。
- PID 已退出时，只清理过期运行文件。
- 验证通过后先发送 SIGTERM；超过 `VLLM_SHUTDOWN_TIMEOUT` 仍未退出才发送 SIGKILL。
- 进程退出后继续检查 endpoint；若 30 秒后仍可访问，则返回失败，防止误报停止成功。

### 输出内容

成功停止后删除 `server.pid` 和 `model_path`，但保留 `server.log` 与 `start_metadata.txt` 供审计和排错。脚本不会使用模糊进程名执行 `pkill`，也不会停止未知服务。

## 9. `19_run_fc_pipeline.sh`

### 输入要求

该脚本需要启动、生成和评测三个阶段的全部输入，包括本地 Qwen/MiniLM/Wikipedia 资源、完整数据、4-GPU vLLM 配置、生成参数和评测参数。开始运行时还要求：

- 不存在受管 PID 文件。
- 目标 endpoint 没有任何健康服务响应。

若要复用已经运行的 Qwen 服务，应分别执行 `11_run_fc_retrieval.sh` 和 `12_evaluate_fc_local_qwen.sh`，不能使用该端到端入口。

### 处理过程

脚本依次执行：

1. `10_start_qwen_vllm.sh`
2. `11_run_fc_retrieval.sh`
3. `12_evaluate_fc_local_qwen.sh`

启动成功后，pipeline 标记自己拥有该服务。它通过 `EXIT`、`SIGINT` 和 `SIGTERM` trap 保证在正常完成、生成失败、评测失败或用户中断时调用 `13_stop_qwen_vllm.sh`。若清理自有服务失败，pipeline 会返回失败状态。

### 输出内容

该入口产生第 5、6、7 节所列的服务运行文件、FC 轨迹、评测结果、日志和元数据。正常结束或任一阶段失败后，自有 vLLM 服务都会被停止，PID 和模型路径记录被清理；服务日志、轨迹和评测结果会保留。

## 10. 正式运行与 smoke 输出规模

| 运行类型 | profile 数 | instruction 数 | 预期轨迹/评测数 |
|---|---:|---:|---:|
| 正式默认数据 | 16 | 50 | 800 |
| `09` 生成的 smoke 数据 | 1 | 1 | 1 |

脚本校验实际读取到的数据规模，不把 800 硬编码为所有运行的要求。因此，只要同时设置匹配的 `PROFILE_FILE`、`INSTRUCTION_FILE` 和独立输出目录，也可以使用其他受控子集进行调试。
