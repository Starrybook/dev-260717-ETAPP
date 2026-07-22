#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

require_command "$PYTHON_BIN"
require_file "default profile file" "$ETAPP_ROOT/profile/profiles.json"
require_file "default instruction file" "$ETAPP_ROOT/data/instruction/instruction.json"

mkdir -p "$SMOKE_INPUT_DIR"
SMOKE_PROFILE_FILE="$SMOKE_INPUT_DIR/profiles.json"
SMOKE_INSTRUCTION_FILE="$SMOKE_INPUT_DIR/instruction.json"

"$PYTHON_BIN" - "$ETAPP_ROOT/profile/profiles.json" "$ETAPP_ROOT/data/instruction/instruction.json" "$SMOKE_PROFILE_FILE" "$SMOKE_INSTRUCTION_FILE" <<'PY'
import json
import sys
from pathlib import Path

profile_source, instruction_source, profile_target, instruction_target = map(Path, sys.argv[1:])
profiles = json.loads(profile_source.read_text(encoding="utf-8"))
instructions = json.loads(instruction_source.read_text(encoding="utf-8"))
if not profiles or not instructions:
    raise SystemExit("Source profiles and instructions must be non-empty")
first_name = next(iter(profiles))
profile_target.write_text(json.dumps({first_name: profiles[first_name]}, ensure_ascii=False, indent=4), encoding="utf-8")
instruction_target.write_text(json.dumps([instructions[0]], ensure_ascii=False, indent=4), encoding="utf-8")
print(f"Prepared FC smoke input: user={first_name!r}, instruction={instructions[0]['query']!r}")
PY

info "Smoke profile: $SMOKE_PROFILE_FILE"
info "Smoke instruction: $SMOKE_INSTRUCTION_FILE"
printf 'export PROFILE_FILE=%q\n' "$SMOKE_PROFILE_FILE"
printf 'export INSTRUCTION_FILE=%q\n' "$SMOKE_INSTRUCTION_FILE"
printf 'export FC_OUTPUT_DIR=%q\n' "$ETAPP_ROOT/output/fc_qwen72b_tool_retrieval_smoke"
