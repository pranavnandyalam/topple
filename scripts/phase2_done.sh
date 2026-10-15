#!/usr/bin/env bash
# shellcheck disable=SC2086  # section 4 splits $S into three submissions
# Phase 2a done-when (docs/design/phase-2.md section 4). Evidence comes from the shipped ONNX
# directory in a separate torch-free environment (.venv-eval, gitignored), so the developer's
# .venv is never modified: a clean tree, the torch-absence check, fresh evaluation and compares,
# a re-export that must rebuild the shipped bytes, G9 (phase2_ci_check.py) and G1-G11 but G9
# (phase2_gate.py). It cannot pass before submissions/ppo_balance, the parity fixture and
# docs/results/ are committed. Run it from anywhere in the checkout.
# Every line below is a simple command, so errexit stops on any failure: the re-export and its
# cmp are two lines, never `export ... && cmp ...` (bash ignores a failure left of `&&`, so a
# failing re-export, e.g. with no runs/ in this checkout, would skip the cmp and carry on).
# The shellcheck directive above applies file-wide (it must follow the shebang directly).
set -euo pipefail
top=$(git rev-parse --show-toplevel)   # an assignment, so errexit sees a failure
cd "$top"
# docs/design/phase-2.md section 4 (test_phase2_gate checks these lines):
test -z "$(git status --porcelain -- submissions docs/results tests/data configs scripts python)"
export UV_PROJECT_ENVIRONMENT=.venv-eval            # gitignored
uv sync --locked --no-dev
uv run --no-sync python -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec("torch") is not None)'
uv run --no-sync topple eval submissions/ppo_balance -p eval_v1 --out out/ppo_eval_v1
S="submissions/pid_example.py submissions/lqr_example.py submissions/ppo_balance"
uv run --no-sync topple compare $S -p balance-v1/standard --seeds 0..199 --workers 4 --json out/cmp_cal.json
uv run --no-sync topple compare $S -p eval_v1 --workers 4 --json out/cmp_eval_v1.json
unset UV_PROJECT_ENVIRONMENT
K=$(uv run python -c 'import json;print(json.load(open("docs/results/phase2.json"))["shipped"]["run"])')
D=$(mktemp -d)/p2x
uv run topple export "runs/$K" --out "$D"
cmp "$D/policy.onnx" submissions/ppo_balance/policy.onnx
uv run python scripts/phase2_ci_check.py --head "$(git rev-parse HEAD)" --workflow ci.yml   # G9
uv run python scripts/phase2_gate.py out/cmp_cal.json out/cmp_eval_v1.json out/ppo_eval_v1/results.json \
    docs/results/phase2.json --ci-dir docs/results/phase2/ci_ubuntu \
    --submission submissions/ppo_balance --results-dir docs/results/phase2
