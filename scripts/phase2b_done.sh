#!/usr/bin/env bash
# shellcheck disable=SC2086  # $A splits into the driver and the five anchors
# Phase 2b done-when (docs/design/phase-2.md section 8.9). Evidence comes from the shipped ONNX
# directory in the torch-free .venv-eval (gitignored), so the developer's .venv is never
# modified: a clean tree, the torch-absence check, fresh compares on standard 0..399, clean
# 0..399 and wp_eval_v1, the item 2/3/9 tests, a re-export that must rebuild the shipped bytes,
# item 10 (phase2_ci_check.py) and the rest (phase2b_gate.py). Run it from anywhere in the
# checkout. As in phase2_done.sh, every line is a simple command so errexit stops on any
# failure, and command output is taken only by an assignment.
set -euo pipefail
top=$(git rev-parse --show-toplevel)
cd "$top"
dirty=$(git status --porcelain -- submissions docs/results tests/data configs scripts python)
test -z "$dirty"
export UV_PROJECT_ENVIRONMENT=.venv-eval            # gitignored
uv sync --locked          # with the dev group (pytest, below); no extras, so still torch-free
uv run --no-sync python -c 'import importlib.util,sys; sys.exit(importlib.util.find_spec("torch") is not None)'
A="submissions/waypoint_pid_example.py submissions/anchors/creeper.py submissions/anchors/gates_only.py submissions/anchors/reckless.py submissions/anchors/balance_in_place.py submissions/anchors/spin.py"
uv run --no-sync topple compare $A submissions/ppo_waypoint -p waypoint-v1/standard --seeds 0..399 --workers 4 --json out/wp_cal.json
uv run --no-sync topple compare submissions/waypoint_pid_example.py submissions/ppo_waypoint -p waypoint-v1/clean --seeds 0..399 --workers 4 --json out/wp_clean.json
uv run --no-sync topple compare submissions/waypoint_pid_example.py submissions/anchors/creeper.py submissions/ppo_waypoint -p wp_eval_v1 --workers 4 --json out/wp_eval.json
uv run --no-sync pytest -m "not train" tests/test_golden.py tests/test_challenge.py tests/test_traj_v2.py tests/test_waypoint_verify.py tests/test_waypoint_driver.py "tests/test_shipped_policy.py::test_the_shipped_policy_loads_and_matches_its_fixture[ppo_waypoint]"
unset UV_PROJECT_ENVIRONMENT
K=$(uv run python -c 'import json;print(json.load(open("docs/results/phase2b.json"))["shipped"]["run"])')
D=$(mktemp -d)/p2bx
uv run topple export "runs/$K" --out "$D"
cmp "$D/policy.onnx" submissions/ppo_waypoint/policy.onnx
head=$(git rev-parse HEAD)
uv run python scripts/phase2_ci_check.py --head "$head" --workflow ci.yml   # item 10
uv run python scripts/phase2b_gate.py out/wp_cal.json out/wp_clean.json out/wp_eval.json \
    docs/results/phase2b.json --submission submissions/ppo_waypoint \
    --fixture tests/data/ppo_waypoint_parity.npz --results-dir docs/results/phase2
