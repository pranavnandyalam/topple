"""The torch-free surface stays torch-free (docs/design/phase-2.md §6 ``test_import_hygiene``).

Eval, compare, shove, pack, export and CI never need torch, Stable-Baselines3, TensorBoard or
psutil (the ``train`` extra), and neither do the torch-free training modules (configs, wrappers,
selection: ``make_train_env`` imports SB3's ``Monitor`` only when called). A fresh interpreter
imports every torch-free module with an import hook that records (and refuses) any attempt to
import one of them, so a guarded ``try: import torch`` counts as much as a plain one, whether or
not the package is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEAVY = ("torch", "stable_baselines3", "tensorboard", "psutil")
MODULES = (
    "topple.cli",
    "topple.pool",
    "topple.controllers.onnx_policy",
    "topple.compare",
    "topple.bench",
    "topple.stats",
    "topple.export",
    "topple._npz",
    "topple.yamlsafe",
    "topple.train",
    "topple.train.config",
    "topple.train.wrappers",
    "topple.train.selection",
    # scripts/: the done-when gates (docs/design/phase-2.md §4), run in .venv-eval and CI
    "phase2_common",
    "phase2_gate",
    "phase2_ci_check",
)
PROBE = """
import importlib, importlib.abc, json, pkgutil, sys

HEAVY = set(json.loads(sys.argv[1]))
attempts = []

class Recorder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            attempts.append(name)
            raise ImportError(f"{name} is not allowed here")
        return None

sys.meta_path.insert(0, Recorder())
import topple.policy
names = json.loads(sys.argv[2]) + [
    f"topple.policy.{m.name}" for m in pkgutil.iter_modules(topple.policy.__path__)
]
for name in names:
    importlib.import_module(name)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
print(json.dumps({"names": names, "attempts": attempts, "loaded": loaded}))
"""


def test_torch_free_modules_import_nothing_heavy() -> None:
    done = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(HEAVY), json.dumps(MODULES)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "scripts")},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["attempts"] == []
    assert report["loaded"] == []
    names = set(report["names"])
    assert set(MODULES) <= names
    for policy_module in ("build", "manifest", "pack", "runtime", "validate"):
        assert f"topple.policy.{policy_module}" in names


def test_the_probe_catches_a_guarded_import() -> None:
    """Positive control: the hook sees an import that its caller swallows."""
    probe = PROBE.replace(
        "for name in names:",
        "try:\n    import psutil\nexcept ImportError:\n    pass\nfor name in names:",
    )
    done = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(HEAVY), json.dumps(["topple.stats"])],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1])["attempts"] == ["psutil"]
