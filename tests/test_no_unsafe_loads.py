"""No unsafe deserialization anywhere (docs/design/phase-2.md §6 ``test_no_unsafe_loads``).

An AST scan of ``python/topple``, ``scripts/`` and ``tests/``. Outside ``topple/train/`` no file
imports a pickle-family module, torch or Stable-Baselines3, except the fixed (file -> modules)
allow-list below. In the torch-free tree the only ONNX entry point is
``onnx.ModelProto.FromString``: no ``onnx.load*``, no external-data loading, no
``numpy_helper`` (its ``to_array`` reads external data), no custom-op libraries, and ONNX
Runtime sessions and their config entries only in ``runtime.make_session``. SB3 and
VecNormalize loads only in ``learn.resume_from``. Everywhere: no unsafe YAML loads, no
``allow_pickle=True``, no ``torch.load`` without ``weights_only=True``.

The scan is name-based: ``getattr`` evasion is out of scope (§6).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("python/topple", "scripts", "tests")
TRAIN_DIR = "python/topple/train/"
BANNED_MODULES = frozenset(
    {"pickle", "cloudpickle", "dill", "joblib", "shelve", "marshal", "torch", "stable_baselines3"}
)
TRAIN_TESTS = (
    "tests/test_train_smoke.py",
    "tests/test_snapshot_reference.py",
    "tests/test_pack_torch_legacy.py",
    "tests/test_resources.py",
    "tests/test_surgery.py",
)
# (file -> banned names it may use). Fixed: an entry is a reviewed exception, never a pattern.
ALLOWED: dict[str, frozenset[str]] = {
    **dict.fromkeys(TRAIN_TESTS, frozenset({"torch", "stable_baselines3"})),
    # TaskSpec pickles and copies (contract §8)
    "tests/test_contract.py": frozenset({"pickle"}),
    # LoadedPolicy, ONNXPolicy and sessions refuse to be pickled (slice 1b)
    "tests/test_onnx_policy.py": frozenset({"pickle"}),
    # a PolicyRef pickles small: bytes, a sha and an id (slice 1b)
    "tests/test_pool.py": frozenset({"pickle"}),
    # §3.3: generated once with the train extra
    "scripts/make_thirdparty_fixtures.py": frozenset({"torch"}),
    # these monkeypatch the readers to raise, proving validation and pack never call them
    "tests/test_policy_validate.py": frozenset({"numpy_helper"}),
    "tests/test_policy_canonical.py": frozenset({"numpy_helper"}),
    "tests/test_policy_pack.py": frozenset({"numpy_helper", "external_data_helper"}),
}
ONNX_REFERENCES = frozenset(
    {
        "numpy_helper",
        "external_data_helper",
        "load_external_data_for_model",
        "register_custom_ops_library",
    }
)
SESSION_SITE = ("python/topple/policy/runtime.py", "make_session")
SESSION_KEYS = frozenset(
    {
        "session.load_model_format",
        "session.allow_released_opsets_only",
        "session.strict_shape_type_inference",
    }
)
SB3_LOADS = frozenset({"PPO.load", "SAC.load", "VecNormalize.load"})
SB3_SITE = ("python/topple/train/learn.py", "resume_from")
UNSAFE_YAML = frozenset(
    {"load", "full_load", "unsafe_load", "load_all", "full_load_all", "unsafe_load_all"}
)


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    what: str


def dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a chain of names and attributes, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


class Scanner(ast.NodeVisitor):
    """Collects the violations of one file (``rel``, repository-relative POSIX path)."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.train = rel.startswith(TRAIN_DIR)
        self.allowed = ALLOWED.get(rel, frozenset())
        self.torch_free = not self.train and "torch" not in self.allowed
        self.functions: list[str] = []
        self.found: list[Violation] = []

    def flag(self, node: ast.AST, what: str) -> None:
        if what not in self.allowed:
            self.found.append(Violation(self.rel, getattr(node, "lineno", 0), what))

    # imports --------------------------------------------------------------------------------

    def module(self, node: ast.AST, name: str) -> None:
        top = name.split(".")[0]
        if top in BANNED_MODULES and not self.train:
            self.flag(node, top)
        if self.torch_free:
            for part in name.split("."):
                if part in ONNX_REFERENCES:
                    self.flag(node, part)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.module(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self.module(node, node.module)
            for alias in node.names:
                self.module(node, f"{node.module}.{alias.name}")
                if node.module == "onnx" and alias.name.startswith("load") and self.torch_free:
                    self.flag(node, "onnx.load")
                if node.module == "yaml" and alias.name in UNSAFE_YAML:
                    self.flag(node, f"yaml.{alias.name}")
        self.generic_visit(node)

    # names ----------------------------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.torch_free and node.attr in ONNX_REFERENCES:
            self.flag(node, node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self.torch_free and node.id in ONNX_REFERENCES:
            self.flag(node, node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 - the visitor's naming

    def inside(self, site: tuple[str, str]) -> bool:
        return self.rel == site[0] and site[1] in self.functions

    # calls ----------------------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted(node.func) or ""
        last = name.rsplit(".", 1)[-1]
        if name in ("__import__", "importlib.import_module") and node.args:
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                self.module(node, target.value)
        if self.torch_free and name.startswith("onnx.load"):
            self.flag(node, "onnx.load")
        if last == "InferenceSession" and not self.inside(SESSION_SITE):
            self.flag(node, "InferenceSession")
        if last == "add_session_config_entry":
            first = node.args[0] if node.args else None
            literal = isinstance(first, ast.Constant) and first.value in SESSION_KEYS
            if not (self.inside(SESSION_SITE) and literal):
                self.flag(node, "add_session_config_entry")
        if (name in SB3_LOADS or last == "load_from_zip_file") and not self.inside(SB3_SITE):
            self.flag(node, name)
        if name.startswith("yaml.") and last in UNSAFE_YAML:
            self.flag(node, name)
        if is_true(keyword(node, "allow_pickle")):
            self.flag(node, "allow_pickle=True")
        if name == "torch.load" and not is_true(keyword(node, "weights_only")):
            self.flag(node, "torch.load without weights_only=True")
        self.generic_visit(node)


def scan_source(rel: str, source: str) -> list[Violation]:
    scanner = Scanner(rel)
    scanner.visit(ast.parse(source, filename=rel))
    return scanner.found


def python_files() -> Iterator[Path]:
    for top in SCANNED:
        root = REPO / top
        if root.is_dir():
            yield from sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_tree_has_no_unsafe_loads() -> None:
    files = list(python_files())
    assert len(files) > 50  # the scan really ran over the tree
    found = [
        v
        for path in files
        for v in scan_source(path.relative_to(REPO).as_posix(), path.read_text("utf-8"))
    ]
    assert found == []


def test_the_allow_list_names_real_reasons() -> None:
    """Every allowed name is one the scan bans; entries for files that exist are still used
    (a stale exception is removed, never kept "just in case")."""
    banned = BANNED_MODULES | ONNX_REFERENCES
    for rel, names in ALLOWED.items():
        assert names <= banned, rel
        path = REPO / rel
        if not path.exists():
            continue  # a deferred or train-only file (§6)
        scanner = Scanner(rel)
        scanner.allowed = frozenset()
        scanner.visit(ast.parse(path.read_text("utf-8")))
        used = {v.what for v in scanner.found}
        assert names <= used, f"{rel}: unused allow-list entries {sorted(names - used)}"


# --- positive controls: each rule fires on a minimal offender ----------------------------------

OFFENDERS = {
    "import pickle": "pickle",
    "import cloudpickle as cp": "cloudpickle",
    "from dill import loads": "dill",
    "import joblib.numpy_pickle": "joblib",
    "import shelve": "shelve",
    "from marshal import loads": "marshal",
    "import torch": "torch",
    "from stable_baselines3 import PPO": "stable_baselines3",
    "__import__('pickle')": "pickle",
    "importlib.import_module('torch.nn')": "torch",
    "onnx.load('m.onnx')": "onnx.load",
    "onnx.load_model_from_string(b'')": "onnx.load",
    "from onnx import load_model": "onnx.load",
    "onnx.load_external_data_for_model(m, '.')": "load_external_data_for_model",
    "from onnx.external_data_helper import load_external_data_for_tensor": "external_data_helper",
    "import onnx.numpy_helper": "numpy_helper",
    "onnx.numpy_helper.to_array(t)": "numpy_helper",
    "from onnx import numpy_helper": "numpy_helper",
    "ort.register_custom_ops_library('x.so')": "register_custom_ops_library",
    "ort.InferenceSession(b'')": "InferenceSession",
    "InferenceSession('m.onnx')": "InferenceSession",
    "o.add_session_config_entry('session.load_model_format', 'ONNX')": "add_session_config_entry",
    "PPO.load('model.zip')": "PPO.load",
    "VecNormalize.load('v.pkl', env)": "VecNormalize.load",
    "load_from_zip_file('model.zip')": "load_from_zip_file",
    "yaml.load(text)": "yaml.load",
    "yaml.unsafe_load(text)": "yaml.unsafe_load",
    "yaml.full_load(text)": "yaml.full_load",
    "from yaml import load": "yaml.load",
    "np.load(path, allow_pickle=True)": "allow_pickle=True",
    "torch.load(path)": "torch.load without weights_only=True",
    "torch.load(path, weights_only=False)": "torch.load without weights_only=True",
}


@pytest.mark.parametrize(("source", "what"), list(OFFENDERS.items()), ids=list(OFFENDERS.values()))
def test_each_rule_fires(source: str, what: str) -> None:
    assert what in {v.what for v in scan_source("python/topple/x.py", source)}


@pytest.mark.parametrize(
    "source",
    [
        "onnx.ModelProto.FromString(data)",
        "yaml.safe_load(text)",
        "np.load(path, allow_pickle=False)",
        "monkeypatch.setattr(ort, 'InferenceSession', fake)",
        "import onnxruntime as ort",
        "from onnx import helper, TensorProto",
    ],
)
def test_safe_code_passes(source: str) -> None:
    assert scan_source("python/topple/x.py", source) == []


def test_sessions_only_in_make_session() -> None:
    site = (
        "def make_session(v):\n"
        "    o.add_session_config_entry('session.load_model_format', 'ONNX')\n"
        "    return ort.InferenceSession(v.canonical)\n"
    )
    assert scan_source("python/topple/policy/runtime.py", site) == []
    assert scan_source("python/topple/policy/other.py", site) != []
    elsewhere = site.replace("make_session", "make_other")
    assert scan_source("python/topple/policy/runtime.py", elsewhere) != []
    other_key = site.replace("session.load_model_format", "session.use_env_allocators")
    found = scan_source("python/topple/policy/runtime.py", other_key)
    assert [v.what for v in found] == ["add_session_config_entry"]


def test_train_code_may_import_torch_but_not_load_unsafely() -> None:
    assert scan_source("python/topple/train/learn.py", "import torch\nimport pickle") == []
    assert scan_source("python/topple/train/learn.py", "torch.load(p, weights_only=True)") == []
    assert scan_source("python/topple/train/x.py", "torch.load(p)") != []
    resume = "def resume_from(run):\n    return PPO.load(run)\n"
    assert scan_source("python/topple/train/learn.py", resume) == []
    assert scan_source("python/topple/train/x.py", resume) != []


def test_allow_listed_files_are_exempt_only_for_their_names() -> None:
    assert scan_source("tests/test_contract.py", "import pickle") == []
    assert scan_source("tests/test_contract.py", "import dill") != []
    assert scan_source("tests/test_train_smoke.py", "import torch\nimport stable_baselines3") == []
    assert scan_source("tests/test_policy_canonical.py", "onnx.numpy_helper.to_array(t)") == []
    assert scan_source("tests/test_policy_canonical.py", "onnx.load('m')") != []
