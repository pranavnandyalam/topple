"""No unsafe deserialization anywhere (docs/design/phase-2.md §6 ``test_no_unsafe_loads``).

An AST scan of ``python/topple``, ``scripts/`` and ``tests/``. Outside ``topple/train/`` no file
imports a pickle-family module, torch or Stable-Baselines3, except the fixed (file -> modules)
allow-list below. In the torch-free tree the only ONNX entry point is
``onnx.ModelProto.FromString``: no ``onnx.load*``, no external-data loading, no
``numpy_helper`` (its ``to_array`` reads external data), no custom-op libraries, and ONNX
Runtime sessions and their config entries only in ``runtime.make_session``; no path-based
readers either (``infer_shapes_path``, ``onnx.hub``, ``check_model`` given a path, §3.4). SB3
and VecNormalize loads (qualified or not) only in ``learn.resume_from``. Everywhere: no unsafe
YAML loads, no ``allow_pickle=True``, no ``torch.load`` without ``weights_only=True``.

The scan is name-based: ``getattr`` evasion is out of scope (§6). Names imported under another
name (``import onnx as ox``, ``from yaml import load as y``) are read as what they import.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("python/topple", "scripts", "tests")
TRAIN_DIR = "python/topple/train/"
# The torch-free half of topple/train/ is scanned like the rest of the tree (§2.1: configs,
# wrappers and selection never import torch; the CLI and ``topple export`` import them).
TORCH_FREE_TRAIN = frozenset(
    f"{TRAIN_DIR}{name}.py" for name in ("__init__", "config", "wrappers", "selection")
)
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
    # a train-marked test that imports torch / SB3 itself (test_train_smoke,
    # test_snapshot_reference, test_resources and test_surgery reach them only through
    # topple.train)
    "tests/test_pack_torch_legacy.py": frozenset({"torch", "stable_baselines3"}),
    # TaskSpec pickles and copies (contract §8)
    "tests/test_contract.py": frozenset({"pickle"}),
    # LoadedPolicy, ONNXPolicy and sessions refuse to be pickled (slice 1b)
    "tests/test_onnx_policy.py": frozenset({"pickle"}),
    # a PolicyRef pickles small: bytes, a sha and an id (slice 1b)
    "tests/test_pool.py": frozenset({"pickle"}),
    # §3.3: generated once with the train extra
    "scripts/make_thirdparty_fixtures.py": frozenset({"torch"}),
    # §2.1: make_train_env imports SB3's Monitor inside the function, never at module import
    "python/topple/train/wrappers.py": frozenset({"stable_baselines3"}),
    # its make_train_env case with SB3's real Monitor is train-marked (skipped without SB3)
    "tests/test_train_wrappers.py": frozenset({"stable_baselines3"}),
    # lr_fn's schedule pickles (SB3 saves it inside model.zip, resume only)
    "tests/test_train_config.py": frozenset({"pickle"}),
    # a selection task's EpisodeRows pickles small, never with its trajectory
    "tests/test_selection.py": frozenset({"pickle"}),
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
        "infer_shapes_path",
        "extract_model",  # onnx.utils.extract_model reads and writes model paths only
    }
)
REFERENCE_EVALUATOR = "ReferenceEvaluator"  # onnx.reference: a path argument means onnx.load
ONNX_HUB = "onnx.hub"
CHECK_MODEL = "onnx.checker.check_model"
# what a ``check_model`` argument that is a path looks like (§3.4 bans ``check_model(path)``)
PATH_CALLS = frozenset({"Path", "PurePath", "PosixPath", "str", "fspath", "join", "joinpath"})
PATH_WORDS = ("path", "file")
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


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """(local name -> what it imports) for every ``import x as y`` and ``from m import n [as
    y]`` in ``tree`` (absolute imports; file-wide, as the scan is name-based)."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def looks_like_a_path(node: ast.expr | None, canonical: Callable[[str], str]) -> bool:
    """Whether a ``check_model`` argument is (by its form or name) a path, not a proto."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.Div)  # Path / "m.onnx"
    if isinstance(node, ast.Call):
        return canonical(dotted(node.func) or "").rsplit(".", 1)[-1] in PATH_CALLS
    name = dotted(node) if node is not None else None
    last = (name or "").rsplit(".", 1)[-1].lower()
    return any(word in last for word in PATH_WORDS)


class Scanner(ast.NodeVisitor):
    """Collects the violations of one file (``rel``, repository-relative POSIX path)."""

    def __init__(self, rel: str, aliases: dict[str, str] | None = None) -> None:
        self.rel = rel
        self.train = rel.startswith(TRAIN_DIR) and rel not in TORCH_FREE_TRAIN
        self.allowed = ALLOWED.get(rel, frozenset())
        self.torch_free = not self.train and "torch" not in self.allowed
        self.aliases = aliases or {}
        self.functions: list[str] = []
        self.found: list[Violation] = []

    def canonical(self, name: str) -> str:
        """``name`` with its first part replaced by what that part imports, if it is an alias
        (``ox.load`` after ``import onnx as ox`` is ``onnx.load``)."""
        first, dot, rest = name.partition(".")
        target = self.aliases.get(first)
        return f"{target}{dot}{rest}" if target else name

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
            if name == ONNX_HUB or name.startswith(f"{ONNX_HUB}."):
                self.flag(node, ONNX_HUB)

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
        name = self.canonical(dotted(node.func) or "")
        last = name.rsplit(".", 1)[-1]
        if name in ("__import__", "importlib.import_module") and node.args:
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                self.module(node, target.value)
        if self.torch_free and name.startswith("onnx.load"):
            self.flag(node, "onnx.load")
        if self.torch_free and name.startswith(f"{ONNX_HUB}."):
            self.flag(node, ONNX_HUB)
        if self.torch_free and last == "infer_shapes_path":
            self.flag(node, "infer_shapes_path")
        if self.torch_free and name == CHECK_MODEL:
            model = node.args[0] if node.args else keyword(node, "model")
            if looks_like_a_path(model, self.canonical):
                self.flag(node, "check_model(path)")
        if self.torch_free and last == REFERENCE_EVALUATOR:
            model = node.args[0] if node.args else keyword(node, "proto")
            if looks_like_a_path(model, self.canonical):
                self.flag(node, "ReferenceEvaluator(path)")
        if last == "InferenceSession" and not self.inside(SESSION_SITE):
            self.flag(node, "InferenceSession")
        if last == "add_session_config_entry":
            first = node.args[0] if node.args else None
            literal = isinstance(first, ast.Constant) and first.value in SESSION_KEYS
            if not (self.inside(SESSION_SITE) and literal):
                self.flag(node, "add_session_config_entry")
        sb3 = next((s for s in SB3_LOADS if name == s or name.endswith(f".{s}")), None)
        if last == "load_from_zip_file":
            sb3 = last
        if sb3 is not None and not self.inside(SB3_SITE):
            self.flag(node, sb3)
        if name.startswith("yaml.") and last in UNSAFE_YAML:
            self.flag(node, name)
        if is_true(keyword(node, "allow_pickle")):
            self.flag(node, "allow_pickle=True")
        if name == "torch.load" and not is_true(keyword(node, "weights_only")):
            self.flag(node, "torch.load without weights_only=True")
        self.generic_visit(node)


def scan_source(rel: str, source: str) -> list[Violation]:
    tree = ast.parse(source, filename=rel)
    scanner = Scanner(rel, import_aliases(tree))
    scanner.visit(tree)
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
        tree = ast.parse(path.read_text("utf-8"))
        scanner = Scanner(rel, import_aliases(tree))
        scanner.allowed = frozenset()
        scanner.visit(tree)
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
    # qualified SB3 loads (spec-SPEC-4)
    "stable_baselines3.PPO.load(p)": "PPO.load",
    "sb3.SAC.load(p)": "SAC.load",
    "vec_env.VecNormalize.load(p, env)": "VecNormalize.load",
    "from stable_baselines3 import PPO as Algo\nAlgo.load(p)": "PPO.load",
    "save_util.load_from_zip_file(p)": "load_from_zip_file",
    # aliased modules and path-based ONNX readers (quality-F9)
    "import onnx as ox\nox.load('m.onnx')": "onnx.load",
    "from onnx import load_model_from_string as lm\nlm(b'')": "onnx.load",
    "import yaml as y\ny.load(text)": "yaml.load",
    "from yaml import unsafe_load as ul\nul(text)": "yaml.unsafe_load",
    "onnx.shape_inference.infer_shapes_path('m.onnx')": "infer_shapes_path",
    "from onnx.shape_inference import infer_shapes_path": "infer_shapes_path",
    "import onnx as ox\nox.shape_inference.infer_shapes_path(p, q)": "infer_shapes_path",
    "from onnx import hub\nhub.load('resnet50')": "onnx.hub",
    "import onnx.hub": "onnx.hub",
    "onnx.hub.load('resnet50')": "onnx.hub",
    "onnx.checker.check_model('m.onnx')": "check_model(path)",
    "onnx.checker.check_model(f'{d}/m.onnx')": "check_model(path)",
    "onnx.checker.check_model(Path(p))": "check_model(path)",
    "onnx.checker.check_model(model_path, full_check=True)": "check_model(path)",
    "onnx.checker.check_model(d / 'm.onnx')": "check_model(path)",
    "onnx.checker.check_model(model=str(p))": "check_model(path)",
    "from onnx.checker import check_model as cm\ncm('m.onnx')": "check_model(path)",
    "import onnx.numpy_helper as nh\nnh.to_array(t)": "numpy_helper",
    # path-based ONNX readers (slice 1c re-review)
    "onnx.utils.extract_model('m.onnx', 'out.onnx', ['obs'], ['action'])": "extract_model",
    "from onnx.utils import extract_model": "extract_model",
    "import onnx.utils as u\nu.extract_model(src, dst, i, o)": "extract_model",
    "onnx.reference.ReferenceEvaluator('m.onnx')": "ReferenceEvaluator(path)",
    "from onnx.reference import ReferenceEvaluator\nReferenceEvaluator(model_path)": (
        "ReferenceEvaluator(path)"
    ),
    "ReferenceEvaluator(proto=Path(p))": "ReferenceEvaluator(path)",
    "ReferenceEvaluator(f'{d}/m.onnx', verbose=1)": "ReferenceEvaluator(path)",
    # the torch-free half of topple/train/ is not exempt (§2.1)
}
TORCH_FREE_TRAIN_OFFENDERS = {
    "import torch": "torch",
    "from stable_baselines3 import PPO": "stable_baselines3",
    "import pickle": "pickle",
    "onnx.load('m.onnx')": "onnx.load",
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
        "onnx.checker.check_model(proto, full_check=True)",
        "import onnx as ox\nox.checker.check_model(model)",
        "onnx.shape_inference.infer_shapes(proto, strict_mode=True)",
        "from yaml import safe_load as load\nload(text)",
        "import numpy as np\nnp.load(path)",
        "model.load(path)",  # no SB3 class named
        "PPO_load(p)",
        "from topple.traj import load_npz\nload_npz(p, max_steps=1000)",
        "from onnx.reference import ReferenceEvaluator\nReferenceEvaluator(proto)",
        "onnx.reference.ReferenceEvaluator(model, verbose=0)",
        "ReferenceEvaluator(proto=model)",
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
    assert (
        scan_source("tests/test_pack_torch_legacy.py", "import torch\nimport stable_baselines3")
        == []
    )
    assert scan_source("tests/test_policy_canonical.py", "onnx.numpy_helper.to_array(t)") == []
    assert scan_source("tests/test_policy_canonical.py", "onnx.load('m')") != []


@pytest.mark.parametrize("name", sorted(TORCH_FREE_TRAIN))
@pytest.mark.parametrize(
    ("source", "what"),
    list(TORCH_FREE_TRAIN_OFFENDERS.items()),
    ids=list(TORCH_FREE_TRAIN_OFFENDERS),
)
def test_the_torch_free_train_modules_are_scanned(name: str, source: str, what: str) -> None:
    allowed = ALLOWED.get(name, frozenset())
    found = {v.what for v in scan_source(name, source)}
    assert what in found or what in allowed


def test_the_train_extra_modules_stay_exempt() -> None:
    assert scan_source("python/topple/train/callbacks.py", "import torch") == []
    assert scan_source("python/topple/train/wrappers.py", "import torch") != []
    lazy = "def make_train_env(cfg, i):\n    from stable_baselines3.common.monitor import Monitor\n"
    assert scan_source("python/topple/train/wrappers.py", lazy) == []
    assert scan_source("python/topple/train/config.py", lazy) != []
