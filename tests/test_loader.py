import hashlib
import os
import sys
import threading
from pathlib import Path

import pytest

import topple.loader as loader_module
from topple.contract import FEATURES, Controller, TaskSpec
from topple.envs.balance import BalanceEnv
from topple.evaluate import run_episode
from topple.loader import LoadError, load_submission, submission_id, submission_sha256

SPEC: TaskSpec = BalanceEnv("ideal").task_spec

GOOD = """
class C:
    def reset(self, rng_seed):
        self.seed = rng_seed
    def act(self, obs):
        return (0.0, 0.0)

def make_controller(spec):
    assert spec.challenge == "balance-v1"
    return C()
"""

ZERO_ARG = """
class C:
    def reset(self, rng_seed): pass
    def act(self, obs): return [0, 0]

def make_controller():
    return C()
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_load_file(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "mine.py", GOOD))
    assert sub.id == "mine"
    assert len(sub.sha256) == 64
    ctrl = sub.factory(SPEC)
    assert isinstance(ctrl, Controller)


def test_zero_arg_factory(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "lesson1.py", ZERO_ARG))
    assert isinstance(sub.factory(SPEC), Controller)


def test_optional_spec_factory(tmp_path: Path) -> None:
    text = ZERO_ARG.replace(
        "def make_controller():", "def make_controller(spec=None):\n    assert spec is not None"
    )
    sub = load_submission(write(tmp_path, "opt.py", text))
    assert isinstance(sub.factory(SPEC), Controller)


def test_directory_submission_with_assets(tmp_path: Path) -> None:
    root = tmp_path / "team_bot"
    write(root, "gains.txt", "0.5")
    write(
        root,
        "controller.py",
        GOOD.replace(
            "def make_controller(spec):",
            "from pathlib import Path\n"
            "GAIN = float((Path(__file__).parent / 'gains.txt').read_text())\n"
            "def make_controller(spec):",
        ),
    )
    sub = load_submission(root)
    assert sub.id == "team_bot"
    assert sub.entry.name == "controller.py"
    assert isinstance(sub.factory(SPEC), Controller)


def test_sha_changes_with_any_file(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "data.txt", "a")
    before = submission_sha256(root)
    write(root, "data.txt", "b")
    assert submission_sha256(root) != before


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("nofactory.py", "x = 1\n", "must define make_controller"),
        ("syntax.py", "def make_controller(:\n", "importing syntax.py failed"),
        ("raises.py", "raise RuntimeError('at import')\n", "RuntimeError: at import"),
        (
            "twoargs.py",
            "def make_controller(a, b):\n    pass\n",
            "must take (spec) or no arguments",
        ),
        ("notpy.txt", "hello", "expected a .py file"),
    ],
)
def test_load_errors(tmp_path: Path, name: str, text: str, message: str) -> None:
    with pytest.raises(LoadError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_submission(write(tmp_path, name, text))


HOSTILE_IMPORT_PRELUDE = """
import sys


class Nameless(type):
    @property
    def __name__(cls):
        raise RuntimeError("no name for you")


class Liar(type):
    @property
    def __name__(cls):
        sys.modules["topple_test_liar_ran"] = sys
        return "Spoofed"


class Bomb(str):
    def __format__(self, spec):
        raise RuntimeError("cannot format myself")

    def __add__(self, other):
        raise RuntimeError("cannot add myself")

    def __radd__(self, other):
        raise RuntimeError("cannot add myself")
"""


def load_error_message(path: Path) -> str:
    """The ``LoadError`` message ``load_submission(path)`` raises, as an exact ``str``.

    Anything else that escapes fails the test outside the ``except`` block, so pytest never has
    to render a hostile exception itself.
    """
    try:
        load_submission(path)
    except LoadError as exc:
        return str.__str__(exc.args[0])
    except BaseException:
        outcome = "raised something other than LoadError"
    else:
        outcome = "accepted it"
    pytest.fail(f"load_submission {outcome}")


HOSTILE_IMPORTS = {
    "exception whose class name raises": (
        "class Boom(Exception, metaclass=Nameless): pass\nraise Boom('why')\n",
        "Boom: why",
    ),
    "exception whose metaclass lies about its name": (
        "class Real(Exception, metaclass=Liar): pass\nraise Real('why')\n",
        "Real: why",
    ),
    "exception whose __str__ raises": (
        "class E(Exception):\n    def __str__(self):\n        raise ValueError\nraise E()\n",
        "E: <str() failed>",
    ),
    "exception whose message is a hostile str": (
        "class E(Exception):\n    def __str__(self):\n        return Bomb('why')\nraise E()\n",
        "E: why",
    ),
    "exception named by a hostile str": (
        "class E(Exception): pass\nE.__name__ = Bomb('Renamed')\nraise E('why')\n",
        "Renamed: why",
    ),
    "sys.exit at import": ("sys.exit(7)\n", "SystemExit: 7"),
    "GeneratorExit at import": ("raise GeneratorExit('stop')\n", "GeneratorExit: stop"),
}


@pytest.mark.parametrize(
    ("body", "detail"), HOSTILE_IMPORTS.values(), ids=list(HOSTILE_IMPORTS.keys())
)
def test_a_hostile_import_failure_is_a_clean_load_error(
    tmp_path: Path, body: str, detail: str
) -> None:
    """Contract §8 and spec H: whatever the submission raises while it is imported (any
    ``BaseException`` but ``KeyboardInterrupt``), the loader raises ``LoadError`` naming the real
    class, without running metaclass or ``str``-subclass code, so the CLI never shows a
    traceback."""
    sys.modules.pop("topple_test_liar_ran", None)
    path = write(tmp_path, "hostile.py", HOSTILE_IMPORT_PRELUDE + body)
    assert load_error_message(path) == f"importing hostile.py failed: {detail}"
    assert "topple_test_liar_ran" not in sys.modules


def test_keyboard_interrupt_at_import_still_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        load_submission(write(tmp_path, "ctrl_c.py", "raise KeyboardInterrupt\n"))


HOSTILE_FACTORIES = {
    "module whose attribute lookup raises": (
        "import sys, types\n"
        "class M(types.ModuleType):\n"
        "    def __getattribute__(self, name):\n"
        "        raise RuntimeError('no attributes for you')\n"
        "sys.modules[__name__].__class__ = M\n",
        "reading make_controller from hostile.py failed: RuntimeError: no attributes for you",
    ),
    "factory whose signature raises": (
        "class Factory:\n"
        "    @property\n"
        "    def __signature__(self):\n"
        "        raise ZeroDivisionError('no signature for you')\n"
        "    def __call__(self, spec):\n"
        "        return None\n"
        "make_controller = Factory()\n",
        "make_controller in hostile.py has no usable signature: "
        "ZeroDivisionError: no signature for you",
    ),
    "factory whose signature exits": (
        "import sys\n"
        "class Factory:\n"
        "    @property\n"
        "    def __signature__(self):\n"
        "        sys.exit(3)\n"
        "    def __call__(self, spec):\n"
        "        return None\n"
        "make_controller = Factory()\n",
        "make_controller in hostile.py has no usable signature: SystemExit: 3",
    ),
}


@pytest.mark.parametrize(
    ("text", "message"), HOSTILE_FACTORIES.values(), ids=list(HOSTILE_FACTORIES.keys())
)
def test_a_hostile_factory_lookup_is_a_clean_load_error(
    tmp_path: Path, text: str, message: str
) -> None:
    """Finding and inspecting ``make_controller`` runs submission code too; whatever it raises
    becomes ``LoadError``, and the half-loaded module is not left in ``sys.modules``."""
    before = set(sys.modules)
    assert load_error_message(write(tmp_path, "hostile.py", text)) == message
    assert [name for name in sys.modules if name not in before] == []


def test_a_submission_without_a_factory_leaves_no_module_behind(tmp_path: Path) -> None:
    before = set(sys.modules)
    message = load_error_message(write(tmp_path, "nofactory.py", "x = 'only here'\n"))
    assert message == "nofactory.py must define make_controller(spec)"
    assert [name for name in sys.modules if name not in before] == []


def test_factory_returning_non_controller(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "bad.py", "def make_controller(spec):\n    return 42\n"))
    with pytest.raises(LoadError, match="lacks reset"):
        sub.factory(SPEC)


CLASSLESS = """
class C:
    @property
    def __class__(self):
        raise RuntimeError("hostile class")

    def reset(self, rng_seed):
        pass

    def act(self, obs):
        return (0.0, 0.0)


def make_controller(spec):
    return C()
"""


def test_a_loaded_controller_is_never_asked_its_class(tmp_path: Path) -> None:
    """Contract §8: the loader's factory, like the harness, never reads ``__class__``, so a
    controller behaves the same loaded from a file as passed in directly."""
    sub = load_submission(write(tmp_path, "classless.py", CLASSLESS))
    run = run_episode(BalanceEnv("ideal", record=True), sub.factory, seed=0, index=0)
    assert run.result.error is None
    assert run.trajectory.meta["termination"]["reason"] == "fell"


NAMELESS = """
class Meta(type):
    @property
    def __name__(cls):
        raise RuntimeError("no name for you")


class C(metaclass=Meta):
    pass


def make_controller(spec):
    return C()
"""


def test_a_non_controller_is_named_by_its_real_class_name(tmp_path: Path) -> None:
    sub = load_submission(write(tmp_path, "nameless.py", NAMELESS))
    with pytest.raises(LoadError, match=r"returned C, which lacks reset\(\)/act\(\)"):
        sub.factory(SPEC)


DYNAMIC = """
import types


class Dynamic:
    \"\"\"Answers every attribute dynamically: a static lookup finds no reset or act.\"\"\"

    def __getattr__(self, name):
        return lambda *args: (0.0, 0.0)


class Exploding:
    \"\"\"Any dynamic attribute read raises: the static lookup never triggers it.\"\"\"

    def __getattribute__(self, name):
        raise RuntimeError("dynamic lookup ran")
"""


@pytest.mark.parametrize(
    "returned",
    [
        "types.SimpleNamespace(reset=lambda seed: None, act=lambda obs: (0.0, 0.0))",
        "types.SimpleNamespace(reset=lambda seed: None)",
        "Dynamic()",
        "Exploding()",
    ],
    ids=["instance attributes", "act missing", "only __getattr__", "raising __getattribute__"],
)
def test_controller_methods_may_live_on_the_instance(tmp_path: Path, returned: str) -> None:
    """Methods are looked up as ``isinstance(x, Controller)`` would (statically, instance dict
    included), only without reading ``__class__``."""
    text = DYNAMIC + f"\n\ndef make_controller(spec):\n    return {returned}\n"
    sub = load_submission(write(tmp_path, "ns.py", text))
    if "act=" in returned:
        assert isinstance(sub.factory(SPEC), Controller)
    else:
        with pytest.raises(LoadError, match="lacks reset"):
            sub.factory(SPEC)


def test_missing_path(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="does not exist"):
        load_submission(tmp_path / "nope.py")


def test_directory_without_entry(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(LoadError, match=r"controller\.py"):
        load_submission(tmp_path / "empty")


def test_too_many_files(tmp_path: Path) -> None:
    root = tmp_path / "big"
    write(root, "controller.py", GOOD)
    for i in range(60):
        write(root, f"f{i}.txt", "x")
    with pytest.raises(LoadError, match="files"):
        load_submission(root)


def test_two_submissions_do_not_collide(tmp_path: Path) -> None:
    a = load_submission(write(tmp_path / "a", "bot.py", GOOD))
    b = load_submission(write(tmp_path / "b", "bot.py", GOOD.replace("0.0, 0.0", "0.1, 0.1")))
    assert a.sha256 != b.sha256
    assert a.factory(SPEC).act(None) != b.factory(SPEC).act(None)  # type: ignore[arg-type]


def test_spec_features_match_contract() -> None:
    assert SPEC.obs_features == FEATURES


# ------------------------------------------------------------------ hashing and directory hygiene


def reference_sha(root: Path, rels: list[str]) -> str:
    """Length-framed (POSIX relpath, size, bytes) records, written out independently."""
    digest = hashlib.sha256(b"topple.submission/1\n")
    for rel in sorted(rels):
        data = (root / rel).read_bytes()
        name = rel.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def test_sha_follows_the_framing(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "assets/gains.txt", "0.5")
    assert submission_sha256(root) == reference_sha(root, ["assets/gains.txt", "controller.py"])
    single = write(tmp_path, "solo.py", GOOD)
    assert submission_sha256(single) == reference_sha(tmp_path, ["solo.py"])


def test_sha_frames_names_and_contents(tmp_path: Path) -> None:
    """Without framing, ("a", "bc") and ("ab", "c") would hash the same bytes."""
    one, two = tmp_path / "one", tmp_path / "two"
    write(one, "controller.py", GOOD)
    write(two, "controller.py", GOOD)
    write(one, "a", "bc")
    write(two, "ab", "c")
    assert submission_sha256(one) != submission_sha256(two)


def _non_utf8_file(directory: Path) -> Path:
    """A file named by the bytes ``x\\xff.txt`` (not UTF-8), as Linux allows; skipped where the
    file system refuses such names (APFS)."""
    raw = os.fsencode(directory) + b"/x\xff.txt"
    try:
        fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        pytest.skip(f"this file system refuses non-UTF-8 names: {exc}")
    os.close(fd)
    return Path(os.fsdecode(raw))  # the name holds the lone surrogate "\\udcff"


def test_a_file_name_that_is_not_utf8_is_a_load_error(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    _non_utf8_file(root)
    with pytest.raises(LoadError, match="not valid UTF-8"):
        submission_sha256(root)
    with pytest.raises(LoadError, match="not valid UTF-8"):
        load_submission(root)


def test_a_submission_whose_own_name_is_not_utf8_is_refused(tmp_path: Path) -> None:
    """Its name becomes the controller id, which must be text (``results.json`` is UTF-8)."""
    with pytest.raises(LoadError, match="not valid UTF-8"):
        load_submission(tmp_path / "ctl\udcff.py")


def test_a_name_that_is_not_utf8_is_refused_before_the_file_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linux decodes the name ``b"x\\xff.txt"`` to ``"x\\udcff.txt"``, which has no UTF-8 form:
    the hash refuses it with a ``LoadError`` (the CLI's clean exit 2), never ``UnicodeError``."""
    root = tmp_path / "bot"
    entry = write(root, "controller.py", GOOD)
    odd = root / "x\udcff.txt"
    monkeypatch.setattr(loader_module, "_files", lambda path: [entry, odd])
    with pytest.raises(LoadError, match=r"x\\udcff\.txt.* not valid UTF-8"):
        submission_sha256(root)


@pytest.mark.parametrize("junk", ["__pycache__/controller.cpython-312.pyc", "old.pyc", "lib/x.pyc"])
def test_compiled_python_is_refused(tmp_path: Path, junk: str) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, junk, "")
    with pytest.raises(LoadError, match=r"__pycache__|\.pyc"):
        load_submission(root)


def test_empty_pycache_directory_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "__pycache__").mkdir()
    with pytest.raises(LoadError, match="__pycache__"):
        load_submission(root)


def test_loading_writes_no_bytecode_so_it_can_load_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    # Start from Python's default: an earlier load in this worker (or PYTHONDONTWRITEBYTECODE)
    # would otherwise have turned bytecode off already and hidden a regression.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    first = load_submission(root)
    assert sorted(p.name for p in root.iterdir()) == ["controller.py"]
    assert load_submission(root).sha256 == first.sha256


LAZY = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class C:
    def reset(self, rng_seed):
        pass

    def act(self, obs):
        import {name}_act

        return {name}_act.ACTION


def make_controller(spec):
    import {name}_factory

    return C()
"""


def test_lazy_imports_from_the_submission_write_no_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sibling modules imported later (in make_controller or act) leave no __pycache__ behind,
    so the submission still loads and hashes the same after it has run."""
    name = f"lazy_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / "bot"
    write(root, "controller.py", LAZY.replace("{name}", name))
    write(root, f"{name}_factory.py", "X = 1\n")
    write(root, f"{name}_act.py", "ACTION = (0.0, 0.0)\n")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        first = load_submission(root)
        controller = first.factory(SPEC)
        controller.reset(0)
        assert controller.act(None) == (0.0, 0.0)  # type: ignore[arg-type]
    finally:
        for module in (f"{name}_factory", f"{name}_act"):
            sys.modules.pop(module, None)
    assert sorted(p.name for p in root.iterdir()) == sorted(
        ["controller.py", f"{name}_factory.py", f"{name}_act.py"]
    )
    assert load_submission(root).sha256 == first.sha256


def test_symlinked_files_are_refused(tmp_path: Path) -> None:
    secret = write(tmp_path, "secret.txt", "token")
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "data.txt").symlink_to(secret)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(root)


def test_symlinked_directories_are_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    write(elsewhere, "big.bin", "x")
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    (root / "lib").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(root)


def test_a_symlinked_submission_path_is_refused(tmp_path: Path) -> None:
    real = write(tmp_path, "real.py", GOOD)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    with pytest.raises(LoadError, match="symlink"):
        load_submission(link)


def test_special_files_are_refused_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    os.mkfifo(root / "pipe")
    outcome: list[BaseException | None] = []

    def load() -> None:
        try:
            load_submission(root)
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)

    # Opening a FIFO blocks until a writer appears: run the load in a daemon thread so that a
    # regression fails this test instead of hanging the worker forever.
    worker = threading.Thread(target=load, daemon=True)
    worker.start()
    worker.join(timeout=10)
    if worker.is_alive():
        with open(root / "pipe", "wb"):  # unblock the reader so the thread can finish
            pass
        pytest.fail("load_submission blocked on a FIFO")
    assert len(outcome) == 1
    assert isinstance(outcome[0], LoadError)
    assert "regular file" in str(outcome[0])


def test_size_cap_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    monkeypatch.setattr(loader_module, "MAX_BYTES", len(GOOD) + 10)
    write(root, "data.bin", "x" * 10)
    assert len(load_submission(root).sha256) == 64
    write(root, "data.bin", "x" * 11)
    with pytest.raises(LoadError, match="larger than"):
        load_submission(root)


def test_size_cap_stops_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap bounds what is read, not just what is counted afterwards."""
    root = tmp_path / "bot"
    write(root, "controller.py", GOOD)
    write(root, "huge.bin", "x" * 100_000)
    monkeypatch.setattr(loader_module, "MAX_BYTES", len(GOOD) + 100)
    sizes: list[int] = []
    real_open = os.fdopen

    class Spy:
        def __init__(self, fh: object) -> None:
            self.fh = fh

        def __enter__(self) -> "Spy":
            return self

        def __exit__(self, *exc: object) -> None:
            self.fh.close()  # type: ignore[attr-defined]

        def read(self, n: int = -1) -> bytes:
            data: bytes = self.fh.read(n)  # type: ignore[attr-defined]
            sizes.append(len(data))
            return data

    monkeypatch.setattr(loader_module.os, "fdopen", lambda fd, mode: Spy(real_open(fd, mode)))
    with pytest.raises(LoadError, match="larger than"):
        submission_sha256(root)
    assert sizes == [len(GOOD), 101]  # the 100 kB file: only one byte past what was left


def test_hashing_a_symlinked_file_is_refused(tmp_path: Path) -> None:
    real = write(tmp_path, "real.py", GOOD)
    link = tmp_path / "link.py"
    link.symlink_to(real)
    with pytest.raises(LoadError, match="symlink"):
        submission_sha256(link)


def test_submission_id_names_the_resolved_file_or_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id is what the path resolves to, however it is spelled (``.``, ``bot/..``)."""
    bot = tmp_path / "mybot"
    (bot / "inner").mkdir(parents=True)
    (bot / "solo.py").write_text("")
    monkeypatch.chdir(bot)
    assert submission_id(Path(".")) == "mybot"
    assert submission_id(Path("inner/..")) == "mybot"
    assert submission_id(bot) == "mybot"
    assert submission_id(Path("solo.py")) == "solo"
    assert submission_id(Path("inner/../solo.py")) == "solo"
