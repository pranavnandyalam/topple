"""Training configs (docs/design/phase-2.md §2.1, §2.2, §6 ``test_train_config``).

The shipped YAMLs validate; every key the spec forbids is refused with a ``ConfigError`` that
names its field; the YAML reader refuses anchors, aliases, deep nesting and oversize files
before anything is built; ``lr_fn`` clamps; the derived files differ from their base in exactly
the listed key paths; ``config_sha256`` hashes the effective config.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from topple.train.config import (
    MAX_CONFIG_BYTES,
    MAX_SELECTION_EVALS,
    MAX_TRAIN_SEED,
    LrSchedule,
    PpoTrainConfig,
    SacTrainConfig,
    config_sha256,
    load_train_config,
    lr_fn,
    parse_train_config,
    read_train_config,
    validate_train_config,
)
from topple.yamlsafe import ConfigError, safe_load_capped

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "train"
SHIPPED = ("ppo_balance", "sac_balance", "smoke_ppo", "smoke_sac", "ablate_nopenalty")


def raw(name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text("utf-8"))
    assert isinstance(data, dict)
    return data


def as_yaml(data: object) -> bytes:
    return yaml.safe_dump(data, sort_keys=False).encode("utf-8")


def refused(data: object, *fields: str) -> str:
    """The ``ConfigError`` text for config ``data``, which must name every one of ``fields``."""
    with pytest.raises(ConfigError) as info:
        parse_train_config(as_yaml(data))
    message = str(info.value)
    for name in fields:
        assert name in message, message
    return message


def edited(name: str, edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data = copy.deepcopy(raw(name))
    edit(data)
    return data


# --- the shipped files --------------------------------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_configs_validate(name: str) -> None:
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    assert cfg.algo == ("sac" if "sac" in name else "ppo")
    assert cfg.n_envs <= 8  # §1: shipped configs use at most 8 envs


def test_the_ppo_config_is_the_measured_layout() -> None:
    cfg = load_train_config(CONFIGS / "ppo_balance.yaml")
    assert isinstance(cfg, PpoTrainConfig)
    assert (cfg.seed, cfg.total_timesteps, cfg.n_envs) == (0, 5_000_000, 8)
    assert (cfg.vec_env, cfg.start_method) == ("subproc", "forkserver")
    assert cfg.env.action_excess.enabled is True
    assert cfg.policy.net_arch.pi == (64, 64)
    assert cfg.policy.net_arch.vf == (64, 64)
    assert cfg.ppo.learning_rate.schedule == "linear"
    assert cfg.ppo.target_kl is None
    assert cfg.selection.seeds.seeds == tuple(range(2000, 2200))
    assert (cfg.selection.gate, cfg.selection.max_saturation) == ("pid", 0.08)
    assert cfg.health is not None
    assert cfg.health.min_std == 0.03
    assert cfg.env.reward.reward_config().model_dump() == {
        "tilt": 0.30,
        "rate": 0.10,
        "drift": 0.25,
        "yaw_rate": 0.05,
        "effort": 0.10,
        "smooth": 0.20,
    }


def test_the_sac_config_has_no_penalty_and_no_health() -> None:
    cfg = load_train_config(CONFIGS / "sac_balance.yaml")
    assert isinstance(cfg, SacTrainConfig)
    assert cfg.env.action_excess.enabled is False
    assert cfg.policy.net_arch.qf == (256, 256)
    assert cfg.sac.ent_coef == "auto"
    assert not hasattr(cfg, "health")


# --- rejected ----------------------------------------------------------------------------------


def test_extra_keys_are_refused() -> None:
    refused({**raw("ppo_balance"), "checkpoint_every": 100_000}, "checkpoint_every")
    refused(edited("ppo_balance", lambda d: d["env"].update(curriculum=True)), "curriculum")
    refused(edited("sac_balance", lambda d: d["sac"].update(use_sde=True)), "use_sde")


@pytest.mark.parametrize("n_envs", [0, 17, -1])
def test_n_envs_outside_1_to_16(n_envs: int) -> None:
    refused({**raw("ppo_balance"), "n_envs": n_envs}, "n_envs")


@pytest.mark.parametrize("n_envs", [1, 16])
def test_n_envs_bounds_are_inclusive(n_envs: int) -> None:
    data = {**raw("ppo_balance"), "n_envs": n_envs}
    assert parse_train_config(as_yaml(data)).n_envs == n_envs


def test_the_other_algorithms_block_is_refused() -> None:
    sac_block = raw("sac_balance")["sac"]
    ppo_block = raw("ppo_balance")["ppo"]
    refused({**raw("ppo_balance"), "sac": sac_block}, "sac")
    refused({**raw("sac_balance"), "ppo": ppo_block}, "ppo")


def test_health_is_ppo_only() -> None:
    refused({**raw("sac_balance"), "health": raw("ppo_balance")["health"]}, "health")
    no_health = {k: v for k, v in raw("ppo_balance").items() if k != "health"}
    cfg = parse_train_config(as_yaml(no_health))
    assert isinstance(cfg, PpoTrainConfig)
    assert cfg.health is None


def test_action_excess_off_takes_no_other_key() -> None:
    def off_with_box(d: dict[str, Any]) -> None:
        d["env"]["action_excess"] = {"enabled": False, "box": 2}

    refused(edited("ppo_balance", off_with_box), "action_excess", "box")


@pytest.mark.parametrize("enabled", [1, 0, "true", None])
def test_action_excess_enabled_is_a_real_bool(enabled: object) -> None:
    def tag(d: dict[str, Any]) -> None:
        d["env"]["action_excess"]["enabled"] = enabled

    refused(edited("ppo_balance", tag), "action_excess")


def test_sac_refuses_the_action_excess_penalty() -> None:
    def on(d: dict[str, Any]) -> None:
        d["env"]["action_excess"] = {"enabled": True, "box": 2.0, "weight": 0.1}

    refused(edited("sac_balance", on), "action_excess")


def test_ppo_with_relu_is_refused() -> None:
    refused(edited("ppo_balance", lambda d: d["policy"].update(activation="relu")), "activation")
    refused(edited("sac_balance", lambda d: d["policy"].update(activation="tanh")), "activation")


@pytest.mark.parametrize("arch", [[64, 64, 64], [64], [0, 64], [64, 257], [64.0, 64]])
def test_net_arch_is_exactly_two_widths_in_1_to_256(arch: list[object]) -> None:
    refused(edited("ppo_balance", lambda d: d["policy"]["net_arch"].update(pi=arch)), "net_arch")


def test_qf_on_ppo_and_vf_on_sac_are_refused() -> None:
    refused(edited("ppo_balance", lambda d: d["policy"]["net_arch"].update(qf=[64, 64])), "qf")
    refused(edited("sac_balance", lambda d: d["policy"]["net_arch"].update(vf=[64, 64])), "vf")


def test_log_std_init_on_sac_is_refused() -> None:
    refused(edited("sac_balance", lambda d: d["policy"].update(log_std_init=0.0)), "log_std_init")


def test_learning_rate_final_above_initial_is_refused() -> None:
    def up(d: dict[str, Any]) -> None:
        d["ppo"]["learning_rate"] = {"schedule": "linear", "initial": 1e-4, "final": 3e-4}

    refused(edited("ppo_balance", up), "learning_rate")


def test_a_constant_learning_rate_has_final_equal_to_initial() -> None:
    def drift(d: dict[str, Any]) -> None:
        d["sac"]["learning_rate"]["final"] = 0.0

    refused(edited("sac_balance", drift), "learning_rate")


def test_the_pid_gate_needs_a_saturation_bar() -> None:
    def no_bar(d: dict[str, Any]) -> None:
        del d["selection"]["max_saturation"]

    refused(edited("ppo_balance", no_bar), "selection")
    data = edited("smoke_ppo", no_bar)  # the none gate does not read it
    assert parse_train_config(as_yaml(data)).selection.max_saturation is None


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("selection", "protocol"), "eval_v1", "protocol"),  # an alias, not the canonical id
        (("selection", "protocol"), "balance-v1", "protocol"),
        (("selection", "seeds"), {"first": 2199, "last": 2000}, "seeds"),
        (("selection", "seeds"), {"first": 0, "last": 10_000}, "seeds"),
        (("selection", "seeds"), {"first": 2**52, "last": 2**52}, "seeds"),  # training seeds
        (("selection", "gate"), "lqr", "gate"),
        (("selection", "every_steps"), 0, "every_steps"),
        (("selection", "max_saturation"), 1.5, "max_saturation"),
        (("env", "challenge"), "nope-v1", "challenge"),
        (("env", "challenge"), "../balance-v1", "challenge"),
        (("env", "preset"), "brutal", "preset"),
        (("env", "max_drift_m"), 0.0, "max_drift_m"),
        (
            ("env", "reward"),
            {"tilt": 0.5, "rate": 0.5, "drift": 0.5, "yaw_rate": 0, "effort": 0, "smooth": 0},
            "reward",
        ),
        (("env", "reward"), {"tilt": 0.3}, "reward"),  # every weight written out
        (("normalize", "obs"), False, "obs"),
        (("normalize", "epsilon"), 0.0, "epsilon"),
        (("limits",), {"max_wall_clock_s": 2400, "max_cpu_s": 4500}, "max_tree_rss_mib"),
        (("limits", "max_cpu_s"), 0, "max_cpu_s"),
        (("ppo", "batch_size"), 32_768, "batch_size"),  # more than one rollout of 16,384
        (("ppo", "gamma"), 1.5, "gamma"),
        (("seed",), MAX_TRAIN_SEED, "seed"),
        (("seed",), -1, "seed"),
        (("total_timesteps",), 0, "total_timesteps"),
        (("vec_env",), "ray", "vec_env"),
        (("start_method",), "thread", "start_method"),
        (("algo",), "td3", "algo"),
    ],
)
def test_bad_values_are_refused(path: tuple[str, ...], value: object, field: str) -> None:
    data = copy.deepcopy(raw("ppo_balance"))
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    refused(data, field)


def test_the_other_challenges_protocol_is_refused() -> None:
    data = edited("ppo_balance", lambda d: d["selection"].update(protocol="other-v1/standard"))
    refused(data, "selection.protocol must be a balance-v1 protocol")


def test_too_many_selection_evals_are_refused() -> None:
    every = 5_000_000 // (MAX_SELECTION_EVALS + 1)
    refused(edited("ppo_balance", lambda d: d["selection"].update(every_steps=every)), "evals")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_envs", "8"),  # no coercion from strings
        ("n_envs", True),  # a bool is not an int
        ("n_envs", 8.0),  # nor a float
        ("total_timesteps", 5.0e6),
        ("seed", None),
    ],
)
def test_types_are_strict(key: str, value: object) -> None:
    refused({**raw("ppo_balance"), key: value}, key)


def test_an_int_is_accepted_for_a_float() -> None:
    data = edited("ppo_balance", lambda d: d["env"].update(max_drift_m=2))
    cfg = parse_train_config(as_yaml(data))
    assert cfg.env.max_drift_m == 2.0
    assert isinstance(cfg.env.max_drift_m, float)


@pytest.mark.parametrize(
    "text",
    [
        b"algo: ppo\nseed: .nan\n",
        b"algo: ppo\nseed: .inf\n",
        b"algo: ppo\nseed: 2026-10-15\n",  # a date
        b"algo: ppo\nseed: !!binary aGVsbG8=\n",
        b"algo: ppo\nseed: !!set {1, 2}\n",
    ],
)
def test_values_that_are_not_plain_data_are_refused(text: bytes) -> None:
    with pytest.raises(ConfigError):
        parse_train_config(text)


@pytest.mark.parametrize(
    "text",
    [
        b"- 1\n- 2\n",
        b"just text\n",
        b"",
        b"algo: ppo\n---\nalgo: sac\n",  # two documents
        b"!!python/object:os.system {}\n",
        b"algo: !!python/name:os.system ''\n",
    ],
)
def test_what_is_not_one_mapping_is_refused(text: bytes) -> None:
    with pytest.raises(ConfigError):
        parse_train_config(text)


def test_a_huge_integer_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        parse_train_config(b"seed: " + b"9" * 6000 + b"\n")


def test_errors_never_quote_more_than_8_fields() -> None:
    data = {key: [] for key in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")}
    data["algo"] = "ppo"  # type: ignore[assignment]
    message = refused(data)
    assert "more)" in message


# --- the YAML reader ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        b"a: &x 1\nb: *x\n",
        b"a: &x [1, 2]\n",
        b"base: &base {n: 1}\nderived:\n  <<: *base\n",
        b"lol: &a [x, x]\nlol2: &b [*a, *a]\n",
        b"a: *undefined\n",  # an alias alone
    ],
)
def test_anchors_and_aliases_are_refused(text: bytes) -> None:
    with pytest.raises(ConfigError, match=r"anchors|aliases"):
        safe_load_capped(text)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (b"algo: ppo\nseed: 0\nseed: 5\n", "repeats the key 'seed'"),
        (b"env:\n  preset: standard\n  preset: ideal\n", "repeats the key 'preset'"),
        (b"a: {x: 1, x: 2}\n", "repeats the key 'x'"),
        (b"a: 1\n'a': 2\n", "repeats the key 'a'"),
        (b"base:\n  <<: {n: 1}\n  n: 2\n", "merge keys"),
        (b"? [a, b]\n: 1\n", "plain scalars"),
        (b"? {a: 1}\n: 1\n", "plain scalars"),
    ],
)
def test_repeated_merge_and_complex_keys_are_refused(text: bytes, match: str) -> None:
    """PyYAML would keep the last of two equal keys without a word."""
    with pytest.raises(ConfigError, match=match):
        safe_load_capped(text)


def test_equal_keys_in_different_mappings_are_fine() -> None:
    text = b"a: {x: 1, y: [{x: 2}, {x: 3}]}\nb: {x: 4}\nc: [x, x]\n"
    assert safe_load_capped(text) == {
        "a": {"x": 1, "y": [{"x": 2}, {"x": 3}]},
        "b": {"x": 4},
        "c": ["x", "x"],
    }


def test_a_duplicated_key_in_a_config_file_is_refused() -> None:
    text = (CONFIGS / "ppo_balance.yaml").read_bytes() + b"seed: 7\n"
    with pytest.raises(ConfigError, match="repeats the key 'seed'"):
        parse_train_config(text)


@pytest.mark.parametrize("opener", [b"[", b"{a: "])
def test_8000_levels_are_a_config_error_not_a_recursion_error(opener: bytes) -> None:
    text = opener * 8000
    with pytest.raises(ConfigError, match="nests"):
        safe_load_capped(text)


def test_block_nesting_is_bounded_too() -> None:
    text = "".join(f"{'  ' * i}k{i}:\n" for i in range(40)).encode()
    with pytest.raises(ConfigError, match="nests more than 32"):
        safe_load_capped(text)
    assert safe_load_capped(text, max_depth=64) is not None


def test_more_than_64_kib_is_refused() -> None:
    text = b"#" * MAX_CONFIG_BYTES + b"\n"
    with pytest.raises(ConfigError, match="more than"):
        safe_load_capped(text)
    assert safe_load_capped(b"#" * (MAX_CONFIG_BYTES - 1)) is None


def test_the_reader_takes_bytes_only() -> None:
    with pytest.raises(ConfigError):
        safe_load_capped("a: 1")  # type: ignore[arg-type]


def test_non_utf8_is_refused() -> None:
    with pytest.raises(ConfigError, match="UTF-8"):
        safe_load_capped(b"a: \xff\xfe\n")


def test_a_syntax_error_names_its_line_without_quoting_the_text() -> None:
    with pytest.raises(ConfigError) as info:
        safe_load_capped(b"a: 1\nb: [1, 2\nc: secret\n")
    assert "line" in str(info.value)
    assert "secret" not in str(info.value)


# --- files --------------------------------------------------------------------------------------


def test_load_reads_a_file_and_hashes_its_bytes(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "c.yaml"
    data = (CONFIGS / "smoke_ppo.yaml").read_bytes()
    path.write_bytes(data)
    loaded = read_train_config(path)
    assert loaded.file_sha256 == hashlib.sha256(data).hexdigest()
    assert loaded.config == load_train_config(CONFIGS / "smoke_ppo.yaml")


def test_load_refuses_a_missing_file_and_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_train_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="cannot read"):
        load_train_config(tmp_path)


def test_load_refuses_an_oversize_file_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.yaml"
    path.write_bytes(b"#" * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="more than"):
        load_train_config(path)


def test_load_never_blocks_on_a_fifo(tmp_path: Path, make_fifo: Callable[[Path], Path]) -> None:
    fifo = make_fifo(tmp_path / "c.yaml")
    with pytest.raises(ConfigError, match="not a regular file"):
        load_train_config(fifo)


# --- --seed and the hash -----------------------------------------------------------------------


def test_seed_replaces_the_files_seed() -> None:
    cfg = load_train_config(CONFIGS / "ppo_balance.yaml", seed=2)
    assert cfg.seed == 2
    assert cfg.model_dump() == {
        **load_train_config(CONFIGS / "ppo_balance.yaml").model_dump(),
        "seed": 2,
    }


@pytest.mark.parametrize("seed", [-1, MAX_TRAIN_SEED, True, 1.0])
def test_a_bad_seed_is_refused(seed: Any) -> None:
    with pytest.raises(ConfigError):
        load_train_config(CONFIGS / "ppo_balance.yaml", seed=seed)


def test_config_sha256_ignores_key_order_and_follows_the_seed() -> None:
    data = raw("ppo_balance")
    shuffled = dict(reversed(list(data.items())))
    shuffled["ppo"] = dict(reversed(list(data["ppo"].items())))
    a = parse_train_config(as_yaml(data))
    b = parse_train_config(as_yaml(shuffled))
    assert config_sha256(a) == config_sha256(b)
    shas = {config_sha256(parse_train_config(as_yaml(data), seed=s)) for s in (0, 1, 2)}
    assert len(shas) == 3
    assert config_sha256(a) in shas  # the file's own seed is 0


def test_config_sha256_is_of_the_canonical_json() -> None:
    import hashlib

    cfg = load_train_config(CONFIGS / "sac_balance.yaml")
    text = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert config_sha256(cfg) == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("name", SHIPPED)
def test_the_effective_config_round_trips_through_json(name: str) -> None:
    """``train_meta.json`` stores the effective config; it validates back to the same one."""
    cfg = load_train_config(CONFIGS / f"{name}.yaml")
    again = validate_train_config(json.loads(json.dumps(cfg.model_dump(mode="json"))))
    assert again == cfg
    assert config_sha256(again) == config_sha256(cfg)


def test_validate_refuses_values_json_cannot_hold() -> None:
    with pytest.raises(ConfigError, match="plain data"):
        validate_train_config({"algo": "ppo", "seed": math.nan})
    with pytest.raises(ConfigError, match="plain data"):
        validate_train_config({"algo": "ppo", "seed": {1, 2}})


# --- the learning rate --------------------------------------------------------------------------


def test_lr_fn_linear_clamps_at_zero_progress() -> None:
    fn = lr_fn(LrSchedule(schedule="linear", initial=3e-4, final=0.0))
    assert fn(1.0) == 3e-4
    assert fn(0.0) == 0.0
    assert fn(1 - 5_013_504 / 5_000_000) == 0.0  # the last PPO update's progress, -0.0027
    assert fn(0.5) == pytest.approx(1.5e-4)
    assert all(fn(p / 1000) >= 0.0 for p in range(-1000, 1001))


def test_lr_fn_linear_to_a_nonzero_final() -> None:
    fn = lr_fn(LrSchedule(schedule="linear", initial=3e-4, final=1e-4))
    assert (fn(1.0), fn(0.0), fn(-0.5)) == (3e-4, 1e-4, 1e-4)


def test_lr_fn_constant() -> None:
    fn = lr_fn(LrSchedule(schedule="constant", initial=3e-4, final=3e-4))
    assert {fn(1.0), fn(0.0), fn(-0.1)} == {3e-4}


def test_lr_fn_pickles() -> None:
    """SB3 saves the schedule with the model (``model.zip``, resume only)."""
    fn = lr_fn(LrSchedule(schedule="linear", initial=3e-4, final=0.0))
    assert pickle.loads(pickle.dumps(fn))(0.5) == fn(0.5)


# --- derived files ------------------------------------------------------------------------------


def _replaced(base: dict[str, Any], changes: dict[str, object]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for path, value in changes.items():
        *parents, last = path.split(".")
        target = out
        for key in parents:
            target = target[key]
        assert last in target, path  # a listed path exists in the base
        assert target[last] != value, path  # and the derived file really changes it
        target[last] = value
    return out


SMOKE_COMMON: dict[str, object] = {
    "total_timesteps": 2048,
    "n_envs": 2,
    "vec_env": "dummy",
    "selection.seeds": {"first": 2000, "last": 2001},
    "selection.every_steps": 1024,
    "selection.workers": 2,
    "selection.gate": "none",
    "limits": {"max_wall_clock_s": 300, "max_cpu_s": 600, "max_tree_rss_mib": 4096},
}
DERIVED: dict[str, tuple[str, dict[str, object]]] = {
    "smoke_ppo": (
        "ppo_balance",
        {**SMOKE_COMMON, "ppo.n_steps": 512, "ppo.batch_size": 256},
    ),
    "smoke_sac": ("sac_balance", {**SMOKE_COMMON, "sac.learning_starts": 256}),
    "ablate_nopenalty": (
        "ppo_balance",
        {"seed": 1, "total_timesteps": 1_000_000, "env.action_excess": {"enabled": False}},
    ),
}


@pytest.mark.parametrize("name", list(DERIVED))
def test_derived_files_differ_from_their_base_in_exactly_the_listed_paths(name: str) -> None:
    base_name, changes = DERIVED[name]
    base = load_train_config(CONFIGS / f"{base_name}.yaml").model_dump(mode="json")
    derived = load_train_config(CONFIGS / f"{name}.yaml").model_dump(mode="json")
    assert derived == _replaced(base, changes)


def test_the_listed_paths_check_has_teeth() -> None:
    """A derived file with one more change fails the comparison above."""
    base = load_train_config(CONFIGS / "ppo_balance.yaml").model_dump(mode="json")
    derived = load_train_config(CONFIGS / "smoke_ppo.yaml").model_dump(mode="json")
    derived["ppo"]["n_epochs"] = 5
    assert derived != _replaced(base, DERIVED["smoke_ppo"][1])
