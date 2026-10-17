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
    MAX_SELECTION_SEEDS,
    MAX_TRAIN_META_BYTES,
    MAX_TRAIN_SEED,
    LrSchedule,
    PpoTrainConfig,
    SacTrainConfig,
    config_doc,
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
PPO_BALANCE_2A_SHA256 = "9ba39c87e3d419ecf13daeff27bf851e9e0e3bd360ec3328ef58451050158874"
SHA256_2A = {
    # recorded by the Phase 2a runs (runs/ppo_s0, sac_s0, ablate_nopenalty)
    "ppo_balance": PPO_BALANCE_2A_SHA256,
    "sac_balance": "987da388b0f6aa316606afc17f28ad481caf94162df6a1a32f5209c751d257f3",
    "ablate_nopenalty": "1808680c6432535600ceb48e59928e04301ac83073a12977be8bfe3dbf2e997e",
    # the train CI job's smoke configs at the 2a merge (386df1e)
    "smoke_ppo": "424f36b7652ea11ed66eccc104af4e1adab86694cedcbc0e12ca649fde74f612",
    "smoke_sac": "5ad044128dfcef83d12791c8f14307af4a7f5989d6a00762f4274bcdac6535da",
}
SHIPPED = (
    "ppo_balance",
    "sac_balance",
    "smoke_ppo",
    "smoke_sac",
    "ablate_nopenalty",
    "ppo_waypoint",
    "ppo_waypoint_ft",
)


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
    every = 5_000_000 // MAX_SELECTION_EVALS  # exactly the most
    parse_train_config(
        as_yaml(edited("ppo_balance", lambda d: d["selection"].update(every_steps=every)))
    )


def test_too_many_selection_seeds_are_refused() -> None:
    def seeds(n: int) -> dict[str, Any]:
        return edited(
            "ppo_balance", lambda d: d["selection"].update(seeds={"first": 2000, "last": 1999 + n})
        )

    refused(seeds(MAX_SELECTION_SEEDS + 1), "selection.seeds", f"at most {MAX_SELECTION_SEEDS}")
    assert len(parse_train_config(as_yaml(seeds(MAX_SELECTION_SEEDS))).selection.seeds.seeds) == (
        MAX_SELECTION_SEEDS
    )


def test_train_meta_fits_its_64_kib_readers_at_the_config_bounds() -> None:
    """``topple export`` and ``topple train`` read ``train_meta.json`` with a 64 KiB cap: at the
    most selection seeds and evals a config allows, with the widest numbers and the waypoint
    gate's two baselines, ``gate_baseline``, ``selection_history`` and the config written with
    ``indent=2`` leave ``MAX_TRAIN_META_BYTES - 56 KiB`` for the remaining identity and resource
    fields."""
    from topple.export import MAX_JSON_BYTES
    from topple.train.selection import GateBaseline, HistoryRow, SelectionResult

    wide = -1.2345678901234568e-300  # 24 characters: the longest repr a float64 has
    extras = {"mean_gates": wide, "docked": wide, "fell": wide, "median_t_finish_s": wide}

    def result(step: int) -> SelectionResult:
        return SelectionResult(
            step=step,
            success_rate=wide,
            score=wide,
            median_energy_j=wide,
            mean_saturation=wide,
            scores=[wide] * MAX_SELECTION_SEEDS,
            extras=extras,
        )

    baseline = GateBaseline("waypoint", reference=result(0), anchor=result(0))
    history = [
        HistoryRow(result(10**10 - k), True, "a" * 64, wide, wide, wide, wide, wide).to_json()
        for k in range(MAX_SELECTION_EVALS + 1, 0, -1)
    ]
    config = load_train_config(CONFIGS / "ppo_balance.yaml").model_dump(mode="json")
    doc = {"config": config, "gate_baseline": baseline.to_json(), "selection_history": history}
    assert MAX_TRAIN_META_BYTES == MAX_JSON_BYTES == 64 * 1024  # the readers' cap
    assert len(json.dumps(doc, indent=2)) <= 56 * 1024


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
        # PyYAML merges any key tagged !!merge, whatever its text: the second a would be dropped
        (b"a: 1\n!!merge x: {b: 2, a: 9}\n", "tag"),
        (b"!!merge x: {seed: 7}\n", "tag"),
        (b"!<tag:yaml.org,2002:merge> x: {seed: 7}\n", "tag"),
        (b"!!str a: 1\n", "tag"),
        (b"! a: 1\n", "tag"),
    ],
)
def test_repeated_merge_and_complex_keys_are_refused(text: bytes, match: str) -> None:
    """PyYAML would keep the last of two equal keys without a word."""
    with pytest.raises(ConfigError, match=match):
        safe_load_capped(text)


@pytest.mark.parametrize(
    "hidden", [b"!!merge hidden: {seed: 12345, n_envs: 16}\n", b"!!merge h: {}\n"]
)
def test_a_tagged_merge_key_in_a_config_file_is_refused(hidden: bytes) -> None:
    text = (CONFIGS / "smoke_ppo.yaml").read_bytes() + hidden
    with pytest.raises(ConfigError, match="tag"):
        parse_train_config(text)
    without_seed = b"".join(
        line
        for line in (CONFIGS / "smoke_ppo.yaml").read_bytes().splitlines(keepends=True)
        if not line.startswith(b"seed:")
    )
    with pytest.raises(ConfigError, match="tag"):
        parse_train_config(without_seed + b"!!merge x: {seed: 7}\n")


def test_tagged_values_are_still_read() -> None:
    assert safe_load_capped(b"a: !!str 1\nb: !!int '2'\n") == {"a": "1", "b": 2}


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


@pytest.mark.parametrize("flow", [True, False])
def test_exactly_32_levels_are_read_and_33_refused(flow: bool) -> None:
    def nested(levels: int) -> bytes:
        if flow:
            return b"[" * levels + b"]" * levels
        return "".join(f"{'  ' * i}k{i}:\n" for i in range(levels)).encode()  # one mapping each

    assert safe_load_capped(nested(32)) is not None
    with pytest.raises(ConfigError, match="nests more than 32"):
        safe_load_capped(nested(33))


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


def clean(message: str, limit: int = 2000) -> None:
    """One bounded line: no control character a terminal or a CI log would act on."""
    assert len(message) <= limit, len(message)
    assert message.isprintable(), message[:200]


def test_a_huge_tag_is_not_quoted_whole() -> None:
    with pytest.raises(ConfigError) as info:
        safe_load_capped(b"a: !" + b"x" * 60_000 + b" 1\n")
    clean(str(info.value), 300)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(algo="x\n::error::pwned\x1b[2J" + "A" * 20_000),
        lambda d: d.update({"bad\nkey\x1b[2J" + "k" * 20_000: 1}),
        lambda d: d["env"].update(challenge="balance\n::error::x\x1b" + "c" * 20_000),
        lambda d: d["selection"].update(protocol="p\n::error::x\x1b" + "p" * 20_000),
        lambda d: d["env"].update(preset="\x1b" + "q" * 20_000),
    ],
)
def test_errors_never_quote_raw_or_unbounded_input(edit: Callable[[dict[str, Any]], None]) -> None:
    """Pydantic's messages quote the input (\"Input tag 'x...' found\"), keys name the field: in
    a ``ConfigError`` (and the ``ExportError`` that wraps it) they are escaped and bounded."""
    data = edited("ppo_balance", edit)
    with pytest.raises(ConfigError) as info:
        validate_train_config(data)
    message = str(info.value)
    clean(message)
    assert "\n" not in message
    assert "\x1b" not in message


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
    text = json.dumps(config_doc(cfg), sort_keys=True, separators=(",", ":"))
    assert config_sha256(cfg) == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("name", list(SHA256_2A))
def test_every_2a_config_keeps_its_sha256(name: str) -> None:
    assert config_sha256(load_train_config(CONFIGS / f"{name}.yaml")) == SHA256_2A[name]


def test_the_2a_config_sha256_is_unchanged_by_the_2b_keys() -> None:
    """The Phase 2a runs (``runs/ppo_s0``, seed 0) recorded this hash: the 2b keys
    (``env.waypoint_reward``, ``init``) are left out when unset, so 2a runs still export."""
    cfg = load_train_config(CONFIGS / "ppo_balance.yaml")
    assert config_sha256(cfg) == PPO_BALANCE_2A_SHA256
    assert "init" not in config_doc(cfg)
    assert "waypoint_reward" not in config_doc(cfg)["env"]
    wp = load_train_config(CONFIGS / "ppo_waypoint.yaml")
    assert config_doc(wp)["env"]["waypoint_reward"]["progress"] == 20.0
    assert "init" in config_doc(load_train_config(CONFIGS / "ppo_waypoint_ft.yaml"))


INIT = {
    "from_runs": ["runs/a", "runs/b"],
    "step": "best",
    "zero_inputs": [2, 4, 5],
    "log_std": -1.0,
}


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"from_runs": []}, "at least 1"),
        ({"from_runs": ["/abs/run"]}, "relative"),
        ({"from_runs": ["runs/../x"]}, "relative"),
        ({"from_runs": ["runs//x"]}, "relative"),
        ({"from_runs": ["runs/.hidden"]}, "relative"),
        ({"from_runs": ["runs/a", "runs/a"]}, "repeat"),
        ({"from_runs": ["runs/a b"]}, "relative"),
        ({"step": 100}, "best"),
        ({"zero_inputs": [4, 2]}, "increasing"),
        ({"zero_inputs": [-1]}, "greater than or equal"),
        ({"log_std": 3.0}, "less than or equal"),
        ({"extra": 1}, "Extra inputs"),
    ],
)
def test_init_is_validated(change: dict[str, object], match: str) -> None:
    data = raw("ppo_waypoint")
    data["init"] = {**INIT, **change}
    with pytest.raises(ConfigError, match=match):
        parse_train_config(as_yaml(data))


def test_init_is_for_a_waypoint_challenge_only() -> None:
    data = raw("ppo_balance")
    data["init"] = INIT
    with pytest.raises(ConfigError, match="waypoint challenge only"):
        parse_train_config(as_yaml(data))
    data = raw("sac_balance")
    data["init"] = INIT
    with pytest.raises(ConfigError, match="Extra inputs"):
        parse_train_config(as_yaml(data))


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
    # §8.8: exactly these key paths (it keeps the LR schedule, start_method and health)
    "ppo_waypoint": (
        "ppo_balance",
        {
            "total_timesteps": 6_000_000,
            "env.challenge": "waypoint-v1",
            "env.max_drift_m": 2.5,
            "env.reward.drift": 0.0,
            "env.waypoint_reward": {
                "balance_scale": 0.1,
                "progress": 20.0,
                "gate": 10.0,
                "dock": 20.0,
                "terminated": -50.0,
                "dock_hold": 0.0,
            },
            "selection": {
                "protocol": "waypoint-v1/standard",
                "seeds": {"first": 2000, "last": 2199},
                "every_steps": 500_000,
                "workers": 4,
                "gate": "waypoint",
                "max_saturation": None,
            },
            "limits": {"max_wall_clock_s": 3600, "max_cpu_s": 7129, "max_tree_rss_mib": 8192},
        },
    ),
    # §8.8 arm (b): (a) plus init
    "ppo_waypoint_ft": (
        "ppo_waypoint",
        {
            "init": {
                "from_runs": ["runs/ppo_s0", "runs/ppo_s1", "runs/ppo_s2"],
                "step": "best",
                "zero_inputs": [2, 4, 5],
                "log_std": -1.0,
            }
        },
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


def test_a_waypoint_config_needs_the_waypoint_reward() -> None:
    def drop(d: dict[str, Any]) -> None:
        del d["env"]["waypoint_reward"]

    refused(edited("ppo_waypoint", drop), "waypoint_reward")


def test_a_balance_config_refuses_the_waypoint_reward() -> None:
    def add(d: dict[str, Any]) -> None:
        d["env"]["waypoint_reward"] = raw("ppo_waypoint")["env"]["waypoint_reward"]

    refused(edited("ppo_balance", add), "waypoint_reward")


@pytest.mark.parametrize(
    ("key", "value"),
    [("terminated", 1.0), ("progress", -1.0), ("balance_scale", 11.0), ("dock_hold", 10.5)],
)
def test_the_waypoint_reward_weights_are_bounded(key: str, value: float) -> None:
    def bad(d: dict[str, Any]) -> None:
        d["env"]["waypoint_reward"][key] = value

    refused(edited("ppo_waypoint", bad), key)


@pytest.mark.parametrize(
    ("name", "gate"),
    [("ppo_waypoint", "pid"), ("ppo_balance", "waypoint"), ("sac_balance", "waypoint")],
)
def test_the_gate_must_fit_the_challenge_kind(name: str, gate: str) -> None:
    def swap(d: dict[str, Any]) -> None:
        d["selection"]["gate"] = gate
        d["selection"]["max_saturation"] = 0.08 if gate == "pid" else None

    refused(edited(name, swap), "selection.gate", "does not fit")


def test_init_refuses_reward_normalization() -> None:
    """apply_init freezes VecNormalize, which would also freeze the reward statistics."""
    data = raw("ppo_waypoint")
    data["init"] = INIT
    data["normalize"] = {**data["normalize"], "reward": True}
    with pytest.raises(ConfigError, match=r"normalize\.reward must be false"):
        parse_train_config(as_yaml(data))
