"""Training configuration (docs/design/phase-2.md §2.1, §2.2).

A ``configs/train/*.yaml`` file is complete: there is no inheritance, merge or default the file
could leave to code, except ``selection.max_saturation`` (only the ``pid`` gate reads it) and
``health`` (PPO only). The models are frozen and strict (``extra="forbid"``, no coercion: ``8``
is not ``"8"`` and ``true`` is not ``1``, no NaN or infinity) and discriminated on ``algo``: a
PPO config has a ``ppo`` block, a ``PpoPolicy`` (tanh, ``pi`` and ``vf`` widths, ``log_std_init``)
and optionally ``health``; a SAC config has a ``sac`` block and a ``SacPolicy`` (relu, ``pi`` and
``qf`` widths), and the other algorithm's block or ``health`` is an extra key. ``action_excess``
is discriminated on ``enabled``: ``{enabled: false}`` takes no other key.

``load_train_config`` reads the file without blocking (``topple._inputs.open_regular``, 64 KiB),
parses it with ``topple.yamlsafe.safe_load_capped`` (no anchors, aliases or deep nesting),
validates it through strict JSON mode (YAML sequences become tuples, an int is accepted for a
float, never the reverse) and applies ``--seed``: the result is the effective config, which
``config_sha256`` hashes. Every failure is a ``ConfigError`` naming fields, never a traceback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from topple._inputs import open_regular
from topple.ids import parse_protocol, resolve_protocol, validate_challenge_id
from topple.train.wrappers import TRAIN_SEED_LO
from topple.yamlsafe import MAX_YAML_BYTES, ConfigError, clean_text, safe_load_capped

if TYPE_CHECKING:
    from topple.envs.balance import RewardConfig

MAX_CONFIG_BYTES: Final = MAX_YAML_BYTES
MAX_WIDTH: Final = 256
MAX_N_ENVS: Final = 16  # shipped configs use <= 8 (§1); 16 is for Phase 13's bigger machines
MAX_TRAIN_SEED: Final = 2**32  # SB3 seeds numpy's legacy global RNG, which takes < 2**32
MAX_TOTAL_TIMESTEPS: Final = 10**10
MAX_TRAIN_META_BYTES: Final = 64 * 1024
"""``train_meta.json``'s size budget: ``topple export`` and ``topple train`` read it with this cap
(§3.3 step 2), so the bounds below keep a run's file under it."""
MAX_SELECTION_EVALS: Final = 40
"""Most periodic selection evals a run may ask for (``total_timesteps // every_steps``); the
shipped configs ask for 19 or 20 (§2.4), waypoint 12 (16 at 8M). With ``MAX_SELECTION_SEEDS``
it bounds ``selection_history`` and ``gate_baseline`` so that ``train_meta.json`` stays under
``MAX_TRAIN_META_BYTES`` (tested at the bounds), and bounds ``selection_scores.npz``."""
MAX_SELECTION_SEEDS: Final = 250
"""Most selection seeds (every config uses 200; smoke 2): ``gate_baseline`` keeps one score per
seed for each baseline in ``train_meta.json``."""
GATES_OF_KIND: Final = {"balance": ("pid", "none"), "waypoint": ("waypoint", "none")}
"""The selection gates each challenge kind takes (``none``: the smoke configs)."""
MAX_MESSAGES: Final = 8
MAX_LOC_CHARS: Final = 96
_PRESET_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _challenge_id(value: str) -> str:
    return validate_challenge_id(value)


def _canonical_protocol(value: str) -> str:
    """A ``<challenge>/<preset>`` id written out (an alias such as ``eval_v1`` is refused: the
    selection protocol is recorded and re-hashed by its canonical id, §2.3)."""
    parse_protocol(value)
    if resolve_protocol(value) != value:
        raise ValueError("write the protocol as <challenge>/<preset>, not an alias")
    return value


def _preset_name(value: str) -> str:
    if not value or not value[0].isalpha() or not set(value) <= _PRESET_CHARS:
        raise ValueError("a preset name is lowercase letters, digits and _")
    return value


ChallengeId = Annotated[str, AfterValidator(_challenge_id)]
ProtocolId = Annotated[str, AfterValidator(_canonical_protocol)]
PresetName = Annotated[str, Field(max_length=32), AfterValidator(_preset_name)]
Width = Annotated[int, Field(ge=1, le=MAX_WIDTH)]
Hidden = tuple[Width, Width]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Weight = Annotated[float, Field(ge=0.0, le=1.0)]
Rate = Annotated[float, Field(gt=0.0, le=1.0)]
Count = Annotated[int, Field(ge=1, le=1 << 20)]
Steps = Annotated[int, Field(ge=0, le=MAX_TOTAL_TIMESTEPS)]


# --- learning rate ------------------------------------------------------------------------------


class LrSchedule(_Frozen):
    """``constant`` (``final == initial``) or ``linear`` from ``initial`` at SB3's
    ``progress_remaining`` 1 to ``final`` at 0 (clamped there, ``lr_fn``)."""

    schedule: Literal["constant", "linear"]
    initial: Rate
    final: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def _ordered(self) -> LrSchedule:
        if self.final > self.initial:
            raise ValueError("learning_rate.final must not exceed learning_rate.initial")
        if self.schedule == "constant" and self.final != self.initial:
            raise ValueError("a constant learning_rate has final == initial")
        return self


@dataclass(frozen=True)
class _Constant:
    value: float

    def __call__(self, progress_remaining: float) -> float:
        return self.value


@dataclass(frozen=True)
class _Linear:
    initial: float
    final: float

    def __call__(self, progress_remaining: float) -> float:
        # §2.2: SB3's progress reaches 1 - 5,013,504/5,000,000 = -0.0027 on the last PPO update;
        # without the clamp the rate would turn negative (gradient ascent).
        return self.final + (self.initial - self.final) * max(progress_remaining, 0.0)


def lr_fn(s: LrSchedule) -> Callable[[float], float]:
    """SB3's learning-rate schedule for ``s``: ``p -> final + (initial - final) * max(p, 0)``
    for ``linear`` (never below ``final >= 0``), ``p -> initial`` for ``constant``. A plain
    picklable object, not a lambda (SB3 saves the schedule with the model)."""
    if s.schedule == "constant":
        return _Constant(s.initial)
    return _Linear(s.initial, s.final)


# --- the environment ----------------------------------------------------------------------------


class RewardWeights(_Frozen):
    """``topple.envs.balance.RewardConfig``'s six weights, all written out (its sum bound
    applies)."""

    tilt: Weight
    rate: Weight
    drift: Weight
    yaw_rate: Weight
    effort: Weight
    smooth: Weight

    @model_validator(mode="after")
    def _bounded(self) -> RewardWeights:
        self.reward_config()  # RewardConfig's own check: the weights sum to at most 1
        return self

    def reward_config(self) -> RewardConfig:
        """The env's ``RewardConfig`` (imported on use: it pulls in MuJoCo)."""
        from topple.envs.balance import RewardConfig

        return RewardConfig(**self.model_dump())


class ActionExcessOn(_Frozen):
    """The PPO action-excess penalty (§2.1): actions in ``Box(-box, box)``, reward minus
    ``weight * min(mean_i(max(|a_i| - 1, 0)^2), 1)``."""

    enabled: Literal[True]
    box: Annotated[float, Field(ge=1.0, le=10.0)]
    weight: Weight


class ActionExcessOff(_Frozen):
    enabled: Literal[False]


def _enabled_tag(value: object) -> str | None:
    """``on``/``off`` for an ``enabled`` that is exactly ``true``/``false`` (pydantic's own
    tag lookup would take ``1`` for ``true``); anything else matches no member."""
    enabled = value.get("enabled") if isinstance(value, dict) else getattr(value, "enabled", None)
    if enabled is True:
        return "on"
    if enabled is False:
        return "off"
    return None


ActionExcess = Annotated[
    Annotated[ActionExcessOn, Tag("on")] | Annotated[ActionExcessOff, Tag("off")],
    Discriminator(
        _enabled_tag,
        custom_error_type="invalid_action_excess",
        custom_error_message="action_excess.enabled must be true or false",
    ),
]


class WaypointRewardWeights(_Frozen):
    """§8.8's waypoint training reward (not frozen; never part of a score)."""

    balance_scale: Annotated[float, Field(ge=0.0, le=10.0)]
    progress: Annotated[float, Field(ge=0.0, le=1000.0)]
    gate: Annotated[float, Field(ge=0.0, le=1000.0)]
    dock: Annotated[float, Field(ge=0.0, le=1000.0)]
    terminated: Annotated[float, Field(ge=-1000.0, le=0.0)]
    dock_hold: Annotated[float, Field(ge=0.0, le=10.0)]


class EnvConfig(_Frozen):
    """The training env: the challenge's env class on ``preset`` with the training-only
    ``reward`` and ``max_drift_m`` termination (§2.1)."""

    challenge: ChallengeId
    preset: PresetName
    max_drift_m: Annotated[float, Field(gt=0.0, le=100.0)]
    reward: RewardWeights
    action_excess: ActionExcess
    waypoint_reward: WaypointRewardWeights | None = None
    """Required exactly for a waypoint challenge (§8.8)."""

    @model_validator(mode="after")
    def _known(self) -> EnvConfig:
        from topple.challenge import load_challenge

        try:
            challenge = load_challenge(self.challenge)
        except (OSError, ValueError):
            raise ValueError(f"there is no challenge {self.challenge!r}") from None
        if self.preset not in challenge.config.presets:
            raise ValueError(f"{self.challenge} has no preset {self.preset!r}")
        if (challenge.kind == "waypoint") != (self.waypoint_reward is not None):
            raise ValueError(
                "env.waypoint_reward is required for, and only for, a waypoint challenge"
            )
        return self


class NormalizeConfig(_Frozen):
    """SB3 ``VecNormalize``. Observations are always normalized: the exported graph (§3.2)
    always holds the normalization."""

    obs: bool
    reward: bool
    clip_obs: Annotated[float, Field(gt=0.0, le=1e6)]
    epsilon: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def _obs(self) -> NormalizeConfig:
        if self.obs is not True:
            raise ValueError("normalize.obs must be true (the exported graph normalizes)")
        return self


# --- the policy and the algorithms --------------------------------------------------------------


class PpoNetArch(_Frozen):
    pi: Hidden
    vf: Hidden


class SacNetArch(_Frozen):
    pi: Hidden
    qf: Hidden


class PpoPolicy(_Frozen):
    net_arch: PpoNetArch
    activation: Literal["tanh"]
    log_std_init: Annotated[float, Field(ge=-10.0, le=2.0)]


class SacPolicy(_Frozen):
    """No ``log_std_init``: SB3's SAC ignores it without SDE (§2.1)."""

    net_arch: SacNetArch
    activation: Literal["relu"]


class PpoParams(_Frozen):
    n_steps: Count
    batch_size: Annotated[int, Field(ge=2, le=1 << 20)]
    n_epochs: Annotated[int, Field(ge=1, le=1000)]
    gamma: Rate
    gae_lambda: Fraction
    learning_rate: LrSchedule
    clip_range: Rate
    ent_coef: Fraction
    vf_coef: Annotated[float, Field(ge=0.0, le=10.0)]
    max_grad_norm: Annotated[float, Field(gt=0.0, le=1000.0)]
    target_kl: Annotated[float, Field(gt=0.0)] | None


class SacParams(_Frozen):
    buffer_size: Annotated[int, Field(ge=1, le=10_000_000)]
    learning_starts: Steps
    batch_size: Count
    gamma: Rate
    learning_rate: LrSchedule
    tau: Rate
    train_freq: Count
    gradient_steps: Count
    ent_coef: Literal["auto"] | Annotated[float, Field(gt=0.0)]


# --- selection, limits, health ------------------------------------------------------------------


class SeedRange(_Frozen):
    """Selection seeds ``first..last`` (inclusive, at most ``MAX_SELECTION_SEEDS``), public and
    below the training seeds (§0)."""

    first: Annotated[int, Field(ge=0, lt=TRAIN_SEED_LO)]
    last: Annotated[int, Field(ge=0, lt=TRAIN_SEED_LO)]

    @model_validator(mode="after")
    def _ordered(self) -> SeedRange:
        if self.last < self.first:
            raise ValueError("seeds.last must not be below seeds.first")
        if self.last - self.first >= MAX_SELECTION_SEEDS:
            raise ValueError(
                f"selection takes at most {MAX_SELECTION_SEEDS} seeds (train_meta.json keeps a "
                "score per seed and stays under 64 KiB)"
            )
        return self

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.first, self.last + 1))


class SelectionConfig(_Frozen):
    """Selection evals (§2.4): every ``every_steps`` env steps and at the end, on
    ``protocol`` with ``seeds``, in a pool of ``workers`` (clamped to 4), gated by ``gate``.
    ``max_saturation`` is the ``pid`` gate's saturation bar and required for it."""

    protocol: ProtocolId
    seeds: SeedRange
    every_steps: Annotated[int, Field(ge=1, le=MAX_TOTAL_TIMESTEPS)]
    workers: Annotated[int, Field(ge=1, le=64)]
    gate: Literal["pid", "waypoint", "none"]
    max_saturation: Fraction | None = None

    @model_validator(mode="after")
    def _gate(self) -> SelectionConfig:
        if self.gate == "pid" and self.max_saturation is None:
            raise ValueError("the pid gate needs selection.max_saturation")
        return self


class LimitsConfig(_Frozen):
    """Hard limits (§1): each stops training with its abort flag and exit code 4."""

    max_wall_clock_s: Annotated[int, Field(ge=1)]
    max_cpu_s: Annotated[int, Field(ge=1)]
    max_tree_rss_mib: Annotated[int, Field(ge=1)]


class HealthConfig(_Frozen):
    """PPO health flags (§2.5): ``std_collapse`` below ``min_std`` before
    ``min_std_before_step``; ``late_regression`` after ``regression_after_step`` by
    ``regression_drop`` below the running best."""

    min_std: Annotated[float, Field(gt=0.0)]
    min_std_before_step: Steps
    regression_after_step: Steps
    regression_drop: Rate


MAX_INIT_RUNS: Final = 8
_RUN_PATH_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/"
)


def _run_path(value: str) -> str:
    """A run directory relative to the working directory: no absolute path, no ``..``, no
    empty component, letters, digits and ``_.-/`` only."""
    parts = value.split("/")
    if (
        not value
        or not set(value) <= _RUN_PATH_CHARS
        or value.startswith("/")
        or any(not part or part.startswith(".") for part in parts)
    ):
        raise ValueError("a run path is relative, with no empty component or one starting '.'")
    return value


RunPath = Annotated[str, Field(max_length=200), AfterValidator(_run_path)]


class InitConfig(_Frozen):
    """§8.8's fine-tune init (``surgery.extend_obs``): seed k starts from ``from_runs[k mod n]``
    at its ``best.json`` step, or the next run whose best step is eligible with no abort flag;
    ``zero_inputs`` are the source features whose first-layer columns are zeroed, and the
    policy's ``log_std`` restarts at ``log_std``."""

    from_runs: Annotated[tuple[RunPath, ...], Field(min_length=1, max_length=MAX_INIT_RUNS)]
    step: Literal["best"]
    zero_inputs: Annotated[tuple[Annotated[int, Field(ge=0, lt=64)], ...], Field(max_length=64)]
    log_std: Annotated[float, Field(ge=-10.0, le=2.0)]

    @model_validator(mode="after")
    def _unique(self) -> InitConfig:
        if len(set(self.from_runs)) != len(self.from_runs):
            raise ValueError("init.from_runs must not repeat a run")
        if list(self.zero_inputs) != sorted(set(self.zero_inputs)):
            raise ValueError("init.zero_inputs must be strictly increasing")
        return self


# --- the config ---------------------------------------------------------------------------------


class _Common(_Frozen):
    algo: Literal["ppo", "sac"]
    seed: Annotated[int, Field(ge=0, lt=MAX_TRAIN_SEED)]
    total_timesteps: Annotated[int, Field(ge=1, le=MAX_TOTAL_TIMESTEPS)]
    n_envs: Annotated[int, Field(ge=1, le=MAX_N_ENVS)]
    vec_env: Literal["subproc", "dummy"]
    start_method: Literal["forkserver", "spawn", "fork"]
    env: EnvConfig
    normalize: NormalizeConfig
    selection: SelectionConfig
    limits: LimitsConfig

    @model_validator(mode="after")
    def _selection(self) -> _Common:
        challenge, _ = parse_protocol(self.selection.protocol)
        if challenge != self.env.challenge:
            raise ValueError(
                f"selection.protocol must be a {self.env.challenge} protocol, "
                "the challenge the policy is trained on"
            )
        from topple.challenge import load_challenge

        kind = load_challenge(self.env.challenge).kind
        allowed = GATES_OF_KIND[kind]
        if self.selection.gate not in allowed:
            raise ValueError(
                f"selection.gate {self.selection.gate!r} does not fit a {kind} challenge "
                f"(one of {', '.join(allowed)})"
            )
        evals = self.total_timesteps // self.selection.every_steps
        if evals > MAX_SELECTION_EVALS:
            raise ValueError(
                f"total_timesteps // selection.every_steps is {evals}: at most "
                f"{MAX_SELECTION_EVALS} selection evals"
            )
        return self


class PpoTrainConfig(_Common):
    algo: Literal["ppo"]
    policy: PpoPolicy
    ppo: PpoParams
    health: HealthConfig | None = None
    init: InitConfig | None = None
    """A fine-tune from balance runs (§8.8): only for a waypoint challenge."""

    @model_validator(mode="after")
    def _rollout(self) -> PpoTrainConfig:
        if self.ppo.batch_size > self.ppo.n_steps * self.n_envs:
            raise ValueError("ppo.batch_size must not exceed ppo.n_steps * n_envs (one rollout)")
        if self.init is not None and self.env.waypoint_reward is None:
            raise ValueError("init extends a balance policy to a waypoint challenge only")
        if self.init is not None and self.normalize.reward:
            # apply_init freezes VecNormalize (training=False), which also freezes ret_rms
            raise ValueError("init freezes VecNormalize, so normalize.reward must be false")
        return self


class SacTrainConfig(_Common):
    algo: Literal["sac"]
    policy: SacPolicy
    sac: SacParams

    @model_validator(mode="after")
    def _no_penalty(self) -> SacTrainConfig:
        if self.env.action_excess.enabled:
            raise ValueError("SAC takes action_excess: {enabled: false} (tanh actions stay in ±1)")
        return self


TrainConfig = Annotated[PpoTrainConfig | SacTrainConfig, Field(discriminator="algo")]
_ADAPTER: Final[TypeAdapter[PpoTrainConfig | SacTrainConfig]] = TypeAdapter(TrainConfig)


@dataclass(frozen=True)
class LoadedTrainConfig:
    """The effective config of a file and the sha256 of the file's raw bytes
    (``train_meta.config_file_sha256``, §2.3)."""

    config: PpoTrainConfig | SacTrainConfig
    file_sha256: str


def load_train_config(path: Path, *, seed: int | None = None) -> PpoTrainConfig | SacTrainConfig:
    """The effective config of the YAML file ``path`` (``seed`` replacing the file's, as
    ``topple train --seed`` does); ``ConfigError`` for anything else."""
    return read_train_config(path, seed=seed).config


def read_train_config(path: Path, *, seed: int | None = None) -> LoadedTrainConfig:
    """``load_train_config`` with the sha256 of the bytes it read (one read)."""
    data = _read(path)
    return LoadedTrainConfig(parse_train_config(data, seed=seed), hashlib.sha256(data).hexdigest())


def parse_train_config(data: bytes, *, seed: int | None = None) -> PpoTrainConfig | SacTrainConfig:
    """The effective config of the YAML ``data``; ``ConfigError`` for anything else."""
    cfg = validate_train_config(safe_load_capped(data, MAX_CONFIG_BYTES))
    if seed is None:
        return cfg
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("--seed must be an int")
    return validate_train_config({**config_doc(cfg), "seed": seed})


def validate_train_config(raw: object) -> PpoTrainConfig | SacTrainConfig:
    """``raw`` (plain data: from YAML, or ``train_meta.json``'s ``config``) as a config,
    validated in strict JSON mode; ``ConfigError`` for anything else."""
    try:
        text = json.dumps(raw, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise ConfigError(
            "the config holds a value that is not plain data (a date, a set, binary, NaN or "
            "infinity)"
        ) from None
    try:
        return _ADAPTER.validate_json(text, strict=True)
    except ValidationError as exc:
        raise ConfigError(_describe(exc)) from None


def config_doc(cfg: PpoTrainConfig | SacTrainConfig) -> dict[str, Any]:
    """The config as plain data, as ``train_meta.json`` records it. The keys added in 2b
    (``env.waypoint_reward``, ``init``) are left out when unset, so a balance config keeps the
    document, and so the ``config_sha256``, it had before them (the Phase 2a runs)."""
    doc = cfg.model_dump(mode="json")
    if doc["env"].get("waypoint_reward") is None:
        doc["env"] = {k: v for k, v in doc["env"].items() if k != "waypoint_reward"}
    if doc.get("init", 0) is None:
        doc = {k: v for k, v in doc.items() if k != "init"}
    return doc


def config_sha256(cfg: PpoTrainConfig | SacTrainConfig) -> str:
    """sha256 of the effective config's canonical JSON (``config_doc``; keys sorted, no
    spaces): the same for any key order in the file, different for every ``--seed``."""
    text = json.dumps(config_doc(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> bytes:
    """The bytes of ``path``: a regular file (symlinks followed) of at most 64 KiB, opened
    without blocking (a FIFO or device is refused, never read)."""
    try:
        fh, _ = open_regular(path, MAX_CONFIG_BYTES)
    except ValueError as exc:
        raise ConfigError(f"cannot read {path.name!r}: {exc}") from None
    except OSError as exc:
        raise ConfigError(
            f"cannot read {path.name!r}: {exc.strerror or type(exc).__name__}"
        ) from None
    with fh:
        data = fh.read(MAX_CONFIG_BYTES + 1)  # it may have grown since it was examined
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError(f"{path.name!r} has more than {MAX_CONFIG_BYTES} bytes")
    return data


def _describe(exc: ValidationError) -> str:
    """At most ``MAX_MESSAGES`` field errors as ``<loc>: <message>``. Pydantic's messages and
    the locations can quote the input (``Input tag 'x' found ...``, an extra key's name), so
    each is escaped and cut (``clean_text``): one bounded, printable line."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False)[
        :MAX_MESSAGES
    ]:
        loc = clean_text(".".join(str(p) for p in error["loc"]), MAX_LOC_CHARS) or "(root)"
        parts.append(f"{loc}: {clean_text(error['msg'])}")
    more = exc.error_count() - MAX_MESSAGES
    return "; ".join(parts) + (f" (and {more} more)" if more > 0 else "")
