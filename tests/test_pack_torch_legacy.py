"""Advisory: torch's legacy exporter (``torch.onnx.export(dynamo=False)``) still writes models
``topple policy pack`` accepts (docs/design/phase-2.md §3.3 "Third-party recipes", §6
``test_pack_torch_legacy``, decision log B26).

``train``-marked (torch, stable-baselines3). The python-train CI job runs this file in its own
``continue-on-error`` step: torch 2.14 deprecates the TorchScript exporter with "will be
removed", and its removal must not fail the job. The default job never needs torch: it packs
the committed ``tests/data/thirdparty_{legacy,dynamo}.onnx`` instead (``test_policy_pack``).

1. The fixture script's own network and legacy export (``scripts/make_thirdparty_fixtures.py``)
   still has the shape the committed fixture has, packs, loads and matches torch.
2. A Linear-Tanh actor as a Stable-Baselines3 user would export it (the ``ActorCriticPolicy``
   actor head behind a VecNormalize-style ``[D]`` normalization, I/O at a static batch of 1)
   packs to ``clip(mean)``, the §3.2 structure the Phase 8 C path reads, and matches torch.
3. The same actor straight from SB3's init (every bias zero) packs and loads as ``other``.
"""

from __future__ import annotations

import io
import json
import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import gymnasium as gym
import onnx
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

import make_thirdparty_fixtures as fixtures
from topple.challenge import load_challenge
from topple.policy.pack import pack_policy
from topple.policy.runtime import load_policy_files

pytestmark = pytest.mark.train

BALANCE = load_challenge("balance-v1")
D, A = fixtures.D, fixtures.A
CLIP_OBS = 10.0
EPSILON = 1e-8
TOLERANCE = 1e-5


def run_packed(files: dict[str, bytes], obs: np.ndarray) -> np.ndarray:
    session = load_policy_files(files, challenge=BALANCE).session
    return np.asarray(session.run(["action"], {"obs": obs})[0])


def legacy_bytes(net: nn.Module, input_name: str, output_name: str) -> bytes:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # "will be removed": the point here
        torch.onnx.export(
            net,
            (torch.zeros(1, D),),
            buf,
            dynamo=False,
            opset_version=20,
            input_names=[input_name],
            output_names=[output_name],
        )
    return buf.getvalue()


def test_the_fixture_scripts_legacy_export_packs_loads_and_matches_torch() -> None:
    torch.manual_seed(0)
    net = fixtures.ThirdPartyPolicy().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = fixtures.export_legacy(net)
    assert fixtures.check_legacy(onnx.ModelProto.FromString(data)) == []
    files = pack_policy(data, challenge=BALANCE)
    assert json.loads(files["policy.json"])["action_rule"] == "other"  # rank-0 scale
    assert fixtures.parity(net, files) <= TOLERANCE


class Sb3Actor(nn.Module):
    """``clip(actor(clamp((obs - mean) / sqrt(var + eps), ±clip)), -1, 1)``: SB3's deterministic
    PPO action behind VecNormalize, without the ``Flatten`` features extractor (a no-op on a
    ``[batch, D]`` observation; ``Flatten`` is not an allowed op)."""

    def __init__(self, policy: ActorCriticPolicy, mean: np.ndarray, var: np.ndarray) -> None:
        super().__init__()
        self.policy = policy
        inv_std = (1.0 / np.sqrt(var + EPSILON)).astype(np.float32)
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("inv_std", torch.from_numpy(inv_std))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((obs - self.mean) * self.inv_std, -CLIP_OBS, CLIP_OBS)
        mu = self.policy.action_net(self.policy.mlp_extractor.forward_actor(z))
        out: torch.Tensor = torch.clamp(mu, -1.0, 1.0)
        return out


def sb3_actor() -> Sb3Actor:
    torch.manual_seed(0)
    policy = ActorCriticPolicy(
        gym.spaces.Box(-np.inf, np.inf, (D,), np.float32),
        gym.spaces.Box(-1.0, 1.0, (A,), np.float32),
        lambda _: 3e-4,
        net_arch={"pi": [16, 16], "vf": [16, 16]},
        activation_fn=nn.Tanh,
    )
    rng = np.random.default_rng(0)
    return Sb3Actor(policy, rng.normal(0.0, 1.0, D), rng.uniform(0.1, 4.0, D)).eval()


def check_parity(net: Sb3Actor, files: dict[str, bytes]) -> None:
    obs = np.random.default_rng(1).normal(0.0, 5.0, (64, D)).astype(np.float32)
    with torch.no_grad():
        want = net(torch.from_numpy(obs)).numpy()
    assert float(np.max(np.abs(run_packed(files, obs) - want))) <= TOLERANCE


def test_an_sb3_actor_exported_by_the_legacy_exporter_packs_to_clip_mean() -> None:
    """Biases as training leaves them (distinct, non-zero): the §3.2 structure."""
    net = sb3_actor()
    with torch.no_grad():
        for linear in (*net.policy.mlp_extractor.policy_net[0::2], net.policy.action_net):
            linear.bias.uniform_(-0.1, 0.1)
    files = pack_policy(legacy_bytes(net, "input", "output"), challenge=BALANCE)
    assert json.loads(files["policy.json"])["action_rule"] == "clip(mean)"
    check_parity(net, files)


def test_a_freshly_initialized_sb3_actor_packs_and_loads_as_other() -> None:
    """SB3's orthogonal init zeroes every bias; the legacy exporter then emits one zero tensor
    and an ``Identity`` for the second hidden bias, pack resolves both Gemms to one
    initializer, and 11 initializers are not the §3.2 sequence's 12: ``other``, still exact."""
    net = sb3_actor()
    data = legacy_bytes(net, "input", "output")
    assert "Identity" in [n.op_type for n in onnx.ModelProto.FromString(data).graph.node]
    files = pack_policy(data, challenge=BALANCE)
    assert json.loads(files["policy.json"])["action_rule"] == "other"
    assert len(onnx.ModelProto.FromString(files["policy.onnx"]).graph.initializer) == 11
    check_parity(net, files)
