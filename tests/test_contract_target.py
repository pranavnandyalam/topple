"""``balancer-target.obs.v1`` (docs/design/phase-2.md §8.4; contract §6).

The 10 ``balancer.obs.v1`` features, then ``gate_x, gate_y, next_x, next_y, final``: 15
float32 features; on the wire 16 float64, ``t`` first (128 bytes). ``controller_obs`` builds it
when given the schema; its 2-argument call still builds ``BalancerObs``.
"""

from __future__ import annotations

import numpy as np
import pytest

from topple.contract import (
    FEATURES,
    OBS_SCHEMA_ID,
    TARGET_FEATURES,
    TARGET_OBS_SCHEMA_ID,
    BalancerObs,
    BalancerTargetObs,
    controller_obs,
    obs_features,
    validate_schema_id,
)

ROW = [0.1, -0.2, 0.3, -0.4, 5.0, 5.5, 0.6, 0.7, 0.8, -0.9, 0.45, -0.25, 1.1, 0.2, 0.0]


def test_the_schema_and_its_15_features() -> None:
    assert validate_schema_id(TARGET_OBS_SCHEMA_ID) == "balancer-target.obs.v1"
    assert (*FEATURES, "gate_x", "gate_y", "next_x", "next_y", "final") == TARGET_FEATURES
    assert obs_features(OBS_SCHEMA_ID) == FEATURES
    assert obs_features(TARGET_OBS_SCHEMA_ID) == TARGET_FEATURES
    with pytest.raises(ValueError, match="unknown"):
        obs_features("other.obs.v1")


def test_controller_obs_builds_the_schema_it_is_given() -> None:
    obs = controller_obs(np.array(ROW, np.float64), 0.25, schema=TARGET_OBS_SCHEMA_ID)
    assert isinstance(obs, BalancerTargetObs)
    assert obs.t == 0.25
    assert obs.gate == (float(np.float32(0.45)), float(np.float32(-0.25)))
    assert obs.next == (float(np.float32(1.1)), float(np.float32(0.2)))
    assert obs.final == 0.0
    assert obs.pitch == float(np.float32(0.1))
    np.testing.assert_array_equal(obs.to_array(), np.asarray(ROW, np.float32))
    two_arg = controller_obs(np.array(ROW[:10]), 0.25)  # Phase 1's call, unchanged
    assert type(two_arg) is BalancerObs


def test_the_wire_is_128_bytes_and_lossless() -> None:
    obs = controller_obs(np.array(ROW), 1.5, schema=TARGET_OBS_SCHEMA_ID)
    data = obs.to_wire()
    assert len(data) == 128
    assert BalancerTargetObs.from_wire(data) == obs
    with pytest.raises(ValueError, match="128"):
        BalancerTargetObs.from_wire(data[:-1])


@pytest.mark.parametrize("width", [10, 14, 16])
def test_a_wrong_width_is_refused(width: int) -> None:
    with pytest.raises(ValueError, match="15"):
        controller_obs(np.zeros(width), 0.0, schema=TARGET_OBS_SCHEMA_ID)


def test_final_must_be_0_or_1() -> None:
    row = [*ROW[:14], 0.5]
    with pytest.raises(ValueError, match="final"):
        controller_obs(np.array(row), 0.0, schema=TARGET_OBS_SCHEMA_ID)
