"""``topple.traj/2`` (docs/design/phase-2.md §8.1a, contract §14a).

``/1`` stays exactly as it is for ``balancer.obs.v1``; every other observation schema writes
``/2``, whose ``obs`` is T x D with 1 <= D <= ``MAX_OBS_WIDTH`` and whose meta names exactly D
features. A trusted reader passes the protocol's ``TrajLayout``; the default (``TRAJ_V1``)
fails closed on a ``/2`` file.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from topple.contract import FEATURES, OBS_SCHEMA_ID, TARGET_FEATURES, TARGET_OBS_SCHEMA_ID
from topple.traj import (
    FORMAT,
    FORMAT_V2,
    MAX_OBS_WIDTH,
    TRAJ_V1,
    Trajectory,
    TrajLayout,
    load_npz,
    read_meta,
    traj_format,
)

WAYPOINT = TrajLayout(FORMAT_V2, len(TARGET_FEATURES))
WP_META = {
    "challenge": "waypoint-v1",
    "obs_schema": TARGET_OBS_SCHEMA_ID,
    "obs_features": list(TARGET_FEATURES),
}


def make(t: int = 5, width: int = 15, meta: dict[str, object] | None = None) -> Trajectory:
    rng = np.random.default_rng(1)
    return Trajectory(
        qpos=rng.normal(size=(t + 1, 9)),
        qvel=rng.normal(size=(t + 1, 8)),
        obs=rng.normal(size=(t, width)).astype(np.float32),
        action=rng.normal(size=(t, 2)),
        ctrl=rng.uniform(-1, 1, size=(t, 2)),
        enc_counts=rng.integers(-100, 100, size=(t, 2)),
        gyro=rng.normal(size=(t, 3)),
        accel=rng.normal(size=(t, 3)),
        push_force=np.zeros((t, 3)),
        epilogue_qpos=rng.normal(size=(2, 9)),
        meta=dict(WP_META if meta is None else meta),
    )


def members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def meta_of(path: Path) -> dict[str, object]:
    raw = np.load(io.BytesIO(members(path)["meta_json.npy"]))
    return json.loads(raw.tobytes())


def test_a_waypoint_record_round_trips_as_v2(tmp_path: Path) -> None:
    path = tmp_path / "wp.npz"
    traj = make()
    traj.save_npz(path)
    assert meta_of(path)["format"] == FORMAT_V2
    back = load_npz(path, max_steps=10, layout=WAYPOINT)
    np.testing.assert_array_equal(back.obs, traj.obs)
    assert back.meta == WP_META


@pytest.mark.parametrize(
    ("meta", "fmt"),
    [
        ({}, FORMAT),  # Phase 1's test files hold no obs_schema
        ({"obs_schema": OBS_SCHEMA_ID}, FORMAT),
        ({"obs_schema": TARGET_OBS_SCHEMA_ID}, FORMAT_V2),
        ({"obs_schema": 7}, FORMAT_V2),  # not exactly the balance string
    ],
)
def test_the_format_rule(meta: dict[str, object], fmt: str) -> None:
    assert traj_format(meta) == fmt


def test_balance_records_are_written_exactly_as_before(tmp_path: Path) -> None:
    """``/1`` members are byte-identical to the merged writer's (array bytes, dtypes, shapes and
    meta_json); whole files differ only by numpy's zip timestamps."""
    meta = {"challenge": "balance-v1", "obs_schema": OBS_SCHEMA_ID, "obs_features": list(FEATURES)}
    traj = make(width=10, meta=meta)
    path = tmp_path / "b.npz"
    traj.save_npz(path)
    assert meta_of(path)["format"] == FORMAT
    expected = json.dumps({"format": FORMAT, **meta}, sort_keys=True).encode()
    raw = np.load(io.BytesIO(members(path)["meta_json.npy"]))
    assert raw.tobytes() == expected
    back = load_npz(path, max_steps=10)  # TRAJ_V1, the default
    np.testing.assert_array_equal(back.obs, traj.obs)


def test_a_15_wide_obs_under_the_v1_layout_is_refused_before_its_data(tmp_path: Path) -> None:
    path = tmp_path / "wp.npz"
    make().save_npz(path)
    with pytest.raises(ValueError, match="obs must have shape"):
        load_npz(path, max_steps=10)  # the default fails closed on a waypoint file


def test_a_10_wide_obs_under_the_waypoint_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "b.npz"
    make(width=10, meta={"challenge": "balance-v1"}).save_npz(path)
    with pytest.raises(ValueError, match="obs must have shape"):
        load_npz(path, max_steps=10, layout=WAYPOINT)


def test_v2_offered_to_a_v1_layout_of_the_right_width_is_refused(tmp_path: Path) -> None:
    """A ``/2`` file that happens to be 10 wide still fails the format check."""
    meta = {**WP_META, "obs_features": list(FEATURES)}
    path = tmp_path / "x.npz"
    make(width=10, meta=meta).save_npz(path)
    with pytest.raises(ValueError, match=r"not a topple\.traj/1"):
        load_npz(path, max_steps=10)


def test_v1_offered_to_a_waypoint_layout_is_refused(tmp_path: Path) -> None:
    layout10 = TrajLayout(FORMAT_V2, 10)
    path = tmp_path / "b.npz"
    make(width=10, meta={"challenge": "balance-v1"}).save_npz(path)
    with pytest.raises(ValueError, match=r"not a topple\.traj/2"):
        load_npz(path, max_steps=10, layout=layout10)


@pytest.mark.parametrize(
    "features",
    [list(TARGET_FEATURES[:14]), "gate_x", [*TARGET_FEATURES[:14], 7], None],
    ids=["14-names", "a-string", "non-string", "missing"],
)
def test_obs_features_must_name_exactly_d_features(tmp_path: Path, features: object) -> None:
    meta = {**WP_META, "obs_features": features}
    if features is None:
        del meta["obs_features"]
    path = tmp_path / "wp.npz"
    make(meta=meta).save_npz(path)
    with pytest.raises(ValueError, match="obs_features"):
        load_npz(path, max_steps=10, layout=WAYPOINT)


def test_widths_beyond_the_cap_are_refused() -> None:
    with pytest.raises(ValueError, match="obs_width"):
        TrajLayout(FORMAT_V2, MAX_OBS_WIDTH + 1)
    with pytest.raises(ValueError, match="obs must have shape"):
        make(width=MAX_OBS_WIDTH + 1)
    with pytest.raises(ValueError, match="10 obs columns"):
        TrajLayout(FORMAT, 15)
    assert TrajLayout(FORMAT, 10) == TRAJ_V1


def test_a_15_wide_obs_claiming_the_balance_schema_cannot_be_built() -> None:
    meta = {**WP_META, "obs_schema": OBS_SCHEMA_ID}
    with pytest.raises(ValueError, match="obs must have shape"):
        make(meta=meta)


def test_read_meta_accepts_both_formats(tmp_path: Path) -> None:
    wp, b = tmp_path / "wp.npz", tmp_path / "b.npz"
    make().save_npz(wp)
    make(width=10, meta={"challenge": "balance-v1"}).save_npz(b)
    assert read_meta(wp)["challenge"] == "waypoint-v1"
    assert read_meta(b)["challenge"] == "balance-v1"


def test_a_65_wide_header_is_refused_before_its_data_is_read(tmp_path: Path) -> None:
    """The header check fires before any obs data: a file whose obs member is only a header
    claiming 65 columns (and no data) is refused for its width, not for truncation."""
    path = tmp_path / "wide.npz"
    make().save_npz(path)
    members_ = members(path)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f4", "fortran_order": False, "shape": (5, 65)}
    )
    members_["obs.npy"] = b"\x93NUMPY\x01\x00" + header.getvalue()[8:]
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members_.items():
            zf.writestr(name, data)
    with pytest.raises(ValueError, match="obs must have shape"):
        load_npz(path, max_steps=10, layout=WAYPOINT)
