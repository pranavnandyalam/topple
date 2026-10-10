import numpy as np
import pytest

from topple.rng import (
    MAX_SEED,
    Stream,
    controller_seed,
    stream_rng,
    validate_seed,
)


def test_streams_are_append_only_ids() -> None:
    assert [s.value for s in Stream] == [0, 1, 2, 3, 4, 5]
    assert [s.name for s in Stream] == ["INIT", "PUSHES", "SENSOR", "MODEL", "DELAY", "CONTROLLER"]


def test_stream_rng_deterministic() -> None:
    a = stream_rng(7, Stream.PUSHES).random(5)
    b = stream_rng(7, Stream.PUSHES).random(5)
    np.testing.assert_array_equal(a, b)


def test_streams_are_independent() -> None:
    a = stream_rng(7, Stream.PUSHES).random(5)
    b = stream_rng(7, Stream.MODEL).random(5)
    c = stream_rng(8, Stream.PUSHES).random(5)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_agent_index_separates_streams() -> None:
    a = stream_rng(7, Stream.INIT, agent=0).random(3)
    b = stream_rng(7, Stream.INIT, agent=1).random(3)
    assert not np.array_equal(a, b)


def test_consuming_one_stream_does_not_shift_another() -> None:
    pushes_alone = stream_rng(3, Stream.PUSHES).random(4)
    noise = stream_rng(3, Stream.SENSOR)
    noise.random(1000)
    np.testing.assert_array_equal(stream_rng(3, Stream.PUSHES).random(4), pushes_alone)


def test_controller_seed_hides_episode_seed() -> None:
    s = controller_seed(1234)
    assert s != 1234
    assert 0 <= s < 2**53
    assert controller_seed(1234) == s
    assert controller_seed(1235) != s


def test_controller_seed_depends_on_salt() -> None:
    assert controller_seed(1234, salt=0) != controller_seed(1234, salt=2**100 + 17)


@pytest.mark.parametrize("seed", [0, 1, MAX_SEED - 1])
def test_validate_seed_ok(seed: int) -> None:
    assert validate_seed(seed) == seed


@pytest.mark.parametrize("seed", [-1, MAX_SEED, 2**64, 1.5, True])
def test_validate_seed_rejects(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_seed(seed)  # type: ignore[arg-type]
