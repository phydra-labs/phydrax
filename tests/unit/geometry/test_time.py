import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.domain import ScalarInterval, TimeInterval


@pytest.fixture
def time_domain() -> ScalarInterval:
    return TimeInterval(0.0, 10.0)


def test_time_interval_construction_alias_and_labels() -> None:
    domain = TimeInterval(0.0, 5.0)
    assert TimeInterval is ScalarInterval
    assert jnp.allclose(domain.start, 0.0)
    assert jnp.allclose(domain.end, 5.0)
    assert ScalarInterval(0.0, 1.0, label="tau").label == "tau"
    with pytest.raises(ValueError):
        TimeInterval(5.0, 0.0)


def test_time_interval_measure_and_membership(time_domain: ScalarInterval) -> None:
    bounds = list(time_domain.bounds)
    assert jnp.allclose(bounds[0], 0.0)
    assert jnp.allclose(bounds[1], 10.0)
    assert jnp.allclose(time_domain.extent, 10.0)
    assert jnp.all(time_domain._contains(jnp.asarray([0.0, 5.0, 10.0])))
    assert not jnp.any(time_domain._contains(jnp.asarray([-1.0, 11.0])))


def test_time_interval_sampling_honors_bounds_and_predicates(
    time_domain: ScalarInterval,
) -> None:
    key = jr.key(0)
    samples = time_domain.sample(5, key=key)
    assert samples.shape == (5,)
    assert jnp.all(samples >= time_domain.start)
    assert jnp.all(samples <= time_domain.end)

    selected = time_domain.sample(5, where=lambda time: time > 5.0, key=key)
    assert selected.shape == (5,)
    assert jnp.all(selected > 5.0)
