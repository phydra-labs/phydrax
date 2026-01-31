#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.domain._scalar import ScalarInterval
from phydrax.domain._time import TimeInterval


@pytest.fixture
def time_domain():
    return TimeInterval(0.0, 10.0)


def test_init_valid():
    td = TimeInterval(0.0, 5.0)
    assert jnp.allclose(td.start, 0.0)
    assert jnp.allclose(td.end, 5.0)


def test_init_invalid():
    with pytest.raises(ValueError):
        TimeInterval(5.0, 0.0)


def test_time_interval_alias():
    assert TimeInterval is ScalarInterval


def test_scalar_interval_label():
    dom = ScalarInterval(0.0, 1.0, label="tau")
    assert dom.label == "tau"


def test_bounds(time_domain):
    bounds = list(time_domain.bounds)
    assert jnp.allclose(bounds[0], 0.0)
    assert jnp.allclose(bounds[1], 10.0)


def test_extent(time_domain):
    extent = time_domain.extent
    assert jnp.allclose(extent, 10.0)


def test_sample_default(time_domain):
    key = jr.key(0)
    samples = time_domain.sample(5, key=key)
    assert samples.shape == (5,)
    assert jnp.all(samples >= time_domain.start)
    assert jnp.all(samples <= time_domain.end)


def test_sample_with_where(time_domain):
    key = jr.key(0)
    where = lambda t: t > 5.0
    samples = time_domain.sample(5, where=where, key=key)
    assert samples.shape == (5,)
    assert jnp.all(samples > 5.0)


def test_contains(time_domain):
    points_inside = jnp.array([0.0, 5.0, 10.0])
    points_outside = jnp.array([-1.0, 11.0])
    assert jnp.all(time_domain._contains(points_inside))
    assert not jnp.any(time_domain._contains(points_outside))
