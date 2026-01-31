#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.domain.geometry1d._primitives import Interval1d


def test_interval():
    start, end = 0.0, 2.0
    interval = Interval1d(start=start, end=end)
    assert np.isclose(float(interval.length), end - start, atol=1e-6)


def test_line_segment_invalid_endpoints():
    start, end = 2.0, 0.0  # Invalid since start >= end
    with pytest.raises(ValueError, match="`start` must be less than `end`."):
        Interval1d(start=start, end=end)


@pytest.fixture
def simple_interval():
    return Interval1d(start=0.0, end=1.0)


def test_sample_interior(simple_interval):
    num_points = 100
    sampled_points = simple_interval.sample_interior(num_points=num_points)
    assert sampled_points.shape == (num_points, 1)
    assert np.all(sampled_points >= simple_interval.start)
    assert np.all(sampled_points <= simple_interval.end)


def test_sample_boundary(simple_interval):
    num_points = 50
    key = jr.key(1)
    sampled_points = simple_interval.sample_boundary(num_points=num_points, key=key)
    assert sampled_points.shape == (num_points, 1)
    boundary_values = [float(simple_interval.start), float(simple_interval.end)]
    assert np.all(np.isin(np.asarray(sampled_points).flatten(), boundary_values))


def test_contains_method(simple_interval):
    inside_point = jnp.array([[0.5]])
    outside_point = jnp.array([[1.5]])
    assert simple_interval._contains(inside_point)[0]
    assert not simple_interval._contains(outside_point)[0]


def test_on_boundary_method(simple_interval):
    boundary_point = jnp.array([[1.0]])
    interior_point = jnp.array([[0.5]])
    assert simple_interval._on_boundary(boundary_point)[0]
    assert not simple_interval._on_boundary(interior_point)[0]


def test_boundary_normals(simple_interval):
    points = jnp.array([[0.0], [1.0]])
    expected_normals = np.array([[-1.0], [1.0]])
    computed_normals = simple_interval._boundary_normals(points)
    assert np.allclose(computed_normals, expected_normals, atol=1e-6)
