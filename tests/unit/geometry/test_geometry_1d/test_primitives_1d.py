import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.domain.geometry1d._primitives import Interval1d


@pytest.fixture
def simple_interval() -> Interval1d:
    return Interval1d(start=0.0, end=1.0)


def test_interval_measure_membership_and_boundary_classification(
    simple_interval: Interval1d,
) -> None:
    assert np.isclose(float(simple_interval.length), 1.0, atol=1e-6)
    assert bool(simple_interval._contains(jnp.asarray([[0.5]]))[0])
    assert not bool(simple_interval._contains(jnp.asarray([[1.5]]))[0])
    assert bool(simple_interval._on_boundary(jnp.asarray([[1.0]]))[0])
    assert not bool(simple_interval._on_boundary(jnp.asarray([[0.5]]))[0])
    with pytest.raises(ValueError, match="`start` must be less than `end`."):
        Interval1d(start=2.0, end=0.0)


def test_interval_sampling_respects_interior_and_boundary_support(
    simple_interval: Interval1d,
) -> None:
    interior = simple_interval.sample_interior(num_points=100)
    assert interior.shape == (100, 1)
    assert np.all(interior >= simple_interval.start)
    assert np.all(interior <= simple_interval.end)

    boundary = simple_interval.sample_boundary(num_points=50, key=jr.key(1))
    assert boundary.shape == (50, 1)
    boundary_values = [float(simple_interval.start), float(simple_interval.end)]
    assert np.all(np.isin(np.asarray(boundary).flatten(), boundary_values))


def test_boundary_normals_and_fields_are_scale_covariant(
    simple_interval: Interval1d,
) -> None:
    np.testing.assert_allclose(
        simple_interval._boundary_normals(jnp.asarray([[0.0], [1.0]])),
        np.asarray([[-1.0], [1.0]]),
        atol=1e-6,
    )

    normalized_points = jnp.asarray([[0.0], [0.25], [0.5], [0.75], [1.0]])
    normalized_factors = []
    gate_values = []
    for scale in (1e-7, 1.0):
        interval = Interval1d(0.0, scale)
        factor = interval.boundary_ansatz_factor
        normalized_factors.append(factor(scale * normalized_points) / scale)
        gate_values.append(interval.make_enforcement_gate()(scale * normalized_points))
        np.testing.assert_allclose(
            jax.grad(factor)(jnp.asarray([0.0])),
            jnp.asarray([-1.0]),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            jax.grad(factor)(jnp.asarray([scale])),
            jnp.asarray([1.0]),
            atol=1e-10,
        )

    np.testing.assert_allclose(normalized_factors[0], normalized_factors[1], atol=1e-10)
    np.testing.assert_allclose(gate_values[0], gate_values[1], atol=1e-10)
