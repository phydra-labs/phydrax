#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.metrix import (
    AbstractStateGeometry,
    EuclideanStateGeometry,
    LocalRetraction,
)


def test_abstract_state_geometry_contract_is_exported_and_abstract():
    with pytest.raises(TypeError):
        AbstractStateGeometry()

    geometry = EuclideanStateGeometry()
    assert isinstance(geometry, AbstractStateGeometry)
    assert geometry.geometry_id == "state-geometry:euclidean"
    assert geometry.retraction_method == "addition"
    assert geometry.trivial


def test_euclidean_local_retraction_preserves_state_shaped_contract():
    geometry = EuclideanStateGeometry(geometry_id="geometry:test-euclidean")
    base = jnp.array([[1.0, -2.0], [0.5, 3.0]])
    increment = jnp.array([[0.25, 1.0], [-0.5, 2.0]])
    retraction = geometry.local_retraction(base)

    assert isinstance(retraction, LocalRetraction)
    assert retraction.retraction_id == "geometry:test-euclidean:local-retraction"
    assert retraction.resolved_method == "addition"
    assert jnp.array_equal(retraction(increment), base + increment)
    assert jnp.array_equal(
        retraction.pullback(increment, 2.0 * increment),
        2.0 * increment,
    )
    assert bool(geometry.contains(base))
    assert jnp.array_equal(geometry.project_tangent(base, increment), increment)
    assert jnp.array_equal(
        geometry.inverse_retract(base, base + increment),
        increment,
    )
    assert jnp.allclose(geometry.interpolate(base, base + increment, 0.25), base + 0.25 * increment)


def test_euclidean_geometry_is_jittable_and_differentiable():
    geometry = EuclideanStateGeometry()
    base = jnp.array([0.5, -1.0])

    @jax.jit
    def objective(increment):
        point = geometry.local_retraction(base).evaluate(increment)
        return jnp.sum(point**2)

    increment = jnp.array([0.25, 0.5])
    assert jnp.allclose(objective(increment), jnp.sum((base + increment) ** 2))
    assert jnp.allclose(jax.grad(objective)(increment), 2.0 * (base + increment))


def test_local_retraction_rejects_shape_changes():
    retraction = EuclideanStateGeometry().local_retraction(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="preserve state shape"):
        retraction.evaluate(jnp.zeros((4,)))
