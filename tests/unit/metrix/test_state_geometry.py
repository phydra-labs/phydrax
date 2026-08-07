#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.metrix import (
    AbstractStateGeometry,
    EmbeddedStateGeometry,
    EuclideanStateGeometry,
    LocalRetraction,
    PointwiseStateGeometry,
    SpecialOrthogonalStateGeometry,
    SymmetricPositiveDefiniteStateGeometry,
)
from tools.metrix_benchmarks import run_state_geometry_benchmarks


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
    assert jnp.allclose(
        geometry.interpolate(base, base + increment, 0.25),
        base + 0.25 * increment,
    )


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


@pytest.mark.parametrize("method", ["exponential", "cayley"])
def test_special_orthogonal_retractions_preserve_group(method):
    geometry = SpecialOrthogonalStateGeometry(3, retraction=method)
    base = jnp.eye(3)
    local = jnp.array(
        [[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]]
    )

    point = jax.jit(geometry.retract)(base, local)
    assert bool(geometry.contains(point))
    assert jnp.allclose(point.T @ point, jnp.eye(3), atol=1e-10)
    assert jnp.linalg.det(point) > 0.0
    assert jnp.allclose(geometry.to_local(base, base @ local), local)


def test_spd_congruence_retraction_and_inverse_are_positive_definite():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    base = jnp.array([[2.0, 0.25], [0.25, 1.5]])
    local = jnp.array([[0.2, -0.1], [-0.1, -0.15]])
    point = jax.jit(geometry.retract)(base, local)
    recovered = geometry.inverse_retract(base, point)

    assert bool(geometry.contains(point))
    assert jnp.min(jnp.linalg.eigvalsh(point)) > 0.0
    assert jnp.allclose(recovered, local, atol=1e-9)
    gradient = jax.grad(
        lambda scale: jnp.trace(geometry.retract(base, scale * local))
    )(jnp.asarray(0.5))
    assert jnp.isfinite(gradient)


def test_spd_retraction_gradient_is_finite_at_repeated_eigenvalues():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    local = jnp.array([[0.15, -0.05], [-0.05, -0.1]])
    gradient = jax.grad(
        lambda base: jnp.sum(geometry.retract(base, local))
    )(jnp.eye(2))
    assert jnp.all(jnp.isfinite(gradient))


def test_spd_inverse_retraction_gradients_are_finite_at_relative_identity():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    identity = jnp.eye(2)
    direction = jnp.array([[0.1, -0.02], [-0.02, 0.05]])
    base_gradient = jax.grad(
        lambda base: jnp.sum(geometry.inverse_retract(base, identity))
    )(identity)
    point_gradient = jax.grad(
        lambda point: jnp.sum(geometry.inverse_retract(identity, point))
    )(identity)
    _, point_jvp = jax.jvp(
        lambda point: geometry.inverse_retract(identity, point),
        (identity,),
        (direction,),
    )
    assert jnp.all(jnp.isfinite(base_gradient))
    assert jnp.all(jnp.isfinite(point_gradient))
    assert jnp.all(jnp.isfinite(point_jvp))
    assert jnp.allclose(point_jvp, direction)


def test_embedded_and_pointwise_adapters_preserve_explicit_contracts():
    sphere = EmbeddedStateGeometry(
        membership=lambda state: jnp.isclose(jnp.linalg.norm(state), 1.0),
        tangent_projection=lambda state, vector: (
            vector - jnp.vdot(state, vector) * state
        ),
        retraction=lambda state, tangent: (
            (state + tangent) / jnp.linalg.norm(state + tangent)
        ),
        geometry_id="state-geometry:sphere:3",
        retraction_method="normalize",
    )
    base = jnp.array([1.0, 0.0, 0.0])
    point = sphere.retract(base, jnp.array([0.0, 0.2, 0.0]))
    assert bool(sphere.contains(point))

    pointwise = PointwiseStateGeometry(
        SpecialOrthogonalStateGeometry(2),
        (2, 2),
    )
    states = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
    locals_ = jnp.broadcast_to(jnp.array([[0.0, -0.2], [0.2, 0.0]]), states.shape)
    points = pointwise.retract(states, locals_)
    assert bool(pointwise.contains(points))
    assert points.shape == states.shape


def test_state_geometry_benchmark_records_manifold_residuals():
    report = run_state_geometry_benchmarks((2,), repeats=1)
    record = report["records"][0]
    assert record["so_exponential"]["orthogonality_error"] < 1e-10
    assert record["so_exponential"]["determinant"] > 0.0
    assert record["spd_congruence_exponential"]["minimum_eigenvalue"] > 0.0
    assert record["so_exponential"]["geometry_id"].startswith("state-geometry:so")
