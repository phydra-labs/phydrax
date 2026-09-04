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
    point = retraction(increment)
    assert jnp.array_equal(
        retraction.inverse_jvp(point, 2.0 * increment),
        2.0 * increment,
    )
    assert jnp.array_equal(
        retraction.vjp(increment, 3.0 * increment),
        3.0 * increment,
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
    local = jnp.array([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]])

    point = jax.jit(geometry.retract)(base, local)
    assert bool(geometry.contains(point))
    assert jnp.allclose(point.T @ point, jnp.eye(3), atol=1e-10)
    assert jnp.linalg.det(point) > 0.0
    assert jnp.allclose(
        geometry.retraction_jvp(base, jnp.zeros_like(local), local),
        base @ local,
    )
    assert jnp.allclose(geometry.inverse_retract(base, point), local, atol=2e-10)
    midpoint = geometry.interpolate(base, point, 0.5)
    assert bool(geometry.contains(midpoint))
    assert jnp.allclose(geometry.interpolate(base, point, 1.0), point, atol=2e-10)


@pytest.mark.parametrize("method", ["exponential", "cayley"])
def test_so_inverse_jvp_inverts_noncommuting_retraction_jvp(method):
    geometry = SpecialOrthogonalStateGeometry(3, retraction=method)
    base = jnp.eye(3)
    local = jnp.array([[0.0, -0.4, 0.2], [0.4, 0.0, -0.3], [-0.2, 0.3, 0.0]])
    direction = jnp.array([[0.0, 0.15, -0.1], [-0.15, 0.0, 0.25], [0.1, -0.25, 0.0]])
    point = geometry.retract(base, local)
    tangent = geometry.retraction_jvp(base, local, direction)
    step = 1e-5
    finite_difference = (
        geometry.retract(base, local + step * direction)
        - geometry.retract(base, local - step * direction)
    ) / (2.0 * step)
    recovered = jax.jit(geometry.retraction_inverse_jvp)(base, point, tangent)

    assert jnp.linalg.norm(local @ direction - direction @ local) > 0.01
    assert jnp.allclose(tangent, finite_difference, atol=2e-10)
    assert jnp.allclose(recovered, direction, atol=2e-9)


def test_so_exponential_inverse_jvp_preserves_tiny_float32_velocity():
    geometry = SpecialOrthogonalStateGeometry(3)
    base = jnp.eye(3, dtype=jnp.float32)
    local = jnp.asarray(
        [[0.0, -0.3, 0.1], [0.3, 0.0, -0.2], [-0.1, 0.2, 0.0]],
        dtype=jnp.float32,
    )
    direction = jnp.asarray(
        [[0.0, 1e-7, -2e-7], [-1e-7, 0.0, 1.5e-7], [2e-7, -1.5e-7, 0.0]],
        dtype=jnp.float32,
    )
    point = geometry.retract(base, local)
    tangent = geometry.retraction_jvp(base, local, direction)
    recovered = geometry.retraction_inverse_jvp(base, point, tangent)

    assert jnp.linalg.norm(recovered) > 0.0
    assert jnp.allclose(recovered, direction, rtol=5e-4, atol=1e-11)


def test_so_exponential_inverse_jvp_solves_heterogeneous_batches_independently():
    geometry = SpecialOrthogonalStateGeometry(3)
    bases = jnp.broadcast_to(jnp.eye(3), (3, 3, 3))
    locals = jnp.array(
        [
            [[0.0, -0.1, 0.2], [0.1, 0.0, -0.05], [-0.2, 0.05, 0.0]],
            [[0.0, 0.7, -0.1], [-0.7, 0.0, 0.3], [0.1, -0.3, 0.0]],
            [[0.0, -0.2, -0.6], [0.2, 0.0, 0.4], [0.6, -0.4, 0.0]],
        ]
    )
    direction_shapes = jnp.array(
        [
            [[0.0, 0.3, -0.2], [-0.3, 0.0, 0.1], [0.2, -0.1, 0.0]],
            [[0.0, -0.1, 0.4], [0.1, 0.0, -0.2], [-0.4, 0.2, 0.0]],
            [[0.0, 0.2, 0.1], [-0.2, 0.0, 0.5], [-0.1, -0.5, 0.0]],
        ]
    )
    scales = jnp.array([1e-8, 0.2, 1e-4])[:, None, None]
    directions = scales * direction_shapes
    points = geometry.retract(bases, locals)
    tangents = geometry.retraction_jvp(bases, locals, directions)
    recovered = jax.jit(geometry.retraction_inverse_jvp)(
        bases,
        points,
        tangents,
    )
    relative_errors = jnp.linalg.norm(
        recovered - directions,
        axis=(-2, -1),
    ) / jnp.linalg.norm(directions, axis=(-2, -1))

    assert jnp.all(jnp.linalg.norm(recovered, axis=(-2, -1)) > 0.0)
    assert jnp.all(relative_errors < 3e-9)


def test_so_exponential_inverse_jvp_does_not_materialize_full_jacobian(monkeypatch):
    geometry = SpecialOrthogonalStateGeometry(5)
    local = jnp.diag(jnp.ones(4), 1) - jnp.diag(jnp.ones(4), -1)
    direction = 0.1 * (jnp.diag(jnp.ones(3), 2) - jnp.diag(jnp.ones(3), -2))
    point, tangent = jax.jvp(
        lambda value: geometry.retract(jnp.eye(5), value),
        (local,),
        (direction,),
    )

    def reject_full_jacobian(*args, **kwargs):
        raise AssertionError("SO inverse JVP must remain matrix-free")

    monkeypatch.setattr(jax, "jacfwd", reject_full_jacobian)
    recovered = geometry.retraction_inverse_jvp(jnp.eye(5), point, tangent)
    _, reconstructed = jax.jvp(
        lambda value: geometry.retract(jnp.eye(5), value),
        (local,),
        (recovered,),
    )

    assert bool(geometry.contains(point))
    assert jnp.allclose(recovered, direction, atol=2e-8)
    assert jnp.linalg.norm(reconstructed - tangent) / jnp.linalg.norm(tangent) < 2e-8


def test_so_exponential_inverse_supports_principal_rotations_and_rejects_cut_locus():
    geometry = SpecialOrthogonalStateGeometry(2)
    radius = jnp.asarray(0.499)
    angle = 2.0 * jnp.arctan(radius)
    supported = jnp.array([[0.0, -angle], [angle, 0.0]])
    direction = jnp.array([[0.0, -0.17], [0.17, 0.0]])
    supported_point, point_tangent = jax.jvp(
        lambda local: geometry.retract(jnp.eye(2), local),
        (supported,),
        (direction,),
    )
    recovered, recovered_tangent = jax.jvp(
        lambda point: geometry.inverse_retract(jnp.eye(2), point),
        (supported_point,),
        (point_tangent,),
    )
    assert jnp.allclose(recovered, supported, atol=2e-12)
    assert jnp.allclose(recovered_tangent, direction, atol=2e-11)

    wide_local = jnp.array([[0.0, -2.0], [2.0, 0.0]])
    wide_point = geometry.retract(jnp.eye(2), wide_local)
    assert jnp.allclose(
        geometry.inverse_retract(jnp.eye(2), wide_point),
        wide_local,
        atol=2e-11,
    )

    cut_locus = -jnp.eye(2)
    with pytest.raises(Exception, match="rotation-by-pi cut locus"):
        geometry.inverse_retract(jnp.eye(2), cut_locus)


def test_spd_congruence_retraction_and_inverse_are_positive_definite():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    base = jnp.array([[2.0, 0.25], [0.25, 1.5]])
    local = jnp.array([[0.2, -0.1], [-0.1, -0.15]])
    point = jax.jit(geometry.retract)(base, local)
    recovered = geometry.inverse_retract(base, point)

    assert bool(geometry.contains(point))
    assert jnp.min(jnp.linalg.eigvalsh(point)) > 0.0
    assert jnp.allclose(recovered, local, atol=1e-9)
    gradient = jax.grad(lambda scale: jnp.trace(geometry.retract(base, scale * local)))(
        jnp.asarray(0.5)
    )
    assert jnp.isfinite(gradient)


def test_spd_inverse_jvp_inverts_retraction_differential():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    base = jnp.array([[2.0, 0.35], [0.35, 1.2]])
    local = jnp.array([[0.25, -0.12], [-0.12, -0.08]])
    direction = jnp.array([[-0.06, 0.09], [0.09, 0.11]])
    epsilon = 1e-5
    tangent = (
        geometry.retract(base, local + epsilon * direction)
        - geometry.retract(base, local - epsilon * direction)
    ) / (2.0 * epsilon)
    point = geometry.retract(base, local)
    recovered = geometry.retraction_inverse_jvp(base, point, tangent)
    assert jnp.allclose(recovered, direction, rtol=2e-7, atol=2e-8)


def test_spd_retraction_gradient_is_finite_at_repeated_eigenvalues():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    local = jnp.array([[0.15, -0.05], [-0.05, -0.1]])
    gradient = jax.grad(lambda base: jnp.sum(geometry.retract(base, local)))(jnp.eye(2))
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


def test_spd_logarithm_jvp_is_stable_for_tiny_separated_spectrum():
    geometry = SymmetricPositiveDefiniteStateGeometry(2)
    identity = jnp.eye(2)
    point = jnp.diag(jnp.array([1e-6, 1e-4]))
    direction = jnp.diag(jnp.array([1e-8, 1e-7]))
    local = geometry.inverse_retract(identity, point)
    _, derivative = jax.jvp(
        lambda value: geometry.inverse_retract(identity, value),
        (point,),
        (direction,),
    )
    assert jnp.allclose(
        jnp.diagonal(local),
        jnp.log(jnp.array([1e-6, 1e-4])),
    )
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.allclose(
        jnp.diagonal(derivative),
        jnp.array([1e-2, 1e-3]),
    )


def test_embedded_and_pointwise_adapters_preserve_explicit_contracts():
    sphere = EmbeddedStateGeometry(
        membership=lambda state: jnp.isclose(jnp.linalg.norm(state), 1.0),
        tangent_projection=lambda state, vector: vector - jnp.vdot(state, vector) * state,
        retraction=lambda state, tangent: (
            (state + tangent) / jnp.linalg.norm(state + tangent)
        ),
        geometry_id="state-geometry:sphere:3",
        retraction_method="normalize",
    )
    base = jnp.array([1.0, 0.0, 0.0])
    point = sphere.retract(base, jnp.array([0.0, 0.2, 0.0]))
    assert bool(sphere.contains(point))
    assert not sphere.supports_exact_inverse
    assert not sphere.supports_exact_differential
    assert not sphere.supports_transport
    assert not sphere.supports_commutator_free
    with pytest.raises(ValueError, match="inverse_retraction callable"):
        sphere.inverse_retract(base, point)
    with pytest.raises(ValueError, match="inverse_retraction callable"):
        sphere.interpolate(base, point, 1.0)
    with pytest.raises(ValueError, match="retraction_jvp_action callable"):
        sphere.retraction_jvp(base, jnp.zeros(3), jnp.ones(3))
    with pytest.raises(ValueError, match="tangent_transport_action callable"):
        sphere.transport_tangent(base, point, jnp.ones(3))

    pointwise = PointwiseStateGeometry(
        SpecialOrthogonalStateGeometry(2),
        (2, 2),
    )
    states = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
    locals_ = jnp.broadcast_to(jnp.array([[0.0, -0.2], [0.2, 0.0]]), states.shape)
    points = pointwise.retract(states, locals_)
    assert bool(pointwise.contains(points))
    assert points.shape == states.shape


def test_unequal_embedded_spaces_preserve_exact_differential_roles():
    scale = jnp.array([2.0, 3.0])
    geometry = EmbeddedStateGeometry(
        membership=lambda state: jnp.all(jnp.isfinite(state)),
        tangent_projection=lambda state, ambient: ambient[:2],
        retraction=lambda state, local: state.at[:2].add(scale * local),
        inverse_retraction=lambda state, point: (point[:2] - state[:2]) / scale,
        retraction_jvp_action=lambda state, local, velocity: scale * velocity,
        retraction_inverse_jvp_action=(lambda state, point, tangent: tangent / scale),
        retraction_vjp_action=lambda state, local, cotangent: scale * cotangent,
        tangent_transport_action=lambda state, point, tangent: tangent,
        cotangent_transport_pullback_action=(lambda state, point, cotangent: cotangent),
        geometry_id="state-geometry:unequal-affine",
        retraction_method="scaled-affine",
        isometric_transport=True,
    )
    source = jnp.array([1.0, -2.0, 4.0])
    local = jnp.array([0.25, -0.5])
    direction = jnp.array([-0.3, 0.2])
    cotangent = jnp.array([1.5, -0.75])
    retraction = geometry.local_retraction(source)
    point = retraction.evaluate(local)
    tangent = retraction.jvp(local, direction)
    local_covector = retraction.vjp(local, cotangent)

    assert point.shape == (3,)
    assert tangent.shape == (2,)
    assert local_covector.shape == (2,)
    assert jnp.allclose(retraction.inverse_jvp(point, tangent), direction)
    assert jnp.allclose(local_covector, scale * cotangent)
    assert jnp.allclose(
        jnp.sum(cotangent * tangent),
        jnp.sum(local_covector * direction),
    )
    assert bool(retraction.chart_evidence(local, direction, cotangent).valid)
    assert bool(
        geometry.transport_evidence(
            source,
            point,
            direction,
            cotangent,
            require_isometry=True,
        ).valid
    )


def test_pointwise_geometry_routes_unequal_role_shapes_over_leading_axes():
    scale = jnp.array([2.0, 3.0])
    geometry = EmbeddedStateGeometry(
        membership=lambda state: jnp.all(jnp.isfinite(state)),
        tangent_projection=lambda state, ambient: ambient[:2],
        retraction=lambda state, local: state.at[:2].add(scale * local),
        inverse_retraction=lambda state, point: (point[:2] - state[:2]) / scale,
        retraction_jvp_action=lambda state, local, velocity: scale * velocity,
        retraction_inverse_jvp_action=(lambda state, point, tangent: tangent / scale),
        retraction_vjp_action=lambda state, local, cotangent: scale * cotangent,
        tangent_transport_action=lambda state, point, tangent: tangent,
        cotangent_transport_pullback_action=(lambda state, point, cotangent: cotangent),
        geometry_id="state-geometry:pointwise-unequal",
        retraction_method="scaled-affine",
    )
    pointwise = PointwiseStateGeometry(
        geometry,
        (3,),
        local_shape=(2,),
        tangent_shape=(2,),
    )
    states = jnp.arange(12.0).reshape((4, 3))
    locals_ = jnp.full((4, 2), 0.1)
    directions = jnp.full((4, 2), -0.2)
    cotangents = jnp.full((4, 2), 0.3)
    points = pointwise.retract(states, locals_)
    tangents = pointwise.retraction_jvp(states, locals_, directions)
    local_covectors = pointwise.retraction_vjp(states, locals_, cotangents)

    assert points.shape == (4, 3)
    assert tangents.shape == (4, 2)
    assert local_covectors.shape == (4, 2)
    assert jnp.allclose(
        pointwise.retraction_inverse_jvp(states, points, tangents),
        directions,
    )
    assert jnp.allclose(
        jnp.sum(cotangents * tangents),
        jnp.sum(local_covectors * directions),
    )


@pytest.mark.parametrize(
    "geometry,base,local,direction,cotangent",
    [
        (
            SpecialOrthogonalStateGeometry(2),
            jnp.eye(2),
            jnp.array([[0.0, -0.1], [0.1, 0.0]]),
            jnp.array([[0.0, 0.2], [-0.2, 0.0]]),
            jnp.array([[0.1, -0.4], [0.3, -0.2]]),
        ),
        (
            SymmetricPositiveDefiniteStateGeometry(2),
            jnp.array([[2.0, 0.1], [0.1, 1.5]]),
            jnp.array([[0.1, -0.02], [-0.02, -0.08]]),
            jnp.array([[-0.04, 0.03], [0.03, 0.05]]),
            jnp.array([[0.2, -0.1], [-0.1, 0.3]]),
        ),
    ],
)
def test_matrix_geometry_vjp_and_transport_certify_algebraic_duality(
    geometry,
    base,
    local,
    direction,
    cotangent,
):
    point = geometry.retract(base, local)
    tangent = geometry.retraction_jvp(base, local, direction)
    local_covector = geometry.retraction_vjp(base, local, cotangent)
    assert jnp.allclose(
        jnp.sum(cotangent * tangent),
        jnp.sum(local_covector * direction),
        rtol=2e-6,
        atol=2e-7,
    )
    evidence = geometry.transport_evidence(
        base,
        point,
        geometry.project_tangent(base, direction),
        cotangent,
    )
    assert bool(evidence.valid)


def test_state_geometry_benchmark_records_manifold_residuals():
    report = run_state_geometry_benchmarks((2,), repeats=1)
    record = report["records"][0]
    assert record["so_exponential"]["orthogonality_error"] < 1e-10
    assert record["so_exponential"]["determinant"] > 0.0
    assert record["spd_congruence_exponential"]["minimum_eigenvalue"] > 0.0
    assert record["so_exponential"]["geometry_id"].startswith("state-geometry:so")
