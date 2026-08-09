#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_manifold_and_state_geometry_law_reports():
    manifold = phx.metrix.SphereManifold(3)
    point = jnp.array([1.0, 0.0, 0.0])
    ambient = jnp.array([0.2, 0.7, -0.1])

    manifold_report = phx.metrix.validate_manifold(manifold, point, ambient)
    geometry_report = phx.metrix.validate_state_geometry(
        phx.metrix.EuclideanStateGeometry(), point, ambient
    )

    assert bool(manifold_report.valid)
    assert bool(geometry_report.valid)
    assert manifold_report.metric_duality_residual < 1e-10
    assert geometry_report.retraction_differential_residual < 1e-8


def test_builtin_manifolds_satisfy_shared_retraction_and_metric_laws():
    matrix_point = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    matrix_ambient = jnp.array([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]])
    cases = (
        (
            phx.metrix.EuclideanManifold((3,)),
            jnp.array([0.2, -0.1, 0.4]),
            jnp.array([0.3, 0.5, -0.2]),
        ),
        (
            phx.metrix.SphereManifold(3),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.2, 0.7, -0.1]),
        ),
        (phx.metrix.StiefelManifold(3, 2), matrix_point, matrix_ambient),
        (phx.metrix.GrassmannManifold(3, 2), matrix_point, matrix_ambient),
        (phx.metrix.ObliqueManifold(3, 2), matrix_point, matrix_ambient),
        (
            phx.metrix.FixedRankManifold(3, 3, 2),
            jnp.diag(jnp.array([2.0, 1.0, 0.0])),
            jnp.array([[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [0.2, 0.5, -0.1]]),
        ),
        (
            phx.metrix.SpecialOrthogonalManifold(3),
            jnp.eye(3),
            jnp.array([[0.0, -0.2, 0.1], [0.2, 0.0, -0.3], [-0.1, 0.3, 0.0]]),
        ),
        (
            phx.metrix.AffineInvariantSPDManifold(3),
            jnp.diag(jnp.array([2.0, 1.5, 1.0])),
            jnp.array([[0.2, -0.1, 0.3], [-0.1, 0.1, -0.2], [0.3, -0.2, 0.4]]),
        ),
    )

    for manifold, point, ambient in cases:
        report = phx.metrix.validate_manifold(
            manifold,
            point,
            ambient,
            tolerance=2e-4,
        )
        assert bool(report.valid), manifold.manifold_id


def test_matrix_manifolds_obey_constraints_and_quotient_invariance():
    key = jax.random.key(4)
    stiefel = phx.metrix.StiefelManifold(5, 2)
    point = jnp.linalg.qr(jax.random.normal(key, (5, 2)))[0]
    tangent = stiefel.project_tangent(point, jax.random.normal(key, (5, 2)))
    destination = stiefel.retract(point, 0.1 * tangent)

    grassmann = phx.metrix.GrassmannManifold(5, 2)
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    horizontal = grassmann.project_tangent(point, tangent)

    assert bool(stiefel.contains(destination))
    assert jnp.max(jnp.abs(destination.T @ destination - jnp.eye(2))) < 1e-10
    assert jnp.allclose(
        grassmann.inner(point, horizontal, horizontal),
        grassmann.inner(point @ rotation, horizontal @ rotation, horizontal @ rotation),
        atol=1e-10,
    )


def test_oblique_and_fixed_rank_manifolds_preserve_their_constraints():
    key = jax.random.key(9)
    oblique = phx.metrix.ObliqueManifold(4, 3)
    raw = jax.random.normal(key, (4, 3))
    point = raw / jnp.linalg.norm(raw, axis=0, keepdims=True)
    tangent = oblique.project_tangent(point, jnp.ones((4, 3)))
    destination = oblique.retract(point, 0.1 * tangent)

    assert bool(oblique.contains(destination))
    assert jnp.max(jnp.abs(jnp.sum(destination**2, axis=0) - 1.0)) < 1e-10

    fixed_rank = phx.metrix.FixedRankManifold(4, 3, 2)
    point = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tangent = fixed_rank.project_tangent(
        point, jax.random.normal(jax.random.key(10), (4, 3))
    )
    destination = fixed_rank.retract(point, 0.01 * tangent)

    assert bool(fixed_rank.contains(destination))
    assert jnp.linalg.matrix_rank(destination, tol=1e-8) == 2


def test_differentiable_maps_compose_pull_back_metrics_and_transform_connections():
    polar = phx.metrix.CoordinateChart("polar", ("r", "theta"))
    cartesian = phx.metrix.CoordinateChart("cartesian", ("x", "y"))
    polar_to_cartesian = phx.metrix.DifferentiableMap(
        polar,
        cartesian,
        lambda q: jnp.array([q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])]),
    )
    identity = phx.metrix.DifferentiableMap(
        cartesian,
        cartesian,
        lambda q: q,
    )
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(2), chart=cartesian)
    point = jnp.array([2.0, 0.3])
    pulled_metric = phx.metrix.pullback_metric(metric, polar_to_cartesian)
    pulled_connection = phx.metrix.pullback_affine_connection(
        phx.metrix.LeviCivitaConnection(metric), polar_to_cartesian
    )

    assert jnp.allclose(
        polar_to_cartesian.compose(identity)(point),
        polar_to_cartesian(point),
    )
    assert jnp.allclose(pulled_metric(point), jnp.diag(jnp.array([1.0, 4.0])))
    assert (
        phx.metrix.connection_transformation_residual(
            phx.metrix.LeviCivitaConnection(pulled_metric),
            phx.metrix.LeviCivitaConnection(metric),
            polar_to_cartesian,
            point,
        )
        < 1e-10
    )
    assert jnp.max(jnp.abs(phx.metrix.torsion_tensor(pulled_connection, point))) < 1e-10
    assert (
        jnp.max(
            jnp.abs(
                phx.metrix.nonmetricity_tensor(pulled_connection, pulled_metric, point)
            )
        )
        < 1e-10
    )


def test_general_affine_connection_exposes_torsion_nonmetricity_and_curvature():
    chart = phx.metrix.CoordinateChart("affine_plane", ("x", "y"))
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(2), chart=chart)

    def coefficients(q):
        values = jnp.zeros((2, 2, 2), dtype=q.dtype)
        values = values.at[0, 0, 1].set(1.0)
        return values.at[0, 1, 1].set(q[0])

    connection = phx.metrix.CallableAffineConnection(
        coefficients,
        chart=chart,
    )
    point = jnp.array([0.2, -0.3])
    torsion = phx.metrix.torsion_tensor(connection, point)
    nonmetricity = phx.metrix.nonmetricity_tensor(connection, metric, point)
    curvature = phx.metrix.connection_riemann_tensor(connection, point)
    ricci = phx.metrix.connection_ricci_tensor(connection, point)

    assert jnp.allclose(torsion[0, 0, 1], 1.0)
    assert jnp.max(jnp.abs(nonmetricity)) > 0.0
    assert jnp.allclose(curvature[0, 1, 0, 1], 1.0)
    assert jnp.allclose(ricci[1, 1], 1.0)
    assert jnp.allclose(
        jax.jit(lambda q: phx.metrix.connection_ricci_tensor(connection, q))(point),
        ricci,
    )
