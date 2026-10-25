#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.metrix._flat_torus import FlatTorusManifold
from phydrax.transport.continuous import ManifoldTransportGeometry


# Rows a, b, c; b - a is the shortest lattice vector (length ~0.51), so the
# componentwise fractional minimum image is not the physical shortest image.
SKEW = np.asarray([[1.0, 0.0, 0.0], [0.9, 0.5, 0.0], [0.3, 0.4, 1.1]])


def _image_lengths(delta: np.ndarray, cell: np.ndarray, extent: int = 4) -> list:
    """Brute-force (length, fractional image) pairs of one fractional displacement."""
    records = []
    for shift in itertools.product(range(-extent, extent + 1), repeat=3):
        image = delta + np.asarray(shift, dtype=np.float64)
        cartesian = image @ cell
        records.append((float(np.sqrt(cartesian @ cartesian)), image))
    records.sort(key=lambda item: item[0])
    return records


def test_flat_torus_manifold_exp_log_round_trip() -> None:
    rng = np.random.default_rng(11)
    point = jnp.asarray(rng.uniform(0.0, 1.0, (4, 3)))
    tangent = jnp.asarray(rng.uniform(-0.4, 0.4, (4, 3)))
    fractional = FlatTorusManifold((4, 3))
    destination = fractional.exp(point, tangent)

    assert fractional.metric_kind == "fractional-euclidean"
    assert bool(fractional.contains(point)) and bool(fractional.contains(destination))
    np.testing.assert_allclose(fractional.log(point, destination), tangent, atol=1e-12)
    lifted = fractional.exp(point, tangent + 1.0)
    np.testing.assert_allclose(fractional.log(destination, lifted), 0.0, atol=1e-12)
    np.testing.assert_allclose(fractional.retract(point, tangent), destination)
    np.testing.assert_allclose(
        fractional.squared_distance(point, destination), jnp.sum(tangent**2), rtol=1e-12
    )
    assert float(fractional.constraint_residual(point)) == 0.0
    assert not bool(fractional.contains(point + 1.0))
    np.testing.assert_allclose(
        fractional.constraint_residual(point + 1.5), np.max(point + 0.5), rtol=1e-12
    )
    wrapped = np.asarray(FlatTorusManifold((3,)).wrap(jnp.asarray([-1e-17, 1.0, 2.25])))
    np.testing.assert_allclose(wrapped, [0.0, 0.0, 0.25], atol=1e-12)
    assert np.all((wrapped >= 0.0) & (wrapped < 1.0))

    physical = FlatTorusManifold((4, 3), cell=PeriodicCell(SKEW))
    small = jnp.asarray(rng.uniform(-0.05, 0.05, (4, 3)))
    np.testing.assert_allclose(
        physical.log(point, physical.exp(point, small)), small, atol=1e-12
    )
    assert physical.metric_kind == "physical-cell"
    assert physical.manifold_id != fractional.manifold_id
    np.testing.assert_array_equal(
        physical.transport(point, small, destination, tangent), tangent
    )


def test_skew_cell_physical_log_differs_from_fractional_minimum_image() -> None:
    point = jnp.asarray([[0.05, 0.10, 0.20]])
    destination = jnp.asarray([[0.45, 0.50, 0.20]])
    fractional = FlatTorusManifold((1, 3))
    physical = FlatTorusManifold((1, 3), cell=PeriodicCell(SKEW))
    raw = np.asarray(destination - point)[0]
    oracle = _image_lengths(raw, SKEW)
    best_length, best_image = oracle[0]

    componentwise = np.asarray(fractional.log(point, destination))[0]
    shortest = np.asarray(physical.log(point, destination))[0]

    np.testing.assert_allclose(componentwise, [0.4, 0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(shortest, best_image, atol=1e-12)
    assert not np.allclose(shortest, componentwise)
    difference = shortest - componentwise
    np.testing.assert_allclose(difference, np.round(difference), atol=1e-12)
    componentwise_length = np.linalg.norm(componentwise @ SKEW)
    assert best_length < componentwise_length - 0.1
    np.testing.assert_allclose(
        physical.squared_distance(point, destination), best_length**2, rtol=1e-12
    )
    evidence = physical.log_with_evidence(point, destination)
    np.testing.assert_allclose(evidence.distance[0], best_length, rtol=1e-12)
    np.testing.assert_allclose(evidence.second_distance[0], oracle[1][0], rtol=1e-12)
    np.testing.assert_allclose(evidence.margin[0], oracle[1][0] - best_length, rtol=1e-10)
    assert bool(evidence.unique[0])


def test_half_period_points_report_and_refuse_branch_ambiguity() -> None:
    point = jnp.zeros((1, 3))
    fractional = FlatTorusManifold((1, 3))
    half = jnp.asarray([[0.5, 0.1, 0.2]])
    evidence = fractional.log_with_evidence(point, half)

    assert not bool(evidence.unique[0])
    np.testing.assert_allclose(evidence.margin[0], 0.0, atol=1e-12)
    with pytest.raises(eqx.EquinoxRuntimeError, match="cut locus"):
        fractional.log(point, half)
    near = jnp.asarray([[0.45, 0.1, 0.2]])
    near_evidence = fractional.log_with_evidence(point, near)
    oracle = _image_lengths(np.asarray(near)[0], np.eye(3))
    assert bool(near_evidence.unique[0])
    np.testing.assert_allclose(
        near_evidence.margin[0], oracle[1][0] - oracle[0][0], rtol=1e-10
    )
    tolerant = FlatTorusManifold((1, 3), branch_tolerance=0.2)
    assert not bool(tolerant.log_with_evidence(point, near).unique[0])

    # a/2 and -a/2 are the two shortest images in the skew cell.
    physical = FlatTorusManifold((1, 3), cell=PeriodicCell(SKEW))
    physical_half = physical.log_with_evidence(point, jnp.asarray([[0.5, 0.0, 0.0]]))
    oracle = _image_lengths(np.asarray([0.5, 0.0, 0.0]), SKEW)
    np.testing.assert_allclose(oracle[0][0], oracle[1][0], rtol=1e-12)
    assert not bool(physical_half.unique[0])
    np.testing.assert_allclose(physical_half.margin[0], 0.0, atol=1e-12)
    with pytest.raises(eqx.EquinoxRuntimeError, match="cut locus"):
        physical.log(point, jnp.asarray([[0.5, 0.0, 0.0]]))
    np.testing.assert_allclose(
        physical.squared_distance(point, jnp.asarray([[0.5, 0.0, 0.0]])),
        oracle[0][0] ** 2,
        rtol=1e-12,
    )


def test_physical_metric_inner_and_gradient_consistency() -> None:
    rng = np.random.default_rng(5)
    manifold = FlatTorusManifold((2, 3), cell=PeriodicCell(SKEW))
    point = jnp.asarray(rng.uniform(0.0, 1.0, (2, 3)))
    left = jnp.asarray(rng.normal(size=(2, 3)))
    right = jnp.asarray(rng.normal(size=(2, 3)))
    expected = np.sum((np.asarray(left) @ SKEW) * (np.asarray(right) @ SKEW))
    np.testing.assert_allclose(manifold.inner(point, left, right), expected, rtol=1e-12)
    np.testing.assert_allclose(
        manifold.norm(point, left) ** 2,
        np.sum((np.asarray(left) @ SKEW) ** 2),
        rtol=1e-12,
    )

    target = jnp.asarray(rng.normal(size=(2, 3)))

    def objective(value: jax.Array) -> jax.Array:
        cartesian = value @ jnp.asarray(SKEW)
        return jnp.sum(jnp.sin(cartesian) * target)

    cotangent = jax.grad(objective)(point)
    gradient = manifold.egrad_to_rgrad(point, cotangent)
    for direction in (left, right):
        np.testing.assert_allclose(
            manifold.inner(point, gradient, direction),
            jnp.sum(cotangent * direction),
            rtol=1e-10,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        manifold.metric @ manifold.inverse_metric, np.eye(3), atol=1e-12
    )
    euclidean = FlatTorusManifold((2, 3))
    np.testing.assert_array_equal(euclidean.egrad_to_rgrad(point, cotangent), cotangent)


def test_flat_torus_manifold_drives_manifold_transport_geometry() -> None:
    manifold = FlatTorusManifold((2, 3), cell=PeriodicCell(SKEW))
    geometry = ManifoldTransportGeometry(manifold)
    source = jnp.asarray([[0.05, 0.10, 0.20], [0.7, 0.2, 0.9]])
    target = jnp.asarray([[0.12, 0.05, 0.25], [0.68, 0.26, 0.95]])
    evaluation = geometry.interpolant.evaluate(jnp.asarray(0.5), source, target)
    expected = manifold.exp(source, 0.5 * manifold.log(source, target))

    assert manifold.manifold_id in geometry.geometry_id
    np.testing.assert_allclose(evaluation.state, expected, atol=1e-12)
    np.testing.assert_allclose(
        evaluation.conditional_velocity, manifold.log(source, target), atol=1e-10
    )
    np.testing.assert_allclose(
        geometry.state_geometry.inverse_retract(source, target),
        manifold.log(source, target),
        atol=1e-12,
    )
