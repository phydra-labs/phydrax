from fractions import Fraction as F

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.geometry.brep._hermite_bridge import (
    build_hermite_bridge,
    build_hermite_bridge_certified,
)
from phydrax.geometry.brep._patches import BSplineSurfacePatch


def _patch(offset: float) -> BSplineSurfacePatch:
    points = jnp.array(
        [
            [[0.0, offset, 0.0], [0.0, offset + 1.0, 0.2]],
            [[1.0, offset, 0.1], [1.0, offset + 1.0, 0.3]],
        ]
    )
    return BSplineSurfacePatch(
        points,
        jnp.array([[1.0, 1.1], [1.2, 1.3]]),
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )


def _bridge(
    left: BSplineSurfacePatch, right: BSplineSurfacePatch, continuity: int = 1
) -> BSplineSurfacePatch:
    return build_hermite_bridge(
        left,
        right,
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(0), F(1)),
        left_collar=F(1, 2),
        right_collar=F(1, 2),
        continuity=continuity,
    )


@pytest.mark.parametrize("continuity", [1, 2])
def test_rational_endpoint_jets_and_exact_banks(continuity: int) -> None:
    left, right = _patch(0.0), _patch(2.0)
    bridge = _bridge(left, right, continuity)
    assert bridge.v_degree == 2 * continuity + 1
    assert bridge.exact_control_point_bank is not None
    assert all(w > 0 for row in bridge.exact_weights for w in row)
    for along in (0.2, 0.7):
        for source, endpoint, boundary in ((left, 0.0, 1.0), (right, 1.0, 0.0)):
            # v1 left outward and v0 right inward are both +source v.
            source_curve = lambda t: source.evaluate(
                jnp.array([along, boundary + t * 0.5])
            )
            bridge_curve = lambda t: bridge.evaluate(jnp.array([along, t]))
            for order in range(continuity + 1):
                np.testing.assert_allclose(
                    bridge_curve(endpoint), source_curve(0.0), rtol=1e-9, atol=1e-9
                )
                source_curve = jax.jacfwd(source_curve)
                bridge_curve = jax.jacfwd(bridge_curve)


def test_traced_coefficients_are_differentiable_without_exact_claim() -> None:
    left, right = _patch(0.0), _patch(2.0)

    def evaluate(points: Array) -> Array:
        changed = eqx.tree_at(lambda p: p.control_points, left, points)
        bridge = _bridge(changed, right)
        assert bridge.exact_control_point_bank is None
        return bridge.control_points

    value, tangent = jax.jvp(
        evaluate, (left.control_points,), (jnp.ones_like(left.control_points),)
    )
    assert value.shape == tangent.shape
    assert np.isfinite(np.asarray(tangent)).all()
    assert np.linalg.norm(np.asarray(tangent)) > 0


@pytest.mark.parametrize("transverse", [0.0, 0.43])
@pytest.mark.parametrize("continuity", [1, 2])
def test_bridge_evaluation_jvp_vjp_and_source_seam(
    transverse: float, continuity: int
) -> None:
    left, right = _patch(0.0), _patch(2.0)
    along = 0.37
    points, weights = left.control_points, left.weights
    point_direction = (
        jnp.arange(points.size, dtype=points.dtype).reshape(points.shape) / 17
    )
    weight_direction = jnp.array([[0.03, -0.02], [0.01, 0.04]])
    cotangent = jnp.array([0.7, -0.4, 0.2])

    def source(p: Array, w: Array) -> BSplineSurfacePatch:
        return eqx.tree_at(
            lambda patch: (patch.control_points, patch.weights), left, (p, w)
        )

    def evaluate(p: Array, w: Array) -> Array:
        bridge = _bridge(source(p, w), right, continuity)
        return bridge.evaluate(jnp.array([along, transverse]))

    _, tangent = jax.jvp(evaluate, (points, weights), (point_direction, weight_direction))
    _, pullback = jax.vjp(evaluate, points, weights)
    point_gradient, weight_gradient = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(point_direction, point_gradient)
        + jnp.vdot(weight_direction, weight_gradient),
        rtol=1e-9,
        atol=1e-9,
    )
    if transverse == 0.0:

        def source_evaluate(p: Array, w: Array) -> Array:
            return source(p, w).evaluate(jnp.array([along, 1.0]))

        _, source_tangent = jax.jvp(
            source_evaluate, (points, weights), (point_direction, weight_direction)
        )
        np.testing.assert_allclose(tangent, source_tangent, rtol=1e-9, atol=1e-9)


def test_mismatched_along_knots_are_rejected() -> None:
    left, right = _patch(0.0), _patch(2.0)
    with pytest.raises(ValueError, match="full along-domain"):
        build_hermite_bridge(
            left,
            right,
            left_side="v1",
            right_side="v0",
            left_along=(F(0), F(1)),
            right_along=(F(0), F(1, 2)),
            left_collar=F(1),
            right_collar=F(1),
        )


def test_exact_solve_evidence_is_bound_to_sources_and_result() -> None:
    left, right = _patch(0.0), _patch(2.0)
    bridge, evidence = build_hermite_bridge_certified(
        left,
        right,
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(0), F(1)),
        left_collar=F(1, 2),
        right_collar=F(1, 2),
        continuity=2,
    )
    assert evidence.left is left and evidence.right is right and evidence.bridge is bridge
    assert evidence.solve.rank == evidence.solve.variable_count == 6
