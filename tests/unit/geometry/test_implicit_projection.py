#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Callable
from typing import Any, final, Literal, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry._capabilities import GeometryCapability
from phydrax.geometry._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from phydrax.geometry._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from phydrax.geometry._validity import GeometryValidityEvidence, unrestricted_validity
from phydrax.geometry.design._schema import (
    DesignState,
    ParameterId,
    ParameterSchema,
    ParameterSpec,
)


@final
class _ParametricLevelKernel(GeometryKernel):
    """``|x|^2 - theta`` (sphere) or ``cos(theta) x0 + sin(theta) x1 - 1`` (plane)."""

    shape: Literal["sphere", "plane"] = eqx.field(static=True)

    def __init__(self, shape: Literal["sphere", "plane"], /) -> None:
        self.shape = shape

    def geometry_validity(self, state: DesignState, /) -> GeometryValidityEvidence:
        del state
        return unrestricted_validity(contract_id=f"parametric-{self.shape}")

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset({GeometryCapability.REGION_QUERY})

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.LEVEL_SET,
            regularity=FieldRegularity.SMOOTH,
            safe_step_factor=None,
            validity_region="all of R^3",
            parameter_differentiable=True,
            provenance=(f"parametric_{self.shape}_level_set",),
        )

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        theta = state.values[0]
        if self.shape == "sphere":
            return jnp.sum(points * points, axis=-1) - theta
        return jnp.cos(theta) * points[..., 0] + jnp.sin(theta) * points[..., 1] - 1.0

    def contains(self, state: DesignState, points: Array, /) -> Array:
        return self.boundary_field(state, points) <= 0.0

    def boundary_normal(self, state: DesignState, points: Array, /) -> NoReturn:
        del state, points
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)))

    def measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")

    def boundary_measure(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")

    def sample_interior(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")

    def sample_boundary(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")

    def boundary_atlas(self, state: DesignState, /) -> NoReturn:
        del state
        raise NotImplementedError("The oracle kernel exposes only its boundary field.")


def _oracle_plan(
    shape: Literal["sphere", "plane"], theta: float, anchors: np.ndarray, /
) -> tuple[phx.geometry.ImplicitPointProjectionPlan, Callable[[Any], DesignState]]:
    schema = ParameterSchema(
        (ParameterSpec(ParameterId("oracle", "theta"), (), "float64", "level-parameter"),)
    )
    state = DesignState(schema, [jnp.asarray(theta, dtype=jnp.float64)])
    plan = phx.geometry.ImplicitPointProjectionPlan(
        CompiledGeometry(_ParametricLevelKernel(shape), state),
        anchors,
        0.25,
        source_id=f"oracle-{shape}",
    )
    return plan, lambda value: state.replace_at(0, value)


def _nested(function: Callable[[Array], Any], modes: str, /) -> Callable[[Array], Any]:
    for mode in reversed(modes):
        function = jax.jacfwd(function) if mode == "f" else jax.jacrev(function)
    return function


# Every composition the consumers can request: nested forward (``jacfwd`` /
# ``jvp``), reverse-over-forward, forward-over-reverse and nested reverse.
_SECOND_ORDER_MODES = ("ff", "rf", "fr", "rr")
_THIRD_ORDER_MODES = ("fff", "rff", "frf")
# The oracles and their derivatives are O(1): a few float64 ulps of rounding.
_ROUNDING = 8.0 * float(np.finfo(np.float64).eps)


@pytest.mark.parametrize("theta", [1.44, 1.5])
def test_sphere_projection_follows_the_level_set_flow_to_third_order(
    theta: float,
) -> None:
    # F = |x|^2 - theta: the flow dx/dtheta = -F_theta grad F / |grad F|^2 =
    # x / (2 |x|^2) keeps x radial, so x(theta) = sqrt(theta) x_hat exactly and
    # every derivative is a closed-form power of theta along x_hat. The anchors
    # lie on the zero set at theta = 1.44; theta = 1.5 checks a projected primal.
    directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [-0.48, 0.6, -0.64]])
    plan, at = _oracle_plan("sphere", 1.44, 1.2 * directions)

    def points(value: Array) -> Array:
        return plan.realize(at(value)).points

    value = jnp.asarray(theta)
    root = np.sqrt(theta)
    oracles = {
        0: root * directions,
        1: directions / (2.0 * root),
        2: -directions / (4.0 * theta * root),
        3: 3.0 * directions / (8.0 * theta**2 * root),
    }
    assert bool(plan.realize(at(value)).accepted)
    np.testing.assert_allclose(points(value), oracles[0], rtol=0.0, atol=_ROUNDING)
    for modes in ("f", "r", *_SECOND_ORDER_MODES, *_THIRD_ORDER_MODES):
        np.testing.assert_allclose(
            _nested(points, modes)(value),
            oracles[len(modes)],
            rtol=0.0,
            atol=_ROUNDING,
            err_msg=modes,
        )


@pytest.mark.parametrize("theta", [0.0, 0.05])
def test_rotating_plane_projection_follows_the_level_set_flow_to_third_order(
    theta: float,
) -> None:
    # F = n . x - 1 with n = (cos theta, sin theta, 0) and t = (-sin theta,
    # cos theta, 0). Along the flow x = s n + q t + z e2 has F_theta = q, so
    # x' = -q n keeps s = 1 and gives q' = -1, z' = 0. Hence x' = -q n,
    # x'' = n - q t and x''' = 2 t + q n with q = t . x at the projected point.
    # The fixed-anchor Newton foot point differs from the flow away from
    # theta = 0, so the oracle is evaluated at the primal it returns.
    anchors = np.asarray([[1.0, 0.5, 0.3], [1.0, -0.25, 0.0], [1.0, 0.0, -0.7]])
    plan, at = _oracle_plan("plane", 0.0, anchors)

    def points(value: Array) -> Array:
        return plan.realize(at(value)).points

    def normals(value: Array) -> Array:
        return plan.realize(at(value)).normals

    value = jnp.asarray(theta)
    projected = np.asarray(points(value))
    normal = np.asarray([np.cos(theta), np.sin(theta), 0.0])
    tangent = np.asarray([-np.sin(theta), np.cos(theta), 0.0])
    q = (projected @ tangent)[:, None]
    oracles = {
        1: -q * normal,
        2: normal - q * tangent,
        3: 2.0 * tangent + q * normal,
    }
    assert bool(plan.realize(at(value)).accepted)
    np.testing.assert_allclose(projected @ normal, 1.0, rtol=0.0, atol=_ROUNDING)
    for modes in ("f", "r", *_SECOND_ORDER_MODES, *_THIRD_ORDER_MODES):
        np.testing.assert_allclose(
            _nested(points, modes)(value),
            oracles[len(modes)],
            rtol=0.0,
            atol=_ROUNDING,
            err_msg=modes,
        )
    # The unit normal n(theta) is independent of the point: n'' = -n.
    for modes in _SECOND_ORDER_MODES:
        np.testing.assert_allclose(
            _nested(normals, modes)(value),
            np.broadcast_to(-normal, projected.shape),
            rtol=0.0,
            atol=_ROUNDING,
            err_msg=modes,
        )


def test_normal_gauge_projection_has_expected_sphere_radius_derivative() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    anchors = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        anchors,
        0.25,
        source_id="sphere-anchors",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def coordinates(radius: Any) -> Any:
        state = geometry.state.replace_at(radius_index, radius)
        return plan.realize(state).proposed_points

    points, tangent = jax.jvp(
        coordinates,
        (jnp.asarray(1.1),),
        (jnp.asarray(1.0),),
    )
    result = eqx.filter_jit(plan.realize)(geometry.state.replace_at(radius_index, 1.1))
    assert jnp.allclose(points, 1.1 * anchors, atol=1.0e-7)
    assert jnp.allclose(tangent, anchors, atol=1.0e-6)
    assert bool(result.accepted)
    assert result.evidence.root_residual <= plan.policy.root_tolerance


def test_projection_rejects_trust_region_expiry_with_finite_fallback() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        1.0,
        feature_id="sphere",
    ).compile()
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    state = geometry.state.replace_at(radius_index, jnp.asarray(1.5))

    result = eqx.filter_jit(plan.realize)(state)

    assert not bool(result.accepted)
    assert bool(result.refresh_required)
    assert jnp.all(jnp.isfinite(result.proposed_points))
    assert jnp.array_equal(result.points, plan.anchors)
    tangent = jax.jvp(
        lambda radius: (
            plan.realize(geometry.state.replace_at(radius_index, radius)).points
        ),
        (jnp.asarray(1.5),),
        (jnp.asarray(1.0),),
    )[1]
    reverse = jax.grad(
        lambda radius: jnp.sum(
            plan.realize(geometry.state.replace_at(radius_index, radius)).points
        )
    )(jnp.asarray(1.5))
    assert jnp.all(jnp.isnan(tangent))
    assert jnp.isnan(reverse)

    def realized(radius: Any) -> tuple[Any, Any, Any]:
        result = plan.realize(geometry.state.replace_at(radius_index, radius))
        return result.proposed_points, result.points, result.normals

    for modes in (*_SECOND_ORDER_MODES, *_THIRD_ORDER_MODES):
        for leaf in jax.tree.leaves(_nested(realized, modes)(jnp.asarray(1.5))):
            assert bool(jnp.all(jnp.isnan(leaf))), modes
