#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ._cone_root import safeguarded_newton_bisection, SafeguardedRootResult
from ._cones import AbstractConvexCone


def _geometric_mean(x: Array, y: Array, exponent: float, /) -> Array:
    positive = (x > 0.0) & (y > 0.0)
    safe_x = jnp.where(positive, x, 1.0)
    safe_y = jnp.where(positive, y, 1.0)
    value = jnp.exp(exponent * jnp.log(safe_x) + (1.0 - exponent) * jnp.log(safe_y))
    return jnp.where(positive, value, jnp.where((x == 0.0) | (y == 0.0), 0.0, jnp.nan))


def _primal_slack(value: Array, exponent: float, /) -> Array:
    x, y, z = value
    geometric = _geometric_mean(x, y, exponent)
    boundary = geometric - jnp.abs(z)
    return jnp.minimum(jnp.minimum(x, y), boundary)


def _dual_slack(value: Array, exponent: float, /) -> Array:
    u, v, w = value
    complement = 1.0 - exponent
    scaled_u = u / exponent
    scaled_v = v / complement
    geometric = _geometric_mean(scaled_u, scaled_v, exponent)
    boundary = geometric - jnp.abs(w)
    return jnp.minimum(jnp.minimum(u, v), boundary)


def _in_primal(value: Array, exponent: float, /) -> Array:
    return _primal_slack(value, exponent) >= 0.0


def _in_dual(value: Array, exponent: float, /) -> Array:
    return _dual_slack(value, exponent) >= 0.0


def _positive_quadratic_root(
    source: Array,
    weight: float,
    root: Array,
    gap: Array,
    /,
) -> Array:
    product = weight * root * gap
    radical = jnp.sqrt(jnp.maximum(source * source + 4.0 * product, 0.0))
    direct = 0.5 * (source + radical)
    stable = (
        2.0
        * product
        / jnp.maximum(
            radical - source,
            jnp.finfo(source.dtype).tiny,
        )
    )
    return jnp.where(source >= 0.0, direct, stable)


def _root_coordinates(
    value: Array,
    transformed_root: Array,
    exponent: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    absolute_z = jnp.abs(value[2])
    root = absolute_z * jax.nn.sigmoid(transformed_root)
    gap = absolute_z * jax.nn.sigmoid(-transformed_root)
    projected_x = _positive_quadratic_root(
        value[0],
        exponent,
        root,
        gap,
    )
    projected_y = _positive_quadratic_root(
        value[1],
        1.0 - exponent,
        root,
        gap,
    )
    return root, gap, projected_x, projected_y


def _root_log_value(value: Array, transformed_root: Array, exponent: float, /) -> Array:
    root, _, projected_x, projected_y = _root_coordinates(
        value,
        transformed_root,
        exponent,
    )
    smallest = jnp.nextafter(
        jnp.asarray(0.0, dtype=value.dtype),
        jnp.asarray(1.0, dtype=value.dtype),
    )
    return (
        exponent * jnp.log(jnp.maximum(projected_x, smallest))
        + (1.0 - exponent) * jnp.log(jnp.maximum(projected_y, smallest))
        - jnp.log(jnp.maximum(root, smallest))
    )


def _root_function(
    value: Array,
    transformed_root: Array,
    exponent: float,
    /,
) -> tuple[Array, Array]:
    function = _root_log_value(value, transformed_root, exponent)
    derivative = jax.grad(lambda current: _root_log_value(value, current, exponent))(
        transformed_root
    )
    return function, derivative


def _power_root_projection(
    value: Array,
    exponent: float,
    /,
) -> tuple[Array, SafeguardedRootResult]:
    precision = jnp.finfo(value.dtype)
    minimum_weight = jnp.minimum(exponent, 1.0 - exponent)
    absolute_z = jnp.maximum(jnp.abs(value[2]), precision.tiny)
    scale_limit = (
        jnp.log(jnp.asarray(minimum_weight, dtype=value.dtype))
        + 2.0 * jnp.log(absolute_z)
        - jnp.log(jnp.asarray(precision.tiny, dtype=value.dtype))
        - 2.0
    )
    maximum_limit = -jnp.log(jnp.asarray(precision.tiny, dtype=value.dtype)) - 2.0
    limit = jnp.clip(scale_limit, 8.0, maximum_limit)
    multiplier = 512.0 if precision.bits <= 32 else 4096.0
    epsilon = multiplier * precision.eps
    result = safeguarded_newton_bisection(
        lambda transformed: _root_function(value, transformed, exponent),
        -limit,
        limit,
        absolute_tolerance=jnp.asarray(epsilon, dtype=value.dtype),
        relative_tolerance=jnp.asarray(0.0, dtype=value.dtype),
        maximum_steps=96,
    )
    root, gap, projected_x, projected_y = _root_coordinates(
        value,
        result.root,
        exponent,
    )
    geometric = _geometric_mean(projected_x, projected_y, exponent)
    projected = jnp.asarray([projected_x, projected_y, jnp.sign(value[2]) * geometric])
    boundary_residual = jnp.abs(geometric - root)
    valid = (
        result.converged
        & jnp.all(jnp.isfinite(projected))
        & (root > 0.0)
        & (gap > 0.0)
        & (boundary_residual <= epsilon * jnp.maximum(jnp.abs(value[2]), 1.0))
    )
    certificate = SafeguardedRootResult(
        root,
        jnp.maximum(result.residual, boundary_residual),
        result.bracket_width,
        result.iterations,
        result.finite & jnp.all(jnp.isfinite(projected)),
        valid,
    )
    return jnp.where(valid, projected, jnp.full_like(value, jnp.nan)), certificate


def _projection_value(value: Array, exponent: float, /) -> Array:
    in_primal = _in_primal(value, exponent)
    in_polar = _in_dual(-value, exponent)
    zero_tail = value[2] == 0.0

    def general(_):
        projected, _ = _power_root_projection(value, exponent)
        return projected

    return jax.lax.cond(
        in_primal,
        lambda _: value,
        lambda _: jax.lax.cond(
            in_polar,
            lambda __: jnp.zeros_like(value),
            lambda __: jax.lax.cond(
                zero_tail,
                lambda ___: jnp.asarray(
                    [jnp.maximum(value[0], 0.0), jnp.maximum(value[1], 0.0), 0.0]
                ),
                general,
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )


def _boundary_kkt_matrix(
    value: Array,
    projected: Array,
    exponent: float,
    /,
) -> Array:
    x, y, z = projected
    complement = 1.0 - exponent
    geometric = _geometric_mean(x, y, exponent)
    safe_x = jnp.maximum(x, jnp.finfo(value.dtype).tiny)
    safe_y = jnp.maximum(y, jnp.finfo(value.dtype).tiny)
    gradient = jnp.asarray(
        [
            exponent * geometric / safe_x,
            complement * geometric / safe_y,
            -jnp.sign(z),
        ],
        dtype=value.dtype,
    )
    hessian = jnp.zeros((3, 3), dtype=value.dtype)
    hessian = hessian.at[0, 0].set(
        exponent * (exponent - 1.0) * geometric / (safe_x * safe_x)
    )
    hessian = hessian.at[1, 1].set(
        complement * (complement - 1.0) * geometric / (safe_y * safe_y)
    )
    cross = exponent * complement * geometric / (safe_x * safe_y)
    hessian = hessian.at[0, 1].set(cross)
    hessian = hessian.at[1, 0].set(cross)
    difference = value - projected
    multiplier = jnp.vdot(difference, gradient) / jnp.maximum(
        jnp.vdot(gradient, gradient), jnp.finfo(value.dtype).tiny
    )
    matrix = jnp.zeros((4, 4), dtype=value.dtype)
    matrix = matrix.at[:3, :3].set(jnp.eye(3, dtype=value.dtype) + multiplier * hessian)
    matrix = matrix.at[:3, 3].set(gradient)
    matrix = matrix.at[3, :3].set(gradient)
    return matrix


def _kkt_solve(matrix: Array, right: Array, /) -> Array:
    return solve(
        LinearSystem(
            DenseLinearOperator(matrix),
            problem_id="power-cone-projection-kkt",
        ),
        right,
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("status"),
        ),
    ).value


def _minimum_singular_value(matrix: Array, /) -> Array:
    decomposition = factorize(
        DenseLinearOperator(matrix),
        FactorizationPolicy("svd"),
    )
    return decomposition.singular_values()[-1]


def _projection_jvp(
    value: Array,
    projected: Array,
    tangent: Array,
    exponent: float,
    /,
) -> Array:
    in_primal = _in_primal(value, exponent)
    in_polar = _in_dual(-value, exponent)
    zero_tail = value[2] == 0.0

    def general(_):
        matrix = _boundary_kkt_matrix(value, projected, exponent)
        right = jnp.concatenate((tangent, jnp.zeros(1, dtype=value.dtype)))
        return _kkt_solve(matrix, right)[:3]

    face_action = jnp.asarray(
        [
            jnp.where(value[0] > 0.0, tangent[0], 0.0),
            jnp.where(value[1] > 0.0, tangent[1], 0.0),
            0.0,
        ]
    )
    return jax.lax.cond(
        in_primal,
        lambda _: tangent,
        lambda _: jax.lax.cond(
            in_polar,
            lambda __: jnp.zeros_like(tangent),
            lambda __: jax.lax.cond(
                zero_tail,
                lambda ___: face_action,
                general,
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )


def _homogeneous_scale(value: Array, /) -> Array:
    scale = jnp.max(jnp.abs(value))
    return jnp.where(scale > 0.0, scale, 1.0)


def _working_value(value: Array, /) -> Array:
    working_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    return value.astype(jnp.result_type(value.dtype, working_dtype))


@jax.custom_jvp
def _project_power_single(value: Array, exponent: float, /) -> Array:
    working = _working_value(value)

    def general(_: None) -> Array:
        scale = _homogeneous_scale(working)
        projected = scale * _projection_value(working / scale, exponent)
        return projected.astype(value.dtype)

    return jax.lax.cond(
        _in_primal(working, exponent),
        lambda _: value,
        lambda _: jax.lax.cond(
            _in_dual(-working, exponent),
            lambda __: jnp.zeros_like(value),
            general,
            operand=None,
        ),
        operand=None,
    )


@_project_power_single.defjvp
def _project_power_single_jvp(primals, tangents):
    value, exponent = primals
    tangent, _ = tangents
    working = _working_value(value)
    working_tangent = tangent.astype(working.dtype)

    def general(_: None) -> tuple[Array, Array]:
        scale = _homogeneous_scale(working)
        normalized = working / scale
        projected = _projection_value(normalized, exponent)
        derivative = _projection_jvp(
            normalized,
            projected,
            working_tangent,
            exponent,
        )
        return (
            (scale * projected).astype(value.dtype),
            derivative.astype(tangent.dtype),
        )

    return jax.lax.cond(
        _in_primal(working, exponent),
        lambda _: (value, tangent),
        lambda _: jax.lax.cond(
            _in_dual(-working, exponent),
            lambda __: (jnp.zeros_like(value), jnp.zeros_like(tangent)),
            general,
            operand=None,
        ),
        operand=None,
    )


def _project_power_dual_single(value: Array, exponent: float, /) -> Array:
    working = _working_value(value)
    scale = _homogeneous_scale(working)
    normalized = working / scale
    candidate = normalized + _project_power_single(-normalized, exponent)
    positive = jnp.maximum(candidate[:2], 0.0)
    boundary = _geometric_mean(
        positive[0] / exponent,
        positive[1] / (1.0 - exponent),
        exponent,
    )
    candidate = candidate.at[:2].set(positive)
    candidate = candidate.at[2].set(jnp.clip(candidate[2], -boundary, boundary))
    return (scale * candidate).astype(value.dtype)


def _smoothness_margin(value: Array, exponent: float, /) -> Array:
    working = _working_value(value)
    scale = _homogeneous_scale(working)
    normalized = working / scale
    primal_slack = _primal_slack(working, exponent)
    polar_slack = _dual_slack(-working, exponent)
    strict_primal = primal_slack > 0.0
    strict_polar = polar_slack > 0.0
    projected, root = _power_root_projection(normalized, exponent)
    kkt = _boundary_kkt_matrix(normalized, projected, exponent)
    minimum_singular = _minimum_singular_value(kkt)
    general_margin = scale * jnp.minimum(
        jnp.minimum(jnp.minimum(projected[0], projected[1]), jnp.abs(projected[2])),
        jnp.where(root.converged, minimum_singular, 0.0),
    )
    margin = jnp.where(
        strict_primal,
        primal_slack,
        jnp.where(strict_polar, polar_slack, general_margin),
    )
    closed_boundary = (_in_primal(working, exponent) & ~strict_primal) | (
        _in_dual(-working, exponent) & ~strict_polar
    )
    margin = jnp.where(
        closed_boundary | (working[2] == 0.0) | ~jnp.isfinite(margin),
        0.0,
        margin,
    )
    exponent_margin = scale * jnp.minimum(exponent, 1.0 - exponent)
    return jnp.maximum(jnp.minimum(margin, exponent_margin), 0.0).astype(value.dtype)


class PowerCone(AbstractConvexCone):
    """Standard three-dimensional power cone with a static exponent."""

    exponent: float = eqx.field(static=True)

    def __init__(self, exponent: float, /):
        if isinstance(exponent, bool) or not isinstance(exponent, Real):
            raise TypeError("PowerCone exponent must be a real scalar.")
        value = float(exponent)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "PowerCone exponent must be finite and strictly between zero and one."
            )
        self.exponent = value
        self.dimension = 3
        self.cone_id = canonical_fingerprint({"kind": "power-cone", "exponent": value})

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        projected = jax.vmap(lambda item: _project_power_single(item, self.exponent))(
            flat
        )
        return projected.reshape(array.shape)

    def project_dual(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        projected = jax.vmap(
            lambda item: _project_power_dual_single(item, self.exponent)
        )(flat)
        return projected.reshape(array.shape)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        margin = jax.vmap(
            lambda item: _primal_slack(_working_value(item), self.exponent)
        )(flat)
        return margin.astype(array.dtype).reshape(array.shape[:-1])

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        margin = jax.vmap(lambda item: _smoothness_margin(-item, self.exponent))(flat)
        return margin.reshape(array.shape[:-1])


__all__ = ["PowerCone"]
