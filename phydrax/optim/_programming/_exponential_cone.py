#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

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


def _exp_limit(dtype) -> Array:
    return 0.25 * jnp.log(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype))


def _log_scaled_exponential(scale: Array, exponent: Array, /) -> Array:
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    return jnp.log(safe_scale) + exponent


def _scaled_exponential(scale: Array, exponent: Array, /) -> Array:
    logarithm = _log_scaled_exponential(scale, exponent)
    maximum_logarithm = jnp.log(
        jnp.asarray(jnp.finfo(logarithm.dtype).max, dtype=logarithm.dtype)
    )
    finite_value = jnp.exp(jnp.minimum(logarithm, maximum_logarithm))
    represented = jnp.where(logarithm <= maximum_logarithm, finite_value, jnp.inf)
    represented = jnp.where(jnp.isnan(logarithm), jnp.nan, represented)
    return jnp.where(scale > 0.0, represented, 0.0)


def _primal_slack(value: Array, /) -> Array:
    x, y, z = value
    safe_y = jnp.where(y > 0.0, y, 1.0)
    boundary = z - _scaled_exponential(y, x / safe_y)
    return jnp.where(y > 0.0, jnp.minimum(y, boundary), y)


def _dual_slack(value: Array, /) -> Array:
    u, v, w = value
    safe_u = jnp.where(u < 0.0, u, -1.0)
    boundary = jnp.e * w - _scaled_exponential(-u, v / safe_u)
    return jnp.where(u < 0.0, jnp.minimum(-u, boundary), -jnp.abs(u))


def _in_primal(value: Array, /) -> Array:
    x, y, z = value
    safe_y = jnp.where(y > 0.0, y, 1.0)
    safe_z = jnp.where(z > 0.0, z, 1.0)
    smooth = (
        (y > 0.0)
        & (z > 0.0)
        & (_log_scaled_exponential(y, x / safe_y) <= jnp.log(safe_z))
    )
    face = (y == 0.0) & (x <= 0.0) & (z >= 0.0)
    return smooth | face


def _in_dual(value: Array, /) -> Array:
    u, v, w = value
    safe_u = jnp.where(u < 0.0, u, -1.0)
    safe_w = jnp.where(w > 0.0, w, 1.0)
    smooth = (
        (u < 0.0)
        & (w > 0.0)
        & (_log_scaled_exponential(-u, v / safe_u) <= 1.0 + jnp.log(safe_w))
    )
    face = (u == 0.0) & (v >= 0.0) & (w >= 0.0)
    return smooth | face


def _primal_candidate(value: Array, /) -> tuple[Array, Array]:
    x, y, z = value
    face = jnp.asarray([jnp.minimum(x, 0.0), 0.0, jnp.maximum(z, 0.0)])
    face_distance = jnp.linalg.norm(face - value)

    def smooth_candidate(_):
        boundary = _scaled_exponential(y, x / y)
        candidate = jnp.asarray([x, y, jnp.maximum(z, boundary)])
        distance = jnp.linalg.norm(candidate - value)
        use = distance < face_distance
        return jnp.where(use, candidate, face), jnp.where(use, distance, face_distance)

    return jax.lax.cond(
        y > 0.0,
        smooth_candidate,
        lambda _: (face, face_distance),
        operand=None,
    )


def _polar_candidate(value: Array, /) -> tuple[Array, Array]:
    x, y, z = value
    face = jnp.asarray([0.0, jnp.minimum(y, 0.0), jnp.minimum(z, 0.0)])
    face_distance = jnp.linalg.norm(face - value)

    def smooth_candidate(_):
        boundary = -_scaled_exponential(x, y / x - 1.0)
        candidate = jnp.asarray([x, y, jnp.minimum(z, boundary)])
        distance = jnp.linalg.norm(candidate - value)
        use = distance < face_distance
        return jnp.where(use, candidate, face), jnp.where(use, distance, face_distance)

    return jax.lax.cond(
        x > 0.0,
        smooth_candidate,
        lambda _: (face, face_distance),
        operand=None,
    )


def _h_and_derivative(value: Array, rho: Array, /) -> tuple[Array, Array]:
    x, y, z = value
    exp_rho = jnp.exp(rho)
    exp_negative = jnp.exp(-rho)
    function = (
        ((rho - 1.0) * x + y) * exp_rho
        - (x - rho * y) * exp_negative
        - (rho * (rho - 1.0) + 1.0) * z
    )
    derivative = (
        (rho * x + y) * exp_rho
        + (x - (rho - 1.0) * y) * exp_negative
        - (2.0 * rho - 1.0) * z
    )
    return function, derivative


def _pomega(rho: Array, /) -> Array:
    denominator = rho * (rho - 1.0) + 1.0
    value = jnp.exp(rho) / denominator
    return jnp.where(
        rho < 2.0,
        jnp.minimum(value, jnp.exp(jnp.asarray(2.0, dtype=rho.dtype)) / 3.0),
        value,
    )


def _domega(rho: Array, /) -> Array:
    denominator = rho * (rho - 1.0) + 1.0
    value = -jnp.exp(-rho) / denominator
    return jnp.where(
        rho > -1.0,
        jnp.maximum(value, -jnp.exp(jnp.asarray(1.0, dtype=rho.dtype)) / 3.0),
        value,
    )


def _ppsi(value: Array, /) -> Array:
    x, y, _ = value
    radical = jnp.sqrt(jnp.maximum(x * x + y * y - x * y, 0.0))
    safe_x = jnp.where(x != 0.0, x, 1.0)
    denominator = x - y - radical
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    psi_first = (x - y + radical) / safe_x
    psi_second = -y / safe_denominator
    psi = jnp.where(x > y, psi_first, psi_second)
    divisor = psi * (psi - 1.0) + 1.0
    return ((psi - 1.0) * x + y) / divisor


def _dpsi(value: Array, /) -> Array:
    x, y, _ = value
    radical = jnp.sqrt(jnp.maximum(x * x + y * y - x * y, 0.0))
    safe_y = jnp.where(y != 0.0, y, 1.0)
    denominator = x + radical
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0)
    psi = jnp.where(y > x, (x - radical) / safe_y, (x - y) / safe_denominator)
    divisor = psi * (psi - 1.0) + 1.0
    return (x - psi * y) / divisor


def _root_bracket(
    value: Array,
    primal_distance: Array,
    polar_distance: Array,
    /,
) -> tuple[Array, Array]:
    x, y, z = value
    limit = _exp_limit(value.dtype)
    epsilon = jnp.sqrt(jnp.finfo(value.dtype).eps)
    lower = -limit
    upper = limit
    negative_y = jnp.minimum(y, 0.0)
    primal_radius = jnp.sqrt(
        jnp.maximum(primal_distance * primal_distance - negative_y * negative_y, 0.0)
    )
    negative_x = jnp.minimum(x, 0.0)
    polar_radius = jnp.sqrt(
        jnp.maximum(polar_distance * polar_distance - negative_x * negative_x, 0.0)
    )

    def positive_z(bounds):
        lo, hi = bounds
        ratio = z / jnp.maximum(_ppsi(value), jnp.finfo(value.dtype).tiny)
        return jnp.maximum(
            lo, jnp.log(jnp.maximum(ratio, jnp.finfo(value.dtype).tiny))
        ), hi

    def negative_z(bounds):
        lo, hi = bounds
        ratio = -z / jnp.maximum(_dpsi(value), jnp.finfo(value.dtype).tiny)
        return lo, jnp.minimum(
            hi, -jnp.log(jnp.maximum(ratio, jnp.finfo(value.dtype).tiny))
        )

    lower, upper = jax.lax.cond(
        z > 0.0,
        positive_z,
        lambda bounds: jax.lax.cond(z < 0.0, negative_z, lambda item: item, bounds),
        (lower, upper),
    )

    def positive_x(bounds):
        lo, hi = bounds
        base = 1.0 - y / x
        lo = jnp.maximum(lo, base)
        tpu = jnp.maximum(epsilon, jnp.minimum(polar_radius, primal_radius + z))
        candidate = jnp.maximum(lo, base + tpu / x / _pomega(lo))
        return lo, jnp.minimum(hi, candidate)

    lower, upper = jax.lax.cond(x > 0.0, positive_x, lambda item: item, (lower, upper))

    def positive_y(bounds):
        lo, hi = bounds
        base = x / y
        hi = jnp.minimum(hi, base)
        tdl = -jnp.maximum(epsilon, jnp.minimum(primal_radius, polar_radius - z))
        candidate = jnp.minimum(hi, base - tdl / y / _domega(hi))
        return jnp.maximum(lo, candidate), hi

    lower, upper = jax.lax.cond(y > 0.0, positive_y, lambda item: item, (lower, upper))
    return (
        jnp.clip(lower, -limit, limit),
        jnp.clip(upper, -limit, limit),
    )


def _boundary_projection(value: Array, rho: Array, /) -> tuple[Array, Array]:
    x, y, _ = value
    linear = (rho - 1.0) * x + y
    denominator = rho * (rho - 1.0) + 1.0
    scale = linear / denominator
    candidate = jnp.asarray([rho * scale, scale, _scaled_exponential(scale, rho)])
    valid = (linear > 0.0) & jnp.all(jnp.isfinite(candidate))
    return candidate, valid


def _exp_root_projection(value: Array, /) -> tuple[Array, SafeguardedRootResult]:
    _, primal_distance = _primal_candidate(value)
    _, polar_distance = _polar_candidate(value)
    lower, upper = _root_bracket(value, primal_distance, polar_distance)
    epsilon = 64.0 * jnp.finfo(value.dtype).eps
    result = safeguarded_newton_bisection(
        lambda rho: _h_and_derivative(value, rho),
        lower,
        upper,
        absolute_tolerance=jnp.asarray(epsilon, dtype=value.dtype),
        relative_tolerance=jnp.asarray(epsilon, dtype=value.dtype),
        maximum_steps=80,
    )
    candidate, candidate_valid = _boundary_projection(value, result.root)
    safe_y = jnp.where(candidate[1] > 0.0, candidate[1], 1.0)
    snapped_boundary = _scaled_exponential(candidate[1], candidate[0] / safe_y)
    candidate = candidate.at[2].set(snapped_boundary)
    valid = result.converged & candidate_valid
    certificate = SafeguardedRootResult(
        result.root,
        result.residual,
        result.bracket_width,
        result.iterations,
        result.finite & candidate_valid,
        valid,
    )
    return jnp.where(valid, candidate, jnp.full_like(value, jnp.nan)), certificate


def _projection_value(value: Array, /) -> Array:
    in_primal = _in_primal(value)
    in_polar = _in_dual(-value)
    face = (value[0] < 0.0) & (value[1] < 0.0)

    def general(_):
        projected, _ = _exp_root_projection(value)
        return projected

    return jax.lax.cond(
        in_primal,
        lambda _: value,
        lambda _: jax.lax.cond(
            in_polar,
            lambda __: jnp.zeros_like(value),
            lambda __: jax.lax.cond(
                face,
                lambda ___: jnp.asarray([value[0], 0.0, jnp.maximum(value[2], 0.0)]),
                general,
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )


def _boundary_kkt_matrix(value: Array, projected: Array, /) -> tuple[Array, Array]:
    x, y, _ = projected
    safe_y = jnp.where(y > 0.0, y, 1.0)
    ratio = x / safe_y
    exponential = jnp.exp(ratio)
    gradient = jnp.asarray(
        [exponential, exponential * (1.0 - ratio), -1.0], dtype=value.dtype
    )
    hessian = jnp.zeros((3, 3), dtype=value.dtype)
    hessian = hessian.at[0, 0].set(exponential / safe_y)
    hessian = hessian.at[0, 1].set(-ratio * exponential / safe_y)
    hessian = hessian.at[1, 0].set(hessian[0, 1])
    hessian = hessian.at[1, 1].set(ratio * ratio * exponential / safe_y)
    difference = value - projected
    multiplier = jnp.vdot(difference, gradient) / jnp.maximum(
        jnp.vdot(gradient, gradient), jnp.finfo(value.dtype).tiny
    )
    matrix = jnp.zeros((4, 4), dtype=value.dtype)
    matrix = matrix.at[:3, :3].set(jnp.eye(3, dtype=value.dtype) + multiplier * hessian)
    matrix = matrix.at[:3, 3].set(gradient)
    matrix = matrix.at[3, :3].set(gradient)
    return matrix, gradient


def _kkt_solve(matrix: Array, right: Array, /) -> Array:
    return solve(
        LinearSystem(
            DenseLinearOperator(matrix),
            problem_id="exponential-cone-projection-kkt",
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


def _projection_jvp(value: Array, projected: Array, tangent: Array, /) -> Array:
    in_primal = _in_primal(value)
    in_polar = _in_dual(-value)
    face = (value[0] < 0.0) & (value[1] < 0.0)

    def general(_):
        matrix, _ = _boundary_kkt_matrix(value, projected)
        right = jnp.concatenate((tangent, jnp.zeros(1, dtype=value.dtype)))
        return _kkt_solve(matrix, right)[:3]

    face_action = jnp.asarray(
        [tangent[0], 0.0, jnp.where(value[2] > 0.0, tangent[2], 0.0)]
    )
    return jax.lax.cond(
        in_primal,
        lambda _: tangent,
        lambda _: jax.lax.cond(
            in_polar,
            lambda __: jnp.zeros_like(tangent),
            lambda __: jax.lax.cond(face, lambda ___: face_action, general, operand=None),
            operand=None,
        ),
        operand=None,
    )


def _working_value(value: Array, /) -> Array:
    working_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    return value.astype(jnp.result_type(value.dtype, working_dtype))


def _homogeneous_scale(value: Array, /) -> Array:
    scale = jnp.max(jnp.abs(value))
    return jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)


@jax.custom_jvp
def _project_exp_single(value: Array, /) -> Array:
    working = _working_value(value)
    scale = _homogeneous_scale(working)
    projected = jax.lax.cond(
        _in_primal(working),
        lambda _: working,
        lambda _: scale * _projection_value(working / scale),
        operand=None,
    )
    return projected.astype(value.dtype)


@_project_exp_single.defjvp
def _project_exp_single_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    working = _working_value(value)
    working_tangent = tangent.astype(working.dtype)

    def exterior(_):
        scale = _homogeneous_scale(working)
        normalized = working / scale
        projected = _projection_value(normalized)
        derivative = _projection_jvp(normalized, projected, working_tangent)
        return scale * projected, derivative

    projected, derivative = jax.lax.cond(
        _in_primal(working),
        lambda _: (working, working_tangent),
        exterior,
        operand=None,
    )
    return projected.astype(value.dtype), derivative.astype(tangent.dtype)


def _project_exp_dual_single(value: Array, /) -> Array:
    working = _working_value(value)

    def exterior(_):
        scale = _homogeneous_scale(working)
        normalized = working / scale
        return scale * (normalized + _project_exp_single(-normalized))

    projected = jax.lax.cond(
        _in_dual(working),
        lambda _: working,
        exterior,
        operand=None,
    )
    return projected.astype(value.dtype)


def _smoothness_margin(value: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(value)), 1.0)
    normalized = value / scale
    primal_slack = _primal_slack(normalized)
    polar_slack = _dual_slack(-normalized)
    strict_primal = primal_slack > 0.0
    strict_polar = polar_slack > 0.0
    face = (normalized[0] < 0.0) & (normalized[1] < 0.0)
    face_margin = jnp.minimum(
        jnp.minimum(-normalized[0], -normalized[1]), jnp.abs(normalized[2])
    )
    projected, root = _exp_root_projection(normalized)
    kkt, _ = _boundary_kkt_matrix(normalized, projected)
    minimum_singular = _minimum_singular_value(kkt)
    general_margin = jnp.minimum(
        jnp.maximum(projected[1], 0.0),
        jnp.where(root.converged, minimum_singular, 0.0),
    )
    margin = jnp.where(
        strict_primal,
        primal_slack,
        jnp.where(
            strict_polar,
            polar_slack,
            jnp.where(face, face_margin, general_margin),
        ),
    )
    closed_boundary = (_in_primal(normalized) & ~strict_primal) | (
        _in_dual(-normalized) & ~strict_polar
    )
    margin = jnp.where(closed_boundary | ~jnp.isfinite(margin), 0.0, margin)
    return scale * jnp.maximum(margin, 0.0)


class ExponentialCone(AbstractConvexCone):
    """Three-dimensional exponential cone in canonical ``(x, y, z)`` order."""

    def __init__(self):
        self.dimension = 3
        self.cone_id = canonical_fingerprint({"kind": "exponential-cone"})

    def project(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        projected = jax.vmap(_project_exp_single)(flat)
        return projected.reshape(array.shape)

    def project_dual(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        projected = jax.vmap(_project_exp_dual_single)(flat)
        return projected.reshape(array.shape)

    def interior_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        margin = jax.vmap(_primal_slack)(flat)
        return margin.reshape(array.shape[:-1])

    def dual_projection_smoothness_margin(self, value: Any, /) -> Array:
        array = self._validate(value)
        flat = array.reshape((-1, 3))
        margin = jax.vmap(_smoothness_margin)(-flat)
        return margin.reshape(array.shape[:-1])


__all__ = ["ExponentialCone"]
