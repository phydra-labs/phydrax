#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rotation factor of the 3D polar decomposition with admission evidence."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from functools import partial
from typing import final

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ._dense_svd import economy_svd
from ._small_batched import (
    determinant_small_linear,
    SmallLinearSolvePlan,
    solve_small_linear,
)


_DEFAULT_PLAN = SmallLinearSolvePlan(3)


class PolarRotation3Status(IntEnum):
    """Admission of one frame ``F`` for its polar rotation (first failure)."""

    SUCCESS = 0
    NONFINITE = 1
    SINGULAR = 2
    ILL_CONDITIONED = 3
    IMPROPER = 4
    DERIVATIVE_OPERATOR_FAILED = 5


@final
class PolarRotation3Result(StrictModule):
    """Rotation factor ``R`` of ``F = R U`` with nondifferentiable evidence.

    ``value`` is ``R`` for an admitted frame and NaN otherwise.
    ``singular_values`` (descending), ``determinant`` (``det F``),
    ``condition_estimate`` (``s_1 / s_3``), ``derivative_condition_estimate``
    (native small-solve estimate of the derivative operator
    ``tr(U) I - U``), ``successful`` and ``status`` (`PolarRotation3Status`)
    are evidence without derivatives.
    """

    value: Array
    singular_values: Array
    determinant: Array
    condition_estimate: Array
    derivative_condition_estimate: Array
    successful: Array
    status: Array


def _derivative_operator(rotation: Array, frame: Array, /) -> Array:
    """Return ``tr(U) I - U`` with ``U = sym(R^T F)``.

    ``[w]_x U + U [w]_x = X - X^T`` is ``(tr(U) I - U) w = vee(X - X^T)``.
    """
    stretch = rotation.T @ frame
    stretch = 0.5 * (stretch + stretch.T)
    return jnp.trace(stretch) * jnp.eye(3, dtype=stretch.dtype) - stretch


def _poisoned_solve(
    plan: SmallLinearSolvePlan,
    operator: Array,
    matvec: Callable[[Array], Array],
    rhs: Array,
    /,
) -> Array:
    del matvec
    solved = solve_small_linear(plan, operator, rhs)
    return jnp.where(solved.successful, solved.value, jnp.nan)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _factor(plan: SmallLinearSolvePlan, frame: Array) -> tuple[Array, Array]:
    left, singular, right_adjoint = economy_svd(frame)
    return left @ right_adjoint, singular


@_factor.defjvp
def _factor_jvp(
    plan: SmallLinearSolvePlan, primals: tuple[Array], tangents: tuple[Array]
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    (frame,), (frame_tangent,) = primals, tangents
    rotation, singular = _factor(plan, frame)
    operator = _derivative_operator(rotation, frame)
    local = rotation.T @ frame_tangent
    axial = jnp.stack(
        (
            local[2, 1] - local[1, 2],
            local[0, 2] - local[2, 0],
            local[1, 0] - local[0, 1],
        )
    )
    omega = jax.lax.custom_linear_solve(
        lambda vector: operator @ vector,
        axial,
        partial(_poisoned_solve, plan, operator),
        symmetric=True,
    )
    generator = jnp.cross(omega, jnp.eye(3, dtype=omega.dtype)).T
    return (rotation, singular), (rotation @ generator, jnp.zeros_like(singular))


def polar_rotation3(
    frame: ArrayLike, /, *, plan: SmallLinearSolvePlan | None = None
) -> PolarRotation3Result:
    """Return the rotation factor ``R`` of the polar decomposition ``F = R U``.

    ``frame`` is one real ``3 x 3`` matrix (vmap for batches).  ``plan``
    (default ``SmallLinearSolvePlan(3)``) owns every admission threshold; the
    first failed condition is the status: finite entries (a non-finite frame
    never reaches the SVD), ``s_3 > singular_tolerance * s_1``,
    ``s_1 / s_3 <= maximum_condition``, ``det F > 0`` and, last, the native
    small-solve admission (scaled determinant and ``maximum_condition``) of
    the derivative operator ``tr(U) I - U``, the very solve its tangents use.
    An inadmissible frame has a NaN value.
    Orientation is tested on the orthogonal factor, whose determinant is
    scale-free. The unscaled determinant evidence may underflow or overflow.

    The value is ``W V^T`` of the native economy SVD ``F = W S V^T``.  SVD
    factors are not differentiable at repeated singular values (``F = I`` and
    every isotropic frame), but ``R`` is smooth on admitted frames: with
    ``U = R^T F`` and ``X = R^T dF``, ``dR = R [omega]_x`` where
    ``(tr(U) I - U) omega = vee(X - X^T)``.  The operator (eigenvalues
    ``s_i + s_j``, ``i != j``) is applied through a symmetric
    ``custom_linear_solve`` with the native small solve; a failed solve
    poisons the tangent with NaN.  The rule re-enters the factorization, so
    every higher derivative is exact as well.
    """
    plan_ = _DEFAULT_PLAN if plan is None else plan
    if not isinstance(plan_, SmallLinearSolvePlan):
        raise TypeError("plan must be a SmallLinearSolvePlan or None.")
    if plan_.dimension != 3:
        raise ValueError("polar_rotation3 requires a three-dimensional plan.")
    value = jnp.asarray(frame)
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("frame must be a real floating matrix.")
    if value.shape != (3, 3):
        raise ValueError("frame must have shape (3, 3).")
    finite = jnp.all(jnp.isfinite(jax.lax.stop_gradient(value)))
    # Guarded execution: a non-finite frame is already refused (NONFINITE).
    operand = jnp.where(finite, value, jnp.eye(3, dtype=value.dtype))
    rotation, singular = _factor(plan_, operand)
    singular = jax.lax.stop_gradient(singular)
    fixed = jax.lax.stop_gradient(operand)
    determinant = determinant_small_linear(plan_, fixed)
    fixed_rotation = jax.lax.stop_gradient(rotation)
    orientation = determinant_small_linear(plan_, fixed_rotation)
    derivative = solve_small_linear(
        plan_,
        _derivative_operator(fixed_rotation, fixed),
        jnp.zeros((3,), dtype=value.dtype),
    )
    largest, smallest = singular[0], singular[2]
    condition = largest / jnp.where(smallest > 0.0, smallest, 1.0)
    status = jnp.where(
        ~finite,
        int(PolarRotation3Status.NONFINITE),
        jnp.where(
            smallest <= plan_.singular_tolerance * largest,
            int(PolarRotation3Status.SINGULAR),
            jnp.where(
                condition > plan_.maximum_condition,
                int(PolarRotation3Status.ILL_CONDITIONED),
                jnp.where(
                    orientation <= 0.0,
                    int(PolarRotation3Status.IMPROPER),
                    jnp.where(
                        derivative.successful,
                        int(PolarRotation3Status.SUCCESS),
                        int(PolarRotation3Status.DERIVATIVE_OPERATOR_FAILED),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(PolarRotation3Status.SUCCESS)
    nan = jnp.asarray(jnp.nan, dtype=value.dtype)
    return PolarRotation3Result(
        jnp.where(successful, rotation, nan),
        jnp.where(finite, singular, nan),
        jnp.where(finite, determinant, nan),
        jnp.where(finite & (smallest > 0.0), condition, jnp.inf),
        jnp.where(finite, derivative.condition_estimate, jnp.inf),
        successful,
        status,
    )


__all__ = ["PolarRotation3Result", "PolarRotation3Status", "polar_rotation3"]
