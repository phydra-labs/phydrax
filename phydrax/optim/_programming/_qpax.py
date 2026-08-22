#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import qpax
from jaxtyping import Array


_PUBLIC_STEP_FRACTION_DEFAULT = 0.995
_QPAX_FIXED_STEP_FRACTION = 0.99


def _validate_step_fraction(step_fraction: float, /) -> None:
    if step_fraction != _PUBLIC_STEP_FRACTION_DEFAULT:
        raise ValueError(
            "method='qpax-implicit' does not support configurable step_fraction; "
            f"QPax 0.1.4 fixes its fraction-to-boundary multiplier at "
            f"{_QPAX_FIXED_STEP_FRACTION}. Omit step_fraction to use the public "
            f"default ({_PUBLIC_STEP_FRACTION_DEFAULT})."
        )


def _regularized_quadratic(quadratic: Array, regularization: float, /) -> Array:
    if quadratic.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("The QPax backend supports only float32 and float64 QP data.")
    return quadratic + regularization * jnp.eye(
        quadratic.shape[-1], dtype=quadratic.dtype
    )


def solve_qpax_implicit(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    /,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Call QPax 0.1.4's public implicit full-result API over flat batches.

    QPax fixes its fraction-to-boundary multiplier at 0.99, so only the public
    default ``step_fraction`` request is accepted.
    """

    _validate_step_fraction(step_fraction)
    quadratic = _regularized_quadratic(quadratic, regularization)

    def solve_one(q, c, a, b, g, h):
        return qpax.solve_qp(
            q,
            c,
            a,
            b,
            g,
            h,
            backend="i",
            solver_tol=tolerance,
            max_iter=max_iterations,
        )

    primal, slack, inequality_dual, equality_dual, converged, iterations = jax.vmap(
        solve_one
    )(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
    )
    return (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        jnp.asarray(converged, dtype=bool),
        jnp.asarray(iterations, dtype=jnp.int32),
    )


def solve_qpax_implicit_primal(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    /,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> Array:
    """Call QPax 0.1.4's public implicit custom-VJP primal API.

    QPax fixes its fraction-to-boundary multiplier at 0.99, so only the public
    default ``step_fraction`` request is accepted.
    """

    _validate_step_fraction(step_fraction)
    quadratic = _regularized_quadratic(quadratic, regularization)

    def solve_one(q, c, a, b, g, h):
        return qpax.solve_qp_primal(
            q,
            c,
            a,
            b,
            g,
            h,
            backend="i",
            solver_tol=tolerance,
            max_iter=max_iterations,
        )

    return jax.vmap(solve_one)(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
    )


__all__ = ["solve_qpax_implicit", "solve_qpax_implicit_primal"]
