#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


KKTForm: TypeAlias = Literal["augmented", "null-space", "range-space"]


class KKTInertia(StrictModule):
    positive: Array
    negative: Array
    zero: Array
    tolerance: Array


class KKTPlan(StrictModule):
    form: KKTForm = eqx.field(static=True)
    primal_dimension: int = eqx.field(static=True)
    constraint_dimension: int = eqx.field(static=True)
    expected_positive: int = eqx.field(static=True)
    expected_negative: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    maximum_regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class KKTSolveResult(StrictModule):
    primal_step: Array
    dual_step: Array
    inertia: KKTInertia
    primal_regularization: Array
    dual_regularization: Array
    residual_norm: Array
    inertia_matches: Array
    finite: Array


class KKTFactorization(StrictModule):
    """Reusable factorization of one regularized canonical KKT matrix."""

    plan: KKTPlan
    matrix: Array
    factor: Array
    pivots: Array
    inertia: KKTInertia
    primal_regularization: Array
    dual_regularization: Array
    inertia_matches: Array
    finite: Array


def kkt_inertia(matrix: Any, /, *, tolerance: float = 1e-10) -> KKTInertia:
    value = jnp.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("KKT inertia requires one square matrix.")
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (value + jnp.conj(value.T)))
    threshold = jnp.asarray(tolerance, dtype=eigenvalues.real.dtype)
    return KKTInertia(
        jnp.sum(eigenvalues > threshold, dtype=jnp.int32),
        jnp.sum(eigenvalues < -threshold, dtype=jnp.int32),
        jnp.sum(jnp.abs(eigenvalues) <= threshold, dtype=jnp.int32),
        threshold,
    )


def plan_kkt(
    primal_dimension: int,
    constraint_dimension: int,
    /,
    *,
    jacobian_density: float = 1.0,
    regularization: float = 1e-10,
    maximum_regularization: float = 1e8,
) -> KKTPlan:
    primal = int(primal_dimension)
    constraints = int(constraint_dimension)
    density = float(jacobian_density)
    regularization_ = float(regularization)
    maximum_ = float(maximum_regularization)
    if primal < 1 or constraints < 0:
        raise ValueError("KKT dimensions are invalid.")
    if not isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("jacobian_density must lie in [0, 1].")
    if not isfinite(regularization_) or regularization_ <= 0.0:
        raise ValueError("regularization must be finite and positive.")
    if not isfinite(maximum_) or maximum_ < regularization_:
        raise ValueError("maximum_regularization must exceed regularization.")
    if constraints == 0:
        form: KKTForm = "augmented"
    elif constraints < primal // 4 and density > 0.25:
        form = "range-space"
    elif constraints > primal // 2:
        form = "null-space"
    else:
        form = "augmented"
    plan_id = canonical_fingerprint(
        {
            "kind": "kkt-plan",
            "form": form,
            "primal": primal,
            "constraints": constraints,
            "density": density,
        }
    )
    return KKTPlan(
        form=form,
        primal_dimension=primal,
        constraint_dimension=constraints,
        expected_positive=primal,
        expected_negative=constraints,
        regularization=regularization_,
        maximum_regularization=maximum_,
        plan_id=plan_id,
    )


def _augmented_matrix(hessian, jacobian, primal_regularization, dual_regularization):
    n = hessian.shape[0]
    m = jacobian.shape[0]
    return jnp.block(
        [
            [
                hessian + primal_regularization * jnp.eye(n, dtype=hessian.dtype),
                jnp.conj(jacobian.T),
            ],
            [jacobian, -dual_regularization * jnp.eye(m, dtype=hessian.dtype)],
        ]
    )


def factor_kkt(
    hessian: Any,
    jacobian: Any,
    plan: KKTPlan,
    /,
) -> KKTFactorization:
    """Factor one canonical KKT matrix for one or more right-hand sides."""
    if not isinstance(plan, KKTPlan):
        raise TypeError("plan must be KKTPlan.")
    hessian_ = jnp.asarray(hessian)
    jacobian_ = jnp.asarray(jacobian)
    if hessian_.shape != (plan.primal_dimension, plan.primal_dimension):
        raise ValueError("Hessian shape does not match KKT plan.")
    if jacobian_.shape != (plan.constraint_dimension, plan.primal_dimension):
        raise ValueError("Jacobian shape does not match KKT plan.")
    primal_regularization = jnp.asarray(plan.regularization, dtype=hessian_.dtype)
    dual_regularization = jnp.asarray(plan.regularization, dtype=hessian_.dtype)
    matrix = _augmented_matrix(
        hessian_,
        jacobian_,
        primal_regularization,
        dual_regularization,
    )
    inertia = kkt_inertia(matrix)
    for _ in range(16):
        matches = (
            (inertia.positive == plan.expected_positive)
            & (inertia.negative == plan.expected_negative)
            & (inertia.zero == 0)
        )
        if bool(matches):
            break
        primal_regularization = jnp.minimum(
            plan.maximum_regularization,
            10.0 * primal_regularization,
        )
        dual_regularization = jnp.minimum(
            plan.maximum_regularization,
            10.0 * dual_regularization,
        )
        matrix = _augmented_matrix(
            hessian_,
            jacobian_,
            primal_regularization,
            dual_regularization,
        )
        inertia = kkt_inertia(matrix)
    factor, pivots = jsp.linalg.lu_factor(matrix)
    matches = (
        (inertia.positive == plan.expected_positive)
        & (inertia.negative == plan.expected_negative)
        & (inertia.zero == 0)
    )
    finite = jnp.all(jnp.isfinite(factor))
    return KKTFactorization(
        plan,
        matrix,
        factor,
        pivots,
        inertia,
        primal_regularization,
        dual_regularization,
        matches,
        finite,
    )


def solve_factored_kkt(
    factorization: KKTFactorization,
    primal_residual: Any,
    constraint_residual: Any,
    /,
) -> KKTSolveResult:
    """Solve one KKT right-hand side without rebuilding its factorization."""
    if not isinstance(factorization, KKTFactorization):
        raise TypeError("factorization must be KKTFactorization.")
    plan = factorization.plan
    primal_rhs = jnp.asarray(primal_residual)
    constraint_rhs = jnp.asarray(constraint_residual)
    if primal_rhs.shape != (plan.primal_dimension,):
        raise ValueError("Primal residual shape does not match KKT plan.")
    if constraint_rhs.shape != (plan.constraint_dimension,):
        raise ValueError("Constraint residual shape does not match KKT plan.")
    rhs = -jnp.concatenate([primal_rhs, constraint_rhs])
    solution = jsp.linalg.lu_solve(
        (factorization.factor, factorization.pivots),
        rhs,
    )
    primal_step = solution[: plan.primal_dimension]
    dual_step = solution[plan.primal_dimension :]
    residual = factorization.matrix @ solution - rhs
    finite = factorization.finite & jnp.all(jnp.isfinite(solution))
    return KKTSolveResult(
        primal_step,
        dual_step,
        factorization.inertia,
        factorization.primal_regularization,
        factorization.dual_regularization,
        jnp.linalg.norm(residual),
        factorization.inertia_matches,
        finite,
    )


def solve_kkt(
    hessian: Any,
    jacobian: Any,
    primal_residual: Any,
    constraint_residual: Any,
    plan: KKTPlan,
    /,
) -> KKTSolveResult:
    """Factor and solve one canonical KKT system."""
    factorization = factor_kkt(hessian, jacobian, plan)
    return solve_factored_kkt(
        factorization,
        primal_residual,
        constraint_residual,
    )


__all__ = [
    "KKTFactorization",
    "KKTForm",
    "KKTInertia",
    "KKTPlan",
    "KKTSolveResult",
    "factor_kkt",
    "kkt_inertia",
    "plan_kkt",
    "solve_kkt",
    "solve_factored_kkt",
]
