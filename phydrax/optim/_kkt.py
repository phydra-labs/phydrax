#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    HermitianPrecisionPolicy,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear,
    PreparedLinearSolve,
    solve as solve_linear,
)


def _spectral_precision(
    precision: NonlinearPrecisionPolicy,
    /,
) -> HermitianPrecisionPolicy:
    compute = (
        precision.residual_dtype
        if precision.residual_dtype is not None
        else precision.state_dtype
    )
    accumulation = (
        precision.accumulation_dtype
        if precision.accumulation_dtype is not None
        else compute
    )
    return HermitianPrecisionPolicy(
        compute_dtype=compute,
        factorization_dtype=accumulation,
        accumulation_dtype=accumulation,
        decision_dtype=precision.decision_dtype,
    )


KKTForm: TypeAlias = Literal["dense-augmented"]


class KKTRegularizationPolicy(StrictModule):
    """Primal/dual shifts and bounded inertia-correction work."""

    initial_primal: float = eqx.field(static=True)
    initial_dual: float = eqx.field(static=True)
    primal_growth: float = eqx.field(static=True)
    dual_growth: float = eqx.field(static=True)
    maximum_primal: float = eqx.field(static=True)
    maximum_dual: float = eqx.field(static=True)
    maximum_corrections: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_primal: float = 1e-10,
        initial_dual: float = 1e-10,
        primal_growth: float = 10.0,
        dual_growth: float = 10.0,
        maximum_primal: float = 1e8,
        maximum_dual: float = 1e8,
        maximum_corrections: int = 16,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_primal,
                initial_dual,
                primal_growth,
                dual_growth,
                maximum_primal,
                maximum_dual,
            )
        )
        corrections = int(maximum_corrections)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("KKT regularization values must be finite and positive.")
        if values[2] <= 1.0 or values[3] <= 1.0:
            raise ValueError("KKT regularization growth factors must exceed one.")
        if values[4] < values[0] or values[5] < values[1]:
            raise ValueError("Maximum KKT regularization must exceed initial values.")
        if corrections < 0:
            raise ValueError("maximum_corrections must be non-negative.")
        (
            self.initial_primal,
            self.initial_dual,
            self.primal_growth,
            self.dual_growth,
            self.maximum_primal,
            self.maximum_dual,
        ) = values
        self.maximum_corrections = corrections


class KKTInertia(StrictModule):
    positive: Array
    negative: Array
    zero: Array
    tolerance: Array
    certified: Array
    source: str = eqx.field(static=True)
    zero_reliable: bool = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)


class KKTPlan(StrictModule):
    form: KKTForm = eqx.field(static=True)
    primal_dimension: int = eqx.field(static=True)
    constraint_dimension: int = eqx.field(static=True)
    expected_positive: int = eqx.field(static=True)
    expected_negative: int = eqx.field(static=True)
    expected_zero: int = eqx.field(static=True)
    regularization: KKTRegularizationPolicy
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

    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)


class KKTFactorization(StrictModule):
    """Reusable factorization of one regularized canonical KKT matrix."""

    plan: KKTPlan
    matrix: Array
    linear: PreparedLinearSolve
    inertia: KKTInertia
    primal_regularization: Array
    dual_regularization: Array
    inertia_matches: Array
    finite: Array
    precision: NonlinearPrecisionPolicy


def kkt_inertia(
    matrix: Any,
    /,
    *,
    tolerance: float = 1e-10,
    precision: NonlinearPrecisionPolicy | None = None,
) -> KKTInertia:
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    value = precision_.accumulation(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("KKT inertia requires one square matrix.")
    spectrum = HermitianSpectrum(
        value,
        tolerance=tolerance,
        precision=_spectral_precision(precision_),
    )
    eigenvalues = spectrum.eigenvalues
    threshold = precision_.decision(tolerance)
    return KKTInertia(
        positive=jnp.sum(eigenvalues > threshold, dtype=jnp.int32),
        negative=jnp.sum(eigenvalues < -threshold, dtype=jnp.int32),
        zero=jnp.sum(jnp.abs(eigenvalues) <= threshold, dtype=jnp.int32),
        tolerance=threshold,
        certified=jnp.all(jnp.isfinite(eigenvalues)),
        source="dense-hermitian-spectrum",
        zero_reliable=True,
        precision_evidence=spectrum.precision_evidence,
    )


def plan_kkt(
    primal_dimension: int,
    constraint_dimension: int,
    /,
    *,
    jacobian_density: float = 1.0,
    regularization: float = 1e-10,
    maximum_regularization: float = 1e8,
    regularization_policy: KKTRegularizationPolicy | None = None,
) -> KKTPlan:
    primal = int(primal_dimension)
    constraints = int(constraint_dimension)
    density = float(jacobian_density)
    if primal < 1 or constraints < 0:
        raise ValueError("KKT dimensions are invalid.")
    if not isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError("jacobian_density must lie in [0, 1].")
    policy = (
        KKTRegularizationPolicy(
            initial_primal=regularization,
            initial_dual=regularization,
            maximum_primal=maximum_regularization,
            maximum_dual=maximum_regularization,
        )
        if regularization_policy is None
        else regularization_policy
    )
    if not isinstance(policy, KKTRegularizationPolicy):
        raise TypeError("regularization_policy must be KKTRegularizationPolicy or None.")
    form: KKTForm = "dense-augmented"
    plan_id = canonical_fingerprint(
        {
            "kind": "kkt-plan",
            "form": form,
            "primal": primal,
            "constraints": constraints,
            "density": density,
            "regularization": {
                "initial_primal": policy.initial_primal,
                "initial_dual": policy.initial_dual,
                "primal_growth": policy.primal_growth,
                "dual_growth": policy.dual_growth,
                "maximum_primal": policy.maximum_primal,
                "maximum_dual": policy.maximum_dual,
                "maximum_corrections": policy.maximum_corrections,
            },
        }
    )
    return KKTPlan(
        form=form,
        primal_dimension=primal,
        constraint_dimension=constraints,
        expected_positive=primal,
        expected_negative=constraints,
        expected_zero=0,
        regularization=policy,
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
    *,
    precision: NonlinearPrecisionPolicy | None = None,
) -> KKTFactorization:
    """Factor one canonical KKT matrix for one or more right-hand sides."""
    if not isinstance(plan, KKTPlan):
        raise TypeError("plan must be KKTPlan.")
    precision_ = NonlinearPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, NonlinearPrecisionPolicy):
        raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
    hessian_ = precision_.accumulation(hessian)
    jacobian_ = precision_.accumulation(jacobian)
    if hessian_.shape != (plan.primal_dimension, plan.primal_dimension):
        raise ValueError("Hessian shape does not match KKT plan.")
    if jacobian_.shape != (plan.constraint_dimension, plan.primal_dimension):
        raise ValueError("Jacobian shape does not match KKT plan.")
    policy = plan.regularization
    primal_regularization = jnp.asarray(
        policy.initial_primal,
        dtype=hessian_.dtype,
    )
    dual_regularization = jnp.asarray(
        policy.initial_dual,
        dtype=hessian_.dtype,
    )
    matrix = _augmented_matrix(
        hessian_,
        jacobian_,
        primal_regularization,
        dual_regularization,
    )
    inertia = kkt_inertia(matrix, precision=precision_)
    for _ in range(policy.maximum_corrections):
        matches = (
            inertia.certified
            & inertia.zero_reliable
            & (inertia.positive == plan.expected_positive)
            & (inertia.negative == plan.expected_negative)
            & (inertia.zero == plan.expected_zero)
        )
        if bool(matches):
            break
        primal_regularization = jnp.minimum(
            policy.maximum_primal,
            policy.primal_growth * primal_regularization,
        )
        dual_regularization = jnp.minimum(
            policy.maximum_dual,
            policy.dual_growth * dual_regularization,
        )
        matrix = _augmented_matrix(
            hessian_,
            jacobian_,
            primal_regularization,
            dual_regularization,
        )
        inertia = kkt_inertia(matrix, precision=precision_)
    matches = (
        inertia.certified
        & inertia.zero_reliable
        & (inertia.positive == plan.expected_positive)
        & (inertia.negative == plan.expected_negative)
        & (inertia.zero == plan.expected_zero)
    )
    linear_problem = LinearSystem(DenseLinearOperator(matrix))
    linear = prepare_linear(
        linear_problem,
        precision_.bind_linear(LinearSolvePolicy(DenseLU())),
    )
    finite = jnp.all(jnp.isfinite(matrix))
    return KKTFactorization(
        plan,
        matrix,
        linear,
        inertia,
        primal_regularization,
        dual_regularization,
        matches,
        finite,
        precision_,
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
    primal_rhs = factorization.precision.accumulation(primal_residual)
    constraint_rhs = factorization.precision.accumulation(constraint_residual)
    if primal_rhs.shape != (plan.primal_dimension,):
        raise ValueError("Primal residual shape does not match KKT plan.")
    if constraint_rhs.shape != (plan.constraint_dimension,):
        raise ValueError("Constraint residual shape does not match KKT plan.")
    rhs = -jnp.concatenate([primal_rhs, constraint_rhs])
    linear_result = solve_linear(factorization.linear, rhs)
    solution = factorization.precision.direction(linear_result.value)
    primal_step = solution[: plan.primal_dimension]
    dual_step = solution[plan.primal_dimension :]
    residual = factorization.matrix @ solution - rhs
    finite = (
        factorization.finite
        & linear_result.diagnostics.converged
        & jnp.all(jnp.isfinite(solution))
    )
    return KKTSolveResult(
        primal_step,
        dual_step,
        factorization.inertia,
        factorization.primal_regularization,
        factorization.dual_regularization,
        factorization.precision.decision(jnp.linalg.norm(residual)),
        factorization.inertia_matches,
        finite,
        factorization.inertia.precision_evidence,
    )


def solve_kkt(
    hessian: Any,
    jacobian: Any,
    primal_residual: Any,
    constraint_residual: Any,
    plan: KKTPlan,
    /,
    *,
    precision: NonlinearPrecisionPolicy | None = None,
) -> KKTSolveResult:
    """Factor and solve one canonical KKT system."""
    factorization = factor_kkt(
        hessian,
        jacobian,
        plan,
        precision=precision,
    )
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
    "KKTRegularizationPolicy",
    "KKTSolveResult",
    "factor_kkt",
    "kkt_inertia",
    "plan_kkt",
    "solve_kkt",
    "solve_factored_kkt",
]
