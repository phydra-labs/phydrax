#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import checked, parse
from ._property_verification import (
    DensePropertyEvidence,
    DensePropertyVerificationPolicy,
    verify_dense_properties,
)


type ConditionTargetBound = Literal["gershgorin", "trace"]


@final
class ConditionTargetPolicy(StrictModule):
    """Explicit model-changing diagonal regularization in normalized coordinates.

    The target concerns ``D**(-1/2) C D**(-1/2) + eta I``, where
    ``D = diag(diag(C))``. It does not bound the condition number of the
    covariance in its original coordinates. Targets with ``target * eps >= 1``
    are refused at execution because the input dtype cannot resolve a useful
    reciprocal condition estimate.
    """

    max_condition_number: float = eqx.field(static=True)
    bound: ConditionTargetBound = eqx.field(static=True)

    def __init__(
        self,
        max_condition_number: float,
        *,
        bound: ConditionTargetBound = "gershgorin",
    ) -> None:
        target = float(max_condition_number)
        if not math.isfinite(target) or target <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one.")
        bound_ = parse(bound, ConditionTargetBound, "bound")
        self.max_condition_number = target
        self.bound = bound_


@final
class ConditionTargetResult(StrictModule):
    """Transformed covariance and explicitly diagonal-normalized evidence.

    ``property_evidence`` describes the diagonal-normalized input, with the
    native verifier's numerical PSD tolerance. Its eigendecomposition is
    diagnostic and is not differentiated. The covariance itself is never
    symmetrized or repaired.
    ``normalization_scale`` is ``sqrt(diag(C))``; ``applied_diagonal`` is
    ``eta * diag(C)``. All numerical result fields remain dynamic PyTree leaves.
    ``normalized_upper_bound`` is the ideal PSD bound used to choose eta, not
    an unconditional spectral certificate for numerically accepted PSD input.
    ``normalized_condition_bound`` includes the verifier's eigenvalue tolerance
    in its denominator and, for trace, its numerator.
    ``normalized_row_sums`` retains the exact primal absolute row sums for
    Gershgorin derivative tie selection, and is ``None`` for trace.
    """

    covariance: Array
    applied_diagonal: Array
    eta: Array
    normalized_upper_bound: Array
    normalized_row_sums: Array | None
    normalized_condition_bound: Array
    normalization_scale: Array
    property_evidence: DensePropertyEvidence
    policy: ConditionTargetPolicy
    coordinate_system: Literal["diagonal-normalized"] = eqx.field(
        static=True, default="diagonal-normalized"
    )


@checked
def condition_target_covariance(
    matrix: ArrayLike,
    policy: ConditionTargetPolicy,
) -> ConditionTargetResult:
    """Return ``C + eta diag(diag(C))`` under an explicit condition policy.

    The complete observation covariance, including any explicit noise and
    jitter, must be supplied by callers. For a non-singleton covariance,
    ``eta = upper_bound(B) / (target - 1)``, with
    ``B = D**(-1/2) C D**(-1/2)``. Gershgorin uses the maximum absolute row
    sum, and trace uses the PSD trace upper bound. The ideal PSD condition bound
    is ``(upper_bound + eta) / eta``. For reported evidence let
    ``lower = min(min_eigenvalue(B) - tolerance, 0)``. Its denominator is
    ``eta + lower``; its numerator uses the Gershgorin upper bound or, for
    trace, ``trace(B) - (n - 1) * lower``, then adds eta. This adjustment is
    diagnostic only: the ideal PSD bound still determines eta and its
    derivative. The reported bound can exceed the requested target near
    numerical singularity. A positive singleton is already condition one and
    is left unchanged.

    Real finite symmetric numerical-PSD matrices with strictly positive
    diagonals are supported. A PSD matrix with a zero diagonal is legitimate,
    but is outside this normalization's domain and is refused, not repaired.
    Native symmetry and PSD tolerances in normalized coordinates are reported
    in ``property_evidence``. Verification performs one dense eigendecomposition
    per invocation. The supplied covariance is retained without symmetrization
    or repair.

    Scaling and eta are differentiated, including the selected row of the
    Gershgorin bound. Absolute values at zero and tied maximizing rows have
    piecewise kinks and follow JAX's corresponding derivative conventions.
    No stochastic approximation, hidden jitter, or raw-coordinate condition
    guarantee is introduced.
    """
    value = jnp.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError("Condition targeting requires a nonempty square matrix.")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("Condition targeting requires a real floating-point covariance.")
    if value.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Condition targeting supports float32 and float64 covariances.")
    eps = jnp.finfo(value.dtype).eps
    if policy.max_condition_number * eps >= 1.0:
        raise ValueError(
            "Condition target cannot support meaningful precision in this dtype."
        )
    target = jnp.asarray(policy.max_condition_number, dtype=value.dtype)
    value = eqx.error_if(
        value,
        ~jnp.isfinite(target) | (target <= 1.0),
        "Condition target must be representably finite and greater than one in this dtype.",
    )
    value = eqx.error_if(
        value,
        ~jnp.all(jnp.isfinite(value)),
        "Condition targeting requires a finite symmetric positive-semidefinite covariance.",
    )
    diagonal = jnp.diag(value)
    diagonal = eqx.error_if(
        diagonal,
        jnp.any(diagonal <= 0.0),
        "Diagonal normalization requires strictly positive diagonal entries; zero-diagonal PSD covariances are outside its domain.",
    )
    scale = jnp.sqrt(diagonal)
    normalized = value / scale[:, None] / scale[None, :]
    normalized = eqx.error_if(
        normalized,
        ~jnp.all(jnp.isfinite(normalized)),
        "Diagonal-normalized covariance is not finite in this dtype.",
    )
    evidence = verify_dense_properties(
        jax.lax.stop_gradient(normalized),
        policy=DensePropertyVerificationPolicy(require_positive_semidefinite=True),
    )
    normalized = eqx.error_if(
        normalized,
        ~evidence.successful,
        "Condition targeting requires a finite symmetric positive-semidefinite covariance in normalized coordinates.",
    )
    match policy.bound:
        case "gershgorin":
            row_sums = jnp.sum(jnp.abs(normalized), axis=1)
            upper_bound = jnp.max(row_sums)
        case "trace":
            row_sums = None
            upper_bound = jnp.trace(normalized)
        case _:
            assert_never(policy.bound)
    if value.shape[0] == 1:
        eta = jnp.zeros((), dtype=value.dtype)
        condition_bound = jnp.ones((), dtype=value.dtype)
    else:
        eta = upper_bound / (target - 1.0)
        lower_bound = jnp.minimum(jnp.min(evidence.eigenvalues) - evidence.tolerance, 0.0)
        shifted_lower_bound = eqx.error_if(
            eta + lower_bound,
            eta + lower_bound <= 0.0,
            "Condition target cannot certify a positive normalized eigenvalue bound in this dtype.",
        )
        match policy.bound:
            case "gershgorin":
                evidence_upper_bound = upper_bound
            case "trace":
                evidence_upper_bound = upper_bound - (value.shape[0] - 1) * lower_bound
            case _:
                assert_never(policy.bound)
        condition_bound = (evidence_upper_bound + eta) / shifted_lower_bound
    applied_diagonal = eta * diagonal
    covariance = value.at[jnp.diag_indices(value.shape[0])].add(applied_diagonal)
    covariance = eqx.error_if(
        covariance,
        ~jnp.all(jnp.isfinite(covariance)) | ~jnp.isfinite(condition_bound),
        "Condition-target covariance or normalized bound overflowed in this dtype.",
    )
    return ConditionTargetResult(
        covariance=covariance,
        applied_diagonal=applied_diagonal,
        eta=eta,
        normalized_upper_bound=upper_bound,
        normalized_row_sums=row_sums,
        normalized_condition_bound=condition_bound,
        normalization_scale=scale,
        property_evidence=evidence,
        policy=policy,
    )


__all__ = [
    "ConditionTargetBound",
    "ConditionTargetPolicy",
    "ConditionTargetResult",
    "condition_target_covariance",
]
