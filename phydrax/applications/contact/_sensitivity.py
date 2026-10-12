#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from math import isfinite
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._strict import StrictModule
from ...nonlinear._sensitivity import (
    root_solution_jvp,
    root_solution_vjp,
    SensitivityEvidence,
    SensitivityPolicy,
    SolutionMapDerivative,
)
from ...typing import checked
from ._solver import _ContactRuntime, MonolithicContactPlan, MonolithicContactResult


@final
class ContactSensitivityResult(StrictModule):
    value: PyTree[ArrayLike]
    root_evidence: SensitivityEvidence
    minimum_gap: Array
    minimum_feature_margin: Array
    route_complete: Array
    branch_qualified: Array
    successful: Array


def _runtime(
    plan: MonolithicContactPlan, result: MonolithicContactResult, /
) -> _ContactRuntime:
    if plan.plan_id != result.plan_id:
        raise ValueError("Result belongs to another monolithic contact plan.")
    return _ContactRuntime(
        plan,
        result.previous,
        result.candidate_epoch,
        result.step_size,
        result.candidate.friction_state,
        False,
    )


def _qualify(
    derivative: SolutionMapDerivative,
    result: MonolithicContactResult,
    tolerance: float,
    /,
) -> ContactSensitivityResult:
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("margin_tolerance must be finite and nonnegative.")
    threshold = jnp.asarray(tolerance, dtype=result.contact.minimum_gap.dtype)
    branch = (
        result.accepted
        & result.candidate_epoch.successful
        & (result.contact.minimum_gap > threshold)
        & (result.contact.minimum_feature_margin > threshold)
    )
    successful = derivative.evidence.successful & branch
    value = jax.tree.map(
        lambda leaf: (
            jnp.where(successful, leaf, jnp.full_like(leaf, jnp.nan))
            if eqx.is_inexact_array(leaf)
            else leaf
        ),
        derivative.value,
    )
    return ContactSensitivityResult(
        value,
        derivative.evidence,
        result.contact.minimum_gap,
        result.contact.minimum_feature_margin,
        result.candidate_epoch.successful,
        branch,
        successful,
    )


@checked
def monolithic_contact_solution_jvp(
    plan: MonolithicContactPlan,
    result: MonolithicContactResult,
    arguments: Any,
    tangent_arguments: Any,
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1e-10,
) -> ContactSensitivityResult:
    """Implicit local derivative with collision routes and friction frame frozen.

    History is frozen at the previous accepted state. Feature switches and failed
    primal steps invalidate the derivative; they never produce a fabricated zero.
    """
    runtime = _runtime(plan, result)
    derivative = root_solution_jvp(
        runtime.problem(),
        result.candidate.coordinates,
        arguments,
        tangent_arguments,
        policy=policy,
    )
    return _qualify(derivative, result, margin_tolerance)


@checked
def monolithic_contact_solution_vjp(
    plan: MonolithicContactPlan,
    result: MonolithicContactResult,
    arguments: Any,
    cotangent_state: PyTree[Array],
    /,
    *,
    policy: SensitivityPolicy | None = None,
    margin_tolerance: float = 1e-10,
) -> ContactSensitivityResult:
    """Transpose implicit derivative of the same fixed-route physical residual."""
    runtime = _runtime(plan, result)
    derivative = root_solution_vjp(
        runtime.problem(),
        result.candidate.coordinates,
        arguments,
        cotangent_state,
        policy=policy,
    )
    return _qualify(derivative, result, margin_tolerance)


__all__ = [
    "ContactSensitivityResult",
    "monolithic_contact_solution_jvp",
    "monolithic_contact_solution_vjp",
]
