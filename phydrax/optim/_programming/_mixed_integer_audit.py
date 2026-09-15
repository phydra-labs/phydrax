#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax import ein

from ._mixed_integer import (
    MixedIntegerCandidate,
    MixedIntegerCandidateAudit,
    MixedIntegerProgram,
)
from ._mixed_integer_policy import MixedIntegerCertification
from ._problem import (
    _conic_matrix_mv,
    _conic_quadratic_mv,
    ConicProgram,
    LinearProgram,
)
from ._quadratic import _max_abs, QuadraticProgram


def _linear_residuals(matrix, rhs, primal, /, *, inequality: bool):
    values = ein.contract("ij,j->i", matrix, primal)
    residual = values - rhs
    if inequality:
        residual = jnp.maximum(residual, 0.0)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(values), _max_abs(rhs)),
    )
    return _max_abs(residual), scale


def audit_mixed_integer_candidate(
    program: MixedIntegerProgram,
    candidate: MixedIntegerCandidate,
    certification: MixedIntegerCertification,
    /,
) -> MixedIntegerCandidateAudit:
    """Independently replay one candidate against the canonical program."""
    if not isinstance(program, MixedIntegerProgram):
        raise TypeError("program must be a MixedIntegerProgram.")
    if not isinstance(candidate, MixedIntegerCandidate):
        raise TypeError("candidate must be a MixedIntegerCandidate.")
    if not isinstance(certification, MixedIntegerCertification):
        raise TypeError("certification must be MixedIntegerCertification.")
    relaxation = program.relaxation
    primal = jnp.asarray(candidate.primal, dtype=relaxation.linear.dtype)
    expected = (relaxation.num_variables,)
    if primal.shape != expected:
        raise ValueError(f"Candidate primal must have shape {expected}.")

    finite = jnp.all(jnp.isfinite(primal))
    bound_residual = jnp.maximum(
        jnp.maximum(relaxation.lower_bounds - primal, 0.0),
        jnp.maximum(primal - relaxation.upper_bounds, 0.0),
    )
    bound_violation = _max_abs(bound_residual)
    primal_scale = jnp.maximum(1.0, _max_abs(primal))
    bounds_valid = finite & (bound_violation <= certification.feasibility * primal_scale)

    discrete = jnp.asarray(program.discrete_indices, dtype=jnp.int32)
    discrete_values = primal[discrete]
    integrality_violation = _max_abs(discrete_values - jnp.rint(discrete_values))
    integrality_valid = finite & (integrality_violation <= certification.integrality)

    if isinstance(relaxation, LinearProgram):
        objective = ein.contract("i,i->", relaxation.linear, primal)
        equality_violation, equality_scale = _linear_residuals(
            relaxation.equality_matrix,
            relaxation.equality_rhs,
            primal,
            inequality=False,
        )
        inequality_violation, inequality_scale = _linear_residuals(
            relaxation.inequality_matrix,
            relaxation.inequality_rhs,
            primal,
            inequality=True,
        )
        cone_violation = jnp.asarray(0.0, dtype=primal.dtype)
        constraints_valid = (
            equality_violation <= certification.feasibility * equality_scale
        ) & (inequality_violation <= certification.feasibility * inequality_scale)
    elif isinstance(relaxation, QuadraticProgram):
        objective = 0.5 * ein.contract(
            "i,ij,j->", primal, relaxation.quadratic, primal
        ) + ein.contract("i,i->", relaxation.linear, primal)
        equality_violation, equality_scale = _linear_residuals(
            relaxation.equality_matrix[: relaxation.num_user_equalities],
            relaxation.equality_rhs[: relaxation.num_user_equalities],
            primal,
            inequality=False,
        )
        inequality_violation, inequality_scale = _linear_residuals(
            relaxation.inequality_matrix[: relaxation.num_user_inequalities],
            relaxation.inequality_rhs[: relaxation.num_user_inequalities],
            primal,
            inequality=True,
        )
        cone_violation = jnp.asarray(0.0, dtype=primal.dtype)
        constraints_valid = (
            equality_violation <= certification.feasibility * equality_scale
        ) & (inequality_violation <= certification.feasibility * inequality_scale)
    elif isinstance(relaxation, ConicProgram):
        matrix_value = _conic_matrix_mv(relaxation.constraint_matrix, primal)
        slack = relaxation.constraint_rhs - matrix_value
        quadratic_value = _conic_quadratic_mv(relaxation.quadratic, primal)
        objective = 0.5 * ein.contract("i,i->", primal, quadratic_value) + ein.contract(
            "i,i->", relaxation.linear, primal
        )
        equality_violation = jnp.asarray(0.0, dtype=primal.dtype)
        inequality_violation = jnp.asarray(0.0, dtype=primal.dtype)
        cone_violation = jnp.asarray(relaxation.cone.residual(slack))
        constraints_valid = cone_violation <= certification.feasibility
    else:
        raise TypeError("Unsupported canonical mixed-integer relaxation.")

    if candidate.reported_objective is None:
        objective_residual = jnp.asarray(0.0, dtype=primal.dtype)
        objective_valid = jnp.isfinite(objective)
    else:
        reported = jnp.asarray(candidate.reported_objective, dtype=primal.dtype)
        objective_residual = jnp.abs(objective - reported)
        objective_scale = jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(objective), jnp.abs(reported)),
        )
        objective_valid = jnp.isfinite(objective) & (
            objective_residual <= certification.objective * objective_scale
        )

    finite = finite & jnp.isfinite(objective)
    valid = (
        finite & bounds_valid & integrality_valid & constraints_valid & objective_valid
    )
    return MixedIntegerCandidateAudit(
        primal,
        objective,
        bound_violation,
        integrality_violation,
        equality_violation,
        inequality_violation,
        cone_violation,
        objective_residual,
        finite,
        bounds_valid,
        integrality_valid,
        constraints_valid,
        objective_valid,
        valid,
        candidate.candidate_id,
        program.structure_id,
    )


__all__ = ["audit_mixed_integer_candidate"]
