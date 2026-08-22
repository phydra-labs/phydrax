#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule


class ConvexProgramStatus(IntEnum):
    """Portable terminal status for canonical convex programs."""

    OPTIMAL = 0
    ITERATION_LIMIT = 1
    PRIMAL_INFEASIBLE = 2
    DUAL_INFEASIBLE = 3
    NONFINITE_INPUT = 4
    NONFINITE_OUTPUT = 5
    NUMERICAL_FAILURE = 6
    BACKEND_FAILED = 7
    INVALID_PROBLEM = 8


_STATUS_MESSAGES = {
    ConvexProgramStatus.OPTIMAL: "optimal solution certified",
    ConvexProgramStatus.ITERATION_LIMIT: "iteration limit reached",
    ConvexProgramStatus.PRIMAL_INFEASIBLE: "primal infeasibility certified",
    ConvexProgramStatus.DUAL_INFEASIBLE: "dual infeasibility or primal recession certified",
    ConvexProgramStatus.NONFINITE_INPUT: "program data contain non-finite values",
    ConvexProgramStatus.NONFINITE_OUTPUT: "solver output contains non-finite values",
    ConvexProgramStatus.NUMERICAL_FAILURE: "numerical method failed",
    ConvexProgramStatus.BACKEND_FAILED: "selected external backend failed",
    ConvexProgramStatus.INVALID_PROBLEM: "program violates the selected method contract",
}


def convex_program_status_message(status: int | ConvexProgramStatus, /) -> str:
    """Return the stable message for one convex-program status value."""

    return _STATUS_MESSAGES[ConvexProgramStatus(int(status))]


class ConvexProgramCapabilities(StrictModule):
    """Static capabilities declared by one mathematical-programming method."""

    linear_program: bool = eqx.field(static=True)
    quadratic_program: bool = eqx.field(static=True)
    conic_program: bool = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    sparse: bool = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    warm_start: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    infeasibility_certificates: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)
    algorithmic_differentiation: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_program: bool,
        quadratic_program: bool,
        conic_program: bool,
        dense: bool,
        sparse: bool,
        matrix_free: bool,
        warm_start: bool,
        prepared_refresh: bool,
        infeasibility_certificates: bool,
        implicit_differentiation: bool,
        algorithmic_differentiation: bool,
    ):
        self.linear_program = bool(linear_program)
        self.quadratic_program = bool(quadratic_program)
        self.conic_program = bool(conic_program)
        self.dense = bool(dense)
        self.sparse = bool(sparse)
        self.matrix_free = bool(matrix_free)
        self.warm_start = bool(warm_start)
        self.prepared_refresh = bool(prepared_refresh)
        self.infeasibility_certificates = bool(infeasibility_certificates)
        self.implicit_differentiation = bool(implicit_differentiation)
        self.algorithmic_differentiation = bool(algorithmic_differentiation)


class ConvexProgramCertificate(StrictModule):
    """Primal and dual rays with independently audited validity evidence."""

    primal_ray: Array
    equality_dual_ray: Array
    inequality_dual_ray: Array
    lower_bound_dual_ray: Array
    upper_bound_dual_ray: Array
    primal_ray_residual_norm: Array
    dual_ray_residual_norm: Array
    primal_ray_objective: Array
    dual_ray_objective: Array
    primal_ray_valid: Array
    dual_ray_valid: Array

    def __init__(
        self,
        *,
        primal_ray: Any,
        equality_dual_ray: Any,
        inequality_dual_ray: Any,
        lower_bound_dual_ray: Any,
        upper_bound_dual_ray: Any,
        primal_ray_residual_norm: Any,
        dual_ray_residual_norm: Any,
        primal_ray_objective: Any,
        dual_ray_objective: Any,
        primal_ray_valid: Any,
        dual_ray_valid: Any,
    ):
        self.primal_ray = jnp.asarray(primal_ray)
        self.equality_dual_ray = jnp.asarray(equality_dual_ray)
        self.inequality_dual_ray = jnp.asarray(inequality_dual_ray)
        self.lower_bound_dual_ray = jnp.asarray(lower_bound_dual_ray)
        self.upper_bound_dual_ray = jnp.asarray(upper_bound_dual_ray)
        self.primal_ray_residual_norm = jnp.asarray(primal_ray_residual_norm)
        self.dual_ray_residual_norm = jnp.asarray(dual_ray_residual_norm)
        self.primal_ray_objective = jnp.asarray(primal_ray_objective)
        self.dual_ray_objective = jnp.asarray(dual_ray_objective)
        self.primal_ray_valid = jnp.asarray(primal_ray_valid, dtype=bool)
        self.dual_ray_valid = jnp.asarray(dual_ray_valid, dtype=bool)


class ConvexWarmStart(StrictModule):
    """Primal and dual initialization for one fixed canonical program topology."""

    primal: Array
    equality_dual: Array
    inequality_dual: Array
    inequality_slack: Array
    lower_bound_dual: Array
    upper_bound_dual: Array
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        primal: Any,
        equality_dual: Any,
        inequality_dual: Any,
        inequality_slack: Any,
        lower_bound_dual: Any,
        upper_bound_dual: Any,
        structure_id: str,
    ):
        identifier = str(structure_id)
        if not identifier:
            raise ValueError("structure_id must be non-empty.")
        arrays = tuple(
            jnp.asarray(value)
            for value in (
                primal,
                equality_dual,
                inequality_dual,
                inequality_slack,
                lower_bound_dual,
                upper_bound_dual,
            )
        )
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
            raise TypeError("Convex warm starts must be real-valued.")
        (
            self.primal,
            self.equality_dual,
            self.inequality_dual,
            self.inequality_slack,
            self.lower_bound_dual,
            self.upper_bound_dual,
        ) = arrays
        self.structure_id = identifier

    @classmethod
    def from_result(
        cls,
        result: Any,
        /,
        *,
        interior_margin: float = 1e-8,
    ) -> "ConvexWarmStart":
        """Build an explicitly interiorized warm start from one audited result."""

        margin = float(interior_margin)
        if not jnp.isfinite(margin) or margin <= 0.0:
            raise ValueError("interior_margin must be finite and positive.")
        return cls(
            primal=result.primal,
            equality_dual=result.equality_dual,
            inequality_dual=jnp.maximum(result.inequality_dual, margin),
            inequality_slack=jnp.maximum(result.inequality_slack, margin),
            lower_bound_dual=jnp.maximum(result.lower_bound_dual, margin),
            upper_bound_dual=jnp.maximum(result.upper_bound_dual, margin),
            structure_id=result.provenance.structure_id,
        )


class ConvexProgramProvenance(StrictModule):
    """Static method identity plus observable numeric execution version."""

    numeric_version: Array
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    backend_version: str = eqx.field(static=True)
    convexity_evidence: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        numeric_version: Any,
        problem_id: str,
        structure_id: str,
        policy_id: str,
        method_id: str,
        backend: str,
        backend_version: str,
        convexity_evidence: str,
        regularization: float,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("numeric_version must be scalar.")
        strings = tuple(
            str(value)
            for value in (
                problem_id,
                structure_id,
                policy_id,
                method_id,
                backend,
                backend_version,
                convexity_evidence,
            )
        )
        if any(not value for value in strings):
            raise ValueError("Convex-program provenance strings must be non-empty.")
        self.numeric_version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        (
            self.problem_id,
            self.structure_id,
            self.policy_id,
            self.method_id,
            self.backend,
            self.backend_version,
            self.convexity_evidence,
        ) = strings
        self.regularization = float(regularization)


__all__ = [
    "ConvexWarmStart",
    "ConvexProgramCapabilities",
    "ConvexProgramCertificate",
    "ConvexProgramProvenance",
    "ConvexProgramStatus",
    "convex_program_status_message",
]
