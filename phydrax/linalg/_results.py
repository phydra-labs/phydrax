#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class LinearSolveStatus(IntEnum):
    SUCCESS = 0
    MAXIMUM_STEPS_REACHED = 1
    STAGNATION = 2
    BREAKDOWN = 3
    SINGULAR = 4
    RANK_DEFICIENT = 5
    NONFINITE_INPUT = 6
    NONFINITE_OUTPUT = 7
    RESIDUAL_TOO_LARGE = 8
    CAPABILITY_REJECTED = 9
    INCOMPATIBLE_STRUCTURE = 10
    ADJOINT_FAILED = 11
    CONDITION_LIMIT_REACHED = 12


_STATUS_MESSAGES = {
    LinearSolveStatus.SUCCESS: "success",
    LinearSolveStatus.MAXIMUM_STEPS_REACHED: "maximum steps reached",
    LinearSolveStatus.STAGNATION: "iteration stagnated",
    LinearSolveStatus.BREAKDOWN: "iterative breakdown",
    LinearSolveStatus.SINGULAR: "operator is singular or numerically singular",
    LinearSolveStatus.RANK_DEFICIENT: "declared rank policy was not satisfied",
    LinearSolveStatus.NONFINITE_INPUT: "operator or right-hand side is non-finite",
    LinearSolveStatus.NONFINITE_OUTPUT: "solution contains non-finite values",
    LinearSolveStatus.RESIDUAL_TOO_LARGE: "true residual exceeds the requested tolerance",
    LinearSolveStatus.CAPABILITY_REJECTED: "operator lacks a required capability",
    LinearSolveStatus.INCOMPATIBLE_STRUCTURE: "problem structure is incompatible",
    LinearSolveStatus.ADJOINT_FAILED: "adjoint solve failed",
    LinearSolveStatus.CONDITION_LIMIT_REACHED: "condition limit reached",
}


def linear_status_message(status: int | LinearSolveStatus, /) -> str:
    return _STATUS_MESSAGES[LinearSolveStatus(int(status))]


class LinearSolveDiagnostics(StrictModule):
    """JAX-compatible evidence retained per operator batch or right-hand side."""

    residual_norm: Array
    relative_residual: Array
    normal_residual_norm: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    finite: Array
    converged: Array
    singular_values: Array | None
    compatibility_residual: Array
    gauge_residual: Array
    nullity: Array
    matvec_count: Array
    adjoint_matvec_count: Array

    def __init__(
        self,
        *,
        residual_norm: Any,
        relative_residual: Any,
        normal_residual_norm: Any = jnp.nan,
        iterations: Any = 0,
        rank: Any = -1,
        condition_estimate: Any = jnp.nan,
        finite: Any = True,
        converged: Any = True,
        singular_values: Any | None = None,
        compatibility_residual: Any = 0.0,
        gauge_residual: Any = 0.0,
        nullity: Any = -1,
        matvec_count: Any = 0,
        adjoint_matvec_count: Any = 0,
    ):
        self.residual_norm = jnp.asarray(residual_norm)
        self.relative_residual = jnp.asarray(relative_residual)
        self.normal_residual_norm = jnp.asarray(normal_residual_norm)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.compatibility_residual = jnp.asarray(compatibility_residual)
        self.gauge_residual = jnp.asarray(gauge_residual)
        self.nullity = jnp.asarray(nullity, dtype=jnp.int32)
        self.matvec_count = jnp.asarray(matvec_count, dtype=jnp.int32)
        self.adjoint_matvec_count = jnp.asarray(adjoint_matvec_count, dtype=jnp.int32)
        self.singular_values = (
            None if singular_values is None else jnp.asarray(singular_values)
        )


class LinearSolveProvenance(StrictModule):
    """Static provider selection, candidate rejections, and plan identity."""

    backend: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    rejected: tuple[str, ...] = eqx.field(static=True)
    prepared: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: str,
        method: str,
        plan_id: str,
        problem_id: str,
        reason: str,
        rejected: tuple[str, ...] = (),
        prepared: bool,
    ):
        values = (
            str(backend),
            str(method),
            str(plan_id),
            str(problem_id),
            str(reason),
        )
        if any(not value for value in values):
            raise ValueError("Solve provenance identifiers and reason must be non-empty.")
        (
            self.backend,
            self.method,
            self.plan_id,
            self.problem_id,
            self.reason,
        ) = values
        self.rejected = tuple(str(value) for value in rejected)
        self.prepared = bool(prepared)


class LinearSolveResult(StrictModule):
    """Numerical value plus portable status, diagnostics, and provenance."""

    value: PyTree[Array]
    status: Array
    diagnostics: LinearSolveDiagnostics
    provenance: LinearSolveProvenance

    def __init__(
        self,
        value: PyTree[Array],
        status: Any,
        diagnostics: LinearSolveDiagnostics,
        provenance: LinearSolveProvenance,
        /,
    ):
        if not isinstance(diagnostics, LinearSolveDiagnostics):
            raise TypeError("diagnostics must be LinearSolveDiagnostics.")
        if not isinstance(provenance, LinearSolveProvenance):
            raise TypeError("provenance must be LinearSolveProvenance.")
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.status == int(LinearSolveStatus.SUCCESS)


__all__ = [
    "LinearSolveDiagnostics",
    "LinearSolveProvenance",
    "LinearSolveResult",
    "LinearSolveStatus",
    "linear_status_message",
]
