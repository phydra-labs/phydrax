#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ._policies import EigenDifferentiationMode, EigenTarget


class EigenSolveStatus(IntEnum):
    SUCCESS = 0
    PARTIAL_CONVERGENCE = 1
    MAXIMUM_STEPS_REACHED = 2
    BREAKDOWN = 3
    RANK_DEFICIENT = 4
    NONFINITE_INPUT = 5
    NONFINITE_OUTPUT = 6
    INSUFFICIENT_SPACE = 7
    CAPABILITY_REJECTED = 8
    DIFFERENTIATION_REJECTED = 9


class EigenSolveDiagnostics(StrictModule):
    """JAX-compatible per-mode residual, convergence, and work evidence."""

    residual_norms: Array
    relative_residuals: Array
    orthogonality_error: Array
    iterations: Array
    operator_matvec_count: Array
    metric_matvec_count: Array
    preconditioner_apply_count: Array
    converged: Array
    mode_mask: Array
    effective_count: Array
    isolation_gaps: Array
    initial_rank: Array

    def __init__(
        self,
        residual_norms: Any,
        relative_residuals: Any,
        orthogonality_error: Any,
        iterations: Any,
        operator_matvec_count: Any,
        metric_matvec_count: Any,
        preconditioner_apply_count: Any,
        converged: Any,
        mode_mask: Any,
        effective_count: Any,
        isolation_gaps: Any,
        initial_rank: Any,
        /,
    ):
        residuals = jnp.asarray(residual_norms)
        relative = jnp.asarray(relative_residuals)
        converged_ = jnp.asarray(converged, dtype=bool)
        mask = jnp.asarray(mode_mask, dtype=bool)
        gaps = jnp.asarray(isolation_gaps)
        if residuals.ndim != 1:
            raise ValueError("residual_norms must be rank one.")
        shape = residuals.shape
        if any(value.shape != shape for value in (relative, converged_, mask, gaps)):
            raise ValueError("Per-mode eigen diagnostics must have identical shapes.")
        orthogonality = jnp.asarray(orthogonality_error)
        iterations_ = jnp.asarray(iterations, dtype=jnp.int32)
        operator_count = jnp.asarray(operator_matvec_count, dtype=jnp.int32)
        metric_count = jnp.asarray(metric_matvec_count, dtype=jnp.int32)
        preconditioner_count = jnp.asarray(preconditioner_apply_count, dtype=jnp.int32)
        effective = jnp.asarray(effective_count, dtype=jnp.int32)
        initial_rank_ = jnp.asarray(initial_rank, dtype=jnp.int32)
        scalars = (
            orthogonality,
            iterations_,
            operator_count,
            metric_count,
            preconditioner_count,
            effective,
            initial_rank_,
        )
        if any(value.shape != () for value in scalars):
            raise ValueError("Aggregate eigen diagnostics must be scalar.")
        self.residual_norms = residuals
        self.relative_residuals = relative
        self.orthogonality_error = orthogonality
        self.iterations = iterations_
        self.operator_matvec_count = operator_count
        self.metric_matvec_count = metric_count
        self.preconditioner_apply_count = preconditioner_count
        self.converged = converged_
        self.mode_mask = mask
        self.effective_count = effective
        self.isolation_gaps = gaps
        self.initial_rank = initial_rank_


class EigenSolveProvenance(StrictModule):
    """Static method selection, identities, versions, and rejected candidates."""

    method: str = eqx.field(static=True)
    which: EigenTarget = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    rejections: tuple[str, ...] = eqx.field(static=True)
    differentiation: EigenDifferentiationMode = eqx.field(static=True)
    symbolic_version: int = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        method: str,
        which: EigenTarget,
        problem_id: str,
        plan_id: str,
        rejections: tuple[str, ...],
        differentiation: EigenDifferentiationMode,
        symbolic_version: int,
        numeric_version: int,
        /,
    ):
        method_, problem_id_, plan_id_ = str(method), str(problem_id), str(plan_id)
        if not method_ or not problem_id_ or not plan_id_:
            raise ValueError("Eigen provenance identifiers must be non-empty.")
        if which not in (
            "smallest-algebraic",
            "largest-algebraic",
            "smallest-magnitude",
            "largest-magnitude",
        ):
            raise ValueError("Unknown eigen provenance target.")
        if differentiation not in ("none", "eigenvalues"):
            raise ValueError("Unknown eigen provenance differentiation mode.")
        rejections_ = tuple(str(value) for value in rejections)
        if any(not value for value in rejections_):
            raise ValueError("Eigen provenance rejection reasons must be non-empty.")
        symbolic, numeric = int(symbolic_version), int(numeric_version)
        if symbolic < 1 or numeric < 0:
            raise ValueError("Eigen provenance versions are invalid.")
        self.method = method_
        self.which = which
        self.problem_id = problem_id_
        self.plan_id = plan_id_
        self.rejections = rejections_
        self.differentiation = differentiation
        self.symbolic_version = symbolic
        self.numeric_version = numeric


class EigenSolveResult(StrictModule):
    """Eigenpairs plus fixed-shape validity masks and complete solve evidence."""

    eigenvalues: Array
    eigenvectors: PyTree[Array]
    mode_mask: Array
    effective_count: Array
    converged: Array
    status: Array
    diagnostics: EigenSolveDiagnostics
    provenance: EigenSolveProvenance

    def __init__(
        self,
        eigenvalues: Any,
        eigenvectors: PyTree[Array],
        mode_mask: Any,
        effective_count: Any,
        converged: Any,
        status: Any,
        diagnostics: EigenSolveDiagnostics,
        provenance: EigenSolveProvenance,
        /,
    ):
        values = jnp.asarray(eigenvalues)
        mask = jnp.asarray(mode_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape:
            raise ValueError(
                "eigenvalues and mode_mask must have one matching mode axis."
            )
        effective = jnp.asarray(effective_count, dtype=jnp.int32)
        converged_ = jnp.asarray(converged, dtype=bool)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if effective.shape != () or status_.shape != ():
            raise ValueError("effective_count and status must be scalar.")
        if converged_.shape != values.shape:
            raise ValueError("converged must provide one flag per requested mode.")
        if not isinstance(diagnostics, EigenSolveDiagnostics):
            raise TypeError("diagnostics must be EigenSolveDiagnostics.")
        if not isinstance(provenance, EigenSolveProvenance):
            raise TypeError("provenance must be EigenSolveProvenance.")
        if diagnostics.mode_mask.shape != values.shape:
            raise ValueError("Result and diagnostic mode capacities must match.")
        self.eigenvalues = values
        self.eigenvectors = eigenvectors
        self.mode_mask = mask
        self.effective_count = effective
        self.converged = converged_
        self.status = status_
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def residual_norms(self) -> Array:
        return self.diagnostics.residual_norms

    @property
    def relative_residuals(self) -> Array:
        return self.diagnostics.relative_residuals

    @property
    def orthogonality_error(self) -> Array:
        return self.diagnostics.orthogonality_error

    @property
    def iterations(self) -> Array:
        return self.diagnostics.iterations

    @property
    def operator_matvec_count(self) -> Array:
        return self.diagnostics.operator_matvec_count

    @property
    def metric_matvec_count(self) -> Array:
        return self.diagnostics.metric_matvec_count

    @property
    def preconditioner_apply_count(self) -> Array:
        return self.diagnostics.preconditioner_apply_count

    @property
    def successful(self) -> Array:
        return self.status == int(EigenSolveStatus.SUCCESS)


class _NativeEigenResult(NamedTuple):
    values: Array
    vectors: Array
    mode_mask: Array
    converged: Array
    residual_norms: Array
    relative_residuals: Array
    orthogonality_error: Array
    iterations: Array
    operator_matvec_count: Array
    metric_matvec_count: Array
    preconditioner_apply_count: Array
    isolation_gaps: Array
    rank_deficient: Array


__all__ = [
    "EigenSolveDiagnostics",
    "EigenSolveProvenance",
    "EigenSolveResult",
    "EigenSolveStatus",
]
