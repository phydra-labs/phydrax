#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from .._materialization import MaterializationPolicy, materialize
from .._operators import AbstractLinearOperator
from .._policies import FailurePolicy
from .._spaces import _coordinate_dtype


SchurBackend: TypeAlias = Literal["jax-cpu"]


class SchurSolveStatus(IntEnum):
    """Portable status for a dense complex Schur decomposition."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONFINITE_OUTPUT = 2
    RESIDUAL_TOLERANCE_NOT_MET = 3
    UNITARITY_TOLERANCE_NOT_MET = 4
    BACKEND_REJECTED = 5


class SchurEigenproblem(StrictModule):
    """General unbatched coordinate eigenproblem with no normality assumption."""

    operator: AbstractLinearOperator
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError("Schur eigenproblems require an unbatched endomorphism.")
        if not jnp.issubdtype(_coordinate_dtype(operator.source), jnp.inexact):
            raise TypeError("Schur eigenproblems require real or complex coordinates.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "schur-eigenproblem",
                    "operator": operator.operator_id,
                    "source": operator.source.space_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.operator = operator
        self.problem_id = identifier

    @property
    def dimension(self) -> int:
        return self.operator.source.size


class SchurTolerancePolicy(StrictModule):
    """Backward-error and Schur-vector unitarity tolerances."""

    relative: float = eqx.field(static=True)
    absolute: float = eqx.field(static=True)
    unitarity: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative: float = 1e-8,
        absolute: float = 1e-10,
        unitarity: float = 1e-8,
    ):
        values = tuple(float(value) for value in (relative, absolute, unitarity))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Schur tolerances must be finite and non-negative.")
        self.relative, self.absolute, self.unitarity = values


class SchurResourcePolicy(StrictModule):
    """Hard retained-state and decomposition-workspace budgets."""

    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        preparation_bytes: int = 512 * 1024 * 1024,
        workspace_bytes: int = 1024 * 1024 * 1024,
    ):
        preparation = int(preparation_bytes)
        workspace = int(workspace_bytes)
        if preparation < 0 or workspace < 0:
            raise ValueError("Schur resource budgets must be non-negative.")
        self.preparation_bytes = preparation
        self.workspace_bytes = workspace


class SchurSolvePolicy(StrictModule):
    """Dense complex-Schur backend, materialization, accuracy, and failure policy."""

    backend: SchurBackend = eqx.field(static=True)
    tolerance: SchurTolerancePolicy = eqx.field(static=True)
    resources: SchurResourcePolicy = eqx.field(static=True)
    materialization: MaterializationPolicy = eqx.field(static=True)
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: SchurBackend = "jax-cpu",
        tolerance: SchurTolerancePolicy | None = None,
        resources: SchurResourcePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        failure: FailurePolicy | None = None,
    ):
        if backend != "jax-cpu":
            raise ValueError("Only backend='jax-cpu' is supported.")
        tolerance_ = SchurTolerancePolicy() if tolerance is None else tolerance
        resources_ = SchurResourcePolicy() if resources is None else resources
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(tolerance_, SchurTolerancePolicy):
            raise TypeError("tolerance must be a SchurTolerancePolicy or None.")
        if not isinstance(resources_, SchurResourcePolicy):
            raise TypeError("resources must be a SchurResourcePolicy or None.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.backend = backend
        self.tolerance = tolerance_
        self.resources = resources_
        self.materialization = materialization_
        self.failure = failure_


class SchurCostEstimate(StrictModule):
    """Static dense input, output, and conservative workspace estimate."""

    backend: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    input_matrix_bytes: int = eqx.field(static=True)
    schur_form_bytes: int = eqx.field(static=True)
    schur_vectors_bytes: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    exact_preparation: bool = eqx.field(static=True)


class SchurSolvePlan(StrictModule):
    """Immutable symbolic dense-Schur plan."""

    policy: SchurSolvePolicy = eqx.field(static=True)
    cost: SchurCostEstimate = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedSchurSolve(StrictModule):
    """Materialized numerical state reusable under one Schur plan."""

    problem: SchurEigenproblem
    matrix: Array
    plan: SchurSolvePlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array


class SchurSolveDiagnostics(StrictModule):
    """Schur relation, unitarity, nonnormality, and spectral-separation evidence."""

    column_residual_norms: Array
    residual_norm: Array
    relative_residual: Array
    unitarity_error: Array
    departure_from_normality: Array
    eigenvalue_separation: Array
    finite: Array
    converged: Array
    decomposition_count: Array
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class SchurSolveProvenance(StrictModule):
    """Backend, coordinate convention, identities, and numerical version."""

    backend: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    ordering: str = eqx.field(static=True)
    numeric_version: Array


class SchurSolveResult(StrictModule):
    """Full complex Schur form and eigenvalues without false eigenvector semantics."""

    eigenvalues: Array
    schur_form: Array
    schur_vectors: Array
    status: Array
    diagnostics: SchurSolveDiagnostics
    provenance: SchurSolveProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SchurSolveStatus.SUCCESS)


def plan_schur_eigensolve(
    problem: SchurEigenproblem,
    policy: SchurSolvePolicy | None = None,
    /,
) -> SchurSolvePlan:
    """Plan a full complex Schur decomposition with explicit CPU capability."""
    if not isinstance(problem, SchurEigenproblem):
        raise TypeError("problem must be a SchurEigenproblem.")
    selected = SchurSolvePolicy() if policy is None else policy
    if not isinstance(selected, SchurSolvePolicy):
        raise TypeError("policy must be a SchurSolvePolicy or None.")
    if jax.default_backend() != "cpu":
        raise ValueError("The JAX Schur backend is CPU-only on this runtime.")
    cost = _schur_cost(problem)
    if cost.preparation_bytes > selected.resources.preparation_bytes:
        raise ValueError(
            f"Schur preparation estimate {cost.preparation_bytes} exceeds budget "
            f"{selected.resources.preparation_bytes}."
        )
    if cost.workspace_bytes > selected.resources.workspace_bytes:
        raise ValueError(
            f"Schur workspace estimate {cost.workspace_bytes} exceeds budget "
            f"{selected.resources.workspace_bytes}."
        )
    payload = {
        "kind": "schur-solve-plan",
        "problem": problem.problem_id,
        "operator": problem.operator.operator_id,
        "backend": selected.backend,
        "dimension": problem.dimension,
        "tolerance": {
            "relative": selected.tolerance.relative,
            "absolute": selected.tolerance.absolute,
            "unitarity": selected.tolerance.unitarity,
        },
    }
    return SchurSolvePlan(
        policy=selected,
        cost=cost,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        plan_id=canonical_fingerprint(payload),
    )


def prepare_schur_eigensolve(
    problem: SchurEigenproblem,
    policy: SchurSolvePolicy | SchurSolvePlan | None = None,
    /,
) -> PreparedSchurSolve:
    """Materialize and bind a general operator to one Schur plan."""
    plan = (
        policy
        if isinstance(policy, SchurSolvePlan)
        else plan_schur_eigensolve(problem, policy)
    )
    _validate_plan(problem, plan)
    matrix = _materialize_matrix(problem, plan)
    return _prepared(
        problem,
        matrix,
        plan,
        numeric_version=0,
        refresh_count=0,
    )


def refresh_schur_eigensolve(
    prepared: PreparedSchurSolve,
    problem: SchurEigenproblem,
    /,
) -> PreparedSchurSolve:
    """Refresh matrix values while preserving one symbolic Schur plan."""
    if not isinstance(prepared, PreparedSchurSolve):
        raise TypeError("prepared must be a PreparedSchurSolve.")
    _validate_plan(problem, prepared.plan)
    matrix = _materialize_matrix(problem, prepared.plan)
    return _prepared(
        problem,
        matrix,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
    )


def schur_eigensolve(
    problem_or_prepared: SchurEigenproblem | PreparedSchurSolve,
    /,
    *,
    policy: SchurSolvePolicy | SchurSolvePlan | None = None,
) -> SchurSolveResult:
    """Compute a full complex Schur form ``A = Q T Qᴴ`` and its eigenvalues."""
    if isinstance(problem_or_prepared, PreparedSchurSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared Schur state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, SchurEigenproblem):
        prepared = prepare_schur_eigensolve(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a SchurEigenproblem or PreparedSchurSolve.")
    matrix = jax.lax.stop_gradient(prepared.matrix)
    schur_form, schur_vectors = jsp.linalg.schur(matrix, output="complex")
    schur_form = jax.lax.stop_gradient(schur_form)
    schur_vectors = jax.lax.stop_gradient(schur_vectors)
    eigenvalues = jnp.diag(schur_form)
    residual_matrix = (
        matrix.astype(schur_form.dtype) @ schur_vectors - schur_vectors @ schur_form
    )
    column_residuals = jnp.linalg.norm(residual_matrix, axis=0)
    residual = jnp.linalg.norm(residual_matrix)
    matrix_norm = jnp.linalg.norm(matrix)
    tiny = jnp.asarray(jnp.finfo(schur_form.real.dtype).tiny)
    relative = residual / jnp.maximum(matrix_norm, tiny)
    identity = jnp.eye(prepared.problem.dimension, dtype=schur_vectors.dtype)
    unitarity = jnp.linalg.norm(jnp.conj(schur_vectors.T) @ schur_vectors - identity)
    triangular_norm_squared = jnp.sum(jnp.abs(schur_form) ** 2)
    diagonal_norm_squared = jnp.sum(jnp.abs(eigenvalues) ** 2)
    departure = jnp.sqrt(
        jnp.maximum(triangular_norm_squared - diagonal_norm_squared, 0)
    ) / jnp.maximum(jnp.sqrt(triangular_norm_squared), tiny)
    separation = _eigenvalue_separation(eigenvalues)
    finite = (
        jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(schur_form))
        & jnp.all(jnp.isfinite(schur_vectors))
        & jnp.isfinite(residual)
        & jnp.isfinite(unitarity)
    )
    residual_ok = residual <= (
        prepared.plan.policy.tolerance.absolute
        + prepared.plan.policy.tolerance.relative * matrix_norm
    )
    unitarity_ok = unitarity <= prepared.plan.policy.tolerance.unitarity
    converged = finite & residual_ok & unitarity_ok
    status = jnp.where(
        ~finite,
        int(SchurSolveStatus.NONFINITE_OUTPUT),
        jnp.where(
            ~residual_ok,
            int(SchurSolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
            jnp.where(
                ~unitarity_ok,
                int(SchurSolveStatus.UNITARITY_TOLERANCE_NOT_MET),
                int(SchurSolveStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    if prepared.plan.policy.failure.mode == "error":
        schur_form = eqx.error_if(
            schur_form,
            status != int(SchurSolveStatus.SUCCESS),
            "Schur eigensolve did not satisfy its numerical contract.",
        )
    diagnostics = SchurSolveDiagnostics(
        column_residual_norms=column_residuals,
        residual_norm=residual,
        relative_residual=relative,
        unitarity_error=unitarity,
        departure_from_normality=departure,
        eigenvalue_separation=separation,
        finite=finite,
        converged=converged,
        decomposition_count=jnp.asarray(1, dtype=jnp.int32),
        preparation_bytes=prepared.plan.cost.preparation_bytes,
        workspace_bytes=prepared.plan.cost.workspace_bytes,
    )
    return SchurSolveResult(
        eigenvalues=eigenvalues,
        schur_form=schur_form,
        schur_vectors=schur_vectors,
        status=status,
        diagnostics=diagnostics,
        provenance=SchurSolveProvenance(
            backend=prepared.plan.policy.backend,
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            coordinate_convention="canonical-coordinate complex Schur form",
            ordering="backend Schur order; no eigenvalue reordering",
            numeric_version=prepared.numeric_version,
        ),
    )


def _schur_cost(problem: SchurEigenproblem, /) -> SchurCostEstimate:
    dimension = problem.dimension
    input_itemsize = _coordinate_dtype(problem.operator.source).itemsize
    output_dtype = jnp.result_type(
        _coordinate_dtype(problem.operator.source), jnp.complex64
    )
    output_itemsize = jnp.dtype(output_dtype).itemsize
    matrix_entries = dimension * dimension
    input_bytes = matrix_entries * input_itemsize
    form_bytes = matrix_entries * output_itemsize
    vectors_bytes = matrix_entries * output_itemsize
    preparation = input_bytes
    workspace = 8 * matrix_entries * output_itemsize
    return SchurCostEstimate(
        backend="jax-cpu",
        dimension=dimension,
        input_matrix_bytes=input_bytes,
        schur_form_bytes=form_bytes,
        schur_vectors_bytes=vectors_bytes,
        preparation_bytes=preparation,
        workspace_bytes=workspace,
        exact_preparation=True,
    )


def _materialize_matrix(
    problem: SchurEigenproblem,
    plan: SchurSolvePlan,
    /,
) -> Array:
    matrix = jnp.asarray(materialize(problem.operator, plan.policy.materialization))
    expected = (problem.dimension, problem.dimension)
    if matrix.shape != expected:
        raise ValueError(f"Materialized Schur operator must have shape {expected}.")
    finite = jnp.all(jnp.isfinite(matrix))
    if isinstance(finite, jax_core.Tracer):
        matrix = eqx.error_if(matrix, ~finite, "Schur input matrix must be finite.")
    elif not bool(finite):
        raise ValueError("Schur input matrix must be finite.")
    return jax.lax.stop_gradient(matrix)


def _prepared(
    problem: SchurEigenproblem,
    matrix: Array,
    plan: SchurSolvePlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
) -> PreparedSchurSolve:
    fingerprint = canonical_fingerprint(array_tree_fingerprint(problem.operator))
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-schur-solve",
                "plan": plan.plan_id,
                "operator_fingerprint": fingerprint,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedSchurSolve(
        problem=problem,
        matrix=matrix,
        plan=plan,
        prepared_id=identifier,
        operator_fingerprint=fingerprint,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


def _validate_plan(problem: SchurEigenproblem, plan: SchurSolvePlan, /) -> None:
    if not isinstance(plan, SchurSolvePlan):
        raise TypeError("plan must be a SchurSolvePlan.")
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
    ):
        raise ValueError("Schur plan belongs to a different symbolic eigenproblem.")


def _eigenvalue_separation(eigenvalues: Array, /) -> Array:
    count = eigenvalues.size
    if count == 1:
        return jnp.asarray([jnp.inf], dtype=eigenvalues.real.dtype)
    distances = jnp.abs(eigenvalues[:, None] - eigenvalues[None, :])
    diagonal = jnp.eye(count, dtype=jnp.bool_)
    return jnp.min(jnp.where(diagonal, jnp.inf, distances), axis=1)


__all__ = [
    "PreparedSchurSolve",
    "SchurBackend",
    "SchurCostEstimate",
    "SchurEigenproblem",
    "SchurResourcePolicy",
    "SchurSolveDiagnostics",
    "SchurSolvePlan",
    "SchurSolvePolicy",
    "SchurSolveProvenance",
    "SchurSolveResult",
    "SchurSolveStatus",
    "SchurTolerancePolicy",
    "plan_schur_eigensolve",
    "prepare_schur_eigensolve",
    "refresh_schur_eigensolve",
    "schur_eigensolve",
]
