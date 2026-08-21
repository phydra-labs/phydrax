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
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._matrix_equations import MatrixEquationProblem
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._pairings import EuclideanPairing
from ._policies import FailurePolicy
from ._shifted import (
    plan_shifted_solve,
    ShiftedLinearSystemFamily,
    ShiftedSolvePlan,
    ShiftedSolvePolicy,
    ShiftedSolveStatus,
    solve_shifted,
)
from ._spaces import _coordinate_dtype, ArraySpace


FactoredMatrixSolutionForm: TypeAlias = Literal[
    "hermitian-positive-semidefinite",
    "general",
]


class FactoredMatrixEquationStatus(IntEnum):
    """Aggregate status for a factored matrix-equation solve."""

    SUCCESS = 0
    SHIFTED_SOLVE_FAILURE = 1
    NONFINITE = 2
    RESIDUAL_TOLERANCE_NOT_MET = 3


class FactoredMatrixSolution(StrictModule):
    """Fixed-capacity low-rank ``U V*`` using its leading ``rank`` columns."""

    left_factor: Array
    right_factor: Array
    rank: Array
    form: FactoredMatrixSolutionForm = eqx.field(static=True)

    def __init__(
        self,
        left_factor: ArrayLike,
        right_factor: ArrayLike | None = None,
        /,
        *,
        rank: ArrayLike | None = None,
        hermitian_positive_semidefinite: bool = False,
    ):
        left = jnp.asarray(left_factor)
        if left.ndim != 2:
            raise ValueError("left_factor must have shape (m, capacity).")
        psd = bool(hermitian_positive_semidefinite)
        if psd:
            if right_factor is not None:
                raise ValueError(
                    "right_factor must be omitted for a Hermitian positive semidefinite factor."
                )
            right = left
            form: FactoredMatrixSolutionForm = "hermitian-positive-semidefinite"
        else:
            if right_factor is None:
                raise ValueError(
                    "right_factor is required for a general factored solution."
                )
            right = jnp.asarray(right_factor)
            if right.ndim != 2 or right.shape[1] != left.shape[1]:
                raise ValueError(
                    "right_factor must have shape (n, capacity) with the same capacity."
                )
            dtype = jnp.result_type(left.dtype, right.dtype)
            left = left.astype(dtype)
            right = right.astype(dtype)
            form = "general"
        rank_value = jnp.asarray(
            left.shape[1] if rank is None else rank,
            dtype=jnp.int32,
        )
        if rank_value.ndim != 0:
            raise ValueError("rank must be scalar.")
        invalid_rank = (rank_value < 0) | (rank_value > left.shape[1])
        if isinstance(invalid_rank, jax_core.Tracer):
            rank_value = eqx.error_if(
                rank_value,
                invalid_rank,
                "rank must be between zero and the factor capacity.",
            )
        elif bool(invalid_rank):
            raise ValueError("rank must be between zero and the factor capacity.")
        self.left_factor = left
        self.right_factor = right
        self.rank = rank_value
        self.form = form

    @property
    def capacity(self) -> int:
        return int(self.left_factor.shape[1])

    @property
    def hermitian_positive_semidefinite(self) -> bool:
        return self.form == "hermitian-positive-semidefinite"

    @property
    def factor(self) -> Array:
        """Return ``Z`` for the Hermitian positive semidefinite form."""
        if not self.hermitian_positive_semidefinite:
            raise ValueError("A general U V* solution has no single PSD factor.")
        return self.left_factor

    def to_dense(self, /) -> Array:
        """Explicitly reconstruct the represented matrix."""
        active = jnp.arange(self.capacity) < self.rank
        left = jnp.where(active[None, :], self.left_factor, 0)
        right = jnp.where(active[None, :], self.right_factor, 0)
        return left @ jnp.conj(right.T)


class FactoredMatrixEquationProblem(StrictModule):
    """Continuous Lyapunov equation ``A X + X A* = -B B*``."""

    operator: AbstractLinearOperator
    source_factor: Array
    problem_id: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator | ArrayLike,
        source_factor: ArrayLike,
        /,
        *,
        problem_id: str | None = None,
    ):
        operator_ = (
            operator
            if isinstance(operator, AbstractLinearOperator)
            else DenseLinearOperator(operator)
        )
        _validate_factored_operator(operator_)
        factor = jnp.asarray(source_factor)
        if factor.ndim != 2 or factor.shape[0] != operator_.source.size:
            raise ValueError(
                "source_factor must have shape (operator dimension, source rank)."
            )
        if factor.shape[1] < 1:
            raise ValueError("source_factor must contain at least one column.")
        if not jnp.issubdtype(factor.dtype, jnp.number) or jnp.issubdtype(
            factor.dtype, jnp.bool_
        ):
            raise TypeError("source_factor must contain real or complex values.")
        coordinate_dtype = _coordinate_dtype(operator_.source)
        promoted_dtype = np.dtype(jnp.result_type(coordinate_dtype, factor.dtype))
        if promoted_dtype != coordinate_dtype:
            raise TypeError(
                "source_factor cannot promote the operator coordinate dtype; "
                f"expected values representable as {coordinate_dtype}."
            )
        factor = factor.astype(coordinate_dtype)
        finite = jnp.all(jnp.isfinite(factor))
        if isinstance(finite, jax_core.Tracer):
            factor = eqx.error_if(
                factor,
                ~finite,
                "Factored Lyapunov source factor must be finite.",
            )
        elif not bool(finite):
            raise ValueError("Factored Lyapunov source factor must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "factored-continuous-lyapunov-problem",
                    "operator": operator_.operator_id,
                    "space": operator_.source.space_id,
                    "source_rank": int(factor.shape[1]),
                    "dtype": coordinate_dtype.str,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.operator = operator_
        self.source_factor = factor
        self.problem_id = identifier
        self.kind = "continuous-lyapunov"

    @property
    def dimension(self) -> int:
        return self.operator.source.size

    @property
    def source_rank(self) -> int:
        return int(self.source_factor.shape[1])


class FactoredMatrixEquationPolicy(StrictModule):
    """Low-rank ADI shifts, delegated Krylov solves, and truncation tolerances."""

    shifts: tuple[float | complex, ...] = eqx.field(static=True)
    shifted: ShiftedSolvePolicy = eqx.field(static=True)
    relative_truncation_tolerance: float = eqx.field(static=True)
    absolute_truncation_tolerance: float = eqx.field(static=True)
    maximum_rank: int | None = eqx.field(static=True)
    relative_residual_tolerance: float = eqx.field(static=True)
    absolute_residual_tolerance: float = eqx.field(static=True)
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        shifts: Any,
        /,
        *,
        shifted: ShiftedSolvePolicy | None = None,
        relative_truncation_tolerance: float = 1e-10,
        absolute_truncation_tolerance: float = 0.0,
        maximum_rank: int | None = None,
        relative_residual_tolerance: float = 1e-6,
        absolute_residual_tolerance: float = 1e-10,
        failure: FailurePolicy | None = None,
    ):
        values = np.asarray(tuple(shifts))
        if values.ndim != 1 or values.size < 1:
            raise ValueError("shifts must be one nonempty rank-one sequence.")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise TypeError("shifts must contain real or complex numbers.")
        if not np.all(np.isfinite(values)):
            raise ValueError("ADI shifts must be finite.")
        if np.any(np.real(values) >= 0.0):
            raise ValueError(
                "Continuous low-rank ADI shifts must lie in the open left half-plane."
            )
        shifted_ = ShiftedSolvePolicy() if shifted is None else shifted
        if not isinstance(shifted_, ShiftedSolvePolicy):
            raise TypeError("shifted must be a ShiftedSolvePolicy or None.")
        relative_truncation = _nonnegative_finite(
            relative_truncation_tolerance,
            "relative_truncation_tolerance",
        )
        absolute_truncation = _nonnegative_finite(
            absolute_truncation_tolerance,
            "absolute_truncation_tolerance",
        )
        relative_residual = _nonnegative_finite(
            relative_residual_tolerance,
            "relative_residual_tolerance",
        )
        absolute_residual = _nonnegative_finite(
            absolute_residual_tolerance,
            "absolute_residual_tolerance",
        )
        if maximum_rank is None:
            maximum_rank_ = None
        else:
            maximum_rank_ = int(maximum_rank)
            if maximum_rank_ < 0:
                raise ValueError("maximum_rank must be non-negative or None.")
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.shifts = tuple(value.item() for value in values)
        self.shifted = shifted_
        self.relative_truncation_tolerance = relative_truncation
        self.absolute_truncation_tolerance = absolute_truncation
        self.maximum_rank = maximum_rank_
        self.relative_residual_tolerance = relative_residual
        self.absolute_residual_tolerance = absolute_residual
        self.failure = failure_


class FactoredMatrixEquationCostEstimate(StrictModule):
    """Low-rank ADI setup, factor storage, and avoided dense-solution costs."""

    dimension: int = eqx.field(static=True)
    source_rank: int = eqx.field(static=True)
    adi_steps: int = eqx.field(static=True)
    raw_rank_capacity: int = eqx.field(static=True)
    shifted_solve_count: int = eqx.field(static=True)
    shifted_setup_matvec_count: int = eqx.field(static=True)
    residual_matvec_count: int = eqx.field(static=True)
    factor_storage_bytes: int = eqx.field(static=True)
    source_factor_storage_bytes: int = eqx.field(static=True)
    shifted_basis_storage_bytes: int = eqx.field(static=True)
    shifted_workspace_bytes: int = eqx.field(static=True)
    small_matrix_workspace_bytes: int = eqx.field(static=True)
    explicit_solution_bytes: int = eqx.field(static=True)
    selected_method: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class FactoredMatrixEquationPlan(StrictModule):
    """Symbolic low-rank ADI plan retaining one reusable shifted-solve plan."""

    shifted_plan: ShiftedSolvePlan = eqx.field(static=True)
    policy: FactoredMatrixEquationPolicy = eqx.field(static=True)
    cost: FactoredMatrixEquationCostEstimate = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    source_rank: int = eqx.field(static=True)
    shift_dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedFactoredMatrixEquation(StrictModule):
    """Numerically bound factored equation with a reusable symbolic ADI plan."""

    problem: FactoredMatrixEquationProblem
    plan: FactoredMatrixEquationPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array


class FactoredMatrixEquationResidualCertificate(StrictModule):
    """Original-equation Frobenius residual from a low-rank Gram identity."""

    residual_norm: Array
    relative_residual: Array
    forcing_norm: Array
    valid: Array
    equation: str = eqx.field(static=True)
    norm: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class FactoredMatrixEquationDiagnostics(StrictModule):
    """Rank, truncation, shifted-solve, storage, and convergence evidence."""

    rank: Array
    raw_rank: Array
    truncation_loss: Array
    relative_truncation_loss: Array
    shifted_status: Array
    shifted_residual_norm: Array
    shifted_relative_residual: Array
    shifted_iterations: Array
    shifted_condition_estimate: Array
    setup_matvec_count: Array
    residual_matvec_count: Array
    retained_factor_storage_bytes: Array
    finite: Array
    converged: Array
    factor_storage_bytes: int = eqx.field(static=True)
    source_factor_storage_bytes: int = eqx.field(static=True)


class FactoredMatrixEquationProvenance(StrictModule):
    """Equation, ADI shifts, delegated method, identities, and materialization evidence."""

    shifts: Array
    numeric_version: Array
    kind: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    algorithm: str = eqx.field(static=True)
    shifted_method: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    source_rank: int = eqx.field(static=True)
    operator_materialized: bool = eqx.field(static=True)
    solution_materialized: bool = eqx.field(static=True)


class FactoredMatrixEquationResult(StrictModule):
    """Factored solution and complete numerical evidence without a dense value."""

    solution: FactoredMatrixSolution
    status: Array
    diagnostics: FactoredMatrixEquationDiagnostics
    certificate: FactoredMatrixEquationResidualCertificate
    provenance: FactoredMatrixEquationProvenance
    cost: FactoredMatrixEquationCostEstimate = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(FactoredMatrixEquationStatus.SUCCESS)

    @property
    def rank(self) -> Array:
        return self.solution.rank

    @property
    def truncation_loss(self) -> Array:
        return self.diagnostics.truncation_loss

    @property
    def shifted_statuses(self) -> Array:
        return self.diagnostics.shifted_status

    @property
    def residual_estimate(self) -> Array:
        return self.certificate.residual_norm


def factored_continuous_lyapunov_equation(
    operator: AbstractLinearOperator | ArrayLike,
    source_factor: ArrayLike,
    /,
    *,
    problem_id: str | None = None,
) -> FactoredMatrixEquationProblem:
    """Construct ``A X + X A* = -B B*`` without forming ``B B*``."""
    return FactoredMatrixEquationProblem(
        operator,
        source_factor,
        problem_id=problem_id,
    )


def plan_factored_matrix_equation(
    problem: FactoredMatrixEquationProblem,
    policy: FactoredMatrixEquationPolicy,
    /,
) -> FactoredMatrixEquationPlan:
    """Plan low-rank ADI and its reusable one-shift Krylov solve."""
    _reject_dense_or_unsupported_problem(problem)
    if not isinstance(policy, FactoredMatrixEquationPolicy):
        raise TypeError("policy must be a FactoredMatrixEquationPolicy.")
    shifts = _adi_shifts(problem, policy)
    family = _shift_family(problem, shifts[0], None)
    shifted_plan = plan_shifted_solve(family, policy.shifted)
    cost = _factored_cost(problem, policy, shifted_plan, shifts.dtype)
    plan_id = canonical_fingerprint(
        {
            "kind": "factored-matrix-equation-plan",
            "problem": problem.problem_id,
            "operator": problem.operator.operator_id,
            "source_space": problem.operator.source.space_id,
            "source_rank": problem.source_rank,
            "shifts": policy.shifts,
            "shifted_plan": shifted_plan.plan_id,
            "maximum_rank": policy.maximum_rank,
            "relative_truncation_tolerance": policy.relative_truncation_tolerance,
            "absolute_truncation_tolerance": policy.absolute_truncation_tolerance,
            "relative_residual_tolerance": policy.relative_residual_tolerance,
            "absolute_residual_tolerance": policy.absolute_residual_tolerance,
        }
    )
    return FactoredMatrixEquationPlan(
        shifted_plan=shifted_plan,
        policy=policy,
        cost=cost,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        source_space_id=problem.operator.source.space_id,
        source_rank=problem.source_rank,
        shift_dtype=np.dtype(shifts.dtype).str,
        plan_id=plan_id,
    )


def prepare_factored_matrix_equation(
    problem: FactoredMatrixEquationProblem,
    policy: FactoredMatrixEquationPolicy | FactoredMatrixEquationPlan,
    /,
) -> PreparedFactoredMatrixEquation:
    """Bind numerical operator and source-factor state to a factored plan."""
    _reject_dense_or_unsupported_problem(problem)
    plan = (
        policy
        if isinstance(policy, FactoredMatrixEquationPlan)
        else plan_factored_matrix_equation(problem, policy)
    )
    _validate_factored_plan(problem, plan)
    return PreparedFactoredMatrixEquation(
        problem=problem,
        plan=plan,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-factored-matrix-equation",
                "plan": plan.plan_id,
                "operator": canonical_fingerprint(
                    array_tree_fingerprint(problem.operator)
                ),
                "source_factor": canonical_fingerprint(
                    array_tree_fingerprint(problem.source_factor)
                ),
            }
        ),
        numeric_version=jnp.asarray(0, dtype=jnp.int32),
        refresh_count=jnp.asarray(0, dtype=jnp.int32),
    )


def refresh_factored_matrix_equation(
    prepared: PreparedFactoredMatrixEquation,
    problem: FactoredMatrixEquationProblem,
    /,
) -> PreparedFactoredMatrixEquation:
    """Refresh numerical operator/source state without rebuilding the shifted plan."""
    if not isinstance(prepared, PreparedFactoredMatrixEquation):
        raise TypeError("prepared must be a PreparedFactoredMatrixEquation.")
    _reject_dense_or_unsupported_problem(problem)
    _validate_factored_plan(problem, prepared.plan)
    return PreparedFactoredMatrixEquation(
        problem=problem,
        plan=prepared.plan,
        prepared_id=prepared.prepared_id,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
    )


def solve_factored_matrix_equation(
    problem_or_prepared: FactoredMatrixEquationProblem
    | PreparedFactoredMatrixEquation
    | MatrixEquationProblem,
    /,
    *,
    policy: FactoredMatrixEquationPolicy | FactoredMatrixEquationPlan | None = None,
) -> FactoredMatrixEquationResult:
    """Solve a factored continuous Lyapunov equation by low-rank ADI."""
    if isinstance(problem_or_prepared, PreparedFactoredMatrixEquation):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared factored equation.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, FactoredMatrixEquationProblem):
        if policy is None:
            raise ValueError(
                "An explicit FactoredMatrixEquationPolicy with left-half-plane ADI shifts is required."
            )
        prepared = prepare_factored_matrix_equation(problem_or_prepared, policy)
    elif isinstance(problem_or_prepared, MatrixEquationProblem):
        raise NotImplementedError(
            "Factored solves currently support only a factored continuous Lyapunov "
            "problem A X + X A* = -B B*; dense/generalized/Sylvester/discrete "
            "problems are not converted or materialized implicitly."
        )
    else:
        raise TypeError(
            "Expected a FactoredMatrixEquationProblem or PreparedFactoredMatrixEquation."
        )
    return _execute_factored(prepared)


def _execute_factored(
    prepared: PreparedFactoredMatrixEquation,
    /,
) -> FactoredMatrixEquationResult:
    problem = prepared.problem
    plan = prepared.plan
    policy = plan.policy
    shifts = _adi_shifts(problem, policy)
    residual_factor = problem.source_factor
    blocks = []
    status_rows = []
    residual_rows = []
    relative_rows = []
    iteration_rows = []
    condition_rows = []
    setup_matvec_count = jnp.asarray(0, dtype=jnp.int32)
    for step in range(len(policy.shifts)):
        shift = shifts[step]
        family = _shift_family(problem, shift, plan.shifted_plan.family_id)
        solve_columns = []
        column_statuses = []
        column_residuals = []
        column_relatives = []
        column_iterations = []
        column_conditions = []
        for column in range(problem.source_rank):
            shifted = solve_shifted(
                family,
                residual_factor[:, column],
                policy=plan.shifted_plan,
            )
            solve_columns.append(-jnp.asarray(shifted.value)[0])
            column_statuses.append(shifted.status[0])
            column_residuals.append(shifted.diagnostics.residual_norm[0])
            column_relatives.append(shifted.diagnostics.relative_residual[0])
            column_iterations.append(shifted.diagnostics.iterations[0])
            column_conditions.append(shifted.diagnostics.condition_estimate[0])
            setup_matvec_count = (
                setup_matvec_count + shifted.diagnostics.setup_matvec_count
            )
        inverse_action = jnp.stack(solve_columns, axis=1)
        scale = jnp.sqrt(-2.0 * jnp.real(shift)).astype(inverse_action.dtype)
        blocks.append(scale * inverse_action)
        residual_factor = (
            residual_factor
            - (2.0 * jnp.real(shift)).astype(inverse_action.dtype) * inverse_action
        )
        status_rows.append(jnp.stack(column_statuses))
        residual_rows.append(jnp.stack(column_residuals))
        relative_rows.append(jnp.stack(column_relatives))
        iteration_rows.append(jnp.stack(column_iterations))
        condition_rows.append(jnp.stack(column_conditions))
    raw_factor = jnp.concatenate(blocks, axis=1)
    factor, rank, raw_rank, truncation_loss, relative_truncation_loss = _compress_factor(
        raw_factor,
        policy,
    )
    certificate = _residual_certificate(problem, factor)
    shifted_status = jnp.stack(status_rows)
    shifted_residual = jnp.stack(residual_rows)
    shifted_relative = jnp.stack(relative_rows)
    shifted_iterations = jnp.stack(iteration_rows)
    shifted_conditions = jnp.stack(condition_rows)
    shifted_success = jnp.all(shifted_status == int(ShiftedSolveStatus.SUCCESS))
    factor_finite = jnp.all(jnp.isfinite(factor))
    finite = factor_finite & certificate.valid
    residual_satisfied = certificate.residual_norm <= (
        policy.absolute_residual_tolerance
        + policy.relative_residual_tolerance * certificate.forcing_norm
    )
    converged = finite & shifted_success & residual_satisfied
    status = jnp.where(
        ~finite,
        int(FactoredMatrixEquationStatus.NONFINITE),
        jnp.where(
            ~shifted_success,
            int(FactoredMatrixEquationStatus.SHIFTED_SOLVE_FAILURE),
            jnp.where(
                ~residual_satisfied,
                int(FactoredMatrixEquationStatus.RESIDUAL_TOLERANCE_NOT_MET),
                int(FactoredMatrixEquationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    if policy.failure.mode == "error":
        factor = eqx.error_if(
            factor,
            status != int(FactoredMatrixEquationStatus.SUCCESS),
            "Factored matrix equation did not satisfy its numerical contract.",
        )
    solution = FactoredMatrixSolution(
        factor,
        rank=rank,
        hermitian_positive_semidefinite=True,
    )
    retained_storage = rank * jnp.asarray(
        problem.dimension * factor.dtype.itemsize,
        dtype=jnp.int32,
    )
    diagnostics = FactoredMatrixEquationDiagnostics(
        rank=rank,
        raw_rank=raw_rank,
        truncation_loss=truncation_loss,
        relative_truncation_loss=relative_truncation_loss,
        shifted_status=shifted_status,
        shifted_residual_norm=shifted_residual,
        shifted_relative_residual=shifted_relative,
        shifted_iterations=shifted_iterations,
        shifted_condition_estimate=shifted_conditions,
        setup_matvec_count=setup_matvec_count,
        residual_matvec_count=jnp.asarray(factor.shape[1], dtype=jnp.int32),
        retained_factor_storage_bytes=retained_storage,
        finite=finite,
        converged=converged,
        factor_storage_bytes=plan.cost.factor_storage_bytes,
        source_factor_storage_bytes=plan.cost.source_factor_storage_bytes,
    )
    return FactoredMatrixEquationResult(
        solution=solution,
        status=status,
        diagnostics=diagnostics,
        certificate=certificate,
        provenance=FactoredMatrixEquationProvenance(
            shifts=shifts,
            numeric_version=prepared.numeric_version,
            kind=problem.kind,
            convention="A X + X A* = -B B*; X approximately Z Z*",
            algorithm="low-rank ADI with Gram-factor compression",
            shifted_method=plan.shifted_plan.selected_method,
            problem_id=problem.problem_id,
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=problem.operator.operator_id,
            source_rank=problem.source_rank,
            operator_materialized=False,
            solution_materialized=False,
        ),
        cost=plan.cost,
    )


def _compress_factor(
    raw_factor: Array,
    policy: FactoredMatrixEquationPolicy,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    gram = jnp.conj(raw_factor.T) @ raw_factor
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(jnp.real(eigenvalues[::-1]), 0)
    eigenvectors = eigenvectors[:, ::-1]
    largest = eigenvalues[0]
    threshold = jnp.maximum(
        policy.absolute_truncation_tolerance,
        policy.relative_truncation_tolerance * largest,
    )
    capacity = raw_factor.shape[1]
    maximum_rank = (
        capacity
        if policy.maximum_rank is None
        else min(
            policy.maximum_rank,
            capacity,
        )
    )
    retained = (jnp.arange(capacity) < maximum_rank) & (eigenvalues > threshold)
    rotated = raw_factor @ eigenvectors
    factor = rotated * retained.astype(rotated.dtype)[None, :]
    rank = jnp.sum(retained, dtype=jnp.int32)
    epsilon = jnp.finfo(eigenvalues.dtype).eps
    raw_threshold = epsilon * max(raw_factor.shape) * largest
    raw_rank = jnp.sum(eigenvalues > raw_threshold, dtype=jnp.int32)
    discarded = jnp.where(retained, 0, eigenvalues)
    truncation_loss = jnp.sqrt(jnp.sum(jnp.square(discarded)))
    solution_norm = jnp.sqrt(jnp.sum(jnp.square(eigenvalues)))
    tiny = jnp.asarray(jnp.finfo(eigenvalues.dtype).tiny)
    relative_loss = truncation_loss / jnp.maximum(solution_norm, tiny)
    return factor, rank, raw_rank, truncation_loss, relative_loss


def _residual_certificate(
    problem: FactoredMatrixEquationProblem,
    factor: Array,
    /,
) -> FactoredMatrixEquationResidualCertificate:
    applied = _operator_columns(problem.operator, factor)
    source = problem.source_factor.astype(factor.dtype)
    columns = jnp.concatenate((applied, factor, source), axis=1)
    factor_capacity = factor.shape[1]
    source_rank = source.shape[1]
    signature = jnp.zeros(
        (2 * factor_capacity + source_rank,) * 2,
        dtype=factor.dtype,
    )
    identity = jnp.eye(factor_capacity, dtype=factor.dtype)
    signature = signature.at[:factor_capacity, factor_capacity : 2 * factor_capacity].set(
        identity
    )
    signature = signature.at[factor_capacity : 2 * factor_capacity, :factor_capacity].set(
        identity
    )
    signature = signature.at[2 * factor_capacity :, 2 * factor_capacity :].set(
        jnp.eye(source_rank, dtype=factor.dtype)
    )
    gram = jnp.conj(columns.T) @ columns
    signed_gram = signature @ gram
    residual_squared = jnp.maximum(
        jnp.real(jnp.trace(signed_gram @ signed_gram)),
        0,
    )
    source_gram = jnp.conj(source.T) @ source
    forcing_squared = jnp.maximum(
        jnp.real(jnp.trace(source_gram @ source_gram)),
        0,
    )
    residual_norm = jnp.sqrt(residual_squared)
    forcing_norm = jnp.sqrt(forcing_squared)
    tiny = jnp.asarray(jnp.finfo(residual_norm.dtype).tiny)
    relative = residual_norm / jnp.maximum(forcing_norm, tiny)
    valid = (
        jnp.isfinite(residual_norm) & jnp.isfinite(relative) & jnp.isfinite(forcing_norm)
    )
    return FactoredMatrixEquationResidualCertificate(
        residual_norm=residual_norm,
        relative_residual=relative,
        forcing_norm=forcing_norm,
        valid=valid,
        equation="A X + X A* + B B* = 0",
        norm="Frobenius",
        method="exact low-rank Gram identity in coordinate arithmetic",
        exact=True,
    )


def _operator_columns(operator: AbstractLinearOperator, columns: Array, /) -> Array:
    return jax.vmap(
        lambda column: operator.target.flatten(
            operator.mv(operator.source.unflatten(column))
        ),
        in_axes=1,
        out_axes=1,
    )(columns)


def _factored_cost(
    problem: FactoredMatrixEquationProblem,
    policy: FactoredMatrixEquationPolicy,
    shifted_plan: ShiftedSolvePlan,
    dtype: Any,
    /,
) -> FactoredMatrixEquationCostEstimate:
    dimension = problem.dimension
    source_rank = problem.source_rank
    steps = len(policy.shifts)
    capacity = source_rank * steps
    itemsize = np.dtype(dtype).itemsize
    certificate_width = 2 * capacity + source_rank
    return FactoredMatrixEquationCostEstimate(
        dimension=dimension,
        source_rank=source_rank,
        adi_steps=steps,
        raw_rank_capacity=capacity,
        shifted_solve_count=steps * source_rank,
        shifted_setup_matvec_count=(steps * source_rank * shifted_plan.cost.matvec_count),
        residual_matvec_count=capacity,
        factor_storage_bytes=dimension * capacity * itemsize,
        source_factor_storage_bytes=dimension * source_rank * itemsize,
        shifted_basis_storage_bytes=shifted_plan.cost.basis_storage_bytes,
        shifted_workspace_bytes=shifted_plan.cost.workspace_bytes,
        small_matrix_workspace_bytes=certificate_width * certificate_width * itemsize,
        explicit_solution_bytes=dimension * dimension * itemsize,
        selected_method=shifted_plan.selected_method,
        exact=False,
    )


def _adi_shifts(
    problem: FactoredMatrixEquationProblem,
    policy: FactoredMatrixEquationPolicy,
    /,
) -> Array:
    values = jnp.asarray(policy.shifts)
    coordinate_dtype = _coordinate_dtype(problem.operator.source)
    if not np.issubdtype(coordinate_dtype, np.complexfloating) and np.issubdtype(
        np.dtype(values.dtype), np.complexfloating
    ):
        imaginary = np.asarray([complex(value).imag for value in policy.shifts])
        if np.any(imaginary != 0.0):
            raise NotImplementedError(
                "Complex ADI shifts for a real-coordinate operator require paired real-block "
                "shifted solves and are not currently supported."
            )
        values = jnp.real(values)
    dtype = jnp.result_type(coordinate_dtype, values.dtype)
    if np.dtype(dtype) != coordinate_dtype:
        if np.issubdtype(coordinate_dtype, np.complexfloating):
            values = values.astype(coordinate_dtype)
        else:
            values = values.astype(coordinate_dtype)
    else:
        values = values.astype(dtype)
    return values


def _shift_family(
    problem: FactoredMatrixEquationProblem,
    adi_shift: Array,
    family_id: str | None,
    /,
) -> ShiftedLinearSystemFamily:
    return ShiftedLinearSystemFamily(
        problem.operator,
        jnp.reshape(-adi_shift, (1,)),
        family_id=family_id,
    )


def _validate_factored_operator(operator: AbstractLinearOperator, /) -> None:
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Factored Lyapunov equations require an unbatched endomorphism.")
    if not isinstance(operator.source, ArraySpace) or operator.source.shape != (
        operator.source.size,
    ):
        raise NotImplementedError(
            "Factored matrix equations currently require a rank-one ArraySpace coordinate vector."
        )
    if not isinstance(operator.source.pairing, EuclideanPairing):
        raise NotImplementedError(
            "Factored Hermitian coordinate solutions currently require Euclidean pairing."
        )


def _validate_factored_plan(
    problem: FactoredMatrixEquationProblem,
    plan: FactoredMatrixEquationPlan,
    /,
) -> None:
    if not isinstance(problem, FactoredMatrixEquationProblem):
        _reject_dense_or_unsupported_problem(problem)
    if not isinstance(plan, FactoredMatrixEquationPlan):
        raise TypeError("plan must be a FactoredMatrixEquationPlan.")
    shifts = _adi_shifts(problem, plan.policy)
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or problem.operator.source.space_id != plan.source_space_id
        or problem.source_rank != plan.source_rank
        or np.dtype(shifts.dtype).str != plan.shift_dtype
    ):
        raise ValueError(
            "Factored matrix-equation plan belongs to a different symbolic problem."
        )


def _reject_dense_or_unsupported_problem(problem: Any, /) -> None:
    if isinstance(problem, MatrixEquationProblem):
        raise NotImplementedError(
            "Factored solves currently support only FactoredMatrixEquationProblem for "
            "A X + X A* = -B B*; no dense fallback is provided for generalized, "
            "Sylvester, discrete Lyapunov, or dense-forcing problems."
        )
    if not isinstance(problem, FactoredMatrixEquationProblem):
        raise TypeError("problem must be a FactoredMatrixEquationProblem.")
    if problem.kind != "continuous-lyapunov":
        raise NotImplementedError(
            "Only continuous Lyapunov structure is supported by the factored solver."
        )


def _nonnegative_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


__all__ = [
    "FactoredMatrixEquationCostEstimate",
    "FactoredMatrixEquationDiagnostics",
    "FactoredMatrixEquationPlan",
    "FactoredMatrixEquationPolicy",
    "FactoredMatrixEquationProblem",
    "FactoredMatrixEquationProvenance",
    "FactoredMatrixEquationResidualCertificate",
    "FactoredMatrixEquationResult",
    "FactoredMatrixEquationStatus",
    "FactoredMatrixSolution",
    "FactoredMatrixSolutionForm",
    "PreparedFactoredMatrixEquation",
    "factored_continuous_lyapunov_equation",
    "plan_factored_matrix_equation",
    "prepare_factored_matrix_equation",
    "refresh_factored_matrix_equation",
    "solve_factored_matrix_equation",
]
