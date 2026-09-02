#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._dense_pseudoinverse import fixed_rank_pseudoinverse_value
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator, DenseLinearOperator
from ._policies import (
    DenseCholesky,
    DenseLU,
    DenseQR,
    DenseSVD,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    MixedPrecisionPolicy,
    RankPolicy,
    SolveResourcePolicy,
    TolerancePolicy,
)
from ._prepared import PreparedLinearSolve
from ._problems import LeastSquaresProblem, LinearSystem, MinimumNormProblem
from ._properties import OperatorProperties
from ._rank import numerical_rank_data
from ._results import (
    LinearSolveDiagnostics,
    LinearSolveProvenance,
    LinearSolveResult,
    LinearSolveStatus,
    MatrixInversionResult,
)
from ._spaces import ArraySpace, RHSLayout
from ._subspaces import LinearSubspace
from .backends._jax_dense import (
    DenseCholeskyState,
    DenseLUState,
    DenseQRState,
    DenseSVDState,
)


FactorizationKind: TypeAlias = Literal["auto", "lu", "cholesky", "qr", "svd"]


class FactorizationPolicy(StrictModule):
    """Dense factorization choice with explicit rank and resource policies."""

    kind: FactorizationKind = eqx.field(static=True)
    rank: RankPolicy
    tolerance: TolerancePolicy
    materialization: MaterializationPolicy
    differentiation: DifferentiationPolicy
    failure: FailurePolicy
    resources: SolveResourcePolicy
    precision: MixedPrecisionPolicy | None

    def __init__(
        self,
        kind: FactorizationKind = "auto",
        /,
        *,
        rank: RankPolicy | None = None,
        tolerance: TolerancePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        differentiation: DifferentiationPolicy | None = None,
        failure: FailurePolicy | None = None,
        resources: SolveResourcePolicy | None = None,
        precision: MixedPrecisionPolicy | None = None,
    ):
        if kind not in ("auto", "lu", "cholesky", "qr", "svd"):
            raise ValueError("Unknown factorization kind.")
        self.kind = kind
        self.rank = RankPolicy() if rank is None else rank
        self.tolerance = TolerancePolicy() if tolerance is None else tolerance
        self.materialization = (
            MaterializationPolicy() if materialization is None else materialization
        )
        self.differentiation = (
            DifferentiationPolicy() if differentiation is None else differentiation
        )
        self.failure = FailurePolicy() if failure is None else failure
        self.resources = SolveResourcePolicy() if resources is None else resources
        self.precision = precision


class FactorizationCapabilities(StrictModule):
    solve: bool = eqx.field(static=True)
    transpose_solve: bool = eqx.field(static=True)
    adjoint_solve: bool = eqx.field(static=True)
    rank: bool = eqx.field(static=True)
    singular_values: bool = eqx.field(static=True)
    determinant: bool = eqx.field(static=True)
    pseudodeterminant: bool = eqx.field(static=True)
    nullspaces: bool = eqx.field(static=True)
    inverse_materialization: bool = eqx.field(static=True)
    pseudoinverse_materialization: bool = eqx.field(static=True)


class PreparedFactorization(StrictModule):
    """Reusable dense numerical factorization with truthful capabilities."""

    operator: AbstractLinearOperator
    prepared_solve: PreparedLinearSolve
    policy: FactorizationPolicy
    capabilities: FactorizationCapabilities
    factorization_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        prepared_solve: PreparedLinearSolve,
        policy: FactorizationPolicy,
        capabilities: FactorizationCapabilities,
        /,
    ):
        self.operator = operator
        self.prepared_solve = prepared_solve
        self.policy = policy
        self.capabilities = capabilities
        numeric_version = prepared_solve.numeric_version
        version_id = (
            "traced"
            if isinstance(numeric_version, jax_core.Tracer)
            else int(numeric_version)
        )
        self.factorization_id = canonical_fingerprint(
            {
                "kind": "prepared-factorization",
                "plan": prepared_solve.plan.plan_id,
                "operator": operator.operator_id,
                "numeric_version": version_id,
            }
        )

    def solve(self, rhs: PyTree[Any], /):
        from ._runtime import solve

        return solve(self.prepared_solve, rhs)

    def solve_transpose(self, rhs: PyTree[Any], /):
        if not self.capabilities.transpose_solve:
            raise ValueError("This factorization does not support transpose solves.")
        from ._runtime import solve_transpose

        return solve_transpose(self.prepared_solve, rhs)

    def solve_adjoint(self, rhs: PyTree[Any], /):
        if not self.capabilities.adjoint_solve:
            raise ValueError("This factorization does not support adjoint solves.")
        from ._runtime import solve_adjoint

        return solve_adjoint(self.prepared_solve, rhs)

    def materialize_inverse(self, /) -> MatrixInversionResult:
        if not self.capabilities.inverse_materialization:
            raise ValueError("This factorization cannot materialize an inverse.")
        if self.operator.source.size != self.operator.target.size:
            raise ValueError("A matrix inverse requires a square operator.")
        return _materialize_with_identity(self, operation="inverse")

    def materialize_pseudoinverse(self, /) -> MatrixInversionResult:
        if not self.capabilities.pseudoinverse_materialization:
            raise ValueError("This factorization cannot materialize a pseudoinverse.")
        return _materialize_with_identity(self, operation="pseudoinverse")

    def rank(self, /) -> Array:
        if not self.capabilities.rank:
            raise ValueError("This factorization does not report numerical rank.")
        state = self.prepared_solve.state
        if isinstance(state, (DenseQRState, DenseSVDState)):
            return state.rank
        size = jnp.asarray(_state_matrix(state).shape[-1], dtype=jnp.int32)
        if isinstance(state, DenseLUState):
            return jnp.where(state.singular, jnp.asarray(-1, dtype=jnp.int32), size)
        if isinstance(state, DenseCholeskyState):
            return jnp.where(state.invalid, jnp.asarray(-1, dtype=jnp.int32), size)
        raise TypeError(f"Unsupported factorization state {type(state).__name__}.")

    def singular_values(self, /) -> Array:
        if not self.capabilities.singular_values:
            raise ValueError("This factorization does not expose singular values.")
        state = self.prepared_solve.state
        if not isinstance(state, DenseSVDState):
            raise TypeError("Singular-value capability requires a dense SVD state.")
        return state.reported_singular_values

    def determinant_sign(self, /) -> Array:
        if not self.capabilities.determinant:
            raise ValueError("This factorization does not define a determinant.")
        return jnp.linalg.slogdet(_state_matrix(self.prepared_solve.state))[0]

    def log_abs_determinant(self, /) -> Array:
        if not self.capabilities.determinant:
            raise ValueError("This factorization does not define a determinant.")
        return jnp.linalg.slogdet(_state_matrix(self.prepared_solve.state))[1]

    def log_pseudodeterminant(self, /) -> Array:
        if not self.capabilities.pseudodeterminant:
            raise ValueError("This factorization does not define a pseudodeterminant.")
        state = self.prepared_solve.state
        if not isinstance(state, DenseSVDState):
            raise TypeError("Pseudodeterminant capability requires a dense SVD state.")
        safe = jnp.where(state.retained, state.singular_values, 1)
        return jnp.sum(jnp.where(state.retained, jnp.log(safe), 0), axis=-1)

    def right_nullspace(self, /) -> LinearSubspace:
        if not self.capabilities.nullspaces:
            raise ValueError("This factorization does not expose nullspaces.")
        return _nullspace(self, right=True)

    def left_nullspace(self, /) -> LinearSubspace:
        if not self.capabilities.nullspaces:
            raise ValueError("This factorization does not expose nullspaces.")
        return _nullspace(self, right=False)


def factorize(
    operator: AbstractLinearOperator,
    policy: FactorizationPolicy | None = None,
    /,
) -> PreparedFactorization:
    """Materialize and prepare one reusable dense factorization."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    policy_ = FactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, FactorizationPolicy):
        raise TypeError("policy must be a FactorizationPolicy or None.")
    rows, columns = operator.target.size, operator.source.size
    kind = policy_.kind
    if kind == "auto":
        kind = (
            "cholesky"
            if rows == columns and operator.properties.certifies("positive_definite")
            else "lu"
            if rows == columns
            else "svd"
        )
    if kind in ("lu", "cholesky") and rows != columns:
        raise ValueError(f"{kind} factorization requires a square operator.")
    if kind == "qr" and rows < columns:
        raise ValueError("QR factorization requires at least as many rows as columns.")
    methods = {
        "lu": DenseLU(),
        "cholesky": DenseCholesky(),
        "qr": DenseQR(),
        "svd": DenseSVD(),
    }
    method = methods[kind]
    if kind in ("lu", "cholesky"):
        problem = LinearSystem(operator)
    elif kind == "svd" and rows < columns:
        problem = MinimumNormProblem(operator)
    else:
        problem = LeastSquaresProblem(operator)
    solve_policy = LinearSolvePolicy(
        method,
        tolerance=policy_.tolerance,
        rank=policy_.rank,
        materialization=policy_.materialization,
        differentiation=policy_.differentiation,
        failure=policy_.failure,
        resources=policy_.resources,
        precision=policy_.precision,
    )
    from ._runtime import prepare

    prepared = prepare(problem, solve_policy)
    direct_solve = isinstance(prepared.state, (DenseLUState, DenseCholeskyState))
    svd = isinstance(prepared.state, DenseSVDState)
    capabilities = FactorizationCapabilities(
        solve=True,
        transpose_solve=direct_solve,
        adjoint_solve=direct_solve,
        rank=isinstance(
            prepared.state,
            (DenseLUState, DenseCholeskyState, DenseQRState, DenseSVDState),
        ),
        singular_values=svd,
        determinant=rows == columns,
        pseudodeterminant=svd,
        nullspaces=svd and not operator.batch_shape,
        inverse_materialization=direct_solve and rows == columns,
        pseudoinverse_materialization=svd,
    )
    return PreparedFactorization(operator, prepared, policy_, capabilities)


def inverse(
    matrix_or_operator: ArrayLike | AbstractLinearOperator,
    policy: FactorizationPolicy | None = None,
    /,
    *,
    properties: OperatorProperties | None = None,
) -> MatrixInversionResult:
    """Materialize an explicit inverse through one prepared dense factorization."""
    operator = _as_factorization_operator(matrix_or_operator, properties=properties)
    policy_ = FactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, FactorizationPolicy):
        raise TypeError("policy must be a FactorizationPolicy or None.")
    if policy_.kind not in ("auto", "lu", "cholesky"):
        raise ValueError("inverse requires an auto, LU, or Cholesky factorization.")
    if policy_.differentiation.mode == "rhs-only":
        raise ValueError("rhs-only differentiation is not defined for an inverse matrix.")
    return factorize(operator, policy_).materialize_inverse()


def pseudoinverse(
    matrix_or_operator: ArrayLike | AbstractLinearOperator,
    policy: FactorizationPolicy | None = None,
    /,
    *,
    properties: OperatorProperties | None = None,
) -> MatrixInversionResult:
    """Materialize a Moore-Penrose pseudoinverse through a prepared dense SVD."""
    operator = _as_factorization_operator(matrix_or_operator, properties=properties)
    policy_ = FactorizationPolicy("svd") if policy is None else policy
    if not isinstance(policy_, FactorizationPolicy):
        raise TypeError("policy must be a FactorizationPolicy or None.")
    if policy_.kind not in ("auto", "svd"):
        raise ValueError("pseudoinverse requires an auto or SVD factorization.")
    if policy_.differentiation.mode == "rhs-only":
        raise ValueError(
            "rhs-only differentiation is not defined for a pseudoinverse matrix."
        )
    if policy_.kind == "auto":
        policy_ = FactorizationPolicy(
            "svd",
            rank=policy_.rank,
            tolerance=policy_.tolerance,
            materialization=policy_.materialization,
            differentiation=policy_.differentiation,
            failure=policy_.failure,
            resources=policy_.resources,
            precision=policy_.precision,
        )
    return factorize(operator, policy_).materialize_pseudoinverse()


def _as_factorization_operator(
    matrix_or_operator: ArrayLike | AbstractLinearOperator,
    /,
    *,
    properties: OperatorProperties | None,
) -> AbstractLinearOperator:
    if isinstance(matrix_or_operator, AbstractLinearOperator):
        if properties is not None:
            raise ValueError("properties must be declared on an operator input.")
        return matrix_or_operator
    return DenseLinearOperator(matrix_or_operator, properties=properties)


def _materialize_with_identity(
    factorization: PreparedFactorization,
    /,
    *,
    operation: Literal["inverse", "pseudoinverse"],
) -> MatrixInversionResult:
    state = factorization.prepared_solve.state
    if operation == "pseudoinverse" and isinstance(state, DenseSVDState):
        return _materialize_prepared_pseudoinverse(factorization, state)
    operator = factorization.operator
    target_size = operator.target.size
    dtype = _state_matrix(factorization.prepared_solve.state).dtype
    identity = jnp.eye(target_size, dtype=dtype)
    if isinstance(operator.target, ArraySpace):
        rhs = identity.reshape(operator.target.shape + (target_size,))
    else:
        rhs = jax.vmap(operator.target.unflatten, in_axes=1, out_axes=-1)(identity)

    from ._runtime import solve

    linear = solve(
        factorization.prepared_solve,
        rhs,
        rhs_layout=RHSLayout((target_size,)),
    )
    value = _canonical_matrix_value(operator, linear.value, target_size)
    if (
        operation == "inverse"
        and factorization.policy.differentiation.mode == "mathematical"
    ):
        value = _mathematical_inverse_value(
            _state_matrix(factorization.prepared_solve.state),
            value,
        )
    return _matrix_inversion_result(
        linear,
        value,
        factorization,
        operation=operation,
    )


def _materialize_prepared_pseudoinverse(
    factorization: PreparedFactorization,
    state: DenseSVDState,
    /,
) -> MatrixInversionResult:
    if state.source_projection is not None:
        raise ValueError(
            "Pseudoinverse materialization does not accept regularized rank projection."
        )
    if state.design.shape[-2] != state.target_size:
        raise ValueError("Pseudoinverse materialization does not accept regularization.")
    safe = jnp.where(
        state.retained,
        state.singular_values,
        jnp.ones_like(state.singular_values),
    )
    reciprocal = jnp.where(state.retained, 1.0 / safe, 0.0)
    right = jnp.conj(jnp.swapaxes(state.vh, -1, -2))
    reduced_value = (right * reciprocal[..., None, :]) @ jnp.conj(
        jnp.swapaxes(state.u, -1, -2)
    )
    reduced_value = fixed_rank_pseudoinverse_value(
        state.design,
        reduced_value,
        state.hermitian,
    )
    value = reduced_value
    if state.source_inverse_square_root is not None:
        value = state.source_inverse_square_root[..., :, None] * value
    if state.square_root_weights is not None:
        value = value * state.square_root_weights[..., None, :]

    matrix = state.original_matrix
    residual = matrix @ value @ matrix - matrix
    residual_norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=(-2, -1)))
    matrix_norm = jnp.sqrt(jnp.sum(jnp.abs(matrix) ** 2, axis=(-2, -1)))
    tiny = jnp.asarray(jnp.finfo(matrix.real.dtype).tiny, dtype=matrix.real.dtype)
    relative_residual = residual_norm / jnp.maximum(matrix_norm, tiny)
    tolerance = factorization.policy.tolerance
    threshold = (
        jnp.asarray(
            tolerance.absolute,
            dtype=matrix.real.dtype,
        )
        + jnp.asarray(tolerance.relative, dtype=matrix.real.dtype) * matrix_norm
    )
    input_finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    output_finite = jnp.all(jnp.isfinite(value), axis=(-2, -1))
    required_rank = min(matrix.shape[-2], matrix.shape[-1])
    rank_deficient = state.rank < required_rank
    status = jnp.zeros(matrix.shape[:-2], dtype=jnp.int32)
    status = jnp.where(
        ~input_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        (status == 0) & ~output_finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        (status == 0) & factorization.policy.rank.require_full_rank & rank_deficient,
        int(LinearSolveStatus.RANK_DEFICIENT),
        status,
    )
    status = jnp.where(
        (status == 0) & (residual_norm > threshold),
        int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
        status,
    )
    rank_data = numerical_rank_data(
        state.reported_singular_values,
        matrix.shape[-2],
        matrix.shape[-1],
        factorization.policy.rank,
    )
    diagnostics = LinearSolveDiagnostics(
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        normal_residual_norm=residual_norm,
        rank=state.rank,
        condition_estimate=state.condition_estimate,
        finite=input_finite & output_finite,
        converged=status == int(LinearSolveStatus.SUCCESS),
        singular_values=state.reported_singular_values,
        rank_cutoff=rank_data.cutoff,
    )
    prepared = factorization.prepared_solve
    plan = prepared.plan
    provenance = LinearSolveProvenance(
        backend=plan.backend,
        method=plan.method,
        plan_id=plan.plan_id,
        problem_id=prepared.problem.problem_id,
        reason=plan.reason,
        rejected=plan.rejected,
        prepared=True,
        rhs_mode="pseudo-block",
        operator_numeric_version=prepared.numeric_version,
        requested_precision=factorization.policy.precision,
    )
    if factorization.policy.failure.mode == "error":
        value = eqx.error_if(
            value,
            jnp.any(status != int(LinearSolveStatus.SUCCESS)),
            "Pseudoinverse materialization failed.",
        )
    return MatrixInversionResult(
        value,
        status,
        diagnostics,
        provenance,
        "pseudoinverse",
    )


@jax.custom_jvp
def _mathematical_inverse_value(matrix: Array, value: Array, /) -> Array:
    del matrix
    return value


@_mathematical_inverse_value.defjvp
def _mathematical_inverse_value_jvp(primals, tangents):
    _, value = primals
    matrix_tangent, _ = tangents
    return value, -value @ matrix_tangent @ value


def _canonical_matrix_value(
    operator: AbstractLinearOperator,
    value: PyTree[Array],
    rhs_count: int,
    /,
) -> Array:
    if isinstance(operator.source, ArraySpace):
        return jnp.asarray(value).reshape(
            operator.batch_shape + (operator.source.size, rhs_count)
        )
    if operator.batch_shape:
        raise TypeError("Batched matrix inversion requires ArraySpace values.")
    return jax.vmap(operator.source.flatten, in_axes=-1, out_axes=1)(value)


def _matrix_inversion_result(
    linear: LinearSolveResult,
    value: Array,
    factorization: PreparedFactorization,
    /,
    *,
    operation: Literal["inverse", "pseudoinverse"],
) -> MatrixInversionResult:
    status = _collapse_matrix_status(linear.status)
    diagnostics = linear.diagnostics
    condition_estimate = diagnostics.condition_estimate[..., 0]
    if operation == "inverse":
        matrix = _state_matrix(factorization.prepared_solve.state)
        matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix), axis=-1), axis=-1)
        inverse_norm = jnp.max(jnp.sum(jnp.abs(value), axis=-1), axis=-1)
        condition_estimate = matrix_norm * inverse_norm
        precision = factorization.policy.precision
        if precision is not None and precision.condition_limit is not None:
            status = jnp.where(
                (status == 0) & (condition_estimate > precision.condition_limit),
                int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
                status,
            )
    singular_values = diagnostics.singular_values
    rank_cutoff = jnp.asarray(jnp.nan, dtype=jnp.real(value).dtype)
    if singular_values is not None:
        rank_data = numerical_rank_data(
            singular_values,
            factorization.operator.target.size,
            factorization.operator.source.size,
            factorization.policy.rank,
        )
        rank_cutoff = rank_data.cutoff
    matrix_diagnostics = LinearSolveDiagnostics(
        residual_norm=jnp.max(diagnostics.residual_norm, axis=-1),
        relative_residual=jnp.max(diagnostics.relative_residual, axis=-1),
        normal_residual_norm=jnp.max(
            diagnostics.normal_residual_norm,
            axis=-1,
        ),
        iterations=jnp.max(diagnostics.iterations, axis=-1),
        rank=diagnostics.rank[..., 0],
        condition_estimate=condition_estimate,
        finite=jnp.all(diagnostics.finite, axis=-1),
        converged=status == int(LinearSolveStatus.SUCCESS),
        singular_values=singular_values,
        rank_cutoff=rank_cutoff,
        compatibility_residual=jnp.max(
            diagnostics.compatibility_residual,
            axis=-1,
        ),
        gauge_residual=jnp.max(diagnostics.gauge_residual, axis=-1),
        nullity=diagnostics.nullity[..., 0],
        matvec_count=jnp.max(diagnostics.matvec_count, axis=-1),
        adjoint_matvec_count=jnp.max(
            diagnostics.adjoint_matvec_count,
            axis=-1,
        ),
        effective_block_rank=jnp.max(
            diagnostics.effective_block_rank,
            axis=-1,
        ),
        deflated_rhs_count=jnp.max(
            diagnostics.deflated_rhs_count,
            axis=-1,
        ),
        refinement_steps=jnp.max(diagnostics.refinement_steps, axis=-1),
    )
    return MatrixInversionResult(
        value,
        status,
        matrix_diagnostics,
        linear.provenance,
        operation,
    )


def _collapse_matrix_status(status: Array, /) -> Array:
    values = jnp.asarray(status, dtype=jnp.int32)
    collapsed = jnp.zeros(values.shape[:-1], dtype=jnp.int32)
    priority = (
        LinearSolveStatus.MAXIMUM_STEPS_REACHED,
        LinearSolveStatus.STAGNATION,
        LinearSolveStatus.BREAKDOWN,
        LinearSolveStatus.CAPABILITY_REJECTED,
        LinearSolveStatus.INCOMPATIBLE_STRUCTURE,
        LinearSolveStatus.ADJOINT_FAILED,
        LinearSolveStatus.RESIDUAL_TOO_LARGE,
        LinearSolveStatus.CONDITION_LIMIT_REACHED,
        LinearSolveStatus.NONFINITE_OUTPUT,
        LinearSolveStatus.RANK_DEFICIENT,
        LinearSolveStatus.SINGULAR,
        LinearSolveStatus.NONFINITE_INPUT,
    )
    for code in priority:
        collapsed = jnp.where(
            jnp.any(values == int(code), axis=-1),
            int(code),
            collapsed,
        )
    return collapsed


def refresh_factorization(
    factorization: PreparedFactorization,
    operator: AbstractLinearOperator,
    /,
) -> PreparedFactorization:
    """Refresh numerical factors while preserving the symbolic plan and versioning."""
    if not isinstance(factorization, PreparedFactorization):
        raise TypeError("factorization must be a PreparedFactorization.")
    previous_problem = factorization.prepared_solve.problem
    if isinstance(previous_problem, LinearSystem):
        problem = LinearSystem(operator, problem_id=previous_problem.problem_id)
    elif isinstance(previous_problem, LeastSquaresProblem):
        problem = LeastSquaresProblem(operator, problem_id=previous_problem.problem_id)
    else:
        problem = MinimumNormProblem(operator, problem_id=previous_problem.problem_id)
    from ._runtime import refresh

    prepared = refresh(factorization.prepared_solve, problem)
    return PreparedFactorization(
        operator,
        prepared,
        factorization.policy,
        factorization.capabilities,
    )


def _state_matrix(state: Any, /) -> Array:
    if isinstance(state, (DenseLUState, DenseCholeskyState)):
        return state.matrix
    if isinstance(state, (DenseQRState, DenseSVDState)):
        return state.original_matrix
    raise TypeError(f"Unsupported factorization state {type(state).__name__}.")


def _nullspace(
    factorization: PreparedFactorization,
    /,
    *,
    right: bool,
) -> LinearSubspace:
    state = factorization.prepared_solve.state
    if not isinstance(state, DenseSVDState):
        raise TypeError("Nullspace extraction requires a dense SVD state.")
    matrix = state.original_matrix
    if right:
        kernel_operator = matrix
        space = factorization.operator.source
    else:
        target_metric = _riesz_matrix(factorization.operator.target)
        kernel_operator = jnp.conj(jnp.swapaxes(matrix, -1, -2)) @ target_metric
        space = factorization.operator.target
    itemsize = matrix.dtype.itemsize
    basis_bytes = space.size * space.size * itemsize
    workspace_bytes = (
        matrix.shape[-2] ** 2
        + matrix.shape[-1] ** 2
        + min(matrix.shape[-2], matrix.shape[-1])
    ) * itemsize
    if basis_bytes > factorization.policy.resources.factorization_bytes:
        raise ValueError(
            f"Nullspace basis requires {basis_bytes} bytes, exceeding the "
            "factorization budget."
        )
    if workspace_bytes > factorization.policy.resources.workspace_bytes:
        raise ValueError(
            f"Full nullspace SVD requires an estimated {workspace_bytes} "
            "workspace bytes, exceeding the workspace budget."
        )
    _, _, vh = jnp.linalg.svd(kernel_operator, full_matrices=True)
    vectors = jnp.conj(jnp.swapaxes(vh, -1, -2))
    rank = state.rank
    capacity = vectors.shape[-1]
    order = (jnp.arange(capacity, dtype=jnp.int32) + rank) % capacity
    basis = vectors[:, order]
    dimension = jnp.asarray(capacity, dtype=jnp.int32) - rank
    basis = _metric_orthonormalize(space, basis, dimension)
    return LinearSubspace(
        space,
        basis,
        dimension=dimension,
        orthonormal=True,
    )


def _riesz_matrix(space, /) -> Array:
    coordinates = space.flatten(space.zeros())
    basis = jnp.eye(space.size, dtype=coordinates.dtype)
    return jax.vmap(
        lambda column: space.flatten(space.riesz(space.unflatten(column))),
        in_axes=1,
        out_axes=1,
    )(basis)


def _metric_orthonormalize(
    space,
    basis: Array,
    dimension: Array,
    /,
) -> Array:
    capacity = basis.shape[-1]
    active = jnp.arange(capacity) < dimension
    masked = jnp.where(active[None, :], basis, 0)

    def inner(left, right):
        return space.inner(space.unflatten(left), space.unflatten(right))

    gram = jax.vmap(
        lambda left: jax.vmap(lambda right: inner(left, right), in_axes=1)(masked),
        in_axes=1,
    )(masked)
    gram = 0.5 * (gram + jnp.conj(jnp.swapaxes(gram, -1, -2)))
    gram = gram + jnp.diag((~active).astype(gram.dtype))
    factor = jnp.linalg.cholesky(gram)
    transform = jnp.linalg.solve(
        jnp.conj(jnp.swapaxes(factor, -1, -2)),
        jnp.eye(capacity, dtype=gram.dtype),
    )
    return masked @ transform


__all__ = [
    "FactorizationCapabilities",
    "FactorizationKind",
    "FactorizationPolicy",
    "PreparedFactorization",
    "factorize",
    "inverse",
    "pseudoinverse",
    "refresh_factorization",
]
