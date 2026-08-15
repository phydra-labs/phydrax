#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator
from ._policies import (
    DenseCholesky,
    DenseLU,
    DenseQR,
    DenseSVD,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    RankPolicy,
    SolveResourcePolicy,
)
from ._prepared import PreparedLinearSolve
from ._problems import LeastSquaresProblem, LinearSystem, MinimumNormProblem
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
    materialization: MaterializationPolicy
    resources: SolveResourcePolicy

    def __init__(
        self,
        kind: FactorizationKind = "auto",
        /,
        *,
        rank: RankPolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        resources: SolveResourcePolicy | None = None,
    ):
        if kind not in ("auto", "lu", "cholesky", "qr", "svd"):
            raise ValueError("Unknown factorization kind.")
        self.kind = kind
        self.rank = RankPolicy() if rank is None else rank
        self.materialization = (
            MaterializationPolicy() if materialization is None else materialization
        )
        self.resources = SolveResourcePolicy() if resources is None else resources


class FactorizationCapabilities(StrictModule):
    solve: bool = eqx.field(static=True)
    transpose_solve: bool = eqx.field(static=True)
    adjoint_solve: bool = eqx.field(static=True)
    rank: bool = eqx.field(static=True)
    singular_values: bool = eqx.field(static=True)
    determinant: bool = eqx.field(static=True)
    pseudodeterminant: bool = eqx.field(static=True)
    nullspaces: bool = eqx.field(static=True)


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
        self.factorization_id = canonical_fingerprint(
            {
                "kind": "prepared-factorization",
                "plan": prepared_solve.plan.plan_id,
                "operator": operator.operator_id,
                "numeric_version": int(prepared_solve.numeric_version),
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
    if operator.batch_shape:
        raise ValueError(
            "Public factorization artifacts currently require no operator batch."
        )
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
        rank=policy_.rank,
        materialization=policy_.materialization,
        differentiation=DifferentiationPolicy("mathematical"),
        failure=FailurePolicy("status"),
        resources=policy_.resources,
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
        nullspaces=svd,
    )
    return PreparedFactorization(operator, prepared, policy_, capabilities)


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
    "refresh_factorization",
]
