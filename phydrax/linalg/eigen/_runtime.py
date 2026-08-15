#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, PyTree

from .._materialization import MaterializationPolicy, materialize
from .._operators import AbstractLinearOperator
from .._preconditioning import prepare_preconditioner, PreparedPreconditioner
from .._spaces import _coordinate_dtype, _coordinate_pairing_matrix
from ._native import _solve_dense_eigh, _solve_lobpcg, _solve_restarted_lanczos
from ._plans import EigenSolvePlan, make_eigensolve_plan
from ._policies import DenseEigh, EigenSolvePolicy, LOBPCG, RestartedLanczos
from ._prepared import DenseEigenState, PreparedEigenSolve
from ._problems import Eigenproblem, EigenproblemLike, GeneralizedEigenproblem
from ._results import (
    _NativeEigenResult,
    EigenSolveDiagnostics,
    EigenSolveProvenance,
    EigenSolveResult,
    EigenSolveStatus,
)


def plan_eigensolve(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy | None = None,
    /,
) -> EigenSolvePlan:
    """Select and resource-check a deterministic matrix-free eigen solve."""
    selected_policy = EigenSolvePolicy() if policy is None else policy
    return make_eigensolve_plan(problem, selected_policy)


def prepare_eigensolve(
    problem: EigenproblemLike,
    policy: EigenSolvePolicy | EigenSolvePlan | None = None,
    /,
) -> PreparedEigenSolve:
    """Prepare reusable fixed-shape numerical state for one eigenproblem."""
    _require_problem(problem)
    if isinstance(policy, EigenSolvePlan):
        selected_plan = policy
        if selected_plan.problem_id != problem.problem_id:
            raise ValueError("Plan and eigenproblem IDs must match.")
        replanned = make_eigensolve_plan(problem, selected_plan.policy)
        if replanned.plan_id != selected_plan.plan_id:
            raise ValueError("Plan does not match the eigenproblem symbolic structure.")
    else:
        selected_policy = EigenSolvePolicy() if policy is None else policy
        selected_plan = make_eigensolve_plan(problem, selected_policy)
    return _prepare_for_plan(problem, selected_plan)


def refresh_eigensolve(
    prepared: PreparedEigenSolve,
    problem: EigenproblemLike,
    /,
    *,
    setup_operator: AbstractLinearOperator | None = None,
) -> PreparedEigenSolve:
    """Refresh numerical state without changing the symbolic eigen solve plan."""
    if not isinstance(prepared, PreparedEigenSolve):
        raise TypeError("prepared must be a PreparedEigenSolve.")
    _require_problem(problem)
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("Numeric refreshes must preserve problem_id.")

    policy = prepared.plan.policy
    preconditioning = policy.preconditioning
    if setup_operator is not None:
        if preconditioning is None:
            raise ValueError("setup_operator requires a preconditioning policy.")
        replacement = preconditioning.with_setup_operator(setup_operator)
        policy = eqx.tree_at(
            lambda selected: selected.preconditioning,
            policy,
            replacement,
        )
    elif (
        preconditioning is not None
        and preconditioning.builder is not None
        and preconditioning.setup_operator is not None
        and preconditioning.refresh_policy != "frozen"
    ):
        raise ValueError(
            "Refreshing a distinct setup operator requires setup_operator=...; "
            "silent reuse after coefficient changes is forbidden."
        )

    refreshed_plan = make_eigensolve_plan(problem, policy)
    if refreshed_plan.plan_id != prepared.plan.plan_id:
        raise ValueError("Numeric refreshes must preserve the symbolic eigen solve plan.")
    return _prepare_for_plan(
        problem,
        refreshed_plan,
        previous_preconditioner=prepared.preconditioning_state,
        symbolic_version=prepared.symbolic_version,
        numeric_version=prepared.numeric_version + 1,
    )


def _prepare_for_plan(
    problem: EigenproblemLike,
    plan: EigenSolvePlan,
    /,
    *,
    previous_preconditioner: PreparedPreconditioner | None = None,
    symbolic_version: int = 1,
    numeric_version: int = 0,
) -> PreparedEigenSolve:
    if isinstance(plan.selected_method, DenseEigh):
        n = problem.dimension
        dtype = _coordinate_dtype(problem.operator.source)
        constraint_basis = jnp.zeros((n, 0), dtype=dtype)
        metric_constraint_basis = constraint_basis
        initial_basis = jnp.zeros((n, plan.block_dimension), dtype=dtype)
        initial_rank = jnp.asarray(0, dtype=jnp.int32)
        preconditioning_state = None
        dense_state = _prepare_dense_state(problem, plan)
    else:
        constraint_basis, metric_constraint_basis = _constraint_state(problem)
        initial_basis, initial_rank = _initial_block(
            problem,
            plan,
            constraint_basis,
            metric_constraint_basis,
        )
        preconditioning_state = _prepare_preconditioning(
            problem,
            plan,
            previous=previous_preconditioner,
            numeric_version=numeric_version,
        )
        dense_state = None
    return PreparedEigenSolve(
        problem,
        plan,
        initial_basis,
        constraint_basis,
        metric_constraint_basis,
        preconditioning_state=preconditioning_state,
        dense_state=dense_state,
        initial_rank=initial_rank,
        symbolic_version=symbolic_version,
        numeric_version=numeric_version,
    )


def _prepare_dense_state(
    problem: EigenproblemLike,
    plan: EigenSolvePlan,
    /,
) -> DenseEigenState:
    space = problem.operator.source
    n = problem.dimension
    dtype = _coordinate_dtype(space)
    pairing = _coordinate_pairing_matrix(space)
    operator_matrix = materialize(problem.operator, plan.policy.materialization)
    metric_matrix = (
        materialize(problem.metric_operator, plan.policy.materialization)
        if isinstance(problem, GeneralizedEigenproblem)
        else jnp.eye(n, dtype=dtype)
    )
    paired_operator = _require_hermitian(
        pairing @ operator_matrix,
        "paired operator",
    )
    paired_metric = _require_hermitian(
        pairing @ metric_matrix,
        "paired metric",
    )
    metric_factor = jnp.linalg.cholesky(
        paired_metric,
        symmetrize_input=False,
    )
    metric_factor = eqx.error_if(
        metric_factor,
        jnp.any(~jnp.isfinite(metric_factor))
        | jnp.any(jnp.real(jnp.diag(metric_factor)) <= 0),
        "Dense generalized eigensolve metric factorization failed.",
    )
    left_reduced = jsp.linalg.solve_triangular(
        metric_factor,
        paired_operator,
        lower=True,
    )
    reduced_operator = jnp.conj(
        jsp.linalg.solve_triangular(
            metric_factor,
            jnp.conj(left_reduced.T),
            lower=True,
        ).T
    )
    reduced_operator = _require_hermitian(
        reduced_operator,
        "reduced Hermitian operator",
    )
    return DenseEigenState(
        reduced_operator,
        metric_factor,
        operator_matvec_count=n,
        metric_matvec_count=(n if isinstance(problem, GeneralizedEigenproblem) else 0),
    )


def _require_hermitian(matrix: Array, name: str, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), 1)
    tolerance = 64 * max(matrix.shape[0], 1) * jnp.finfo(matrix.real.dtype).eps * scale
    error = jnp.max(jnp.abs(matrix - jnp.conj(matrix.T)))
    return eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix)) | (error > tolerance),
        f"Dense eigensolve {name} contradicts its self-adjoint certificate.",
    )


def _prepare_preconditioning(
    problem: EigenproblemLike,
    plan: EigenSolvePlan,
    /,
    *,
    previous: PreparedPreconditioner | None,
    numeric_version: int,
) -> PreparedPreconditioner | None:
    if plan.preconditioner_plan is None:
        return None
    byte_budget = plan.policy.resources.preconditioner_bytes
    itemsize = np.dtype(_coordinate_dtype(problem.operator.source)).itemsize
    materialization = MaterializationPolicy(
        max_entries=max(1, byte_budget // itemsize),
        max_bytes=max(1, byte_budget),
    )
    return prepare_preconditioner(
        plan.preconditioner_plan,
        problem.operator,
        materialization=materialization,
        previous=previous,
        numeric_version=numeric_version,
    )


def _constraint_state(problem: EigenproblemLike, /) -> tuple[Array, Array]:
    space = problem.operator.source
    constraints = problem.constraints
    if constraints is None or constraints.capacity == 0:
        empty = jnp.zeros((space.size, 0), dtype=_coordinate_dtype(space))
        return empty, empty
    mask = jnp.arange(constraints.capacity) < constraints.dimension
    basis = jnp.where(mask[None, :], constraints.basis, 0)
    metric_basis = _metric_coordinate_columns(problem, basis)
    gram_diagonal = jax.vmap(
        lambda column, metric_column: jnp.real(
            _coordinate_inner(space, column, metric_column)
        ),
        in_axes=1,
    )(basis, metric_basis)
    norms = jnp.sqrt(jnp.maximum(gram_diagonal, 0))
    safe_norms = jnp.where(mask & (norms > 0), norms, 1)
    basis = jnp.where(mask[None, :], basis / safe_norms[None, :], 0)
    metric_basis = jnp.where(
        mask[None, :],
        metric_basis / safe_norms[None, :],
        0,
    )
    gram = jax.vmap(
        lambda left: jax.vmap(
            lambda right: _coordinate_inner(space, left, right),
            in_axes=1,
        )(metric_basis),
        in_axes=1,
    )(basis)
    gram = 0.5 * (gram + jnp.conj(gram.T))
    gram = gram + jnp.diag((~mask).astype(gram.dtype))
    singular_values = jnp.linalg.svd(gram, compute_uv=False)
    cutoff = (
        jnp.finfo(basis.real.dtype).eps
        * max(constraints.capacity, 1)
        * jnp.max(singular_values)
    )
    dual_diagonal = jax.vmap(
        lambda column: jnp.real(_coordinate_inner(space, column, column)),
        in_axes=1,
    )(metric_basis)
    dual_norms = jnp.sqrt(jnp.maximum(dual_diagonal, 0))
    safe_dual_norms = jnp.where(mask & (dual_norms > 0), dual_norms, 1)
    normalized_dual = jnp.where(
        mask[None, :],
        metric_basis / safe_dual_norms[None, :],
        0,
    )
    dual_gram = jax.vmap(
        lambda left: jax.vmap(
            lambda right: _coordinate_inner(space, left, right),
            in_axes=1,
        )(normalized_dual),
        in_axes=1,
    )(normalized_dual)
    dual_gram = 0.5 * (dual_gram + jnp.conj(dual_gram.T))
    dual_gram = dual_gram + jnp.diag((~mask).astype(dual_gram.dtype))
    dual_singular_values = jnp.linalg.svd(dual_gram, compute_uv=False)
    dual_cutoff = (
        jnp.finfo(basis.real.dtype).eps
        * max(constraints.capacity, 1)
        * jnp.max(dual_singular_values)
    )
    unresolved = (
        jnp.any(~jnp.isfinite(singular_values))
        | (jnp.min(singular_values) <= cutoff)
        | jnp.any(~jnp.isfinite(dual_singular_values))
        | (jnp.min(dual_singular_values) <= dual_cutoff)
    )
    basis = eqx.error_if(
        basis,
        unresolved,
        "Active constraints are numerically rank-deficient in the metric or "
        "residual-dual pairing.",
    )
    return basis, metric_basis


def _initial_block(
    problem: EigenproblemLike,
    plan: EigenSolvePlan,
    constraint_basis: Array,
    metric_constraint_basis: Array,
    /,
) -> tuple[Array, Array]:
    space = problem.operator.source
    policy = plan.policy
    dtype = _coordinate_dtype(space)
    supplied = policy.initial_basis
    if supplied is not None:
        if supplied.shape[0] != space.size:
            raise ValueError(
                "initial_basis leading dimension must equal problem dimension."
            )
        if np.dtype(supplied.dtype) != np.dtype(dtype):
            raise TypeError(
                "initial_basis dtype must match the eigenproblem coordinates."
            )

    key = policy.key
    if key is None:
        key = jax.random.key(int(plan.plan_id[:8], 16) & 0x7FFFFFFF)
    repair = jax.random.normal(
        key,
        (space.size, plan.block_dimension),
        dtype=dtype,
    )
    if supplied is None:
        candidates = repair
    elif supplied.shape[1] == plan.block_dimension and (
        plan.block_dimension < plan.available_dimension
    ):
        exploration = supplied[:, -1:] + repair[:, :1]
        candidates = jnp.concatenate(
            (
                supplied[:, :-1],
                exploration,
                repair[:, 1:],
            ),
            axis=1,
        )
    else:
        candidates = jnp.concatenate((supplied, repair), axis=1)
    metric_candidates = _metric_coordinate_columns(problem, candidates)
    return _orthonormal_initial_block(
        space,
        candidates,
        metric_candidates,
        constraint_basis,
        metric_constraint_basis,
        plan.block_dimension,
    )


def _orthonormal_initial_block(
    space: Any,
    candidates: Array,
    metric_candidates: Array,
    constraint_basis: Array,
    metric_constraint_basis: Array,
    width: int,
    /,
) -> tuple[Array, Array]:
    basis = jnp.zeros((space.size, width), dtype=candidates.dtype)
    metric_basis = jnp.zeros_like(basis)
    rank = jnp.asarray(0, dtype=jnp.int32)
    constraint_gram = _constraint_gram(
        space,
        constraint_basis,
        metric_constraint_basis,
    )
    real_dtype = candidates.real.dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    tiny = jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype)
    rank_scale = jnp.asarray(max(space.size, candidates.shape[1]), dtype=real_dtype)

    def add_candidate(index, state):
        current_basis, current_metric_basis, current_rank = state
        vector = candidates[:, index]
        metric_vector = metric_candidates[:, index]
        vector, metric_vector = _project_constraint_columns(
            space,
            vector,
            metric_vector,
            constraint_basis,
            metric_constraint_basis,
            constraint_gram,
        )
        reference_norm = jnp.maximum(
            jnp.real(_coordinate_inner(space, vector, metric_vector)),
            0,
        )
        vector, metric_vector = _project_initial_columns(
            space,
            vector,
            metric_vector,
            current_basis,
            current_metric_basis,
            current_rank,
        )
        vector, metric_vector = _project_initial_columns(
            space,
            vector,
            metric_vector,
            current_basis,
            current_metric_basis,
            current_rank,
        )
        norm_squared = jnp.maximum(
            jnp.real(_coordinate_inner(space, vector, metric_vector)),
            0,
        )
        cutoff = eps * rank_scale * jnp.maximum(reference_norm, tiny)
        accepted = (
            (current_rank < width) & jnp.isfinite(norm_squared) & (norm_squared > cutoff)
        )
        scale = jnp.sqrt(jnp.maximum(norm_squared, tiny))
        normalized = vector / scale
        normalized_metric = metric_vector / scale
        slot = jnp.minimum(current_rank, width - 1)
        current_basis = current_basis.at[:, slot].set(
            jnp.where(accepted, normalized, current_basis[:, slot])
        )
        current_metric_basis = current_metric_basis.at[:, slot].set(
            jnp.where(accepted, normalized_metric, current_metric_basis[:, slot])
        )
        return (
            current_basis,
            current_metric_basis,
            current_rank + accepted.astype(jnp.int32),
        )

    basis, _, rank = jax.lax.fori_loop(
        0,
        candidates.shape[1],
        add_candidate,
        (basis, metric_basis, rank),
    )
    return basis, rank


def _constraint_gram(
    space: Any,
    basis: Array,
    metric_basis: Array,
    /,
) -> Array:
    capacity = basis.shape[1]
    if capacity == 0:
        return jnp.zeros((0, 0), dtype=basis.dtype)
    gram = jax.vmap(
        lambda left: jax.vmap(
            lambda metric_right: _coordinate_inner(space, left, metric_right),
            in_axes=1,
        )(metric_basis),
        in_axes=1,
    )(basis)
    active = jnp.any(basis != 0, axis=0)
    return gram + jnp.diag((~active).astype(gram.dtype))


def _project_constraint_columns(
    space: Any,
    vector: Array,
    metric_vector: Array,
    basis: Array,
    metric_basis: Array,
    gram: Array,
    /,
) -> tuple[Array, Array]:
    if basis.shape[1] == 0:
        return vector, metric_vector
    right_hand_side = jax.vmap(
        lambda column: _coordinate_inner(space, column, metric_vector),
        in_axes=1,
    )(basis)
    coefficients = jnp.linalg.solve(gram, right_hand_side)
    return vector - basis @ coefficients, metric_vector - metric_basis @ coefficients


def _project_initial_columns(
    space: Any,
    vector: Array,
    metric_vector: Array,
    basis: Array,
    metric_basis: Array,
    rank: Array,
    /,
) -> tuple[Array, Array]:
    mask = jnp.arange(basis.shape[1]) < rank
    coefficients = jax.vmap(
        lambda column: _coordinate_inner(space, column, metric_vector),
        in_axes=1,
    )(basis)
    coefficients = jnp.where(mask, coefficients, 0)
    return vector - basis @ coefficients, metric_vector - metric_basis @ coefficients


def _coordinate_inner(space: Any, left: Array, right: Array, /) -> Array:
    return space.inner(space.unflatten(left), space.unflatten(right))


def _metric_coordinate_columns(problem: EigenproblemLike, block: Array, /) -> Array:
    if isinstance(problem, Eigenproblem):
        return block
    space = problem.operator.source

    def apply(column):
        return space.flatten(problem.metric_operator.mv(space.unflatten(column)))

    return jax.vmap(apply, in_axes=1, out_axes=1)(block)


def eigensolve(
    problem_or_prepared: EigenproblemLike | PreparedEigenSolve,
    /,
    *,
    policy: EigenSolvePolicy | EigenSolvePlan | None = None,
) -> EigenSolveResult:
    """Solve a certified standard or generalized self-adjoint eigenproblem."""
    if isinstance(problem_or_prepared, PreparedEigenSolve):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, (Eigenproblem, GeneralizedEigenproblem)):
        prepared = prepare_eigensolve(problem_or_prepared, policy)
    else:
        raise TypeError("Expected an Eigenproblem or PreparedEigenSolve.")

    native = _stop_arrays(_dispatch_native(prepared))
    status = _solve_status(native, prepared.plan.policy.count)
    values = native.values
    vectors = native.vectors
    if prepared.plan.policy.differentiation == "eigenvalues":
        differentiation_valid, denominators = _differentiation_evidence(
            prepared.problem,
            values,
            vectors,
            native.mode_mask,
            native.converged,
            native.isolation_gaps,
            native.residual_norms,
            native.relative_residuals,
        )
        differentiation_rejected = (
            status == int(EigenSolveStatus.SUCCESS)
        ) & ~differentiation_valid
        status = jnp.where(
            differentiation_rejected,
            int(EigenSolveStatus.DIFFERENTIATION_REJECTED),
            status,
        ).astype(jnp.int32)
        values = jax.lax.cond(
            differentiation_valid,
            lambda payload: _mathematical_eigenvalues(
                prepared.problem,
                jax.lax.stop_gradient(payload[0]),
                jax.lax.stop_gradient(payload[1]),
                payload[2],
            ),
            lambda payload: jax.lax.stop_gradient(payload[0]),
            (values, vectors, denominators),
        )

    if prepared.plan.policy.failure.mode == "error":
        failed = status != int(EigenSolveStatus.SUCCESS)
        message = (
            "Eigen solve failed; inspect status-mode diagnostics for the failure class."
        )
        status = eqx.error_if(status, failed, message)
        values = eqx.error_if(values, failed, message)
        vectors = eqx.error_if(vectors, failed, message)

    effective_count = jnp.sum(native.mode_mask, dtype=jnp.int32)
    eigenvectors = _unflatten_mode_columns(prepared.problem.operator.source, vectors)
    diagnostics = EigenSolveDiagnostics(
        native.residual_norms,
        native.relative_residuals,
        native.orthogonality_error,
        native.iterations,
        native.operator_matvec_count,
        native.metric_matvec_count
        + int(
            isinstance(prepared.problem, GeneralizedEigenproblem)
            and prepared.plan.policy.differentiation == "eigenvalues"
        )
        * prepared.plan.policy.count,
        native.preconditioner_apply_count,
        native.converged,
        native.mode_mask,
        effective_count,
        native.isolation_gaps,
        prepared.initial_rank,
    )
    provenance = EigenSolveProvenance(
        prepared.plan.selected_method.name,
        prepared.plan.policy.which,
        prepared.problem.problem_id,
        prepared.plan.plan_id,
        prepared.plan.rejections,
        prepared.plan.policy.differentiation,
        prepared.symbolic_version,
        prepared.numeric_version,
    )
    result = EigenSolveResult(
        values,
        eigenvectors,
        native.mode_mask,
        effective_count,
        native.converged,
        status,
        diagnostics,
        provenance,
    )
    return (
        _stop_arrays(result) if prepared.plan.policy.differentiation == "none" else result
    )


def _dispatch_native(prepared: PreparedEigenSolve, /) -> _NativeEigenResult:
    method = prepared.plan.selected_method
    if isinstance(method, DenseEigh):
        return _solve_dense_eigh(prepared)
    if isinstance(method, LOBPCG):
        return _solve_lobpcg(prepared)
    if isinstance(method, RestartedLanczos):
        return _solve_restarted_lanczos(prepared)
    raise TypeError("Eigen solve plan did not select a supported eigen method.")


def _solve_status(native: _NativeEigenResult, count: int, /) -> Array:
    finite = (
        jnp.all(jnp.isfinite(native.values))
        & jnp.all(jnp.isfinite(native.vectors))
        & jnp.all(jnp.isfinite(native.residual_norms) | ~native.mode_mask)
        & jnp.all(jnp.isfinite(native.relative_residuals) | ~native.mode_mask)
        & jnp.isfinite(native.orthogonality_error)
    )
    converged = native.converged & native.mode_mask
    effective_count = jnp.sum(native.mode_mask, dtype=jnp.int32)
    converged_count = jnp.sum(converged, dtype=jnp.int32)
    complete = (effective_count == count) & (converged_count == count)
    partial = converged_count > 0
    status = jnp.where(
        complete,
        int(EigenSolveStatus.SUCCESS),
        jnp.where(
            native.rank_deficient,
            int(EigenSolveStatus.RANK_DEFICIENT),
            jnp.where(
                partial,
                int(EigenSolveStatus.PARTIAL_CONVERGENCE),
                int(EigenSolveStatus.MAXIMUM_STEPS_REACHED),
            ),
        ),
    )
    return jnp.where(
        finite,
        status,
        int(EigenSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)


def _unflatten_mode_columns(space: Any, coordinates: Array, /) -> PyTree[Array]:
    return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(coordinates)


def _differentiation_evidence(
    problem: EigenproblemLike,
    values: Array,
    vectors: Array,
    mode_mask: Array,
    converged: Array,
    isolation_gaps: Array,
    residual_norms: Array,
    relative_residuals: Array,
    /,
) -> tuple[Array, Array]:
    scale = jnp.maximum(jnp.abs(values), 1)
    roundoff = jnp.sqrt(jnp.finfo(values.dtype).eps) * max(problem.dimension, 1) * scale
    residual_uncertainty = 4 * jnp.maximum(
        residual_norms,
        relative_residuals * scale,
    )
    cutoff = jnp.maximum(roundoff, residual_uncertainty)
    denominators = jax.lax.stop_gradient(_metric_denominators(problem, vectors))
    valid = (
        jnp.all(mode_mask & converged)
        & jnp.all(jnp.isfinite(isolation_gaps) & (isolation_gaps > cutoff))
        & jnp.all(jnp.isfinite(denominators) & (denominators > 0))
    )
    return valid, denominators


def _metric_denominators(problem: EigenproblemLike, vectors: Array, /) -> Array:
    space = problem.operator.source
    metric_vectors = _metric_coordinate_columns(problem, vectors)
    return jax.vmap(
        lambda vector, metric_vector: jnp.real(
            _coordinate_inner(space, vector, metric_vector)
        ),
        in_axes=1,
    )(vectors, metric_vectors)


@jax.custom_jvp
def _mathematical_eigenvalues(
    problem: EigenproblemLike,
    values: Array,
    vectors: Array,
    denominators: Array,
    /,
) -> Array:
    del problem, vectors, denominators
    return values


@_mathematical_eigenvalues.defjvp
def _mathematical_eigenvalues_jvp(primals, tangents):
    problem, values, vectors, denominators = primals
    problem_tangent, _, _, _ = tangents
    space = problem.operator.source

    def perturbation(current_problem):
        contributions = []
        for index in range(values.shape[0]):
            vector = vectors[:, index]
            mathematical_vector = space.unflatten(vector)
            operator_image = current_problem.operator.mv(mathematical_vector)
            numerator = space.inner(mathematical_vector, operator_image)
            if isinstance(problem, GeneralizedEigenproblem):
                metric_image = current_problem.metric_operator.mv(mathematical_vector)
                numerator = numerator - values[index] * space.inner(
                    mathematical_vector,
                    metric_image,
                )
            contributions.append(jnp.real(numerator) / denominators[index])
        return jnp.stack(contributions)

    _, value_tangent = jax.jvp(
        perturbation,
        (problem,),
        (problem_tangent,),
    )
    return values, value_tangent


def _stop_arrays(value: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        value,
    )


def _require_problem(problem: object, /) -> None:
    if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
        raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")


__all__ = [
    "eigensolve",
    "plan_eigensolve",
    "prepare_eigensolve",
    "refresh_eigensolve",
]
