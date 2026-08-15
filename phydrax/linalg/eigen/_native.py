#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from .._spaces import _coordinate_pairing_matrix
from ..krylov._decompositions import _block_inner, _orthonormalize_block
from ._problems import GeneralizedEigenproblem
from ._results import _NativeEigenResult


def _solve_dense_eigh(prepared: Any, /) -> _NativeEigenResult:
    """Solve a prepared standard or generalized dense Hermitian problem."""
    problem = prepared.problem
    policy = prepared.plan.policy
    state = prepared.dense_state
    if problem.batch_shape:
        return _solve_batched_dense_eigh(prepared)
    values, reduced_vectors = jnp.linalg.eigh(
        state.reduced_operator,
        symmetrize_input=False,
    )
    vectors = jsp.linalg.solve_triangular(
        jnp.conj(state.metric_factor.T),
        reduced_vectors,
        lower=False,
    )
    if policy.which == "smallest-algebraic":
        criterion = values
    elif policy.which == "largest-algebraic":
        criterion = -values
    elif policy.which == "smallest-magnitude":
        criterion = jnp.abs(values)
    else:
        criterion = -jnp.abs(values)
    order = jnp.argsort(criterion, stable=True)
    selected_indices = order[: policy.count]
    selected_values = jnp.real(values[selected_indices])
    selected_vectors = vectors[:, selected_indices]
    mode_mask = jnp.ones((policy.count,), dtype=bool)
    operator_vectors, operator_count = _operator_columns(
        problem.operator,
        selected_vectors,
        mode_mask,
    )
    metric_vectors, metric_count = _metric_columns(
        problem,
        selected_vectors,
        mode_mask,
    )
    _, residual_norms, relative_residuals = _residual_evidence(
        problem.operator.source,
        selected_values,
        selected_vectors,
        operator_vectors,
        metric_vectors,
        mode_mask,
        prepared.metric_constraint_basis,
    )
    orthogonality = _orthogonality_error(
        problem.operator.source,
        selected_vectors,
        metric_vectors,
        mode_mask,
    )
    converged = _mode_convergence(
        residual_norms,
        relative_residuals,
        mode_mask,
        policy.tolerance.absolute,
        policy.tolerance.relative,
    ) & (
        orthogonality <= jnp.asarray(policy.tolerance.orthogonality, orthogonality.dtype)
    )
    isolation_gaps = _dense_isolation_gaps(
        selected_values,
        values,
        selected_indices,
        residual_norms,
        relative_residuals,
        policy.which,
    )
    return _NativeEigenResult(
        values=selected_values,
        vectors=selected_vectors,
        mode_mask=mode_mask,
        converged=converged,
        residual_norms=residual_norms,
        relative_residuals=relative_residuals,
        orthogonality_error=orthogonality,
        iterations=jnp.asarray(1, dtype=jnp.int32),
        operator_matvec_count=jnp.asarray(
            state.operator_matvec_count + operator_count,
            dtype=jnp.int32,
        ),
        metric_matvec_count=jnp.asarray(
            state.metric_matvec_count + metric_count,
            dtype=jnp.int32,
        ),
        preconditioner_apply_count=jnp.asarray(0, dtype=jnp.int32),
        isolation_gaps=isolation_gaps,
        rank_deficient=jnp.asarray(False),
    )


def _solve_batched_dense_eigh(prepared: Any, /) -> _NativeEigenResult:
    problem = prepared.problem
    policy = prepared.plan.policy
    state = prepared.dense_state
    batch_shape = problem.batch_shape
    values, reduced_vectors = jnp.linalg.eigh(
        state.reduced_operator,
        symmetrize_input=False,
    )
    vectors = jsp.linalg.solve_triangular(
        jnp.conj(jnp.swapaxes(state.metric_factor, -1, -2)),
        reduced_vectors,
        lower=False,
    )
    if policy.which == "smallest-algebraic":
        criterion = values
    elif policy.which == "largest-algebraic":
        criterion = -values
    elif policy.which == "smallest-magnitude":
        criterion = jnp.abs(values)
    else:
        criterion = -jnp.abs(values)
    order = jnp.argsort(criterion, axis=-1, stable=True)
    selected_indices = order[..., : policy.count]
    selected_values = jnp.real(
        jnp.take_along_axis(values, selected_indices, axis=-1)
    )
    selected_vectors = jnp.take_along_axis(
        vectors,
        selected_indices[..., None, :],
        axis=-1,
    )
    mode_mask = jnp.ones(batch_shape + (policy.count,), dtype=bool)
    operator_vectors = _batched_operator_images(
        problem.operator,
        selected_vectors,
    )
    metric_vectors = (
        _batched_operator_images(problem.metric_operator, selected_vectors)
        if isinstance(problem, GeneralizedEigenproblem)
        else selected_vectors
    )
    residual = operator_vectors - metric_vectors * selected_values[..., None, :]
    pairing = _coordinate_pairing_matrix(problem.operator.source)
    residual_norms = _batched_column_norms(pairing, residual)
    operator_norms = _batched_column_norms(pairing, operator_vectors)
    metric_norms = _batched_column_norms(pairing, metric_vectors)
    scale = operator_norms + jnp.abs(selected_values) * metric_norms
    relative_residuals = residual_norms / jnp.maximum(
        scale,
        jnp.finfo(residual_norms.dtype).tiny,
    )
    gram = jnp.einsum(
        "...ni,nm,...mj->...ij",
        jnp.conj(selected_vectors),
        pairing,
        metric_vectors,
    )
    identity = jnp.eye(policy.count, dtype=gram.dtype)
    orthogonality = jnp.max(
        jnp.abs(gram - identity),
        axis=(-2, -1),
    )
    converged = _mode_convergence(
        residual_norms,
        relative_residuals,
        mode_mask,
        policy.tolerance.absolute,
        policy.tolerance.relative,
    ) & (
        orthogonality[..., None]
        <= jnp.asarray(policy.tolerance.orthogonality, orthogonality.dtype)
    )
    isolation_gaps = _batched_dense_isolation_gaps(
        selected_values,
        values,
        selected_indices,
        residual_norms,
        relative_residuals,
        policy.which,
    )
    per_batch = lambda value: jnp.full(batch_shape, value, dtype=jnp.int32)
    return _NativeEigenResult(
        values=selected_values,
        vectors=selected_vectors,
        mode_mask=mode_mask,
        converged=converged,
        residual_norms=residual_norms,
        relative_residuals=relative_residuals,
        orthogonality_error=orthogonality,
        iterations=per_batch(1),
        operator_matvec_count=per_batch(
            state.operator_matvec_count + policy.count
        ),
        metric_matvec_count=per_batch(
            state.metric_matvec_count
            + (
                policy.count
                if isinstance(problem, GeneralizedEigenproblem)
                else 0
            )
        ),
        preconditioner_apply_count=per_batch(0),
        isolation_gaps=isolation_gaps,
        rank_deficient=jnp.zeros(batch_shape, dtype=bool),
    )


def _batched_operator_images(operator: Any, vectors: Array, /) -> Array:
    batch_shape = operator.batch_shape
    space = operator.source
    width = vectors.shape[-1]
    structured = vectors.reshape(batch_shape + space.shape + (width,))
    images = operator.mv(structured)
    return jnp.asarray(images).reshape(batch_shape + (space.size, width))


def _batched_column_norms(pairing: Array, block: Array, /) -> Array:
    squared = jnp.einsum(
        "...ni,nm,...mi->...i",
        jnp.conj(block),
        pairing,
        block,
    )
    return jnp.sqrt(jnp.maximum(jnp.real(squared), 0))


def _batched_dense_isolation_gaps(
    selected_values: Array,
    all_values: Array,
    selected_indices: Array,
    residual_norms: Array,
    relative_residuals: Array,
    which: str,
    /,
) -> Array:
    distances = jnp.abs(
        selected_values[..., :, None] - all_values[..., None, :]
    )
    if which in ("smallest-magnitude", "largest-magnitude"):
        target_distances = jnp.abs(
            jnp.abs(selected_values)[..., :, None]
            - jnp.abs(all_values)[..., None, :]
        )
        distances = jnp.minimum(distances, target_distances)
    selected_scale = jnp.maximum(jnp.abs(selected_values), 1)
    selected_uncertainty = 4 * jnp.maximum(
        residual_norms,
        relative_residuals * selected_scale,
    )
    all_uncertainty = (
        jnp.sqrt(jnp.finfo(all_values.dtype).eps)
        * max(all_values.shape[-1], 1)
        * jnp.maximum(jnp.abs(all_values), 1)
    )
    distances = (
        distances
        - selected_uncertainty[..., :, None]
        - all_uncertainty[..., None, :]
    )
    neighbors = (
        selected_indices[..., :, None]
        != jnp.arange(all_values.shape[-1])[None, :]
    )
    distances = jnp.where(neighbors, distances, jnp.asarray(jnp.inf))
    return jnp.min(distances, axis=-1)


def _dense_isolation_gaps(
    selected_values: Array,
    all_values: Array,
    selected_indices: Array,
    residual_norms: Array,
    relative_residuals: Array,
    which: str,
    /,
) -> Array:
    distances = jnp.abs(selected_values[:, None] - all_values[None, :])
    if which in ("smallest-magnitude", "largest-magnitude"):
        target_distances = jnp.abs(
            jnp.abs(selected_values)[:, None] - jnp.abs(all_values)[None, :]
        )
        distances = jnp.minimum(distances, target_distances)
    scale = jnp.maximum(jnp.abs(selected_values), 1)
    selected_uncertainty = 4 * jnp.maximum(
        residual_norms,
        relative_residuals * scale,
    )
    all_uncertainty = (
        jnp.sqrt(jnp.finfo(all_values.dtype).eps)
        * max(all_values.shape[0], 1)
        * jnp.maximum(jnp.abs(all_values), 1)
    )
    distances = distances - selected_uncertainty[:, None] - all_uncertainty[None, :]
    neighbors = selected_indices[:, None] != jnp.arange(all_values.shape[0])[None, :]
    distances = jnp.where(neighbors, distances, jnp.asarray(jnp.inf))
    return jnp.min(distances, axis=1)


def _solve_lobpcg(prepared: Any, /) -> _NativeEigenResult:
    """Run a fixed-capacity, pairing-aware block LOBPCG iteration."""
    problem = prepared.problem
    plan = prepared.plan
    policy = plan.policy
    space = problem.operator.source
    width = plan.block_dimension
    count = policy.count
    required = count + int(policy.differentiation == "eigenvalues")
    real_dtype = prepared.initial_basis.real.dtype
    rank_tolerance = jnp.sqrt(jnp.finfo(real_dtype).eps)
    initial_mask = jnp.arange(width) < jnp.asarray(prepared.initial_rank, dtype=jnp.int32)
    initial = jnp.where(initial_mask[None, :], prepared.initial_basis, 0)
    metric_initial, metric_count = _metric_columns(problem, initial, initial_mask)
    initial, metric_initial = _project_constraints(
        space,
        initial,
        metric_initial,
        prepared.constraint_basis,
        prepared.metric_constraint_basis,
    )
    basis, metric_basis, basis_mask = _orthonormalize_seed(
        space,
        initial,
        metric_initial,
        initial_mask,
        rank_tolerance,
        generalized=problem.kind == "generalized",
    )
    operator_basis, operator_count = _operator_columns(
        problem.operator, basis, basis_mask
    )
    (
        values,
        basis,
        operator_basis,
        metric_basis,
        basis_mask,
    ) = _rayleigh_ritz(
        space,
        basis,
        operator_basis,
        metric_basis,
        basis_mask,
        policy.which,
    )
    residuals, residual_norms, relative_residuals = _residual_evidence(
        space,
        values,
        basis,
        operator_basis,
        metric_basis,
        basis_mask,
        prepared.metric_constraint_basis,
    )
    converged = _mode_convergence(
        residual_norms,
        relative_residuals,
        basis_mask,
        policy.tolerance.absolute,
        policy.tolerance.relative,
    )
    locked = converged
    zeros = jnp.zeros_like(basis)
    direction_mask = jnp.zeros((width,), dtype=jnp.bool_)
    state = (
        basis,
        operator_basis,
        metric_basis,
        zeros,
        zeros,
        zeros,
        direction_mask,
        basis_mask,
        locked,
        values,
        residuals,
        residual_norms,
        relative_residuals,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(operator_count, dtype=jnp.int32),
        jnp.asarray(metric_count, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(iteration, current):
        (
            x,
            ax,
            bx,
            direction,
            adirection,
            bdirection,
            has_direction,
            mode_mask,
            locked_modes,
            theta,
            residual,
            residual_norm,
            relative_residual,
            iterations,
            matvecs,
            metric_matvecs,
            preconditioner_applies,
        ) = current
        orthogonality_ok = _orthogonality_error(
            space,
            x[:, :required],
            bx[:, :required],
            mode_mask[:required],
        ) <= jnp.asarray(
            policy.tolerance.orthogonality,
            real_dtype,
        )
        incomplete = ~(
            jnp.all(locked_modes[:required])
            & jnp.all(mode_mask[:required])
            & orthogonality_ok
        )
        execute = jnp.any(mode_mask) & incomplete

        def update(operand):
            (
                x_i,
                ax_i,
                bx_i,
                direction_i,
                adirection_i,
                bdirection_i,
                has_direction_i,
                mode_mask_i,
                locked_i,
                theta_i,
                residual_i,
                _,
                _,
                iterations_i,
                matvecs_i,
                metric_matvecs_i,
                preconditioner_applies_i,
            ) = operand
            locked_i = locked_i & orthogonality_ok
            active = mode_mask_i & ~locked_i
            search, applied = _precondition_columns(
                prepared,
                residual_i,
                active,
                jnp.asarray(iteration, dtype=jnp.int32),
            )
            metric_search, metric_used = _metric_columns(problem, search, active)
            search, metric_search = _project_constraints(
                space,
                search,
                metric_search,
                prepared.constraint_basis,
                prepared.metric_constraint_basis,
            )
            operator_search, operator_used = _operator_columns(
                problem.operator, search, active
            )
            trial = jnp.concatenate((x_i, search, direction_i), axis=1)
            operator_trial = jnp.concatenate(
                (ax_i, operator_search, adirection_i), axis=1
            )
            metric_trial = jnp.concatenate((bx_i, metric_search, bdirection_i), axis=1)
            trial_mask = jnp.concatenate((mode_mask_i, active, has_direction_i), axis=0)
            (
                trial,
                operator_trial,
                metric_trial,
                trial_mask,
            ) = _orthonormalize_images(
                space,
                trial,
                operator_trial,
                metric_trial,
                trial_mask,
                rank_tolerance,
            )
            (
                trial_values,
                trial,
                operator_trial,
                metric_trial,
                trial_mask,
            ) = _rayleigh_ritz(
                space,
                trial,
                operator_trial,
                metric_trial,
                trial_mask,
                policy.which,
            )
            candidate = trial[:, :width]
            operator_candidate = operator_trial[:, :width]
            metric_candidate = metric_trial[:, :width]
            candidate_mask = trial_mask[:width]
            candidate_values = trial_values[:width]
            locked_columns = locked_i[None, :]
            next_x = jnp.where(locked_columns, x_i, candidate)
            next_ax = jnp.where(locked_columns, ax_i, operator_candidate)
            next_bx = jnp.where(locked_columns, bx_i, metric_candidate)
            next_values = jnp.where(locked_i, theta_i, candidate_values)
            next_mask = jnp.where(locked_i, mode_mask_i, candidate_mask)
            next_direction = jnp.where(active[None, :], next_x - x_i, jnp.zeros_like(x_i))
            next_adirection = jnp.where(
                active[None, :], next_ax - ax_i, jnp.zeros_like(ax_i)
            )
            next_bdirection = jnp.where(
                active[None, :], next_bx - bx_i, jnp.zeros_like(bx_i)
            )
            next_direction_mask = active & next_mask
            (
                next_residual,
                next_residual_norm,
                next_relative_residual,
            ) = _residual_evidence(
                space,
                next_values,
                next_x,
                next_ax,
                next_bx,
                next_mask,
                prepared.metric_constraint_basis,
            )
            newly_converged = _mode_convergence(
                next_residual_norm,
                next_relative_residual,
                next_mask,
                policy.tolerance.absolute,
                policy.tolerance.relative,
            )
            return (
                next_x,
                next_ax,
                next_bx,
                next_direction,
                next_adirection,
                next_bdirection,
                next_direction_mask,
                next_mask,
                locked_i | newly_converged,
                next_values,
                next_residual,
                next_residual_norm,
                next_relative_residual,
                iterations_i + 1,
                matvecs_i + operator_used,
                metric_matvecs_i + metric_used,
                preconditioner_applies_i + applied,
            )

        return jax.lax.cond(execute, update, lambda operand: operand, current)

    state = jax.lax.fori_loop(0, policy.max_steps, step, state)
    (
        basis,
        _,
        metric_basis,
        _,
        _,
        _,
        _,
        mode_mask,
        locked,
        values,
        _,
        residual_norms,
        relative_residuals,
        iterations,
        operator_count,
        metric_count,
        preconditioner_count,
    ) = state
    return _final_result(
        space,
        values,
        basis,
        metric_basis,
        mode_mask,
        locked,
        residual_norms,
        relative_residuals,
        count=count,
        required=required,
        retained=width,
        which=policy.which,
        orthogonality_tolerance=policy.tolerance.orthogonality,
        iterations=iterations,
        operator_count=operator_count,
        metric_count=metric_count,
        preconditioner_count=preconditioner_count,
    )


def _solve_restarted_lanczos(prepared: Any, /) -> _NativeEigenResult:
    """Run fixed-capacity thick-restarted, fully reorthogonalized Lanczos."""
    problem = prepared.problem
    plan = prepared.plan
    policy = plan.policy
    space = problem.operator.source
    seed_width = plan.block_dimension
    retained = plan.restart_dimension
    capacity = plan.subspace_dimension
    count = policy.count
    required = count + int(policy.differentiation == "eigenvalues")
    real_dtype = prepared.initial_basis.real.dtype
    rank_tolerance = jnp.sqrt(jnp.finfo(real_dtype).eps)
    kept_seed_width = min(seed_width, retained)
    seed = jnp.zeros(
        (prepared.initial_basis.shape[0], retained), dtype=prepared.initial_basis.dtype
    )
    seed = seed.at[:, :kept_seed_width].set(prepared.initial_basis[:, :kept_seed_width])
    initial_rank = jnp.minimum(
        jnp.asarray(prepared.initial_rank, dtype=jnp.int32), kept_seed_width
    )
    seed_mask = jnp.arange(retained) < initial_rank
    metric_seed, metric_count = _metric_columns(problem, seed, seed_mask)
    seed, metric_seed = _project_constraints(
        space,
        seed,
        metric_seed,
        prepared.constraint_basis,
        prepared.metric_constraint_basis,
    )
    seed, metric_seed, seed_mask = _orthonormalize_seed(
        space,
        seed,
        metric_seed,
        seed_mask,
        rank_tolerance,
        generalized=problem.kind == "generalized",
    )
    operator_seed, operator_count = _operator_columns(problem.operator, seed, seed_mask)
    (
        values,
        seed,
        operator_seed,
        metric_seed,
        seed_mask,
    ) = _rayleigh_ritz(
        space,
        seed,
        operator_seed,
        metric_seed,
        seed_mask,
        policy.which,
    )
    residuals, residual_norms, relative_residuals = _residual_evidence(
        space,
        values,
        seed,
        operator_seed,
        metric_seed,
        seed_mask,
        prepared.metric_constraint_basis,
    )
    converged = _mode_convergence(
        residual_norms,
        relative_residuals,
        seed_mask,
        policy.tolerance.absolute,
        policy.tolerance.relative,
    )
    state = (
        seed,
        operator_seed,
        metric_seed,
        seed_mask,
        converged,
        values,
        residuals,
        residual_norms,
        relative_residuals,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(operator_count, dtype=jnp.int32),
        jnp.asarray(metric_count, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def restart(cycle, current):
        (
            retained_basis,
            retained_operator_basis,
            retained_metric_basis,
            retained_mask,
            locked,
            retained_values,
            retained_residuals,
            _,
            _,
            iterations,
            matvecs,
            metric_matvecs,
            preconditioner_applies,
        ) = current
        orthogonality_ok = _orthogonality_error(
            space,
            retained_basis[:, :required],
            retained_metric_basis[:, :required],
            retained_mask[:required],
        ) <= jnp.asarray(
            policy.tolerance.orthogonality,
            real_dtype,
        )
        incomplete = ~(
            jnp.all(locked[:required])
            & jnp.all(retained_mask[:required])
            & orthogonality_ok
        )
        execute = jnp.any(retained_mask) & incomplete

        def expand_and_restart(operand):
            (
                retained_basis_i,
                retained_operator_basis_i,
                retained_metric_basis_i,
                retained_mask_i,
                locked_i,
                retained_values_i,
                retained_residuals_i,
                _,
                _,
                iterations_i,
                matvecs_i,
                metric_matvecs_i,
                preconditioner_applies_i,
            ) = operand
            locked_i = locked_i & orthogonality_ok
            basis = jnp.zeros(
                (retained_basis_i.shape[0], capacity), dtype=retained_basis_i.dtype
            )
            operator_basis = jnp.zeros_like(basis)
            metric_basis = jnp.zeros_like(basis)
            basis = basis.at[:, :retained].set(retained_basis_i)
            operator_basis = operator_basis.at[:, :retained].set(
                retained_operator_basis_i
            )
            metric_basis = metric_basis.at[:, :retained].set(retained_metric_basis_i)
            basis_mask = jnp.zeros((capacity,), dtype=jnp.bool_)
            basis_mask = basis_mask.at[:retained].set(retained_mask_i)
            expansion_state = (
                basis,
                operator_basis,
                metric_basis,
                basis_mask,
                matvecs_i,
                metric_matvecs_i,
                preconditioner_applies_i,
            )

            def expand(index, expansion):
                (
                    basis_i,
                    operator_basis_i,
                    metric_basis_i,
                    basis_mask_i,
                    matvecs_j,
                    metric_matvecs_j,
                    preconditioner_applies_j,
                ) = expansion
                nominal_source = index - retained
                unlocked_retained = retained_mask_i & ~locked_i
                unlocked_ordinal = jnp.cumsum(unlocked_retained.astype(jnp.int32)) - 1
                residual_source_mask = unlocked_retained & (
                    unlocked_ordinal == nominal_source
                )
                use_residual = jnp.any(residual_source_mask)
                residual_source = jnp.argmax(residual_source_mask.astype(jnp.int32))
                prior = (jnp.arange(capacity) < index) & basis_mask_i
                latest_source = jnp.max(jnp.where(prior, jnp.arange(capacity), 0))
                source_index = jnp.where(
                    use_residual,
                    residual_source,
                    latest_source,
                )
                active = jax.lax.cond(
                    use_residual,
                    lambda i: retained_mask_i[i],
                    lambda i: basis_mask_i[i],
                    source_index,
                )
                raw = jax.lax.cond(
                    use_residual,
                    lambda i: retained_residuals_i[:, i],
                    lambda i: operator_basis_i[:, i],
                    source_index,
                )
                raw = jnp.where(active, raw, 0)
                applied = jnp.asarray(0, dtype=jnp.int32)
                raw_block = raw[:, None]
                raw_mask = active[None]
                metric_raw, metric_used = _metric_columns(problem, raw_block, raw_mask)
                raw_block, metric_raw = _project_constraints(
                    space,
                    raw_block,
                    metric_raw,
                    prepared.constraint_basis,
                    prepared.metric_constraint_basis,
                )
                candidate = raw_block[:, 0]
                metric_candidate = metric_raw[:, 0]
                initial_norm = _paired_norm(space, candidate, metric_candidate)

                def orthogonalize(_, vectors):
                    candidate_i, metric_candidate_i = vectors
                    coefficients = _block_inner(
                        basis_i,
                        metric_candidate_i[:, None],
                        _coordinate_inner(space),
                    )[:, 0]
                    coefficients = jnp.where(basis_mask_i, coefficients, 0)
                    return (
                        candidate_i - basis_i @ coefficients,
                        metric_candidate_i - metric_basis_i @ coefficients,
                    )

                candidate, metric_candidate = jax.lax.fori_loop(
                    0,
                    2,
                    orthogonalize,
                    (candidate, metric_candidate),
                )
                norm = _paired_norm(space, candidate, metric_candidate)
                threshold = rank_tolerance * jnp.where(
                    initial_norm > 0.0, initial_norm, 1.0
                )
                finite = (
                    jnp.all(jnp.isfinite(candidate))
                    & jnp.all(jnp.isfinite(metric_candidate))
                    & jnp.isfinite(norm)
                )
                independent = active & finite & (norm > threshold)
                safe_norm = jnp.where(independent, norm, 1.0)
                candidate = jnp.where(independent, candidate / safe_norm, 0)
                metric_candidate = jnp.where(independent, metric_candidate / safe_norm, 0)
                operator_candidate, operator_used = _operator_columns(
                    problem.operator,
                    candidate[:, None],
                    independent[None],
                )
                basis_i = basis_i.at[:, index].set(candidate)
                metric_basis_i = metric_basis_i.at[:, index].set(metric_candidate)
                operator_basis_i = operator_basis_i.at[:, index].set(
                    operator_candidate[:, 0]
                )
                basis_mask_i = basis_mask_i.at[index].set(independent)
                return (
                    basis_i,
                    operator_basis_i,
                    metric_basis_i,
                    basis_mask_i,
                    matvecs_j + operator_used,
                    metric_matvecs_j + metric_used,
                    preconditioner_applies_j + applied,
                )

            (
                basis,
                operator_basis,
                metric_basis,
                basis_mask,
                next_matvecs,
                next_metric_matvecs,
                next_preconditioner_applies,
            ) = jax.lax.fori_loop(
                retained,
                capacity,
                expand,
                expansion_state,
            )
            (
                ritz_values,
                ritz_basis,
                ritz_operator_basis,
                ritz_metric_basis,
                ritz_mask,
            ) = _rayleigh_ritz(
                space,
                basis,
                operator_basis,
                metric_basis,
                basis_mask,
                policy.which,
            )
            next_basis = ritz_basis[:, :retained]
            next_operator_basis = ritz_operator_basis[:, :retained]
            next_metric_basis = ritz_metric_basis[:, :retained]
            next_mask = ritz_mask[:retained]
            next_values = ritz_values[:retained]
            locked_columns = locked_i[None, :]
            next_basis = jnp.where(locked_columns, retained_basis_i, next_basis)
            next_operator_basis = jnp.where(
                locked_columns, retained_operator_basis_i, next_operator_basis
            )
            next_metric_basis = jnp.where(
                locked_columns, retained_metric_basis_i, next_metric_basis
            )
            next_mask = jnp.where(locked_i, retained_mask_i, next_mask)
            next_values = jnp.where(locked_i, retained_values_i, next_values)
            (
                next_residuals,
                next_residual_norms,
                next_relative_residuals,
            ) = _residual_evidence(
                space,
                next_values,
                next_basis,
                next_operator_basis,
                next_metric_basis,
                next_mask,
                prepared.metric_constraint_basis,
            )
            newly_converged = _mode_convergence(
                next_residual_norms,
                next_relative_residuals,
                next_mask,
                policy.tolerance.absolute,
                policy.tolerance.relative,
            )
            return (
                next_basis,
                next_operator_basis,
                next_metric_basis,
                next_mask,
                locked_i | newly_converged,
                next_values,
                next_residuals,
                next_residual_norms,
                next_relative_residuals,
                iterations_i + 1,
                next_matvecs,
                next_metric_matvecs,
                next_preconditioner_applies,
            )

        return jax.lax.cond(execute, expand_and_restart, lambda operand: operand, current)

    state = jax.lax.fori_loop(0, policy.max_steps, restart, state)
    (
        basis,
        _,
        metric_basis,
        mode_mask,
        locked,
        values,
        _,
        residual_norms,
        relative_residuals,
        iterations,
        operator_count,
        metric_count,
        preconditioner_count,
    ) = state
    return _final_result(
        space,
        values,
        basis,
        metric_basis,
        mode_mask,
        locked,
        residual_norms,
        relative_residuals,
        count=count,
        required=required,
        retained=retained,
        which=policy.which,
        orthogonality_tolerance=policy.tolerance.orthogonality,
        iterations=iterations,
        operator_count=operator_count,
        metric_count=metric_count,
        preconditioner_count=preconditioner_count,
    )


def _coordinate_inner(space: Any):
    def inner(left, right):
        return space.inner(space.unflatten(left), space.unflatten(right))

    return inner


def _operator_columns(operator: Any, block: Array, mask: Array, /):
    output = jnp.zeros_like(block)

    def apply(index, images):
        def active_action(value):
            vector = operator.source.unflatten(block[:, index])
            image = operator.mv(vector)
            return value.at[:, index].set(operator.target.flatten(image))

        return jax.lax.cond(mask[index], active_action, lambda value: value, images)

    output = jax.lax.fori_loop(0, block.shape[1], apply, output)
    return output, jnp.sum(mask, dtype=jnp.int32)


def _metric_columns(problem: Any, block: Array, mask: Array, /):
    if problem.kind == "standard":
        return jnp.where(mask[None, :], block, 0), jnp.asarray(0, dtype=jnp.int32)
    return _operator_columns(problem.metric_operator, block, mask)


def _precondition_columns(
    prepared: Any,
    block: Array,
    mask: Array,
    iteration: Array,
    /,
):
    if prepared.preconditioning_state is None:
        return jnp.where(mask[None, :], block, 0), jnp.asarray(0, dtype=jnp.int32)
    action = prepared.preconditioning_state.action
    space = prepared.problem.operator.source
    output = jnp.zeros_like(block)

    def apply(index, images):
        def active_action(value):
            vector = space.unflatten(block[:, index])
            image = action.apply(vector, iteration=iteration)
            return value.at[:, index].set(space.flatten(image))

        return jax.lax.cond(mask[index], active_action, lambda value: value, images)

    output = jax.lax.fori_loop(0, block.shape[1], apply, output)
    return output, jnp.sum(mask, dtype=jnp.int32)


def _project_constraints(
    space: Any,
    block: Array,
    metric_block: Array,
    constraints: Array,
    metric_constraints: Array,
    /,
):
    if constraints.shape[1] == 0:
        return block, metric_block
    inner = _coordinate_inner(space)
    gram = _block_inner(constraints, metric_constraints, inner)
    gram = 0.5 * (gram + jnp.conj(gram.T))
    right = _block_inner(constraints, metric_block, inner)
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(jnp.real(eigenvalues), 0.0)
    largest = jnp.max(eigenvalues)
    tolerance = (
        constraints.shape[1]
        * jnp.finfo(block.real.dtype).eps
        * jnp.where(largest > 0.0, largest, 1.0)
    )
    active = jnp.isfinite(eigenvalues) & (eigenvalues > tolerance)
    inverse_values = jnp.where(active, 1.0 / jnp.where(active, eigenvalues, 1.0), 0)
    inverse = (eigenvectors * inverse_values[None, :]) @ jnp.conj(eigenvectors.T)
    coefficients = inverse @ right
    return (
        block - constraints @ coefficients,
        metric_block - metric_constraints @ coefficients,
    )


def _orthonormalize_seed(
    space: Any,
    block: Array,
    metric_block: Array,
    mask: Array,
    tolerance: Array,
    /,
    *,
    generalized: bool,
):
    block = jnp.where(mask[None, :], block, 0)
    metric_block = jnp.where(mask[None, :], metric_block, 0)
    if not generalized:
        basis, _, rank = _orthonormalize_block(block, _coordinate_inner(space), tolerance)
        active = jnp.arange(block.shape[1]) < rank
        basis = jnp.where(active[None, :], basis, 0)
        return basis, basis, active
    basis, _, metric_basis, active = _orthonormalize_images(
        space,
        block,
        block,
        metric_block,
        mask,
        tolerance,
    )
    return basis, metric_basis, active


def _orthonormalize_images(
    space: Any,
    block: Array,
    operator_block: Array,
    metric_block: Array,
    mask: Array,
    tolerance: Array,
    /,
):
    block = jnp.where(mask[None, :], block, 0)
    operator_block = jnp.where(mask[None, :], operator_block, 0)
    metric_block = jnp.where(mask[None, :], metric_block, 0)
    diagonal = jnp.real(
        jnp.diag(_block_inner(block, metric_block, _coordinate_inner(space)))
    )
    column_norms = jnp.sqrt(jnp.maximum(diagonal, 0))
    normalizable = mask & jnp.isfinite(column_norms) & (column_norms > 0)
    safe_column_norms = jnp.where(normalizable, column_norms, 1)
    block = jnp.where(
        normalizable[None, :],
        block / safe_column_norms[None, :],
        0,
    )
    operator_block = jnp.where(
        normalizable[None, :],
        operator_block / safe_column_norms[None, :],
        0,
    )
    metric_block = jnp.where(
        normalizable[None, :],
        metric_block / safe_column_norms[None, :],
        0,
    )
    gram = _block_inner(block, metric_block, _coordinate_inner(space))
    gram = 0.5 * (gram + jnp.conj(gram.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(jnp.real(eigenvalues[order]), 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = jnp.sqrt(eigenvalues)
    largest = singular_values[0]
    valid_scale = jnp.isfinite(largest) & (largest > 0.0)
    threshold = tolerance * jnp.where(valid_scale, largest, 1.0)
    active = valid_scale & jnp.isfinite(singular_values) & (singular_values > threshold)
    safe = jnp.where(active, singular_values, 1.0)
    transform = eigenvectors / safe[None, :]
    transform = jnp.where(active[None, :], transform, 0)
    return (
        block @ transform,
        operator_block @ transform,
        metric_block @ transform,
        active,
    )


def _rayleigh_ritz(
    space: Any,
    basis: Array,
    operator_basis: Array,
    metric_basis: Array,
    mask: Array,
    which: str,
    /,
):
    inner = _coordinate_inner(space)
    projected = _block_inner(basis, operator_basis, inner)
    gram = _block_inner(basis, metric_basis, inner)
    projected = 0.5 * (projected + jnp.conj(projected.T))
    gram = 0.5 * (gram + jnp.conj(gram.T))
    pair_mask = mask[:, None] & mask[None, :]
    projected = jnp.where(pair_mask, projected, 0)
    gram = jnp.where(pair_mask, gram, 0)
    scale = jnp.max(jnp.abs(projected)) + jnp.asarray(1, projected.real.dtype)
    padding = scale * (basis.shape[1] + 2)
    padded_projected = projected + jnp.diag(
        jnp.where(mask, jnp.asarray(0, projected.real.dtype), padding)
    )
    padded_gram = gram + jnp.diag(jnp.where(mask, jnp.asarray(0, gram.real.dtype), 1))
    gram_values, gram_vectors = jnp.linalg.eigh(padded_gram)
    gram_floor = jnp.finfo(padded_gram.real.dtype).eps * jnp.maximum(
        jnp.max(jnp.abs(gram_values)), 1
    )
    safe_gram_values = jnp.where(
        jnp.isfinite(gram_values),
        jnp.maximum(jnp.real(gram_values), gram_floor),
        jnp.asarray(jnp.nan, gram_values.real.dtype),
    )
    whitening = (gram_vectors / jnp.sqrt(safe_gram_values)[None, :]) @ jnp.conj(
        gram_vectors.T
    )
    whitened = jnp.conj(whitening.T) @ padded_projected @ whitening
    whitened = 0.5 * (whitened + jnp.conj(whitened.T))
    values, vectors = jnp.linalg.eigh(whitened)
    coefficients = whitening @ vectors
    active_energy = jnp.sum(
        jnp.where(mask[:, None], jnp.abs(coefficients) ** 2, 0), axis=0
    )
    total_energy = jnp.sum(jnp.abs(coefficients) ** 2, axis=0)
    finite_ritz = jnp.isfinite(values) & jnp.all(
        jnp.isfinite(coefficients),
        axis=0,
    )
    valid = finite_ritz & (active_energy > 0.5 * total_energy)
    if which == "smallest-algebraic":
        criterion = values
    elif which == "largest-algebraic":
        criterion = -values
    elif which == "smallest-magnitude":
        criterion = jnp.abs(values)
    else:
        criterion = -jnp.abs(values)
    sort_key = jnp.where(valid, criterion, jnp.asarray(jnp.inf, criterion.dtype))
    order = jnp.argsort(sort_key, stable=True)
    values = jnp.real(values[order])
    coefficients = coefficients[:, order]
    valid = valid[order]
    finite_ritz = finite_ritz[order]
    rotated_basis = basis @ coefficients
    rotated_operator = operator_basis @ coefficients
    rotated_metric = metric_basis @ coefficients
    rotated_basis = jnp.where(valid[None, :], rotated_basis, 0)
    rotated_operator = jnp.where(valid[None, :], rotated_operator, 0)
    rotated_metric = jnp.where(valid[None, :], rotated_metric, 0)
    values = jnp.where(
        valid,
        values,
        jnp.where(finite_ritz, 0, jnp.asarray(jnp.nan, values.dtype)),
    )
    return values, rotated_basis, rotated_operator, rotated_metric, valid


def _residual_evidence(
    space: Any,
    values: Array,
    basis: Array,
    operator_basis: Array,
    metric_basis: Array,
    mask: Array,
    constraint_dual_basis: Array,
    /,
):
    width = operator_basis.shape[1]
    projected_images = _project_dual_residual(
        space,
        jnp.concatenate((operator_basis, metric_basis), axis=1),
        constraint_dual_basis,
    )
    projected_operator = projected_images[:, :width]
    projected_metric = projected_images[:, width:]
    residual = projected_operator - projected_metric * values[None, :]
    residual = jnp.where(mask[None, :], residual, 0)
    residual_norms = _column_norms(space, residual)
    operator_norms = _column_norms(space, projected_operator)
    metric_norms = _column_norms(space, projected_metric)
    scale = operator_norms + jnp.abs(values) * metric_norms
    tiny = jnp.finfo(residual_norms.dtype).tiny
    relative = residual_norms / jnp.maximum(scale, tiny)
    residual_norms = jnp.where(mask, residual_norms, jnp.asarray(jnp.inf))
    relative = jnp.where(mask, relative, jnp.asarray(jnp.inf))
    return residual, residual_norms, relative


def _project_dual_residual(
    space: Any,
    residual: Array,
    constraint_dual_basis: Array,
    /,
) -> Array:
    if constraint_dual_basis.shape[1] == 0:
        return residual
    inner = _coordinate_inner(space)
    column_norms = _column_norms(space, constraint_dual_basis)
    normalizable = jnp.isfinite(column_norms) & (column_norms > 0)
    safe_norms = jnp.where(normalizable, column_norms, 1)
    normalized_dual = jnp.where(
        normalizable[None, :],
        constraint_dual_basis / safe_norms[None, :],
        0,
    )
    gram = _block_inner(normalized_dual, normalized_dual, inner)
    gram = 0.5 * (gram + jnp.conj(gram.T))
    right = _block_inner(normalized_dual, residual, inner)
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(jnp.real(eigenvalues), 0.0)
    largest = jnp.max(eigenvalues)
    cutoff = (
        constraint_dual_basis.shape[1]
        * jnp.finfo(residual.real.dtype).eps
        * jnp.where(largest > 0.0, largest, 1.0)
    )
    active = jnp.isfinite(eigenvalues) & (eigenvalues > cutoff)
    inverse_values = jnp.where(active, 1.0 / jnp.where(active, eigenvalues, 1.0), 0)
    inverse = (eigenvectors * inverse_values[None, :]) @ jnp.conj(eigenvectors.T)
    return residual - normalized_dual @ (inverse @ right)


def _column_norms(space: Any, block: Array, /):
    gram = _block_inner(block, block, _coordinate_inner(space))
    return jnp.sqrt(jnp.maximum(jnp.real(jnp.diag(gram)), 0.0))


def _paired_norm(space: Any, vector: Array, metric_vector: Array, /):
    value = _coordinate_inner(space)(vector, metric_vector)
    return jnp.sqrt(jnp.maximum(jnp.real(value), 0.0))


def _mode_convergence(
    residual_norms: Array,
    relative_residuals: Array,
    mask: Array,
    absolute: float,
    relative: float,
    /,
):
    return mask & (
        (residual_norms <= jnp.asarray(absolute, residual_norms.dtype))
        | (relative_residuals <= jnp.asarray(relative, relative_residuals.dtype))
    )


def _orthogonality_error(
    space: Any,
    basis: Array,
    metric_basis: Array,
    mask: Array,
    /,
):
    gram = _block_inner(basis, metric_basis, _coordinate_inner(space))
    identity = jnp.eye(basis.shape[1], dtype=gram.dtype)
    pair_mask = mask[:, None] & mask[None, :]
    difference = jnp.where(pair_mask, gram - identity, 0)
    return jnp.max(jnp.abs(difference))


def _isolation_gaps(
    values: Array,
    mask: Array,
    converged: Array,
    residual_norms: Array,
    relative_residuals: Array,
    count: int,
    which: str,
    /,
):
    distances = jnp.abs(values[:, None] - values[None, :])
    if which in ("smallest-magnitude", "largest-magnitude"):
        target_distances = jnp.abs(jnp.abs(values)[:, None] - jnp.abs(values)[None, :])
        distances = jnp.minimum(distances, target_distances)
    scale = jnp.maximum(jnp.abs(values), 1)
    uncertainty = 4 * jnp.maximum(
        residual_norms,
        relative_residuals * scale,
    )
    distances = distances - uncertainty[:, None] - uncertainty[None, :]
    indices = jnp.arange(values.shape[0])
    certified = mask & converged
    neighbors = (
        certified[:, None] & certified[None, :] & (indices[:, None] != indices[None, :])
    )
    distances = jnp.where(neighbors, distances, jnp.asarray(jnp.inf))
    gaps = jnp.min(distances, axis=1)
    available = jnp.any(neighbors, axis=1)
    gaps = jnp.where(available, gaps, jnp.asarray(jnp.nan))
    return gaps[:count]


def _final_result(
    space: Any,
    values: Array,
    basis: Array,
    metric_basis: Array,
    mode_mask: Array,
    locked: Array,
    residual_norms: Array,
    relative_residuals: Array,
    /,
    *,
    count: int,
    required: int,
    retained: int,
    which: str,
    orthogonality_tolerance: float,
    iterations: Array,
    operator_count: Array,
    metric_count: Array,
    preconditioner_count: Array,
):
    retained_values = values[:retained]
    retained_mask = mode_mask[:retained]
    requested_mask = mode_mask[:count]
    orthogonality = _orthogonality_error(
        space,
        basis[:, :count],
        metric_basis[:, :count],
        requested_mask,
    )
    orthogonality_ok = orthogonality <= jnp.asarray(
        orthogonality_tolerance, orthogonality.dtype
    )
    converged = locked[:count] & requested_mask & orthogonality_ok
    isolation_mask = retained_mask[:required]
    isolation_orthogonality = _orthogonality_error(
        space,
        basis[:, :required],
        metric_basis[:, :required],
        isolation_mask,
    )
    isolation_certified = locked[:retained] & (
        isolation_orthogonality
        <= jnp.asarray(orthogonality_tolerance, isolation_orthogonality.dtype)
    )
    return _NativeEigenResult(
        values=values[:count],
        vectors=jnp.where(requested_mask[None, :], basis[:, :count], 0),
        mode_mask=requested_mask,
        converged=converged,
        residual_norms=residual_norms[:count],
        relative_residuals=relative_residuals[:count],
        orthogonality_error=orthogonality,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        operator_matvec_count=jnp.asarray(operator_count, dtype=jnp.int32),
        metric_matvec_count=jnp.asarray(metric_count, dtype=jnp.int32),
        preconditioner_apply_count=jnp.asarray(preconditioner_count, dtype=jnp.int32),
        isolation_gaps=_isolation_gaps(
            retained_values,
            retained_mask,
            isolation_certified,
            residual_norms[:retained],
            relative_residuals[:retained],
            count,
            which,
        ),
        rank_deficient=jnp.sum(requested_mask, dtype=jnp.int32) < count,
    )


__all__ = ["_solve_lobpcg", "_solve_restarted_lanczos"]
