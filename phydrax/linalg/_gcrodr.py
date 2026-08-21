#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from ._plans import LinearSolvePlan
from ._policies import FGMRES, GMRES, LinearSolvePolicy
from ._prepared import PreparedLinearSolve
from ._problems import AbstractLinearProblem, LinearSystem
from ._recycling import (
    RecyclingState,
    RecyclingSubspace,
    RecyclingUpdateStatus,
)
from ._recycling_policy import (
    RecyclingExtraction,
    RecyclingPolicy,
    RecyclingRefresh,
)
from ._results import (
    LinearSolveDiagnostics,
    LinearSolveProvenance,
    LinearSolveResult,
    LinearSolveStatus,
    RecycledLinearSolveResult,
)
from ._spaces import _coordinate_dtype
from .krylov._decompositions import arnoldi
from .krylov._results import KrylovBreakdownStatus


def solve_recycled(
    prepared_or_problem: PreparedLinearSolve | AbstractLinearProblem,
    rhs: PyTree[Any],
    /,
    *,
    recycling: RecyclingState | RecyclingSubspace | None = None,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
) -> RecycledLinearSolveResult:
    """Solve one right-hand side and return an updated GCRO-DR space."""
    from ._runtime import prepare

    if isinstance(prepared_or_problem, PreparedLinearSolve):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = prepared_or_problem
    elif isinstance(prepared_or_problem, AbstractLinearProblem):
        selected_policy = (
            LinearSolvePolicy(GMRES(), recycling=RecyclingPolicy())
            if policy is None
            else policy
        )
        prepared = prepare(prepared_or_problem, selected_policy)
    else:
        raise TypeError("Expected an AbstractLinearProblem or PreparedLinearSolve.")
    recycling_policy = prepared.plan.policy.recycling
    if recycling_policy is None:
        raise ValueError("solve_recycled requires a RecyclingPolicy in prepared state.")
    return _solve_recycled(
        prepared,
        rhs,
        recycling,
        capacity=recycling_policy.capacity,
        extraction=recycling_policy.extraction,
        refresh=recycling_policy.refresh,
    )


def refresh_recycling(
    recycling: RecyclingState | RecyclingSubspace,
    prepared: PreparedLinearSolve,
    /,
    *,
    extraction: RecyclingExtraction = "harmonic-ritz",
    refresh: RecyclingRefresh = "reuse-source",
) -> RecyclingState:
    """Refresh numerical images or discard them under one unchanged solve plan."""
    extraction_ = _validate_extraction(extraction)
    refresh_ = _validate_refresh(refresh)
    _validate_prepared(prepared)
    capacity = (
        recycling.capacity
        if isinstance(recycling, RecyclingState)
        else recycling.dimension
        if isinstance(recycling, RecyclingSubspace)
        else 0
    )
    state = _coerce_state(recycling, prepared, capacity, extraction_)
    _validate_state(state, prepared, state.capacity, extraction_)
    if refresh_ == "rebuild":
        return _stop_state(
            _empty_state(
                prepared,
                state.capacity,
                extraction_,
                recycling_id=state.recycling_id,
                update_count=state.update_count + 1,
            )
        )
    return _stop_state(
        _refresh_source_images(
            state,
            prepared,
            preserve_status=isinstance(recycling, RecyclingSubspace),
        )
    )


def _solve_recycled(
    prepared: PreparedLinearSolve,
    rhs: PyTree[Any],
    recycling: RecyclingState | RecyclingSubspace | None,
    *,
    capacity: int,
    extraction: RecyclingExtraction,
    refresh: RecyclingRefresh,
) -> RecycledLinearSolveResult:

    extraction_ = _validate_extraction(extraction)
    refresh_ = _validate_refresh(refresh)
    _validate_prepared(prepared)
    capacity_ = int(capacity)
    if capacity_ < 1:
        raise ValueError("Recycling capacity must be positive.")
    operator = prepared.problem.operator
    target = operator.target
    right_hand_side = target.validate(rhs)
    rhs_coordinates = target.flatten(right_hand_side)
    if rhs_coordinates.ndim != 1:
        raise ValueError(
            "GCRO-DR accepts one right-hand side; use independent solves for batches."
        )

    refresh_matvecs = jnp.asarray(0, dtype=jnp.int32)
    if recycling is None:
        state = _empty_state(prepared, capacity_, extraction_)
    else:
        state = _coerce_state(recycling, prepared, capacity_, extraction_)
        _validate_state(state, prepared, capacity_, extraction_)
        if refresh_ == "rebuild":
            state = _empty_state(
                prepared,
                capacity_,
                extraction_,
                recycling_id=state.recycling_id,
                update_count=state.update_count,
            )
        else:
            refresh_matvecs = state.effective_dimension
            state = _refresh_source_images(
                state,
                prepared,
                increment=False,
                preserve_status=isinstance(recycling, RecyclingSubspace),
            )
    state = _stop_state(state)

    source_basis = state.source_basis
    image_basis = state.image_basis
    coarse_coefficients = _basis_coefficients(
        target,
        image_basis,
        rhs_coordinates,
        state.effective_dimension,
    )
    coarse_value = source_basis @ coarse_coefficients
    projected_rhs = rhs_coordinates - image_basis @ coarse_coefficients

    action = lambda vector: _operator_coordinates(prepared, vector)
    projected_action = lambda vector: _project_coordinates(
        target,
        image_basis,
        state.effective_dimension,
        action(vector),
    )
    inner = lambda left, right: _coordinate_inner(target, left, right)
    max_steps = prepared.plan.policy.tolerance.max_steps or operator.source.size
    restart, stagnation = _gmres_configuration(prepared, max_steps)
    from .backends._native_krylov import _fgmres_raw

    correction, auxiliary = _fgmres_raw(
        projected_action,
        projected_rhs,
        jnp.zeros_like(projected_rhs),
        inner,
        lambda vector, _: vector,
        max_steps,
        restart,
        stagnation,
        prepared.plan.policy.tolerance.relative,
        prepared.plan.policy.tolerance.absolute,
        identity_preconditioner=True,
    )
    image_correction = action(correction)
    correction_coefficients = _basis_coefficients(
        target,
        image_basis,
        image_correction,
        state.effective_dimension,
    )
    augmented_correction = correction - source_basis @ correction_coefficients
    value_coordinates = coarse_value + augmented_correction
    updated, extraction_matvecs = _extract_state(
        state,
        prepared,
        projected_action,
        projected_rhs,
        restart,
        extraction_,
    )
    return RecycledLinearSolveResult(
        _build_result(
            prepared,
            rhs_coordinates,
            value_coordinates,
            auxiliary,
            max_steps,
            restart,
            refresh_matvecs + extraction_matvecs,
        ),
        _stop_state(updated),
    )


def _build_result(
    prepared: PreparedLinearSolve,
    rhs: Array,
    initial: Array,
    auxiliary,
    max_steps: int,
    restart: int,
    /,
    setup_matvecs: Array,
) -> LinearSolveResult:
    from ._runtime import _implicit_root_value

    problem = prepared.problem
    if not isinstance(problem, LinearSystem):
        raise TypeError("GCRO-DR result construction requires a LinearSystem.")
    mode = prepared.plan.policy.differentiation.mode
    if mode in ("mathematical", "rhs-only"):
        execution_problem = (
            jax.tree.map(
                lambda value: (
                    jax.lax.stop_gradient(value) if eqx.is_array(value) else value
                ),
                problem,
            )
            if mode == "rhs-only"
            else problem
        )
        value = _implicit_root_value(
            prepared,
            execution_problem,
            rhs[:, None],
            initial[:, None],
        )[:, 0]
    elif mode == "none":
        execution_problem = problem
        value = jax.lax.stop_gradient(initial)
    else:
        raise ValueError("Algorithmic differentiation through GCRO-DR is unsupported.")

    operator = execution_problem.operator
    residual = (
        operator.target.flatten(operator.mv(operator.source.unflatten(value))) - rhs
    )
    residual_norm = _coordinate_norm(operator.target, residual)
    rhs_norm = _coordinate_norm(operator.target, rhs)
    relative_residual = jnp.where(
        rhs_norm > 0.0,
        residual_norm / rhs_norm,
        residual_norm,
    )
    roundoff_relative = (
        10.0
        * jnp.finfo(rhs.real.dtype).eps
        * float(max(operator.source.size, operator.target.size))
    )
    relative_tolerance = jnp.maximum(
        prepared.plan.policy.tolerance.relative,
        roundoff_relative,
    )
    threshold = prepared.plan.policy.tolerance.absolute + relative_tolerance * rhs_norm
    converged = residual_norm <= threshold
    status = jnp.where(
        converged,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
    ).astype(jnp.int32)
    breakdown = auxiliary[4]
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.NONFINITE_ACTION),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.STAGNATION),
        int(LinearSolveStatus.STAGNATION),
        status,
    )
    status = jnp.where(
        (breakdown != int(KrylovBreakdownStatus.NONE))
        & (breakdown != int(KrylovBreakdownStatus.HAPPY))
        & (breakdown != int(KrylovBreakdownStatus.NONFINITE_ACTION))
        & (breakdown != int(KrylovBreakdownStatus.STAGNATION)),
        int(LinearSolveStatus.BREAKDOWN),
        status,
    )
    rhs_finite = jnp.all(jnp.isfinite(rhs))
    value_finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual_norm)
    finite = rhs_finite & value_finite
    status = jnp.where(
        ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        rhs_finite & ~value_finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    ).astype(jnp.int32)
    cycles = (max_steps + restart - 1) // restart
    matvec_count = (
        2 * auxiliary[0] + jnp.asarray(cycles + 3, dtype=jnp.int32) + setup_matvecs
    )
    properties = operator.properties
    rank = (
        properties.rank
        if properties.certifies("rank") and properties.rank is not None
        else -1
    )
    diagnostics = LinearSolveDiagnostics(
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        normal_residual_norm=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        iterations=auxiliary[0],
        rank=rank,
        condition_estimate=auxiliary[3],
        finite=finite,
        converged=status == int(LinearSolveStatus.SUCCESS),
        compatibility_residual=0.0,
        gauge_residual=0.0,
        nullity=-1,
        matvec_count=matvec_count,
        adjoint_matvec_count=0,
    )
    provenance = LinearSolveProvenance(
        backend=prepared.plan.backend,
        method=f"gcro-dr-{prepared.plan.method}",
        plan_id=prepared.plan.plan_id,
        problem_id=problem.problem_id,
        reason=f"{prepared.plan.reason}; GCRO-DR augmented projected solve",
        rejected=prepared.plan.rejected,
        prepared=True,
        rhs_mode="single",
        operator_numeric_version=prepared.numeric_version,
        recycling_capacity=prepared.plan.recycling_capacity,
        recycling_state_bytes=prepared.plan.recycling_state_bytes,
    )
    if prepared.plan.policy.failure.mode == "error":
        value = eqx.error_if(
            value,
            status != int(LinearSolveStatus.SUCCESS),
            "Linear solve failed.",
        )
    return LinearSolveResult(
        operator.source.unflatten(value),
        status,
        diagnostics,
        provenance,
    )


def _validate_prepared(prepared: PreparedLinearSolve, /) -> None:
    if not isinstance(prepared, PreparedLinearSolve):
        raise TypeError("prepared must be a PreparedLinearSolve.")
    if not isinstance(prepared.problem, LinearSystem):
        raise ValueError("GCRO-DR is available only for square LinearSystem problems.")
    operator = prepared.problem.operator
    if operator.batch_shape:
        raise ValueError("GCRO-DR recycling space is unavailable for batched operators.")
    if not operator.source.compatible(operator.target):
        raise ValueError(
            "GCRO-DR requires identical source and target spaces, including pairing."
        )
    if prepared.plan.method not in ("gmres", "fgmres"):
        raise ValueError("GCRO-DR requires a GMRES-like solve plan.")
    if prepared.problem.nullspace_policy is not None:
        raise ValueError("GCRO-DR with a declared nullspace is not yet supported.")
    if prepared.plan.backend != "native-krylov":
        raise ValueError("GCRO-DR currently requires the native-krylov backend.")
    if prepared.plan.policy.preconditioning is not None:
        raise ValueError("GCRO-DR with a preconditioner is not yet supported.")
    if prepared.plan.policy.differentiation.mode == "algorithmic":
        raise ValueError("Algorithmic differentiation through GCRO-DR is unsupported.")


def _validate_state(
    state: RecyclingState,
    prepared: PreparedLinearSolve,
    capacity: int,
    extraction: RecyclingExtraction,
    /,
) -> None:
    operator = prepared.problem.operator
    expected_id = _recycling_id(prepared, capacity, extraction)
    if state.capacity != capacity:
        raise ValueError(
            f"Recycling capacity mismatch: state has {state.capacity}, plan requires {capacity}."
        )
    if not state.source.compatible(operator.source) or not state.target.compatible(
        operator.target
    ):
        raise ValueError("Recycling state vector spaces are structurally incompatible.")
    if jnp.dtype(state.source_basis.dtype) != _coordinate_dtype(operator.source):
        raise ValueError("Recycling source basis dtype does not match its source space.")
    if jnp.dtype(state.image_basis.dtype) != _coordinate_dtype(operator.target):
        raise ValueError("Recycling image basis dtype does not match its target space.")
    if state.operator_id != operator.operator_id:
        raise ValueError("Stale recycling operator ID; structural reuse is rejected.")
    if state.solve_plan_id != prepared.plan.plan_id:
        raise ValueError("Stale recycling solve-plan ID; structural reuse is rejected.")
    if state.recycling_id != expected_id:
        raise ValueError("Stale recycling ID; the state does not belong to this plan.")
    if state.extraction != extraction:
        raise ValueError("Recycling extraction policy does not match the state.")


def _coerce_state(
    recycling: RecyclingState | RecyclingSubspace,
    prepared: PreparedLinearSolve,
    capacity: int | Array,
    extraction: RecyclingExtraction,
    /,
) -> RecyclingState:
    if isinstance(recycling, RecyclingState):
        return recycling
    if not isinstance(recycling, RecyclingSubspace):
        raise TypeError("recycling must be RecyclingState, RecyclingSubspace, or None.")
    capacity_ = int(capacity)
    operator = prepared.problem.operator
    if recycling.operator_id != operator.operator_id:
        raise ValueError("Stale recycling operator ID; structural reuse is rejected.")
    if not recycling.source.compatible(
        operator.source
    ) or not recycling.target.compatible(operator.target):
        raise ValueError(
            "Recycling subspace vector spaces are structurally incompatible."
        )
    if jnp.dtype(recycling.source_basis.dtype) != _coordinate_dtype(operator.source):
        raise ValueError("Recycling source basis dtype does not match its source space.")
    if jnp.dtype(recycling.image_basis.dtype) != _coordinate_dtype(operator.target):
        raise ValueError("Recycling image basis dtype does not match its target space.")
    source = jnp.zeros(
        (operator.source.size, capacity_), dtype=recycling.source_basis.dtype
    )
    image = jnp.zeros(
        (operator.target.size, capacity_), dtype=recycling.image_basis.dtype
    )
    copied = min(recycling.dimension, capacity_)
    source = source.at[:, :copied].set(recycling.source_basis[:, :copied])
    image = image.at[:, :copied].set(recycling.image_basis[:, :copied])
    status = (
        RecyclingUpdateStatus.CAPACITY_TRUNCATED
        if recycling.dimension > capacity_
        else RecyclingUpdateStatus.CURRENT
    )
    state = RecyclingState(
        source=operator.source,
        target=operator.target,
        source_basis=source,
        image_basis=image,
        effective_dimension=copied,
        operator_id=operator.operator_id,
        solve_plan_id=prepared.plan.plan_id,
        operator_numeric_version=prepared.numeric_version,
        recycling_id=_recycling_id(prepared, capacity_, extraction),
        update_status=status,
        extraction=extraction,
    )
    return state


def _empty_state(
    prepared: PreparedLinearSolve,
    capacity: int,
    extraction: RecyclingExtraction,
    /,
    *,
    recycling_id: str | None = None,
    update_count: Any = 0,
) -> RecyclingState:
    operator = prepared.problem.operator
    dtype = _coordinate_dtype(operator.source)
    return RecyclingState(
        source=operator.source,
        target=operator.target,
        source_basis=jnp.zeros((operator.source.size, capacity), dtype=dtype),
        image_basis=jnp.zeros((operator.target.size, capacity), dtype=dtype),
        effective_dimension=0,
        operator_id=operator.operator_id,
        solve_plan_id=prepared.plan.plan_id,
        operator_numeric_version=prepared.numeric_version,
        recycling_id=(
            _recycling_id(prepared, capacity, extraction)
            if recycling_id is None
            else recycling_id
        ),
        update_count=update_count,
        update_status=RecyclingUpdateStatus.EMPTY,
        extraction=extraction,
    )


def _refresh_source_images(
    state: RecyclingState,
    prepared: PreparedLinearSolve,
    /,
    *,
    increment: bool = True,
    preserve_status: bool = False,
) -> RecyclingState:
    operator = prepared.problem.operator
    images = _apply_basis(
        lambda vector: jax.lax.stop_gradient(
            _operator_coordinates(prepared, jax.lax.stop_gradient(vector))
        ),
        state.source_basis,
        state.effective_dimension,
        operator.target.size,
    )
    active = jnp.arange(state.capacity) < state.effective_dimension
    source, image, rank, status = _compress_candidates(
        operator.target,
        state.source_basis,
        images,
        active,
        state.capacity,
    )
    refreshed_status = jnp.where(
        status == int(RecyclingUpdateStatus.RANK_LOSS),
        status,
        jnp.where(
            status == int(RecyclingUpdateStatus.EMPTY),
            status,
            int(RecyclingUpdateStatus.REFRESHED),
        ),
    )
    if preserve_status:
        refreshed_status = _carry_update_status(
            state.update_status,
            refreshed_status,
        )
    return RecyclingState(
        source=operator.source,
        target=operator.target,
        source_basis=source,
        image_basis=image,
        effective_dimension=rank,
        operator_id=operator.operator_id,
        solve_plan_id=prepared.plan.plan_id,
        operator_numeric_version=prepared.numeric_version,
        recycling_id=state.recycling_id,
        update_count=state.update_count + int(increment),
        update_status=refreshed_status,
        extraction=state.extraction,
    )


def _extract_state(
    state: RecyclingState,
    prepared: PreparedLinearSolve,
    projected_action,
    initial: Array,
    restart: int,
    extraction: RecyclingExtraction,
    /,
) -> tuple[RecyclingState, Array]:
    operator = prepared.problem.operator
    target = operator.target
    algorithmic_action = lambda vector: jax.lax.stop_gradient(
        projected_action(jax.lax.stop_gradient(vector))
    )
    decomposition = arnoldi(
        algorithmic_action,
        jax.lax.stop_gradient(initial),
        max_dimension=restart,
        inner=lambda left, right: _algorithmic_inner(target, left, right),
        orthogonalization="double",
    )
    krylov_sources = decomposition.basis[:-1].T
    krylov_images = _apply_basis(
        lambda vector: jax.lax.stop_gradient(
            _operator_coordinates(prepared, jax.lax.stop_gradient(vector))
        ),
        krylov_sources,
        decomposition.effective_dimension,
        operator.target.size,
    )
    retained_active = jnp.arange(state.capacity) < state.effective_dimension
    krylov_active = jnp.arange(restart) < decomposition.effective_dimension
    search_sources = jnp.concatenate((state.source_basis, krylov_sources), axis=1)
    search_images = jnp.concatenate((state.image_basis, krylov_images), axis=1)
    search_active = jnp.concatenate((retained_active, krylov_active))
    candidates, candidate_images, candidate_active = _augmented_harmonic_ritz_sources(
        target,
        search_sources,
        search_images,
        search_active,
        state.capacity,
    )
    source, image, rank, status = _compress_candidates(
        target,
        candidates,
        candidate_images,
        candidate_active,
        state.capacity,
    )
    status = jnp.where(
        status == int(RecyclingUpdateStatus.RANK_LOSS),
        status,
        jnp.where(
            jnp.sum(search_active, dtype=jnp.int32) > state.capacity,
            int(RecyclingUpdateStatus.CAPACITY_TRUNCATED),
            status,
        ),
    ).astype(jnp.int32)
    status = _carry_update_status(state.update_status, status)
    updated = RecyclingState(
        source=operator.source,
        target=operator.target,
        source_basis=source,
        image_basis=image,
        effective_dimension=rank,
        operator_id=operator.operator_id,
        solve_plan_id=prepared.plan.plan_id,
        operator_numeric_version=prepared.numeric_version,
        recycling_id=state.recycling_id,
        update_count=state.update_count + 1,
        update_status=status,
        extraction=extraction,
    )
    setup_matvecs = decomposition.matvec_count + decomposition.effective_dimension
    return updated, setup_matvecs


def _augmented_harmonic_ritz_sources(
    space,
    search_sources: Array,
    search_images: Array,
    search_active: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array]:
    dimension = int(search_sources.shape[1])
    active = search_active[:, None] & search_active[None, :]
    image_gram = _coordinate_cross_gram(space, search_images, search_images)
    coupling = _coordinate_cross_gram(space, search_images, search_sources)
    image_gram = jnp.where(active, image_gram, 0)
    coupling = jnp.where(active, coupling, 0)
    safe_coupling = coupling + jnp.diag((~search_active).astype(coupling.dtype))
    harmonic = jnp.linalg.pinv(safe_coupling) @ image_gram
    scale = jnp.maximum(jnp.linalg.norm(harmonic), 1.0)
    inactive_value = scale / jnp.sqrt(jnp.finfo(harmonic.real.dtype).eps)
    harmonic = harmonic + jnp.diag(
        (~search_active).astype(harmonic.dtype) * inactive_value
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(harmonic)
    order = jnp.argsort(jnp.abs(eigenvalues))
    selected_count = min(capacity, dimension)
    coefficients = eigenvectors[:, order[:selected_count]]
    selected_values = eigenvalues[order[:selected_count]]
    selected_active = jnp.arange(selected_count) < jnp.minimum(
        jnp.sum(search_active, dtype=jnp.int32),
        selected_count,
    )
    if jnp.issubdtype(search_sources.dtype, jnp.complexfloating):
        sources = search_sources @ coefficients.astype(search_sources.dtype)
        images = search_images @ coefficients.astype(search_images.dtype)
        active_columns = selected_active
        output_sources = jnp.zeros(
            (search_sources.shape[0], capacity),
            dtype=search_sources.dtype,
        )
        output_images = jnp.zeros(
            (search_images.shape[0], capacity),
            dtype=search_images.dtype,
        )
        output_sources = output_sources.at[:, :selected_count].set(sources)
        output_images = output_images.at[:, :selected_count].set(images)
        output_active = jnp.zeros((capacity,), dtype=bool)
        output_active = output_active.at[:selected_count].set(active_columns)
        return output_sources, output_images, output_active

    real_coefficients = jnp.real(coefficients).astype(search_sources.dtype)
    imaginary_coefficients = jnp.imag(coefficients).astype(search_sources.dtype)
    coefficients_pool = jnp.stack(
        (real_coefficients, imaginary_coefficients),
        axis=2,
    ).reshape((dimension, 2 * selected_count))
    component_norms = jnp.linalg.norm(coefficients_pool, axis=0)
    tolerance = jnp.sqrt(jnp.finfo(search_sources.dtype).eps)

    def select_real_block(index, carry):
        used, active_pool_ = carry
        selected_value = selected_values[index]
        imaginary_part = jnp.imag(selected_value)
        eigenvalue_tolerance = jnp.finfo(search_sources.dtype).eps * jnp.maximum(
            jnp.abs(selected_value),
            jnp.finfo(search_sources.dtype).tiny,
        )
        conjugate_pair = imaginary_part > eigenvalue_tolerance
        eligible = selected_active[index] & (imaginary_part >= -eigenvalue_tolerance)
        width = jnp.where(conjugate_pair, 2, 1).astype(jnp.int32)
        real_valid = component_norms[2 * index] > tolerance
        imaginary_valid = component_norms[2 * index + 1] > tolerance
        accepted = (
            eligible
            & (used + width <= capacity)
            & real_valid
            & (~conjugate_pair | imaginary_valid)
        )
        active_pool_ = active_pool_.at[2 * index].set(accepted)
        active_pool_ = active_pool_.at[2 * index + 1].set(accepted & conjugate_pair)
        used = used + jnp.where(accepted, width, 0)
        return used, active_pool_

    _, active_pool = jax.lax.fori_loop(
        0,
        selected_count,
        select_real_block,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((2 * selected_count,), dtype=bool),
        ),
    )
    sources = search_sources @ coefficients_pool
    images = search_images @ coefficients_pool
    output_sources = jnp.zeros(
        (search_sources.shape[0], 2 * capacity),
        dtype=search_sources.dtype,
    )
    output_images = jnp.zeros(
        (search_images.shape[0], 2 * capacity),
        dtype=search_images.dtype,
    )
    output_sources = output_sources.at[:, : 2 * selected_count].set(sources)
    output_images = output_images.at[:, : 2 * selected_count].set(images)
    output_active = jnp.zeros((2 * capacity,), dtype=bool)
    output_active = output_active.at[: 2 * selected_count].set(active_pool)
    return output_sources, output_images, output_active


def _compress_candidates(
    space,
    source_candidates: Array,
    image_candidates: Array,
    candidate_active: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    sources = jnp.zeros(
        (source_candidates.shape[0], capacity), dtype=source_candidates.dtype
    )
    images = jnp.zeros(
        (image_candidates.shape[0], capacity), dtype=image_candidates.dtype
    )
    rank = jnp.asarray(0, dtype=jnp.int32)
    active_count = jnp.sum(candidate_active, dtype=jnp.int32)
    real_dtype = image_candidates.real.dtype
    tolerance = jnp.sqrt(jnp.finfo(real_dtype).eps)

    def step(index, carry):
        sources_, images_, rank_ = carry
        source = source_candidates[:, index]
        image = image_candidates[:, index]
        original_norm = _algorithmic_norm(space, image)
        slots = jnp.arange(capacity) < rank_

        def orthogonalize(values):
            source_, image_ = values
            coefficients = jax.vmap(
                lambda column: _algorithmic_inner(space, column, image_), in_axes=1
            )(images_)
            coefficients = jnp.where(slots, coefficients, 0)
            return source_ - sources_ @ coefficients, image_ - images_ @ coefficients

        source, image = orthogonalize((source, image))
        source, image = orthogonalize((source, image))
        norm = _algorithmic_norm(space, image)
        independent = norm > tolerance * jnp.maximum(original_norm, 1.0)
        finite = (
            jnp.isfinite(norm)
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(image))
        )
        accepted = candidate_active[index] & independent & finite & (rank_ < capacity)
        safe_norm = jnp.where(accepted, norm, 1.0)
        slot = jnp.minimum(rank_, capacity - 1)
        sources_ = jax.lax.cond(
            accepted,
            lambda value: value.at[:, slot].set(source / safe_norm),
            lambda value: value,
            sources_,
        )
        images_ = jax.lax.cond(
            accepted,
            lambda value: value.at[:, slot].set(image / safe_norm),
            lambda value: value,
            images_,
        )
        return sources_, images_, rank_ + accepted.astype(jnp.int32)

    sources, images, rank = jax.lax.fori_loop(
        0,
        source_candidates.shape[1],
        step,
        (sources, images, rank),
    )
    status = jnp.where(
        rank < jnp.minimum(active_count, capacity),
        int(RecyclingUpdateStatus.RANK_LOSS),
        jnp.where(
            active_count > capacity,
            int(RecyclingUpdateStatus.CAPACITY_TRUNCATED),
            jnp.where(
                rank == 0,
                int(RecyclingUpdateStatus.EMPTY),
                int(RecyclingUpdateStatus.CURRENT),
            ),
        ),
    ).astype(jnp.int32)
    return sources, images, rank, status


def _carry_update_status(previous: Array, current: Array, /) -> Array:
    rank_loss = (previous == int(RecyclingUpdateStatus.RANK_LOSS)) | (
        current == int(RecyclingUpdateStatus.RANK_LOSS)
    )
    truncated = (previous == int(RecyclingUpdateStatus.CAPACITY_TRUNCATED)) | (
        current == int(RecyclingUpdateStatus.CAPACITY_TRUNCATED)
    )
    return jnp.where(
        rank_loss,
        int(RecyclingUpdateStatus.RANK_LOSS),
        jnp.where(
            truncated,
            int(RecyclingUpdateStatus.CAPACITY_TRUNCATED),
            current,
        ),
    ).astype(jnp.int32)


def _coordinate_cross_gram(space, left: Array, right: Array, /) -> Array:
    return jax.vmap(
        lambda left_column: jax.vmap(
            lambda right_column: _algorithmic_inner(
                space,
                left_column,
                right_column,
            ),
            in_axes=1,
        )(right),
        in_axes=1,
    )(left)


def _apply_basis(
    action,
    basis: Array,
    effective_dimension: Array,
    output_size: int,
    /,
) -> Array:
    output = jax.eval_shape(action, basis[:, 0])
    if not isinstance(output, jax.ShapeDtypeStruct) or output.shape != (output_size,):
        raise ValueError("Recycling basis action returned an incompatible vector.")
    images = jnp.zeros(
        (output_size, basis.shape[1]),
        dtype=output.dtype,
    )

    def step(index, value):
        return jax.lax.cond(
            index < effective_dimension,
            lambda output: output.at[:, index].set(action(basis[:, index])),
            lambda output: output,
            value,
        )

    return jax.lax.fori_loop(0, basis.shape[1], step, images)


def _basis_coefficients(
    space,
    basis: Array,
    vector: Array,
    effective_dimension: Array,
    /,
) -> Array:
    coefficients = jax.vmap(
        lambda column: _coordinate_inner(space, column, vector), in_axes=1
    )(basis)
    return jnp.where(jnp.arange(basis.shape[1]) < effective_dimension, coefficients, 0)


def _project_coordinates(
    space,
    basis: Array,
    effective_dimension: Array,
    vector: Array,
    /,
) -> Array:
    return vector - basis @ _basis_coefficients(space, basis, vector, effective_dimension)


def _operator_coordinates(prepared: PreparedLinearSolve, vector: Array, /) -> Array:
    operator = prepared.problem.operator
    return operator.target.flatten(operator.mv(operator.source.unflatten(vector)))


def _coordinate_inner(space, left: Array, right: Array, /) -> Array:
    return space.inner(space.unflatten(left), space.unflatten(right))


def _coordinate_norm(space, vector: Array, /) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.real(_coordinate_inner(space, vector, vector)), 0.0))


def _algorithmic_inner(space, left: Array, right: Array, /) -> Array:
    return jax.lax.stop_gradient(_coordinate_inner(space, left, right))


def _algorithmic_norm(space, vector: Array, /) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.real(_algorithmic_inner(space, vector, vector)), 0.0))


def _gmres_configuration(
    prepared: PreparedLinearSolve, max_steps: int, /
) -> tuple[int, int]:
    method = prepared.plan.policy.method
    if isinstance(method, (GMRES, FGMRES)):
        restart = method.restart
        stagnation = method.stagnation_iterations
    else:
        restart = 30
        stagnation = 8
    return min(restart, max_steps, prepared.problem.operator.source.size), stagnation


def _recycling_id(
    prepared: PreparedLinearSolve, capacity: int, extraction: RecyclingExtraction, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "gcro-dr",
            "operator": prepared.problem.operator.operator_id,
            "plan": prepared.plan.plan_id,
            "source": prepared.problem.operator.source.space_id,
            "target": prepared.problem.operator.target.space_id,
            "capacity": int(capacity),
            "extraction": extraction,
        }
    )


def _validate_extraction(value: str, /) -> RecyclingExtraction:
    if value != "harmonic-ritz":
        raise ValueError("Only extraction='harmonic-ritz' is supported.")
    return value


def _validate_refresh(value: str, /) -> RecyclingRefresh:
    if value not in ("reuse-source", "rebuild"):
        raise ValueError("refresh must be 'reuse-source' or 'rebuild'.")
    return value


def _stop_state(state: RecyclingState, /) -> RecyclingState:
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        state,
    )


__all__ = ["refresh_recycling", "solve_recycled"]
