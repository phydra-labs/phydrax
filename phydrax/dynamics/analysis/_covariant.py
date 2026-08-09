#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._evolution import AbstractDifferentiableEvolution
from .._grid import EvolutionGrid, IterationGrid, TimeGrid


CovariantDirectionKind: TypeAlias = Literal["clv", "adjoint"]
CovariantMemoryMode: TypeAlias = Literal["store", "recompute"]

COVARIANT_SUCCESS = 0
COVARIANT_EVOLUTION_FAILED = 1
COVARIANT_SINGULAR_FACTOR = 2
COVARIANT_NONFINITE = 3
COVARIANT_INSUFFICIENT_BACKWARD_DEPTH = 4


class CovariantDirectionResult(StrictModule):
    """Ginelli covariant directions with explicit backward and memory evidence."""

    coordinates: Array
    states: Array
    directions: Array
    direction_valid: Array
    local_growth_rates: Array
    covariance_error: Array
    backward_convergence_drift: Array
    saved_checkpoint_index: Array
    valid: Array
    converged: Array
    status: Array
    evolution: AbstractDifferentiableEvolution
    grid: EvolutionGrid
    kind: CovariantDirectionKind = eqx.field(static=True)
    memory_mode: CovariantMemoryMode = eqx.field(static=True)
    leading_k: int = eqx.field(static=True)
    full_basis: bool = eqx.field(static=True)
    qr_interval: int = eqx.field(static=True)
    save_every: int = eqx.field(static=True)
    backward_discard: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    stored_frame_count: int = eqx.field(static=True)
    tangent_evaluations: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    tangent_method_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)


class _ForwardInterval(StrictModule):
    state: Array
    basis: Array
    triangular: Array
    valid: Array


def _thin_qr(matrix: Array, /) -> tuple[Array, Array]:
    basis, triangular = jnp.linalg.qr(matrix, mode="reduced")
    diagonal = jnp.diag(triangular)
    signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
    return basis * signs[None, :], signs[:, None] * triangular


def _initial_basis(
    state: Array,
    rank: int,
    supplied: ArrayLike | None,
    seed: int,
    /,
) -> Array:
    dimension = int(state.size)
    if supplied is None:
        matrix = jax.random.normal(
            jax.random.PRNGKey(int(seed)),
            (dimension, rank),
            dtype=jnp.result_type(state, float),
        )
    else:
        matrix = jnp.asarray(supplied)
        if matrix.shape == tuple(state.shape) + (rank,):
            matrix = matrix.reshape((dimension, rank))
        if matrix.shape != (dimension, rank):
            raise ValueError("initial_basis has an incompatible shape.")
    basis, triangular = _thin_qr(matrix)
    if bool(jnp.any(jnp.abs(jnp.diag(triangular)) == 0.0)):
        raise ValueError("initial_basis must have full column rank.")
    return basis


def _advance_interval(
    evolution: AbstractDifferentiableEvolution,
    state: Array,
    basis: Array,
    coordinates: Array,
    start: int,
    stop: int,
    args: Any,
    /,
) -> _ForwardInterval:
    state_shape = evolution.state_layout.shape
    run_valid = jnp.asarray(True)
    for step_index in range(start, stop):
        source = coordinates[step_index]
        target = coordinates[step_index + 1]

        def propagate(vector):
            tangent_step = evolution.tangent_action(
                state,
                vector.reshape(state_shape),
                source,
                target,
                args,
            )
            return (
                tangent_step.primal.final_state,
                tangent_step.tangent.reshape((-1,)),
                tangent_step.valid,
            )

        states, propagated, step_valid = jax.vmap(
            propagate, in_axes=1, out_axes=(0, 1, 0)
        )(basis)
        state = states[0]
        basis = propagated
        run_valid = (
            run_valid
            & jnp.all(step_valid)
            & jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(basis))
        )
    orthonormal, triangular = _thin_qr(basis)
    diagonal = jnp.abs(jnp.diag(triangular))
    valid = (
        run_valid
        & jnp.all(jnp.isfinite(orthonormal))
        & jnp.all(jnp.isfinite(triangular))
        & jnp.all(diagonal > 0.0)
    )
    return _ForwardInterval(
        state=state,
        basis=orthonormal,
        triangular=triangular,
        valid=valid,
    )


def covariant_directions(
    evolution: AbstractDifferentiableEvolution,
    initial_state: ArrayLike,
    grid: EvolutionGrid,
    /,
    *,
    args: Any = None,
    leading_k: int | None = None,
    initial_basis: ArrayLike | None = None,
    seed: int = 0,
    kind: CovariantDirectionKind = "clv",
    memory_mode: CovariantMemoryMode = "store",
    qr_interval: int = 1,
    save_every: int = 1,
    backward_discard: int = 1,
    convergence_tolerance: float = 1e-6,
) -> CovariantDirectionResult:
    """Compute CLVs or full dual adjoint directions by backward triangular solves.

    ``store`` retains every forward QR frame and is linear-time with O(checkpoints ×
    state × rank) working storage. ``recompute`` retains no QR history and rebuilds
    prefixes during the backward sweep, using O(state × rank) working storage and
    quadratic forward work. Returned saved directions consume the same output storage.
    """
    if not isinstance(evolution, AbstractDifferentiableEvolution):
        raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
    if not isinstance(grid, (TimeGrid, IterationGrid)):
        raise TypeError("grid must be a TimeGrid or IterationGrid.")
    if kind not in ("clv", "adjoint"):
        raise ValueError("kind must be 'clv' or 'adjoint'.")
    if memory_mode not in ("store", "recompute"):
        raise ValueError("memory_mode must be 'store' or 'recompute'.")
    cadence = int(qr_interval)
    saving = int(save_every)
    discard = int(backward_discard)
    if cadence < 1 or saving < 1 or discard < 0:
        raise ValueError("QR/save intervals must be positive and discard nonnegative.")
    state = jnp.asarray(initial_state)
    if state.shape != evolution.state_layout.shape:
        raise ValueError("initial_state has the wrong state shape.")
    if not jnp.issubdtype(state.dtype, jnp.floating):
        raise TypeError("initial_state must have a real floating dtype.")
    dimension = evolution.state_layout.size
    rank = dimension if leading_k is None else int(leading_k)
    if rank < 1 or rank > dimension:
        raise ValueError("leading_k must lie in [1, state dimension].")
    if kind == "adjoint" and rank != dimension:
        raise ValueError("Adjoint covariant directions require a full basis.")
    basis0 = _initial_basis(state, rank, initial_basis, int(seed))
    boundaries = [0]
    while boundaries[-1] < grid.num_steps:
        boundaries.append(min(grid.num_steps, boundaries[-1] + cadence))
    intervals = len(boundaries) - 1
    saved = list(range(0, intervals + 1, saving))
    if saved[-1] != intervals:
        saved.append(intervals)
    saved_set = set(saved)
    coordinates = grid.coordinates
    frame_history: list[Array] = []
    state_history: list[Array] = []
    triangular_history: list[Array] = []
    forward_valid = True
    tangent_evaluations = 0
    if memory_mode == "store":
        frame_history.append(basis0)
        state_history.append(state)
        current_state = state
        current_basis = basis0
        for interval in range(intervals):
            advanced = _advance_interval(
                evolution,
                current_state,
                current_basis,
                coordinates,
                boundaries[interval],
                boundaries[interval + 1],
                args,
            )
            tangent_evaluations += (
                boundaries[interval + 1] - boundaries[interval]
            ) * rank
            current_state = advanced.state
            current_basis = advanced.basis
            frame_history.append(current_basis)
            state_history.append(current_state)
            triangular_history.append(advanced.triangular)
            forward_valid = forward_valid and bool(advanced.valid)

    def prefix(count: int) -> tuple[Array, Array, bool, int]:
        current_state = state
        current_basis = basis0
        prefix_valid = True
        evaluations = 0
        for interval in range(count):
            advanced = _advance_interval(
                evolution,
                current_state,
                current_basis,
                coordinates,
                boundaries[interval],
                boundaries[interval + 1],
                args,
            )
            current_state = advanced.state
            current_basis = advanced.basis
            prefix_valid = prefix_valid and bool(advanced.valid)
            evaluations += (boundaries[interval + 1] - boundaries[interval]) * rank
        return current_state, current_basis, prefix_valid, evaluations

    if memory_mode == "store":
        final_basis = frame_history[-1]
    else:
        _, final_basis, final_valid, evaluations = prefix(intervals)
        forward_valid = forward_valid and final_valid
        tangent_evaluations += evaluations
    coefficients = jnp.eye(rank, dtype=basis0.dtype)
    probe_coefficients = jnp.eye(rank, dtype=basis0.dtype) + 0.25 * jnp.triu(
        jax.random.normal(
            jax.random.PRNGKey(int(seed) + 1),
            (rank, rank),
            dtype=basis0.dtype,
        ),
        k=1,
    )
    probe_coefficients = (
        probe_coefficients / jnp.linalg.norm(probe_coefficients, axis=0)[None, :]
    )
    saved_directions: dict[int, Array] = {intervals: final_basis @ coefficients}
    saved_probe_directions: dict[int, Array] = {
        intervals: final_basis @ probe_coefficients
    }
    saved_states: dict[int, Array] = {}
    if memory_mode == "store":
        if intervals in saved_set:
            saved_states[intervals] = state_history[-1]
    else:
        final_state, _, _, evaluations = prefix(intervals)
        tangent_evaluations += evaluations
        if intervals in saved_set:
            saved_states[intervals] = final_state
    local_rates: dict[int, Array] = {}
    covariance_errors: dict[int, Array] = {}
    backward_valid = True
    for interval in range(intervals - 1, -1, -1):
        if memory_mode == "store":
            basis_before = frame_history[interval]
            basis_after = frame_history[interval + 1]
            triangular = triangular_history[interval]
            state_before = state_history[interval]
        else:
            state_before, basis_before, prefix_valid, evaluations = prefix(interval)
            tangent_evaluations += evaluations
            advanced = _advance_interval(
                evolution,
                state_before,
                basis_before,
                coordinates,
                boundaries[interval],
                boundaries[interval + 1],
                args,
            )
            tangent_evaluations += (
                boundaries[interval + 1] - boundaries[interval]
            ) * rank
            basis_after = advanced.basis
            triangular = advanced.triangular
            forward_valid = forward_valid and prefix_valid and bool(advanced.valid)
        raw = jnp.linalg.solve(triangular, coefficients)
        probe_raw = jnp.linalg.solve(triangular, probe_coefficients)
        column_norm = jnp.linalg.norm(raw, axis=0)
        probe_column_norm = jnp.linalg.norm(probe_raw, axis=0)
        solve_valid = (
            jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(column_norm))
            & jnp.all(column_norm > 0.0)
            & jnp.all(jnp.isfinite(probe_raw))
            & jnp.all(jnp.isfinite(probe_column_norm))
            & jnp.all(probe_column_norm > 0.0)
        )
        backward_valid = backward_valid and bool(solve_valid)
        next_coefficients = raw / column_norm[None, :]
        next_probe_coefficients = probe_raw / probe_column_norm[None, :]
        directions_before = basis_before @ next_coefficients
        probe_directions_before = basis_before @ next_probe_coefficients
        propagated = basis_after @ (triangular @ next_coefficients)
        propagated = propagated / jnp.linalg.norm(propagated, axis=0)[None, :]
        target = basis_after @ coefficients
        target = target / jnp.linalg.norm(target, axis=0)[None, :]
        covariance_errors[interval] = jnp.linalg.norm(
            jnp.abs(propagated) - jnp.abs(target), axis=0
        )
        duration = (
            coordinates[boundaries[interval + 1]] - coordinates[boundaries[interval]]
        )
        local_rates[interval] = -jnp.log(column_norm) / duration
        coefficients = next_coefficients
        probe_coefficients = next_probe_coefficients
        if interval in saved_set:
            saved_directions[interval] = directions_before
            saved_probe_directions[interval] = probe_directions_before
            saved_states[interval] = state_before
    direction_values = []
    state_values = []
    growth_values = []
    error_values = []
    drift_values = []
    validity_values = []
    for checkpoint in saved:
        values = saved_directions[checkpoint]
        probe_values = saved_probe_directions[checkpoint]
        if kind == "adjoint":
            values = jnp.linalg.inv(values).T
            values = values / jnp.linalg.norm(values, axis=0)[None, :]
            probe_values = jnp.linalg.inv(probe_values).T
            probe_values = probe_values / jnp.linalg.norm(probe_values, axis=0)[None, :]
        direction_values.append(values.reshape(evolution.state_layout.shape + (rank,)))
        drift_values.append(
            jnp.linalg.norm(
                jnp.abs(values.reshape((dimension, rank)))
                - jnp.abs(probe_values.reshape((dimension, rank))),
                axis=0,
            )
        )
        state_values.append(saved_states[checkpoint])
        if checkpoint < intervals:
            growth_values.append(local_rates[checkpoint])
            error_values.append(covariance_errors[checkpoint])
        else:
            growth_values.append(jnp.full((rank,), jnp.nan))
            error_values.append(jnp.full((rank,), jnp.nan))
        validity_values.append(
            forward_valid
            and backward_valid
            and checkpoint <= intervals - discard
            and bool(jnp.all(jnp.isfinite(values)))
        )
    directions = jnp.stack(tuple(direction_values), axis=0)
    saved_states_array = jnp.stack(tuple(state_values), axis=0)
    growth = jnp.stack(tuple(growth_values), axis=0)
    errors = jnp.stack(tuple(error_values), axis=0)
    convergence_drift = jnp.stack(tuple(drift_values), axis=0)
    direction_valid = jnp.asarray(validity_values, dtype=bool)
    finite_errors = jnp.where(direction_valid[:, None], errors, jnp.nan)
    finite_drift = jnp.where(direction_valid[:, None], convergence_drift, jnp.nan)
    maximum_error = jnp.nanmax(finite_errors)
    maximum_drift = jnp.nanmax(finite_drift)
    enough_depth = bool(jnp.any(direction_valid))
    valid = jnp.asarray(forward_valid and backward_valid and enough_depth)
    converged = valid & (
        jnp.maximum(maximum_error, maximum_drift) <= float(convergence_tolerance)
    )
    status = jnp.asarray(
        COVARIANT_SUCCESS
        if bool(converged)
        else COVARIANT_INSUFFICIENT_BACKWARD_DEPTH
        if forward_valid and backward_valid
        else COVARIANT_SINGULAR_FACTOR
        if forward_valid
        else COVARIANT_EVOLUTION_FAILED,
        dtype=jnp.int32,
    )
    saved_indices = jnp.asarray(saved, dtype=jnp.int32)
    saved_coordinates = coordinates[
        jnp.asarray([boundaries[index] for index in saved], dtype=jnp.int32)
    ]
    return CovariantDirectionResult(
        coordinates=saved_coordinates,
        states=saved_states_array,
        directions=directions,
        direction_valid=direction_valid,
        local_growth_rates=growth,
        covariance_error=errors,
        backward_convergence_drift=convergence_drift,
        saved_checkpoint_index=saved_indices,
        valid=valid,
        converged=converged,
        status=status,
        evolution=evolution,
        grid=grid,
        kind=kind,
        memory_mode=memory_mode,
        leading_k=rank,
        full_basis=rank == dimension,
        qr_interval=cadence,
        convergence_tolerance=float(convergence_tolerance),
        save_every=saving,
        backward_discard=discard,
        stored_frame_count=(intervals + 1 if memory_mode == "store" else 0),
        tangent_evaluations=tangent_evaluations,
        method_id=f"ginelli-periodic-qr:{kind}:{memory_mode}",
        evolution_id=evolution.evolution_id,
        tangent_method_id=evolution.tangent_method_id,
        approximation_id=(
            "finite-window-full-covariant-basis"
            if rank == dimension
            else "finite-window-leading-covariant-subspace"
        ),
    )


__all__ = [
    "COVARIANT_EVOLUTION_FAILED",
    "COVARIANT_INSUFFICIENT_BACKWARD_DEPTH",
    "COVARIANT_NONFINITE",
    "COVARIANT_SINGULAR_FACTOR",
    "COVARIANT_SUCCESS",
    "CovariantDirectionKind",
    "CovariantDirectionResult",
    "CovariantMemoryMode",
    "covariant_directions",
]
