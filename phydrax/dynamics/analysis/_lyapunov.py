#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._evolution import AbstractDifferentiableEvolution
from .._grid import EvolutionGrid, IterationGrid, TimeGrid
from .._system import ContinuousSystem, DiscreteSystem


LyapunovSpectrumMethod: TypeAlias = Literal["periodic_qr"]

LYAPUNOV_SUCCESS = 0
LYAPUNOV_NONFINITE_TANGENT = 1
LYAPUNOV_SINGULAR_TANGENT = 2
LYAPUNOV_INSUFFICIENT_ACCUMULATION = 3
LYAPUNOV_EVOLUTION_FAILED = 4


class LyapunovSpectrumCheckpoint(StrictModule):
    """Complete periodic-QR state for exact continuation on one evolution."""

    state: Array
    basis: Array
    log_stretch: Array
    accumulated_time: Array
    current_coordinate: Array
    valid: Array
    status: Array
    step_index: int = eqx.field(static=True)
    accumulated_intervals: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    leading_k: int = eqx.field(static=True)
    qr_interval: int = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    accumulation_interval: int = eqx.field(static=True)
    system_kind: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    tangent_method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: ArrayLike,
        basis: ArrayLike,
        log_stretch: ArrayLike,
        accumulated_time: ArrayLike,
        current_coordinate: ArrayLike,
        step_index: int,
        accumulated_intervals: int,
        state_shape: tuple[int, ...],
        leading_k: int,
        qr_interval: int,
        burn_in: int,
        accumulation_interval: int,
        system_kind: str,
        system_id: str,
        evolution_id: str,
        tangent_method: str,
        backend: str,
        discretization_id: str,
        approximation_id: str,
        valid: ArrayLike = True,
        status: ArrayLike = LYAPUNOV_SUCCESS,
    ):
        state_array = jnp.asarray(state)
        shape = tuple(int(size) for size in state_shape)
        dimension = int(state_array.size)
        rank = int(leading_k)
        basis_array = jnp.asarray(basis)
        logs = jnp.asarray(log_stretch)
        elapsed = jnp.asarray(accumulated_time, dtype=float)
        coordinate = jnp.asarray(current_coordinate)
        valid_array = jnp.asarray(valid, dtype=bool)
        status_array = jnp.asarray(status, dtype=jnp.int32)
        if not jnp.issubdtype(state_array.dtype, jnp.floating):
            raise TypeError("Checkpoint state must have a real floating dtype.")
        if tuple(state_array.shape) != shape:
            raise ValueError("Checkpoint state does not match state_shape.")
        if basis_array.shape != (dimension, rank):
            raise ValueError("Checkpoint basis has an incompatible shape.")
        if logs.shape != (rank,):
            raise ValueError("Checkpoint log_stretch has an incompatible shape.")
        if elapsed.shape != () or coordinate.shape != ():
            raise ValueError("Checkpoint coordinates and elapsed time must be scalar.")
        if valid_array.shape != () or status_array.shape != ():
            raise ValueError("Checkpoint validity and status must be scalar.")
        if int(step_index) < 0 or int(accumulated_intervals) < 0:
            raise ValueError("Checkpoint counters must be non-negative.")
        identifiers = (
            system_kind,
            system_id,
            evolution_id,
            tangent_method,
            backend,
            discretization_id,
            approximation_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError(
                "Checkpoint provenance identifiers must be non-empty strings."
            )
        self.state = state_array
        self.basis = basis_array
        self.log_stretch = logs
        self.accumulated_time = elapsed
        self.current_coordinate = coordinate
        self.valid = valid_array
        self.status = status_array
        self.step_index = int(step_index)
        self.accumulated_intervals = int(accumulated_intervals)
        self.state_shape = shape
        self.state_dimension = dimension
        self.leading_k = rank
        self.qr_interval = int(qr_interval)
        self.burn_in = int(burn_in)
        self.accumulation_interval = int(accumulation_interval)
        self.system_kind = system_kind
        self.system_id = system_id
        self.evolution_id = evolution_id
        self.method_id = "periodic_qr"
        self.tangent_method = tangent_method
        self.backend = backend
        self.discretization_id = discretization_id
        self.approximation_id = approximation_id

    @property
    def current_time(self) -> Array:
        return self.current_coordinate


class LyapunovSpectrumResult(StrictModule):
    """Finite-time Lyapunov spectrum with numerical and evolution provenance."""

    exponents: Array
    finite_time_exponents: Array
    accumulation_times: Array
    convergence_drift: Array
    kaplan_yorke_dimension: Array
    kaplan_yorke_valid: Array
    valid: Array
    status: Array
    final_state: Array
    checkpoint: LyapunovSpectrumCheckpoint
    method: LyapunovSpectrumMethod = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    system_kind: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    tangent_method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    leading_k: int = eqx.field(static=True)
    full_spectrum: bool = eqx.field(static=True)
    qr_interval: int = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    accumulation_interval: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        exponents: ArrayLike,
        finite_time_exponents: ArrayLike,
        accumulation_times: ArrayLike,
        convergence_drift: ArrayLike,
        kaplan_yorke_dimension: ArrayLike,
        kaplan_yorke_valid: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        checkpoint: LyapunovSpectrumCheckpoint,
    ):
        if not isinstance(checkpoint, LyapunovSpectrumCheckpoint):
            raise TypeError("checkpoint must be a LyapunovSpectrumCheckpoint.")
        spectrum = jnp.asarray(exponents)
        history = jnp.asarray(finite_time_exponents)
        times = jnp.asarray(accumulation_times, dtype=float)
        if spectrum.shape != (checkpoint.leading_k,):
            raise ValueError("Spectrum shape does not match the checkpoint rank.")
        if history.ndim != 2 or history.shape[1:] != spectrum.shape:
            raise ValueError("finite_time_exponents must have shape (interval, k).")
        if times.shape != (history.shape[0],):
            raise ValueError("accumulation_times must align with finite-time estimates.")
        self.exponents = spectrum
        self.finite_time_exponents = history
        self.accumulation_times = times
        self.convergence_drift = jnp.asarray(convergence_drift)
        self.kaplan_yorke_dimension = jnp.asarray(kaplan_yorke_dimension)
        self.kaplan_yorke_valid = jnp.asarray(kaplan_yorke_valid, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.final_state = checkpoint.state
        self.checkpoint = checkpoint
        self.method = "periodic_qr"
        self.method_id = checkpoint.method_id
        self.system_kind = checkpoint.system_kind
        self.system_id = checkpoint.system_id
        self.evolution_id = checkpoint.evolution_id
        self.tangent_method = checkpoint.tangent_method
        self.backend = checkpoint.backend
        self.discretization_id = checkpoint.discretization_id
        self.approximation = (
            "full_finite_time_spectrum"
            if checkpoint.leading_k == checkpoint.state_dimension
            else "leading_k_finite_time_spectrum"
        )
        self.state_shape = checkpoint.state_shape
        self.state_dimension = checkpoint.state_dimension
        self.leading_k = checkpoint.leading_k
        self.full_spectrum = checkpoint.leading_k == checkpoint.state_dimension
        self.qr_interval = checkpoint.qr_interval
        self.burn_in = checkpoint.burn_in
        self.accumulation_interval = checkpoint.accumulation_interval


def kaplan_yorke_dimension(exponents: ArrayLike, /) -> Array:
    """Return the Kaplan--Yorke dimension for one complete ordered spectrum."""
    values = jnp.asarray(exponents)
    if values.ndim != 1 or int(values.size) == 0:
        raise ValueError("exponents must be a non-empty rank-1 array.")
    values = jnp.sort(values)[::-1]
    cumulative = jnp.cumsum(values)
    count = jnp.sum(cumulative >= 0.0, dtype=jnp.int32)
    dimension = int(values.size)
    index = jnp.minimum(count, dimension - 1)
    fractional = cumulative[jnp.maximum(count - 1, 0)] / jnp.abs(values[index])
    interior = count.astype(values.dtype) + fractional
    return jnp.where(
        count == 0,
        0.0,
        jnp.where(count == dimension, dimension, interior),
    )


def _thin_qr(matrix: Array, /) -> tuple[Array, Array]:
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    raw_diagonal = jnp.diag(r)
    diagonal = jnp.abs(raw_diagonal)
    signs = jnp.where(raw_diagonal < 0.0, -1.0, 1.0)
    return q * signs[None, :], diagonal


def _initial_basis(state: Array, rank: int, supplied: ArrayLike | None, /) -> Array:
    dimension = int(state.size)
    dtype = jnp.result_type(state, float)
    if supplied is None:
        if rank == dimension:
            return jnp.eye(dimension, dtype=dtype)
        basis, _ = _thin_qr(
            jax.random.normal(jax.random.PRNGKey(0), (dimension, rank), dtype=dtype)
        )
        return basis
    basis = jnp.asarray(supplied)
    if basis.shape == tuple(state.shape) + (rank,):
        basis = basis.reshape((dimension, rank))
    if basis.shape != (dimension, rank):
        raise ValueError("initial_basis must have shape state_shape + (leading_k,).")
    basis, diagonal = _thin_qr(basis)
    if bool(jnp.any(diagonal == 0.0)):
        raise ValueError("initial_basis must have full column rank.")
    return basis


def _schedule(qr_interval: int, burn_in: int, accumulation_interval: int) -> None:
    if qr_interval <= 0:
        raise ValueError("qr_interval must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if accumulation_interval <= 0:
        raise ValueError("accumulation_interval must be positive.")
    if accumulation_interval % qr_interval != 0:
        raise ValueError("accumulation_interval must be a multiple of qr_interval.")


def _provenance(evolution: AbstractDifferentiableEvolution, grid: EvolutionGrid, /):
    if isinstance(evolution.system, ContinuousSystem):
        system_kind = "continuous"
    elif isinstance(evolution.system, DiscreteSystem):
        system_kind = "discrete"
    else:
        raise TypeError("Unsupported dynamical system type for Lyapunov analysis.")
    return (
        system_kind,
        evolution.system.system_id,
        evolution.evolution_id,
        evolution.tangent_method_id,
        evolution.backend_id,
        f"{evolution.discretization_id}:{grid.grid_id}",
        evolution.approximation_id,
    )


def finite_time_lyapunov_spectrum(
    evolution: AbstractDifferentiableEvolution,
    initial_state: ArrayLike | None,
    grid: EvolutionGrid,
    /,
    *,
    args=None,
    leading_k: int | None = None,
    qr_interval: int = 1,
    burn_in: int = 0,
    accumulation_interval: int | None = None,
    initial_basis: ArrayLike | None = None,
    checkpoint: LyapunovSpectrumCheckpoint | None = None,
) -> LyapunovSpectrumResult:
    """Estimate a pathwise finite-time spectrum over one declared evolution grid.

    Scheduling values count adjacent grid segments. Physical-time normalization comes
    from the grid coordinates, so irregular ``TimeGrid`` intervals remain explicit.
    """
    if not isinstance(evolution, AbstractDifferentiableEvolution):
        raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
    if not isinstance(grid, (TimeGrid, IterationGrid)):
        raise TypeError("grid must be a TimeGrid or IterationGrid.")
    cadence = int(qr_interval)
    burn = int(burn_in)
    accumulation = (
        cadence if accumulation_interval is None else int(accumulation_interval)
    )
    _schedule(cadence, burn, accumulation)
    provenance = _provenance(evolution, grid)
    state_shape = evolution.state_layout.shape
    dimension = evolution.state_layout.size

    if checkpoint is None:
        if initial_state is None:
            raise ValueError("initial_state is required without a checkpoint.")
        state = jnp.asarray(initial_state)
        if state.shape != state_shape:
            raise ValueError(
                f"initial_state must have shape {state_shape}; got {state.shape}."
            )
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("initial_state must have a real floating dtype.")
        rank = dimension if leading_k is None else int(leading_k)
        if rank <= 0 or rank > dimension:
            raise ValueError("leading_k must lie between one and the state dimension.")
        basis = _initial_basis(state, rank, initial_basis)
        logs = jnp.zeros((rank,), dtype=jnp.result_type(state, float))
        elapsed = jnp.asarray(0.0, dtype=grid.coordinates.dtype)
        start_step = 0
        previous_intervals = 0
        initial_member = jnp.asarray(
            evolution.state_layout.geometry.contains(state), dtype=bool
        )
        valid = jnp.all(jnp.isfinite(state)) & initial_member
        status = jnp.where(valid, LYAPUNOV_SUCCESS, LYAPUNOV_EVOLUTION_FAILED).astype(
            jnp.int32
        )
    else:
        if not isinstance(checkpoint, LyapunovSpectrumCheckpoint):
            raise TypeError("checkpoint must be a LyapunovSpectrumCheckpoint or None.")
        expected = (
            checkpoint.state_shape == state_shape
            and checkpoint.qr_interval == cadence
            and checkpoint.burn_in == burn
            and checkpoint.accumulation_interval == accumulation
            and (
                checkpoint.system_kind,
                checkpoint.system_id,
                checkpoint.evolution_id,
                checkpoint.tangent_method,
                checkpoint.backend,
                checkpoint.discretization_id,
                checkpoint.approximation_id,
            )
            == provenance
        )
        if not expected:
            raise ValueError("Checkpoint provenance is incompatible with this run.")
        if leading_k is not None and int(leading_k) != checkpoint.leading_k:
            raise ValueError("leading_k is incompatible with the checkpoint.")
        if initial_basis is not None:
            raise ValueError("initial_basis cannot be supplied when resuming.")
        if initial_state is not None and tuple(jnp.shape(initial_state)) != state_shape:
            raise ValueError("initial_state shape is incompatible with the checkpoint.")
        state = eqx.error_if(
            checkpoint.state,
            grid.coordinates[0] != checkpoint.current_coordinate,
            "Checkpoint coordinate must equal the first grid coordinate.",
        )
        rank = checkpoint.leading_k
        basis = checkpoint.basis
        logs = checkpoint.log_stretch
        elapsed = checkpoint.accumulated_time
        start_step = checkpoint.step_index
        previous_intervals = checkpoint.accumulated_intervals
        valid = checkpoint.valid
        status = checkpoint.status

    num_steps = grid.num_steps
    final_step = start_step + num_steps
    report_indices = tuple(
        local_index
        for local_index in range(num_steps)
        if (
            start_step + local_index + 1 > burn
            and (
                (start_step + local_index + 1 - burn) % accumulation == 0
                or local_index + 1 == num_steps
            )
        )
    )
    new_intervals = sum(
        1
        for local_index in range(num_steps)
        if (
            start_step + local_index + 1 > burn
            and (
                start_step + local_index + 1 == final_step
                or (start_step + local_index + 1 - burn) % cadence == 0
            )
        )
    )

    def scan_step(carry, scan_input):
        current, frame, stretch, total_time, pending_time, run_valid, run_status = carry
        local_index, source, target = scan_input
        global_step = start_step + local_index + 1

        def propagate(vector):
            tangent_step = evolution.tangent_action(
                current,
                vector.reshape(state_shape),
                source,
                target,
                args,
            )
            return (
                tangent_step.primal.final_state,
                tangent_step.tangent.reshape((-1,)),
                tangent_step.primal.valid,
                jnp.all(jnp.isfinite(tangent_step.tangent)),
            )

        states, propagated, primal_valid, tangent_finite = jax.vmap(
            propagate, in_axes=1, out_axes=(0, 1, 0, 0)
        )(frame)
        next_state = states[0]
        evolution_valid = jnp.all(primal_valid)
        tangents_valid = jnp.all(tangent_finite) & jnp.all(jnp.isfinite(propagated))
        step_valid = evolution_valid & tangents_valid
        step_status = jnp.where(
            ~evolution_valid,
            LYAPUNOV_EVOLUTION_FAILED,
            jnp.where(
                tangents_valid,
                LYAPUNOV_SUCCESS,
                LYAPUNOV_NONFINITE_TANGENT,
            ),
        ).astype(jnp.int32)
        next_valid = run_valid & step_valid
        next_status = jnp.where(run_status == LYAPUNOV_SUCCESS, step_status, run_status)
        after_burn = global_step > burn
        next_pending = pending_time + jnp.where(
            after_burn, target - source, jnp.asarray(0.0, dtype=target.dtype)
        )
        at_burn = global_step == burn
        before_burn_qr = (global_step < burn) & (global_step % cadence == 0)
        after_burn_qr = (global_step > burn) & ((global_step - burn) % cadence == 0)
        is_final = local_index + 1 == num_steps
        should_qr = at_burn | before_burn_qr | after_burn_qr | is_final

        def orthonormalize(values):
            (
                candidate_frame,
                candidate_stretch,
                candidate_time,
                candidate_pending,
                candidate_valid,
                candidate_status,
            ) = values
            orthonormal, diagonal = _thin_qr(candidate_frame)
            finite = jnp.all(jnp.isfinite(next_state)) & jnp.all(
                jnp.isfinite(orthonormal)
            )
            nonsingular = jnp.all(diagonal > 0.0)
            qr_status = jnp.where(
                finite,
                jnp.where(
                    nonsingular,
                    LYAPUNOV_SUCCESS,
                    LYAPUNOV_SINGULAR_TANGENT,
                ),
                LYAPUNOV_NONFINITE_TANGENT,
            ).astype(jnp.int32)
            updated_valid = candidate_valid & finite & nonsingular
            updated_status = jnp.where(
                candidate_status == LYAPUNOV_SUCCESS,
                qr_status,
                candidate_status,
            )
            accumulate = candidate_pending > 0.0
            updated_stretch = jnp.where(
                accumulate,
                candidate_stretch + jnp.log(diagonal),
                candidate_stretch,
            )
            updated_time = candidate_time + jnp.where(
                accumulate,
                candidate_pending,
                jnp.asarray(0.0, dtype=candidate_time.dtype),
            )
            updated_pending = jnp.where(
                accumulate,
                jnp.asarray(0.0, dtype=candidate_pending.dtype),
                candidate_pending,
            )
            return (
                orthonormal,
                updated_stretch,
                updated_time,
                updated_pending,
                updated_valid,
                updated_status,
            )

        frame, stretch, total_time, next_pending, next_valid, next_status = jax.lax.cond(
            should_qr,
            orthonormalize,
            lambda values: values,
            (
                propagated,
                stretch,
                total_time,
                next_pending,
                next_valid,
                next_status,
            ),
        )
        estimate = jnp.where(
            total_time > 0.0,
            stretch / total_time,
            jnp.full_like(stretch, jnp.nan),
        )
        return (
            next_state,
            frame,
            stretch,
            total_time,
            next_pending,
            next_valid,
            next_status,
        ), (estimate, total_time)

    initial_carry = (
        state,
        basis,
        logs,
        elapsed,
        jnp.asarray(0.0, dtype=grid.coordinates.dtype),
        valid,
        status,
    )
    final_carry, (all_estimates, all_times) = jax.lax.scan(
        scan_step,
        initial_carry,
        (
            jnp.arange(num_steps, dtype=jnp.int32),
            grid.coordinates[:-1],
            grid.coordinates[1:],
        ),
    )
    state, basis, logs, elapsed, _, valid, status = final_carry
    if report_indices:
        indices = jnp.asarray(report_indices, dtype=jnp.int32)
        history = jnp.take(all_estimates, indices, axis=0)
        history_times = jnp.take(all_times, indices, axis=0)
        exponents = history[-1]
        result_valid = valid
        result_status = status
    else:
        history = jnp.empty((0, rank), dtype=logs.dtype)
        history_times = jnp.empty((0,), dtype=grid.coordinates.dtype)
        exponents = jnp.full((rank,), jnp.nan, dtype=logs.dtype)
        result_valid = jnp.asarray(False)
        result_status = jnp.where(
            valid,
            jnp.asarray(LYAPUNOV_INSUFFICIENT_ACCUMULATION, dtype=jnp.int32),
            status,
        )
    drift = (
        jnp.max(jnp.abs(history[-1] - history[-2]))
        if int(history.shape[0]) >= 2
        else jnp.asarray(jnp.nan, dtype=logs.dtype)
    )
    full_spectrum = rank == dimension
    ky_valid = (
        jnp.asarray(full_spectrum) & result_valid & jnp.all(jnp.isfinite(exponents))
    )
    ky_dimension = jnp.where(
        ky_valid,
        kaplan_yorke_dimension(exponents),
        jnp.asarray(jnp.nan, dtype=exponents.dtype),
    )
    checkpoint_result = LyapunovSpectrumCheckpoint(
        state=state,
        basis=basis,
        log_stretch=logs,
        accumulated_time=elapsed,
        current_coordinate=grid.coordinates[-1],
        step_index=final_step,
        accumulated_intervals=previous_intervals + new_intervals,
        state_shape=state_shape,
        leading_k=rank,
        qr_interval=cadence,
        burn_in=burn,
        accumulation_interval=accumulation,
        system_kind=provenance[0],
        system_id=provenance[1],
        evolution_id=provenance[2],
        tangent_method=provenance[3],
        backend=provenance[4],
        discretization_id=provenance[5],
        approximation_id=provenance[6],
        valid=valid,
        status=status,
    )
    return LyapunovSpectrumResult(
        exponents=exponents,
        finite_time_exponents=history,
        accumulation_times=history_times,
        convergence_drift=drift,
        kaplan_yorke_dimension=ky_dimension,
        kaplan_yorke_valid=ky_valid,
        valid=result_valid,
        status=result_status,
        checkpoint=checkpoint_result,
    )


__all__ = [
    "LYAPUNOV_EVOLUTION_FAILED",
    "LYAPUNOV_INSUFFICIENT_ACCUMULATION",
    "LYAPUNOV_NONFINITE_TANGENT",
    "LYAPUNOV_SINGULAR_TANGENT",
    "LYAPUNOV_SUCCESS",
    "LyapunovSpectrumCheckpoint",
    "LyapunovSpectrumMethod",
    "LyapunovSpectrumResult",
    "finite_time_lyapunov_spectrum",
    "kaplan_yorke_dimension",
]
