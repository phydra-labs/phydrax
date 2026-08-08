#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._differential import DifferentialProblem


LyapunovSpectrumMethod: TypeAlias = Literal["periodic_qr"]

LYAPUNOV_SUCCESS = 0
LYAPUNOV_NONFINITE_TANGENT = 1
LYAPUNOV_SINGULAR_TANGENT = 2
LYAPUNOV_INSUFFICIENT_ACCUMULATION = 3


class LyapunovSpectrumCheckpoint(StrictModule):
    """Complete periodic-QR state for an exact continuation of an estimate."""

    state: Array
    basis: Array
    log_stretch: Array
    accumulated_time: Array
    current_time: Array
    step_index: int = eqx.field(static=True)
    accumulated_intervals: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    leading_k: int = eqx.field(static=True)
    qr_interval: float = eqx.field(static=True)
    burn_in: float = eqx.field(static=True)
    accumulation_interval: float = eqx.field(static=True)
    system_kind: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    tangent_method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    valid: Array
    status: Array

    def __init__(
        self,
        *,
        state: ArrayLike,
        basis: ArrayLike,
        log_stretch: ArrayLike,
        accumulated_time: ArrayLike,
        current_time: ArrayLike,
        step_index: int,
        accumulated_intervals: int,
        state_shape: tuple[int, ...],
        leading_k: int,
        qr_interval: float,
        burn_in: float,
        accumulation_interval: float,
        system_kind: str,
        system_id: str,
        tangent_method: str,
        backend: str,
        discretization_id: str,
        valid: ArrayLike = True,
        status: ArrayLike = LYAPUNOV_SUCCESS,
    ):
        state_array = jnp.asarray(state)
        if not jnp.issubdtype(state_array.dtype, jnp.floating):
            raise TypeError("Checkpoint state must have a real floating dtype.")
        shape = tuple(int(size) for size in state_shape)
        dimension = int(state_array.size)
        if tuple(state_array.shape) != shape:
            raise ValueError("Checkpoint state does not match state_shape.")
        basis_array = jnp.asarray(basis)
        logs = jnp.asarray(log_stretch)
        if basis_array.shape != (dimension, int(leading_k)):
            raise ValueError("Checkpoint basis has an incompatible shape.")
        if logs.shape != (int(leading_k),):
            raise ValueError("Checkpoint log_stretch has an incompatible shape.")
        elapsed = jnp.asarray(accumulated_time, dtype=float)
        time = jnp.asarray(current_time, dtype=float)
        valid_array = jnp.asarray(valid, dtype=bool)
        status_array = jnp.asarray(status, dtype=jnp.int32)
        if elapsed.shape != () or time.shape != ():
            raise ValueError("Checkpoint times must be scalar.")
        if valid_array.shape != () or status_array.shape != ():
            raise ValueError("Checkpoint validity and status must be scalar.")
        if int(step_index) < 0 or int(accumulated_intervals) < 0:
            raise ValueError("Checkpoint counters must be non-negative.")
        if not system_id:
            raise ValueError("system_id must be non-empty.")
        self.state = state_array
        self.basis = basis_array
        self.log_stretch = logs
        self.accumulated_time = elapsed
        self.current_time = time
        self.step_index = int(step_index)
        self.accumulated_intervals = int(accumulated_intervals)
        self.state_shape = shape
        self.state_dimension = dimension
        self.leading_k = int(leading_k)
        self.qr_interval = float(qr_interval)
        self.burn_in = float(burn_in)
        self.accumulation_interval = float(accumulation_interval)
        self.system_kind = str(system_kind)
        self.system_id = str(system_id)
        self.method_id = "periodic_qr"
        self.tangent_method = str(tangent_method)
        self.backend = str(backend)
        self.discretization_id = str(discretization_id)
        self.valid = valid_array
        self.status = status_array


class LyapunovSpectrumResult(StrictModule):
    """Finite-time Lyapunov spectrum and its complete numerical provenance."""

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
    tangent_method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    leading_k: int = eqx.field(static=True)
    full_spectrum: bool = eqx.field(static=True)
    qr_interval: float = eqx.field(static=True)
    burn_in: float = eqx.field(static=True)
    accumulation_interval: float = eqx.field(static=True)

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
    """Return the Kaplan--Yorke dimension for a complete spectrum."""
    values = jnp.asarray(exponents)
    if values.ndim != 1 or int(values.size) == 0:
        raise ValueError("exponents must be a non-empty rank-1 array.")
    values = jnp.sort(values)[::-1]
    cumulative = jnp.cumsum(values)
    nonnegative = cumulative >= 0.0
    count = jnp.sum(nonnegative, dtype=jnp.int32)
    dimension = int(values.size)
    index = jnp.minimum(count, dimension - 1)
    fractional = cumulative[jnp.maximum(count - 1, 0)] / jnp.abs(values[index])
    interior = count.astype(values.dtype) + fractional
    return jnp.where(
        count == 0,
        0.0,
        jnp.where(count == dimension, dimension, interior),
    )


def lyapunov_spectrum_map(
    map_fn: Callable[[Array, Any], ArrayLike],
    initial_state: ArrayLike | None,
    /,
    *,
    num_steps: int,
    args: Any = None,
    leading_k: int | None = None,
    qr_interval: int = 1,
    burn_in: int = 0,
    accumulation_interval: int | None = None,
    tangent_action: Callable[[Array, Array, Any], ArrayLike] | None = None,
    initial_basis: ArrayLike | None = None,
    checkpoint: LyapunovSpectrumCheckpoint | None = None,
    system_id: str = "map",
) -> LyapunovSpectrumResult:
    """Estimate map exponents with matrix-free JVPs and periodic thin QR.

    ``map_fn(state, args)`` advances one iterate. A supplied
    ``tangent_action(state, vector, args)`` must apply its Jacobian to one vector;
    otherwise JAX JVP supplies the matrix-free action.
    """
    if not callable(map_fn):
        raise TypeError("map_fn must be callable.")
    if tangent_action is not None and not callable(tangent_action):
        raise TypeError("tangent_action must be callable or None.")
    steps = int(num_steps)
    cadence = int(qr_interval)
    burn = int(burn_in)
    accumulation = (
        cadence if accumulation_interval is None else int(accumulation_interval)
    )
    if steps <= 0:
        raise ValueError("num_steps must be positive.")
    _validate_integer_schedule(cadence, burn, accumulation)
    tangent_method = "jax_jvp" if tangent_action is None else "user_jvp"
    state, basis, logs, elapsed, step_index, accumulated, valid, status = _initial_state(
        initial_state,
        initial_basis=initial_basis,
        leading_k=leading_k,
        checkpoint=checkpoint,
        system_kind="map",
        system_id=system_id,
        qr_interval=float(cadence),
        burn_in=float(burn),
        accumulation_interval=float(accumulation),
        tangent_method=tangent_method,
        backend="jax",
        discretization_id="map_iterate",
    )
    state_shape = tuple(state.shape)
    history: list[Array] = []
    history_times: list[Array] = []
    steps_since_qr = 0
    start_step = step_index
    for local_index in range(steps):
        previous = state
        if tangent_action is None:
            state, linearized_map = jax.linearize(
                lambda value: jnp.asarray(map_fn(value, args)),
                previous,
            )

            def one_action(vector):
                return linearized_map(vector.reshape(state_shape)).reshape(-1)
        else:
            state = jnp.asarray(map_fn(previous, args))

            def one_action(vector):
                value = jnp.asarray(
                    tangent_action(previous, vector.reshape(state_shape), args)
                )
                if tuple(value.shape) != state_shape:
                    raise ValueError("tangent_action changed the tangent shape.")
                return value.reshape(-1)

        basis = jax.vmap(one_action, in_axes=1, out_axes=1)(basis)
        if tuple(state.shape) != state_shape:
            raise ValueError("map_fn changed the state shape.")
        step_index = start_step + local_index + 1
        steps_since_qr += 1
        at_burn = step_index == burn
        before_burn_qr = step_index < burn and step_index % cadence == 0
        after_burn_qr = step_index > burn and (step_index - burn) % cadence == 0
        scheduled_qr = before_burn_qr or after_burn_qr
        is_final = local_index + 1 == steps
        if scheduled_qr or at_burn or is_final:
            basis, diagonal = _thin_qr(basis)
            finite = jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(basis))
            nonsingular = jnp.all(diagonal > 0.0)
            next_status = jnp.where(
                finite,
                jnp.where(nonsingular, LYAPUNOV_SUCCESS, LYAPUNOV_SINGULAR_TANGENT),
                LYAPUNOV_NONFINITE_TANGENT,
            )
            valid = valid & finite & nonsingular
            status = jnp.where(status == LYAPUNOV_SUCCESS, next_status, status)
            left_step = step_index - steps_since_qr
            accumulation_steps = max(0, step_index - max(left_step, burn))
            if accumulation_steps > 0:
                logs = logs + jnp.log(diagonal)
                elapsed = elapsed + float(accumulation_steps)
                accumulated += 1
            steps_since_qr = 0
        if step_index > burn and ((step_index - burn) % accumulation == 0 or is_final):
            estimate = logs / elapsed
            history.append(estimate)
            history_times.append(elapsed)
    return _make_result(
        state=state,
        basis=basis,
        logs=logs,
        elapsed=elapsed,
        current_time=float(step_index),
        step_index=step_index,
        accumulated=accumulated,
        history=history,
        history_times=history_times,
        state_shape=state_shape,
        cadence=float(cadence),
        burn=float(burn),
        accumulation=float(accumulation),
        system_kind="map",
        system_id=system_id,
        tangent_method=tangent_method,
        backend="jax",
        discretization_id="map_iterate",
        valid=valid,
        status=status,
    )


def lyapunov_spectrum_flow(
    problem: DifferentialProblem,
    /,
    *,
    step_size: float,
    leading_k: int | None = None,
    qr_interval: float,
    burn_in: float = 0.0,
    accumulation_interval: float | None = None,
    tangent_action: Callable[[Array, Array, Array, Any], ArrayLike] | None = None,
    initial_basis: ArrayLike | None = None,
    checkpoint: LyapunovSpectrumCheckpoint | None = None,
    system_id: str = "differential_flow",
) -> LyapunovSpectrumResult:
    """Estimate deterministic-flow exponents using fixed-step RK4 tangent actions.

    The flow is the supplied :class:`DifferentialProblem`. ``tangent_action`` has
    signature ``(time, state, vector, args)``; when omitted, a JVP of the declared
    drift is used. Cadences and burn-in are physical times and must be integer
    multiples of ``step_size``.
    """
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("problem must be a DifferentialProblem.")
    if problem.stochastic:
        raise ValueError("Lyapunov flow spectra require a deterministic problem.")
    if tangent_action is not None and not callable(tangent_action):
        raise TypeError("tangent_action must be callable or None.")
    step = float(step_size)
    cadence_time = float(qr_interval)
    burn_time = float(burn_in)
    accumulation_time = (
        cadence_time if accumulation_interval is None else float(accumulation_interval)
    )
    if not jnp.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    cadence_steps = _time_to_steps(cadence_time, step, owner="qr_interval")
    burn_steps = _time_to_steps(burn_time, step, owner="burn_in", allow_zero=True)
    accumulation_steps = _time_to_steps(
        accumulation_time, step, owner="accumulation_interval"
    )
    _validate_integer_schedule(cadence_steps, burn_steps, accumulation_steps)
    total_time = float(problem.t1 - problem.t0)
    total_steps = _time_to_steps(total_time, step, owner="problem duration")
    tangent_method = "jax_jvp" if tangent_action is None else "user_jvp"
    state, basis, logs, elapsed, step_index, accumulated, valid, status = _initial_state(
        problem.initial_state,
        initial_basis=initial_basis,
        leading_k=leading_k,
        checkpoint=checkpoint,
        system_kind="flow",
        system_id=system_id,
        qr_interval=cadence_time,
        burn_in=burn_time,
        accumulation_interval=accumulation_time,
        tangent_method=tangent_method,
        backend="phydrax.solver",
        discretization_id=f"fixed_rk4_dt={step:.17g}",
    )
    if checkpoint is not None:
        tolerance = 32.0 * jnp.finfo(float).eps * max(1.0, abs(float(problem.t0)))
        if abs(float(checkpoint.current_time) - float(problem.t0)) > tolerance:
            raise ValueError(
                "A resumed flow problem must start at checkpoint.current_time."
            )
    continuation_steps = total_steps
    state_shape = tuple(state.shape)

    def augmented_rhs(time, current_state, current_basis):
        if tangent_action is None:
            derivative, linearized_drift = jax.linearize(
                lambda value: jnp.asarray(problem.drift(time, value, problem.args)),
                current_state,
            )

            def one_action(vector):
                return linearized_drift(vector.reshape(state_shape)).reshape(-1)
        else:
            derivative = jnp.asarray(problem.drift(time, current_state, problem.args))

            def one_action(vector):
                value = jnp.asarray(
                    tangent_action(
                        time,
                        current_state,
                        vector.reshape(state_shape),
                        problem.args,
                    )
                )
                if tuple(value.shape) != state_shape:
                    raise ValueError("tangent_action changed the tangent shape.")
                return value.reshape(-1)

        if tuple(derivative.shape) != state_shape:
            raise ValueError("DifferentialProblem drift changed the state shape.")
        tangent = jax.vmap(one_action, in_axes=1, out_axes=1)(current_basis)
        return derivative, tangent

    history: list[Array] = []
    history_times: list[Array] = []
    steps_since_qr = 0
    start_step = step_index
    start_time = jnp.asarray(problem.t0)
    for local_index in range(continuation_steps):
        time = start_time + local_index * step
        k1_state, k1_basis = augmented_rhs(time, state, basis)
        k2_state, k2_basis = augmented_rhs(
            time + 0.5 * step,
            state + 0.5 * step * k1_state,
            basis + 0.5 * step * k1_basis,
        )
        k3_state, k3_basis = augmented_rhs(
            time + 0.5 * step,
            state + 0.5 * step * k2_state,
            basis + 0.5 * step * k2_basis,
        )
        k4_state, k4_basis = augmented_rhs(
            time + step,
            state + step * k3_state,
            basis + step * k3_basis,
        )
        state = state + (step / 6.0) * (
            k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state
        )
        basis = basis + (step / 6.0) * (
            k1_basis + 2.0 * k2_basis + 2.0 * k3_basis + k4_basis
        )
        step_index = start_step + local_index + 1
        steps_since_qr += 1
        at_burn = step_index == burn_steps
        before_burn_qr = step_index < burn_steps and step_index % cadence_steps == 0
        after_burn_qr = (
            step_index > burn_steps and (step_index - burn_steps) % cadence_steps == 0
        )
        scheduled_qr = before_burn_qr or after_burn_qr
        is_final = local_index + 1 == continuation_steps
        if scheduled_qr or at_burn or is_final:
            basis, diagonal = _thin_qr(basis)
            finite = jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(basis))
            nonsingular = jnp.all(diagonal > 0.0)
            next_status = jnp.where(
                finite,
                jnp.where(nonsingular, LYAPUNOV_SUCCESS, LYAPUNOV_SINGULAR_TANGENT),
                LYAPUNOV_NONFINITE_TANGENT,
            )
            valid = valid & finite & nonsingular
            status = jnp.where(status == LYAPUNOV_SUCCESS, next_status, status)
            left_step = step_index - steps_since_qr
            effective_steps = max(0, step_index - max(left_step, burn_steps))
            if effective_steps > 0:
                logs = logs + jnp.log(diagonal)
                elapsed = elapsed + effective_steps * step
                accumulated += 1
            steps_since_qr = 0
        if step_index > burn_steps and (
            (step_index - burn_steps) % accumulation_steps == 0 or is_final
        ):
            estimate = logs / elapsed
            history.append(estimate)
            history_times.append(elapsed)
    current_time = float(problem.t1)
    return _make_result(
        state=state,
        basis=basis,
        logs=logs,
        elapsed=elapsed,
        current_time=current_time,
        step_index=step_index,
        accumulated=accumulated,
        history=history,
        history_times=history_times,
        state_shape=state_shape,
        cadence=cadence_time,
        burn=burn_time,
        accumulation=accumulation_time,
        system_kind="flow",
        system_id=system_id,
        tangent_method=tangent_method,
        backend="phydrax.solver",
        discretization_id=f"fixed_rk4_dt={step:.17g}",
        valid=valid,
        status=status,
    )


def _validate_integer_schedule(cadence: int, burn: int, accumulation: int) -> None:
    if cadence <= 0:
        raise ValueError("qr_interval must be positive.")
    if burn < 0:
        raise ValueError("burn_in must be non-negative.")
    if accumulation <= 0:
        raise ValueError("accumulation_interval must be positive.")
    if accumulation % cadence != 0:
        raise ValueError("accumulation_interval must be a multiple of qr_interval.")


def _time_to_steps(
    value: float,
    step: float,
    /,
    *,
    owner: str,
    allow_zero: bool = False,
) -> int:
    ratio = float(value) / step
    count = int(round(ratio))
    if (count == 0 and not allow_zero) or count < 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{owner} must be {qualifier}.")
    tolerance = 32.0 * jnp.finfo(float).eps * max(1.0, abs(ratio))
    if abs(ratio - count) > tolerance:
        raise ValueError(f"{owner} must be an integer multiple of step_size.")
    return count


def _initial_state(
    initial_state: ArrayLike | None,
    /,
    *,
    initial_basis: ArrayLike | None,
    leading_k: int | None,
    checkpoint: LyapunovSpectrumCheckpoint | None,
    system_kind: str,
    system_id: str,
    qr_interval: float,
    burn_in: float,
    accumulation_interval: float,
    tangent_method: str,
    backend: str,
    discretization_id: str,
):
    if not isinstance(system_id, str) or not system_id:
        raise ValueError("system_id must be a non-empty string.")
    if checkpoint is not None:
        if not isinstance(checkpoint, LyapunovSpectrumCheckpoint):
            raise TypeError("checkpoint must be a LyapunovSpectrumCheckpoint or None.")
        expected = (
            checkpoint.system_kind == system_kind
            and checkpoint.system_id == system_id
            and checkpoint.qr_interval == qr_interval
            and checkpoint.burn_in == burn_in
            and checkpoint.accumulation_interval == accumulation_interval
            and checkpoint.tangent_method == tangent_method
            and checkpoint.backend == backend
            and checkpoint.discretization_id == discretization_id
        )
        if not expected:
            raise ValueError("Checkpoint provenance is incompatible with this run.")
        if leading_k is not None and int(leading_k) != checkpoint.leading_k:
            raise ValueError("leading_k is incompatible with the checkpoint.")
        if initial_basis is not None:
            raise ValueError("initial_basis cannot be supplied when resuming.")
        if (
            initial_state is not None
            and tuple(jnp.shape(initial_state)) != checkpoint.state_shape
        ):
            raise ValueError("initial_state shape is incompatible with the checkpoint.")
        return (
            checkpoint.state,
            checkpoint.basis,
            checkpoint.log_stretch,
            checkpoint.accumulated_time,
            checkpoint.step_index,
            checkpoint.accumulated_intervals,
            checkpoint.valid,
            checkpoint.status,
        )
    if initial_state is None:
        raise ValueError("initial_state is required without a checkpoint.")
    state = jnp.asarray(initial_state)
    if state.size == 0:
        raise ValueError("initial_state must contain at least one scalar.")
    if not jnp.issubdtype(state.dtype, jnp.floating):
        raise TypeError("initial_state must have a real floating dtype.")
    dimension = int(state.size)
    rank = dimension if leading_k is None else int(leading_k)
    if rank <= 0 or rank > dimension:
        raise ValueError("leading_k must lie between one and the state dimension.")
    if initial_basis is None:
        basis_dtype = jnp.result_type(state, float)
        if rank == dimension:
            basis = jnp.eye(dimension, dtype=basis_dtype)
        else:
            # A fixed random-looking frame avoids coordinate-axis bias. The
            # evolved frame itself is retained in every checkpoint.
            basis, _ = _thin_qr(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (dimension, rank),
                    dtype=basis_dtype,
                )
            )
    else:
        basis = jnp.asarray(initial_basis)
        if basis.shape == tuple(state.shape) + (rank,):
            basis = basis.reshape((dimension, rank))
        if basis.shape != (dimension, rank):
            raise ValueError("initial_basis must have shape state_shape + (leading_k,).")
        basis, diagonal = _thin_qr(basis)
        if bool(jnp.any(diagonal == 0.0)):
            raise ValueError("initial_basis must have full column rank.")
    return (
        state,
        basis,
        jnp.zeros((rank,), dtype=jnp.result_type(state, float)),
        jnp.asarray(0.0),
        0,
        0,
        jnp.asarray(True),
        jnp.asarray(LYAPUNOV_SUCCESS, dtype=jnp.int32),
    )


def _thin_qr(matrix: Array, /) -> tuple[Array, Array]:
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    diagonal = jnp.abs(jnp.diag(r))
    signs = jnp.where(jnp.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :], diagonal


def _make_result(
    *,
    state: Array,
    basis: Array,
    logs: Array,
    elapsed: Array,
    current_time: float,
    step_index: int,
    accumulated: int,
    history: list[Array],
    history_times: list[Array],
    state_shape: tuple[int, ...],
    cadence: float,
    burn: float,
    accumulation: float,
    system_kind: str,
    system_id: str,
    tangent_method: str,
    backend: str,
    discretization_id: str,
    valid: Array,
    status: Array,
) -> LyapunovSpectrumResult:
    rank = int(logs.shape[0])
    result_valid = valid
    result_status = status
    if history:
        finite_history = jnp.stack(tuple(history), axis=0)
        times = jnp.stack(tuple(history_times), axis=0)
        exponents = finite_history[-1]
    else:
        finite_history = jnp.empty((0, rank), dtype=logs.dtype)
        times = jnp.empty((0,), dtype=float)
        exponents = jnp.full((rank,), jnp.nan, dtype=logs.dtype)
        result_valid = jnp.asarray(False)
        result_status = jnp.where(
            valid,
            jnp.asarray(LYAPUNOV_INSUFFICIENT_ACCUMULATION, dtype=jnp.int32),
            status,
        )
    if int(finite_history.shape[0]) >= 2:
        drift = jnp.max(jnp.abs(finite_history[-1] - finite_history[-2]))
    else:
        drift = jnp.asarray(jnp.nan, dtype=logs.dtype)
    full = rank == int(state.size)
    ky_valid = jnp.asarray(full) & result_valid & jnp.all(jnp.isfinite(exponents))
    ky_dimension = jnp.where(
        ky_valid,
        kaplan_yorke_dimension(exponents),
        jnp.asarray(jnp.nan, dtype=exponents.dtype),
    )
    checkpoint = LyapunovSpectrumCheckpoint(
        state=state,
        basis=basis,
        log_stretch=logs,
        accumulated_time=elapsed,
        current_time=current_time,
        step_index=step_index,
        accumulated_intervals=accumulated,
        state_shape=state_shape,
        leading_k=rank,
        qr_interval=cadence,
        burn_in=burn,
        accumulation_interval=accumulation,
        system_kind=system_kind,
        system_id=system_id,
        tangent_method=tangent_method,
        backend=backend,
        discretization_id=discretization_id,
        valid=valid,
        status=status,
    )
    return LyapunovSpectrumResult(
        exponents=exponents,
        finite_time_exponents=finite_history,
        accumulation_times=times,
        convergence_drift=drift,
        kaplan_yorke_dimension=ky_dimension,
        kaplan_yorke_valid=ky_valid,
        valid=result_valid,
        status=result_status,
        checkpoint=checkpoint,
    )


__all__ = [
    "LYAPUNOV_INSUFFICIENT_ACCUMULATION",
    "LYAPUNOV_NONFINITE_TANGENT",
    "LYAPUNOV_SINGULAR_TANGENT",
    "LYAPUNOV_SUCCESS",
    "LyapunovSpectrumCheckpoint",
    "LyapunovSpectrumMethod",
    "LyapunovSpectrumResult",
    "kaplan_yorke_dimension",
    "lyapunov_spectrum_flow",
    "lyapunov_spectrum_map",
]
