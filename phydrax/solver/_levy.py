#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil, isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic import (
    AbstractLevyProcess,
    LevyJumpSeries,
    LevyProcessRealization,
    StochasticTrajectory,
)
from ..stochastic._trajectory import _TrajectoryRecord
from ._solution_validation import validate_solution_arrays


LevySDEVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
LevySDEScheme: TypeAlias = Literal["euler", "tamed_euler"]
LevySmallJumpApproximation: TypeAlias = Literal["truncate", "gaussian"]


class _IdentityLevyDispersion(eqx.Module):
    dimension: int = eqx.field(static=True)

    def __call__(self, time, state, args):
        del time, args
        return jnp.eye(self.dimension, dtype=jnp.asarray(state).dtype)


class LevySDEProblem(StrictModule):
    """Itô SDE driven by one finite-dimensional Lévy process.

    The equation is ``dX = drift(t, X, args) dt + dispersion(t, X, args) dL``.
    The dispersion must return ``state_shape + (driver.dimension,)``. A missing
    dispersion denotes the identity and therefore requires a vector state matching
    the driver dimension.
    """

    drift: LevySDEVectorField
    dispersion: LevySDEVectorField
    initial_state: Array
    driver: AbstractLevyProcess
    t0: Array
    t1: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: LevySDEVectorField,
        initial_state: ArrayLike,
        driver: AbstractLevyProcess,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        dispersion: LevySDEVectorField | None = None,
        args: Any = None,
        problem_id: str | None = None,
    ):
        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not isinstance(driver, AbstractLevyProcess):
            raise TypeError("driver must implement AbstractLevyProcess.")
        state = jnp.asarray(initial_state)
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("initial_state must have a non-empty positive shape.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("t0 and t1 must be scalar.")
        if not bool(jnp.isfinite(start) & jnp.isfinite(end) & (end > start)):
            raise ValueError("LevySDEProblem requires finite t1 > t0.")
        if dispersion is None:
            if state_shape != (driver.dimension,):
                raise ValueError(
                    "Identity Lévy dispersion requires state shape (driver.dimension,)."
                )
            resolved_dispersion: LevySDEVectorField = _IdentityLevyDispersion(
                driver.dimension
            )
        else:
            if not callable(dispersion):
                raise TypeError("dispersion must be callable or None.")
            resolved_dispersion = dispersion
        drift_value = jnp.asarray(drift(start, state, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve initial_state shape.")
        dispersion_value = jnp.asarray(resolved_dispersion(start, state, args))
        expected_dispersion = state_shape + (driver.dimension,)
        if dispersion_value.shape != expected_dispersion:
            raise ValueError(
                f"dispersion must return shape {expected_dispersion}; "
                f"got {dispersion_value.shape}."
            )
        identifier = (
            f"{driver.process_id}:levy-sde" if problem_id is None else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.drift = drift
        self.dispersion = resolved_dispersion
        self.initial_state = state
        self.driver = driver
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_shape = state_shape
        self.problem_id = identifier


class LevySDESolverDiagnostics(StrictModule):
    """Cutoff, capacity, and small-jump diagnostics for a Lévy SDE solve."""

    complete_above_cutoff: Array
    num_large_jumps: Array
    smallest_radius: Array
    small_jump_covariance: Array
    cutoff: Array
    num_steps: int = eqx.field(static=True)
    scheme: LevySDEScheme = eqx.field(static=True)
    small_jump_approximation: LevySmallJumpApproximation = eqx.field(static=True)

    def __init__(
        self,
        *,
        complete_above_cutoff: ArrayLike,
        num_large_jumps: ArrayLike,
        smallest_radius: ArrayLike,
        small_jump_covariance: ArrayLike,
        cutoff: ArrayLike,
        num_steps: int,
        scheme: LevySDEScheme,
        small_jump_approximation: LevySmallJumpApproximation,
    ):
        complete = jnp.asarray(complete_above_cutoff, dtype=bool)
        counts = jnp.asarray(num_large_jumps, dtype=jnp.int32)
        radii = jnp.asarray(smallest_radius, dtype=float)
        covariance = jnp.asarray(small_jump_covariance, dtype=float)
        threshold = jnp.asarray(cutoff, dtype=float)
        if counts.shape != complete.shape or radii.shape != complete.shape:
            raise ValueError("Per-path Lévy diagnostics must have matching shapes.")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("small_jump_covariance must be square.")
        if threshold.shape != () or not bool(jnp.isfinite(threshold) & (threshold > 0.0)):
            raise ValueError("cutoff must be a finite positive scalar.")
        steps = int(num_steps)
        if steps < 0:
            raise ValueError("num_steps must be non-negative.")
        if scheme not in ("euler", "tamed_euler"):
            raise ValueError("Unknown Lévy SDE scheme.")
        if small_jump_approximation not in ("truncate", "gaussian"):
            raise ValueError("Unknown small-jump approximation.")
        self.complete_above_cutoff = complete
        self.num_large_jumps = counts
        self.smallest_radius = radii
        self.small_jump_covariance = covariance
        self.cutoff = threshold
        self.num_steps = steps
        self.scheme = scheme
        self.small_jump_approximation = small_jump_approximation

    @property
    def capacity_sufficient(self) -> Array:
        return self.complete_above_cutoff


class LevySDESolution(StrictModule):
    """Fixed-step Lévy SDE trajectory with series and approximation provenance."""

    times: Array
    states: Array
    valid: Array
    realization: LevyProcessRealization
    series: LevyJumpSeries
    diagnostics: LevySDESolverDiagnostics
    metadata: frozendict[str, Any]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        realization: LevyProcessRealization,
        series: LevyJumpSeries,
        diagnostics: LevySDESolverDiagnostics,
        state_shape: Sequence[int],
        solver_name: str,
        approximation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(realization, LevyProcessRealization):
            raise TypeError("realization must be a LevyProcessRealization.")
        if not isinstance(series, LevyJumpSeries):
            raise TypeError("series must be a LevyJumpSeries.")
        if not isinstance(diagnostics, LevySDESolverDiagnostics):
            raise TypeError("diagnostics must be LevySDESolverDiagnostics.")
        arrays = validate_solution_arrays(
            times,
            states,
            valid,
            sample_shape=realization.sample_shape,
            state_shape=state_shape,
            time_layout="shared",
            owner="LevySDESolution",
        )
        time_values = arrays.times
        state_values = arrays.states
        valid_values = arrays.valid
        shape = arrays.state_shape
        if series.realization_id != realization.realization_id:
            raise ValueError("series and realization identities must match.")
        if not isinstance(solver_name, str) or not solver_name:
            raise ValueError("solver_name must be non-empty.")
        if not isinstance(approximation_id, str) or not approximation_id:
            raise ValueError("approximation_id must be non-empty.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.realization = realization
        self.series = series
        self.diagnostics = diagnostics
        self.metadata = frozendict({} if metadata is None else metadata)
        self.sample_shape = realization.sample_shape
        self.state_shape = shape
        self.solver_name = solver_name
        self.approximation_id = approximation_id

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1)

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        resolved_realization_axes = (
            tuple(f"path_{index}" for index in range(len(self.sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        resolved_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        record = _TrajectoryRecord(
            self.times,
            self.states,
            state_shape=self.state_shape,
            realization_shape=self.sample_shape,
            valid=self.valid,
            realizations=(self.realization,),
            approximation_id=self.approximation_id,
            solver_name=self.solver_name,
            uncertainty_source="process",
            metadata={
                **dict(self.metadata),
                "driver_process_id": self.realization.process_id,
                "small_jump_approximation": (
                    self.diagnostics.small_jump_approximation
                ),
            },
        )
        return record.to_stochastic_trajectory(
            realization_axes=resolved_realization_axes,
            state_axes=resolved_state_axes,
        )


def _step_schedule(
    start: float,
    end: float,
    save_times: ArrayLike,
    max_step: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    saved = np.asarray(save_times, dtype=float)
    if saved.ndim != 1 or saved.size <= 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    if np.any(~np.isfinite(saved)) or np.any(np.diff(saved) <= 0.0):
        raise ValueError("save_times must be finite and strictly increasing.")
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, abs(start), abs(end))
    if float(saved[0]) < start - tolerance or float(saved[-1]) > end + tolerance:
        raise ValueError("save_times must lie in the problem interval.")
    current = start
    boundaries = [start]
    save_indices: list[int] = []
    for target in saved:
        target_value = float(target)
        interval = target_value - current
        if interval < -tolerance:
            raise ValueError("save_times cannot precede the current solve time.")
        if abs(interval) <= tolerance:
            save_indices.append(len(boundaries) - 1)
            current = target_value
            continue
        count = max(1, int(ceil(interval / max_step)))
        step = interval / count
        boundaries.extend(current + step * index for index in range(1, count + 1))
        boundaries[-1] = target_value
        current = target_value
        save_indices.append(len(boundaries) - 1)
    boundary_values = jnp.asarray(boundaries, dtype=float)
    return (
        boundary_values[:-1],
        boundary_values[1:],
        jnp.asarray(save_indices, dtype=jnp.int32),
        jnp.asarray(saved, dtype=float),
    )


def _covariance_factor(covariance: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    tolerance = 100.0 * jnp.finfo(covariance.dtype).eps
    eigenvalues = jnp.where(
        (eigenvalues < 0.0) & (eigenvalues > -tolerance),
        0.0,
        eigenvalues,
    )
    eigenvalues = eqx.error_if(
        eigenvalues,
        jnp.any(eigenvalues < 0.0),
        "small-jump covariance must be positive semidefinite.",
    )
    return eigenvectors * jnp.sqrt(eigenvalues)[None, :]


def solve_levy_sde(
    problem: LevySDEProblem,
    realization: LevyProcessRealization,
    /,
    *,
    save_times: ArrayLike,
    dt: float,
    cutoff: float,
    scheme: LevySDEScheme = "euler",
    small_jumps: LevySmallJumpApproximation = "truncate",
    throw: bool = True,
) -> LevySDESolution:
    """Solve an infinite-activity Lévy SDE by coupled series truncation.

    Large jumps above ``cutoff`` are taken from the reusable decreasing Poisson
    series. ``small_jumps="gaussian"`` adds a coupled covariance-matched Wiener
    closure for omitted jumps. ``tamed_euler`` bounds only the deterministic drift
    increment and leaves the Itô jump map unchanged.
    """
    if not isinstance(problem, LevySDEProblem):
        raise TypeError("problem must be a LevySDEProblem.")
    if not isinstance(realization, LevyProcessRealization):
        raise TypeError("realization must be a LevyProcessRealization.")
    if realization.process_id != problem.driver.process_id:
        raise ValueError("Problem driver and realization process_id values must match.")
    if realization.dimension != problem.driver.dimension:
        raise ValueError("Problem driver and realization dimensions must match.")
    if realization.support[0] > float(problem.t0) or realization.support[1] < float(
        problem.t1
    ):
        raise ValueError("Lévy realization support must cover the problem interval.")
    step_limit = float(dt)
    threshold = float(cutoff)
    if not isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("dt must be finite and positive.")
    if not isfinite(threshold) or threshold <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    if scheme not in ("euler", "tamed_euler"):
        raise ValueError("scheme must be 'euler' or 'tamed_euler'.")
    if small_jumps not in ("truncate", "gaussian"):
        raise ValueError("small_jumps must be 'truncate' or 'gaussian'.")
    if not isinstance(throw, bool):
        raise TypeError("throw must be a bool.")

    starts, ends, save_indices, saved = _step_schedule(
        float(problem.t0),
        float(problem.t1),
        save_times,
        step_limit,
    )
    steps = ends - starts
    num_steps = int(steps.size)
    series = realization.series(problem.driver)
    complete = series.complete_above(threshold)
    if throw and not bool(jnp.all(complete)):
        failed = int(jnp.sum(~complete))
        raise RuntimeError(
            f"Lévy series capacity misses jumps above cutoff on {failed} path(s); "
            "extend the realization or increase cutoff."
        )
    jump_increments = series.increments(starts, ends, cutoff=threshold)
    deterministic_rate = problem.driver.drift + problem.driver.truncation_drift(threshold)
    duration_shape = (1,) * len(realization.sample_shape) + steps.shape + (1,)
    rate_shape = (1,) * (len(realization.sample_shape) + 1) + (problem.driver.dimension,)
    driver_increments = jump_increments + steps.reshape(
        duration_shape
    ) * deterministic_rate.reshape(rate_shape)
    small_covariance = problem.driver.small_jump_covariance(threshold)
    if small_jumps == "gaussian":
        gaussian = realization.gaussian_realization().increments(
            starts,
            ends,
            dtype=problem.initial_state.real.dtype,
        )
        driver_increments = driver_increments + jnp.einsum(
            "ij,...j->...i",
            _covariance_factor(small_covariance),
            gaussian,
        )

    def one_path(increments):
        def advance(state, item):
            time, step, driver_increment = item
            drift = jnp.asarray(problem.drift(time, state, problem.args))
            if drift.shape != problem.state_shape:
                raise ValueError("drift must preserve the declared state shape.")
            if scheme == "tamed_euler":
                norm = jnp.linalg.norm(drift.reshape((-1,)))
                drift_update = step * drift / (1.0 + step * norm)
            else:
                drift_update = step * drift
            dispersion = jnp.asarray(problem.dispersion(time, state, problem.args))
            expected = problem.state_shape + (problem.driver.dimension,)
            if dispersion.shape != expected:
                raise ValueError(
                    f"dispersion must return shape {expected}; got {dispersion.shape}."
                )
            jump_update = jnp.tensordot(
                dispersion,
                driver_increment,
                axes=((-1,), (0,)),
            )
            next_state = state + drift_update + jump_update
            return next_state, next_state

        _, stepped = jax.lax.scan(
            advance,
            problem.initial_state,
            (starts, steps, increments),
        )
        complete_states = jnp.concatenate(
            (problem.initial_state[None, ...], stepped),
            axis=0,
        )
        return complete_states[save_indices]

    if realization.sample_shape:
        flat_increments = driver_increments.reshape(
            (realization.num_paths, num_steps, problem.driver.dimension)
        )
        states = jax.vmap(one_path)(flat_increments).reshape(
            realization.sample_shape + (int(saved.size),) + problem.state_shape
        )
    else:
        states = one_path(driver_increments)
    state_axes = tuple(range(len(realization.sample_shape) + 1, states.ndim))
    finite = jnp.all(jnp.isfinite(states), axis=state_axes)
    path_complete = complete.reshape(realization.sample_shape)
    valid = finite & path_complete[..., None]
    diagnostics = LevySDESolverDiagnostics(
        complete_above_cutoff=complete,
        num_large_jumps=series.num_jumps_above(threshold),
        smallest_radius=series.smallest_radius,
        small_jump_covariance=small_covariance,
        cutoff=threshold,
        num_steps=num_steps,
        scheme=scheme,
        small_jump_approximation=small_jumps,
    )
    approximation_id = (
        f"{problem.problem_id}:{realization.coupling_id}:"
        f"{scheme}:{small_jumps}:dt={step_limit:.17g}:cutoff={threshold:.17g}:"
        f"terms={realization.max_terms}"
    )
    return LevySDESolution(
        times=saved,
        states=states,
        valid=valid,
        realization=realization,
        series=series,
        diagnostics=diagnostics,
        state_shape=problem.state_shape,
        solver_name=("LevyEuler" if scheme == "euler" else "LevyTamedEuler"),
        approximation_id=approximation_id,
        metadata={
            "problem_id": problem.problem_id,
            "driver_process_id": problem.driver.process_id,
            "dt": step_limit,
            "cutoff": threshold,
        },
    )


__all__ = [
    "LevySDEProblem",
    "LevySDEScheme",
    "LevySDESolution",
    "LevySDESolverDiagnostics",
    "LevySDEVectorField",
    "LevySmallJumpApproximation",
    "solve_levy_sde",
]
