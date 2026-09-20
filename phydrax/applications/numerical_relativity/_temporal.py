#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._ssp_runge_kutta import (
    ssprk33_step_with_evidence,
    ssprk54_step_with_evidence,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._boundaries import (
    AbstractZ4cBoundary,
    PeriodicBoundary,
    Z4cBoundaryEvidence,
)
from ._derivatives import FourthOrderDerivatives
from ._enforcement import (
    Z4cAlgebraicEnforcement,
    Z4cEnforcementEvidence,
)
from ._gauge import AbstractZ4cGauge
from ._grid import FixedGridGeometry
from ._state import Z4cState
from ._status import NumericalRelativityStatus
from ._z4c import (
    evaluate_z4c_rhs,
    z4c_adm_geometry,
    z4c_snapshot_token,
    Z4cConstraintEvidence,
    Z4cSystem,
)


Z4cIntegrator: TypeAlias = Literal["ssprk33", "ssprk54"]
StressEnergyProvider: TypeAlias = Callable[
    [Array, ADMGridGeometry], StressEnergyProjection
]
_SSPRK33_RHS_TIME_FRACTIONS = (0.0, 1.0, 0.5)
_SSPRK54_RHS_TIME_FRACTIONS = (
    0.0,
    0.391752226571890,
    0.586079689311540,
    0.474542363026870,
    0.935010631009240,
)


def _stage_snapshot_token(
    runtime: FixedGridZ4cRuntime,
    state: Z4cRuntimeState,
    stage_time: Array,
    /,
) -> Array:
    fractions = (
        _SSPRK33_RHS_TIME_FRACTIONS
        if runtime.integrator == "ssprk33"
        else _SSPRK54_RHS_TIME_FRACTIONS
    )
    normalized = (stage_time - state.time) / jnp.asarray(
        runtime.time_step, dtype=state.time.dtype
    )
    distances = jnp.abs(normalized - jnp.asarray(fractions, dtype=state.time.dtype))
    stage_slot = jnp.argmin(distances).astype(jnp.int32) + 1
    return z4c_snapshot_token(state.step_index, stage_slot)


def _stage_stress_energy(
    provider: StressEnergyProvider | None,
    stage_time: Array,
    geometry: ADMGridGeometry,
    /,
) -> StressEnergyProjection | None:
    if provider is None:
        return None
    result = provider(stage_time, geometry)
    if not isinstance(result, StressEnergyProjection):
        raise TypeError("stress_energy_provider must return StressEnergyProjection.")
    return result


class Z4cRuntimeState(StrictModule):
    """One committed spacetime state with fixed-grid logical time."""

    state: Z4cState
    time: Array
    step_index: Array
    runtime_id: str = eqx.field(static=True)


class Z4cStepResult(StrictModule):
    """Uncommitted candidate and fail-closed internally accepted proposal."""

    source: Z4cRuntimeState
    candidate: Z4cRuntimeState
    accepted: Z4cRuntimeState
    constraints: Z4cConstraintEvidence
    boundary: Z4cBoundaryEvidence
    enforcement: Z4cEnforcementEvidence
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class FixedGridZ4cRuntime(StrictModule, NonTrainableState):
    """Prepared method-of-lines Z4c runtime with no dynamic topology changes."""

    system: Z4cSystem
    grid: FixedGridGeometry
    derivatives: FourthOrderDerivatives
    gauge: AbstractZ4cGauge
    boundary: AbstractZ4cBoundary
    enforcement: Z4cAlgebraicEnforcement
    time_step: float = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    integrator: Z4cIntegrator = eqx.field(static=True)
    courant_number: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Z4cSystem,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        gauge: AbstractZ4cGauge,
        boundary: AbstractZ4cBoundary,
        enforcement: Z4cAlgebraicEnforcement,
        /,
        *,
        time_step: float,
        start_time: float = 0.0,
        integrator: Z4cIntegrator = "ssprk54",
        maximum_courant_number: float = 0.25,
    ):
        if not isinstance(system, Z4cSystem):
            raise TypeError("system must be a Z4cSystem.")
        if not isinstance(grid, FixedGridGeometry):
            raise TypeError("grid must be a FixedGridGeometry.")
        if not isinstance(derivatives, FourthOrderDerivatives):
            raise TypeError("derivatives must be FourthOrderDerivatives.")
        if not isinstance(gauge, AbstractZ4cGauge):
            raise TypeError("gauge must implement AbstractZ4cGauge.")
        if not isinstance(boundary, AbstractZ4cBoundary):
            raise TypeError("boundary must implement AbstractZ4cBoundary.")
        if not isinstance(enforcement, Z4cAlgebraicEnforcement):
            raise TypeError("enforcement must be Z4cAlgebraicEnforcement.")
        step = float(time_step)
        start = float(start_time)
        maximum_courant = float(maximum_courant_number)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if not isfinite(start):
            raise ValueError("start_time must be finite.")
        if not isfinite(maximum_courant) or maximum_courant <= 0.0:
            raise ValueError("maximum_courant_number must be finite and positive.")
        if integrator not in ("ssprk33", "ssprk54"):
            raise ValueError("integrator must be 'ssprk33' or 'ssprk54'.")
        if derivatives.grid_shape != grid.shape or derivatives.spacing != grid.spacing:
            raise ValueError("derivatives must be prepared for the exact fixed grid.")
        if grid.periodic != (derivatives.boundary == "periodic"):
            raise ValueError("grid topology and derivative boundary mode disagree.")
        if grid.periodic != isinstance(boundary, PeriodicBoundary):
            raise ValueError("Periodic grids require PeriodicBoundary and conversely.")
        courant = step / min(grid.spacing)
        if courant > maximum_courant:
            raise ValueError("time_step exceeds the declared fixed-grid Courant limit.")
        self.system = system
        self.grid = grid
        self.derivatives = derivatives
        self.gauge = gauge
        self.boundary = boundary
        self.enforcement = enforcement
        self.time_step = step
        self.start_time = start
        self.integrator = integrator
        self.courant_number = courant
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-z4c-method-of-lines",
                "system": system.system_id,
                "grid": grid.grid_id,
                "derivatives": derivatives.derivative_id,
                "gauge": gauge.gauge_id,
                "boundary": boundary.boundary_id,
                "enforcement": enforcement.enforcement_id,
                "time_step": step,
                "start_time": start,
                "integrator": integrator,
                "maximum_courant_number": maximum_courant,
            }
        )

    def initialize(
        self,
        state: Z4cState,
        /,
        *,
        time: float | None = None,
        step_index: int = 0,
    ) -> Z4cRuntimeState:
        return initialize_z4c_runtime_state(self, state, time=time, step_index=step_index)

    def evaluate(
        self,
        state: Z4cRuntimeState,
        /,
        *,
        stress_energy_provider: StressEnergyProvider | None = None,
    ) -> Z4cStepResult:
        return evaluate_z4c_step(
            self,
            state,
            stress_energy_provider=stress_energy_provider,
        )

    def accept(
        self,
        result: Z4cStepResult,
        accept: ArrayLike = True,
        /,
    ) -> Z4cRuntimeState:
        return accept_z4c_step(self, result, accept)


def initialize_z4c_runtime_state(
    runtime: FixedGridZ4cRuntime,
    state: Z4cState,
    /,
    *,
    time: float | None = None,
    step_index: int = 0,
) -> Z4cRuntimeState:
    if not isinstance(runtime, FixedGridZ4cRuntime):
        raise TypeError("runtime must be a FixedGridZ4cRuntime.")
    if not isinstance(state, Z4cState) or state.grid_id != runtime.grid.grid_id:
        raise ValueError("state must belong to the runtime grid.")
    index = int(step_index)
    if index < 0:
        raise ValueError("step_index must be non-negative.")
    expected_time = runtime.start_time + index * runtime.time_step
    time_ = expected_time if time is None else float(time)
    tolerance = 64.0 * jnp.finfo(state.values.dtype).eps * max(abs(expected_time), 1.0)
    if not isfinite(time_) or abs(time_ - expected_time) > tolerance:
        raise ValueError("time must match start_time + step_index*time_step.")
    bounded = runtime.boundary.apply_state(
        jnp.asarray(time_, dtype=state.values.dtype), state, runtime.grid
    )
    if not bool(bounded.evidence.successful):
        raise ValueError("Initial boundary application failed.")
    enforced = runtime.enforcement.apply(bounded.state)
    if not bool(enforced.evidence.successful):
        raise ValueError("Initial algebraic enforcement failed.")
    return Z4cRuntimeState(
        enforced.state,
        jnp.asarray(time_, dtype=state.values.dtype),
        jnp.asarray(index, dtype=jnp.int32),
        runtime.runtime_id,
    )


def _time_consistent(runtime: FixedGridZ4cRuntime, state: Z4cRuntimeState, /) -> Array:
    expected = (
        runtime.start_time + state.step_index.astype(state.time.dtype) * runtime.time_step
    )
    tolerance = (
        64.0 * jnp.finfo(state.time.dtype).eps * jnp.maximum(jnp.abs(expected), 1.0)
    )
    return jnp.isfinite(state.time) & (jnp.abs(state.time - expected) <= tolerance)


def evaluate_z4c_step(
    runtime: FixedGridZ4cRuntime,
    state: Z4cRuntimeState,
    /,
    *,
    stress_energy_provider: StressEnergyProvider | None = None,
) -> Z4cStepResult:
    """Propose one fixed-grid SSPRK step; no caller state is committed here."""

    if not isinstance(runtime, FixedGridZ4cRuntime):
        raise TypeError("runtime must be a FixedGridZ4cRuntime.")
    if not isinstance(state, Z4cRuntimeState):
        raise TypeError("state must be a Z4cRuntimeState.")
    if state.runtime_id != runtime.runtime_id:
        raise ValueError("state does not belong to this runtime.")
    if state.state.grid_id != runtime.grid.grid_id:
        raise ValueError("runtime state has an incompatible grid identity.")
    if stress_energy_provider is not None and not callable(stress_energy_provider):
        raise TypeError("stress_energy_provider must be callable or None.")
    time_consistent = _time_consistent(runtime, state)
    stage_finite: list[Array] = []
    stage_source_valid: list[Array] = []
    stage_boundary_valid: list[Array] = []

    def vector_field(time, values, args):
        del args
        stage_state = Z4cState(values, grid_id=runtime.grid.grid_id)
        bounded = runtime.boundary.apply_state(time, stage_state, runtime.grid)
        snapshot_token = _stage_snapshot_token(runtime, state, time)
        geometry = z4c_adm_geometry(
            runtime.system,
            runtime.grid,
            bounded.state,
            snapshot_token=snapshot_token,
        )
        stress_energy = _stage_stress_energy(stress_energy_provider, time, geometry)
        evaluation = evaluate_z4c_rhs(
            runtime.system,
            runtime.grid,
            runtime.derivatives,
            runtime.gauge,
            bounded.state,
            snapshot_token=snapshot_token,
            stress_energy=stress_energy,
        )
        boundary_rates = runtime.boundary.apply_rates(
            time,
            bounded.state,
            evaluation.rates,
            runtime.grid,
            runtime.derivatives,
        )
        stage_finite.append(evaluation.finite & boundary_rates.evidence.finite)
        stage_source_valid.append(evaluation.source_valid)
        stage_boundary_valid.append(
            bounded.evidence.successful & boundary_rates.evidence.successful
        )
        return boundary_rates.state.values

    step_size = jnp.asarray(runtime.time_step, dtype=state.time.dtype)
    if runtime.integrator == "ssprk33":
        step = ssprk33_step_with_evidence(
            vector_field, state.time, state.state.values, step_size
        )
    else:
        step = ssprk54_step_with_evidence(
            vector_field, state.time, state.state.values, step_size
        )
    candidate_time = state.time + step_size
    raw_candidate = Z4cState(step.state, grid_id=runtime.grid.grid_id)
    bounded_candidate = runtime.boundary.apply_state(
        candidate_time, raw_candidate, runtime.grid
    )
    enforcement = runtime.enforcement.apply(bounded_candidate.state)
    final_snapshot_token = z4c_snapshot_token(state.step_index, 7)
    final_geometry = z4c_adm_geometry(
        runtime.system,
        runtime.grid,
        enforcement.state,
        snapshot_token=final_snapshot_token,
    )
    final_stress_energy = _stage_stress_energy(
        stress_energy_provider, candidate_time, final_geometry
    )
    final_evaluation = evaluate_z4c_rhs(
        runtime.system,
        runtime.grid,
        runtime.derivatives,
        runtime.gauge,
        enforcement.state,
        snapshot_token=final_snapshot_token,
        stress_energy=final_stress_energy,
    )
    final_boundary = runtime.boundary.apply_rates(
        candidate_time,
        enforcement.state,
        final_evaluation.rates,
        runtime.grid,
        runtime.derivatives,
    ).evidence
    candidate = Z4cRuntimeState(
        enforcement.state,
        candidate_time,
        state.step_index + jnp.asarray(1, dtype=jnp.int32),
        runtime.runtime_id,
    )
    all_stage_finite = jnp.all(jnp.stack(tuple(stage_finite)))
    all_stage_source_valid = jnp.all(jnp.stack(tuple(stage_source_valid)))
    all_stage_boundary_valid = jnp.all(jnp.stack(tuple(stage_boundary_valid)))
    finite = (
        jnp.all(jnp.isfinite(step.state))
        & all_stage_finite
        & final_evaluation.finite
        & bounded_candidate.evidence.finite
        & enforcement.evidence.finite
        & final_boundary.finite
    )
    converged = step.successful
    physically_valid = final_evaluation.physically_valid
    derivative_valid = final_evaluation.derivative_valid
    qualified = (
        final_evaluation.constraints.qualified
        & all_stage_source_valid
        & all_stage_boundary_valid
        & bounded_candidate.evidence.successful
        & final_boundary.successful
        & enforcement.evidence.successful
        & time_consistent
    )
    status = jnp.asarray(int(NumericalRelativityStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.NONFINITE_STATE)),
    )
    status = jnp.where(
        jnp.all(candidate.state.lapse > 0.0),
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.NONPOSITIVE_LAPSE)),
    )
    status = jnp.where(
        jnp.all(candidate.state.chi > 0.0),
        status,
        jnp.bitwise_or(
            status, int(NumericalRelativityStatus.NONPOSITIVE_CONFORMAL_FACTOR)
        ),
    )
    status = jnp.where(
        physically_valid,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.SINGULAR_CONFORMAL_METRIC)),
    )
    status = jnp.where(
        final_evaluation.constraints.within_tolerance,
        status,
        jnp.bitwise_or(
            status, int(NumericalRelativityStatus.CONSTRAINT_TOLERANCE_EXCEEDED)
        ),
    )
    boundary_successful = (
        bounded_candidate.evidence.successful
        & final_boundary.successful
        & all_stage_boundary_valid
    )
    status = jnp.where(
        boundary_successful,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.BOUNDARY_FAILURE)),
    )
    status = jnp.where(
        enforcement.evidence.successful,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.ENFORCEMENT_FAILURE)),
    )
    status = jnp.where(
        derivative_valid,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.DERIVATIVE_INVALID)),
    )
    status = jnp.where(
        time_consistent,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.TIME_GRID_MISMATCH)),
    )
    status = jnp.where(
        final_evaluation.source_valid & all_stage_source_valid,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.SOURCE_INVALID)),
    )
    successful = (
        (status == int(NumericalRelativityStatus.SUCCESS)) & converged & qualified
    )
    status = jnp.where(
        successful,
        status,
        jnp.bitwise_or(status, int(NumericalRelativityStatus.STEP_REJECTED)),
    )
    accepted = Z4cRuntimeState(
        state.state.with_values(
            jnp.where(successful, candidate.state.values, state.state.values)
        ),
        jnp.where(successful, candidate.time, state.time),
        jnp.where(successful, candidate.step_index, state.step_index),
        runtime.runtime_id,
    )
    return Z4cStepResult(
        state,
        candidate,
        accepted,
        final_evaluation.constraints,
        final_boundary,
        enforcement.evidence,
        status,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        successful,
        runtime.runtime_id,
    )


def accept_z4c_step(
    runtime: FixedGridZ4cRuntime,
    result: Z4cStepResult,
    accept: ArrayLike = True,
    /,
) -> Z4cRuntimeState:
    """Atomically commit or roll back an evaluated proposal with a scalar mask."""

    if not isinstance(runtime, FixedGridZ4cRuntime):
        raise TypeError("runtime must be a FixedGridZ4cRuntime.")
    if not isinstance(result, Z4cStepResult) or result.runtime_id != runtime.runtime_id:
        raise ValueError("result does not belong to this runtime.")
    mask = jnp.asarray(accept, dtype=jnp.bool_)
    if mask.shape != ():
        raise ValueError("accept must be scalar.")
    commit = mask & result.successful
    return Z4cRuntimeState(
        result.source.state.with_values(
            jnp.where(
                commit,
                result.accepted.state.values,
                result.source.state.values,
            )
        ),
        jnp.where(commit, result.accepted.time, result.source.time),
        jnp.where(commit, result.accepted.step_index, result.source.step_index),
        runtime.runtime_id,
    )


__all__ = [
    "FixedGridZ4cRuntime",
    "StressEnergyProvider",
    "Z4cIntegrator",
    "Z4cRuntimeState",
    "Z4cStepResult",
    "accept_z4c_step",
    "evaluate_z4c_step",
    "initialize_z4c_runtime_state",
]
