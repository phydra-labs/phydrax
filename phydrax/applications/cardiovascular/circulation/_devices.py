#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cardiovascular pumps, controllers, hydraulic lines, and ECMO circuits."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite, pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....dynamics import (
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)
from ._components import PressureFlowComponent
from ._oxygen import exchange_membrane_oxygen, MembraneOxygenatorModel


class PumpMapStatus(IntFlag):
    """Fail-closed status bits for pump-map evaluation."""

    SUCCESS = 0
    NONFINITE = 1
    FLOW_OUT_OF_DOMAIN = 2
    SPEED_OUT_OF_DOMAIN = 4


class ControllerStatus(IntFlag):
    """Status bits shared by causal pacemaker and pump controllers."""

    SUCCESS = 0
    NONFINITE = 1
    NONCAUSAL_SAMPLE = 2
    SAMPLE_PERIOD_MISMATCH = 4


class ControlEvent(IntFlag):
    """Fixed event mask emitted by controllers."""

    NONE = 0
    SENSED = 1
    PACED = 2
    PULSE_ACTIVE = 4
    SPEED_LOWER_LIMIT = 8
    SPEED_UPPER_LIMIT = 16
    SLEW_LIMIT = 32
    INTEGRAL_LOWER_LIMIT = 64
    INTEGRAL_UPPER_LIMIT = 128


class ECMOHydraulicStatus(IntFlag):
    """Fail-closed ECMO operating-point status."""

    SUCCESS = 0
    NONFINITE = 1
    PUMP_MAP_REFUSAL = 2
    NO_OPERATING_POINT = 4
    RESIDUAL_FAILURE = 8
    OXYGEN_INPUT_INVALID = 16
    OXYGEN_EXCHANGE_FAILURE = 32


class HydraulicDeviceComponent(PressureFlowComponent):
    """Concrete canonical DAE wrapper for advanced hydraulic devices."""

    def __init__(
        self,
        dae_component: DAEComponent,
        /,
        *,
        component_kind: str,
        parameters: tuple[tuple[str, float | str], ...],
    ):
        PressureFlowComponent.__init__(
            self,
            dae_component,
            component_kind=component_kind,
            parameters=parameters,
        )


class PumpHeadFlowMap(StrictModule, NonTrainableState):
    """Validated rectangular head–flow–speed map with refusal outside its domain."""

    flow_axis_mm3_per_ms: Array
    speed_axis_rpm: Array
    head_kPa: Array
    map_name: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        map_name: str,
        flow_axis_mm3_per_ms: ArrayLike,
        speed_axis_rpm: ArrayLike,
        head_kPa: ArrayLike,
        /,
    ):
        name = str(map_name).strip()
        flow_host = np.asarray(flow_axis_mm3_per_ms, dtype=float)
        speed_host = np.asarray(speed_axis_rpm, dtype=float)
        head_host = np.asarray(head_kPa, dtype=float)
        if not name:
            raise ValueError("map_name must be non-empty.")
        if flow_host.ndim != 1 or speed_host.ndim != 1:
            raise ValueError("Pump-map axes must be one-dimensional.")
        if flow_host.size < 2 or speed_host.size < 2:
            raise ValueError("Pump-map axes need at least two points.")
        if head_host.shape != (speed_host.size, flow_host.size):
            raise ValueError("head_kPa must have shape [speed, flow].")
        if not (
            np.all(np.isfinite(flow_host))
            and np.all(np.isfinite(speed_host))
            and np.all(np.isfinite(head_host))
        ):
            raise ValueError("Pump-map data must be finite.")
        if (
            np.any(flow_host < 0.0)
            or np.any(speed_host <= 0.0)
            or np.any(head_host < 0.0)
            or np.any(np.diff(flow_host) <= 0.0)
            or np.any(np.diff(speed_host) <= 0.0)
        ):
            raise ValueError("Pump-map axes and heads are outside their domains.")
        if np.any(np.diff(head_host, axis=1) > 0.0):
            raise ValueError("Pump head must be nonincreasing with flow at each speed.")
        if np.any(np.diff(head_host, axis=0) < 0.0):
            raise ValueError("Pump head must be nondecreasing with speed at each flow.")
        self.flow_axis_mm3_per_ms = jnp.asarray(flow_host)
        self.speed_axis_rpm = jnp.asarray(speed_host)
        self.head_kPa = jnp.asarray(head_host)
        self.map_name = name
        self.map_id = canonical_fingerprint(
            {
                "kind": "pump-head-flow-speed-map-v1",
                "map_name": name,
                "flow_axis_mm3_per_ms": flow_host.tolist(),
                "speed_axis_rpm": speed_host.tolist(),
                "head_kPa": head_host.tolist(),
            }
        )


class PumpMapResult(StrictModule):
    """Interpolated head and pump-map domain evidence."""

    head_kPa: Array
    status: Array
    successful: Array


def evaluate_pump_map(
    pump_map: PumpHeadFlowMap,
    flow_mm3_per_ms: ArrayLike,
    speed_rpm: ArrayLike,
    /,
) -> PumpMapResult:
    """Bilinearly evaluate a map, returning NaN rather than extrapolating."""
    flow = jnp.asarray(flow_mm3_per_ms, dtype=pump_map.head_kPa.dtype)
    speed = jnp.asarray(speed_rpm, dtype=pump_map.head_kPa.dtype)
    if flow.shape != () or speed.shape != ():
        raise ValueError("Pump-map flow and speed must be scalars.")
    finite = jnp.isfinite(flow) & jnp.isfinite(speed)
    flow_valid = (flow >= pump_map.flow_axis_mm3_per_ms[0]) & (
        flow <= pump_map.flow_axis_mm3_per_ms[-1]
    )
    speed_valid = (speed >= pump_map.speed_axis_rpm[0]) & (
        speed <= pump_map.speed_axis_rpm[-1]
    )
    row_heads = jax.vmap(
        lambda row: jnp.interp(flow, pump_map.flow_axis_mm3_per_ms, row)
    )(pump_map.head_kPa)
    head = jnp.interp(speed, pump_map.speed_axis_rpm, row_heads)
    successful = finite & flow_valid & speed_valid
    status = jnp.asarray(int(PumpMapStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(PumpMapStatus.NONFINITE)),
    )
    status = jnp.where(
        flow_valid,
        status,
        jnp.bitwise_or(status, int(PumpMapStatus.FLOW_OUT_OF_DOMAIN)),
    )
    status = jnp.where(
        speed_valid,
        status,
        jnp.bitwise_or(status, int(PumpMapStatus.SPEED_OUT_OF_DOMAIN)),
    )
    return PumpMapResult(
        head_kPa=jnp.where(successful, head, jnp.asarray(jnp.nan, dtype=head.dtype)),
        status=status,
        successful=successful,
    )


class PacemakerControllerPlan(StrictModule, NonTrainableState):
    """Causal inhibited pacemaker with fixed sensing and pulse limits."""

    lower_rate_bpm: float = eqx.field(static=True)
    upper_rate_bpm: float = eqx.field(static=True)
    refractory_period_ms: float = eqx.field(static=True)
    pulse_width_ms: float = eqx.field(static=True)
    pulse_amplitude_mA: float = eqx.field(static=True)
    escape_interval_ms: float = eqx.field(static=True)
    minimum_interval_ms: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower_rate_bpm: float,
        upper_rate_bpm: float,
        refractory_period_ms: float,
        pulse_width_ms: float,
        pulse_amplitude_mA: float,
        /,
    ):
        lower = float(lower_rate_bpm)
        upper = float(upper_rate_bpm)
        refractory = float(refractory_period_ms)
        width = float(pulse_width_ms)
        amplitude = float(pulse_amplitude_mA)
        if not all(
            isfinite(value) for value in (lower, upper, refractory, width, amplitude)
        ):
            raise ValueError("Pacemaker parameters must be finite.")
        if (
            lower <= 0.0
            or upper < lower
            or refractory <= 0.0
            or width <= 0.0
            or amplitude <= 0.0
        ):
            raise ValueError("Pacemaker parameters are outside their domains.")
        escape = 60_000.0 / lower
        minimum = 60_000.0 / upper
        if refractory >= escape or width >= minimum:
            raise ValueError("Pacemaker refractory and pulse widths violate rate limits.")
        self.lower_rate_bpm = lower
        self.upper_rate_bpm = upper
        self.refractory_period_ms = refractory
        self.pulse_width_ms = width
        self.pulse_amplitude_mA = amplitude
        self.escape_interval_ms = escape
        self.minimum_interval_ms = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "causal-inhibited-pacemaker-v1",
                "lower_rate_bpm": lower,
                "upper_rate_bpm": upper,
                "refractory_period_ms": refractory,
                "pulse_width_ms": width,
                "pulse_amplitude_mA": amplitude,
            }
        )


class PacemakerControllerState(StrictModule):
    """Fixed pacemaker state at the latest committed sample."""

    time_ms: Array
    last_activation_time_ms: Array
    pulse_end_time_ms: Array
    paced_count: Array
    sample_index: Array


class PacemakerStepResult(StrictModule):
    """Committed pacemaker state, output pulse, and event mask."""

    state: PacemakerControllerState
    pacing_output_mA: Array
    event: Array
    accepted_sense: Array
    status: Array
    successful: Array


class PacemakerReplayTrace(StrictModule):
    """Deterministic replay output for a fixed sample sequence."""

    final_state: PacemakerControllerState
    pacing_output_mA: Array
    event: Array
    accepted_sense: Array
    successful: Array


def initialize_pacemaker_controller(
    plan: PacemakerControllerPlan,
    /,
    *,
    start_time_ms: float = 0.0,
) -> PacemakerControllerState:
    start = float(start_time_ms)
    if not isfinite(start):
        raise ValueError("start_time_ms must be finite.")
    dtype = jnp.asarray(start).dtype
    return PacemakerControllerState(
        time_ms=jnp.asarray(start, dtype=dtype),
        last_activation_time_ms=jnp.asarray(start, dtype=dtype),
        pulse_end_time_ms=jnp.asarray(start, dtype=dtype),
        paced_count=jnp.asarray(0, dtype=jnp.int32),
        sample_index=jnp.asarray(0, dtype=jnp.int32),
    )


def step_pacemaker_controller(
    plan: PacemakerControllerPlan,
    state: PacemakerControllerState,
    sample_time_ms: ArrayLike,
    sensed_depolarization: ArrayLike,
    /,
) -> PacemakerStepResult:
    """Consume only the current sample; invalid/noncausal samples do not commit."""
    sample_time = jnp.asarray(sample_time_ms, dtype=state.time_ms.dtype).reshape(())
    sensed = jnp.asarray(sensed_depolarization, dtype=bool).reshape(())
    finite = jnp.isfinite(sample_time)
    causal = sample_time > state.time_ms
    elapsed = sample_time - state.last_activation_time_ms
    accepted_sense = sensed & (elapsed >= plan.refractory_period_ms)
    pace_due = (
        ~accepted_sense
        & (elapsed >= plan.escape_interval_ms)
        & (elapsed >= plan.minimum_interval_ms)
    )
    pulse_end = jnp.where(
        pace_due, sample_time + plan.pulse_width_ms, state.pulse_end_time_ms
    )
    pulse_active = pace_due | (sample_time < state.pulse_end_time_ms)
    event = jnp.asarray(int(ControlEvent.NONE), dtype=jnp.int32)
    event = jnp.where(
        accepted_sense,
        jnp.bitwise_or(event, int(ControlEvent.SENSED)),
        event,
    )
    event = jnp.where(
        pace_due,
        jnp.bitwise_or(event, int(ControlEvent.PACED)),
        event,
    )
    event = jnp.where(
        pulse_active,
        jnp.bitwise_or(event, int(ControlEvent.PULSE_ACTIVE)),
        event,
    )
    activation_time = jnp.where(
        accepted_sense | pace_due, sample_time, state.last_activation_time_ms
    )
    candidate = PacemakerControllerState(
        time_ms=sample_time,
        last_activation_time_ms=activation_time,
        pulse_end_time_ms=pulse_end,
        paced_count=state.paced_count + pace_due.astype(jnp.int32),
        sample_index=state.sample_index + jnp.asarray(1, dtype=state.sample_index.dtype),
    )
    successful = finite & causal
    accepted = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior), candidate, state
    )
    status = jnp.asarray(int(ControllerStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(ControllerStatus.NONFINITE)),
    )
    status = jnp.where(
        causal,
        status,
        jnp.bitwise_or(status, int(ControllerStatus.NONCAUSAL_SAMPLE)),
    )
    return PacemakerStepResult(
        state=accepted,
        pacing_output_mA=jnp.where(
            successful & pulse_active, plan.pulse_amplitude_mA, 0.0
        ),
        event=jnp.where(successful, event, int(ControlEvent.NONE)),
        accepted_sense=successful & accepted_sense,
        status=status,
        successful=successful,
    )


def replay_pacemaker_controller(
    plan: PacemakerControllerPlan,
    initial_state: PacemakerControllerState,
    sample_time_ms: ArrayLike,
    sensed_depolarization: ArrayLike,
    /,
) -> PacemakerReplayTrace:
    """Replay a fixed sequence through the same causal transition function."""
    times = jnp.asarray(sample_time_ms, dtype=initial_state.time_ms.dtype)
    sensed = jnp.asarray(sensed_depolarization, dtype=bool)
    if times.ndim != 1 or sensed.shape != times.shape:
        raise ValueError("Pacemaker replay arrays must be equal one-dimensional shapes.")

    def transition(state, sample):
        time, sensed_now = sample
        result = step_pacemaker_controller(plan, state, time, sensed_now)
        return result.state, (
            result.pacing_output_mA,
            result.event,
            result.accepted_sense,
            result.successful,
        )

    final_state, outputs = jax.lax.scan(transition, initial_state, (times, sensed))
    return PacemakerReplayTrace(
        final_state=final_state,
        pacing_output_mA=outputs[0],
        event=outputs[1],
        accepted_sense=outputs[2],
        successful=outputs[3],
    )


class PumpControllerPlan(StrictModule, NonTrainableState):
    """Causal fixed-period PI flow controller with integral, speed, and slew limits."""

    sample_period_ms: float = eqx.field(static=True)
    proportional_gain_rpm_ms_per_mm3: float = eqx.field(static=True)
    integral_gain_rpm_per_mm3: float = eqx.field(static=True)
    bias_speed_rpm: float = eqx.field(static=True)
    minimum_speed_rpm: float = eqx.field(static=True)
    maximum_speed_rpm: float = eqx.field(static=True)
    maximum_slew_rpm_per_ms: float = eqx.field(static=True)
    minimum_integral_mm3: float = eqx.field(static=True)
    maximum_integral_mm3: float = eqx.field(static=True)
    timing_tolerance_ms: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_period_ms: float,
        proportional_gain_rpm_ms_per_mm3: float,
        integral_gain_rpm_per_mm3: float,
        bias_speed_rpm: float,
        minimum_speed_rpm: float,
        maximum_speed_rpm: float,
        maximum_slew_rpm_per_ms: float,
        /,
        *,
        minimum_integral_mm3: float,
        maximum_integral_mm3: float,
        timing_tolerance_ms: float = 1.0e-9,
    ):
        values = tuple(
            float(value)
            for value in (
                sample_period_ms,
                proportional_gain_rpm_ms_per_mm3,
                integral_gain_rpm_per_mm3,
                bias_speed_rpm,
                minimum_speed_rpm,
                maximum_speed_rpm,
                maximum_slew_rpm_per_ms,
                minimum_integral_mm3,
                maximum_integral_mm3,
                timing_tolerance_ms,
            )
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Pump controller parameters must be finite.")
        period, kp, ki, bias, minimum, maximum, slew, imin, imax, tolerance = values
        if (
            period <= 0.0
            or kp < 0.0
            or ki < 0.0
            or minimum <= 0.0
            or maximum <= minimum
            or bias < minimum
            or bias > maximum
            or slew <= 0.0
            or imin >= imax
            or tolerance < 0.0
            or tolerance >= period
        ):
            raise ValueError("Pump controller parameters are outside their domains.")
        self.sample_period_ms = period
        self.proportional_gain_rpm_ms_per_mm3 = kp
        self.integral_gain_rpm_per_mm3 = ki
        self.bias_speed_rpm = bias
        self.minimum_speed_rpm = minimum
        self.maximum_speed_rpm = maximum
        self.maximum_slew_rpm_per_ms = slew
        self.minimum_integral_mm3 = imin
        self.maximum_integral_mm3 = imax
        self.timing_tolerance_ms = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "causal-pump-pi-controller-v1",
                "sample_period_ms": period,
                "proportional_gain_rpm_ms_per_mm3": kp,
                "integral_gain_rpm_per_mm3": ki,
                "bias_speed_rpm": bias,
                "minimum_speed_rpm": minimum,
                "maximum_speed_rpm": maximum,
                "maximum_slew_rpm_per_ms": slew,
                "minimum_integral_mm3": imin,
                "maximum_integral_mm3": imax,
                "timing_tolerance_ms": tolerance,
            }
        )


class PumpControllerState(StrictModule):
    """Committed fixed-size state for the pump PI controller."""

    time_ms: Array
    integral_error_mm3: Array
    speed_rpm: Array
    sample_index: Array


class PumpControllerStepResult(StrictModule):
    """Committed pump command and limiting evidence."""

    state: PumpControllerState
    speed_command_rpm: Array
    error_mm3_per_ms: Array
    event: Array
    status: Array
    successful: Array


class PumpControllerReplayTrace(StrictModule):
    """Deterministic pump-controller replay trace."""

    final_state: PumpControllerState
    speed_command_rpm: Array
    error_mm3_per_ms: Array
    event: Array
    successful: Array


def initialize_pump_controller(
    plan: PumpControllerPlan,
    /,
    *,
    start_time_ms: float = 0.0,
    initial_speed_rpm: float | None = None,
) -> PumpControllerState:
    start = float(start_time_ms)
    speed = plan.bias_speed_rpm if initial_speed_rpm is None else float(initial_speed_rpm)
    if not isfinite(start) or not isfinite(speed):
        raise ValueError("Pump controller initial values must be finite.")
    if speed < plan.minimum_speed_rpm or speed > plan.maximum_speed_rpm:
        raise ValueError("Initial pump speed is outside controller limits.")
    dtype = jnp.asarray(start).dtype
    return PumpControllerState(
        time_ms=jnp.asarray(start, dtype=dtype),
        integral_error_mm3=jnp.asarray(0.0, dtype=dtype),
        speed_rpm=jnp.asarray(speed, dtype=dtype),
        sample_index=jnp.asarray(0, dtype=jnp.int32),
    )


def step_pump_controller(
    plan: PumpControllerPlan,
    state: PumpControllerState,
    sample_time_ms: ArrayLike,
    measured_flow_mm3_per_ms: ArrayLike,
    setpoint_flow_mm3_per_ms: ArrayLike,
    /,
) -> PumpControllerStepResult:
    """Advance one fixed-period PI sample without reading future observations."""
    time = jnp.asarray(sample_time_ms, dtype=state.time_ms.dtype).reshape(())
    measured = jnp.asarray(measured_flow_mm3_per_ms, dtype=state.speed_rpm.dtype).reshape(
        ()
    )
    setpoint = jnp.asarray(setpoint_flow_mm3_per_ms, dtype=state.speed_rpm.dtype).reshape(
        ()
    )
    elapsed = time - state.time_ms
    finite = jnp.isfinite(time) & jnp.isfinite(measured) & jnp.isfinite(setpoint)
    causal = elapsed > 0.0
    period_valid = jnp.abs(elapsed - plan.sample_period_ms) <= plan.timing_tolerance_ms
    error = setpoint - measured
    unconstrained_integral = state.integral_error_mm3 + plan.sample_period_ms * error
    integral = jnp.clip(
        unconstrained_integral,
        plan.minimum_integral_mm3,
        plan.maximum_integral_mm3,
    )
    raw_target = (
        plan.bias_speed_rpm
        + plan.proportional_gain_rpm_ms_per_mm3 * error
        + plan.integral_gain_rpm_per_mm3 * integral
    )
    bounded_target = jnp.clip(raw_target, plan.minimum_speed_rpm, plan.maximum_speed_rpm)
    maximum_change = plan.maximum_slew_rpm_per_ms * plan.sample_period_ms
    speed = jnp.clip(
        bounded_target,
        state.speed_rpm - maximum_change,
        state.speed_rpm + maximum_change,
    )
    speed = jnp.clip(speed, plan.minimum_speed_rpm, plan.maximum_speed_rpm)
    event = jnp.asarray(int(ControlEvent.NONE), dtype=jnp.int32)
    event = jnp.where(
        raw_target < plan.minimum_speed_rpm,
        jnp.bitwise_or(event, int(ControlEvent.SPEED_LOWER_LIMIT)),
        event,
    )
    event = jnp.where(
        raw_target > plan.maximum_speed_rpm,
        jnp.bitwise_or(event, int(ControlEvent.SPEED_UPPER_LIMIT)),
        event,
    )
    event = jnp.where(
        jnp.abs(bounded_target - state.speed_rpm) > maximum_change,
        jnp.bitwise_or(event, int(ControlEvent.SLEW_LIMIT)),
        event,
    )
    event = jnp.where(
        unconstrained_integral < plan.minimum_integral_mm3,
        jnp.bitwise_or(event, int(ControlEvent.INTEGRAL_LOWER_LIMIT)),
        event,
    )
    event = jnp.where(
        unconstrained_integral > plan.maximum_integral_mm3,
        jnp.bitwise_or(event, int(ControlEvent.INTEGRAL_UPPER_LIMIT)),
        event,
    )
    candidate = PumpControllerState(
        time_ms=time,
        integral_error_mm3=integral,
        speed_rpm=speed,
        sample_index=state.sample_index + jnp.asarray(1, dtype=state.sample_index.dtype),
    )
    successful = finite & causal & period_valid
    accepted = jax.tree.map(
        lambda proposed, prior: jnp.where(successful, proposed, prior), candidate, state
    )
    status = jnp.asarray(int(ControllerStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(ControllerStatus.NONFINITE)),
    )
    status = jnp.where(
        causal,
        status,
        jnp.bitwise_or(status, int(ControllerStatus.NONCAUSAL_SAMPLE)),
    )
    status = jnp.where(
        period_valid,
        status,
        jnp.bitwise_or(status, int(ControllerStatus.SAMPLE_PERIOD_MISMATCH)),
    )
    return PumpControllerStepResult(
        state=accepted,
        speed_command_rpm=jnp.where(successful, speed, state.speed_rpm),
        error_mm3_per_ms=jnp.where(successful, error, jnp.nan),
        event=jnp.where(successful, event, int(ControlEvent.NONE)),
        status=status,
        successful=successful,
    )


def replay_pump_controller(
    plan: PumpControllerPlan,
    initial_state: PumpControllerState,
    sample_time_ms: ArrayLike,
    measured_flow_mm3_per_ms: ArrayLike,
    setpoint_flow_mm3_per_ms: ArrayLike,
    /,
) -> PumpControllerReplayTrace:
    """Replay recorded pump samples through the production transition."""
    times = jnp.asarray(sample_time_ms, dtype=initial_state.time_ms.dtype)
    measured = jnp.asarray(measured_flow_mm3_per_ms, dtype=initial_state.speed_rpm.dtype)
    setpoint = jnp.asarray(setpoint_flow_mm3_per_ms, dtype=initial_state.speed_rpm.dtype)
    if times.ndim != 1 or measured.shape != times.shape or setpoint.shape != times.shape:
        raise ValueError("Pump replay arrays must have equal one-dimensional shapes.")

    def transition(state, sample):
        result = step_pump_controller(plan, state, sample[0], sample[1], sample[2])
        return result.state, (
            result.speed_command_rpm,
            result.error_mm3_per_ms,
            result.event,
            result.successful,
        )

    final_state, outputs = jax.lax.scan(
        transition, initial_state, (times, measured, setpoint)
    )
    return PumpControllerReplayTrace(
        final_state=final_state,
        speed_command_rpm=outputs[0],
        error_mm3_per_ms=outputs[1],
        event=outputs[2],
        successful=outputs[3],
    )


def _poiseuille_resistance(
    length_mm: float,
    inner_diameter_mm: float,
    dynamic_viscosity_mg_per_mm_ms: float,
) -> float:
    radius = 0.5 * inner_diameter_mm
    return 8.0 * dynamic_viscosity_mg_per_mm_ms * length_mm / (pi * radius**4)


def _validated_hydraulic_parameters(
    element_id: str,
    length_mm: float,
    inner_diameter_mm: float,
    dynamic_viscosity_mg_per_mm_ms: float,
    quadratic_loss_kPa_ms2_per_mm6: float,
) -> tuple[str, float, float, float, float, float]:
    identifier = str(element_id).strip()
    length = float(length_mm)
    diameter = float(inner_diameter_mm)
    viscosity = float(dynamic_viscosity_mg_per_mm_ms)
    quadratic = float(quadratic_loss_kPa_ms2_per_mm6)
    if not identifier:
        raise ValueError("Hydraulic element ID must be non-empty.")
    if not all(isfinite(value) for value in (length, diameter, viscosity, quadratic)):
        raise ValueError("Hydraulic element parameters must be finite.")
    if length <= 0.0 or diameter <= 0.0 or viscosity <= 0.0 or quadratic < 0.0:
        raise ValueError("Hydraulic element parameters are outside their domains.")
    linear = _poiseuille_resistance(length, diameter, viscosity)
    return identifier, length, diameter, viscosity, quadratic, linear


class Cannula(StrictModule, NonTrainableState):
    """Circular cannula with Poiseuille and directional quadratic loss."""

    cannula_id: str = eqx.field(static=True)
    length_mm: float = eqx.field(static=True)
    inner_diameter_mm: float = eqx.field(static=True)
    dynamic_viscosity_mg_per_mm_ms: float = eqx.field(static=True)
    linear_resistance_kPa_ms_per_mm3: float = eqx.field(static=True)
    quadratic_loss_kPa_ms2_per_mm6: float = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        cannula_id: str,
        length_mm: float,
        inner_diameter_mm: float,
        /,
        *,
        dynamic_viscosity_mg_per_mm_ms: float = 3.5e-3,
        quadratic_loss_kPa_ms2_per_mm6: float = 0.0,
    ):
        values = _validated_hydraulic_parameters(
            cannula_id,
            length_mm,
            inner_diameter_mm,
            dynamic_viscosity_mg_per_mm_ms,
            quadratic_loss_kPa_ms2_per_mm6,
        )
        self.cannula_id, self.length_mm, self.inner_diameter_mm = values[:3]
        self.dynamic_viscosity_mg_per_mm_ms = values[3]
        self.quadratic_loss_kPa_ms2_per_mm6 = values[4]
        self.linear_resistance_kPa_ms_per_mm3 = values[5]
        self.element_id = canonical_fingerprint(
            {
                "kind": "cannula-v1",
                "cannula_id": self.cannula_id,
                "length_mm": self.length_mm,
                "inner_diameter_mm": self.inner_diameter_mm,
                "dynamic_viscosity_mg_per_mm_ms": self.dynamic_viscosity_mg_per_mm_ms,
                "quadratic_loss_kPa_ms2_per_mm6": self.quadratic_loss_kPa_ms2_per_mm6,
            }
        )

    def pressure_drop(self, flow_mm3_per_ms: ArrayLike, /) -> Array:
        flow = jnp.asarray(flow_mm3_per_ms)
        return (
            self.linear_resistance_kPa_ms_per_mm3 * flow
            + self.quadratic_loss_kPa_ms2_per_mm6 * flow * jnp.abs(flow)
        )

    def as_pressure_flow_component(
        self, name: str | None = None, /
    ) -> PressureFlowComponent:
        return _hydraulic_pressure_flow_component(
            self.cannula_id if name is None else name,
            self.linear_resistance_kPa_ms_per_mm3,
            self.quadratic_loss_kPa_ms2_per_mm6,
            "cannula",
        )


class TubingSegment(StrictModule, NonTrainableState):
    """Circular tubing segment with distributed and connector losses."""

    tubing_id: str = eqx.field(static=True)
    length_mm: float = eqx.field(static=True)
    inner_diameter_mm: float = eqx.field(static=True)
    dynamic_viscosity_mg_per_mm_ms: float = eqx.field(static=True)
    linear_resistance_kPa_ms_per_mm3: float = eqx.field(static=True)
    quadratic_loss_kPa_ms2_per_mm6: float = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        tubing_id: str,
        length_mm: float,
        inner_diameter_mm: float,
        /,
        *,
        dynamic_viscosity_mg_per_mm_ms: float = 3.5e-3,
        quadratic_loss_kPa_ms2_per_mm6: float = 0.0,
    ):
        values = _validated_hydraulic_parameters(
            tubing_id,
            length_mm,
            inner_diameter_mm,
            dynamic_viscosity_mg_per_mm_ms,
            quadratic_loss_kPa_ms2_per_mm6,
        )
        self.tubing_id, self.length_mm, self.inner_diameter_mm = values[:3]
        self.dynamic_viscosity_mg_per_mm_ms = values[3]
        self.quadratic_loss_kPa_ms2_per_mm6 = values[4]
        self.linear_resistance_kPa_ms_per_mm3 = values[5]
        self.element_id = canonical_fingerprint(
            {
                "kind": "tubing-segment-v1",
                "tubing_id": self.tubing_id,
                "length_mm": self.length_mm,
                "inner_diameter_mm": self.inner_diameter_mm,
                "dynamic_viscosity_mg_per_mm_ms": self.dynamic_viscosity_mg_per_mm_ms,
                "quadratic_loss_kPa_ms2_per_mm6": self.quadratic_loss_kPa_ms2_per_mm6,
            }
        )

    def pressure_drop(self, flow_mm3_per_ms: ArrayLike, /) -> Array:
        flow = jnp.asarray(flow_mm3_per_ms)
        return (
            self.linear_resistance_kPa_ms_per_mm3 * flow
            + self.quadratic_loss_kPa_ms2_per_mm6 * flow * jnp.abs(flow)
        )

    def as_pressure_flow_component(
        self, name: str | None = None, /
    ) -> PressureFlowComponent:
        return _hydraulic_pressure_flow_component(
            self.tubing_id if name is None else name,
            self.linear_resistance_kPa_ms_per_mm3,
            self.quadratic_loss_kPa_ms2_per_mm6,
            "tubing-segment",
        )


class HydraulicOxygenator(StrictModule, NonTrainableState):
    """Hydraulic pressure-loss model; it never implies gas exchange by itself."""

    oxygenator_id: str = eqx.field(static=True)
    linear_resistance_kPa_ms_per_mm3: float = eqx.field(static=True)
    quadratic_loss_kPa_ms2_per_mm6: float = eqx.field(static=True)
    supports_gas_exchange: bool = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        oxygenator_id: str,
        linear_resistance_kPa_ms_per_mm3: float,
        /,
        *,
        quadratic_loss_kPa_ms2_per_mm6: float = 0.0,
    ):
        identifier = str(oxygenator_id).strip()
        linear = float(linear_resistance_kPa_ms_per_mm3)
        quadratic = float(quadratic_loss_kPa_ms2_per_mm6)
        if not identifier:
            raise ValueError("oxygenator_id must be non-empty.")
        if (
            not isfinite(linear)
            or not isfinite(quadratic)
            or linear <= 0.0
            or quadratic < 0.0
        ):
            raise ValueError("Hydraulic oxygenator losses are outside their domains.")
        self.oxygenator_id = identifier
        self.linear_resistance_kPa_ms_per_mm3 = linear
        self.quadratic_loss_kPa_ms2_per_mm6 = quadratic
        self.supports_gas_exchange = False
        self.element_id = canonical_fingerprint(
            {
                "kind": "hydraulic-oxygenator-v1",
                "oxygenator_id": identifier,
                "linear_resistance_kPa_ms_per_mm3": linear,
                "quadratic_loss_kPa_ms2_per_mm6": quadratic,
                "supports_gas_exchange": False,
            }
        )

    def pressure_drop(self, flow_mm3_per_ms: ArrayLike, /) -> Array:
        flow = jnp.asarray(flow_mm3_per_ms)
        return (
            self.linear_resistance_kPa_ms_per_mm3 * flow
            + self.quadratic_loss_kPa_ms2_per_mm6 * flow * jnp.abs(flow)
        )

    def as_pressure_flow_component(
        self, name: str | None = None, /
    ) -> PressureFlowComponent:
        return _hydraulic_pressure_flow_component(
            self.oxygenator_id if name is None else name,
            self.linear_resistance_kPa_ms_per_mm3,
            self.quadratic_loss_kPa_ms2_per_mm6,
            "hydraulic-oxygenator",
        )


class _FlowConservationResidual(StrictModule):
    def __call__(self, time: Array, jet, args, /) -> Array:
        del time, args
        return jet.value("flow_in") - jet.value("flow_out")


class _HydraulicDropResidual(StrictModule):
    linear_resistance: Array
    quadratic_loss: Array

    def __call__(self, time: Array, jet, args, /) -> Array:
        del time, args
        flow = jet.value("flow_out")
        return (
            jet.value("pressure_in")
            - jet.value("pressure_out")
            - self.linear_resistance * flow
            - self.quadratic_loss * flow * jnp.abs(flow)
        )


class _PumpHeadResidual(StrictModule):
    pump_map: PumpHeadFlowMap
    speed_rpm: Array

    def __call__(self, time: Array, jet, args, /) -> Array:
        del time, args
        result = evaluate_pump_map(self.pump_map, jet.value("flow_out"), self.speed_rpm)
        return jet.value("pressure_out") - jet.value("pressure_in") - result.head_kPa


def _incidence(*entries: tuple[str, int]) -> tuple[DAEDerivativeIncidence, ...]:
    return tuple(DAEDerivativeIncidence(name, order) for name, order in entries)


def _two_port_variables() -> tuple[DAEVariableBlock, ...]:
    return (
        DAEVariableBlock("pressure_in", (), 0, 10.0),
        DAEVariableBlock("pressure_out", (), 0, 10.0),
        DAEVariableBlock("flow_in", (), 0, 1.0),
        DAEVariableBlock("flow_out", (), 0, 1.0),
    )


def _two_ports() -> tuple[DAEPort, ...]:
    return (
        DAEPort("inlet", ("pressure_in",), ("flow_in",)),
        DAEPort("outlet", ("pressure_out",), ("flow_out",)),
    )


def _hydraulic_pressure_flow_component(
    name: str,
    linear_resistance: float,
    quadratic_loss: float,
    component_kind: str,
) -> PressureFlowComponent:
    component = DAEComponent(
        name,
        _two_port_variables(),
        (
            DAEEquationBlock(
                "conserve_flow",
                _FlowConservationResidual(),
                _incidence(("flow_in", 0), ("flow_out", 0)),
            ),
            DAEEquationBlock(
                "pressure_drop",
                _HydraulicDropResidual(
                    jnp.asarray(linear_resistance), jnp.asarray(quadratic_loss)
                ),
                _incidence(
                    ("pressure_in", 0),
                    ("pressure_out", 0),
                    ("flow_out", 0),
                ),
            ),
        ),
        _two_ports(),
    )
    return HydraulicDeviceComponent(
        component,
        component_kind=component_kind,
        parameters=(
            ("linear_resistance_kPa_ms_per_mm3", linear_resistance),
            ("quadratic_loss_kPa_ms2_per_mm6", quadratic_loss),
        ),
    )


def pump_pressure_flow_component(
    name: str,
    pump_map: PumpHeadFlowMap,
    speed_rpm: float,
    /,
) -> PressureFlowComponent:
    """Lower one fixed-speed pump map to the canonical pressure/flow DAE substrate."""
    speed = float(speed_rpm)
    if (
        not isfinite(speed)
        or speed < float(pump_map.speed_axis_rpm[0])
        or speed > float(pump_map.speed_axis_rpm[-1])
    ):
        raise ValueError("Fixed pump speed is outside the map domain.")
    component = DAEComponent(
        name,
        _two_port_variables(),
        (
            DAEEquationBlock(
                "conserve_flow",
                _FlowConservationResidual(),
                _incidence(("flow_in", 0), ("flow_out", 0)),
            ),
            DAEEquationBlock(
                "pump_head",
                _PumpHeadResidual(pump_map, jnp.asarray(speed)),
                _incidence(
                    ("pressure_in", 0),
                    ("pressure_out", 0),
                    ("flow_out", 0),
                ),
            ),
        ),
        _two_ports(),
    )
    return HydraulicDeviceComponent(
        component,
        component_kind="head-flow-speed-pump",
        parameters=(("pump_map_id", pump_map.map_id), ("speed_rpm", speed)),
    )


class ECMOCircuitPlan(StrictModule, NonTrainableState):
    """Fixed hydraulic ECMO circuit with optional explicit oxygen exchange."""

    pump_map: PumpHeadFlowMap
    drainage_cannula: Cannula
    return_cannula: Cannula
    tubing: tuple[TubingSegment, ...]
    oxygenator: HydraulicOxygenator
    oxygen_model: MembraneOxygenatorModel | None
    bisection_steps: int = eqx.field(static=True)
    residual_tolerance_kPa: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pump_map: PumpHeadFlowMap,
        drainage_cannula: Cannula,
        return_cannula: Cannula,
        tubing: tuple[TubingSegment, ...],
        oxygenator: HydraulicOxygenator,
        /,
        *,
        oxygen_model: MembraneOxygenatorModel | None = None,
        bisection_steps: int = 64,
        residual_tolerance_kPa: float = 1.0e-8,
    ):
        if not isinstance(pump_map, PumpHeadFlowMap):
            raise TypeError("pump_map must be a PumpHeadFlowMap.")
        if not isinstance(drainage_cannula, Cannula) or not isinstance(
            return_cannula, Cannula
        ):
            raise TypeError("ECMO drainage and return elements must be Cannula objects.")
        segments = tuple(tubing)
        if any(not isinstance(segment, TubingSegment) for segment in segments):
            raise TypeError("Every ECMO tubing element must be a TubingSegment.")
        if not isinstance(oxygenator, HydraulicOxygenator):
            raise TypeError("oxygenator must be a HydraulicOxygenator.")
        if oxygen_model is not None and not isinstance(
            oxygen_model, MembraneOxygenatorModel
        ):
            raise TypeError("oxygen_model must be a MembraneOxygenatorModel or None.")
        steps = int(bisection_steps)
        tolerance = float(residual_tolerance_kPa)
        if steps <= 0 or not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("ECMO operating-point solver configuration is invalid.")
        self.pump_map = pump_map
        self.drainage_cannula = drainage_cannula
        self.return_cannula = return_cannula
        self.tubing = segments
        self.oxygenator = oxygenator
        self.oxygen_model = oxygen_model
        self.bisection_steps = steps
        self.residual_tolerance_kPa = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ecmo-circuit-plan-v1",
                "pump_map": pump_map.map_id,
                "drainage_cannula": drainage_cannula.element_id,
                "return_cannula": return_cannula.element_id,
                "tubing": [segment.element_id for segment in segments],
                "hydraulic_oxygenator": oxygenator.element_id,
                "oxygen_model": None if oxygen_model is None else oxygen_model.model_id,
                "bisection_steps": steps,
                "residual_tolerance_kPa": tolerance,
            }
        )

    @property
    def gas_exchange_enabled(self) -> bool:
        return self.oxygen_model is not None


class ECMOHydraulicResult(StrictModule):
    """ECMO operating point and componentwise hydraulic conservation evidence."""

    flow_mm3_per_ms: Array
    pump_head_kPa: Array
    pressure_load_kPa: Array
    drainage_drop_kPa: Array
    tubing_drop_kPa: Array
    oxygenator_drop_kPa: Array
    return_drop_kPa: Array
    circuit_drop_kPa: Array
    pressure_balance_residual_kPa: Array
    status: Array
    successful: Array


class ECMOCircuitResult(StrictModule):
    """Hydraulic result plus explicitly routed or bypassed oxygen content."""

    hydraulics: ECMOHydraulicResult
    inlet_oxygen_content_mL_per_dL: Array
    outlet_oxygen_content_mL_per_dL: Array
    oxygen_transfer_mL_per_ms: Array
    gas_exchange_enabled: Array
    gas_exchange_performed: Array
    status: Array
    successful: Array


def _ecmo_component_drops(
    plan: ECMOCircuitPlan,
    flow_mm3_per_ms: Array,
) -> tuple[Array, Array, Array, Array]:
    drainage = plan.drainage_cannula.pressure_drop(flow_mm3_per_ms)
    tubing = jnp.asarray(0.0, dtype=flow_mm3_per_ms.dtype)
    for segment in plan.tubing:
        tubing = tubing + segment.pressure_drop(flow_mm3_per_ms)
    oxygenator = plan.oxygenator.pressure_drop(flow_mm3_per_ms)
    return_drop = plan.return_cannula.pressure_drop(flow_mm3_per_ms)
    return drainage, tubing, oxygenator, return_drop


def solve_ecmo_hydraulics(
    plan: ECMOCircuitPlan,
    speed_rpm: ArrayLike,
    inlet_pressure_kPa: ArrayLike,
    outlet_pressure_kPa: ArrayLike,
    /,
) -> ECMOHydraulicResult:
    """Solve pump head = pressure load + passive circuit loss by bisection."""
    speed = jnp.asarray(speed_rpm, dtype=plan.pump_map.head_kPa.dtype).reshape(())
    inlet_pressure = jnp.asarray(inlet_pressure_kPa, dtype=speed.dtype).reshape(())
    outlet_pressure = jnp.asarray(outlet_pressure_kPa, dtype=speed.dtype).reshape(())
    pressure_load = outlet_pressure - inlet_pressure
    minimum_flow = plan.pump_map.flow_axis_mm3_per_ms[0]
    maximum_flow = plan.pump_map.flow_axis_mm3_per_ms[-1]

    def residual(flow):
        map_result = evaluate_pump_map(plan.pump_map, flow, speed)
        drops = _ecmo_component_drops(plan, flow)
        total_drop = sum(drops, start=jnp.asarray(0.0, dtype=flow.dtype))
        return map_result.head_kPa - pressure_load - total_drop

    lower_residual = residual(minimum_flow)
    upper_residual = residual(maximum_flow)
    finite = (
        jnp.isfinite(speed)
        & jnp.isfinite(inlet_pressure)
        & jnp.isfinite(outlet_pressure)
        & jnp.isfinite(lower_residual)
        & jnp.isfinite(upper_residual)
    )
    speed_map_result = evaluate_pump_map(plan.pump_map, minimum_flow, speed)
    bracketed = (lower_residual >= 0.0) & (upper_residual <= 0.0)

    def bisect(_, bracket):
        lower, upper = bracket
        middle = 0.5 * (lower + upper)
        middle_residual = residual(middle)
        lower = jnp.where(middle_residual >= 0.0, middle, lower)
        upper = jnp.where(middle_residual >= 0.0, upper, middle)
        return lower, upper

    lower, upper = jax.lax.fori_loop(
        0,
        plan.bisection_steps,
        bisect,
        (minimum_flow, maximum_flow),
    )
    candidate_flow = 0.5 * (lower + upper)
    map_result = evaluate_pump_map(plan.pump_map, candidate_flow, speed)
    drainage, tubing, oxygenator, return_drop = _ecmo_component_drops(
        plan, candidate_flow
    )
    circuit_drop = drainage + tubing + oxygenator + return_drop
    balance_residual = map_result.head_kPa - pressure_load - circuit_drop
    residual_valid = jnp.abs(balance_residual) <= plan.residual_tolerance_kPa
    successful = (
        finite
        & speed_map_result.successful
        & bracketed
        & map_result.successful
        & residual_valid
    )
    status = jnp.asarray(int(ECMOHydraulicStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.NONFINITE)),
    )
    status = jnp.where(
        speed_map_result.successful & map_result.successful,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.PUMP_MAP_REFUSAL)),
    )
    status = jnp.where(
        bracketed,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.NO_OPERATING_POINT)),
    )
    status = jnp.where(
        residual_valid,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.RESIDUAL_FAILURE)),
    )
    nan = jnp.asarray(jnp.nan, dtype=speed.dtype)
    return ECMOHydraulicResult(
        flow_mm3_per_ms=jnp.where(successful, candidate_flow, nan),
        pump_head_kPa=jnp.where(successful, map_result.head_kPa, nan),
        pressure_load_kPa=jnp.where(successful, pressure_load, nan),
        drainage_drop_kPa=jnp.where(successful, drainage, nan),
        tubing_drop_kPa=jnp.where(successful, tubing, nan),
        oxygenator_drop_kPa=jnp.where(successful, oxygenator, nan),
        return_drop_kPa=jnp.where(successful, return_drop, nan),
        circuit_drop_kPa=jnp.where(successful, circuit_drop, nan),
        pressure_balance_residual_kPa=jnp.where(successful, balance_residual, nan),
        status=status,
        successful=successful,
    )


def run_ecmo_circuit(
    plan: ECMOCircuitPlan,
    speed_rpm: ArrayLike,
    inlet_pressure_kPa: ArrayLike,
    outlet_pressure_kPa: ArrayLike,
    inlet_oxygen_content_mL_per_dL: ArrayLike,
    /,
) -> ECMOCircuitResult:
    """Run hydraulics and perform gas exchange only when an oxygen model exists."""
    hydraulics = solve_ecmo_hydraulics(
        plan, speed_rpm, inlet_pressure_kPa, outlet_pressure_kPa
    )
    inlet_oxygen = jnp.asarray(
        inlet_oxygen_content_mL_per_dL, dtype=plan.pump_map.head_kPa.dtype
    ).reshape(())
    oxygen_input_valid = jnp.isfinite(inlet_oxygen) & (inlet_oxygen >= 0.0)
    if plan.oxygen_model is None:
        outlet_oxygen = inlet_oxygen
        transfer = jnp.asarray(0.0, dtype=inlet_oxygen.dtype)
        exchange_successful = oxygen_input_valid
        exchange_performed = jnp.asarray(False)
    else:
        exchange = exchange_membrane_oxygen(
            plan.oxygen_model, inlet_oxygen, hydraulics.flow_mm3_per_ms
        )
        outlet_oxygen = exchange.outlet.total_mL_per_dL
        transfer = exchange.oxygen_transfer_mL_per_ms
        exchange_successful = exchange.successful
        exchange_performed = exchange.successful
    successful = hydraulics.successful & oxygen_input_valid & exchange_successful
    status = hydraulics.status
    status = jnp.where(
        oxygen_input_valid,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.OXYGEN_INPUT_INVALID)),
    )
    status = jnp.where(
        exchange_successful,
        status,
        jnp.bitwise_or(status, int(ECMOHydraulicStatus.OXYGEN_EXCHANGE_FAILURE)),
    )
    nan = jnp.asarray(jnp.nan, dtype=inlet_oxygen.dtype)
    return ECMOCircuitResult(
        hydraulics=hydraulics,
        inlet_oxygen_content_mL_per_dL=jnp.where(successful, inlet_oxygen, nan),
        outlet_oxygen_content_mL_per_dL=jnp.where(successful, outlet_oxygen, nan),
        oxygen_transfer_mL_per_ms=jnp.where(successful, transfer, nan),
        gas_exchange_enabled=jnp.asarray(plan.gas_exchange_enabled),
        gas_exchange_performed=successful & exchange_performed,
        status=status,
        successful=successful,
    )


__all__ = [
    "Cannula",
    "ControlEvent",
    "ControllerStatus",
    "ECMOCircuitPlan",
    "ECMOCircuitResult",
    "ECMOHydraulicResult",
    "ECMOHydraulicStatus",
    "HydraulicOxygenator",
    "HydraulicDeviceComponent",
    "PacemakerControllerPlan",
    "PacemakerControllerState",
    "PacemakerReplayTrace",
    "PacemakerStepResult",
    "PumpControllerPlan",
    "PumpControllerReplayTrace",
    "PumpControllerState",
    "PumpControllerStepResult",
    "PumpHeadFlowMap",
    "PumpMapResult",
    "PumpMapStatus",
    "TubingSegment",
    "evaluate_pump_map",
    "initialize_pacemaker_controller",
    "initialize_pump_controller",
    "pump_pressure_flow_component",
    "replay_pacemaker_controller",
    "replay_pump_controller",
    "run_ecmo_circuit",
    "solve_ecmo_hydraulics",
    "step_pacemaker_controller",
    "step_pump_controller",
]
