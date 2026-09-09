#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical integrate-and-fire segment maps, explicit resets, and spike arming.

Units are ms, mV, nA, uS and nF. Injected current is positive inward;
``synaptic_conductance_uS * voltage_mV + synaptic_current_offset_nA`` is
positive outward. Segment maps never emit spikes or apply implicit resets.
"""

from __future__ import annotations

from math import isfinite

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LeakyIntegrateAndFire(StrictModule, NonTrainableState):
    """Capacitive point neuron with passive leak and an explicit spike reset."""

    capacitance_nF: Array
    leak_conductance_uS: Array
    resting_mV: Array
    threshold_mV: Array
    reset_mV: Array
    refractory_ms: Array

    def __init__(
        self,
        capacitance_nF: float,
        leak_conductance_uS: float,
        resting_mV: float,
        threshold_mV: float,
        reset_mV: float,
        refractory_ms: float = 0.0,
    ):
        capacitance, leak, resting, threshold, reset, refractory = _base_parameters(
            capacitance_nF,
            leak_conductance_uS,
            resting_mV,
            threshold_mV,
            reset_mV,
            refractory_ms,
        )
        self.capacitance_nF = jnp.asarray(capacitance)
        self.leak_conductance_uS = jnp.asarray(leak)
        self.resting_mV = jnp.asarray(resting)
        self.threshold_mV = jnp.asarray(threshold)
        self.reset_mV = jnp.asarray(reset)
        self.refractory_ms = jnp.asarray(refractory)


class AdaptiveExponentialIntegrateAndFire(StrictModule, NonTrainableState):
    """AdEx point neuron with separate exponential onset and spike cutoff.

    ``exponential_threshold_mV`` is the exponential-current onset V_T;
    ``threshold_mV`` is the upper event cutoff. They must not be conflated.
    Adaptation follows tau_w dw/dt = a (V - E_L) - w and receives an
    increment b only when :func:`reset_point_neuron` is explicitly called.
    """

    capacitance_nF: Array
    leak_conductance_uS: Array
    resting_mV: Array
    threshold_mV: Array
    reset_mV: Array
    refractory_ms: Array
    slope_mV: Array
    adaptation_conductance_uS: Array
    adaptation_time_constant_ms: Array
    adaptation_increment_nA: Array
    exponential_threshold_mV: Array

    def __init__(
        self,
        capacitance_nF: float,
        leak_conductance_uS: float,
        resting_mV: float,
        threshold_mV: float,
        reset_mV: float,
        slope_mV: float,
        adaptation_conductance_uS: float,
        adaptation_time_constant_ms: float,
        adaptation_increment_nA: float,
        refractory_ms: float = 0.0,
        *,
        exponential_threshold_mV: float = -50.0,
    ):
        capacitance, leak, resting, threshold, reset, refractory = _base_parameters(
            capacitance_nF,
            leak_conductance_uS,
            resting_mV,
            threshold_mV,
            reset_mV,
            refractory_ms,
        )
        slope = _positive(slope_mV, "slope_mV")
        adaptation = _positive(
            adaptation_conductance_uS, "adaptation_conductance_uS", allow_zero=True
        )
        tau = _positive(adaptation_time_constant_ms, "adaptation_time_constant_ms")
        increment = _positive(
            adaptation_increment_nA, "adaptation_increment_nA", allow_zero=True
        )
        onset = _finite(exponential_threshold_mV, "exponential_threshold_mV")
        if onset >= threshold:
            raise ValueError("exponential_threshold_mV must be below threshold_mV.")
        self.capacitance_nF = jnp.asarray(capacitance)
        self.leak_conductance_uS = jnp.asarray(leak)
        self.resting_mV = jnp.asarray(resting)
        self.threshold_mV = jnp.asarray(threshold)
        self.reset_mV = jnp.asarray(reset)
        self.refractory_ms = jnp.asarray(refractory)
        self.slope_mV = jnp.asarray(slope)
        self.adaptation_conductance_uS = jnp.asarray(adaptation)
        self.adaptation_time_constant_ms = jnp.asarray(tau)
        self.adaptation_increment_nA = jnp.asarray(increment)
        self.exponential_threshold_mV = jnp.asarray(onset)


PointNeuronModel = LeakyIntegrateAndFire | AdaptiveExponentialIntegrateAndFire


class PointNeuronState(StrictModule):
    """Scalar or broadcast-compatible vector voltage, adaptation and deadline."""

    voltage_mV: Array
    adaptation_nA: Array
    refractory_until_ms: Array


def _finite(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not bool.")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _positive(value, name, *, allow_zero=False):
    resolved = _finite(value, name)
    if resolved < 0 if allow_zero else resolved <= 0:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}.")
    return resolved


def _base_parameters(capacitance, leak, resting, threshold, reset, refractory):
    capacitance = _positive(capacitance, "capacitance_nF")
    leak = _positive(leak, "leak_conductance_uS", allow_zero=True)
    resting = _finite(resting, "resting_mV")
    threshold = _finite(threshold, "threshold_mV")
    reset = _finite(reset, "reset_mV")
    refractory = _positive(refractory, "refractory_ms", allow_zero=True)
    if reset >= threshold:
        raise ValueError("reset_mV must be below threshold_mV.")
    return capacitance, leak, resting, threshold, reset, refractory


def initialize_point_neuron(
    model: PointNeuronModel, voltage_mV=None, time_ms=0.0
) -> PointNeuronState:
    """Initialize at rest (or supplied voltage), with no pending refractory hold."""
    if not isinstance(
        model, (LeakyIntegrateAndFire, AdaptiveExponentialIntegrateAndFire)
    ):
        raise TypeError("model must be a physical point-neuron model.")
    voltage, time = jnp.broadcast_arrays(
        jnp.asarray(model.resting_mV if voltage_mV is None else voltage_mV),
        jnp.asarray(time_ms),
    )
    dtype = jnp.result_type(voltage, model.capacitance_nF, jnp.asarray(0.0))
    voltage = voltage.astype(dtype)
    return PointNeuronState(voltage, jnp.zeros_like(voltage), time.astype(dtype))


def _charge_increment(rate_times_step: Array) -> Array:
    """Entire phi_1(-x), including a smooth and accurate zero-leak limit."""
    small = jnp.abs(rate_times_step) < 1.0e-4
    safe = jnp.where(small, 1.0, rate_times_step)
    regular = -jnp.expm1(-rate_times_step) / safe
    series = 1.0 + rate_times_step * (
        -0.5 + rate_times_step * (1.0 / 6.0 - rate_times_step / 24.0)
    )
    return jnp.where(small, series, regular)


def _passive_advance(voltage, elapsed, capacitance, conductance, inward_drive):
    rate_times_step = conductance * elapsed / capacitance
    return voltage + (elapsed / capacitance) * _charge_increment(rate_times_step) * (
        inward_drive - conductance * voltage
    )


def _adaptation_advance(model, adaptation, voltage, elapsed):
    equilibrium = model.adaptation_conductance_uS * (voltage - model.resting_mV)
    return adaptation + (-jnp.expm1(-elapsed / model.adaptation_time_constant_ms)) * (
        equilibrium - adaptation
    )


def _adex_advance(model, voltage, adaptation, elapsed, conductance, inward_drive):
    """Second-order exponential midpoint, with 32 fixed differentiable substeps.

    Leak and frozen-voltage adaptation are treated exponentially, not with a
    stiff explicit-Euler update. Above the event cutoff only, the exponential
    current is continued constantly. That continuation leaves the physical
    subthreshold ODE unchanged and gives the event localizer a finite bracket
    without pretending that the post-spike AdEx blow-up is a physical segment.
    The network must localize the first cutoff and apply its explicit reset.
    """
    step = elapsed / 32

    def exponential_current(value):
        exponent = (
            jnp.minimum(value, model.threshold_mV) - model.exponential_threshold_mV
        ) / model.slope_mV
        return model.leak_conductance_uS * model.slope_mV * jnp.exp(exponent)

    def substep(_, values):
        current_voltage, current_adaptation = values
        midpoint_voltage = _passive_advance(
            current_voltage,
            step / 2,
            model.capacitance_nF,
            conductance,
            inward_drive + exponential_current(current_voltage) - current_adaptation,
        )
        midpoint_adaptation = _adaptation_advance(
            model, current_adaptation, current_voltage, step / 2
        )
        next_voltage = _passive_advance(
            current_voltage,
            step,
            model.capacitance_nF,
            conductance,
            inward_drive + exponential_current(midpoint_voltage) - midpoint_adaptation,
        )
        next_adaptation = _adaptation_advance(
            model, current_adaptation, midpoint_voltage, step
        )
        return next_voltage, next_adaptation

    return jax.lax.fori_loop(0, 32, substep, (voltage, adaptation))


def advance_point_neuron(
    model: PointNeuronModel,
    state: PointNeuronState,
    elapsed_ms,
    injected_current_nA=0.0,
    synaptic_conductance_uS=0.0,
    synaptic_current_offset_nA=0.0,
    time_ms=0.0,
) -> PointNeuronState:
    """Advance one constant-input segment without spike detection or reset.

    LIF charge is analytic, including zero total conductance. Refractory voltage
    is held exactly at reset; AdEx adaptation continues evolving during that
    hold. A segment crossing release integrates only its remaining free time.
    ``time_ms`` is the start of the supplied state, not the segment end.
    All arguments broadcast over independent point neurons. Invalid numerical
    inputs propagate nonfinite state for the caller's atomic-step validation.
    """
    if not isinstance(
        model, (LeakyIntegrateAndFire, AdaptiveExponentialIntegrateAndFire)
    ):
        raise TypeError("model must be a physical point-neuron model.")
    dtype = state.voltage_mV.dtype
    voltage, adaptation, deadline, elapsed, time, injected, synaptic_g, synaptic_i = (
        jnp.broadcast_arrays(
            state.voltage_mV,
            state.adaptation_nA,
            state.refractory_until_ms,
            jnp.asarray(elapsed_ms, dtype=dtype),
            jnp.asarray(time_ms, dtype=dtype),
            jnp.asarray(injected_current_nA, dtype=dtype),
            jnp.asarray(synaptic_conductance_uS, dtype=dtype),
            jnp.asarray(synaptic_current_offset_nA, dtype=dtype),
        )
    )
    valid = (
        jnp.isfinite(elapsed)
        & (elapsed >= 0)
        & jnp.isfinite(time)
        & jnp.isfinite(injected)
        & jnp.isfinite(synaptic_g)
        & (synaptic_g >= 0)
        & jnp.isfinite(synaptic_i)
        & jnp.isfinite(deadline)
        & jnp.isfinite(voltage)
        & jnp.isfinite(adaptation)
    )
    held = time < deadline
    held_elapsed = jnp.minimum(jnp.maximum(deadline - time, 0), elapsed)
    free_elapsed = elapsed - held_elapsed
    voltage = jnp.where(held, model.reset_mV, voltage)
    conductance = model.leak_conductance_uS + synaptic_g
    inward_drive = model.leak_conductance_uS * model.resting_mV + injected - synaptic_i
    if isinstance(model, AdaptiveExponentialIntegrateAndFire):
        adaptation = _adaptation_advance(model, adaptation, model.reset_mV, held_elapsed)
        voltage, adaptation = _adex_advance(
            model, voltage, adaptation, free_elapsed, conductance, inward_drive
        )
    else:
        voltage = _passive_advance(
            voltage, free_elapsed, model.capacitance_nF, conductance, inward_drive
        )
    return PointNeuronState(
        jnp.where(valid, voltage, jnp.nan),
        jnp.where(valid, adaptation, jnp.nan),
        deadline,
    )


def reset_point_neuron(
    model: PointNeuronModel, state: PointNeuronState, time_ms
) -> PointNeuronState:
    """Apply a spike reset, AdEx increment, and absolute refractory deadline."""
    voltage, adaptation, time = jnp.broadcast_arrays(
        state.voltage_mV,
        state.adaptation_nA,
        jnp.asarray(time_ms, dtype=state.voltage_mV.dtype),
    )
    if isinstance(model, AdaptiveExponentialIntegrateAndFire):
        adaptation = adaptation + model.adaptation_increment_nA
    elif not isinstance(model, LeakyIntegrateAndFire):
        raise TypeError("model must be a physical point-neuron model.")
    return PointNeuronState(
        jnp.full_like(voltage, model.reset_mV), adaptation, time + model.refractory_ms
    )


class ThresholdDetector(StrictModule, NonTrainableState):
    """Observation-only upward spike detector with optional rearming hysteresis."""

    threshold_mV: Array
    rearm_mV: Array

    def __init__(self, threshold_mV: float, rearm_mV: float | None = None):
        threshold = _finite(threshold_mV, "threshold_mV")
        rearm = threshold if rearm_mV is None else _finite(rearm_mV, "rearm_mV")
        if rearm > threshold:
            raise ValueError("rearm_mV must not exceed threshold_mV.")
        self.threshold_mV = jnp.asarray(threshold)
        self.rearm_mV = jnp.asarray(rearm)


def initialize_threshold_detector(detector: ThresholdDetector, voltage_mV) -> Array:
    """Arm only below threshold and at/below the configured rearming level."""
    voltage = jnp.asarray(voltage_mV)
    return (voltage < detector.threshold_mV) & (voltage <= detector.rearm_mV)


def advance_threshold_detector(
    detector: ThresholdDetector, armed: Array, voltage_mV
) -> tuple[Array, Array]:
    """Return ``(armed, fired)``; a threshold plateau never fires repeatedly."""
    voltage = jnp.asarray(voltage_mV)
    fired = jnp.asarray(armed, dtype=jnp.bool_) & (voltage >= detector.threshold_mV)
    rearmed = initialize_threshold_detector(detector, voltage)
    return (jnp.asarray(armed, dtype=jnp.bool_) & ~fired) | rearmed, fired


__all__ = [
    "AdaptiveExponentialIntegrateAndFire",
    "LeakyIntegrateAndFire",
    "PointNeuronModel",
    "PointNeuronState",
    "ThresholdDetector",
    "advance_point_neuron",
    "advance_threshold_detector",
    "initialize_point_neuron",
    "initialize_threshold_detector",
    "reset_point_neuron",
]
