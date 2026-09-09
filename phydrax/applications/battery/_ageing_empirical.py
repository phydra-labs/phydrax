#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class EmpiricalAgeingStatus(IntEnum):
    """Fail-closed disposition for one physical ageing macrostep."""

    SUCCESS = 0
    INVALID_STATE = 1
    INVALID_TIMES = 2
    TIME_GAP_EXCEEDED = 3
    MACROSTEP_EXCEEDED = 4
    TEMPERATURE_OUT_OF_SUPPORT = 5
    STATE_OF_CHARGE_OUT_OF_SUPPORT = 6
    CURRENT_OUT_OF_SUPPORT = 7
    THROUGHPUT_OUT_OF_SUPPORT = 8
    NONFINITE_TRANSITION = 9


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _support_bounds(
    values: ArrayLike,
    name: str,
    /,
    *,
    positive_lower: bool = False,
    nonnegative_lower: bool = False,
) -> Array:
    bounds = jnp.asarray(values)
    if bounds.shape != (2,) or jnp.issubdtype(bounds.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must contain two real scalars.")
    host = np.asarray(bounds, dtype=float)
    if not np.all(np.isfinite(host)) or not host[1] > host[0]:
        raise ValueError(f"{name} must be finite and strictly increasing.")
    if positive_lower and not host[0] > 0.0:
        raise ValueError(f"{name} must have a positive lower bound.")
    if nonnegative_lower and host[0] < 0.0:
        raise ValueError(f"{name} must have a nonnegative lower bound.")
    return jax.lax.stop_gradient(bounds.astype(jnp.result_type(bounds, float)))


def _real_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must be one real scalar.")
    if not jnp.issubdtype(scalar.dtype, jnp.inexact):
        scalar = scalar.astype(float)
    return scalar


class EmpiricalAgeingSupport(StrictModule, NonTrainableState):
    """Immutable calibration support for one empirical cell/product law."""

    temperature_bounds_k: Array
    state_of_charge_bounds: Array
    current_magnitude_bounds_a: Array
    charge_throughput_bounds_c: Array
    maximum_time_gap_s: float = eqx.field(static=True)
    maximum_macrostep_s: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature_bounds_k: ArrayLike,
        state_of_charge_bounds: ArrayLike,
        current_magnitude_bounds_a: ArrayLike,
        charge_throughput_bounds_c: ArrayLike,
        /,
        *,
        maximum_time_gap_s: float,
        maximum_macrostep_s: float,
        source_id: str,
    ):
        temperature = _support_bounds(
            temperature_bounds_k,
            "Temperature support",
            positive_lower=True,
        )
        state_of_charge = _support_bounds(
            state_of_charge_bounds,
            "State-of-charge support",
            nonnegative_lower=True,
        )
        current = _support_bounds(
            current_magnitude_bounds_a,
            "Current-magnitude support",
            nonnegative_lower=True,
        )
        throughput = _support_bounds(
            charge_throughput_bounds_c,
            "Charge-throughput support",
            nonnegative_lower=True,
        )
        soc_host = np.asarray(state_of_charge, dtype=float)
        if soc_host[1] > 1.0:
            raise ValueError("State-of-charge support must lie inside [0, 1].")
        maximum_gap = float(maximum_time_gap_s)
        maximum_macrostep = float(maximum_macrostep_s)
        if not isfinite(maximum_gap) or maximum_gap <= 0.0:
            raise ValueError("maximum_time_gap_s must be finite and positive.")
        if not isfinite(maximum_macrostep) or maximum_macrostep <= 0.0:
            raise ValueError("maximum_macrostep_s must be finite and positive.")
        if maximum_gap > maximum_macrostep:
            raise ValueError("maximum_time_gap_s cannot exceed maximum_macrostep_s.")
        source = _identifier(source_id, "Empirical ageing support source ID")
        self.temperature_bounds_k = temperature
        self.state_of_charge_bounds = state_of_charge
        self.current_magnitude_bounds_a = current
        self.charge_throughput_bounds_c = throughput
        self.maximum_time_gap_s = maximum_gap
        self.maximum_macrostep_s = maximum_macrostep
        self.source_id = source
        self.support_id = canonical_fingerprint(
            {
                "kind": "battery-empirical-ageing-support",
                "temperature_bounds_k": np.asarray(temperature, dtype=float).tolist(),
                "state_of_charge_bounds": np.asarray(
                    state_of_charge, dtype=float
                ).tolist(),
                "current_magnitude_bounds_a": np.asarray(current, dtype=float).tolist(),
                "charge_throughput_bounds_c": np.asarray(
                    throughput, dtype=float
                ).tolist(),
                "maximum_time_gap_s": maximum_gap,
                "maximum_macrostep_s": maximum_macrostep,
                "source_id": source,
            }
        )


class EmpiricalAgeingTopology(StrictModule, NonTrainableState):
    """Fixed stress-history shape bound to an immutable calibration support."""

    support: EmpiricalAgeingSupport
    history_node_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(self, support: EmpiricalAgeingSupport, history_node_count: int, /):
        if not isinstance(support, EmpiricalAgeingSupport):
            raise TypeError("support must be EmpiricalAgeingSupport.")
        if isinstance(history_node_count, bool) or not isinstance(
            history_node_count, int
        ):
            raise TypeError("history_node_count must be an integer.")
        if history_node_count < 2:
            raise ValueError("Empirical ageing histories require at least two nodes.")
        self.support = support
        self.history_node_count = history_node_count
        self.topology_id = canonical_fingerprint(
            {
                "kind": "battery-empirical-ageing-topology",
                "support_id": support.support_id,
                "history_node_count": history_node_count,
            }
        )

    def advance(
        self,
        coefficients: "EmpiricalAgeingCoefficients",
        state: "EmpiricalAgeingState",
        history: "EmpiricalAgeingStressHistory",
        /,
    ) -> "EmpiricalAgeingTransitionResult":
        return advance_empirical_ageing(self, coefficients, state, history)


class EmpiricalAgeingCoefficients(StrictModule):
    """Dynamic user coefficients for the declared additive exposure law.

    Calendar exposure is integrated per second and throughput exposure per
    coulomb. The temperature factors use ``exp(a * (1 / T_ref - 1 / T))``;
    SOC and current-magnitude factors use centered exponentials. The two latent
    exposure clocks are raised to their respective powers and then added.
    """

    calendar_rate_per_s: Array
    throughput_rate_per_c: Array
    calendar_activation_temperature_k: Array
    throughput_activation_temperature_k: Array
    calendar_soc_coefficient: Array
    throughput_soc_coefficient: Array
    throughput_current_coefficient_per_a: Array
    reference_temperature_k: Array
    reference_state_of_charge: Array
    reference_current_magnitude_a: Array
    calendar_exponent: Array
    throughput_exponent: Array
    capacity_loss_scale: Array
    resistance_growth_scale: Array

    def __init__(
        self,
        calendar_rate_per_s: ArrayLike,
        throughput_rate_per_c: ArrayLike,
        calendar_activation_temperature_k: ArrayLike,
        throughput_activation_temperature_k: ArrayLike,
        calendar_soc_coefficient: ArrayLike,
        throughput_soc_coefficient: ArrayLike,
        throughput_current_coefficient_per_a: ArrayLike,
        reference_temperature_k: ArrayLike,
        reference_state_of_charge: ArrayLike,
        reference_current_magnitude_a: ArrayLike,
        calendar_exponent: ArrayLike,
        throughput_exponent: ArrayLike,
        capacity_loss_scale: ArrayLike,
        resistance_growth_scale: ArrayLike,
        /,
    ):
        values = tuple(
            _real_scalar(value, name)
            for value, name in (
                (calendar_rate_per_s, "calendar_rate_per_s"),
                (throughput_rate_per_c, "throughput_rate_per_c"),
                (
                    calendar_activation_temperature_k,
                    "calendar_activation_temperature_k",
                ),
                (
                    throughput_activation_temperature_k,
                    "throughput_activation_temperature_k",
                ),
                (calendar_soc_coefficient, "calendar_soc_coefficient"),
                (throughput_soc_coefficient, "throughput_soc_coefficient"),
                (
                    throughput_current_coefficient_per_a,
                    "throughput_current_coefficient_per_a",
                ),
                (reference_temperature_k, "reference_temperature_k"),
                (reference_state_of_charge, "reference_state_of_charge"),
                (
                    reference_current_magnitude_a,
                    "reference_current_magnitude_a",
                ),
                (calendar_exponent, "calendar_exponent"),
                (throughput_exponent, "throughput_exponent"),
                (capacity_loss_scale, "capacity_loss_scale"),
                (resistance_growth_scale, "resistance_growth_scale"),
            )
        )
        dtype = jnp.result_type(*values)
        values = tuple(value.astype(dtype) for value in values)
        (
            calendar_rate,
            throughput_rate,
            calendar_activation,
            throughput_activation,
            calendar_soc,
            throughput_soc,
            throughput_current,
            reference_temperature,
            reference_soc,
            reference_current,
            calendar_power,
            throughput_power,
            capacity_scale,
            resistance_scale,
        ) = values
        finite = jnp.all(jnp.isfinite(jnp.stack(values)))
        valid = (
            finite
            & (calendar_rate >= 0.0)
            & (throughput_rate >= 0.0)
            & (reference_temperature > 0.0)
            & (reference_soc >= 0.0)
            & (reference_soc <= 1.0)
            & (reference_current >= 0.0)
            & (calendar_power > 0.0)
            & (throughput_power > 0.0)
            & (capacity_scale >= 0.0)
            & (resistance_scale >= 0.0)
        )
        calendar_rate = eqx.error_if(
            calendar_rate,
            ~valid,
            "Empirical ageing coefficients are nonfinite or outside their "
            "physical domain.",
        )
        self.calendar_rate_per_s = calendar_rate
        self.throughput_rate_per_c = throughput_rate
        self.calendar_activation_temperature_k = calendar_activation
        self.throughput_activation_temperature_k = throughput_activation
        self.calendar_soc_coefficient = calendar_soc
        self.throughput_soc_coefficient = throughput_soc
        self.throughput_current_coefficient_per_a = throughput_current
        self.reference_temperature_k = reference_temperature
        self.reference_state_of_charge = reference_soc
        self.reference_current_magnitude_a = reference_current
        self.calendar_exponent = calendar_power
        self.throughput_exponent = throughput_power
        self.capacity_loss_scale = capacity_scale
        self.resistance_growth_scale = resistance_scale


class EmpiricalAgeingState(StrictModule):
    """Latent exposure clocks and accepted physical-history coordinates."""

    calendar_exposure: Array
    throughput_exposure: Array
    time_s: Array
    charge_throughput_c: Array

    def __init__(
        self,
        calendar_exposure: ArrayLike,
        throughput_exposure: ArrayLike,
        time_s: ArrayLike,
        charge_throughput_c: ArrayLike,
        /,
    ):
        values = tuple(
            _real_scalar(value, name)
            for value, name in (
                (calendar_exposure, "calendar_exposure"),
                (throughput_exposure, "throughput_exposure"),
                (time_s, "time_s"),
                (charge_throughput_c, "charge_throughput_c"),
            )
        )
        dtype = jnp.result_type(*values)
        (
            self.calendar_exposure,
            self.throughput_exposure,
            self.time_s,
            self.charge_throughput_c,
        ) = tuple(value.astype(dtype) for value in values)


class EmpiricalAgeingStressHistory(StrictModule):
    """Supplied fixed-shape stress nodes for one physical macrostep."""

    times_s: Array
    temperature_k: Array
    state_of_charge: Array
    current_a: Array

    def __init__(
        self,
        times_s: ArrayLike,
        temperature_k: ArrayLike,
        state_of_charge: ArrayLike,
        current_a: ArrayLike,
        /,
    ):
        arrays = tuple(
            jnp.asarray(value)
            for value in (times_s, temperature_k, state_of_charge, current_a)
        )
        if arrays[0].ndim != 1 or int(arrays[0].size) < 2:
            raise ValueError(
                "Empirical ageing stress histories require at least two nodes."
            )
        if any(value.shape != arrays[0].shape for value in arrays[1:]):
            raise ValueError(
                "Empirical ageing stress histories must share one fixed shape."
            )
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
            raise TypeError("Empirical ageing stress histories must be real-valued.")
        dtype = jnp.result_type(*arrays, float)
        self.times_s, self.temperature_k, self.state_of_charge, self.current_a = (
            value.astype(dtype) for value in arrays
        )


class EmpiricalAgeingStressSummary(StrictModule):
    """Time-weighted macrostep stress and exposure increments."""

    duration_s: Array
    mean_temperature_k: Array
    mean_state_of_charge: Array
    mean_current_magnitude_a: Array
    charge_throughput_c: Array
    calendar_exposure: Array
    throughput_exposure: Array


class EmpiricalAgeingObservables(StrictModule):
    """One-way damage, capacity, resistance, and state-of-health outputs."""

    calendar_damage: Array
    throughput_damage: Array
    total_damage: Array
    capacity_loss_fraction: Array
    resistance_growth_fraction: Array
    state_of_health: Array


class EmpiricalAgeingTransitionResult(StrictModule):
    """Candidate, accepted, stress, and fail-closed status for one macrostep."""

    candidate_state: EmpiricalAgeingState
    accepted_state: EmpiricalAgeingState
    stress_summary: EmpiricalAgeingStressSummary
    observables: EmpiricalAgeingObservables
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == int(EmpiricalAgeingStatus.SUCCESS)


def initial_empirical_ageing_state(
    time_s: ArrayLike = 0.0,
    /,
    *,
    charge_throughput_c: ArrayLike = 0.0,
) -> EmpiricalAgeingState:
    """Create a zero-damage state at caller-owned physical coordinates."""

    time = _real_scalar(time_s, "time_s")
    throughput = _real_scalar(charge_throughput_c, "charge_throughput_c")
    dtype = jnp.result_type(time, throughput)
    zero = jnp.zeros((), dtype=dtype)
    return EmpiricalAgeingState(zero, zero, time, throughput)


def empirical_ageing_observables(
    coefficients: EmpiricalAgeingCoefficients,
    state: EmpiricalAgeingState,
    /,
) -> EmpiricalAgeingObservables:
    """Evaluate the exact nonlinear law from its latent exposure state."""

    if not isinstance(coefficients, EmpiricalAgeingCoefficients):
        raise TypeError("coefficients must be EmpiricalAgeingCoefficients.")
    if not isinstance(state, EmpiricalAgeingState):
        raise TypeError("state must be EmpiricalAgeingState.")
    calendar_damage = jnp.power(state.calendar_exposure, coefficients.calendar_exponent)
    throughput_damage = jnp.power(
        state.throughput_exposure, coefficients.throughput_exponent
    )
    total_damage = calendar_damage + throughput_damage
    capacity_argument = coefficients.capacity_loss_scale * total_damage
    resistance_argument = coefficients.resistance_growth_scale * total_damage
    capacity_loss = -jnp.expm1(-capacity_argument)
    state_of_health = jnp.exp(-capacity_argument)
    resistance_growth = jnp.expm1(resistance_argument)
    return EmpiricalAgeingObservables(
        calendar_damage,
        throughput_damage,
        total_damage,
        capacity_loss,
        resistance_growth,
        state_of_health,
    )


def _trapezoid(values: Array, time_gaps_s: Array, /) -> Array:
    return jnp.sum(0.5 * (values[:-1] + values[1:]) * time_gaps_s)


def _accepted_state(
    successful: Array,
    candidate: EmpiricalAgeingState,
    previous: EmpiricalAgeingState,
    /,
) -> EmpiricalAgeingState:
    return EmpiricalAgeingState(
        jnp.where(successful, candidate.calendar_exposure, previous.calendar_exposure),
        jnp.where(
            successful, candidate.throughput_exposure, previous.throughput_exposure
        ),
        jnp.where(successful, candidate.time_s, previous.time_s),
        jnp.where(
            successful, candidate.charge_throughput_c, previous.charge_throughput_c
        ),
    )


def advance_empirical_ageing(
    topology: EmpiricalAgeingTopology,
    coefficients: EmpiricalAgeingCoefficients,
    state: EmpiricalAgeingState,
    history: EmpiricalAgeingStressHistory,
    /,
) -> EmpiricalAgeingTransitionResult:
    """Advance one declared physical macrostep without altering its stress inputs.

    Node-wise nonlinear stress rates are integrated with trapezoidal, time-weighted
    quadrature. Latent exposures are additive, so replaying adjacent partitions of
    the same supplied nodes gives the same final nonlinear damage as one macrostep.
    Unsupported histories are never accepted and are never clipped or extrapolated.
    """

    if not isinstance(topology, EmpiricalAgeingTopology):
        raise TypeError("topology must be EmpiricalAgeingTopology.")
    if not isinstance(coefficients, EmpiricalAgeingCoefficients):
        raise TypeError("coefficients must be EmpiricalAgeingCoefficients.")
    if not isinstance(state, EmpiricalAgeingState):
        raise TypeError("state must be EmpiricalAgeingState.")
    if not isinstance(history, EmpiricalAgeingStressHistory):
        raise TypeError("history must be EmpiricalAgeingStressHistory.")
    if int(history.times_s.size) != topology.history_node_count:
        raise ValueError("Stress history shape does not match the ageing topology.")

    dtype = jnp.result_type(
        history.times_s,
        state.calendar_exposure,
        coefficients.calendar_rate_per_s,
    )
    times = history.times_s.astype(dtype)
    temperature = history.temperature_k.astype(dtype)
    state_of_charge = history.state_of_charge.astype(dtype)
    current = history.current_a.astype(dtype)
    magnitude = jnp.abs(current)
    gaps = jnp.diff(times)
    duration = times[-1] - times[0]
    support = topology.support

    state_finite = (
        jnp.isfinite(state.calendar_exposure)
        & jnp.isfinite(state.throughput_exposure)
        & jnp.isfinite(state.time_s)
        & jnp.isfinite(state.charge_throughput_c)
    )
    state_valid = (
        state_finite
        & (state.calendar_exposure >= 0.0)
        & (state.throughput_exposure >= 0.0)
        & (state.charge_throughput_c >= 0.0)
    )
    times_valid = (
        jnp.all(jnp.isfinite(times)) & jnp.all(gaps > 0.0) & (times[0] == state.time_s)
    )
    gaps_valid = jnp.all(gaps <= support.maximum_time_gap_s)
    macrostep_valid = duration <= support.maximum_macrostep_s
    temperature_valid = jnp.all(
        jnp.isfinite(temperature)
        & (temperature >= support.temperature_bounds_k[0])
        & (temperature <= support.temperature_bounds_k[1])
    )
    state_of_charge_valid = jnp.all(
        jnp.isfinite(state_of_charge)
        & (state_of_charge >= support.state_of_charge_bounds[0])
        & (state_of_charge <= support.state_of_charge_bounds[1])
    )
    current_valid = jnp.all(
        jnp.isfinite(magnitude)
        & (magnitude >= support.current_magnitude_bounds_a[0])
        & (magnitude <= support.current_magnitude_bounds_a[1])
    )

    safe_gaps = jnp.where(jnp.isfinite(gaps) & (gaps > 0.0), gaps, 0.0)
    safe_temperature = jnp.where(
        jnp.isfinite(temperature)
        & (temperature >= support.temperature_bounds_k[0])
        & (temperature <= support.temperature_bounds_k[1]),
        temperature,
        coefficients.reference_temperature_k,
    )
    safe_soc = jnp.where(
        jnp.isfinite(state_of_charge)
        & (state_of_charge >= support.state_of_charge_bounds[0])
        & (state_of_charge <= support.state_of_charge_bounds[1]),
        state_of_charge,
        coefficients.reference_state_of_charge,
    )
    safe_magnitude = jnp.where(
        jnp.isfinite(magnitude)
        & (magnitude >= support.current_magnitude_bounds_a[0])
        & (magnitude <= support.current_magnitude_bounds_a[1]),
        magnitude,
        coefficients.reference_current_magnitude_a,
    )
    calendar_log_stress = coefficients.calendar_activation_temperature_k * (
        1.0 / coefficients.reference_temperature_k - 1.0 / safe_temperature
    ) + coefficients.calendar_soc_coefficient * (
        safe_soc - coefficients.reference_state_of_charge
    )
    throughput_log_stress = (
        coefficients.throughput_activation_temperature_k
        * (1.0 / coefficients.reference_temperature_k - 1.0 / safe_temperature)
        + coefficients.throughput_soc_coefficient
        * (safe_soc - coefficients.reference_state_of_charge)
        + coefficients.throughput_current_coefficient_per_a
        * (safe_magnitude - coefficients.reference_current_magnitude_a)
    )
    calendar_rate = coefficients.calendar_rate_per_s * jnp.exp(calendar_log_stress)
    throughput_rate_per_c = coefficients.throughput_rate_per_c * jnp.exp(
        throughput_log_stress
    )
    calendar_increment_raw = _trapezoid(calendar_rate, safe_gaps)
    throughput_increment_raw = _trapezoid(
        throughput_rate_per_c * safe_magnitude, safe_gaps
    )
    charge_throughput = _trapezoid(magnitude, safe_gaps)
    candidate_throughput = state.charge_throughput_c + charge_throughput
    throughput_valid = (
        jnp.isfinite(candidate_throughput)
        & (state.charge_throughput_c >= support.charge_throughput_bounds_c[0])
        & (candidate_throughput >= support.charge_throughput_bounds_c[0])
        & (candidate_throughput <= support.charge_throughput_bounds_c[1])
    )
    rate_domain_valid = (
        state_valid
        & times_valid
        & gaps_valid
        & macrostep_valid
        & temperature_valid
        & state_of_charge_valid
        & current_valid
        & throughput_valid
    )
    calendar_increment = jnp.where(
        rate_domain_valid, calendar_increment_raw, jnp.zeros((), dtype=dtype)
    )
    throughput_increment = jnp.where(
        rate_domain_valid, throughput_increment_raw, jnp.zeros((), dtype=dtype)
    )
    candidate = EmpiricalAgeingState(
        state.calendar_exposure + calendar_increment,
        state.throughput_exposure + throughput_increment,
        jnp.where(times_valid, times[-1], state.time_s),
        candidate_throughput,
    )
    candidate_observables = empirical_ageing_observables(coefficients, candidate)
    transition_finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    candidate.calendar_exposure,
                    candidate.throughput_exposure,
                    candidate.time_s,
                    candidate.charge_throughput_c,
                    candidate_observables.calendar_damage,
                    candidate_observables.throughput_damage,
                    candidate_observables.total_damage,
                    candidate_observables.capacity_loss_fraction,
                    candidate_observables.resistance_growth_fraction,
                    candidate_observables.state_of_health,
                )
            )
        )
    )

    status = jnp.asarray(int(EmpiricalAgeingStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(~state_valid, int(EmpiricalAgeingStatus.INVALID_STATE), status)
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~times_valid,
        int(EmpiricalAgeingStatus.INVALID_TIMES),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~gaps_valid,
        int(EmpiricalAgeingStatus.TIME_GAP_EXCEEDED),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~macrostep_valid,
        int(EmpiricalAgeingStatus.MACROSTEP_EXCEEDED),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~temperature_valid,
        int(EmpiricalAgeingStatus.TEMPERATURE_OUT_OF_SUPPORT),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~state_of_charge_valid,
        int(EmpiricalAgeingStatus.STATE_OF_CHARGE_OUT_OF_SUPPORT),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~current_valid,
        int(EmpiricalAgeingStatus.CURRENT_OUT_OF_SUPPORT),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~throughput_valid,
        int(EmpiricalAgeingStatus.THROUGHPUT_OUT_OF_SUPPORT),
        status,
    )
    status = jnp.where(
        (status == int(EmpiricalAgeingStatus.SUCCESS)) & ~transition_finite,
        int(EmpiricalAgeingStatus.NONFINITE_TRANSITION),
        status,
    )
    successful = status == int(EmpiricalAgeingStatus.SUCCESS)
    accepted = _accepted_state(successful, candidate, state)
    observables = empirical_ageing_observables(coefficients, accepted)

    safe_duration = jnp.where(
        jnp.isfinite(duration) & (duration > 0.0), duration, jnp.ones((), dtype=dtype)
    )
    summary = EmpiricalAgeingStressSummary(
        duration,
        _trapezoid(temperature, safe_gaps) / safe_duration,
        _trapezoid(state_of_charge, safe_gaps) / safe_duration,
        _trapezoid(magnitude, safe_gaps) / safe_duration,
        charge_throughput,
        calendar_increment,
        throughput_increment,
    )
    return EmpiricalAgeingTransitionResult(
        candidate,
        accepted,
        summary,
        observables,
        status,
    )


__all__ = [
    "EmpiricalAgeingCoefficients",
    "EmpiricalAgeingObservables",
    "EmpiricalAgeingState",
    "EmpiricalAgeingStatus",
    "EmpiricalAgeingStressHistory",
    "EmpiricalAgeingStressSummary",
    "EmpiricalAgeingSupport",
    "EmpiricalAgeingTopology",
    "EmpiricalAgeingTransitionResult",
    "advance_empirical_ageing",
    "empirical_ageing_observables",
    "initial_empirical_ageing_state",
]
