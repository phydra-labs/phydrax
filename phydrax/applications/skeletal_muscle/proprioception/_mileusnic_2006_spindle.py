#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mileusnic--Brown--Lan--Loeb 2006 feline muscle-spindle model."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


MILEUSNIC_SPINDLE_2006_DOI = "10.1152/jn.00868.2005"
_BRANCHES = ("bag1", "bag2", "chain")


class MileusnicSpindleStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    INVALID_STATE = 2
    INVALID_INPUT = 4
    INVALID_STEP = 8
    INVALID_PARAMETERS = 16


def _scalar(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


class MileusnicSpindle2006Parameters(StrictModule):
    """Source Table-1 parameters; force-unit terms remain arbitrary and scalable."""

    series_stiffness: Array
    polar_stiffness: Array
    polar_mass: Array
    velocity_exponent: Array
    lengthening_damping: Array
    shortening_damping: Array
    polar_rest_length: Array
    sensory_rest_length: Array
    polar_offset: Array
    sensory_threshold: Array
    beta_zero: Array
    beta_dynamic: Array
    beta_static: Array
    gamma_dynamic: Array
    gamma_static: Array
    primary_gain_pps: Array
    secondary_gain_pps: Array
    secondary_mix: Array
    secondary_polar_threshold: Array
    secondary_region_length: Array
    primary_occlusion: Array
    dynamic_half_frequency_pps: Array
    static_half_frequency_pps: Array
    dynamic_time_constant_s: Array
    static_time_constant_s: Array

    @classmethod
    def feline_published(cls) -> MileusnicSpindle2006Parameters:
        return cls(
            jnp.asarray(10.4649),
            jnp.asarray(0.1500),
            jnp.asarray(0.0002),
            jnp.asarray(0.3),
            jnp.asarray(1.0),
            jnp.asarray(0.4200),
            jnp.asarray(0.76),
            jnp.asarray(0.04),
            jnp.asarray(0.46),
            jnp.asarray(0.0423),
            jnp.asarray((0.0605, 0.0822, 0.0822)),
            jnp.asarray((0.2592, 0.0, 0.0)),
            jnp.asarray((0.0, 0.0460, 0.0690)),
            jnp.asarray((0.0289, 0.0, 0.0)),
            jnp.asarray((0.0, 0.0636, 0.0954)),
            jnp.asarray((20_000.0, 10_000.0, 10_000.0)),
            jnp.asarray((0.0, 7_250.0, 7_250.0)),
            jnp.asarray(0.7),
            jnp.asarray(0.89),
            jnp.asarray(0.04),
            jnp.asarray(0.156),
            jnp.asarray(60.0),
            jnp.asarray((60.0, 90.0)),
            jnp.asarray(0.149),
            jnp.asarray(0.205),
        )

def _parameters_admissible(parameters: MileusnicSpindle2006Parameters, /) -> Array:
    vector_values = (
        parameters.beta_zero,
        parameters.beta_dynamic,
        parameters.beta_static,
        parameters.gamma_dynamic,
        parameters.gamma_static,
        parameters.primary_gain_pps,
        parameters.secondary_gain_pps,
        parameters.static_half_frequency_pps,
    )
    scalar_values = (
        parameters.series_stiffness,
        parameters.polar_stiffness,
        parameters.polar_mass,
        parameters.velocity_exponent,
        parameters.lengthening_damping,
        parameters.shortening_damping,
        parameters.polar_rest_length,
        parameters.sensory_rest_length,
        parameters.polar_offset,
        parameters.sensory_threshold,
        parameters.secondary_mix,
        parameters.secondary_polar_threshold,
        parameters.secondary_region_length,
        parameters.primary_occlusion,
        parameters.dynamic_half_frequency_pps,
        parameters.dynamic_time_constant_s,
        parameters.static_time_constant_s,
    )
    finite = jnp.all(
        jnp.stack(
            tuple(jnp.all(jnp.isfinite(value)) for value in vector_values)
            + tuple(jnp.isfinite(value) for value in scalar_values)
        )
    )
    return (
        finite
        & (parameters.series_stiffness > 0.0)
        & (parameters.polar_stiffness > 0.0)
        & (parameters.polar_mass > 0.0)
        & (parameters.velocity_exponent > 0.0)
        & (parameters.lengthening_damping > 0.0)
        & (parameters.shortening_damping > 0.0)
        & jnp.all(parameters.beta_zero >= 0.0)
        & jnp.all(parameters.beta_dynamic >= 0.0)
        & jnp.all(parameters.beta_static >= 0.0)
        & jnp.all(parameters.gamma_dynamic >= 0.0)
        & jnp.all(parameters.gamma_static >= 0.0)
        & jnp.all(parameters.primary_gain_pps >= 0.0)
        & jnp.all(parameters.secondary_gain_pps >= 0.0)
        & (parameters.secondary_mix >= 0.0)
        & (parameters.secondary_mix <= 1.0)
        & (parameters.primary_occlusion >= 0.0)
        & (parameters.primary_occlusion <= 1.0)
        & (parameters.dynamic_half_frequency_pps > 0.0)
        & jnp.all(parameters.static_half_frequency_pps > 0.0)
        & (parameters.dynamic_time_constant_s > 0.0)
        & (parameters.static_time_constant_s > 0.0)
    )


def _validate_parameters_host(parameters: MileusnicSpindle2006Parameters, /) -> None:
    three_vectors = (
        parameters.beta_zero,
        parameters.beta_dynamic,
        parameters.beta_static,
        parameters.gamma_dynamic,
        parameters.gamma_static,
        parameters.primary_gain_pps,
        parameters.secondary_gain_pps,
    )
    if any(value.shape != (3,) for value in three_vectors):
        raise ValueError("Branch parameter vectors must have shape (3,).")
    if parameters.static_half_frequency_pps.shape != (2,):
        raise ValueError("static_half_frequency_pps must have shape (2,).")
    if not bool(np.asarray(_parameters_admissible(parameters))):
        raise ValueError("Mileusnic spindle parameters violate the model domain.")


class MileusnicSpindleInput(StrictModule, NonTrainableState):
    """Normalized fascicle kinematics and fusimotor frequencies."""

    fascicle_length_over_optimal: Array
    fascicle_velocity_per_s: Array
    fascicle_acceleration_per_s2: Array
    gamma_dynamic_pps: Array
    gamma_static_pps: Array

    def __init__(
        self,
        fascicle_length_over_optimal: ArrayLike,
        fascicle_velocity_per_s: ArrayLike,
        fascicle_acceleration_per_s2: ArrayLike,
        gamma_dynamic_pps: ArrayLike,
        gamma_static_pps: ArrayLike,
        /,
    ):
        self.fascicle_length_over_optimal = _scalar(
            fascicle_length_over_optimal, "fascicle_length_over_optimal"
        )
        self.fascicle_velocity_per_s = _scalar(
            fascicle_velocity_per_s, "fascicle_velocity_per_s"
        )
        self.fascicle_acceleration_per_s2 = _scalar(
            fascicle_acceleration_per_s2, "fascicle_acceleration_per_s2"
        )
        self.gamma_dynamic_pps = _scalar(gamma_dynamic_pps, "gamma_dynamic_pps")
        self.gamma_static_pps = _scalar(gamma_static_pps, "gamma_static_pps")


class MileusnicSpindleState(StrictModule, NonTrainableState):
    branch_tension_force_unit: Array
    branch_tension_rate_force_unit_per_s: Array
    bag1_dynamic_activation: Array
    bag2_static_activation: Array


class MileusnicSpindleRates(StrictModule, NonTrainableState):
    branch_tension_rate_force_unit_per_s: Array
    branch_tension_acceleration_force_unit_per_s2: Array
    bag1_dynamic_activation_per_s: Array
    bag2_static_activation_per_s: Array


class MileusnicSpindleOutput(StrictModule, NonTrainableState):
    primary_branch_pps: Array
    secondary_branch_pps: Array
    primary_afferent_pps: Array
    secondary_afferent_pps: Array
    polar_length: Array
    polar_velocity_per_s: Array
    dynamic_activation: Array
    static_activation: Array


class MileusnicSpindleEvidence(StrictModule, NonTrainableState):
    status: Array
    finite: Array
    state_admissible: Array
    input_admissible: Array
    parameters_admissible: Array
    step_admissible: Array
    source_doi: str = eqx.field(static=True, default=MILEUSNIC_SPINDLE_2006_DOI)
    species_scope: str = eqx.field(
        static=True, default="feline soleus fit; feline medial-gastrocnemius validation"
    )

    @property
    def successful(self) -> Array:
        return self.status == int(MileusnicSpindleStatus.SUCCESS)


class MileusnicSpindleCandidate(StrictModule, NonTrainableState):
    source_state: MileusnicSpindleState
    candidate_state: MileusnicSpindleState
    source_output: MileusnicSpindleOutput
    evidence: MileusnicSpindleEvidence

    def commit(self, /) -> MileusnicSpindleState:
        accepted = self.evidence.successful
        return MileusnicSpindleState(
            jnp.where(
                accepted,
                self.candidate_state.branch_tension_force_unit,
                self.source_state.branch_tension_force_unit,
            ),
            jnp.where(
                accepted,
                self.candidate_state.branch_tension_rate_force_unit_per_s,
                self.source_state.branch_tension_rate_force_unit_per_s,
            ),
            jnp.where(
                accepted,
                self.candidate_state.bag1_dynamic_activation,
                self.source_state.bag1_dynamic_activation,
            ),
            jnp.where(
                accepted,
                self.candidate_state.bag2_static_activation,
                self.source_state.bag2_static_activation,
            ),
        )


class MileusnicSpindle2006Plan(StrictModule, NonTrainableState):
    maximum_step_s: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_step_s: float = 1.0e-4):
        maximum = float(maximum_step_s)
        if not isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_step_s must be positive and finite.")
        self.maximum_step_s = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mileusnic-spindle-2006-plan",
                "source_doi": MILEUSNIC_SPINDLE_2006_DOI,
                "branches": _BRANCHES,
                "maximum_step_s": maximum.hex(),
                "force_scale": "arbitrary-force-unit",
            }
        )

    def prepare(
        self, parameters: MileusnicSpindle2006Parameters | None = None, /
    ) -> PreparedMileusnicSpindle2006:
        return PreparedMileusnicSpindle2006(
            self,
            MileusnicSpindle2006Parameters.feline_published()
            if parameters is None
            else parameters,
        )


class PreparedMileusnicSpindle2006(StrictModule):
    plan: MileusnicSpindle2006Plan
    parameters: MileusnicSpindle2006Parameters
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MileusnicSpindle2006Plan,
        parameters: MileusnicSpindle2006Parameters,
        /,
    ):
        if not isinstance(plan, MileusnicSpindle2006Plan):
            raise TypeError("plan must be MileusnicSpindle2006Plan.")
        if not isinstance(parameters, MileusnicSpindle2006Parameters):
            raise TypeError("parameters must be MileusnicSpindle2006Parameters.")
        _validate_parameters_host(parameters)
        self.plan = plan
        self.parameters = parameters
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-mileusnic-spindle-2006", "plan": plan.plan_id}
        )

    def initialize(
        self,
        input_value: MileusnicSpindleInput | None = None,
        /,
    ) -> MileusnicSpindleState:
        value = (
            MileusnicSpindleInput(1.0, 0.0, 0.0, 0.0, 0.0)
            if input_value is None
            else input_value
        )
        dynamic_target, static_target, chain_target = self._fusimotor_targets(value)
        active_force = (
            self.parameters.gamma_dynamic * dynamic_target
            + self.parameters.gamma_static
            * jnp.asarray((0.0, static_target, chain_target))
        )
        numerator = (
            self.parameters.polar_stiffness
            * (
                value.fascicle_length_over_optimal
                - self.parameters.sensory_rest_length
                - self.parameters.polar_rest_length
            )
            + active_force
        )
        denominator = 1.0 + self.parameters.polar_stiffness / self.parameters.series_stiffness
        tension = numerator / denominator
        return MileusnicSpindleState(
            tension,
            jnp.zeros((3,), dtype=tension.dtype),
            dynamic_target,
            static_target,
        )

    def _fusimotor_targets(
        self, value: MileusnicSpindleInput, /
    ) -> tuple[Array, Array, Array]:
        dynamic_square = value.gamma_dynamic_pps**2
        static_square = value.gamma_static_pps**2
        dynamic = dynamic_square / (
            dynamic_square + self.parameters.dynamic_half_frequency_pps**2
        )
        static = static_square / (
            static_square + self.parameters.static_half_frequency_pps[0] ** 2
        )
        chain = static_square / (
            static_square + self.parameters.static_half_frequency_pps[1] ** 2
        )
        return dynamic, static, chain

    def rates(
        self,
        state: MileusnicSpindleState,
        input_value: MileusnicSpindleInput,
        /,
    ) -> MileusnicSpindleRates:
        dynamic_target, static_target, chain_target = self._fusimotor_targets(
            input_value
        )
        dynamic_activation = jnp.asarray(
            (state.bag1_dynamic_activation, 0.0, 0.0)
        )
        static_activation = jnp.asarray(
            (0.0, state.bag2_static_activation, chain_target)
        )
        beta = (
            self.parameters.beta_zero
            + self.parameters.beta_dynamic * dynamic_activation
            + self.parameters.beta_static * static_activation
        )
        active_force = (
            self.parameters.gamma_dynamic * dynamic_activation
            + self.parameters.gamma_static * static_activation
        )
        polar_length = (
            input_value.fascicle_length_over_optimal
            - self.parameters.sensory_rest_length
            - state.branch_tension_force_unit / self.parameters.series_stiffness
        )
        polar_velocity = (
            input_value.fascicle_velocity_per_s
            - state.branch_tension_rate_force_unit_per_s
            / self.parameters.series_stiffness
        )
        damping_direction = jnp.where(
            polar_velocity >= 0.0,
            self.parameters.lengthening_damping,
            self.parameters.shortening_damping,
        )
        damping = (
            beta
            * damping_direction
            * jnp.sign(polar_velocity)
            * jnp.abs(polar_velocity) ** self.parameters.velocity_exponent
            * (polar_length - self.parameters.polar_offset)
        )
        passive = self.parameters.polar_stiffness * (
            polar_length - self.parameters.polar_rest_length
        )
        tension_acceleration = self.parameters.series_stiffness * (
            input_value.fascicle_acceleration_per_s2
            - (
                state.branch_tension_force_unit - damping - passive - active_force
            )
            / self.parameters.polar_mass
        )
        return MileusnicSpindleRates(
            state.branch_tension_rate_force_unit_per_s,
            tension_acceleration,
            (dynamic_target - state.bag1_dynamic_activation)
            / self.parameters.dynamic_time_constant_s,
            (static_target - state.bag2_static_activation)
            / self.parameters.static_time_constant_s,
        )

    def output(
        self,
        state: MileusnicSpindleState,
        input_value: MileusnicSpindleInput,
        /,
    ) -> MileusnicSpindleOutput:
        dynamic, static, chain = self._fusimotor_targets(input_value)
        polar_length = (
            input_value.fascicle_length_over_optimal
            - self.parameters.sensory_rest_length
            - state.branch_tension_force_unit / self.parameters.series_stiffness
        )
        polar_velocity = (
            input_value.fascicle_velocity_per_s
            - state.branch_tension_rate_force_unit_per_s
            / self.parameters.series_stiffness
        )
        sensory_stretch = jnp.maximum(
            state.branch_tension_force_unit / self.parameters.series_stiffness
            + self.parameters.sensory_rest_length
            - self.parameters.sensory_threshold,
            0.0,
        )
        primary_branch = self.parameters.primary_gain_pps * sensory_stretch
        first = primary_branch[0]
        remaining = primary_branch[1] + primary_branch[2]
        primary = jnp.maximum(first, remaining) + self.parameters.primary_occlusion * jnp.minimum(
            first, remaining
        )
        polar_stretch = jnp.maximum(
            polar_length - self.parameters.secondary_polar_threshold,
            0.0,
        )
        secondary_branch = self.parameters.secondary_gain_pps * (
            self.parameters.secondary_mix
            * sensory_stretch
            * self.parameters.secondary_region_length
            / self.parameters.sensory_rest_length
            + (1.0 - self.parameters.secondary_mix)
            * polar_stretch
            * self.parameters.secondary_region_length
            / self.parameters.polar_rest_length
        )
        secondary = secondary_branch[1] + secondary_branch[2]
        return MileusnicSpindleOutput(
            primary_branch,
            secondary_branch,
            primary,
            secondary,
            polar_length,
            polar_velocity,
            dynamic,
            jnp.asarray((0.0, static, chain)),
        )

    def candidate(
        self,
        state: MileusnicSpindleState,
        input_value: MileusnicSpindleInput,
        step_s: ArrayLike,
        /,
    ) -> MileusnicSpindleCandidate:
        step = _scalar(step_s, "step_s")
        output = self.output(state, input_value)
        step_valid = jnp.isfinite(step) & (step > 0.0) & (
            step <= self.plan.maximum_step_s
        )
        input_values = jnp.stack(
            (
                input_value.fascicle_length_over_optimal,
                input_value.fascicle_velocity_per_s,
                input_value.fascicle_acceleration_per_s2,
                input_value.gamma_dynamic_pps,
                input_value.gamma_static_pps,
            )
        )
        input_valid = (
            jnp.all(jnp.isfinite(input_values))
            & (input_value.fascicle_length_over_optimal > 0.0)
            & (input_value.gamma_dynamic_pps >= 0.0)
            & (input_value.gamma_static_pps >= 0.0)
        )
        state_values = jnp.concatenate(
            (
                state.branch_tension_force_unit,
                state.branch_tension_rate_force_unit_per_s,
                jnp.asarray(
                    (
                        state.bag1_dynamic_activation,
                        state.bag2_static_activation,
                    )
                ),
            )
        )
        state_valid = (
            jnp.all(jnp.isfinite(state_values))
            & (state.bag1_dynamic_activation >= 0.0)
            & (state.bag1_dynamic_activation <= 1.0)
            & (state.bag2_static_activation >= 0.0)
            & (state.bag2_static_activation <= 1.0)
        )
        parameters_valid = _parameters_admissible(self.parameters)
        safe_step = jnp.where(step_valid, step, 0.0)
        proposed = _rk4_state(self, state, input_value, safe_step)
        candidate_values = jnp.concatenate(
            (
                proposed.branch_tension_force_unit,
                proposed.branch_tension_rate_force_unit_per_s,
                jnp.asarray(
                    (
                        proposed.bag1_dynamic_activation,
                        proposed.bag2_static_activation,
                    )
                ),
            )
        )
        finite = jnp.all(jnp.isfinite(candidate_values)) & jnp.all(
            jnp.isfinite(output.primary_branch_pps)
        )
        status = jnp.asarray(int(MileusnicSpindleStatus.SUCCESS), dtype=jnp.int32)
        status |= jnp.where(
            finite, 0, int(MileusnicSpindleStatus.NONFINITE)
        ).astype(jnp.int32)
        status |= jnp.where(
            state_valid, 0, int(MileusnicSpindleStatus.INVALID_STATE)
        ).astype(jnp.int32)
        status |= jnp.where(
            input_valid, 0, int(MileusnicSpindleStatus.INVALID_INPUT)
        ).astype(jnp.int32)
        status |= jnp.where(
            parameters_valid, 0, int(MileusnicSpindleStatus.INVALID_PARAMETERS)
        ).astype(jnp.int32)
        status |= jnp.where(
            step_valid, 0, int(MileusnicSpindleStatus.INVALID_STEP)
        ).astype(jnp.int32)
        evidence = MileusnicSpindleEvidence(
            status,
            finite,
            state_valid,
            input_valid,
            parameters_valid,
            step_valid,
        )
        return MileusnicSpindleCandidate(state, proposed, output, evidence)


def _state_increment(
    state: MileusnicSpindleState,
    rates: MileusnicSpindleRates,
    scale: Array,
    /,
) -> MileusnicSpindleState:
    return MileusnicSpindleState(
        state.branch_tension_force_unit
        + scale * rates.branch_tension_rate_force_unit_per_s,
        state.branch_tension_rate_force_unit_per_s
        + scale * rates.branch_tension_acceleration_force_unit_per_s2,
        state.bag1_dynamic_activation
        + scale * rates.bag1_dynamic_activation_per_s,
        state.bag2_static_activation + scale * rates.bag2_static_activation_per_s,
    )


def _rk4_state(
    prepared: PreparedMileusnicSpindle2006,
    state: MileusnicSpindleState,
    input_value: MileusnicSpindleInput,
    step: Array,
    /,
) -> MileusnicSpindleState:
    first = prepared.rates(state, input_value)
    second = prepared.rates(_state_increment(state, first, 0.5 * step), input_value)
    third = prepared.rates(_state_increment(state, second, 0.5 * step), input_value)
    fourth = prepared.rates(_state_increment(state, third, step), input_value)
    return MileusnicSpindleState(
        state.branch_tension_force_unit
        + step
        / 6.0
        * (
            first.branch_tension_rate_force_unit_per_s
            + 2.0 * second.branch_tension_rate_force_unit_per_s
            + 2.0 * third.branch_tension_rate_force_unit_per_s
            + fourth.branch_tension_rate_force_unit_per_s
        ),
        state.branch_tension_rate_force_unit_per_s
        + step
        / 6.0
        * (
            first.branch_tension_acceleration_force_unit_per_s2
            + 2.0 * second.branch_tension_acceleration_force_unit_per_s2
            + 2.0 * third.branch_tension_acceleration_force_unit_per_s2
            + fourth.branch_tension_acceleration_force_unit_per_s2
        ),
        state.bag1_dynamic_activation
        + step
        / 6.0
        * (
            first.bag1_dynamic_activation_per_s
            + 2.0 * second.bag1_dynamic_activation_per_s
            + 2.0 * third.bag1_dynamic_activation_per_s
            + fourth.bag1_dynamic_activation_per_s
        ),
        state.bag2_static_activation
        + step
        / 6.0
        * (
            first.bag2_static_activation_per_s
            + 2.0 * second.bag2_static_activation_per_s
            + 2.0 * third.bag2_static_activation_per_s
            + fourth.bag2_static_activation_per_s
        ),
    )


__all__ = [
    "MILEUSNIC_SPINDLE_2006_DOI",
    "MileusnicSpindle2006Parameters",
    "MileusnicSpindle2006Plan",
    "MileusnicSpindleCandidate",
    "MileusnicSpindleEvidence",
    "MileusnicSpindleInput",
    "MileusnicSpindleOutput",
    "MileusnicSpindleRates",
    "MileusnicSpindleState",
    "MileusnicSpindleStatus",
    "PreparedMileusnicSpindle2006",
]
