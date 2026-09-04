#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Potvin--Fuglevand deterministic mean-rate isometric motor-unit fatigue."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....dynamics import (
    DiscreteSystem,
    DiscreteTransitionResult,
    InputLayout,
    StateLayout,
)


POTVIN_FUGLEVAND_2017_DOI = "10.1371/journal.pcbi.1005581"
POTVIN_FUGLEVAND_2017_REFERENCE_SHA = (
    "15462f85106ed9ebde3d78ab6fe665c88bf8b32e"
)
POTVIN_FUGLEVAND_2017_MODEL_ID = "potvin-fuglevand-2017-sustained-isometric"
_PARAMETER_SCHEMA_ID = canonical_fingerprint(
    {
        "kind": "potvin-fuglevand-2017-parameter-schema",
        "source": POTVIN_FUGLEVAND_2017_DOI,
        "fields": (
            "recruitment_threshold",
            "rested_twitch_force",
            "resting_contraction_time_s",
            "maximum_firing_rate_hz",
            "nominal_twitch_force_loss_per_s",
            "minimum_firing_rate_hz",
            "firing_rate_gain_hz",
            "derecruitment_delta_hz",
            "adaptation_scale",
            "adaptation_time_constant_s",
            "contraction_time_change_ratio",
        ),
    }
)


class PotvinFuglevand2017Status(IntFlag):
    """Failure bits for one candidate population transition."""

    SUCCESS = 0
    NONFINITE = 1
    INVALID_STATE = 2
    INVALID_PARAMETERS = 4
    INVALID_EXCITATION = 8
    INVALID_STEP = 16


def _scalar(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return result if jnp.issubdtype(result.dtype, jnp.inexact) else result.astype(float)


def _vector(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{name} must be one nonempty motor-unit vector.")
    return result if jnp.issubdtype(result.dtype, jnp.inexact) else result.astype(float)


class PotvinFuglevand2017Parameters(StrictModule):
    """Numeric model parameters; every array remains an optimizable JAX leaf."""

    recruitment_threshold: Array
    rested_twitch_force: Array
    resting_contraction_time_s: Array
    maximum_firing_rate_hz: Array
    nominal_twitch_force_loss_per_s: Array
    minimum_firing_rate_hz: Array
    firing_rate_gain_hz: Array
    derecruitment_delta_hz: Array
    adaptation_scale: Array
    adaptation_time_constant_s: Array
    contraction_time_change_ratio: Array

    def __init__(
        self,
        recruitment_threshold: ArrayLike,
        rested_twitch_force: ArrayLike,
        resting_contraction_time_s: ArrayLike,
        maximum_firing_rate_hz: ArrayLike,
        nominal_twitch_force_loss_per_s: ArrayLike,
        /,
        *,
        minimum_firing_rate_hz: ArrayLike,
        firing_rate_gain_hz: ArrayLike,
        derecruitment_delta_hz: ArrayLike,
        adaptation_scale: ArrayLike,
        adaptation_time_constant_s: ArrayLike,
        contraction_time_change_ratio: ArrayLike,
    ):
        vectors = (
            _vector(recruitment_threshold, "recruitment_threshold"),
            _vector(rested_twitch_force, "rested_twitch_force"),
            _vector(resting_contraction_time_s, "resting_contraction_time_s"),
            _vector(maximum_firing_rate_hz, "maximum_firing_rate_hz"),
            _vector(
                nominal_twitch_force_loss_per_s,
                "nominal_twitch_force_loss_per_s",
            ),
        )
        shape = vectors[0].shape
        if any(value.shape != shape for value in vectors[1:]):
            raise ValueError("All per-motor-unit parameter vectors must agree in shape.")
        (
            self.recruitment_threshold,
            self.rested_twitch_force,
            self.resting_contraction_time_s,
            self.maximum_firing_rate_hz,
            self.nominal_twitch_force_loss_per_s,
        ) = vectors
        self.minimum_firing_rate_hz = _scalar(
            minimum_firing_rate_hz, "minimum_firing_rate_hz"
        )
        self.firing_rate_gain_hz = _scalar(
            firing_rate_gain_hz, "firing_rate_gain_hz"
        )
        self.derecruitment_delta_hz = _scalar(
            derecruitment_delta_hz, "derecruitment_delta_hz"
        )
        self.adaptation_scale = _scalar(adaptation_scale, "adaptation_scale")
        self.adaptation_time_constant_s = _scalar(
            adaptation_time_constant_s, "adaptation_time_constant_s"
        )
        self.contraction_time_change_ratio = _scalar(
            contraction_time_change_ratio, "contraction_time_change_ratio"
        )

    @property
    def unit_count(self) -> int:
        return self.recruitment_threshold.shape[0]

    @property
    def schema_id(self) -> str:
        return _PARAMETER_SCHEMA_ID


class PotvinFuglevand2017State(StrictModule, NonTrainableState):
    """Independent population state at one accepted protocol time."""

    recruitment_duration_s: Array
    current_twitch_force: Array

    def __init__(
        self,
        recruitment_duration_s: ArrayLike,
        current_twitch_force: ArrayLike,
        /,
    ):
        duration = _vector(recruitment_duration_s, "recruitment_duration_s")
        capacity = _vector(current_twitch_force, "current_twitch_force")
        if duration.shape != capacity.shape:
            raise ValueError("Motor-unit state vectors must agree in shape.")
        self.recruitment_duration_s = duration
        self.current_twitch_force = capacity


class PotvinFuglevand2017Output(StrictModule, NonTrainableState):
    """Per-unit and aggregate outputs evaluated at the interval source state."""

    unadapted_firing_rate_hz: Array
    firing_rate_adaptation_hz: Array
    firing_rate_hz: Array
    contraction_time_s: Array
    normalized_firing_rate: Array
    normalized_force: Array
    motor_unit_force: Array
    total_force: Array
    force_capacity_fraction: Array
    total_force_capacity_fraction: Array
    recruited: Array
    saturated: Array
    maximum_excitation: Array


class PotvinFuglevand2017Evidence(StrictModule, NonTrainableState):
    """Validity, branch-margin, and aggregation evidence for one transition."""

    status: Array
    finite: Array
    state_admissible: Array
    parameters_admissible: Array
    excitation_admissible: Array
    step_admissible: Array
    force_aggregation_residual: Array
    minimum_recruitment_margin: Array
    minimum_saturation_margin: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(PotvinFuglevand2017Status.SUCCESS)


class PotvinFuglevand2017Candidate(StrictModule, NonTrainableState):
    """Source and candidate states plus source-time outputs and evidence."""

    source_state: PotvinFuglevand2017State
    candidate_state: PotvinFuglevand2017State
    output: PotvinFuglevand2017Output
    evidence: PotvinFuglevand2017Evidence

    def commit(self, /) -> PotvinFuglevand2017State:
        """Commit the complete candidate exactly when all evidence succeeds."""
        accept = self.evidence.successful
        return PotvinFuglevand2017State(
            jnp.where(
                accept,
                self.candidate_state.recruitment_duration_s,
                self.source_state.recruitment_duration_s,
            ),
            jnp.where(
                accept,
                self.candidate_state.current_twitch_force,
                self.source_state.current_twitch_force,
            ),
        )


class PotvinFuglevand2017Plan(StrictModule, NonTrainableState):
    """Static identity and transition policy for the published isometric model."""

    unit_count: int = eqx.field(static=True)
    central_adaptation: bool = eqx.field(static=True)
    peripheral_fatigue: bool = eqx.field(static=True)
    maximum_step_s: float = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        unit_count: int = 120,
        /,
        *,
        central_adaptation: bool = True,
        peripheral_fatigue: bool = True,
        maximum_step_s: float = 0.1,
        dtype: Any = np.float64,
    ):
        if isinstance(unit_count, bool) or not isinstance(unit_count, int):
            raise TypeError("unit_count must be an integer.")
        if unit_count < 2:
            raise ValueError("Potvin--Fuglevand populations require at least two units.")
        if not isinstance(central_adaptation, bool) or not isinstance(
            peripheral_fatigue, bool
        ):
            raise TypeError("Fatigue mechanism selections must be boolean.")
        maximum_step = float(maximum_step_s)
        if not isfinite(maximum_step) or maximum_step <= 0.0:
            raise ValueError("maximum_step_s must be positive and finite.")
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.kind != "f":
            raise TypeError("Potvin--Fuglevand dtype must be floating point.")
        self.unit_count = unit_count
        self.central_adaptation = central_adaptation
        self.peripheral_fatigue = peripheral_fatigue
        self.maximum_step_s = maximum_step
        self.dtype = resolved_dtype
        self.plan_id = canonical_fingerprint(
            {
                "kind": POTVIN_FUGLEVAND_2017_MODEL_ID,
                "source_doi": POTVIN_FUGLEVAND_2017_DOI,
                "reference_sha": POTVIN_FUGLEVAND_2017_REFERENCE_SHA,
                "unit_count": unit_count,
                "central_adaptation": central_adaptation,
                "peripheral_fatigue": peripheral_fatigue,
                "maximum_step_s": maximum_step,
                "dtype": resolved_dtype.str,
            }
        )

    @property
    def state_layout(self) -> StateLayout:
        names = tuple(
            f"{field}[{index}]"
            for field in ("recruitment_duration_s", "current_twitch_force")
            for index in range(self.unit_count)
        )
        return StateLayout(
            (2, self.unit_count),
            axes=("state_component", "motor_unit"),
            component_names=names,
            layout_id=f"{self.plan_id}:state",
        )

    @property
    def input_layout(self) -> InputLayout:
        return InputLayout(
            (1,),
            axes=("control",),
            component_names=("common_excitation",),
            roles="control",
            layout_id=f"{self.plan_id}:input",
        )

    def prepare(
        self,
        parameters: PotvinFuglevand2017Parameters | None = None,
        /,
    ) -> PreparedPotvinFuglevand2017:
        selected = (
            potvin_fuglevand_2017_default_parameters(
                self.unit_count, dtype=self.dtype
            )
            if parameters is None
            else parameters
        )
        return PreparedPotvinFuglevand2017(self, selected)

    def as_discrete_system(self, /) -> DiscreteSystem:
        """Return the canonical one-population array-state dynamics view."""
        plan = self

        def transition(context, packed_state, inputs, parameters):
            if not isinstance(parameters, PotvinFuglevand2017Parameters):
                raise TypeError(
                    "Potvin--Fuglevand DiscreteSystem args must be model parameters."
                )
            state = _unpack_state(plan, packed_state)
            candidate = _candidate(
                plan,
                parameters,
                state,
                inputs[0],
                context.duration,
            )
            return DiscreteTransitionResult(
                _pack_state(plan, candidate.candidate_state),
                _pack_state(plan, candidate.commit()),
                candidate.evidence.successful,
                candidate.evidence.status,
            )

        return DiscreteSystem(
            transition,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=self.plan_id,
            maximum_step_size=self.maximum_step_s,
        )


class PreparedPotvinFuglevand2017(StrictModule):
    """Prepared model with fixed topology and differentiable numeric parameters."""

    plan: PotvinFuglevand2017Plan
    parameters: PotvinFuglevand2017Parameters
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PotvinFuglevand2017Plan,
        parameters: PotvinFuglevand2017Parameters,
        /,
    ):
        if not isinstance(plan, PotvinFuglevand2017Plan):
            raise TypeError("plan must be PotvinFuglevand2017Plan.")
        if not isinstance(parameters, PotvinFuglevand2017Parameters):
            raise TypeError("parameters must be PotvinFuglevand2017Parameters.")
        _validate_parameters_host(parameters, plan.unit_count)
        self.plan = plan
        self.parameters = parameters
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-potvin-fuglevand-2017",
                "plan": plan.plan_id,
                "parameter_schema": parameters.schema_id,
            }
        )

    @property
    def state_layout(self) -> StateLayout:
        return self.plan.state_layout

    @property
    def input_layout(self) -> InputLayout:
        return self.plan.input_layout

    @property
    def discrete_system(self) -> DiscreteSystem:
        return self.plan.as_discrete_system()

    def initialize(self, /) -> PotvinFuglevand2017State:
        dtype = self.parameters.rested_twitch_force.dtype
        return PotvinFuglevand2017State(
            jnp.zeros((self.plan.unit_count,), dtype=dtype),
            self.parameters.rested_twitch_force,
        )

    def pack_state(self, state: PotvinFuglevand2017State, /) -> Array:
        return _pack_state(self.plan, state)

    def unpack_state(self, state: ArrayLike, /) -> PotvinFuglevand2017State:
        return _unpack_state(self.plan, state)

    def evaluate(
        self,
        state: PotvinFuglevand2017State,
        common_excitation: ArrayLike,
        /,
    ) -> PotvinFuglevand2017Output:
        _require_state_shape(state, self.plan.unit_count)
        excitation = _scalar(common_excitation, "common_excitation")
        admissible = (
            _parameters_admissible(self.parameters)
            & _state_admissible(self.parameters, state)
            & jnp.isfinite(excitation)
            & (excitation >= 0.0)
            & (excitation <= _maximum_excitation(self.parameters))
        )
        checked_excitation = eqx.error_if(
            excitation,
            ~admissible,
            "Potvin--Fuglevand evaluation inputs are outside the model domain.",
        )
        return _evaluate(
            self.plan,
            self.parameters,
            state,
            checked_excitation,
        )

    def candidate(
        self,
        state: PotvinFuglevand2017State,
        common_excitation: ArrayLike,
        step_s: ArrayLike,
        /,
    ) -> PotvinFuglevand2017Candidate:
        _require_state_shape(state, self.plan.unit_count)
        return _candidate(
            self.plan,
            self.parameters,
            state,
            _scalar(common_excitation, "common_excitation"),
            _scalar(step_s, "step_s"),
        )

    def advance(
        self,
        state: PotvinFuglevand2017State,
        common_excitation: ArrayLike,
        step_s: ArrayLike,
        /,
    ) -> tuple[PotvinFuglevand2017State, PotvinFuglevand2017Output, PotvinFuglevand2017Evidence]:
        """Evaluate, advance, and atomically commit one interval."""
        candidate = self.candidate(state, common_excitation, step_s)
        return candidate.commit(), candidate.output, candidate.evidence

    def maximum_excitation(self, /) -> Array:
        return _maximum_excitation(self.parameters)

    def rested_maximum_force(self, /) -> Array:
        state = self.initialize()
        return self.evaluate(state, self.maximum_excitation()).total_force


def potvin_fuglevand_2017_default_parameters(
    unit_count: int = 120,
    /,
    *,
    dtype: Any = np.float64,
) -> PotvinFuglevand2017Parameters:
    """Construct the exact parameter population used in the 2017 study."""
    if isinstance(unit_count, bool) or not isinstance(unit_count, int):
        raise TypeError("unit_count must be an integer.")
    if unit_count < 2:
        raise ValueError("Potvin--Fuglevand populations require at least two units.")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise TypeError("Potvin--Fuglevand dtype must be floating point.")
    fraction = jnp.arange(unit_count, dtype=resolved_dtype) / (unit_count - 1)

    minimum_threshold = 1.0
    recruitment_range = 50.0
    recruitment_threshold = jnp.exp(jnp.log(recruitment_range) * fraction)
    rested_twitch_force = jnp.exp(jnp.log(100.0) * fraction)
    contraction_scale = jnp.log(100.0) / jnp.log(3.0)
    resting_contraction_time_s = (
        0.09 * (1.0 / rested_twitch_force) ** (1.0 / contraction_scale)
    )
    maximum_firing_rate_hz = 35.0 - 10.0 * (
        (recruitment_threshold - minimum_threshold)
        / (recruitment_range - minimum_threshold)
    )
    fatigability_scale = jnp.exp(jnp.log(180.0) * fraction)
    nominal_twitch_force_loss_per_s = (
        fatigability_scale * (0.0225 / 180.0) * rested_twitch_force
    )
    return PotvinFuglevand2017Parameters(
        recruitment_threshold,
        rested_twitch_force,
        resting_contraction_time_s,
        maximum_firing_rate_hz,
        nominal_twitch_force_loss_per_s,
        minimum_firing_rate_hz=jnp.asarray(8.0, dtype=resolved_dtype),
        firing_rate_gain_hz=jnp.asarray(1.0, dtype=resolved_dtype),
        derecruitment_delta_hz=jnp.asarray(2.0, dtype=resolved_dtype),
        adaptation_scale=jnp.asarray(0.67, dtype=resolved_dtype),
        adaptation_time_constant_s=jnp.asarray(22.0, dtype=resolved_dtype),
        contraction_time_change_ratio=jnp.asarray(0.379, dtype=resolved_dtype),
    )


def _validate_parameters_host(
    parameters: PotvinFuglevand2017Parameters,
    unit_count: int,
    /,
) -> None:
    vectors = tuple(
        np.asarray(value)
        for value in (
            parameters.recruitment_threshold,
            parameters.rested_twitch_force,
            parameters.resting_contraction_time_s,
            parameters.maximum_firing_rate_hz,
            parameters.nominal_twitch_force_loss_per_s,
        )
    )
    scalars = tuple(
        float(np.asarray(value))
        for value in (
            parameters.minimum_firing_rate_hz,
            parameters.firing_rate_gain_hz,
            parameters.derecruitment_delta_hz,
            parameters.adaptation_scale,
            parameters.adaptation_time_constant_s,
            parameters.contraction_time_change_ratio,
        )
    )
    if any(value.shape != (unit_count,) for value in vectors):
        raise ValueError("Parameter vectors must match the plan motor-unit count.")
    if any(np.any(~np.isfinite(value)) for value in vectors) or any(
        not isfinite(value) for value in scalars
    ):
        raise ValueError("Potvin--Fuglevand parameters must be finite.")
    thresholds, rested, contraction, maximum_rate, fatigue = vectors
    minimum_rate, gain, delta, scale, tau, contraction_ratio = scalars
    if (
        np.any(thresholds <= 0.0)
        or np.any(np.diff(thresholds) <= 0.0)
        or np.any(rested <= 0.0)
        or np.any(np.diff(rested) <= 0.0)
        or np.any(contraction <= 0.0)
        or np.any(np.diff(contraction) >= 0.0)
        or minimum_rate <= 0.0
        or gain <= 0.0
        or np.any(maximum_rate < minimum_rate)
        or np.any(fatigue < 0.0)
        or delta < 0.0
        or scale < 0.0
        or tau <= 0.0
        or contraction_ratio < 0.0
    ):
        raise ValueError("Potvin--Fuglevand parameters violate model bounds.")


def _parameters_admissible(parameters: PotvinFuglevand2017Parameters, /) -> Array:
    vector_finite = jnp.asarray(True)
    for value in (
        parameters.recruitment_threshold,
        parameters.rested_twitch_force,
        parameters.resting_contraction_time_s,
        parameters.maximum_firing_rate_hz,
        parameters.nominal_twitch_force_loss_per_s,
    ):
        vector_finite = vector_finite & jnp.all(jnp.isfinite(value))
    scalar_finite = jnp.asarray(True)
    for value in (
        parameters.minimum_firing_rate_hz,
        parameters.firing_rate_gain_hz,
        parameters.derecruitment_delta_hz,
        parameters.adaptation_scale,
        parameters.adaptation_time_constant_s,
        parameters.contraction_time_change_ratio,
    ):
        scalar_finite = scalar_finite & jnp.isfinite(value)
    return (
        vector_finite
        & scalar_finite
        & jnp.all(parameters.recruitment_threshold > 0.0)
        & jnp.all(jnp.diff(parameters.recruitment_threshold) > 0.0)
        & jnp.all(parameters.rested_twitch_force > 0.0)
        & jnp.all(jnp.diff(parameters.rested_twitch_force) > 0.0)
        & jnp.all(parameters.resting_contraction_time_s > 0.0)
        & jnp.all(jnp.diff(parameters.resting_contraction_time_s) < 0.0)
        & (parameters.minimum_firing_rate_hz > 0.0)
        & (parameters.firing_rate_gain_hz > 0.0)
        & jnp.all(
            parameters.maximum_firing_rate_hz
            >= parameters.minimum_firing_rate_hz
        )
        & jnp.all(parameters.nominal_twitch_force_loss_per_s >= 0.0)
        & (parameters.derecruitment_delta_hz >= 0.0)
        & (parameters.adaptation_scale >= 0.0)
        & (parameters.adaptation_time_constant_s > 0.0)
        & (parameters.contraction_time_change_ratio >= 0.0)
    )

def _state_finite(state: PotvinFuglevand2017State, /) -> Array:
    return jnp.all(jnp.isfinite(state.recruitment_duration_s)) & jnp.all(
        jnp.isfinite(state.current_twitch_force)
    )


def _state_admissible(
    parameters: PotvinFuglevand2017Parameters,
    state: PotvinFuglevand2017State,
    /,
) -> Array:
    rested = jnp.asarray(
        parameters.rested_twitch_force,
        dtype=state.current_twitch_force.dtype,
    )
    capacity_tolerance = (
        16.0
        * jnp.finfo(state.current_twitch_force.dtype).eps
        * jnp.maximum(1.0, jnp.abs(rested))
    )
    return (
        _state_finite(state)
        & jnp.all(state.recruitment_duration_s >= 0.0)
        & jnp.all(state.current_twitch_force >= 0.0)
        & jnp.all(state.current_twitch_force <= rested + capacity_tolerance)
    )


def _require_state_shape(state: PotvinFuglevand2017State, unit_count: int, /) -> None:
    if not isinstance(state, PotvinFuglevand2017State):
        raise TypeError("state must be PotvinFuglevand2017State.")
    expected = (unit_count,)
    if (
        state.recruitment_duration_s.shape != expected
        or state.current_twitch_force.shape != expected
    ):
        raise ValueError(f"Motor-unit state vectors must have shape {expected}.")


def _pack_state(
    plan: PotvinFuglevand2017Plan,
    state: PotvinFuglevand2017State,
    /,
) -> Array:
    _require_state_shape(state, plan.unit_count)
    return jnp.stack(
        (state.recruitment_duration_s, state.current_twitch_force), axis=0
    )


def _unpack_state(
    plan: PotvinFuglevand2017Plan,
    packed_state: ArrayLike,
    /,
) -> PotvinFuglevand2017State:
    packed = jnp.asarray(packed_state)
    expected = (2, plan.unit_count)
    if packed.shape != expected:
        raise ValueError(f"Packed motor-unit state must have shape {expected}.")
    return PotvinFuglevand2017State(packed[0], packed[1])


def _maximum_excitation(parameters: PotvinFuglevand2017Parameters, /) -> Array:
    safe_gain = jnp.maximum(
        parameters.firing_rate_gain_hz,
        jnp.finfo(parameters.firing_rate_gain_hz.dtype).tiny,
    )
    return parameters.recruitment_threshold[-1] + (
        parameters.maximum_firing_rate_hz[-1]
        - parameters.minimum_firing_rate_hz
    ) / safe_gain

def _force_frequency(normalized_rate: Array, /) -> Array:
    switch_force = 1.0 - jnp.exp(-2.0 * 0.4**3)
    return jnp.where(
        normalized_rate <= 0.4,
        normalized_rate / 0.4 * switch_force,
        1.0 - jnp.exp(-2.0 * normalized_rate**3),
    )


def _evaluate(
    plan: PotvinFuglevand2017Plan,
    parameters: PotvinFuglevand2017Parameters,
    state: PotvinFuglevand2017State,
    common_excitation: Array,
    /,
) -> PotvinFuglevand2017Output:
    dtype = state.current_twitch_force.dtype
    tiny = jnp.finfo(dtype).tiny
    rested = jnp.asarray(parameters.rested_twitch_force, dtype=dtype)
    duration = state.recruitment_duration_s
    capacity = state.current_twitch_force
    minimum_rate = jnp.asarray(parameters.minimum_firing_rate_hz, dtype=dtype)
    gain = jnp.asarray(parameters.firing_rate_gain_hz, dtype=dtype)
    threshold = jnp.asarray(parameters.recruitment_threshold, dtype=dtype)
    maximum_rate = jnp.asarray(parameters.maximum_firing_rate_hz, dtype=dtype)

    recruited = common_excitation >= threshold
    unadapted = gain * (common_excitation - threshold) + minimum_rate
    unadapted = jnp.where(recruited, jnp.minimum(unadapted, maximum_rate), 0.0)
    saturated = recruited & (unadapted >= maximum_rate)

    threshold_span = jnp.maximum(threshold[-1] - threshold[0], tiny)
    threshold_fraction = (threshold - threshold[0]) / threshold_span
    maximum_adaptation = (
        jnp.asarray(parameters.adaptation_scale, dtype=dtype)
        * (unadapted - minimum_rate + parameters.derecruitment_delta_hz)
        * threshold_fraction
    )
    maximum_adaptation = jnp.maximum(maximum_adaptation, 0.0)
    safe_tau = jnp.maximum(
        jnp.asarray(parameters.adaptation_time_constant_s, dtype=dtype), tiny
    )
    adaptation = maximum_adaptation * (1.0 - jnp.exp(-duration / safe_tau))
    if not plan.central_adaptation:
        adaptation = jnp.zeros_like(adaptation)
    firing_rate = jnp.maximum(unadapted - adaptation, 0.0)

    capacity_fraction = capacity / jnp.maximum(rested, tiny)
    resting_contraction_time = jnp.asarray(
        parameters.resting_contraction_time_s, dtype=dtype
    )
    contraction_time = resting_contraction_time * (
        1.0
        + jnp.asarray(parameters.contraction_time_change_ratio, dtype=dtype)
        * (1.0 - capacity_fraction)
    )
    normalized_rate = contraction_time * firing_rate
    normalized_force = _force_frequency(normalized_rate)
    normalized_force = jnp.where(recruited, normalized_force, 0.0)
    motor_unit_force = normalized_force * capacity
    total_force = jnp.sum(motor_unit_force)
    rested_maximum_force_fraction = _force_frequency(
        resting_contraction_time * maximum_rate
    )
    total_capacity_fraction = jnp.sum(
        rested_maximum_force_fraction * capacity
    ) / jnp.maximum(jnp.sum(rested_maximum_force_fraction * rested), tiny)
    return PotvinFuglevand2017Output(
        unadapted,
        adaptation,
        firing_rate,
        contraction_time,
        normalized_rate,
        normalized_force,
        motor_unit_force,
        total_force,
        capacity_fraction,
        total_capacity_fraction,
        recruited,
        saturated,
        _maximum_excitation(parameters),
    )


def _candidate(
    plan: PotvinFuglevand2017Plan,
    parameters: PotvinFuglevand2017Parameters,
    state: PotvinFuglevand2017State,
    common_excitation: Array,
    step_s: Array,
    /,
) -> PotvinFuglevand2017Candidate:
    _require_state_shape(state, plan.unit_count)
    parameters_admissible = _parameters_admissible(parameters)
    state_finite = _state_finite(state)
    state_admissible = _state_admissible(parameters, state)
    maximum_excitation = _maximum_excitation(parameters)
    excitation_admissible = (
        jnp.isfinite(common_excitation)
        & (common_excitation >= 0.0)
        & (common_excitation <= maximum_excitation)
    )
    step_admissible = (
        jnp.isfinite(step_s) & (step_s > 0.0) & (step_s <= plan.maximum_step_s)
    )

    safe_excitation = jnp.where(excitation_admissible, common_excitation, 0.0)
    safe_step = jnp.where(step_admissible, step_s, 0.0)
    output = _evaluate(plan, parameters, state, safe_excitation)
    was_recruited = state.recruitment_duration_s > 0.0
    duration = jnp.where(
        was_recruited | output.recruited,
        state.recruitment_duration_s + safe_step,
        0.0,
    )
    if not plan.central_adaptation:
        duration = jnp.zeros_like(duration)
    capacity = state.current_twitch_force
    if plan.peripheral_fatigue:
        capacity = jnp.clip(
            capacity
            - jnp.asarray(parameters.nominal_twitch_force_loss_per_s)
            * output.normalized_force
            * safe_step,
            0.0,
            jnp.asarray(parameters.rested_twitch_force),
        )
    candidate_state = PotvinFuglevand2017State(duration, capacity)

    finite = (
        state_finite
        & jnp.isfinite(common_excitation)
        & jnp.isfinite(step_s)
        & jnp.all(jnp.isfinite(candidate_state.recruitment_duration_s))
        & jnp.all(jnp.isfinite(candidate_state.current_twitch_force))
        & jnp.all(jnp.isfinite(output.firing_rate_hz))
        & jnp.all(jnp.isfinite(output.motor_unit_force))
        & jnp.isfinite(output.total_force)
    )
    status = jnp.asarray(int(PotvinFuglevand2017Status.SUCCESS), dtype=jnp.int32)
    status = jnp.bitwise_or(
        status,
        jnp.where(
            finite,
            0,
            int(PotvinFuglevand2017Status.NONFINITE),
        ).astype(jnp.int32),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            state_admissible,
            0,
            int(PotvinFuglevand2017Status.INVALID_STATE),
        ).astype(jnp.int32),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            parameters_admissible,
            0,
            int(PotvinFuglevand2017Status.INVALID_PARAMETERS),
        ).astype(jnp.int32),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            excitation_admissible,
            0,
            int(PotvinFuglevand2017Status.INVALID_EXCITATION),
        ).astype(jnp.int32),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            step_admissible,
            0,
            int(PotvinFuglevand2017Status.INVALID_STEP),
        ).astype(jnp.int32),
    )
    saturation_excitation = parameters.recruitment_threshold + (
        parameters.maximum_firing_rate_hz - parameters.minimum_firing_rate_hz
    ) / jnp.maximum(
        parameters.firing_rate_gain_hz,
        jnp.finfo(parameters.firing_rate_gain_hz.dtype).tiny,
    )
    aggregation_residual = output.total_force - jnp.sum(output.motor_unit_force)
    evidence = PotvinFuglevand2017Evidence(
        status,
        finite,
        state_admissible,
        parameters_admissible,
        excitation_admissible,
        step_admissible,
        aggregation_residual,
        jnp.min(jnp.abs(safe_excitation - parameters.recruitment_threshold)),
        jnp.min(jnp.abs(safe_excitation - saturation_excitation)),
        plan.plan_id,
    )
    return PotvinFuglevand2017Candidate(state, candidate_state, output, evidence)


__all__ = [
    "POTVIN_FUGLEVAND_2017_DOI",
    "POTVIN_FUGLEVAND_2017_MODEL_ID",
    "POTVIN_FUGLEVAND_2017_REFERENCE_SHA",
    "PotvinFuglevand2017Candidate",
    "PotvinFuglevand2017Evidence",
    "PotvinFuglevand2017Output",
    "PotvinFuglevand2017Parameters",
    "PotvinFuglevand2017Plan",
    "PotvinFuglevand2017State",
    "PotvinFuglevand2017Status",
    "PreparedPotvinFuglevand2017",
    "potvin_fuglevand_2017_default_parameters",
]
