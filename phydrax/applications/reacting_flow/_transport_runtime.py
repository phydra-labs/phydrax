#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_transport_properties import (
    AbstractGasTransportPropertyPlan,
    GasTransportPropertyEvaluation,
)


class TransportPropertyReuseState(StrictModule):
    temperature: Array
    pressure: Array
    properties: GasTransportPropertyEvaluation
    reuse_count: Array
    plan_id: str = eqx.field(static=True)


class TransportPropertyReuseCandidate(StrictModule):
    properties: GasTransportPropertyEvaluation
    proposed_state: TransportPropertyReuseState
    viscosity_reused: Array
    conductivity_reused: Array
    diffusion_reused: Array
    viscosity_error_bound: Array
    conductivity_error_bound: Array
    diffusion_error_bound: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TransportPropertyReusePlan(StrictModule, NonTrainableState):
    properties: AbstractGasTransportPropertyPlan
    temperature_bounds: tuple[float, float] = eqx.field(static=True)
    pressure_bounds: tuple[float, float] = eqx.field(static=True)
    logarithmic_sensitivities: tuple[tuple[float, float], ...] = eqx.field(static=True)
    maximum_relative_errors: tuple[float, float, float] = eqx.field(static=True)
    maximum_reuse_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        properties: AbstractGasTransportPropertyPlan,
        /,
        *,
        temperature_bounds: tuple[float, float],
        pressure_bounds: tuple[float, float],
        logarithmic_sensitivities: tuple[tuple[float, float], ...],
        maximum_relative_errors: tuple[float, float, float],
        maximum_reuse_count: int,
    ):
        if not isinstance(properties, AbstractGasTransportPropertyPlan):
            raise TypeError("properties must implement AbstractGasTransportPropertyPlan.")
        temperature = tuple(float(value) for value in temperature_bounds)
        pressure = tuple(float(value) for value in pressure_bounds)
        sensitivities = tuple(
            tuple(float(component) for component in value)
            for value in logarithmic_sensitivities
        )
        errors = tuple(float(value) for value in maximum_relative_errors)
        count = int(maximum_reuse_count)
        if (
            len(temperature) != 2
            or len(pressure) != 2
            or not 0.0 < temperature[0] < temperature[1]
            or not 0.0 < pressure[0] < pressure[1]
            or len(sensitivities) != 3
            or any(len(value) != 2 for value in sensitivities)
            or any(
                not isfinite(component) or component < 0.0
                for value in sensitivities
                for component in value
            )
            or len(errors) != 3
            or any(not isfinite(value) or value < 0.0 for value in errors)
            or count < 0
        ):
            raise ValueError(
                "Transport reuse support, sensitivities, or limits are invalid."
            )
        self.properties = properties
        self.temperature_bounds = temperature
        self.pressure_bounds = pressure
        self.logarithmic_sensitivities = sensitivities
        self.maximum_relative_errors = errors
        self.maximum_reuse_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transport-property-reuse",
                "properties": properties.property_id,
                "temperature_bounds": temperature,
                "pressure_bounds": pressure,
                "logarithmic_sensitivities": sensitivities,
                "maximum_relative_errors": errors,
                "maximum_reuse_count": count,
            }
        )

    def initialize(
        self, temperature: ArrayLike, pressure: ArrayLike, /
    ) -> TransportPropertyReuseState:
        temperature_, pressure_ = jnp.broadcast_arrays(
            jnp.asarray(temperature), jnp.asarray(pressure)
        )
        evaluated = self.properties.evaluate(temperature_, pressure_)
        return TransportPropertyReuseState(
            temperature_,
            pressure_,
            evaluated,
            jnp.zeros(temperature_.shape, dtype=jnp.int32),
            self.plan_id,
        )

    def propose(
        self,
        accepted: TransportPropertyReuseState,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> TransportPropertyReuseCandidate:
        if not isinstance(accepted, TransportPropertyReuseState):
            raise TypeError("accepted must be TransportPropertyReuseState.")
        if accepted.plan_id != self.plan_id:
            raise ValueError("Transport reuse state belongs to another plan.")
        temperature_, pressure_ = jnp.broadcast_arrays(
            jnp.asarray(temperature, dtype=accepted.temperature.dtype),
            jnp.asarray(pressure, dtype=accepted.pressure.dtype),
        )
        if temperature_.shape != accepted.temperature.shape:
            raise ValueError("Transport reuse query shape changed after preparation.")
        tiny = jnp.finfo(temperature_.dtype).tiny
        relative_temperature = jnp.abs(
            jnp.log(
                jnp.maximum(temperature_, tiny) / jnp.maximum(accepted.temperature, tiny)
            )
        )
        relative_pressure = jnp.abs(
            jnp.log(jnp.maximum(pressure_, tiny) / jnp.maximum(accepted.pressure, tiny))
        )
        estimated = tuple(
            sensitivity[0] * relative_temperature + sensitivity[1] * relative_pressure
            for sensitivity in self.logarithmic_sensitivities
        )
        in_support = (
            (temperature_ >= self.temperature_bounds[0])
            & (temperature_ <= self.temperature_bounds[1])
            & (pressure_ >= self.pressure_bounds[0])
            & (pressure_ <= self.pressure_bounds[1])
            & jnp.isfinite(temperature_)
            & jnp.isfinite(pressure_)
        )
        within_count = accepted.reuse_count < self.maximum_reuse_count
        reused = tuple(
            in_support & within_count & (bound <= maximum)
            for bound, maximum in zip(
                estimated, self.maximum_relative_errors, strict=True
            )
        )
        reuse_all = reused[0] & reused[1] & reused[2]

        def evaluate_fresh(_):
            return self.properties.evaluate(temperature_, pressure_)

        fresh = jax.lax.cond(
            jnp.all(reuse_all),
            lambda _: accepted.properties,
            evaluate_fresh,
            operand=None,
        )
        old = accepted.properties
        viscosity = jnp.where(
            reused[0][..., None], old.species_viscosity, fresh.species_viscosity
        )
        conductivity = jnp.where(
            reused[1][..., None],
            old.species_thermal_conductivity,
            fresh.species_thermal_conductivity,
        )
        diffusion = jnp.where(
            reused[2][..., None, None],
            old.binary_diffusion_coefficients,
            fresh.binary_diffusion_coefficients,
        )
        supported = in_support & jnp.where(reuse_all, old.supported, fresh.supported)
        finite = jnp.where(reuse_all, old.finite, fresh.finite)
        successful = (
            supported & finite & jnp.where(reuse_all, old.successful, fresh.successful)
        )
        evaluation = GasTransportPropertyEvaluation(
            viscosity,
            conductivity,
            diffusion,
            jnp.where(reused[0], estimated[0], fresh.viscosity_relative_error_bound),
            jnp.where(reused[1], estimated[1], fresh.conductivity_relative_error_bound),
            jnp.where(reused[2], estimated[2], fresh.diffusion_relative_error_bound),
            supported,
            finite,
            successful,
            self.properties.property_id,
        )
        proposed = TransportPropertyReuseState(
            temperature_,
            pressure_,
            evaluation,
            jnp.where(reuse_all, accepted.reuse_count + 1, 0),
            self.plan_id,
        )
        return TransportPropertyReuseCandidate(
            evaluation,
            proposed,
            reused[0],
            reused[1],
            reused[2],
            evaluation.viscosity_relative_error_bound,
            evaluation.conductivity_relative_error_bound,
            evaluation.diffusion_relative_error_bound,
            successful,
            self.plan_id,
        )

    def commit(
        self,
        accepted: TransportPropertyReuseState,
        candidate: TransportPropertyReuseCandidate,
        commit: ArrayLike,
        /,
    ) -> TransportPropertyReuseState:
        if candidate.plan_id != self.plan_id or accepted.plan_id != self.plan_id:
            raise ValueError("Transport reuse state/candidate belongs to another plan.")
        decision = jnp.asarray(commit, dtype=bool)
        if decision.shape != ():
            raise ValueError("Transport reuse commit decision must be scalar.")
        return jax.tree.map(
            lambda proposed, prior: jnp.where(decision, proposed, prior),
            candidate.proposed_state,
            accepted,
        )


__all__ = [
    "TransportPropertyReuseCandidate",
    "TransportPropertyReusePlan",
    "TransportPropertyReuseState",
]
