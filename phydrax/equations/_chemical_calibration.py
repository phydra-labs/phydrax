#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_mechanism import PreparedChemicalMechanism
from ._chemical_rates import ArrheniusRatePlan


class ChemicalParameterCoordinate(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    LOG_MULTIPLICATIVE = "log_multiplicative"
    BOUNDED = "bounded"


class ChemicalCalibrationParameter(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    reaction_index: int = eqx.field(static=True)
    direction: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    coordinate: ChemicalParameterCoordinate = eqx.field(static=True)
    baseline: Array
    lower: Array
    upper: Array
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reaction_index: int,
        field_name: str,
        coordinate: ChemicalParameterCoordinate,
        baseline: ArrayLike,
        /,
        *,
        direction: str = "forward",
        lower: ArrayLike = -jnp.inf,
        upper: ArrayLike = jnp.inf,
    ):
        name_ = str(name)
        index = int(reaction_index)
        field = str(field_name)
        direction_ = str(direction)
        if not name_ or index < 0:
            raise ValueError("Calibration parameter name/index is invalid.")
        if field not in (
            "pre_exponential",
            "temperature_exponent",
            "activation_energy",
        ):
            raise ValueError("Unsupported Arrhenius calibration field.")
        if direction_ not in ("forward", "reverse"):
            raise ValueError("direction must be forward or reverse.")
        if not isinstance(coordinate, ChemicalParameterCoordinate):
            raise TypeError("coordinate must be ChemicalParameterCoordinate.")
        baseline_ = jnp.asarray(baseline)
        lower_ = jnp.asarray(lower, dtype=baseline_.dtype)
        upper_ = jnp.asarray(upper, dtype=baseline_.dtype)
        lower_host = float(lower_)
        upper_host = float(upper_)
        if baseline_.shape != () or lower_.shape != () or upper_.shape != ():
            raise ValueError("Calibration bounds and baseline must be scalar.")
        if coordinate is ChemicalParameterCoordinate.BOUNDED and not bool(
            jnp.isfinite(lower_) & jnp.isfinite(upper_) & (upper_ > lower_)
        ):
            raise ValueError("Bounded coordinates require finite ordered bounds.")
        self.name = name_
        self.reaction_index = index
        self.direction = direction_
        self.field_name = field
        self.coordinate = coordinate
        self.baseline = baseline_
        self.lower = lower_
        self.upper = upper_
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "chemical-calibration-parameter",
                "name": name_,
                "reaction": index,
                "direction": direction_,
                "field": field,
                "coordinate": coordinate.value,
                "baseline": float(baseline_),
                "lower": lower_host if np.isfinite(lower_host) else None,
                "upper": upper_host if np.isfinite(upper_host) else None,
            }
        )

    def decode(self, coordinate_value: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinate_value, dtype=self.baseline.dtype)
        if value.shape != ():
            raise ValueError("Calibration coordinate must be scalar.")
        if self.coordinate is ChemicalParameterCoordinate.ADDITIVE:
            return self.baseline + value
        if self.coordinate is ChemicalParameterCoordinate.MULTIPLICATIVE:
            return self.baseline * (1.0 + value)
        if self.coordinate is ChemicalParameterCoordinate.LOG_MULTIPLICATIVE:
            return self.baseline * jnp.exp(value)
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(value)


class ChemicalCalibrationPlan(StrictModule, NonTrainableState):
    mechanism: PreparedChemicalMechanism
    parameters: tuple[ChemicalCalibrationParameter, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, mechanism: PreparedChemicalMechanism, parameters, /):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        values = tuple(parameters)
        if not values or any(
            not isinstance(value, ChemicalCalibrationParameter) for value in values
        ):
            raise TypeError("parameters must contain calibration parameter objects.")
        if len({value.name for value in values}) != len(values):
            raise ValueError("Calibration parameter names must be unique.")
        for value in values:
            if value.reaction_index >= mechanism.reaction_count:
                raise ValueError("Calibration reaction index is out of range.")
            reaction = mechanism.reactions[value.reaction_index]
            rate = (
                reaction.forward_rate
                if value.direction == "forward"
                else reaction.reverse_rate
            )
            if not isinstance(rate, ArrheniusRatePlan):
                raise TypeError(
                    "Direct calibration currently requires an Arrhenius rate."
                )
        self.mechanism = mechanism
        self.parameters = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "chemical-calibration-plan",
                "mechanism": mechanism.mechanism_id,
                "parameters": [value.parameter_id for value in values],
            }
        )

    def apply(self, coordinates: ArrayLike, /) -> PreparedChemicalMechanism:
        values = jnp.asarray(coordinates)
        if values.shape != (len(self.parameters),):
            raise ValueError("coordinates must have one entry per parameter.")
        reactions = list(self.mechanism.reactions)
        for parameter_index, parameter in enumerate(self.parameters):
            reaction = reactions[parameter.reaction_index]
            rate = (
                reaction.forward_rate
                if parameter.direction == "forward"
                else reaction.reverse_rate
            )
            if not isinstance(rate, ArrheniusRatePlan):
                raise TypeError("Calibration target ceased to be Arrhenius.")
            decoded = parameter.decode(values[parameter_index])
            if parameter.field_name == "pre_exponential":
                updated_rate = eqx.tree_at(
                    lambda value: value.pre_exponential, rate, decoded
                )
            elif parameter.field_name == "temperature_exponent":
                updated_rate = eqx.tree_at(
                    lambda value: value.temperature_exponent, rate, decoded
                )
            else:
                updated_rate = eqx.tree_at(
                    lambda value: value.activation_energy, rate, decoded
                )
            if parameter.direction == "forward":
                reaction = eqx.tree_at(
                    lambda value: value.forward_rate, reaction, updated_rate
                )
            else:
                reaction = eqx.tree_at(
                    lambda value: value.reverse_rate, reaction, updated_rate
                )
            reactions[parameter.reaction_index] = reaction
        return eqx.tree_at(
            lambda mechanism: mechanism.reactions,
            self.mechanism,
            tuple(reactions),
        )


__all__ = [
    "ChemicalCalibrationParameter",
    "ChemicalCalibrationPlan",
    "ChemicalParameterCoordinate",
]
