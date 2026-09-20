#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ParameterConstraintKind(StrEnum):
    UNCONSTRAINED = "unconstrained"
    GAUSSIAN = "gaussian"


class BinnedModifierMode(StrEnum):
    EXPONENTIAL = "exponential"
    ADDITIVE = "additive"


class StatisticalParameter(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    initial: float = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    constraint_kind: ParameterConstraintKind = eqx.field(static=True)
    constraint_mean: float = eqx.field(static=True)
    constraint_standard_deviation: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        initial: float,
        lower: float,
        upper: float,
        constraint_kind: ParameterConstraintKind = ParameterConstraintKind.UNCONSTRAINED,
        constraint_mean: float = 0.0,
        constraint_standard_deviation: float = 1.0,
    ):
        name_ = str(name).strip()
        values = tuple(
            map(
                float,
                (initial, lower, upper, constraint_mean, constraint_standard_deviation),
            )
        )
        if not name_ or not isinstance(constraint_kind, ParameterConstraintKind):
            raise ValueError("Parameter name and constraint kind are required.")
        if (
            any(math.isnan(value) for value in values)
            or not values[1] < values[2]
            or not values[1] <= values[0] <= values[2]
        ):
            raise ValueError("Parameter initial value and bounds are invalid.")
        if constraint_kind is ParameterConstraintKind.GAUSSIAN and (
            not math.isfinite(values[4]) or values[4] <= 0.0
        ):
            raise ValueError(
                "Gaussian constraints require positive finite standard deviation."
            )
        self.name = name_
        (
            self.initial,
            self.lower,
            self.upper,
            self.constraint_mean,
            self.constraint_standard_deviation,
        ) = values
        self.constraint_kind = constraint_kind
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "statistical-parameter",
                "name": name_,
                "values": list(values),
                "constraint": constraint_kind.value,
            }
        )


class BinnedStatisticalModel(StrictModule, NonTrainableState):
    nominal_samples: Array
    observations: Array
    modifier_effects: Array
    bin_active: Array
    parameters: tuple[StatisticalParameter, ...]
    modifier_modes: tuple[BinnedModifierMode, ...] = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    sample_names: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        nominal_samples: ArrayLike,
        observations: ArrayLike,
        modifier_effects: ArrayLike,
        /,
        *,
        parameters: Sequence[StatisticalParameter],
        modifier_modes: Sequence[BinnedModifierMode],
        channel_names: Sequence[str],
        sample_names: Sequence[str],
        bin_active: ArrayLike | None = None,
    ):
        nominal = np.asarray(nominal_samples, dtype=np.float64)
        observed = np.asarray(observations, dtype=np.float64)
        effects = np.asarray(modifier_effects, dtype=np.float64)
        parameters_ = tuple(parameters)
        modes = tuple(modifier_modes)
        channels = tuple(str(value).strip() for value in channel_names)
        samples = tuple(str(value).strip() for value in sample_names)
        if (
            nominal.ndim != 3
            or nominal.shape[0] < 1
            or nominal.shape[1] < 1
            or nominal.shape[2] < 1
        ):
            raise ValueError("nominal_samples must have shape (channel, sample, bin).")
        channel_count, sample_count, bin_count = nominal.shape
        if (
            observed.shape != (channel_count, bin_count)
            or effects.shape != (len(parameters_),) + nominal.shape
        ):
            raise ValueError(
                "Observations or modifier effects do not align with the model."
            )
        if not parameters_ or any(
            not isinstance(value, StatisticalParameter) for value in parameters_
        ):
            raise TypeError("parameters must contain typed non-empty values.")
        if len(modes) != len(parameters_) or any(
            not isinstance(value, BinnedModifierMode) for value in modes
        ):
            raise TypeError("modifier_modes must align with parameters.")
        if (
            len(channels) != channel_count
            or len(samples) != sample_count
            or any(not value for value in channels + samples)
        ):
            raise ValueError("Channel/sample names must align and be non-empty.")
        if (
            len(set(channels)) != len(channels)
            or len(set(samples)) != len(samples)
            or len({value.name for value in parameters_}) != len(parameters_)
        ):
            raise ValueError("Channel, sample, and parameter names must be unique.")
        active = (
            np.ones(observed.shape, dtype=np.bool_)
            if bin_active is None
            else np.asarray(bin_active, dtype=np.bool_)
        )
        if active.shape != observed.shape:
            raise ValueError("bin_active must align with observations.")
        if (
            np.any(~np.isfinite(nominal))
            or np.any(nominal < 0.0)
            or np.any(~np.isfinite(observed[active]))
            or np.any(observed[active] < 0.0)
            or np.any(~np.isfinite(effects))
        ):
            raise ValueError(
                "Binned model inputs must be finite with nonnegative nominal/observed values."
            )
        self.nominal_samples = jnp.asarray(nominal)
        self.observations = jnp.asarray(observed)
        self.modifier_effects = jnp.asarray(effects)
        self.bin_active = jnp.asarray(active)
        self.parameters = parameters_
        self.modifier_modes = modes
        self.channel_names = channels
        self.sample_names = samples
        self.model_id = canonical_fingerprint(
            {
                "kind": "hep-binned-statistical-model",
                "arrays": array_tree_fingerprint((nominal, observed, effects, active)),
                "parameters": [value.parameter_id for value in parameters_],
                "modes": [value.value for value in modes],
                "channels": list(channels),
                "samples": list(samples),
            }
        )

    @property
    def initial_parameters(self) -> Array:
        return jnp.asarray(
            [value.initial for value in self.parameters], dtype=self.nominal_samples.dtype
        )

    @property
    def lower_bounds(self) -> Array:
        return jnp.asarray(
            [value.lower for value in self.parameters], dtype=self.nominal_samples.dtype
        )

    @property
    def upper_bounds(self) -> Array:
        return jnp.asarray(
            [value.upper for value in self.parameters], dtype=self.nominal_samples.dtype
        )


class BinnedLikelihoodEvaluation(StrictModule, NonTrainableState):
    sample_expectations: Array
    channel_expectations: Array
    poisson_log_likelihood: Array
    constraint_log_likelihood: Array
    total_log_likelihood: Array
    finite: Array
    valid: Array
    model_id: str = eqx.field(static=True)


def evaluate_binned_model(
    model: BinnedStatisticalModel,
    parameters: ArrayLike,
    /,
) -> BinnedLikelihoodEvaluation:
    if not isinstance(model, BinnedStatisticalModel):
        raise TypeError("model must be BinnedStatisticalModel.")
    values = jnp.asarray(parameters, dtype=model.nominal_samples.dtype)
    if values.shape != (len(model.parameters),):
        raise ValueError("parameters must align with the model parameter axis.")
    samples = model.nominal_samples
    for index, mode in enumerate(model.modifier_modes):
        effect = model.modifier_effects[index]
        if mode is BinnedModifierMode.EXPONENTIAL:
            samples = samples * jnp.exp(values[index] * effect)
        else:
            samples = samples + values[index] * effect
    expected = jnp.sum(samples, axis=1)
    physical = jnp.all(jnp.where(model.bin_active, expected > 0.0, True)) & jnp.all(
        samples >= 0.0
    )
    safe_expected = jnp.maximum(expected, jnp.finfo(expected.dtype).tiny)
    poisson_terms = (
        model.observations * jnp.log(safe_expected)
        - safe_expected
        - jsp.special.gammaln(model.observations + 1.0)
    )
    poisson = jnp.sum(jnp.where(model.bin_active, poisson_terms, 0.0))
    constraint = jnp.asarray(0.0, dtype=values.dtype)
    for index, parameter in enumerate(model.parameters):
        if parameter.constraint_kind is ParameterConstraintKind.GAUSSIAN:
            standardized = (
                values[index] - parameter.constraint_mean
            ) / parameter.constraint_standard_deviation
            constraint = constraint - 0.5 * (
                standardized * standardized
                + jnp.log(2.0 * jnp.pi * parameter.constraint_standard_deviation**2)
            )
    bounded = jnp.all((values >= model.lower_bounds) & (values <= model.upper_bounds))
    total = poisson + constraint
    finite = jnp.all(jnp.isfinite(samples)) & jnp.isfinite(total)
    return BinnedLikelihoodEvaluation(
        samples,
        expected,
        poisson,
        constraint,
        total,
        finite,
        finite & physical & bounded,
        model.model_id,
    )


__all__ = [
    "BinnedLikelihoodEvaluation",
    "BinnedModifierMode",
    "BinnedStatisticalModel",
    "ParameterConstraintKind",
    "StatisticalParameter",
    "evaluate_binned_model",
]
