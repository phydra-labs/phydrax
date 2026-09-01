#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..observation import CovarianceAction


class SimulationSensitivityReport(StrictModule):
    value: Array
    jvp: Array
    finite_difference: Array
    jvp_residual: Array
    vjp_pairing_residual: Array
    finite: Array

    @classmethod
    def evaluate(
        cls,
        function: Callable,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        epsilon: float = 1e-5,
    ) -> SimulationSensitivityReport:
        parameter = jnp.asarray(parameters)
        tangent = jnp.asarray(direction, dtype=parameter.dtype)
        value, jvp = jax.jvp(function, (parameter,), (tangent,))
        finite_difference = (
            function(parameter + epsilon * tangent)
            - function(parameter - epsilon * tangent)
        ) / (2.0 * epsilon)
        cotangent = jnp.ones_like(value)
        _, pullback = jax.vjp(function, parameter)
        vjp = pullback(cotangent)[0]
        pairing = jnp.sum(cotangent * jvp) - jnp.sum(vjp * tangent)
        return cls(
            value=value,
            jvp=jvp,
            finite_difference=finite_difference,
            jvp_residual=jnp.max(jnp.abs(jvp - finite_difference), initial=0.0),
            vjp_pairing_residual=jnp.abs(pairing),
            finite=jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(jvp)),
        )


class FieldObservationPlan(StrictModule, NonTrainableState):
    operator: Callable = eqx.field(static=True)
    observed: Array
    covariance: CovarianceAction
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: Callable,
        observed: ArrayLike,
        covariance: CovarianceAction,
        /,
        *,
        observation_id: str,
    ):
        if not callable(operator) or not observation_id:
            raise ValueError("Field observation metadata is invalid.")
        observed_ = jnp.asarray(observed).reshape((-1,))
        if observed_.shape != (covariance.layout.size,):
            raise ValueError("Observation and covariance layouts disagree.")
        observed_ = eqx.error_if(
            observed_,
            jnp.any(~jnp.isfinite(observed_)),
            "Observed field values must be finite.",
        )
        self.operator = operator
        self.observed = observed_
        self.covariance = covariance
        self.observation_id = canonical_fingerprint(
            {
                "kind": "field-observation",
                "declared_id": observation_id,
                "covariance": covariance.action_id,
            }
        )

    def log_likelihood(self, state: Any, args: Any = None, /) -> Array:
        predicted = jnp.asarray(self.operator(state, args)).reshape((-1,))
        if predicted.shape != self.observed.shape:
            raise ValueError("Predicted field observation shape is invalid.")
        residual = predicted - self.observed
        normalization = 0.5 * (
            residual.size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.dtype))
            + self.covariance.logdet_covariance
        )
        return -0.5 * self.covariance.quadratic(residual) - normalization


class WhitenedFieldInferencePlan(StrictModule, NonTrainableState):
    simulate: Callable = eqx.field(static=True)
    observation: FieldObservationPlan
    prior_transform: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        simulate: Callable,
        observation: FieldObservationPlan,
        prior_transform: ArrayLike,
        /,
        *,
        plan_id: str,
    ):
        transform = jnp.asarray(prior_transform)
        if (
            not callable(simulate)
            or not isinstance(observation, FieldObservationPlan)
            or transform.ndim != 2
            or not plan_id
        ):
            raise ValueError("Whitened field inference plan is invalid.")
        self.simulate = simulate
        self.observation = observation
        self.prior_transform = transform
        self.plan_id = canonical_fingerprint(
            {
                "kind": "whitened-field-inference",
                "declared_id": plan_id,
                "observation": observation.observation_id,
                "latent_size": transform.shape[1],
                "field_size": transform.shape[0],
            }
        )

    def physical_field(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if value.shape != (self.prior_transform.shape[1],):
            raise ValueError("Whitened latent field shape is invalid.")
        return self.prior_transform @ value

    def log_density(self, latent: ArrayLike, args: Any = None, /) -> Array:
        value = jnp.asarray(latent)
        state = self.simulate(self.physical_field(value), args)
        return -0.5 * jnp.sum(value**2) + self.observation.log_likelihood(state, args)

    def value_and_gradient(
        self, latent: ArrayLike, args: Any = None, /
    ) -> tuple[Array, Array]:
        return jax.value_and_grad(lambda value: self.log_density(value, args))(
            jnp.asarray(latent)
        )


class ParticleMarginalLikelihoodPlan(StrictModule, NonTrainableState):
    log_weight: Callable = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, log_weight: Callable, /, *, plan_id: str):
        if not callable(log_weight) or not plan_id:
            raise ValueError("Particle marginal likelihood plan is invalid.")
        self.log_weight = log_weight
        self.plan_id = canonical_fingerprint(
            {"kind": "particle-marginal-likelihood", "declared_id": plan_id}
        )

    def estimate(self, parameters: Any, realizations: Any, /) -> Array:
        weights = jax.vmap(lambda realization: self.log_weight(parameters, realization))(
            realizations
        )
        maximum = jnp.max(weights)
        return maximum + jnp.log(jnp.mean(jnp.exp(weights - maximum)))


__all__ = [
    "FieldObservationPlan",
    "ParticleMarginalLikelihoodPlan",
    "SimulationSensitivityReport",
    "WhitenedFieldInferencePlan",
]
