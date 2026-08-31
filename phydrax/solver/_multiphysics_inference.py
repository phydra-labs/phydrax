#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


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
    inverse_covariance: Array
    normalization: Array
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: Callable,
        observed: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        observation_id: str,
    ):
        if not callable(operator) or not observation_id:
            raise ValueError("Field observation metadata is invalid.")
        observed_host = np.asarray(observed)
        covariance_host = np.asarray(covariance, dtype=observed_host.dtype)
        if covariance_host.shape != (observed_host.size, observed_host.size):
            raise ValueError("Observation covariance shape is invalid.")
        sign, logdet = np.linalg.slogdet(covariance_host)
        if sign <= 0.0:
            raise ValueError("Observation covariance must be positive definite.")
        inverse = np.linalg.inv(covariance_host)
        self.operator = operator
        self.observed = jnp.asarray(observed_host).reshape((-1,))
        self.inverse_covariance = jnp.asarray(inverse)
        self.normalization = jnp.asarray(
            0.5 * (observed_host.size * np.log(2.0 * np.pi) + logdet)
        )
        self.observation_id = canonical_fingerprint(
            {"kind": "field-observation", "declared_id": observation_id}
        )

    def log_likelihood(self, state: Any, args: Any = None, /) -> Array:
        predicted = jnp.asarray(self.operator(state, args)).reshape((-1,))
        residual = predicted - self.observed
        quadratic = oe.contract("i,ij,j->", residual, self.inverse_covariance, residual)
        return -0.5 * quadratic - self.normalization


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
