# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._strict import StrictModule
from .._system import DiscreteStepContext


class LearnedMarginalTransition(StrictModule):
    """Model-declared Gaussian next-state/increment law with explicit factor scaling."""

    model: Any
    state_size: int = eqx.field(static=True)
    event: Literal["next_state", "increment"] = eqx.field(static=True)
    factor_scaling: Literal["covariance", "diffusion"] = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        state_size: int,
        /,
        *,
        event: Literal["next_state", "increment"] = "next_state",
        factor_scaling: Literal["covariance", "diffusion"] = "covariance",
    ):
        if not callable(model) or int(state_size) <= 0:
            raise ValueError(
                "Marginal transition requires callable model and state size."
            )
        if event not in ("next_state", "increment") or factor_scaling not in (
            "covariance",
            "diffusion",
        ):
            raise ValueError("Unknown stochastic transition semantics.")
        self.model = model
        self.state_size = int(state_size)
        self.event = event
        self.factor_scaling = factor_scaling

    def parameters(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        control: ArrayLike | None = None,
        /,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(state)
        model_input = (
            (value, context.duration)
            if control is None
            else (value, context.duration, jnp.asarray(control))
        )
        location, factor = self.model(model_input, key=None)
        location = jnp.asarray(location)
        factor = jnp.asarray(factor)
        if location.shape != (self.state_size,) or factor.shape != (
            self.state_size,
            self.state_size,
        ):
            raise ValueError(
                "Marginal transition model returned incompatible event shapes."
            )
        factor = jnp.tril(factor)
        factor = eqx.error_if(
            factor, jnp.any(~jnp.isfinite(factor)), "Transition factor must be finite."
        )
        if self.factor_scaling == "diffusion":
            factor = jnp.sqrt(context.duration) * factor
        return location, factor

    def sample(
        self,
        key: Key[Array, ""],
        context: DiscreteStepContext,
        state: ArrayLike,
        control: ArrayLike | None = None,
        /,
    ) -> Array:
        location, factor = self.parameters(context, state, control)
        noise = jr.normal(key, (self.state_size,), dtype=location.real.dtype)
        event = location + contract("ij,j->i", factor, noise)
        return event if self.event == "next_state" else jnp.asarray(state) + event


class LearnedPathwiseTransition(StrictModule):
    """Learned drift/noise action driven by a caller-supplied Wiener increment."""

    drift_model: Any
    noise_model: Any
    state_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)
    interpretation: Literal["ito", "stratonovich"] = eqx.field(static=True)

    def __init__(
        self,
        drift_model: Any,
        noise_model: Any,
        state_size: int,
        noise_size: int,
        /,
        *,
        interpretation: Literal["ito", "stratonovich"] = "ito",
    ):
        if not callable(drift_model) or not callable(noise_model):
            raise TypeError("Pathwise transition models must be callable.")
        if min(int(state_size), int(noise_size)) <= 0 or interpretation not in (
            "ito",
            "stratonovich",
        ):
            raise ValueError(
                "Pathwise transition dimensions or interpretation are invalid."
            )
        self.drift_model = drift_model
        self.noise_model = noise_model
        self.state_size = int(state_size)
        self.noise_size = int(noise_size)
        self.interpretation = interpretation

    def __call__(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        wiener_increment: ArrayLike,
        control: ArrayLike | None = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        increment = jnp.asarray(wiener_increment)
        if increment.shape != (self.noise_size,):
            raise ValueError("Wiener increment shape does not match noise_size.")
        model_input = (
            (value, context.duration)
            if control is None
            else (value, context.duration, jnp.asarray(control))
        )
        drift = jnp.asarray(self.drift_model(model_input, key=None))
        diffusion = jnp.asarray(self.noise_model(model_input, key=None))
        if drift.shape != (self.state_size,) or diffusion.shape != (
            self.state_size,
            self.noise_size,
        ):
            raise ValueError("Pathwise transition model returned incompatible shapes.")
        return (
            value + context.duration * drift + contract("ij,j->i", diffusion, increment)
        )


__all__ = ["LearnedMarginalTransition", "LearnedPathwiseTransition"]
