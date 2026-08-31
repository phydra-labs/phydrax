#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._probability import AbstractProbabilityLaw, DiagonalNormalLaw
from .._strict import StrictModule
from ._process import (
    AbstractMarginalTransitionLaw,
    DiagonalGaussianProcessDistribution,
)


TerminalReferenceRelationship = Literal["exact", "asymptotic", "external"]


def _identifier(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _inexact_state(value: ArrayLike, state_shape: tuple[int, ...], /) -> Array:
    state = jnp.asarray(value)
    if state.ndim < len(state_shape) or tuple(state.shape[-len(state_shape) :]) != state_shape:
        raise ValueError(f"state must end in shape {state_shape}; got {state.shape}.")
    if jnp.iscomplexobj(state):
        raise TypeError("Gaussian score diffusion initially requires real state coordinates.")
    return state if jnp.issubdtype(state.dtype, jnp.inexact) else state.astype(float)


class DiffusionTerminalReference(StrictModule):
    """Declared terminal source law and its relationship to the forward marginal."""

    law: AbstractProbabilityLaw
    residual_signal_scale: Array
    relationship: TerminalReferenceRelationship = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        law: AbstractProbabilityLaw,
        /,
        *,
        relationship: TerminalReferenceRelationship,
        residual_signal_scale: ArrayLike,
        reference_id: str,
        process_id: str,
    ):
        if not isinstance(law, AbstractProbabilityLaw):
            raise TypeError("law must implement AbstractProbabilityLaw.")
        if tuple(law.batch_shape):
            raise ValueError("A diffusion terminal reference must be unbatched.")
        if law.density_measure_kind != "lebesgue":
            raise ValueError("A Gaussian diffusion terminal reference must be Lebesgue.")
        if relationship not in ("exact", "asymptotic", "external"):
            raise ValueError("Unknown terminal-reference relationship.")
        residual = jnp.asarray(residual_signal_scale, dtype=float).reshape(())
        if bool(~jnp.isfinite(residual)) or float(residual) < 0.0:
            raise ValueError("residual_signal_scale must be finite and nonnegative.")
        self.law = law
        self.residual_signal_scale = residual
        self.relationship = relationship
        self.reference_id = _identifier(reference_id, owner="reference_id")
        self.process_id = _identifier(process_id, owner="process_id")


class AbstractGaussianDiffusion(AbstractMarginalTransitionLaw):
    """Real vector Itô diffusion with scalar state-independent noise."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    terminal_time: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, terminal_time: float, process_id: str):
        size = int(dimension)
        horizon = float(terminal_time)
        if size <= 0:
            raise ValueError("dimension must be positive.")
        if not isfinite(horizon) or horizon <= 0.0:
            raise ValueError("terminal_time must be finite and positive.")
        self.state_shape = (size,)
        self.dimension = size
        self.terminal_time = horizon
        self.process_id = _identifier(process_id, owner="process_id")

    def _time(self, value: ArrayLike, /) -> Array:
        time = jnp.asarray(value, dtype=float)
        if time.shape != ():
            raise ValueError("Diffusion times must be scalar.")
        return eqx.error_if(
            time,
            ~jnp.isfinite(time) | (time < 0.0) | (time > self.terminal_time),
            "Diffusion time lies outside [0, terminal_time].",
        )

    def _interval(self, t0: ArrayLike, t1: ArrayLike, /) -> tuple[Array, Array]:
        start = self._time(t0)
        end = self._time(t1)
        end = eqx.error_if(
            end,
            end <= start,
            "Diffusion transitions require t1 > t0.",
        )
        return start, end

    @abstractmethod
    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def diffusion_scale(self, time: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def transition_mean_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def transition_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def asymptotic_terminal_reference(self) -> DiffusionTerminalReference:
        raise NotImplementedError

    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> DiagonalGaussianProcessDistribution:
        state_array = _inexact_state(state, self.state_shape)
        mean_scale = self.transition_mean_scale(t0, t1).astype(state_array.dtype)
        scale = self.transition_scale(t0, t1).astype(state_array.dtype)
        return DiagonalGaussianProcessDistribution(
            mean_scale * state_array,
            scale,
            event_shape=self.state_shape,
        )

    def perturb(
        self,
        key: Key[Array, ""],
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike = 0.0,
        t1: ArrayLike,
    ) -> Array:
        return self.marginal_transition(state, t0=t0, t1=t1).sample(key)

    def conditional_score(
        self,
        value: ArrayLike,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike = 0.0,
        t1: ArrayLike,
    ) -> Array:
        return self.marginal_transition(state, t0=t0, t1=t1).score(value)


class VariancePreservingDiffusion(AbstractGaussianDiffusion):
    """Linear-beta variance-preserving diffusion with exact Gaussian transitions."""

    beta_minimum: float = eqx.field(static=True)
    beta_maximum: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        beta_minimum: float = 0.1,
        beta_maximum: float = 20.0,
        terminal_time: float = 1.0,
        process_id: str | None = None,
    ):
        minimum = float(beta_minimum)
        maximum = float(beta_maximum)
        if not isfinite(minimum) or not isfinite(maximum) or minimum <= 0.0:
            raise ValueError("VP beta bounds must be finite and beta_minimum positive.")
        if maximum < minimum:
            raise ValueError("beta_maximum must be at least beta_minimum.")
        resolved_id = process_id or canonical_fingerprint(
            {
                "kind": "variance-preserving-diffusion",
                "dimension": int(dimension),
                "beta_minimum": minimum,
                "beta_maximum": maximum,
                "terminal_time": float(terminal_time),
            }
        )
        super().__init__(dimension, terminal_time, resolved_id)
        self.beta_minimum = minimum
        self.beta_maximum = maximum

    def beta(self, time: ArrayLike, /) -> Array:
        value = self._time(time)
        fraction = value / self.terminal_time
        return self.beta_minimum + fraction * (self.beta_maximum - self.beta_minimum)

    def integrated_beta(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        start, end = self._interval(t0, t1)
        slope = (self.beta_maximum - self.beta_minimum) / self.terminal_time
        return self.beta_minimum * (end - start) + 0.5 * slope * (end**2 - start**2)

    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        state_array = _inexact_state(state, self.state_shape)
        return -0.5 * self.beta(time).astype(state_array.dtype) * state_array

    def diffusion_scale(self, time: ArrayLike, /) -> Array:
        return jnp.sqrt(self.beta(time))

    def transition_mean_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        return jnp.exp(-0.5 * self.integrated_beta(t0, t1))

    def transition_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        integrated = self.integrated_beta(t0, t1)
        return jnp.sqrt(-jnp.expm1(-integrated))

    def asymptotic_terminal_reference(self) -> DiffusionTerminalReference:
        law = DiagonalNormalLaw(
            jnp.zeros(self.state_shape),
            jnp.ones(self.state_shape),
            event_shape=self.state_shape,
        )
        residual = self.transition_mean_scale(0.0, self.terminal_time)
        reference_id = canonical_fingerprint(
            {
                "kind": "vp-asymptotic-terminal-reference",
                "process_id": self.process_id,
                "residual_signal_scale": float(residual),
            }
        )
        return DiffusionTerminalReference(
            law,
            relationship="asymptotic",
            residual_signal_scale=residual,
            reference_id=reference_id,
            process_id=self.process_id,
        )


class VarianceExplodingDiffusion(AbstractGaussianDiffusion):
    """Geometric-scale variance-exploding diffusion with zero initial variance."""

    initial_scale: float = eqx.field(static=True)
    terminal_scale: float = eqx.field(static=True)
    log_scale_ratio: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        initial_scale: float = 0.01,
        terminal_scale: float = 50.0,
        terminal_time: float = 1.0,
        process_id: str | None = None,
    ):
        initial = float(initial_scale)
        terminal = float(terminal_scale)
        if not isfinite(initial) or not isfinite(terminal) or initial <= 0.0:
            raise ValueError("VE scales must be finite and initial_scale positive.")
        if terminal <= initial:
            raise ValueError("terminal_scale must exceed initial_scale.")
        ratio = float(jnp.log(terminal / initial))
        resolved_id = process_id or canonical_fingerprint(
            {
                "kind": "variance-exploding-diffusion",
                "dimension": int(dimension),
                "initial_scale": initial,
                "terminal_scale": terminal,
                "terminal_time": float(terminal_time),
            }
        )
        super().__init__(dimension, terminal_time, resolved_id)
        self.initial_scale = initial
        self.terminal_scale = terminal
        self.log_scale_ratio = ratio

    def reference_scale(self, time: ArrayLike, /) -> Array:
        value = self._time(time)
        return self.initial_scale * jnp.exp(self.log_scale_ratio * value / self.terminal_time)

    def transition_variance(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        start, end = self._interval(t0, t1)
        start_scale = self.reference_scale(start)
        exponent = 2.0 * self.log_scale_ratio * (end - start) / self.terminal_time
        return start_scale**2 * jnp.expm1(exponent)

    def drift(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        self._time(time)
        state_array = _inexact_state(state, self.state_shape)
        return jnp.zeros_like(state_array)

    def diffusion_scale(self, time: ArrayLike, /) -> Array:
        rate = 2.0 * self.log_scale_ratio / self.terminal_time
        return self.reference_scale(time) * jnp.sqrt(rate)

    def transition_mean_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        self._interval(t0, t1)
        return jnp.asarray(1.0)

    def transition_scale(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        return jnp.sqrt(self.transition_variance(t0, t1))

    def asymptotic_terminal_reference(self) -> DiffusionTerminalReference:
        scale = self.transition_scale(0.0, self.terminal_time)
        law = DiagonalNormalLaw(
            jnp.zeros(self.state_shape),
            jnp.full(self.state_shape, scale),
            event_shape=self.state_shape,
        )
        reference_id = canonical_fingerprint(
            {
                "kind": "ve-asymptotic-terminal-reference",
                "process_id": self.process_id,
                "terminal_noise_scale": float(scale),
            }
        )
        return DiffusionTerminalReference(
            law,
            relationship="asymptotic",
            residual_signal_scale=1.0,
            reference_id=reference_id,
            process_id=self.process_id,
        )


__all__ = [
    "AbstractGaussianDiffusion",
    "DiffusionTerminalReference",
    "TerminalReferenceRelationship",
    "VarianceExplodingDiffusion",
    "VariancePreservingDiffusion",
]
