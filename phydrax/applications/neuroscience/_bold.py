#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persistent positive-flow Balloon-Windkessel state and fractional BOLD."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._regional import _parameter


class NeuralBOLDDrive(StrictModule):
    """Explicit affine neural-to-vasodilatory drive.

    ``gain * sum(component_weights * (neural - baseline), axis=-1)`` is the
    drive in inverse seconds squared. Weights/baseline are [2] or [region,2];
    gain is a scalar or [region]. Baseline is required, never estimated from
    the observed trajectory or reset at a continuation boundary.
    """

    component_weights: Array
    baseline: Array
    gain: Array

    def __init__(
        self,
        component_weights: ArrayLike,
        baseline: ArrayLike,
        /,
        *,
        gain: ArrayLike = 1.0,
    ):
        weights = jnp.asarray(component_weights)
        reference = jnp.asarray(baseline)
        for value in (weights, reference):
            if (
                value.ndim not in (1, 2)
                or value.shape[-1] != 2
                or jnp.iscomplexobj(value)
            ):
                raise ValueError(
                    "BOLD drive weights/baseline require real [2] or [region,2] arrays."
                )
        weights = weights.astype(jnp.result_type(weights.dtype, float))
        reference = reference.astype(jnp.result_type(reference.dtype, float))
        self.component_weights = eqx.error_if(
            weights, ~jnp.all(jnp.isfinite(weights)), "BOLD drive weights must be finite."
        )
        self.baseline = eqx.error_if(
            reference,
            ~jnp.all(jnp.isfinite(reference)),
            "BOLD drive baseline must be finite.",
        )
        self.gain = _parameter(gain, "BOLD drive gain")

    def __call__(self, neural: Array, /) -> Array:
        return self.gain * jnp.sum(
            self.component_weights * (neural - self.baseline), axis=-1
        )


class BalloonWindkessel(StrictModule):
    """Balloon model in state ``(s, log(f), log(v), log(q))`` per region.

    ``s`` is the vasodilatory signal (1/s); flow f, volume v and deoxyhemoglobin
    q are positive ratios to resting values. ``s'=u-kappa*s-gamma*(f-1)`` and
    ``f'=s``. Transit time is seconds, alpha/extraction/resting_volume are
    dimensionless fractions. Every numerical parameter is scalar or [region].

    BOLD is a fractional signal change, not percent: multiply by 100 explicitly
    if measurements are percent. Default coefficients are k1=7*E0, k2=2,
    k3=2*E0-0.2; custom acquisition coefficients must be declared explicitly.
    """

    kappa_per_s: Array
    gamma_per_s2: Array
    transit_s: Array
    alpha: Array
    extraction: Array
    resting_volume: Array
    k1: Array | None
    k2: Array
    k3: Array | None

    def __init__(
        self,
        *,
        kappa_per_s: ArrayLike = 0.65,
        gamma_per_s2: ArrayLike = 0.41,
        transit_s: ArrayLike = 0.98,
        alpha: ArrayLike = 0.32,
        extraction: ArrayLike = 0.34,
        resting_volume: ArrayLike = 0.02,
        k1: ArrayLike | None = None,
        k2: ArrayLike = 2.0,
        k3: ArrayLike | None = None,
    ):
        self.kappa_per_s = _parameter(kappa_per_s, "kappa_per_s", positive=True)
        self.gamma_per_s2 = _parameter(gamma_per_s2, "gamma_per_s2", positive=True)
        self.transit_s = _parameter(transit_s, "transit_s", positive=True)
        self.alpha = _parameter(alpha, "alpha", positive=True)
        extraction_ = _parameter(extraction, "extraction", positive=True)
        self.extraction = eqx.error_if(
            extraction_,
            jnp.any(extraction_ >= 1.0),
            "extraction must lie strictly between zero and one.",
        )
        volume = _parameter(resting_volume, "resting_volume", positive=True)
        self.resting_volume = eqx.error_if(
            volume,
            jnp.any(volume >= 1.0),
            "resting_volume must lie strictly between zero and one.",
        )
        self.k1 = None if k1 is None else _parameter(k1, "k1")
        self.k2 = _parameter(k2, "k2")
        self.k3 = None if k3 is None else _parameter(k3, "k3")

    def __call__(self, state: ArrayLike, neural_drive: ArrayLike, /) -> Array:
        values = jnp.asarray(state)
        if values.ndim < 1 or values.shape[-1] != 4:
            raise ValueError(
                "Balloon state must have a trailing (s,log(f),log(v),log(q)) axis."
            )
        signal, log_flow, log_volume, log_deoxy = (
            values[..., index] for index in range(4)
        )
        flow = jnp.exp(log_flow)
        volume_outflow_ratio = jnp.exp((1.0 / self.alpha - 1.0) * log_volume)
        # expm1 retains oxygen-extraction accuracy near small E0 / high flow.
        extracted = -jnp.expm1(jnp.log1p(-self.extraction) / flow)
        d_signal = (
            jnp.asarray(neural_drive)
            - self.kappa_per_s * signal
            - self.gamma_per_s2 * jnp.expm1(log_flow)
        )
        d_log_flow = signal * jnp.exp(-log_flow)
        d_log_volume = (
            jnp.exp(log_flow - log_volume) - volume_outflow_ratio
        ) / self.transit_s
        d_log_deoxy = (
            jnp.exp(log_flow - log_deoxy) * extracted / self.extraction
            - volume_outflow_ratio
        ) / self.transit_s
        return jnp.stack((d_signal, d_log_flow, d_log_volume, d_log_deoxy), axis=-1)

    def bold(self, state: ArrayLike, /) -> Array:
        """Fractional BOLD without clipping or hiding invalid physical states."""
        values = jnp.asarray(state)
        if values.ndim < 1 or values.shape[-1] != 4:
            raise ValueError("Balloon state must have a trailing four-coordinate axis.")
        log_volume, log_deoxy = values[..., 2], values[..., 3]
        k1 = 7.0 * self.extraction if self.k1 is None else self.k1
        k3 = 2.0 * self.extraction - 0.2 if self.k3 is None else self.k3
        return -self.resting_volume * (
            k1 * jnp.expm1(log_deoxy)
            + self.k2 * jnp.expm1(log_deoxy - log_volume)
            + k3 * jnp.expm1(log_volume)
        )


def balloon_equilibrium(region_count: int, /, *, dtype=None) -> Array:
    """Resting state s=0, f=v=q=1, exact zero fractional BOLD."""
    if (
        isinstance(region_count, bool)
        or not isinstance(region_count, int)
        or region_count < 1
    ):
        raise ValueError("region_count must be a positive integer.")
    return jnp.zeros((region_count, 4), dtype=dtype)


def _balloon_valid(state: Array, /) -> Array:
    ratios = jnp.exp(state[..., 1:])
    return jnp.all(jnp.isfinite(state), axis=-1) & jnp.all(
        jnp.isfinite(ratios) & (ratios > 0.0), axis=-1
    )


__all__ = ["BalloonWindkessel", "NeuralBOLDDrive", "balloon_equilibrium"]
