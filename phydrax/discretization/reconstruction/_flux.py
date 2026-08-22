#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._weno import WENOReconstructionPlan


class RusanovFluxPlan(StrictModule, NonTrainableState):
    """Local Lax–Friedrichs flux with explicit physical flux and wave speed."""

    flux: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    wave_speed: Callable[[Array, Array, Any], ArrayLike] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux: Callable[[Array, Any], ArrayLike],
        wave_speed: Callable[[Array, Array, Any], ArrayLike],
        /,
    ):
        if not callable(flux) or not callable(wave_speed):
            raise TypeError("flux and wave_speed must be callable.")
        self.flux = flux
        self.wave_speed = wave_speed
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rusanov-flux",
                "flux": repr(flux),
                "wave_speed": repr(wave_speed),
            }
        )

    def face_flux(self, left: ArrayLike, right: ArrayLike, args: Any = None) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape:
            raise ValueError("Rusanov left and right states must align.")
        left_flux = jnp.asarray(self.flux(left_, args))
        right_flux = jnp.asarray(self.flux(right_, args))
        if left_flux.shape != left_.shape or right_flux.shape != right_.shape:
            raise ValueError("Physical flux must preserve the state shape.")
        speed = jnp.asarray(self.wave_speed(left_, right_, args))
        if speed.shape == left_.shape[:-1] and left_.ndim > 1:
            speed = speed[..., None]
        if speed.shape == ():
            speed = jnp.broadcast_to(speed, left_.shape)
        if speed.shape != left_.shape:
            raise ValueError(
                "Wave speed must be scalar or broadcast over state channels."
            )
        speed = eqx.error_if(
            speed,
            jnp.any(~jnp.isfinite(speed)) | jnp.any(speed < 0.0),
            "Rusanov wave speed must be finite and non-negative.",
        )
        return 0.5 * (left_flux + right_flux) - 0.5 * speed * (right_ - left_)


class FluxDifferenceDynamics1D(StrictModule):
    """Periodic conservative WENO/Rusanov method with SSPRK3 stepping."""

    reconstruction: WENOReconstructionPlan
    numerical_flux: RusanovFluxPlan
    spacing: Array
    source: Callable[[Array, Array, Any], ArrayLike] | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: WENOReconstructionPlan,
        numerical_flux: RusanovFluxPlan,
        spacing: ArrayLike,
        /,
        *,
        source: Callable[[Array, Array, Any], ArrayLike] | None = None,
    ):
        if not isinstance(reconstruction, WENOReconstructionPlan) or not isinstance(
            numerical_flux, RusanovFluxPlan
        ):
            raise TypeError("Flux dynamics requires WENO and Rusanov plans.")
        spacing_ = jnp.asarray(spacing)
        if spacing_.shape != () or not bool(np.isfinite(np.asarray(spacing_))):
            raise ValueError("spacing must be one finite scalar.")
        if float(spacing_) <= 0.0:
            raise ValueError("spacing must be positive.")
        if source is not None and not callable(source):
            raise TypeError("source must be callable or None.")
        self.reconstruction = reconstruction
        self.numerical_flux = numerical_flux
        self.spacing = spacing_
        self.source = source
        self.method_id = canonical_fingerprint(
            {
                "kind": "flux-difference-dynamics",
                "reconstruction": reconstruction.plan_id,
                "numerical_flux": numerical_flux.plan_id,
                "spacing": float(spacing_),
                "source": None if source is None else repr(source),
            }
        )

    def face_flux(self, state: ArrayLike, args: Any = None, /) -> Array:
        left, right = self.reconstruction.reconstruct(state)
        return self.numerical_flux.face_flux(left, right, args)

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        value = jnp.asarray(state)
        flux = self.face_flux(value, args)
        derivative = -(flux - jnp.roll(flux, 1, axis=0)) / self.spacing
        if self.source is not None:
            source = jnp.asarray(self.source(jnp.asarray(time), value, args))
            if source.shape != value.shape:
                raise ValueError("Flux-difference source must match the state shape.")
            derivative = derivative + source
        return derivative

    def stable_step(self, state: ArrayLike, cfl: float = 0.4, args: Any = None) -> Array:
        value = jnp.asarray(state)
        left, right = self.reconstruction.reconstruct(value)
        speed = jnp.asarray(self.numerical_flux.wave_speed(left, right, args))
        maximum = jnp.max(speed)
        return (
            float(cfl) * self.spacing / jnp.maximum(maximum, jnp.finfo(value.dtype).tiny)
        )

    def ssprk3_step(
        self,
        time: Array,
        state: Array,
        step_size: ArrayLike,
        args: Any = None,
    ) -> Array:
        dt = jnp.asarray(step_size)
        first = state + dt * self(time, state, args)
        second = 0.75 * state + 0.25 * (
            first + dt * self(jnp.asarray(time) + dt, first, args)
        )
        return (1.0 / 3.0) * state + (2.0 / 3.0) * (
            second + dt * self(jnp.asarray(time) + 0.5 * dt, second, args)
        )


__all__ = ["FluxDifferenceDynamics1D", "RusanovFluxPlan"]
