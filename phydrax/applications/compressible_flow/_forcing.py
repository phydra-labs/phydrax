#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CompressibleForcingResult(StrictModule):
    """Conservative source and exact mass/momentum/energy work ledger."""

    source: Array
    mass_source: Array
    momentum_source: Array
    total_energy_source: Array
    acceleration_work: Array
    injected_total_energy: Array
    volumetric_heating: Array
    work_identity_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class CompressibleForcingPlan(StrictModule, NonTrainableState):
    """Body acceleration, mass injection, and heat in conserved variables."""

    acceleration: tuple[float, ...] = eqx.field(static=True)
    injection_velocity: tuple[float, ...] = eqx.field(static=True)
    mass_rate: float = eqx.field(static=True)
    injection_specific_internal_energy: float = eqx.field(static=True)
    volumetric_heating: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        acceleration: Sequence[float] | None = None,
        mass_rate: float = 0.0,
        injection_velocity: Sequence[float] | None = None,
        injection_specific_internal_energy: float = 0.0,
        volumetric_heating: float = 0.0,
    ):
        dimension_ = int(dimension)
        acceleration_ = (
            (0.0,) * dimension_
            if acceleration is None
            else tuple(float(value) for value in acceleration)
        )
        injection_velocity_ = (
            (0.0,) * dimension_
            if injection_velocity is None
            else tuple(float(value) for value in injection_velocity)
        )
        mass_rate_ = float(mass_rate)
        injection_internal = float(injection_specific_internal_energy)
        heating = float(volumetric_heating)
        if (
            dimension_ not in (1, 2, 3)
            or len(acceleration_) != dimension_
            or len(injection_velocity_) != dimension_
            or any(
                not np.isfinite(value)
                for value in (
                    *acceleration_,
                    *injection_velocity_,
                    mass_rate_,
                    injection_internal,
                    heating,
                )
            )
            or injection_internal < 0.0
        ):
            raise ValueError("Compressible forcing values are invalid.")
        self.dimension = dimension_
        self.acceleration = acceleration_
        self.mass_rate = mass_rate_
        self.injection_velocity = injection_velocity_
        self.injection_specific_internal_energy = injection_internal
        self.volumetric_heating = heating
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-conservative-forcing",
                "dimension": dimension_,
                "acceleration": acceleration_,
                "mass_rate": mass_rate_,
                "injection_velocity": injection_velocity_,
                "injection_specific_internal_energy": injection_internal,
                "volumetric_heating": heating,
            }
        )

    def evaluate(
        self,
        conserved: ArrayLike,
        /,
        *,
        acceleration: ArrayLike | None = None,
        mass_rate: ArrayLike | None = None,
        injection_velocity: ArrayLike | None = None,
        injection_specific_internal_energy: ArrayLike | None = None,
        volumetric_heating: ArrayLike | None = None,
    ) -> CompressibleForcingResult:
        state = jnp.asarray(conserved)
        if state.ndim < 1 or state.shape[-1] != self.dimension + 2:
            raise ValueError("Compressible forcing state has the wrong component count.")
        state = eqx.error_if(
            state,
            jnp.any(~jnp.isfinite(state) | (state[..., :1] <= 0.0)),
            "Compressible forcing state must be finite with positive density.",
        )
        field_shape = state.shape[:-1]
        density = state[..., 0]
        momentum = state[..., 1 : 1 + self.dimension]
        acceleration_ = jnp.asarray(
            self.acceleration if acceleration is None else acceleration,
            dtype=state.dtype,
        )
        injection_velocity_ = jnp.asarray(
            self.injection_velocity if injection_velocity is None else injection_velocity,
            dtype=state.dtype,
        )
        mass_rate_ = jnp.asarray(
            self.mass_rate if mass_rate is None else mass_rate, dtype=state.dtype
        )
        injection_internal = jnp.asarray(
            self.injection_specific_internal_energy
            if injection_specific_internal_energy is None
            else injection_specific_internal_energy,
            dtype=state.dtype,
        )
        heating = jnp.asarray(
            self.volumetric_heating if volumetric_heating is None else volumetric_heating,
            dtype=state.dtype,
        )
        acceleration_ = jnp.broadcast_to(acceleration_, field_shape + (self.dimension,))
        injection_velocity_ = jnp.broadcast_to(
            injection_velocity_, field_shape + (self.dimension,)
        )
        mass_rate_ = jnp.broadcast_to(mass_rate_, field_shape)
        injection_internal = jnp.broadcast_to(injection_internal, field_shape)
        heating = jnp.broadcast_to(heating, field_shape)
        acceleration_work = oe.contract(
            "...d,...d->...", momentum, acceleration_, backend="jax"
        )
        acceleration_momentum = density[..., None] * acceleration_
        injected_momentum = mass_rate_[..., None] * injection_velocity_
        momentum_source = acceleration_momentum + injected_momentum
        injection_kinetic = 0.5 * oe.contract(
            "...d,...d->...",
            injection_velocity_,
            injection_velocity_,
            backend="jax",
        )
        injected_energy = mass_rate_ * (injection_internal + injection_kinetic)
        energy_source = acceleration_work + injected_energy + heating
        source = jnp.concatenate(
            (
                mass_rate_[..., None],
                momentum_source,
                energy_source[..., None],
            ),
            axis=-1,
        )
        residual = energy_source - acceleration_work - injected_energy - heating
        finite = jnp.all(jnp.isfinite(source))
        return CompressibleForcingResult(
            source,
            mass_rate_,
            momentum_source,
            energy_source,
            acceleration_work,
            injected_energy,
            heating,
            residual,
            finite,
            self.plan_id,
        )

    __call__ = evaluate


__all__ = ["CompressibleForcingPlan", "CompressibleForcingResult"]
