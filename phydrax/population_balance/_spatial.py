#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


def spatial_population_rate(
    flux_divergence: ArrayLike, internal_rate: ArrayLike, source: ArrayLike = 0.0, /
):
    flux = jnp.asarray(flux_divergence)
    internal = jnp.asarray(internal_rate)
    source_ = jnp.asarray(source)
    if flux.shape != internal.shape or source_.shape not in ((), flux.shape):
        raise ValueError("Spatial population rates must be scalar or shape aligned.")
    flux = eqx.error_if(
        flux,
        jnp.any(~jnp.isfinite(flux) | ~jnp.isfinite(internal) | ~jnp.isfinite(source_)),
        "Spatial population rates must be finite.",
    )
    return -flux + internal + source_


@dataclass(frozen=True, slots=True)
class SpatialPopulationStep:
    cell_number: Array
    boundary_balance_residual: Array
    minimum_cell_number: Array
    candidate_cell_number: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialPopulationTransport:
    """One-dimensional conservative finite-volume transport for every section."""

    cell_volumes: Array

    @classmethod
    def create(cls, cell_volumes: ArrayLike, /) -> SpatialPopulationTransport:
        volumes = np.asarray(cell_volumes, dtype=np.float64)
        if (
            volumes.ndim != 1
            or volumes.size == 0
            or not np.all(np.isfinite(volumes))
            or np.any(volumes <= 0)
        ):
            raise ValueError(
                "Population control volumes must be a finite positive vector."
            )
        return cls(jnp.asarray(volumes))

    def face_number_flux(
        self,
        cell_number: ArrayLike,
        volume_flux_m3_s: ArrayLike,
        /,
        *,
        left_inflow_density: ArrayLike = 0.0,
        right_inflow_density: ArrayLike = 0.0,
    ) -> Array:
        number = jnp.asarray(cell_number)
        volume_flux = jnp.asarray(volume_flux_m3_s)
        if number.ndim != 2 or number.shape[0] != self.cell_volumes.size:
            raise ValueError("Spatial population requires shape (cell, section).")
        if volume_flux.shape != (self.cell_volumes.size + 1,):
            raise ValueError("Volume flux requires one value per control-volume face.")
        number = eqx.error_if(
            number,
            jnp.any(~jnp.isfinite(number) | (number < 0))
            | jnp.any(~jnp.isfinite(volume_flux)),
            "Population numbers must be finite/nonnegative and fluxes finite.",
        )
        density = number / self.cell_volumes[:, None]
        left_boundary = jnp.broadcast_to(
            jnp.asarray(left_inflow_density), (number.shape[1],)
        )
        right_boundary = jnp.broadcast_to(
            jnp.asarray(right_inflow_density), (number.shape[1],)
        )
        number = eqx.error_if(
            number,
            jnp.any(
                ~jnp.isfinite(left_boundary)
                | ~jnp.isfinite(right_boundary)
                | (left_boundary < 0)
                | (right_boundary < 0)
            ),
            "Population inflow densities must be finite and nonnegative.",
        )
        interior = jnp.where(volume_flux[1:-1, None] >= 0, density[:-1], density[1:])
        left = jnp.where(volume_flux[0] >= 0, left_boundary, density[0])
        right = jnp.where(volume_flux[-1] >= 0, density[-1], right_boundary)
        upwind = jnp.concatenate((left[None, :], interior, right[None, :]), axis=0)
        return volume_flux[:, None] * upwind

    def advance(
        self,
        cell_number: ArrayLike,
        volume_flux_m3_s: ArrayLike,
        step_size_s: float,
        /,
        *,
        left_inflow_density: ArrayLike = 0.0,
        right_inflow_density: ArrayLike = 0.0,
    ) -> SpatialPopulationStep:
        if not isfinite(step_size_s) or step_size_s <= 0:
            raise ValueError(
                "Population transport step size must be finite and positive."
            )
        number = jnp.asarray(cell_number)
        flux = self.face_number_flux(
            number,
            volume_flux_m3_s,
            left_inflow_density=left_inflow_density,
            right_inflow_density=right_inflow_density,
        )
        rate = flux[:-1] - flux[1:]
        candidate = number + float(step_size_s) * rate
        successful = jnp.all(jnp.isfinite(candidate)) & jnp.all(candidate >= -1e-12)
        accepted = jnp.where(successful, jnp.maximum(candidate, 0), number)
        boundary_change = float(step_size_s) * (flux[0] - flux[-1])
        balance = jnp.sum(accepted - number, axis=0) - jnp.where(
            successful, boundary_change, jnp.zeros_like(boundary_change)
        )
        successful = (
            successful
            & jnp.all(jnp.isfinite(balance))
            & jnp.all(jnp.abs(balance) <= 1e-10)
        )
        accepted = jnp.where(successful, accepted, number)
        return SpatialPopulationStep(
            accepted,
            balance,
            jnp.min(accepted),
            candidate,
            successful,
        )


__all__ = [
    "SpatialPopulationStep",
    "SpatialPopulationTransport",
    "spatial_population_rate",
]
