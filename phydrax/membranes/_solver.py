#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Species-conservative segmented membrane modules."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class MembraneModuleResult:
    feed_molar_flow_mol_s: Array
    permeate_molar_flow_mol_s: Array
    species_flux_mol_m2_s: Array
    species_balance_residual_mol_s: Array
    stage_cut: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CrossflowMembraneModule:
    segment_area_m2: Array
    species_permeance_mol_m2_s_pa: Array
    feed_pressure_pa: Array
    permeate_pressure_pa: Array

    @classmethod
    def create(
        cls,
        segment_area_m2: ArrayLike,
        species_permeance_mol_m2_s_pa: ArrayLike,
        feed_pressure_pa: ArrayLike,
        permeate_pressure_pa: ArrayLike,
        /,
    ) -> CrossflowMembraneModule:
        area = np.asarray(segment_area_m2, dtype=float)
        permeance = np.asarray(species_permeance_mol_m2_s_pa, dtype=float)
        feed_pressure = np.asarray(feed_pressure_pa, dtype=float)
        permeate_pressure = np.asarray(permeate_pressure_pa, dtype=float)
        if area.ndim != 1 or area.size == 0 or np.any(area <= 0):
            raise ValueError("Membrane segment areas must be a positive vector.")
        if permeance.ndim == 1:
            permeance = np.broadcast_to(permeance, (area.size, permeance.size))
        if (
            permeance.ndim != 2
            or permeance.shape[0] != area.size
            or np.any(permeance < 0)
        ):
            raise ValueError(
                "Membrane permeance must be non-negative and segment aligned."
            )
        if feed_pressure.shape != area.shape or permeate_pressure.shape != area.shape:
            raise ValueError("Membrane pressure profiles must align with segments.")
        if np.any(feed_pressure <= 0) or np.any(permeate_pressure < 0):
            raise ValueError("Membrane pressures are outside physical bounds.")
        return cls(
            jnp.asarray(area),
            jnp.asarray(permeance),
            jnp.asarray(feed_pressure),
            jnp.asarray(permeate_pressure),
        )

    def solve(
        self,
        feed_inlet_molar_flow_mol_s: ArrayLike,
        permeate_inlet_molar_flow_mol_s: ArrayLike,
        /,
    ) -> MembraneModuleResult:
        feed = jnp.asarray(feed_inlet_molar_flow_mol_s)
        permeate = jnp.asarray(permeate_inlet_molar_flow_mol_s)
        species = self.species_permeance_mol_m2_s_pa.shape[1]
        if feed.shape != (species,) or permeate.shape != (species,):
            raise ValueError("Membrane inlet species flows do not match permeance data.")
        if bool(jnp.any(feed < 0) | jnp.any(permeate < 0)):
            raise ValueError("Membrane inlet species flows must be non-negative.")
        initial = feed + permeate
        feed_profile = [feed]
        permeate_profile = [permeate]
        flux_profile = []
        for segment in range(self.segment_area_m2.size):
            feed_total = jnp.sum(feed)
            permeate_total = jnp.sum(permeate)
            feed_fraction = jnp.where(
                feed_total > 0,
                feed / jnp.maximum(feed_total, jnp.finfo(feed.dtype).tiny),
                jnp.zeros_like(feed),
            )
            permeate_fraction = jnp.where(
                permeate_total > 0,
                permeate / jnp.maximum(permeate_total, jnp.finfo(permeate.dtype).tiny),
                jnp.zeros_like(permeate),
            )
            raw_flux = self.species_permeance_mol_m2_s_pa[segment] * (
                self.feed_pressure_pa[segment] * feed_fraction
                - self.permeate_pressure_pa[segment] * permeate_fraction
            )
            raw_transfer = self.segment_area_m2[segment] * raw_flux
            transfer = jnp.where(
                raw_transfer >= 0,
                jnp.minimum(raw_transfer, feed),
                jnp.maximum(raw_transfer, -permeate),
            )
            flux = transfer / self.segment_area_m2[segment]
            feed = feed - transfer
            permeate = permeate + transfer
            feed_profile.append(feed)
            permeate_profile.append(permeate)
            flux_profile.append(flux)
        feed_values = jnp.stack(feed_profile)
        permeate_values = jnp.stack(permeate_profile)
        flux_values = jnp.stack(flux_profile)
        balance = feed + permeate - initial
        transferred_total = jnp.sum(permeate - permeate_profile[0])
        stage_cut = transferred_total / jnp.maximum(
            jnp.sum(feed_inlet_molar_flow_mol_s),
            jnp.finfo(feed.dtype).tiny,
        )
        successful = (
            jnp.all(jnp.isfinite(feed_values))
            & jnp.all(jnp.isfinite(permeate_values))
            & jnp.all(feed_values >= -1e-12)
            & jnp.all(permeate_values >= -1e-12)
        )
        return MembraneModuleResult(
            feed_values,
            permeate_values,
            flux_values,
            balance,
            stage_cut,
            successful,
        )


__all__ = ["CrossflowMembraneModule", "MembraneModuleResult"]
