#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative segmented catalytic plug-flow reactors."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract


@dataclass(frozen=True, slots=True)
class CatalyticReactorResult:
    species_molar_flow_mol_s: Array
    temperature_k: Array
    reaction_extent_mol_s: Array
    conserved_quantity_residual: Array
    minimum_species_flow_mol_s: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SegmentedCatalyticReactor:
    catalyst_area_m2: Array
    stoichiometry: Array
    reaction_orders: Array
    rate_constants_mol_m2_s: Array
    reaction_enthalpy_j_mol: Array
    conservation_matrix: Array
    volumetric_flow_m3_s: float
    heat_capacity_flow_w_k: float
    tolerance: float

    @classmethod
    def create(
        cls,
        catalyst_area_m2: ArrayLike,
        stoichiometry: ArrayLike,
        reaction_orders: ArrayLike,
        rate_constants_mol_m2_s: ArrayLike,
        reaction_enthalpy_j_mol: ArrayLike,
        conservation_matrix: ArrayLike,
        volumetric_flow_m3_s: float,
        heat_capacity_flow_w_k: float,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> SegmentedCatalyticReactor:
        area = np.asarray(catalyst_area_m2, dtype=np.float64)
        stoichiometry_ = np.asarray(stoichiometry, dtype=np.float64)
        orders = np.asarray(reaction_orders, dtype=np.float64)
        constants = np.asarray(rate_constants_mol_m2_s, dtype=np.float64)
        enthalpy = np.asarray(reaction_enthalpy_j_mol, dtype=np.float64)
        conservation = np.asarray(conservation_matrix, dtype=np.float64)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Catalytic tolerance must be finite and positive.")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                area,
                stoichiometry_,
                orders,
                constants,
                enthalpy,
                conservation,
            )
        ):
            raise ValueError("Catalytic operators and properties must be finite.")
        if area.ndim != 1 or area.size == 0 or np.any(area <= 0):
            raise ValueError("Catalyst segment areas must be a positive vector.")
        if stoichiometry_.ndim != 2 or stoichiometry_.shape[0] == 0:
            raise ValueError("Catalyst stoichiometry requires reaction-by-species shape.")
        if orders.shape != stoichiometry_.shape or np.any(orders < 0):
            raise ValueError("Catalyst reaction orders must be non-negative and aligned.")
        reaction_count = stoichiometry_.shape[0]
        if constants.ndim == 1:
            constants = np.broadcast_to(constants, (area.size, reaction_count))
        if constants.shape != (area.size, reaction_count) or np.any(constants < 0):
            raise ValueError(
                "Catalyst rate constants must be non-negative and segment aligned."
            )
        if enthalpy.shape != (reaction_count,):
            raise ValueError("Catalyst reaction enthalpies must align with reactions.")
        if conservation.ndim != 2 or conservation.shape[1] != stoichiometry_.shape[1]:
            raise ValueError("Catalyst conservation matrix must map species quantities.")
        if not np.allclose(
            conservation @ stoichiometry_.T, 0, atol=tolerance, rtol=tolerance
        ):
            raise ValueError(
                "Catalyst stoichiometry violates declared conservation laws."
            )
        if (
            not np.isfinite(volumetric_flow_m3_s)
            or volumetric_flow_m3_s <= 0
            or not np.isfinite(heat_capacity_flow_w_k)
            or heat_capacity_flow_w_k <= 0
        ):
            raise ValueError(
                "Catalyst flow and heat-capacity rates must be finite and positive."
            )
        return cls(
            jnp.asarray(area),
            jnp.asarray(stoichiometry_),
            jnp.asarray(orders),
            jnp.asarray(constants),
            jnp.asarray(enthalpy),
            jnp.asarray(conservation),
            float(volumetric_flow_m3_s),
            float(heat_capacity_flow_w_k),
            float(tolerance),
        )

    def solve(
        self, inlet_species_molar_flow_mol_s: ArrayLike, inlet_temperature_k: float, /
    ) -> CatalyticReactorResult:
        flow = jnp.asarray(inlet_species_molar_flow_mol_s)
        species_count = self.stoichiometry.shape[1]
        if flow.shape != (species_count,):
            raise ValueError("Catalyst inlet species flows are invalid.")
        flow = eqx.error_if(
            flow,
            jnp.any(~jnp.isfinite(flow) | (flow < 0)),
            "Catalyst inlet species flows must be finite and nonnegative.",
        )
        temperature = jnp.asarray(inlet_temperature_k, dtype=flow.dtype)
        temperature = eqx.error_if(
            temperature,
            ~jnp.isfinite(temperature) | (temperature <= 0),
            "Catalyst inlet temperature must be finite and positive.",
        )
        initial = flow
        successful = jnp.asarray(True)
        flow_profile = [flow]
        temperature_profile = [temperature]
        extent_profile = []
        for segment in range(self.catalyst_area_m2.size):
            concentration = flow / self.volumetric_flow_m3_s
            rates = self.rate_constants_mol_m2_s[segment] * jnp.prod(
                jnp.where(
                    self.reaction_orders > 0,
                    concentration[None, :] ** self.reaction_orders,
                    1.0,
                ),
                axis=1,
            )
            requested_extent = self.catalyst_area_m2[segment] * rates
            consumption = contract(
                "rs,r->s",
                jnp.maximum(-self.stoichiometry, 0),
                requested_extent,
            )
            ratios = jnp.where(
                consumption > 0,
                flow / jnp.maximum(consumption, jnp.finfo(flow.dtype).tiny),
                jnp.inf,
            )
            scale = jnp.minimum(1.0, jnp.min(ratios))
            extent = scale * requested_extent
            candidate_flow = flow + self.stoichiometry.T @ extent
            heat_release = -contract("r,r->", self.reaction_enthalpy_j_mol, extent)
            candidate_temperature = (
                temperature + heat_release / self.heat_capacity_flow_w_k
            )
            segment_valid = (
                jnp.all(jnp.isfinite(candidate_flow))
                & jnp.all(candidate_flow >= -self.tolerance)
                & jnp.isfinite(candidate_temperature)
                & (candidate_temperature > 0)
            )
            successful = successful & segment_valid
            flow = jnp.where(segment_valid, candidate_flow, flow)
            temperature = jnp.where(segment_valid, candidate_temperature, temperature)
            extent = jnp.where(segment_valid, extent, jnp.zeros_like(extent))
            flow_profile.append(flow)
            temperature_profile.append(temperature)
            extent_profile.append(extent)
        flows = jnp.stack(flow_profile)
        temperatures = jnp.stack(temperature_profile)
        extents = jnp.stack(extent_profile)
        conservation_residual = self.conservation_matrix @ (flow - initial)
        minimum = jnp.min(flows)
        successful = (
            successful
            & jnp.all(jnp.isfinite(flows))
            & jnp.all(jnp.isfinite(temperatures))
            & (minimum >= -self.tolerance)
            & jnp.all(temperatures > 0)
            & jnp.all(jnp.isfinite(conservation_residual))
            & (jnp.linalg.norm(conservation_residual) <= self.tolerance)
        )
        return CatalyticReactorResult(
            flows,
            temperatures,
            extents,
            conservation_residual,
            minimum,
            successful,
        )


__all__ = ["CatalyticReactorResult", "SegmentedCatalyticReactor"]
