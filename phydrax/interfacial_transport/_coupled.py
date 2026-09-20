#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative bulk–surface species transport."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import AdsorptionKinetics


@dataclass(frozen=True, slots=True)
class BulkSurfaceTransportStep:
    bulk_concentration_mol_m3: Array
    surface_concentration_mol_m2: Array
    transferred_to_surface_mol: Array
    total_mole_balance_residual: Array
    minimum_concentration: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CoupledBulkSurfaceTransport:
    """Implicit surface transport with conservative finite-volume adsorption."""

    bulk_volumes_m3: Array
    surface_areas_m2: Array
    surface_transport_generator_s_inv: Array
    bulk_from_surface: Array
    kinetics: AdsorptionKinetics

    @classmethod
    def create(
        cls,
        bulk_volumes_m3: ArrayLike,
        surface_areas_m2: ArrayLike,
        surface_transport_generator_s_inv: ArrayLike,
        bulk_from_surface: ArrayLike,
        kinetics: AdsorptionKinetics,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> CoupledBulkSurfaceTransport:
        volumes = np.asarray(bulk_volumes_m3, dtype=np.float64)
        areas = np.asarray(surface_areas_m2, dtype=np.float64)
        generator = np.asarray(surface_transport_generator_s_inv, dtype=np.float64)
        coupling = np.asarray(bulk_from_surface, dtype=np.float64)
        if volumes.ndim != 1 or volumes.size == 0 or np.any(volumes <= 0):
            raise ValueError("Bulk control volumes must be a positive vector.")
        if areas.ndim != 1 or areas.size == 0 or np.any(areas <= 0):
            raise ValueError("Surface control areas must be a positive vector.")
        if generator.shape != (areas.size, areas.size):
            raise ValueError("Surface transport generator has incompatible shape.")
        if not np.allclose(areas @ generator, 0, atol=tolerance, rtol=tolerance):
            raise ValueError(
                "Surface transport generator must conserve area-weighted content."
            )
        if coupling.shape != (volumes.size, areas.size) or np.any(coupling < 0):
            raise ValueError("Bulk–surface coupling has incompatible shape or signs.")
        if not np.allclose(coupling.sum(axis=0), 1, atol=tolerance, rtol=0):
            raise ValueError("Every surface control area must map completely to bulk.")
        return cls(
            jnp.asarray(volumes),
            jnp.asarray(areas),
            jnp.asarray(generator),
            jnp.asarray(coupling),
            kinetics,
        )

    def advance(
        self,
        bulk_concentration_mol_m3: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        step_size_s: float,
        /,
    ) -> BulkSurfaceTransportStep:
        bulk = jnp.asarray(bulk_concentration_mol_m3)
        surface = jnp.asarray(surface_concentration_mol_m2)
        if bulk.shape != self.bulk_volumes_m3.shape:
            raise ValueError("Bulk concentration does not match transport topology.")
        if surface.shape != self.surface_areas_m2.shape:
            raise ValueError("Surface concentration does not match transport topology.")
        if step_size_s <= 0 or bool(jnp.any(bulk < 0) | jnp.any(surface < 0)):
            raise ValueError("Transport state and step size must be non-negative.")

        initial_total = contract("b,b->", self.bulk_volumes_m3, bulk) + contract(
            "s,s->", self.surface_areas_m2, surface
        )
        matrix = jnp.eye(surface.size, dtype=surface.dtype) - float(
            step_size_s
        ) * self.surface_transport_generator_s_inv.astype(surface.dtype)
        transported = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            surface,
            policy=LinearSolvePolicy(DenseLU()),
        )
        if bool(jnp.any(transported.value < -1e-12)):
            raise ValueError("Surface transport operator violated positivity.")
        transported_surface = jnp.maximum(transported.value, 0)

        local_bulk = self.bulk_from_surface.T @ bulk
        requested = (
            float(step_size_s)
            * self.surface_areas_m2
            * self.kinetics.rate(local_bulk, transported_surface)
        )
        surface_moles = self.surface_areas_m2 * transported_surface
        maximum_surface_moles = (
            self.surface_areas_m2 * self.kinetics.maximum_surface_concentration_mol_m2
        )
        desorption = jnp.maximum(-requested, 0)
        desorption = jnp.minimum(desorption, surface_moles)
        surface_moles = surface_moles - desorption
        bulk_moles = self.bulk_volumes_m3 * bulk + self.bulk_from_surface @ desorption

        adsorption = jnp.maximum(requested, 0)
        adsorption = jnp.minimum(adsorption, maximum_surface_moles - surface_moles)
        requested_bulk = self.bulk_from_surface @ adsorption
        ratios = jnp.where(
            requested_bulk > 0,
            bulk_moles / jnp.maximum(requested_bulk, jnp.finfo(bulk.dtype).tiny),
            jnp.inf,
        )
        scale = jnp.minimum(1.0, jnp.min(ratios))
        adsorption = scale * adsorption
        bulk_moles = bulk_moles - self.bulk_from_surface @ adsorption
        surface_moles = surface_moles + adsorption
        transferred = adsorption - desorption

        next_bulk = bulk_moles / self.bulk_volumes_m3
        next_surface = surface_moles / self.surface_areas_m2
        final_total = jnp.sum(bulk_moles) + jnp.sum(surface_moles)
        minimum = jnp.minimum(jnp.min(next_bulk), jnp.min(next_surface))
        successful = (
            transported.successful
            & jnp.all(jnp.isfinite(next_bulk))
            & jnp.all(jnp.isfinite(next_surface))
            & (minimum >= 0)
        )
        return BulkSurfaceTransportStep(
            next_bulk,
            next_surface,
            transferred,
            final_total - initial_total,
            minimum,
            successful,
        )


__all__ = ["BulkSurfaceTransportStep", "CoupledBulkSurfaceTransport"]
