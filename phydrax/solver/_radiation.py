#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedTensorGrid
from ..discretization.finite_volume import ConservativeDiffusionPlan


class RadiationMatterState(StrictModule):
    radiation_energy: Array
    material_energy: Array
    time: Array


class RadiationDiffusionDiagnostics(StrictModule):
    radiation_change: Array
    material_change: Array
    combined_energy_defect: Array
    minimum_radiation_energy: Array
    successful: Array


class GrayLinearRadiationDiffusionPlan(StrictModule, NonTrainableState):
    """Gray linear diffusion with frozen-equilibrium local matter exchange."""

    grid: PreparedTensorGrid
    diffusion: object
    transport_extinction: float = eqx.field(static=True)
    absorption_coefficient: float = eqx.field(static=True)
    reduced_light_speed: float = eqx.field(static=True)
    eddington_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        transport_extinction: float,
        absorption_coefficient: float,
        reduced_light_speed: float = 1.0,
        eddington_factor: float = 1.0 / 3.0,
    ):
        transport = float(transport_extinction)
        absorption = float(absorption_coefficient)
        speed = float(reduced_light_speed)
        factor = float(eddington_factor)
        if (
            not isinstance(grid, PreparedTensorGrid)
            or not np.isfinite(transport)
            or transport <= 0.0
            or not np.isfinite(absorption)
            or absorption <= 0.0
            or not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(factor)
            or not 0.0 < factor <= 1.0
        ):
            raise ValueError("Gray linear radiation parameters are invalid.")
        coefficient = speed * factor / transport
        diffusion = ConservativeDiffusionPlan(grid).prepare(coefficient)
        self.grid = grid
        self.diffusion = diffusion
        self.transport_extinction = transport
        self.absorption_coefficient = absorption
        self.reduced_light_speed = speed
        self.eddington_factor = factor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gray-linear-radiation-diffusion",
                "grid": grid.prepared_id,
                "transport_extinction": transport,
                "absorption_coefficient": absorption,
                "reduced_light_speed": speed,
                "eddington_factor": factor,
                "diffusion": diffusion.operator_id,
            }
        )

    def initialize(
        self,
        radiation_energy: ArrayLike,
        material_energy: ArrayLike,
        time: ArrayLike = 0.0,
        /,
    ) -> RadiationMatterState:
        radiation = jnp.asarray(radiation_energy)
        material = jnp.asarray(material_energy, dtype=radiation.dtype)
        if radiation.shape != self.grid.shape or material.shape != self.grid.shape:
            raise ValueError("Radiation and material energy must match the grid shape.")
        radiation = eqx.error_if(
            radiation,
            jnp.any(~jnp.isfinite(radiation))
            | jnp.any(~jnp.isfinite(material))
            | jnp.any(radiation <= 0.0)
            | jnp.any(material <= 0.0),
            "Radiation-matter initial energy is inadmissible.",
        )
        return RadiationMatterState(
            radiation,
            material,
            jnp.asarray(time, dtype=radiation.dtype).reshape(()),
        )

    def advance(
        self,
        state: RadiationMatterState,
        end_time: ArrayLike,
        frozen_equilibrium_radiation_energy: ArrayLike,
        /,
    ) -> tuple[RadiationMatterState, RadiationDiffusionDiagnostics]:
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - state.time
        equilibrium = jnp.broadcast_to(
            jnp.asarray(
                frozen_equilibrium_radiation_energy,
                dtype=state.radiation_energy.dtype,
            ),
            state.radiation_energy.shape,
        )
        diffusion_change = step * self.diffusion.mv(state.radiation_energy)
        diffused = state.radiation_energy + diffusion_change
        relaxation = 1.0 - jnp.exp(
            -self.absorption_coefficient * self.reduced_light_speed * step
        )
        exchange = relaxation * (equilibrium - diffused)
        radiation = diffused + exchange
        material = state.material_energy - exchange
        measure = self.grid.quadrature_weights
        combined_defect = jnp.sum(
            measure
            * (radiation + material - state.radiation_energy - state.material_energy)
        )
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(jnp.isfinite(radiation))
            & jnp.all(jnp.isfinite(material))
            & jnp.all(radiation > 0.0)
            & jnp.all(material > 0.0)
        )
        accepted = RadiationMatterState(
            jnp.where(successful, radiation, state.radiation_energy),
            jnp.where(successful, material, state.material_energy),
            jnp.where(successful, end, state.time),
        )
        diagnostics = RadiationDiffusionDiagnostics(
            radiation_change=accepted.radiation_energy - state.radiation_energy,
            material_change=accepted.material_energy - state.material_energy,
            combined_energy_defect=combined_defect,
            minimum_radiation_energy=jnp.min(accepted.radiation_energy),
            successful=successful,
        )
        return accepted, diagnostics


__all__ = [
    "GrayLinearRadiationDiffusionPlan",
    "RadiationDiffusionDiagnostics",
    "RadiationMatterState",
]
