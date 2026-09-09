#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._ablating_material import AblatingMaterialState


class ConjugateAerothermalExchange(StrictModule):
    fluid_heat_loss: Array
    material_heat_gain: Array
    pore_gas_mass_flux: Array
    interface_energy_defect: Array
    interface_mass_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ConjugateAerothermalInterfacePlan(StrictModule, NonTrainableState):
    """Common-refinement extensive heat and pore-gas transfer."""

    fluid_to_mortar: Array
    material_to_mortar: Array
    mortar_measures: Array
    material_face_to_cell: Array
    material_cell_volumes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid_to_mortar: ArrayLike,
        material_to_mortar: ArrayLike,
        mortar_measures: ArrayLike,
        material_face_to_cell: ArrayLike,
        material_cell_volumes: ArrayLike,
        /,
    ):
        fluid = np.asarray(fluid_to_mortar, dtype=float)
        material = np.asarray(material_to_mortar, dtype=float)
        measures = np.asarray(mortar_measures, dtype=float)
        face_to_cell = np.asarray(material_face_to_cell, dtype=float)
        volumes = np.asarray(material_cell_volumes, dtype=float)
        mortar_count = measures.size
        if (
            fluid.ndim != 2
            or material.ndim != 2
            or fluid.shape[0] != mortar_count
            or material.shape[0] != mortar_count
            or face_to_cell.shape != (volumes.size, material.shape[1])
            or np.any(~np.isfinite(fluid))
            or np.any(~np.isfinite(material))
            or np.any(~np.isfinite(face_to_cell))
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0.0)
            or np.any(~np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or not np.allclose(np.sum(fluid, axis=1), 1.0, atol=1.0e-12)
            or not np.allclose(np.sum(material, axis=1), 1.0, atol=1.0e-12)
        ):
            raise ValueError("Conjugate mortar and cell projection arrays are invalid.")
        self.fluid_to_mortar = jnp.asarray(fluid)
        self.material_to_mortar = jnp.asarray(material)
        self.mortar_measures = jnp.asarray(measures)
        self.material_face_to_cell = jnp.asarray(face_to_cell)
        self.material_cell_volumes = jnp.asarray(volumes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conjugate-aerothermal-interface",
                "fluid_to_mortar": array_tree_fingerprint(self.fluid_to_mortar),
                "material_to_mortar": array_tree_fingerprint(self.material_to_mortar),
                "mortar_measures": array_tree_fingerprint(self.mortar_measures),
                "material_face_to_cell": array_tree_fingerprint(
                    self.material_face_to_cell
                ),
                "material_cell_volumes": array_tree_fingerprint(
                    self.material_cell_volumes
                ),
            }
        )

    def exchange(
        self,
        fluid_heat_flux_to_material: ArrayLike,
        material_face_heat_flux: ArrayLike,
        pore_gas_face_mass_flux: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> ConjugateAerothermalExchange:
        fluid_heat = jnp.asarray(fluid_heat_flux_to_material)
        material_heat = jnp.asarray(material_face_heat_flux, dtype=fluid_heat.dtype)
        pore_mass = jnp.asarray(pore_gas_face_mass_flux, dtype=fluid_heat.dtype)
        if (
            fluid_heat.shape != (self.fluid_to_mortar.shape[1],)
            or material_heat.shape != (self.material_to_mortar.shape[1],)
            or pore_mass.shape[0] != self.material_to_mortar.shape[1]
        ):
            raise ValueError("Conjugate interface fields do not match face topology.")
        fluid_mortar = contract(
            "mf,f->m", self.fluid_to_mortar, fluid_heat, backend="jax"
        )
        material_mortar = contract(
            "ms,s->m", self.material_to_mortar, material_heat, backend="jax"
        )
        fluid_loss = jnp.sum(self.mortar_measures * fluid_mortar)
        material_gain = jnp.sum(self.mortar_measures * material_mortar)
        energy_defect = material_gain - fluid_loss
        mass_flux = jnp.sum(pore_mass, axis=0)
        mass_defect = jnp.zeros_like(mass_flux)
        scale = jnp.maximum(jnp.maximum(jnp.abs(fluid_loss), jnp.abs(material_gain)), 1.0)
        finite = (
            jnp.isfinite(fluid_loss)
            & jnp.isfinite(material_gain)
            & jnp.all(jnp.isfinite(mass_flux))
        )
        successful = finite & (jnp.abs(energy_defect) <= tolerance * scale)
        return ConjugateAerothermalExchange(
            fluid_loss,
            material_gain,
            mass_flux,
            energy_defect,
            mass_defect,
            finite,
            successful,
            self.plan_id,
        )

    def apply_material_heat(
        self,
        state: AblatingMaterialState,
        face_heat_flux: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> AblatingMaterialState:
        heat = jnp.asarray(face_heat_flux, dtype=state.energy_density.dtype)
        step = jnp.asarray(step_size, dtype=state.energy_density.dtype)
        integrated_face = heat
        cell_power = contract(
            "cf,f->c", self.material_face_to_cell, integrated_face, backend="jax"
        )
        energy = state.energy_density + step * cell_power / self.material_cell_volumes
        return AblatingMaterialState(
            state.solid_component_densities,
            state.pore_gas_molar_densities,
            energy,
            state.porosity,
            state.finite & jnp.all(jnp.isfinite(energy)),
        )


class RecessionEvaluation(StrictModule):
    normal_speed: Array
    displacement: Array
    candidate_vertices: Array
    minimum_edge_length: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class FixedConnectivityRecessionPlan(StrictModule, NonTrainableState):
    minimum_edge_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, minimum_edge_length: float = 1.0e-10):
        minimum = float(minimum_edge_length)
        if not np.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("Minimum recession edge length must be positive.")
        self.minimum_edge_length = minimum
        self.plan_id = canonical_fingerprint(
            {"kind": "fixed-connectivity-recession", "minimum_edge_length": minimum}
        )

    def evaluate(
        self,
        vertices: ArrayLike,
        vertex_normals: ArrayLike,
        surface_mass_flux: ArrayLike,
        solid_density: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RecessionEvaluation:
        points = jnp.asarray(vertices)
        normals = jnp.asarray(vertex_normals, dtype=points.dtype)
        mass_flux = jnp.asarray(surface_mass_flux, dtype=points.dtype)
        density = jnp.asarray(solid_density, dtype=points.dtype)
        step = jnp.asarray(step_size, dtype=points.dtype)
        if (
            points.ndim != 2
            or normals.shape != points.shape
            or mass_flux.shape != points.shape[:-1]
            or density.shape != mass_flux.shape
        ):
            raise ValueError(
                "Recession vertices, normals, mass flux, and density must align."
            )
        speed = mass_flux / jnp.maximum(density, jnp.finfo(points.dtype).tiny)
        displacement = -step * speed[..., None] * normals
        candidate = points + displacement
        edge = jnp.roll(candidate, -1, axis=0) - candidate
        minimum_edge = jnp.min(jnp.sqrt(jnp.sum(edge * edge, axis=-1)))
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(minimum_edge)
        successful = (
            finite & jnp.all(density > 0.0) & (minimum_edge >= self.minimum_edge_length)
        )
        return RecessionEvaluation(
            speed, displacement, candidate, minimum_edge, finite, successful, self.plan_id
        )


class ConservativeRecessionRemapResult(StrictModule):
    remapped_gas: Array
    remapped_material: Array
    gas_conservation_defect: Array
    material_conservation_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ConservativeRecessionRemapPlan(StrictModule, NonTrainableState):
    gas_overlap: Array
    material_overlap: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, gas_overlap: ArrayLike, material_overlap: ArrayLike, /):
        gas = np.asarray(gas_overlap, dtype=float)
        material = np.asarray(material_overlap, dtype=float)
        if (
            gas.ndim != 2
            or material.ndim != 2
            or np.any(~np.isfinite(gas))
            or np.any(gas < 0.0)
            or np.any(~np.isfinite(material))
            or np.any(material < 0.0)
            or not np.allclose(np.sum(gas, axis=0), 1.0, atol=1.0e-12)
            or not np.allclose(np.sum(material, axis=0), 1.0, atol=1.0e-12)
        ):
            raise ValueError(
                "Recession remap overlaps must conservatively cover old cells."
            )
        self.gas_overlap = jnp.asarray(gas)
        self.material_overlap = jnp.asarray(material)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-recession-remap",
                "gas_overlap": array_tree_fingerprint(self.gas_overlap),
                "material_overlap": array_tree_fingerprint(self.material_overlap),
            }
        )

    def apply(
        self,
        gas_extensive: ArrayLike,
        material_extensive: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> ConservativeRecessionRemapResult:
        gas = jnp.asarray(gas_extensive)
        material = jnp.asarray(material_extensive, dtype=gas.dtype)
        if (
            gas.shape[0] != self.gas_overlap.shape[1]
            or material.shape[0] != self.material_overlap.shape[1]
        ):
            raise ValueError("Remap state cells do not match old topology.")
        remapped_gas = contract("no,o...->n...", self.gas_overlap, gas, backend="jax")
        remapped_material = contract(
            "no,o...->n...", self.material_overlap, material, backend="jax"
        )
        gas_defect = jnp.sum(remapped_gas, axis=0) - jnp.sum(gas, axis=0)
        material_defect = jnp.sum(remapped_material, axis=0) - jnp.sum(material, axis=0)
        finite = jnp.all(jnp.isfinite(remapped_gas)) & jnp.all(
            jnp.isfinite(remapped_material)
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(gas)), jnp.max(jnp.abs(material))), 1.0
        )
        successful = (
            finite
            & jnp.all(jnp.abs(gas_defect) <= tolerance * scale)
            & jnp.all(jnp.abs(material_defect) <= tolerance * scale)
        )
        return ConservativeRecessionRemapResult(
            remapped_gas,
            remapped_material,
            gas_defect,
            material_defect,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "ConjugateAerothermalExchange",
    "ConjugateAerothermalInterfacePlan",
    "ConservativeRecessionRemapPlan",
    "ConservativeRecessionRemapResult",
    "FixedConnectivityRecessionPlan",
    "RecessionEvaluation",
]
