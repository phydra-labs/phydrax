#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume import UnstructuredFiniteVolumeDiscretization


class MultiphaseComponentState(StrictModule):
    saturation: Array
    phase_density_kg_m3: Array
    phase_composition: Array
    phase_internal_energy_J_kg: Array
    rock_internal_energy_J_m3: Array
    porosity: Array
    component_content_kg: Array
    energy_content_J: Array
    plan_id: str = eqx.field(static=True)


class MultiphaseFaceFluxes(StrictModule):
    phase_volume_rate_m3_s: Array
    upstream_density_kg_m3: Array
    upstream_composition: Array
    upstream_enthalpy_J_kg: Array
    component_mass_rate_kg_s: Array
    energy_rate_W: Array
    plan_id: str = eqx.field(static=True)


class MultiphaseConservationResidual(StrictModule):
    component_kg_s: Array
    energy_W: Array
    global_component_balance_kg_s: Array
    global_energy_balance_W: Array
    finite: Array


class MultiphaseConservationPlan(StrictModule, NonTrainableState):
    """Component and total-energy conservation over accepted phase face rates."""

    discretization: UnstructuredFiniteVolumeDiscretization
    phase_names: tuple[str, ...] = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    boundary_faces: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        phase_names: tuple[str, ...],
        component_names: tuple[str, ...],
        /,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "Multiphase conservation requires unstructured finite volume geometry."
            )
        phases = tuple(str(value).strip() for value in phase_names)
        components = tuple(str(value).strip() for value in component_names)
        if (
            not phases
            or not components
            or any(not value for value in (*phases, *components))
            or len(set(phases)) != len(phases)
            or len(set(components)) != len(components)
        ):
            raise ValueError(
                "Multiphase phase/component names must be nonempty and unique."
            )
        self.discretization = discretization
        self.phase_names, self.component_names = phases, components
        self.boundary_faces = jnp.asarray(
            np.flatnonzero(np.asarray(discretization.neighbour_cells) < 0),
            dtype=jnp.int32,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiphase-component-energy-conservation",
                "geometry": discretization.geometry_id,
                "phases": phases,
                "components": components,
            }
        )

    @property
    def cell_count(self) -> int:
        return self.discretization.cell_volumes.size

    @property
    def phase_count(self) -> int:
        return len(self.phase_names)

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    def state(
        self,
        saturation: ArrayLike,
        phase_density_kg_m3: ArrayLike,
        phase_composition: ArrayLike,
        phase_internal_energy_J_kg: ArrayLike,
        rock_internal_energy_J_m3: ArrayLike,
        porosity: ArrayLike,
        /,
    ) -> MultiphaseComponentState:
        cells, phases, components = (
            self.cell_count,
            self.phase_count,
            self.component_count,
        )
        saturation_ = jnp.asarray(saturation)
        density = jnp.asarray(phase_density_kg_m3)
        composition = jnp.asarray(phase_composition)
        internal = jnp.asarray(phase_internal_energy_J_kg)
        rock = jnp.broadcast_to(jnp.asarray(rock_internal_energy_J_m3), (cells,))
        porosity_ = jnp.broadcast_to(jnp.asarray(porosity), (cells,))
        if (
            saturation_.shape != (cells, phases)
            or density.shape != saturation_.shape
            or composition.shape != (cells, phases, components)
            or internal.shape != saturation_.shape
        ):
            raise ValueError(
                "Multiphase state arrays have incompatible cell/phase/component shapes."
            )
        invalid = (
            jnp.any(~jnp.isfinite(saturation_))
            | jnp.any(saturation_ < 0)
            | jnp.any(jnp.abs(jnp.sum(saturation_, axis=1) - 1.0) > 1e-10)
            | jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0)
            | jnp.any(~jnp.isfinite(composition))
            | jnp.any(composition < 0)
            | jnp.any(jnp.abs(jnp.sum(composition, axis=2) - 1.0) > 1e-10)
            | jnp.any(~jnp.isfinite(internal))
            | jnp.any(~jnp.isfinite(rock))
            | jnp.any(~jnp.isfinite(porosity_))
            | jnp.any((porosity_ <= 0) | (porosity_ >= 1))
        )
        saturation_ = eqx.error_if(
            saturation_,
            invalid,
            "Multiphase state violates saturation/composition/physical constraints.",
        )
        pore_volume = self.discretization.cell_volumes * porosity_
        phase_mass = pore_volume[:, None] * saturation_ * density
        component = jnp.sum(phase_mass[:, :, None] * composition, axis=1)
        energy = self.discretization.cell_volumes * rock + jnp.sum(
            phase_mass * internal, axis=1
        )
        return MultiphaseComponentState(
            saturation_,
            density,
            composition,
            internal,
            rock,
            porosity_,
            component,
            energy,
            self.plan_id,
        )

    def fluxes(
        self,
        phase_volume_rate_m3_s: ArrayLike,
        upstream_density_kg_m3: ArrayLike,
        upstream_composition: ArrayLike,
        upstream_enthalpy_J_kg: ArrayLike,
        /,
    ) -> MultiphaseFaceFluxes:
        faces = self.discretization.face_measures.size
        rate = jnp.asarray(phase_volume_rate_m3_s)
        density = jnp.asarray(upstream_density_kg_m3)
        composition = jnp.asarray(upstream_composition)
        enthalpy = jnp.asarray(upstream_enthalpy_J_kg)
        if (
            rate.shape != (faces, self.phase_count)
            or density.shape != rate.shape
            or composition.shape != (faces, self.phase_count, self.component_count)
            or enthalpy.shape != rate.shape
        ):
            raise ValueError("Multiphase face flux arrays have incompatible shapes.")
        rate = eqx.error_if(
            rate,
            jnp.any(~jnp.isfinite(rate))
            | jnp.any(~jnp.isfinite(density))
            | jnp.any(density <= 0)
            | jnp.any(~jnp.isfinite(composition))
            | jnp.any(composition < 0)
            | jnp.any(jnp.abs(jnp.sum(composition, axis=2) - 1.0) > 1e-10)
            | jnp.any(~jnp.isfinite(enthalpy)),
            "Multiphase upstream face properties and rates must be physical.",
        )
        phase_mass = rate * density
        component_rate = jnp.sum(phase_mass[:, :, None] * composition, axis=1)
        energy_rate = jnp.sum(phase_mass * enthalpy, axis=1)
        return MultiphaseFaceFluxes(
            rate,
            density,
            composition,
            enthalpy,
            component_rate,
            energy_rate,
            self.plan_id,
        )

    def _divergence(self, face_rates: Array) -> Array:
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        result = jnp.zeros(
            (self.cell_count,) + face_rates.shape[1:], dtype=face_rates.dtype
        )
        result = result.at[owner].add(face_rates)
        interior = neighbour >= 0
        safe = jnp.where(interior, neighbour, 0)
        return result.at[safe].add(
            jnp.where(
                interior.reshape((-1,) + (1,) * (face_rates.ndim - 1)), -face_rates, 0
            )
        )

    def residual(
        self,
        candidate: MultiphaseComponentState,
        previous: MultiphaseComponentState,
        dt_s: ArrayLike,
        fluxes: MultiphaseFaceFluxes,
        /,
        *,
        component_source_kg_s: ArrayLike = 0.0,
        energy_source_W: ArrayLike = 0.0,
    ) -> MultiphaseConservationResidual:
        if (
            not isinstance(candidate, MultiphaseComponentState)
            or not isinstance(previous, MultiphaseComponentState)
            or not isinstance(fluxes, MultiphaseFaceFluxes)
        ):
            raise TypeError("Multiphase residual requires prepared states and fluxes.")
        if (
            candidate.plan_id != self.plan_id
            or previous.plan_id != self.plan_id
            or fluxes.plan_id != self.plan_id
        ):
            raise ValueError("Multiphase states and fluxes belong to a different plan.")
        dt = jnp.asarray(dt_s)
        component_source = jnp.broadcast_to(
            jnp.asarray(component_source_kg_s), (self.cell_count, self.component_count)
        )
        energy_source = jnp.broadcast_to(jnp.asarray(energy_source_W), (self.cell_count,))
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt)
            | (dt <= 0)
            | jnp.any(~jnp.isfinite(component_source))
            | jnp.any(~jnp.isfinite(energy_source)),
            "Multiphase timestep and sources must be finite and timestep positive.",
        )
        component = (
            (candidate.component_content_kg - previous.component_content_kg) / dt
            + self._divergence(fluxes.component_mass_rate_kg_s)
            - component_source
        )
        energy = (
            (candidate.energy_content_J - previous.energy_content_J) / dt
            + self._divergence(fluxes.energy_rate_W)
            - energy_source
        )
        global_component = jnp.sum(component, axis=0)
        global_energy = jnp.sum(energy)
        finite = jnp.all(jnp.isfinite(component)) & jnp.all(jnp.isfinite(energy))
        return MultiphaseConservationResidual(
            component, energy, global_component, global_energy, finite
        )


__all__ = [
    "MultiphaseComponentState",
    "MultiphaseConservationPlan",
    "MultiphaseConservationResidual",
    "MultiphaseFaceFluxes",
]
