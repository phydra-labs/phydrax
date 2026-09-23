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
from ._compressible_contracts import CompressibleKineticPopulationState
from ._compressible_execution import IntegerLatticeTransportPlan
from ._positive_kinetic import PositiveCompressibleKineticPlan


class KineticAuxiliaryState(StrictModule):
    species_densities: Array
    mode_energies: Array
    electron_energy: Array
    turbulence_variables: Array
    radiation_energy: Array


class KineticSpeciesTransportEvidence(StrictModule):
    mixture_mass_defect: Array
    minimum_species_density: Array
    element_defect: Array
    charge_defect: Array
    successful: Array


class KineticSpeciesTransportPlan(StrictModule, NonTrainableState):
    transport: IntegerLatticeTransportPlan
    element_matrix: Array
    species_charges: Array
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: IntegerLatticeTransportPlan,
        element_matrix: ArrayLike,
        species_charges: ArrayLike,
        /,
    ):
        if not isinstance(transport, IntegerLatticeTransportPlan):
            raise TypeError("transport must be an IntegerLatticeTransportPlan.")
        elements = np.asarray(element_matrix)
        charges = np.asarray(species_charges)
        if elements.ndim != 2 or elements.shape[1] < 1:
            raise ValueError("element_matrix must have shape (elements, species).")
        species = elements.shape[1]
        if charges.shape != (species,):
            raise ValueError("species_charges must match the species count.")
        if not np.all(np.isfinite(elements)) or not np.all(np.isfinite(charges)):
            raise ValueError("Species element and charge data must be finite.")
        self.transport = transport
        self.element_matrix = jnp.asarray(elements)
        self.species_charges = jnp.asarray(charges)
        self.species_count = species
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-species-transport",
                "transport": transport.transport_id,
                "element_matrix": elements.tolist(),
                "species_charges": charges.tolist(),
            }
        )

    def advect(
        self,
        state: CompressibleKineticPopulationState,
        species_densities: ArrayLike,
        /,
    ) -> tuple[Array, KineticSpeciesTransportEvidence]:
        species = jnp.asarray(species_densities)
        if species.shape != state.spatial_shape + (self.species_count,):
            raise ValueError("species_densities do not match the kinetic state.")
        if state.rule_id != self.transport.rule.rule_id:
            raise ValueError("Species transport and kinetic state rules differ.")
        particle = state.population("particle")
        density = jnp.sum(particle, axis=-1)
        safe_density = jnp.where(density > 0.0, density, 1.0)
        fractions = species / safe_density[..., None]
        velocities = np.asarray(self.transport.rule.velocities, dtype=np.int32)
        transported = jnp.zeros_like(species)
        for direction, velocity in enumerate(velocities):
            source_fraction = jnp.roll(
                fractions,
                shift=tuple(int(value) for value in velocity),
                axis=tuple(range(self.transport.rule.dimension)),
            )
            source_population = jnp.roll(
                particle[..., direction],
                shift=tuple(int(value) for value in velocity),
                axis=tuple(range(self.transport.rule.dimension)),
            )
            transported = transported + source_population[..., None] * source_fraction
        streamed_density = jnp.sum(transported, axis=-1)
        particle_streamed = sum(
            jnp.roll(
                particle[..., direction],
                shift=tuple(int(value) for value in velocity),
                axis=tuple(range(self.transport.rule.dimension)),
            )
            for direction, velocity in enumerate(velocities)
        )
        mass_defect = streamed_density - particle_streamed
        old_elements = species @ jnp.swapaxes(
            self.element_matrix.astype(species.dtype), -1, -2
        )
        new_elements = transported @ jnp.swapaxes(
            self.element_matrix.astype(species.dtype), -1, -2
        )
        old_charge = species @ self.species_charges.astype(species.dtype)
        new_charge = transported @ self.species_charges.astype(species.dtype)
        element_defect = jnp.sum(
            new_elements, axis=tuple(range(transported.ndim - 1))
        ) - jnp.sum(old_elements, axis=tuple(range(species.ndim - 1)))
        charge_defect = jnp.sum(new_charge) - jnp.sum(old_charge)
        minimum = jnp.min(transported, axis=-1)
        tolerance = (
            512.0 * jnp.finfo(species.dtype).eps * jnp.maximum(particle_streamed, 1.0)
        )
        element_scale = jnp.maximum(
            jnp.abs(jnp.sum(old_elements, axis=tuple(range(species.ndim - 1)))),
            1.0,
        )
        charge_scale = jnp.maximum(jnp.abs(jnp.sum(old_charge)), 1.0)
        conservation_tolerance = 512.0 * jnp.finfo(species.dtype).eps
        element_conserved = jnp.all(
            jnp.abs(element_defect) <= conservation_tolerance * element_scale
        )
        charge_conserved = jnp.abs(charge_defect) <= conservation_tolerance * charge_scale
        successful = (
            jnp.all(jnp.isfinite(transported), axis=-1)
            & (minimum >= 0.0)
            & (jnp.abs(mass_defect) <= tolerance)
            & element_conserved
            & charge_conserved
        )
        return transported, KineticSpeciesTransportEvidence(
            mixture_mass_defect=mass_defect,
            minimum_species_density=minimum,
            element_defect=element_defect,
            charge_defect=charge_defect,
            successful=successful,
        )


class KineticSourceEvidence(StrictModule):
    mass_change: Array
    momentum_change: Array
    energy_change: Array
    discarded_nonequilibrium_norm: Array
    successful: Array


class EquilibratingKineticSourceLiftPlan(StrictModule, NonTrainableState):
    model: PositiveCompressibleKineticPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, model: PositiveCompressibleKineticPlan, /):
        if not isinstance(model, PositiveCompressibleKineticPlan):
            raise TypeError("model must be a PositiveCompressibleKineticPlan.")
        self.model = model
        self.plan_id = canonical_fingerprint(
            {"kind": "equilibrating-kinetic-source-lift", "model": model.model_id}
        )

    def apply(
        self,
        state: CompressibleKineticPopulationState,
        mass_source: ArrayLike,
        momentum_source: ArrayLike,
        energy_source: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> tuple[CompressibleKineticPopulationState, KineticSourceEvidence]:
        old = self.model.moments(state)
        dt = jnp.asarray(time_step, dtype=old.density.dtype)
        mass_delta = dt * jnp.asarray(mass_source, dtype=old.density.dtype)
        momentum_delta = dt * jnp.asarray(momentum_source, dtype=old.momentum.dtype)
        energy_delta = dt * jnp.asarray(energy_source, dtype=old.total_energy.dtype)
        if (
            mass_delta.shape != old.density.shape
            or energy_delta.shape != old.density.shape
        ):
            raise ValueError("Scalar kinetic sources must have the spatial shape.")
        if momentum_delta.shape != old.momentum.shape:
            raise ValueError("momentum_source must match the momentum shape.")
        density = old.density + mass_delta
        momentum = old.momentum + momentum_delta
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        total_energy = old.total_energy + energy_delta
        temperature = (
            total_energy / safe_density - 0.5 * jnp.sum(velocity * velocity, axis=-1)
        ) / self.model.heat_capacity_cv
        equilibrium, dual, _ = self.model.equilibrium(density, velocity, temperature)
        candidate = CompressibleKineticPopulationState(
            equilibrium,
            dual,
            jnp.full(density.shape, 2.0, dtype=density.dtype),
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
            self.model.model_id,
            self.model.rule.rule_id,
        )
        candidate_macro = self.model.moments(candidate)
        successful = (
            candidate_macro.admissible
            & jnp.isfinite(dt)
            & (dt >= 0.0)
            & jnp.all(
                jnp.stack(
                    tuple(jnp.min(value, axis=-1) > 0.0 for value in equilibrium),
                    axis=0,
                ),
                axis=0,
            )
        )
        accepted = CompressibleKineticPopulationState(
            tuple(
                jnp.where(successful[..., None], new, old_value)
                for new, old_value in zip(equilibrium, state.populations, strict=True)
            ),
            jnp.where(successful[..., None], dual, state.equilibrium_dual),
            jnp.where(successful, 2.0, state.stabilizer),
            state.frame_velocity,
            state.frame_temperature_scale,
            self.model.layout,
            self.model.model_id,
            self.model.rule.rule_id,
        )
        old_equilibrium, _, _ = self.model.equilibrium(
            old.density,
            old.velocity,
            old.temperature,
            initial_dual=state.equilibrium_dual,
        )
        discarded = jnp.sqrt(
            sum(
                jnp.sum((value - equilibrium_value) ** 2, axis=-1)
                for value, equilibrium_value in zip(
                    state.populations, old_equilibrium, strict=True
                )
            )
        )
        return accepted, KineticSourceEvidence(
            mass_change=candidate_macro.density - old.density,
            momentum_change=candidate_macro.momentum - old.momentum,
            energy_change=candidate_macro.total_energy - old.total_energy,
            discarded_nonequilibrium_norm=discarded,
            successful=successful,
        )


class KineticEffectiveTransportEvidence(StrictModule):
    molecular_viscosity: Array
    turbulent_viscosity: Array
    numerical_viscosity: Array
    effective_viscosity: Array
    thermal_conductivity: Array
    prandtl_number: Array
    relaxation_rate: Array
    successful: Array


class KineticEffectiveTransportPlan(StrictModule, NonTrainableState):
    lattice_time_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, lattice_time_step: float = 1.0, /):
        step = float(lattice_time_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("lattice_time_step must be finite and positive.")
        self.lattice_time_step = step
        self.plan_id = canonical_fingerprint(
            {"kind": "kinetic-effective-transport", "lattice_time_step": step}
        )

    def evaluate(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        molecular_viscosity: ArrayLike,
        thermal_conductivity: ArrayLike,
        heat_capacity_cp: ArrayLike,
        /,
        *,
        turbulent_viscosity: ArrayLike = 0.0,
        numerical_viscosity: ArrayLike = 0.0,
    ) -> KineticEffectiveTransportEvidence:
        rho = jnp.asarray(density)
        thermal = jnp.asarray(temperature, dtype=rho.dtype)
        molecular = jnp.asarray(molecular_viscosity, dtype=rho.dtype)
        turbulent = jnp.asarray(turbulent_viscosity, dtype=rho.dtype)
        numerical = jnp.asarray(numerical_viscosity, dtype=rho.dtype)
        conductivity = jnp.asarray(thermal_conductivity, dtype=rho.dtype)
        cp = jnp.asarray(heat_capacity_cp, dtype=rho.dtype)
        effective = molecular + turbulent + numerical
        pressure = rho * thermal
        beta = 1.0 / (1.0 + 2.0 * effective / (pressure * self.lattice_time_step))
        prandtl = effective * cp / conductivity
        finite = (
            jnp.isfinite(beta)
            & jnp.isfinite(prandtl)
            & (rho > 0.0)
            & (thermal > 0.0)
            & (effective >= 0.0)
            & (conductivity > 0.0)
            & (beta > 0.0)
            & (beta < 1.0)
        )
        return KineticEffectiveTransportEvidence(
            molecular_viscosity=molecular,
            turbulent_viscosity=turbulent,
            numerical_viscosity=numerical,
            effective_viscosity=effective,
            thermal_conductivity=conductivity,
            prandtl_number=prandtl,
            relaxation_rate=2.0 * beta,
            successful=finite,
        )


class KineticRadiationAblationEvidence(StrictModule):
    radiation_energy_exchange: Array
    ablated_mass: Array
    wall_momentum: Array
    wall_energy: Array
    charge_exchange: Array
    successful: Array


class KineticRadiationAblationPlan(StrictModule, NonTrainableState):
    plan_id: str = "kinetic-radiation-ablation-exchange"

    def exchange(
        self,
        auxiliary: KineticAuxiliaryState,
        radiation_energy_exchange: ArrayLike,
        ablated_species: ArrayLike,
        wall_momentum: ArrayLike,
        wall_energy: ArrayLike,
        /,
        *,
        species_charges: ArrayLike,
    ) -> tuple[KineticAuxiliaryState, KineticRadiationAblationEvidence]:
        radiation = jnp.asarray(radiation_energy_exchange)
        ablated = jnp.asarray(ablated_species)
        momentum = jnp.asarray(wall_momentum)
        energy = jnp.asarray(wall_energy)
        charges = jnp.asarray(species_charges, dtype=ablated.dtype)
        if ablated.shape != auxiliary.species_densities.shape:
            raise ValueError("ablated_species must match species densities.")
        if charges.shape != (ablated.shape[-1],):
            raise ValueError("species_charges must match the species count.")
        if radiation.shape != auxiliary.radiation_energy.shape:
            raise ValueError(
                "radiation_energy_exchange must match radiation group energy."
            )
        matter_energy = jnp.sum(radiation, axis=-1)
        updated = KineticAuxiliaryState(
            species_densities=auxiliary.species_densities + ablated,
            mode_energies=auxiliary.mode_energies,
            electron_energy=auxiliary.electron_energy + matter_energy,
            turbulence_variables=auxiliary.turbulence_variables,
            radiation_energy=auxiliary.radiation_energy - radiation,
        )
        charge = ablated @ charges
        successful = (
            jnp.all(jnp.isfinite(updated.species_densities), axis=-1)
            & (jnp.min(updated.species_densities, axis=-1) >= 0.0)
            & jnp.isfinite(updated.electron_energy)
            & jnp.all(jnp.isfinite(momentum), axis=-1)
            & jnp.isfinite(energy)
        )
        return updated, KineticRadiationAblationEvidence(
            radiation_energy_exchange=matter_energy,
            ablated_mass=jnp.sum(ablated, axis=-1),
            wall_momentum=momentum,
            wall_energy=energy,
            charge_exchange=charge,
            successful=successful,
        )


class KineticSpectrum(StrictModule):
    shell_energy: Array
    total_energy: Array
    enstrophy: Array
    finite: Array


class KineticSpectralAnalysisPlan(StrictModule, NonTrainableState):
    shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], /):
        grid = tuple(int(value) for value in shape)
        if len(grid) not in (2, 3) or any(value < 2 for value in grid):
            raise ValueError("Spectral kinetic grid must be two- or three-dimensional.")
        self.shape = grid
        self.plan_id = canonical_fingerprint(
            {"kind": "kinetic-spectral-analysis", "shape": list(grid)}
        )

    def evaluate(self, velocity: ArrayLike, /) -> KineticSpectrum:
        field = jnp.asarray(velocity)
        if field.shape != self.shape + (len(self.shape),):
            raise ValueError("velocity shape does not match the spectral plan.")
        coefficients = jnp.fft.fftn(
            field, axes=tuple(range(len(self.shape))), norm="forward"
        )
        modal_energy = 0.5 * jnp.sum(jnp.abs(coefficients) ** 2, axis=-1)
        modes = jnp.meshgrid(
            *(jnp.fft.fftfreq(size) * size for size in self.shape), indexing="ij"
        )
        radius = jnp.rint(jnp.sqrt(sum(mode * mode for mode in modes))).astype(jnp.int32)
        maximum_shell = min(self.shape) // 2
        shell = jnp.bincount(
            jnp.minimum(radius.reshape(-1), maximum_shell),
            weights=modal_energy.reshape(-1),
            length=maximum_shell + 1,
        )
        if len(self.shape) == 3:
            kx, ky, kz = modes
            curl_x = 1j * (ky * coefficients[..., 2] - kz * coefficients[..., 1])
            curl_y = 1j * (kz * coefficients[..., 0] - kx * coefficients[..., 2])
            curl_z = 1j * (kx * coefficients[..., 1] - ky * coefficients[..., 0])
            enstrophy = 0.5 * jnp.sum(
                jnp.abs(curl_x) ** 2 + jnp.abs(curl_y) ** 2 + jnp.abs(curl_z) ** 2
            )
        else:
            kx, ky = modes
            curl = 1j * (kx * coefficients[..., 1] - ky * coefficients[..., 0])
            enstrophy = 0.5 * jnp.sum(jnp.abs(curl) ** 2)
        finite = jnp.all(jnp.isfinite(shell)) & jnp.isfinite(enstrophy)
        return KineticSpectrum(
            shell_energy=shell,
            total_energy=jnp.sum(modal_energy),
            enstrophy=enstrophy,
            finite=finite,
        )


__all__ = [
    "EquilibratingKineticSourceLiftPlan",
    "KineticAuxiliaryState",
    "KineticEffectiveTransportEvidence",
    "KineticEffectiveTransportPlan",
    "KineticRadiationAblationEvidence",
    "KineticRadiationAblationPlan",
    "KineticSourceEvidence",
    "KineticSpeciesTransportEvidence",
    "KineticSpeciesTransportPlan",
    "KineticSpectralAnalysisPlan",
    "KineticSpectrum",
]
