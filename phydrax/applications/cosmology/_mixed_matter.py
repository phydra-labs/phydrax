#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional fixed-grid wave, particle, and gas cosmology coupling."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...discretization.splatting import ParticleGridSplatState
from ...solver import ParticleMeshGravityPlan
from ._coupled import ComovingEulerPlan, ComovingEulerState
from ._particles import CosmologicalKDKPlan, CosmologicalParticleState
from ._wave_dark_matter import PreparedPeriodicWaveDarkMatter, WaveDarkMatterState


_COMPONENT_NAMES = ("wave", "particles", "gas")


class WaveParticleCosmologyState(StrictModule):
    """Synchronized wave and fixed-capacity collisionless-particle state."""

    wave: WaveDarkMatterState
    particles: CosmologicalParticleState


class WaveParticleGasCosmologyState(StrictModule):
    """Synchronized wave, fixed-capacity particles, and comoving Euler state."""

    wave: WaveDarkMatterState
    particles: CosmologicalParticleState
    gas: ComovingEulerState


class MixedDensityAssembly(StrictModule):
    """All named comoving density contributions before one mean subtraction."""

    wave_density: Array
    particle_density: Array
    gas_density: Array
    total_density: Array
    component_mass: Array
    total_mass: Array
    particle_source_mass: Array
    particle_deposited_mass: Array
    particle_mass_balance_defect: Array
    particle_routes: ParticleGridSplatState
    scale_factor: Array
    density_nonnegative: Array
    finite: Array
    successful: Array
    component_names: tuple[str, str, str] = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    density_convention: str = eqx.field(static=True)
    assembler_id: str = eqx.field(static=True)


class MixedDensityAssembler(StrictModule):
    """Bind wave density and the existing matched PM deposit to one cell layout."""

    wave: PreparedPeriodicWaveDarkMatter
    particle_gravity: ParticleMeshGravityPlan
    cell_volumes: Array = fixed_field()
    include_gas: bool = eqx.field(static=True)
    assembler_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave: PreparedPeriodicWaveDarkMatter,
        particle_gravity: ParticleMeshGravityPlan,
        /,
        *,
        include_gas: bool,
    ):
        if not isinstance(wave, PreparedPeriodicWaveDarkMatter):
            raise TypeError("wave must be PreparedPeriodicWaveDarkMatter.")
        if not isinstance(particle_gravity, ParticleMeshGravityPlan):
            raise TypeError("particle_gravity must be ParticleMeshGravityPlan.")
        gravity_owner = particle_gravity.gravity.plan
        if (
            gravity_owner.gravity_argument is not None
            or wave.gravitational_constant != gravity_owner.gravitational_constant
        ):
            raise ValueError(
                "Mixed density assembly requires exactly one static gravitational "
                "constant shared by wave and PM owners."
            )
        wave_grid = wave.discretization.grid
        gravity_grid = particle_gravity.gravity.dynamics.discretization.grid
        if wave.discretization.physical_shape != particle_gravity.gravity.cell_shape:
            raise ValueError("Wave and particle gravity cell shapes must match exactly.")
        if wave_grid.axis_names != gravity_grid.axis_names:
            raise ValueError("Wave and particle gravity axes must have identical names.")
        if not np.allclose(
            np.asarray(wave_grid.points),
            np.asarray(gravity_grid.points),
            rtol=2.0e-13,
            atol=2.0e-13,
        ):
            raise ValueError("Wave and particle gravity coordinates must be identical.")
        cell_volumes = np.asarray(
            particle_gravity.gravity.dynamics.discretization.cell_volumes
        )
        if not np.allclose(
            np.asarray(wave.discretization.quadrature_weights),
            cell_volumes,
            rtol=2.0e-13,
            atol=2.0e-13,
        ):
            raise ValueError("Wave quadrature and gravity cell measures must agree.")
        include = bool(include_gas)
        self.wave = wave
        self.particle_gravity = particle_gravity
        self.cell_volumes = jnp.asarray(cell_volumes)
        self.include_gas = include
        self.assembler_id = canonical_fingerprint(
            {
                "kind": "mixed-comoving-density-assembler",
                "wave": wave.prepared_id,
                "particle_transfer": particle_gravity.transfer.prepared_id,
                "gravity": particle_gravity.gravity.process_id,
                "gravitational_constant": wave.gravitational_constant,
                "include_gas": include,
                "component_order": list(_COMPONENT_NAMES),
            }
        )

    def assemble(
        self,
        state: WaveParticleCosmologyState | WaveParticleGasCosmologyState,
        /,
    ) -> MixedDensityAssembly:
        if self.include_gas:
            if not isinstance(state, WaveParticleGasCosmologyState):
                raise TypeError("This assembler requires WaveParticleGasCosmologyState.")
            gas_density = jnp.asarray(
                state.gas.cell_average[..., 0], dtype=state.wave.psi.real.dtype
            )
            gas_scale = state.gas.scale_factor
        else:
            if not isinstance(state, WaveParticleCosmologyState):
                raise TypeError("This assembler requires WaveParticleCosmologyState.")
            gas_density = jnp.zeros(
                self.particle_gravity.gravity.cell_shape,
                dtype=state.wave.psi.real.dtype,
            )
            gas_scale = state.wave.scale_factor
        wave_density = self.wave.density(state.wave).astype(gas_density.dtype)
        deposited, routes = self.particle_gravity.density(state.particles.positions)
        particle_density = deposited.density.astype(gas_density.dtype)
        if gas_density.shape != wave_density.shape:
            raise ValueError("Gas density must use the authoritative mixed cell layout.")
        scale = jnp.asarray(state.wave.scale_factor, dtype=gas_density.dtype)
        tolerance = 32.0 * jnp.finfo(scale.dtype).eps * jnp.maximum(scale, 1.0)
        time_aligned = (jnp.abs(state.particles.scale_factor - scale) <= tolerance) & (
            jnp.abs(gas_scale - scale) <= tolerance
        )
        total_density = wave_density + particle_density + gas_density
        volumes = self.cell_volumes.astype(total_density.dtype)
        wave_mass = contract("...,...->", wave_density, volumes)
        gas_mass = contract("...,...->", gas_density, volumes)
        particle_source_mass = jnp.asarray(
            deposited.balance.active_source_total, dtype=total_density.dtype
        )
        particle_deposited_mass = contract("...,...->", particle_density, volumes)
        component_mass = jnp.stack((wave_mass, particle_source_mass, gas_mass))
        total_mass = jnp.sum(component_mass)
        nonnegative = (
            jnp.all(wave_density >= 0.0)
            & jnp.all(particle_density >= 0.0)
            & jnp.all(gas_density >= 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(wave_density))
            & jnp.all(jnp.isfinite(particle_density))
            & jnp.all(jnp.isfinite(gas_density))
            & jnp.all(jnp.isfinite(component_mass))
            & jnp.isfinite(total_mass)
        )
        successful = deposited.successful & time_aligned & nonnegative & finite
        return MixedDensityAssembly(
            wave_density=wave_density,
            particle_density=particle_density,
            gas_density=gas_density,
            total_density=total_density,
            component_mass=component_mass,
            total_mass=total_mass,
            particle_source_mass=particle_source_mass,
            particle_deposited_mass=particle_deposited_mass,
            particle_mass_balance_defect=deposited.balance.maximum_absolute_balance_defect,
            particle_routes=routes,
            scale_factor=scale,
            density_nonnegative=nonnegative,
            finite=finite,
            successful=successful,
            component_names=_COMPONENT_NAMES,
            scale_id=self.wave.scale_id,
            coordinate_convention=self.wave.coordinate_convention,
            density_convention="comoving-mass-per-comoving-volume",
            assembler_id=self.assembler_id,
        )


class SharedPeriodicGravityResult(StrictModule):
    """One mean-zero periodic solve and matched forces for every component."""

    assembly: MixedDensityAssembly
    potential: Array
    cell_acceleration: Array
    particle_acceleration: Array
    mean_density: Array
    source_integral: Array
    poisson_relative_residual: Array
    gauge_defect: Array
    component_force: Array
    total_force: Array
    particle_force_adjoint_defect: Array
    particle_support_complete: Array
    finite: Array
    successful: Array
    mean_removal_count: int = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    potential_convention: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class SharedPeriodicGravityPlan(StrictModule):
    """Apply one existing periodic PM Poisson owner to one mixed density assembly."""

    assembler: MixedDensityAssembler
    particle_gravity: ParticleMeshGravityPlan
    cell_volumes: Array = fixed_field()
    gravitational_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        assembler: MixedDensityAssembler,
        particle_gravity: ParticleMeshGravityPlan,
        /,
    ):
        if not isinstance(assembler, MixedDensityAssembler):
            raise TypeError("assembler must be MixedDensityAssembler.")
        if not isinstance(particle_gravity, ParticleMeshGravityPlan):
            raise TypeError("particle_gravity must be ParticleMeshGravityPlan.")
        if assembler.particle_gravity.plan_id != particle_gravity.plan_id:
            raise ValueError("Density assembly and gravity must share one PM owner.")
        if not particle_gravity.gravity.poisson.diagonalization.nullspace_dimension:
            raise ValueError("Shared mixed gravity requires a mean-zero periodic solve.")
        coupling = float(particle_gravity.gravity.plan.gravitational_constant)
        self.assembler = assembler
        self.particle_gravity = particle_gravity
        self.cell_volumes = assembler.cell_volumes
        self.gravitational_constant = coupling
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shared-periodic-mixed-gravity",
                "assembler": assembler.assembler_id,
                "gravity": particle_gravity.gravity.process_id,
                "gravitational_constant": coupling,
                "mean_removal_count": 1,
                "potential_convention": assembler.wave.potential_convention,
            }
        )

    def solve(
        self,
        assembly: MixedDensityAssembly,
        args: Any = None,
        /,
    ) -> SharedPeriodicGravityResult:
        if not isinstance(assembly, MixedDensityAssembly):
            raise TypeError("assembly must be MixedDensityAssembly.")
        if assembly.assembler_id != self.assembler.assembler_id:
            raise ValueError("Density assembly belongs to a different mixed plan.")
        potential, _, cell_acceleration, solved = (
            self.particle_gravity.gravity.solve_density(assembly.total_density, args)
        )
        gathered = self.particle_gravity.transfer.gather(
            assembly.particle_routes, cell_acceleration
        )
        particles = self.particle_gravity.transfer.particles
        active = particles.active_mask
        particle_acceleration = jnp.where(active[:, None], gathered.values, 0.0)
        volumes = self.cell_volumes.astype(assembly.total_density.dtype)
        volume = jnp.sum(volumes)
        mean_density = contract("...,...->", assembly.total_density, volumes) / volume
        density_source = assembly.total_density - mean_density
        coupling = jnp.asarray(
            self.gravitational_constant,
            dtype=assembly.total_density.dtype,
        )
        poisson_source = 4.0 * jnp.pi * coupling * density_source
        laplacian = self.particle_gravity.gravity.diffusion.mv(potential)
        residual = laplacian - poisson_source
        residual_norm = jnp.sqrt(
            jnp.maximum(contract("...,...,...->", residual, residual, volumes), 0.0)
        )
        source_norm = jnp.sqrt(
            jnp.maximum(
                contract("...,...,...->", poisson_source, poisson_source, volumes),
                0.0,
            )
        )
        relative_residual = jnp.where(
            source_norm > 0.0,
            residual_norm / jnp.where(source_norm > 0.0, source_norm, 1.0),
            residual_norm,
        )
        source_integral = contract("...,...->", density_source, volumes)
        gauge_defect = jnp.abs(contract("...,...->", potential, volumes) / volume)
        wave_force = contract(
            "...,...,...d->d",
            assembly.wave_density,
            volumes,
            cell_acceleration,
        )
        particle_grid_force = contract(
            "...,...,...d->d",
            assembly.particle_density,
            volumes,
            cell_acceleration,
        )
        physical_masses = particles.masses.astype(particle_acceleration.dtype)
        particle_force = jnp.sum(
            physical_masses[:, None] * particle_acceleration,
            axis=0,
        )
        gas_force = contract(
            "...,...,...d->d",
            assembly.gas_density,
            volumes,
            cell_acceleration,
        )
        component_force = jnp.stack((wave_force, particle_force, gas_force))
        total_force = jnp.sum(component_force, axis=0)
        adjoint_defect = jnp.sqrt(jnp.sum((particle_force - particle_grid_force) ** 2))
        support_complete = jnp.all(gathered.support | ~active)
        finite = (
            assembly.finite
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(cell_acceleration))
            & jnp.all(jnp.isfinite(particle_acceleration))
            & jnp.isfinite(mean_density)
            & jnp.isfinite(source_integral)
            & jnp.isfinite(relative_residual)
            & jnp.isfinite(gauge_defect)
            & jnp.all(jnp.isfinite(component_force))
            & jnp.isfinite(adjoint_defect)
        )
        successful = assembly.successful & solved.converged & support_complete & finite
        return SharedPeriodicGravityResult(
            assembly=assembly,
            potential=potential,
            cell_acceleration=cell_acceleration,
            particle_acceleration=particle_acceleration,
            mean_density=mean_density,
            source_integral=source_integral,
            poisson_relative_residual=relative_residual,
            gauge_defect=gauge_defect,
            component_force=component_force,
            total_force=total_force,
            particle_force_adjoint_defect=adjoint_defect,
            particle_support_complete=support_complete,
            finite=finite,
            successful=successful,
            mean_removal_count=1,
            potential_convention=self.assembler.wave.potential_convention,
            gravitational_constant=self.gravitational_constant,
            plan_id=self.plan_id,
        )


class MixedCosmologyDiagnostics(StrictModule):
    """Fixed-shape interval evidence for transactional mixed evolution."""

    attempted: Array
    accepted: Array
    rolled_back: Array
    step_status: Array
    start_scale_factor: Array
    end_scale_factor: Array
    wave_scale_factor: Array
    particle_scale_factor: Array
    gas_scale_factor: Array
    time_level_consistent: Array
    component_mass_start: Array
    component_mass_end: Array
    total_mass_start: Array
    total_mass_end: Array
    mass_balance_defect: Array
    particle_deposition_defect: Array
    mean_density: Array
    source_integral: Array
    poisson_relative_residual: Array
    gauge_defect: Array
    component_force: Array
    total_force: Array
    particle_force_adjoint_defect: Array
    component_gravity_work: Array
    total_gravity_work: Array
    gas_homogeneous_energy_change: Array
    wave_norm_start: Array
    wave_norm_end: Array
    wave_norm_relative_error: Array
    maximum_wave_kinetic_phase: Array
    maximum_wave_potential_phase: Array
    wave_phase_resolved: Array
    wave_norm_conserved: Array
    density_nonnegative: Array
    gas_density_positive: Array
    gas_pressure_positive: Array
    gas_homogeneous_successful: Array
    component_successful: Array
    gravity_successful: Array
    accepted_steps: Array
    completed: Array
    first_failed_step: Array
    component_names: tuple[str, str, str] = eqx.field(static=True)


class WaveParticleCosmologyResult(StrictModule):
    state: WaveParticleCosmologyState
    gravity: SharedPeriodicGravityResult
    diagnostics: MixedCosmologyDiagnostics
    successful: Array
    prepared_id: str = eqx.field(static=True)
    wave_prepared_id: str = eqx.field(static=True)
    particle_plan_id: str = eqx.field(static=True)
    particle_support_id: str = eqx.field(static=True)
    gravity_plan_id: str = eqx.field(static=True)
    assembler_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)


class WaveParticleGasCosmologyResult(StrictModule):
    state: WaveParticleGasCosmologyState
    gravity: SharedPeriodicGravityResult
    diagnostics: MixedCosmologyDiagnostics
    successful: Array
    prepared_id: str = eqx.field(static=True)
    wave_prepared_id: str = eqx.field(static=True)
    particle_plan_id: str = eqx.field(static=True)
    gas_plan_id: str = eqx.field(static=True)
    particle_support_id: str = eqx.field(static=True)
    gravity_plan_id: str = eqx.field(static=True)
    assembler_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)


def _validate_owner_bindings(
    wave: PreparedPeriodicWaveDarkMatter,
    particles: CosmologicalKDKPlan,
    gravity: ParticleMeshGravityPlan,
    gas: ComovingEulerPlan | None,
    /,
) -> None:
    if not isinstance(wave, PreparedPeriodicWaveDarkMatter):
        raise TypeError("wave must be PreparedPeriodicWaveDarkMatter.")
    if not isinstance(particles, CosmologicalKDKPlan):
        raise TypeError("particles must be CosmologicalKDKPlan.")
    if not isinstance(gravity, ParticleMeshGravityPlan):
        raise TypeError("gravity must be ParticleMeshGravityPlan.")
    if particles.particles.prepared_id != gravity.transfer.particles.prepared_id:
        raise ValueError("Mixed KDK and PM transfer must share one particle support.")
    if particles.scale.scale_id != wave.scale_id:
        raise ValueError("Wave and particle cosmology scale contracts must agree.")
    gravity_owner = gravity.gravity.plan
    if gravity_owner.gravity_argument is not None:
        raise ValueError(
            "Mixed shared gravity requires one static gravitational constant owner."
        )
    if wave.gravitational_constant != gravity_owner.gravitational_constant:
        raise ValueError(
            "Wave and PM gravity owners must use exactly one gravitational constant."
        )
    grid = gravity.transfer.plan.target
    if len(particles.box_size) != len(grid.axes):
        raise ValueError("Mixed particle and gravity dimensions disagree.")
    for axis, length in zip(grid.axes, particles.box_size, strict=True):
        if not axis.periodic or axis.bounds is None:
            raise ValueError("Mixed fixed-grid cosmology requires periodic PM axes.")
        bounds = np.asarray(axis.bounds, dtype=np.float64)
        if not np.allclose(bounds, (0.0, length), rtol=0.0, atol=2.0e-13):
            raise ValueError("Mixed PM bounds must be [0, particle box size].")
    if not np.array_equal(
        np.asarray(wave.scale_factors), np.asarray(wave.plan.scale_factors)
    ):
        raise ValueError("Prepared wave schedule identity changed.")
    if gas is not None:
        if not isinstance(gas, ComovingEulerPlan):
            raise TypeError("gas must be ComovingEulerPlan.")
        if (
            gas.dynamics.discretization.grid.prepared_id
            != gravity.transfer.plan.target.prepared_id
        ):
            raise ValueError("Gas and PM transfer must share one prepared grid.")


class WaveParticleCosmologyPlan(StrictModule):
    """Prepare symmetric fixed-grid wave-particle evolution on the wave schedule."""

    wave: PreparedPeriodicWaveDarkMatter
    particles: CosmologicalKDKPlan
    gravity: ParticleMeshGravityPlan
    plan_id: str = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)

    def __init__(
        self,
        wave: PreparedPeriodicWaveDarkMatter,
        particles: CosmologicalKDKPlan,
        gravity: ParticleMeshGravityPlan,
        /,
    ):
        _validate_owner_bindings(wave, particles, gravity, None)
        self.wave = wave
        self.particles = particles
        self.gravity = gravity
        self.gravitational_constant = wave.gravitational_constant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-particle-cosmology-plan",
                "wave": wave.prepared_id,
                "particles": particles.plan_id,
                "gravity": gravity.plan_id,
                "gravitational_constant": wave.gravitational_constant,
                "schedule": np.asarray(wave.scale_factors).tolist(),
                "integrator": "shared-endpoint-kick-drift-kick",
            }
        )

    def prepare(self, /) -> "PreparedWaveParticleCosmology":
        return PreparedWaveParticleCosmology(self)


class WaveParticleGasCosmologyPlan(StrictModule):
    """Prepare symmetric fixed-grid wave-particle-gas predictor/corrector evolution."""

    wave: PreparedPeriodicWaveDarkMatter
    particles: CosmologicalKDKPlan
    gas: ComovingEulerPlan
    gravity: ParticleMeshGravityPlan
    plan_id: str = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)

    def __init__(
        self,
        wave: PreparedPeriodicWaveDarkMatter,
        particles: CosmologicalKDKPlan,
        gas: ComovingEulerPlan,
        gravity: ParticleMeshGravityPlan,
        /,
    ):
        _validate_owner_bindings(wave, particles, gravity, gas)
        self.wave = wave
        self.particles = particles
        self.gas = gas
        self.gravity = gravity
        self.gravitational_constant = wave.gravitational_constant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-particle-gas-cosmology-plan",
                "wave": wave.prepared_id,
                "particles": particles.plan_id,
                "gas": gas.plan_id,
                "gravity": gravity.plan_id,
                "schedule": np.asarray(wave.scale_factors).tolist(),
                "gravitational_constant": wave.gravitational_constant,
                "integrator": "shared-endpoint-predictor-corrector-kdk",
            }
        )

    def prepare(self, /) -> "PreparedWaveParticleGasCosmology":
        return PreparedWaveParticleGasCosmology(self)


def _wave_norm(
    wave: PreparedPeriodicWaveDarkMatter, state: WaveDarkMatterState, /
) -> Array:
    return jnp.real(
        contract(
            "...,...->",
            jnp.abs(state.psi) ** 2,
            wave.discretization.quadrature_weights,
        )
    )


def _wave_kinetic_energy(
    wave: PreparedPeriodicWaveDarkMatter,
    state: WaveDarkMatterState,
    /,
) -> Array:
    coefficients = wave.discretization.project(state.psi)
    kinetic_integral = jnp.real(
        contract(
            "...,...->",
            wave.wavenumber_squared,
            jnp.abs(coefficients) ** 2,
        )
    )
    return (
        wave.reduced_planck_constant**2
        * kinetic_integral
        / (2.0 * wave.boson_mass * state.scale_factor**2)
    )


def _particle_kinetic_energy(
    particles: CosmologicalKDKPlan,
    momenta: Array,
    scale_factor: Array,
    /,
) -> Array:
    masses = particles.particles.safe_masses.astype(momenta.dtype)
    active = particles.particles.active_mask
    per_particle = jnp.sum(momenta**2, axis=-1) / (2.0 * masses * scale_factor**2)
    return jnp.sum(jnp.where(active, per_particle, 0.0))


def _wave_phase_evidence(
    wave: PreparedPeriodicWaveDarkMatter,
    drift_state: WaveDarkMatterState,
    start: Array,
    end: Array,
    first_potential: Array,
    second_potential: Array,
    /,
) -> tuple[Array, Array, Array]:
    drift = wave.background.drift_factor(start, end).astype(first_potential.dtype)
    kick = wave.background.kick_factor(start, end).astype(first_potential.dtype)
    coefficients = wave.discretization.project(drift_state.psi)
    power = jnp.abs(coefficients) ** 2
    occupied = power >= wave.step_policy.relative_amplitude_floor * jnp.max(power)
    kinetic_phase = (
        0.5
        * wave.reduced_planck_constant
        * wave.wavenumber_squared
        * drift
        / wave.boson_mass
    )
    kinetic = jnp.max(jnp.where(occupied, jnp.abs(kinetic_phase), 0.0))
    potential = (
        0.5
        * wave.boson_mass
        * jnp.abs(kick)
        * jnp.maximum(
            jnp.max(jnp.abs(first_potential)), jnp.max(jnp.abs(second_potential))
        )
        / wave.reduced_planck_constant
    )
    resolved = (kinetic <= wave.step_policy.maximum_phase_radians) & (
        potential <= wave.step_policy.maximum_phase_radians
    )
    return kinetic, potential, resolved


def _select_wave_particle(
    successful: Array,
    candidate: WaveParticleCosmologyState,
    previous: WaveParticleCosmologyState,
    /,
) -> WaveParticleCosmologyState:
    return WaveParticleCosmologyState(
        WaveDarkMatterState(
            jnp.where(successful, candidate.wave.psi, previous.wave.psi),
            jnp.where(
                successful, candidate.wave.scale_factor, previous.wave.scale_factor
            ),
        ),
        CosmologicalParticleState(
            jnp.where(
                successful, candidate.particles.positions, previous.particles.positions
            ),
            jnp.where(
                successful,
                candidate.particles.canonical_momenta,
                previous.particles.canonical_momenta,
            ),
            jnp.where(
                successful,
                candidate.particles.scale_factor,
                previous.particles.scale_factor,
            ),
        ),
    )


def _select_wave_particle_gas(
    successful: Array,
    candidate: WaveParticleGasCosmologyState,
    previous: WaveParticleGasCosmologyState,
    /,
) -> WaveParticleGasCosmologyState:
    selected = _select_wave_particle(
        successful,
        WaveParticleCosmologyState(candidate.wave, candidate.particles),
        WaveParticleCosmologyState(previous.wave, previous.particles),
    )
    return WaveParticleGasCosmologyState(
        selected.wave,
        selected.particles,
        ComovingEulerState(
            jnp.where(successful, candidate.gas.cell_average, previous.gas.cell_average),
            jnp.where(successful, candidate.gas.scale_factor, previous.gas.scale_factor),
        ),
    )


def _diagnostics(recorded: tuple[Array, ...], accepted_steps: Array, completed: Array, /):
    (
        attempted,
        accepted,
        rolled_back,
        status,
        start_scale,
        end_scale,
        wave_scale,
        particle_scale,
        gas_scale,
        time_consistent,
        mass_start,
        mass_end,
        total_start,
        total_end,
        mass_defect,
        deposition_defect,
        mean_density,
        source_integral,
        poisson_residual,
        gauge_defect,
        component_force,
        total_force,
        adjoint_defect,
        component_work,
        total_work,
        homogeneous_change,
        norm_start,
        norm_end,
        norm_error,
        kinetic_phase,
        potential_phase,
        phase_resolved,
        norm_conserved,
        density_nonnegative,
        gas_density_positive,
        gas_pressure_positive,
        gas_homogeneous_successful,
        component_successful,
        gravity_successful,
    ) = recorded
    failed = attempted & ~accepted
    first_failed = jnp.where(
        jnp.any(failed),
        jnp.argmax(failed).astype(jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    return MixedCosmologyDiagnostics(
        attempted=attempted,
        accepted=accepted,
        rolled_back=rolled_back,
        step_status=status,
        start_scale_factor=start_scale,
        end_scale_factor=end_scale,
        wave_scale_factor=wave_scale,
        particle_scale_factor=particle_scale,
        gas_scale_factor=gas_scale,
        time_level_consistent=time_consistent,
        component_mass_start=mass_start,
        component_mass_end=mass_end,
        total_mass_start=total_start,
        total_mass_end=total_end,
        mass_balance_defect=mass_defect,
        particle_deposition_defect=deposition_defect,
        mean_density=mean_density,
        source_integral=source_integral,
        poisson_relative_residual=poisson_residual,
        gauge_defect=gauge_defect,
        component_force=component_force,
        total_force=total_force,
        particle_force_adjoint_defect=adjoint_defect,
        component_gravity_work=component_work,
        total_gravity_work=total_work,
        gas_homogeneous_energy_change=homogeneous_change,
        wave_norm_start=norm_start,
        wave_norm_end=norm_end,
        wave_norm_relative_error=norm_error,
        maximum_wave_kinetic_phase=kinetic_phase,
        maximum_wave_potential_phase=potential_phase,
        wave_phase_resolved=phase_resolved,
        wave_norm_conserved=norm_conserved,
        density_nonnegative=density_nonnegative,
        gas_density_positive=gas_density_positive,
        gas_pressure_positive=gas_pressure_positive,
        gas_homogeneous_successful=gas_homogeneous_successful,
        component_successful=component_successful,
        gravity_successful=gravity_successful,
        accepted_steps=accepted_steps,
        completed=completed,
        first_failed_step=first_failed,
        component_names=_COMPONENT_NAMES,
    )


class PreparedWaveParticleCosmology(StrictModule):
    """Prepared atomic wave-particle evolution using one shared potential per stage."""

    plan: WaveParticleCosmologyPlan
    density: MixedDensityAssembler
    gravity: SharedPeriodicGravityPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: WaveParticleCosmologyPlan, /):
        if not isinstance(plan, WaveParticleCosmologyPlan):
            raise TypeError("plan must be WaveParticleCosmologyPlan.")
        density = MixedDensityAssembler(plan.wave, plan.gravity, include_gas=False)
        gravity = SharedPeriodicGravityPlan(density, plan.gravity)
        self.plan = plan
        self.density = density
        self.gravity = gravity
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-wave-particle-cosmology",
                "plan": plan.plan_id,
                "density": density.assembler_id,
                "gravity": gravity.plan_id,
            }
        )

    def rollout(
        self,
        state: WaveParticleCosmologyState,
        args: Any = None,
        /,
    ) -> WaveParticleCosmologyResult:
        if not isinstance(state, WaveParticleCosmologyState):
            raise TypeError("state must be WaveParticleCosmologyState.")
        wave = self.plan.wave
        particles = self.plan.particles
        initial_scale = wave.scale_factors[0].astype(state.wave.psi.real.dtype)
        tolerance = (
            32.0 * jnp.finfo(initial_scale.dtype).eps * jnp.maximum(initial_scale, 1.0)
        )
        wave_scale = jnp.asarray(state.wave.scale_factor, dtype=initial_scale.dtype)
        particle_scale = jnp.asarray(
            state.particles.scale_factor, dtype=initial_scale.dtype
        )
        if wave_scale.shape != () or particle_scale.shape != ():
            raise ValueError("Mixed component scale factors must be scalar.")
        initial_scale = eqx.error_if(
            initial_scale,
            ~jnp.isfinite(wave_scale)
            | (wave_scale <= 0.0)
            | ~jnp.isfinite(particle_scale)
            | (particle_scale <= 0.0)
            | (jnp.abs(wave_scale - initial_scale) > tolerance)
            | (jnp.abs(particle_scale - initial_scale) > tolerance),
            "Mixed wave and particle scale factors must be finite, positive, "
            "and start at the first scheduled scale factor.",
        )
        initial = WaveParticleCosmologyState(
            WaveDarkMatterState(state.wave.psi, initial_scale),
            CosmologicalParticleState(
                state.particles.positions,
                state.particles.canonical_momenta,
                initial_scale,
            ),
        )

        def step(carry, end):
            current, active, count = carry

            def attempt(_):
                start = current.wave.scale_factor
                assembly_0 = self.density.assemble(current)
                gravity_0 = self.gravity.solve(assembly_0, args)
                first_wave = wave.potential_kick(
                    current.wave, gravity_0.potential, start, end, 0.5
                )
                drifted_wave = wave.kinetic_drift(first_wave, start, end)
                proposal = particles.propose(
                    wave.background,
                    current.particles,
                    end,
                    gravity_0.particle_acceleration,
                )
                endpoint_predictor = WaveParticleCosmologyState(
                    drifted_wave,
                    CosmologicalParticleState(
                        proposal.positions,
                        proposal.half_momenta,
                        proposal.end_scale_factor,
                    ),
                )
                assembly_1 = self.density.assemble(endpoint_predictor)
                gravity_1 = self.gravity.solve(assembly_1, args)
                final_wave = wave.potential_kick(
                    drifted_wave, gravity_1.potential, start, end, 0.5
                )
                final_particles, particle_evidence = particles.complete(
                    current.particles,
                    proposal,
                    gravity_1.particle_acceleration,
                )
                candidate = WaveParticleCosmologyState(final_wave, final_particles)
                norm_0 = _wave_norm(wave, current.wave)
                norm_1 = _wave_norm(wave, final_wave)
                norm_error = jnp.abs(norm_1 - norm_0) / norm_0
                norm_closed = norm_error <= wave.step_policy.norm_relative_tolerance
                kinetic_phase, potential_phase, phase_resolved = _wave_phase_evidence(
                    wave,
                    first_wave,
                    start,
                    end,
                    gravity_0.potential,
                    gravity_1.potential,
                )
                wave_work = (
                    _wave_kinetic_energy(wave, first_wave)
                    - _wave_kinetic_energy(wave, current.wave)
                    + _wave_kinetic_energy(wave, final_wave)
                    - _wave_kinetic_energy(wave, drifted_wave)
                )
                particle_work = (
                    _particle_kinetic_energy(particles, proposal.half_momenta, start)
                    - _particle_kinetic_energy(
                        particles, current.particles.canonical_momenta, start
                    )
                    + _particle_kinetic_energy(
                        particles, final_particles.canonical_momenta, end
                    )
                    - _particle_kinetic_energy(particles, proposal.half_momenta, end)
                )
                component_work = jnp.stack(
                    (wave_work, particle_work, jnp.asarray(0.0, dtype=wave_work.dtype))
                )
                mass_defect = assembly_1.total_mass - assembly_0.total_mass
                mass_tolerance = (
                    256.0
                    * jnp.finfo(mass_defect.dtype).eps
                    * jnp.maximum(jnp.abs(assembly_0.total_mass), 1.0)
                )
                scale_tolerance = 32.0 * jnp.finfo(end.dtype).eps * jnp.maximum(end, 1.0)
                time_consistent = (
                    jnp.abs(final_wave.scale_factor - end) <= scale_tolerance
                ) & (jnp.abs(final_particles.scale_factor - end) <= scale_tolerance)
                components_ok = (
                    proposal.successful
                    & particle_evidence.successful
                    & norm_closed
                    & phase_resolved
                    & time_consistent
                    & (jnp.abs(mass_defect) <= mass_tolerance)
                    & jnp.all(jnp.isfinite(component_work))
                )
                gravity_ok = gravity_0.successful & gravity_1.successful
                successful = active & gravity_ok & components_ok
                accepted_state = _select_wave_particle(successful, candidate, current)
                status = jnp.where(
                    ~gravity_ok,
                    2,
                    jnp.where(~components_ok, 3, 0),
                ).astype(jnp.int32)
                recorded = (
                    jnp.asarray(True),
                    successful,
                    ~successful,
                    status,
                    start,
                    end,
                    candidate.wave.scale_factor,
                    candidate.particles.scale_factor,
                    candidate.wave.scale_factor,
                    time_consistent,
                    assembly_0.component_mass,
                    assembly_1.component_mass,
                    assembly_0.total_mass,
                    assembly_1.total_mass,
                    mass_defect,
                    jnp.maximum(
                        assembly_0.particle_mass_balance_defect,
                        assembly_1.particle_mass_balance_defect,
                    ),
                    gravity_1.mean_density,
                    jnp.maximum(
                        jnp.abs(gravity_0.source_integral),
                        jnp.abs(gravity_1.source_integral),
                    ),
                    jnp.maximum(
                        gravity_0.poisson_relative_residual,
                        gravity_1.poisson_relative_residual,
                    ),
                    jnp.maximum(gravity_0.gauge_defect, gravity_1.gauge_defect),
                    gravity_1.component_force,
                    gravity_1.total_force,
                    jnp.maximum(
                        gravity_0.particle_force_adjoint_defect,
                        gravity_1.particle_force_adjoint_defect,
                    ),
                    component_work,
                    jnp.sum(component_work),
                    jnp.asarray(0.0, dtype=wave_work.dtype),
                    norm_0,
                    norm_1,
                    norm_error,
                    kinetic_phase,
                    potential_phase,
                    phase_resolved,
                    norm_closed,
                    assembly_0.density_nonnegative & assembly_1.density_nonnegative,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(True),
                    components_ok,
                    gravity_ok,
                )
                return (
                    accepted_state,
                    successful,
                    count + successful.astype(jnp.int32),
                ), recorded

            def stopped(_):
                assembly = self.density.assemble(current)
                zero = jnp.asarray(0.0, dtype=current.wave.psi.real.dtype)
                zeros_component = jnp.zeros((3,), dtype=zero.dtype)
                zeros_force = jnp.zeros(
                    (3, particles.particles.ambient_dimension), dtype=zero.dtype
                )
                recorded = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(1, dtype=jnp.int32),
                    current.wave.scale_factor,
                    end,
                    current.wave.scale_factor,
                    current.particles.scale_factor,
                    current.wave.scale_factor,
                    jnp.asarray(False),
                    assembly.component_mass,
                    assembly.component_mass,
                    assembly.total_mass,
                    assembly.total_mass,
                    zero,
                    assembly.particle_mass_balance_defect,
                    zero,
                    zero,
                    zero,
                    zero,
                    zeros_force,
                    jnp.zeros((particles.particles.ambient_dimension,), dtype=zero.dtype),
                    zero,
                    zeros_component,
                    zero,
                    zero,
                    _wave_norm(wave, current.wave),
                    _wave_norm(wave, current.wave),
                    zero,
                    zero,
                    zero,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    assembly.density_nonnegative,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(False),
                    jnp.asarray(False),
                )
                return carry, recorded

            return jax.lax.cond(active, attempt, stopped, operand=None)

        final_carry, recorded = jax.lax.scan(
            step,
            (initial, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32)),
            wave.scale_factors[1:].astype(initial_scale.dtype),
        )
        final_state, active, accepted_steps = final_carry
        completed = active & (accepted_steps == wave.scale_factors.size - 1)
        final_assembly = self.density.assemble(final_state)
        final_gravity = self.gravity.solve(final_assembly, args)
        diagnostics = _diagnostics(recorded, accepted_steps, completed)
        return WaveParticleCosmologyResult(
            final_state,
            final_gravity,
            diagnostics,
            completed,
            self.prepared_id,
            self.plan.wave.prepared_id,
            self.plan.particles.plan_id,
            self.plan.particles.particles.prepared_id,
            self.gravity.plan_id,
            self.density.assembler_id,
            self.plan.wave.scale_id,
        )


class PreparedWaveParticleGasCosmology(StrictModule):
    """Prepared atomic wave-particle-gas evolution with endpoint gravity correction."""

    plan: WaveParticleGasCosmologyPlan
    density: MixedDensityAssembler
    gravity: SharedPeriodicGravityPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: WaveParticleGasCosmologyPlan, /):
        if not isinstance(plan, WaveParticleGasCosmologyPlan):
            raise TypeError("plan must be WaveParticleGasCosmologyPlan.")
        density = MixedDensityAssembler(plan.wave, plan.gravity, include_gas=True)
        gravity = SharedPeriodicGravityPlan(density, plan.gravity)
        self.plan = plan
        self.density = density
        self.gravity = gravity
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-wave-particle-gas-cosmology",
                "plan": plan.plan_id,
                "density": density.assembler_id,
                "gravity": gravity.plan_id,
            }
        )

    def rollout(
        self,
        state: WaveParticleGasCosmologyState,
        args: Any = None,
        /,
    ) -> WaveParticleGasCosmologyResult:
        if not isinstance(state, WaveParticleGasCosmologyState):
            raise TypeError("state must be WaveParticleGasCosmologyState.")
        wave = self.plan.wave
        particles = self.plan.particles
        gas = self.plan.gas
        initial_scale = wave.scale_factors[0].astype(state.wave.psi.real.dtype)
        tolerance = (
            32.0 * jnp.finfo(initial_scale.dtype).eps * jnp.maximum(initial_scale, 1.0)
        )
        wave_scale = jnp.asarray(state.wave.scale_factor, dtype=initial_scale.dtype)
        particle_scale = jnp.asarray(
            state.particles.scale_factor, dtype=initial_scale.dtype
        )
        gas_scale = jnp.asarray(state.gas.scale_factor, dtype=initial_scale.dtype)
        if wave_scale.shape != () or particle_scale.shape != () or gas_scale.shape != ():
            raise ValueError("Mixed component scale factors must be scalar.")
        initial_scale = eqx.error_if(
            initial_scale,
            ~jnp.isfinite(wave_scale)
            | (wave_scale <= 0.0)
            | ~jnp.isfinite(particle_scale)
            | (particle_scale <= 0.0)
            | ~jnp.isfinite(gas_scale)
            | (gas_scale <= 0.0)
            | (jnp.abs(wave_scale - initial_scale) > tolerance)
            | (jnp.abs(particle_scale - initial_scale) > tolerance)
            | (jnp.abs(gas_scale - initial_scale) > tolerance),
            "Mixed wave, particle, and gas scale factors must be finite, positive, "
            "and start at the first scheduled scale factor.",
        )
        initial = WaveParticleGasCosmologyState(
            WaveDarkMatterState(state.wave.psi, initial_scale),
            CosmologicalParticleState(
                state.particles.positions,
                state.particles.canonical_momenta,
                initial_scale,
            ),
            ComovingEulerState(state.gas.cell_average, initial_scale),
        )

        def step(carry, end):
            current, active, count = carry

            def attempt(_):
                start = current.wave.scale_factor
                assembly_0 = self.density.assemble(current)
                gravity_0 = self.gravity.solve(assembly_0, args)
                first_wave = wave.potential_kick(
                    current.wave, gravity_0.potential, start, end, 0.5
                )
                drifted_wave = wave.kinetic_drift(first_wave, start, end)
                proposal = particles.propose(
                    wave.background,
                    current.particles,
                    end,
                    gravity_0.particle_acceleration,
                )
                predicted_gas, predicted_gas_evidence = gas.advance(
                    wave.background,
                    current.gas,
                    end,
                    gravity_0.cell_acceleration,
                    gravity_0.cell_acceleration,
                    args,
                )
                predicted_state = WaveParticleGasCosmologyState(
                    drifted_wave,
                    CosmologicalParticleState(
                        proposal.positions,
                        proposal.half_momenta,
                        proposal.end_scale_factor,
                    ),
                    predicted_gas,
                )
                predicted_assembly = self.density.assemble(predicted_state)
                predicted_gravity = self.gravity.solve(predicted_assembly, args)
                corrected_gas, gas_evidence = gas.advance(
                    wave.background,
                    current.gas,
                    end,
                    gravity_0.cell_acceleration,
                    predicted_gravity.cell_acceleration,
                    args,
                )
                zero_gravity = jnp.zeros_like(gravity_0.cell_acceleration)
                homogeneous_gas, homogeneous_evidence = gas.advance(
                    wave.background,
                    current.gas,
                    end,
                    zero_gravity,
                    zero_gravity,
                    args,
                )
                endpoint_state = WaveParticleGasCosmologyState(
                    drifted_wave,
                    predicted_state.particles,
                    corrected_gas,
                )
                assembly_1 = self.density.assemble(endpoint_state)
                gravity_1 = self.gravity.solve(assembly_1, args)
                final_wave = wave.potential_kick(
                    drifted_wave, gravity_1.potential, start, end, 0.5
                )
                final_particles, particle_evidence = particles.complete(
                    current.particles,
                    proposal,
                    gravity_1.particle_acceleration,
                )
                candidate = WaveParticleGasCosmologyState(
                    final_wave, final_particles, corrected_gas
                )
                norm_0 = _wave_norm(wave, current.wave)
                norm_1 = _wave_norm(wave, final_wave)
                norm_error = jnp.abs(norm_1 - norm_0) / norm_0
                norm_closed = norm_error <= wave.step_policy.norm_relative_tolerance
                kinetic_phase, potential_phase, phase_resolved = _wave_phase_evidence(
                    wave,
                    first_wave,
                    start,
                    end,
                    gravity_0.potential,
                    gravity_1.potential,
                )
                wave_work = (
                    _wave_kinetic_energy(wave, first_wave)
                    - _wave_kinetic_energy(wave, current.wave)
                    + _wave_kinetic_energy(wave, final_wave)
                    - _wave_kinetic_energy(wave, drifted_wave)
                )
                particle_work = (
                    _particle_kinetic_energy(particles, proposal.half_momenta, start)
                    - _particle_kinetic_energy(
                        particles, current.particles.canonical_momenta, start
                    )
                    + _particle_kinetic_energy(
                        particles, final_particles.canonical_momenta, end
                    )
                    - _particle_kinetic_energy(particles, proposal.half_momenta, end)
                )
                volumes = self.density.cell_volumes.astype(
                    corrected_gas.cell_average.dtype
                )
                gas_work = contract(
                    "...,...->",
                    corrected_gas.cell_average[..., -1]
                    - homogeneous_gas.cell_average[..., -1],
                    volumes,
                )
                homogeneous_change = contract(
                    "...,...->",
                    homogeneous_gas.cell_average[..., -1]
                    - current.gas.cell_average[..., -1],
                    volumes,
                )
                component_work = jnp.stack((wave_work, particle_work, gas_work))
                mass_defect = assembly_1.total_mass - assembly_0.total_mass
                mass_tolerance = (
                    512.0
                    * jnp.finfo(mass_defect.dtype).eps
                    * jnp.maximum(jnp.abs(assembly_0.total_mass), 1.0)
                    * gas.substeps
                )
                scale_tolerance = 32.0 * jnp.finfo(end.dtype).eps * jnp.maximum(end, 1.0)
                time_consistent = (
                    (jnp.abs(final_wave.scale_factor - end) <= scale_tolerance)
                    & (jnp.abs(final_particles.scale_factor - end) <= scale_tolerance)
                    & (jnp.abs(corrected_gas.scale_factor - end) <= scale_tolerance)
                )
                components_ok = (
                    proposal.successful
                    & particle_evidence.successful
                    & predicted_gas_evidence.successful
                    & gas_evidence.successful
                    & homogeneous_evidence.successful
                    & norm_closed
                    & phase_resolved
                    & time_consistent
                    & (jnp.abs(mass_defect) <= mass_tolerance)
                    & jnp.all(jnp.isfinite(component_work))
                    & jnp.isfinite(homogeneous_change)
                )
                gravity_ok = (
                    gravity_0.successful
                    & predicted_gravity.successful
                    & gravity_1.successful
                )
                successful = active & gravity_ok & components_ok
                accepted_state = _select_wave_particle_gas(successful, candidate, current)
                status = jnp.where(
                    ~gravity_ok,
                    2,
                    jnp.where(~components_ok, 3, 0),
                ).astype(jnp.int32)
                recorded = (
                    jnp.asarray(True),
                    successful,
                    ~successful,
                    status,
                    start,
                    end,
                    candidate.wave.scale_factor,
                    candidate.particles.scale_factor,
                    candidate.gas.scale_factor,
                    time_consistent,
                    assembly_0.component_mass,
                    assembly_1.component_mass,
                    assembly_0.total_mass,
                    assembly_1.total_mass,
                    mass_defect,
                    jnp.maximum(
                        jnp.maximum(
                            assembly_0.particle_mass_balance_defect,
                            predicted_assembly.particle_mass_balance_defect,
                        ),
                        assembly_1.particle_mass_balance_defect,
                    ),
                    gravity_1.mean_density,
                    jnp.maximum(
                        jnp.maximum(
                            jnp.abs(gravity_0.source_integral),
                            jnp.abs(predicted_gravity.source_integral),
                        ),
                        jnp.abs(gravity_1.source_integral),
                    ),
                    jnp.maximum(
                        jnp.maximum(
                            gravity_0.poisson_relative_residual,
                            predicted_gravity.poisson_relative_residual,
                        ),
                        gravity_1.poisson_relative_residual,
                    ),
                    jnp.maximum(
                        jnp.maximum(
                            gravity_0.gauge_defect, predicted_gravity.gauge_defect
                        ),
                        gravity_1.gauge_defect,
                    ),
                    gravity_1.component_force,
                    gravity_1.total_force,
                    jnp.maximum(
                        jnp.maximum(
                            gravity_0.particle_force_adjoint_defect,
                            predicted_gravity.particle_force_adjoint_defect,
                        ),
                        gravity_1.particle_force_adjoint_defect,
                    ),
                    component_work,
                    jnp.sum(component_work),
                    homogeneous_change,
                    norm_0,
                    norm_1,
                    norm_error,
                    kinetic_phase,
                    potential_phase,
                    phase_resolved,
                    norm_closed,
                    assembly_0.density_nonnegative
                    & predicted_assembly.density_nonnegative
                    & assembly_1.density_nonnegative,
                    gas_evidence.density_positive,
                    gas_evidence.pressure_positive,
                    homogeneous_evidence.successful,
                    components_ok,
                    gravity_ok,
                )
                return (
                    accepted_state,
                    successful,
                    count + successful.astype(jnp.int32),
                ), recorded

            def stopped(_):
                assembly = self.density.assemble(current)
                zero = jnp.asarray(0.0, dtype=current.wave.psi.real.dtype)
                zeros_component = jnp.zeros((3,), dtype=zero.dtype)
                zeros_force = jnp.zeros(
                    (3, particles.particles.ambient_dimension), dtype=zero.dtype
                )
                recorded = (
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(1, dtype=jnp.int32),
                    current.wave.scale_factor,
                    end,
                    current.wave.scale_factor,
                    current.particles.scale_factor,
                    current.gas.scale_factor,
                    jnp.asarray(False),
                    assembly.component_mass,
                    assembly.component_mass,
                    assembly.total_mass,
                    assembly.total_mass,
                    zero,
                    assembly.particle_mass_balance_defect,
                    zero,
                    zero,
                    zero,
                    zero,
                    zeros_force,
                    jnp.zeros((particles.particles.ambient_dimension,), dtype=zero.dtype),
                    zero,
                    zeros_component,
                    zero,
                    zero,
                    _wave_norm(wave, current.wave),
                    _wave_norm(wave, current.wave),
                    zero,
                    zero,
                    zero,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    assembly.density_nonnegative,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(False),
                )
                return carry, recorded

            return jax.lax.cond(active, attempt, stopped, operand=None)

        final_carry, recorded = jax.lax.scan(
            step,
            (initial, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32)),
            wave.scale_factors[1:].astype(initial_scale.dtype),
        )
        final_state, active, accepted_steps = final_carry
        completed = active & (accepted_steps == wave.scale_factors.size - 1)
        final_assembly = self.density.assemble(final_state)
        final_gravity = self.gravity.solve(final_assembly, args)
        diagnostics = _diagnostics(recorded, accepted_steps, completed)
        return WaveParticleGasCosmologyResult(
            final_state,
            final_gravity,
            diagnostics,
            completed,
            self.prepared_id,
            self.plan.wave.prepared_id,
            self.plan.particles.plan_id,
            self.plan.gas.plan_id,
            self.plan.particles.particles.prepared_id,
            self.gravity.plan_id,
            self.density.assembler_id,
            self.plan.wave.scale_id,
        )


__all__ = [
    "MixedCosmologyDiagnostics",
    "MixedDensityAssembler",
    "MixedDensityAssembly",
    "PreparedWaveParticleCosmology",
    "PreparedWaveParticleGasCosmology",
    "SharedPeriodicGravityPlan",
    "SharedPeriodicGravityResult",
    "WaveParticleCosmologyPlan",
    "WaveParticleCosmologyResult",
    "WaveParticleCosmologyState",
    "WaveParticleGasCosmologyPlan",
    "WaveParticleGasCosmologyResult",
    "WaveParticleGasCosmologyState",
]
