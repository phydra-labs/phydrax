#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._phase_field import (
    double_well_chemical_derivative,
    double_well_free_energy_density,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import PreparedLatticeBoltzmannBoundary
from ._collision import macroscopic_raw_moments, quadratic_equilibrium
from ._discretization import LatticeBoltzmannDiscretization
from ._interfacial import (
    isotropic_divergence,
    isotropic_gradient,
    isotropic_laplacian,
    korteweg_stress,
    natural_wetting_gradient,
)
from ._lattice import LatticeBoltzmannVelocitySet
from ._method import (
    LatticeBoltzmannMethodPlan,
    PreparedLatticeBoltzmannMethodPlan,
)
from ._precision import LatticeBoltzmannPrecisionPolicy
from ._scaling import LatticeBoltzmannScaling


class FreeEnergyLBMState(StrictModule):
    """Hydrodynamic and phase populations with independent trailing-Q axes."""

    hydrodynamic_populations: Array
    phase_populations: Array


class FreeEnergyLBMRuntimeParameters(StrictModule):
    """Differentiable free-energy, mobility, viscosity, and wetting controls.

    The free-energy coefficients and phase mobility use lattice units. Viscosity
    and moving-wall velocity use physical units through the compiled scaling.
    """

    kinematic_viscosity: Array
    phase_mobility: Array
    bulk_coefficient: Array
    gradient_coefficient: Array
    wetting_strength: Array
    moving_wall_velocities: Array
    wall_normal: Array
    wetting_mask: Array

    def __init__(
        self,
        kinematic_viscosity: ArrayLike,
        phase_mobility: ArrayLike,
        bulk_coefficient: ArrayLike,
        gradient_coefficient: ArrayLike,
        /,
        *,
        wetting_strength: ArrayLike = 0.0,
        moving_wall_velocities: ArrayLike | None = None,
        wall_normal: ArrayLike | None = None,
        wetting_mask: ArrayLike | None = None,
    ):
        viscosity = jnp.asarray(kinematic_viscosity)
        if viscosity.shape != () or not jnp.issubdtype(viscosity.dtype, jnp.inexact):
            raise ValueError("kinematic_viscosity must be one inexact scalar array.")
        mobility, bulk, gradient, wetting = tuple(
            jnp.asarray(value, dtype=viscosity.dtype)
            for value in (
                phase_mobility,
                bulk_coefficient,
                gradient_coefficient,
                wetting_strength,
            )
        )
        if any(value.shape != () for value in (mobility, bulk, gradient, wetting)):
            raise ValueError("Free-energy coefficients must be scalar arrays.")
        if (wall_normal is None) != (wetting_mask is None):
            raise ValueError("wall_normal and wetting_mask must be supplied together.")
        self.kinematic_viscosity = viscosity
        self.phase_mobility = mobility
        self.bulk_coefficient = bulk
        self.gradient_coefficient = gradient
        self.wetting_strength = wetting
        self.moving_wall_velocities = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if moving_wall_velocities is None
            else jnp.asarray(moving_wall_velocities, dtype=viscosity.dtype)
        )
        self.wall_normal = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if wall_normal is None
            else jnp.asarray(wall_normal, dtype=viscosity.dtype)
        )
        self.wetting_mask = (
            jnp.empty((0,), dtype=bool)
            if wetting_mask is None
            else jnp.asarray(wetting_mask, dtype=bool)
        )


class FreeEnergyLBMMethod(StrictModule, NonTrainableState):
    """Coupled hydrodynamic and conservative Cahn-Hilliard population method."""

    hydrodynamic_method: LatticeBoltzmannMethodPlan
    density_floor: float = eqx.field(static=True)
    maximum_absolute_phase: float = eqx.field(static=True)
    minimum_interface_cells: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    maximum_capillary_number: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    relative_energy_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_method: LatticeBoltzmannMethodPlan,
        /,
        *,
        density_floor: float = 1.0e-12,
        maximum_absolute_phase: float = 1.25,
        minimum_interface_cells: float = 1.0,
        maximum_mach: float = 0.3,
        maximum_capillary_number: float = 1.0,
        conservation_tolerance: float = 1.0e-11,
        relative_energy_tolerance: float = 1.0e-8,
    ):
        if not isinstance(hydrodynamic_method, LatticeBoltzmannMethodPlan):
            raise TypeError("hydrodynamic_method must be LatticeBoltzmannMethodPlan.")
        if hydrodynamic_method.forcing is None:
            raise ValueError("Free-energy chemical forcing requires a forced LBM method.")
        values = tuple(
            float(value)
            for value in (
                density_floor,
                maximum_absolute_phase,
                minimum_interface_cells,
                maximum_mach,
                maximum_capillary_number,
                conservation_tolerance,
                relative_energy_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Free-energy method limits must be finite and positive.")
        rho_floor, max_phase, min_width, mach, capillary, mass_tol, energy_tol = values
        if max_phase < 1.0:
            raise ValueError("maximum_absolute_phase cannot be smaller than one.")
        if mach >= 1.0:
            raise ValueError("maximum_mach must be smaller than one.")
        self.hydrodynamic_method = hydrodynamic_method
        self.density_floor = rho_floor
        self.maximum_absolute_phase = max_phase
        self.minimum_interface_cells = min_width
        self.maximum_mach = mach
        self.maximum_capillary_number = capillary
        self.conservation_tolerance = mass_tol
        self.relative_energy_tolerance = energy_tol
        self.method_id = canonical_fingerprint(
            {
                "kind": "free-energy-lattice-boltzmann-method",
                "hydrodynamic_method": hydrodynamic_method.method_id,
                "density_floor": rho_floor,
                "maximum_absolute_phase": max_phase,
                "minimum_interface_cells": min_width,
                "maximum_mach": mach,
                "maximum_capillary_number": capillary,
                "conservation_tolerance": mass_tol,
                "relative_energy_tolerance": energy_tol,
            }
        )


class FreeEnergyPhaseFields(StrictModule):
    phase: Array
    gradient: Array
    laplacian: Array
    bulk_energy_density: Array
    gradient_energy_density: Array
    chemical_potential: Array
    chemical_force_density: Array
    korteweg_stress: Array


class PhasePopulationMoments(StrictModule):
    phase: Array
    phase_flux: Array
    second_moment: Array


class FreeEnergyLedger(StrictModule):
    mixture_mass: Array
    phase_mass: Array
    bulk_free_energy: Array
    gradient_free_energy: Array
    free_energy: Array
    kinetic_energy: Array
    total_energy: Array


class FreeEnergyMacroscopicState(StrictModule):
    density: Array
    velocity: Array
    pressure: Array
    phase_fields: FreeEnergyPhaseFields
    ledger: FreeEnergyLedger


class FreeEnergyDiagnostics(StrictModule):
    ledger: FreeEnergyLedger
    mixture_mass_defect: Array
    phase_mass_defect: Array
    energy_change: Array
    minimum_density: Array
    maximum_absolute_phase: Array
    maximum_mach: Array
    maximum_capillary_number: Array
    interface_cells: Array
    phase_equilibrium_mass_defect: Array
    phase_equilibrium_flux_defect: Array


class FreeEnergyStepResult(StrictModule):
    candidate_state: FreeEnergyLBMState
    accepted_state: FreeEnergyLBMState
    successful: Array
    residual: Array
    work: Array
    diagnostics: FreeEnergyDiagnostics


class _FreeEnergyFields(StrictModule):
    density: Array
    raw_momentum: Array
    velocity: Array
    phase_fields: FreeEnergyPhaseFields


def phase_population_moments(
    populations: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> PhasePopulationMoments:
    """Return explicit zeroth, first, and second phase-population moments."""

    values = precision.accumulation(populations)
    if values.shape[-1] != velocity_set.population_count:
        raise ValueError("Phase populations have an incompatible trailing axis.")
    velocities = precision.accumulation(velocity_set.velocities)
    return PhasePopulationMoments(
        jnp.sum(values, axis=-1),
        oe.contract("...q,qa->...a", values, velocities),
        oe.contract("...q,qa,qb->...ab", values, velocities, velocities),
    )


def phase_field_equilibrium(
    phase: ArrayLike,
    chemical_potential: ArrayLike,
    velocity: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Construct a phase equilibrium with explicit phase and advective moments."""

    phi = precision.compute(phase)
    chemical = precision.compute(chemical_potential)
    flow = precision.compute(velocity)
    if chemical.shape != phi.shape or flow.shape != (*phi.shape, velocity_set.dimension):
        raise ValueError("Phase-equilibrium fields have incompatible shapes.")
    weights = precision.coefficient(velocity_set.weights)
    velocities = precision.coefficient(velocity_set.velocities)
    cs2 = precision.coefficient(velocity_set.sound_speed_squared)
    cu = oe.contract("...d,qd->...q", flow, velocities)
    equilibrium = weights * (chemical[..., None] / cs2 + phi[..., None] * cu / cs2)
    rest = velocity_set.velocity_tuples.index((0,) * velocity_set.dimension)
    equilibrium = equilibrium.at[..., rest].add(phi - chemical / cs2)
    return precision.compute(equilibrium)


def phase_relaxation_rate(phase_mobility: ArrayLike, /) -> Array:
    """Map lattice Cahn-Hilliard mobility to its BGK relaxation rate."""

    mobility = jnp.asarray(phase_mobility)
    if mobility.shape != ():
        raise ValueError("phase_mobility must be scalar.")
    rate = 1.0 / (0.5 + mobility)
    return eqx.error_if(
        rate,
        ~jnp.isfinite(mobility)
        | (mobility <= 0.0)
        | ~jnp.isfinite(rate)
        | (rate <= 0.0)
        | (rate >= 2.0),
        "phase_mobility must produce a phase relaxation rate in (0, 2).",
    )


def free_energy_phase_fields(
    phase: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    bulk_coefficient: ArrayLike,
    gradient_coefficient: ArrayLike,
    /,
    *,
    wall_normal: ArrayLike | None = None,
    wetting_mask: ArrayLike | None = None,
    wetting_strength: ArrayLike = 0.0,
) -> FreeEnergyPhaseFields:
    """Evaluate bulk energy, natural wetting, chemical potential, force, and stress."""

    phi = jnp.asarray(phase)
    bulk = jnp.asarray(bulk_coefficient, dtype=phi.dtype)
    kappa = jnp.asarray(gradient_coefficient, dtype=phi.dtype)
    if bulk.shape != () or kappa.shape != ():
        raise ValueError("Free-energy coefficients must be scalar.")
    bulk = eqx.error_if(
        bulk,
        ~jnp.isfinite(bulk) | (bulk <= 0.0),
        "bulk_coefficient must be finite and positive.",
    )
    kappa = eqx.error_if(
        kappa,
        ~jnp.isfinite(kappa) | (kappa <= 0.0),
        "gradient_coefficient must be finite and positive.",
    )
    gradient = isotropic_gradient(phi, velocity_set, 1.0)
    if (wall_normal is None) != (wetting_mask is None):
        raise ValueError("wall_normal and wetting_mask must be supplied together.")
    if wall_normal is None:
        laplacian = isotropic_laplacian(phi, velocity_set, 1.0)
    elif wetting_mask is not None:
        gradient = natural_wetting_gradient(
            phi,
            gradient,
            wall_normal,
            wetting_strength,
            kappa,
            wetting_mask,
        )
        laplacian = isotropic_divergence(gradient, velocity_set, 1.0)
    bulk_density = double_well_free_energy_density(phi, bulk)
    gradient_squared = oe.contract("...d,...d->...", gradient, gradient)
    gradient_density = 0.5 * kappa * gradient_squared
    chemical = double_well_chemical_derivative(phi, bulk) - kappa * laplacian
    force = -phi[..., None] * isotropic_gradient(chemical, velocity_set, 1.0)
    stress = korteweg_stress(phi, chemical, gradient, bulk, kappa)
    return FreeEnergyPhaseFields(
        phi,
        gradient,
        laplacian,
        bulk_density,
        gradient_density,
        chemical,
        force,
        stress,
    )


def free_energy_surface_tension(
    bulk_coefficient: ArrayLike, gradient_coefficient: ArrayLike, /
) -> Array:
    """Return the planar-interface surface tension in lattice units."""

    bulk = jnp.asarray(bulk_coefficient)
    gradient = jnp.asarray(gradient_coefficient, dtype=bulk.dtype)
    if bulk.shape != () or gradient.shape != ():
        raise ValueError("Free-energy coefficients must be scalar.")
    value = 2.0 * jnp.sqrt(2.0 * bulk * gradient) / 3.0
    return eqx.error_if(
        value,
        ~jnp.isfinite(bulk) | ~jnp.isfinite(gradient) | (bulk <= 0.0) | (gradient <= 0.0),
        "Free-energy coefficients must be finite and positive.",
    )


class PreparedFreeEnergyLBMDynamics(StrictModule, NonTrainableState):
    """Pure coupled hydrodynamic and free-energy phase-population dynamics."""

    discretization: LatticeBoltzmannDiscretization
    scaling: LatticeBoltzmannScaling
    method: FreeEnergyLBMMethod
    hydrodynamic_method: PreparedLatticeBoltzmannMethodPlan
    boundary: PreparedLatticeBoltzmannBoundary
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        scaling: LatticeBoltzmannScaling,
        method: FreeEnergyLBMMethod,
        boundary: PreparedLatticeBoltzmannBoundary,
        /,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be an LBM discretization.")
        if not isinstance(scaling, LatticeBoltzmannScaling):
            raise TypeError("scaling must be LatticeBoltzmannScaling.")
        if not isinstance(method, FreeEnergyLBMMethod):
            raise TypeError("method must be FreeEnergyLBMMethod.")
        if not isinstance(boundary, PreparedLatticeBoltzmannBoundary):
            raise TypeError("boundary must be a prepared LBM boundary.")
        if boundary.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Boundary and free-energy discretizations do not match.")
        if not np.isclose(
            float(scaling.sound_speed_squared),
            float(discretization.velocity_set.sound_speed_squared),
        ):
            raise ValueError("Scaling and velocity-set sound speeds do not match.")
        if not np.isclose(float(scaling.cell_size), float(discretization.cell_size)):
            raise ValueError("Scaling and discretization cell sizes do not match.")
        hydrodynamic_method = method.hydrodynamic_method.prepare(
            discretization.velocity_set,
            discretization.precision,
        )
        self.discretization = discretization
        self.scaling = scaling
        self.method = method
        self.hydrodynamic_method = hydrodynamic_method
        self.boundary = boundary
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-free-energy-lattice-boltzmann-dynamics",
                "discretization": discretization.prepared_id,
                "scaling": scaling.scaling_id,
                "method": method.method_id,
                "prepared_hydrodynamic_method": hydrodynamic_method.method_id,
                "boundary": boundary.boundary_id,
            }
        )

    def _parameters(self, args: Any, /) -> FreeEnergyLBMRuntimeParameters:
        if not isinstance(args, FreeEnergyLBMRuntimeParameters):
            raise TypeError(
                "Free-energy fixed-step args must be FreeEnergyLBMRuntimeParameters."
            )
        return args

    def _validate_state(self, state: FreeEnergyLBMState, /) -> FreeEnergyLBMState:
        if not isinstance(state, FreeEnergyLBMState):
            raise TypeError("state must be FreeEnergyLBMState.")
        return FreeEnergyLBMState(
            self.discretization.validate_populations(state.hydrodynamic_populations),
            self.discretization.validate_populations(state.phase_populations),
        )

    def _wetting_data(
        self, parameters: FreeEnergyLBMRuntimeParameters, dtype, /
    ) -> tuple[Array | None, Array | None, Array, Array]:
        strength = jnp.asarray(parameters.wetting_strength, dtype=dtype)
        strength_valid = jnp.isfinite(strength)
        safe_strength = jnp.where(strength_valid, strength, 0.0)
        if parameters.wetting_mask.size == 0:
            return None, None, safe_strength, strength_valid
        shape = self.discretization.grid.shape
        dimension = self.discretization.velocity_set.dimension
        if parameters.wetting_mask.shape != shape:
            raise ValueError("wetting_mask must match the lattice grid shape.")
        if parameters.wall_normal.shape != (*shape, dimension):
            raise ValueError("wall_normal must contain one vector per lattice cell.")
        mask = parameters.wetting_mask
        wall = jnp.asarray(parameters.wall_normal, dtype=dtype)
        norm = jnp.sqrt(oe.contract("...d,...d->...", wall, wall))
        normal_valid = jnp.all(jnp.isfinite(wall), axis=-1) & (norm > 0.0)
        fallback = jnp.zeros_like(wall).at[..., 0].set(1.0)
        safe_wall = jnp.where((~mask | normal_valid)[..., None], wall, fallback)
        valid = strength_valid & jnp.all(~mask | normal_valid)
        return safe_wall, mask, safe_strength, valid

    def _safe_coefficients(
        self, parameters: FreeEnergyLBMRuntimeParameters, dtype, /
    ) -> tuple[Array, Array, Array]:
        bulk = jnp.asarray(parameters.bulk_coefficient, dtype=dtype)
        gradient = jnp.asarray(parameters.gradient_coefficient, dtype=dtype)
        valid = (
            jnp.isfinite(bulk) & (bulk > 0.0) & jnp.isfinite(gradient) & (gradient > 0.0)
        )
        return jnp.where(valid, bulk, 1.0), jnp.where(valid, gradient, 1.0), valid

    def _fields(
        self,
        state: FreeEnergyLBMState,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> tuple[_FreeEnergyFields, Array]:
        density, raw_momentum = macroscopic_raw_moments(
            state.hydrodynamic_populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        phase_moments = phase_population_moments(
            state.phase_populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        wall, mask, wetting, wetting_valid = self._wetting_data(
            parameters, state.hydrodynamic_populations.dtype
        )
        bulk, gradient, coefficient_valid = self._safe_coefficients(
            parameters, state.hydrodynamic_populations.dtype
        )
        phase_fields = free_energy_phase_fields(
            phase_moments.phase,
            self.discretization.velocity_set,
            bulk,
            gradient,
            wall_normal=wall,
            wetting_mask=mask,
            wetting_strength=wetting,
        )
        safe_density = jnp.maximum(density, self.method.density_floor)
        velocity = (
            raw_momentum + 0.5 * phase_fields.chemical_force_density
        ) / safe_density[..., None]
        return (
            _FreeEnergyFields(density, raw_momentum, velocity, phase_fields),
            wetting_valid & coefficient_valid,
        )

    def _ledger(self, fields: _FreeEnergyFields, /) -> FreeEnergyLedger:
        fluid = self.boundary.geometry.fluid_mask
        kinetic_density = (
            0.5
            * fields.density
            * oe.contract("...d,...d->...", fields.velocity, fields.velocity)
        )
        mixture_mass = jnp.sum(jnp.where(fluid, fields.density, 0.0))
        phase_mass = jnp.sum(jnp.where(fluid, fields.phase_fields.phase, 0.0))
        bulk = jnp.sum(jnp.where(fluid, fields.phase_fields.bulk_energy_density, 0.0))
        gradient = jnp.sum(
            jnp.where(fluid, fields.phase_fields.gradient_energy_density, 0.0)
        )
        kinetic = jnp.sum(jnp.where(fluid, kinetic_density, 0.0))
        return FreeEnergyLedger(
            mixture_mass,
            phase_mass,
            bulk,
            gradient,
            bulk + gradient,
            kinetic,
            bulk + gradient + kinetic,
        )

    def initialize_state(
        self,
        density: ArrayLike,
        phase: ArrayLike,
        velocity: ArrayLike,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyLBMState:
        parameters_ = self._parameters(parameters)
        dtype = jnp.dtype(self.discretization.precision.population_dtype)
        shape = self.discretization.grid.shape
        physical_density = jnp.asarray(density, dtype=dtype)
        phase_field = jnp.asarray(phase, dtype=dtype)
        if physical_density.shape == ():
            physical_density = jnp.broadcast_to(physical_density, shape)
        if phase_field.shape == ():
            phase_field = jnp.broadcast_to(phase_field, shape)
        if physical_density.shape != shape or phase_field.shape != shape:
            raise ValueError(
                "Initial density and phase must be scalar or match the grid."
            )
        dimension = self.discretization.velocity_set.dimension
        physical_velocity = jnp.asarray(velocity, dtype=dtype)
        if physical_velocity.shape == (dimension,):
            physical_velocity = jnp.broadcast_to(physical_velocity, (*shape, dimension))
        if physical_velocity.shape != (*shape, dimension):
            raise ValueError(
                "Initial velocity must be one vector or one vector per cell."
            )
        wall, mask, wetting, wetting_valid = self._wetting_data(parameters_, dtype)
        bulk, gradient, coefficient_valid = self._safe_coefficients(parameters_, dtype)
        phase_fields = free_energy_phase_fields(
            phase_field,
            self.discretization.velocity_set,
            bulk,
            gradient,
            wall_normal=wall,
            wetting_mask=mask,
            wetting_strength=wetting,
        )
        lattice_density = self.scaling.lattice_density(physical_density)
        lattice_velocity = self.scaling.lattice_velocity(physical_velocity)
        safe_density = jnp.maximum(lattice_density, self.method.density_floor)
        raw_velocity = (
            lattice_velocity
            - 0.5 * phase_fields.chemical_force_density / safe_density[..., None]
        )
        hydrodynamic = quadratic_equilibrium(
            lattice_density,
            raw_velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        phase_populations = phase_field_equilibrium(
            phase_field,
            phase_fields.chemical_potential,
            lattice_velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        fluid = self.boundary.geometry.fluid_mask
        solid_hydrodynamic = quadratic_equilibrium(
            jnp.ones_like(lattice_density),
            jnp.zeros_like(lattice_velocity),
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        hydrodynamic = jnp.where(fluid[..., None], hydrodynamic, solid_hydrodynamic)
        phase_populations = jnp.where(
            fluid[..., None], phase_populations, jnp.zeros_like(phase_populations)
        )
        state = FreeEnergyLBMState(
            self.discretization.precision.population(hydrodynamic),
            self.discretization.precision.population(phase_populations),
        )
        interface_cells = jnp.sqrt(2.0 * gradient / bulk)
        valid = (
            coefficient_valid
            & wetting_valid
            & jnp.all(jnp.isfinite(hydrodynamic))
            & jnp.all(jnp.isfinite(phase_populations))
            & jnp.all((~fluid) | (lattice_density > self.method.density_floor))
            & jnp.all(
                (~fluid) | (jnp.abs(phase_field) <= self.method.maximum_absolute_phase)
            )
            & (interface_cells >= self.method.minimum_interface_cells)
        )
        checked = eqx.error_if(
            state.hydrodynamic_populations,
            ~valid,
            "Initial free-energy state is not finite, resolved, and admissible.",
        )
        return FreeEnergyLBMState(checked, state.phase_populations)

    def macroscopic_state(
        self,
        state: FreeEnergyLBMState,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyMacroscopicState:
        values = self._validate_state(state)
        fields, _ = self._fields(values, self._parameters(parameters))
        return FreeEnergyMacroscopicState(
            self.scaling.physical_density(fields.density),
            self.scaling.physical_velocity(fields.velocity),
            self.scaling.physical_pressure(fields.density),
            fields.phase_fields,
            self._ledger(fields),
        )

    def _physical_surface_tension(self, lattice_tension: Array, /) -> Array:
        dtype = lattice_tension.dtype
        return (
            lattice_tension
            * self.scaling.reference_density.astype(dtype)
            * self.scaling.cell_size.astype(dtype) ** 3
            / self.scaling.time_step.astype(dtype) ** 2
        )

    def _diagnostics(
        self,
        fields: _FreeEnergyFields,
        ledger: FreeEnergyLedger,
        mixture_defect: Array,
        phase_defect: Array,
        energy_change: Array,
        equilibrium_mass_defect: Array,
        equilibrium_flux_defect: Array,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyDiagnostics:
        fluid = self.boundary.geometry.fluid_mask
        speed = jnp.sqrt(oe.contract("...d,...d->...", fields.velocity, fields.velocity))
        cs = jnp.sqrt(
            jnp.asarray(
                self.discretization.velocity_set.sound_speed_squared,
                dtype=speed.dtype,
            )
        )
        bulk, gradient, _ = self._safe_coefficients(parameters, speed.dtype)
        interface_cells = jnp.sqrt(2.0 * gradient / bulk)
        physical_tension = self._physical_surface_tension(
            free_energy_surface_tension(bulk, gradient)
        )
        capillary = (
            self.scaling.physical_density(fields.density)
            * jnp.asarray(parameters.kinematic_viscosity, dtype=speed.dtype)
            * self.scaling.physical_velocity(speed)
            / physical_tension
        )
        return FreeEnergyDiagnostics(
            ledger,
            mixture_defect,
            phase_defect,
            energy_change,
            jnp.min(jnp.where(fluid, fields.density, jnp.inf)),
            jnp.max(jnp.where(fluid, jnp.abs(fields.phase_fields.phase), 0.0)),
            jnp.max(jnp.where(fluid, speed / cs, 0.0)),
            jnp.max(jnp.where(fluid, capillary, 0.0)),
            interface_cells,
            equilibrium_mass_defect,
            equilibrium_flux_defect,
        )

    def scalar_diagnostics(
        self,
        step_index: Array,
        time: Array,
        state: FreeEnergyLBMState,
        parameters: FreeEnergyLBMRuntimeParameters,
        /,
    ) -> FreeEnergyDiagnostics:
        del step_index, time
        values = self._validate_state(state)
        parameters_ = self._parameters(parameters)
        fields, _ = self._fields(values, parameters_)
        zero = jnp.zeros((), dtype=values.hydrodynamic_populations.dtype)
        return self._diagnostics(
            fields, self._ledger(fields), zero, zero, zero, zero, zero, parameters_
        )

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: FreeEnergyLBMState,
        step_size: Array,
        args: Any,
        /,
    ) -> FreeEnergyStepResult:
        del step_index, time
        values = self._validate_state(state)
        parameters = self._parameters(args)
        dtype = values.hydrodynamic_populations.dtype
        dt = jnp.asarray(step_size, dtype=dtype)
        expected_dt = jnp.asarray(self.scaling.time_step, dtype=dtype)
        fields, fields_valid = self._fields(values, parameters)
        previous_ledger = self._ledger(fields)
        fluid = self.boundary.geometry.fluid_mask
        viscosity = jnp.asarray(parameters.kinematic_viscosity, dtype=dtype)
        mobility = jnp.asarray(parameters.phase_mobility, dtype=dtype)
        viscosity_valid = jnp.isfinite(viscosity) & (viscosity > 0.0)
        mobility_valid = jnp.isfinite(mobility) & (mobility > 0.0)
        hydro_rate = self.scaling.relaxation_rate(
            jnp.where(viscosity_valid, viscosity, 1.0)
        )
        phase_rate = phase_relaxation_rate(jnp.where(mobility_valid, mobility, 1.0))
        collision_result = self.hydrodynamic_method.collide(
            self.discretization.precision.compute(values.hydrodynamic_populations),
            fields.density,
            fields.velocity,
            fields.phase_fields.chemical_force_density,
            hydro_rate,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        hydrodynamic_post = collision_result.candidate_populations
        phase_equilibrium = phase_field_equilibrium(
            fields.phase_fields.phase,
            fields.phase_fields.chemical_potential,
            fields.velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        phase_post = values.phase_populations - phase_rate * (
            values.phase_populations - phase_equilibrium
        )
        hydrodynamic_post = jnp.where(
            fluid[..., None], hydrodynamic_post, values.hydrodynamic_populations
        )
        phase_post = jnp.where(fluid[..., None], phase_post, values.phase_populations)
        equilibrium_moments = phase_population_moments(
            phase_equilibrium,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        equilibrium_mass_defect = jnp.max(
            jnp.abs(equilibrium_moments.phase - fields.phase_fields.phase)
        )
        equilibrium_flux_defect = jnp.max(
            jnp.abs(
                equilibrium_moments.phase_flux
                - fields.phase_fields.phase[..., None] * fields.velocity
            )
        )
        wall_velocity = jnp.asarray(parameters.moving_wall_velocities, dtype=dtype)
        wall_valid = jnp.all(jnp.isfinite(wall_velocity))
        safe_wall = jnp.where(jnp.isfinite(wall_velocity), wall_velocity, 0.0)
        hydrodynamic_candidate = self.boundary.route(
            self.discretization.precision.population(hydrodynamic_post),
            fields.density,
            self.scaling.lattice_velocity(safe_wall),
        )
        phase_candidate = self.boundary.route(
            self.discretization.precision.population(phase_post),
            jnp.zeros_like(fields.density),
            jnp.zeros(
                (
                    self.boundary.moving_face_count,
                    self.discretization.velocity_set.dimension,
                ),
                dtype=dtype,
            ),
        )
        candidate = FreeEnergyLBMState(
            self.discretization.precision.population(hydrodynamic_candidate),
            self.discretization.precision.population(phase_candidate),
        )
        candidate_fields, candidate_fields_valid = self._fields(candidate, parameters)
        candidate_ledger = self._ledger(candidate_fields)
        mixture_defect = jnp.abs(
            candidate_ledger.mixture_mass - previous_ledger.mixture_mass
        ) / jnp.maximum(jnp.abs(previous_ledger.mixture_mass), 1.0)
        phase_defect = jnp.abs(
            candidate_ledger.phase_mass - previous_ledger.phase_mass
        ) / jnp.maximum(jnp.abs(previous_ledger.phase_mass), 1.0)
        energy_change = candidate_ledger.total_energy - previous_ledger.total_energy
        provisional = self._diagnostics(
            candidate_fields,
            candidate_ledger,
            mixture_defect,
            phase_defect,
            energy_change,
            equilibrium_mass_defect,
            equilibrium_flux_defect,
            parameters,
        )
        mass_tolerance = jnp.asarray(self.method.conservation_tolerance, dtype=dtype)
        energy_tolerance = jnp.asarray(
            self.method.relative_energy_tolerance, dtype=dtype
        ) * jnp.maximum(jnp.abs(previous_ledger.total_energy), 1.0)
        successful = (
            collision_result.successful
            & jnp.isclose(dt, expected_dt, rtol=1.0e-12, atol=1.0e-12)
            & viscosity_valid
            & mobility_valid
            & fields_valid
            & candidate_fields_valid
            & wall_valid
            & jnp.all(jnp.isfinite(hydrodynamic_candidate))
            & jnp.all(jnp.isfinite(phase_candidate))
            & jnp.all((~fluid) | (candidate_fields.density > self.method.density_floor))
            & (provisional.maximum_absolute_phase <= self.method.maximum_absolute_phase)
            & (provisional.interface_cells >= self.method.minimum_interface_cells)
            & (provisional.maximum_mach <= self.method.maximum_mach)
            & (
                provisional.maximum_capillary_number
                <= self.method.maximum_capillary_number
            )
            & (mixture_defect <= mass_tolerance)
            & (phase_defect <= mass_tolerance)
            & (equilibrium_mass_defect <= mass_tolerance)
            & (equilibrium_flux_defect <= mass_tolerance)
            & (energy_change <= energy_tolerance)
        )
        accepted = FreeEnergyLBMState(
            jnp.where(
                successful,
                candidate.hydrodynamic_populations,
                values.hydrodynamic_populations,
            ),
            jnp.where(successful, candidate.phase_populations, values.phase_populations),
        )
        accepted_fields, _ = self._fields(accepted, parameters)
        accepted_ledger = self._ledger(accepted_fields)
        diagnostics = self._diagnostics(
            accepted_fields,
            accepted_ledger,
            jnp.where(successful, mixture_defect, 0.0),
            jnp.where(successful, phase_defect, 0.0),
            jnp.where(successful, energy_change, 0.0),
            jnp.where(successful, equilibrium_mass_defect, 0.0),
            jnp.where(successful, equilibrium_flux_defect, 0.0),
            parameters,
        )
        residual = jnp.maximum(
            jnp.maximum(mixture_defect, phase_defect),
            jnp.maximum(
                equilibrium_mass_defect,
                jnp.maximum(equilibrium_flux_defect, jnp.maximum(energy_change, 0.0)),
            ),
        )
        work = jnp.asarray(
            2
            * self.boundary.geometry.fluid_count
            * self.discretization.velocity_set.population_count,
            dtype=jnp.int32,
        )
        return FreeEnergyStepResult(
            candidate,
            accepted,
            successful,
            residual,
            work,
            diagnostics,
        )


__all__ = [
    "FreeEnergyDiagnostics",
    "FreeEnergyLBMMethod",
    "FreeEnergyLBMRuntimeParameters",
    "FreeEnergyLBMState",
    "FreeEnergyLedger",
    "FreeEnergyMacroscopicState",
    "FreeEnergyPhaseFields",
    "FreeEnergyStepResult",
    "PhasePopulationMoments",
    "PreparedFreeEnergyLBMDynamics",
    "free_energy_phase_fields",
    "free_energy_surface_tension",
    "phase_field_equilibrium",
    "phase_population_moments",
    "phase_relaxation_rate",
]
