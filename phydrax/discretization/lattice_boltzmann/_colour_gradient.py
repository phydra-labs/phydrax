#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import PreparedLatticeBoltzmannBoundary
from ._collision import macroscopic_raw_moments, quadratic_equilibrium
from ._discretization import LatticeBoltzmannDiscretization
from ._interfacial import continuum_surface_force, InterfacialFields
from ._method import (
    LatticeBoltzmannMethodPlan,
    PreparedLatticeBoltzmannMethodPlan,
)
from ._program import coupled_population_manifest, KineticProgramManifest
from ._scaling import LatticeBoltzmannScaling


class ColourGradientLBMState(StrictModule):
    """Binary component populations, each with a trailing population axis."""

    red_populations: Array
    blue_populations: Array


class ColourGradientLBMRuntimeParameters(StrictModule):
    """Differentiable physical and wetting controls for one fixed-step rollout."""

    kinematic_viscosity: Array
    surface_tension: Array
    moving_wall_velocities: Array
    wall_normal: Array
    wetting_mask: Array
    contact_angle: Array

    def __init__(
        self,
        kinematic_viscosity: ArrayLike,
        surface_tension: ArrayLike,
        /,
        *,
        moving_wall_velocities: ArrayLike | None = None,
        wall_normal: ArrayLike | None = None,
        wetting_mask: ArrayLike | None = None,
        contact_angle: ArrayLike = 0.5 * jnp.pi,
    ):
        viscosity = jnp.asarray(kinematic_viscosity)
        if viscosity.shape != () or not jnp.issubdtype(viscosity.dtype, jnp.inexact):
            raise ValueError("kinematic_viscosity must be one inexact scalar array.")
        tension = jnp.asarray(surface_tension, dtype=viscosity.dtype)
        angle = jnp.asarray(contact_angle, dtype=viscosity.dtype)
        if tension.shape != () or angle.shape != ():
            raise ValueError("surface_tension and contact_angle must be scalar.")
        if (wall_normal is None) != (wetting_mask is None):
            raise ValueError("wall_normal and wetting_mask must be supplied together.")
        walls = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if moving_wall_velocities is None
            else jnp.asarray(moving_wall_velocities, dtype=viscosity.dtype)
        )
        normals = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if wall_normal is None
            else jnp.asarray(wall_normal, dtype=viscosity.dtype)
        )
        mask = (
            jnp.empty((0,), dtype=bool)
            if wetting_mask is None
            else jnp.asarray(wetting_mask, dtype=bool)
        )
        self.kinematic_viscosity = viscosity
        self.surface_tension = tension
        self.moving_wall_velocities = walls
        self.wall_normal = normals
        self.wetting_mask = mask
        self.contact_angle = angle


class ColourGradientLBMMethod(StrictModule, NonTrainableState):
    """Conservative binary recolouring layered over one forced LBM method."""

    hydrodynamic_method: LatticeBoltzmannMethodPlan
    recolouring_strength: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    gradient_floor: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    maximum_capillary_number: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_method: LatticeBoltzmannMethodPlan,
        /,
        *,
        recolouring_strength: float = 0.7,
        density_floor: float = 1.0e-12,
        gradient_floor: float = 1.0e-14,
        maximum_mach: float = 0.3,
        maximum_capillary_number: float = 1.0,
        conservation_tolerance: float = 1.0e-11,
    ):
        if not isinstance(hydrodynamic_method, LatticeBoltzmannMethodPlan):
            raise TypeError("hydrodynamic_method must be LatticeBoltzmannMethodPlan.")
        if hydrodynamic_method.forcing is None:
            raise ValueError("Colour-gradient CSF requires a forced LBM method.")
        values = tuple(
            float(value)
            for value in (
                recolouring_strength,
                density_floor,
                gradient_floor,
                maximum_mach,
                maximum_capillary_number,
                conservation_tolerance,
            )
        )
        beta, rho_floor, grad_floor, mach, capillary, tolerance = values
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Colour-gradient method limits must be finite and positive.")
        if beta > 1.0:
            raise ValueError("recolouring_strength must lie in (0, 1].")
        if mach >= 1.0:
            raise ValueError("maximum_mach must be smaller than one.")
        self.hydrodynamic_method = hydrodynamic_method
        self.recolouring_strength = beta
        self.density_floor = rho_floor
        self.gradient_floor = grad_floor
        self.maximum_mach = mach
        self.maximum_capillary_number = capillary
        self.conservation_tolerance = tolerance
        self.method_id = canonical_fingerprint(
            {
                "kind": "colour-gradient-lattice-boltzmann-method",
                "hydrodynamic_method": hydrodynamic_method.method_id,
                "recolouring_strength": beta,
                "density_floor": rho_floor,
                "gradient_floor": grad_floor,
                "maximum_mach": mach,
                "maximum_capillary_number": capillary,
                "conservation_tolerance": tolerance,
            }
        )


class ColourGradientMacroscopicState(StrictModule):
    red_density: Array
    blue_density: Array
    density: Array
    colour: Array
    velocity: Array
    pressure: Array
    interfacial: InterfacialFields


class RecolouringConservation(StrictModule):
    red_mass_defect: Array
    blue_mass_defect: Array
    population_closure_defect: Array
    momentum_closure_defect: Array


class ColourGradientDiagnostics(StrictModule):
    red_mass: Array
    blue_mass: Array
    total_mass: Array
    red_mass_defect: Array
    blue_mass_defect: Array
    total_mass_defect: Array
    minimum_component_density: Array
    minimum_density: Array
    maximum_mach: Array
    maximum_capillary_number: Array
    force_norm: Array
    recolouring: RecolouringConservation


class ColourGradientStepResult(StrictModule):
    candidate_state: ColourGradientLBMState
    accepted_state: ColourGradientLBMState
    successful: Array
    residual: Array
    work: Array
    diagnostics: ColourGradientDiagnostics


class _ColourGradientFields(StrictModule):
    red_density: Array
    blue_density: Array
    density: Array
    colour: Array
    raw_momentum: Array
    velocity: Array
    interfacial: InterfacialFields


def recolour_populations(
    total_populations: ArrayLike,
    red_density: ArrayLike,
    blue_density: ArrayLike,
    interface_normal: ArrayLike,
    velocity_set,
    recolouring_strength: ArrayLike,
    /,
    *,
    density_floor: ArrayLike = 1.0e-14,
) -> ColourGradientLBMState:
    """Conservatively split a mixture population into red and blue populations.

    The split preserves both component zeroth moments and the complete mixture
    population direction by direction.  Consequently every mixture moment,
    including momentum, is unchanged by recolouring.
    """

    populations = jnp.asarray(total_populations)
    red = jnp.asarray(red_density, dtype=populations.dtype)
    blue = jnp.asarray(blue_density, dtype=populations.dtype)
    normal = jnp.asarray(interface_normal, dtype=populations.dtype)
    expected = (*red.shape, velocity_set.population_count)
    if populations.shape != expected or blue.shape != red.shape:
        raise ValueError("Population and component-density shapes are incompatible.")
    if normal.shape != (*red.shape, velocity_set.dimension):
        raise ValueError("interface_normal has an incompatible shape.")
    beta = jnp.asarray(recolouring_strength, dtype=populations.dtype)
    floor = jnp.asarray(density_floor, dtype=populations.dtype)
    if beta.shape != () or floor.shape != ():
        raise ValueError("Recolouring coefficients must be scalar.")
    beta = eqx.error_if(
        beta,
        ~jnp.isfinite(beta) | (beta <= 0.0) | (beta > 1.0),
        "recolouring_strength must lie in (0, 1].",
    )
    floor = eqx.error_if(
        floor,
        ~jnp.isfinite(floor) | (floor <= 0.0),
        "density_floor must be finite and positive.",
    )
    density = red + blue
    safe_density = jnp.maximum(density, floor)
    velocities = jnp.asarray(velocity_set.velocities, dtype=populations.dtype)
    speed = jnp.sqrt(ein.contract("qd,qd->q", velocities, velocities))
    direction = velocities / jnp.where(speed > 0.0, speed, 1.0)[:, None]
    cosine = ein.contract("...d,qd->...q", normal, direction)
    weights = jnp.asarray(velocity_set.weights, dtype=populations.dtype)
    segregation = beta * (red * blue / safe_density)[..., None] * weights * cosine
    red_fraction = red / safe_density
    blue_fraction = blue / safe_density
    red_populations = red_fraction[..., None] * populations + segregation
    blue_populations = blue_fraction[..., None] * populations - segregation
    occupied = density > floor
    red_populations = jnp.where(occupied[..., None], red_populations, 0.0)
    blue_populations = jnp.where(occupied[..., None], blue_populations, 0.0)
    return ColourGradientLBMState(red_populations, blue_populations)


class PreparedColourGradientLBMDynamics(StrictModule, NonTrainableState):
    """Pure matched-density colour-gradient collide, recolour, and route dynamics."""

    discretization: LatticeBoltzmannDiscretization
    scaling: LatticeBoltzmannScaling
    method: ColourGradientLBMMethod
    hydrodynamic_method: PreparedLatticeBoltzmannMethodPlan
    program_manifest: KineticProgramManifest
    boundary: PreparedLatticeBoltzmannBoundary
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        scaling: LatticeBoltzmannScaling,
        method: ColourGradientLBMMethod,
        boundary: PreparedLatticeBoltzmannBoundary,
        /,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be an LBM discretization.")
        if not isinstance(scaling, LatticeBoltzmannScaling):
            raise TypeError("scaling must be LatticeBoltzmannScaling.")
        if not isinstance(method, ColourGradientLBMMethod):
            raise TypeError("method must be ColourGradientLBMMethod.")
        if not isinstance(boundary, PreparedLatticeBoltzmannBoundary):
            raise TypeError("boundary must be a prepared LBM boundary.")
        if boundary.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Boundary and colour-gradient discretizations do not match.")
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
        program_manifest = coupled_population_manifest(
            "colour_gradient_lattice_boltzmann",
            discretization.velocity_set.lattice_id,
            discretization.precision.policy_id,
            discretization.velocity_set.population_count,
            discretization.velocity_set.dimension,
            ("red_populations", "blue_populations"),
            (("red_mass", "momentum"), ("blue_mass", "momentum")),
        )
        self.discretization = discretization
        self.scaling = scaling
        self.method = method
        self.hydrodynamic_method = hydrodynamic_method
        self.program_manifest = program_manifest
        self.boundary = boundary
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-colour-gradient-lattice-boltzmann-dynamics",
                "discretization": discretization.prepared_id,
                "scaling": scaling.scaling_id,
                "method": method.method_id,
                "prepared_hydrodynamic_method": hydrodynamic_method.method_id,
                "program_manifest": program_manifest.manifest_id,
                "boundary": boundary.boundary_id,
            }
        )

    def _parameters(self, args: Any, /) -> ColourGradientLBMRuntimeParameters:
        if not isinstance(args, ColourGradientLBMRuntimeParameters):
            raise TypeError(
                "Colour-gradient fixed-step args must be ColourGradientLBMRuntimeParameters."
            )
        return args

    def _validate_state(self, state: ColourGradientLBMState, /) -> ColourGradientLBMState:
        if not isinstance(state, ColourGradientLBMState):
            raise TypeError("state must be ColourGradientLBMState.")
        red = self.discretization.validate_populations(state.red_populations)
        blue = self.discretization.validate_populations(state.blue_populations)
        return ColourGradientLBMState(red, blue)

    def _wetting_data(
        self, parameters: ColourGradientLBMRuntimeParameters, dtype, /
    ) -> tuple[Array | None, Array | None, Array, Array]:
        shape = self.discretization.grid.shape
        dimension = self.discretization.velocity_set.dimension
        angle = jnp.asarray(parameters.contact_angle, dtype=dtype)
        angle_valid = jnp.isfinite(angle) & (angle > 0.0) & (angle < jnp.pi)
        safe_angle = jnp.where(angle_valid, angle, 0.5 * jnp.pi)
        if parameters.wetting_mask.size == 0:
            return None, None, safe_angle, angle_valid
        if parameters.wetting_mask.shape != shape:
            raise ValueError("wetting_mask must match the lattice grid shape.")
        if parameters.wall_normal.shape != (*shape, dimension):
            raise ValueError("wall_normal must contain one vector per lattice cell.")
        mask = parameters.wetting_mask
        wall = jnp.asarray(parameters.wall_normal, dtype=dtype)
        norm = jnp.sqrt(ein.contract("...d,...d->...", wall, wall))
        normal_valid = jnp.all(jnp.isfinite(wall), axis=-1) & (norm > 0.0)
        fallback = jnp.zeros_like(wall).at[..., 0].set(1.0)
        safe_wall = jnp.where((~mask | normal_valid)[..., None], wall, fallback)
        valid = angle_valid & jnp.all(~mask | normal_valid)
        return safe_wall, mask, safe_angle, valid

    def _lattice_surface_tension(self, physical: Array, /) -> Array:
        dtype = physical.dtype
        dt = self.scaling.time_step.astype(dtype)
        dx = self.scaling.cell_size.astype(dtype)
        rho0 = self.scaling.reference_density.astype(dtype)
        return physical * dt**2 / (rho0 * dx**3)

    def _fields(
        self,
        state: ColourGradientLBMState,
        parameters: ColourGradientLBMRuntimeParameters,
        /,
    ) -> tuple[_ColourGradientFields, Array]:
        red_density, red_momentum = macroscopic_raw_moments(
            state.red_populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        blue_density, blue_momentum = macroscopic_raw_moments(
            state.blue_populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        density = red_density + blue_density
        safe_density = jnp.maximum(
            density,
            jnp.asarray(self.method.density_floor, dtype=density.dtype),
        )
        colour = (red_density - blue_density) / safe_density
        wall, mask, angle, wetting_valid = self._wetting_data(
            parameters, state.red_populations.dtype
        )
        physical_tension = jnp.asarray(parameters.surface_tension, dtype=density.dtype)
        tension_valid = jnp.isfinite(physical_tension) & (physical_tension >= 0.0)
        safe_tension = jnp.where(tension_valid, physical_tension, 0.0)
        interfacial = continuum_surface_force(
            colour,
            self.discretization.velocity_set,
            self._lattice_surface_tension(safe_tension),
            1.0,
            wall_normal=wall,
            wetting_mask=mask,
            contact_angle=angle,
            epsilon=self.method.gradient_floor,
        )
        raw_momentum = red_momentum + blue_momentum
        velocity = (raw_momentum + 0.5 * interfacial.force_density) / safe_density[
            ..., None
        ]
        fields = _ColourGradientFields(
            red_density,
            blue_density,
            density,
            colour,
            raw_momentum,
            velocity,
            interfacial,
        )
        return fields, tension_valid & wetting_valid

    def initialize_state(
        self,
        red_density: ArrayLike,
        blue_density: ArrayLike,
        velocity: ArrayLike,
        parameters: ColourGradientLBMRuntimeParameters,
        /,
    ) -> ColourGradientLBMState:
        parameters_ = self._parameters(parameters)
        dtype = jnp.dtype(self.discretization.precision.population_dtype)
        shape = self.discretization.grid.shape
        red_physical = jnp.asarray(red_density, dtype=dtype)
        blue_physical = jnp.asarray(blue_density, dtype=dtype)
        if red_physical.shape == ():
            red_physical = jnp.broadcast_to(red_physical, shape)
        if blue_physical.shape == ():
            blue_physical = jnp.broadcast_to(blue_physical, shape)
        if red_physical.shape != shape or blue_physical.shape != shape:
            raise ValueError(
                "Initial component densities must be scalar or match the grid."
            )
        dimension = self.discretization.velocity_set.dimension
        physical_velocity = jnp.asarray(velocity, dtype=dtype)
        if physical_velocity.shape == (dimension,):
            physical_velocity = jnp.broadcast_to(physical_velocity, (*shape, dimension))
        if physical_velocity.shape != (*shape, dimension):
            raise ValueError(
                "Initial velocity must be one vector or one vector per cell."
            )
        red_lattice = self.scaling.lattice_density(red_physical)
        blue_lattice = self.scaling.lattice_density(blue_physical)
        density = red_lattice + blue_lattice
        safe_density = jnp.maximum(density, self.method.density_floor)
        colour = (red_lattice - blue_lattice) / safe_density
        wall, mask, angle, wetting_valid = self._wetting_data(parameters_, dtype)
        tension = jnp.asarray(parameters_.surface_tension, dtype=dtype)
        tension_valid = jnp.isfinite(tension) & (tension >= 0.0)
        safe_tension = jnp.where(tension_valid, tension, 0.0)
        interfacial = continuum_surface_force(
            colour,
            self.discretization.velocity_set,
            self._lattice_surface_tension(safe_tension),
            1.0,
            wall_normal=wall,
            wetting_mask=mask,
            contact_angle=angle,
            epsilon=self.method.gradient_floor,
        )
        lattice_velocity = self.scaling.lattice_velocity(physical_velocity)
        raw_velocity = (
            lattice_velocity - 0.5 * interfacial.force_density / safe_density[..., None]
        )
        total = quadratic_equilibrium(
            density,
            raw_velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        initial = recolour_populations(
            total,
            red_lattice,
            blue_lattice,
            interfacial.normal,
            self.discretization.velocity_set,
            self.method.recolouring_strength,
            density_floor=self.method.density_floor,
        )
        fluid = self.boundary.geometry.fluid_mask
        solid_total = quadratic_equilibrium(
            jnp.ones_like(density),
            jnp.zeros_like(lattice_velocity),
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        red_populations = jnp.where(
            fluid[..., None], initial.red_populations, 0.5 * solid_total
        )
        blue_populations = jnp.where(
            fluid[..., None], initial.blue_populations, 0.5 * solid_total
        )
        state = ColourGradientLBMState(
            self.discretization.precision.population(red_populations),
            self.discretization.precision.population(blue_populations),
        )
        valid = (
            tension_valid
            & wetting_valid
            & jnp.all(jnp.isfinite(red_populations))
            & jnp.all(jnp.isfinite(blue_populations))
            & jnp.all((~fluid) | (red_lattice >= 0.0))
            & jnp.all((~fluid) | (blue_lattice >= 0.0))
            & jnp.all((~fluid) | (density > self.method.density_floor))
        )
        red_checked = eqx.error_if(
            state.red_populations,
            ~valid,
            "Initial colour-gradient state is not finite and admissible.",
        )
        return ColourGradientLBMState(red_checked, state.blue_populations)

    def macroscopic_state(
        self,
        state: ColourGradientLBMState,
        parameters: ColourGradientLBMRuntimeParameters,
        /,
    ) -> ColourGradientMacroscopicState:
        values = self._validate_state(state)
        fields, _ = self._fields(values, self._parameters(parameters))
        return ColourGradientMacroscopicState(
            self.scaling.physical_density(fields.red_density),
            self.scaling.physical_density(fields.blue_density),
            self.scaling.physical_density(fields.density),
            fields.colour,
            self.scaling.physical_velocity(fields.velocity),
            self.scaling.physical_pressure(fields.density),
            fields.interfacial,
        )

    def _recolouring_conservation(
        self,
        recoloured: ColourGradientLBMState,
        total: Array,
        red_density: Array,
        blue_density: Array,
        /,
    ) -> RecolouringConservation:
        red_moment = jnp.sum(recoloured.red_populations, axis=-1)
        blue_moment = jnp.sum(recoloured.blue_populations, axis=-1)
        closure = recoloured.red_populations + recoloured.blue_populations - total
        velocities = jnp.asarray(
            self.discretization.velocity_set.velocities, dtype=total.dtype
        )
        momentum_closure = ein.contract("...q,qd->...d", closure, velocities)
        return RecolouringConservation(
            jnp.max(jnp.abs(red_moment - red_density)),
            jnp.max(jnp.abs(blue_moment - blue_density)),
            jnp.max(jnp.abs(closure)),
            jnp.max(jnp.abs(momentum_closure)),
        )

    def _diagnostics(
        self,
        fields: _ColourGradientFields,
        red_defect: Array,
        blue_defect: Array,
        total_defect: Array,
        conservation: RecolouringConservation,
        parameters: ColourGradientLBMRuntimeParameters,
        /,
    ) -> ColourGradientDiagnostics:
        fluid = self.boundary.geometry.fluid_mask
        speed = jnp.sqrt(ein.contract("...d,...d->...", fields.velocity, fields.velocity))
        cs = jnp.sqrt(
            jnp.asarray(
                self.discretization.velocity_set.sound_speed_squared,
                dtype=speed.dtype,
            )
        )
        physical_speed = self.scaling.physical_velocity(speed)
        physical_density = self.scaling.physical_density(fields.density)
        tension = jnp.asarray(parameters.surface_tension, dtype=speed.dtype)
        capillary = jnp.where(
            tension > 0.0,
            physical_density
            * jnp.asarray(parameters.kinematic_viscosity, dtype=speed.dtype)
            * physical_speed
            / tension,
            0.0,
        )
        return ColourGradientDiagnostics(
            red_mass=jnp.sum(jnp.where(fluid, fields.red_density, 0.0)),
            blue_mass=jnp.sum(jnp.where(fluid, fields.blue_density, 0.0)),
            total_mass=jnp.sum(jnp.where(fluid, fields.density, 0.0)),
            red_mass_defect=red_defect,
            blue_mass_defect=blue_defect,
            total_mass_defect=total_defect,
            minimum_component_density=jnp.min(
                jnp.where(
                    fluid,
                    jnp.minimum(fields.red_density, fields.blue_density),
                    jnp.inf,
                )
            ),
            minimum_density=jnp.min(jnp.where(fluid, fields.density, jnp.inf)),
            maximum_mach=jnp.max(jnp.where(fluid, speed / cs, 0.0)),
            maximum_capillary_number=jnp.max(jnp.where(fluid, capillary, 0.0)),
            force_norm=jnp.sqrt(
                jnp.sum(
                    jnp.where(fluid[..., None], fields.interfacial.force_density, 0.0)
                    ** 2
                )
            ),
            recolouring=conservation,
        )

    def scalar_diagnostics(
        self,
        step_index: Array,
        time: Array,
        state: ColourGradientLBMState,
        parameters: ColourGradientLBMRuntimeParameters,
        /,
    ) -> ColourGradientDiagnostics:
        del step_index, time
        values = self._validate_state(state)
        parameters_ = self._parameters(parameters)
        fields, _ = self._fields(values, parameters_)
        zero = jnp.zeros((), dtype=values.red_populations.dtype)
        conservation = RecolouringConservation(zero, zero, zero, zero)
        return self._diagnostics(fields, zero, zero, zero, conservation, parameters_)

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: ColourGradientLBMState,
        step_size: Array,
        args: Any,
        /,
    ) -> ColourGradientStepResult:
        del step_index, time
        values = self._validate_state(state)
        parameters = self._parameters(args)
        dtype = values.red_populations.dtype
        dt = jnp.asarray(step_size, dtype=dtype)
        expected_dt = jnp.asarray(self.scaling.time_step, dtype=dtype)
        fields, interfacial_valid = self._fields(values, parameters)
        fluid = self.boundary.geometry.fluid_mask
        viscosity = jnp.asarray(parameters.kinematic_viscosity, dtype=dtype)
        viscosity_valid = jnp.isfinite(viscosity) & (viscosity > 0.0)
        safe_viscosity = jnp.where(viscosity_valid, viscosity, 1.0)
        even_rate = self.scaling.relaxation_rate(safe_viscosity)
        total = values.red_populations + values.blue_populations
        collision_result = self.hydrodynamic_method.collide(
            self.discretization.precision.compute(total),
            fields.density,
            fields.velocity,
            fields.interfacial.force_density,
            even_rate,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        post_collision = jnp.where(
            fluid[..., None], collision_result.candidate_populations, total
        )
        post_density = jnp.sum(post_collision, axis=-1)
        red_fraction = fields.red_density / jnp.maximum(
            fields.density, self.method.density_floor
        )
        target_red = red_fraction * post_density
        target_blue = post_density - target_red
        recoloured = recolour_populations(
            post_collision,
            target_red,
            target_blue,
            fields.interfacial.normal,
            self.discretization.velocity_set,
            self.method.recolouring_strength,
            density_floor=self.method.density_floor,
        )
        conservation = self._recolouring_conservation(
            recoloured, post_collision, target_red, target_blue
        )
        wall_velocity = jnp.asarray(parameters.moving_wall_velocities, dtype=dtype)
        wall_valid = jnp.all(jnp.isfinite(wall_velocity))
        safe_wall = jnp.where(jnp.isfinite(wall_velocity), wall_velocity, 0.0)
        lattice_wall = self.scaling.lattice_velocity(safe_wall)
        red_candidate = self.boundary.route(
            self.discretization.precision.population(recoloured.red_populations),
            target_red,
            lattice_wall,
        )
        blue_candidate = self.boundary.route(
            self.discretization.precision.population(recoloured.blue_populations),
            target_blue,
            lattice_wall,
        )
        candidate = ColourGradientLBMState(
            self.discretization.precision.population(red_candidate),
            self.discretization.precision.population(blue_candidate),
        )
        candidate_fields, candidate_valid = self._fields(candidate, parameters)
        previous_red = jnp.sum(jnp.where(fluid, fields.red_density, 0.0))
        previous_blue = jnp.sum(jnp.where(fluid, fields.blue_density, 0.0))
        candidate_red = jnp.sum(jnp.where(fluid, candidate_fields.red_density, 0.0))
        candidate_blue = jnp.sum(jnp.where(fluid, candidate_fields.blue_density, 0.0))
        red_defect = jnp.abs(candidate_red - previous_red) / jnp.maximum(
            jnp.abs(previous_red), 1.0
        )
        blue_defect = jnp.abs(candidate_blue - previous_blue) / jnp.maximum(
            jnp.abs(previous_blue), 1.0
        )
        previous_total = previous_red + previous_blue
        candidate_total = candidate_red + candidate_blue
        total_defect = jnp.abs(candidate_total - previous_total) / jnp.maximum(
            jnp.abs(previous_total), 1.0
        )
        provisional = self._diagnostics(
            candidate_fields,
            red_defect,
            blue_defect,
            total_defect,
            conservation,
            parameters,
        )
        tolerance = jnp.asarray(self.method.conservation_tolerance, dtype=dtype)
        conservation_valid = (
            (conservation.red_mass_defect <= tolerance)
            & (conservation.blue_mass_defect <= tolerance)
            & (conservation.population_closure_defect <= tolerance)
            & (conservation.momentum_closure_defect <= tolerance)
        )
        successful = (
            collision_result.successful
            & jnp.isclose(dt, expected_dt, rtol=1.0e-12, atol=1.0e-12)
            & viscosity_valid
            & interfacial_valid
            & candidate_valid
            & wall_valid
            & conservation_valid
            & jnp.all(jnp.isfinite(red_candidate))
            & jnp.all(jnp.isfinite(blue_candidate))
            & jnp.all((~fluid) | (candidate_fields.red_density >= 0.0))
            & jnp.all((~fluid) | (candidate_fields.blue_density >= 0.0))
            & jnp.all((~fluid) | (candidate_fields.density > self.method.density_floor))
            & (provisional.maximum_mach <= self.method.maximum_mach)
            & (
                provisional.maximum_capillary_number
                <= self.method.maximum_capillary_number
            )
        )
        accepted = ColourGradientLBMState(
            jnp.where(successful, candidate.red_populations, values.red_populations),
            jnp.where(successful, candidate.blue_populations, values.blue_populations),
        )
        accepted_fields, _ = self._fields(accepted, parameters)
        diagnostics = self._diagnostics(
            accepted_fields,
            jnp.where(successful, red_defect, 0.0),
            jnp.where(successful, blue_defect, 0.0),
            jnp.where(successful, total_defect, 0.0),
            RecolouringConservation(
                jnp.where(successful, conservation.red_mass_defect, 0.0),
                jnp.where(successful, conservation.blue_mass_defect, 0.0),
                jnp.where(successful, conservation.population_closure_defect, 0.0),
                jnp.where(successful, conservation.momentum_closure_defect, 0.0),
            ),
            parameters,
        )
        residual = jnp.maximum(
            total_defect,
            jnp.maximum(red_defect, blue_defect),
        )
        work = jnp.asarray(
            2
            * self.boundary.geometry.fluid_count
            * self.discretization.velocity_set.population_count,
            dtype=jnp.int32,
        )
        return ColourGradientStepResult(
            candidate,
            accepted,
            successful,
            residual,
            work,
            diagnostics,
        )


__all__ = [
    "ColourGradientDiagnostics",
    "ColourGradientLBMMethod",
    "ColourGradientLBMRuntimeParameters",
    "ColourGradientLBMState",
    "ColourGradientMacroscopicState",
    "ColourGradientStepResult",
    "PreparedColourGradientLBMDynamics",
    "RecolouringConservation",
    "recolour_populations",
]
