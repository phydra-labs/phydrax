#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._boundary import (
    LatticeBoltzmannBoundaryParameters,
    PreparedLatticeBoltzmannBoundary,
    PreparedStagedLatticeBoltzmannBoundary,
)
from ._collision import (
    LatticeBoltzmannCollisionDiagnostics,
    macroscopic_raw_moments,
    quadratic_equilibrium,
)
from ._discretization import LatticeBoltzmannDiscretization
from ._implicit_forcing import (
    LocalRootSolver,
    VelocityDependentAccelerationPlan,
    VelocityDependentAccelerationProblem,
)
from ._method import (
    LatticeBoltzmannMethodPlan,
    PreparedLatticeBoltzmannMethodPlan,
)
from ._scaling import LatticeBoltzmannScaling


LatticeAcceleration: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class LatticeBoltzmannRuntimeParameters(StrictModule):
    """Differentiable physical controls supplied through fixed-step ``args``."""

    local_root_solver: LocalRootSolver | None

    kinematic_viscosity: Array
    force_parameters: Any
    moving_wall_velocities: Array
    boundary_parameters: LatticeBoltzmannBoundaryParameters | None

    def __init__(
        self,
        kinematic_viscosity: ArrayLike,
        /,
        *,
        force_parameters: Any = None,
        moving_wall_velocities: ArrayLike | None = None,
        local_root_solver: LocalRootSolver | None = None,
        boundary_parameters: LatticeBoltzmannBoundaryParameters | None = None,
    ):
        viscosity = jnp.asarray(kinematic_viscosity)
        walls = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if moving_wall_velocities is None
            else jnp.asarray(moving_wall_velocities, dtype=viscosity.dtype)
        )
        if viscosity.shape != () or not jnp.issubdtype(viscosity.dtype, jnp.inexact):
            raise ValueError("kinematic_viscosity must be one inexact scalar array.")
        self.kinematic_viscosity = viscosity
        self.force_parameters = force_parameters
        self.moving_wall_velocities = walls
        self.local_root_solver = local_root_solver
        self.boundary_parameters = boundary_parameters


class LatticeBoltzmannMacroscopicState(StrictModule):
    density: Array
    velocity: Array
    pressure: Array


class LatticeBoltzmannDiagnostics(StrictModule):
    minimum_density: Array
    maximum_mach: Array
    minimum_population: Array
    total_mass: Array
    mass_defect: Array
    momentum_norm: Array
    force_norm: Array


class LatticeBoltzmannStepResult(StrictModule):
    candidate_state: Array
    accepted_state: Array
    successful: Array
    residual: Array
    work: Array
    diagnostics: LatticeBoltzmannDiagnostics
    collision_diagnostics: LatticeBoltzmannCollisionDiagnostics


class _LatticeFields(StrictModule):
    density: Array
    raw_momentum: Array
    velocity: Array
    force_density: Array
    force_successful: Array


class PreparedLatticeBoltzmannDynamics(StrictModule, NonTrainableState):
    """Pure collide-and-route dynamics for one frozen lattice and geometry."""

    discretization: LatticeBoltzmannDiscretization
    scaling: LatticeBoltzmannScaling
    method: PreparedLatticeBoltzmannMethodPlan
    boundary: PreparedLatticeBoltzmannBoundary | PreparedStagedLatticeBoltzmannBoundary
    acceleration: LatticeAcceleration | None
    acceleration_id: str | None = eqx.field(static=True)
    implicit_acceleration: VelocityDependentAccelerationPlan | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        scaling: LatticeBoltzmannScaling,
        method: LatticeBoltzmannMethodPlan | PreparedLatticeBoltzmannMethodPlan,
        boundary: PreparedLatticeBoltzmannBoundary
        | PreparedStagedLatticeBoltzmannBoundary,
        /,
        *,
        acceleration: LatticeAcceleration | None = None,
        acceleration_id: str | None = None,
        implicit_acceleration: VelocityDependentAccelerationPlan | None = None,
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be an LBM discretization.")
        if not isinstance(scaling, LatticeBoltzmannScaling):
            raise TypeError("scaling must be LatticeBoltzmannScaling.")
        if not isinstance(
            method, (LatticeBoltzmannMethodPlan, PreparedLatticeBoltzmannMethodPlan)
        ):
            raise TypeError("method must be a lattice-Boltzmann method plan.")
        prepared_method = (
            method.prepare(discretization.velocity_set, discretization.precision)
            if isinstance(method, LatticeBoltzmannMethodPlan)
            else method
        )
        if (
            prepared_method.collision.lattice_id != discretization.velocity_set.lattice_id
            or prepared_method.collision.precision_policy_id
            != discretization.precision.policy_id
        ):
            raise ValueError("Prepared method and discretization do not match.")
        if not isinstance(
            boundary,
            (PreparedLatticeBoltzmannBoundary, PreparedStagedLatticeBoltzmannBoundary),
        ):
            raise TypeError("boundary must be a prepared LBM boundary.")
        if boundary.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Boundary and dynamics discretizations do not match.")
        if acceleration is not None and not callable(acceleration):
            raise TypeError("acceleration must be callable or None.")
        if implicit_acceleration is not None and not isinstance(
            implicit_acceleration, VelocityDependentAccelerationPlan
        ):
            raise TypeError(
                "implicit_acceleration must be VelocityDependentAccelerationPlan or None."
            )
        if implicit_acceleration is not None and acceleration is not None:
            raise ValueError(
                "Explicit and velocity-dependent acceleration are mutually exclusive."
            )
        if acceleration is None:
            if acceleration_id is not None:
                raise ValueError("acceleration_id requires an acceleration callable.")
            acceleration_identifier = None
        else:
            acceleration_identifier = (
                "" if acceleration_id is None else str(acceleration_id)
            )
            if not acceleration_identifier:
                raise ValueError("Acceleration requires a non-empty acceleration_id.")
        has_acceleration = acceleration is not None or implicit_acceleration is not None
        if (prepared_method.forcing is None) == has_acceleration:
            raise ValueError(
                "GuoForcingPlan and exactly one acceleration model must be supplied together."
            )
        lattice_cs2 = float(discretization.velocity_set.sound_speed_squared)
        if not jnp.isclose(scaling.sound_speed_squared, lattice_cs2):
            raise ValueError("Scaling and velocity-set sound speeds do not match.")
        self.discretization = discretization
        self.scaling = scaling
        self.method = prepared_method
        self.boundary = boundary
        self.acceleration = acceleration
        self.acceleration_id = acceleration_identifier
        self.implicit_acceleration = implicit_acceleration
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-lattice-boltzmann-dynamics",
                "discretization": discretization.prepared_id,
                "scaling": scaling.scaling_id,
                "method": prepared_method.method_id,
                "boundary": boundary.boundary_id,
                "acceleration": acceleration_identifier,
                "implicit_acceleration": (
                    None
                    if implicit_acceleration is None
                    else implicit_acceleration.plan_id
                ),
            }
        )

    @property
    def coordinates(self) -> Array:
        return self.discretization.grid.points.reshape(
            (*self.discretization.grid.shape, self.discretization.velocity_set.dimension)
        )

    def _parameters(self, args: Any, /) -> LatticeBoltzmannRuntimeParameters:
        if not isinstance(args, LatticeBoltzmannRuntimeParameters):
            raise TypeError(
                "LBM fixed-step args must be LatticeBoltzmannRuntimeParameters."
            )
        return args

    def _physical_acceleration(
        self,
        time: Array,
        parameters: LatticeBoltzmannRuntimeParameters,
        dtype: Any,
        /,
    ) -> Array:
        dimension = self.discretization.velocity_set.dimension
        if self.acceleration is None:
            return jnp.zeros((*self.discretization.grid.shape, dimension), dtype=dtype)
        value = jnp.asarray(
            self.acceleration(time, self.coordinates, parameters.force_parameters),
            dtype=dtype,
        )
        if value.shape == (dimension,):
            return jnp.broadcast_to(value, (*self.discretization.grid.shape, dimension))
        expected = (*self.discretization.grid.shape, dimension)
        if value.shape != expected:
            raise ValueError(
                f"Acceleration must have shape {(dimension,)} or {expected}."
            )
        return value

    def _lattice_fields(
        self,
        time: Array,
        populations: Array,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
    ) -> _LatticeFields:
        density, raw_momentum = macroscopic_raw_moments(
            populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        if self.implicit_acceleration is not None:
            if parameters.local_root_solver is None:
                raise ValueError(
                    "Velocity-dependent acceleration requires local_root_solver."
                )
            problem = VelocityDependentAccelerationProblem(
                time,
                self.coordinates,
                density,
                raw_momentum,
                parameters=parameters.force_parameters,
            )
            result = self.implicit_acceleration.solve(
                problem, parameters.local_root_solver
            )
            return _LatticeFields(
                density,
                raw_momentum,
                result.velocity,
                result.force_density,
                jnp.all(result.root.converged),
            )
        acceleration = self._physical_acceleration(time, parameters, populations.dtype)
        lattice_acceleration = self.scaling.lattice_acceleration(acceleration)
        force_density = density[..., None] * lattice_acceleration
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = (raw_momentum + 0.5 * force_density) / safe_density[..., None]
        successful = jnp.all(jnp.isfinite(force_density))
        return _LatticeFields(density, raw_momentum, velocity, force_density, successful)

    def initialize_state(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> Array:
        parameters_ = self._parameters(parameters)
        dtype = jnp.dtype(self.discretization.precision.population_dtype)
        rho_physical = jnp.asarray(density, dtype=dtype)
        if rho_physical.shape == ():
            rho_physical = jnp.broadcast_to(rho_physical, self.discretization.grid.shape)
        if rho_physical.shape != self.discretization.grid.shape:
            raise ValueError("Initial density must be scalar or match the LBM grid.")
        dimension = self.discretization.velocity_set.dimension
        velocity_physical = jnp.asarray(velocity, dtype=dtype)
        if velocity_physical.shape == (dimension,):
            velocity_physical = jnp.broadcast_to(
                velocity_physical,
                (*self.discretization.grid.shape, dimension),
            )
        expected_velocity = (*self.discretization.grid.shape, dimension)
        if velocity_physical.shape != expected_velocity:
            raise ValueError(
                "Initial velocity must be one vector or one vector per grid cell."
            )
        rho_lattice = self.scaling.lattice_density(rho_physical)
        velocity_lattice = self.scaling.lattice_velocity(velocity_physical)
        if self.implicit_acceleration is None:
            acceleration = self._physical_acceleration(
                jnp.asarray(time, dtype=dtype), parameters_, dtype
            )
            force_density = rho_lattice[..., None] * self.scaling.lattice_acceleration(
                acceleration
            )
        else:
            problem = VelocityDependentAccelerationProblem(
                jnp.asarray(time, dtype=dtype),
                self.coordinates,
                rho_lattice,
                rho_lattice[..., None] * velocity_lattice,
                parameters=parameters_.force_parameters,
                initial_velocity=velocity_lattice,
            )
            acceleration = self.implicit_acceleration.evaluate_acceleration(
                velocity_lattice, problem
            )
            force_density = rho_lattice[..., None] * acceleration
        safe_density = jnp.where(rho_lattice > 0.0, rho_lattice, 1.0)
        raw_velocity = velocity_lattice - 0.5 * force_density / safe_density[..., None]
        populations = quadratic_equilibrium(
            rho_lattice,
            raw_velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        populations = self.discretization.precision.population(populations)
        fluid = self.boundary.geometry.fluid_mask
        safe_solid = quadratic_equilibrium(
            jnp.ones_like(rho_lattice),
            jnp.zeros_like(velocity_lattice),
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        populations = jnp.where(fluid[..., None], populations, safe_solid)
        valid = (
            jnp.all(jnp.isfinite(populations))
            & jnp.all((~fluid) | (rho_lattice > 0.0))
            & jnp.all(jnp.isfinite(force_density))
        )
        return eqx.error_if(
            populations,
            ~valid,
            "Initial LBM density, velocity, and force must be finite and admissible.",
        )

    def macroscopic_state(
        self,
        time: ArrayLike,
        populations: ArrayLike,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
    ) -> LatticeBoltzmannMacroscopicState:
        values = self.discretization.validate_populations(populations)
        fields = self._lattice_fields(
            jnp.asarray(time), values, self._parameters(parameters)
        )
        return LatticeBoltzmannMacroscopicState(
            self.scaling.physical_density(fields.density),
            self.scaling.physical_velocity(fields.velocity),
            self.scaling.physical_pressure(fields.density),
        )

    def _diagnostics(
        self,
        fields: _LatticeFields,
        populations: Array,
        mass_defect: Array,
        /,
    ) -> LatticeBoltzmannDiagnostics:
        fluid = self.boundary.geometry.fluid_mask
        density = jnp.where(fluid, fields.density, jnp.inf)
        speed = jnp.sqrt(oe.contract("...d,...d->...", fields.velocity, fields.velocity))
        cs = jnp.sqrt(
            jnp.asarray(
                self.discretization.velocity_set.sound_speed_squared,
                dtype=speed.dtype,
            )
        )
        momentum = jnp.where(
            fluid[..., None],
            fields.density[..., None] * fields.velocity,
            0.0,
        )
        force = jnp.where(fluid[..., None], fields.force_density, 0.0)
        return LatticeBoltzmannDiagnostics(
            minimum_density=jnp.min(density),
            maximum_mach=jnp.max(jnp.where(fluid, speed / cs, 0.0)),
            minimum_population=jnp.min(jnp.where(fluid[..., None], populations, jnp.inf)),
            total_mass=jnp.sum(jnp.where(fluid, fields.density, 0.0)),
            mass_defect=mass_defect,
            momentum_norm=jnp.sqrt(jnp.sum(momentum**2)),
            force_norm=jnp.sqrt(jnp.sum(force**2)),
        )

    def scalar_diagnostics(
        self,
        step_index: Array,
        time: Array,
        populations: Array,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
    ) -> LatticeBoltzmannDiagnostics:
        del step_index
        values = self.discretization.validate_populations(populations)
        fields = self._lattice_fields(time, values, self._parameters(parameters))
        zero = jnp.zeros((), dtype=values.dtype)
        return self._diagnostics(fields, values, zero)

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        populations: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> LatticeBoltzmannStepResult:
        del step_index
        values = self.discretization.validate_populations(populations)
        parameters = self._parameters(args)
        time_ = jnp.asarray(time, dtype=values.dtype)
        dt = jnp.asarray(step_size, dtype=values.dtype)
        expected_dt = jnp.asarray(self.scaling.time_step, dtype=values.dtype)
        fields = self._lattice_fields(time_, values, parameters)
        fluid = self.boundary.geometry.fluid_mask
        even_rate = self.scaling.relaxation_rate(parameters.kinematic_viscosity)
        collision_result = self.method.collide(
            self.discretization.precision.compute(values),
            fields.density,
            fields.velocity,
            fields.force_density,
            even_rate,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        post_collision = jnp.where(
            fluid[..., None], collision_result.candidate_populations, values
        )
        wall_velocity = self.scaling.lattice_velocity(parameters.moving_wall_velocities)
        if isinstance(self.boundary, PreparedStagedLatticeBoltzmannBoundary):
            boundary_parameters = parameters.boundary_parameters
            if boundary_parameters is None:
                dimension = self.discretization.velocity_set.dimension
                body_count = len(self.boundary.body_ids)
                boundary_parameters = LatticeBoltzmannBoundaryParameters(
                    velocity_targets=jnp.zeros(
                        (len(self.boundary.velocity_normals), dimension),
                        dtype=values.dtype,
                    ),
                    pressure_densities=jnp.ones(
                        (len(self.boundary.pressure_normals),),
                        dtype=values.dtype,
                    ),
                    pressure_tangential_velocities=jnp.zeros(
                        (len(self.boundary.pressure_normals), dimension),
                        dtype=values.dtype,
                    ),
                    convective_speeds=jnp.zeros(
                        (len(self.boundary.convective_normals),),
                        dtype=values.dtype,
                    ),
                    half_force_density=jnp.zeros(
                        self.discretization.grid.shape + (dimension,),
                        dtype=values.dtype,
                    ),
                    body_centers=jnp.zeros((body_count, dimension), dtype=values.dtype),
                    body_linear_velocities=jnp.zeros(
                        (body_count, dimension), dtype=values.dtype
                    ),
                    body_angular_velocities=jnp.zeros(
                        (body_count, 1 if dimension == 2 else 3),
                        dtype=values.dtype,
                    ),
                    time_step=self.scaling.time_step,
                )
            boundary_result = self.boundary.apply(
                self.discretization.precision.population(post_collision),
                fields.density,
                self.boundary.initial_state(values),
                boundary_parameters,
            )
            candidate = boundary_result.populations
        else:
            candidate = self.boundary.route(
                self.discretization.precision.population(post_collision),
                fields.density,
                wall_velocity,
            )
        candidate = self.discretization.precision.population(candidate)
        candidate_fields = self._lattice_fields(time_ + dt, candidate, parameters)
        previous_mass = jnp.sum(jnp.where(fluid, fields.density, 0.0))
        candidate_mass = jnp.sum(jnp.where(fluid, candidate_fields.density, 0.0))
        scale = jnp.maximum(jnp.abs(previous_mass), 1.0)
        mass_defect = jnp.abs(candidate_mass - previous_mass) / scale
        successful = (
            collision_result.successful
            & fields.force_successful
            & candidate_fields.force_successful
            & jnp.isclose(dt, expected_dt, rtol=1e-12, atol=1e-12)
            & jnp.all(jnp.isfinite(fields.force_density))
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all((~fluid) | (candidate_fields.density > 0.0))
        )
        accepted = jnp.where(successful, candidate, values)
        accepted_fields = self._lattice_fields(time_ + dt, accepted, parameters)
        diagnostics = self._diagnostics(
            accepted_fields,
            accepted,
            jnp.where(successful, mass_defect, 0.0),
        )
        work = jnp.asarray(
            self.boundary.geometry.fluid_count
            * self.discretization.velocity_set.population_count,
            dtype=jnp.int32,
        )
        return LatticeBoltzmannStepResult(
            candidate,
            accepted,
            successful,
            mass_defect,
            work,
            diagnostics,
            collision_result.diagnostics,
        )


__all__ = [
    "LatticeAcceleration",
    "LatticeBoltzmannDiagnostics",
    "LatticeBoltzmannMacroscopicState",
    "LatticeBoltzmannRuntimeParameters",
    "LatticeBoltzmannStepResult",
    "PreparedLatticeBoltzmannDynamics",
]
