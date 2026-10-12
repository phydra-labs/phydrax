#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization.mpm import (
    MPMGridState,
    MPMParticleState,
    MPMRunStatus,
    MPMRuntimeState,
    PreparedMPMDynamics,
)
from ..discretization.mpm._loads import MPMBoundaryLoadResult, MPMDashpotResult
from ..discretization.mpm._phases import normalize_grid_momentum
from ..discretization.mpm._transfer import build_apic_route_payload, gather_apic
from ..discretization.mpm._volumetric import MPMVolumetricStabilizationPlan
from ..discretization.spatial import SparseBlockTopologyState
from ..discretization.splatting import ParticleGridSplatState
from ..equations import (
    AbstractImplicitMPMConstitutivePlan,
    MaterialPointArguments,
)
from ..linalg import (
    GMRES,
    LinearSolvePolicy,
    SmallLinearSolvePlan,
    solve_small_linear,
    TolerancePolicy,
)
from ..nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ..typing import checked


class ImplicitMPMMethodPlan(StrictModule, NonTrainableState):
    nonlinear_method: NewtonKrylov
    termination: NonlinearTermination
    volumetric_stabilization: MPMVolumetricStabilizationPlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        nonlinear_method: NewtonKrylov | None = None,
        termination: NonlinearTermination | None = None,
        /,
        *,
        volumetric_stabilization: MPMVolumetricStabilizationPlan | None = None,
    ) -> None:
        method = (
            NewtonKrylov(
                linear_policy=LinearSolvePolicy(
                    GMRES(restart=64),
                    tolerance=TolerancePolicy(
                        relative=1.0e-9,
                        absolute=1.0e-11,
                        max_steps=512,
                    ),
                )
            )
            if nonlinear_method is None
            else nonlinear_method
        )
        termination_ = (
            NonlinearTermination(
                maximum_steps=25,
                maximum_linear_iterations=512,
            )
            if termination is None
            else termination
        )
        if not isinstance(method, NewtonKrylov):
            raise TypeError("nonlinear_method must be NewtonKrylov or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        stabilization = (
            MPMVolumetricStabilizationPlan()
            if volumetric_stabilization is None
            else volumetric_stabilization
        )
        if not isinstance(stabilization, MPMVolumetricStabilizationPlan):
            raise TypeError(
                "volumetric_stabilization must be MPMVolumetricStabilizationPlan or None."
            )
        self.nonlinear_method = method
        self.termination = termination_
        self.volumetric_stabilization = stabilization
        self.method_id = canonical_fingerprint(
            {
                "kind": "implicit-mpm-method",
                "integrator": "backward-euler-grid-velocity",
                "nonlinear_method": method.method_id,
                "maximum_steps": termination_.maximum_steps,
                "absolute_residual": termination_.absolute_residual,
                "relative_residual": termination_.relative_residual,
                "volumetric_stabilization": stabilization.plan_id,
            }
        )


class ImplicitMPMDiagnostics(StrictModule):
    """Convergence evidence and endpoint boundary impulse/work evaluations."""

    residual_norm: Array
    nonlinear_status: Array
    nonlinear_steps: Array
    linear_iterations: Array
    material_successful: Array
    tangent_successful: Array
    minimum_jacobian: Array
    finite: Array
    boundary_loads: tuple[MPMBoundaryLoadResult, ...] = ()
    dashpots: tuple[MPMDashpotResult, ...] = ()


class ImplicitMPMStepResult(StrictModule):
    candidate_state: MPMRuntimeState
    accepted_state: MPMRuntimeState
    grid: MPMGridState
    nonlinear_result: NonlinearResult
    diagnostics: ImplicitMPMDiagnostics
    successful: Array
    suggested_step: Array


@final
class PreparedImplicitMPMResidual(StrictModule, NonTrainableState):
    """Pure backward-Euler residual over a frozen particle/grid route topology.

    ``evaluate`` neither runs Newton nor mutates material or boundary history. Its
    JVP is the full discrete tangent, including the stabilized deformation update,
    force adjoint, follower-boundary geometry, and endpoint dashpot damping. Arrays
    captured during preparation remain dynamic PyTree leaves.
    """

    dynamics: PreparedMPMDynamics
    state: MPMRuntimeState
    arguments: MaterialPointArguments
    step_size: Array
    routes: ParticleGridSplatState
    execution_routes: ParticleGridSplatState
    storage_state: SparseBlockTopologyState | None
    mass: Array
    grid_mass: Array
    grid_momentum: Array
    normalized_velocity: Array
    active_nodes: Array
    initial_velocity: Array
    density: Array
    external: Array
    external_ok: Array
    initial_physical_admissible: Array | None = None

    def evaluate(
        self, grid_velocity: Array, /, *, include_inertia: bool = True
    ) -> tuple[Array, Array]:
        dynamics = self.dynamics
        particle = self.state.particles
        active = dynamics.particles.active_mask
        dimension = dynamics.dimension
        material_plan = dynamics.material
        if not isinstance(material_plan, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("Implicit MPM residual requires an implicit material.")
        if grid_velocity.shape != self.initial_velocity.shape:
            raise ValueError("Implicit MPM residual grid velocity shape changed.")
        kinematic_velocity = dynamics._axis_velocity(grid_velocity, self.storage_state)
        gathered = gather_apic(
            self.execution_routes,
            kinematic_velocity.reshape((dynamics.grid_count, dimension)),
            active,
            dynamics.method.transfer.maximum_condition,
        )
        deformation = dynamics._update_deformation(
            self.execution_routes,
            particle.deformation_gradient,
            gathered.velocity_gradient,
            self.step_size,
            self.mass,
            active,
            reference_volume=particle.reference_volume,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        linearized = material_plan.evaluate_linearized(
            deformation,
            particle.material_state,
            self.density,
            self.arguments.material_parameters,
            self.state.time + self.step_size if include_inertia else self.state.time,
            self.step_size,
        )
        material = linearized.response
        physical = dynamics._update_physical_deformation(
            particle.physical_deformation_gradient,
            gathered.velocity_gradient,
            self.step_size,
        )
        payload = build_apic_route_payload(
            self.execution_routes,
            self.mass,
            particle.velocity,
            particle.affine_velocity,
            particle.reference_volume,
            material.first_piola,
            deformation,
            self.external,
            active,
            physical_deformation_gradient=physical,
        )
        scattered = dynamics._scatter_route_payload(
            self.routes, self.storage_state, payload
        )
        force = (
            scattered.values[..., dimension : 2 * dimension]
            + scattered.values[..., 2 * dimension :]
        )
        load_state = (
            self.state
            if include_inertia
            else eqx.tree_at(
                lambda state: state.time, self.state, self.state.time - self.step_size
            )
        )
        boundary_loads, dashpots = dynamics._boundary_load_results(
            load_state,
            kinematic_velocity,
            self.step_size,
            self.arguments,
            self.storage_state,
            implicit=True,
        )
        if not include_inertia:
            dashpots = ()
        for load_result in (*boundary_loads, *dashpots):
            force = force + load_result.force
        value = self._grid_residual(grid_velocity, force, include_inertia)
        valid = (
            gathered.successful
            & material.successful.all()
            & material.admissible.all()
            & linearized.tangent_successful.all()
            & jnp.all(jnp.isfinite(value))
        )
        for load_result in (*boundary_loads, *dashpots):
            valid = (
                valid
                & load_result.successful
                & jnp.all(self.active_nodes[..., None] | (load_result.force == 0.0))
            )
        if dynamics.geometry.kind == "axisymmetric":
            candidate_position = particle.position + self.step_size * gathered.velocity
            valid = (
                valid
                & jnp.all(
                    (~active)
                    | (
                        dynamics.particle_domain.contains(particle.position)
                        & (particle.position[:, 0] > 0.0)
                    )
                )
                & jnp.all(
                    (~active)
                    | (
                        dynamics.particle_domain.contains(candidate_position)
                        & (candidate_position[:, 0] > 0.0)
                    )
                )
            )
        if physical is not None:
            if self.initial_physical_admissible is None:
                raise RuntimeError(
                    "F-bar residual requires physical deformation evidence."
                )
            valid = (
                valid
                & self.initial_physical_admissible
                & dynamics._physical_deformation_admissible(physical, active)
            )
        return value, valid

    def __call__(
        self, grid_velocity: Array, arguments: object = None, /
    ) -> tuple[Array, Array]:
        return self.evaluate(grid_velocity)

    def residual_jvp(
        self, grid_velocity: Array, direction: Array, /, *, include_inertia: bool = True
    ) -> Array:
        """Exact directional action of the same frozen-route physical update."""
        return jax.jvp(
            lambda value: self.evaluate(value, include_inertia=include_inertia)[0],
            (grid_velocity,),
            (direction,),
        )[1]

    def _grid_residual(
        self, grid_velocity: Array, force: Array, include_inertia: bool, /
    ) -> Array:
        dynamics = self.dynamics
        dimension = dynamics.dimension
        value = (
            self.grid_mass[..., None] * (grid_velocity - self.normalized_velocity)
            - self.step_size * force
            if include_inertia
            else -self.step_size * force
        )
        value = jnp.where(self.active_nodes[..., None], value, grid_velocity)
        if dynamics.boundary is not None:
            if dynamics.compact_storage:
                if self.storage_state is None:
                    raise RuntimeError("Compact implicit MPM requires storage state.")
                logical = self.storage_state.logical_node_ids.reshape((-1,))
                valid_nodes = self.storage_state.node_valid.reshape((-1,))
                mask = (
                    dynamics.boundary.mask.reshape((-1, dimension))[logical]
                    & valid_nodes[:, None]
                )
                prescribed = dynamics.boundary.values.reshape((-1, dimension))[
                    logical
                ].astype(grid_velocity.dtype)
            else:
                mask = dynamics.boundary.mask
                prescribed = dynamics.boundary.values.astype(grid_velocity.dtype)
            value = jnp.where(mask, grid_velocity - prescribed, value)
        if dynamics.geometry.kind == "axisymmetric":
            value = jnp.where(
                dynamics._axis_mask(self.storage_state), grid_velocity, value
            )
        return value

    def evaluate_detailed(
        self, grid_velocity: Array, /, *, include_inertia: bool = True
    ) -> ImplicitMPMResidualEvaluation:
        """Return candidate history without a root solve, commit, or route rebuild."""
        if grid_velocity.shape != self.initial_velocity.shape:
            raise ValueError("Implicit MPM residual grid velocity shape changed.")
        dynamics = self.dynamics
        particle = self.state.particles
        active = dynamics.particles.active_mask
        velocity = dynamics._axis_velocity(grid_velocity, self.storage_state)
        gathered = gather_apic(
            self.execution_routes,
            velocity.reshape((dynamics.grid_count, dynamics.dimension)),
            active,
            dynamics.method.transfer.maximum_condition,
        )
        deformation = dynamics._update_deformation(
            self.execution_routes,
            particle.deformation_gradient,
            gathered.velocity_gradient,
            self.step_size,
            self.mass,
            active,
            reference_volume=particle.reference_volume,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        material_plan = dynamics.material
        if not isinstance(material_plan, AbstractImplicitMPMConstitutivePlan):
            raise TypeError(
                "Implicit residual requires an implicit constitutive material."
            )
        linearized = material_plan.evaluate_linearized(
            deformation,
            particle.material_state,
            self.density,
            self.arguments.material_parameters,
            self.state.time + self.step_size if include_inertia else self.state.time,
            self.step_size,
        )
        material = linearized.response
        physical = dynamics._update_physical_deformation(
            particle.physical_deformation_gradient,
            gathered.velocity_gradient,
            self.step_size,
        )
        candidate_particle = MPMParticleState(
            particle.position + self.step_size * gathered.velocity,
            gathered.velocity if include_inertia else particle.velocity,
            deformation,
            gathered.affine_velocity if include_inertia else particle.affine_velocity,
            particle.reference_volume,
            material.first_piola,
            material.reference_energy_density,
            material.maximum_wave_speed,
            material.trial_state,
            physical,
        )
        load_state = (
            self.state
            if include_inertia
            else eqx.tree_at(
                lambda state: state.time, self.state, self.state.time - self.step_size
            )
        )
        loads, dashpots = dynamics._boundary_load_results(
            load_state,
            velocity,
            self.step_size,
            self.arguments,
            self.storage_state,
            implicit=True,
        )
        candidate_input = dynamics.splat.plan.assignment.update_input(
            candidate_particle.position,
            deformation if physical is None else physical,
            self.state.assignment_input,
        )
        candidate = MPMRuntimeState(
            candidate_particle,
            self.state.time + self.step_size if include_inertia else self.state.time,
            self.state.accepted_step + 1 if include_inertia else self.state.accepted_step,
            self.state.last_status,
            self.state.topology_generation
            if self.storage_state is None
            else self.storage_state.generation,
            candidate_input,
            self.state.material_slots,
            self.state.body_ids,
            self.state.velocity_field_slots,
            self.storage_state,
            self.state.lifecycle_state,
            tuple(load.candidate_state for load in loads)
            if include_inertia
            else self.state.boundary_states,
        )
        from ..linalg import determinant_small_linear

        determinant = determinant_small_linear(
            SmallLinearSolvePlan(dynamics.constitutive_dimension), deformation
        )
        admissible = jnp.all(
            (~active)
            | (
                material.admissible
                & (determinant > 0.0)
                & dynamics.particle_domain.contains(candidate_particle.position)
            )
        )
        tangent = jnp.all((~active) | linearized.tangent_successful)
        payload = build_apic_route_payload(
            self.execution_routes,
            self.mass,
            particle.velocity,
            particle.affine_velocity,
            particle.reference_volume,
            material.first_piola,
            deformation,
            self.external,
            active,
            physical_deformation_gradient=physical,
        )
        scattered = dynamics._scatter_route_payload(
            self.routes, self.storage_state, payload
        )
        dimension = dynamics.dimension
        internal = scattered.values[..., dimension : 2 * dimension]
        external = scattered.values[..., 2 * dimension :]
        force = internal + external
        for load in (*loads, *(dashpots if include_inertia else ())):
            external = external + load.force
            force = force + load.force
        grid = MPMGridState(
            self.grid_mass[None, ...],
            self.grid_momentum[None, ...],
            self.normalized_velocity[None, ...],
            internal[None, ...],
            external[None, ...],
            velocity[None, ...],
            self.active_nodes[None, ...],
        )
        residual = self._grid_residual(grid_velocity, force, include_inertia)
        successful = (
            self.external_ok
            & self.execution_routes.successful
            & gathered.successful
            & material.successful.all()
            & material.admissible.all()
            & linearized.tangent_successful.all()
            & tree_allfinite(candidate_particle)
            & jnp.all(jnp.isfinite(residual))
            & admissible
        )
        for load in (*loads, *(dashpots if include_inertia else ())):
            successful = (
                successful
                & load.successful
                & jnp.all(self.active_nodes[..., None] | (load.force == 0.0))
            )
        if dynamics.geometry.kind == "axisymmetric":
            successful = successful & jnp.all(
                (~active)
                | (
                    (particle.position[:, 0] > 0.0)
                    & (candidate_particle.position[:, 0] > 0.0)
                    & dynamics.particle_domain.contains(particle.position)
                )
            )
        if physical is not None:
            if self.initial_physical_admissible is None:
                raise RuntimeError(
                    "F-bar residual requires physical deformation evidence."
                )
            successful = (
                successful
                & self.initial_physical_admissible
                & dynamics._physical_deformation_admissible(physical, active)
            )
        return ImplicitMPMResidualEvaluation(
            residual,
            candidate,
            grid,
            successful,
            admissible,
            tangent,
            jnp.min(jnp.where(active, determinant, jnp.inf)),
            jnp.where(successful, self.step_size, 0.5 * self.step_size),
        )


@final
class ImplicitMPMResidualEvaluation(StrictModule):
    residual: Array
    candidate_state: MPMRuntimeState
    grid: MPMGridState
    successful: Array
    admissible: Array
    tangent_successful: Array
    minimum_jacobian: Array
    suggested_step: Array


@final
class PreparedImplicitMPMDynamics(StrictModule, NonTrainableState):
    explicit: PreparedMPMDynamics
    method: ImplicitMPMMethodPlan
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        explicit: PreparedMPMDynamics,
        method: ImplicitMPMMethodPlan | None = None,
        /,
    ) -> None:
        method_ = (
            ImplicitMPMMethodPlan(
                volumetric_stabilization=explicit.method.volumetric_stabilization
            )
            if method is None
            else method
        )
        if not isinstance(method_, ImplicitMPMMethodPlan):
            raise TypeError("method must be ImplicitMPMMethodPlan or None.")
        if (
            method_.volumetric_stabilization.plan_id
            != explicit.method.volumetric_stabilization.plan_id
        ):
            raise ValueError(
                "Implicit and explicit MPM volumetric stabilization plans must match."
            )
        if not isinstance(explicit.material, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("Implicit MPM requires an implicit constitutive plan.")
        if explicit.nodal_fields.field_count != 1:
            raise ValueError(
                "The single-field implicit adapter requires one nodal field; "
                "use MPMImplicitTopologyPlan for K-way implicit field layouts."
            )
        if explicit.contact is not None:
            raise ValueError(
                "The single-field implicit adapter excludes sharp rigid contact; "
                "use MPMImplicitContactOperator for complementarity coupling."
            )
        if explicit.splat.plan.assignment.capabilities.source_geometry_kind not in (
            "point",
            "uGIMP",
        ):
            raise ValueError(
                "The single-field implicit adapter requires point or fixed uGIMP "
                "routes; use MPMRouteSupersetPlan for moving-domain derivatives."
            )
        self.explicit = explicit
        self.method = method_
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-implicit-mpm",
                "explicit": explicit.prepared_id,
                "method": method_.method_id,
            }
        )

    def prepare_residual(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: MaterialPointArguments,
        /,
    ) -> PreparedImplicitMPMResidual:
        """Prepare the pure grid residual without solving or committing a step."""
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        dynamics = self.explicit
        particle = state.particles
        active_particles = dynamics.particles.active_mask
        mass = dynamics.particles.safe_masses.astype(particle.position.dtype)
        routes = dynamics._build_routes(
            particle.position, state.assignment_input, active_particles
        )
        storage_state = dynamics._build_storage(
            routes, state.storage_state, state.boundary_states
        )
        execution_routes = dynamics._mapped_routes(routes, storage_state)
        external, external_ok = dynamics._external(state.time, particle, arguments)
        mass_result = dynamics._deposit_content(routes, storage_state, mass)
        initial_payload = build_apic_route_payload(
            execution_routes,
            mass,
            particle.velocity,
            particle.affine_velocity,
            particle.reference_volume,
            particle.first_piola,
            particle.deformation_gradient,
            external,
            active_particles,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        initial_scatter = dynamics._scatter_route_payload(
            routes, storage_state, initial_payload
        )
        dimension = dynamics.dimension
        grid_mass = mass_result.content
        grid_momentum = initial_scatter.values[..., :dimension]
        normalized = normalize_grid_momentum(
            grid_mass,
            grid_momentum,
            mass_tolerance_factor=dynamics.method.mass_tolerance_factor,
        )
        node_valid = dynamics._storage_node_valid(storage_state).reshape(
            dynamics.nodal_shape
        )
        normalized = eqx.tree_at(
            lambda value: (value.active, value.velocity),
            normalized,
            (
                normalized.active & node_valid,
                jnp.where(
                    (normalized.active & node_valid)[..., None],
                    normalized.velocity,
                    0.0,
                ),
            ),
        )
        initial_guess = normalized.velocity
        if dynamics.boundary is not None:
            if dynamics.compact_storage:
                # Compact storage is block-sparse, whose build always returns a state.
                if not (storage_state is not None):
                    raise RuntimeError(
                        "Internal invariant failed: storage_state is not None."
                    )
                initial_guess = dynamics.boundary.apply_indexed(
                    initial_guess,
                    grid_mass,
                    dt,
                    storage_state.logical_node_ids.reshape((-1,)),
                    storage_state.node_valid.reshape((-1,)),
                ).velocity
            else:
                initial_guess = dynamics.boundary.apply(
                    initial_guess, grid_mass, dt
                ).velocity
        initial_guess = dynamics._axis_velocity(initial_guess, storage_state)
        density = mass / jnp.where(active_particles, particle.reference_volume, 1.0)

        return PreparedImplicitMPMResidual(
            dynamics,
            state,
            arguments,
            dt,
            routes,
            execution_routes,
            storage_state,
            mass,
            grid_mass,
            grid_momentum,
            normalized.velocity,
            normalized.active,
            initial_guess,
            density,
            external,
            external_ok,
            (
                None
                if particle.physical_deformation_gradient is None
                else dynamics._physical_deformation_admissible(
                    particle.physical_deformation_gradient, active_particles
                )
            ),
        )

    def step_detailed(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: MaterialPointArguments,
        /,
    ) -> ImplicitMPMStepResult:
        prepared = self.prepare_residual(state, step_size, arguments)
        dynamics = self.explicit
        particle = state.particles
        dt = prepared.step_size
        execution_routes = prepared.execution_routes
        storage_state = prepared.storage_state
        active_particles = dynamics.particles.active_mask
        mass = prepared.mass
        dimension = dynamics.dimension
        grid_mass = prepared.grid_mass
        grid_momentum = prepared.grid_momentum
        density = prepared.density
        material_plan = dynamics.material
        if not isinstance(material_plan, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("Implicit MPM requires an implicit material.")
        problem = NonlinearSystemProblem(
            prepared,
            has_aux=True,
            validity=lambda current, value, auxiliary, args: auxiliary,
            problem_id="implicit-mpm-grid-velocity",
        )
        nonlinear = implicit_root_result(
            problem,
            prepared.initial_velocity,
            method=self.method.nonlinear_method,
            termination=self.method.termination,
        )
        root = dynamics._axis_velocity(nonlinear.state, storage_state)
        gathered = gather_apic(
            execution_routes,
            root.reshape((dynamics.grid_count, dimension)),
            active_particles,
            dynamics.method.transfer.maximum_condition,
        )
        candidate_deformation = dynamics._update_deformation(
            execution_routes,
            particle.deformation_gradient,
            gathered.velocity_gradient,
            dt,
            mass,
            active_particles,
            reference_volume=particle.reference_volume,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        material = dynamics.material.evaluate(
            candidate_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
        linearized = material_plan.evaluate_linearized(
            candidate_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
        identity = jnp.broadcast_to(
            jnp.eye(dynamics.constitutive_dimension, dtype=particle.position.dtype),
            candidate_deformation.shape,
        )
        determinant = solve_small_linear(
            SmallLinearSolvePlan(dynamics.constitutive_dimension),
            candidate_deformation,
            identity,
        ).determinant
        material_ok = jnp.all(
            (~active_particles) | (material.successful & material.admissible)
        )
        tangent_ok = jnp.all((~active_particles) | linearized.tangent_successful)
        determinant_ok = jnp.all(
            (~active_particles) | (jnp.isfinite(determinant) & (determinant > 0.0))
        )
        candidate_particle = MPMParticleState(
            particle.position + dt * gathered.velocity,
            gathered.velocity,
            candidate_deformation,
            gathered.affine_velocity,
            particle.reference_volume,
            material.first_piola,
            material.reference_energy_density,
            material.maximum_wave_speed,
            material.trial_state,
            dynamics._update_physical_deformation(
                particle.physical_deformation_gradient, gathered.velocity_gradient, dt
            ),
        )
        boundary_loads, dashpots = dynamics._boundary_load_results(
            state, root, dt, arguments, storage_state, implicit=True
        )
        candidate_boundary_states = tuple(
            result.candidate_state for result in boundary_loads
        )
        finite = tree_allfinite(candidate_particle)
        successful = (
            nonlinear.successful
            & execution_routes.successful
            & prepared.external_ok
            & gathered.successful
            & material_ok
            & tangent_ok
            & determinant_ok
            & finite
        )
        for load_result in (*boundary_loads, *dashpots):
            successful = (
                successful
                & load_result.successful
                & jnp.all(prepared.active_nodes[..., None] | (load_result.force == 0.0))
            )
        if dynamics.geometry.kind == "axisymmetric":
            successful = (
                successful
                & jnp.all(
                    (~active_particles)
                    | (
                        dynamics.particle_domain.contains(particle.position)
                        & (particle.position[:, 0] > 0.0)
                    )
                )
                & jnp.all(
                    (~active_particles)
                    | (
                        dynamics.particle_domain.contains(candidate_particle.position)
                        & (candidate_particle.position[:, 0] > 0.0)
                    )
                )
            )
        candidate_input = dynamics.splat.plan.assignment.update_input(
            candidate_particle.position,
            (
                candidate_particle.deformation_gradient
                if candidate_particle.physical_deformation_gradient is None
                else candidate_particle.physical_deformation_gradient
            ),
            state.assignment_input,
        )
        accepted_particle = tree_where(successful, candidate_particle, particle)
        accepted_input = tree_where(successful, candidate_input, state.assignment_input)
        accepted_storage = tree_where(successful, storage_state, state.storage_state)
        accepted_boundary_states = tree_where(
            successful, candidate_boundary_states, state.boundary_states
        )
        candidate_generation = (
            state.topology_generation
            if storage_state is None
            else storage_state.generation
        )
        accepted_generation = jnp.where(
            successful, candidate_generation, state.topology_generation
        )
        status = jnp.where(
            successful,
            int(MPMRunStatus.SUCCESS),
            int(MPMRunStatus.NONLINEAR_FAILED),
        ).astype(jnp.int32)
        accepted_state = MPMRuntimeState(
            accepted_particle,
            jnp.where(successful, state.time + dt, state.time),
            jnp.where(successful, state.accepted_step + 1, state.accepted_step),
            status,
            accepted_generation,
            accepted_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            accepted_storage,
            state.lifecycle_state,
            accepted_boundary_states,
        )
        candidate_state = MPMRuntimeState(
            candidate_particle,
            state.time + dt,
            state.accepted_step + 1,
            status,
            candidate_generation,
            candidate_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            storage_state,
            state.lifecycle_state,
            candidate_boundary_states,
        )
        residual_norm = jnp.linalg.norm(nonlinear.residual.reshape((-1,)))
        diagnostics = ImplicitMPMDiagnostics(
            residual_norm,
            nonlinear.status,
            nonlinear.diagnostics.iterations,
            nonlinear.diagnostics.linear_iterations,
            material_ok,
            tangent_ok,
            jnp.min(jnp.where(active_particles, determinant, jnp.inf)),
            finite,
            boundary_loads,
            dashpots,
        )
        zero_force = jnp.zeros_like(root)
        boundary_force = zero_force
        for load_result in (*boundary_loads, *dashpots):
            boundary_force = boundary_force + load_result.force
        grid = MPMGridState(
            grid_mass[None, ...],
            grid_momentum[None, ...],
            prepared.normalized_velocity[None, ...],
            zero_force[None, ...],
            boundary_force[None, ...],
            root[None, ...],
            prepared.active_nodes[None, ...],
        )
        return ImplicitMPMStepResult(
            candidate_state,
            accepted_state,
            grid,
            nonlinear,
            diagnostics,
            successful,
            jnp.where(successful, dt, 0.5 * dt),
        )


__all__ = [
    "ImplicitMPMDiagnostics",
    "ImplicitMPMMethodPlan",
    "ImplicitMPMResidualEvaluation",
    "ImplicitMPMStepResult",
    "PreparedImplicitMPMDynamics",
    "PreparedImplicitMPMResidual",
]
