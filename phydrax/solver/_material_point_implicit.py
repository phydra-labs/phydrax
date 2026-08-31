#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

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
from ..discretization.mpm._phases import normalize_grid_momentum, update_deformation
from ..discretization.mpm._transfer import build_apic_route_payload, gather_apic
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


class ImplicitMPMMethodPlan(StrictModule, NonTrainableState):
    nonlinear_method: NewtonKrylov
    termination: NonlinearTermination
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        nonlinear_method: NewtonKrylov | None = None,
        termination: NonlinearTermination | None = None,
        /,
    ):
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
        self.nonlinear_method = method
        self.termination = termination_
        self.method_id = canonical_fingerprint(
            {
                "kind": "implicit-mpm-method",
                "integrator": "backward-euler-grid-velocity",
                "nonlinear_method": method.method_id,
                "maximum_steps": termination_.maximum_steps,
                "absolute_residual": termination_.absolute_residual,
                "relative_residual": termination_.relative_residual,
            }
        )


class ImplicitMPMDiagnostics(StrictModule):
    residual_norm: Array
    nonlinear_status: Array
    nonlinear_steps: Array
    linear_iterations: Array
    material_successful: Array
    tangent_successful: Array
    minimum_jacobian: Array
    finite: Array


class ImplicitMPMStepResult(StrictModule):
    candidate_state: MPMRuntimeState
    accepted_state: MPMRuntimeState
    grid: MPMGridState
    nonlinear_result: NonlinearResult
    diagnostics: ImplicitMPMDiagnostics
    successful: Array
    suggested_step: Array


class PreparedImplicitMPMDynamics(StrictModule, NonTrainableState):
    explicit: PreparedMPMDynamics
    method: ImplicitMPMMethodPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        explicit: PreparedMPMDynamics,
        method: ImplicitMPMMethodPlan | None = None,
        /,
    ):
        if not isinstance(explicit, PreparedMPMDynamics):
            raise TypeError("explicit must be PreparedMPMDynamics.")
        method_ = ImplicitMPMMethodPlan() if method is None else method
        if not isinstance(method_, ImplicitMPMMethodPlan):
            raise TypeError("method must be ImplicitMPMMethodPlan or None.")
        if not isinstance(explicit.material, AbstractImplicitMPMConstitutivePlan):
            raise TypeError("Implicit MPM requires an implicit constitutive plan.")
        if explicit.nodal_fields.field_count != 1:
            raise ValueError("Initial implicit MPM supports one nodal field.")
        if explicit.contact is not None:
            raise ValueError("Initial implicit MPM does not support sharp contact.")
        if explicit.splat.plan.assignment.capabilities.source_geometry_kind not in (
            "point",
            "uGIMP",
        ):
            raise ValueError(
                "Initial implicit MPM supports point or fixed uGIMP routes only."
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

    def step_detailed(
        self,
        state: MPMRuntimeState,
        step_size: ArrayLike,
        arguments: MaterialPointArguments,
        /,
    ) -> ImplicitMPMStepResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        dynamics = self.explicit
        particle = state.particles
        active_particles = dynamics.particles.active_mask
        mass = dynamics.particles.safe_masses.astype(particle.position.dtype)
        routes = dynamics.splat.build(
            particle.position, assignment_input=state.assignment_input
        )
        external, external_ok = dynamics._external(state.time, particle, arguments)
        mass_result = dynamics.splat.deposit_content(routes, mass)
        initial_payload = build_apic_route_payload(
            routes,
            mass,
            particle.velocity,
            particle.affine_velocity,
            particle.reference_volume,
            particle.first_piola,
            particle.deformation_gradient,
            external,
            active_particles,
        )
        initial_scatter = dynamics.splat.scatter_route_payload(routes, initial_payload)
        dimension = dynamics.dimension
        grid_mass = mass_result.content
        grid_momentum = initial_scatter.values[..., :dimension]
        normalized = normalize_grid_momentum(
            grid_mass,
            grid_momentum,
            mass_tolerance_factor=dynamics.method.mass_tolerance_factor,
        )
        initial_guess = normalized.velocity
        if dynamics.boundary is not None:
            initial_guess = dynamics.boundary.apply(initial_guess, grid_mass, dt).velocity
        density = mass / jnp.where(active_particles, particle.reference_volume, 1.0)

        def residual(grid_velocity, _):
            gathered = gather_apic(
                routes,
                grid_velocity.reshape((dynamics.splat.target_size, dimension)),
                active_particles,
                dynamics.method.transfer.maximum_condition,
            )
            deformation = update_deformation(
                particle.deformation_gradient, gathered.velocity_gradient, dt
            )
            material = dynamics.material.evaluate(
                deformation,
                particle.material_state,
                density,
                arguments.material_parameters,
                state.time + dt,
                dt,
            )
            payload = build_apic_route_payload(
                routes,
                mass,
                particle.velocity,
                particle.affine_velocity,
                particle.reference_volume,
                material.first_piola,
                deformation,
                external,
                active_particles,
            )
            scattered = dynamics.splat.scatter_route_payload(routes, payload)
            force = (
                scattered.values[..., dimension : 2 * dimension]
                + scattered.values[..., 2 * dimension :]
            )
            value = (
                grid_mass[..., None] * (grid_velocity - normalized.velocity) - dt * force
            )
            value = jnp.where(normalized.active[..., None], value, grid_velocity)
            if dynamics.boundary is not None:
                value = jnp.where(
                    dynamics.boundary.mask,
                    grid_velocity - dynamics.boundary.values.astype(grid_velocity.dtype),
                    value,
                )
            valid = (
                gathered.successful
                & material.successful.all()
                & material.admissible.all()
                & jnp.all(jnp.isfinite(value))
            )
            return value, valid

        problem = NonlinearSystemProblem(
            residual,
            has_aux=True,
            validity=lambda current, value, auxiliary, args: auxiliary,
            problem_id="implicit-mpm-grid-velocity",
        )
        nonlinear = implicit_root_result(
            problem,
            initial_guess,
            method=self.method.nonlinear_method,
            termination=self.method.termination,
        )
        root = nonlinear.state
        gathered = gather_apic(
            routes,
            root.reshape((dynamics.splat.target_size, dimension)),
            active_particles,
            dynamics.method.transfer.maximum_condition,
        )
        candidate_deformation = update_deformation(
            particle.deformation_gradient, gathered.velocity_gradient, dt
        )
        material = dynamics.material.evaluate(
            candidate_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
        linearized = dynamics.material.evaluate_linearized(
            candidate_deformation,
            particle.material_state,
            density,
            arguments.material_parameters,
            state.time + dt,
            dt,
        )
        identity = jnp.broadcast_to(
            jnp.eye(dimension, dtype=particle.position.dtype),
            candidate_deformation.shape,
        )
        determinant = solve_small_linear(
            SmallLinearSolvePlan(dimension), candidate_deformation, identity
        ).determinant
        material_ok = jnp.all(
            (~active_particles) | (material.successful & material.admissible)
        )
        tangent_ok = jnp.all((~active_particles) | linearized.tangent_successful)
        jacobian_ok = jnp.all(
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
        )
        finite = tree_allfinite(candidate_particle)
        successful = (
            nonlinear.successful
            & routes.successful
            & external_ok
            & gathered.successful
            & material_ok
            & finite
        )
        candidate_input = dynamics.splat.plan.assignment.update_input(
            candidate_particle.position,
            candidate_particle.deformation_gradient,
            state.assignment_input,
        )
        accepted_particle = tree_where(successful, candidate_particle, particle)
        accepted_input = tree_where(successful, candidate_input, state.assignment_input)
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
            state.topology_generation,
            accepted_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            state.storage_state,
        )
        candidate_state = MPMRuntimeState(
            candidate_particle,
            state.time + dt,
            state.accepted_step + 1,
            status,
            state.topology_generation,
            candidate_input,
            state.material_slots,
            state.body_ids,
            state.velocity_field_slots,
            state.storage_state,
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
        )
        zero_force = jnp.zeros_like(root)
        grid = MPMGridState(
            grid_mass[None, ...],
            grid_momentum[None, ...],
            normalized.velocity[None, ...],
            zero_force[None, ...],
            zero_force[None, ...],
            root[None, ...],
            normalized.active[None, ...],
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
    "ImplicitMPMStepResult",
    "PreparedImplicitMPMDynamics",
]
