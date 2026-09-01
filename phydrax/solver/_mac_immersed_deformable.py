#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.fem._immersed_marker import (
    PreparedFiniteElementImmersedMarkerMap,
)
from ..dynamics import SecondOrderDifferentialSystem
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..nonlinear import (
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._mac_immersed_boundary import MACImmersedBoundaryProjectionPlan
from ._mac_viscous import MACHelmholtzResult, MACHelmholtzSolvePlan


StructuralEnergy = Callable[[Array, Any], Array]


class MACDeformableImmersedEnergyLedger(StrictModule):
    fluid_kinetic_before: Array
    fluid_kinetic_after: Array
    structural_energy_before: Array
    structural_energy_after: Array
    fluid_coupling_power: Array
    structural_coupling_power: Array
    coupling_power_residual: Array
    total_energy_change: Array


class MACDeformableImmersedState(StrictModule):
    fluid_state: Array
    configuration: Array
    structural_velocity: Array
    pressure: Array
    marker_force_density: Array
    accepted_steps: Array


class MACDeformableImmersedStepResult(StrictModule):
    candidate_state: MACDeformableImmersedState
    accepted_state: MACDeformableImmersedState
    nonlinear: NonlinearResult
    helmholtz: MACHelmholtzResult
    energy: MACDeformableImmersedEnergyLedger
    divergence: Array
    marker_slip: Array
    route_unchanged: Array
    finite: Array
    accepted: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACDeformableImmersedBackwardEulerMethod(StrictModule, NonTrainableState):
    """Synchronized contact-free fluid/FE backward-Euler immersed coupling."""

    dynamics: CompiledMACIncompressibleDynamics
    projection: MACImmersedBoundaryProjectionPlan
    marker_map: PreparedFiniteElementImmersedMarkerMap
    structure: SecondOrderDifferentialSystem
    structural_energy: StructuralEnergy
    energy_id: str = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    helmholtz: MACHelmholtzSolvePlan
    nonlinear_method: NewtonKrylov
    termination: NonlinearTermination
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        projection: MACImmersedBoundaryProjectionPlan,
        marker_map: PreparedFiniteElementImmersedMarkerMap,
        structure: SecondOrderDifferentialSystem,
        structural_energy: StructuralEnergy,
        step_size: float,
        /,
        *,
        energy_id: str,
        nonlinear_method: NewtonKrylov | None = None,
        termination: NonlinearTermination | None = None,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(projection, MACImmersedBoundaryProjectionPlan):
            raise TypeError("projection must be MACImmersedBoundaryProjectionPlan.")
        if not isinstance(marker_map, PreparedFiniteElementImmersedMarkerMap):
            raise TypeError("marker_map must be PreparedFiniteElementImmersedMarkerMap.")
        if not isinstance(structure, SecondOrderDifferentialSystem):
            raise TypeError("structure must be SecondOrderDifferentialSystem.")
        if marker_map.configuration_space.size != int(jnp.prod(jnp.asarray(structure.state_shape))):
            raise ValueError("FE marker-map and structural state dimensions differ.")
        if marker_map.markers.prepared_id != projection.transfer.markers.prepared_id:
            raise ValueError("FE map and immersed projection must share markers.")
        if not callable(structural_energy):
            raise TypeError("structural_energy must be callable.")
        identifier = str(energy_id)
        if not identifier:
            raise ValueError("energy_id must be nonempty.")
        step = float(step_size)
        if step <= 0.0:
            raise ValueError("step_size must be positive.")
        viscosity = float(jnp.asarray(dynamics.problem.viscosity))
        self.dynamics = dynamics
        self.projection = projection
        self.marker_map = marker_map
        self.structure = structure
        self.structural_energy = structural_energy
        self.energy_id = identifier
        self.step_size = step
        self.helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            fixed_mass_coefficient=1.0,
            fixed_diffusion_coefficient=step * viscosity,
        )
        self.nonlinear_method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
        self.termination = (
            NonlinearTermination(maximum_steps=25)
            if termination is None
            else termination
        )
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-deformable-immersed-backward-euler",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "marker_map": marker_map.prepared_id,
                "structure": structure.system_id,
                "energy": identifier,
                "step_size": step,
            }
        )

    def initialize(
        self,
        fluid_state: ArrayLike,
        configuration: ArrayLike,
        structural_velocity: ArrayLike,
        /,
    ) -> MACDeformableImmersedState:
        fluid = self.dynamics.validate_state(fluid_state)
        q = jnp.asarray(configuration)
        v = jnp.asarray(structural_velocity)
        if q.shape != self.structure.state_shape or v.shape != q.shape:
            raise ValueError("Structural initial arrays must match the second-order system.")
        dtype = fluid.dtype
        pressure = jnp.zeros(
            self.dynamics.momentum.operators.discretization.cell_shape, dtype=dtype
        )
        marker_force = jnp.zeros(
            self.marker_map.markers.active_velocity_space.structure().shape,
            dtype=dtype,
        )
        return MACDeformableImmersedState(
            fluid, q, v, pressure, marker_force, jnp.zeros((), dtype=jnp.int32)
        )

    def step(
        self,
        time: ArrayLike,
        state: MACDeformableImmersedState,
        /,
        *,
        args: Any = None,
    ) -> MACDeformableImmersedStepResult:
        if not isinstance(state, MACDeformableImmersedState):
            raise TypeError("state must be MACDeformableImmersedState.")
        step = jnp.asarray(self.step_size, dtype=state.fluid_state.dtype)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        attempted_time = time_ + step
        current_fluid = self.dynamics.validate_state(state.fluid_state)
        current_velocity = self.dynamics.unpack_velocity(current_fluid)
        _, convection, _, forcing = self.dynamics.rate_components(
            time_, current_fluid, args
        )
        explicit = tuple(
            -advective + source
            for advective, source in zip(convection, forcing, strict=True)
        )
        rhs = tuple(
            value + step * rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(
            attempted_time, args
        )
        helmholtz = self.helmholtz.solve(
            rhs, boundary_stage, initial_guess=current_velocity
        )
        tentative_coordinates = self.dynamics.momentum.operators.velocity_space.flatten(
            helmholtz.value
        )
        initial_q = state.configuration
        initial_v = state.structural_velocity
        predicted_q = initial_q + step * initial_v
        predictor_kinematics = self.marker_map.kinematics(predicted_q, initial_v)
        predictor_relation = self.projection.transfer.relation(
            predictor_kinematics.position
        )
        initial_guess = (
            tentative_coordinates,
            predicted_q,
            initial_v,
            state.pressure,
            state.marker_force_density,
        )
        operators = self.dynamics.momentum.operators
        boundaries = self.dynamics.momentum.boundaries

        def residual(unknown, _):
            fluid_coordinates, q, v, pressure, multiplier = unknown
            fluid_velocity = tuple(operators.velocity_space.unflatten(fluid_coordinates))
            marker_kinematics = self.marker_map.kinematics(q, v)
            relation = self.projection.transfer.relation(marker_kinematics.position)
            gradient = boundaries.pressure_gradient(
                pressure,
                boundary_stage,
                homogeneous=boundaries.closure_kind == "neumann",
            )
            spread = self.projection.transfer.spread(relation, multiplier)
            fluid_residual = boundaries.homogeneous_rate(
                tuple(
                    value
                    - predictor
                    + step * derivative
                    - step * force
                    for value, predictor, derivative, force in zip(
                        fluid_velocity,
                        helmholtz.value,
                        gradient,
                        spread,
                        strict=True,
                    )
                )
            )
            acceleration = (v - initial_v) / step
            structural = self.structure.evaluate(
                attempted_time, q, v, acceleration, args
            )
            load = self.marker_map.structural_load(multiplier)
            structural_residual = (
                structural
                + self.marker_map.configuration_space.flatten(load).reshape(
                    structural.shape
                )
            )
            configuration_residual = q - initial_q - step * v
            divergence = operators.divergence(fluid_velocity)
            marker_slip = self.projection.transfer.gather(
                relation, fluid_velocity
            ) - self.marker_map.active_velocity(v)
            return (
                operators.velocity_space.flatten(fluid_residual),
                configuration_residual,
                structural_residual,
                divergence,
                marker_slip,
            )

        problem = NonlinearSystemProblem(
            residual,
            problem_id=f"mac-deformable-immersed/{self.method_id}",
        ).bind_spaces(initial_guess)
        nonlinear = self.nonlinear_method.solve(
            problem,
            initial_guess,
            termination=self.termination,
        )
        fluid_coordinates, q_candidate, v_candidate, pressure_candidate, multiplier_candidate = nonlinear.state
        candidate_velocity = tuple(operators.velocity_space.unflatten(fluid_coordinates))
        marker_candidate = self.marker_map.kinematics(q_candidate, v_candidate)
        relation_candidate = self.projection.transfer.relation(marker_candidate.position)
        marker_slip = self.projection.transfer.gather(
            relation_candidate, candidate_velocity
        ) - self.marker_map.active_velocity(v_candidate)
        divergence = operators.divergence(candidate_velocity)
        route_unchanged = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(candidate_indices == predictor_indices)
                    & jnp.all(candidate_valid == predictor_valid)
                    for candidate_indices, predictor_indices, candidate_valid, predictor_valid in zip(
                        relation_candidate.face_indices,
                        predictor_relation.face_indices,
                        relation_candidate.valid,
                        predictor_relation.valid,
                        strict=True,
                    )
                )
            )
        )
        finite = (
            nonlinear.successful
            & relation_candidate.successful
            & jnp.all(jnp.isfinite(fluid_coordinates))
            & jnp.all(jnp.isfinite(q_candidate))
            & jnp.all(jnp.isfinite(v_candidate))
            & jnp.all(jnp.isfinite(pressure_candidate))
            & jnp.all(jnp.isfinite(multiplier_candidate))
        )
        accepted = finite & route_unchanged
        candidate_state = MACDeformableImmersedState(
            fluid_coordinates,
            q_candidate,
            v_candidate,
            pressure_candidate,
            multiplier_candidate,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted_state = jax_tree_where(accepted, candidate_state, state)
        fluid_before = 0.5 * jnp.real(
            operators.velocity_space.inner(current_velocity, current_velocity)
        )
        fluid_after = 0.5 * jnp.real(
            operators.velocity_space.inner(candidate_velocity, candidate_velocity)
        )
        structure_before = jnp.asarray(self.structural_energy(initial_q, args))
        structure_after = jnp.asarray(self.structural_energy(q_candidate, args))
        marker_fluid_velocity = self.projection.transfer.gather(
            relation_candidate, candidate_velocity
        )
        fluid_power = jnp.real(
            self.marker_map.markers.active_velocity_space.inner(
                marker_fluid_velocity, multiplier_candidate
            )
        )
        structural_load = self.marker_map.structural_load(multiplier_candidate)
        structural_power = -jnp.real(
            self.marker_map.configuration_space.inner(v_candidate, structural_load)
        )
        energy = MACDeformableImmersedEnergyLedger(
            fluid_before,
            fluid_after,
            structure_before,
            structure_after,
            fluid_power,
            structural_power,
            fluid_power + structural_power,
            fluid_after
            + structure_after
            - fluid_before
            - structure_before,
        )
        return MACDeformableImmersedStepResult(
            candidate_state,
            accepted_state,
            nonlinear,
            helmholtz,
            energy,
            divergence,
            marker_slip,
            route_unchanged,
            finite,
            accepted,
            self.method_id,
        )


def jax_tree_where(condition: Array, candidate, fallback):
    import jax

    return jax.tree.map(
        lambda accepted, rejected: jnp.where(condition, accepted, rejected),
        candidate,
        fallback,
    )


__all__ = [
    "MACDeformableImmersedBackwardEulerMethod",
    "MACDeformableImmersedEnergyLedger",
    "MACDeformableImmersedState",
    "MACDeformableImmersedStepResult",
    "StructuralEnergy",
]
