#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
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
from ..discretization.finite_volume._mac_marker_transfer import (
    MACMarkerTransferDiagnostics,
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
StructuralContactResidual = Callable[[Array, Array, Any], Array]


class MACDeformableImmersedStatus(IntFlag):
    SUCCESS = 0
    NONLINEAR_FAILED = 1
    TRANSFER_FAILED = 2
    ROUTE_CHANGED = 4
    DIVERGENCE_FAILED = 8
    SLIP_FAILED = 16
    PRESSURE_GAUGE_FAILED = 32
    COUPLING_WORK_FAILED = 64
    NONFINITE = 128


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
    time: Array
    fluid_state: Array
    configuration: Array
    structural_velocity: Array
    pressure: Array
    marker_force_density: Array
    accepted_steps: Array
    status: Array


class MACDeformableImmersedStepResult(StrictModule):
    candidate_state: MACDeformableImmersedState
    accepted_state: MACDeformableImmersedState
    nonlinear: NonlinearResult
    helmholtz: MACHelmholtzResult
    energy: MACDeformableImmersedEnergyLedger
    transfer_diagnostics: MACMarkerTransferDiagnostics
    divergence: Array
    marker_slip: Array
    divergence_norm: Array
    slip_norm: Array
    gauge_defect: Array
    kkt_residual_norm: Array
    status: Array
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
    structural_contact_residual: StructuralContactResidual | None
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
        structural_contact_residual: StructuralContactResidual | None = None,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(projection, MACImmersedBoundaryProjectionPlan):
            raise TypeError("projection must be MACImmersedBoundaryProjectionPlan.")
        if not isinstance(marker_map, PreparedFiniteElementImmersedMarkerMap):
            raise TypeError("marker_map must be PreparedFiniteElementImmersedMarkerMap.")
        if not isinstance(structure, SecondOrderDifferentialSystem):
            raise TypeError("structure must be SecondOrderDifferentialSystem.")
        if marker_map.configuration_space.size != int(
            jnp.prod(jnp.asarray(structure.state_shape))
        ):
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
        if structural_contact_residual is not None and not callable(
            structural_contact_residual
        ):
            raise TypeError("structural_contact_residual must be callable or None.")
        self.structural_contact_residual = structural_contact_residual
        self.step_size = step
        self.helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            fixed_mass_coefficient=1.0,
            fixed_diffusion_coefficient=step * viscosity,
        )
        self.nonlinear_method = (
            NewtonKrylov() if nonlinear_method is None else nonlinear_method
        )
        self.termination = (
            NonlinearTermination(maximum_steps=25) if termination is None else termination
        )
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-deformable-immersed-backward-euler",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "marker_map": marker_map.prepared_id,
                "structure": structure.system_id,
                "energy": identifier,
                "contact": structural_contact_residual is not None,
                "step_size": step,
            }
        )

    def initialize(
        self,
        fluid_state: ArrayLike,
        configuration: ArrayLike,
        structural_velocity: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> MACDeformableImmersedState:
        fluid = self.dynamics.validate_state(fluid_state)
        q = jnp.asarray(configuration)
        v = jnp.asarray(structural_velocity)
        if q.shape != self.structure.state_shape or v.shape != q.shape:
            raise ValueError(
                "Structural initial arrays must match the second-order system."
            )
        dtype = fluid.dtype
        pressure = jnp.zeros(
            self.dynamics.momentum.operators.discretization.cell_shape, dtype=dtype
        )
        marker_force = jnp.zeros(
            self.marker_map.markers.active_velocity_space.structure().shape,
            dtype=dtype,
        )
        return MACDeformableImmersedState(
            jnp.asarray(time, dtype=dtype).reshape(()),
            fluid,
            q,
            v,
            pressure,
            marker_force,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(int(MACDeformableImmersedStatus.SUCCESS), dtype=jnp.int32),
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
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
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
            relation = self.projection.transfer.relation_on_routes(
                marker_kinematics.position,
                self.projection.transfer.route_state(predictor_relation),
            )
            if boundaries.closure_kind == "neumann":
                volumes = operators.discretization.cell_volumes.astype(pressure.dtype)
                pressure_mean = jnp.sum(volumes * pressure) / jnp.sum(volumes)
                pressure_value = pressure - pressure_mean
            else:
                pressure_mean = jnp.asarray(0.0, dtype=pressure.dtype)
                pressure_value = pressure
            gradient = boundaries.pressure_gradient(
                pressure_value,
                boundary_stage,
                homogeneous=boundaries.closure_kind == "neumann",
            )
            spread = self.projection.transfer.spread(relation, multiplier)
            fluid_residual = boundaries.homogeneous_rate(
                tuple(
                    value - predictor + step * derivative - step * force
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
            structural = self.structure.evaluate(attempted_time, q, v, acceleration, args)
            contact_residual = (
                jnp.zeros_like(structural)
                if self.structural_contact_residual is None
                else self.structural_contact_residual(q, v, args)
            )
            load = self.marker_map.structural_load(multiplier)
            structural_residual = (
                structural
                + contact_residual
                + self.marker_map.configuration_space.flatten(load).reshape(
                    structural.shape
                )
            )
            configuration_residual = q - initial_q - step * v
            divergence = operators.divergence(fluid_velocity) + pressure_mean
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
        (
            fluid_coordinates,
            q_candidate,
            v_candidate,
            pressure_candidate,
            multiplier_candidate,
        ) = nonlinear.state
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
        volumes = operators.discretization.cell_volumes.astype(fluid_coordinates.dtype)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        slip_norm = jnp.sqrt(
            jnp.real(
                self.marker_map.markers.active_velocity_space.inner(
                    marker_slip, marker_slip
                )
            )
        )
        pressure_value = (
            operators.gauge_project(pressure_candidate)
            if boundaries.closure_kind == "neumann"
            else pressure_candidate
        )
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_value))
            if boundaries.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=pressure_value.dtype)
        )
        transfer_diagnostics = self.projection.transfer.diagnostics(
            relation_candidate,
            candidate_velocity,
            multiplier_candidate,
        )
        kkt_residual_norm = nonlinear.diagnostics.final_residual_norm
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
        coupling_power_residual = fluid_power + structural_power
        energy = MACDeformableImmersedEnergyLedger(
            fluid_before,
            fluid_after,
            structure_before,
            structure_after,
            fluid_power,
            structural_power,
            coupling_power_residual,
            fluid_after + structure_after - fluid_before - structure_before,
        )
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.stack(
                    (
                        kkt_residual_norm,
                        divergence_norm,
                        slip_norm,
                        jnp.abs(fluid_power),
                        jnp.abs(structural_power),
                    )
                )
            ),
        )
        tolerance = self.projection.tolerance * scale
        finite = (
            relation_candidate.successful
            & transfer_diagnostics.finite
            & jnp.all(jnp.isfinite(fluid_coordinates))
            & jnp.all(jnp.isfinite(q_candidate))
            & jnp.all(jnp.isfinite(v_candidate))
            & jnp.all(jnp.isfinite(pressure_value))
            & jnp.all(jnp.isfinite(multiplier_candidate))
            & jnp.isfinite(coupling_power_residual)
        )
        checks = (
            (nonlinear.successful, MACDeformableImmersedStatus.NONLINEAR_FAILED),
            (
                relation_candidate.successful & transfer_diagnostics.successful,
                MACDeformableImmersedStatus.TRANSFER_FAILED,
            ),
            (route_unchanged, MACDeformableImmersedStatus.ROUTE_CHANGED),
            (
                divergence_norm <= tolerance,
                MACDeformableImmersedStatus.DIVERGENCE_FAILED,
            ),
            (slip_norm <= tolerance, MACDeformableImmersedStatus.SLIP_FAILED),
            (
                gauge_defect <= self.projection.tolerance,
                MACDeformableImmersedStatus.PRESSURE_GAUGE_FAILED,
            ),
            (
                jnp.abs(coupling_power_residual) <= tolerance,
                MACDeformableImmersedStatus.COUPLING_WORK_FAILED,
            ),
            (finite, MACDeformableImmersedStatus.NONFINITE),
        )
        status = jnp.asarray(int(MACDeformableImmersedStatus.SUCCESS), dtype=jnp.int32)
        accepted = jnp.asarray(True)
        for passed, flag in checks:
            accepted = accepted & passed
            status = status | jnp.where(passed, 0, int(flag)).astype(jnp.int32)
        candidate_state = MACDeformableImmersedState(
            attempted_time,
            fluid_coordinates,
            q_candidate,
            v_candidate,
            pressure_value,
            multiplier_candidate,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
            status,
        )
        accepted_state = jax_tree_where(accepted, candidate_state, state)
        return MACDeformableImmersedStepResult(
            candidate_state=candidate_state,
            accepted_state=accepted_state,
            nonlinear=nonlinear,
            helmholtz=helmholtz,
            energy=energy,
            transfer_diagnostics=transfer_diagnostics,
            divergence=divergence,
            marker_slip=marker_slip,
            divergence_norm=divergence_norm,
            slip_norm=slip_norm,
            gauge_defect=gauge_defect,
            kkt_residual_norm=kkt_residual_norm,
            status=status,
            route_unchanged=route_unchanged,
            finite=finite,
            accepted=accepted,
            method_id=self.method_id,
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
    "StructuralContactResidual",
]
