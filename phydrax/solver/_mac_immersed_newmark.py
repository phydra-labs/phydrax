#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import MACMarkerTransferDiagnostics
from ..nonlinear import NonlinearResult, NonlinearSystemProblem
from ._mac_immersed_deformable import (
    jax_tree_where,
    MACDeformableImmersedBackwardEulerMethod,
    MACDeformableImmersedEnergyLedger,
    MACDeformableImmersedStatus,
)
from ._mac_viscous import MACHelmholtzResult


class MACDeformableImmersedNewmarkState(StrictModule):
    time: Array
    fluid_state: Array
    configuration: Array
    structural_velocity: Array
    structural_acceleration: Array
    pressure: Array
    marker_force_density: Array
    accepted_steps: Array
    status: Array


class MACDeformableImmersedNewmarkResult(StrictModule):
    candidate_state: MACDeformableImmersedNewmarkState
    accepted_state: MACDeformableImmersedNewmarkState
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


class MACDeformableImmersedNewmarkMethod(StrictModule, NonTrainableState):
    """Monolithic fluid/FE Newmark solve with accepted-time marker constraints."""

    base: MACDeformableImmersedBackwardEulerMethod
    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: MACDeformableImmersedBackwardEulerMethod,
        /,
        *,
        beta: float = 0.25,
        gamma: float = 0.5,
    ):
        if not isinstance(base, MACDeformableImmersedBackwardEulerMethod):
            raise TypeError("base must be MACDeformableImmersedBackwardEulerMethod.")
        beta_ = float(beta)
        gamma_ = float(gamma)
        if beta_ <= 0.0 or gamma_ <= 0.0 or 2.0 * beta_ < gamma_:
            raise ValueError(
                "Newmark parameters must be positive and satisfy 2 beta >= gamma."
            )
        self.base = base
        self.beta = beta_
        self.gamma = gamma_
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-deformable-immersed-newmark",
                "base": base.method_id,
                "beta": beta_,
                "gamma": gamma_,
            }
        )

    def initialize(
        self,
        fluid_state: ArrayLike,
        configuration: ArrayLike,
        structural_velocity: ArrayLike,
        /,
        *,
        structural_acceleration: ArrayLike | None = None,
        time: ArrayLike = 0.0,
    ) -> MACDeformableImmersedNewmarkState:
        initialized = self.base.initialize(
            fluid_state, configuration, structural_velocity, time=time
        )
        acceleration = (
            jnp.zeros_like(initialized.configuration)
            if structural_acceleration is None
            else jnp.asarray(
                structural_acceleration, dtype=initialized.configuration.dtype
            )
        )
        if acceleration.shape != initialized.configuration.shape:
            raise ValueError("Initial structural acceleration has the wrong shape.")
        return MACDeformableImmersedNewmarkState(
            initialized.time,
            initialized.fluid_state,
            initialized.configuration,
            initialized.structural_velocity,
            acceleration,
            initialized.pressure,
            initialized.marker_force_density,
            initialized.accepted_steps,
            initialized.status,
        )

    def step(
        self,
        time: ArrayLike,
        state: MACDeformableImmersedNewmarkState,
        /,
        *,
        args: Any = None,
    ) -> MACDeformableImmersedNewmarkResult:
        if not isinstance(state, MACDeformableImmersedNewmarkState):
            raise TypeError("state must be MACDeformableImmersedNewmarkState.")
        base = self.base
        step = jnp.asarray(base.step_size, dtype=state.fluid_state.dtype)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        attempted_time = time_ + step
        current_fluid = base.dynamics.validate_state(state.fluid_state)
        current_velocity = base.dynamics.unpack_velocity(current_fluid)
        components = base.dynamics.rate_components(time_, current_fluid, args)
        explicit = tuple(
            -advective + source
            for advective, source in zip(
                components.convection, components.forcing, strict=True
            )
        )
        right_hand_side = tuple(
            value + step * rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        boundary_stage = base.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        helmholtz = base.helmholtz.solve(
            right_hand_side,
            boundary_stage,
            initial_guess=current_velocity,
        )
        operators = base.dynamics.momentum.operators
        boundaries = base.dynamics.momentum.boundaries
        tentative = operators.velocity_space.flatten(helmholtz.value)
        q0 = state.configuration
        v0 = state.structural_velocity
        a0 = state.structural_acceleration
        q_predictor = q0 + step * v0 + step**2 * (0.5 - self.beta) * a0
        v_predictor = v0 + step * (1.0 - self.gamma) * a0
        marker_predictor = base.marker_map.kinematics(q_predictor, v_predictor)
        predictor_relation = base.projection.transfer.relation(marker_predictor.position)
        predictor_routes = base.projection.transfer.route_state(predictor_relation)
        initial_guess = (
            tentative,
            q_predictor + self.beta * step**2 * a0,
            v_predictor + self.gamma * step * a0,
            a0,
            state.pressure,
            state.marker_force_density,
        )

        def residual(unknown, _):
            fluid_coordinates, q, v, acceleration, pressure, multiplier = unknown
            fluid_velocity = tuple(operators.velocity_space.unflatten(fluid_coordinates))
            marker = base.marker_map.kinematics(q, v)
            relation = base.projection.transfer.relation_on_routes(
                marker.position, predictor_routes
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
            spread = base.projection.transfer.spread(relation, multiplier)
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
            structural = base.structure.evaluate(attempted_time, q, v, acceleration, args)
            contact = (
                jnp.zeros_like(structural)
                if base.structural_contact_residual is None
                else base.structural_contact_residual(q, v, args)
            )
            marker_load = base.marker_map.structural_load(multiplier)
            structural_residual = (
                structural
                + contact
                + base.marker_map.configuration_space.flatten(marker_load).reshape(
                    structural.shape
                )
            )
            q_residual = q - q_predictor - self.beta * step**2 * acceleration
            v_residual = v - v_predictor - self.gamma * step * acceleration
            divergence = operators.divergence(fluid_velocity) + pressure_mean
            slip = base.projection.transfer.gather(
                relation, fluid_velocity
            ) - base.marker_map.active_velocity(v)
            return (
                operators.velocity_space.flatten(fluid_residual),
                q_residual,
                v_residual,
                structural_residual,
                divergence,
                slip,
            )

        problem = NonlinearSystemProblem(
            residual,
            problem_id=f"mac-deformable-newmark/{self.method_id}",
        ).bind_spaces(initial_guess)
        nonlinear = base.nonlinear_method.solve(
            problem,
            initial_guess,
            termination=base.termination,
        )
        (
            fluid_coordinates,
            q,
            v,
            acceleration,
            pressure,
            multiplier,
        ) = nonlinear.state
        fluid_velocity = tuple(operators.velocity_space.unflatten(fluid_coordinates))
        marker = base.marker_map.kinematics(q, v)
        relation = base.projection.transfer.relation(marker.position)
        route_unchanged = base.projection.transfer.routes_match(
            relation, predictor_routes
        )
        slip = base.projection.transfer.gather(
            relation, fluid_velocity
        ) - base.marker_map.active_velocity(v)
        divergence = operators.divergence(fluid_velocity)
        volumes = operators.discretization.cell_volumes.astype(fluid_coordinates.dtype)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        slip_norm = jnp.sqrt(
            jnp.real(base.marker_map.markers.active_velocity_space.inner(slip, slip))
        )
        pressure_value = (
            operators.gauge_project(pressure)
            if boundaries.closure_kind == "neumann"
            else pressure
        )
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_value))
            if boundaries.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=pressure_value.dtype)
        )
        transfer = base.projection.transfer.diagnostics(
            relation, fluid_velocity, multiplier
        )
        fluid_before = 0.5 * jnp.real(
            operators.velocity_space.inner(current_velocity, current_velocity)
        )
        fluid_after = 0.5 * jnp.real(
            operators.velocity_space.inner(fluid_velocity, fluid_velocity)
        )
        structure_before = jnp.asarray(base.structural_energy(q0, args))
        structure_after = jnp.asarray(base.structural_energy(q, args))
        marker_velocity = base.projection.transfer.gather(relation, fluid_velocity)
        fluid_power = jnp.real(
            base.marker_map.markers.active_velocity_space.inner(
                marker_velocity, multiplier
            )
        )
        structural_load = base.marker_map.structural_load(multiplier)
        structural_power = -jnp.real(
            base.marker_map.configuration_space.inner(v, structural_load)
        )
        coupling_residual = fluid_power + structural_power
        energy = MACDeformableImmersedEnergyLedger(
            fluid_before,
            fluid_after,
            structure_before,
            structure_after,
            fluid_power,
            structural_power,
            coupling_residual,
            fluid_after + structure_after - fluid_before - structure_before,
        )
        kkt_norm = nonlinear.diagnostics.final_residual_norm
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.stack(
                    (
                        kkt_norm,
                        divergence_norm,
                        slip_norm,
                        jnp.abs(fluid_power),
                        jnp.abs(structural_power),
                    )
                )
            ),
        )
        tolerance = base.projection.tolerance * scale
        finite = (
            relation.successful
            & transfer.finite
            & jnp.all(jnp.isfinite(fluid_coordinates))
            & jnp.all(jnp.isfinite(q))
            & jnp.all(jnp.isfinite(v))
            & jnp.all(jnp.isfinite(acceleration))
            & jnp.all(jnp.isfinite(pressure_value))
            & jnp.all(jnp.isfinite(multiplier))
            & jnp.isfinite(coupling_residual)
        )
        checks = (
            (nonlinear.successful, MACDeformableImmersedStatus.NONLINEAR_FAILED),
            (
                relation.successful & transfer.successful,
                MACDeformableImmersedStatus.TRANSFER_FAILED,
            ),
            (route_unchanged, MACDeformableImmersedStatus.ROUTE_CHANGED),
            (
                divergence_norm <= tolerance,
                MACDeformableImmersedStatus.DIVERGENCE_FAILED,
            ),
            (slip_norm <= tolerance, MACDeformableImmersedStatus.SLIP_FAILED),
            (
                gauge_defect <= base.projection.tolerance,
                MACDeformableImmersedStatus.PRESSURE_GAUGE_FAILED,
            ),
            (
                jnp.abs(coupling_residual) <= tolerance,
                MACDeformableImmersedStatus.COUPLING_WORK_FAILED,
            ),
            (finite, MACDeformableImmersedStatus.NONFINITE),
        )
        status = jnp.asarray(int(MACDeformableImmersedStatus.SUCCESS), dtype=jnp.int32)
        accepted = jnp.asarray(True)
        for passed, flag in checks:
            accepted = accepted & passed
            status = status | jnp.where(passed, 0, int(flag)).astype(jnp.int32)
        candidate = MACDeformableImmersedNewmarkState(
            attempted_time,
            fluid_coordinates,
            q,
            v,
            acceleration,
            pressure_value,
            multiplier,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
            status,
        )
        accepted_state = jax_tree_where(accepted, candidate, state)
        return MACDeformableImmersedNewmarkResult(
            candidate,
            accepted_state,
            nonlinear,
            helmholtz,
            energy,
            transfer,
            divergence,
            slip,
            divergence_norm,
            slip_norm,
            gauge_defect,
            kkt_norm,
            status,
            route_unchanged,
            finite,
            accepted,
            self.method_id,
        )


__all__ = [
    "MACDeformableImmersedNewmarkMethod",
    "MACDeformableImmersedNewmarkResult",
    "MACDeformableImmersedNewmarkState",
]
