#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_marker_transfer import (
    MACMarkerRelation,
    MACMarkerRouteState,
    MACMarkerTransferDiagnostics,
    PreparedMACMarkerTransfer,
)
from ..discretization.particle._rigid_body import (
    rigid_body_drift,
    rigid_body_world_inertia,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ..discretization.particle._rigid_marker import (
    PreparedRigidMarkerMap,
    RigidGeneralizedVelocity,
)
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..linalg import (
    BlockSpace,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)
from ._mac_stage_inverse_general import (
    MACOperatorStageInverseMomentum,
    MACVariableDensityStageInverseMomentum,
)
from ._mac_stage_inverse_momentum import (
    MACDiagonalStageInverseMomentum,
    MACHelmholtzStageInverseMomentum,
    MACStageInverseMomentum,
    MACStageInverseMomentumDiagnostics,
)
from ._mac_viscous import MACHelmholtzResult, MACHelmholtzSolvePlan
from ._structured_incompressible import MACPressureClosureReport


class MACRigidImmersedStatus(IntFlag):
    SUCCESS = 0
    BOUNDARY_FAILED = 1
    TRANSFER_FAILED = 2
    LINEAR_SOLVE_FAILED = 4
    DIVERGENCE_FAILED = 8
    SLIP_FAILED = 16
    PRESSURE_GAUGE_FAILED = 32
    KKT_RESIDUAL_FAILED = 64
    ROUTE_CHANGED = 128
    NONFINITE = 256


class MACRigidImmersedProjectionResult(StrictModule):
    fluid_velocity: FaceVelocity
    body_kinematics: RigidBodyKinematics
    pressure: Array
    marker_force_density: Array
    marker_slip: Array
    divergence: Array
    relation: MACMarkerRelation
    route_state: MACMarkerRouteState
    transfer_diagnostics: MACMarkerTransferDiagnostics
    inverse_momentum_diagnostics: MACStageInverseMomentumDiagnostics
    stage_inverse_id: str = eqx.field(static=True)
    kkt_residual_norm: Array
    divergence_norm: Array
    slip_norm: Array
    gauge_defect: Array
    route_unchanged: Array
    status: Array
    linear: LinearSolveResult
    closure: MACPressureClosureReport
    finite: Array
    converged: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class MACRigidImmersedProjectionPlan(StrictModule, NonTrainableState):
    """Simultaneous unit-density MAC and generic rigid-body velocity projection."""

    dynamics: CompiledMACIncompressibleDynamics
    rigid_markers: PreparedRigidMarkerMap
    transfer: PreparedMACMarkerTransfer
    constraint_length: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        rigid_markers: PreparedRigidMarkerMap,
        transfer: PreparedMACMarkerTransfer,
        /,
        *,
        constraint_length: float,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):

        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(rigid_markers, PreparedRigidMarkerMap):
            raise TypeError("rigid_markers must be PreparedRigidMarkerMap.")
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        if transfer.markers.prepared_id != rigid_markers.markers.prepared_id:
            raise ValueError("Rigid map and transfer must share markers.")
        if transfer.operators.prepared_id != dynamics.momentum.operators.prepared_id:
            raise ValueError("Rigid projection and fluid must share MAC operators.")
        length = float(constraint_length)
        tolerance_ = float(tolerance)
        if length <= 0.0 or tolerance_ <= 0.0:
            raise ValueError("constraint_length and tolerance must be positive.")
        policy = (
            LinearSolvePolicy(
                GMRES(restart=min(50, int(maximum_iterations))),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=int(maximum_iterations),
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )
        self.dynamics = dynamics
        self.rigid_markers = rigid_markers
        self.transfer = transfer
        self.constraint_length = length
        self.tolerance = tolerance_
        self.linear_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-projection",
                "dynamics": dynamics.compilation_id,
                "rigid_markers": rigid_markers.prepared_id,
                "transfer": transfer.prepared_id,
                "constraint_length": length,
                "linear_method": policy.method.name,
            }
        )

    def project(
        self,
        fluid_velocity: FaceVelocity,
        body_kinematics: RigidBodyKinematics,
        inverse_momentum: MACStageInverseMomentum | ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        args: Any = None,
        expected_routes: MACMarkerRouteState | None = None,
        allow_route_refresh: bool = False,
        body_inverse_coefficient: ArrayLike | None = None,
    ) -> MACRigidImmersedProjectionResult:
        operators = self.dynamics.momentum.operators
        boundaries = self.dynamics.momentum.boundaries
        stage = boundaries.evaluate(time, args)
        stage_inverse = (
            inverse_momentum
            if isinstance(
                inverse_momentum,
                (
                    MACDiagonalStageInverseMomentum,
                    MACHelmholtzStageInverseMomentum,
                    MACOperatorStageInverseMomentum,
                    MACVariableDensityStageInverseMomentum,
                ),
            )
            else MACDiagonalStageInverseMomentum(
                operators,
                boundaries,
                stage,
                inverse_momentum,
            )
        )
        if stage_inverse.operators.prepared_id != operators.prepared_id:
            raise ValueError("Rigid stage inverse and fluid operators differ.")
        if body_inverse_coefficient is None and isinstance(
            inverse_momentum,
            (
                MACDiagonalStageInverseMomentum,
                MACHelmholtzStageInverseMomentum,
                MACOperatorStageInverseMomentum,
                MACVariableDensityStageInverseMomentum,
            ),
        ):
            raise ValueError("body_inverse_coefficient is required with a stage inverse.")
        body_coefficient = (
            jnp.asarray(inverse_momentum, dtype=operators.pressure_space.dtype)
            if body_inverse_coefficient is None
            and not isinstance(
                inverse_momentum,
                (
                    MACDiagonalStageInverseMomentum,
                    MACHelmholtzStageInverseMomentum,
                    MACOperatorStageInverseMomentum,
                    MACVariableDensityStageInverseMomentum,
                ),
            )
            else jnp.asarray(
                body_inverse_coefficient,
                dtype=operators.pressure_space.dtype,
            )
        ).reshape(())
        body_coefficient = eqx.error_if(
            body_coefficient,
            ~jnp.isfinite(body_coefficient) | (body_coefficient <= 0.0),
            "Rigid-body inverse coefficient must be positive and finite.",
        )
        bounded = boundaries.enforce(operators.validate_velocity(fluid_velocity), stage)
        marker_state = self.rigid_markers.evaluate(body_kinematics)
        relation = self.transfer.relation(marker_state.position)
        marker_before = self.transfer.gather(relation, bounded)
        route_state = self.transfer.route_state(relation)
        route_unchanged = (
            jnp.asarray(True)
            if expected_routes is None
            else self.transfer.routes_match(relation, expected_routes)
        )
        route_acceptable = route_unchanged | bool(allow_route_refresh)
        rigid_velocity = self.rigid_markers.generalized_velocity(body_kinematics)
        k_operator = self.rigid_markers.velocity_operator(body_kinematics)
        rigid_marker_velocity = k_operator.mv(rigid_velocity)
        ell = jnp.asarray(self.constraint_length, dtype=operators.pressure_space.dtype)
        incoming_pressure = (
            jnp.zeros(operators.discretization.cell_shape, dtype=ell.dtype)
            if pressure is None
            else operators.validate_pressure(pressure)
        )
        if boundaries.closure_kind == "neumann":
            incoming_pressure = operators.gauge_project(incoming_pressure)
        initial_multiplier = (
            jnp.zeros_like(marker_before)
            if marker_force_density is None
            else self.transfer.markers.active_velocity_space.validate(
                jnp.asarray(marker_force_density)
            )
        )
        dual_space = BlockSpace(
            (operators.pressure_space, self.transfer.markers.active_velocity_space),
            names=("pressure", "marker"),
        )
        mobile = self.rigid_markers.mobile_indices
        inverse_mass = self.rigid_markers.bodies.inverse_masses[mobile]
        if self.rigid_markers.bodies.ambient_dimension == 2:
            inverse_inertia_mobile = self.rigid_markers.bodies.inverse_inertia_body[
                mobile
            ]

            def angular_inverse(rotation):
                return inverse_inertia_mobile[:, None] * rotation

        else:
            _, inverse_inertia = rigid_body_world_inertia(
                self.rigid_markers.bodies, body_kinematics.orientation
            )
            inverse_inertia_mobile = inverse_inertia[mobile]

            def angular_inverse(rotation):
                return contract(
                    "...ij,...j->...i",
                    inverse_inertia_mobile,
                    rotation,
                )

        def body_inverse(load: RigidGeneralizedVelocity):
            return RigidGeneralizedVelocity(
                body_coefficient * inverse_mass[:, None] * load.translation,
                body_coefficient * angular_inverse(load.rotation),
            )

        def response(dual):
            scaled_pressure, multiplier = dual
            if boundaries.closure_kind == "neumann":
                projected_pressure = operators.gauge_project(scaled_pressure)
                volumes = operators.discretization.cell_volumes.astype(ell.dtype)
                mean = jnp.sum(volumes * scaled_pressure) / jnp.sum(volumes)
            else:
                projected_pressure = scaled_pressure
                mean = jnp.asarray(0.0, dtype=ell.dtype)
            gradient = boundaries.pressure_gradient(
                ell * projected_pressure,
                stage,
                homogeneous=boundaries.closure_kind == "neumann",
            )
            spread = self.transfer.spread(relation, multiplier)
            fluid_rhs = tuple(
                derivative - force
                for derivative, force in zip(gradient, spread, strict=True)
            )
            fluid_mass_image = boundaries.homogeneous_rate(
                stage_inverse.apply_inverse(fluid_rhs)
            )
            body_load = k_operator.adjoint_mv(multiplier)
            body_mass_image = body_inverse(body_load)
            return fluid_mass_image, body_mass_image, mean, fluid_rhs

        def action(dual):
            fluid_image, body_image, mean, _ = response(dual)
            pressure_image = -ell * operators.divergence(fluid_image)
            if boundaries.closure_kind == "neumann":
                pressure_image = pressure_image + mean
            marker_image = k_operator.mv(body_image) - self.transfer.gather(
                relation, fluid_image
            )
            return pressure_image, marker_image

        operator = FunctionLinearOperator(
            action,
            source=dual_space,
            target=dual_space,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"mac-rigid-immersed-dual/{self.plan_id}",
        )
        pressure_rhs = -ell * operators.divergence(bounded)
        if boundaries.closure_kind == "neumann":
            pressure_rhs = operators.compatibility_project(pressure_rhs)
        marker_rhs = rigid_marker_velocity - marker_before
        linear = solve(
            LinearSystem(operator, problem_id=f"mac-rigid-immersed/{self.plan_id}"),
            (pressure_rhs, marker_rhs),
            policy=self.linear_policy,
            initial_guess=(incoming_pressure / ell, initial_multiplier),
        )
        fluid_image, body_image, _, fluid_rhs = response(linear.value)
        candidate_fluid = tuple(
            value - image for value, image in zip(bounded, fluid_image, strict=True)
        )
        candidate_generalized = RigidGeneralizedVelocity(
            rigid_velocity.translation - body_image.translation,
            rigid_velocity.rotation - body_image.rotation,
        )
        candidate_body = self.rigid_markers.with_generalized_velocity(
            body_kinematics, candidate_generalized
        )
        pressure_candidate = ell * linear.value[0]
        if boundaries.closure_kind == "neumann":
            pressure_candidate = operators.gauge_project(
                incoming_pressure + pressure_candidate
            )
        marker_after = self.transfer.gather(relation, candidate_fluid)
        body_marker_after = k_operator.mv(candidate_generalized)
        slip = marker_after - body_marker_after
        divergence = operators.divergence(candidate_fluid)
        residual = operator(linear.value)
        residual_value = (
            residual[0] - pressure_rhs,
            residual[1] - marker_rhs,
        )
        residual_norm = jnp.sqrt(
            jnp.real(dual_space.inner(residual_value, residual_value))
        )
        volumes = operators.discretization.cell_volumes.astype(ell.dtype)
        scale = jnp.maximum(
            1.0,
            jnp.sqrt(
                jnp.sum(volumes * pressure_rhs**2)
                + jnp.real(
                    self.transfer.markers.active_velocity_space.inner(
                        marker_rhs, marker_rhs
                    )
                )
            ),
        )
        tolerance = self.tolerance * scale
        divergence_tolerance = tolerance / ell
        transfer_diagnostics = self.transfer.diagnostics(
            relation, candidate_fluid, linear.value[1]
        )
        inverse_momentum_diagnostics = stage_inverse.diagnostics(fluid_rhs, fluid_image)
        divergence_norm = jnp.sqrt(jnp.sum(volumes * divergence**2))
        slip_norm = jnp.sqrt(
            jnp.real(self.transfer.markers.active_velocity_space.inner(slip, slip))
        )
        gauge_defect = (
            jnp.abs(jnp.sum(volumes * pressure_candidate))
            if boundaries.closure_kind == "neumann"
            else jnp.asarray(0.0, dtype=ell.dtype)
        )
        finite = (
            stage.finite
            & transfer_diagnostics.finite
            & inverse_momentum_diagnostics.finite
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in candidate_fluid)
                )
            )
            & jnp.all(jnp.isfinite(candidate_body.velocity))
            & jnp.all(jnp.isfinite(candidate_body.angular_velocity))
            & jnp.all(jnp.isfinite(pressure_candidate))
            & jnp.all(jnp.isfinite(linear.value[1]))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(gauge_defect)
        )
        checks = (
            (stage.successful, MACRigidImmersedStatus.BOUNDARY_FAILED),
            (
                relation.successful & transfer_diagnostics.successful,
                MACRigidImmersedStatus.TRANSFER_FAILED,
            ),
            (linear.successful, MACRigidImmersedStatus.LINEAR_SOLVE_FAILED),
            (
                inverse_momentum_diagnostics.converged,
                MACRigidImmersedStatus.LINEAR_SOLVE_FAILED,
            ),
            (
                divergence_norm <= divergence_tolerance,
                MACRigidImmersedStatus.DIVERGENCE_FAILED,
            ),
            (slip_norm <= tolerance, MACRigidImmersedStatus.SLIP_FAILED),
            (
                gauge_defect <= self.tolerance,
                MACRigidImmersedStatus.PRESSURE_GAUGE_FAILED,
            ),
            (
                residual_norm <= tolerance,
                MACRigidImmersedStatus.KKT_RESIDUAL_FAILED,
            ),
            (route_acceptable, MACRigidImmersedStatus.ROUTE_CHANGED),
            (finite, MACRigidImmersedStatus.NONFINITE),
        )
        status = jnp.asarray(int(MACRigidImmersedStatus.SUCCESS), dtype=jnp.int32)
        converged = jnp.asarray(True)
        for passed, flag in checks:
            converged = converged & passed
            status = status | jnp.where(passed, 0, int(flag)).astype(jnp.int32)
        accepted_fluid = tuple(
            jnp.where(converged, candidate, original)
            for candidate, original in zip(candidate_fluid, bounded, strict=True)
        )
        accepted_body = jax_tree_where(converged, candidate_body, body_kinematics)
        pressure_value = jnp.where(converged, pressure_candidate, incoming_pressure)
        multiplier_value = jnp.where(converged, linear.value[1], initial_multiplier)
        closure = MACPressureClosureReport(
            kind=boundaries.closure_kind,
            gauge="zero-mean" if boundaries.closure_kind == "neumann" else "none",
            compatibility="projected"
            if boundaries.closure_kind == "neumann"
            else "unprojected",
            integrated_mass_flux=jnp.sum(volumes * operators.divergence(accepted_fluid)),
            mass_defect=jnp.abs(jnp.sum(volumes * operators.divergence(accepted_fluid))),
            gauge_defect=gauge_defect,
            finite=finite,
            successful=converged,
            closure_id=self.plan_id,
        )
        return MACRigidImmersedProjectionResult(
            fluid_velocity=accepted_fluid,
            body_kinematics=accepted_body,
            pressure=pressure_value,
            marker_force_density=multiplier_value,
            marker_slip=slip,
            divergence=operators.divergence(accepted_fluid),
            relation=relation,
            route_state=route_state,
            transfer_diagnostics=transfer_diagnostics,
            inverse_momentum_diagnostics=inverse_momentum_diagnostics,
            stage_inverse_id=stage_inverse.stage_id,
            kkt_residual_norm=residual_norm,
            divergence_norm=divergence_norm,
            slip_norm=slip_norm,
            gauge_defect=gauge_defect,
            route_unchanged=route_unchanged,
            status=status,
            linear=linear,
            closure=closure,
            finite=finite,
            converged=converged,
            plan_id=self.plan_id,
        )


class MACRigidImmersedEnergyLedger(StrictModule):
    fluid_kinetic_before: Array
    fluid_kinetic_after: Array
    rigid_kinetic_before: Array
    rigid_kinetic_after: Array
    fluid_coupling_power: Array
    rigid_coupling_power: Array
    coupling_power_residual: Array
    external_work: Array
    total_energy_change: Array


class MACRigidImmersedStepResult(StrictModule):
    fluid_state: Array
    body_kinematics: RigidBodyKinematics
    pressure: Array
    marker_force_density: Array
    helmholtz: MACHelmholtzResult
    projection: MACRigidImmersedProjectionResult
    time: Array
    attempted_time: Array
    energy: MACRigidImmersedEnergyLedger
    status: Array
    accepted: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACRigidImmersedEulerMethod(StrictModule, NonTrainableState):
    """Contact-free synchronized fluid/rigid Euler step with exact velocity coupling."""

    dynamics: CompiledMACIncompressibleDynamics
    projection: MACRigidImmersedProjectionPlan
    helmholtz: MACHelmholtzSolvePlan
    step_size: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        projection: MACRigidImmersedProjectionPlan,
        step_size: float,
        /,
    ):
        step = float(step_size)
        if step <= 0.0:
            raise ValueError("step_size must be positive.")
        if projection.dynamics.compilation_id != dynamics.compilation_id:
            raise ValueError("Rigid projection and method must share dynamics.")
        viscosity = float(jnp.asarray(dynamics.problem.viscosity))
        self.dynamics = dynamics
        self.projection = projection
        self.helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            fixed_mass_coefficient=1.0,
            fixed_diffusion_coefficient=step * viscosity,
        )
        self.step_size = step
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-euler",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "step_size": step,
            }
        )

    def _rigid_kinetic(self, kinematics: RigidBodyKinematics, /) -> Array:
        bodies = self.projection.rigid_markers.bodies
        mobile = self.projection.rigid_markers.mobile_indices
        linear = 0.5 * jnp.sum(
            bodies.particles.safe_masses[mobile, None] * kinematics.velocity[mobile] ** 2
        )
        if bodies.ambient_dimension == 2:
            angular = 0.5 * jnp.sum(
                bodies.inertia_body[mobile, None]
                * kinematics.angular_velocity[mobile] ** 2
            )
        else:
            inertia, _ = rigid_body_world_inertia(bodies, kinematics.orientation)
            angular = 0.5 * jnp.sum(
                kinematics.angular_velocity[mobile]
                * contract(
                    "...ij,...j->...i",
                    inertia[mobile],
                    kinematics.angular_velocity[mobile],
                )
            )
        return linear + angular

    def step(
        self,
        time: ArrayLike,
        fluid_state: ArrayLike,
        body_kinematics: RigidBodyKinematics,
        /,
        *,
        body_load: RigidBodyLoad | None = None,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        args: Any = None,
        expected_routes: MACMarkerRouteState | None = None,
        allow_route_refresh: bool = False,
    ) -> MACRigidImmersedStepResult:
        step = jnp.asarray(
            self.step_size,
            dtype=self.dynamics.momentum.operators.pressure_space.dtype,
        )
        current = self.dynamics.validate_state(fluid_state)
        velocity = self.dynamics.unpack_velocity(current)
        _, convection, _, forcing = self.dynamics.rate_components(time, current, args)
        explicit = tuple(
            -advective + source
            for advective, source in zip(convection, forcing, strict=True)
        )
        rhs = tuple(
            value + step * rate for value, rate in zip(velocity, explicit, strict=True)
        )
        attempted_time = jnp.asarray(time, dtype=step.dtype) + step
        stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        helmholtz = self.helmholtz.solve(rhs, stage, initial_guess=velocity)
        stage_inverse = MACHelmholtzStageInverseMomentum(
            self.helmholtz,
            stage,
            rhs_scale=step,
            stage_id=f"{self.method_id}/accepted-stage",
        )
        predicted_pose = rigid_body_drift(
            self.projection.rigid_markers.bodies, body_kinematics, step
        )
        if body_load is not None:
            bodies = self.projection.rigid_markers.bodies
            mobile = self.projection.rigid_markers.mobile_indices
            if bodies.ambient_dimension == 2:
                angular_increment = (
                    step
                    * bodies.inverse_inertia_body[mobile, None]
                    * body_load.torque[mobile]
                )
            else:
                _, inverse_inertia = rigid_body_world_inertia(
                    bodies, predicted_pose.orientation
                )
                angular_increment = step * contract(
                    "...ij,...j->...i",
                    inverse_inertia[mobile],
                    body_load.torque[mobile],
                )
            predicted_pose = RigidBodyKinematics(
                predicted_pose.position,
                predicted_pose.velocity.at[mobile].add(
                    step
                    * bodies.inverse_masses[mobile][:, None]
                    * body_load.force[mobile]
                ),
                predicted_pose.orientation,
                predicted_pose.angular_velocity.at[mobile].add(angular_increment),
            )
        projected = self.projection.project(
            helmholtz.value,
            predicted_pose,
            stage_inverse,
            pressure=pressure,
            marker_force_density=marker_force_density,
            time=attempted_time,
            args=args,
            expected_routes=expected_routes,
            allow_route_refresh=allow_route_refresh,
            body_inverse_coefficient=step,
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            projected.fluid_velocity
        )
        accepted = helmholtz.converged & projected.converged
        fluid_before = 0.5 * jnp.real(
            self.dynamics.momentum.operators.velocity_space.inner(velocity, velocity)
        )
        fluid_after = 0.5 * jnp.real(
            self.dynamics.momentum.operators.velocity_space.inner(
                projected.fluid_velocity, projected.fluid_velocity
            )
        )
        rigid_before = self._rigid_kinetic(body_kinematics)
        rigid_after = self._rigid_kinetic(projected.body_kinematics)
        marker_velocity = self.projection.transfer.gather(
            projected.relation, projected.fluid_velocity
        )
        fluid_power = jnp.real(
            self.projection.transfer.markers.active_velocity_space.inner(
                marker_velocity, projected.marker_force_density
            )
        )
        rigid_operator = self.projection.rigid_markers.velocity_operator(
            projected.body_kinematics
        )
        rigid_marker_velocity = rigid_operator.mv(
            self.projection.rigid_markers.generalized_velocity(projected.body_kinematics)
        )
        rigid_power = -jnp.real(
            self.projection.transfer.markers.active_velocity_space.inner(
                rigid_marker_velocity, projected.marker_force_density
            )
        )
        external_work = (
            jnp.asarray(0.0, dtype=step.dtype)
            if body_load is None
            else step
            * (
                jnp.sum(body_load.force * projected.body_kinematics.velocity)
                + jnp.sum(body_load.torque * projected.body_kinematics.angular_velocity)
            )
        )
        energy = MACRigidImmersedEnergyLedger(
            fluid_before,
            fluid_after,
            rigid_before,
            rigid_after,
            fluid_power,
            rigid_power,
            fluid_power + rigid_power,
            external_work,
            fluid_after + rigid_after - fluid_before - rigid_before,
        )
        status = jnp.where(
            helmholtz.converged,
            projected.status,
            projected.status | int(MACRigidImmersedStatus.LINEAR_SOLVE_FAILED),
        ).astype(jnp.int32)
        return MACRigidImmersedStepResult(
            fluid_state=jnp.where(accepted, candidate_state, current),
            body_kinematics=jax_tree_where(
                accepted, projected.body_kinematics, body_kinematics
            ),
            pressure=projected.pressure,
            marker_force_density=projected.marker_force_density,
            helmholtz=helmholtz,
            projection=projected,
            time=jnp.where(accepted, attempted_time, jnp.asarray(time)),
            attempted_time=attempted_time,
            energy=energy,
            status=status,
            accepted=accepted,
            method_id=self.method_id,
        )


class MACRigidImmersedBackwardEulerMethod(StrictModule, NonTrainableState):
    """Accepted-time fixed-point backward-Euler rigid immersed coupling."""

    base: MACRigidImmersedEulerMethod
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: MACRigidImmersedEulerMethod,
        /,
        *,
        maximum_iterations: int = 8,
        tolerance: float = 1.0e-9,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        if iterations <= 0 or tolerance_ <= 0.0:
            raise ValueError("Rigid fixed-point limits must be positive.")
        self.base = base
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-backward-euler",
                "base": base.method_id,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
            }
        )

    def step(
        self,
        time: ArrayLike,
        fluid_state: ArrayLike,
        body_kinematics: RigidBodyKinematics,
        /,
        *,
        body_load: RigidBodyLoad | None = None,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        args: Any = None,
    ) -> MACRigidImmersedStepResult:
        guess = body_kinematics
        previous_velocity = guess.velocity
        previous_angular = guess.angular_velocity
        result = None
        route_state = None
        residual = jnp.asarray(jnp.inf, dtype=guess.position.dtype)
        for _ in range(self.maximum_iterations):
            iteration_state = RigidBodyKinematics(
                body_kinematics.position,
                guess.velocity,
                body_kinematics.orientation,
                guess.angular_velocity,
            )
            result = self.base.step(
                time,
                fluid_state,
                iteration_state,
                body_load=body_load,
                pressure=pressure,
                marker_force_density=marker_force_density,
                args=args,
                expected_routes=route_state,
                allow_route_refresh=True,
            )
            route_state = result.projection.route_state
            residual = jnp.maximum(
                jnp.max(jnp.abs(result.body_kinematics.velocity - previous_velocity)),
                jnp.max(
                    jnp.abs(result.body_kinematics.angular_velocity - previous_angular)
                ),
            )
            previous_velocity = result.body_kinematics.velocity
            previous_angular = result.body_kinematics.angular_velocity
            guess = result.body_kinematics
        if result is None:
            raise RuntimeError("Rigid backward-Euler iteration did not execute.")
        converged = result.accepted & (residual <= self.tolerance)
        status = result.status | jnp.where(
            residual <= self.tolerance,
            0,
            int(MACRigidImmersedStatus.KKT_RESIDUAL_FAILED),
        ).astype(jnp.int32)
        return MACRigidImmersedStepResult(
            fluid_state=jnp.where(
                converged, result.fluid_state, jnp.asarray(fluid_state)
            ),
            body_kinematics=jax_tree_where(
                converged, result.body_kinematics, body_kinematics
            ),
            pressure=result.pressure,
            marker_force_density=result.marker_force_density,
            helmholtz=result.helmholtz,
            projection=result.projection,
            time=jnp.where(converged, result.time, jnp.asarray(time)),
            attempted_time=result.attempted_time,
            energy=result.energy,
            status=status,
            accepted=converged,
            method_id=self.method_id,
        )


class MACRigidImmersedMidpointMethod(StrictModule, NonTrainableState):
    """Second-order pose-centred rigid coupling built on accepted-time iterations."""

    backward_euler: MACRigidImmersedBackwardEulerMethod
    method_id: str = eqx.field(static=True)

    def __init__(self, backward_euler: MACRigidImmersedBackwardEulerMethod, /):
        self.backward_euler = backward_euler
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-midpoint",
                "backward_euler": backward_euler.method_id,
            }
        )

    def step(self, *args, **kwargs) -> MACRigidImmersedStepResult:
        first = self.backward_euler.step(*args, **kwargs)
        initial = args[2]
        midpoint_velocity = 0.5 * (initial.velocity + first.body_kinematics.velocity)
        midpoint_angular = 0.5 * (
            initial.angular_velocity + first.body_kinematics.angular_velocity
        )
        midpoint_state = RigidBodyKinematics(
            initial.position,
            midpoint_velocity,
            initial.orientation,
            midpoint_angular,
        )
        second_args = (args[0], args[1], midpoint_state)
        second = self.backward_euler.step(
            *second_args,
            **kwargs,
        )
        return MACRigidImmersedStepResult(
            fluid_state=second.fluid_state,
            body_kinematics=second.body_kinematics,
            pressure=second.pressure,
            marker_force_density=second.marker_force_density,
            helmholtz=second.helmholtz,
            projection=second.projection,
            time=second.time,
            attempted_time=second.attempted_time,
            energy=second.energy,
            status=second.status,
            accepted=second.accepted,
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
    "MACRigidImmersedBackwardEulerMethod",
    "MACRigidImmersedEulerMethod",
    "MACRigidImmersedEnergyLedger",
    "MACRigidImmersedMidpointMethod",
    "MACRigidImmersedProjectionPlan",
    "MACRigidImmersedProjectionResult",
    "MACRigidImmersedStatus",
    "MACRigidImmersedStepResult",
]
