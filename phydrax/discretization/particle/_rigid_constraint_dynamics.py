#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from math import isfinite, pi
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...linalg import (
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    OperatorProperties,
    PyTreeSpace,
    RankPolicy,
    saddle_point_system,
    solve as solve_linear,
    svd as svd_linalg,
    TolerancePolicy,
)
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)
from ._rigid_body import (
    _quaternion_relative_rotation_vector,
    _rigid_body_close_kick,
    _rigid_body_drift,
    _rigid_body_half_kick,
    _rigid_body_world_inertia,
    PreparedRigidBodySet,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._rigid_joints import (
    _RigidMobileIncrement,
    PreparedRigidJointGraph,
    rigid_joint_maximum_residual,
    rigid_joint_pairing,
    RigidJointGraphPlan,
    RigidJointMultipliers,
    RigidJointResiduals,
)


ExternalRigidBodyLoad = Callable[[Array, RigidBodyKinematics, Any], RigidBodyLoad]


class RigidConstraintRejectionReason(IntFlag):
    NONE = 0
    INVALID_STEP = 1 << 0
    INVALID_STATE = 1 << 1
    INITIAL_LOAD = 1 << 2
    POSITION_SOLVE = 1 << 3
    POSITION_CONSTRAINT = 1 << 4
    POSITION_STATIONARITY = 1 << 5
    ROTATION_CHART = 1 << 6
    HINGE_ALIGNMENT = 1 << 7
    CLOSING_LOAD = 1 << 8
    VELOCITY_SOLVE = 1 << 9
    VELOCITY_CONSTRAINT = 1 << 10
    FIXED_BODY = 1 << 11
    QUATERNION = 1 << 12
    ENERGY_PROJECTION = 1 << 13
    NONFINITE = 1 << 14
    RANK_OR_CONDITION = 1 << 15


class RigidConstraintSolverPlan(StrictModule, NonTrainableState):
    position_method: AbstractNonlinearMethod
    position_termination: NonlinearTermination
    velocity_policy: LinearSolvePolicy
    characteristic_length: float = eqx.field(static=True)
    position_tolerance: float = eqx.field(static=True)
    stationarity_tolerance: float = eqx.field(static=True)
    velocity_tolerance: float = eqx.field(static=True)
    quaternion_tolerance: float = eqx.field(static=True)
    fixed_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    hinge_alignment_margin: float = eqx.field(static=True)
    rotation_chart_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        position_method: AbstractNonlinearMethod | None = None,
        position_termination: NonlinearTermination | None = None,
        velocity_policy: LinearSolvePolicy | None = None,
        characteristic_length: float = 1.0,
        position_tolerance: float = 1.0e-8,
        stationarity_tolerance: float = 1.0e-8,
        velocity_tolerance: float = 1.0e-8,
        quaternion_tolerance: float = 1.0e-8,
        fixed_tolerance: float = 1.0e-12,
        energy_tolerance: float = 1.0e-10,
        rank_tolerance: float = 1.0e-10,
        hinge_alignment_margin: float = 1.0e-6,
        rotation_chart_margin: float = 1.0e-4,
        plan_id: str | None = None,
    ):
        default_linear = LinearSolvePolicy(
            GMRES(restart=32, stagnation_iterations=32),
            tolerance=TolerancePolicy(relative=1.0e-9, absolute=1.0e-11, max_steps=256),
            rank=RankPolicy(),
        )
        method = (
            NewtonKrylov(linear_policy=default_linear)
            if position_method is None
            else position_method
        )
        termination = (
            NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=1.0e-9,
                absolute_step=1.0e-12,
                relative_step=1.0e-10,
                maximum_steps=32,
                maximum_evaluations=128,
                maximum_linear_iterations=256,
            )
            if position_termination is None
            else position_termination
        )
        velocity = default_linear if velocity_policy is None else velocity_policy
        if not isinstance(method, AbstractNonlinearMethod):
            raise TypeError("position_method must be an AbstractNonlinearMethod or None.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("position_termination must be NonlinearTermination or None.")
        if not isinstance(velocity, LinearSolvePolicy):
            raise TypeError("velocity_policy must be LinearSolvePolicy or None.")
        positive = tuple(
            float(value)
            for value in (
                characteristic_length,
                position_tolerance,
                stationarity_tolerance,
                velocity_tolerance,
                quaternion_tolerance,
                fixed_tolerance,
                energy_tolerance,
                rank_tolerance,
                hinge_alignment_margin,
                rotation_chart_margin,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "Rigid constraint scales and tolerances must be positive finite values."
            )
        if positive[-3] >= 1.0:
            raise ValueError("rank_tolerance must be smaller than one.")
        if positive[-2] >= 1.0:
            raise ValueError("hinge_alignment_margin must be smaller than one.")
        if positive[-1] >= pi:
            raise ValueError("rotation_chart_margin must be smaller than pi.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-constraint-solver-plan",
                "position_method": method.method_id,
                "position_termination": {
                    "absolute_residual": termination.absolute_residual,
                    "relative_residual": termination.relative_residual,
                    "maximum_steps": termination.maximum_steps,
                },
                "velocity_method": velocity.method.name,
                "values": positive,
            }
        )
        self.position_method = method
        self.position_termination = termination
        self.velocity_policy = velocity
        (
            self.characteristic_length,
            self.position_tolerance,
            self.stationarity_tolerance,
            self.velocity_tolerance,
            self.quaternion_tolerance,
            self.fixed_tolerance,
            self.energy_tolerance,
            self.rank_tolerance,
            self.hinge_alignment_margin,
            self.rotation_chart_margin,
        ) = positive
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")


class RigidConstraintDynamicsPlan(StrictModule, NonTrainableState):
    joints: RigidJointGraphPlan
    solver: RigidConstraintSolverPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        joints: RigidJointGraphPlan,
        /,
        *,
        solver: RigidConstraintSolverPlan | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(joints, RigidJointGraphPlan):
            raise TypeError("joints must be a RigidJointGraphPlan.")
        solver_ = RigidConstraintSolverPlan() if solver is None else solver
        if not isinstance(solver_, RigidConstraintSolverPlan):
            raise TypeError("solver must be a RigidConstraintSolverPlan or None.")
        generated = canonical_fingerprint(
            {
                "kind": "rigid-constraint-dynamics-plan",
                "joints": joints.plan_id,
                "solver": solver_.plan_id,
            }
        )
        self.joints = joints
        self.solver = solver_
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self,
        bodies: PreparedRigidBodySet,
        reference: RigidBodyKinematics,
        /,
        *,
        external_load: ExternalRigidBodyLoad | None = None,
        external_load_id: str | None = None,
    ) -> PreparedRigidConstraintDynamics:
        if external_load is not None and not callable(external_load):
            raise TypeError("external_load must be callable or None.")
        if external_load is None and external_load_id is not None:
            raise ValueError("external_load_id requires external_load.")
        if external_load is not None and not external_load_id:
            raise ValueError("External rigid-body load requires a stable nonempty ID.")
        return PreparedRigidConstraintDynamics(
            bodies,
            self.joints.prepare(bodies, reference),
            self.solver,
            reference,
            external_load,
            None if external_load_id is None else str(external_load_id),
            self.plan_id,
        )


class RigidConstraintState(StrictModule):
    kinematics: RigidBodyKinematics
    position_multiplier_guess: RigidJointMultipliers
    velocity_multiplier_guess: RigidJointMultipliers


class RigidConstraintDiagnostics(StrictModule):
    maximum_position_residual: Array
    maximum_stationarity_residual: Array
    maximum_velocity_residual: Array
    maximum_rotation_increment: Array
    minimum_hinge_alignment: Array
    quaternion_defect: Array
    fixed_pose_defect: Array
    fixed_velocity_defect: Array
    kinetic_energy_before_projection: Array
    kinetic_energy_after_projection: Array
    velocity_projection_energy_increase: Array
    constraint_rank: Array
    constraint_condition: Array
    finite: Array
    locally_valid: Array


class RigidConstraintEvaluation(StrictModule):
    initial_load: RigidBodyLoad
    closing_load: RigidBodyLoad
    position_result: NonlinearResult | None
    velocity_result: LinearSolveResult | None
    position_residuals: RigidJointResiduals
    velocity_residuals: RigidJointResiduals
    position_multipliers: RigidJointMultipliers
    velocity_multipliers: RigidJointMultipliers
    diagnostics: RigidConstraintDiagnostics
    successful: Array
    rejection_reasons: Array
    prepared_id: str = eqx.field(static=True)


class RigidConstraintStepResult(StrictModule):
    candidate_state: RigidConstraintState
    accepted_state: RigidConstraintState
    evaluation: RigidConstraintEvaluation
    successful: Array
    rejection_reasons: Array


class _PositionProjectionUnknown(StrictModule):
    increment: _RigidMobileIncrement
    multipliers: RigidJointMultipliers


class _PositionProjectionResidual(StrictModule):
    stationarity: _RigidMobileIncrement
    constraints: RigidJointResiduals


class _PositionProjectionArguments(StrictModule):
    predicted: RigidBodyKinematics
    inertia_world: Array


class _PositionProjectionAuxiliary(StrictModule):
    kinematics: RigidBodyKinematics
    physical_residuals: RigidJointResiduals
    minimum_hinge_alignment: Array
    maximum_rotation_increment: Array


def _scaled_residuals(
    residuals: RigidJointResiduals,
    length_scale: float,
    /,
) -> RigidJointResiduals:
    return RigidJointResiduals(
        residuals.fixed_translation / length_scale,
        residuals.fixed_rotation,
        residuals.ball_anchor / length_scale,
        residuals.hinge_anchor / length_scale,
        residuals.hinge_axis,
    )


def _multipliers_to_residuals(value: RigidJointMultipliers, /) -> RigidJointResiduals:
    return RigidJointResiduals(
        value.fixed_translation,
        value.fixed_rotation,
        value.ball_anchor,
        value.hinge_anchor,
        value.hinge_axis,
    )


def _residuals_to_multipliers(value: RigidJointResiduals, /) -> RigidJointMultipliers:
    return RigidJointMultipliers(
        value.fixed_translation,
        value.fixed_rotation,
        value.ball_anchor,
        value.hinge_anchor,
        value.hinge_axis,
    )


def _negate_residuals(value: RigidJointResiduals, /) -> RigidJointResiduals:
    return jax.tree.map(lambda item: -item, value)


def _maximum_increment(value: _RigidMobileIncrement, /) -> Array:
    translation = jnp.max(jnp.abs(value.translation), initial=0.0)
    rotation = jnp.max(jnp.abs(value.rotation), initial=0.0)
    return jnp.maximum(translation, rotation)


class _PositionProjectionFunction(StrictModule, NonTrainableState):
    graph: PreparedRigidJointGraph
    characteristic_length: float = eqx.field(static=True)

    def __call__(
        self,
        state: _PositionProjectionUnknown,
        args: _PositionProjectionArguments,
        /,
    ):
        graph = self.graph
        masses = graph.bodies.particles.safe_masses[graph.mobile_indices]

        def lagrangian(increment):
            candidate = graph.retract(args.predicted, increment)
            physical = graph.residuals(candidate)
            scaled = _scaled_residuals(physical, self.characteristic_length)
            translation_energy = 0.5 * jnp.sum(
                masses[:, None] * increment.translation * increment.translation
            )
            rotation_energy = 0.5 * jnp.sum(
                increment.rotation
                * contract("...ij,...j->...i", args.inertia_world, increment.rotation)
            )
            return (
                translation_energy
                + rotation_energy
                + rigid_joint_pairing(state.multipliers, scaled)
            )

        stationarity = jax.grad(lagrangian)(state.increment)
        candidate = graph.retract(args.predicted, state.increment)
        physical = graph.residuals(candidate)
        constraints = _scaled_residuals(physical, self.characteristic_length)
        alignment = graph.hinge_alignment(candidate)
        minimum_alignment = jnp.min(alignment, initial=1.0)
        maximum_rotation = jnp.max(
            jnp.linalg.norm(state.increment.rotation, axis=-1), initial=0.0
        )
        return _PositionProjectionResidual(
            stationarity, constraints
        ), _PositionProjectionAuxiliary(
            candidate,
            physical,
            minimum_alignment,
            maximum_rotation,
        )


class _PositionProjectionValidity(StrictModule, NonTrainableState):
    solver: RigidConstraintSolverPlan

    def __call__(self, state, residual, auxiliary, args, /):
        del residual, args
        quaternion_norm = jnp.linalg.norm(auxiliary.kinematics.orientation, axis=-1)
        return (
            tree_allfinite(state)
            & tree_allfinite(auxiliary)
            & jnp.all(jnp.abs(quaternion_norm - 1.0) <= self.solver.quaternion_tolerance)
            & (
                auxiliary.maximum_rotation_increment
                < pi - self.solver.rotation_chart_margin
            )
            & (auxiliary.minimum_hinge_alignment > self.solver.hinge_alignment_margin)
        )


class PreparedRigidConstraintDynamics(StrictModule, NonTrainableState):
    bodies: PreparedRigidBodySet
    joints: PreparedRigidJointGraph
    solver: RigidConstraintSolverPlan
    external_load: ExternalRigidBodyLoad | None
    external_load_id: str | None = eqx.field(static=True)
    position_problem: NonlinearSystemProblem | None
    position_template: PreparedNonlinearSolve | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: PreparedRigidBodySet,
        joints: PreparedRigidJointGraph,
        solver: RigidConstraintSolverPlan,
        reference: RigidBodyKinematics,
        external_load: ExternalRigidBodyLoad | None,
        external_load_id: str | None,
        plan_id: str,
        /,
    ):
        if joints.bodies.prepared_id != bodies.prepared_id:
            raise ValueError("Rigid joint graph and body set do not match.")
        if joints.constraint_count == 0:
            problem = None
            template = None
        else:
            residual = _PositionProjectionFunction(joints, solver.characteristic_length)
            problem = NonlinearSystemProblem(
                residual,
                has_aux=True,
                validity=_PositionProjectionValidity(solver),
                problem_id=f"rigid-position-projection/{joints.prepared_id}",
            )
            _, inertia_world = _rigid_body_world_inertia(bodies, reference.orientation)
            args = _PositionProjectionArguments(
                reference,
                inertia_world[joints.mobile_indices],
            )
            initial = _PositionProjectionUnknown(
                joints.empty_increment(reference.position.dtype),
                joints.empty_multipliers(reference.position.dtype),
            )
            template = prepare_nonlinear(
                problem,
                initial,
                method=solver.position_method,
                termination=solver.position_termination,
                args=args,
            )
        self.bodies = bodies
        self.joints = joints
        self.solver = solver
        self.external_load = external_load
        self.external_load_id = external_load_id
        self.position_problem = problem
        self.position_template = template
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-constraint-dynamics",
                "plan": plan_id,
                "bodies": bodies.prepared_id,
                "joints": joints.prepared_id,
                "solver": solver.plan_id,
                "external_load": external_load_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        orientation: ArrayLike,
        angular_velocity: ArrayLike,
        /,
    ) -> RigidConstraintState:
        kinematics = self.bodies.kinematics(
            position, velocity, orientation, angular_velocity
        )
        multipliers = self.joints.empty_multipliers(kinematics.position.dtype)
        return RigidConstraintState(kinematics, multipliers, multipliers)

    def _load(self, time: Array, kinematics: RigidBodyKinematics, args: Any, /):
        if self.external_load is None:
            return RigidBodyLoad(
                jnp.zeros_like(kinematics.position),
                jnp.zeros_like(kinematics.angular_velocity),
            )
        result = self.external_load(time, kinematics, args)
        if not isinstance(result, RigidBodyLoad):
            raise TypeError("external_load must return RigidBodyLoad.")
        if result.force.shape != kinematics.position.shape or (
            result.torque.shape != kinematics.angular_velocity.shape
        ):
            raise ValueError("External rigid-body load returned incompatible shapes.")
        return result

    def _solve_position(
        self,
        predicted: RigidBodyKinematics,
        multiplier_guess: RigidJointMultipliers,
        /,
    ) -> NonlinearResult:
        if self.position_problem is None or self.position_template is None:
            raise ValueError("An empty joint graph has no position projection solve.")
        inertia_world, _ = _rigid_body_world_inertia(self.bodies, predicted.orientation)
        arguments = _PositionProjectionArguments(
            predicted,
            inertia_world[self.joints.mobile_indices],
        )
        initial = _PositionProjectionUnknown(
            self.joints.empty_increment(predicted.position.dtype),
            multiplier_guess,
        )
        refreshed = refresh_nonlinear(
            self.position_template,
            self.position_problem,
            initial,
            args=arguments,
        )
        return implicit_root_result(refreshed)

    def _kinetic_energy(self, kinematics: RigidBodyKinematics, /) -> Array:
        mobile = self.joints.mobile_indices
        masses = self.bodies.particles.safe_masses[mobile]
        inertia_world, _ = _rigid_body_world_inertia(self.bodies, kinematics.orientation)
        linear = 0.5 * jnp.sum(masses[:, None] * kinematics.velocity[mobile] ** 2)
        angular = 0.5 * jnp.sum(
            kinematics.angular_velocity[mobile]
            * contract(
                "...ij,...j->...i",
                inertia_world[mobile],
                kinematics.angular_velocity[mobile],
            )
        )
        return linear + angular

    def _solve_velocity(
        self,
        kinematics: RigidBodyKinematics,
        multiplier_guess: RigidJointMultipliers,
        /,
    ):
        graph = self.joints
        zero = graph.empty_increment(kinematics.position.dtype)
        constraint_template = _scaled_residuals(
            graph.residuals(kinematics), self.solver.characteristic_length
        )
        primal_space = PyTreeSpace(zero)
        constraint_space = PyTreeSpace(constraint_template)
        mobile = graph.mobile_indices
        masses = self.bodies.particles.safe_masses[mobile]
        inertia_world, _ = _rigid_body_world_inertia(self.bodies, kinematics.orientation)
        inertia = inertia_world[mobile]

        mass = FunctionLinearOperator(
            lambda value: _RigidMobileIncrement(
                masses[:, None] * value.translation,
                contract("...ij,...j->...i", inertia, value.rotation),
            ),
            source=primal_space,
            target=primal_space,
            transpose_action=lambda value: _RigidMobileIncrement(
                masses[:, None] * value.translation,
                contract("...ij,...j->...i", inertia, value.rotation),
            ),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                block_diagonal=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "block_diagonal": "construction",
                },
            ),
            operator_id=f"rigid-mass/{self.prepared_id}",
        )
        constraint = FunctionLinearOperator(
            lambda value: _scaled_residuals(
                graph.velocity_residuals(kinematics, value.translation, value.rotation),
                self.solver.characteristic_length,
            ),
            source=primal_space,
            target=constraint_space,
            operator_id=f"rigid-joint-jacobian/{self.prepared_id}",
        )
        system = saddle_point_system(
            mass,
            constraint,
            problem_id=f"rigid-velocity-projection/{self.prepared_id}",
        )
        free_residual = _scaled_residuals(
            graph.current_velocity_residuals(kinematics),
            self.solver.characteristic_length,
        )
        initial_dual = _multipliers_to_residuals(multiplier_guess)
        result = solve_linear(
            system,
            (zero, _negate_residuals(free_residual)),
            policy=self.solver.velocity_policy,
            initial_guess=(zero, initial_dual),
        )
        correction, dual = result.value
        velocity = kinematics.velocity.at[mobile].add(correction.translation)
        angular = kinematics.angular_velocity.at[mobile].add(correction.rotation)
        projected = RigidBodyKinematics(
            kinematics.position,
            velocity,
            kinematics.orientation,
            angular,
        )
        rank_result = svd_linalg.svd(
            svd_linalg.SVDProblem(
                constraint,
                problem_id=f"rigid-joint-rank/{self.prepared_id}",
            ),
            policy=svd_linalg.SVDSolvePolicy(
                count=graph.constraint_count,
                rank=RankPolicy(
                    relative_cutoff=self.solver.rank_tolerance,
                    require_full_rank=True,
                ),
            ),
        )
        singular_values = rank_result.singular_values
        smallest = jnp.min(singular_values)
        condition = jnp.max(singular_values) / jnp.maximum(
            smallest, jnp.finfo(singular_values.dtype).tiny
        )
        rank_valid = (
            rank_result.successful
            & (rank_result.numerical_rank == graph.constraint_count)
            & jnp.isfinite(condition)
        )
        return (
            result,
            projected,
            _residuals_to_multipliers(dual),
            graph.current_velocity_residuals(projected),
            rank_valid,
            rank_result.numerical_rank,
            condition,
        )

    def _empty_evaluation(
        self,
        initial_load,
        closing_load,
        candidate,
        successful,
        reasons,
        kinetic_before,
        kinetic_after,
        /,
    ):
        residuals = self.joints.residuals(candidate.kinematics)
        multipliers = self.joints.empty_multipliers(candidate.kinematics.position.dtype)
        quaternion_defect = jnp.max(
            jnp.abs(jnp.linalg.norm(candidate.kinematics.orientation, axis=-1) - 1.0),
            initial=0.0,
        )
        diagnostics = RigidConstraintDiagnostics(
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            jnp.asarray(1.0, dtype=candidate.kinematics.position.dtype),
            quaternion_defect,
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            jnp.asarray(0.0, dtype=candidate.kinematics.position.dtype),
            kinetic_before,
            kinetic_after,
            jnp.maximum(kinetic_after - kinetic_before, 0.0),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=candidate.kinematics.position.dtype),
            tree_allfinite(candidate),
            successful,
        )
        return RigidConstraintEvaluation(
            initial_load,
            closing_load,
            None,
            None,
            residuals,
            residuals,
            multipliers,
            multipliers,
            diagnostics,
            successful,
            reasons,
            self.prepared_id,
        )

    def step(
        self,
        state: RigidConstraintState,
        time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> RigidConstraintStepResult:
        if not isinstance(state, RigidConstraintState):
            raise TypeError("state must be a RigidConstraintState.")
        time_ = jnp.asarray(time, dtype=state.kinematics.position.dtype)
        step_ = jnp.asarray(step_size, dtype=state.kinematics.position.dtype)
        valid_step = jnp.isfinite(time_) & jnp.isfinite(step_) & (step_ > 0.0)
        safe_step = jnp.where(valid_step, step_, 1.0)
        input_quaternion_defect = jnp.max(
            jnp.abs(jnp.linalg.norm(state.kinematics.orientation, axis=-1) - 1.0),
            initial=0.0,
        )
        input_valid = tree_allfinite(state) & (
            input_quaternion_defect <= self.solver.quaternion_tolerance
        )
        initial_load = self._load(time_, state.kinematics, args)
        initial_load_valid = tree_allfinite(initial_load)
        half = _rigid_body_half_kick(
            self.bodies, state.kinematics, initial_load, safe_step
        )
        predicted = _rigid_body_drift(self.bodies, half, safe_step)

        if self.joints.constraint_count == 0:
            closing_load = self._load(time_ + safe_step, predicted, args)
            candidate_kinematics = _rigid_body_close_kick(
                self.bodies, predicted, closing_load, safe_step
            )
            candidate = RigidConstraintState(
                candidate_kinematics,
                state.position_multiplier_guess,
                state.velocity_multiplier_guess,
            )
            finite = tree_allfinite(candidate) & tree_allfinite(closing_load)
            candidate_quaternion_defect = jnp.max(
                jnp.abs(jnp.linalg.norm(candidate_kinematics.orientation, axis=-1) - 1.0),
                initial=0.0,
            )
            quaternion_valid = (
                candidate_quaternion_defect <= self.solver.quaternion_tolerance
            )
            successful = (
                valid_step & input_valid & initial_load_valid & finite & quaternion_valid
            )
            reasons = jnp.asarray(0, dtype=jnp.int32)
            reasons |= jnp.where(
                valid_step, 0, int(RigidConstraintRejectionReason.INVALID_STEP)
            ).astype(jnp.int32)
            reasons |= jnp.where(
                input_valid, 0, int(RigidConstraintRejectionReason.INVALID_STATE)
            ).astype(jnp.int32)
            reasons |= jnp.where(
                initial_load_valid, 0, int(RigidConstraintRejectionReason.INITIAL_LOAD)
            ).astype(jnp.int32)
            reasons |= jnp.where(
                finite, 0, int(RigidConstraintRejectionReason.NONFINITE)
            ).astype(jnp.int32)
            reasons |= jnp.where(
                quaternion_valid, 0, int(RigidConstraintRejectionReason.QUATERNION)
            ).astype(jnp.int32)
            evaluation = self._empty_evaluation(
                initial_load,
                closing_load,
                candidate,
                successful,
                reasons,
                self._kinetic_energy(state.kinematics),
                self._kinetic_energy(candidate_kinematics),
            )
            accepted = tree_where(successful, candidate, state)
            return RigidConstraintStepResult(
                candidate, accepted, evaluation, successful, reasons
            )

        position_result = self._solve_position(predicted, state.position_multiplier_guess)
        position_aux = position_result.auxiliary
        projected_pose = position_aux.kinematics
        position_residuals = position_aux.physical_residuals
        position_maximum = rigid_joint_maximum_residual(position_residuals)
        stationarity_maximum = _maximum_increment(position_result.residual.stationarity)
        position_success = (
            position_result.successful
            & (position_maximum <= self.solver.position_tolerance)
            & (stationarity_maximum <= self.solver.stationarity_tolerance)
        )

        linear_velocity = (
            projected_pose.position - state.kinematics.position
        ) / safe_step
        angular_velocity = (
            _quaternion_relative_rotation_vector(
                state.kinematics.orientation, projected_pose.orientation
            )
            / safe_step
        )
        mobile = (self.bodies.particles.active_mask & ~self.bodies.fixed_mask)[:, None]
        drifted = RigidBodyKinematics(
            projected_pose.position,
            jnp.where(mobile, linear_velocity, 0.0),
            projected_pose.orientation,
            jnp.where(mobile, angular_velocity, 0.0),
        )
        full_rotation = jnp.max(
            jnp.linalg.norm(safe_step * drifted.angular_velocity, axis=-1), initial=0.0
        )
        rotation_valid = (
            position_aux.maximum_rotation_increment
            < pi - self.solver.rotation_chart_margin
        ) & (full_rotation < pi - self.solver.rotation_chart_margin)
        hinge_valid = (
            position_aux.minimum_hinge_alignment > self.solver.hinge_alignment_margin
        )

        closing_load = self._load(time_ + safe_step, drifted, args)
        closing_load_valid = tree_allfinite(closing_load)
        closed = _rigid_body_close_kick(self.bodies, drifted, closing_load, safe_step)
        kinetic_before = self._kinetic_energy(closed)
        (
            velocity_result,
            candidate_kinematics,
            velocity_multipliers,
            velocity_residuals,
            rank_valid,
            constraint_rank,
            constraint_condition,
        ) = self._solve_velocity(closed, state.velocity_multiplier_guess)
        kinetic_after = self._kinetic_energy(candidate_kinematics)
        energy_increase = jnp.maximum(kinetic_after - kinetic_before, 0.0)
        velocity_maximum = rigid_joint_maximum_residual(velocity_residuals)
        velocity_success = (
            velocity_result.successful
            & (velocity_maximum <= self.solver.velocity_tolerance)
            & (energy_increase <= self.solver.energy_tolerance)
            & rank_valid
        )

        position_multipliers = position_result.state.multipliers
        candidate = RigidConstraintState(
            candidate_kinematics,
            position_multipliers,
            velocity_multipliers,
        )
        quaternion_defect = jnp.max(
            jnp.abs(jnp.linalg.norm(candidate_kinematics.orientation, axis=-1) - 1.0),
            initial=0.0,
        )
        fixed = self.bodies.fixed_mask[:, None]
        fixed_pose_defect = jnp.maximum(
            jnp.max(
                jnp.where(
                    fixed,
                    jnp.abs(candidate_kinematics.position - state.kinematics.position),
                    0.0,
                ),
                initial=0.0,
            ),
            jnp.max(
                jnp.where(
                    fixed,
                    jnp.abs(
                        candidate_kinematics.orientation - state.kinematics.orientation
                    ),
                    0.0,
                ),
                initial=0.0,
            ),
        )
        fixed_velocity_defect = jnp.maximum(
            jnp.max(
                jnp.where(fixed, jnp.abs(candidate_kinematics.velocity), 0.0),
                initial=0.0,
            ),
            jnp.max(
                jnp.where(
                    fixed,
                    jnp.abs(candidate_kinematics.angular_velocity),
                    0.0,
                ),
                initial=0.0,
            ),
        )
        fixed_valid = (fixed_pose_defect <= self.solver.fixed_tolerance) & (
            fixed_velocity_defect <= self.solver.fixed_tolerance
        )
        quaternion_valid = quaternion_defect <= self.solver.quaternion_tolerance
        finite = (
            tree_allfinite(candidate)
            & tree_allfinite(position_residuals)
            & tree_allfinite(velocity_residuals)
            & closing_load_valid
        )
        successful = (
            valid_step
            & input_valid
            & initial_load_valid
            & position_success
            & rotation_valid
            & hinge_valid
            & closing_load_valid
            & velocity_success
            & fixed_valid
            & quaternion_valid
            & finite
            & rank_valid
        )

        reasons = jnp.asarray(0, dtype=jnp.int32)
        checks = (
            (valid_step, RigidConstraintRejectionReason.INVALID_STEP),
            (input_valid, RigidConstraintRejectionReason.INVALID_STATE),
            (initial_load_valid, RigidConstraintRejectionReason.INITIAL_LOAD),
            (position_result.successful, RigidConstraintRejectionReason.POSITION_SOLVE),
            (
                position_maximum <= self.solver.position_tolerance,
                RigidConstraintRejectionReason.POSITION_CONSTRAINT,
            ),
            (
                stationarity_maximum <= self.solver.stationarity_tolerance,
                RigidConstraintRejectionReason.POSITION_STATIONARITY,
            ),
            (rotation_valid, RigidConstraintRejectionReason.ROTATION_CHART),
            (hinge_valid, RigidConstraintRejectionReason.HINGE_ALIGNMENT),
            (closing_load_valid, RigidConstraintRejectionReason.CLOSING_LOAD),
            (velocity_result.successful, RigidConstraintRejectionReason.VELOCITY_SOLVE),
            (
                velocity_maximum <= self.solver.velocity_tolerance,
                RigidConstraintRejectionReason.VELOCITY_CONSTRAINT,
            ),
            (rank_valid, RigidConstraintRejectionReason.RANK_OR_CONDITION),
            (fixed_valid, RigidConstraintRejectionReason.FIXED_BODY),
            (quaternion_valid, RigidConstraintRejectionReason.QUATERNION),
            (
                energy_increase <= self.solver.energy_tolerance,
                RigidConstraintRejectionReason.ENERGY_PROJECTION,
            ),
            (finite, RigidConstraintRejectionReason.NONFINITE),
        )
        for condition, reason in checks:
            reasons |= jnp.where(condition, 0, int(reason)).astype(jnp.int32)

        diagnostics = RigidConstraintDiagnostics(
            position_maximum,
            stationarity_maximum,
            velocity_maximum,
            jnp.maximum(position_aux.maximum_rotation_increment, full_rotation),
            position_aux.minimum_hinge_alignment,
            quaternion_defect,
            fixed_pose_defect,
            fixed_velocity_defect,
            kinetic_before,
            kinetic_after,
            energy_increase,
            constraint_rank,
            constraint_condition,
            finite,
            successful,
        )
        evaluation = RigidConstraintEvaluation(
            initial_load,
            closing_load,
            position_result,
            velocity_result,
            position_residuals,
            velocity_residuals,
            position_multipliers,
            velocity_multipliers,
            diagnostics,
            successful,
            reasons,
            self.prepared_id,
        )
        accepted = tree_where(successful, candidate, state)
        return RigidConstraintStepResult(
            candidate,
            accepted,
            evaluation,
            successful,
            reasons,
        )


__all__ = [
    "PreparedRigidConstraintDynamics",
    "RigidConstraintDiagnostics",
    "RigidConstraintDynamicsPlan",
    "RigidConstraintEvaluation",
    "RigidConstraintRejectionReason",
    "RigidConstraintSolverPlan",
    "RigidConstraintState",
    "RigidConstraintStepResult",
]
