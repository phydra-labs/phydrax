#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..equations._mac_enthalpy_porosity import (
    CompiledMACEnthalpyPorosityDynamics,
    MACEnthalpyPorosityStage,
)
from ..linalg import LinearSolvePolicy
from ._mac_composite_projection import (
    CompositeMACProjectionPlan,
    CompositeMACProjectionResult,
)
from ._mac_stage_inverse_general import (
    MACOperatorStageSolveResult,
    MACVariableViscosityStagePlan,
)
from ._mac_viscous import (
    _constraint_operators,
    _incoming_pressure,
    _variable_viscosity_policy,
)


class MACEnthalpyPorosityStepStatus(IntEnum):
    SUCCESS = 0
    INVALID_STATE = 1
    BOUNDARY_FAILURE = 2
    COEFFICIENT_FAILURE = 3
    MOMENTUM_FAILURE = 4
    PROJECTION_FAILURE = 5
    ENTHALPY_FAILURE = 6


class MACEnthalpyPorosityIMEXResult(StrictModule):
    time: Array
    attempted_time: Array
    step_size: Array
    previous_state: Array
    state: Array
    velocity: FaceVelocity
    enthalpy: Array
    pressure: Array
    explicit_velocity_rate: FaceVelocity
    enthalpy_rate: Array
    coefficient_stage: MACEnthalpyPorosityStage
    predictor: MACOperatorStageSolveResult
    projection: CompositeMACProjectionResult
    finite: Array
    accepted: Array
    status: Array
    stage_plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


def _status(
    state_valid: Array,
    boundary_valid: Array,
    coefficient_valid: Array,
    momentum_valid: Array,
    projection_valid: Array,
    enthalpy_valid: Array,
    /,
) -> Array:
    return jnp.where(
        ~state_valid,
        int(MACEnthalpyPorosityStepStatus.INVALID_STATE),
        jnp.where(
            ~boundary_valid,
            int(MACEnthalpyPorosityStepStatus.BOUNDARY_FAILURE),
            jnp.where(
                ~coefficient_valid,
                int(MACEnthalpyPorosityStepStatus.COEFFICIENT_FAILURE),
                jnp.where(
                    ~momentum_valid,
                    int(MACEnthalpyPorosityStepStatus.MOMENTUM_FAILURE),
                    jnp.where(
                        ~projection_valid,
                        int(MACEnthalpyPorosityStepStatus.PROJECTION_FAILURE),
                        jnp.where(
                            ~enthalpy_valid,
                            int(MACEnthalpyPorosityStepStatus.ENTHALPY_FAILURE),
                            int(MACEnthalpyPorosityStepStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _explicit_velocity_rate(
    dynamics: CompiledMACEnthalpyPorosityDynamics,
    stage: MACEnthalpyPorosityStage,
    /,
) -> FaceVelocity:
    return dynamics.momentum.boundaries.homogeneous_rate(
        tuple(
            -convection + forcing + buoyancy
            for convection, forcing, buoyancy in zip(
                stage.momentum_components.convection,
                stage.momentum_components.forcing,
                stage.buoyancy_force,
                strict=True,
            )
        )
    )


def _face_resistance(
    dynamics: CompiledMACEnthalpyPorosityDynamics,
    stage: MACEnthalpyPorosityStage,
    /,
) -> FaceVelocity:
    return tuple(
        dynamics.problem.material.evaluate(value).mushy_resistance
        for value in stage.enthalpy_flux.enthalpy_face_values
    )


class MACEnthalpyPorosityIMEXEulerMethod(StrictModule, NonTrainableState):
    dynamics: CompiledMACEnthalpyPorosityDynamics
    face_density: FaceVelocity
    divergence_operator: object
    gradient_operator: object
    variable_linear_policy: LinearSolvePolicy
    pressure_linear_policy: LinearSolvePolicy
    fixed_step_size: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACEnthalpyPorosityDynamics,
        /,
        *,
        fixed_step_size: float | None = None,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(dynamics, CompiledMACEnthalpyPorosityDynamics):
            raise TypeError("dynamics must be CompiledMACEnthalpyPorosityDynamics.")
        fixed = None if fixed_step_size is None else float(fixed_step_size)
        tolerance_ = float(tolerance)
        if fixed is not None and (not np.isfinite(fixed) or fixed <= 0.0):
            raise ValueError("fixed_step_size must be positive and finite.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        policy = _variable_viscosity_policy(tolerance_, maximum_iterations, linear_policy)
        pressure_policy = _variable_viscosity_policy(tolerance_, maximum_iterations, None)
        density = tuple(
            jnp.ones(
                layout.shape,
                dtype=dynamics.momentum.operators.pressure_space.dtype,
            )
            for layout in dynamics.momentum.operators.discretization.face_layouts
        )
        divergence, gradient = _constraint_operators(dynamics.base_dynamics)
        identifier = canonical_fingerprint(
            {
                "kind": "mac-enthalpy-porosity-imex-euler",
                "dynamics": dynamics.compilation_id,
                "fixed_step_size": fixed,
                "tolerance": tolerance_,
            }
        )
        self.dynamics = dynamics
        self.face_density = density
        self.divergence_operator = divergence
        self.gradient_operator = gradient
        self.variable_linear_policy = policy
        self.pressure_linear_policy = pressure_policy
        self.fixed_step_size = fixed
        self.tolerance = tolerance_
        self.method_id = identifier

    def _step_size(self, value: ArrayLike | None, /) -> Array:
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        if self.fixed_step_size is None:
            if value is None:
                raise ValueError("Dynamic MAC enthalpy IMEX Euler requires step_size.")
            step = jnp.asarray(value, dtype=dtype).reshape(())
        else:
            step = jnp.asarray(self.fixed_step_size, dtype=dtype)
            if value is not None:
                supplied = jnp.asarray(value, dtype=dtype).reshape(())
                step = eqx.error_if(
                    step,
                    supplied != step,
                    "Fixed MAC enthalpy IMEX Euler step cannot change.",
                )
        return eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "MAC enthalpy IMEX Euler step must be positive and finite.",
        )

    def step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        step_size: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        args: Any = None,
    ) -> MACEnthalpyPorosityIMEXResult:
        step = self._step_size(step_size)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        current_state = self.dynamics.validate_state(state)
        current_velocity, current_enthalpy = self.dynamics.unpack_state(current_state)
        coefficient_stage = self.dynamics.stage(time_, current_state, args)
        explicit_velocity = _explicit_velocity_rate(self.dynamics, coefficient_stage)
        enthalpy_rate = coefficient_stage.enthalpy_rate
        attempted_time = time_ + step
        boundary_stage = self.dynamics.base_dynamics.boundary_stage(attempted_time, args)
        stage_plan = MACVariableViscosityStagePlan(
            self.dynamics.momentum,
            self.face_density,
            coefficient_stage.thermodynamics.viscosity,
            step,
            rhs_scale=step,
            face_resistance=_face_resistance(self.dynamics, coefficient_stage),
            stage_id=f"{self.method_id}/accepted-state",
        )
        inverse = stage_plan.inverse(
            boundary_stage, linear_policy=self.variable_linear_policy
        )
        physical_rhs = tuple(
            value / step + rate
            for value, rate in zip(current_velocity, explicit_velocity, strict=True)
        )
        predictor = inverse.solve_affine(physical_rhs)
        projection_plan = CompositeMACProjectionPlan(
            self.divergence_operator,
            self.gradient_operator,
            inverse.operator(),
            self.dynamics.momentum.operators.gauge_project,
            linear_policy=self.pressure_linear_policy,
            tolerance=self.tolerance,
        )
        incoming_pressure = _incoming_pressure(
            self.dynamics.base_dynamics, pressure, step.dtype
        )
        projection = projection_plan.project(predictor.value, pressure=incoming_pressure)
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_enthalpy = current_enthalpy + step * enthalpy_rate
        candidate_thermodynamics = self.dynamics.problem.material.evaluate(
            candidate_enthalpy
        )
        candidate_state = self.dynamics.pack_state(candidate_velocity, candidate_enthalpy)
        state_valid = jnp.all(jnp.isfinite(current_state))
        boundary_valid = (
            coefficient_stage.boundary_stage.successful & boundary_stage.successful
        )
        coefficient_valid = coefficient_stage.successful
        momentum_valid = predictor.converged
        projection_valid = projection.accepted
        enthalpy_valid = jnp.all(candidate_thermodynamics.successful) & jnp.all(
            jnp.isfinite(candidate_enthalpy)
        )
        finite = (
            state_valid
            & boundary_stage.finite
            & coefficient_stage.finite
            & predictor.finite
            & projection.finite
            & enthalpy_valid
            & jnp.all(jnp.isfinite(candidate_state))
        )
        accepted = (
            finite
            & boundary_valid
            & coefficient_valid
            & momentum_valid
            & projection_valid
            & enthalpy_valid
        )
        status = _status(
            state_valid,
            boundary_valid,
            coefficient_valid,
            momentum_valid,
            projection_valid,
            enthalpy_valid,
        )
        return MACEnthalpyPorosityIMEXResult(
            jnp.where(accepted, attempted_time, time_),
            attempted_time,
            step,
            current_state,
            jnp.where(accepted, candidate_state, current_state),
            tuple(
                jnp.where(accepted, candidate, current)
                for candidate, current in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            jnp.where(accepted, candidate_enthalpy, current_enthalpy),
            jnp.where(accepted, projection.pressure, incoming_pressure),
            explicit_velocity,
            enthalpy_rate,
            coefficient_stage,
            predictor,
            projection,
            finite,
            accepted,
            status,
            stage_plan.plan_id,
            self.method_id,
        )


class MACEnthalpyPorositySBDF2State(StrictModule):
    time: Array
    state: Array
    previous_state: Array
    previous_explicit_velocity_rate: FaceVelocity
    previous_enthalpy_rate: Array
    pressure: Array
    step_size: Array
    finite: Array
    method_id: str = eqx.field(static=True)


class MACEnthalpyPorositySBDF2Result(StrictModule):
    previous: MACEnthalpyPorositySBDF2State
    state: MACEnthalpyPorositySBDF2State
    predictor: MACOperatorStageSolveResult
    projection: CompositeMACProjectionResult
    coefficient_stage: MACEnthalpyPorosityStage
    finite: Array
    accepted: Array
    status: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACEnthalpyPorositySBDF2Method(StrictModule, NonTrainableState):
    dynamics: CompiledMACEnthalpyPorosityDynamics
    startup: MACEnthalpyPorosityIMEXEulerMethod
    divergence_operator: object
    gradient_operator: object
    variable_linear_policy: LinearSolvePolicy
    pressure_linear_policy: LinearSolvePolicy
    step_size: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACEnthalpyPorosityDynamics,
        step_size: float,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("SBDF2 step_size must be positive and finite.")
        startup = MACEnthalpyPorosityIMEXEulerMethod(
            dynamics,
            fixed_step_size=step,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
        )
        divergence, gradient = _constraint_operators(dynamics.base_dynamics)
        identifier = canonical_fingerprint(
            {
                "kind": "mac-enthalpy-porosity-sbdf2",
                "dynamics": dynamics.compilation_id,
                "step_size": step,
                "tolerance": float(tolerance),
            }
        )
        self.dynamics = dynamics
        self.startup = startup
        self.divergence_operator = divergence
        self.gradient_operator = gradient
        self.variable_linear_policy = startup.variable_linear_policy
        self.pressure_linear_policy = startup.pressure_linear_policy
        self.step_size = step
        self.tolerance = float(tolerance)
        self.method_id = identifier

    def initialize(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        args: Any = None,
    ) -> MACEnthalpyPorositySBDF2Result:
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        state_ = self.dynamics.validate_state(state)
        initial_stage = self.dynamics.stage(time_, state_, args)
        initial_explicit = _explicit_velocity_rate(self.dynamics, initial_stage)
        initial_pressure = _incoming_pressure(
            self.dynamics.base_dynamics, pressure, dtype
        )
        initial_history = MACEnthalpyPorositySBDF2State(
            time_,
            state_,
            state_,
            initial_explicit,
            initial_stage.enthalpy_rate,
            initial_pressure,
            jnp.asarray(self.step_size, dtype=dtype),
            initial_stage.finite,
            self.method_id,
        )
        startup = self.startup.step(time_, state_, pressure=initial_pressure, args=args)
        next_history = MACEnthalpyPorositySBDF2State(
            startup.time,
            startup.state,
            state_,
            startup.explicit_velocity_rate,
            startup.enthalpy_rate,
            startup.pressure,
            initial_history.step_size,
            startup.finite,
            self.method_id,
        )
        return MACEnthalpyPorositySBDF2Result(
            initial_history,
            next_history,
            startup.predictor,
            startup.projection,
            startup.coefficient_stage,
            startup.finite,
            startup.accepted,
            startup.status,
            self.method_id,
        )

    def step(
        self,
        state: MACEnthalpyPorositySBDF2State,
        /,
        *,
        args: Any = None,
    ) -> MACEnthalpyPorositySBDF2Result:
        if not isinstance(state, MACEnthalpyPorositySBDF2State):
            raise TypeError("state must be MACEnthalpyPorositySBDF2State.")
        if state.method_id != self.method_id:
            raise ValueError("SBDF2 state belongs to another method.")
        step = state.step_size
        current_velocity, current_enthalpy = self.dynamics.unpack_state(state.state)
        previous_velocity, previous_enthalpy = self.dynamics.unpack_state(
            state.previous_state
        )
        current_stage = self.dynamics.stage(state.time, state.state, args)
        current_explicit = _explicit_velocity_rate(self.dynamics, current_stage)
        extrapolated_enthalpy = 2.0 * current_enthalpy - previous_enthalpy
        coefficient_state = self.dynamics.pack_state(
            current_velocity, extrapolated_enthalpy
        )
        coefficient_stage = self.dynamics.stage(
            state.time + step, coefficient_state, args
        )
        attempted_time = state.time + step
        boundary_stage = self.dynamics.base_dynamics.boundary_stage(attempted_time, args)
        mass = tuple(
            jnp.full_like(component, 1.5 / step) for component in current_velocity
        )
        stage_plan = MACVariableViscosityStagePlan(
            self.dynamics.momentum,
            mass,
            coefficient_stage.thermodynamics.viscosity,
            jnp.asarray(1.0, dtype=step.dtype),
            face_resistance=_face_resistance(self.dynamics, coefficient_stage),
            stage_id=f"{self.method_id}/extrapolated-state",
        )
        inverse = stage_plan.inverse(
            boundary_stage, linear_policy=self.variable_linear_policy
        )
        physical_rhs = tuple(
            2.0 * current / step
            - 0.5 * previous / step
            + 2.0 * current_rate
            - previous_rate
            for current, previous, current_rate, previous_rate in zip(
                current_velocity,
                previous_velocity,
                current_explicit,
                state.previous_explicit_velocity_rate,
                strict=True,
            )
        )
        predictor = inverse.solve_affine(physical_rhs)
        projection_plan = CompositeMACProjectionPlan(
            self.divergence_operator,
            self.gradient_operator,
            inverse.operator(),
            self.dynamics.momentum.operators.gauge_project,
            linear_policy=self.pressure_linear_policy,
            tolerance=self.tolerance,
        )
        projection = projection_plan.project(predictor.value, pressure=state.pressure)
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            projection.velocity, boundary_stage
        )
        candidate_enthalpy = (
            4.0 / 3.0 * current_enthalpy
            - 1.0 / 3.0 * previous_enthalpy
            + 2.0
            / 3.0
            * step
            * (2.0 * current_stage.enthalpy_rate - state.previous_enthalpy_rate)
        )
        candidate_thermodynamics = self.dynamics.problem.material.evaluate(
            candidate_enthalpy
        )
        candidate_coordinates = self.dynamics.pack_state(
            candidate_velocity, candidate_enthalpy
        )
        state_valid = state.finite & current_stage.finite
        boundary_valid = boundary_stage.successful
        coefficient_valid = coefficient_stage.successful
        momentum_valid = predictor.converged
        projection_valid = projection.accepted
        enthalpy_valid = jnp.all(candidate_thermodynamics.successful) & jnp.all(
            jnp.isfinite(candidate_enthalpy)
        )
        finite = (
            state_valid
            & boundary_stage.finite
            & coefficient_stage.finite
            & predictor.finite
            & projection.finite
            & enthalpy_valid
            & jnp.all(jnp.isfinite(candidate_coordinates))
        )
        accepted = (
            finite
            & boundary_valid
            & coefficient_valid
            & momentum_valid
            & projection_valid
            & enthalpy_valid
        )
        status = _status(
            state_valid,
            boundary_valid,
            coefficient_valid,
            momentum_valid,
            projection_valid,
            enthalpy_valid,
        )
        next_state = MACEnthalpyPorositySBDF2State(
            jnp.where(accepted, attempted_time, state.time),
            jnp.where(accepted, candidate_coordinates, state.state),
            jnp.where(accepted, state.state, state.previous_state),
            tuple(
                jnp.where(accepted, current, previous)
                for current, previous in zip(
                    current_explicit,
                    state.previous_explicit_velocity_rate,
                    strict=True,
                )
            ),
            jnp.where(
                accepted, current_stage.enthalpy_rate, state.previous_enthalpy_rate
            ),
            jnp.where(accepted, projection.pressure, state.pressure),
            step,
            jnp.where(accepted, finite, state.finite),
            self.method_id,
        )
        return MACEnthalpyPorositySBDF2Result(
            state,
            next_state,
            predictor,
            projection,
            coefficient_stage,
            finite,
            accepted,
            status,
            self.method_id,
        )


__all__ = [
    "MACEnthalpyPorosityIMEXEulerMethod",
    "MACEnthalpyPorosityIMEXResult",
    "MACEnthalpyPorositySBDF2Method",
    "MACEnthalpyPorositySBDF2Result",
    "MACEnthalpyPorositySBDF2State",
    "MACEnthalpyPorosityStepStatus",
]
