#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._lagrangian_marker import LagrangianMarkerKinematics
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_marker_transfer import MACMarkerRouteState
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..linalg import LinearSolvePolicy
from ._mac_immersed_boundary import (
    MACImmersedBoundaryProjectionPlan,
    MACImmersedBoundaryProjectionResult,
)
from ._mac_stage_inverse_momentum import MACHelmholtzStageInverseMomentum
from ._mac_viscous import (
    MACHelmholtzResult,
    MACHelmholtzSolveMethod,
    MACHelmholtzSolvePlan,
)


MarkerMotionProvider = Callable[[Array, Any], LagrangianMarkerKinematics]


class MACImmersedBoundaryStepStatus(IntFlag):
    SUCCESS = 0
    INVALID_TIME = 1
    MOTION_FAILED = 2
    BOUNDARY_FAILED = 4
    HELMHOLTZ_FAILED = 8
    PROJECTION_FAILED = 16
    HISTORY_INVALID = 32
    NONFINITE = 64


class MACImmersedBoundaryIMEXEulerResult(StrictModule):
    time: Array
    attempted_time: Array
    step_size: Array
    previous_state: Array
    state: Array
    velocity: FaceVelocity
    pressure: Array
    marker_force_density: Array
    marker_kinematics: LagrangianMarkerKinematics
    route_state: MACMarkerRouteState
    explicit_rate: FaceVelocity
    helmholtz: MACHelmholtzResult
    projection: MACImmersedBoundaryProjectionResult
    finite: Array
    accepted: Array
    status: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACImmersedBoundaryIMEXEulerMethod(StrictModule, NonTrainableState):
    """Backward-Euler diffusion plus explicit forcing and marker projection.

    Algebraic SGS rates are retained in the explicit partition. Optional fixed
    marker normals select the projection's normal-only constraint.
    """

    dynamics: CompiledMACIncompressibleDynamics
    projection: MACImmersedBoundaryProjectionPlan
    marker_motion: MarkerMotionProvider
    marker_constraint_normals: Array | None
    motion_id: str = eqx.field(static=True)
    helmholtz: MACHelmholtzSolvePlan
    fixed_step_size: float | None = eqx.field(static=True)
    allow_route_refresh: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        projection: MACImmersedBoundaryProjectionPlan,
        marker_motion: MarkerMotionProvider,
        /,
        *,
        motion_id: str,
        fixed_step_size: float | None = None,
        solve_method: MACHelmholtzSolveMethod = "auto",
        hybrid_line_axis: int | None = None,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        allow_route_refresh: bool = False,
        marker_constraint_normals: ArrayLike | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if not isinstance(projection, MACImmersedBoundaryProjectionPlan):
            raise TypeError("projection must be MACImmersedBoundaryProjectionPlan.")
        if projection.operators.prepared_id != dynamics.momentum.operators.prepared_id:
            raise ValueError("Immersed projection and dynamics must share MAC operators.")
        if projection.boundaries.prepared_id != dynamics.momentum.boundaries.prepared_id:
            raise ValueError("Immersed projection and dynamics must share boundaries.")
        if not callable(marker_motion):
            raise TypeError("marker_motion must be callable.")
        identifier = str(motion_id)
        if not identifier:
            raise ValueError("motion_id must be nonempty.")
        fixed = None if fixed_step_size is None else float(fixed_step_size)
        if fixed is not None and (not np.isfinite(fixed) or fixed <= 0.0):
            raise ValueError("fixed_step_size must be positive and finite.")
        viscosity = float(np.asarray(dynamics.problem.viscosity))
        if marker_constraint_normals is None:
            constraint_normals = None
        else:
            constraint_normals = jnp.asarray(
                marker_constraint_normals,
                dtype=projection.operators.pressure_space.dtype,
            )
            expected = (
                projection.transfer.markers.capacity,
                projection.transfer.markers.ambient_dimension,
            )
            if constraint_normals.shape != expected:
                raise ValueError(f"marker_constraint_normals must have shape {expected}.")
        helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            solve_method=solve_method,
            hybrid_line_axis=hybrid_line_axis,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
            fixed_mass_coefficient=None if fixed is None else 1.0,
            fixed_diffusion_coefficient=None if fixed is None else fixed * viscosity,
            maximum_resource_bytes=maximum_resource_bytes,
        )
        self.dynamics = dynamics
        self.projection = projection
        self.marker_motion = marker_motion
        self.marker_constraint_normals = constraint_normals
        self.motion_id = identifier
        self.helmholtz = helmholtz
        self.fixed_step_size = fixed
        self.allow_route_refresh = bool(allow_route_refresh)
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-boundary-imex-euler",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "motion": identifier,
                "helmholtz": helmholtz.plan_id,
                "fixed_step_size": fixed,
                "allow_route_refresh": bool(allow_route_refresh),
                "marker_constraint_normals": None
                if constraint_normals is None
                else array_tree_fingerprint(constraint_normals),
            }
        )

    def _step_size(self, value: ArrayLike | None, /) -> Array:
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        if self.fixed_step_size is None:
            if value is None:
                raise ValueError("Dynamic immersed IMEX Euler requires step_size.")
            step = jnp.asarray(value, dtype=dtype).reshape(())
        else:
            step = jnp.asarray(self.fixed_step_size, dtype=dtype)
            if value is not None:
                supplied = jnp.asarray(value, dtype=dtype).reshape(())
                step = eqx.error_if(
                    step,
                    supplied != step,
                    "Fixed immersed IMEX Euler step cannot change.",
                )
        return eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Immersed IMEX Euler step must be positive and finite.",
        )

    def _explicit_rate(self, time: Array, state: Array, args: Any, /) -> FaceVelocity:
        components = self.dynamics.rate_components(time, state, args)
        return tuple(
            -advective + sgs + source
            for advective, sgs, source in zip(
                components.convection,
                components.sgs,
                components.forcing,
                strict=True,
            )
        )

    def step(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        step_size: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        expected_routes: MACMarkerRouteState | None = None,
        args: Any = None,
    ) -> MACImmersedBoundaryIMEXEulerResult:
        step = self._step_size(step_size)
        current_state = self.dynamics.validate_state(state)
        current_velocity = self.dynamics.unpack_velocity(current_state)
        time_ = jnp.asarray(time, dtype=step.dtype).reshape(())
        attempted_time = time_ + step
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        explicit = self._explicit_rate(time_, current_state, args)
        rhs = tuple(
            value + step * rate
            for value, rate in zip(current_velocity, explicit, strict=True)
        )
        viscosity = self.dynamics.problem.viscosity.astype(step.dtype)
        helmholtz = self.helmholtz.solve(
            rhs,
            boundary_stage,
            mass_coefficient=None if self.fixed_step_size is not None else 1.0,
            diffusion_coefficient=None
            if self.fixed_step_size is not None
            else step * viscosity,
            initial_guess=current_velocity,
        )
        marker_kinematics = self.marker_motion(attempted_time, args)
        if not isinstance(marker_kinematics, LagrangianMarkerKinematics):
            raise TypeError("marker_motion must return LagrangianMarkerKinematics.")
        stage_inverse = MACHelmholtzStageInverseMomentum(
            self.helmholtz,
            boundary_stage,
            mass_coefficient=(
                None if self.fixed_step_size is not None else jnp.asarray(1.0)
            ),
            diffusion_coefficient=(
                None if self.fixed_step_size is not None else step * viscosity
            ),
            rhs_scale=step,
            stage_id=f"{self.method_id}/stage",
        )
        immersed = self.projection.project(
            helmholtz.value,
            stage_inverse,
            marker_kinematics,
            pressure=pressure,
            marker_force_density=marker_force_density,
            boundary_stage=boundary_stage,
            expected_routes=expected_routes,
            allow_route_refresh=self.allow_route_refresh,
            marker_constraint_normals=self.marker_constraint_normals,
        )
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            immersed.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        finite = (
            boundary_stage.finite
            & helmholtz.finite
            & immersed.finite
            & jnp.all(jnp.isfinite(candidate_state))
        )
        accepted = (
            boundary_stage.successful & helmholtz.converged & immersed.converged & finite
        )
        status = jnp.where(
            ~boundary_stage.successful,
            int(MACImmersedBoundaryStepStatus.BOUNDARY_FAILED),
            jnp.where(
                ~helmholtz.converged,
                int(MACImmersedBoundaryStepStatus.HELMHOLTZ_FAILED),
                jnp.where(
                    ~immersed.converged,
                    int(MACImmersedBoundaryStepStatus.PROJECTION_FAILED),
                    jnp.where(
                        ~finite,
                        int(MACImmersedBoundaryStepStatus.NONFINITE),
                        int(MACImmersedBoundaryStepStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        accepted_velocity = tuple(
            jnp.where(accepted, candidate, current)
            for candidate, current in zip(
                candidate_velocity, current_velocity, strict=True
            )
        )
        return MACImmersedBoundaryIMEXEulerResult(
            time=jnp.where(accepted, attempted_time, time_),
            attempted_time=attempted_time,
            step_size=step,
            previous_state=current_state,
            state=jnp.where(accepted, candidate_state, current_state),
            velocity=accepted_velocity,
            pressure=jnp.where(
                accepted,
                immersed.pressure,
                jnp.zeros_like(immersed.pressure)
                if pressure is None
                else jnp.asarray(pressure),
            ),
            marker_force_density=jnp.where(
                accepted,
                immersed.marker_force_density,
                jnp.zeros_like(immersed.marker_force_density)
                if marker_force_density is None
                else jnp.asarray(marker_force_density),
            ),
            marker_kinematics=marker_kinematics,
            route_state=immersed.route_state,
            explicit_rate=explicit,
            helmholtz=helmholtz,
            projection=immersed,
            finite=finite,
            accepted=accepted,
            status=status,
            method_id=self.method_id,
        )


class MACImmersedBoundarySBDF2State(StrictModule):
    time: Array
    previous_state: Array
    state: Array
    previous_explicit_rate: FaceVelocity
    explicit_rate: FaceVelocity
    pressure: Array
    marker_force_density: Array
    route_state: MACMarkerRouteState
    accepted_steps: Array
    valid: Array
    status: Array
    method_id: str = eqx.field(static=True)


class MACImmersedBoundarySBDF2Result(StrictModule):
    history: MACImmersedBoundarySBDF2State
    attempted_time: Array
    step_size: Array
    pressure_correction_coefficient: Array
    velocity: FaceVelocity
    pressure: Array
    marker_force_density: Array
    marker_kinematics: LagrangianMarkerKinematics
    route_state: MACMarkerRouteState
    helmholtz: MACHelmholtzResult
    projection: MACImmersedBoundaryProjectionResult
    finite: Array
    accepted: Array
    status: Array
    startup: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class MACImmersedBoundarySBDF2Method(StrictModule, NonTrainableState):
    """Fixed-step SBDF2 retaining algebraic SGS and marker constraints."""

    dynamics: CompiledMACIncompressibleDynamics
    projection: MACImmersedBoundaryProjectionPlan
    marker_motion: MarkerMotionProvider
    motion_id: str = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    startup_method: MACImmersedBoundaryIMEXEulerMethod
    helmholtz: MACHelmholtzSolvePlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        projection: MACImmersedBoundaryProjectionPlan,
        marker_motion: MarkerMotionProvider,
        step_size: float,
        /,
        *,
        motion_id: str,
        solve_method: MACHelmholtzSolveMethod = "auto",
        hybrid_line_axis: int | None = None,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 500,
        linear_policy: LinearSolvePolicy | None = None,
        allow_route_refresh: bool = False,
        marker_constraint_normals: ArrayLike | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
    ):
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("SBDF2 step_size must be positive and finite.")
        startup = MACImmersedBoundaryIMEXEulerMethod(
            dynamics,
            projection,
            marker_motion,
            motion_id=motion_id,
            fixed_step_size=step,
            solve_method=solve_method,
            hybrid_line_axis=hybrid_line_axis,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
            allow_route_refresh=allow_route_refresh,
            marker_constraint_normals=marker_constraint_normals,
            maximum_resource_bytes=maximum_resource_bytes,
        )
        viscosity = float(np.asarray(dynamics.problem.viscosity))
        helmholtz = MACHelmholtzSolvePlan(
            dynamics.momentum,
            solve_method=solve_method,
            hybrid_line_axis=hybrid_line_axis,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            linear_policy=linear_policy,
            fixed_mass_coefficient=1.5,
            fixed_diffusion_coefficient=step * viscosity,
            maximum_resource_bytes=maximum_resource_bytes,
        )
        self.dynamics = dynamics
        self.projection = projection
        self.marker_motion = marker_motion
        self.motion_id = str(motion_id)
        self.step_size = step
        self.startup_method = startup
        self.helmholtz = helmholtz
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-immersed-boundary-sbdf2",
                "dynamics": dynamics.compilation_id,
                "projection": projection.plan_id,
                "motion": str(motion_id),
                "step_size": step,
                "allow_route_refresh": bool(allow_route_refresh),
                "marker_constraint_normals": None
                if startup.marker_constraint_normals is None
                else array_tree_fingerprint(startup.marker_constraint_normals),
                "helmholtz": helmholtz.plan_id,
            }
        )

    def _explicit_rate(self, time: Array, state: Array, args: Any, /) -> FaceVelocity:
        components = self.dynamics.rate_components(time, state, args)
        return tuple(
            -advective + sgs + source
            for advective, sgs, source in zip(
                components.convection,
                components.sgs,
                components.forcing,
                strict=True,
            )
        )

    def initialize(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        pressure: ArrayLike | None = None,
        marker_force_density: ArrayLike | None = None,
        args: Any = None,
    ) -> MACImmersedBoundarySBDF2Result:
        initial_state = self.dynamics.validate_state(state)
        dtype = self.dynamics.momentum.operators.pressure_space.dtype
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        initial_explicit = self._explicit_rate(time_, initial_state, args)
        startup = self.startup_method.step(
            time_,
            initial_state,
            pressure=pressure,
            marker_force_density=marker_force_density,
            args=args,
        )
        following_explicit = jax.lax.cond(
            startup.accepted,
            lambda _: self._explicit_rate(startup.time, startup.state, args),
            lambda _: initial_explicit,
            operand=None,
        )
        history = MACImmersedBoundarySBDF2State(
            time=startup.time,
            previous_state=initial_state,
            state=startup.state,
            previous_explicit_rate=initial_explicit,
            explicit_rate=following_explicit,
            pressure=startup.pressure,
            marker_force_density=startup.marker_force_density,
            route_state=startup.route_state,
            accepted_steps=startup.accepted.astype(jnp.int32),
            valid=startup.accepted,
            status=startup.status,
            method_id=self.method_id,
        )
        return MACImmersedBoundarySBDF2Result(
            history=history,
            attempted_time=startup.attempted_time,
            step_size=startup.step_size,
            pressure_correction_coefficient=startup.step_size,
            velocity=startup.velocity,
            pressure=startup.pressure,
            marker_force_density=startup.marker_force_density,
            marker_kinematics=startup.marker_kinematics,
            route_state=startup.route_state,
            helmholtz=startup.helmholtz,
            projection=startup.projection,
            finite=startup.finite,
            accepted=startup.accepted,
            status=startup.status,
            startup=True,
            method_id=self.method_id,
        )

    def step(
        self, history: MACImmersedBoundarySBDF2State, /, *, args: Any = None
    ) -> MACImmersedBoundarySBDF2Result:
        if not isinstance(history, MACImmersedBoundarySBDF2State):
            raise TypeError("history must be MACImmersedBoundarySBDF2State.")
        if history.method_id != self.method_id:
            raise ValueError("SBDF2 history belongs to another immersed method.")
        current = self.dynamics.validate_state(history.state)
        previous = self.dynamics.validate_state(history.previous_state)
        current_velocity = self.dynamics.unpack_velocity(current)
        previous_velocity = self.dynamics.unpack_velocity(previous)
        step = jnp.asarray(
            self.step_size,
            dtype=self.dynamics.momentum.operators.pressure_space.dtype,
        )
        attempted_time = history.time + step
        boundary_stage = self.dynamics.momentum.boundaries.evaluate(attempted_time, args)
        rhs = tuple(
            2.0 * now - 0.5 * old + step * (2.0 * now_rate - old_rate)
            for now, old, now_rate, old_rate in zip(
                current_velocity,
                previous_velocity,
                history.explicit_rate,
                history.previous_explicit_rate,
                strict=True,
            )
        )
        helmholtz = self.helmholtz.solve(
            rhs, boundary_stage, initial_guess=current_velocity
        )
        coefficient = (2.0 / 3.0) * step
        marker_kinematics = self.marker_motion(attempted_time, args)
        if not isinstance(marker_kinematics, LagrangianMarkerKinematics):
            raise TypeError("marker_motion must return LagrangianMarkerKinematics.")
        stage_inverse = MACHelmholtzStageInverseMomentum(
            self.helmholtz,
            boundary_stage,
            rhs_scale=step,
            stage_id=f"{self.method_id}/stage",
        )
        immersed = self.projection.project(
            helmholtz.value,
            stage_inverse,
            marker_kinematics,
            pressure=history.pressure,
            marker_force_density=history.marker_force_density,
            boundary_stage=boundary_stage,
            expected_routes=history.route_state,
            allow_route_refresh=self.startup_method.allow_route_refresh,
            marker_constraint_normals=self.startup_method.marker_constraint_normals,
        )
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            immersed.velocity, boundary_stage
        )
        candidate_state = self.dynamics.momentum.operators.velocity_space.flatten(
            candidate_velocity
        )
        finite = (
            boundary_stage.finite
            & helmholtz.finite
            & immersed.finite
            & jnp.all(jnp.isfinite(candidate_state))
        )
        accepted = (
            history.valid
            & boundary_stage.successful
            & helmholtz.converged
            & immersed.converged
            & finite
        )
        status = jnp.where(
            history.valid,
            jnp.where(
                ~boundary_stage.successful,
                int(MACImmersedBoundaryStepStatus.BOUNDARY_FAILED),
                jnp.where(
                    ~helmholtz.converged,
                    int(MACImmersedBoundaryStepStatus.HELMHOLTZ_FAILED),
                    jnp.where(
                        ~immersed.converged,
                        int(MACImmersedBoundaryStepStatus.PROJECTION_FAILED),
                        jnp.where(
                            ~finite,
                            int(MACImmersedBoundaryStepStatus.NONFINITE),
                            int(MACImmersedBoundaryStepStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
            history.status,
        ).astype(jnp.int32)
        next_explicit = jax.lax.cond(
            accepted,
            lambda _: self._explicit_rate(attempted_time, candidate_state, args),
            lambda _: history.explicit_rate,
            operand=None,
        )
        next_history = MACImmersedBoundarySBDF2State(
            time=jnp.where(accepted, attempted_time, history.time),
            previous_state=jnp.where(accepted, current, previous),
            state=jnp.where(accepted, candidate_state, current),
            previous_explicit_rate=tuple(
                jnp.where(accepted, now, old)
                for now, old in zip(
                    history.explicit_rate,
                    history.previous_explicit_rate,
                    strict=True,
                )
            ),
            explicit_rate=next_explicit,
            pressure=jnp.where(accepted, immersed.pressure, history.pressure),
            marker_force_density=jnp.where(
                accepted,
                immersed.marker_force_density,
                history.marker_force_density,
            ),
            route_state=immersed.route_state,
            accepted_steps=history.accepted_steps + accepted.astype(jnp.int32),
            valid=accepted,
            status=status,
            method_id=self.method_id,
        )
        return MACImmersedBoundarySBDF2Result(
            history=next_history,
            attempted_time=attempted_time,
            step_size=step,
            pressure_correction_coefficient=coefficient,
            velocity=tuple(
                jnp.where(accepted, candidate, original)
                for candidate, original in zip(
                    candidate_velocity, current_velocity, strict=True
                )
            ),
            pressure=next_history.pressure,
            route_state=next_history.route_state,
            marker_force_density=next_history.marker_force_density,
            marker_kinematics=marker_kinematics,
            helmholtz=helmholtz,
            projection=immersed,
            finite=finite,
            accepted=accepted,
            status=status,
            startup=False,
            method_id=self.method_id,
        )


__all__ = [
    "MACImmersedBoundaryIMEXEulerMethod",
    "MACImmersedBoundaryIMEXEulerResult",
    "MACImmersedBoundarySBDF2Method",
    "MACImmersedBoundarySBDF2Result",
    "MACImmersedBoundarySBDF2State",
    "MACImmersedBoundaryStepStatus",
    "MarkerMotionProvider",
]
