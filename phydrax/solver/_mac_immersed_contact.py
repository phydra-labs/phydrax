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
from ..discretization.particle import (
    HardContactState,
    HardContactStepResult,
    PreparedHardContact,
    PreparedRigidConstraintDynamics,
    RigidBodyKinematics,
    RigidConstraintState,
    RigidConstraintStepResult,
    RigidContactGeometry,
)
from ._mac_immersed_rigid import (
    MACRigidImmersedBackwardEulerMethod,
    MACRigidImmersedMidpointMethod,
    MACRigidImmersedStepResult,
)


RigidContactGeometryProvider = Callable[[object], RigidContactGeometry]
RigidImmersedAcceptedMethod = (
    MACRigidImmersedBackwardEulerMethod | MACRigidImmersedMidpointMethod
)


class MACRigidImmersedContactResult(StrictModule):
    immersed: MACRigidImmersedStepResult
    contact: HardContactStepResult
    contact_state: HardContactState
    coupling_residual: Array
    iterations: Array
    accepted: Array
    method_id: str = eqx.field(static=True)


class MACRigidImmersedContactMethod(StrictModule, NonTrainableState):
    """Accepted-time fixed-point coupling of fluid constraints and hard contact."""

    immersed: RigidImmersedAcceptedMethod
    contact: PreparedHardContact
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        immersed: RigidImmersedAcceptedMethod,
        contact: PreparedHardContact,
        /,
        *,
        maximum_iterations: int = 8,
        tolerance: float = 1.0e-9,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        if iterations <= 0 or tolerance_ <= 0.0:
            raise ValueError("Coupled-contact iteration limits must be positive.")
        if contact.bodies.prepared_id != (
            immersed.backward_euler.base.projection.rigid_markers.bodies.prepared_id
            if isinstance(immersed, MACRigidImmersedMidpointMethod)
            else immersed.base.projection.rigid_markers.bodies.prepared_id
        ):
            raise ValueError("Contact and immersed methods use different rigid bodies.")
        self.immersed = immersed
        self.contact = contact
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-hard-contact",
                "immersed": immersed.method_id,
                "contact": contact.prepared_id,
                "iterations": iterations,
                "tolerance": tolerance_,
            }
        )

    def step(
        self,
        time: ArrayLike,
        fluid_state: ArrayLike,
        body_kinematics,
        contact_state: HardContactState,
        geometry: RigidContactGeometryProvider,
        /,
        **kwargs: Any,
    ) -> MACRigidImmersedContactResult:
        guess = body_kinematics
        residual = jnp.asarray(jnp.inf, dtype=body_kinematics.position.dtype)
        immersed_result = None
        contact_result = None
        for iteration in range(self.maximum_iterations):
            immersed_result = self.immersed.step(
                time,
                fluid_state,
                guess,
                **kwargs,
            )
            contact_result = self.contact.evaluate(
                contact_state,
                immersed_result.body_kinematics,
                geometry(immersed_result.body_kinematics),
                self.immersed.backward_euler.base.step_size
                if isinstance(self.immersed, MACRigidImmersedMidpointMethod)
                else self.immersed.base.step_size,
            )
            corrected = contact_result.accepted_kinematics
            residual = jnp.maximum(
                jnp.max(
                    jnp.abs(corrected.velocity - immersed_result.body_kinematics.velocity)
                ),
                jnp.max(
                    jnp.abs(
                        corrected.angular_velocity
                        - immersed_result.body_kinematics.angular_velocity
                    )
                ),
            )
            guess = RigidBodyKinematics(
                body_kinematics.position,
                corrected.velocity,
                body_kinematics.orientation,
                corrected.angular_velocity,
            )
        if immersed_result is None or contact_result is None:
            raise RuntimeError("Coupled contact iteration did not execute.")
        accepted = (
            immersed_result.accepted
            & contact_result.successful
            & (residual <= self.tolerance)
        )
        return MACRigidImmersedContactResult(
            immersed_result,
            contact_result,
            contact_result.accepted_state,
            residual,
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            accepted,
            self.method_id,
        )


class MACRigidImmersedJointResult(StrictModule):
    immersed: MACRigidImmersedStepResult
    constraints: RigidConstraintStepResult
    constraint_state: RigidConstraintState
    coupling_residual: Array
    iterations: Array
    accepted: Array
    method_id: str = eqx.field(static=True)


class MACRigidImmersedJointMethod(StrictModule, NonTrainableState):
    """Accepted-time fixed-point coupling of immersed and rigid-joint solves."""

    immersed: RigidImmersedAcceptedMethod
    constraints: PreparedRigidConstraintDynamics
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        immersed: RigidImmersedAcceptedMethod,
        constraints: PreparedRigidConstraintDynamics,
        /,
        *,
        maximum_iterations: int = 8,
        tolerance: float = 1.0e-9,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        if iterations <= 0 or tolerance_ <= 0.0:
            raise ValueError("Coupled-joint iteration limits must be positive.")
        self.immersed = immersed
        self.constraints = constraints
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.method_id = canonical_fingerprint(
            {
                "kind": "mac-rigid-immersed-joints",
                "immersed": immersed.method_id,
                "constraints": constraints.prepared_id,
                "iterations": iterations,
                "tolerance": tolerance_,
            }
        )

    def step(
        self,
        time: ArrayLike,
        fluid_state: ArrayLike,
        state: RigidConstraintState,
        /,
        *,
        args: Any = None,
        **kwargs: Any,
    ) -> MACRigidImmersedJointResult:
        iterate = state
        residual = jnp.asarray(jnp.inf, dtype=state.kinematics.position.dtype)
        immersed_result = None
        constraint_result = None
        step_size = (
            self.immersed.backward_euler.base.step_size
            if isinstance(self.immersed, MACRigidImmersedMidpointMethod)
            else self.immersed.base.step_size
        )
        for iteration in range(self.maximum_iterations):
            constraint_result = self.constraints.step(iterate, time, step_size, args)
            immersed_result = self.immersed.step(
                time,
                fluid_state,
                constraint_result.accepted_state.kinematics,
                args=args,
                **kwargs,
            )
            residual = jnp.maximum(
                jnp.max(
                    jnp.abs(
                        immersed_result.body_kinematics.velocity
                        - constraint_result.accepted_state.kinematics.velocity
                    )
                ),
                jnp.max(
                    jnp.abs(
                        immersed_result.body_kinematics.angular_velocity
                        - constraint_result.accepted_state.kinematics.angular_velocity
                    )
                ),
            )
            iterate_kinematics = RigidBodyKinematics(
                state.kinematics.position,
                immersed_result.body_kinematics.velocity,
                state.kinematics.orientation,
                immersed_result.body_kinematics.angular_velocity,
            )
            iterate = RigidConstraintState(
                iterate_kinematics,
                constraint_result.accepted_state.position_multiplier_guess,
                constraint_result.accepted_state.velocity_multiplier_guess,
            )
        if immersed_result is None or constraint_result is None:
            raise RuntimeError("Coupled joint iteration did not execute.")
        accepted = (
            immersed_result.accepted
            & constraint_result.successful
            & (residual <= self.tolerance)
        )
        return MACRigidImmersedJointResult(
            immersed_result,
            constraint_result,
            iterate,
            residual,
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            accepted,
            self.method_id,
        )


__all__ = [
    "MACRigidImmersedContactMethod",
    "MACRigidImmersedContactResult",
    "MACRigidImmersedJointMethod",
    "MACRigidImmersedJointResult",
    "RigidContactGeometryProvider",
    "RigidImmersedAcceptedMethod",
]
