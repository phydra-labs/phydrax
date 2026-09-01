#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


class SpacecraftDynamicsResult(StrictModule):
    times: Array
    position: Array
    velocity: Array
    orientation: Array
    angular_velocity: Array
    force: Array
    torque: Array
    valid: Array
    status: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SpacecraftDynamicsPlan(StrictModule, NonTrainableState):
    bodies: PreparedRigidBodySet
    context: AstrodynamicsContext
    times: Array
    load_function: Callable
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: PreparedRigidBodySet,
        context: AstrodynamicsContext,
        times: ArrayLike,
        load_function: Callable[[Array, RigidBodyKinematics, Any], RigidBodyLoad],
        /,
        *,
        load_id: str,
    ):
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("bodies must be a PreparedRigidBodySet.")
        if bodies.ambient_dimension != 3:
            raise ValueError(
                "Spacecraft dynamics requires three-dimensional rigid bodies."
            )
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        if not callable(load_function):
            raise TypeError("load_function must be callable.")
        identifier = str(load_id).strip()
        if not identifier:
            raise ValueError("load_id must be non-empty.")
        times_host = np.asarray(times, dtype=float)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
        ):
            raise ValueError("Spacecraft times must be finite and strictly increasing.")
        self.bodies = bodies
        self.context = context
        self.times = jnp.asarray(times_host)
        self.load_function = load_function
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spacecraft-dynamics-plan",
                "bodies": bodies.prepared_id,
                "context": context.context_id,
                "load_id": identifier,
                "num_times": int(times_host.size),
            }
        )

    def rollout(
        self,
        initial: RigidBodyKinematics,
        args: Any = None,
        /,
    ) -> SpacecraftDynamicsResult:
        if not isinstance(initial, RigidBodyKinematics):
            raise TypeError("initial must be RigidBodyKinematics.")
        initial_load = self.load_function(self.times[0], initial, args)
        if not isinstance(initial_load, RigidBodyLoad):
            raise TypeError("load_function must return RigidBodyLoad.")

        def step(carry, interval):
            kinematics, load, active = carry
            start, end = interval

            def advance(_):
                result = rigid_body_kick_drift_kick(
                    self.bodies,
                    kinematics,
                    load,
                    start,
                    end - start,
                    self.load_function,
                    args,
                )
                return result.kinematics, result.load, result.successful

            next_kinematics, next_load, valid = jax.lax.cond(
                active,
                advance,
                lambda _: (kinematics, load, jnp.asarray(False)),
                operand=None,
            )
            accepted = active & valid
            return (next_kinematics, next_load, accepted), (
                next_kinematics,
                next_load,
                accepted,
            )

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        (_, _, completed), outputs = jax.lax.scan(
            step,
            (initial, initial_load, jnp.asarray(True)),
            intervals,
        )
        kinematics, loads, valid_tail = outputs
        position = jnp.concatenate((initial.position[None], kinematics.position), axis=0)
        velocity = jnp.concatenate((initial.velocity[None], kinematics.velocity), axis=0)
        orientation = jnp.concatenate(
            (initial.orientation[None], kinematics.orientation), axis=0
        )
        angular = jnp.concatenate(
            (initial.angular_velocity[None], kinematics.angular_velocity), axis=0
        )
        force = jnp.concatenate((initial_load.force[None], loads.force), axis=0)
        torque = jnp.concatenate((initial_load.torque[None], loads.torque), axis=0)
        initial_valid = jnp.all(jnp.isfinite(position[0])) & jnp.all(
            jnp.isfinite(orientation[0])
        )
        valid = jnp.concatenate((initial_valid[None], valid_tail))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return SpacecraftDynamicsResult(
            self.times,
            position,
            velocity,
            orientation,
            angular,
            force,
            torque,
            valid,
            status,
            completed,
            self.plan_id,
        )


class FiniteBurnEvaluation(StrictModule):
    load: RigidBodyLoad
    mass_flow_rate: Array
    thrust: Array
    saturated: Array
    valid: Array
    burn_id: str = eqx.field(static=True)


class FiniteBurnPlan(StrictModule, NonTrainableState):
    direction_body: Array
    lever_arm_body: Array
    maximum_thrust: Array
    specific_impulse: Array
    standard_gravity: Array
    burn_id: str = eqx.field(static=True)

    def __init__(
        self,
        direction_body: ArrayLike,
        /,
        *,
        maximum_thrust: ArrayLike,
        specific_impulse: ArrayLike,
        lever_arm_body: ArrayLike | tuple[float, float, float] = (0.0, 0.0, 0.0),
        standard_gravity: ArrayLike = 9.80665,
        burn_id: str,
    ):
        direction = np.asarray(direction_body, dtype=float)
        lever = np.asarray(lever_arm_body, dtype=float)
        if direction.shape != (3,) or lever.shape != (3,):
            raise ValueError("Burn direction and lever arm must have shape (3,).")
        norm = float(np.sqrt(np.sum(direction * direction)))
        if not np.isfinite(norm) or norm <= 0.0 or np.any(~np.isfinite(lever)):
            raise ValueError("Burn geometry must be finite and nondegenerate.")
        identifier = str(burn_id).strip()
        if not identifier:
            raise ValueError("burn_id must be non-empty.")
        self.direction_body = jnp.asarray(direction / norm)
        self.lever_arm_body = jnp.asarray(lever)
        self.maximum_thrust = jnp.asarray(maximum_thrust).reshape(())
        self.specific_impulse = jnp.asarray(specific_impulse).reshape(())
        self.standard_gravity = jnp.asarray(standard_gravity).reshape(())
        self.burn_id = identifier

    def evaluate(
        self,
        kinematics: RigidBodyKinematics,
        throttle: ArrayLike,
        /,
    ) -> FiniteBurnEvaluation:
        throttle_ = jnp.asarray(throttle)
        if throttle_.shape not in ((), (kinematics.position.shape[0],)):
            raise ValueError("throttle must be scalar or body-capacity shaped.")
        throttle_ = jnp.broadcast_to(throttle_, (kinematics.position.shape[0],))
        clipped = jnp.clip(throttle_, 0.0, 1.0)
        saturated = clipped != throttle_
        rotation = quaternion_rotation_matrix(kinematics.orientation)
        direction = contract("...ij,j->...i", rotation, self.direction_body)
        lever = contract("...ij,j->...i", rotation, self.lever_arm_body)
        thrust = clipped * self.maximum_thrust
        force = thrust[:, None] * direction
        torque = jnp.cross(lever, force)
        denominator = self.specific_impulse * self.standard_gravity
        valid = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(torque))
            & jnp.isfinite(denominator)
            & (denominator > 0.0)
            & (self.maximum_thrust >= 0.0)
        )
        mass_flow = jnp.where(valid, thrust / denominator, 0.0)
        return FiniteBurnEvaluation(
            RigidBodyLoad(
                jnp.where(valid, force, jnp.zeros_like(force)),
                jnp.where(valid, torque, jnp.zeros_like(torque)),
            ),
            mass_flow,
            thrust,
            saturated,
            valid,
            self.burn_id,
        )


class VariableMassSpacecraftState(StrictModule):
    kinematics: RigidBodyKinematics
    propellant_mass: Array

    def __init__(self, kinematics: RigidBodyKinematics, propellant_mass: ArrayLike, /):
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        mass = jnp.asarray(propellant_mass)
        if mass.shape != (kinematics.position.shape[0],):
            raise ValueError("propellant_mass must have body-capacity shape.")
        self.kinematics = kinematics
        self.propellant_mass = mass


def deplete_propellant(
    state: VariableMassSpacecraftState,
    burn: FiniteBurnEvaluation,
    step_size: ArrayLike,
    /,
) -> tuple[VariableMassSpacecraftState, Array]:
    if not isinstance(state, VariableMassSpacecraftState):
        raise TypeError("state must be VariableMassSpacecraftState.")
    if not isinstance(burn, FiniteBurnEvaluation):
        raise TypeError("burn must be FiniteBurnEvaluation.")
    step = jnp.asarray(step_size).reshape(())
    required = burn.mass_flow_rate * step
    sufficient = state.propellant_mass >= required
    next_mass = jnp.where(
        sufficient & burn.valid, state.propellant_mass - required, state.propellant_mass
    )
    return VariableMassSpacecraftState(state.kinematics, next_mass), jnp.all(
        sufficient & burn.valid
    )


class ReactionWheelSet(StrictModule, NonTrainableState):
    axes_body: Array
    inertias: Array
    maximum_torque: Array
    maximum_momentum: Array
    wheel_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes_body: ArrayLike,
        inertias: ArrayLike,
        maximum_torque: ArrayLike,
        maximum_momentum: ArrayLike,
        /,
        *,
        wheel_id: str,
    ):
        axes = np.asarray(axes_body, dtype=float)
        inertias_ = np.asarray(inertias, dtype=float)
        torque = np.asarray(maximum_torque, dtype=float)
        momentum = np.asarray(maximum_momentum, dtype=float)
        if axes.ndim != 2 or axes.shape[1] != 3:
            raise ValueError("Reaction-wheel axes must have shape (W,3).")
        count = axes.shape[0]
        if (
            inertias_.shape != (count,)
            or torque.shape != (count,)
            or momentum.shape != (count,)
        ):
            raise ValueError("Reaction-wheel properties must match wheel count.")
        norms = np.sqrt(np.sum(axes * axes, axis=1))
        if (
            np.any(~np.isfinite(axes))
            or np.any(norms <= 0.0)
            or np.any(inertias_ <= 0.0)
            or np.any(torque <= 0.0)
            or np.any(momentum <= 0.0)
        ):
            raise ValueError("Reaction-wheel properties must be finite and positive.")
        self.axes_body = jnp.asarray(axes / norms[:, None])
        self.inertias = jnp.asarray(inertias_)
        self.maximum_torque = jnp.asarray(torque)
        self.maximum_momentum = jnp.asarray(momentum)
        self.wheel_id = str(wheel_id)
        if not self.wheel_id:
            raise ValueError("wheel_id must be non-empty.")

    def actuate(
        self,
        commanded_torque: ArrayLike,
        wheel_momentum: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        command = jnp.asarray(commanded_torque)
        momentum = jnp.asarray(wheel_momentum)
        if command.shape != self.maximum_torque.shape or momentum.shape != command.shape:
            raise ValueError("Wheel command and momentum must have wheel shape.")
        applied = jnp.clip(command, -self.maximum_torque, self.maximum_torque)
        acceleration = applied / self.inertias
        body_torque = -contract("wi,w->i", self.axes_body, applied)
        saturated = (applied != command) | (jnp.abs(momentum) >= self.maximum_momentum)
        return body_torque, acceleration, saturated


__all__ = [
    "FiniteBurnEvaluation",
    "FiniteBurnPlan",
    "ReactionWheelSet",
    "SpacecraftDynamicsPlan",
    "SpacecraftDynamicsResult",
    "VariableMassSpacecraftState",
    "deplete_propellant",
]
