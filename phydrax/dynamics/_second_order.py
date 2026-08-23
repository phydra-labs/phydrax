#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import DiscretizationBundle


SecondOrderResidual: TypeAlias = Callable[[Array, Array, Array, Array, Any], ArrayLike]


def _scale(value: ArrayLike | None, shape: tuple[int, ...], owner: str) -> Array:
    result = jnp.ones(shape) if value is None else jnp.asarray(value, dtype=float)
    result = jnp.broadcast_to(result, shape)
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)) | jnp.any(result <= 0.0),
        f"{owner} must be finite and positive.",
    )


class SecondOrderDifferentialSystem(StrictModule):
    """State-shaped residual ``F(t, q, v, a, args) = 0``."""

    residual: SecondOrderResidual
    configuration_scale: Array
    velocity_scale: Array
    acceleration_scale: Array
    residual_scale: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual: SecondOrderResidual,
        /,
        *,
        state_shape: Sequence[int],
        configuration_scale: ArrayLike | None = None,
        velocity_scale: ArrayLike | None = None,
        acceleration_scale: ArrayLike | None = None,
        residual_scale: ArrayLike | None = None,
        system_id: str,
    ):
        if not callable(residual):
            raise TypeError("Second-order residual must be callable.")
        shape = tuple(int(size) for size in state_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        identifier = str(system_id)
        if not identifier:
            raise ValueError("system_id must be non-empty.")
        self.residual = residual
        self.configuration_scale = _scale(
            configuration_scale, shape, "configuration_scale"
        )
        self.velocity_scale = _scale(velocity_scale, shape, "velocity_scale")
        self.acceleration_scale = _scale(acceleration_scale, shape, "acceleration_scale")
        self.residual_scale = _scale(residual_scale, shape, "residual_scale")
        self.state_shape = shape
        self.system_id = identifier

    def evaluate(
        self,
        time: ArrayLike,
        configuration: ArrayLike,
        velocity: ArrayLike,
        acceleration: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        time_ = jnp.asarray(time)
        q = jnp.asarray(configuration)
        v = jnp.asarray(velocity)
        a = jnp.asarray(acceleration)
        if time_.shape != () or q.shape != self.state_shape:
            raise ValueError("Second-order time/state shape is invalid.")
        if v.shape != self.state_shape or a.shape != self.state_shape:
            raise ValueError("Velocity and acceleration must match state_shape.")
        value = jnp.asarray(self.residual(time_, q, v, a, args))
        if value.shape != self.state_shape:
            raise ValueError("Second-order residual must preserve state_shape.")
        return value

    def scaled_residual(self, time, configuration, velocity, acceleration, args=None):
        value = self.evaluate(time, configuration, velocity, acceleration, args)
        return value / self.residual_scale.astype(value.dtype)


class SecondOrderDifferentialProblem(StrictModule):
    """Initial-value problem for one second-order residual system."""

    system: SecondOrderDifferentialSystem
    initial_configuration: Array
    initial_velocity: Array
    initial_acceleration: Array
    args: Any
    discretization_bundle: DiscretizationBundle | None
    problem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        system: SecondOrderDifferentialSystem,
        initial_configuration: ArrayLike,
        initial_velocity: ArrayLike,
        /,
        *,
        initial_acceleration: ArrayLike | None = None,
        args: Any = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(system, SecondOrderDifferentialSystem):
            raise TypeError("system must be SecondOrderDifferentialSystem.")
        configuration = jnp.asarray(initial_configuration)
        velocity = jnp.asarray(initial_velocity)
        acceleration = (
            jnp.zeros_like(configuration)
            if initial_acceleration is None
            else jnp.asarray(initial_acceleration)
        )
        if (
            configuration.shape != system.state_shape
            or velocity.shape != system.state_shape
            or acceleration.shape != system.state_shape
        ):
            raise ValueError("Initial second-order arrays must match system state_shape.")
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError("discretization_bundle must be DiscretizationBundle or None.")
        bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )
        payload = {
            "system_id": system.system_id,
            "state_shape": list(system.state_shape),
            "state_dtype": str(configuration.dtype),
            "discretization_bundle_id": bundle_id,
        }
        identifier = (
            f"second-order-problem:{canonical_fingerprint(payload)}"
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty or None.")
        self.system = system
        self.initial_configuration = configuration
        self.initial_velocity = velocity
        self.initial_acceleration = acceleration
        self.args = args
        self.discretization_bundle = discretization_bundle
        self.problem_id = identifier
        self.discretization_bundle_id = bundle_id


__all__ = [
    "SecondOrderDifferentialProblem",
    "SecondOrderDifferentialSystem",
    "SecondOrderResidual",
]
