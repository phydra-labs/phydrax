#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import StateLayout
from ._context import AstrodynamicsContext


class CartesianOrbitState(StrictModule):
    """Cartesian position and velocity in one explicit astrodynamics context."""

    position: Array
    velocity: Array
    context: AstrodynamicsContext

    def __init__(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        context: AstrodynamicsContext,
        /,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        if position_.shape != (3,) or velocity_.shape != (3,):
            raise ValueError(
                "Cartesian orbit position and velocity must have shape (3,)."
            )
        if not jnp.issubdtype(position_.dtype, jnp.inexact):
            position_ = position_.astype(float)
            velocity_ = velocity_.astype(float)
        position_ = eqx.error_if(
            position_,
            ~jnp.all(jnp.isfinite(position_)) | ~jnp.all(jnp.isfinite(velocity_)),
            "Cartesian orbit state must be finite.",
        )
        self.position = position_
        self.velocity = velocity_
        self.context = context

    def packed(self) -> Array:
        return jnp.concatenate((self.position, self.velocity))


class CartesianOrbitTrajectory(StrictModule, NonTrainableState):
    """Dense Cartesian trajectory with one shared context and validity evidence."""

    times: Array
    states: Array
    valid: Array
    status: Array
    context: AstrodynamicsContext
    trajectory_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        trajectory_id: str | None = None,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        times_ = jnp.asarray(times)
        states_ = jnp.asarray(states)
        valid_ = jnp.asarray(valid, dtype=bool)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if times_.ndim != 1 or states_.shape != (times_.shape[0], 6):
            raise ValueError("Trajectory states must have shape (num_times, 6).")
        if valid_.shape != times_.shape or status_.shape != times_.shape:
            raise ValueError("Trajectory validity and status must match the time axis.")
        self.times = times_
        self.states = states_
        self.valid = valid_
        self.status = status_
        self.context = context
        generated = canonical_fingerprint(
            {
                "kind": "cartesian-orbit-trajectory",
                "context": context.context_id,
                "num_times": int(times_.shape[0]),
            }
        )
        self.trajectory_id = generated if trajectory_id is None else str(trajectory_id)
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty.")


CARTESIAN_ORBIT_STATE_LAYOUT = StateLayout(
    (6,),
    axes=("phase_component",),
    component_names=("x", "y", "z", "vx", "vy", "vz"),
    layout_id="astrodynamics:cartesian-orbit-state",
)


def pack_cartesian_state(state: CartesianOrbitState, /) -> Array:
    if not isinstance(state, CartesianOrbitState):
        raise TypeError("state must be a CartesianOrbitState.")
    return state.packed()


def unpack_cartesian_state(
    values: ArrayLike,
    context: AstrodynamicsContext,
    /,
) -> CartesianOrbitState:
    packed = jnp.asarray(values)
    if packed.shape != (6,):
        raise ValueError("Packed Cartesian orbit state must have shape (6,).")
    return CartesianOrbitState(packed[:3], packed[3:], context)


__all__ = [
    "CARTESIAN_ORBIT_STATE_LAYOUT",
    "CartesianOrbitState",
    "CartesianOrbitTrajectory",
    "pack_cartesian_state",
    "unpack_cartesian_state",
]
