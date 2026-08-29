#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import AbstractInputPolicy, DifferentialAlgebraicSystem


class ImplicitStageArguments(StrictModule):
    """One affine state-rate stage ``state_rate = shift * state + offset``."""

    time: Array
    shift: Array
    rate_offset: Array
    explicit_value: Array
    fallback_state: Array
    active: Array
    model_args: Any

    def __init__(
        self,
        *,
        time: ArrayLike,
        shift: ArrayLike,
        rate_offset: ArrayLike,
        explicit_value: ArrayLike,
        fallback_state: ArrayLike,
        active: ArrayLike,
        model_args: Any,
    ):
        time_ = jnp.asarray(time)
        shift_ = jnp.asarray(shift)
        offset = jnp.asarray(rate_offset)
        explicit = jnp.asarray(explicit_value)
        fallback = jnp.asarray(fallback_state)
        active_ = jnp.asarray(active, dtype=bool)
        if time_.shape != () or shift_.shape != () or active_.shape != ():
            raise ValueError(
                "Implicit stage time, shift, and active flag must be scalar."
            )
        if offset.shape != fallback.shape or explicit.shape != fallback.shape:
            raise ValueError(
                "Implicit stage rate offset, explicit value, and fallback state must align."
            )
        self.time = time_
        self.shift = shift_
        self.rate_offset = offset
        self.explicit_value = explicit
        self.fallback_state = fallback
        self.active = active_
        self.model_args = model_args

    def state_rate(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.rate_offset.shape:
            raise ValueError("Implicit stage state shape does not match its rate offset.")
        return self.shift * value + self.rate_offset


class ImplicitStageResidual(StrictModule):
    """Prepared residual form shared by BDF, theta, DIRK, and IMEX stages."""

    system: DifferentialAlgebraicSystem
    input_policy: AbstractInputPolicy | None

    def __init__(
        self,
        system: DifferentialAlgebraicSystem,
        input_policy: AbstractInputPolicy | None = None,
        /,
    ):
        if not isinstance(system, DifferentialAlgebraicSystem):
            raise TypeError("ImplicitStageResidual requires DifferentialAlgebraicSystem.")
        if input_policy is not None and not isinstance(
            input_policy, AbstractInputPolicy
        ):
            raise TypeError("input_policy must be an AbstractInputPolicy or None.")
        self.system = system
        self.input_policy = input_policy

    def __call__(
        self,
        state: Array,
        arguments: ImplicitStageArguments,
        /,
    ) -> Array:
        if not isinstance(arguments, ImplicitStageArguments):
            raise TypeError("arguments must be ImplicitStageArguments.")
        state_rate = arguments.state_rate(state)
        inputs = (
            None
            if self.input_policy is None
            else self.input_policy.evaluate(
                arguments.time,
                state,
                arguments.model_args,
            )
        )
        physical = self.system.scaled_residual(
            arguments.time,
            state,
            state_rate,
            arguments.model_args,
            inputs=inputs,
        )
        residual = physical - arguments.explicit_value / self.system.residual_scale
        return jnp.where(
            arguments.active,
            residual,
            state - arguments.fallback_state,
        )


__all__ = ["ImplicitStageArguments", "ImplicitStageResidual"]
