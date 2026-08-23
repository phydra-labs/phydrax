#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._implicit_stage import ImplicitStageArguments
from ._temporal_method import TemporalMethodCapabilities


class ThetaMethod(StrictModule, NonTrainableState):
    """Implicit theta method in midpoint or endpoint form."""

    capabilities: TemporalMethodCapabilities
    theta: float = eqx.field(static=True)
    endpoint: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, theta: float = 0.5, /, *, endpoint: bool = False):
        value = float(theta)
        if not isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("theta must be finite and lie in (0, 1].")
        if not isinstance(endpoint, bool):
            raise TypeError("endpoint must be a bool.")
        form = "endpoint" if endpoint else "midpoint"
        identifier = f"temporal:theta:{value.hex()}:{form}"
        second_order = value == 0.5
        self.theta = value
        self.endpoint = endpoint
        self.method_id = identifier
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("implicit-residual", "split-residual"),
            method_class="theta",
            order=2 if second_order else 1,
            adaptive=False,
            history_depth=1,
            stage_abscissae=(0.0, 1.0) if endpoint else (value,),
            causal_stage_extent=1.0,
            a_stable=value >= 0.5,
            l_stable=value == 1.0,
            stiffly_accurate=endpoint or value == 1.0,
            symplectic=second_order and not endpoint,
            reversible=second_order,
            verified=True,
            method_id=identifier,
        )


def endpoint_theta_stage_arguments(
    method: ThetaMethod,
    /,
    *,
    target_time: Array,
    previous: Array,
    previous_rate: Array,
    step_size: Array,
    model_args: Any,
    active: Array = jnp.asarray(True),
) -> ImplicitStageArguments:
    """Form the stiffly accurate endpoint theta stage."""
    if not isinstance(method, ThetaMethod) or not method.endpoint:
        raise ValueError("Endpoint stage arguments require endpoint ThetaMethod.")
    theta = method.theta
    shift = 1.0 / (theta * step_size)
    offset = -previous / (theta * step_size) - ((1.0 - theta) / theta) * previous_rate
    return ImplicitStageArguments(
        time=target_time,
        shift=shift,
        rate_offset=offset,
        explicit_value=jnp.zeros_like(previous),
        fallback_state=previous,
        active=active,
        model_args=model_args,
    )


def endpoint_theta_rate(
    method: ThetaMethod,
    state: Array,
    previous: Array,
    previous_rate: Array,
    step_size: Array,
    /,
) -> Array:
    arguments = endpoint_theta_stage_arguments(
        method,
        target_time=jnp.asarray(0.0, dtype=step_size.dtype),
        previous=previous,
        previous_rate=previous_rate,
        step_size=step_size,
        model_args=None,
    )
    return arguments.state_rate(state)


__all__ = [
    "ThetaMethod",
    "endpoint_theta_rate",
    "endpoint_theta_stage_arguments",
]
