#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import LinearDescriptorSystem
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    plan,
    prepare,
    solve,
)


class DescriptorFrequencyResponse(StrictModule):
    response: Array
    state_response: Array
    residual_norm: Array
    relative_residual: Array
    condition_estimate: Array
    linear_status: Array
    finite: Array
    system_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.linear_status == int(LinearSolveStatus.SUCCESS)) & jnp.all(
            self.finite
        )


def descriptor_frequency_response(
    system: LinearDescriptorSystem,
    angular_frequency: ArrayLike,
    /,
    *,
    linear_policy: LinearSolvePolicy | None = None,
) -> DescriptorFrequencyResponse:
    if not isinstance(system, LinearDescriptorSystem):
        raise TypeError("system must be LinearDescriptorSystem.")
    omega = jnp.asarray(angular_frequency, dtype=float)
    case_shape = system.batch_shape + omega.shape
    mass = jnp.broadcast_to(
        system.mass_matrix.reshape(
            system.batch_shape + (1,) * omega.ndim + system.mass_matrix.shape[-2:]
        ),
        case_shape + system.mass_matrix.shape[-2:],
    )
    state = jnp.broadcast_to(
        system.state_matrix.reshape(
            system.batch_shape + (1,) * omega.ndim + system.state_matrix.shape[-2:]
        ),
        case_shape + system.state_matrix.shape[-2:],
    )
    inputs = jnp.broadcast_to(
        system.input_matrix.reshape(
            system.batch_shape + (1,) * omega.ndim + system.input_matrix.shape[-2:]
        ),
        case_shape + system.input_matrix.shape[-2:],
    )
    outputs = jnp.broadcast_to(
        system.output_matrix.reshape(
            system.batch_shape + (1,) * omega.ndim + system.output_matrix.shape[-2:]
        ),
        case_shape + system.output_matrix.shape[-2:],
    )
    feedthrough = jnp.broadcast_to(
        system.feedthrough_matrix.reshape(
            system.batch_shape + (1,) * omega.ndim + system.feedthrough_matrix.shape[-2:]
        ),
        case_shape + system.feedthrough_matrix.shape[-2:],
    )
    frequency = omega.reshape((1,) * len(system.batch_shape) + omega.shape)
    pencil = -1j * frequency[..., None, None] * mass - state
    inputs = inputs.astype(pencil.dtype)
    outputs = outputs.astype(pencil.dtype)
    feedthrough = feedthrough.astype(pencil.dtype)
    policy = (
        LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status"))
        if linear_policy is None
        else linear_policy
    )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    problem = LinearSystem(
        DenseLinearOperator(pencil, operator_id=f"{system.system_id}/frequency-pencil"),
        problem_id=f"{system.system_id}/frequency-response",
    )
    prepared = prepare(problem, plan(problem, policy))
    result = solve(prepared, inputs)
    state_response = jnp.asarray(result.value)
    response = outputs @ state_response + feedthrough
    residual = pencil @ state_response - inputs
    residual_norm = jnp.linalg.norm(residual, axis=(-2, -1))
    scale = jnp.maximum(
        jnp.linalg.norm(inputs, axis=(-2, -1))
        + jnp.linalg.norm(state_response, axis=(-2, -1)),
        1.0,
    )
    relative = residual_norm / scale
    condition = jnp.linalg.cond(pencil)
    finite = (
        jnp.all(jnp.isfinite(response), axis=(-2, -1))
        & jnp.all(jnp.isfinite(state_response), axis=(-2, -1))
        & jnp.isfinite(relative)
    )
    return DescriptorFrequencyResponse(
        response,
        state_response,
        residual_norm,
        relative,
        condition,
        jnp.asarray(result.status, dtype=jnp.int32),
        finite,
        system.system_id,
    )


__all__ = ["DescriptorFrequencyResponse", "descriptor_frequency_response"]
