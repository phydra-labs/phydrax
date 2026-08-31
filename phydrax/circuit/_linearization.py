#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..control._descriptor_frequency import (
    descriptor_frequency_response,
    DescriptorFrequencyResponse,
)
from ..dynamics._linear_descriptor import LinearDescriptorSystem
from ..linalg import DenseLinearOperator
from ..linalg.eigen import general_eigensolve, GeneralEigenproblem
from ._dae import PreparedCircuitDAE


class CircuitLinearizationResult(StrictModule):
    descriptor: LinearDescriptorSystem
    operating_state: Array
    operating_residual: Array
    residual_norm: Array
    prepared_dae_id: str = eqx.field(static=True)
    linearization_id: str = eqx.field(static=True)


class DescriptorPoleResult(StrictModule):
    poles: Array
    finite: Array
    stable: Array
    eigensolve: Any
    system_id: str = eqx.field(static=True)


def linearize_circuit(
    prepared_dae: PreparedCircuitDAE,
    state: ArrayLike,
    /,
    *,
    time: ArrayLike = 0.0,
    args: Any = None,
) -> CircuitLinearizationResult:
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    value = jnp.asarray(state, dtype=float)
    if value.shape != (prepared_dae.plan.layout.size,):
        raise ValueError("Linearization state has the wrong shape.")
    time_ = jnp.asarray(time, dtype=float)
    if time_.shape != ():
        raise ValueError("Linearization time must be scalar.")
    zero_rate = jnp.zeros_like(value)

    def residual(current_state, current_rate):
        return prepared_dae.system.evaluate(time_, current_state, current_rate, args)

    state_jacobian, rate_jacobian = jax.jacfwd(residual, argnums=(0, 1))(value, zero_rate)
    operating_residual = residual(value, zero_rate)
    size = value.size
    identity = jnp.eye(size, dtype=value.dtype)
    descriptor = LinearDescriptorSystem(
        rate_jacobian,
        -state_jacobian,
        identity,
        identity,
        jnp.zeros((size, size), dtype=value.dtype),
        system_id=f"{prepared_dae.plan.circuit.circuit_id}/linearization",
    )
    identifier = canonical_fingerprint(
        {
            "kind": "circuit-linearization",
            "dae": prepared_dae.prepared_id,
            "descriptor": descriptor.system_id,
        }
    )
    return CircuitLinearizationResult(
        descriptor,
        value,
        operating_residual,
        jnp.linalg.norm(operating_residual),
        prepared_dae.prepared_id,
        identifier,
    )


def circuit_small_signal_response(
    linearization: CircuitLinearizationResult,
    angular_frequency: ArrayLike,
    /,
) -> DescriptorFrequencyResponse:
    if not isinstance(linearization, CircuitLinearizationResult):
        raise TypeError("linearization must be CircuitLinearizationResult.")
    return descriptor_frequency_response(linearization.descriptor, angular_frequency)


def descriptor_poles(
    system: LinearDescriptorSystem,
    /,
    *,
    policy: Any = None,
) -> DescriptorPoleResult:
    if not isinstance(system, LinearDescriptorSystem):
        raise TypeError("system must be LinearDescriptorSystem.")
    if system.batch_shape:
        raise ValueError("Descriptor pole analysis requires one unbatched system.")
    state = DenseLinearOperator(
        system.state_matrix, operator_id=f"{system.system_id}/state"
    )
    mass = DenseLinearOperator(system.mass_matrix, operator_id=f"{system.system_id}/mass")
    result = general_eigensolve(
        GeneralEigenproblem(
            state,
            mass,
            problem_id=f"{system.system_id}/poles",
        ),
        policy=policy,
    )
    poles = result.eigenvalues
    finite = jnp.isfinite(poles)
    stable = jnp.all(jnp.where(finite, jnp.real(poles) < 0.0, True))
    return DescriptorPoleResult(poles, finite, stable, result, system.system_id)


__all__ = [
    "CircuitLinearizationResult",
    "DescriptorPoleResult",
    "circuit_small_signal_response",
    "descriptor_poles",
    "linearize_circuit",
]
