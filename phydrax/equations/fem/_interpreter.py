#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._ir import FiniteElementActionIR
from ._operators import (
    average,
    curl,
    divergence,
    FieldJet,
    jump,
    normal_trace,
    symmetric_gradient,
    tangential_trace,
)


def evaluate_differential_operator(
    operation: str,
    jet: FieldJet,
    /,
    *,
    normal: ArrayLike | None = None,
    other: FieldJet | None = None,
) -> Array:
    if operation == "value":
        return jet.value
    if operation == "grad":
        if jet.gradient is None:
            raise ValueError("Gradient was not prepared for this field slot.")
        return jet.gradient
    if operation == "sym-grad":
        if jet.gradient is None:
            raise ValueError("Symmetric gradient requires a prepared gradient.")
        return symmetric_gradient(jet.gradient)
    if operation == "div":
        if jet.divergence is not None:
            return jet.divergence
        if jet.gradient is None:
            raise ValueError("Divergence requires a prepared gradient/divergence.")
        return divergence(jet.gradient)
    if operation == "curl":
        if jet.curl is not None:
            return jet.curl
        if jet.gradient is None:
            raise ValueError("Curl requires a prepared gradient/curl.")
        return curl(jet.gradient)
    if operation == "normal-trace":
        if normal is None:
            raise ValueError("Normal trace requires a normal.")
        return normal_trace(jet.value, normal)
    if operation == "tangential-trace":
        if normal is None:
            raise ValueError("Tangential trace requires a normal.")
        return tangential_trace(jet.value, normal)
    if operation == "jump":
        if other is None:
            raise ValueError("Jump requires plus and minus field jets.")
        return jump(jet.value, other.value)
    if operation == "average":
        if other is None:
            raise ValueError("Average requires plus and minus field jets.")
        return average(jet.value, other.value)
    raise ValueError(f"Operation {operation!r} requires a method-specific reducer.")


def execute_local_action(
    action: FiniteElementActionIR,
    kernels: Mapping[str, Callable],
    field_jets: Mapping[str, FieldJet],
    context: object,
    /,
    *,
    normals: ArrayLike | None = None,
    other_jets: Mapping[str, FieldJet] | None = None,
) -> Array:
    if not isinstance(action, FiniteElementActionIR):
        raise TypeError("action must be FiniteElementActionIR.")
    if action.kernel_id not in kernels:
        raise KeyError(f"No local kernel registered for {action.kernel_id!r}.")
    evaluated = {}
    for slot_name, operation in action.operators:
        if slot_name not in field_jets:
            raise KeyError(f"No field jet exists for slot {slot_name!r}.")
        other = None if other_jets is None else other_jets.get(slot_name)
        evaluated[(slot_name, operation)] = evaluate_differential_operator(
            operation,
            field_jets[slot_name],
            normal=normals,
            other=other,
        )
    result = jnp.asarray(kernels[action.kernel_id](evaluated, context))
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


__all__ = ["evaluate_differential_operator", "execute_local_action"]
