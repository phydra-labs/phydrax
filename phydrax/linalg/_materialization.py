#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ._properties import LinearCapabilityError


class MaterializationPolicy(StrictModule):
    """Explicit dense-materialization permission and memory limits."""

    max_entries: int = eqx.field(static=True)
    max_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_entries: int = 1_000_000,
        max_bytes: int = 64 * 1024 * 1024,
    ):
        entries = int(max_entries)
        byte_count = int(max_bytes)
        if entries < 1 or byte_count < 1:
            raise ValueError("Materialization limits must be positive.")
        self.max_entries = entries
        self.max_bytes = byte_count


def materialize(operator, policy: MaterializationPolicy, /) -> Array:
    """Materialize an operator only under an explicit bounded policy."""
    from ._operators import AbstractLinearOperator

    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not isinstance(policy, MaterializationPolicy):
        raise TypeError("policy must be a MaterializationPolicy.")
    if not operator.capabilities.materialize:
        raise LinearCapabilityError("Operator does not support dense materialization.")
    entries = prod(operator.batch_shape) * operator.source.size * operator.target.size
    if entries > policy.max_entries:
        raise LinearCapabilityError(
            f"Dense materialization requires {entries} entries, exceeding "
            f"the policy limit {policy.max_entries}."
        )
    expected = operator.batch_shape + (operator.target.size, operator.source.size)
    target_dtypes = [spec.dtype for spec in jax.tree.leaves(operator.target.structure())]
    expected_dtype = jnp.dtype(jnp.result_type(*target_dtypes))
    required_bytes = entries * expected_dtype.itemsize
    if required_bytes > policy.max_bytes:
        raise LinearCapabilityError(
            f"Dense materialization requires {required_bytes} bytes, exceeding "
            f"the policy limit {policy.max_bytes}."
        )
    matrix = jnp.asarray(operator._materialize())
    if matrix.shape != expected or matrix.dtype != expected_dtype:
        raise ValueError(
            "Operator materialization must have shape and dtype "
            f"{expected} and {expected_dtype}; got {matrix.shape} and {matrix.dtype}."
        )
    if matrix.nbytes > policy.max_bytes:
        raise LinearCapabilityError(
            f"Dense materialization produced {matrix.nbytes} bytes, exceeding "
            f"the policy limit {policy.max_bytes}."
        )
    return matrix


__all__ = ["MaterializationPolicy", "materialize"]
