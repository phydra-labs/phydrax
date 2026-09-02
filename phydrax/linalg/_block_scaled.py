#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp

from phydrax.ein import contract

from .._precision import dequantize_mx, MicroscaledArray


def contract_block_scaled(
    subscripts: str,
    *operands: Any,
    compute_dtype: Any = jnp.float32,
    provider: Literal["portable", "fused"] = "portable",
):
    """Contract scalar or block-scaled operands with explicit wide accumulation.

    The portable path is the correctness contract. Fused execution is rejected until a
    concrete public provider supports the exact format, shape, and device contract.
    """

    if not isinstance(subscripts, str) or not subscripts:
        raise ValueError("subscripts must be a non-empty contraction expression.")
    if len(operands) < 1:
        raise ValueError("At least one contraction operand is required.")
    if provider != "portable":
        raise ValueError(
            "No fused block-scaled provider is available for this runtime contract."
        )
    dtype = jnp.dtype(compute_dtype)
    if not jnp.issubdtype(dtype, jnp.floating) or dtype.itemsize < 4:
        raise TypeError(
            "Block-scaled contraction compute dtype must be float32 or wider."
        )
    decoded = tuple(
        dequantize_mx(value, dtype=dtype)
        if isinstance(value, MicroscaledArray)
        else jnp.asarray(value, dtype=dtype)
        for value in operands
    )
    return contract(subscripts, *decoded, backend="jax").astype(dtype)


__all__ = ["contract_block_scaled"]
