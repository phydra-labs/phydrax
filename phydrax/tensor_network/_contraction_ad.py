# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Native-owned multilinear rules, independent of vendor backward/cache APIs.

The primitive operates on unique modes. Native diagonal gathers precede it, so
JAX owns the matching diagonal scatter in transposition. Ordinary complex
contraction is bilinear: its JAX transpose does not conjugate either operand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ._contraction_provider import TensorOperation


_tensor = core.Primitive("phydrax_prepared_tensor")


def _fixed_path(count: int, /) -> tuple[tuple[int, ...], ...]:
    # A deterministic algebraic lowering, never an optimizer/search in tracing.
    return tuple((0, 1) for _ in range(count - 1))


@lru_cache(maxsize=512)
def transpose_operation(
    operation: TensorOperation, index: int, /
) -> tuple[TensorOperation, tuple[int, ...]]:
    others = tuple(
        modes for position, modes in enumerate(operation.inputs) if position != index
    )
    available = set(operation.output).union(*(set(modes) for modes in others))
    missing = tuple(mode for mode in operation.inputs[index] if mode not in available)
    inputs = (operation.output, *others, *((mode,) for mode in missing))
    transposed = replace(
        operation,
        inputs=inputs,
        output=operation.inputs[index],
        path=_fixed_path(len(inputs)),
    )
    admit_operation(transposed)
    return transposed, missing


def admit_operation(operation: TensorOperation, /) -> int:
    """Bound explicit live buffers, algebraic intermediates, and provider scratch."""
    extents = dict(operation.dimensions)
    sizes = [prod(extents[mode] for mode in modes) for modes in operation.inputs]
    live = list(operation.inputs)
    peak = sum(sizes)
    for positions in operation.path:
        if (
            not positions
            or len(set(positions)) != len(positions)
            or min(positions) < 0
            or max(positions) >= len(live)
        ):
            raise ValueError("Invalid provider evolving-live-list path.")
        selected = tuple(live[index] for index in positions)
        remaining = tuple(
            modes for index, modes in enumerate(live) if index not in positions
        )
        needed = set(operation.output).union(*(set(modes) for modes in remaining))
        output = tuple(
            mode
            for mode in dict.fromkeys(mode for modes in selected for mode in modes)
            if mode in needed
        )
        result_size = prod(extents[mode] for mode in output)
        peak = max(peak, sum(sizes) + result_size)
        for index in sorted(positions, reverse=True):
            live.pop(index)
            sizes.pop(index)
        live.append(output)
        sizes.append(result_size)
    itemsize = np.dtype(operation.dtype).itemsize
    # A tangent sum retains the primal and one accumulator beside one term.
    required = (
        peak + 2 * prod(operation.output_shape)
    ) * itemsize + operation.cache_limit_bytes
    if required > operation.maximum_derivative_bytes:
        raise MemoryError("Tensor derivative lowering exceeds maximum_derivative_bytes.")
    return required


def admit_derivatives(operation: TensorOperation, /) -> int:
    """Prepare first/second transpose recipes and return their largest byte bound."""
    required = admit_operation(operation)
    for index in range(len(operation.inputs)):
        first, _ = transpose_operation(operation, index)
        required = max(required, admit_operation(first))
        for second_index in range(len(first.inputs)):
            second, _ = transpose_operation(first, second_index)
            required = max(required, admit_operation(second))
    return required


@lru_cache(maxsize=512)
def _ffi_callable(operation: TensorOperation) -> Any:
    return jax.ffi.ffi_call(
        f"phydrax_{operation.provider}_contract",
        jax.ShapeDtypeStruct(operation.output_shape, np.dtype(operation.dtype)),
        vmap_method="sequential",
        input_layouts=tuple(tuple(range(len(modes))) for modes in operation.inputs),
        output_layouts=tuple(range(len(operation.output))),
    )


def _ffi(*operands: Array, operation: TensorOperation) -> Array:
    from .._cuda_native import load_cuda_component

    load_cuda_component(operation.provider)
    return _ffi_callable(operation)(
        *operands,
        recipe=operation.recipe,
        workspace_limit_bytes=np.int64(operation.workspace_limit_bytes),
        cache_limit_bytes=np.int64(operation.cache_limit_bytes),
        partition=np.int64(operation.partition),
        capacity=np.int64(operation.capacity),
    )


def _implementation(*operands: Array, operation: TensorOperation) -> Array:
    if operation.provider == "cutensornet":
        return _ffi(*operands, operation=operation)
    if len(operands) <= 2:
        return _ffi(*operands, operation=replace(operation, path=()))
    # A derivative may add explicit broadcast vectors. cuTENSOR is binary; lower
    # the declared fixed path, preserving every retained shared/hyperedge mode.
    live_modes = list(operation.inputs)
    values = list(operands)
    for positions in operation.path:
        selected = tuple(values[index] for index in positions)
        modes = tuple(live_modes[index] for index in positions)
        remaining = tuple(
            value for index, value in enumerate(live_modes) if index not in positions
        )
        needed = set(operation.output).union(*(set(value) for value in remaining))
        output = tuple(
            mode
            for mode in dict.fromkeys(mode for value in modes for mode in value)
            if mode in needed
        )
        child = replace(operation, inputs=modes, output=output, path=())
        result = _ffi(*selected, operation=child)
        for index in sorted(positions, reverse=True):
            values.pop(index)
            live_modes.pop(index)
        values.append(result)
        live_modes.append(output)
    result = values[0]
    if live_modes[0] != operation.output:
        result = _ffi(
            result, operation=replace(operation, inputs=(live_modes[0],), path=())
        )
    return result


def _abstract(*operands: Any, operation: TensorOperation) -> Any:
    extents = dict(operation.dimensions)
    if len(operands) != len(operation.inputs):
        raise ValueError("Provider operand count differs from prepared recipe.")
    for operand, modes in zip(operands, operation.inputs, strict=True):
        if operand.shape != tuple(
            extents[mode] for mode in modes
        ) or operand.dtype != np.dtype(operation.dtype):
            raise TypeError("Provider operand shape/dtype differs from prepared recipe.")
    return ShapedArray(operation.output_shape, np.dtype(operation.dtype))


def _lower(
    ctx: mlir.LoweringRuleContext, *operands: Any, operation: TensorOperation
) -> Any:
    return mlir.lower_fun(_implementation, multiple_results=False)(
        ctx, *operands, operation=operation
    )


def _unsupported(
    ctx: mlir.LoweringRuleContext, *operands: Any, operation: TensorOperation
) -> Any:
    del ctx, operands
    raise ValueError(
        f"Explicit {operation.provider} contraction requires a CUDA device; no native fallback is allowed."
    )


def _jvp(
    primals: Sequence[Array],
    tangents: Sequence[Array | ad.Zero],
    *,
    operation: TensorOperation,
) -> tuple[Array, Array | ad.Zero]:
    output = _tensor.bind(*primals, operation=operation)
    active = tuple(
        (index, tangent)
        for index, tangent in enumerate(tangents)
        if not isinstance(tangent, ad.Zero)
    )
    if not active:
        return output, ad.Zero(jax.typeof(output))
    if operation.derivative_depth >= 2:
        raise ValueError(
            "Prepared CUDA tensor contractions admit derivatives through order two."
        )
    child = replace(operation, derivative_depth=operation.derivative_depth + 1)
    total = None
    for index, tangent in active:
        values = list(primals)
        values[index] = tangent
        term = _tensor.bind(*values, operation=child)
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("An active tensor tangent was lost.")
    return output, total


def _transpose(
    cotangent: Array | ad.Zero,
    *operands: Array | ad.UndefinedPrimal,
    operation: TensorOperation,
) -> tuple[Array | ad.Zero | None, ...]:
    undefined = tuple(
        index for index, value in enumerate(operands) if ad.is_undefined_primal(value)
    )
    if len(undefined) != 1:
        raise ValueError("Multilinear tensor transpose requires exactly one linear slot.")
    index = undefined[0]
    linear = operands[index]
    if not isinstance(linear, ad.UndefinedPrimal):
        raise RuntimeError("Tensor transpose lost its linear slot.")
    if isinstance(cotangent, ad.Zero):
        result = ad.Zero(linear.aval)
    else:
        child, missing = transpose_operation(operation, index)
        values = [cotangent]
        for position, value in enumerate(operands):
            if position != index:
                if isinstance(value, ad.UndefinedPrimal):
                    raise RuntimeError("Only one tensor input may be linear.")
                values.append(value)
        extents = dict(operation.dimensions)
        values.extend(
            jnp.ones((extents[mode],), dtype=operation.dtype) for mode in missing
        )
        result = _tensor.bind(*values, operation=child)
    return tuple(
        result if position == index else None for position in range(len(operands))
    )


def _batch(
    operands: Sequence[Array], axes: Sequence[int | None], *, operation: TensorOperation
) -> tuple[Array, int | None]:
    mapped = tuple(index for index, axis in enumerate(axes) if axis is not None)
    if not mapped:
        return _tensor.bind(*operands, operation=operation), None
    moved = tuple(
        value if axis is None else jnp.moveaxis(value, axis, 0)
        for value, axis in zip(operands, axes, strict=True)
    )
    count = moved[mapped[0]].shape[0]
    if count > operation.maximum_batch_size:
        raise MemoryError("Tensor batching exceeds maximum_batch_size.")
    batch_retained = sum(value.nbytes for value in operands)
    if (
        batch_retained
        + count * prod(operation.output_shape) * np.dtype(operation.dtype).itemsize
        + admit_operation(operation)
        > operation.maximum_derivative_bytes
    ):
        raise MemoryError("Tensor batching exceeds maximum_derivative_bytes.")

    def lane(values: tuple[Array, ...]) -> Array:
        current = list(moved)
        for index, value in zip(mapped, values, strict=True):
            current[index] = value
        return _tensor.bind(*current, operation=operation)

    return jax.lax.map(lane, tuple(moved[index] for index in mapped)), 0


_tensor.def_impl(_implementation)
_tensor.def_abstract_eval(_abstract)
mlir.register_lowering(_tensor, _lower, platform="cuda")
for _platform in ("cpu", "rocm", "tpu"):
    mlir.register_lowering(_tensor, _unsupported, platform=_platform)
ad.primitive_jvps[_tensor] = _jvp
ad.primitive_transposes[_tensor] = _transpose
batching.primitive_batchers[_tensor] = _batch


def execute_operation(operation: TensorOperation, operands: Sequence[Array], /) -> Array:
    return _tensor.bind(*operands, operation=operation)
