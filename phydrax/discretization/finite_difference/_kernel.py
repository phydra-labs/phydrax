#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


PatchExecutionKind: TypeAlias = Literal["vmap", "lax_map"]
SweepDirection: TypeAlias = Literal["forward", "backward"]


class PatchKernelPlan(StrictModule, NonTrainableState):
    """Prepared parallel local patch kernel without materialized index matrices."""

    kernel_shape: tuple[int, ...] = eqx.field(static=True)
    kernel_functions: tuple[Callable[[Array, Any], ArrayLike], ...] = eqx.field(
        static=True
    )
    execution: PatchExecutionKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel_shape: Sequence[int],
        kernel: Callable[[Array, Any], ArrayLike]
        | Sequence[Callable[[Array, Any], ArrayLike]],
        /,
        *,
        execution: PatchExecutionKind = "vmap",
        plan_id: str | None = None,
    ):
        shape = tuple(int(size) for size in kernel_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("kernel_shape dimensions must be positive.")
        functions = cast(
            tuple[Callable[[Array, Any], ArrayLike], ...],
            (kernel,) if callable(kernel) else tuple(kernel),
        )
        if not functions or not all(callable(function) for function in functions):
            raise TypeError("kernel must contain one or more callables.")
        if execution not in ("vmap", "lax_map"):
            raise ValueError("Unknown patch execution kind.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "patch-kernel-plan",
                    "kernel_shape": list(shape),
                    "functions": [repr(function) for function in functions],
                    "execution": execution,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.kernel_shape = shape
        self.kernel_functions = functions
        self.execution = execution
        self.plan_id = identifier

    def prepare(self, input_shape: Sequence[int], /) -> "PreparedPatchKernel":
        return PreparedPatchKernel(self, input_shape)


class PreparedPatchKernel(StrictModule, NonTrainableState):
    """Parallel patch execution with optional per-output kernel dispatch."""

    plan: PatchKernelPlan
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PatchKernelPlan, input_shape: Sequence[int], /):
        if not isinstance(plan, PatchKernelPlan):
            raise TypeError("plan must be a PatchKernelPlan.")
        shape = tuple(int(size) for size in input_shape)
        if len(shape) != len(plan.kernel_shape) or any(
            kernel > size
            for kernel, size in zip(
                plan.kernel_shape,
                shape,
                strict=True,
            )
        ):
            raise ValueError("Kernel dimensions must fit aligned input dimensions.")
        output_shape = tuple(
            size - kernel + 1
            for size, kernel in zip(shape, plan.kernel_shape, strict=True)
        )
        offsets = tuple(product(*(range(size) for size in plan.kernel_shape)))
        self.plan = plan
        self.input_shape = shape
        self.output_shape = output_shape
        self.offsets = offsets
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-patch-kernel",
                "plan": plan.plan_id,
                "input_shape": list(shape),
                "output_shape": list(output_shape),
            }
        )

    def patches(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape != self.input_shape:
            raise ValueError(f"Patch input must have shape {self.input_shape}.")
        windows = []
        for offset in self.offsets:
            slices = tuple(
                slice(start, start + output)
                for start, output in zip(offset, self.output_shape, strict=True)
            )
            windows.append(array[slices])
        stacked = jnp.stack(windows, axis=-1)
        return stacked.reshape(self.output_shape + self.plan.kernel_shape)

    def __call__(
        self,
        values: ArrayLike,
        args: Any = None,
        /,
        *,
        kernel_indices: ArrayLike | None = None,
    ) -> Array:
        patches = self.patches(values)
        flattened = patches.reshape((-1,) + self.plan.kernel_shape)
        if kernel_indices is None:
            if len(self.plan.kernel_functions) != 1:
                raise ValueError("Multiple patch kernels require kernel_indices.")
            function = self.plan.kernel_functions[0]
            mapped = (
                jax.vmap(lambda patch: function(patch, args))
                if self.plan.execution == "vmap"
                else lambda collection: jax.lax.map(
                    lambda patch: function(patch, args), collection
                )
            )
            result = mapped(flattened)
        else:
            indices = jnp.asarray(kernel_indices, dtype=jnp.int32)
            if indices.shape != self.output_shape:
                raise ValueError("kernel_indices must match patch output shape.")
            flat_indices = indices.reshape((-1,))
            functions = tuple(
                lambda patch, function=function: function(patch, args)
                for function in self.plan.kernel_functions
            )
            mapper = lambda data: jax.lax.switch(data[0], functions, data[1])
            result = jax.vmap(mapper)((flat_indices, flattened))
        result_array = jnp.asarray(result)
        return result_array.reshape(self.output_shape + result_array.shape[1:])


class OrderedPatchKernelPlan(StrictModule, NonTrainableState):
    """One-dimensional causal patch sweep lowered through ``lax.scan``."""

    kernel_size: int = eqx.field(static=True)
    kernel: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    direction: SweepDirection = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel_size: int,
        kernel: Callable[[Array, Any], ArrayLike],
        /,
        *,
        direction: SweepDirection = "forward",
        plan_id: str | None = None,
    ):
        size = int(kernel_size)
        if size <= 0 or size % 2 == 0:
            raise ValueError("Ordered patch kernels require a positive odd kernel size.")
        if not callable(kernel):
            raise TypeError("kernel must be callable.")
        if direction not in ("forward", "backward"):
            raise ValueError("Unknown sweep direction.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "ordered-patch-kernel",
                    "kernel_size": size,
                    "kernel": repr(kernel),
                    "direction": direction,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.kernel_size = size
        self.kernel = kernel
        self.direction = direction
        self.plan_id = identifier

    def __call__(self, values: ArrayLike, args: Any = None, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim != 1 or array.size < self.kernel_size:
            raise ValueError("Ordered patch input must be rank-1 and fit the kernel.")
        radius = self.kernel_size // 2
        padded = jnp.pad(array, (radius, radius))
        indices = jnp.arange(array.size)
        reverse = self.direction == "backward"

        def step(state: Array, index: Array) -> tuple[Array, Array]:
            patch = jax.lax.dynamic_slice(state, (index,), (self.kernel_size,))
            updated = jnp.asarray(self.kernel(patch, args))
            if updated.shape != ():
                raise ValueError("Ordered patch kernels must return a scalar.")
            state = state.at[index + radius].set(updated)
            return state, updated

        final, _ = jax.lax.scan(step, padded, indices, reverse=reverse)
        return final[radius:-radius]


__all__ = [
    "OrderedPatchKernelPlan",
    "PatchExecutionKind",
    "PatchKernelPlan",
    "PreparedPatchKernel",
    "SweepDirection",
]
