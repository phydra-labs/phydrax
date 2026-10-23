# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compile-target admission for internal native Mosaic GPU lowerings."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jax import Array
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.sharding_impls import ShardingContext, SPMDAxisContext
from jax.interpreters import mlir


def lower_mosaic_or_reference(
    native: Callable[..., tuple[Array, ...]],
    reference: Callable[..., tuple[Array, ...]],
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    **parameters: int | tuple[int, ...],
) -> Sequence[ir.Value]:
    """Use Mosaic only for explicitly resolved Ampere-or-newer CUDA targets."""
    axis_context = ctx.module_context.axis_context
    if isinstance(axis_context, ShardingContext):
        devices = axis_context.device_assignment
    elif isinstance(axis_context, SPMDAxisContext):
        devices = tuple(axis_context.mesh.devices.flat)
    else:
        devices = None
    if devices is None:
        backend = ctx.module_context.backend
        if isinstance(backend, xla_client.Client) and backend.platform in ("cuda", "gpu"):
            # Ordinary jit can omit an assignment while supplying its selected
            # compilation client. Every possible device must support the kernel.
            devices = tuple(backend.devices())
    admitted = bool(devices)
    for device in devices or ():
        # Compute capability is an optional provider attribute, absent on CPU.
        capability = getattr(device, "compute_capability", None)
        parts = capability.split(".") if isinstance(capability, str) else ()
        if (
            len(parts) != 2
            or not all(part.isdecimal() for part in parts)
            or int(parts[0]) < 8
        ):
            admitted = False
            break
    implementation = native if admitted else reference
    return mlir.lower_fun(implementation, multiple_results=True)(
        ctx, *operands, **parameters
    )
