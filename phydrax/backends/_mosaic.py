# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compile-target admission for internal native Mosaic GPU lowerings."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jax import Array
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.sharding_impls import ShardingContext, SPMDAxisContext
from jax.interpreters import mlir

from ..execution import default_device


def lower_mosaic_or_reference(
    native: Callable[..., Sequence[Array]],
    reference: Callable[..., Sequence[Array]],
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    **parameters: object,
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
    if admitted and devices:
        # Mosaic infers its architecture while constructing the kernel module;
        # the caller's default device need not be the compilation target.
        with default_device(devices[0]):
            return mlir.lower_fun(native, multiple_results=True)(
                ctx, *operands, **parameters
            )
    return mlir.lower_fun(reference, multiple_results=True)(ctx, *operands, **parameters)
