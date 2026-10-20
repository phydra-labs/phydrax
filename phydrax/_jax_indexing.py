#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Platform-scoped JAX indexing correctness rules, without backend discovery.

JAX 0.12 narrows fill-gather indices before checking bounds. The same lowering
is used by scatter transposes. Register this correction explicitly for a
platform; importing this module neither imports JAX nor changes its callables.
The Metal wheel packages a private, byte-identical copy so plugin discovery
never needs to import the Phydrax core package.
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import Any, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from jax import Array


_registered_platforms: set[str] = set()
_registration_lock = Lock()
_installation_lock = Lock()
_accelerator_callback_installed = False


def _gather_fill_preserving_indices(
    operand: Array,
    indices: Array,
    *,
    dimension_numbers: Any,
    slice_sizes: Sequence[Any],
    unique_indices: bool,
    indices_are_sorted: bool,
    fill_value: Any,
    output_shape: Sequence[Any],
) -> Array:
    """Mask whole windows before any address-width conversion."""
    from jax._src.lax import lax, slicing

    # JAX 0.12's generic fill lowering narrows indices to a shape-selected
    # signed dtype before checking bounds. Small operands consequently admit
    # e.g. int64(2**32) as zero, including in scatter's reverse-mode gather.
    # Widen comparisons without changing signedness; the native gather owns
    # clamping and the eventual int32 address conversion.
    unsigned = np.issubdtype(indices.dtype, np.unsignedinteger)
    index_dtype = np.dtype(np.uint64 if unsigned else np.int64)
    wide_indices = lax.convert_element_type(indices, index_dtype)
    mapped_dims = np.asarray(dimension_numbers.start_index_map, dtype=np.int32)
    operand_dims = lax.convert_element_type(lax.shape_as_value(operand.shape), np.int64)
    window_dims = lax.convert_element_type(lax.shape_as_value(slice_sizes), np.int64)
    upper_bound = operand_dims[mapped_dims] - window_dims[mapped_dims]
    num_batch_dims = len(indices.shape) - 1
    batch_axes = tuple(range(num_batch_dims))
    upper = lax.expand_dims(
        lax.convert_element_type(upper_bound, index_dtype), batch_axes
    )
    nonempty_bound = lax.expand_dims(lax.ge(upper_bound, np.int64(0)), batch_axes)
    mask = lax.bitwise_and(
        nonempty_bound,
        lax.bitwise_and(
            lax.ge(wide_indices, index_dtype.type(0)),
            lax.le(wide_indices, upper),
        ),
    )
    mask = lax.reduce_and(mask, [num_batch_dims])
    gathered = slicing.gather(
        operand,
        indices,
        dimension_numbers,
        slice_sizes,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        mode=slicing.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    batch_dims_in_output = [
        axis
        for axis in range(len(output_shape))
        if axis not in dimension_numbers.offset_dims
    ]
    return lax.select(
        lax.broadcast_in_dim(
            mask,
            output_shape,
            batch_dims_in_output,
            out_sharding=gathered.aval.sharding,
        ),
        gathered,
        lax.full_like(gathered, fill_value=fill_value),
    )


def _gather_rule(ctx: Any, operand: Any, indices: Any, **params: Any) -> Sequence[Any]:
    from jax._src import dtypes
    from jax._src.lax import slicing
    from jax.interpreters import mlir

    (aval_out,) = ctx.avals_out
    if params["mode"] != slicing.GatherScatterMode.FILL_OR_DROP or dtypes.issubdtype(
        aval_out.dtype, dtypes.extended
    ):
        # Keep canonical clip/promise, empty-shape, opaque-dtype and dynamic
        # shape handling. The backend's dtype admission is unchanged.
        return slicing._gather_lower(ctx, operand, indices, **params)
    fill_params = dict(params)
    del fill_params["mode"]
    return mlir.lower_fun(_gather_fill_preserving_indices, multiple_results=False)(
        ctx, operand, indices, **fill_params, output_shape=aval_out.shape
    )


def register_gather_lowering(platform: str) -> None:
    """Idempotently register fill-gather correctness for one explicit platform.

    This is host-side lowering registration, not backend initialization. It is
    safe before platform selection, including when selection came from the
    environment. Other platforms and public JAX numerical callables are intact.
    """
    if not isinstance(platform, str):
        raise TypeError("platform must be a string")
    if not platform or platform != platform.strip():
        raise ValueError("platform must be a non-empty canonical platform name")
    with _registration_lock:
        if platform in _registered_platforms:
            return
        from jax import lax
        from jax.interpreters import mlir

        mlir.register_lowering(lax.gather_p, _gather_rule, platform=platform)
        _registered_platforms.add(platform)


def _register_discovered_oneapi(*, c_api: Any = None) -> None:
    """Install only after upstream discovery has made the platform known."""
    del c_api
    from jax._src import xla_bridge

    if xla_bridge.is_known_platform("oneapi"):
        register_gather_lowering("oneapi")


def install_accelerator_lowerings() -> None:
    """Install TPU rules and defer oneAPI rules until upstream plugin discovery.

    JAX 0.12 exposes platform lowerings publicly, but its discovery-completion
    hook is owned by ``xla_bridge``. The callback runs after provider registration
    and before client creation. Installing the hook does not discover plugins,
    load their libraries, select a platform, or initialize any backend.
    """
    global _accelerator_callback_installed
    register_gather_lowering("tpu")
    with _installation_lock:
        if _accelerator_callback_installed:
            return
        from jax._src import xla_bridge

        # Cover a provider registered before this call, including completed
        # discovery; the upstream hook covers discovery still to come.
        _register_discovered_oneapi()
        xla_bridge.register_plugin_callbacks(_register_discovered_oneapi)
        _accelerator_callback_installed = True
