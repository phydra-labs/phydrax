#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array


def _leaf_static_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, (bool, int, float, str, type(None))):
        return a == b
    if isinstance(a, tuple):
        if len(a) != len(b):
            return False
        return all(_leaf_static_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_leaf_static_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_leaf_static_equal(a[k], b[k]) for k in a)
    return a is b


def pack_scan_modules(modules: Sequence[Any]) -> tuple[Any | None, Any | None, bool]:
    """Pack a sequence of Equinox modules into scan-friendly dynamic/static pytrees."""
    if len(modules) == 0:
        return None, None, False

    dynamic0, static0 = eqx.partition(modules[0], eqx.is_array)
    dynamic_treedef = jtu.tree_structure(dynamic0)
    static_treedef = jtu.tree_structure(static0)
    dynamic0_leaves = jtu.tree_leaves(dynamic0)
    static0_leaves = jtu.tree_leaves(static0)
    if len(dynamic0_leaves) == 0:
        return None, None, False

    dynamics = [dynamic0]
    for module in modules[1:]:
        dynamic_i, static_i = eqx.partition(module, eqx.is_array)
        if jtu.tree_structure(dynamic_i) != dynamic_treedef:
            return None, None, False
        if jtu.tree_structure(static_i) != static_treedef:
            return None, None, False

        dynamic_i_leaves = jtu.tree_leaves(dynamic_i)
        for x0, xi in zip(dynamic0_leaves, dynamic_i_leaves, strict=True):
            if x0.shape != xi.shape:
                return None, None, False
            if x0.dtype != xi.dtype:
                return None, None, False

        static_i_leaves = jtu.tree_leaves(static_i)
        for s0, si in zip(static0_leaves, static_i_leaves, strict=True):
            if not _leaf_static_equal(s0, si):
                return None, None, False

        dynamics.append(dynamic_i)

    stacked_dynamic = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *dynamics)
    return stacked_dynamic, static0, True


def stack_scan_dynamics(modules: Sequence[Any]) -> Any | None:
    """Stack dynamic leaves across compatible modules along a new scan axis."""
    if len(modules) == 0:
        return None
    dynamics = [eqx.partition(module, eqx.is_array)[0] for module in modules]
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *dynamics)


def scan_apply(
    dynamic: Any, static: Any, x: Array, fn: Callable[[Array, Any], Array]
) -> Array:
    """Apply a scan over packed module dynamics, reconstructing modules each step."""

    def _step(carry: Array, dynamic_layer: Any) -> tuple[Array, None]:
        layer = eqx.combine(dynamic_layer, static)
        out = fn(carry, layer)
        return out, None

    carry, _ = jax.lax.scan(_step, x, dynamic)
    return carry


def scan_apply_with_data(
    dynamic: Any,
    static: Any,
    carry: Any,
    data: Any,
    fn: Callable[[Any, Any, Any], Any],
) -> Any:
    """Apply a scan over packed module dynamics with aligned per-step scan data."""

    def _step(carry_i: Any, packed: tuple[Any, Any]) -> tuple[Any, None]:
        dynamic_layer, data_i = packed
        layer = eqx.combine(dynamic_layer, static)
        out = fn(carry_i, layer, data_i)
        return out, None

    carry_out, _ = jax.lax.scan(_step, carry, (dynamic, data))
    return carry_out
