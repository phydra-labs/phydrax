#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Literal

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike

from ._syntax import (
    _Axis,
    _axis_tokens,
    _Ellipsis,
    _ellipsis_tokens,
    _parse_pattern,
    _Pattern,
    _pattern_error,
    _PhysicalAxis,
    _Singleton,
)


_Operation = Literal["rearrange", "reduce", "repeat"]


@dataclass(frozen=True, slots=True)
class _EllipsisAxis:
    offset: int


_AxisKey = str | _EllipsisAxis


@dataclass(frozen=True, slots=True)
class _TransformPlan:
    elementary_input_shape: tuple[int, ...]
    reduction_axes: tuple[int, ...]
    permutation: tuple[int, ...]
    broadcast_shape: tuple[int, ...]
    broadcast_dimensions: tuple[int, ...]
    output_shape: tuple[int, ...]
    reduction: str | None


_Reduction = Callable[..., Array]
_REDUCTIONS: Mapping[str, _Reduction] = {
    "sum": jnp.sum,
    "mean": jnp.mean,
    "prod": jnp.prod,
    "min": jnp.min,
    "max": jnp.max,
    "all": jnp.all,
    "any": jnp.any,
}


def _axis_map(pattern: _Pattern, *, right: bool) -> dict[str, _Axis]:
    expression = pattern.right if right else pattern.left
    return {axis.name: axis for axis in _axis_tokens(expression)}


def _normalize_sizes(
    pattern: _Pattern, sizes: Mapping[str, object]
) -> tuple[tuple[str, int], ...]:
    axes = _axis_map(pattern, right=False) | _axis_map(pattern, right=True)
    normalized: list[tuple[str, int]] = []
    for name, value in sizes.items():
        if name not in axes:
            raise _pattern_error(
                pattern.source,
                0,
                f"size was provided for unused axis {name!r}",
            )
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise _pattern_error(
                pattern.source,
                axes[name].start,
                f"axis size for {name!r} must be a static integer, got {type(value).__name__}",
            )
        integer = int(value)
        if integer <= 0:
            raise _pattern_error(
                pattern.source,
                axes[name].start,
                f"axis size for {name!r} must be positive, got {integer}",
            )
        normalized.append((name, integer))
    return tuple(sorted(normalized))


def _concrete_shape(pattern: _Pattern, shape: tuple[object, ...]) -> tuple[int, ...]:
    concrete: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise _pattern_error(
                pattern.source,
                0,
                "ein transforms require concrete static integer dimensions",
            )
        concrete.append(int(dimension))
    return tuple(concrete)


def _validate_operation(
    operation: _Operation,
    pattern: _Pattern,
    sizes: Mapping[str, int],
) -> None:
    left_axes = _axis_map(pattern, right=False)
    right_axes = _axis_map(pattern, right=True)
    left_names = set(left_axes)
    right_names = set(right_axes)

    if operation == "rearrange":
        removed = left_names - right_names
        added = right_names - left_names
        if removed:
            name = min(removed)
            raise _pattern_error(
                pattern.source,
                left_axes[name].start,
                f"rearrange cannot remove axis {name!r}",
            )
        if added:
            name = min(added)
            raise _pattern_error(
                pattern.source,
                right_axes[name].start,
                f"rearrange cannot add axis {name!r}",
            )
        return

    if operation == "reduce":
        added = right_names - left_names
        if added:
            name = min(added)
            raise _pattern_error(
                pattern.source,
                right_axes[name].start,
                f"reduce cannot add axis {name!r}",
            )
        return

    removed = left_names - right_names
    if removed:
        name = min(removed)
        raise _pattern_error(
            pattern.source,
            left_axes[name].start,
            f"repeat cannot remove axis {name!r}",
        )
    for name in sorted(right_names - left_names):
        if name not in sizes:
            raise _pattern_error(
                pattern.source,
                right_axes[name].start,
                f"repeat requires a size for new axis {name!r}",
            )


def _bind_input_physical_axis(
    pattern: _Pattern,
    physical_axis: _PhysicalAxis,
    physical_size: int,
    explicit_sizes: Mapping[str, int],
    bindings: dict[str, int],
    elementary_keys: list[_AxisKey],
    elementary_sizes: list[int],
) -> None:
    known_product = 1
    unknown: list[_Axis] = []

    for factor in physical_axis.factors:
        if isinstance(factor, _Singleton):
            continue
        if isinstance(factor, _Ellipsis):
            raise _pattern_error(
                pattern.source,
                factor.start,
                "input ellipsis must be a standalone physical axis",
            )
        if factor.name in explicit_sizes:
            known_product *= explicit_sizes[factor.name]
        else:
            unknown.append(factor)

    if len(unknown) > 1:
        names = ", ".join(repr(axis.name) for axis in unknown)
        raise _pattern_error(
            pattern.source,
            physical_axis.start,
            f"input group has multiple unresolved factors: {names}",
        )

    if unknown:
        if physical_size % known_product != 0:
            raise _pattern_error(
                pattern.source,
                physical_axis.start,
                f"physical size {physical_size} is not divisible by known factor product {known_product}",
            )
        bindings[unknown[0].name] = physical_size // known_product
    elif physical_size != known_product:
        raise _pattern_error(
            pattern.source,
            physical_axis.start,
            f"physical size {physical_size} does not match factor product {known_product}",
        )

    for factor in physical_axis.factors:
        if not isinstance(factor, _Axis):
            continue
        if factor.name in explicit_sizes:
            bindings[factor.name] = explicit_sizes[factor.name]
        elementary_keys.append(factor.name)
        elementary_sizes.append(bindings[factor.name])


def _expanded_input(
    pattern: _Pattern,
    shape: tuple[int, ...],
    explicit_sizes: Mapping[str, int],
) -> tuple[
    tuple[_AxisKey, ...],
    tuple[int, ...],
    dict[str, int],
    tuple[_EllipsisAxis, ...],
    tuple[int, ...],
]:
    ellipses = _ellipsis_tokens(pattern.left)
    explicit_physical_axes = len(pattern.left.axes) - (1 if ellipses else 0)
    if ellipses:
        ellipsis_width = len(shape) - explicit_physical_axes
        if ellipsis_width < 0:
            raise _pattern_error(
                pattern.source,
                ellipses[0].start,
                f"input rank {len(shape)} is too small for {explicit_physical_axes} explicit physical axes",
            )
    else:
        ellipsis_width = 0
        if len(shape) != explicit_physical_axes:
            raise _pattern_error(
                pattern.source,
                0,
                f"input rank {len(shape)} does not match {explicit_physical_axes} pattern axes",
            )

    bindings: dict[str, int] = {}
    elementary_keys: list[_AxisKey] = []
    elementary_sizes: list[int] = []
    ellipsis_keys = tuple(_EllipsisAxis(offset) for offset in range(ellipsis_width))
    ellipsis_sizes: tuple[int, ...] = ()
    shape_position = 0

    for physical_axis in pattern.left.axes:
        factor = physical_axis.factors[0]
        if isinstance(factor, _Ellipsis):
            ellipsis_sizes = shape[shape_position : shape_position + ellipsis_width]
            elementary_keys.extend(ellipsis_keys)
            elementary_sizes.extend(ellipsis_sizes)
            shape_position += ellipsis_width
            continue

        _bind_input_physical_axis(
            pattern,
            physical_axis,
            shape[shape_position],
            explicit_sizes,
            bindings,
            elementary_keys,
            elementary_sizes,
        )
        shape_position += 1

    return (
        tuple(elementary_keys),
        tuple(elementary_sizes),
        bindings,
        ellipsis_keys,
        ellipsis_sizes,
    )


def _expanded_output(
    pattern: _Pattern,
    bindings: Mapping[str, int],
    ellipsis_keys: tuple[_EllipsisAxis, ...],
    ellipsis_sizes: tuple[int, ...],
) -> tuple[tuple[_AxisKey, ...], tuple[int, ...], tuple[int, ...]]:
    output_keys: list[_AxisKey] = []
    elementary_sizes: list[int] = []
    physical_shape: list[int] = []

    for physical_axis in pattern.right.axes:
        factor = physical_axis.factors[0]
        if (
            not physical_axis.grouped
            and len(physical_axis.factors) == 1
            and isinstance(factor, _Ellipsis)
        ):
            output_keys.extend(ellipsis_keys)
            elementary_sizes.extend(ellipsis_sizes)
            physical_shape.extend(ellipsis_sizes)
            continue

        physical_size = 1
        for factor in physical_axis.factors:
            if isinstance(factor, _Singleton):
                continue
            if isinstance(factor, _Ellipsis):
                output_keys.extend(ellipsis_keys)
                elementary_sizes.extend(ellipsis_sizes)
                for size in ellipsis_sizes:
                    physical_size *= size
                continue
            size = bindings[factor.name]
            output_keys.append(factor.name)
            elementary_sizes.append(size)
            physical_size *= size
        physical_shape.append(physical_size)

    return tuple(output_keys), tuple(elementary_sizes), tuple(physical_shape)


def _key_position(pattern: _Pattern, key: _AxisKey, *, right: bool) -> int:
    if isinstance(key, str):
        return _axis_map(pattern, right=right)[key].start
    ellipses = _ellipsis_tokens(pattern.right if right else pattern.left)
    return ellipses[0].start if ellipses else 0


@lru_cache(maxsize=256)
def _specialize_transform(
    operation: _Operation,
    source: str,
    shape: tuple[int, ...],
    size_items: tuple[tuple[str, int], ...],
    reduction: str | None,
) -> _TransformPlan:
    pattern = _parse_pattern(source)
    explicit_sizes = dict(size_items)
    _validate_operation(operation, pattern, explicit_sizes)

    (
        input_keys,
        input_elementary_shape,
        bindings,
        ellipsis_keys,
        ellipsis_sizes,
    ) = _expanded_input(pattern, shape, explicit_sizes)

    if operation == "repeat":
        for name, axis in _axis_map(pattern, right=True).items():
            if name not in bindings:
                if name not in explicit_sizes:
                    raise _pattern_error(
                        pattern.source,
                        axis.start,
                        f"repeat requires a size for new axis {name!r}",
                    )
                bindings[name] = explicit_sizes[name]

    output_keys, output_elementary_shape, output_shape = _expanded_output(
        pattern,
        bindings,
        ellipsis_keys,
        ellipsis_sizes,
    )

    input_key_set = set(input_keys)
    output_key_set = set(output_keys)
    removed = tuple(key for key in input_keys if key not in output_key_set)
    added = tuple(key for key in output_keys if key not in input_key_set)

    if operation == "rearrange":
        if removed:
            key = removed[0]
            raise _pattern_error(
                pattern.source,
                _key_position(pattern, key, right=False),
                "rearrange cannot remove expanded ellipsis axes",
            )
        if added:
            key = added[0]
            raise _pattern_error(
                pattern.source,
                _key_position(pattern, key, right=True),
                "rearrange cannot add expanded axes",
            )
    elif operation == "reduce":
        if added:
            key = added[0]
            raise _pattern_error(
                pattern.source,
                _key_position(pattern, key, right=True),
                "reduce cannot add axes",
            )
        if not removed:
            raise _pattern_error(
                pattern.source,
                0,
                "reduce must remove at least one axis after ellipsis expansion",
            )
    else:
        if removed:
            key = removed[0]
            raise _pattern_error(
                pattern.source,
                _key_position(pattern, key, right=False),
                "repeat cannot remove axes",
            )
        if not added:
            raise _pattern_error(
                pattern.source,
                0,
                "repeat must add at least one named axis after ellipsis expansion",
            )

    reduction_axes = tuple(
        position for position, key in enumerate(input_keys) if key not in output_key_set
    )
    surviving_keys = tuple(key for key in input_keys if key in output_key_set)
    desired_existing_keys = tuple(key for key in output_keys if key in input_key_set)
    surviving_positions = {key: position for position, key in enumerate(surviving_keys)}
    permutation = tuple(surviving_positions[key] for key in desired_existing_keys)
    output_positions = {key: position for position, key in enumerate(output_keys)}
    broadcast_dimensions = tuple(output_positions[key] for key in desired_existing_keys)

    return _TransformPlan(
        elementary_input_shape=input_elementary_shape,
        reduction_axes=reduction_axes,
        permutation=permutation,
        broadcast_shape=output_elementary_shape,
        broadcast_dimensions=broadcast_dimensions,
        output_shape=output_shape,
        reduction=reduction,
    )


def _execute(array: Array, plan: _TransformPlan) -> Array:
    result = array
    if tuple(result.shape) != plan.elementary_input_shape:
        result = jnp.reshape(result, plan.elementary_input_shape)

    if plan.reduction is not None:
        result = _REDUCTIONS[plan.reduction](result, axis=plan.reduction_axes)

    identity_permutation = tuple(range(result.ndim))
    if plan.permutation != identity_permutation:
        result = jnp.transpose(result, plan.permutation)

    if len(plan.broadcast_dimensions) != len(plan.broadcast_shape):
        result = lax.broadcast_in_dim(
            result,
            shape=plan.broadcast_shape,
            broadcast_dimensions=plan.broadcast_dimensions,
        )

    if tuple(result.shape) != plan.output_shape:
        result = jnp.reshape(result, plan.output_shape)
    return result


def _prepare(
    operation: _Operation,
    x: ArrayLike,
    pattern_source: str,
    sizes: Mapping[str, object],
    reduction: str | None,
) -> tuple[Array, _TransformPlan]:
    if not isinstance(pattern_source, str):
        raise TypeError(
            f"ein pattern must be a string, got {type(pattern_source).__name__}"
        )
    pattern = _parse_pattern(pattern_source)
    normalized_sizes = _normalize_sizes(pattern, sizes)
    explicit_sizes = dict(normalized_sizes)
    _validate_operation(operation, pattern, explicit_sizes)

    if reduction is not None:
        if not isinstance(reduction, str):
            raise TypeError(
                f"ein reduction must be a string, got {type(reduction).__name__}"
            )
        if reduction not in _REDUCTIONS:
            raise _pattern_error(
                pattern.source,
                0,
                f"unsupported reduction {reduction!r}; expected one of {tuple(_REDUCTIONS)}",
            )

    array = jnp.asarray(x)
    shape = _concrete_shape(pattern, tuple(array.shape))
    plan = _specialize_transform(
        operation,
        pattern.source,
        shape,
        normalized_sizes,
        reduction,
    )
    return array, plan


def rearrange(x: ArrayLike, pattern: str, /, **sizes: int) -> Array:
    """Reorder and regroup logical axes without adding or removing data axes."""
    array, plan = _prepare("rearrange", x, pattern, sizes, None)
    return _execute(array, plan)


def reduce(
    x: ArrayLike,
    pattern: str,
    reduction: str,
    /,
    **sizes: int,
) -> Array:
    """Reduce the logical axes omitted from the output pattern."""
    array, plan = _prepare("reduce", x, pattern, sizes, reduction)
    return _execute(array, plan)


def repeat(x: ArrayLike, pattern: str, /, **sizes: int) -> Array:
    """Broadcast new, explicitly sized logical axes into an array."""
    array, plan = _prepare("repeat", x, pattern, sizes, None)
    return _execute(array, plan)
