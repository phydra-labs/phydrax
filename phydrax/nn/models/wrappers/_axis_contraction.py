#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule


class AxisGather(StrictModule):
    """Gather one factor axis onto one or more target axes before contraction."""

    source_axis: str
    target_axes: tuple[str, ...]
    indices: Array

    def __init__(
        self,
        source_axis: str,
        target_axes: Sequence[str],
        indices: ArrayLike,
        /,
    ):
        axes = tuple(str(axis) for axis in target_axes)
        if not axes:
            raise ValueError("AxisGather target_axes must be non-empty.")
        self.source_axis = str(source_axis)
        self.target_axes = axes
        self.indices = jnp.asarray(indices, dtype=jnp.int32)


class AxisFactor(StrictModule):
    """A latent factor tensor with named leading sample axes."""

    name: str
    tensor: Array
    axes: tuple[str, ...]
    gathers: tuple[AxisGather, ...]

    def __init__(
        self,
        name: str,
        tensor: ArrayLike,
        axes: Sequence[str],
        /,
        *,
        gathers: Sequence[AxisGather] = (),
    ):
        axes_ = tuple(str(axis) for axis in axes)
        if len(set(axes_)) != len(axes_):
            raise ValueError(f"AxisFactor {name!r} has duplicate axes {axes_!r}.")
        arr = jnp.asarray(tensor)
        if arr.ndim != len(axes_) + 2:
            raise ValueError(
                f"AxisFactor {name!r} tensor must have rank len(axes)+2; "
                f"got tensor shape {arr.shape} and axes {axes_!r}."
            )
        self.name = str(name)
        self.tensor = arr
        self.axes = axes_
        self.gathers = tuple(gathers)


class AxisProductTerm(StrictModule):
    """One product of factors, reduced over the latent axis."""

    factor_names: tuple[str, ...]
    coefficient: Array

    def __init__(
        self,
        factor_names: Sequence[str],
        /,
        *,
        coefficient: ArrayLike = 1.0,
    ):
        names = tuple(str(name) for name in factor_names)
        if not names:
            raise ValueError("AxisProductTerm requires at least one factor.")
        self.factor_names = names
        self.coefficient = jnp.asarray(coefficient)


class AxisContractionPlan(StrictModule):
    """A sum of latent product terms over named axes."""

    terms: tuple[AxisProductTerm, ...]
    output_axes: tuple[str, ...] | None

    def __init__(
        self,
        terms: Sequence[AxisProductTerm],
        /,
        *,
        output_axes: Sequence[str] | None = None,
    ):
        terms_ = tuple(terms)
        if not terms_:
            raise ValueError("AxisContractionPlan requires at least one term.")
        axes = None if output_axes is None else tuple(str(axis) for axis in output_axes)
        if axes is not None and len(set(axes)) != len(axes):
            raise ValueError(f"AxisContractionPlan has duplicate output axes {axes!r}.")
        self.terms = terms_
        self.output_axes = axes


class AxisContractionResult(StrictModule):
    """The array result of an axis-aware contraction and its named axes."""

    data: Array
    axes: tuple[str, ...]

    def __init__(self, data: ArrayLike, axes: Sequence[str], /):
        axes_ = tuple(str(axis) for axis in axes)
        arr = jnp.asarray(data)
        if arr.ndim < len(axes_):
            raise ValueError(
                "AxisContractionResult data rank is smaller than the named-axis rank."
            )
        self.data = arr
        self.axes = axes_


def _merge_axes(axis_groups: Sequence[Sequence[str]], /) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in axis_groups:
        for axis in group:
            if axis in seen:
                continue
            seen.add(axis)
            out.append(axis)
    return tuple(out)


def _apply_gathers(factor: AxisFactor, /) -> AxisFactor:
    data = factor.tensor
    axes = factor.axes
    for gather in factor.gathers:
        if gather.source_axis not in axes:
            raise ValueError(
                f"Factor {factor.name!r} cannot gather missing axis "
                f"{gather.source_axis!r}."
            )
        for axis in gather.target_axes:
            if axis in axes and axis != gather.source_axis:
                raise ValueError(
                    f"Gather target axis {axis!r} already exists in factor "
                    f"{factor.name!r} axes {axes!r}."
                )
        if gather.indices.ndim != len(gather.target_axes):
            raise ValueError(
                f"Gather indices for factor {factor.name!r} must have rank "
                f"{len(gather.target_axes)}, got {gather.indices.ndim}."
            )
        pos = axes.index(gather.source_axis)
        data = jnp.take(data, gather.indices, axis=pos)
        axes = axes[:pos] + gather.target_axes + axes[pos + 1 :]
    return AxisFactor(factor.name, data, axes)


def _axis_sizes(factors: Sequence[AxisFactor], /) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for factor in factors:
        leading_shape = factor.tensor.shape[: len(factor.axes)]
        for axis, size in zip(factor.axes, leading_shape, strict=True):
            n = int(size)
            if axis in sizes and sizes[axis] != n:
                raise ValueError(
                    f"Axis {axis!r} has inconsistent sizes {sizes[axis]} and {n}."
                )
            sizes[axis] = n
    return sizes


def _broadcast_factor(factor: AxisFactor, output_axes: tuple[str, ...], /) -> Array:
    leading_shape = factor.tensor.shape[: len(factor.axes)]
    axis_sizes = dict(zip(factor.axes, leading_shape, strict=True))
    shape: list[int] = []
    for axis in output_axes:
        shape.append(int(axis_sizes[axis]) if axis in axis_sizes else 1)
    shape.extend([int(factor.tensor.shape[-2]), int(factor.tensor.shape[-1])])
    return factor.tensor.reshape(tuple(shape))


def contract_axis_factors(
    factors: Mapping[str, AxisFactor],
    plan: AxisContractionPlan,
    /,
) -> AxisContractionResult:
    """Evaluate a sum of named-axis latent product contractions."""
    prepared_by_name: dict[str, AxisFactor] = {}
    for name, factor in factors.items():
        prepared_by_name[str(name)] = _apply_gathers(factor)

    all_term_axes: list[tuple[str, ...]] = []
    for term in plan.terms:
        term_factors = tuple(prepared_by_name[name] for name in term.factor_names)
        all_term_axes.append(_merge_axes(tuple(f.axes for f in term_factors)))
    output_axes = plan.output_axes or _merge_axes(all_term_axes)

    terms_out: list[Array] = []
    for term in plan.terms:
        term_factors = tuple(prepared_by_name[name] for name in term.factor_names)
        _axis_sizes(term_factors)
        acc = jnp.asarray(term.coefficient)
        for factor in term_factors:
            acc = acc * _broadcast_factor(factor, output_axes)
        terms_out.append(jnp.sum(acc, axis=-2))

    total = terms_out[0]
    for term_out in terms_out[1:]:
        total = total + term_out
    return AxisContractionResult(total, output_axes)


__all__ = [
    "AxisContractionPlan",
    "AxisContractionResult",
    "AxisFactor",
    "AxisGather",
    "AxisProductTerm",
    "contract_axis_factors",
]
