#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from phydrax._callable import _ensure_special_kwonly_args
from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.representations import (
    FiniteOrthogonalGroup,
    TensorFieldLayout,
)


class GroupAveragedOperator(StrictModule):
    """Reference equivariantization by averaging a lattice model over a finite group."""

    model: Callable
    group: FiniteOrthogonalGroup
    input_layout: TensorFieldLayout
    output_layout: TensorFieldLayout
    spatial_axes: tuple[int, ...] | None

    def __init__(
        self,
        model: Callable,
        group: FiniteOrthogonalGroup,
        input_layout: TensorFieldLayout,
        output_layout: TensorFieldLayout,
        /,
        *,
        spatial_axes: Sequence[int] | None = None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(group, FiniteOrthogonalGroup):
            raise TypeError("group must be a FiniteOrthogonalGroup.")
        if not isinstance(input_layout, TensorFieldLayout) or not isinstance(
            output_layout, TensorFieldLayout
        ):
            raise TypeError("input_layout and output_layout must be tensor layouts.")
        if (
            input_layout.dimension != group.dimension
            or output_layout.dimension != group.dimension
        ):
            raise ValueError("Group and tensor layout dimensions must agree.")
        if spatial_axes is None:
            resolved_axes = None
        else:
            resolved_axes = tuple(int(axis) for axis in spatial_axes)
            if len(resolved_axes) != group.dimension or len(set(resolved_axes)) != len(
                resolved_axes
            ):
                raise ValueError(
                    "spatial_axes must uniquely identify every group dimension."
                )
        self.model = _ensure_special_kwonly_args(model)
        self.group = group
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.spatial_axes = resolved_axes

    def __call__(
        self,
        x: Array | tuple[Any, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if isinstance(x, tuple):
            if not x:
                raise ValueError(
                    "Tuple model inputs must begin with tensor field values."
                )
            values = jnp.asarray(x[0])
            context = x[1:]
        else:
            values = jnp.asarray(x)
            context = None
        if values.ndim < self.group.dimension + 1:
            raise ValueError("Input values do not contain spatial and channel axes.")
        axes = (
            tuple(range(values.ndim - self.group.dimension - 1, values.ndim - 1))
            if self.spatial_axes is None
            else self.spatial_axes
        )
        averaged = None
        for element in range(self.group.order):
            transformed = self.group.field_action(
                values,
                self.input_layout,
                element,
                spatial_axes=axes,
            )
            model_input = transformed if context is None else (transformed,) + context
            prediction = jnp.asarray(self.model(model_input, key=key))
            restored = self.group.field_action(
                prediction,
                self.output_layout,
                self.group.inverse(element),
                spatial_axes=axes,
            )
            averaged = restored if averaged is None else averaged + restored
        if averaged is None:
            raise RuntimeError("Finite groups must contain at least one element.")
        return averaged / float(self.group.order)


__all__ = ["GroupAveragedOperator"]
