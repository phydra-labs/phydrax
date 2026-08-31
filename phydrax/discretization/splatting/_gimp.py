#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._assignment import (
    _basis_and_derivative,
    _tensor_product_state,
    _uniform_spacing,
    AbstractStructuredSplatAssignment,
    SplatAssignmentCapabilities,
    SplatAssignmentState,
)


class GIMPAssignmentInput(StrictModule):
    half_widths: Array

    def __init__(self, half_widths: ArrayLike, /):
        value = jnp.asarray(half_widths)
        if value.ndim != 2:
            raise ValueError("GIMP half widths must have shape (particles, dimension).")
        self.half_widths = value


def _hat_antiderivative(value: Array, /) -> Array:
    lower = 0.5 * jnp.maximum(value + 1.0, 0.0) ** 2
    upper = 1.0 - 0.5 * jnp.maximum(1.0 - value, 0.0) ** 2
    return jnp.where(
        value <= -1.0,
        0.0,
        jnp.where(value < 0.0, lower, jnp.where(value < 1.0, upper, 1.0)),
    )


def _gimp_axis_stencil(
    coordinates,
    bounds,
    periodic,
    position,
    half_width,
    active,
    route_axis_width,
    maximum_half_width_cells,
):
    count = int(coordinates.size)
    spacing_value = _uniform_spacing(coordinates, bounds, periodic)
    spacing = jnp.asarray(spacing_value, dtype=position.dtype)
    lower, upper = bounds
    evaluated = jnp.mod(position - lower, upper - lower) + lower if periodic else position
    center = jnp.floor((evaluated - coordinates[0]) / spacing).astype(jnp.int32)
    slots = jnp.arange(route_axis_width, dtype=jnp.int32) - route_axis_width // 2
    raw = center[:, None] + slots[None, :]
    target = coordinates[0] + raw.astype(position.dtype) * spacing
    safe_half = jnp.maximum(half_width, jnp.finfo(position.dtype).eps * spacing)
    plus = (evaluated[:, None] + safe_half[:, None] - target) / spacing
    minus = (evaluated[:, None] - safe_half[:, None] - target) / spacing
    integrated = spacing * (_hat_antiderivative(plus) - _hat_antiderivative(minus))
    weights = integrated / (2.0 * safe_half[:, None])
    plus_value, _ = _basis_and_derivative(1, plus)
    minus_value, _ = _basis_and_derivative(1, minus)
    derivatives = (plus_value - minus_value) / (2.0 * safe_half[:, None])
    point_coordinate = (evaluated[:, None] - target) / spacing
    point_weight, point_derivative = _basis_and_derivative(1, point_coordinate)
    point = half_width <= jnp.finfo(position.dtype).eps * spacing
    weights = jnp.where(point[:, None], point_weight, weights)
    derivatives = jnp.where(point[:, None], point_derivative / spacing, derivatives)
    width_valid = (
        jnp.isfinite(half_width)
        & (half_width >= 0.0)
        & (half_width <= maximum_half_width_cells * spacing)
    )
    source_in_domain = (
        active
        & width_valid
        & (
            jnp.ones_like(active)
            if periodic
            else (position >= lower) & (position <= upper)
        )
    )
    if periodic:
        indices = jnp.mod(raw, count)
        route_valid = jnp.broadcast_to(source_in_domain[:, None], raw.shape)
    else:
        route_valid = source_in_domain[:, None] & (raw >= 0) & (raw < count)
        indices = jnp.clip(raw, 0, count - 1)
    offsets = target - evaluated[:, None]
    return indices, weights, derivatives, offsets, route_valid, source_in_domain


class UniformGIMPSplatAssignment(AbstractStructuredSplatAssignment):
    """Tensor uGIMP or cpGIMP/AABB convolution over a linear nodal basis."""

    reference_half_widths: Array
    maximum_half_width_cells: float = eqx.field(static=True)
    evolving: bool = eqx.field(static=True)
    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_half_widths: ArrayLike,
        /,
        *,
        maximum_half_width_cells: float = 1.0,
        evolving: bool = False,
    ):
        widths = np.asarray(reference_half_widths, dtype=float)
        maximum = float(maximum_half_width_cells)
        if (
            widths.ndim != 2
            or np.any(~np.isfinite(widths))
            or np.any(widths < 0.0)
            or not np.isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError("GIMP reference widths or support envelope are invalid.")
        self.reference_half_widths = jnp.asarray(widths)
        self.maximum_half_width_cells = maximum
        self.evolving = bool(evolving)
        self.capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=False,
            supports_mixed_entities=False,
            apic_compatible=True,
            source_geometry_kind="cpGIMP/AABB" if evolving else "uGIMP",
            domain_differentiable=bool(evolving),
            maximum_support_radius_cells=1.0 + maximum,
        )
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "uniform-gimp-splat-assignment",
                "reference_half_widths": array_tree_fingerprint(widths),
                "maximum_half_width_cells": maximum,
                "evolving": bool(evolving),
                "capabilities": self.capabilities.capability_id,
            }
        )

    def route_width(self, dimension: int, /) -> int:
        axis_width = 2 * ceil(1.0 + self.maximum_half_width_cells) + 1
        return axis_width ** int(dimension)

    def validate(self, layout, axes, /) -> None:
        if tuple(layout.axis_entities) != ("point",) * len(axes):
            raise ValueError("GIMP requires a nodal tensor-grid target.")
        if self.reference_half_widths.shape[1] != len(axes):
            raise ValueError("GIMP width dimension differs from the target.")
        for coordinates, axis in zip(layout.coordinates_by_axis, axes, strict=True):
            _uniform_spacing(
                coordinates,
                tuple(float(value) for value in np.asarray(axis.bounds)),
                axis.periodic,
            )

    def validate_input(self, assignment_input, source_count, dimension, /) -> None:
        if self.reference_half_widths.shape != (source_count, dimension):
            raise ValueError("GIMP reference widths must match prepared particles.")
        if self.evolving:
            if not isinstance(assignment_input, GIMPAssignmentInput):
                raise TypeError("cpGIMP requires GIMPAssignmentInput.")
            if assignment_input.half_widths.shape != (source_count, dimension):
                raise ValueError("cpGIMP runtime widths changed shape.")
        elif assignment_input is not None:
            raise ValueError("uGIMP owns fixed prepared widths and accepts no input.")

    def update_input(self, position, deformation_gradient, committed_input, /):
        del position, committed_input
        if not self.evolving:
            return None
        widths = oe.contract(
            "pij,pj->pi", jnp.abs(deformation_gradient), self.reference_half_widths
        )
        return GIMPAssignmentInput(widths)

    def build(
        self,
        layout,
        axes,
        axis_bounds,
        position,
        active,
        *,
        assignment_input=None,
    ) -> SplatAssignmentState:
        self.validate_input(
            assignment_input, int(position.shape[0]), int(position.shape[1])
        )
        widths = (
            assignment_input.half_widths
            if self.evolving
            else self.reference_half_widths.astype(position.dtype)
        )
        axis_width = 2 * ceil(1.0 + self.maximum_half_width_cells) + 1
        stencils = tuple(
            _gimp_axis_stencil(
                coordinates,
                bounds,
                axis.periodic,
                position[:, axis_index],
                widths[:, axis_index],
                active,
                axis_width,
                self.maximum_half_width_cells,
            )
            for axis_index, (coordinates, bounds, axis) in enumerate(
                zip(layout.coordinates_by_axis, axis_bounds, axes, strict=True)
            )
        )
        return _tensor_product_state(stencils, layout.shape, active)


__all__ = ["GIMPAssignmentInput", "UniformGIMPSplatAssignment"]
