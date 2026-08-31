#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ._assignment import (
    _tensor_product_state,
    _uniform_axis_stencil,
    _uniform_spacing,
    AbstractStructuredSplatAssignment,
    SplatAssignmentCapabilities,
    SplatAssignmentState,
)


class CPDIAssignmentInput(StrictModule):
    current_edges: Array

    def __init__(self, current_edges: ArrayLike, /):
        value = jnp.asarray(current_edges)
        if value.ndim != 3 or value.shape[-1] != value.shape[-2]:
            raise ValueError("CPDI edge matrices must have shape (particles, d, d).")
        self.current_edges = value


class CPDI2AssignmentInput(StrictModule):
    corners: Array
    center: Array
    deformation_gradient: Array

    def __init__(
        self,
        corners: ArrayLike,
        center: ArrayLike,
        deformation_gradient: ArrayLike,
        /,
    ):
        corners_ = jnp.asarray(corners)
        center_ = jnp.asarray(center)
        deformation = jnp.asarray(deformation_gradient)
        if corners_.ndim != 3 or center_.ndim != 2 or deformation.ndim != 3:
            raise ValueError("CPDI2 input ranks are invalid.")
        if corners_.shape[0] != center_.shape[0] or center_.shape[1] != corners_.shape[2]:
            raise ValueError("CPDI2 corner and center dimensions differ.")
        if deformation.shape != (center_.shape[0], center_.shape[1], center_.shape[1]):
            raise ValueError("CPDI2 deformation shape changed.")
        self.corners = corners_
        self.center = center_
        self.deformation_gradient = deformation


def _corner_signs(dimension: int):
    return jnp.asarray(tuple(product((-1.0, 1.0), repeat=dimension)))


def _corner_point_state(layout, axes, axis_bounds, corners, active):
    particle_count, corner_count, dimension = corners.shape
    flat = corners.reshape((particle_count * corner_count, dimension))
    flat_active = jnp.repeat(active, corner_count)
    stencils = tuple(
        _uniform_axis_stencil(
            1,
            coordinates,
            bounds,
            axis.periodic,
            flat[:, axis_index],
            flat_active,
        )
        for axis_index, (coordinates, bounds, axis) in enumerate(
            zip(layout.coordinates_by_axis, axis_bounds, axes, strict=True)
        )
    )
    state = _tensor_product_state(stencils, layout.shape, flat_active)
    route_width = state.indices.shape[1]
    return state, route_width


def _combine_corner_routes(
    point_state,
    route_width,
    corners,
    center,
    parent_gradients,
    domain_valid,
    target_size,
):
    particle_count, corner_count, dimension = corners.shape
    route_count = corner_count * route_width
    indices = point_state.indices.reshape((particle_count, corner_count, route_width))
    point_weights = point_state.weights.reshape(
        (particle_count, corner_count, route_width)
    )
    point_valid = point_state.valid.reshape((particle_count, corner_count, route_width))
    point_offsets = point_state.route_offsets.reshape(
        (particle_count, corner_count, route_width, dimension)
    )
    corner_offsets = corners - center[:, None, :]
    offsets = point_offsets + corner_offsets[:, :, None, :]
    weights = point_weights / corner_count
    gradients = point_weights[..., None] * parent_gradients[:, :, None, :]
    valid = point_valid & domain_valid[:, None, None]
    flat_indices = indices.reshape((particle_count, route_count))
    flat_weights = weights.reshape((particle_count, route_count))
    flat_gradients = gradients.reshape((particle_count, route_count, dimension))
    flat_offsets = offsets.reshape((particle_count, route_count, dimension))
    flat_valid = valid.reshape((particle_count, route_count))

    def combine(
        source_indices, source_weights, source_gradients, source_offsets, source_valid
    ):
        sentinel = jnp.asarray(target_size, dtype=jnp.int32)
        safe_indices = jnp.where(source_valid, source_indices, sentinel)
        order = jnp.argsort(safe_indices, stable=True)
        sorted_indices = safe_indices[order]
        sorted_weights = source_weights[order]
        sorted_gradients = source_gradients[order]
        sorted_offsets = source_offsets[order]
        sorted_valid = source_valid[order]
        starts = sorted_valid & jnp.concatenate(
            (jnp.ones((1,), dtype=bool), sorted_indices[1:] != sorted_indices[:-1])
        )
        groups = jnp.cumsum(starts.astype(jnp.int32)) - 1
        safe_groups = jnp.maximum(groups, 0)
        output_weights = (
            jnp.zeros_like(sorted_weights)
            .at[safe_groups]
            .add(jnp.where(sorted_valid, sorted_weights, 0.0))
        )
        output_gradients = (
            jnp.zeros_like(sorted_gradients)
            .at[safe_groups]
            .add(jnp.where(sorted_valid[:, None], sorted_gradients, 0.0))
        )
        output_indices = (
            jnp.zeros_like(sorted_indices)
            .at[safe_groups]
            .set(jnp.where(sorted_valid, sorted_indices, 0))
        )
        output_offsets = (
            jnp.zeros_like(sorted_offsets)
            .at[safe_groups]
            .set(jnp.where(sorted_valid[:, None], sorted_offsets, 0.0))
        )
        group_count = jnp.sum(starts.astype(jnp.int32))
        output_valid = jnp.arange(route_count, dtype=jnp.int32) < group_count
        return (
            output_indices,
            output_weights,
            output_gradients,
            output_offsets,
            output_valid,
        )

    indices_, weights_, gradients_, offsets_, valid_ = jax.vmap(combine)(
        flat_indices, flat_weights, flat_gradients, flat_offsets, flat_valid
    )
    captured = jnp.sum(jnp.where(valid_, weights_, 0.0), axis=1)
    first = jnp.sum(
        jnp.where(valid_[..., None], weights_[..., None] * offsets_, 0.0), axis=1
    )
    second = oe.contract(
        "pr,pri,prj->pij", jnp.where(valid_, weights_, 0.0), offsets_, offsets_
    )
    gradient_sum = jnp.sum(jnp.where(valid_[..., None], gradients_, 0.0), axis=1)
    source_in_domain = domain_valid & jnp.all(
        point_state.source_in_domain.reshape((particle_count, corner_count)), axis=1
    )
    return SplatAssignmentState(
        indices=indices_,
        weights=weights_,
        weight_gradients=gradients_,
        route_offsets=offsets_,
        valid=valid_,
        source_in_domain=source_in_domain,
        captured_fractions=captured,
        full_support=source_in_domain & jnp.isclose(captured, 1.0, atol=1.0e-10),
        first_moments=first,
        second_moments=second,
        gradient_sums=gradient_sum,
    )


class _AbstractBaseCPDIAssignment(AbstractStructuredSplatAssignment):
    maximum_extent_cells: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def route_width(self, dimension: int, /) -> int:
        return 4 ** int(dimension)

    def validate(self, layout, axes, /):
        if tuple(layout.axis_entities) != ("point",) * len(axes):
            raise ValueError("CPDI requires a nodal tensor-grid target.")
        for coordinates, axis in zip(layout.coordinates_by_axis, axes, strict=True):
            _uniform_spacing(
                coordinates,
                tuple(float(value) for value in np.asarray(axis.bounds)),
                axis.periodic,
            )

    def _spacing(self, layout, axes):
        return jnp.asarray(
            tuple(
                _uniform_spacing(
                    coordinates,
                    tuple(float(value) for value in np.asarray(axis.bounds)),
                    axis.periodic,
                )
                for coordinates, axis in zip(
                    layout.coordinates_by_axis, axes, strict=True
                )
            )
        )


class AffineCPDISplatAssignment(_AbstractBaseCPDIAssignment):
    reference_edges: Array

    def __init__(
        self,
        reference_edges: ArrayLike,
        /,
        *,
        maximum_extent_cells: float = 2.0,
        maximum_condition: float = 1.0e6,
    ):
        edges = np.asarray(reference_edges, dtype=float)
        if edges.ndim != 3 or edges.shape[-1] != edges.shape[-2]:
            raise ValueError("CPDI reference edges must have shape (particles, d, d).")
        self.reference_edges = jnp.asarray(edges)
        self.maximum_extent_cells = float(maximum_extent_cells)
        self.maximum_condition = float(maximum_condition)
        self.capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=False,
            supports_mixed_entities=False,
            apic_compatible=True,
            source_geometry_kind="CPDI",
            domain_differentiable=True,
            maximum_support_radius_cells=1.0 + self.maximum_extent_cells,
        )
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "affine-cpdi-splat",
                "reference_edges": array_tree_fingerprint(edges),
                "maximum_extent_cells": self.maximum_extent_cells,
                "maximum_condition": self.maximum_condition,
            }
        )

    def validate_input(self, assignment_input, source_count, dimension, /):
        if self.reference_edges.shape != (source_count, dimension, dimension):
            raise ValueError("CPDI reference edges differ from prepared particles.")
        if not isinstance(assignment_input, CPDIAssignmentInput):
            raise TypeError("CPDI requires CPDIAssignmentInput.")
        if assignment_input.current_edges.shape != self.reference_edges.shape:
            raise ValueError("CPDI current edges changed shape.")

    def update_input(self, position, deformation_gradient, committed_input, /):
        del position, committed_input
        edges = oe.contract("pij,pjk->pik", deformation_gradient, self.reference_edges)
        return CPDIAssignmentInput(edges)

    def build(
        self, layout, axes, axis_bounds, position, active, *, assignment_input=None
    ):
        self.validate_input(
            assignment_input, int(position.shape[0]), int(position.shape[1])
        )
        edges = assignment_input.current_edges
        dimension = position.shape[1]
        signs = _corner_signs(dimension).astype(position.dtype)
        corner_offsets = oe.contract("pij,cj->pci", edges, signs)
        corners = position[:, None, :] + corner_offsets
        inverse = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            edges,
            jnp.broadcast_to(jnp.eye(dimension, dtype=position.dtype), edges.shape),
        )
        parent = signs / signs.shape[0]
        gradients = oe.contract("ca,pai->pci", parent, inverse.value)
        spacing = self._spacing(layout, axes).astype(position.dtype)
        extent = jnp.max(jnp.abs(corner_offsets) / spacing[None, None, :], axis=(1, 2))
        valid = (
            active
            & inverse.successful
            & (inverse.determinant > 0.0)
            & (inverse.condition_estimate <= self.maximum_condition)
            & (extent <= self.maximum_extent_cells)
        )
        point_state, width = _corner_point_state(
            layout, axes, axis_bounds, corners, active
        )
        return _combine_corner_routes(
            point_state,
            width,
            corners,
            position,
            gradients,
            valid,
            int(np.prod(layout.shape)),
        )


class CPDI2SplatAssignment(_AbstractBaseCPDIAssignment):
    reference_corner_offsets: Array

    def __init__(
        self,
        reference_corner_offsets: ArrayLike,
        /,
        *,
        maximum_extent_cells: float = 2.0,
        maximum_condition: float = 1.0e6,
    ):
        offsets = np.asarray(reference_corner_offsets, dtype=float)
        if offsets.ndim != 3 or offsets.shape[1] != 2 ** offsets.shape[2]:
            raise ValueError("CPDI2 reference corners must contain all tensor corners.")
        self.reference_corner_offsets = jnp.asarray(offsets)
        self.maximum_extent_cells = float(maximum_extent_cells)
        self.maximum_condition = float(maximum_condition)
        self.capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=False,
            supports_mixed_entities=False,
            apic_compatible=True,
            source_geometry_kind="CPDI2",
            domain_differentiable=True,
            maximum_support_radius_cells=1.0 + self.maximum_extent_cells,
        )
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "cpdi2-splat",
                "reference_corner_offsets": array_tree_fingerprint(offsets),
                "maximum_extent_cells": self.maximum_extent_cells,
                "maximum_condition": self.maximum_condition,
            }
        )

    def validate_input(self, assignment_input, source_count, dimension, /):
        expected = (source_count, 2**dimension, dimension)
        if self.reference_corner_offsets.shape != expected:
            raise ValueError("CPDI2 reference corners differ from prepared particles.")
        if not isinstance(assignment_input, CPDI2AssignmentInput):
            raise TypeError("CPDI2 requires CPDI2AssignmentInput.")
        if assignment_input.corners.shape != expected:
            raise ValueError("CPDI2 current corners changed shape.")

    def update_input(self, position, deformation_gradient, committed_input, /):
        if committed_input is None:
            corners = position[:, None, :] + oe.contract(
                "pij,pcj->pci", deformation_gradient, self.reference_corner_offsets
            )
        else:
            inverse_old = solve_small_linear(
                SmallLinearSolvePlan(position.shape[1]),
                committed_input.deformation_gradient,
                jnp.broadcast_to(
                    jnp.eye(position.shape[1], dtype=position.dtype),
                    committed_input.deformation_gradient.shape,
                ),
            )
            increment = oe.contract(
                "pij,pjk->pik", deformation_gradient, inverse_old.value
            )
            relative = committed_input.corners - committed_input.center[:, None, :]
            corners = position[:, None, :] + oe.contract(
                "pij,pcj->pci", increment, relative
            )
        return CPDI2AssignmentInput(corners, position, deformation_gradient)

    def build(
        self, layout, axes, axis_bounds, position, active, *, assignment_input=None
    ):
        self.validate_input(
            assignment_input, int(position.shape[0]), int(position.shape[1])
        )
        corners = assignment_input.corners
        dimension = position.shape[1]
        signs = _corner_signs(dimension).astype(position.dtype)
        parent = signs / signs.shape[0]
        jacobian = oe.contract("pci,ca->pia", corners, parent)
        inverse = solve_small_linear(
            SmallLinearSolvePlan(dimension),
            jacobian,
            jnp.broadcast_to(jnp.eye(dimension, dtype=position.dtype), jacobian.shape),
        )
        gradients = oe.contract("ca,pai->pci", parent, inverse.value)
        spacing = self._spacing(layout, axes).astype(position.dtype)
        offsets = corners - position[:, None, :]
        extent = jnp.max(jnp.abs(offsets) / spacing[None, None, :], axis=(1, 2))
        valid = (
            active
            & inverse.successful
            & (inverse.determinant > 0.0)
            & (inverse.condition_estimate <= self.maximum_condition)
            & (extent <= self.maximum_extent_cells)
        )
        point_state, width = _corner_point_state(
            layout, axes, axis_bounds, corners, active
        )
        return _combine_corner_routes(
            point_state,
            width,
            corners,
            position,
            gradients,
            valid,
            int(np.prod(layout.shape)),
        )


__all__ = [
    "AffineCPDISplatAssignment",
    "CPDI2AssignmentInput",
    "CPDI2SplatAssignment",
    "CPDIAssignmentInput",
]
