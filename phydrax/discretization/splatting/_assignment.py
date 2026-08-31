#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from itertools import product
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from .._tensor_entities import StructuredAxis, TensorEntityLayout


class SplatAssignmentCapabilities(StrictModule, NonTrainableState):
    """Static numerical properties of one structured splat assignment family."""

    partition_of_unity: bool = eqx.field(static=True)
    nonnegative_weights: bool = eqx.field(static=True)
    local_support: bool = eqx.field(static=True)
    polynomial_reproduction_order: int = eqx.field(static=True)
    maximum_explicit_derivative_order: int = eqx.field(static=True)
    supports_nonuniform: bool = eqx.field(static=True)
    supports_mixed_entities: bool = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)
    apic_compatible: bool = eqx.field(static=True)
    source_geometry_kind: str = eqx.field(static=True)
    domain_differentiable: bool = eqx.field(static=True)
    maximum_support_radius_cells: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        partition_of_unity: bool,
        nonnegative_weights: bool,
        local_support: bool,
        polynomial_reproduction_order: int,
        maximum_explicit_derivative_order: int,
        supports_nonuniform: bool,
        supports_mixed_entities: bool,
        apic_compatible: bool = True,
        source_geometry_kind: str = "point",
        domain_differentiable: bool = False,
        maximum_support_radius_cells: float = 2.0,
    ):
        reproduction = int(polynomial_reproduction_order)
        derivative = int(maximum_explicit_derivative_order)
        if reproduction < 0 or derivative < 0:
            raise ValueError(
                "Assignment reproduction and derivative orders must be nonnegative."
            )
        self.partition_of_unity = bool(partition_of_unity)
        self.nonnegative_weights = bool(nonnegative_weights)
        self.local_support = bool(local_support)
        self.polynomial_reproduction_order = reproduction
        self.maximum_explicit_derivative_order = derivative
        self.supports_nonuniform = bool(supports_nonuniform)
        self.supports_mixed_entities = bool(supports_mixed_entities)
        geometry = str(source_geometry_kind)
        if not geometry:
            raise ValueError("source_geometry_kind must be non-empty.")
        self.apic_compatible = bool(apic_compatible)
        self.source_geometry_kind = geometry
        self.domain_differentiable = bool(domain_differentiable)
        support_radius = float(maximum_support_radius_cells)
        if not np.isfinite(support_radius) or support_radius <= 0.0:
            raise ValueError("maximum_support_radius_cells must be positive.")
        self.maximum_support_radius_cells = support_radius
        self.capability_id = canonical_fingerprint(
            {
                "kind": "splat-assignment-capabilities",
                "partition_of_unity": self.partition_of_unity,
                "nonnegative_weights": self.nonnegative_weights,
                "local_support": self.local_support,
                "polynomial_reproduction_order": reproduction,
                "maximum_explicit_derivative_order": derivative,
                "supports_nonuniform": self.supports_nonuniform,
                "supports_mixed_entities": self.supports_mixed_entities,
                "apic_compatible": self.apic_compatible,
                "source_geometry_kind": geometry,
                "domain_differentiable": self.domain_differentiable,
                "maximum_support_radius_cells": support_radius,
            }
        )


class SplatAssignmentState(StrictModule):
    """Fixed source-target routes, derivatives, and moments for one assignment."""

    indices: Array
    weights: Array
    weight_gradients: Array
    route_offsets: Array
    valid: Array
    source_in_domain: Array
    captured_fractions: Array
    full_support: Array
    first_moments: Array
    second_moments: Array
    gradient_sums: Array

    def __init__(
        self,
        *,
        indices: ArrayLike,
        weights: ArrayLike,
        weight_gradients: ArrayLike,
        route_offsets: ArrayLike,
        valid: ArrayLike,
        source_in_domain: ArrayLike,
        captured_fractions: ArrayLike,
        full_support: ArrayLike,
        first_moments: ArrayLike,
        second_moments: ArrayLike,
        gradient_sums: ArrayLike,
    ):
        indices_ = jnp.asarray(indices)
        weights_ = jnp.asarray(weights)
        gradients = jnp.asarray(weight_gradients)
        offsets = jnp.asarray(route_offsets)
        valid_ = jnp.asarray(valid, dtype=bool)
        if indices_.ndim != 2 or weights_.shape != indices_.shape:
            raise ValueError(
                "Assignment indices and weights must have shape (sources, routes)."
            )
        if valid_.shape != indices_.shape:
            raise ValueError("Assignment validity must match the route shape.")
        source_count, route_count = indices_.shape
        if gradients.shape[:2] != (source_count, route_count):
            raise ValueError(
                "Assignment weight gradients must begin with the route shape."
            )
        if offsets.shape != gradients.shape:
            raise ValueError("Assignment route offsets and gradients must match.")
        dimension = gradients.shape[-1]
        source_shape = (source_count,)
        vectors = {
            "source_in_domain": jnp.asarray(source_in_domain, dtype=bool),
            "captured_fractions": jnp.asarray(captured_fractions),
            "full_support": jnp.asarray(full_support, dtype=bool),
        }
        if any(value.shape != source_shape for value in vectors.values()):
            raise ValueError("Assignment source evidence must match the source count.")
        first = jnp.asarray(first_moments)
        second = jnp.asarray(second_moments)
        gradient_sum = jnp.asarray(gradient_sums)
        if first.shape != (source_count, dimension):
            raise ValueError("Assignment first moments have an incompatible shape.")
        if second.shape != (source_count, dimension, dimension):
            raise ValueError("Assignment second moments have an incompatible shape.")
        if gradient_sum.shape != first.shape:
            raise ValueError("Assignment gradient sums have an incompatible shape.")
        self.indices = indices_.astype(jnp.int32)
        self.weights = weights_
        self.weight_gradients = gradients
        self.route_offsets = offsets
        self.valid = valid_
        self.source_in_domain = vectors["source_in_domain"]
        self.captured_fractions = vectors["captured_fractions"]
        self.full_support = vectors["full_support"]
        self.first_moments = first
        self.second_moments = second
        self.gradient_sums = gradient_sum


class AbstractStructuredSplatAssignment(StrictModule, NonTrainableState):
    """Shape-function contract for one structured particle-grid assignment."""

    capabilities: AbstractAttribute[SplatAssignmentCapabilities]
    assignment_id: AbstractAttribute[str]

    @abc.abstractmethod
    def route_width(self, dimension: int, /) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        /,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_input(
        self,
        assignment_input: object,
        source_count: int,
        dimension: int,
        /,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update_input(
        self,
        position: Array,
        deformation_gradient: Array,
        committed_input: object,
        /,
    ) -> object:
        raise NotImplementedError

    @abc.abstractmethod
    def build(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        axis_bounds: tuple[tuple[float, float], ...],
        position: Array,
        active: Array,
        *,
        assignment_input: object = None,
    ) -> SplatAssignmentState:
        del assignment_input
        raise NotImplementedError


def _uniform_spacing(
    coordinates: ArrayLike,
    bounds: tuple[float, float],
    periodic: bool,
    /,
) -> float:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError(
            "Uniform splat assignment requires at least two finite coordinates."
        )
    differences = np.diff(values)
    spacing = float(differences[0])
    tolerance = np.finfo(float).eps * max(32.0, abs(spacing) * values.size)
    if spacing <= 0.0 or not np.allclose(
        differences, spacing, rtol=1e-10, atol=tolerance
    ):
        raise ValueError(
            "Structured B-spline assignment requires uniformly spaced coordinates."
        )
    if periodic:
        expected = float(bounds[1] - bounds[0]) / values.size
        if not np.isclose(spacing, expected, rtol=1e-10, atol=tolerance):
            raise ValueError(
                "Periodic assignment coordinates do not span their declared period."
            )
    return spacing


def _basis_and_derivative(degree: int, coordinate: Array, /) -> tuple[Array, Array]:
    absolute = jnp.abs(coordinate)
    sign = jnp.sign(coordinate)
    if degree == 1:
        value = jnp.maximum(1.0 - absolute, 0.0)
        derivative = jnp.where(absolute < 1.0, -sign, 0.0)
        return value, derivative
    if degree == 2:
        central = 0.75 - absolute * absolute
        outer = 0.5 * jnp.maximum(1.5 - absolute, 0.0) ** 2
        value = jnp.where(absolute < 0.5, central, jnp.where(absolute < 1.5, outer, 0.0))
        central_derivative = -2.0 * coordinate
        outer_derivative = -jnp.maximum(1.5 - absolute, 0.0) * sign
        derivative = jnp.where(
            absolute < 0.5,
            central_derivative,
            jnp.where(absolute < 1.5, outer_derivative, 0.0),
        )
        return value, derivative
    if degree == 3:
        central = 2.0 / 3.0 - absolute**2 + 0.5 * absolute**3
        outer = jnp.maximum(2.0 - absolute, 0.0) ** 3 / 6.0
        value = jnp.where(absolute < 1.0, central, jnp.where(absolute < 2.0, outer, 0.0))
        central_derivative = (-2.0 * absolute + 1.5 * absolute**2) * sign
        outer_derivative = -0.5 * jnp.maximum(2.0 - absolute, 0.0) ** 2 * sign
        derivative = jnp.where(
            absolute < 1.0,
            central_derivative,
            jnp.where(absolute < 2.0, outer_derivative, 0.0),
        )
        return value, derivative
    raise ValueError("Structured splat basis degree must be one, two, or three.")


def _uniform_axis_stencil(
    degree: int,
    coordinates: Array,
    bounds: tuple[float, float],
    periodic: bool,
    position: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    count = int(coordinates.size)
    spacing = jnp.asarray(
        _uniform_spacing(coordinates, bounds, periodic), dtype=position.dtype
    )
    lower, upper = bounds
    evaluated = jnp.mod(position - lower, upper - lower) + lower if periodic else position
    source_in_domain = active & (
        jnp.ones_like(active) if periodic else (position >= lower) & (position <= upper)
    )
    normalized = (evaluated - coordinates[0]) / spacing
    if degree == 1:
        base = jnp.floor(normalized)
    elif degree == 2:
        base = jnp.floor(normalized - 0.5)
    else:
        base = jnp.floor(normalized - 1.0)
    raw = (
        base.astype(jnp.int32)[:, None] + jnp.arange(degree + 1, dtype=jnp.int32)[None, :]
    )
    target_coordinates = coordinates[0] + raw.astype(position.dtype) * spacing
    local = normalized[:, None] - raw.astype(position.dtype)
    weights, normalized_derivatives = _basis_and_derivative(degree, local)
    derivatives = normalized_derivatives / spacing
    if periodic:
        indices = jnp.mod(raw, count)
        route_valid = jnp.broadcast_to(source_in_domain[:, None], raw.shape)
    else:
        route_valid = source_in_domain[:, None] & (raw >= 0) & (raw < count)
        indices = jnp.clip(raw, 0, count - 1)
    offsets = target_coordinates - evaluated[:, None]
    return indices, weights, derivatives, offsets, route_valid, source_in_domain


def _linear_nonuniform_axis_stencil(
    coordinates: Array,
    bounds: tuple[float, float],
    periodic: bool,
    position: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    count = int(coordinates.size)
    lower, upper = bounds
    if periodic:
        period = upper - lower
        evaluated = jnp.mod(position - lower, period) + lower
        upper_raw = jnp.searchsorted(coordinates, evaluated, side="right")
        lower_index = jnp.mod(upper_raw - 1, count)
        upper_index = jnp.mod(upper_raw, count)
        lower_coordinate = coordinates[lower_index] - jnp.where(
            upper_raw == 0, period, 0.0
        )
        upper_coordinate = coordinates[upper_index] + jnp.where(
            upper_raw == count, period, 0.0
        )
        source_in_domain = active
    else:
        source_in_domain = active & (position >= lower) & (position <= upper)
        evaluated = jnp.clip(position, lower, upper)
        upper_raw = jnp.searchsorted(coordinates, evaluated, side="right")
        upper_index = jnp.clip(upper_raw, 1, count - 1)
        lower_index = upper_index - 1
        lower_coordinate = coordinates[lower_index]
        upper_coordinate = coordinates[upper_index]
    width = upper_coordinate - lower_coordinate
    fraction = jnp.clip((evaluated - lower_coordinate) / width, 0.0, 1.0)
    indices = jnp.stack((lower_index, upper_index), axis=-1).astype(jnp.int32)
    weights = jnp.stack((1.0 - fraction, fraction), axis=-1)
    derivatives = jnp.stack((-1.0 / width, 1.0 / width), axis=-1)
    target_coordinates = jnp.stack((lower_coordinate, upper_coordinate), axis=-1)
    offsets = target_coordinates - evaluated[:, None]
    valid = jnp.broadcast_to(source_in_domain[:, None], weights.shape)
    return indices, weights, derivatives, offsets, valid, source_in_domain


def _tensor_product_state(
    axis_stencils: tuple[tuple[Array, Array, Array, Array, Array, Array], ...],
    target_shape: tuple[int, ...],
    active: Array,
    /,
) -> SplatAssignmentState:
    source_count = int(active.size)
    dimension = len(axis_stencils)
    widths = tuple(int(stencil[0].shape[1]) for stencil in axis_stencils)
    route_count = prod(widths)
    route_indices = []
    route_weights = []
    route_gradients = []
    route_offsets = []
    route_validity = []
    source_in_domain = active.copy()
    for stencil in axis_stencils:
        source_in_domain = source_in_domain & stencil[5]
    for slots in product(*(range(width) for width in widths)):
        axis_indices = [
            axis_stencils[axis][0][:, slot] for axis, slot in enumerate(slots)
        ]
        flat_index = axis_indices[0]
        for axis in range(1, dimension):
            flat_index = flat_index * target_shape[axis] + axis_indices[axis]
        weight = jnp.ones((source_count,), dtype=axis_stencils[0][1].dtype)
        valid = active.copy()
        offsets = []
        for axis, slot in enumerate(slots):
            weight = weight * axis_stencils[axis][1][:, slot]
            valid = valid & axis_stencils[axis][4][:, slot]
            offsets.append(axis_stencils[axis][3][:, slot])
        gradient_components = []
        for derivative_axis in range(dimension):
            derivative = axis_stencils[derivative_axis][2][:, slots[derivative_axis]]
            for axis, slot in enumerate(slots):
                if axis != derivative_axis:
                    derivative = derivative * axis_stencils[axis][1][:, slot]
            gradient_components.append(derivative)
        route_indices.append(flat_index)
        route_weights.append(weight)
        route_gradients.append(jnp.stack(gradient_components, axis=-1))
        route_offsets.append(jnp.stack(offsets, axis=-1))
        route_validity.append(valid)
    indices = jnp.stack(route_indices, axis=-1)
    weights = jnp.stack(route_weights, axis=-1)
    gradients = jnp.stack(route_gradients, axis=1)
    offsets = jnp.stack(route_offsets, axis=1)
    valid = jnp.stack(route_validity, axis=-1)
    masked_weights = jnp.where(valid, weights, 0.0)
    captured = jnp.sum(masked_weights, axis=-1)
    tolerance = jnp.finfo(weights.dtype).eps * max(16, route_count)
    full_support = active & (jnp.abs(captured - 1.0) <= tolerance)
    first = jnp.sum(masked_weights[..., None] * offsets, axis=1)
    masked_gradients = jnp.where(valid[..., None], gradients, 0.0)
    gradient_sums = jnp.sum(masked_gradients, axis=1)
    second = jnp.sum(
        masked_weights[..., None, None] * offsets[..., :, None] * offsets[..., None, :],
        axis=1,
    )
    return SplatAssignmentState(
        indices=indices,
        weights=weights,
        weight_gradients=gradients,
        route_offsets=offsets,
        valid=valid,
        source_in_domain=source_in_domain,
        captured_fractions=captured,
        full_support=full_support,
        first_moments=first,
        second_moments=second,
        gradient_sums=gradient_sums,
    )


class MultilinearSplatAssignment(AbstractStructuredSplatAssignment):
    """First-order tensor assignment on nodal nonuniform or uniform mixed layouts."""

    capabilities: SplatAssignmentCapabilities = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(self):
        capabilities = SplatAssignmentCapabilities(
            partition_of_unity=True,
            nonnegative_weights=True,
            local_support=True,
            polynomial_reproduction_order=1,
            maximum_explicit_derivative_order=1,
            supports_nonuniform=True,
            supports_mixed_entities=True,
            maximum_support_radius_cells=1.0,
        )
        self.capabilities = capabilities
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "multilinear-splat-assignment",
                "capabilities": capabilities.capability_id,
            }
        )

    def route_width(self, dimension: int, /) -> int:
        dimension_ = int(dimension)
        if dimension_ <= 0:
            raise ValueError("Splat assignment dimension must be positive.")
        return 2**dimension_

    def validate(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        /,
    ) -> None:
        if len(axes) != len(layout.shape):
            raise ValueError("Assignment axes must match the target layout dimension.")
        if any(int(coordinates.size) < 2 for coordinates in layout.coordinates_by_axis):
            raise ValueError("Multilinear assignment requires two targets on every axis.")
        if any(entity == "interval" for entity in layout.axis_entities):
            for coordinates, axis in zip(layout.coordinates_by_axis, axes, strict=True):
                _uniform_spacing(
                    coordinates,
                    (
                        float(np.asarray(axis.bounds)[0]),
                        float(np.asarray(axis.bounds)[1]),
                    ),
                    axis.periodic,
                )

    def validate_input(self, assignment_input, source_count, dimension, /) -> None:
        del source_count, dimension
        if assignment_input is not None:
            raise ValueError("Multilinear assignment accepts no source-domain input.")

    def update_input(self, position, deformation_gradient, committed_input, /) -> object:
        del position, deformation_gradient, committed_input
        return None

    def build(
        self,
        layout: TensorEntityLayout,
        axes: tuple[StructuredAxis, ...],
        axis_bounds: tuple[tuple[float, float], ...],
        position: Array,
        active: Array,
        *,
        assignment_input: object = None,
    ) -> SplatAssignmentState:
        self.validate_input(
            assignment_input, int(position.shape[0]), int(position.shape[1])
        )
        uniform_mixed = any(entity == "interval" for entity in layout.axis_entities)
        stencils = []
        for axis, (coordinates, bounds, geometry_axis) in enumerate(
            zip(layout.coordinates_by_axis, axis_bounds, axes, strict=True)
        ):
            if uniform_mixed:
                stencil = _uniform_axis_stencil(
                    1,
                    coordinates,
                    bounds,
                    geometry_axis.periodic,
                    position[:, axis],
                    active,
                )
            else:
                stencil = _linear_nonuniform_axis_stencil(
                    coordinates,
                    bounds,
                    geometry_axis.periodic,
                    position[:, axis],
                    active,
                )
            stencils.append(stencil)
        return _tensor_product_state(tuple(stencils), layout.shape, active)


__all__ = [
    "AbstractStructuredSplatAssignment",
    "MultilinearSplatAssignment",
    "SplatAssignmentCapabilities",
    "SplatAssignmentState",
]
