#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import ceil, prod
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import gather_patches, GatherStencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import parse
from .._tensor_index import TensorIndexLayout
from ..splatting import (
    CPDI2AssignmentInput,
    MultilinearSplatAssignment,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    SplatAssignmentState,
    TensorBSplineSplatAssignment,
)
from ..splatting._assignment import _basis_and_derivative
from ._phases import update_deformation


type MPMVolumetricStabilizationKind = Literal["none", "b_bar", "f_bar"]


@final
class MPMVolumetricStabilizationPlan(StrictModule, NonTrainableState):
    """Cell-gradient B-bar or Zhao--Jiang--Choo incremental F-bar.

    B-bar adapts Hughes (1980), IJNME 15:1413--1418 to background cells.
    F-bar follows arXiv:2209.02466v3, Eq. (27) and Algorithm 1: project
    the trial *total assumed* Jacobian with current physical volume weights.
    The optional blend interpolates between local and projected Jacobians.
    """

    kind: MPMVolumetricStabilizationKind = eqx.field(static=True)
    f_bar_blend: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MPMVolumetricStabilizationKind = "none",
        /,
        *,
        f_bar_blend: float = 1.0,
    ) -> None:
        kind_ = parse(kind, MPMVolumetricStabilizationKind, "kind")
        blend = float(f_bar_blend)
        if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("f_bar_blend must lie in [0, 1].")
        self.kind = kind_
        self.f_bar_blend = blend
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-volumetric-stabilization", "mode": kind_, "f_bar_blend": blend}
        )


@final
class PreparedMPMCellAxis(StrictModule, NonTrainableState):
    """One-dimensional factors of a background-cell average."""

    indices: Array
    values: Array
    derivatives: Array
    valid: Array
    supported: Array
    base: Array
    width: int = eqx.field(static=True)


@final
class PreparedMPMCellAverage(StrictModule, NonTrainableState):
    """Factorized cell tables; never a dense copy of a virtual logical grid."""

    layout: TensorIndexLayout
    axes: tuple[PreparedMPMCellAxis, ...]
    columns: Array
    route_width: int = eqx.field(static=True)


def _quadrature_nodes(count: int, /) -> tuple[float, ...]:
    root = 1.0 / np.sqrt(3.0)
    if count == 2:
        return ((1.0 - root) / 2.0, (1.0 + root) / 2.0)
    return tuple(
        (half + (1.0 + sign * root) / 2.0) / 2.0
        for half in (0.0, 1.0)
        for sign in (-1.0, 1.0)
    )


def _prepare_cell_axis(
    coordinates: NDArray[np.float64],
    degree: int,
    periodic: bool,
    period: float,
    /,
) -> PreparedMPMCellAxis:
    # Gauss points avoid all basis knots. The half-cell split makes quadratic
    # cardinal splines polynomial on each integration interval.
    offsets = np.asarray((0, 1) if degree < 2 else (-1, 0, 1, 2), dtype=np.int32)
    quadrature = jnp.asarray(
        _quadrature_nodes(4 if degree == 2 else 2), dtype=jnp.float64
    )
    values, derivatives = _basis_and_derivative(
        degree, quadrature[:, None] - jnp.asarray(offsets)[None, :]
    )
    average_values = np.asarray(values, dtype=np.float64).mean(axis=0)
    average_derivatives = np.asarray(derivatives, dtype=np.float64).mean(axis=0)
    count = coordinates.size if periodic else coordinates.size - 1
    widths = np.diff(
        np.concatenate((coordinates, np.asarray([coordinates[0] + period])))
        if periodic
        else coordinates
    )
    width = min(offsets.size, coordinates.size) if periodic else offsets.size
    # Periodic aliases are combined on the host, so even a minimal periodic
    # grid has one route per physical node.
    columns = np.arange(offsets.size, dtype=np.int32)
    destinations = columns % coordinates.size if periodic else columns
    integrated_values = np.zeros((width,), dtype=np.float64)
    integrated_derivatives = np.zeros((width,), dtype=np.float64)
    np.add.at(integrated_values, destinations, average_values)
    np.add.at(integrated_derivatives, destinations, average_derivatives)
    base = np.arange(count, dtype=np.int32) + offsets[0]
    raw = base[:, None] + np.arange(width, dtype=np.int32)[None, :]
    valid = (
        np.ones(raw.shape, dtype=np.bool_)
        if periodic
        else (raw >= 0) & (raw < coordinates.size)
    )
    indices = (
        raw % coordinates.size if periodic else np.clip(raw, 0, coordinates.size - 1)
    )
    table_values = np.where(valid, integrated_values[None, :], 0.0)
    table_derivatives = np.where(
        valid, integrated_derivatives[None, :] / widths[:, None], 0.0
    )
    supported = (
        np.abs(np.sum(table_values, axis=1) - 1.0) <= 64.0 * np.finfo(np.float64).eps
    )
    return PreparedMPMCellAxis(
        jnp.asarray(indices, dtype=jnp.int32),
        jnp.asarray(table_values, dtype=jnp.float64),
        jnp.asarray(table_derivatives, dtype=jnp.float64),
        jnp.asarray(valid, dtype=jnp.bool_),
        jnp.asarray(supported, dtype=jnp.bool_),
        jnp.asarray(base, dtype=jnp.int32),
        width,
    )


def prepare_cell_average(splat: PreparedParticleGridSplat, /) -> PreparedMPMCellAverage:
    """Integrate particle-independent assignments once, per axis and cell."""
    layout = TensorIndexLayout(
        splat.layout.axis_names,
        splat.plan.target.structured_axes,
        splat.layout.axis_entities,
    )
    assignment = splat.plan.assignment
    dimension = len(splat.target_shape)
    if isinstance(assignment, TensorBSplineSplatAssignment):
        degrees = assignment._axis_degrees(dimension)
    elif isinstance(assignment, MultilinearSplatAssignment):
        degrees = (1,) * dimension
    else:
        radius = assignment.capabilities.maximum_support_radius_cells
        widths = tuple(
            min(2 * ceil(radius) + 2, size) if axis.periodic else 2 * ceil(radius) + 2
            for size, axis in zip(
                splat.target_shape, splat.plan.target.structured_axes, strict=True
            )
        )
        return PreparedMPMCellAverage(
            layout, (), jnp.empty((0, dimension), dtype=jnp.int32), prod(widths) + 1
        )
    axes = tuple(
        _prepare_cell_axis(
            np.asarray(coordinates, dtype=np.float64),
            degree,
            axis.periodic,
            bounds[1] - bounds[0],
        )
        for coordinates, degree, axis, bounds in zip(
            splat.layout.coordinates_by_axis,
            degrees,
            splat.plan.target.structured_axes,
            splat.axis_bounds,
            strict=True,
        )
    )
    columns = jnp.asarray(
        tuple(product(*(range(axis.width) for axis in axes))), dtype=jnp.int32
    )
    return PreparedMPMCellAverage(
        layout, axes, columns, prod(axis.width for axis in axes)
    )


def _cell_geometry(
    splat: PreparedParticleGridSplat, position: Array, /
) -> tuple[tuple[Array, ...], Array, Array]:
    cells = []
    lower_axes = []
    widths = []
    for axis, (coordinates, grid_axis, bounds) in enumerate(
        zip(
            splat.layout.coordinates_by_axis,
            splat.plan.target.structured_axes,
            splat.axis_bounds,
            strict=True,
        )
    ):
        coordinate = position[:, axis]
        if grid_axis.periodic:
            period = bounds[1] - bounds[0]
            coordinate = jnp.mod(coordinate - coordinates[0], period) + coordinates[0]
            cell = jnp.clip(
                jnp.searchsorted(coordinates, coordinate, side="right") - 1,
                0,
                coordinates.size - 1,
            )
            upper = jnp.where(
                cell == coordinates.size - 1,
                coordinates[0] + period,
                coordinates[jnp.minimum(cell + 1, coordinates.size - 1)],
            )
        else:
            cell = jnp.clip(
                jnp.searchsorted(coordinates, coordinate, side="right") - 1,
                0,
                coordinates.size - 2,
            )
            upper = coordinates[cell + 1]
        cells.append(cell)
        lower_axes.append(coordinates[cell])
        widths.append(upper - coordinates[cell])
    return tuple(cells), jnp.stack(lower_axes, axis=-1), jnp.stack(widths, axis=-1)


def _coalesce_payload(
    splat: PreparedParticleGridSplat,
    routes: ParticleGridSplatState,
    slots: Array,
    valid: Array,
    payload: Array,
    width: int,
    /,
) -> Array:
    """Reduce fixed source routes onto particle-local canonical node slots."""
    capacity = slots.shape[0]
    stencil = GatherStencil(
        indices=jnp.arange(capacity, dtype=jnp.int32)[:, None] * width + slots,
        weights=jnp.zeros_like(slots, dtype=payload.dtype),
        source_size=capacity * width,
        valid=valid,
        support=routes.stencil.support,
        case_shape=routes.stencil.case_shape,
    )
    masked = jnp.where(valid[..., None], payload, 0.0)
    return splat.scatter_route_payload_mapped(routes, masked, stencil).values.reshape(
        (capacity, width, payload.shape[-1])
    )


def _replace_cell_routes(
    routes: ParticleGridSplatState,
    indices: Array,
    weights: Array,
    ordinary: Array,
    averaged: Array,
    offsets: Array,
    valid: Array,
    supported: Array,
    /,
) -> ParticleGridSplatState:
    stencil = GatherStencil(
        indices=indices,
        weights=weights,
        source_size=routes.stencil.source_size,
        valid=valid,
        support=routes.stencil.support & supported,
        case_shape=routes.stencil.case_shape,
    )
    assignment = eqx.tree_at(
        lambda value: (
            value.indices,
            value.weights,
            value.weight_gradients,
            value.route_offsets,
            value.valid,
        ),
        routes.assignment_state,
        (indices, weights, ordinary, offsets, valid),
    )
    result = eqx.tree_at(
        lambda value: (
            value.stencil,
            value.assignment_state,
            value.successful,
            value.valid_route_count,
        ),
        routes,
        (
            stencil,
            assignment,
            routes.successful & jnp.all(~routes.source_active_mask | supported),
            jnp.sum(valid, dtype=jnp.int32),
        ),
    )
    return eqx.tree_at(
        lambda value: value.cell_average_gradients,
        result,
        averaged,
        is_leaf=lambda value: value is None,
    )


def _cached_cell_routes(
    splat: PreparedParticleGridSplat,
    cache: PreparedMPMCellAverage,
    routes: ParticleGridSplatState,
    position: Array,
    cells: tuple[Array, ...],
    /,
) -> ParticleGridSplatState:
    capacity, dimension = position.shape
    coordinates = []
    factors = []
    derivative_factors = []
    valid = jnp.broadcast_to(
        routes.source_active_mask[:, None], (capacity, cache.route_width)
    )
    supported = jnp.ones((capacity,), dtype=jnp.bool_)
    for index, (axis, cell) in enumerate(zip(cache.axes, cells, strict=True)):
        column = cache.columns[:, index]
        coordinates.append(axis.indices[cell[:, None], column[None, :]])
        factors.append(axis.values[cell[:, None], column[None, :]].astype(position.dtype))
        derivative_factors.append(
            axis.derivatives[cell[:, None], column[None, :]].astype(position.dtype)
        )
        valid = valid & axis.valid[cell[:, None], column[None, :]]
        supported = supported & axis.supported[cell]
    indices, _ = cache.layout.flat_indices(jnp.stack(coordinates, axis=-1))
    components = []
    for index in range(dimension):
        component = derivative_factors[index]
        for other in range(dimension):
            if other != index:
                component = component * factors[other]
        components.append(component)
    averaged = jnp.stack(components, axis=-1)
    original_coordinates, _ = cache.layout.integer_coordinates(routes.stencil.indices)
    slots = jnp.zeros_like(routes.stencil.indices)
    mapped_valid = routes.stencil.valid
    for index, (axis, cell, grid_axis) in enumerate(
        zip(cache.axes, cells, splat.plan.target.structured_axes, strict=True)
    ):
        column = original_coordinates[..., index] - axis.base[cell][:, None]
        if grid_axis.periodic:
            column = jnp.mod(column, splat.target_shape[index])
        mapped_valid = mapped_valid & (column >= 0) & (column < axis.width)
        slots = slots + jnp.clip(column, 0, axis.width - 1) * prod(
            factor.width for factor in cache.axes[index + 1 :]
        )
    payload = jnp.concatenate(
        (
            routes.stencil.weights[..., None],
            routes.weight_gradients,
            routes.stencil.weights[..., None] * routes.route_offsets,
        ),
        axis=-1,
    )
    reduced = _coalesce_payload(
        splat, routes, slots, mapped_valid, payload, cache.route_width
    )
    weights = reduced[..., 0]
    ordinary = reduced[..., 1 : 1 + dimension]
    node_coordinates, _ = cache.layout.coordinates_at(indices)
    offsets = jnp.where(
        (weights > 0.0)[..., None],
        reduced[..., 1 + dimension :] / jnp.where(weights > 0.0, weights, 1.0)[..., None],
        node_coordinates - position[:, None, :],
    )
    supported = supported & jnp.all(~routes.stencil.valid | mapped_valid, axis=-1)
    return _replace_cell_routes(
        routes, indices, weights, ordinary, averaged, offsets, valid, supported
    )


def _domain_cell_routes(
    splat: PreparedParticleGridSplat,
    cache: PreparedMPMCellAverage,
    routes: ParticleGridSplatState,
    position: Array,
    assignment_input: object,
    lower: Array,
    width: Array,
    /,
) -> ParticleGridSplatState:
    """Integrate moving domains, then combine every occurrence of each node."""
    dimension = position.shape[-1]
    quadrature = jnp.asarray(
        tuple(product(_quadrature_nodes(4), repeat=dimension)), dtype=position.dtype
    )
    points = lower[None, :, :] + quadrature[:, None, :] * width[None, :, :]
    active = routes.source_active_mask & ~routes.invalid_geometry_mask

    def at_point(point: Array) -> SplatAssignmentState:
        domain = assignment_input
        if isinstance(domain, CPDI2AssignmentInput):
            domain = CPDI2AssignmentInput(
                domain.corners + (point - position)[:, None, :],
                point,
                domain.deformation_gradient,
            )
        return splat.plan.assignment.build(
            splat.layout,
            splat.plan.target.structured_axes,
            splat.axis_bounds,
            point,
            active,
            assignment_input=domain,
        )

    sampled = jax.vmap(at_point)(points)
    capacity = position.shape[0]

    def flatten(value: Array) -> Array:
        return value.swapaxes(0, 1).reshape((capacity, -1) + value.shape[3:])

    source_indices = jnp.concatenate(
        (routes.stencil.indices, flatten(sampled.indices)), axis=1
    )
    source_valid = jnp.concatenate((routes.stencil.valid, flatten(sampled.valid)), axis=1)

    def identify(indices: Array, valid: Array) -> tuple[Array, Array]:
        return jnp.unique(
            jnp.where(valid, indices, splat.target_size),
            return_inverse=True,
            size=cache.route_width,
            fill_value=splat.target_size,
        )

    indices, slots = jax.vmap(identify)(source_indices, source_valid)
    within_capacity = slots < cache.route_width
    source_valid = source_valid & within_capacity
    source_weights = jnp.concatenate(
        (routes.stencil.weights, jnp.zeros_like(flatten(sampled.weights))), axis=1
    )
    ordinary = jnp.concatenate(
        (routes.weight_gradients, jnp.zeros_like(flatten(sampled.weight_gradients))),
        axis=1,
    )
    averaged = jnp.concatenate(
        (
            jnp.zeros_like(routes.weight_gradients),
            flatten(sampled.weight_gradients) / quadrature.shape[0],
        ),
        axis=1,
    )
    source_offsets = jnp.concatenate(
        (
            routes.route_offsets,
            flatten(
                sampled.route_offsets + (points - position[None, :, :])[:, :, None, :]
            ),
        ),
        axis=1,
    )
    payload = jnp.concatenate(
        (
            source_weights[..., None],
            ordinary,
            source_weights[..., None] * source_offsets,
            averaged,
        ),
        axis=-1,
    )
    reduced = _coalesce_payload(
        splat,
        routes,
        jnp.minimum(slots, cache.route_width - 1),
        source_valid,
        payload,
        cache.route_width,
    )
    weights = reduced[..., 0]
    ordinary = reduced[..., 1 : 1 + dimension]
    averaged = reduced[..., 1 + 2 * dimension :]
    node_coordinates, _ = cache.layout.coordinates_at(indices)
    offsets = jnp.where(
        (weights > 0.0)[..., None],
        reduced[..., 1 + dimension : 1 + 2 * dimension]
        / jnp.where(weights > 0.0, weights, 1.0)[..., None],
        node_coordinates - position[:, None, :],
    )
    valid = (indices < splat.target_size) & active[:, None]
    tolerance = jnp.finfo(position.dtype).eps * max(32, splat.route_width)
    supported = jnp.all(jnp.abs(sampled.captured_fractions - 1.0) <= tolerance, axis=0)
    supported = supported & jnp.all(within_capacity, axis=-1)
    return _replace_cell_routes(
        routes,
        jnp.minimum(indices, splat.target_size - 1),
        weights,
        ordinary,
        averaged,
        offsets,
        valid,
        supported,
    )


def build_cell_average_routes(
    splat: PreparedParticleGridSplat,
    cache: PreparedMPMCellAverage,
    routes: ParticleGridSplatState,
    position: Array,
    assignment_input: object,
    /,
) -> ParticleGridSplatState:
    """One canonical node route for both the ordinary and cell-average operators."""
    position = splat.plan.precision.geometry(position)
    if splat.plan.execution.geometry_ad == "frozen":
        position = jax.lax.stop_gradient(position)
        assignment_input = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            assignment_input,
            is_leaf=lambda value: value is None,
        )
    cells, lower, width = _cell_geometry(splat, position)
    if cache.axes:
        return _cached_cell_routes(splat, cache, routes, position, cells)
    return _domain_cell_routes(
        splat, cache, routes, position, assignment_input, lower, width
    )


def stabilized_deformation(
    plan: MPMVolumetricStabilizationPlan,
    splat: PreparedParticleGridSplat,
    routes: ParticleGridSplatState,
    deformation: Array,
    gradient: Array,
    step: Array,
    reference_volume: Array | None,
    physical_deformation: Array | None,
    active: Array,
    field_slots: Array | None,
    field_count: int,
    /,
) -> Array:
    match plan.kind:
        case "none" | "b_bar":
            return update_deformation(deformation, gradient, step)
        case "f_bar":
            if reference_volume is None or physical_deformation is None:
                raise ValueError(
                    "F-bar requires reference volumes and physical deformation gradients."
                )
            if plan.f_bar_blend == 0.0:
                return update_deformation(deformation, gradient, step)
            return _f_bar_deformation(
                plan,
                splat,
                routes,
                deformation,
                gradient,
                step,
                reference_volume,
                physical_deformation,
                active,
                field_slots,
                field_count,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _f_bar_deformation(
    plan: MPMVolumetricStabilizationPlan,
    splat: PreparedParticleGridSplat,
    routes: ParticleGridSplatState,
    deformation: Array,
    gradient: Array,
    step: Array,
    reference_volume: Array,
    physical_deformation: Array,
    active: Array,
    field_slots: Array | None,
    field_count: int,
    /,
) -> Array:
    dimension = deformation.shape[-1]
    identity = jnp.broadcast_to(
        jnp.eye(dimension, dtype=deformation.dtype), deformation.shape
    )
    incremental = identity + step * gradient
    solve_plan = SmallLinearSolvePlan(dimension)
    delta_j = solve_small_linear(solve_plan, incremental, identity).determinant
    assumed_j = solve_small_linear(solve_plan, deformation, identity).determinant
    physical_j = solve_small_linear(
        solve_plan, physical_deformation, identity
    ).determinant
    valid = (
        (delta_j > 0.0)
        & jnp.isfinite(delta_j)
        & (assumed_j > 0.0)
        & jnp.isfinite(assumed_j)
        & (physical_j > 0.0)
        & jnp.isfinite(physical_j)
    )
    trial_j = jnp.where(active & valid, assumed_j * delta_j, 1.0)
    volume = jnp.where(active & valid, reference_volume * physical_j, 0.0)
    projected = jnp.zeros_like(trial_j)
    for field in range(field_count):
        owned = active if field_slots is None else active & (field_slots == field)
        weights = jnp.where(
            owned[:, None] & routes.stencil.valid,
            routes.stencil.weights * volume[:, None],
            0.0,
        )
        payload = jnp.stack((weights * trial_j[:, None], weights), axis=-1)
        reduced = splat.scatter_route_payload_mapped(
            routes, payload, routes.stencil
        ).values
        denominator = reduced[:, 1]
        nodal = jnp.where(
            denominator > 0.0,
            reduced[:, 0] / jnp.where(denominator > 0.0, denominator, 1.0),
            0.0,
        )
        gathered, supported = gather_patches(nodal, routes.stencil)
        particle_j = jnp.sum(
            jnp.where(supported, routes.stencil.weights * gathered, 0.0), axis=1
        )
        projected = jnp.where(owned, particle_j, projected)
    blended = plan.f_bar_blend * projected + (1.0 - plan.f_bar_blend) * trial_j
    factor = jnp.where(active & valid, blended / trial_j, 1.0) ** (1.0 / dimension)
    assumed_incremental = factor[:, None, None] * incremental
    # Inadmissible increments remain inadmissible, so the existing Jacobian
    # evidence rejects them transactionally; projection never repairs inversion.
    return ein.contract("pij,pjk->pik", assumed_incremental, deformation)


__all__ = ["MPMVolumetricStabilizationKind", "MPMVolumetricStabilizationPlan"]
