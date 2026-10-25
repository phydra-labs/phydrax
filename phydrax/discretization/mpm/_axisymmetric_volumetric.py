#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from math import ceil, prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..splatting import (
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    TensorBSplineSplatAssignment,
    UniformGIMPSplatAssignment,
)
from ..splatting._assignment import _uniform_spacing
from ._axisymmetric_routes import build_axisymmetric_assignment
from ._volumetric import _cell_geometry, _coalesce_payload, _replace_cell_routes


@final
class PreparedAxisymmetricCellAverage(StrictModule, NonTrainableState):
    """Fixed Gauss integration of the full cylindrical divergence.

    A cell's physical ring measure is proportional to r dr dz. Spatial
    gradients average with this measure; the hoop operator averages N/r, so
    its radial factor cancels the measure's r. Keeping ghost routes distinct
    makes odd radial reflection commute with this operator and its adjoint.
    """

    quadrature: Array
    quadrature_weights: Array
    columns: Array
    axis_width: int = eqx.field(static=True)
    start_offset: int = eqx.field(static=True)
    route_width: int = eqx.field(static=True)
    quadrature_count: int = eqx.field(static=True)

    def __init__(self, splat: PreparedParticleGridSplat, /) -> None:
        for coordinates, bounds, axis in zip(
            splat.layout.coordinates_by_axis,
            splat.axis_bounds,
            splat.plan.target.structured_axes,
            strict=True,
        ):
            _uniform_spacing(coordinates, bounds, axis.periodic)
        assignment = splat.plan.assignment
        if isinstance(assignment, TensorBSplineSplatAssignment):
            degrees = assignment._axis_degrees(2)
            if max(degrees) > 3:
                # The ring rules below are exact only through cubic bases.
                raise ValueError(
                    "Axisymmetric B-bar ring averaging supports tensor B-spline "
                    "degrees up to three."
                )
            factors = []
            for degree in degrees:
                # Quadratic basis knots bisect cells; cubic bases are quartic
                # after ring weighting, requiring a three-point Gauss rule.
                if degree == 2:
                    nodes, weights = np.polynomial.legendre.leggauss(2)
                    factors.append(
                        (
                            np.concatenate(((nodes + 1.0) / 4.0, (nodes + 3.0) / 4.0)),
                            np.tile(weights / 4.0, 2),
                        )
                    )
                else:
                    nodes, weights = np.polynomial.legendre.leggauss(
                        3 if degree == 3 else 2
                    )
                    factors.append(((nodes + 1.0) / 2.0, weights / 2.0))
        else:
            nodes, weights = np.polynomial.legendre.leggauss(
                8 if isinstance(assignment, UniformGIMPSplatAssignment) else 2
            )
            factors = [((nodes + 1.0) / 2.0, weights / 2.0)] * 2
        columns = tuple(product(*(range(nodes.size) for nodes, _ in factors)))
        quadrature = np.asarray(
            tuple(
                tuple(factors[axis][0][column[axis]] for axis in range(2))
                for column in columns
            ),
            dtype=np.float64,
        )
        quadrature_weights = np.asarray(
            tuple(
                prod(factors[axis][1][column[axis]] for axis in range(2))
                for column in columns
            ),
            dtype=np.float64,
        )
        radius = ceil(assignment.capabilities.maximum_support_radius_cells)
        width = 2 * radius + 2
        self.quadrature = jnp.asarray(quadrature, dtype=jnp.float64)
        self.quadrature_weights = jnp.asarray(quadrature_weights, dtype=jnp.float64)
        self.columns = jnp.asarray(
            tuple(product(range(width), repeat=2)), dtype=jnp.int32
        )
        self.axis_width = width
        self.start_offset = -radius
        self.route_width = width**2
        self.quadrature_count = len(columns)


def _ghost_columns(
    splat: PreparedParticleGridSplat,
    coordinates: Array,
    base: Array,
    cache: PreparedAxisymmetricCellAverage,
    /,
) -> tuple[Array, Array]:
    spacing = jnp.stack(
        tuple(axis[1] - axis[0] for axis in splat.layout.coordinates_by_axis)
    )
    origin = jnp.stack(tuple(axis[0] for axis in splat.layout.coordinates_by_axis))
    integer = jnp.rint((coordinates - origin) / spacing).astype(jnp.int32)
    columns = integer - base[:, None, :]
    valid = jnp.all((columns >= 0) & (columns < cache.axis_width), axis=-1)
    slots = jnp.clip(
        columns[..., 0], 0, cache.axis_width - 1
    ) * cache.axis_width + jnp.clip(columns[..., 1], 0, cache.axis_width - 1)
    return slots, valid


def build_axisymmetric_cell_average_routes(
    splat: PreparedParticleGridSplat,
    cache: PreparedAxisymmetricCellAverage,
    routes: ParticleGridSplatState,
    position: Array,
    /,
) -> ParticleGridSplatState:
    position = splat.plan.precision.geometry(position)
    if splat.plan.execution.geometry_ad == "frozen":
        position = jax.lax.stop_gradient(position)
    capacity = position.shape[0]
    cells, lower, widths = _cell_geometry(splat, position)
    base = jnp.stack(cells, axis=-1) + cache.start_offset
    original_slots, original_supported = _ghost_columns(
        splat, position[:, None, :] + routes.route_offsets, base, cache
    )
    if routes.hoop_gradients is None:
        raise RuntimeError("Axisymmetric B-bar requires ordinary hoop gradients.")
    original_payload = jnp.concatenate(
        (
            routes.stencil.weights[..., None],
            routes.weight_gradients,
            routes.hoop_gradients[..., None],
            routes.stencil.weights[..., None] * routes.route_offsets,
        ),
        axis=-1,
    )
    reduced = _coalesce_payload(
        splat,
        routes,
        original_slots,
        routes.stencil.valid & original_supported,
        original_payload,
        cache.route_width,
    )
    points = (
        lower[:, None, :]
        + cache.quadrature.astype(position.dtype)[None, :, :] * widths[:, None, :]
    )
    quadrature_active = jnp.broadcast_to(
        routes.source_active_mask[:, None], points.shape[:2]
    ).reshape((-1,))
    assignment = splat.plan.assignment
    half_widths = (
        jnp.broadcast_to(
            assignment.reference_half_widths[:, None, :], points.shape
        ).reshape((-1, 2))
        if isinstance(assignment, UniformGIMPSplatAssignment)
        else None
    )
    quadrature = build_axisymmetric_assignment(
        splat, points.reshape((-1, 2)), quadrature_active, half_widths=half_widths
    )
    quadrature_state = quadrature.assignment
    ghost = (
        points.reshape((-1, 2))[:, None, :] + quadrature_state.route_offsets
    ).reshape((capacity, -1, 2))
    quadrature_slots, quadrature_supported = _ghost_columns(splat, ghost, base, cache)
    cell_radius = lower[:, 0] + 0.5 * widths[:, 0]
    measures = (
        cache.quadrature_weights.astype(position.dtype)[None, :]
        * points[..., 0]
        / cell_radius[:, None]
    )
    operators = jnp.concatenate(
        (quadrature_state.weight_gradients, quadrature.hoop_gradients[..., None]), axis=-1
    ).reshape((capacity, cache.quadrature_count, -1, 3))
    payload = (measures[:, :, None, None] * operators).reshape((capacity, -1, 3))
    quadrature_valid = (
        quadrature_state.valid.reshape((capacity, -1)) & quadrature_supported
    )
    averaged = _coalesce_payload(
        splat, routes, quadrature_slots, quadrature_valid, payload, cache.route_width
    )
    raw_coordinates = base[:, None, :] + cache.columns[None, :, :]
    radial = raw_coordinates[..., 0]
    reflected = splat.axis_bounds[0][0] == 0.0
    physical_radial = jnp.abs(radial) if reflected else radial
    axial = raw_coordinates[..., 1]
    if splat.plan.target.structured_axes[1].periodic:
        axial = jnp.mod(axial, splat.target_shape[1])
    valid = (
        routes.source_active_mask[:, None]
        & (physical_radial >= 0)
        & (physical_radial < splat.target_shape[0])
        & (axial >= 0)
        & (axial < splat.target_shape[1])
    )
    indices = jnp.clip(
        physical_radial, 0, splat.target_shape[0] - 1
    ) * splat.target_shape[1] + jnp.clip(axial, 0, splat.target_shape[1] - 1)
    spacing = jnp.stack(
        tuple(axis[1] - axis[0] for axis in splat.layout.coordinates_by_axis)
    )
    origin = jnp.stack(tuple(axis[0] for axis in splat.layout.coordinates_by_axis))
    offsets = (
        origin + raw_coordinates.astype(position.dtype) * spacing - position[:, None, :]
    )
    supported = (
        jnp.all(~routes.stencil.valid | original_supported, axis=-1)
        & jnp.all(
            quadrature_state.full_support.reshape((capacity, cache.quadrature_count)),
            axis=-1,
        )
        & jnp.all(
            ~quadrature_state.valid.reshape((capacity, -1)) | quadrature_supported,
            axis=-1,
        )
    )
    result = _replace_cell_routes(
        routes,
        indices,
        reduced[..., 0],
        reduced[..., 1:3],
        averaged[..., :2],
        offsets,
        valid,
        supported,
    )
    return eqx.tree_at(
        lambda value: (
            value.hoop_gradients,
            value.radial_parity,
            value.cell_average_hoop_gradients,
        ),
        result,
        (
            reduced[..., 3],
            jnp.where(reflected & (radial < 0), -1.0, 1.0).astype(position.dtype),
            averaged[..., 2],
        ),
        is_leaf=lambda value: value is None,
    )
