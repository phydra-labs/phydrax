#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared finite-element geometry of the fuel and cladding bodies of a rod.

Two reductions share one interface. In the layered reduction every axial zone
of a body carries an independent copy of one reference radial P1 interval mesh.
In the RZ reduction each body is one connected P1 quadrilateral mesh with
``axial_cells_per_zone`` element rows per axial zone. Integration sites are
Gauss points; finite-element point-interpolation routes evaluate nodal fields
and their exact derivatives at the sites, and weak forms are assembled by the
routes' exact transpose scatter. Site and trace weights include the
cylindrical measure ``2 pi R`` of the reference configuration.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellBlock, CellMesh
from ...discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    prepare_finite_element_point_interpolation,
    PreparedFiniteElementPointInterpolation,
)
from ...integration import (
    CoordinateMeasurePlan,
    GaussLegendreRule,
    ReferenceIntervalRule,
    ReferenceQuadrilateralRule,
)
from ...sparse import EdgeRelation, SparseLinearMap
from ...typing import ConvertibleToArray
from ._irradiation import require_increasing_from_zero
from ._numerical_views import NumericalPointInterpolation
from ._specification import FuelRodSpecification


type FuelPointRoute = (
    PreparedFiniteElementPointInterpolation | NumericalPointInterpolation
)


class _RadialBodyGeometry(StrictModule, NonTrainableState):
    """Reference radial mesh of one body with its site routes and site weights.

    ``site_weight`` [m^2] is the Gauss weight times the cell length times the
    cylindrical measure ``2 pi R`` at the site.
    """

    nodes: Array
    site_radius: Array
    site_weight: Array
    value_route: FuelPointRoute
    gradient_route: FuelPointRoute
    body: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @property
    def site_count(self) -> int:
        return self.site_radius.shape[0]


def _radial_nodes(
    body: str,
    inner_radius: float,
    outer_radius: float,
    cell_count: int,
    quadrature_order: int,
    /,
    *,
    equal_area: bool,
) -> NDArray[np.float64]:
    """Validated reference node radii [m] of one radial body."""

    if not body or cell_count < 2 or quadrature_order < 1:
        raise ValueError("A radial body requires a name, >= 2 cells, and order >= 1.")
    if not (math.isfinite(inner_radius) and math.isfinite(outer_radius)) or not (
        0.0 <= inner_radius < outer_radius
    ):
        raise ValueError("Radial body bounds must satisfy 0 <= inner < outer.")
    fraction = np.linspace(0.0, 1.0, cell_count + 1)
    if equal_area:
        return np.sqrt(inner_radius**2 + fraction * (outer_radius**2 - inner_radius**2))
    return inner_radius + fraction * (outer_radius - inner_radius)


def _interval_space(
    body: str, nodes: NDArray[np.float64], /
) -> FiniteElementDiscretization:
    """Continuous P1 nodal field on the interval mesh through ``nodes``."""

    cell_count = nodes.shape[0] - 1
    connectivity = np.stack(
        (
            np.arange(cell_count, dtype=np.int32),
            np.arange(1, cell_count + 1, dtype=np.int32),
        ),
        axis=1,
    )
    mesh = CellMesh(nodes[:, None], (CellBlock(body, "interval", connectivity),))
    return FiniteElementPlan(
        mesh, FiniteElementFieldSpec("nodal", lagrange_element("interval", 1))
    ).prepare()


def _radial_body_geometry(
    body: str,
    inner_radius: float,
    outer_radius: float,
    cells: int,
    /,
    *,
    equal_area: bool,
    quadrature_order: int = 2,
) -> _RadialBodyGeometry:
    """Prepare one body mesh on ``[inner_radius, outer_radius]``.

    ``equal_area`` places nodes at equal cross-sectional area increments,
    which refines toward the outer surface (the fuel rim); otherwise nodes are
    uniform in radius.
    """

    name = str(body)
    cell_count = int(cells)
    order = int(quadrature_order)
    nodes = _radial_nodes(
        name,
        float(inner_radius),
        float(outer_radius),
        cell_count,
        order,
        equal_area=equal_area,
    )
    space = _interval_space(name, nodes)
    reference = ReferenceIntervalRule(GaussLegendreRule(order)).materialize()
    reference_points = np.asarray(reference.points)
    reference_weights = np.asarray(reference.weights)
    points_per_cell = reference_weights.shape[0]
    site_cells = np.repeat(np.arange(cell_count, dtype=np.int32), points_per_cell)
    site_points = np.tile(reference_points, (cell_count, 1))
    value_route = prepare_finite_element_point_interpolation(
        space, "nodal", name, site_cells, site_points
    )
    gradient_route = prepare_finite_element_point_interpolation(
        space, "nodal", name, site_cells, site_points, derivative_axis=0
    )
    # Straight P1 intervals: the physical site radius is the interpolated node
    # coordinate and the cell measure is the cell length.
    site_radius = np.asarray(value_route.interpolate(jnp.asarray(nodes)))
    cell_length = np.repeat(np.diff(nodes), points_per_cell)
    site_weight = (
        np.tile(reference_weights, cell_count) * cell_length * 2.0 * math.pi * site_radius
    )
    return _RadialBodyGeometry(
        nodes=jnp.asarray(nodes),
        site_radius=jnp.asarray(site_radius),
        site_weight=jnp.asarray(site_weight),
        value_route=value_route,
        gradient_route=gradient_route,
        body=name,
        geometry_id=canonical_fingerprint(
            {
                "kind": "nuclear-fuel-radial-body",
                "body": name,
                "nodes": nodes.tolist(),
                "quadrature_order": order,
            }
        ),
    )


def point_field_values(route: FuelPointRoute, nodal: Array, /) -> Array:
    """Interpolate a nodal field with any trailing component axes."""

    if nodal.ndim == 1:
        return route.interpolate(nodal)
    matrix = nodal.reshape((nodal.shape[0], -1))
    values = jax.vmap(route.interpolate, in_axes=1, out_axes=1)(matrix)
    return values.reshape((values.shape[0],) + nodal.shape[1:])


def _point_field_scatter(route: FuelPointRoute, dual: Array, /) -> Array:
    """Transpose of `point_field_values` for one route."""

    if dual.ndim == 1:
        return route.transpose_scatter(dual)
    matrix = dual.reshape((dual.shape[0], -1))
    values = jax.vmap(route.transpose_scatter, in_axes=1, out_axes=1)(matrix)
    return values.reshape((values.shape[0],) + dual.shape[1:])


def _layered_point_values(
    route: FuelPointRoute,
    nodal: Array,
    layer_count: int,
    radial_node_count: int,
    /,
) -> Array:
    """Apply one radial route to every axial layer of a flattened layered field."""

    per_layer = nodal.reshape((layer_count, radial_node_count) + nodal.shape[1:])
    values = jax.vmap(lambda layer_nodal: point_field_values(route, layer_nodal))(
        per_layer
    )
    return values.reshape((-1,) + nodal.shape[1:])


def _layered_point_scatter(
    route: FuelPointRoute, dual: Array, layer_count: int, /
) -> Array:
    """Transpose of `_layered_point_values` for one route."""

    per_layer = dual.reshape((layer_count, -1) + dual.shape[1:])
    output = jax.vmap(lambda layer_dual: _point_field_scatter(route, layer_dual))(
        per_layer
    )
    return output.reshape((-1,) + dual.shape[1:])


class FuelBodyGeometry(StrictModule, NonTrainableState):
    """Finite-element routes and site data of one body in either reduction.

    Layered fields are flattened collections of independent radial spaces, one
    per axial layer. RZ fields live on one connected quadrilateral mesh: axial
    gradients and shear are resolved, not reconstructed from independent radial
    solutions. ``site_zone`` is the axial zone of each integration site;
    ``site_inner_radius``/``site_outer_radius`` are the reference radii [m] of
    the cell containing it.
    """

    nodes: Array
    site_coordinates: Array
    site_weight: Array
    site_zone: Array
    site_inner_radius: Array
    site_outer_radius: Array
    value_route: FuelPointRoute
    radial_route: FuelPointRoute
    axial_route: FuelPointRoute | None
    radial_node_count: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    layered: bool = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @property
    def node_count(self) -> int:
        return self.nodes.shape[0]

    @property
    def site_count(self) -> int:
        return self.site_weight.shape[0]

    def values(self, nodal: Array, /) -> Array:
        """Field values at the sites."""

        if self.layered:
            return _layered_point_values(
                self.value_route, nodal, self.layer_count, self.radial_node_count
            )
        return point_field_values(self.value_route, nodal)

    def gradients(self, nodal: Array, /) -> Array:
        """Reference gradients ``(..., 2)`` (radial, axial) at the sites."""

        if self.layered:
            radial = _layered_point_values(
                self.radial_route, nodal, self.layer_count, self.radial_node_count
            )
            return jnp.stack((radial, jnp.zeros_like(radial)), axis=-1)
        if self.axial_route is None:
            raise ValueError("An RZ body requires its prepared axial derivative route.")
        return jnp.stack(
            (
                point_field_values(self.radial_route, nodal),
                point_field_values(self.axial_route, nodal),
            ),
            axis=-1,
        )

    def assemble(self, value_dual: Array, gradient_dual: Array, /) -> Array:
        """Nodal residual ``sum_s w_s (value_dual_s N + gradient_dual_s . grad N)``."""

        weight = self.site_weight.reshape((-1,) + (1,) * (value_dual.ndim - 1))
        value = weight * value_dual
        radial = weight * gradient_dual[..., 0]
        if self.layered:
            return _layered_point_scatter(
                self.value_route, value, self.layer_count
            ) + _layered_point_scatter(self.radial_route, radial, self.layer_count)
        if self.axial_route is None:
            raise ValueError("An RZ body requires its prepared axial derivative route.")
        return (
            _point_field_scatter(self.value_route, value)
            + _point_field_scatter(self.radial_route, radial)
            + _point_field_scatter(self.axial_route, weight * gradient_dual[..., 1])
        )

    def integrate(self, density: Array, /) -> Array:
        """Weighted sum of a site density over the body."""

        return jnp.sum(
            self.site_weight.reshape((-1,) + (1,) * (density.ndim - 1)) * density, axis=0
        )


class FuelTraceGeometry(StrictModule, NonTrainableState):
    """Routes to points on one body surface, with cylindrical surface weights.

    ``zone`` is the axial zone of each trace point (indexes zone material
    data); ``interface_index`` is its interface cell (indexes the per-cell
    contact and gap unknowns). The two coincide in the layered reduction.
    """

    route: FuelPointRoute
    axial_route: FuelPointRoute | None
    reference_radius: Array
    reference_z: Array
    weight: Array
    zone: Array
    interface_index: Array
    radial_node_count: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    layered: bool = eqx.field(static=True)

    def values(self, nodal: Array, /) -> Array:
        """Field values at the trace points."""

        if self.layered:
            return _layered_point_values(
                self.route, nodal, self.layer_count, self.radial_node_count
            )
        return point_field_values(self.route, nodal)

    def axial_gradient(self, nodal: Array, /) -> Array:
        """Axial derivative at the trace points; zero in the layered reduction."""

        if self.layered:
            return jnp.zeros_like(self.values(nodal))
        if self.axial_route is None:
            raise ValueError("This trace has no prepared axial derivative route.")
        return point_field_values(self.axial_route, nodal)

    def assemble(self, traction: Array, /) -> Array:
        """Nodal residual ``sum_p w_p traction_p N``."""

        weighted = self.weight.reshape((-1,) + (1,) * (traction.ndim - 1)) * traction
        if self.layered:
            return _layered_point_scatter(self.route, weighted, self.layer_count)
        return _point_field_scatter(self.route, weighted)


class FuelRodGeometry(StrictModule, NonTrainableState):
    """Fuel and cladding bodies of one rod with their interfaces and surfaces.

    ``layer_lengths`` [m] are the axial zone lengths; ``interface_lengths`` [m]
    and ``interface_zone`` give the length and axial zone of each interface
    cell (one per zone when layered, ``axial_cells_per_zone`` per zone in RZ).
    ``fuel_axis_nodes`` are fuel nodes on the centerline; the axial anchors are
    the bottom-row nodes whose axial displacement is fixed in RZ (empty when
    layered). ``coolant_nodes`` are the cladding outer-surface nodes carrying
    the coolant heat-flux unknowns, ``coolant_node_weights`` their surface
    weights, and ``coolant_flux_map`` interpolates them to the outer trace
    (``None`` when layered, where nodes and trace points coincide).
    ``fuel_end``/``cladding_end`` are the top end faces in RZ.
    """

    fuel: FuelBodyGeometry
    cladding: FuelBodyGeometry
    fuel_interface: FuelTraceGeometry
    cladding_interface: FuelTraceGeometry
    cladding_outer: FuelTraceGeometry
    layer_lengths: Array
    interface_lengths: Array
    interface_zone: Array
    fuel_axis_nodes: Array
    axial_anchor_fuel: Array
    axial_anchor_cladding: Array
    coolant_nodes: Array
    coolant_node_weights: Array
    coolant_flux_map: SparseLinearMap | None
    fuel_end: FuelTraceGeometry | None
    cladding_end: FuelTraceGeometry | None
    geometry_id: str = eqx.field(static=True)
    layered: bool = eqx.field(static=True)

    def coolant_flux_values(self, nodal_flux: Array, /) -> Array:
        """Coolant heat flux at the cladding outer trace points."""

        if self.layered:
            return nodal_flux
        flux_map = self.coolant_flux_map
        if flux_map is None:
            raise RuntimeError("An RZ coolant trace requires its outer-surface flux map.")
        return flux_map.mv(nodal_flux)

    def coolant_flux_rows(
        self, point_residual: Array, current_measure: Array, /
    ) -> Array:
        """Weighted trace equations with one row per coolant flux unknown."""

        if self.layered:
            return point_residual
        flux_map = self.coolant_flux_map
        if flux_map is None:
            raise RuntimeError("An RZ coolant trace requires its outer-surface flux map.")
        weighted = self.cladding_outer.weight * current_measure * point_residual
        return flux_map.transpose_mv(weighted) / self.coolant_node_weights


def _layered_trace(
    space: FiniteElementDiscretization,
    body: str,
    side: int,
    radius: float,
    radial_cell_count: int,
    zone_edges: NDArray[np.float64],
    /,
) -> FuelTraceGeometry:
    """Inner (``side == 0``) or outer (``side == 1``) surface of every layer."""

    zone_count = zone_edges.shape[0] - 1
    route = prepare_finite_element_point_interpolation(
        space,
        "nodal",
        body,
        np.asarray([0 if side == 0 else radial_cell_count - 1], dtype=np.int32),
        np.asarray([[float(side)]], dtype=np.float64),
    )
    return FuelTraceGeometry(
        route=route,
        axial_route=None,
        reference_radius=jnp.full((zone_count,), radius, dtype=jnp.float64),
        reference_z=jnp.asarray((zone_edges[:-1] + zone_edges[1:]) / 2),
        weight=jnp.asarray(2 * math.pi * radius * np.diff(zone_edges)),
        zone=jnp.arange(zone_count, dtype=jnp.int32),
        interface_index=jnp.arange(zone_count, dtype=jnp.int32),
        radial_node_count=radial_cell_count + 1,
        layer_count=zone_count,
        layered=True,
    )


def _layered_body_and_traces(
    body: str,
    inner_radius: float,
    outer_radius: float,
    radial_cell_count: int,
    zone_edges: NDArray[np.float64],
    quadrature_order: int,
    /,
) -> tuple[FuelBodyGeometry, FuelTraceGeometry, FuelTraceGeometry]:
    """Layered body with its inner and outer surface traces."""

    zone_count = zone_edges.shape[0] - 1
    radial = _radial_body_geometry(
        body,
        inner_radius,
        outer_radius,
        radial_cell_count,
        equal_area=inner_radius == 0.0,
        quadrature_order=quadrature_order,
    )
    radii = np.asarray(radial.nodes)
    zone_midpoints = (zone_edges[:-1] + zone_edges[1:]) / 2
    coordinates = np.stack(
        (np.tile(radii, zone_count), np.repeat(zone_midpoints, radial_cell_count + 1)),
        axis=-1,
    )
    site_coordinates = np.stack(
        (
            np.tile(np.asarray(radial.site_radius), zone_count),
            np.repeat(zone_midpoints, radial.site_count),
        ),
        axis=-1,
    )
    geometry = FuelBodyGeometry(
        nodes=jnp.asarray(coordinates),
        site_coordinates=jnp.asarray(site_coordinates),
        site_weight=jnp.asarray(
            (np.diff(zone_edges)[:, None] * np.asarray(radial.site_weight)).reshape(-1)
        ),
        site_zone=jnp.asarray(
            np.repeat(np.arange(zone_count, dtype=np.int32), radial.site_count)
        ),
        site_inner_radius=jnp.asarray(
            np.tile(np.repeat(radii[:-1], quadrature_order), zone_count)
        ),
        site_outer_radius=jnp.asarray(
            np.tile(np.repeat(radii[1:], quadrature_order), zone_count)
        ),
        value_route=radial.value_route,
        radial_route=radial.gradient_route,
        axial_route=None,
        radial_node_count=radial_cell_count + 1,
        layer_count=zone_count,
        layered=True,
        geometry_id=canonical_fingerprint(
            {"body": radial.geometry_id, "edges": zone_edges}
        ),
    )
    space = _interval_space(body, radii)
    return (
        geometry,
        _layered_trace(space, body, 0, inner_radius, radial_cell_count, zone_edges),
        _layered_trace(space, body, 1, outer_radius, radial_cell_count, zone_edges),
    )


def _rz_axial_nodes(
    zone_edges: NDArray[np.float64], axial_cells_per_zone: int, /
) -> NDArray[np.float64]:
    """Axial node coordinates [m]: ``axial_cells_per_zone`` equal cells per zone."""

    zone_count = zone_edges.shape[0] - 1
    return np.concatenate(
        tuple(
            np.linspace(zone_edges[zone], zone_edges[zone + 1], axial_cells_per_zone + 1)[
                :-1
            ]
            for zone in range(zone_count)
        )
        + (zone_edges[-1:],)
    )


def _rz_side_trace(
    space: FiniteElementDiscretization,
    body: str,
    side: int,
    radius: float,
    radial_cell_count: int,
    axial_nodes: NDArray[np.float64],
    line_points: NDArray[np.float64],
    line_weights: NDArray[np.float64],
    zone_count: int,
    axial_cells_per_zone: int,
    /,
) -> FuelTraceGeometry:
    """Inner (``side == 0``) or outer (``side == 1``) lateral surface in RZ."""

    axial_cell_count = axial_nodes.shape[0] - 1
    points_per_edge = line_points.shape[0]
    cells = np.repeat(
        np.arange(axial_cell_count, dtype=np.int32) * radial_cell_count
        + (0 if side == 0 else radial_cell_count - 1),
        points_per_edge,
    )
    reference_points = np.stack(
        (
            np.full(axial_cell_count * points_per_edge, side, dtype=np.float64),
            np.tile(line_points, axial_cell_count),
        ),
        axis=-1,
    )
    axial_cell_length = np.diff(axial_nodes)[:, None]
    return FuelTraceGeometry(
        route=prepare_finite_element_point_interpolation(
            space, "nodal", body, cells, reference_points
        ),
        axial_route=prepare_finite_element_point_interpolation(
            space, "nodal", body, cells, reference_points, derivative_axis=1
        ),
        reference_radius=jnp.full(
            (axial_cell_count * points_per_edge,), radius, dtype=jnp.float64
        ),
        reference_z=jnp.asarray(
            axial_nodes[:-1, None] + axial_cell_length * line_points
        ).reshape(-1),
        weight=jnp.asarray(
            2 * math.pi * radius * (axial_cell_length * line_weights)
        ).reshape(-1),
        zone=jnp.repeat(
            jnp.arange(axial_cell_count, dtype=jnp.int32) // axial_cells_per_zone,
            points_per_edge,
        ),
        interface_index=jnp.repeat(
            jnp.arange(axial_cell_count, dtype=jnp.int32), points_per_edge
        ),
        radial_node_count=radial_cell_count + 1,
        layer_count=zone_count,
        layered=False,
    )


def _rz_top_trace(
    space: FiniteElementDiscretization,
    body: str,
    coordinates: NDArray[np.float64],
    radii: NDArray[np.float64],
    axial_cell_count: int,
    line_points: NDArray[np.float64],
    line_weights: NDArray[np.float64],
    zone_count: int,
    /,
) -> FuelTraceGeometry:
    """Top end face of the body in RZ."""

    radial_cell_count = radii.shape[0] - 1
    points_per_edge = line_points.shape[0]
    cells = np.repeat(
        np.arange(radial_cell_count, dtype=np.int32)
        + (axial_cell_count - 1) * radial_cell_count,
        points_per_edge,
    )
    reference_points = np.stack(
        (
            np.tile(line_points, radial_cell_count),
            np.ones(radial_cell_count * points_per_edge),
        ),
        axis=-1,
    )
    route = prepare_finite_element_point_interpolation(
        space, "nodal", body, cells, reference_points
    )
    point_coordinates = np.asarray(point_field_values(route, jnp.asarray(coordinates)))
    return FuelTraceGeometry(
        route=route,
        axial_route=None,
        reference_radius=jnp.asarray(point_coordinates[:, 0]),
        reference_z=jnp.asarray(point_coordinates[:, 1]),
        weight=jnp.asarray(
            2
            * math.pi
            * point_coordinates[:, 0]
            * np.repeat(np.diff(radii), points_per_edge)
            * np.tile(line_weights, radial_cell_count)
        ),
        zone=jnp.full(
            (radial_cell_count * points_per_edge,), zone_count - 1, dtype=jnp.int32
        ),
        interface_index=jnp.zeros(
            (radial_cell_count * points_per_edge,), dtype=jnp.int32
        ),
        radial_node_count=radial_cell_count + 1,
        layer_count=zone_count,
        layered=False,
    )


def _rz_body_and_traces(
    body: str,
    inner_radius: float,
    outer_radius: float,
    radial_cell_count: int,
    zone_edges: NDArray[np.float64],
    quadrature_order: int,
    axial_cells_per_zone: int,
    /,
) -> tuple[FuelBodyGeometry, FuelTraceGeometry, FuelTraceGeometry, FuelTraceGeometry]:
    """RZ body with its inner, outer, and top surface traces."""

    zone_count = zone_edges.shape[0] - 1
    radii = _radial_nodes(
        body,
        inner_radius,
        outer_radius,
        radial_cell_count,
        quadrature_order,
        equal_area=inner_radius == 0.0,
    )
    radial_node_count = radial_cell_count + 1
    axial_nodes = _rz_axial_nodes(zone_edges, axial_cells_per_zone)
    axial_cell_count = axial_nodes.shape[0] - 1
    cell_count = radial_cell_count * axial_cell_count
    coordinates = np.stack(
        (
            np.tile(radii, axial_cell_count + 1),
            np.repeat(axial_nodes, radial_node_count),
        ),
        axis=-1,
    )
    # Counterclockwise quadrilaterals, radial index fastest.
    connectivity = np.asarray(
        [
            (
                row * radial_node_count + column,
                row * radial_node_count + column + 1,
                (row + 1) * radial_node_count + column + 1,
                (row + 1) * radial_node_count + column,
            )
            for row in range(axial_cell_count)
            for column in range(radial_cell_count)
        ],
        dtype=np.int32,
    )
    space = FiniteElementPlan(
        CellMesh(coordinates, (CellBlock(body, "quadrilateral", connectivity),)),
        FiniteElementFieldSpec("nodal", lagrange_element("quadrilateral", 1)),
    ).prepare()
    quadrature = ReferenceQuadrilateralRule(
        GaussLegendreRule(quadrature_order)
    ).materialize()
    points = np.asarray(quadrature.points)
    weights = np.asarray(quadrature.weights)
    points_per_cell = weights.shape[0]
    site_cells = np.repeat(np.arange(cell_count, dtype=np.int32), points_per_cell)
    site_points = np.tile(points, (cell_count, 1))
    value_route = prepare_finite_element_point_interpolation(
        space, "nodal", body, site_cells, site_points
    )
    sites = np.asarray(point_field_values(value_route, jnp.asarray(coordinates)))
    # Straight-sided rectangles: the reference-to-physical Jacobian is the
    # product of the axial and radial cell lengths.
    cell_area = np.repeat(
        (np.diff(axial_nodes)[:, None] * np.diff(radii)[None, :]).reshape(-1),
        points_per_cell,
    )
    site_weights = np.asarray(
        CoordinateMeasurePlan("axisymmetric-rz", radial_axis=0).weight(
            jnp.asarray(sites),
            jnp.asarray(np.tile(weights, cell_count) * cell_area),
            frame="reference",
        )
    )
    geometry = FuelBodyGeometry(
        nodes=jnp.asarray(coordinates),
        site_coordinates=jnp.asarray(sites),
        site_weight=jnp.asarray(site_weights),
        site_zone=jnp.asarray(
            np.repeat(
                np.arange(axial_cell_count, dtype=np.int32) // axial_cells_per_zone,
                radial_cell_count * points_per_cell,
            )
        ),
        site_inner_radius=jnp.asarray(
            np.tile(np.repeat(radii[:-1], points_per_cell), axial_cell_count)
        ),
        site_outer_radius=jnp.asarray(
            np.tile(np.repeat(radii[1:], points_per_cell), axial_cell_count)
        ),
        value_route=value_route,
        radial_route=prepare_finite_element_point_interpolation(
            space, "nodal", body, site_cells, site_points, derivative_axis=0
        ),
        axial_route=prepare_finite_element_point_interpolation(
            space, "nodal", body, site_cells, site_points, derivative_axis=1
        ),
        radial_node_count=radial_node_count,
        layer_count=zone_count,
        layered=False,
        geometry_id=canonical_fingerprint(
            {"kind": "fuel-rz-body", "nodes": coordinates, "order": quadrature_order}
        ),
    )
    line_rule = ReferenceIntervalRule(GaussLegendreRule(quadrature_order)).materialize()
    line_points = np.asarray(line_rule.points)[:, 0]
    line_weights = np.asarray(line_rule.weights)
    side_traces = tuple(
        _rz_side_trace(
            space,
            body,
            side,
            radius,
            radial_cell_count,
            axial_nodes,
            line_points,
            line_weights,
            zone_count,
            axial_cells_per_zone,
        )
        for side, radius in ((0, inner_radius), (1, outer_radius))
    )
    top_trace = _rz_top_trace(
        space,
        body,
        coordinates,
        radii,
        axial_cell_count,
        line_points,
        line_weights,
        zone_count,
    )
    return geometry, side_traces[0], side_traces[1], top_trace


def _rod_body_and_traces(
    body: str,
    inner_radius: float,
    outer_radius: float,
    radial_cell_count: int,
    zone_edges: NDArray[np.float64],
    /,
    *,
    layered: bool,
    quadrature_order: int,
    axial_cells_per_zone: int,
) -> tuple[
    FuelBodyGeometry, FuelTraceGeometry, FuelTraceGeometry, FuelTraceGeometry | None
]:
    """Body, inner and outer surface traces, and RZ top end face of one body.

    Host-only topology preparation. The layered reduction has no end face.
    """

    if layered:
        geometry, inner, outer = _layered_body_and_traces(
            body,
            inner_radius,
            outer_radius,
            radial_cell_count,
            zone_edges,
            quadrature_order,
        )
        return geometry, inner, outer, None
    return _rz_body_and_traces(
        body,
        inner_radius,
        outer_radius,
        radial_cell_count,
        zone_edges,
        quadrature_order,
        axial_cells_per_zone,
    )


def _coolant_trace_map(
    body: FuelBodyGeometry,
    trace: FuelTraceGeometry,
    nodes: NDArray[np.int32],
    /,
) -> SparseLinearMap:
    """Restrict the outer-trace interpolation to the outer-surface ``nodes``."""

    routes = np.asarray(trace.route.dof_routes, dtype=np.int32)
    weights = np.asarray(trace.route.weights, dtype=np.float64)
    surface_column = np.full((body.node_count,), -1, dtype=np.int32)
    surface_column[nodes] = np.arange(nodes.shape[0], dtype=np.int32)
    columns = surface_column[routes]
    retained = columns >= 0
    if np.any((~retained) & (weights != 0.0)):
        raise ValueError(
            "The outer FE trace has nonzero support outside its nodal boundary."
        )
    rows = np.broadcast_to(
        np.arange(routes.shape[0], dtype=np.int32)[:, None], routes.shape
    )
    relation = EdgeRelation(
        columns[retained],
        rows[retained],
        source_size=nodes.shape[0],
        target_size=routes.shape[0],
    )
    return SparseLinearMap(relation, weights[retained])


def fuel_rod_geometry(
    specification: FuelRodSpecification,
    zone_edges: ConvertibleToArray,
    /,
    *,
    layered: bool,
    fuel_cells: int,
    cladding_cells: int,
    quadrature_order: int,
    axial_cells_per_zone: int = 1,
) -> FuelRodGeometry:
    """Prepare the fuel and cladding geometry of one rod on axial ``zone_edges`` [m]."""

    if (
        fuel_cells < 2
        or cladding_cells < 2
        or quadrature_order < 1
        or axial_cells_per_zone < 1
    ):
        raise ValueError("Mesh counts must be positive; radial bodies require >=2 cells.")
    edges = np.asarray(zone_edges, dtype=np.float64)
    require_increasing_from_zero(edges, "zone_edges")
    fuel_mesh, _, fuel_interface, fuel_end = _rod_body_and_traces(
        "fuel",
        0.0,
        specification.pellet.radius,
        fuel_cells,
        edges,
        layered=layered,
        quadrature_order=quadrature_order,
        axial_cells_per_zone=axial_cells_per_zone,
    )
    cladding_mesh, cladding_interface, cladding_outer_surface, cladding_end = (
        _rod_body_and_traces(
            "cladding",
            specification.cladding.inner_radius,
            specification.cladding.outer_radius,
            cladding_cells,
            edges,
            layered=layered,
            quadrature_order=quadrature_order,
            axial_cells_per_zone=axial_cells_per_zone,
        )
    )
    zone_lengths = np.diff(edges)
    zone_indices = np.arange(zone_lengths.size, dtype=np.int32)
    # Last radial node of every axial node row of the cladding.
    coolant_nodes = (
        np.arange(
            cladding_mesh.node_count // cladding_mesh.radial_node_count, dtype=np.int32
        )
        * cladding_mesh.radial_node_count
        + cladding_mesh.radial_node_count
        - 1
    )
    coolant_flux_map = (
        None
        if layered
        else _coolant_trace_map(cladding_mesh, cladding_outer_surface, coolant_nodes)
    )
    coolant_node_weights = (
        cladding_outer_surface.weight
        if coolant_flux_map is None
        else coolant_flux_map.transpose_mv(cladding_outer_surface.weight)
    )
    return FuelRodGeometry(
        fuel=fuel_mesh,
        cladding=cladding_mesh,
        fuel_interface=fuel_interface,
        cladding_interface=cladding_interface,
        cladding_outer=cladding_outer_surface,
        layer_lengths=jnp.asarray(zone_lengths),
        interface_lengths=jnp.asarray(
            zone_lengths
            if layered
            else np.repeat(zone_lengths / axial_cells_per_zone, axial_cells_per_zone)
        ),
        interface_zone=jnp.asarray(
            zone_indices if layered else np.repeat(zone_indices, axial_cells_per_zone)
        ),
        fuel_axis_nodes=jnp.asarray(
            np.flatnonzero(np.asarray(fuel_mesh.nodes)[:, 0] == 0), dtype=jnp.int32
        ),
        axial_anchor_fuel=jnp.arange(0 if layered else fuel_cells + 1, dtype=jnp.int32),
        axial_anchor_cladding=jnp.arange(
            0 if layered else cladding_cells + 1, dtype=jnp.int32
        ),
        coolant_nodes=jnp.asarray(coolant_nodes),
        coolant_node_weights=coolant_node_weights,
        coolant_flux_map=coolant_flux_map,
        fuel_end=fuel_end,
        cladding_end=cladding_end,
        geometry_id=canonical_fingerprint(
            {
                "fuel": fuel_mesh.geometry_id,
                "cladding": cladding_mesh.geometry_id,
                "layered": layered,
                "coolant_flux_space": "nodal-cladding-outer-trace",
            }
        ),
        layered=layered,
    )


__all__ = [
    "FuelBodyGeometry",
    "FuelRodGeometry",
    "FuelTraceGeometry",
    "fuel_rod_geometry",
]
