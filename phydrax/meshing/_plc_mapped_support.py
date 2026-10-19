#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational source support of degree-one mixed coordinate-map images.

Association identities are inputs, never outputs of geometric searches. Positive
maps, global embedding and whole oriented PLC coverage establish whole-cell
region support; source-polynomial restrictions establish edge and face support.
All uncertain enclosures and exhausted resources fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import final

import jax
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_plc_geometry as _dependency_phydrax_discretization__exact_plc_geometry
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_coverage as _dependency_phydrax_geometry__mapped_coverage
import phydrax.geometry._planar_coverage as _dependency_phydrax_geometry__planar_coverage
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.meshing._association as _owner_phydrax_meshing__association
import phydrax.meshing._association_routes as association_routes
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._certification as _owner_phydrax_meshing__certification
import phydrax.meshing._certification_inputs as _owner_phydrax_meshing__certification_inputs
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._host_banks as _owner_phydrax_meshing__host_banks
import phydrax.meshing._lineage as _owner_phydrax_meshing__lineage
import phydrax.meshing._organization_records as organization_owner
import phydrax.meshing._result as result_owner
from phydrax.discretization._coordinate_identity import coordinate_scope_key
from phydrax.discretization._reference_cell_types import CellGeometryElement

from .._strict import StrictModule
from ..discretization import _coordinate_enclosure as algebra, CellMesh
from ..discretization._cell_complex import PolyhedralConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellBlock, PolyhedralBlock
from ..typing import checked
from ._support_work import spend_support_work


@final
class CellMap(StrictModule):
    kind: str
    vertices: tuple[int, ...]
    coordinates: tuple[algebra.Expression, ...] | None
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank
    root_cell_id: int
    root_vertex_ids: tuple[int, ...]


type _Point = tuple[algebra.ExactScalar, ...]


@final
class MappedCellPreparation(StrictModule):
    """Resolved coordinate routes and root-source ownership for one PLC carrier."""

    elements: tuple[CellGeometryElement, ...]
    routes: tuple[np.ndarray, ...]
    controls: algebra.CoordinateSourceBank
    carrier: np.ndarray
    global_vertices: np.ndarray
    record: (
        _dependency_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource
        | None
    )


@final
class MappedCellImages(StrictModule):
    """Coordinate-map records and the shared rational image of every carrier vertex."""

    cells: tuple[CellMap, ...]
    vertex_images: tuple[_Point, ...]
    degree_one_exactness_work_units: int


@final
class MappedBlockMaps(StrictModule):
    """Validated cell maps of a single coordinate route in block row order."""

    cells: tuple[CellMap, ...]
    degree_one_exactness_work_units: int


def _prepare_mapped_cell_ownership(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedCellPreparation:
    """Resolve source routes and periodic/restricted root ownership once."""
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "MappedPlcSupport.mesh must have topology and ambient dimension 3; got different dimensions"
        )
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    controls = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
    carrier = np.ascontiguousarray(
        jax.device_get(mesh.coordinates), dtype=np.float64
    ).view(np.uint64)
    identity_geometry = geometry
    record = identity_geometry.restriction_source
    while record is None and identity_geometry.periodic_source is not None:
        parent_geometry = identity_geometry.periodic_source.source_geometry
        match parent_geometry:
            case _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec():
                identity_geometry = parent_geometry
            case _:
                raise TypeError(
                    "MappedPlcSupport.periodic_source must contain CellGeometrySpec; got another geometry kind"
                )
        record = identity_geometry.restriction_source
    global_vertices = np.asarray(jax.device_get(mesh.vertex_global_ids), dtype=np.int64)
    host_routes = tuple(
        np.asarray(route, dtype=np.int64) for route in jax.device_get(routes)
    )
    return MappedCellPreparation(
        elements, host_routes, controls, carrier, global_vertices, record
    )


def _mapped_cell_image(
    block: CellBlock,
    element: CellGeometryElement,
    local: algebra.CoordinateSourceBank,
    vertices: np.ndarray,
    cell_row: int,
    parent_ids: np.ndarray,
    parent_vertices: np.ndarray | None,
    prepared: MappedCellPreparation,
    images: list[_Point | None],
    *,
    prepare_expressions: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellMap:
    """Validate shared corner images and retain this cell's root-source ancestry."""
    ledger, carrier, global_vertices = budget, prepared.carrier, prepared.global_vertices
    # Sparse source-basis corner weights evaluate the whole represented
    # map chain, including rational/pyramid charts, without forming
    # unused interior physical polynomials.
    with ledger.temporary_scope():
        polynomial, corners = algebra.coordinate_map_images(
            element, local, include_expressions=prepare_expressions, budget=budget
        )
        if prepare_expressions and polynomial is None:
            raise ValueError(
                "MappedPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
            )
    if corners is None:
        raise ValueError(
            "MappedPlcSupport.coordinates must resolve corner images; got unresolved corners"
        )
    ledger.reserve(sum(len(image) for image in corners))
    for vertex, image in zip(vertices.tolist(), corners, strict=True):
        known = images[vertex]
        if known is None:
            if not np.array_equal(
                algebra.rounded_point(image).view(np.uint64), carrier[vertex]
            ):
                raise ValueError(
                    "MappedPlcSupport.carrier must equal correctly rounded coordinate-map images; got different corner bits"
                )
            images[vertex] = image
        elif known != image:
            raise ValueError(
                "MappedPlcSupport.coordinates must agree on shared corner images; got unequal rational points"
            )
    roots = (
        tuple(int(identifier) for identifier in global_vertices[vertices])
        if parent_vertices is None
        else tuple(
            int(identifier) for identifier in parent_vertices[cell_row] if identifier >= 0
        )
    )
    return CellMap(
        block.cell_kind,
        tuple(vertices.tolist()),
        polynomial,
        element,
        local,
        int(parent_ids[cell_row]),
        roots,
    )


def _require_degree_one_image(
    cell_kind: str,
    element: CellGeometryElement,
    local: algebra.CoordinateSourceBank,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> int:
    """Prove the full represented map equals its degree-one corner interpolant."""
    started = budget.work_units
    coordinates, corners = algebra.coordinate_map_images(element, local, budget=budget)
    if coordinates is None or corners is None:
        raise ValueError(
            "MappedPlcSupport.coordinates must resolve rational source expressions; "
            "got unresolved higher-order source expressions"
        )
    reference = algebra.coordinate_expressions(
        budget.degree_one_coordinate_element(cell_kind),
        corners,
        budget=budget,
    )
    if (
        reference is None
        or len(coordinates) != len(reference)
        or not all(
            algebra.expression_equal(actual, expected, budget=budget)
            for actual, expected in zip(coordinates, reference, strict=True)
        )
    ):
        raise ValueError(
            "MappedPlcSupport.degree_one_exactness must prove equality to the "
            "degree-one corner interpolant as full rational expressions; "
            "got a nonlinear source map"
        )
    return budget.work_units - started


def _mapped_block_maps(
    block: CellBlock | PolyhedralBlock,
    element: CellGeometryElement,
    route: np.ndarray,
    prepared: MappedCellPreparation,
    images: list[_Point | None],
    local_controls: dict[tuple[int, ...], algebra.CoordinateSourceBank],
    *,
    prepare_expressions: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedBlockMaps:
    """Validate a degree-one map block and share its routed source controls."""
    ledger, controls, record = budget, prepared.controls, prepared.record
    element = _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
        element, "Mapped PLC support"
    )
    match block:
        case CellBlock(cell_kind=("tetrahedron" | "hexahedron" | "prism" | "pyramid")):
            pass
        case _:
            raise ValueError(
                "MappedPlcSupport.elements must use supported mixed volume families; got an unsupported element"
            )
    root = element
    while isinstance(
        root,
        _dependency_phydrax_discretization__cell_geometry.RestrictedCellGeometryElement,
    ):
        root = root.source_element
    local_values: list[algebra.CoordinateSourceBank] = []
    for row in np.asarray(route, dtype=np.int64):
        # Tuple slots, Python int64-sized integers and the route index.
        ledger.reserve(row.size, 128 + 40 * row.size)
        key = tuple(row.tolist())
        local = local_controls.get(key)
        if local is None:
            local = tuple(controls[index] for index in key)
            local_controls[key] = local
        local_values.append(local)
    parent_ids = (
        np.asarray(block.global_ids, dtype=np.int64)
        if record is None
        else np.asarray(record.block_parent_cell_ids[block.name], dtype=np.int64)
    )
    parent_vertices = (
        None
        if record is None
        else np.asarray(record.block_parent_vertex_ids[block.name], dtype=np.int64)
    )
    cells: list[CellMap] = []
    exactness_work = 0
    for cell_row, (local, vertices) in enumerate(
        zip(local_values, np.asarray(block.vertices, dtype=np.int64), strict=True)
    ):
        vertices = np.asarray(vertices, dtype=np.int64).reshape(-1)
        if element.degree != 1 or root.degree != 1:
            exactness_work += _require_degree_one_image(
                block.cell_kind, element, local, budget=budget
            )
        cells.append(
            _mapped_cell_image(
                block,
                element,
                local,
                vertices,
                cell_row,
                parent_ids,
                parent_vertices,
                prepared,
                images,
                prepare_expressions=prepare_expressions,
                budget=budget,
            )
        )
    return MappedBlockMaps(tuple(cells), exactness_work)


def _cell_maps(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    *,
    prepare_expressions: bool = True,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedCellImages:
    """Prepare rational map records and require agreement at every shared corner."""
    prepared = _prepare_mapped_cell_ownership(mesh, geometry, budget=budget)
    images: list[_Point | None] = [None] * prepared.carrier.shape[0]
    cells: list[CellMap] = []
    exactness_work = 0
    local_controls: dict[tuple[int, ...], algebra.CoordinateSourceBank] = {}
    for block, element, route in zip(
        mesh.blocks, prepared.elements, prepared.routes, strict=True
    ):
        mapped = _mapped_block_maps(
            block,
            element,
            route,
            prepared,
            images,
            local_controls,
            prepare_expressions=prepare_expressions,
            budget=budget,
        )
        cells.extend(mapped.cells)
        exactness_work += mapped.degree_one_exactness_work_units
    exact: list[_Point] = []
    for image in images:
        if image is None:
            raise ValueError(
                "MappedPlcSupport.vertices must be mapped cell corners; got an unused carrier vertex"
            )
        exact.append(image)
    return MappedCellImages(tuple(cells), tuple(exact), exactness_work)


def _exact_affine_plc_validity(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    cells: tuple[CellMap, ...],
    vertex_images: tuple[_Point, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> (
    _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None
):
    """Certify constant rational tetrahedron Jacobians without polynomial materialization."""
    match geometry.exact_source:
        case (
            _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource()
            | _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometryConvexSource()
        ):
            if any(cell.kind != "tetrahedron" for cell in cells):
                return None
        case _:
            return None
    ledger = budget
    policy = (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy()
    )
    floor = Fraction(policy.relative_determinant_floor)
    status = np.empty((len(cells),), dtype=np.int32)
    lower = np.empty((len(cells),), dtype=np.float64)
    upper = np.empty((len(cells),), dtype=np.float64)
    prepared: dict[
        tuple[tuple[algebra.ExactScalar, ...], ...],
        tuple[algebra.ExactScalar, algebra.ExactScalar],
    ] = {}
    for row, cell in enumerate(cells):
        corners = tuple(vertex_images[vertex] for vertex in cell.vertices)
        ledger.reserve(sum(len(point) for point in corners) + 9)
        columns = tuple(
            tuple(corners[column + 1][axis] - corners[0][axis] for column in range(3))
            for axis in range(3)
        )
        values = prepared.get(columns)
        if values is None:
            bits = max(
                algebra._scalar_bits(value) for column in columns for value in column
            )
            _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                34, 16, 0, 3 * bits + 4, budget=budget
            )
            determinant = (
                columns[0][0]
                * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
                - columns[0][1]
                * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
                + columns[0][2]
                * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
            )
            squared_scale = math.prod(
                (
                    sum(
                        (columns[axis][column] ** 2 for axis in range(3)),
                        Fraction(0),
                    )
                    for column in range(3)
                ),
                start=Fraction(1),
            )
            values = determinant, squared_scale
            prepared[columns] = values
        determinant, squared_scale = values
        status[row] = (
            _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
            if determinant > 0
            and determinant * determinant > floor * floor * squared_scale
            else _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.INVALID
        )
        lower[row] = algebra._scalar_endpoint(determinant, upper=False, budget=budget)
        upper[row] = algebra._scalar_endpoint(determinant, upper=True, budget=budget)
    offsets = [0]
    for block in mesh.blocks:
        offsets.append(offsets[-1] + block.vertices.shape[0])
    return _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate(
        status,
        lower,
        upper,
        np.zeros((len(cells),), dtype=np.int32),
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=tuple(offsets),
        unsupported_block_names=(),
        unresolved_reasons=(),
        geometry_id=_dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            geometry
        ),
        geometry_layout_id=geometry.geometry_layout_id,
        topology_id=mesh.topology_id,
        policy_id=policy.policy_id,
    )


def _cell_coordinates(
    cell: CellMap,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    if cell.coordinates is not None:
        return cell.coordinates
    coordinates = algebra.coordinate_expressions(
        cell.element, cell.controls, budget=budget
    )
    if coordinates is None:
        raise ValueError(
            "MappedPlcSupport.coordinates must resolve source expressions; got unresolved expressions"
        )
    return coordinates


@checked
def plc_mapped_carrier(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    """Validate represented Q1/restricted expressions and their rounded carrier corners."""
    _cell_maps(mesh, geometry, budget=budget)
    return True


@checked
def plc_mapped_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Return entity-set ordered vertex rows, padding mixed faces with -1."""
    if dimension == 0:
        return np.arange(mesh.entity_set(0).count, dtype=np.int64)[:, None]
    connectivity = mesh.connectivity
    match dimension, connectivity:
        case (
            1,
            TetrahedralConnectivity()
            | HexahedralConnectivity()
            | PolyhedralConnectivity(),
        ):
            return np.asarray(connectivity.edges, dtype=np.int64)
        case 2, TetrahedralConnectivity() | HexahedralConnectivity():
            return np.asarray(connectivity.faces, dtype=np.int64)
        case 2, PolyhedralConnectivity():
            offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            rows = np.full(
                (mesh.entity_set(2).count, connectivity.maximum_face_arity),
                -1,
                dtype=np.int64,
            )
            for row, (start, stop) in enumerate(
                zip(offsets[:-1], offsets[1:], strict=True)
            ):
                rows[row, : stop - start] = values[start:stop]
            return rows
        case 3, _:
            ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
            positions = {identifier: row for row, identifier in enumerate(ids.tolist())}
            rows = np.full(
                (ids.size, max(block.vertices.shape[1] for block in mesh.blocks)),
                -1,
                dtype=np.int64,
            )
            for block in mesh.blocks:
                for identifier, vertices in zip(
                    np.asarray(block.global_ids).tolist(),
                    np.asarray(block.vertices, dtype=np.int64),
                    strict=True,
                ):
                    rows[positions[identifier], : vertices.size] = vertices
            return rows
        case _:
            raise ValueError(
                "MappedPlcSupport.mesh must expose fixed mixed-cell incidence; got an unsupported connectivity"
            )


@final
class MappedReferenceCharts(StrictModule):
    """Reference charts of a represented mesh entity in incidence order."""

    charts: tuple[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...]


def _mapped_face_charts(
    cell: CellMap,
    coordinates: tuple[algebra.Expression, ...],
    entity: tuple[int, ...],
    ordered: tuple[int, ...],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedReferenceCharts:
    """Prepare face charts with the entity's retained cyclic orientation."""
    if cell.kind == "pyramid":
        patches = _dependency_phydrax_geometry__mapped_coverage.face_charts(
            coordinates, cell.kind, entity, budget=budget
        )
        position = entity.index(ordered[0])
        forward = entity[(position + 1) % len(entity)] == ordered[1]
        if not forward:
            arguments = tuple(reversed(algebra.axes(2, budget=budget)))
            patches = tuple(
                (
                    (
                        tuple(
                            (
                                algebra.expression_compose(
                                    value, arguments, budget=budget
                                )
                                for value in chart
                            )
                        ),
                        domain,
                    )
                    for chart, domain in patches
                )
            )
        return MappedReferenceCharts(patches)
    else:
        return MappedReferenceCharts(
            _dependency_phydrax_geometry__mapped_coverage.face_charts(
                coordinates, cell.kind, ordered, budget=budget
            )
        )


def _mapped_edge_chart(
    cell: CellMap,
    coordinates: tuple[algebra.Expression, ...],
    topology_vertices: tuple[tuple[float, ...], ...],
    ordered: tuple[int, ...],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedReferenceCharts:
    """Restrict a coordinate map to an oriented reference edge."""
    corners = np.asarray(topology_vertices, dtype=np.float64)[
        np.asarray(ordered, dtype=np.int64)
    ]
    if cell.kind == "pyramid" and 4 in ordered:
        apex = ordered.index(4)
        corners[apex, :2] = corners[1 - apex, :2]
    arguments = algebra.affine_arguments(
        corners[0], (corners[1] - corners[0])[:, None], budget=budget
    )
    return MappedReferenceCharts(
        (
            (
                tuple(
                    (
                        algebra.expression_compose(value, arguments, budget=budget)
                        for value in coordinates
                    )
                ),
                "box",
            ),
        )
    )


@checked
def mapped_reference_charts(
    cells: tuple[CellMap, ...],
    rows: np.ndarray,
    dimension: int,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...]:
    vertices = tuple(rows[rows >= 0].tolist())
    result: list[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain]] = []
    for cell in cells:
        topology = reference_cell_topology(cell.kind)
        entities = topology.entities[dimension]
        for entity in entities:
            if {cell.vertices[index] for index in entity} != set(vertices):
                continue
            ordered = tuple(cell.vertices.index(vertex) for vertex in vertices)
            coordinates = _cell_coordinates(cell, budget=budget)
            charts = (
                _mapped_face_charts(cell, coordinates, entity, ordered, budget=budget)
                if dimension == 2
                else _mapped_edge_chart(
                    cell, coordinates, topology.vertices, ordered, budget=budget
                )
            )
            result.extend(charts.charts)
    if not result:
        raise ValueError(
            "MappedPlcSupport.entity must have coordinate-map incidence charts; got no chart"
        )
    # Equal normalized coefficients prove equal entire charts. Neighbor incidence
    # remains checked; only mathematically identical image proofs are reused.
    unique: list[tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain]] = []
    for chart in result:
        if not any(
            len(chart[0]) == len(prior[0])
            and all(
                algebra.expression_equal(a, b, budget=budget)
                for a, b in zip(chart[0], prior[0], strict=True)
            )
            and chart[1] == prior[1]
            for prior in unique
        ):
            unique.append(chart)
    return tuple(unique)


def _nonnegative(
    value: algebra.Expression,
    dimension: int,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    work: list[int],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    used = _dependency_phydrax_geometry__mapped_coverage.SubdivisionLedger()
    result = _dependency_phydrax_geometry__mapped_coverage.nonnegative(
        value,
        "box",
        dimension,
        limits.maximum_bernstein_nodes,
        limits.maximum_subdivision_depth,
        min(limits.maximum_subdivision_pieces, work[0]),
        used,
        budget=budget,
    )
    work[0] = spend_support_work(
        work[0],
        used.pieces,
        "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
    )
    return result


@checked
def edge_support(
    chart: tuple[algebra.Expression, ...],
    edge: np.ndarray,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    work: list[int],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    endpoints = planar_primitives_owner.rational_points(edge, budget=budget)
    planar_primitives_owner.reserve_fraction_work(
        endpoints, len(endpoints[0]), len(endpoints[0]), 2, budget=budget
    )
    direction = tuple(b - a for a, b in zip(*endpoints, strict=True))
    axis = next((index for index, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError(
            "MappedPlcSupport.source_edge must have distinct endpoints; got a collapsed edge"
        )
    parameter = algebra.expression_scale(
        algebra.expression_add(
            chart[axis],
            algebra.constant(-endpoints[0][axis], 1, budget=budget),
            budget=budget,
        ),
        1 / direction[axis],
        budget=budget,
    )
    for coordinate, origin, delta in zip(chart, endpoints[0], direction, strict=True):
        if algebra.expression_add(
            coordinate,
            algebra.expression_scale(
                algebra.expression_add(
                    algebra.constant(origin, 1, budget=budget),
                    algebra.expression_scale(parameter, delta, budget=budget),
                    budget=budget,
                ),
                -1,
                budget=budget,
            ),
            budget=budget,
        ):
            return False, 0
    if not _nonnegative(parameter, 1, limits, work, budget=budget) or not _nonnegative(
        algebra.expression_add(
            algebra.constant(1, 1, budget=budget),
            algebra.expression_scale(parameter, -1, budget=budget),
            budget=budget,
        ),
        1,
        limits,
        work,
        budget=budget,
    ):
        return False, 0
    derivative = algebra.expression_derivative(parameter, 0, budget=budget)
    start = algebra.expression_evaluate(parameter, (Fraction(0),), budget=budget)
    end = algebra.expression_evaluate(parameter, (Fraction(1),), budget=budget)
    planar_primitives_owner.reserve_fraction_work(((start, end),), 1, 1, 2, budget=budget)
    displacement = end - start
    orientation = 1 if displacement > 0 else -1 if displacement < 0 else 0
    return orientation != 0 and _nonnegative(
        algebra.expression_scale(derivative, orientation, budget=budget),
        1,
        limits,
        work,
        budget=budget,
    ), orientation


def _exact_segment_support(
    segment: tuple[_Point, _Point],
    endpoints: tuple[_Point, _Point],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    planar_primitives_owner.reserve_fraction_work(
        (*segment, *endpoints),
        4 * len(segment[0]),
        2 + 2 * len(segment[0]),
        8,
        budget=budget,
    )
    direction = tuple(b - a for a, b in zip(*endpoints, strict=True))
    axis = next((index for index, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError(
            "MappedPlcSupport.source_edge must have distinct endpoints; got a collapsed edge"
        )
    parameters = tuple(
        (point[axis] - endpoints[0][axis]) / direction[axis] for point in segment
    )
    supported = all(
        0 <= parameter <= 1
        and all(
            value == origin + parameter * delta
            for value, origin, delta in zip(
                point,
                endpoints[0],
                direction,
                strict=True,
            )
        )
        for point, parameter in zip(segment, parameters, strict=True)
    )
    displacement = parameters[1] - parameters[0]
    orientation = 1 if displacement > 0 else -1 if displacement < 0 else 0
    return supported and orientation != 0, orientation


def _segment_in_union(
    segment: tuple[_Point, ...],
    triangles: tuple[tuple[_Point, ...], ...],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    if len(segment) == 1:
        return any(
            all(
                (
                    1
                    if planar_primitives_owner.signed_measure(triangle, budget=budget) > 0
                    else -1
                )
                * planar_primitives_owner.turn(a, b, segment[0], budget=budget)
                >= 0
                for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True)
            )
            for triangle in triangles
        )
    intervals: list[tuple[algebra.ExactScalar, algebra.ExactScalar]] = []
    for triangle in triangles:
        lower, upper = Fraction(0), Fraction(1)
        sign = (
            1
            if planar_primitives_owner.signed_measure(triangle, budget=budget) > 0
            else -1
        )
        for a, b in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
            start = sign * planar_primitives_owner.turn(a, b, segment[0], budget=budget)
            end = sign * planar_primitives_owner.turn(a, b, segment[-1], budget=budget)
            planar_primitives_owner.reserve_fraction_work(
                ((start, end),), 1, 1, 2, budget=budget
            )
            slope = end - start
            if slope > 0:
                planar_primitives_owner.reserve_fraction_work(
                    ((start, slope),), 2, 1, 2, budget=budget
                )
                lower = max(lower, -start / slope)
            elif slope < 0:
                planar_primitives_owner.reserve_fraction_work(
                    ((start, slope),), 2, 1, 2, budget=budget
                )
                upper = min(upper, -start / slope)
            elif start < 0:
                lower, upper = Fraction(1), Fraction(0)
        if lower <= upper:
            intervals.append((lower, upper))
    covered = Fraction(0)
    for lower, upper in sorted(intervals):
        if lower > covered:
            return False
        covered = max(covered, upper)
    return covered >= 1


def _edge_on_facet(
    chart: tuple[algebra.Expression, ...],
    group: _dependency_phydrax_geometry__planar_coverage.SourceGroup,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    work: list[int],
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bool:
    plane = algebra.expression_add(
        algebra.expression_sum(
            tuple(
                algebra.expression_scale(value, weight, budget=budget)
                for value, weight in zip(chart, group.plane[:-1], strict=True)
            ),
            budget=budget,
        ),
        algebra.constant(group.plane[-1], 1, budget=budget),
        budget=budget,
    )
    if plane:
        return False
    pending = [(chart, 0)]
    while pending:
        current, depth = pending.pop()
        work[0] = spend_support_work(
            work[0],
            1 + len(group.simplices),
            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        controls = algebra.expression_control_points(
            tuple(current[axis] for axis in group.axes),
            "box",
            1,
            limits.maximum_bernstein_nodes,
            budget=budget,
        )
        if controls is None:
            return False
        hull = planar_primitives_owner.convex_hull(controls, budget=budget)
        area = abs(planar_primitives_owner.signed_measure(hull, budget=budget))
        contained = (
            sum(
                (
                    planar_primitives_owner.intersection(hull, triangle, budget=budget)
                    for triangle in group.simplices
                ),
                Fraction(0),
            )
            == area
            if area > 0
            else _segment_in_union(hull, group.simplices, budget=budget)
        )
        if contained:
            continue
        if depth >= limits.maximum_subdivision_depth:
            return False
        pending.extend(
            (
                tuple(
                    algebra.expression_compose(value, child, budget=budget)
                    for value in current
                ),
                depth + 1,
            )
            for child in _dependency_phydrax_geometry__mapped_coverage.subdivision_children(
                "box", 1, budget=budget
            )
        )
    return True


@final
class SupportVerdict(StrictModule):
    """Whole-entity support and its consistently oriented source stratum."""

    supported: bool
    orientation: int


@final
class MappedEntityQuery(StrictModule):
    """Incidence and rational corners shared by this entity's source proof."""

    parents: tuple[int, ...]
    entity_vertices: np.ndarray
    exact_points: tuple[_Point, ...]


@final
class MappedSourceQuery(StrictModule):
    """Explicit source stratum and its shared query allowance."""

    dimension: int
    source_index: int
    authority: np.ndarray
    edge_vertices: np.ndarray
    triangle_vertices: np.ndarray
    triangle_facets: np.ndarray
    work: list[int]


@dataclass(frozen=True)
class MappedPlcSupport:
    """Current geometry-owner evidence and its rational source-expression charts."""

    mesh: CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    domain: domain_coverage_owner.PiecewiseLinearDomain
    cells: tuple[CellMap, ...]
    cell_regions: np.ndarray
    embedding: global_embedding_owner.GlobalEmbeddingCertificate
    coverage: domain_coverage_owner.DomainCoverageCertificate
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    entity_vertices: tuple[np.ndarray, ...]
    # Per dimension, immutable CSR (offsets, cell rows) of incident cells.
    entity_parents: tuple[tuple[np.ndarray, np.ndarray], ...]
    boundary_entities: tuple[np.ndarray, ...]
    prepared: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence
    vertex_images: tuple[_Point, ...]
    degree_one_exactness_work_units: int
    # One rational-expression ledger under the source request limits owns every
    # support query and retained source group of this preparation; it is never
    # renewed per query.
    ledger: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget = field(
        repr=False, compare=False
    )
    work_start: int = field(repr=False, compare=False)
    _facet_groups: dict[
        tuple[int, int, str, bytes, bytes],
        tuple[_dependency_phydrax_geometry__planar_coverage.SourceGroup, ...],
    ] = field(default_factory=dict, repr=False, compare=False)
    _source_edges: dict[tuple[int, int, str, bytes], tuple[_Point, _Point]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _source_vertex_banks: dict[int, tuple[np.ndarray, tuple[_Point, ...]]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _edge_midpoints: dict[int, _Point] = field(
        default_factory=dict, repr=False, compare=False
    )

    def edge_midpoint(self, row: int, /) -> _Point:
        """Retain the half-reference image shared by every incident edge chart."""
        if row < 0 or row >= self.entity_vertices[1].shape[0]:
            raise ValueError(
                "MappedPlcSupport.edge_midpoint must name a declared edge row; "
                f"got row {row} for {self.entity_vertices[1].shape[0]} edges"
            )
        cached = self._edge_midpoints.get(row)
        if cached is not None:
            return cached
        with self.ledger.activate(), self.ledger.temporary_scope():
            charts = mapped_reference_charts(
                tuple(self.cells[parent] for parent in self.parents(1, row)),
                self.entity_vertices[1][row],
                1,
                budget=self.ledger,
            )
            self.ledger.reserve_geometry_queries(len(charts))
            points = tuple(
                tuple(
                    algebra.expression_evaluate(
                        value, (Fraction(1, 2),), budget=self.ledger
                    )
                    for value in coordinates
                )
                for coordinates, _ in charts
            )
            if not points or any(point != points[0] for point in points[1:]):
                raise ValueError(
                    "MappedPlcSupport.edge_midpoint must have incident-chart consensus; "
                    f"got {len(points)} incident images without agreement"
                )
            result = points[0]
            self.ledger.retain_basis((row, result))
            self._edge_midpoints[row] = result
            return result

    def parents(self, dimension: int, row: int, /) -> tuple[int, ...]:
        """Incident cell rows of one entity, in incidence order."""
        if dimension == 3:
            return (row,)
        offsets, values = self.entity_parents[dimension]
        return tuple(values[offsets[row] : offsets[row + 1]].tolist())

    def _source_points(self, vertices: np.ndarray, /) -> tuple[_Point, ...]:
        budget = self.ledger
        if vertices.flags.writeable:
            return planar_primitives_owner.rational_points(vertices, budget=budget)
        identifier = id(vertices)
        cached = self._source_vertex_banks.get(identifier)
        if cached is not None and cached[0] is vertices:
            return cached[1]
        exact = planar_primitives_owner.rational_points(vertices, budget=budget)
        self._source_vertex_banks[identifier] = (vertices, exact)
        self.ledger.retain_basis(exact)
        return exact

    def _source_edge(
        self,
        source_index: int,
        edge: np.ndarray,
        vertices: np.ndarray,
        /,
    ) -> tuple[_Point, _Point]:
        exact = self._source_points(vertices)
        self.ledger.reserve(edge.size, 256 + edge.nbytes)
        key = (
            source_index,
            id(exact),
            edge.dtype.str,
            edge.tobytes(),
        )
        cached = self._source_edges.get(key)
        if cached is not None:
            return cached
        result = exact[int(edge[0])], exact[int(edge[1])]
        self._source_edges[key] = result
        self.ledger.retain_basis(key)
        self.ledger.retain_basis(result)
        return result

    def _source_simplex(
        self,
        triangle: np.ndarray,
        vertices: np.ndarray,
        /,
    ) -> _dependency_phydrax_geometry__planar_coverage.AffineSourceSimplex:
        budget = self.ledger
        exact = self._source_points(vertices)
        self.ledger.reserve(triangle.size)
        points = tuple(exact[int(vertex)] for vertex in triangle)
        return _dependency_phydrax_geometry__planar_coverage.prepare_affine_exact_source_simplex(
            points, budget=budget
        )

    def _source_facet_groups(
        self,
        source_index: int,
        triangles: np.ndarray,
        vertices: np.ndarray,
        work: list[int],
    ) -> tuple[_dependency_phydrax_geometry__planar_coverage.SourceGroup, ...]:
        # The explicit source index selects identity. Full source ownership-row and
        # coordinate bytes bind only reuse of its already-proved planar union;
        # mutation of either bank cannot reuse a previous source proof.
        budget = self.ledger
        points = vertices[triangles]
        # The key's byte copies are this query's workspace, charged before use.
        self.ledger.reserve(
            triangles.size + points.size, 256 + triangles.nbytes + points.nbytes
        )
        key = (
            source_index,
            triangles.shape[0],
            triangles.dtype.str,
            triangles.tobytes(),
            points.tobytes(),
        )
        cached = self._facet_groups.get(key)
        if cached is not None:
            return cached
        work[0] = spend_support_work(
            work[0],
            1,
            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        if len(self._facet_groups) >= self.limits.maximum_candidate_pairs:
            raise ValueError(
                "MappedPlcSupport.facet_groups must respect maximum_candidate_pairs; got capacity exhaustion"
            )
        groups, overlaps, used, exceeded = (
            _dependency_phydrax_geometry__planar_coverage.source_groups(
                vertices,
                triangles,
                np.tile(np.asarray(((0, -1),), dtype=np.int64), (triangles.shape[0], 1)),
                min(self.limits.maximum_candidate_pairs, work[0]),
                budget=budget,
            )
        )
        work[0] = spend_support_work(
            work[0],
            used,
            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        if overlaps or exceeded:
            raise ValueError(
                "MappedPlcSupport.source_facet must form a nonoverlapping bounded union; got overlap or unresolved work"
            )
        self._facet_groups[key] = groups
        self.ledger.retain_basis(key)
        self.ledger.retain_basis(groups)
        return groups

    def entity_support(
        self,
        dimension: int,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> tuple[bool, int]:
        """Prove support of an explicitly identified stratum; return orientation.

        rational chart algebra is charged to this preparation's ledger; a refusal
        raises ``CoordinateEnclosureResourceError`` with its requested and
        completed work.
        """
        with (
            self.ledger.activate(),
            self.ledger.bound_stage(
                self.prepared.request.limits.maximum_work_units,
                self.prepared.request.limits.maximum_scratch_bytes,
                starting_work_units=self.work_start,
            ),
            self.ledger.temporary_scope(),
        ):
            verdict = self._entity_support(
                dimension,
                row,
                source_dimension,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            )
            return verdict.supported, verdict.orientation

    def _entity_support(
        self,
        dimension: int,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> SupportVerdict:
        budget = self.ledger
        if (
            dimension not in range(4)
            or row < 0
            or row >= self.mesh.entity_set(dimension).count
        ):
            raise ValueError(
                "MappedPlcSupport.entity must index a declared mesh stratum; got an undeclared entity"
            )
        if source_dimension not in range(4) or source_index < 0:
            raise ValueError(
                "MappedPlcSupport.source_stratum must have dimension 0 through 3 and nonnegative index; got invalid indices"
            )
        authority = self.domain.vertices if source_vertices is None else source_vertices
        if source_dimension == 1 and source_index >= edge_vertices.shape[0]:
            raise ValueError(
                "MappedPlcSupport.source_edge must index the declared edge bank; got an undeclared edge"
            )
        parents = self.parents(dimension, row)
        if source_dimension == 3:
            if dimension < 3 and self.boundary_entities[dimension][row]:
                return SupportVerdict(False, 0)
            work[0] = spend_support_work(
                work[0],
                len(parents),
                "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
            )
            budget.reserve_geometry_queries(len(parents))
            return SupportVerdict(
                bool(parents)
                and bool(
                    np.all(
                        self.cell_regions[np.asarray(parents, dtype=np.int64)]
                        == source_index
                    )
                ),
                0,
            )
        if dimension not in (1, 2) or source_dimension < dimension:
            return SupportVerdict(False, 0)
        entity_vertices = self.entity_vertices[dimension][row]
        exact_points = (
            tuple(
                (
                    self.vertex_images[int(vertex)]
                    for vertex in entity_vertices
                    if vertex >= 0
                )
            )
            if all((self.cells[parent].kind == "tetrahedron" for parent in parents))
            else ()
        )
        query = MappedEntityQuery(parents, entity_vertices, exact_points)
        source = MappedSourceQuery(
            dimension,
            source_index,
            authority,
            edge_vertices,
            triangle_vertices,
            triangle_facets,
            work,
        )
        match source_dimension:
            case 1:
                return self._edge_entity_support(query, source)
            case 2:
                return self._facet_entity_support(query, source)
            case _:
                return SupportVerdict(False, 0)

    def _edge_entity_support(
        self, query: MappedEntityQuery, source: MappedSourceQuery, /
    ) -> SupportVerdict:
        """Prove affine or mapped support on an explicitly identified source edge."""
        budget = self.ledger
        dimension, source_index, authority, work = (
            source.dimension,
            source.source_index,
            source.authority,
            source.work,
        )
        parents, entity_vertices, exact_points = (
            query.parents,
            query.entity_vertices,
            query.exact_points,
        )
        edge_vertices = source.edge_vertices
        if dimension == 1 and len(exact_points) == 2:
            work[0] = spend_support_work(
                work[0],
                1,
                "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
            )
            budget.reserve_geometry_queries(1)
            supported, orientation = _exact_segment_support(
                (exact_points[0], exact_points[1]),
                self._source_edge(source_index, edge_vertices[source_index], authority),
                budget=budget,
            )
            return SupportVerdict(supported, orientation)
        charts = mapped_reference_charts(
            tuple((self.cells[parent] for parent in parents)),
            entity_vertices,
            dimension,
            budget=budget,
        )
        budget.reserve_geometry_queries(len(charts))
        outcomes = tuple(
            (
                edge_support(
                    chart,
                    authority[edge_vertices[source_index]],
                    self.limits,
                    work,
                    budget=budget,
                )
                for chart, _ in charts
            )
        )
        signs = {orientation for supported, orientation in outcomes if supported}
        return SupportVerdict(
            all((supported for supported, _ in outcomes)) and len(signs) == 1,
            next(iter(signs)) if len(signs) == 1 else 0,
        )

    def _facet_entity_support(
        self, query: MappedEntityQuery, source: MappedSourceQuery, /
    ) -> SupportVerdict:
        """Use rational affine containment or prepare the represented facet union."""
        budget = self.ledger
        dimension, source_index, authority, work = (
            source.dimension,
            source.source_index,
            source.authority,
            source.work,
        )
        parents, entity_vertices, exact_points = (
            query.parents,
            query.entity_vertices,
            query.exact_points,
        )
        triangle_vertices, triangle_facets = (
            source.triangle_vertices,
            source.triangle_facets,
        )
        triangles = triangle_vertices[triangle_facets == source_index]
        if triangles.shape[0] == 0:
            return SupportVerdict(False, 0)
        if triangles.shape[0] == 1 and len(exact_points) == dimension + 1:
            work[0] = spend_support_work(
                work[0],
                2,
                "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
            )
            budget.reserve_geometry_queries(1)
            source_simplex = self._source_simplex(triangles[0], authority)
            if dimension == 1:
                return SupportVerdict(
                    _dependency_phydrax_geometry__planar_coverage.affine_segment_containment(
                        (exact_points[0], exact_points[1]), source_simplex, budget=budget
                    ),
                    0,
                )
            direct = _dependency_phydrax_geometry__planar_coverage.affine_simplex_corner_containment(
                (exact_points[0], exact_points[1], exact_points[2]),
                source_simplex,
                budget=budget,
            )
            return SupportVerdict(direct is not None, 0 if direct is None else direct[1])
        charts = mapped_reference_charts(
            tuple((self.cells[parent] for parent in parents)),
            entity_vertices,
            dimension,
            budget=budget,
        )
        groups = self._source_facet_groups(source_index, triangles, authority, work)
        return self._facet_chart_support(dimension, charts, groups, work)

    def _facet_chart_support(
        self,
        dimension: int,
        charts: tuple[
            tuple[tuple[algebra.Expression, ...], algebra.BernsteinDomain], ...
        ],
        groups: tuple[_dependency_phydrax_geometry__planar_coverage.SourceGroup, ...],
        work: list[int],
        /,
    ) -> SupportVerdict:
        """Prove every incident chart is covered and agrees in source orientation."""
        budget = self.ledger
        orientations: set[int] = set()
        for chart, chart_domain in charts:
            supported = False
            for group in groups:
                if dimension == 1:
                    budget.reserve_geometry_queries(1)
                    supported = _edge_on_facet(
                        chart, group, self.limits, work, budget=budget
                    )
                else:
                    for own, other in (group.regions, group.regions[::-1]):
                        budget.reserve_geometry_queries(1)
                        used_work = _dependency_phydrax_geometry__mapped_coverage.SubdivisionLedger()
                        outcome, _, _ = (
                            _dependency_phydrax_geometry__planar_coverage.mapped_containment(
                                chart,
                                chart_domain,
                                group,
                                own,
                                other,
                                self.limits.maximum_bernstein_nodes,
                                self.limits.maximum_subdivision_depth,
                                min(self.limits.maximum_subdivision_pieces, work[0]),
                                min(self.limits.maximum_candidate_pairs, work[0]),
                                used_work,
                                budget=budget,
                            )
                        )
                        work[0] = spend_support_work(
                            work[0],
                            used_work.pieces + used_work.candidate_pairs,
                            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
                        )
                        if outcome != "proven":
                            continue
                        target_orientation = group.orientation * (
                            1 if own == group.regions[0] else -1
                        )
                        orientations.add(target_orientation * group.orientations[0])
                        supported = True
                        break
                if supported:
                    break
            if not supported:
                return SupportVerdict(False, 0)
        return SupportVerdict(
            len(orientations) <= 1, next(iter(orientations)) if orientations else 0
        )

    def vertex_support(
        self,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> bool:
        """Prove stratum support of a vertex on its rational map corner, not its rounding.

        ``source_index`` is the local domain vertex/edge row for strata 0/1 and
        the declared facet/region identifier for strata 2/3. rational algebra is
        charged to this preparation's ledger.
        """
        with (
            self.ledger.activate(),
            self.ledger.bound_stage(
                self.prepared.request.limits.maximum_work_units,
                self.prepared.request.limits.maximum_scratch_bytes,
                starting_work_units=self.work_start,
            ),
            self.ledger.temporary_scope(),
        ):
            return self._vertex_support(
                row,
                source_dimension,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            )

    def _vertex_support(
        self,
        row: int,
        source_dimension: int,
        source_index: int,
        /,
        *,
        edge_vertices: np.ndarray,
        triangle_vertices: np.ndarray,
        triangle_facets: np.ndarray,
        work: list[int],
        source_vertices: np.ndarray | None = None,
    ) -> bool:
        budget = self.ledger
        if row < 0 or row >= self.mesh.entity_set(0).count:
            raise ValueError(
                "MappedPlcSupport.vertex must index the carrier vertex bank; got an undeclared vertex"
            )
        if source_dimension == 3:
            return self._entity_support(
                0,
                row,
                3,
                source_index,
                edge_vertices=edge_vertices,
                triangle_vertices=triangle_vertices,
                triangle_facets=triangle_facets,
                work=work,
                source_vertices=source_vertices,
            ).supported
        point = self.vertex_images[int(self.entity_vertices[0][row][0])]
        work[0] = spend_support_work(
            work[0],
            1,
            "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
        )
        budget.reserve_geometry_queries(1)
        authority = self.domain.vertices if source_vertices is None else source_vertices
        if source_dimension == 0:
            return (
                point
                == planar_primitives_owner.rational_points(
                    authority[source_index : source_index + 1], budget=budget
                )[0]
            )
        if source_dimension == 1:
            start, end = planar_primitives_owner.rational_points(
                authority[edge_vertices[source_index]], budget=budget
            )
            planar_primitives_owner.reserve_fraction_work(
                (point, start, end),
                2 + 3 * len(point),
                1 + 2 * len(point),
                8,
                budget=budget,
            )
            direction = tuple(b - a for a, b in zip(start, end, strict=True))
            axis = max(range(len(direction)), key=lambda value: abs(direction[value]))
            if not direction[axis]:
                raise ValueError(
                    "MappedPlcSupport.source_edge must have distinct endpoints; got a collapsed edge"
                )
            parameter = (point[axis] - start[axis]) / direction[axis]
            return 0 <= parameter <= 1 and all(
                value == origin + parameter * delta
                for value, origin, delta in zip(point, start, direction, strict=True)
            )
        if source_dimension != 2:
            raise ValueError(
                "MappedPlcSupport.source_dimension must be 0 through 3; got another dimension"
            )
        triangles = triangle_vertices[triangle_facets == source_index]
        if triangles.shape[0] == 0:
            return False
        for group in self._source_facet_groups(source_index, triangles, authority, work):
            work[0] = spend_support_work(
                work[0],
                len(group.simplices),
                "MappedPlcSupport.work must respect the support query allowance; got resource exhaustion",
            )
            budget.reserve_geometry_queries(len(group.simplices))
            planar_primitives_owner.reserve_fraction_work(
                (group.plane, point),
                2 * len(point),
                1,
                2 * len(point) + 1,
                budget=budget,
            )
            if sum(
                (
                    weight * value
                    for weight, value in zip(group.plane[:-1], point, strict=True)
                ),
                group.plane[-1],
            ) == 0 and _segment_in_union(
                planar_primitives_owner.project((point,), group.axes, budget=budget),
                group.simplices,
                budget=budget,
            ):
                return True
        return False


def _require_support_work_bounds(
    prepared: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    vertices: tuple[np.ndarray, ...],
    boundary: tuple[np.ndarray, ...],
    embedding_was_supplied: bool,
) -> None:
    """Preserve the support owner's stricter work caps on normalized premises."""
    charged = [
        (
            prepared.coverage.candidate_pair_count,
            limits.maximum_candidate_pairs,
            "coverage candidates",
        ),
        (
            prepared.coverage.subdivision_piece_count,
            limits.maximum_subdivision_pieces,
            "coverage subdivision pieces",
        ),
        (
            prepared.coverage.maximum_subdivision_depth_reached,
            limits.maximum_subdivision_depth,
            "coverage subdivision depth",
        ),
    ]
    if not embedding_was_supplied:
        embedding = prepared.embedding
        charged.extend(
            (
                (
                    embedding.candidate_pair_count,
                    limits.maximum_candidate_pairs,
                    "embedding candidates",
                ),
                (
                    embedding.ray_test_count,
                    limits.maximum_ray_tests,
                    "embedding ray tests",
                ),
                (
                    embedding.subdivision_piece_count,
                    limits.maximum_subdivision_pieces,
                    "embedding subdivision pieces",
                ),
            )
        )
        if "exterior_degree" in embedding.evaluated_checks:
            # The bound affine degree proof preflights every shell against all
            # triangulated boundary pieces, not just the rays eventually used.
            pieces = sum(
                int(np.count_nonzero(row >= 0)) - 2 for row in vertices[2][boundary[2]]
            )
            charged.append(
                (
                    embedding.shell_count * pieces,
                    limits.maximum_ray_tests,
                    "exterior degree preflight",
                )
            )
    for actual, maximum, name in charged:
        if actual > maximum:
            raise ValueError(
                f"MappedPlcSupport.work must respect the {name} bound; got {actual} > {maximum}"
            )
    # Neither evidence reports a Bernstein-node ledger and the embedding reports
    # no depth ledger. A no-larger normalized policy is itself a rigorous bound;
    # do not guess usage beyond that policy.
    if prepared.request.limits.maximum_bernstein_nodes > limits.maximum_bernstein_nodes:
        raise ValueError(
            "MappedPlcSupport.coverage must establish the coefficient bound; got a larger prepared policy"
        )
    if (
        not embedding_was_supplied
        and prepared.request.limits.maximum_subdivision_depth
        > limits.maximum_subdivision_depth
    ):
        raise ValueError(
            "MappedPlcSupport.embedding must establish the subdivision-depth bound; got a larger prepared policy"
        )


@checked
def prepare_mapped_plc_support(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    region_indices: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    certification_request: _owner_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None = None,
    predecessor: CellGeometryPredecessor | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedPlcSupport:
    """Validate source evidence or independently certify the whole target image.

    Labels are explicit region identifiers in entity-set order. bound positive
    validity and embedding must bind the same represented map. The normalized prepared
    acceptance owner computes coverage once with explicit domain-row labels in
    block order. Its normalized request policy is distinct from the row budget,
    while represented charged work must still meet the source stricter support caps.
    Publication supplies its rational limits; existing embedding requires validity.
    """
    if certification_request is not None:
        certification_request.validate_source_integrity(budget=budget)
        if (
            certificate_limits is not None
            and certificate_limits.limits_id != certification_request.limits.limits_id
        ):
            raise ValueError(
                "MappedPlcSupport.limits must match the target request; got different limit identities"
            )
        certificate_limits_ = certification_request.limits
    else:
        certificate_limits_ = (
            _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
            if certificate_limits is None
            else certificate_limits
        )
    ledger = budget
    work_start = ledger.work_units
    with (
        ledger.activate(),
        ledger.bound_stage(
            certificate_limits_.maximum_work_units,
            certificate_limits_.maximum_scratch_bytes,
            starting_work_units=work_start,
        ),
    ):
        return _prepare_mapped_plc_support(
            mesh,
            geometry,
            domain,
            cell_regions,
            region_indices,
            maximum_support_queries=maximum_support_queries,
            embedding=embedding,
            validity=validity,
            certificate_limits=certificate_limits_,
            certification_request=certification_request,
            predecessor=predecessor,
            ledger=ledger,
            work_start=work_start,
            budget=budget,
        )


@final
class MappedRegionRows(StrictModule):
    """Declared region indices mapped into coordinate block row order."""

    labels: np.ndarray
    regions: np.ndarray
    order: np.ndarray
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits


@final
class MappedSupportTables(StrictModule):
    """Immutable entity incidence and region tables retained by source support."""

    cells: tuple[CellMap, ...]
    cell_regions: np.ndarray
    vertices: tuple[np.ndarray, ...]
    parents: tuple[tuple[np.ndarray, np.ndarray], ...]
    boundary: tuple[np.ndarray, ...]


def _retain_mapped_cell_images(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedCellImages:
    """Retain shared source controls and rational carrier corner images once."""
    ledger = budget
    with ledger.activate():
        with ledger.temporary_scope():
            mapped = _cell_maps(mesh, geometry, prepare_expressions=False, budget=budget)
        cells, vertex_images = (mapped.cells, mapped.vertex_images)
        retained_controls: set[int] = set()
        for cell in cells:
            if id(cell.controls) not in retained_controls:
                ledger.retain_basis(cell.controls)
                retained_controls.add(id(cell.controls))
        ledger.retain_basis(vertex_images)
        if geometry.exact_source is not None and all(
            (cell.kind == "tetrahedron" for cell in cells)
        ):
            ledger.reserve(sum((len(point) for point in vertex_images)))
            if all(
                (
                    Fraction(float(value)) == value
                    for point in vertex_images
                    for value in point
                )
            ):
                scope_key = coordinate_scope_key(mesh, geometry)
                affine_cells = np.empty((0,), dtype=np.int64)
                affine_cells.setflags(write=False)
                prepared_scope = (False, affine_cells, vertex_images)
                ledger.retain_basis((scope_key, prepared_scope))
                ledger.scope_cache[scope_key] = prepared_scope
        ledger.reserve(
            sum((2 + len(cell.vertices) + len(cell.root_vertex_ids) for cell in cells)),
            128
            + sum(
                (
                    256 + 40 * (len(cell.vertices) + len(cell.root_vertex_ids))
                    for cell in cells
                )
            ),
        )
    return mapped


def _mapped_region_rows(
    mesh: CellMesh,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    region_indices: np.ndarray,
    maximum_support_queries: int,
    /,
) -> MappedRegionRows:
    """Validate a region-index bijection and prepare labels in block order."""
    labels = np.asarray(cell_regions, dtype=np.int64)
    declared = np.asarray(region_indices, dtype=np.int64)
    if labels.shape != (mesh.entity_set(3).count,) or not np.all(
        np.isin(labels, declared)
    ):
        raise ValueError(
            "MappedPlcSupport.cell_regions must label every cell with a declared region; got missing or undeclared labels"
        )
    if (
        declared.shape != (len(domain.region_ids),)
        or np.unique(declared).size != declared.size
    ):
        raise ValueError(
            "MappedPlcSupport.region_indices must bijectively label domain regions; got duplicate or missing indices"
        )
    limits = _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits(
        maximum_candidate_pairs=maximum_support_queries,
        maximum_ray_tests=maximum_support_queries,
        maximum_subdivision_pieces=maximum_support_queries,
    )
    block_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    order = association_state_owner.entity_rows(mesh, 3, block_ids)
    indices = {label: index for index, label in enumerate(declared.tolist())}
    regions = np.asarray(
        [indices[label] for label in labels[order].tolist()], dtype=np.int64
    )
    return MappedRegionRows(labels, regions, order, limits)


def _mapped_support_validity(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    mapped: MappedCellImages,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate:
    """Bind or certify positivity of the represented coordinate map."""
    ledger, cells, vertex_images = budget, mapped.cells, mapped.vertex_images
    with ledger.activate():
        if validity is None:
            validity = _exact_affine_plc_validity(
                mesh, geometry, cells, vertex_images, budget=budget
            )
            if validity is None:
                validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                    geometry, mesh=mesh, budget=budget
                )
        validity.require_bound(geometry, mesh=mesh)
        if not validity.all_certified:
            raise ValueError(
                "MappedPlcSupport.validity must certify coordinate-map positivity; got invalid or unresolved cells"
            )
    return validity


def _mapped_support_evidence(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    regions: np.ndarray,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    certification_request: _owner_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None,
) -> _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence:
    """Prepare acceptance evidence at the current retained-storage charge."""
    if embedding is None and predecessor is not None:
        embedding = global_embedding_owner.certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=certificate_limits,
            junction_vertices=(
                None
                if certification_request is None
                else certification_request.junction_vertices
            ),
            predecessor=predecessor,
            budget=budget,
        )
    return _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        validity,
        schedule=_leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
            "volume_plc"
        )
        if certification_request is None
        else certification_request.schedule,
        domain=domain,
        cell_regions=regions,
        embedding=embedding,
        limits=certificate_limits,
        source=None if certification_request is None else certification_request.source,
        fidelity_tolerance=None
        if certification_request is None
        else certification_request.fidelity_tolerance,
        fidelity_sample_order=4
        if certification_request is None
        else certification_request.fidelity_sample_order,
        budget=budget,
        scoped_fidelity=()
        if certification_request is None
        else certification_request.scoped_fidelity,
    )


def _mapped_support_tables(
    mesh: CellMesh,
    mapped: MappedCellImages,
    rows: MappedRegionRows,
    prepared: _owner_phydrax_meshing__certification.MeshCertificationPreparedEvidence,
    embedding_was_supplied: bool,
    /,
) -> MappedSupportTables:
    """Build immutable support incidence and enforce the source proof work caps."""
    labels, order, limits, cells = rows.labels, rows.order, rows.limits, mapped.cells
    frozen = np.array(labels, dtype=np.int64, copy=True)
    frozen.setflags(write=False)
    inverse_order = np.argsort(order, kind="stable")
    ordered_cells = tuple(cells[index] for index in inverse_order.tolist())
    unused_top_vertices = np.empty(
        (mesh.entity_set(3).count, 0),
        dtype=np.int64,
    )
    unused_top_vertices.setflags(write=False)
    vertices = (
        *(plc_mapped_vertices(mesh, dimension) for dimension in range(3)),
        unused_top_vertices,
    )

    # Parent cells of every lower-dimensional entity, in incidence order.
    parents: list[tuple[np.ndarray, np.ndarray]] = []
    for dimension in range(3):
        incidence = association_routes.incidence_pairs(mesh, dimension, 3)
        incidence_order = np.argsort(incidence[:, 0], kind="stable")
        offsets = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum(
                    np.bincount(
                        incidence[:, 0], minlength=mesh.entity_set(dimension).count
                    ),
                    dtype=np.int64,
                ),
            )
        )
        values = np.ascontiguousarray(incidence[incidence_order, 1], dtype=np.int64)
        offsets.setflags(write=False)
        values.setflags(write=False)
        parents.append((offsets, values))
    unused_top_parents = (
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )
    unused_top_parents[0].setflags(write=False)
    unused_top_parents[1].setflags(write=False)
    parents.append(unused_top_parents)
    boundary = (
        *(association_routes.boundary_mask(mesh, dimension) for dimension in range(3)),
        np.empty((0,), dtype=np.bool_),
    )
    boundary[3].setflags(write=False)
    _require_support_work_bounds(
        prepared, limits, vertices, boundary, embedding_was_supplied
    )
    return MappedSupportTables(ordered_cells, frozen, vertices, tuple(parents), boundary)


def _prepare_mapped_plc_support(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    cell_regions: np.ndarray,
    region_indices: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    certification_request: _owner_phydrax_meshing__certification_inputs.MeshCertificationInputs
    | None,
    predecessor: CellGeometryPredecessor | None,
    ledger: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    work_start: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedPlcSupport:

    embedding_was_supplied = embedding is not None
    mapped = _retain_mapped_cell_images(mesh, geometry, budget=budget)
    rows = _mapped_region_rows(
        mesh, domain, cell_regions, region_indices, maximum_support_queries
    )
    if embedding is not None and validity is None:
        raise ValueError(
            "MappedPlcSupport.embedding must retain bound validity evidence; got no validity certificate"
        )
    validity = _mapped_support_validity(mesh, geometry, mapped, validity, budget=budget)
    with ledger.activate():
        prepared = _mapped_support_evidence(
            mesh,
            geometry,
            validity,
            domain,
            rows.regions,
            embedding,
            certificate_limits,
            certification_request,
            budget=budget,
            predecessor=predecessor,
        )
    if certification_request is not None:
        prepared.require(mesh, geometry, certification_request, budget=budget)
    tables = _mapped_support_tables(mesh, mapped, rows, prepared, embedding_was_supplied)
    retained = (
        tables.cell_regions,
        *tables.vertices,
        *(array for pair in tables.parents for array in pair),
        *tables.boundary,
    )
    ledger.reserve(
        sum(array.size for array in retained),
        128 * (4 + len(retained)) + sum(array.nbytes for array in retained),
    )
    # Coverage records its earned proof phase. The live request ledger retains
    # support-table charges without recomputing the unchanged geometric proof.
    return MappedPlcSupport(
        mesh,
        geometry,
        domain,
        tables.cells,
        tables.cell_regions,
        prepared.embedding,
        prepared.coverage,
        rows.limits,
        tables.vertices,
        tables.parents,
        tables.boundary,
        prepared,
        mapped.vertex_images,
        mapped.degree_one_exactness_work_units,
        ledger,
        work_start,
    )


def _mapped_association_orientations(
    transfer: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    mesh: CellMesh,
    proof: MappedPlcSupport,
    association: association_state_owner.GeometryAssociation,
    dimension: int,
    work: list[int],
) -> np.ndarray:

    rows = association_state_owner.entity_rows(
        mesh, dimension, np.asarray(association.target_global_ids, dtype=np.int64)
    )
    if np.any(rows < 0):
        raise ValueError(
            "MappedPlcSupport.association must index declared target entities; got an undeclared target ID"
        )
    orientations = np.zeros(rows.shape, dtype=np.int8)
    kinds = np.asarray(association.source_dimensions, dtype=np.int64)
    indices = np.asarray(association.source_indices, dtype=np.int64)
    for position, row, kind, index in zip(
        range(rows.size), rows.tolist(), kinds.tolist(), indices.tolist(), strict=True
    ):
        local_index = index
        if kind in (0, 1):
            matches = np.flatnonzero(
                (transfer.vertex_indices if kind == 0 else transfer.edge_indices) == index
            )
            if matches.size != 1:
                raise ValueError(
                    "MappedPlcSupport.association must resolve the source vertex or edge row; got an undeclared source ID"
                )
            local_index = int(matches[0])
        if dimension == 0:
            supported = proof.vertex_support(
                row,
                kind,
                local_index,
                edge_vertices=transfer.edge_vertices,
                triangle_vertices=transfer.triangle_vertices,
                triangle_facets=transfer.triangle_facets,
                work=work,
            )
            orientation = 0
        else:
            supported, orientation = proof.entity_support(
                dimension,
                row,
                kind,
                local_index,
                edge_vertices=transfer.edge_vertices,
                triangle_vertices=transfer.triangle_vertices,
                triangle_facets=transfer.triangle_facets,
                work=work,
            )
        if not supported:
            raise association_routes.AssociationPropagationError(
                "MappedPlcSupport.association must support the declared PLC stratum; got an unsupported Q1 image",
                np.asarray(association.target_global_ids)[position : position + 1],
            )
        orientations[position] = orientation
    return orientations


@dataclass(frozen=True)
class MappedPlcAssociationProof:
    """rational source rows together with their normalized labelled acceptance premises."""

    associations: tuple[association_state_owner.GeometryAssociation, ...]
    support: MappedPlcSupport


@checked
def certify_mapped_plc_associations(
    transfer: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    associations: tuple[association_state_owner.GeometryAssociation, ...],
    /,
    *,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate | None = None,
    validity: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedPlcAssociationProof:
    """Publish rational represented-source rows only after proving represented Q1 support.

    This is a source-publication proof, not relaxation of a transfer policy.
    The input rows supply declared stratum identities and lineage. Rounded
    simplex-parent containment does not decide membership in a different PLC
    stratum. Every point and whole coordinate-map image is instead checked
    against the explicitly named declared PLC source before residual and
    orientation evidence is issued. Input rows and their source remain unchanged.
    """

    if transfer.domain.ambient_dimension != 3:
        raise ValueError(
            "MappedPlcSupport.domain must have ambient dimension 3; got a nonvolume source"
        )
    roles = (
        association_state_owner.GeometrySourceEntityRole.VERTEX,
        association_state_owner.GeometrySourceEntityRole.EDGE,
        association_state_owner.GeometrySourceEntityRole.FACET,
        association_state_owner.GeometrySourceEntityRole.REGION,
    )
    dimensions: list[int] = []
    regions: np.ndarray | None = None
    for association in associations:
        if (
            association.association_kind
            is not association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != transfer.domain.source_id
            or association.source_revision != transfer.source_revision
        ):
            raise ValueError(
                "MappedPlcSupport.association must retain the represented source namespace; got a different source binding"
            )
        dimension = association_state_owner.target_dimension(mesh, association)
        association.validate_target(mesh.entity_set(dimension))
        if dimension in dimensions or not association.complete:
            raise ValueError(
                "MappedPlcSupport.association must contain unambiguous explicit source rows; got unresolved or ambiguous rows"
            )
        kinds = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        if np.any(kinds < 0) or np.any(kinds > 3) or np.any(indices < 0):
            raise ValueError(
                "MappedPlcSupport.association must declare source-stratum identities; got no source role or namespace"
            )
        expected_roles = tuple(roles[kind] for kind in kinds.tolist())
        expected_ids = tuple(
            _owner_phydrax_meshing__host_banks.entity_key(
                transfer.source_revision, role.value, index
            )
            for role, index in zip(expected_roles, indices.tolist(), strict=True)
        )
        if (
            association.source_entities.roles() != expected_roles
            or association.source_entities.entity_ids() != expected_ids
        ):
            raise ValueError(
                "MappedPlcSupport.association must align roles and namespace IDs with indices; got mismatched metadata"
            )
        if (
            np.any(~np.isin(indices[kinds == 0], transfer.vertex_indices))
            or np.any(~np.isin(indices[kinds == 1], transfer.edge_indices))
            or np.any(indices[kinds == 2] >= transfer.facet_regions.shape[0])
            or np.any(~np.isin(indices[kinds == 3], transfer.region_indices))
        ):
            raise ValueError(
                "MappedPlcSupport.association must reference declared source strata; got an undeclared source ID"
            )
        if dimension == 3:
            if np.any(kinds != 3):
                raise ValueError(
                    "MappedPlcSupport.cells must name source regions; got a nonregion cell association"
                )
            regions = indices[
                association.target_rows(
                    np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
                )
            ]
        dimensions.append(dimension)
    if 0 not in dimensions or regions is None:
        raise ValueError(
            "MappedPlcSupport.associations must cover vertices and cells; got missing dimensions"
        )
    proof = prepare_mapped_plc_support(
        mesh,
        geometry,
        transfer.domain,
        regions,
        transfer.region_indices,
        maximum_support_queries=transfer.maximum_support_queries,
        embedding=embedding,
        validity=validity,
        certificate_limits=certificate_limits,
        budget=budget,
    )
    work = [transfer.maximum_support_queries]
    output: list[association_state_owner.GeometryAssociation] = []
    for association in associations:
        dimension = association_state_owner.target_dimension(mesh, association)
        orientations = _mapped_association_orientations(
            transfer, mesh, proof, association, dimension, work
        )
        kinds = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        output.append(
            association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id,
                transfer.source_revision,
                association.target_entity_set_id,
                association.target_global_ids,
                association.source_entities.entity_ids(),
                np.zeros(association.target_global_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=kinds,
                source_indices=indices,
                source_entity_roles=association.source_entities.roles(),
                parameters=association.parameters,
                orientations=orientations,
                parent_dimensions=association.parent_dimensions,
                parent_ids=association.parent_ids,
                parent_association_id=association.parent_association_id,
                provenance=association.provenance,
            )
        )
    return MappedPlcAssociationProof(tuple(output), proof)


def _require_mapped_source_translation(
    transfer: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    source: result_owner.CellMeshingResult,
    predecessor: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    target: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    shift: np.ndarray,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    if (
        transfer.domain.ambient_dimension != 3
        or predecessor.domain.ambient_dimension != 3
    ):
        raise ValueError(
            "MappedPlcTranslation.domain must have ambient dimension 3; got a nonvolume source"
        )
    if source.mesh.storage is not None or target.storage is not None:
        raise ValueError(
            "MappedPlcTranslation.mesh must be serial host-resident carriers; got mesh storage"
        )
    if (
        transfer.domain.source_id != predecessor.domain.source_id
        or transfer.coordinate_contract.spatial_id
        != predecessor.coordinate_contract.spatial_id
        or source.coordinate_contract.spatial_id
        != predecessor.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "MappedPlcTranslation.source must retain identity, frame, and units; got a different source contract"
        )
    if transfer.source_revision == predecessor.source_revision:
        raise ValueError(
            "MappedPlcTranslation.source_revision must identify a successor; got the predecessor revision"
        )
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError(
            "MappedPlcTranslation.displacement must be finite with shape (3,); got an invalid displacement"
        )
    if (
        target.topology_id != source.mesh.topology_id
        or geometry.geometry_layout_id != source.geometry.geometry_layout_id
    ):
        raise ValueError(
            "MappedPlcTranslation.geometry must retain topology and coordinate-map layout; got different layouts"
        )
    for dimension in range(4):
        if not np.array_equal(
            target.entity_set(dimension).entity_ids,
            source.mesh.entity_set(dimension).entity_ids,
        ):
            raise ValueError(
                "MappedPlcTranslation.mesh must retain scientific entity identities; got different entity IDs"
            )
    tables = (
        (transfer.edge_vertices, predecessor.edge_vertices),
        (transfer.vertex_indices, predecessor.vertex_indices),
        (transfer.edge_indices, predecessor.edge_indices),
        (transfer.triangle_vertices, predecessor.triangle_vertices),
        (transfer.triangle_facets, predecessor.triangle_facets),
        (transfer.facet_regions, predecessor.facet_regions),
        (transfer.region_indices, predecessor.region_indices),
        (transfer.domain.facets, predecessor.domain.facets),
        (transfer.domain.facet_regions, predecessor.domain.facet_regions),
    )
    if any(not np.array_equal(actual, original) for actual, original in tables):
        raise ValueError(
            "MappedPlcTranslation.incidence must retain stratum rows and source indices; got different tables"
        )
    if transfer.domain.region_ids != predecessor.domain.region_ids:
        raise ValueError(
            "MappedPlcTranslation.region_ids must retain domain-region identities; got different region IDs"
        )
    for actual, original in (
        (transfer.source_vertices, predecessor.source_vertices),
        (transfer.domain.vertices, predecessor.domain.vertices),
        (np.asarray(target.coordinates), np.asarray(source.mesh.coordinates)),
        (np.asarray(geometry.coordinates), np.asarray(source.geometry.coordinates)),
    ):
        if actual.shape != original.shape or not np.array_equal(actual, original + shift):
            raise ValueError(
                "MappedPlcTranslation.coordinates must equal source coordinates plus displacement; got different coordinates"
            )
    original_maps = _cell_maps(source.mesh, source.geometry, budget=budget)
    target_maps = _cell_maps(target, geometry, budget=budget)
    for original, actual in zip(original_maps.cells, target_maps.cells, strict=True):
        expected_map = tuple(
            algebra.expression_add(
                value,
                algebra.constant(Fraction(float(displacement)), 3, budget=budget),
                budget=budget,
            )
            for value, displacement in zip(
                _cell_coordinates(original, budget=budget), shift, strict=True
            )
        )
        if (
            original.kind != actual.kind
            or original.vertices != actual.vertices
            or not all(
                algebra.expression_equal(a, b, budget=budget)
                for a, b in zip(
                    _cell_coordinates(actual, budget=budget), expected_map, strict=True
                )
            )
        ):
            raise ValueError(
                "MappedPlcTranslation.coordinates must represent the declared translation; got different map expressions"
            )


@checked
def transition_mapped_source_associations(
    transfer: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    source: result_owner.CellMeshingResult,
    predecessor: _owner_phydrax_meshing__association.PlcAssociationTransfer,
    target: CellMesh,
    /,
    *,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    translation: ArrayLike,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    tuple[organization_owner.MeshPatch, ...],
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshLabel, ...],
    tuple[association_state_owner.GeometryAssociation, ...],
]:
    """Carry explicit represented volume strata through a proven physical translation."""

    shift = np.asarray(translation, dtype=np.float64)
    _require_mapped_source_translation(
        transfer,
        source,
        predecessor,
        target,
        geometry,
        shift,
        budget=budget,
    )
    predecessor.classes(source, budget=budget)
    cell = next(
        association
        for association in source.associations
        if association_state_owner.target_dimension(source.mesh, association) == 3
    )
    regions = np.asarray(cell.source_indices, dtype=np.int64)[
        cell.target_rows(np.asarray(target.entity_set(3).entity_ids, dtype=np.int64))
    ]
    proof = prepare_mapped_plc_support(
        target,
        geometry,
        transfer.domain,
        regions,
        transfer.region_indices,
        maximum_support_queries=transfer.maximum_support_queries,
        budget=budget,
    )
    work = [transfer.maximum_support_queries]
    output: list[association_state_owner.GeometryAssociation] = []
    for previous in source.associations:
        dimension = association_state_owner.target_dimension(source.mesh, previous)
        orientations = _mapped_association_orientations(
            transfer, target, proof, previous, dimension, work
        )
        roles = previous.source_entities.roles()
        if roles is None or any(role is None for role in roles):
            raise ValueError(
                "MappedPlcTranslation.association must declare volume source roles; got incompatible roles"
            )
        names = tuple(
            _owner_phydrax_meshing__host_banks.entity_key(
                transfer.source_revision, role.value, int(index)
            )
            for role, index in zip(
                roles, np.asarray(previous.source_indices), strict=True
            )
            if role is not None
        )
        output.append(
            association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                transfer.domain.source_id,
                transfer.source_revision,
                target.entity_set(dimension).entity_set_id,
                previous.target_global_ids,
                names,
                np.zeros(previous.target_global_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=previous.source_dimensions,
                source_indices=previous.source_indices,
                source_entity_roles=roles,
                parameters=previous.parameters,
                orientations=orientations,
                parent_dimensions=np.full(
                    previous.target_global_ids.shape, dimension, dtype=np.int64
                ),
                parent_ids=previous.target_global_ids,
                parent_association_id=previous.association_id,
                provenance=association_state_owner.GeometryAssociationProvenance.LINEAGE,
            )
        )
    patches, zones, labels = _owner_phydrax_meshing__lineage.inherit_mesh_organization(
        source,
        target,
        _owner_phydrax_meshing__lineage.identity_lineage(source.mesh, target),
    )
    return patches, zones, labels, tuple(output)
