#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate pure hexahedral grids and source-chart restrictions.

The dual route partitions an affine tetrahedral complex into vertex-star hexes.
Balanced-grid and frame-grid routes solve a feature frame and prove a
coordinate-plane block decomposition before integer extraction. The balanced
route consumes the spatial octree and bounded 1:8 face-parity templates; the
frame route retains feature-plane knots. Mapped source charts retain their
coordinate expressions through rational restrictions.

Nonpolycube cuts, varying-frame singular graphs, and CAD projection without a
coverage chart refuse through the selected route; extraction never switches
to dual after a failure.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax._meshcore as _owner_phydrax__meshcore
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _dependency_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._simplicial_locator as _dependency_phydrax_discretization__simplicial_locator
import phydrax.discretization.fem._cell_map as _dependency_phydrax_discretization_fem__cell_map
import phydrax.geometry._atlas as _dependency_phydrax_geometry__atlas
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.linalg._small_batched as _dependency_phydrax_linalg__small_batched
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._dual_extraction as dual_extraction_owner
import phydrax.meshing._family_execution as family_execution_owner
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._reference_root_composition as _dependency_phydrax_meshing__reference_root_composition
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.geometry._rational import fraction_endpoint
from phydrax.linalg._exact import solve_exact

from .._bvh import point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_integer
from ..discretization import (
    CellBlock,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellGeometryElement
from ..discretization.fem._reference_types import FiniteElementSpec
from ..discretization.spatial._level_octree import refined_octree_leaves
from ..discretization.spatial._morton import morton_encode_integer_host, MortonAddressPlan
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
from ..optim._iterative._types import MinimizationResult
from ..typing import checked, parse
from ._hex_frame_records import SourceBlockIntegerGrid
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed


@checked
def extract_volume_hexes(
    source: CellMesh,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    boundary_subdivision: bool = True,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dual_extraction_owner.DualExtraction:
    """Partition a positive affine tetrahedral complex into hexahedra.

    Source validity, embedding, and coverage must independently certify before
    publication. Positive input tetrahedra alone do not establish nonoverlap.
    """

    if source.topology.dimension != 3:
        raise TypeError(
            f"hex.source.topological_dimension must be three; got {source.topology.dimension}"
        )
    if source.coordinates.shape[1] != 3:
        raise ValueError(
            f"hex.source.coordinates must have three physical axes; got {source.coordinates.shape[1]}"
        )
    if source.periodic_topology is not None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.periodic_topology must provide quotient subdivision ancestry; got periodic topology without subdivision ancestry"
        )
    if any(block.cell_kind != "tetrahedron" for block in source.blocks):
        raise ValueError(
            f"hex.source.blocks must contain only tetrahedra; got {tuple(block.cell_kind for block in source.blocks)}"
        )
    if not boundary_subdivision:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.boundary_subdivision must be enabled for triangular-facet closure; got {boundary_subdivision}"
        )
    cells = _dependency_phydrax_meshing__host_banks.entities(source, 3)
    count, original = cells.shape[0], source.coordinates.shape[0]
    scratch, work = 8192 * count + 256 * original, 128 * count
    family_execution_owner.check_family_limits(
        limits, 4 * count, original, 32 * count, scratch, work
    )
    source_geometry = (
        _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(source)
        if source_geometry is None
        else source_geometry
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        certificate = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            source_geometry, mesh=source, budget=budget
        )
    if certificate.invalid_count or certificate.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.validity must certify every tetrahedron; got {certificate.invalid_count} invalid and {certificate.unresolved_count} unresolved"
        )
    edges, faces = (
        _dependency_phydrax_meshing__host_banks.entities(source, 1),
        _dependency_phydrax_meshing__host_banks.entities(source, 2),
    )
    edge_count, face_count = edges.shape[0], faces.shape[0]
    added = edge_count + face_count + count
    output_edges, output_faces = (
        2 * edge_count + 3 * face_count + 4 * count,
        3 * face_count + 6 * count,
    )
    family_execution_owner.check_family_limits(
        limits, 4 * count, original + added, 32 * count, scratch, work
    )
    if output_edges > limits.maximum_edges or output_faces > limits.maximum_faces:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.topology must fit the edge and face limits; got {output_edges} edges and {output_faces} faces",
            resource=True,
        )
    # Allocate only after bounding all template work and published entities.
    edge_lookup = {
        tuple(sorted(edge)): original + row for row, edge in enumerate(edges.tolist())
    }
    face_lookup = {
        tuple(sorted(face)): original + edge_count + row
        for row, face in enumerate(faces.tolist())
    }
    local_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    nodes = _dependency_phydrax_meshing__host_banks.family_host_array(
        (count, 15), np.int64, scope=scope
    )
    nodes[:, :4] = cells
    face_incidence = np.zeros((face_count,), dtype=np.int64)
    for row, tetrahedron in enumerate(cells.tolist()):
        nodes[row, 4:10] = [
            edge_lookup[tuple(sorted((tetrahedron[a], tetrahedron[b])))]
            for a, b in local_edges
        ]
        nodes[row, 10:14] = [
            face_lookup[tuple(sorted(tetrahedron[a] for a in face))]
            for face in local_faces
        ]
        face_incidence[nodes[row, 10:14] - original - edge_count] += 1
    if np.any(face_incidence > 2):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.face_incidence must be at most two; got {np.max(face_incidence)}"
        )
    nodes[:, 14] = original + edge_count + face_count + np.arange(count, dtype=np.int64)
    points = np.asarray(source.coordinates, dtype=np.float64)
    coordinates = _dependency_phydrax_meshing__host_banks.family_host_array(
        (original + added, 3), np.float64, scope=scope
    )
    coordinates[:original] = points
    coordinates[original:] = np.nan
    hexes = _owner_phydrax__meshcore.tetrahedron_dual_hexes(
        nodes, coordinates.shape[0], maximum_cells=limits.maximum_cells
    )
    supports = _dependency_phydrax_meshing__host_banks.family_host_array(
        (coordinates.shape[0], 4), np.int64, scope=scope
    )
    supports.fill(-1)
    supports[:original, 0] = np.arange(original, dtype=np.int64)
    supports[original : original + edge_count, :2] = edges
    supports[original + edge_count : original + edge_count + face_count, :3] = faces
    supports[original + edge_count + face_count :] = cells
    initial = (CellBlock("cells", "hexahedron", hexes),)
    mesh, geometry, parents = parent_coordinate_geometry(
        source,
        source_geometry,
        initial,
        np.repeat(np.arange(count, dtype=np.int64), 4),
        coordinates,
        supports,
        family_execution_owner.vertex_ids(source, added),
        scope=scope,
        budget=budget,
    )
    dimensions, rows = _dependency_phydrax_meshing__host_banks.ancestry(
        source, mesh, supports
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget
        )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.validity must certify positive represented-coordinate maps; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
        )
    boundary_faces = (dimensions[2] == 2) & (
        face_incidence[np.minimum(rows[2], face_count - 1)] == 1
    )
    boundary = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    boundary[
        _dependency_phydrax_meshing__host_banks.entities(mesh, 2)[boundary_faces].reshape(
            -1
        )
    ] = True
    valences = np.bincount(hexes.reshape(-1), minlength=coordinates.shape[0]).astype(
        np.int64
    )
    singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    data = (
        coordinates.nbytes
        + hexes.nbytes
        + supports.nbytes
        + sum(array.nbytes for array in dimensions + rows)
    )
    if data > limits.maximum_data_bytes:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.publication.data_bytes must fit maximum_data_bytes={limits.maximum_data_bytes}; got {data}",
            resource=True,
        )
    return dual_extraction_owner.DualExtraction(
        mesh,
        source,
        supports,
        parents,
        dimensions,
        rows,
        validity,
        valences,
        boundary,
        singular,
        None,
        work,
        geometry,
        source_geometry,
    )


class _ParentSource(StrictModule):
    elements: tuple[CellGeometryElement, ...]
    routes: tuple[np.ndarray, ...]
    values: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank
    cells: np.ndarray
    root_blocks: np.ndarray
    root_rows: np.ndarray


class _ParentCharts(StrictModule):
    groups: dict[str, list[tuple[np.ndarray, int, int]]]
    charts: dict[str, FiniteElementSpec]
    controls: dict[str, tuple[np.ndarray, np.ndarray]]


class _ParentPlacement(StrictModule):
    coordinates: np.ndarray


class _ParentGeometry(StrictModule):
    mesh: CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    parents: np.ndarray


def _prepare_parent_source(
    source: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ParentSource:
    source_elements, source_routes, _ = source_geometry.resolve(
        source, budget=budget, exact_source_prepared=False
    )
    host_routes = tuple(
        np.asarray(route, dtype=np.int64) for route in jax.device_get(source_routes)
    )
    source_values = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        source_geometry, budget=budget
    )
    source_cells = _dependency_phydrax_meshing__host_banks.entities(source, 3)
    root_blocks = np.repeat(
        np.arange(len(source.blocks), dtype=np.int64),
        [block.cell_count for block in source.blocks],
    )
    root_rows = np.concatenate(
        [np.arange(block.cell_count, dtype=np.int64) for block in source.blocks]
    )
    return _ParentSource(
        source_elements, host_routes, source_values, source_cells, root_blocks, root_rows
    )


class _ParentChartDefinition(StrictModule):
    controls: list[list[tuple[int, int]]]
    name: str = eqx.field(static=True)


class _ParentChartBanks(StrictModule):
    numerators: np.ndarray
    denominators: np.ndarray


def _define_parent_chart(
    chart: FiniteElementSpec,
    vertices: np.ndarray,
    tetrahedron: np.ndarray,
    supports: np.ndarray,
    parent: int,
    element: CellGeometryElement,
    /,
) -> _ParentChartDefinition:
    controls = [[(0, 1)] * 3 for _ in range(chart.local_dof_count)]
    for corner, vertex in enumerate(vertices):
        original = supports[vertex]
        original = original[original >= 0]
        if original.size == 0 or np.any(~np.isin(original, tetrahedron)):
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.target.support must belong to the parent tetrahedron; got {original.tolist()} for parent {parent}"
            )
        dof = chart.entity_dofs[0][corner][0]
        for axis in range(3):
            value = Fraction(int(tetrahedron[axis + 1] in original), original.size)
            controls[dof][axis] = value.numerator, value.denominator
    name = "source:" + canonical_fingerprint(
        {
            "element": element.element_id,
            "chart": chart.element_id,
            "controls": controls,
        }
    )
    return _ParentChartDefinition(controls, name)


def _materialize_parent_chart(
    chart: FiniteElementSpec,
    controls: list[list[tuple[int, int]]],
    /,
    *,
    scope: NativeExecutionScope,
) -> _ParentChartBanks:
    numerators = _dependency_phydrax_meshing__host_banks.family_host_array(
        (chart.local_dof_count, 3), np.int64, scope=scope
    )
    denominators = _dependency_phydrax_meshing__host_banks.family_host_array(
        numerators.shape, np.int64, scope=scope
    )
    for dof, values in enumerate(controls):
        for axis, (numerator, denominator) in enumerate(values):
            numerators[dof, axis], denominators[dof, axis] = (
                numerator,
                denominator,
            )
    return _ParentChartBanks(numerators, denominators)


def _group_parent_charts(
    prepared: _ParentSource,
    target_blocks: tuple[CellBlock, ...],
    target_parents: np.ndarray,
    supports: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> _ParentCharts:
    source_elements, source_cells = prepared.elements, prepared.cells
    root_blocks = prepared.root_blocks
    groups: dict[str, list[tuple[np.ndarray, int, int]]] = {}
    group_charts = {}
    chart_controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    offset = 0
    for block in target_blocks:
        chart = (
            _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                block.cell_kind, 1
            )
        )
        host_vertices, host_ids = jax.device_get((block.vertices, block.global_ids))
        block_vertices = np.asarray(host_vertices, dtype=np.int64)
        block_ids = np.asarray(host_ids, dtype=np.int64)
        for row in range(block_vertices.shape[0]):
            vertices = np.asarray(block_vertices[row], dtype=np.int64).reshape(-1)
            parent = int(target_parents[offset + row])
            tetrahedron = np.asarray(source_cells[parent], dtype=np.int64).reshape(-1)
            definition = _define_parent_chart(
                chart,
                vertices,
                tetrahedron,
                supports,
                parent,
                source_elements[root_blocks[parent]],
            )
            name = definition.name
            groups.setdefault(name, []).append((vertices, parent, int(block_ids[row])))
            if name not in chart_controls:
                banks = _materialize_parent_chart(chart, definition.controls, scope=scope)
                chart_controls[name] = banks.numerators, banks.denominators
                group_charts[name] = chart
        offset += block.cell_count
    if offset != target_parents.size:
        raise ValueError(
            f"hex.target.parents must match the target cell count; got {target_parents.size} parent rows for {offset} cells"
        )
    return _ParentCharts(groups, group_charts, chart_controls)


def _place_parent_coordinates(
    source: CellMesh,
    prepared: _ParentSource,
    target_blocks: tuple[CellBlock, ...],
    target_parents: np.ndarray,
    coordinates: np.ndarray,
    supports: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ParentPlacement:
    source_elements, host_routes, source_values = (
        prepared.elements,
        prepared.routes,
        prepared.values,
    )
    source_cells, root_blocks, root_rows = (
        prepared.cells,
        prepared.root_blocks,
        prepared.root_rows,
    )
    source_images: dict[
        int,
        tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...],
    ] = {}
    used_parents = np.unique(target_parents)
    family_execution_owner.count_geometry_queries(
        4 * used_parents.size, work_units=4 * used_parents.size, budget=budget
    )
    for parent in used_parents:
        source_element = (
            _dependency_phydrax_discretization__cell_geometry.require_full_p1_source(
                source_elements[root_blocks[parent]]
            )
        )
        route = host_routes[root_blocks[parent]][root_rows[parent]]
        images = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
            source_element,
            tuple(source_values[int(index)] for index in route),
            budget=budget,
        )
        if images is None:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.source.element must support rational P1 corner evaluation; got {source_element.element_id}"
            )
        for vertex, image in zip(source_cells[parent], images, strict=True):
            previous = source_images.setdefault(int(vertex), image)
            if previous != image:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.source.traces must agree at shared vertices; got different images at vertex {vertex}"
                )
    # P1 evaluation uses the authored barycentric chart with exact arithmetic,
    # uniform barycentric chart, not fitting or averaging rounded carriers.
    used_vertices = np.unique(
        np.concatenate(
            [np.asarray(block.vertices).reshape(-1) for block in target_blocks]
        )
    )
    for vertex in used_vertices:
        original = supports[vertex]
        original = original[original >= 0]
        image = tuple(
            sum((source_images[int(index)][axis] for index in original), Fraction(0))
            / original.size
            for axis in range(3)
        )
        represented = (
            _dependency_phydrax_discretization__coordinate_enclosure.rounded_point(image)
        )
        if original.size == 1:
            if not np.array_equal(coordinates[vertex], represented):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.target.coordinates must retain source corner coordinates; got a changed source corner at vertex {vertex}"
                )
        else:
            coordinates[vertex] = represented
    return _ParentPlacement(coordinates)


def _compose_parent_geometry(
    source: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    prepared: _ParentSource,
    charts: _ParentCharts,
    placed: _ParentPlacement,
    target_parents: np.ndarray,
    vertex_ids: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ParentGeometry:
    source_elements, host_routes = prepared.elements, prepared.routes
    source_cells, root_blocks, root_rows = (
        prepared.cells,
        prepared.root_blocks,
        prepared.root_rows,
    )
    groups, group_charts, chart_controls = charts.groups, charts.charts, charts.controls
    coordinates = placed.coordinates
    blocks, elements, dofs, parent_ids, parent_vertices = [], {}, {}, {}, {}
    source_ids = np.asarray(source.topology.entities(3).entity_ids, dtype=np.int64)
    source_vertices = np.asarray(source.vertex_global_ids, dtype=np.int64)
    output_parents = _dependency_phydrax_meshing__host_banks.family_host_array(
        (target_parents.size,), np.int64, scope=scope
    )
    output_offset = 0
    for name, cells in groups.items():
        vertices = np.stack([cell[0] for cell in cells])
        parents = np.asarray([cell[1] for cell in cells], dtype=np.int64)
        identities = np.asarray([cell[2] for cell in cells], dtype=np.int64)
        chart = group_charts[name]
        source_element = (
            _dependency_phydrax_discretization__cell_geometry.require_full_p1_source(
                source_elements[root_blocks[parents[0]]]
            )
        )
        if chart.cell_kind == "pyramid":
            element = _dependency_phydrax_discretization__cell_geometry.RationalComposedCellGeometryElement(
                source_element, chart, *chart_controls[name]
            )
        else:
            element = _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
                source_element, chart, *chart_controls[name], budget=budget
            )
        routes = np.stack(
            [host_routes[root_blocks[parent]][root_rows[parent]] for parent in parents]
        )
        blocks.append(CellBlock(name, chart.cell_kind, vertices, global_ids=identities))
        elements[name], dofs[name] = element, routes
        parent_ids[name] = source_ids[parents]
        parent_vertices[name] = source_vertices[source_cells[parents]]
        output_parents[output_offset : output_offset + parents.size] = parents
        output_offset += parents.size
    mesh = CellMesh(
        coordinates,
        tuple(blocks),
        vertex_global_ids=vertex_ids,
        numeric_version=source.numeric_version,
    )
    provenance = (
        _dependency_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source_geometry
            ),
            source.topology_id,
            parent_ids,
            parent_vertices,
        )
    )
    geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec(
        elements,
        dofs,
        source_geometry.coordinates,
        exact_source=source_geometry.exact_source,
        restriction_source=provenance,
        budget=budget,
    )
    return _ParentGeometry(mesh, geometry, output_parents)


@checked
def parent_coordinate_geometry(
    source: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    target_blocks: tuple[CellBlock, ...],
    target_parents: np.ndarray,
    coordinates: np.ndarray,
    supports: np.ndarray,
    vertex_ids: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    CellMesh,
    _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    np.ndarray,
]:
    """Compose target charts with unchanged source tetrahedral laws."""
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        prepared = _prepare_parent_source(source, source_geometry, budget=budget)
        charts = _group_parent_charts(
            prepared, target_blocks, target_parents, supports, scope=scope
        )
        placed = _place_parent_coordinates(
            source,
            prepared,
            target_blocks,
            target_parents,
            coordinates,
            supports,
            budget=budget,
        )
        composed = _compose_parent_geometry(
            source,
            source_geometry,
            prepared,
            charts,
            placed,
            target_parents,
            vertex_ids,
            scope=scope,
            budget=budget,
        )
    return composed.mesh, composed.geometry, composed.parents


type NativeHexGridRoute = Literal["balanced_grid", "frame_grid"]


class NativeHexGridSchedule(StrictModule):
    """Explicit field/grid portfolio selection, never switched after failure."""

    route: NativeHexGridRoute = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    maximum_field_steps: int = eqx.field(static=True)
    maximum_location_candidates: int = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        route: NativeHexGridRoute,
        /,
        *,
        maximum_depth: int = 10,
        maximum_field_steps: int = 100,
        maximum_location_candidates: int = 32,
        minimum_scaled_jacobian: float = 0.25,
        minimum_mean_ratio: float = 0.5,
        maximum_aspect_ratio: float = 4.0,
    ) -> None:
        route = parse(route, NativeHexGridRoute, "route")
        maximum_depth = positive_integer(maximum_depth, "maximum_depth")
        maximum_field_steps = positive_integer(maximum_field_steps, "maximum_field_steps")
        maximum_location_candidates = positive_integer(
            maximum_location_candidates, "maximum_location_candidates"
        )
        minimum_scaled_jacobian = finite_real_scalar(
            minimum_scaled_jacobian, "minimum_scaled_jacobian"
        )
        minimum_mean_ratio = finite_real_scalar(minimum_mean_ratio, "minimum_mean_ratio")
        maximum_aspect_ratio = finite_real_scalar(
            maximum_aspect_ratio, "maximum_aspect_ratio"
        )
        if maximum_depth > 20:
            raise ValueError(
                f"hex.schedule.maximum_depth must be at most 20; got {maximum_depth}"
            )
        if (
            not np.isfinite(minimum_scaled_jacobian)
            or not 0.0 < minimum_scaled_jacobian <= 1.0
        ):
            raise ValueError(
                f"hex.schedule.minimum_scaled_jacobian must lie in (0, 1]; got {minimum_scaled_jacobian}"
            )
        if (
            not np.isfinite(minimum_mean_ratio)
            or not 0.0 < minimum_mean_ratio <= 1.0
            or not np.isfinite(maximum_aspect_ratio)
            or maximum_aspect_ratio < 1.0
        ):
            raise ValueError(
                f"hex.schedule.quality must have mean ratio in (0, 1] and aspect ratio at least one; got mean={minimum_mean_ratio}, aspect={maximum_aspect_ratio}"
            )
        self.route = route
        self.maximum_depth = maximum_depth
        self.maximum_field_steps = maximum_field_steps
        self.maximum_location_candidates = maximum_location_candidates
        self.minimum_scaled_jacobian = minimum_scaled_jacobian
        self.minimum_mean_ratio = minimum_mean_ratio
        self.maximum_aspect_ratio = maximum_aspect_ratio
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-hex-grid-schedule",
                "route": route,
                "maximum_depth": maximum_depth,
                "maximum_field_steps": maximum_field_steps,
                "maximum_location_candidates": maximum_location_candidates,
                "minimum_scaled_jacobian": minimum_scaled_jacobian,
                "minimum_mean_ratio": minimum_mean_ratio,
                "maximum_aspect_ratio": maximum_aspect_ratio,
            }
        )


class HexFrameField(StrictModule):
    """Feature-aligned octahedral frame and explicit orthogonal patch graph."""

    basis: np.ndarray
    normals: np.ndarray
    patch_axes: np.ndarray
    alignment_residuals: np.ndarray
    singular_vertices: np.ndarray
    solve: _dependency_phydrax_linalg__small_batched.SmallLinearSolveResult
    optimization: MinimizationResult
    source_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)


class GridHexConstruction(StrictModule):
    """Integer-map mesh, region proof worksets, and independently auditable grid."""

    mesh: CellMesh
    cell_regions: np.ndarray
    logical_vertices: np.ndarray
    integer_cells: np.ndarray
    face_source_facets: np.ndarray
    face_source_polygons: np.ndarray
    frame: HexFrameField
    axis_knots: tuple[np.ndarray, ...]
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    subdivisions: int = eqx.field(static=True)
    closure_rounds: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)


def frame_rotation(angles: Array, /) -> Array:
    x, y, z = angles
    sx, sy, sz = jnp.sin(x), jnp.sin(y), jnp.sin(z)
    cx, cy, cz = jnp.cos(x), jnp.cos(y), jnp.cos(z)
    return jnp.stack(
        (
            jnp.stack((cy * cz, -cy * sz, sy)),
            jnp.stack((cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy)),
            jnp.stack((sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy)),
        )
    )


def _frame_energy(angles: Array, arguments: tuple[Array, Array], /) -> Array:
    initial, normals = arguments
    projected = normals @ (initial @ frame_rotation(angles)).T
    # Fourth-order octahedral energy is invariant to axis signs/permutations.
    return jnp.sum(1.0 - jnp.sum(projected**4, axis=1))


@checked
def prepare_hex_frame(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    schedule: NativeHexGridSchedule,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> tuple[HexFrameField, np.ndarray]:
    """Solve the feature frame and pull source vertices through native linalg."""
    measurement_started = phase_started(record_phase)
    loops = [
        complex_.polygon_vertices[
            complex_.polygon_offsets[row] : complex_.polygon_offsets[row + 1]
        ]
        for row in range(complex_.polygon_facets.size)
    ]
    normals = np.stack(
        [
            np.sum(
                np.cross(
                    complex_.vertices[loop], np.roll(complex_.vertices[loop], -1, axis=0)
                ),
                axis=0,
            )
            for loop in loops
        ]
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.feature_patch must have regular frame orientation; got a degenerate feature normal"
        )
    normals /= lengths[:, None]
    first = normals[0]
    transverse = np.flatnonzero(np.abs(normals @ first) < 0.5)
    if transverse.size == 0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.feature_normals must span a volume frame; got a rank-deficient normal bank"
        )
    second = normals[transverse[0]] - np.sum(normals[transverse[0]] * first) * first
    second /= np.linalg.norm(second)
    initial = np.stack((first, second, np.cross(first, second)))
    method = ProjectedLBFGS()
    with family_execution_owner.field_work_allowance(
        method, normals.shape[0], schedule.maximum_field_steps, scope=scope
    ) as field_ticket:
        optimization = minimize(
            _frame_energy,
            jnp.zeros((3,), dtype=jnp.float64),
            method=method,
            args=(jnp.asarray(initial), jnp.asarray(normals)),
            bounds=Bounds(
                jnp.full((3,), -np.pi / 4.0, dtype=jnp.float64),
                jnp.full((3,), np.pi / 4.0, dtype=jnp.float64),
            ),
            termination=OptimizationTermination(
                maximum_steps=schedule.maximum_field_steps
            ),
        )
        family_execution_owner.charge_field_work(
            field_ticket, optimization, normals.shape[0]
        )
    angles = optimization.parameters
    if not isinstance(angles, Array):
        raise RuntimeError(
            f"hex.frame.parameters must be an array; got {type(optimization.parameters).__name__}"
        )
    basis = np.asarray(
        jax.device_get(jnp.asarray(initial) @ frame_rotation(angles)), dtype=np.float64
    )
    projected = normals @ basis.T
    axes = np.argmax(np.abs(projected), axis=1).astype(np.int64)
    residuals = 1.0 - np.max(np.abs(projected), axis=1)
    solve = _dependency_phydrax_linalg__small_batched.solve_small_linear(
        _dependency_phydrax_linalg__small_batched.SmallLinearSolvePlan(3),
        np.broadcast_to(basis.T, (complex_.vertices.shape[0], 3, 3)),
        complex_.vertices,
    )
    logical, successful = jax.device_get((solve.value, solve.successful))
    if not np.all(np.asarray(successful, dtype=np.bool_)):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.frame.pullback must have successful native solves; got singular or ill-conditioned basis"
        )
    logical_ = np.asarray(logical, dtype=np.float64)
    # Predicate postclassification, not epsilon snapping: every admitted source
    # polygon must lie on one represented coordinate plane of the solved frame.
    for row, loop in enumerate(loops):
        if np.any(logical_[loop, axes[row]] != logical_[loop[0], axes[row]]):
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.frame.patch must lie on a coordinate plane; got a non-grid planar patch at row {row}"
            )
    incidence = [set() for _ in range(complex_.vertices.shape[0])]
    for row, loop in enumerate(loops):
        for vertex in loop.tolist():
            incidence[vertex].add(int(axes[row]))
    singular = np.asarray(
        [row for row, values in enumerate(incidence) if len(values) != 3], dtype=np.int64
    )
    field = HexFrameField(
        basis,
        normals,
        axes,
        residuals,
        singular,
        solve,
        optimization,
        complex_.complex_id,
        canonical_fingerprint(
            {
                "kind": "hex-frame-field",
                "source": complex_.complex_id,
                "schedule": schedule.schedule_id,
                "basis": array_tree_fingerprint(basis),
                "patch_axes": array_tree_fingerprint(axes),
            }
        ),
    )
    record_elapsed(record_phase, "frame_solve", measurement_started)
    return field, logical_


def _grid_vertex_coordinates(
    vertices: np.ndarray,
    knots: tuple[np.ndarray, ...],
    subdivisions: int,
    basis: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    # Logical integer addresses determine vertex identity. Interpolation places vertices;
    # it never decides identity or merges two nearby geometry strata.
    logical = _dependency_phydrax_meshing__host_banks.family_host_array(
        vertices.shape, np.float64, scope=scope
    )
    for axis in range(3):
        position = vertices[:, axis] / subdivisions
        lower = np.floor(position).astype(np.int64)
        capped = np.minimum(lower, knots[axis].size - 2)
        fraction = position - capped
        logical[:, axis] = knots[axis][capped] + fraction * (
            knots[axis][capped + 1] - knots[axis][capped]
        )
    physical = _dependency_phydrax_meshing__host_banks.family_host_array(
        vertices.shape, np.float64, scope=scope
    )
    np.matmul(logical, basis, out=physical)
    return physical


def _grid_face_sources(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    source_logical: np.ndarray,
    grid: np.ndarray,
    faces: np.ndarray,
    frame: HexFrameField,
    knots: tuple[np.ndarray, ...],
    subdivisions: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify face-center trim membership after a boundary-plane exclusion proof."""
    face_points = np.mean(grid[faces], axis=1)
    source_faces = _dependency_phydrax_meshing__host_banks.family_host_array(
        (faces.shape[0],), np.int64, scope=scope
    )
    source_faces.fill(-1)
    source_polygons = _dependency_phydrax_meshing__host_banks.family_host_array(
        (faces.shape[0],), np.int64, scope=scope
    )
    source_polygons.fill(-1)
    for polygon in range(complex_.polygon_facets.size):
        loop = complex_.polygon_vertices[
            complex_.polygon_offsets[polygon] : complex_.polygon_offsets[polygon + 1]
        ]
        axis = int(frame.patch_axes[polygon])
        plane = source_logical[loop[0], axis]
        index = np.flatnonzero(knots[axis] == plane)
        if index.size != 1:
            raise RuntimeError(
                "hex.source.patch must retain a coordinate-plane representation; got a varying grid-plane coordinate"
            )
        logical_plane = index[0] * subdivisions
        on_plane = np.all(grid[faces, axis] == logical_plane, axis=1)
        rows = np.flatnonzero(on_plane)
        if rows.size == 0:
            continue
        remaining = [coordinate for coordinate in range(3) if coordinate != axis]
        ordinal = np.stack(
            [
                np.searchsorted(knots[coordinate], source_logical[loop, coordinate])
                * subdivisions
                for coordinate in remaining
            ],
            axis=1,
        ).astype(np.float64)
        trim = _dependency_phydrax_geometry__atlas.TrimDomain(
            _dependency_phydrax_geometry__atlas.PolygonTrimLoop(ordinal)
        )
        classified = trim.classify(face_points[rows][:, remaining])
        if not np.all(np.asarray(classified.resolved)):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.source.trim must resolve integer-grid membership; got unresolved trim queries"
            )
        chosen = rows[np.asarray(classified.inside) | np.asarray(classified.boundary)]
        facet = complex_.polygon_facets[polygon]
        occupied = source_faces[chosen]
        if np.any((occupied >= 0) & (occupied != facet)):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.face_sources must assign one source facet per face; got incompatible source facets"
            )
        source_faces[chosen] = facet
        source_polygons[chosen] = polygon
    return source_faces, source_polygons


class _GridSource(StrictModule):
    mesh: CellMesh
    cell_regions: np.ndarray
    numeric_version: str = eqx.field(static=True)
    mapped: bool = eqx.field(static=True)


class _FeatureGridKnots(StrictModule):
    knots: tuple[np.ndarray, ...]


class _GridLayout(StrictModule):
    knots: tuple[np.ndarray, ...]
    sizes: np.ndarray
    depth: int = eqx.field(static=True)
    required: int = eqx.field(static=True)
    subdivisions: int = eqx.field(static=True)


class _IntegerGridTopology(StrictModule):
    coordinates: np.ndarray
    cells: np.ndarray
    centers: np.ndarray
    integer_vertices: np.ndarray
    integer_cells: np.ndarray
    offsets: np.ndarray
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    closure_rounds: int = eqx.field(static=True)


class _GridClassification(StrictModule):
    regions: np.ndarray


class _RetainedIntegerGrid(StrictModule):
    mesh: CellMesh
    cells: np.ndarray
    regions: np.ndarray
    coordinates: np.ndarray
    integer_vertices: np.ndarray
    integer_cells: np.ndarray


def _resolve_grid_source(
    construction: _dependency_phydrax_meshing__volume_generation.VolumeConstruction
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    /,
) -> _GridSource:
    match construction:
        case _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain():
            return _GridSource(
                construction.reference_mesh,
                construction.cell_regions,
                construction.source_revision,
                True,
            )
        case _dependency_phydrax_meshing__volume_generation.VolumeConstruction():
            return _GridSource(
                construction.mesh,
                construction.cell_regions,
                construction.mesh.numeric_version,
                False,
            )
        case _:
            raise TypeError(
                f"hex.construction must be VolumeConstruction or MappedReferenceDomain; got {type(construction).__name__}"
            )


def _feature_grid_knots(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    source: _GridSource,
    frame: HexFrameField,
    logical: np.ndarray,
    /,
) -> _FeatureGridKnots:
    if source.mapped:
        root_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
        pullback = _dependency_phydrax_linalg__small_batched.solve_small_linear(
            _dependency_phydrax_linalg__small_batched.SmallLinearSolvePlan(3),
            np.broadcast_to(frame.basis.T, (root_points.shape[0], 3, 3)),
            root_points,
        )
        pulled, successful = jax.device_get((pullback.value, pullback.successful))
        if not np.all(successful):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.source.pullback must have successful native solves; got failed source-reference chart pullback"
            )
        breaks = np.concatenate((logical, np.asarray(pulled, dtype=np.float64)))
    else:
        breaks = logical
    knots = tuple(np.unique(breaks[:, axis]) for axis in range(3))
    if any(values.size < 2 for values in knots):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.knots must have at least two knots per volume axis; got {tuple(values.size for values in knots)}"
        )
    # Only physical facet-group boundary edges impose feature chains. Internal
    # polygon triangulation diagonals of the same facet cancel topologically.
    edge_owners: dict[tuple[int, int, int], int] = {}
    for polygon, facet in enumerate(complex_.polygon_facets.tolist()):
        loop = complex_.polygon_vertices[
            complex_.polygon_offsets[polygon] : complex_.polygon_offsets[polygon + 1]
        ].tolist()
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            key = (facet, min(a, b), max(a, b))
            edge_owners[key] = edge_owners.get(key, 0) + 1
    for (_, a, b), count in edge_owners.items():
        if count == 1 and np.count_nonzero(logical[a] != logical[b]) != 1:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.source.feature_chain must follow integer-grid edges; got a diagonal block-face cut from {a} to {b}"
            )
    for a, b in complex_.segments.tolist():
        if np.count_nonzero(logical[a] != logical[b]) != 1:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.source.segments must follow integer-grid edges; got a diagonal segment from {a} to {b}"
            )
    return _FeatureGridKnots(knots)


def _prepare_grid_layout(
    knots: tuple[np.ndarray, ...],
    schedule: NativeHexGridSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    target_size: float,
    /,
    *,
    scope: NativeExecutionScope,
) -> _GridLayout:
    smallest = min(float(np.min(np.diff(values))) for values in knots)
    placement_size = min(target_size, smallest * schedule.maximum_aspect_ratio)
    axis_counts: list[list[int]] = []
    for values in knots:
        counts: list[int] = []
        axis_size = 0
        for first, last in zip(values[:-1], values[1:], strict=True):
            if placement_size < (last - first) / (1 << schedule.maximum_depth):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.grid.interval must fit the maximum dyadic depth; got interval width {last - first} at placement size {placement_size}",
                    resource=True,
                )
            count = max(1, int(np.ceil((last - first) / placement_size)))
            axis_size += count
            if axis_size > (1 << schedule.maximum_depth):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.grid.axis_size must fit the maximum dyadic depth; got {axis_size}",
                    resource=True,
                )
            counts.append(count)
        axis_counts.append(counts)
    subdivisions = 1
    sizes = np.asarray([sum(counts) for counts in axis_counts], dtype=np.int64)
    depth = max(
        1,
        max(
            int(value).bit_length() - (1 if int(value) & (int(value) - 1) == 0 else 0)
            for value in sizes
        ),
    )
    if depth > schedule.maximum_depth:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.depth must be at most {schedule.maximum_depth}; got {depth}",
            resource=True,
        )
    required = (
        (1 << depth) ** 3
        if schedule.route == "balanced_grid"
        else prod(int(value) for value in sizes)
    )
    family_execution_owner.check_family_limits(
        limits, required, 0, required * 8, required * 1024, required * 32
    )
    refined_knots = []
    for values, counts in zip(knots, axis_counts, strict=True):
        refined_axis = _dependency_phydrax_meshing__host_banks.family_host_array(
            (sum(counts) + 1,), np.float64, scope=scope
        )
        offset = 0
        for first, last, count in zip(values[:-1], values[1:], counts, strict=True):
            refined_axis[offset : offset + count] = (
                first + (last - first) * np.arange(count, dtype=np.float64) / count
            )
            offset += count
        refined_axis[-1] = values[-1]
        refined_knots.append(refined_axis)
    knots = tuple(refined_knots)
    return _GridLayout(knots, sizes, depth, required, subdivisions)


class _IntegerCellExtraction(StrictModule):
    cells: np.ndarray
    integer_vertices: np.ndarray
    integer_cells: np.ndarray
    offsets: np.ndarray
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    closure_rounds: int = eqx.field(static=True)


def _extract_balanced_cells(
    layout: _GridLayout,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    logical: np.ndarray,
    /,
) -> _IntegerCellExtraction:
    knots, sizes, depth, subdivisions = (
        layout.knots,
        layout.sizes,
        layout.depth,
        layout.subdivisions,
    )
    octree_prefixes = np.empty((0,), dtype=np.uint64)
    octree_levels = np.empty((0,), dtype=np.int64)
    closure_rounds = 0
    address = MortonAddressPlan(
        (0.0, 0.0, 0.0), tuple(float(1 << depth) for _ in range(3)), depth
    )
    # Source feature points seed full fine-level paths; octree balance
    # adds the neighbor paths. Refinement is owned by spatial, not meshing.
    feature = np.stack(
        [
            np.searchsorted(knots[axis], logical[:, axis]) * subdivisions
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.int64)
    feature = np.minimum(feature, (1 << depth) - 1)
    codes = morton_encode_integer_host(feature, depth)
    refined = [
        np.unique(codes >> np.uint64(3 * (depth - level))) for level in range(depth)
    ]
    octree_prefixes, octree_levels, lower = refined_octree_leaves(
        address, refined, balanced=True
    )
    # Independent 1:8 closure: a coarse face cannot match a fine neighbor's
    # four faces. Refining all leaf paths to the selected depth guarantees
    # the same four-child face template on both sides, including cycles.
    for level in range(depth):
        all_corners = np.asarray(
            list(product(range(1 << level), repeat=3)), dtype=np.int64
        ) * (1 << (depth - level))
        refined[level] = morton_encode_integer_host(all_corners, depth) >> np.uint64(
            3 * (depth - level)
        )
        closure_rounds += 1
    octree_prefixes, octree_levels, lower = refined_octree_leaves(
        address, refined, balanced=True
    )
    active_box = np.all(lower < sizes, axis=1)
    integer_cells = lower[active_box].astype(np.int64)
    offsets = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.int64,
    )
    parent_corners = np.unique((integer_cells // 2) * 2, axis=0)
    grid_offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int64)
    integer_vertices, inverse = np.unique(
        (parent_corners[:, None, :] + grid_offsets[None, :, :]).reshape((-1, 3)),
        axis=0,
        return_inverse=True,
    )
    nodes = inverse.reshape((-1, 27)).astype(np.int64)
    cells = _owner_phydrax__meshcore.subdivide_hex_grid(
        nodes, integer_vertices.shape[0], maximum_cells=limits.maximum_cells
    )
    child_corners = (
        parent_corners[:, None, :]
        + np.asarray(list(product(range(2), repeat=3)), dtype=np.int64)[None, :, :]
    ).reshape((-1, 3))
    active = np.all(child_corners < sizes, axis=1)
    cells, integer_cells = cells[active], child_corners[active]
    return _IntegerCellExtraction(
        cells,
        integer_vertices,
        integer_cells,
        offsets,
        octree_prefixes,
        octree_levels,
        closure_rounds,
    )


def _extract_frame_cells(layout: _GridLayout, /) -> _IntegerCellExtraction:
    sizes = layout.sizes
    octree_prefixes = np.empty((0,), dtype=np.uint64)
    octree_levels = np.empty((0,), dtype=np.int64)
    closure_rounds = 0
    integer_cells = np.asarray(
        list(product(*(range(int(value)) for value in sizes))), dtype=np.int64
    )
    offsets = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.int64,
    )
    integer_vertices, inverse = np.unique(
        (integer_cells[:, None, :] + offsets[None, :, :]).reshape((-1, 3)),
        axis=0,
        return_inverse=True,
    )
    cells = inverse.reshape((-1, 8))
    return _IntegerCellExtraction(
        cells,
        integer_vertices,
        integer_cells,
        offsets,
        octree_prefixes,
        octree_levels,
        closure_rounds,
    )


def _extract_integer_grid(
    layout: _GridLayout,
    schedule: NativeHexGridSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    frame: HexFrameField,
    logical: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> _IntegerGridTopology:
    match schedule.route:
        case "balanced_grid":
            extracted = _extract_balanced_cells(layout, limits, logical)
        case "frame_grid":
            extracted = _extract_frame_cells(layout)
        case _:
            raise ValueError(
                f"hex.schedule.route must name a supported grid route; got {schedule.route}"
            )
    integer_vertices, cells = extracted.integer_vertices, extracted.cells
    knots, subdivisions = layout.knots, layout.subdivisions
    coordinates = _grid_vertex_coordinates(
        integer_vertices, knots, subdivisions, frame.basis, scope=scope
    )
    centers = np.mean(coordinates[cells], axis=1)
    return _IntegerGridTopology(
        coordinates,
        cells,
        centers,
        integer_vertices,
        extracted.integer_cells,
        extracted.offsets,
        extracted.octree_prefixes,
        extracted.octree_levels,
        extracted.closure_rounds,
    )


def _classify_mapped_grid(
    source: _GridSource,
    topology: _IntegerGridTopology,
    schedule: NativeHexGridSchedule,
    regions: np.ndarray,
    /,
) -> _GridClassification:
    coordinates, cells, offsets = topology.coordinates, topology.cells, topology.offsets
    centers = topology.centers
    root_cells = _dependency_phydrax_meshing__host_banks.entities(source.mesh, 3)
    root_corners = np.asarray(source.mesh.coordinates, dtype=np.float64)[root_cells]
    root_matrices = np.stack(
        (
            root_corners[:, 1] - root_corners[:, 0],
            root_corners[:, 3] - root_corners[:, 0],
            root_corners[:, 4] - root_corners[:, 0],
        ),
        axis=-1,
    )
    diagonal = np.diagonal(root_matrices, axis1=1, axis2=2)
    expected_matrices = np.zeros_like(root_matrices)
    expected_matrices[:, np.arange(3), np.arange(3)] = diagonal
    expected = root_corners[:, :1] + offsets[None, :, :] * diagonal[:, None, :]
    if (
        np.any(diagonal <= 0.0)
        or not np.array_equal(root_matrices, expected_matrices)
        or not np.array_equal(expected, root_corners)
    ):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.roots must be positive axis-aligned affine reference charts; got nonpositive or non-axis-aligned root corners"
        )
    root_lower, root_upper = root_corners[:, 0], root_corners[:, 6]
    root_bvh = prepare_bvh(root_lower, root_upper, dtype=np.float64)
    candidates, valid, complete = jax.device_get(
        point_select_leaf_items(
            centers,
            bvh=root_bvh,
            maximum_candidates=schedule.maximum_location_candidates,
        )
    )
    if not np.all(complete):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.classification must fit the candidate capacity; got incomplete root candidate queries",
            resource=True,
        )
    for row in range(cells.shape[0]):
        values = candidates[row][valid[row]]
        containing = [
            parent
            for parent in values.tolist()
            if np.all(coordinates[cells[row]] >= root_lower[parent])
            and np.all(coordinates[cells[row]] <= root_upper[parent])
        ]
        if len(containing) > 1:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.parent must have one containing reference root; got {len(containing)} roots at cell {row}"
            )
        if containing:
            regions[row] = source.cell_regions[containing[0]]
        elif values.size:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.cell must stay within one source-reference chart; got a chart-boundary crossing at cell {row}"
            )
    return _GridClassification(regions)


def _classify_volume_grid(
    source: _GridSource,
    topology: _IntegerGridTopology,
    schedule: NativeHexGridSchedule,
    regions: np.ndarray,
    /,
) -> _GridClassification:
    centers = topology.centers
    space = FiniteElementPlan(
        source.mesh,
        FiniteElementFieldSpec("grid-region", lagrange_element("tetrahedron", 1)),
    ).prepare()
    locator = _dependency_phydrax_discretization__simplicial_locator.PreparedSimplicialCellLocator(
        _dependency_phydrax_discretization_fem__cell_map.PreparedFiniteElementCellMap(
            space, 0
        ),
        space.default_runtime.coordinates,
        _dependency_phydrax_discretization__simplicial_locator.SimplicialLocationPolicy(
            schedule.maximum_location_candidates, 16, 1
        ),
    )
    located = locator.locate(centers)
    statuses, inside, candidates = jax.device_get(
        (located.status, located.inside, located.candidate_cells)
    )
    if np.any(
        (
            statuses
            != _dependency_phydrax_discretization__simplicial_locator.CellLocationStatus.LOCATED
        )
        & (
            statuses
            != _dependency_phydrax_discretization__simplicial_locator.CellLocationStatus.OUTSIDE
        )
    ):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.region_location must resolve within the candidate and inverse-map workset; got unresolved location statuses",
            resource=True,
        )
    for row in np.flatnonzero(inside).tolist():
        containing = np.asarray(candidates[row], dtype=np.int64)
        containing = containing[containing >= 0]
        values = source.cell_regions[containing]
        if values.size == 0 or np.any(values != values[0]):
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.regions must agree for every containing source cell; got {values.tolist()} at cell {row}"
            )
        regions[row] = values[0]
    return _GridClassification(regions)


def _retain_integer_grid(
    topology: _IntegerGridTopology,
    classified: _GridClassification,
    source: _GridSource,
    layout: _GridLayout,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _RetainedIntegerGrid:
    cells, integer_cells = topology.cells, topology.integer_cells
    coordinates, integer_vertices = topology.coordinates, topology.integer_vertices
    regions, required, numeric_version = (
        classified.regions,
        layout.required,
        source.numeric_version,
    )
    kept = regions >= 0
    cells, regions, integer_cells = cells[kept], regions[kept], integer_cells[kept]
    if cells.shape[0] == 0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.grid.cells must cover a source region; got zero retained cells"
        )
    used, compact = np.unique(cells, return_inverse=True)
    cells = compact.reshape(cells.shape)
    family_execution_owner.check_family_limits(
        limits, cells.shape[0], used.size, cells.size, required * 1024, required * 32
    )
    if used.size != coordinates.shape[0]:
        retained_coordinates = _dependency_phydrax_meshing__host_banks.family_host_array(
            (used.size, 3), coordinates.dtype, scope=scope
        )
        retained_vertices = _dependency_phydrax_meshing__host_banks.family_host_array(
            (used.size, 3), integer_vertices.dtype, scope=scope
        )
        np.take(coordinates, used, axis=0, out=retained_coordinates)
        np.take(integer_vertices, used, axis=0, out=retained_vertices)
        coordinates, integer_vertices = retained_coordinates, retained_vertices
    # Count retained entities, not the bounding box: cavities and
    # disconnected components may discard most candidate cells. Admission must
    # nevertheless precede construction of the connectivity owner.
    reference_topology = reference_cell_topology("hexahedron")
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = np.asarray(
            reference_topology.entities[dimension], dtype=np.int64
        )
        entity_rows = np.sort(
            cells[:, local_entities].reshape((-1, local_entities.shape[1])), axis=1
        )
        count = np.unique(entity_rows, axis=0).shape[0]
        if count > bound:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.entity_count must fit dimension-{dimension} limit {bound}; got {count}",
                resource=True,
            )
    del entity_rows
    mesh = CellMesh(
        coordinates,
        (CellBlock("integer_blocks", "hexahedron", cells),),
        numeric_version=numeric_version,
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            mesh, budget=budget
        )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.validity must certify positive cell determinant enclosures; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
        )
    return _RetainedIntegerGrid(
        mesh, cells, regions, coordinates, integer_vertices, integer_cells
    )


@checked
def generate_integer_grid_hexes(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    construction: _dependency_phydrax_meshing__volume_generation.VolumeConstruction
    | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    schedule: NativeHexGridSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    target_size: float,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> GridHexConstruction:
    """Decompose polycube patches and extract a closed integer grid.

    Boundary and material polygons lie on grid planes, so the source region is
    constant on each open cell. Balanced-grid closure uses the spatial octree
    and bounded repeated 1:8 templates; frame-grid extraction retains the
    feature-plane layout. The selected route does not change after refusal.
    """
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise ValueError(
            f"hex.target_size must be positive and finite; got {target_size}"
        )
    frame, logical = prepare_hex_frame(
        complex_, schedule, record_phase=record_phase, scope=scope
    )
    topology_started = phase_started(record_phase)
    source = _resolve_grid_source(construction)
    features = _feature_grid_knots(complex_, source, frame, logical)
    layout = _prepare_grid_layout(
        features.knots, schedule, limits, target_size, scope=scope
    )
    topology = _extract_integer_grid(
        layout, schedule, limits, frame, logical, scope=scope
    )
    record_elapsed(record_phase, "topology_construction", topology_started)
    classification_started = phase_started(record_phase)
    regions = _dependency_phydrax_meshing__host_banks.family_host_array(
        (topology.cells.shape[0],), np.int64, scope=scope
    )
    regions.fill(-1)
    classification_ticket = scope.batch(
        topology.cells.shape[0], queries_upper=topology.cells.shape[0]
    )
    with classification_ticket:
        classification_ticket.consume(
            topology.cells.shape[0], geometry_queries=topology.cells.shape[0]
        )
        if source.mapped:
            classified = _classify_mapped_grid(source, topology, schedule, regions)
        else:
            classified = _classify_volume_grid(source, topology, schedule, regions)
    record_elapsed(record_phase, "region_classification", classification_started)
    retained_grid = _retain_integer_grid(
        topology, classified, source, layout, limits, scope=scope, budget=budget
    )
    mesh, cells, regions = retained_grid.mesh, retained_grid.cells, retained_grid.regions
    coordinates, integer_vertices = (
        retained_grid.coordinates,
        retained_grid.integer_vertices,
    )
    integer_cells = retained_grid.integer_cells
    knots, subdivisions, required = layout.knots, layout.subdivisions, layout.required
    octree_prefixes, octree_levels = topology.octree_prefixes, topology.octree_levels
    closure_rounds = topology.closure_rounds
    face_sources, face_polygons = _grid_face_sources(
        complex_,
        logical,
        integer_vertices,
        _dependency_phydrax_meshing__host_banks.entities(mesh, 2),
        frame,
        knots,
        subdivisions,
        scope=scope,
    )
    retained = (
        coordinates,
        cells,
        regions,
        integer_vertices,
        integer_cells,
        face_sources,
        face_polygons,
        octree_prefixes,
        octree_levels,
        *knots,
        frame.basis,
        frame.normals,
        frame.patch_axes,
        frame.alignment_residuals,
        frame.singular_vertices,
    )
    data_bytes = sum(array.nbytes for array in retained)
    if data_bytes > limits.maximum_data_bytes:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.data_bytes must fit maximum_data_bytes={limits.maximum_data_bytes}; got {data_bytes}",
            resource=True,
        )
    return GridHexConstruction(
        mesh,
        regions,
        integer_vertices,
        integer_cells,
        face_sources,
        face_polygons,
        frame,
        knots,
        octree_prefixes,
        octree_levels,
        subdivisions,
        closure_rounds,
        required * 32,
        canonical_fingerprint(
            {
                "kind": "integer-grid-hex-construction",
                "source": complex_.complex_id,
                "schedule": schedule.schedule_id,
                "mesh": mesh.mesh_id,
                "regions": array_tree_fingerprint(regions),
            }
        ),
    )


class MappedGridHexConstruction(StrictModule):
    """Store curved restrictions with the declared source coefficient bank."""

    mesh: CellMesh
    reference_mesh: CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    cell_regions: np.ndarray
    source_parent_rows: np.ndarray
    target_reference_rows: np.ndarray
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    mapped_id: str = eqx.field(static=True)


@checked
def mapped_source_lipschitz_bound(
    reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> float:
    """Bound physical placement in source reference charts.

    Affine reference-frame solves and source Jacobians use exact arithmetic.
    Interval bounds cover the source expression throughout every reference cell.
    """
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        elements, routes, _ = source_geometry.resolve(
            reference_mesh, budget=budget, exact_source_prepared=False
        )
        corners = np.asarray(reference_mesh.coordinates, dtype=np.float64)
        source_values = source_geometry.source_coordinates(budget=budget)
        root_count = sum(block.cell_count for block in reference_mesh.blocks)
        family_execution_owner.count_geometry_queries(
            root_count, work_units=9 * root_count, budget=budget
        )
        host_routes = jax.device_get(routes)
        upper = 0.0
        for block, element, route in zip(
            reference_mesh.blocks, elements, host_routes, strict=True
        ):
            cells = np.asarray(block.vertices, dtype=np.int64)
            block_routes = np.asarray(route, dtype=np.int64)
            for row in range(cells.shape[0]):
                matrix, _ = (
                    _dependency_phydrax_meshing__reference_root_composition.affine_reference_root_frame(
                        corners[cells[row]], block.cell_kind
                    )
                )
                inverse = solve_exact(
                    matrix,
                    [
                        [Fraction(int(axis == column)) for column in range(3)]
                        for axis in range(3)
                    ],
                )
                polynomials = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
                    element,
                    tuple(source_values[index] for index in block_routes[row]),
                    budget=budget,
                )
                if polynomials is None:
                    raise _dependency_phydrax_meshing__host_banks.failure(
                        f"hex.source.element must provide rational coordinate polynomials; got {element.element_id}"
                    )
                bounds = tuple(
                    tuple(
                        _polynomial_magnitude(
                            _dependency_phydrax_discretization__coordinate_enclosure.add(
                                _dependency_phydrax_discretization__coordinate_enclosure.add(
                                    _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                        _dependency_phydrax_discretization__coordinate_enclosure.derivative(
                                            value, 0, budget=budget
                                        ),
                                        inverse[0][axis],
                                        budget=budget,
                                    ),
                                    _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                        _dependency_phydrax_discretization__coordinate_enclosure.derivative(
                                            value, 1, budget=budget
                                        ),
                                        inverse[1][axis],
                                        budget=budget,
                                    ),
                                    budget=budget,
                                ),
                                _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                    _dependency_phydrax_discretization__coordinate_enclosure.derivative(
                                        value, 2, budget=budget
                                    ),
                                    inverse[2][axis],
                                    budget=budget,
                                ),
                                budget=budget,
                            ),
                            budget=budget,
                        )
                        for axis in range(3)
                    )
                    for value in polynomials
                )
                row_sum = max(sum(row, Fraction(0)) for row in bounds)
                column_sum = max(
                    sum((row[axis] for row in bounds), Fraction(0)) for axis in range(3)
                )
                upper = max(upper, _sqrt_upper(row_sum * column_sum))
    if not np.isfinite(upper) or upper <= 0.0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.lipschitz_bound must be positive and finite; got {upper}"
        )
    return upper


@checked
def source_block_interval_counts(
    reference_mesh: CellMesh,
    physical_lipschitz_bound: float,
    target_size: float,
    maximum_depth: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    """Size source-local axes by rational reference-edge norm bounds."""
    cells = _dependency_phydrax_meshing__host_banks.entities(reference_mesh, 3)
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    counts = _dependency_phydrax_meshing__host_banks.family_host_array(
        (cells.shape[0], 3), np.int64, scope=scope
    )
    for root, vertices in enumerate(cells):
        matrix, _ = (
            _dependency_phydrax_meshing__reference_root_composition.affine_reference_root_frame(
                points[vertices], "hexahedron"
            )
        )
        for axis in range(3):
            length = _sqrt_upper(sum((row[axis] ** 2 for row in matrix), Fraction(0)))
            ratio = (
                Fraction(physical_lipschitz_bound)
                * Fraction(length)
                / Fraction(target_size)
            )
            count = max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
            if count > 1 << maximum_depth:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.source.axis_count must fit the maximum integer depth; got {count} at root {root}, axis {axis}",
                    resource=True,
                )
            counts[root, axis] = count
    return counts


class _MappedSourcePreparation(StrictModule):
    elements: tuple[CellGeometryElement, ...]
    routes: tuple[np.ndarray, ...]
    values: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank
    cells: np.ndarray
    regions: np.ndarray
    root_corners: np.ndarray
    reference_mesh: CellMesh
    reference_cells: np.ndarray
    points: np.ndarray
    chart: FiniteElementSpec


class _MappedRestrictionCharts(StrictModule):
    parents: np.ndarray
    numerators: np.ndarray
    denominators: np.ndarray


class _MappedRestrictionGroups(StrictModule):
    block_of_root: np.ndarray
    local_of_root: np.ndarray
    groups: dict[str, list[int]]


class _MappedGeometryPreparation(StrictModule):
    mesh: CellMesh
    reference_mesh: CellMesh
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    order: np.ndarray


def _prepare_mapped_source(
    grid: GridHexConstruction | SourceBlockIntegerGrid,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_cell_regions: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedSourcePreparation:
    source_elements, source_routes, _ = source_geometry.resolve(
        source_reference_mesh, budget=budget, exact_source_prepared=False
    )
    source_values = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        source_geometry, budget=budget
    )
    host_routes = tuple(
        np.asarray(route, dtype=np.int64) for route in jax.device_get(source_routes)
    )
    source_cells = np.concatenate(
        [
            np.asarray(block.vertices, dtype=np.int64)
            for block in source_reference_mesh.blocks
        ]
    )
    source_regions = np.asarray(source_cell_regions, dtype=np.int64)
    if source_regions.shape != (source_cells.shape[0],):
        raise ValueError(
            f"hex.source.regions must match the reference cell count; got shape {source_regions.shape} for {source_cells.shape[0]} cells"
        )
    source_points = np.asarray(source_reference_mesh.coordinates, dtype=np.float64)
    root_corners = source_points[source_cells]
    roots = np.stack(
        (
            root_corners[:, 1] - root_corners[:, 0],
            root_corners[:, 3] - root_corners[:, 0],
            root_corners[:, 4] - root_corners[:, 0],
        ),
        axis=-1,
    )
    offsets = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.float64,
    )
    represented = root_corners[:, :1] + np.sum(
        roots[:, None, :, :] * offsets[None, :, None, :], axis=-1
    )
    if not np.array_equal(represented, root_corners):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.roots must be affine hexahedral charts; got non-affine root corners"
        )
    reference_mesh = grid.mesh
    reference_cells = np.asarray(reference_mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)

    chart = _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
        "hexahedron", 1
    )
    return _MappedSourcePreparation(
        source_elements,
        host_routes,
        source_values,
        source_cells,
        source_regions,
        root_corners,
        reference_mesh,
        reference_cells,
        points,
        chart,
    )


def _locate_mapped_charts(
    prepared: _MappedSourcePreparation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedRestrictionCharts:
    points, reference_cells, root_corners, chart = (
        prepared.points,
        prepared.reference_cells,
        prepared.root_corners,
        prepared.chart,
    )
    target_corners = points[reference_cells]
    bvh = prepare_bvh(
        np.min(root_corners, axis=1),
        np.max(root_corners, axis=1),
        dtype=np.float64,
    )
    family_execution_owner.count_geometry_queries(
        reference_cells.shape[0],
        work_units=reference_cells.shape[0],
        budget=budget,
    )
    candidates, valid, complete = jax.device_get(
        point_select_leaf_items(
            np.mean(target_corners, axis=1), bvh=bvh, maximum_candidates=16
        )
    )
    if not np.all(complete):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.root_lookup must fit the candidate capacity; got incomplete root queries",
            resource=True,
        )
    parent_rows = _dependency_phydrax_meshing__host_banks.family_host_array(
        (reference_cells.shape[0],), np.int64, scope=scope
    )
    parent_rows.fill(-1)
    numerators = _dependency_phydrax_meshing__host_banks.family_host_array(
        (reference_cells.shape[0], chart.local_dof_count, 3),
        np.int64,
        scope=scope,
    )
    denominators = _dependency_phydrax_meshing__host_banks.family_host_array(
        numerators.shape, np.int64, scope=scope
    )
    for row in range(reference_cells.shape[0]):
        containing = []
        for slot in np.flatnonzero(valid[row]).tolist():
            parent = int(candidates[row, slot])
            controls = _dependency_phydrax_meshing__reference_root_composition.exact_reference_chart_controls(
                target_corners[row],
                root_corners[parent],
                "hexahedron",
                chart,
                budget=budget,
            )
            if controls is not None:
                containing.append((parent, controls))
        if len(containing) != 1:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.target.parent must have unique source-reference chart containment; got {len(containing)} containing roots at cell {row}"
            )
        parent, controls = containing[0]
        if any(values.dtype.kind not in "iu" for values in controls):
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.target.chart_controls must fit signed-64-bit rational banks; got {tuple(values.dtype for values in controls)}"
            )
        parent_rows[row] = parent
        numerators[row], denominators[row] = controls
    return _MappedRestrictionCharts(parent_rows, numerators, denominators)


def _resolve_mapped_charts(
    grid: GridHexConstruction | SourceBlockIntegerGrid,
    prepared: _MappedSourcePreparation,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedRestrictionCharts:

    reference_cells, chart, source_cells = (
        prepared.reference_cells,
        prepared.chart,
        prepared.cells,
    )
    match grid:
        case SourceBlockIntegerGrid():
            if (
                grid.source_frame.reference_mesh.mesh_id != source_reference_mesh.mesh_id
                or grid.source_frame.source_id
                != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    source_geometry
                )
            ):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.grid.source_frame must bind the source mesh and coordinate bank; got a different source frame binding"
                )
            parent_rows = grid.source_parent_rows
            numerators, denominators = grid.chart_numerators, grid.chart_denominators
            shape = (reference_cells.shape[0], chart.local_dof_count, 3)
            if (
                parent_rows.shape != (reference_cells.shape[0],)
                or np.any(parent_rows < 0)
                or np.any(parent_rows >= source_cells.shape[0])
                or numerators.shape != shape
                or denominators.shape != shape
            ):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.grid.chart_rows must match all target cells and source roots; got incompatible parent or control shapes"
                )
            if (
                np.any(denominators <= 0)
                or np.any(numerators < 0)
                or np.any(numerators > denominators)
            ):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.grid.chart_controls must lie within the reference root; got nonpositive denominators or ratios outside [0, 1]"
                )
            charts = _MappedRestrictionCharts(parent_rows, numerators, denominators)
        case GridHexConstruction():
            charts = _locate_mapped_charts(prepared, scope=scope, budget=budget)
        case _:
            raise TypeError(
                f"hex.grid must be GridHexConstruction or SourceBlockIntegerGrid; got {type(grid).__name__}"
            )
    if np.any(prepared.regions[charts.parents] != grid.cell_regions):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.grid.regions must agree with source-reference regions; got different extracted and parent region IDs"
        )
    return charts


def _group_mapped_restrictions(
    prepared: _MappedSourcePreparation,
    charts: _MappedRestrictionCharts,
    source_reference_mesh: CellMesh,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
) -> _MappedRestrictionGroups:
    source_elements = prepared.elements
    parent_rows, numerators, denominators = (
        charts.parents,
        charts.numerators,
        charts.denominators,
    )
    block_of_root = np.concatenate(
        [
            np.full((block.cell_count,), row, dtype=np.int64)
            for row, block in enumerate(source_reference_mesh.blocks)
        ]
    )
    local_of_root = np.concatenate(
        [
            np.arange(block.cell_count, dtype=np.int64)
            for block in source_reference_mesh.blocks
        ]
    )
    groups: dict[str, list[int]] = {}
    for row, parent in enumerate(parent_rows.tolist()):
        element = source_elements[block_of_root[parent]]
        key = canonical_fingerprint(
            {
                "source_element": element.element_id,
                "chart_numerators": array_tree_fingerprint(numerators[row]),
                "chart_denominators": array_tree_fingerprint(denominators[row]),
            }
        )
        groups.setdefault(f"chart:{key}", []).append(row)
    if len(groups) > limits.maximum_work_units:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.restriction_groups must fit maximum_work_units={limits.maximum_work_units}; got {len(groups)}",
            resource=True,
        )
    return _MappedRestrictionGroups(block_of_root, local_of_root, groups)


def _compose_mapped_geometry(
    prepared: _MappedSourcePreparation,
    charts: _MappedRestrictionCharts,
    grouped: _MappedRestrictionGroups,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedGeometryPreparation:
    source_elements, source_routes, source_values = (
        prepared.elements,
        prepared.routes,
        prepared.values,
    )
    source_cells, reference_mesh, reference_cells, points = (
        prepared.cells,
        prepared.reference_mesh,
        prepared.reference_cells,
        prepared.points,
    )
    chart = prepared.chart
    parent_rows, numerators, denominators = (
        charts.parents,
        charts.numerators,
        charts.denominators,
    )
    block_of_root, local_of_root, groups = (
        grouped.block_of_root,
        grouped.local_of_root,
        grouped.groups,
    )
    blocks, elements, dofs, parent_ids, parent_vertices = [], {}, {}, {}, {}
    source_ids = np.asarray(
        source_reference_mesh.topology.entities(3).entity_ids, dtype=np.int64
    )
    source_vertex_ids = np.asarray(
        source_reference_mesh.vertex_global_ids, dtype=np.int64
    )
    target_ids = np.asarray(
        reference_mesh.topology.entities(3).entity_ids, dtype=np.int64
    )
    physical_points = _dependency_phydrax_meshing__host_banks.family_host_array(
        points.shape, np.float64, scope=scope
    )
    physical_points.fill(np.nan)
    exact_points: dict[
        int,
        tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...],
    ] = {}
    family_execution_owner.count_geometry_queries(
        8 * reference_cells.shape[0],
        work_units=8 * reference_cells.shape[0],
        budget=budget,
    )
    order = []
    for name, rows_ in sorted(groups.items()):
        rows = np.asarray(rows_, dtype=np.int64)
        rows = rows[np.argsort(target_ids[rows], kind="stable")]
        order.extend(rows.tolist())
        parents = parent_rows[rows]
        prototype = rows[0]
        source_element = _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
            source_elements[block_of_root[parents[0]]], "mapped source"
        )
        element = _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
            source_element,
            chart,
            numerators[prototype],
            denominators[prototype],
            budget=budget,
        )
        route = np.stack(
            [
                np.asarray(source_routes[block_of_root[parent]], dtype=np.int64)[
                    local_of_root[parent]
                ]
                for parent in parents.tolist()
            ]
        )
        for cell, source_route in zip(reference_cells[rows].tolist(), route, strict=True):
            images = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                element,
                tuple(source_values[int(index)] for index in source_route),
                budget=budget,
            )
            if images is None:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.source.restriction must support rational corner evaluation; got {element.element_id}"
                )
            for vertex, image in zip(cell, images, strict=True):
                previous = exact_points.get(vertex)
                if previous is not None and previous != image:
                    raise _dependency_phydrax_meshing__host_banks.failure(
                        f"hex.source.restriction_traces must agree at shared coordinate vertices; got different images at vertex {vertex}"
                    )
                exact_points[vertex] = image
                physical_points[vertex] = (
                    _dependency_phydrax_discretization__coordinate_enclosure.rounded_point(
                        image
                    )
                )
        blocks.append(
            CellBlock(
                name, "hexahedron", reference_cells[rows], global_ids=target_ids[rows]
            )
        )
        elements[name], dofs[name] = element, route
        parent_ids[name] = source_ids[parents]
        parent_vertices[name] = source_vertex_ids[source_cells[parents]]
    order_ = np.asarray(order, dtype=np.int64)
    grouped_reference = CellMesh(
        points,
        tuple(blocks),
        vertex_global_ids=reference_mesh.vertex_global_ids,
        numeric_version=reference_mesh.numeric_version,
    )
    mesh = CellMesh(
        physical_points,
        tuple(blocks),
        vertex_global_ids=reference_mesh.vertex_global_ids,
        numeric_version=reference_mesh.numeric_version,
    )
    provenance = (
        _dependency_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
            _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source_geometry
            ),
            source_reference_mesh.topology_id,
            parent_ids,
            parent_vertices,
        )
    )
    geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec(
        elements,
        dofs,
        source_geometry.coordinates,
        exact_source=source_geometry.exact_source,
        restriction_source=provenance,
        budget=budget,
    )
    return _MappedGeometryPreparation(mesh, grouped_reference, geometry, order_)


@checked
def realize_mapped_grid_hexes(
    grid: GridHexConstruction | SourceBlockIntegerGrid,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_cell_regions: np.ndarray,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> MappedGridHexConstruction:
    """Restrict source maps to integer cells with rational chart controls.

    Containment and inverse coordinates use exact arithmetic. Nonbinary inverse
    ratios retain the source coefficients. Shared coordinate images must agree;
    physical embedding and reference-chart coverage certify independently.
    """
    if any(block.cell_kind != "hexahedron" for block in source_reference_mesh.blocks):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.blocks must contain only hexahedral reference roots; got {tuple(block.cell_kind for block in source_reference_mesh.blocks)}"
        )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        prepared = _prepare_mapped_source(
            grid,
            source_reference_mesh,
            source_geometry,
            source_cell_regions,
            budget=budget,
        )
        charts = _resolve_mapped_charts(
            grid,
            prepared,
            source_reference_mesh,
            source_geometry,
            scope=scope,
            budget=budget,
        )
        grouped = _group_mapped_restrictions(
            prepared, charts, source_reference_mesh, limits
        )
        realized = _compose_mapped_geometry(
            prepared,
            charts,
            grouped,
            source_reference_mesh,
            source_geometry,
            scope=scope,
            budget=budget,
        )
        mesh, grouped_reference, geometry, order_ = (
            realized.mesh,
            realized.reference_mesh,
            realized.geometry,
            realized.order,
        )
        parent_rows = charts.parents
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget
        )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.mapped.validity must certify physical cell maps; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
        )
    scaled, mean, aspect = exact_mapped_volume_quality(
        mesh, geometry, scope=scope, budget=budget
    )
    return MappedGridHexConstruction(
        mesh,
        grouped_reference,
        geometry,
        grid.cell_regions[order_],
        parent_rows[order_],
        order_,
        scaled,
        mean,
        aspect,
        canonical_fingerprint(
            {
                "kind": "mapped-grid-hex",
                "source_geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    source_geometry
                ),
                "source_topology": source_reference_mesh.topology_id,
                "mesh": mesh.mesh_id,
                "geometry": _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    geometry
                ),
            }
        ),
    )


class _MappedQualitySource(StrictModule):
    jacobian: tuple[
        tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...],
        ...,
    ]
    determinant_lower: Fraction
    domain: str = eqx.field(static=True)


class _MappedQualityBounds(StrictModule):
    scaled: float
    mean: float
    aspect: float


def _prepare_mapped_quality_source(
    element: CellGeometryElement,
    coordinates: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    route: np.ndarray,
    cell_kind: str,
    domain: str,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource:
    polynomials = (
        _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
            element,
            tuple(coordinates[int(index)] for index in route),
            budget=budget,
        )
    )
    if polynomials is None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.element must provide polynomial interval quality data; got {element.element_id}"
        )
    jacobian = _dependency_phydrax_discretization__coordinate_enclosure.physical_jacobian(
        polynomials, cell_kind, 3, budget=budget
    )
    if jacobian is None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.jacobian must support physical-reference interval evaluation; got unsupported cell kind {cell_kind}"
        )
    determinant = (
        _dependency_phydrax_discretization__coordinate_enclosure.determinant_polynomial(
            polynomials, cell_kind, 3, budget=budget
        )
    )
    if determinant is None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.determinant must support interval evaluation; got unsupported cell kind {cell_kind}"
        )
    determinant_lower = min(
        rational_coefficient(value, role="hex.quality.determinant_controls")
        for value in _dependency_phydrax_discretization__coordinate_enclosure.bernstein_coefficients(
            determinant, domain, 3, budget=budget
        )
    )
    if determinant_lower <= 0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.determinant_lower must be positive; got {determinant_lower}"
        )
    return _MappedQualitySource(jacobian, determinant_lower, domain)


def _enclose_mapped_cell_quality(
    source: _MappedQualitySource,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualityBounds:
    jacobian, determinant_lower, domain = (
        source.jacobian,
        source.determinant_lower,
        source.domain,
    )
    bounds = tuple(
        tuple(_polynomial_magnitude(value, domain, budget=budget) for value in row)
        for row in jacobian
    )
    column_squared = tuple(
        sum((row[axis] ** 2 for row in bounds), Fraction(0)) for axis in range(3)
    )
    norm_squared = sum(column_squared, Fraction(0))
    product_squared = column_squared[0] * column_squared[1] * column_squared[2]
    column_product = _sqrt_upper(product_squared)
    scaled = fraction_endpoint(
        determinant_lower / Fraction(column_product),
        upper=False,
        saturate=True,
    )
    root = float(determinant_lower) ** (2.0 / 3.0)
    squared = determinant_lower**2
    for _ in range(8):
        if Fraction(root) ** 3 <= squared:
            break
        root = float(np.nextafter(root, -np.inf))
    if root < 0.0 or Fraction(root) ** 3 > squared:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.mean_ratio_root must enclose the determinant power downward; got {root}"
        )
    mean = fraction_endpoint(
        3 * Fraction(root) / norm_squared, upper=False, saturate=True
    )
    cofactors = []
    for row in range(3):
        a, b = [index for index in range(3) if index != row]
        cofactor_row = []
        for column in range(3):
            c, d = [index for index in range(3) if index != column]
            minor = _dependency_phydrax_discretization__coordinate_enclosure.add(
                _dependency_phydrax_discretization__coordinate_enclosure.multiply(
                    jacobian[a][c], jacobian[b][d], budget=budget
                ),
                _dependency_phydrax_discretization__coordinate_enclosure.scale(
                    _dependency_phydrax_discretization__coordinate_enclosure.multiply(
                        jacobian[a][d], jacobian[b][c], budget=budget
                    ),
                    Fraction(-1),
                    budget=budget,
                ),
                budget=budget,
            )
            cofactor_row.append(_polynomial_magnitude(minor, domain, budget=budget))
        cofactors.append(tuple(cofactor_row))
    aspect = _sqrt_upper(
        _matrix_norm_product(bounds) * _matrix_norm_product(tuple(cofactors)) / squared
    )
    return _MappedQualityBounds(scaled, mean, aspect)


@checked
def exact_mapped_volume_quality(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enclose whole-cell volume quality from source expressions.

    The aspect bound encloses the physical Jacobian's spectral condition number
    using matrix one/infinity norms and polynomial cofactors.
    """
    if any(
        block.cell_kind not in ("hexahedron", "tetrahedron", "pyramid")
        for block in mesh.blocks
    ):
        raise ValueError(
            f"hex.quality.blocks must contain hex, tet, or pyramid cells; got {tuple(block.cell_kind for block in mesh.blocks)}"
        )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        elements, routes, _ = geometry.resolve(
            mesh, budget=budget, exact_source_prepared=False
        )
        coordinates = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
        count = sum(block.cell_count for block in mesh.blocks)
        scaled = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.float64, scope=scope
        )
        mean = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.float64, scope=scope
        )
        aspect = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.float64, scope=scope
        )
        family_execution_owner.count_geometry_queries(
            count, work_units=count, budget=budget
        )
        host_routes = jax.device_get(routes)
        row_index = 0
        for block, element, block_routes in zip(
            mesh.blocks, elements, host_routes, strict=True
        ):
            domain = "simplex" if block.cell_kind == "tetrahedron" else "box"
            for route in np.asarray(block_routes, dtype=np.int64):
                source = _prepare_mapped_quality_source(
                    element, coordinates, route, block.cell_kind, domain, budget=budget
                )
                bounds = _enclose_mapped_cell_quality(source, budget=budget)
                scaled[row_index], mean[row_index], aspect[row_index] = (
                    bounds.scaled,
                    bounds.mean,
                    bounds.aspect,
                )
                row_index += 1
    return scaled, mean, aspect


def _matrix_norm_product(bounds: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    row_norm = max(sum(row, Fraction(0)) for row in bounds)
    column_norm = max(
        sum((row[column] for row in bounds), Fraction(0)) for column in range(3)
    )
    return row_norm * column_norm


def _polynomial_magnitude(
    polynomial: _dependency_phydrax_discretization__coordinate_enclosure.Polynomial,
    domain: str = "box",
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Fraction:
    return max(
        abs(rational_coefficient(value, role="hex.quality.magnitude_controls"))
        for value in _dependency_phydrax_discretization__coordinate_enclosure.bernstein_coefficients(
            polynomial, domain, 3, budget=budget
        )
    )


def _sqrt_upper(value: Fraction, /) -> float:
    result = float(np.sqrt(fraction_endpoint(value, upper=True, saturate=True)))
    if np.isfinite(result) and Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
    if not np.isfinite(result) or Fraction(result) ** 2 < value:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.spectral_norm must enclose the square root upward; got {result}"
        )
    return result
