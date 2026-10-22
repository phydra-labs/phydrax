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

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import comb, lcm, prod
from typing import cast, Literal

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
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_bits,
    rational_coefficient,
)
from phydrax.geometry._rational import fraction_endpoint
from phydrax.linalg._exact import solve_exact

from .._bvh import point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_storage import declared_integer_storage_bytes
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
from ..discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_mesh_construction_bounds,
    cell_mesh_reuse_bounds,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellGeometryElement
from ..discretization.fem._reference_types import FiniteElementSpec
from ..discretization.spatial._level_octree import refined_octree_leaves
from ..discretization.spatial._morton import morton_encode_integer_host, MortonAddressPlan
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
from ..optim._iterative._types import MinimizationResult
from ..typing import checked, parse
from ._hex_frame_records import _SourceBlockTensorGrid, SourceBlockIntegerGrid
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed


class DualHexCandidate(StrictModule):
    """Dual hexahedra with ancestry and coordinate maps before their validity proof.

    ``extract_volume_hexes`` certifies every candidate cell. A caller that
    publishes only a subset of the cells certifies exactly what it publishes.
    """

    mesh: CellMesh
    source: CellMesh
    vertex_supports: np.ndarray
    parent_cells: np.ndarray
    entity_parent_dimensions: tuple[np.ndarray, ...]
    entity_parent_rows: tuple[np.ndarray, ...]
    valences: np.ndarray
    boundary_vertices: np.ndarray
    singular_vertices: np.ndarray
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    work_units: int = eqx.field(static=True)


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
    candidate = extract_dual_hex_candidate(
        source,
        limits,
        boundary_subdivision=boundary_subdivision,
        source_geometry=source_geometry,
        scope=scope,
        budget=budget,
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            candidate.geometry, mesh=candidate.mesh, budget=budget
        )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.validity must certify positive represented-coordinate maps; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
        )
    return dual_extraction_owner.DualExtraction(
        candidate.mesh,
        candidate.source,
        candidate.vertex_supports,
        candidate.parent_cells,
        candidate.entity_parent_dimensions,
        candidate.entity_parent_rows,
        validity,
        candidate.valences,
        candidate.boundary_vertices,
        candidate.singular_vertices,
        None,
        candidate.work_units,
        candidate.geometry,
        candidate.source_geometry,
    )


@checked
def extract_dual_hex_candidate(
    source: CellMesh,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    boundary_subdivision: bool = True,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> DualHexCandidate:
    """Build the vertex-star hexahedra of a certified positive tetrahedral source."""

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
    count, original = source.topology.entities(3).count, source.coordinates.shape[0]
    scratch, work = 8192 * count + 256 * original, 128 * count
    family_execution_owner.check_family_limits(
        limits, 4 * count, original, 32 * count, scratch, work
    )
    if source_geometry is None:
        source_shapes = tuple(
            (block.cell_kind, block.cell_count, block.arity) for block in source.blocks
        )
        with scope.host_workspace() as workspace:
            # Periodic sources are refused above.
            workspace.set_bound(
                _dependency_phydrax_discretization__cell_geometry.affine_geometry_construction_bounds(
                    original, source.ambient_dimension, source_shapes
                )
            )
            source_geometry = (
                _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
                    source
                )
            )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        certificate = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            source_geometry, mesh=source, budget=budget
        )
    if certificate.invalid_count or certificate.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.validity must certify every tetrahedron; got {certificate.invalid_count} invalid and {certificate.unresolved_count} unresolved"
        )
    edge_count, face_count = (
        source.topology.entities(1).count,
        source.topology.entities(2).count,
    )
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
    # Allocate only after admitting the declared per-cell template work, its
    # host scratch envelope and every published entity count.
    with (
        scope.batch(work) as ticket,
        scope.host_workspace() as workspace,
    ):
        workspace.set_bound(scratch)
        cells = _dependency_phydrax_meshing__host_banks.entities(source, 3)
        edges, faces = (
            _dependency_phydrax_meshing__host_banks.entities(source, 1),
            _dependency_phydrax_meshing__host_banks.entities(source, 2),
        )
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
        ticket.consume(work)
        supports = _dependency_phydrax_meshing__host_banks.family_host_array(
            (coordinates.shape[0], 4), np.int64, scope=scope
        )
        supports.fill(-1)
        supports[:original, 0] = np.arange(original, dtype=np.int64)
        supports[original : original + edge_count, :2] = edges
        supports[original + edge_count : original + edge_count + face_count, :3] = faces
        supports[original + edge_count + face_count :] = cells
        # The edge and face lookups stay live beside the block's temporaries.
        workspace.set_bound(scratch + cell_block_construction_bounds(hexes.shape[0], 8))
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
        boundary_faces = (dimensions[2] == 2) & (
            face_incidence[np.minimum(rows[2], face_count - 1)] == 1
        )
        boundary = np.zeros((coordinates.shape[0],), dtype=np.bool_)
        boundary[
            _dependency_phydrax_meshing__host_banks.entities(mesh, 2)[
                boundary_faces
            ].reshape(-1)
        ] = True
        valences = np.bincount(
            hexes.reshape(-1), minlength=coordinates.shape[0]
        ).astype(np.int64)
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
        return DualHexCandidate(
            mesh,
            source,
            supports,
            parents,
            dimensions,
            rows,
            valences,
            boundary,
            singular,
            geometry,
            source_geometry,
            work,
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
    images: dict[
        int, tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
    ]


class _ParentGeometry(StrictModule):
    # Target rows as (name, cell kind, vertex rows, global IDs); their blocks
    # and mesh are admitted and built after the coordinate phase.
    blocks: tuple[tuple[str, str, np.ndarray, np.ndarray], ...]
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
    images = {}
    for vertex in used_vertices:
        original = supports[vertex]
        original = original[original >= 0]
        image = tuple(
            sum((source_images[int(index)][axis] for index in original), Fraction(0))
            / original.size
            for axis in range(3)
        )
        budget.retain_basis((image,))
        images[int(vertex)] = image
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
    return _ParentPlacement(coordinates, images)


def _compose_parent_geometry(
    source: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    prepared: _ParentSource,
    charts: _ParentCharts,
    target_parents: np.ndarray,
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
        blocks.append((name, chart.cell_kind, vertices, identities))
        elements[name], dofs[name] = element, routes
        parent_ids[name] = source_ids[parents]
        parent_vertices[name] = source_vertices[source_cells[parents]]
        output_parents[output_offset : output_offset + parents.size] = parents
        output_offset += parents.size
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
    return _ParentGeometry(tuple(blocks), geometry, output_parents)


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
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.live_storage() as placement_storage,
    ):
        prepared = _prepare_parent_source(source, source_geometry, budget=budget)
        charts = _group_parent_charts(
            prepared, target_blocks, target_parents, supports, scope=scope
        )
        placement_storage.set_bound(256 + 128 * coordinates.shape[0])
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
            target_parents,
            scope=scope,
            budget=budget,
        )
        # These are the same exact P1 barycentric corner images used to place
        # the rounded carrier. _define_parent_chart assigns their authored
        # weights to every chart's cardinal corner (including a pyramid apex).
        # Bind reuse to the full composed element and exact coefficient bank;
        # carrier validation still compares every cached image to that carrier.
        budget.reserve(len(composed.blocks), 256 + 128 * len(composed.blocks))
        routes = dict(zip(
            composed.geometry.block_names,
            zip(composed.geometry.elements, composed.geometry.geometry_dofs, strict=True),
            strict=True,
        ))
        for name, _, rows, _ in composed.blocks:
            element, route = routes[name]
            for vertices, indices in zip(rows, np.asarray(route), strict=True):
                budget.reserve(len(vertices) + len(indices))
                local = tuple(prepared.values[int(index)] for index in indices)
                _, key = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_preparation_key(
                    element, local, budget=budget
                )
                images = tuple(placed.images[int(vertex)] for vertex in vertices)
                previous = budget.coordinate_corner_cache.get(key)
                if previous is not None and previous != images:
                    raise _dependency_phydrax_meshing__host_banks.failure(
                        "hex.target.corner_images must equal the previously earned exact map images; got conflicting placement images"
                    )
                budget.retain_basis((key, images))
                budget.coordinate_corner_cache[key] = images
        placed_coordinates = placed.coordinates
        del placed, prepared, charts, routes
    # Target blocks and their mesh are admitted outside the coordinate phase.
    vertex_count, ambient = placed_coordinates.shape
    shapes = tuple(
        (kind, vertices.shape[0], vertices.shape[1])
        for _, kind, vertices, _ in composed.blocks
    )
    with scope.host_workspace() as workspace:
        blocks = []
        for name, kind, vertices, identities in composed.blocks:
            workspace.set_bound(
                cell_block_construction_bounds(vertices.shape[0], vertices.shape[1])
            )
            blocks.append(CellBlock(name, kind, vertices, global_ids=identities))
        workspace.set_bound(cell_mesh_construction_bounds(vertex_count, ambient, shapes))
        mesh = CellMesh(
            placed_coordinates,
            tuple(blocks),
            vertex_global_ids=vertex_ids,
            numeric_version=source.numeric_version,
        )
    return mesh, composed.geometry, composed.parents


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


type _RationalGridBank = tuple[tuple[Fraction, ...], ...]


class GridHexConstruction(StrictModule):
    """Integer-map mesh, region proof worksets, and independently auditable grid."""

    mesh: CellMesh
    cell_regions: np.ndarray
    logical_vertices: np.ndarray
    exact_vertex_coordinates: _RationalGridBank = eqx.field(static=True)
    integer_cells: np.ndarray
    face_source_facets: np.ndarray
    face_source_polygons: np.ndarray
    frame: HexFrameField
    axis_knots: tuple[np.ndarray, ...]
    exact_axis_knots: _RationalGridBank = eqx.field(static=True)
    octree_prefixes: np.ndarray
    octree_levels: np.ndarray
    subdivisions: int = eqx.field(static=True)
    closure_rounds: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)


# Conservative CPython storage for one host container entry holding a small
# tuple of machine integers: container slot, tuple header, and int objects.
_HOST_ENTRY_BYTES = 256


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
    polygons = complex_.polygon_facets.size
    entries = complex_.polygon_vertices.size
    vertex_count = complex_.vertices.shape[0]
    with scope.host_workspace() as workspace:
        # Loop views, the normal bank with its length/transverse vectors, each
        # loop's gather/roll/cross rows, the projection banks, host pullback
        # copies, per-loop plane masks, and the vertex-axis incidence bank.
        workspace.set_bound(
            (_HOST_ENTRY_BYTES + 136) * polygons
            + 81 * entries
            + 45 * vertex_count
            + 4096
        )
        loops = [
            complex_.polygon_vertices[
                complex_.polygon_offsets[row] : complex_.polygon_offsets[row + 1]
            ]
            for row in range(polygons)
        ]
        normals = np.stack(
            [
                np.sum(
                    np.cross(
                        complex_.vertices[loop],
                        np.roll(complex_.vertices[loop], -1, axis=0),
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
            jax.device_get(jnp.asarray(initial) @ frame_rotation(angles)),
            dtype=np.float64,
        )
        projected = normals @ basis.T
        axes = np.argmax(np.abs(projected), axis=1).astype(np.int64)
        residuals = 1.0 - np.max(np.abs(projected), axis=1)
        solve = _dependency_phydrax_linalg__small_batched.solve_small_linear(
            _dependency_phydrax_linalg__small_batched.SmallLinearSolvePlan(3),
            np.broadcast_to(basis.T, (vertex_count, 3, 3)),
            complex_.vertices,
        )
        logical, successful = jax.device_get((solve.value, solve.successful))
        if not np.all(np.asarray(successful, dtype=np.bool_)):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.frame.pullback must have successful native solves; got singular or ill-conditioned basis"
            )
        logical_ = np.asarray(logical, dtype=np.float64)
        # Predicate postclassification, not epsilon snapping: every admitted
        # source polygon must lie on one represented coordinate plane.
        for row, loop in enumerate(loops):
            if np.any(logical_[loop, axes[row]] != logical_[loop[0], axes[row]]):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.frame.patch must lie on a coordinate plane; got a non-grid planar patch at row {row}"
                )
        incidence = np.zeros((vertex_count, 3), dtype=np.bool_)
        for row, loop in enumerate(loops):
            incidence[loop, axes[row]] = True
        singular = np.flatnonzero(np.count_nonzero(incidence, axis=1) != 3).astype(
            np.int64
        )
        del incidence
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


def _grid_coordinate_bytes(
    count: int,
    axis_upper: tuple[int, ...],
    basis: _RationalGridBank,
    bits: int,
    /,
    *,
    promotion: bool = False,
) -> int:
    """Bound axis-shared atoms, sparse weighted columns and vertex row owners."""
    per_scalar = 512 + 2 * declared_integer_storage_bytes(bits)
    weighted_entries = sum(
        axis_upper[axis]
        for component in range(3) for axis in range(3)
        if basis[axis][component]
    )
    weighted_atoms = sum(
        axis_upper[axis]
        for component in range(3) for axis in range(3)
        if basis[axis][component] not in (0, 1)
    )
    new_components = sum(
        sum(bool(basis[axis][component]) for axis in range(3)) > 1
        for component in range(3)
    )
    atoms = sum(axis_upper) + weighted_atoms + count * new_components + 9
    construction = (
        4096 + 1024 * count
        + _HOST_ENTRY_BYTES * (sum(axis_upper) + weighted_entries)
        + per_scalar * atoms
    )
    if not promotion:
        return construction
    # retain_basis traverses the still-live bank before promoting it. Include
    # both traversal/retained hash indices, tuple rows, Fraction owners and
    # their two integer children at the canonical owner's unchanged prices.
    return (
        construction + 4096 + 1024 * count
        + (2048 + 4 * declared_integer_storage_bytes(bits)) * atoms
    )


def _grid_vertex_coordinates(
    vertices: np.ndarray,
    knots: _RationalGridBank,
    subdivisions: int,
    basis: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, _RationalGridBank]:
    """Place authenticated integer addresses once, before binary64 rounding."""
    count = vertices.shape[0]
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.temporary_scope(),
    ):
        endpoint_bits = max(coefficient_bits(value) for axis in knots for value in axis)
        basis_values = tuple(tuple(Fraction(float(value)) for value in row) for row in basis)
        basis_bits = max(coefficient_bits(value) for row in basis_values for value in row)
        bits = 3 * (endpoint_bits + basis_bits + subdivisions.bit_length() + 2)
        # Admission and construction share the same sparse storage model.
        axis_upper = tuple(min(count, len(axis) * subdivisions + 1) for axis in knots)
        budget.reserve(
            9, _grid_coordinate_bytes(count, axis_upper, basis_values, bits)
        )
        addresses = tuple(np.unique(vertices[:, axis]).tolist() for axis in range(3))
        logical = []
        for axis, indices in enumerate(addresses):
            budget.reserve(3 * len(indices))
            values = {}
            for address in indices:
                lower = min(address // subdivisions, len(knots[axis]) - 2)
                fraction = Fraction(address - subdivisions * lower, subdivisions)
                values[address] = knots[axis][lower] + fraction * (
                    knots[axis][lower + 1] - knots[axis][lower]
                )
            logical.append(values)
        weighted = []
        for component in range(3):
            columns = []
            for axis in range(3):
                coefficient = basis_values[axis][component]
                if coefficient:
                    budget.reserve(len(addresses[axis]))
                    columns.append(
                        (axis, {
                            address: value if coefficient == 1 else coefficient * value
                            for address, value in logical[axis].items()
                        })
                    )
            weighted.append(columns)
        budget.reserve(count * sum(max(1, len(columns)) for columns in weighted))
        points = []
        zero = Fraction(0)
        for address in vertices:
            point = []
            for columns in weighted:
                terms = tuple(values[address[axis]] for axis, values in columns)
                point.append(
                    zero if not terms else terms[0]
                    if len(terms) == 1 else sum(terms[1:], terms[0])
                )
            points.append(tuple(point))
        exact = tuple(points)
        budget.retain_basis(exact)
        physical = _dependency_phydrax_meshing__host_banks.family_host_array(
            vertices.shape, np.float64, scope=scope
        )
        for row, point in enumerate(exact):
            physical[row] = tuple(float(value) for value in point)
    return physical, exact


def _grid_face_sources(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    source_logical: np.ndarray,
    grid: np.ndarray,
    mesh: CellMesh,
    frame: HexFrameField,
    knots: tuple[np.ndarray, ...],
    subdivisions: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify face-center trim membership after a boundary-plane exclusion proof."""
    face_count = mesh.topology.entities(2).count
    polygons = complex_.polygon_facets.size
    entries = complex_.polygon_vertices.size
    widest_axis = max(values.size for values in knots)
    # Trim queries are admitted on the request's coordinate ticket once each
    # polygon's on-plane faces are known.
    with scope.host_workspace() as workspace:
        # Face rows (32 bytes) and their 96-byte corner gather precede the
        # 24-byte center bank. Per polygon the corner-axis gather and masks
        # (37 bytes) or the row, center, chart and classification selections
        # (at most 80 bytes) coexist with the knot mask and ordinal banks.
        workspace.set_bound(
            (32 + 24 + 96) * face_count + widest_axis + 48 * entries + 4096
        )
        faces = _dependency_phydrax_meshing__host_banks.entities(mesh, 2)
        face_points = np.mean(grid[faces], axis=1)
        source_faces = _dependency_phydrax_meshing__host_banks.family_host_array(
            (face_count,), np.int64, scope=scope
        )
        source_faces.fill(-1)
        source_polygons = _dependency_phydrax_meshing__host_banks.family_host_array(
            (face_count,), np.int64, scope=scope
        )
        source_polygons.fill(-1)
        for polygon in range(polygons):
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
            del on_plane
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
            with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
                # One trim query per on-plane face center; its exact winding
                # visits every loop edge once.
                family_execution_owner.count_geometry_queries(
                    rows.size, work_units=rows.size * loop.size, budget=budget
                )
                classified = trim.classify(
                    face_points[rows][:, remaining], budget=budget
                )
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
    exact_axis_knots: _RationalGridBank = eqx.field(static=True)
    sizes: np.ndarray
    depth: int = eqx.field(static=True)
    required: int = eqx.field(static=True)
    subdivisions: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    refined_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    extraction_work: int = eqx.field(static=True)
    extraction_bytes: int = eqx.field(static=True)


class _IntegerGridTopology(StrictModule):
    coordinates: np.ndarray
    exact_coordinates: _RationalGridBank = eqx.field(static=True)
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
    exact_coordinates: _RationalGridBank = eqx.field(static=True)
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
    *,
    scope: NativeExecutionScope,
) -> _FeatureGridKnots:
    roots = source.mesh.coordinates.shape[0] if source.mapped else 0
    breaks_count = logical.shape[0] + roots
    entries = complex_.polygon_vertices.size
    segments = complex_.segments.shape[0]
    with scope.host_workspace() as workspace:
        # Root points and host pullback copies (49 bytes per root), the
        # concatenated breaks, one axis's sorted copy, mask and unique output,
        # the three knot vectors (65 bytes per break), and the owner map with
        # one polygon's loop list and the segment list.
        workspace.set_bound(
            49 * roots
            + 65 * breaks_count
            + _HOST_ENTRY_BYTES * (2 * entries + segments)
            + 4096
        )
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
            del root_points, pulled, successful
        else:
            breaks = logical
        knots = tuple(np.unique(breaks[:, axis]) for axis in range(3))
        del breaks
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
        del edge_owners
        for a, b in complex_.segments.tolist():
            if np.count_nonzero(logical[a] != logical[b]) != 1:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.source.segments must follow integer-grid edges; got a diagonal segment from {a} to {b}"
                )
        return _FeatureGridKnots(knots)


def source_block_interval_counts(
    lengths: tuple[Fraction, ...],
    target_size: float,
    maximum_size: float,
    admitted: Callable[[tuple[int, ...]], bool],
    /,
    *,
    scope: NativeExecutionScope,
    desired_lengths: tuple[Fraction, ...] | None = None,
    compatible_groups: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, ...]:
    """Preserve an admitted target, otherwise choose a maximal compatible grid.

    Exact count breakpoints first find the finest admitted common aimed size.
    Then integer bisection refines each independent axis class toward its
    target. Monotone admission proves that a rejected unit increment stays
    rejected as other classes refine: the result is componentwise maximal,
    without assuming a cube or claiming a unique global integer optimum.
    Every candidate is metadata only; construction and its ledger happen once.
    """
    def counts_at(size: Fraction, spans: tuple[Fraction, ...] = lengths) -> tuple[int, ...]:
        with scope.batch(4 * len(lengths) + 1) as ticket:
            ticket.consume(4 * len(lengths) + 1)
            return tuple(
                max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
                for length in spans
                for ratio in (length / size,)
            )

    with scope.host_workspace() as workspace:
        # Length Fractions, two endpoints, one candidate, and search lists.
        with scope.batch(2 * len(lengths) + 2) as ticket:
            ticket.consume(2 * len(lengths) + 2)
            bits = max(
                (coefficient_bits(value) for spans in (lengths, desired_lengths or ()) for value in spans),
                default=1,
            ) + coefficient_bits(Fraction(target_size)) + coefficient_bits(Fraction(maximum_size)) + 64
        workspace.set_bound(
            4096 + len(lengths) * (8 * _HOST_ENTRY_BYTES + 8 * declared_integer_storage_bytes(bits))
        )
        if desired_lengths is not None:
            desired = counts_at(Fraction(target_size), desired_lengths)
            if admitted(desired):
                return desired
        fine = counts_at(Fraction(target_size))
        aimed = fine
        if admitted(fine):
            return fine
        coarse = counts_at(Fraction(maximum_size))
        if coarse == fine or not admitted(coarse):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.aimed_counts must admit a construction within the authored size band; got no admitted count family member",
                resource=True,
            )
        while True:
            with scope.batch(3 * len(lengths) + 1) as ticket:
                ticket.consume(3 * len(lengths) + 1)
                differences = tuple(a - b for a, b in zip(fine, coarse, strict=True))
                axis = max(range(len(lengths)), key=differences.__getitem__)
                if differences[axis] <= 1:
                    # First coarsening event after the refused endpoint. If
                    # admitted it is the finest possible remaining count bank.
                    size = min(
                        length / (count - 1)
                        for length, count, difference in zip(lengths, fine, differences, strict=True)
                        if difference
                    )
                    next_counts = counts_at(size)
                    if admitted(next_counts):
                        coarse = next_counts
                        break
                    fine = next_counts
                else:
                    middle = (fine[axis] + coarse[axis]) // 2
                    next_counts = counts_at(lengths[axis] / middle)
                    if admitted(next_counts):
                        coarse = next_counts
                    else:
                        fine = next_counts
        with scope.batch(3 * len(lengths) + 1) as ticket:
            ticket.consume(3 * len(lengths) + 1)
        groups = compatible_groups
        if groups is None:
            groups = tuple((axis,) for axis in range(len(lengths)))
        selected = list(coarse)
        for group in groups:
            low, high = selected[group[0]], aimed[group[0]]
            while low < high:
                with scope.batch(2 * len(lengths) + len(group) + 4) as ticket:
                    ticket.consume(2 * len(lengths) + len(group) + 4)
                    middle = (low + high + 1) // 2
                    candidate = selected.copy()
                    for axis in group:
                        candidate[axis] = middle
                    if admitted(tuple(candidate)):
                        selected = candidate
                        low = middle
                    else:
                        high = middle - 1
        return tuple(selected)


def _grid_layout_bounds(
    sizes: tuple[int, ...], balanced: bool, /
) -> tuple[int, int, tuple[tuple[int, ...], ...], int, int, int, int]:
    """Static banks and work for the actual active lattice and sparse closure."""
    depth = max(1, max((value - 1).bit_length() for value in sizes))
    required = prod(sizes)
    refined_shapes: tuple[tuple[int, ...], ...] = ()
    if balanced:
        parent_shape = tuple((value + 1) // 2 for value in sizes)
        parents = prod(parent_shape)
        vertices = prod(2 * value + 1 for value in parent_shape)
        shapes = [parent_shape]
        for level in range(depth - 2, -1, -1):
            shapes.append(tuple(min(1 << level, value // 2 + 1) for value in shapes[-1]))
        refined_shapes = tuple(reversed(shapes))
        refinement_counts = tuple(prod(shape) for shape in refined_shapes)
        selected = sum(refinement_counts)
        nodes, leaves = 1 + 8 * selected, 1 + 7 * selected
        topology_bytes = 12 * leaves
        balance_work = sum(
            30 * refinement_counts[level]
            + sum(
                refinement_counts[ancestor] + min(26 * refinement_counts[level], refinement_counts[ancestor])
                for ancestor in range(level)
            )
            for level in range(1, depth)
        )
        # Per parent: 27 node entries; 54 native-wrapper index checks; eight
        # native children; 24 corner additions and comparisons each; eight
        # active counts; gather+assignment of 64 connectivity and 24 lower
        # entries; and sixteen parent/dispatch entries.
        extraction_work = 4 * selected + balance_work + 4 * nodes + 2 * leaves + 337 * parents
        tree_bytes = 16 * selected + 256 * nodes + 4096 * max(refinement_counts)
        # Full parent node/dedupe banks remain charged. Only the native child
        # output is streamed; every native child and every copy is still paid.
        template_bytes = topology_bytes + 2048 * parents + 24 * vertices + 192 * 8 * parents
        extraction_bytes = max(tree_bytes, template_bytes) + 4096 * (depth + 1) ** 2
    else:
        vertices = prod(value + 1 for value in sizes)
        topology_bytes = 0
        extraction_work, extraction_bytes = required, 1024 * required
    retained_bytes = topology_bytes + 208 * required + 48 * vertices + (512 * (depth + 1) if balanced else 0)
    return depth, required, refined_shapes, extraction_work, extraction_bytes, retained_bytes, vertices




def _prepare_grid_layout(
    knots: tuple[np.ndarray, ...],
    schedule: NativeHexGridSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    target_size: float,
    /,
    *,
    maximum_size: float | None = None,
    axis_scales: tuple[float, ...] = (1.0, 1.0, 1.0),
    desired_axis_scale: float = 1.0,
    placement_basis_bits: int = 1,
    placement_basis: np.ndarray | None = None,
    fixed_data_bytes: int = 0,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _GridLayout:
    intervals = sum(values.size - 1 for values in knots)
    knot_bytes = sum(values.nbytes for values in knots)
    with scope.host_workspace() as workspace:
        # The caller's knot vectors stay live; one axis's width differences
        # and the per-interval count lists are temporary.
        workspace.set_bound(knot_bytes + 1024 * intervals + 4096)
        with scope.batch(8 * intervals + 1) as ticket:
            ticket.consume(8 * intervals + 1)
            lengths = tuple(
                width * Fraction(scale)
                for values, scale in zip(knots, axis_scales, strict=True)
                for width in (
                    Fraction(float(last)) - Fraction(float(first))
                    for first, last in zip(values[:-1], values[1:], strict=True)
                )
            )
            smallest_reference = min(float(np.min(np.diff(values))) for values in knots)
            desired_placement = min(
                target_size / desired_axis_scale,
                smallest_reference * schedule.maximum_aspect_ratio,
            )
            desired_counts = tuple(
                max(1, int(np.ceil((last - first) / desired_placement)))
                for values in knots
                for first, last in zip(values[:-1], values[1:], strict=True)
            )
            knot_bits = max(coefficient_bits(Fraction(float(value))) for values in knots for value in values)
        # A transformed point is a sum of three products of an exact axis
        # knot and a represented basis entry. Bound numerator/denominator bits
        # before declaring its retained scalar bank, including nonunit roots.
        point_bits = 3 * (knot_bits + schedule.maximum_depth + placement_basis_bits + 8)
        exact_scalar_bytes = 16 + 2 * declared_integer_storage_bytes(point_bits)
        basis_values = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in (np.eye(3) if placement_basis is None else placement_basis)
        )
        placement_columns = tuple(
            sum(bool(basis_values[axis][component]) for axis in range(3))
            for component in range(3)
        )
        # Row evaluation and canonical retention each visit dense component
        # sums; sparse components share their axis atoms instead.
        placement_vertex_work = max(
            9, sum(max(1, columns) for columns in placement_columns)
            + 1 + 3 * sum(columns > 1 for columns in placement_columns),
        )
        smallest = min(lengths)
        aspect_ceiling = float(smallest * Fraction(schedule.maximum_aspect_ratio))
        placement_size = min(target_size, aspect_ceiling)
        ceiling = min(target_size if maximum_size is None else maximum_size, aspect_ceiling)
        balanced = schedule.route == "balanced_grid"

        def axis_sizes(counts: tuple[int, ...]) -> tuple[int, ...]:
            offset = 0
            result = []
            for values in knots:
                stop = offset + values.size - 1
                result.append(sum(counts[offset:stop]))
                offset = stop
            return tuple(result)

        def admitted(counts: tuple[int, ...]) -> bool:
            sizes_ = axis_sizes(counts)
            depth_ = max(1, max((value - 1).bit_length() for value in sizes_))
            # Charge every metadata scan and the triangular sparse-closure
            # cost calculation before using it, including rejected candidates.
            with scope.batch(8 * intervals + 16 * (depth_ + 1) ** 2) as ticket:
                ticket.consume(8 * intervals + 16 * (depth_ + 1) ** 2)
                if depth_ > schedule.maximum_depth:
                    return False
                _, cells_, shapes_, work_, scratch_, retained_, vertices_ = _grid_layout_bounds(sizes_, balanced)
                placement_work = (
                    placement_vertex_work * vertices_
                    + 15 * (sum(sizes_) + 6) + 2 * intervals + 12
                )
                placement_axes = tuple(
                    min(vertices_, value + 2) for value in sizes_
                )
                placement_bytes = _grid_coordinate_bytes(
                    vertices_, placement_axes, basis_values, point_bits, promotion=True
                )
                remaining_ = scope.remaining()
                return (
                    cells_ <= limits.maximum_cells
                    and 8 * cells_ <= limits.maximum_connectivity_entries
                    and vertices_ <= limits.maximum_vertices
                    and (not balanced or limits.maximum_cells >= 8)
                    and work_ + placement_work <= remaining_.remaining_work_units
                    and retained_ + max(scratch_, placement_bytes) <= remaining_.remaining_managed_bytes
                    and retained_ + max(scratch_, placement_bytes) <= limits.maximum_scratch_bytes
                    and (
                        fixed_data_bytes + 192 * cells_
                        + (48 + 3 * exact_scalar_bytes) * vertices_
                        + (8 + exact_scalar_bytes) * (sum(sizes_) + 3)
                        + (12 * (1 + 7 * sum(prod(shape) for shape in shapes_)) if balanced else 0)
                    ) <= limits.maximum_data_bytes
                )

        # Legacy standalone calls have no authored coarsening band. They keep
        # their desired counts and the original typed capacity checks below.
        if maximum_size is None or admitted(desired_counts):
            flat_counts = desired_counts
        else:
            flat_counts = source_block_interval_counts(
                lengths, placement_size, max(placement_size, ceiling), admitted, scope=scope
            )
        sizes_tuple = axis_sizes(flat_counts)
        final_depth = max(1, max((value - 1).bit_length() for value in sizes_tuple))
        with scope.batch(8 * intervals + 16 * (final_depth + 1) ** 2) as ticket:
            ticket.consume(8 * intervals + 16 * (final_depth + 1) ** 2)
            depth, required, refined_shapes, extraction_work, extraction_bytes, retained_bytes, vertices = _grid_layout_bounds(sizes_tuple, balanced)
        if depth > schedule.maximum_depth:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.depth must be at most {schedule.maximum_depth}; got {depth}",
                resource=True,
            )
        family_execution_owner.check_family_limits(limits, required, 0, required * 8, 0, 0)
        remaining = scope.remaining()
        if extraction_work > remaining.remaining_work_units or extraction_bytes + retained_bytes > remaining.remaining_managed_bytes:
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.extraction must fit the remaining request allowance; got excessive active-lattice work or banks",
                resource=True,
            )
        axis_counts: list[list[int]] = []
        offset = 0
        for values in knots:
            stop = offset + values.size - 1
            axis_counts.append(list(flat_counts[offset:stop]))
            offset = stop
        subdivisions = 1
        sizes = np.asarray(sizes_tuple, dtype=np.int64)
        del lengths, desired_counts, flat_counts, admitted
        widest = max(max(counts) for counts in axis_counts)
        workspace.set_bound(
            knot_bytes + (8 + _HOST_ENTRY_BYTES) * intervals + 24 * widest
            + (512 * (depth + 1) if balanced else 0) + 4096
        )
        refined_knots = []
        exact_axes = []
        bits = max(
            coefficient_bits(Fraction(float(value))) for axis in knots for value in axis
        ) + schedule.maximum_depth + 4
        per_scalar = 512 + 2 * declared_integer_storage_bytes(bits)
        with (
            family_execution_owner.coordinate_phase(scope=scope, budget=budget),
            budget.temporary_scope(),
        ):
            budget.reserve(
                3 * sum(map(int, sizes)) + 2 * intervals + 3,
                4096 + per_scalar * (sum(map(int, sizes)) + 3),
            )
            for values, counts in zip(knots, axis_counts, strict=True):
                refined_axis = _dependency_phydrax_meshing__host_banks.family_host_array(
                    (sum(counts) + 1,), np.float64, scope=scope
                )
                exact_axis = []
                for first, last, count in zip(values[:-1], values[1:], counts, strict=True):
                    start, end = Fraction(float(first)), Fraction(float(last))
                    step = (end - start) / count
                    exact_axis.extend(start + index * step for index in range(count))
                exact_axis.append(Fraction(float(values[-1])))
                for index, value in enumerate(exact_axis):
                    refined_axis[index] = float(value)
                refined_knots.append(refined_axis)
                exact_axes.append(tuple(exact_axis))
            exact_knots = tuple(exact_axes)
            budget.retain_basis(exact_knots)
    return _GridLayout(
        tuple(refined_knots), exact_knots, sizes, depth, required, subdivisions,
        retained_bytes, refined_shapes, extraction_work, extraction_bytes,
    )


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
    /,
    *,
    scope: NativeExecutionScope,
) -> _IntegerCellExtraction:
    sizes, depth = layout.sizes, layout.depth
    parents = prod((int(value) + 1) // 2 for value in sizes)
    if limits.maximum_cells < 8:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.template_cells must admit one real 1:8 template; got maximum_cells={limits.maximum_cells}",
            resource=True,
        )
    # One original entry-pass allowance covers all parents and all native
    # children, including the inactive halo. Batching changes simultaneous
    # native output capacity only; it does not reseed work or storage.
    with (
        scope.batch(layout.extraction_work) as ticket,
        scope.host_workspace() as workspace,
    ):
        workspace.set_bound(layout.extraction_bytes)
        ticket.consume(layout.extraction_work)
        address = MortonAddressPlan(
            (0.0, 0.0, 0.0), tuple(float(1 << depth) for _ in range(3)), depth
        )
        # These are genuine selected paths, not a manufactured uniform tree.
        # Independent balanced=True still performs the full touching-leaf
        # closure. The colleague-parent rectangles prove its bank upper bounds.
        refined = []
        for level, shape in enumerate(layout.refined_shapes):
            corners = np.indices(shape, dtype=np.int64).reshape((3, -1)).T
            corners *= 1 << (depth - level)
            refined.append(
                np.sort(
                    morton_encode_integer_host(corners, depth)
                    >> np.uint64(3 * (depth - level))
                )
            )
            del corners
        closure_rounds = depth
        octree_prefixes, octree_levels, lower = refined_octree_leaves(
            address, refined, balanced=True
        )
        del refined
        active_box = np.all(lower < sizes, axis=1)
        if np.any(octree_levels[active_box] != depth):
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.active_paths must reach the selected depth; got a coarse active leaf"
            )
        integer_cells = lower[active_box].astype(np.int64)
        del lower, active_box
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
        del integer_cells
        grid_offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int64)
        integer_vertices, inverse = np.unique(
            (parent_corners[:, None, :] + grid_offsets[None, :, :]).reshape((-1, 3)),
            axis=0,
            return_inverse=True,
        )
        nodes = inverse.reshape((-1, 27)).astype(np.int64)
        del inverse
        cells = np.empty((layout.required, 8), dtype=np.int64)
        integer_cells = np.empty((layout.required, 3), dtype=np.int64)
        child_offsets = np.asarray(list(product(range(2), repeat=3)), dtype=np.int64)
        batch_parents = min(parents, limits.maximum_cells // 8)
        cursor = 0
        for first in range(0, parents, batch_parents):
            last = min(parents, first + batch_parents)
            children = _owner_phydrax__meshcore.subdivide_hex_grid(
                nodes[first:last], integer_vertices.shape[0], maximum_cells=limits.maximum_cells
            )
            child_corners = (
                parent_corners[first:last, None, :] + child_offsets[None, :, :]
            ).reshape((-1, 3))
            active = np.all(child_corners < sizes, axis=1)
            count = int(np.count_nonzero(active))
            cells[cursor : cursor + count] = children[active]
            integer_cells[cursor : cursor + count] = child_corners[active]
            cursor += count
            del children, child_corners, active
        del nodes, parent_corners
        if cursor != layout.required:
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.active_templates must fill the admitted lattice exactly; got mismatched active children"
            )
    return _IntegerCellExtraction(
        cells,
        integer_vertices,
        integer_cells,
        offsets,
        octree_prefixes,
        octree_levels,
        closure_rounds,
    )


def _extract_frame_cells(
    layout: _GridLayout, /, *, scope: NativeExecutionScope
) -> _IntegerCellExtraction:
    sizes, required = layout.sizes, layout.required
    octree_prefixes = np.empty((0,), dtype=np.uint64)
    octree_levels = np.empty((0,), dtype=np.int64)
    closure_rounds = 0
    # One unit per generated cell. The layout's 1024 bytes per required cell
    # bound the 192-byte corner bank, its flattened and sorted copies, the
    # permutation, masks, inverse, and unique output.
    with scope.batch(required) as ticket, scope.host_workspace() as workspace:
        workspace.set_bound(1024 * required)
        ticket.consume(required)
        integer_cells = np.indices(
            tuple(int(value) for value in sizes), dtype=np.int64
        ).reshape((3, -1)).T
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
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _IntegerGridTopology:
    match schedule.route:
        case "balanced_grid":
            extracted = _extract_balanced_cells(layout, limits, scope=scope)
        case "frame_grid":
            extracted = _extract_frame_cells(layout, scope=scope)
        case _:
            raise ValueError(
                f"hex.schedule.route must name a supported grid route; got {schedule.route}"
            )
    integer_vertices, cells = extracted.integer_vertices, extracted.cells
    subdivisions = layout.subdivisions
    coordinates, exact_coordinates = _grid_vertex_coordinates(
        integer_vertices, layout.exact_axis_knots, subdivisions, frame.basis,
        scope=scope, budget=budget
    )
    cell_count = cells.shape[0]
    # Each cell's 192-byte corner gather is temporary.
    with scope.host_workspace() as workspace:
        workspace.set_bound(192 * cell_count + 4096)
        centers = _dependency_phydrax_meshing__host_banks.family_host_array(
            (cell_count, 3), np.float64, scope=scope
        )
        np.mean(coordinates[cells], axis=1, out=centers)
    return _IntegerGridTopology(
        coordinates,
        exact_coordinates,
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
    *,
    scope: NativeExecutionScope,
) -> _GridClassification:
    coordinates, cells, offsets = topology.coordinates, topology.cells, topology.offsets
    centers = topology.centers
    root_count = sum(block.cell_count for block in source.mesh.blocks)
    cell_count = cells.shape[0]
    slots = cell_count * schedule.maximum_location_candidates
    # One hierarchy query per integer-cell center.
    with (
        scope.batch(cell_count, queries_upper=cell_count) as ticket,
        scope.host_workspace() as workspace,
    ):
        # Root connectivity and point copies, corner, matrix and expectation
        # banks with their comparison masks, and the host hierarchy (1536
        # bytes per root); candidate and validity host copies (9 bytes per
        # slot) and completeness flags.
        workspace.set_bound(1536 * root_count + 9 * slots + cell_count + 4096)
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
        del root_matrices, expected_matrices, expected, diagonal
        root_lower, root_upper = root_corners[:, 0], root_corners[:, 6]
        root_bvh = prepare_bvh(root_lower, root_upper, dtype=np.float64)
        ticket.consume(cell_count, geometry_queries=cell_count)
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
        tests = int(np.count_nonzero(valid))
        # Each tested candidate is one root-box containment query on the
        # cell's eight corners.
        with scope.batch(tests, queries_upper=tests) as tested:
            tested.consume(tests, geometry_queries=tests)
            for row in range(cell_count):
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
    *,
    scope: NativeExecutionScope,
) -> _GridClassification:
    centers = topology.centers
    cell_count = centers.shape[0]
    policy = _dependency_phydrax_discretization__simplicial_locator.SimplicialLocationPolicy(
        schedule.maximum_location_candidates, 16, 1
    )
    slots = cell_count * policy.maximum_candidates * policy.maximum_seeds
    # Reserve the bounded inverse-map candidate queries, not solver instructions.
    with (
        scope.batch(slots, queries_upper=slots) as ticket,
        scope.host_workspace() as workspace,
    ):
        # Status, inside, and inside-row host copies (at most 13 bytes per
        # cell), candidate host copies (8 bytes per slot), and one row's
        # containing/region selections.
        workspace.set_bound(
            13 * cell_count + 8 * slots + 24 * policy.maximum_candidates + 4096
        )
        space = FiniteElementPlan(
            source.mesh,
            FiniteElementFieldSpec("grid-region", lagrange_element("tetrahedron", 1)),
        ).prepare()
        locator = _dependency_phydrax_discretization__simplicial_locator.PreparedSimplicialCellLocator(
            _dependency_phydrax_discretization_fem__cell_map.PreparedFiniteElementCellMap(
                space, 0
            ),
            space.default_runtime.coordinates,
            policy,
        )
        ticket.consume(slots, geometry_queries=slots)
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
        inside_rows = np.flatnonzero(inside)
        for row in inside_rows:
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
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _RetainedIntegerGrid:
    cells, integer_cells = topology.cells, topology.integer_cells
    coordinates, integer_vertices = topology.coordinates, topology.integer_vertices
    exact_coordinates = topology.exact_coordinates
    regions, numeric_version = classified.regions, source.numeric_version
    candidates = cells.shape[0]
    with scope.host_workspace() as workspace:
        # The layout's 1024 bytes per cell bound the retained-row copies, the
        # vertex dedupe (flattened, permutation, sorted, mask, inverse), and
        # one dimension's entity gather, sorted rows, and dedupe copies.
        workspace.set_bound(1024 * candidates + 4096)
        kept = regions >= 0
        cells, regions, integer_cells = cells[kept], regions[kept], integer_cells[kept]
        del kept
        retained = cells.shape[0]
        if retained == 0:
            raise _dependency_phydrax_meshing__host_banks.failure(
                "hex.grid.cells must cover a source region; got zero retained cells"
            )
        used, compact = np.unique(cells, return_inverse=True)
        cells = compact.reshape(cells.shape)
        del compact
        family_execution_owner.check_family_limits(
            limits, retained, used.size, cells.size, 0, 0
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
            exact_coordinates = tuple(exact_coordinates[int(index)] for index in used)
        del used
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
            del entity_rows
            if count > bound:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.grid.entity_count must fit dimension-{dimension} limit {bound}; got {count}",
                    resource=True,
                )
    vertex_count = coordinates.shape[0]
    shapes = (("hexahedron", cells.shape[0], 8),)
    block_bytes = cell_block_construction_bounds(cells.shape[0], 8)
    mesh_bytes = cell_mesh_construction_bounds(vertex_count, 3, shapes)
    with scope.host_workspace() as workspace:
        workspace.set_bound(block_bytes)
        block = CellBlock("integer_blocks", "hexahedron", cells)
        workspace.set_bound(mesh_bytes)
        mesh = CellMesh(coordinates, (block,), numeric_version=numeric_version)
    # This intermediate reference-grid certificate is consumed here. Its
    # expanded differential and Bernstein memos are not part of the physical
    # mapped carrier, which earns its own validity after root composition.
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.discarded_preparation_scope(),
    ):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            mesh, budget=budget
        )
        try:
            if validity.invalid_count or validity.unresolved_count:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"hex.grid.validity must certify positive cell determinant enclosures; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
                )
        finally:
            del validity
    return _RetainedIntegerGrid(
        mesh, cells, regions, coordinates, exact_coordinates, integer_vertices, integer_cells
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
    maximum_size: float | None = None,
    reference_axis_bounds: tuple[float, ...] | None = None,
    desired_axis_scale: float = 1.0,
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
    initial_work = scope.remaining().remaining_work_units
    frame, logical = prepare_hex_frame(
        complex_, schedule, record_phase=record_phase, scope=scope
    )
    topology_started = phase_started(record_phase)
    source = _resolve_grid_source(construction)
    features = _feature_grid_knots(complex_, source, frame, logical, scope=scope)
    axis_scales: tuple[float, ...] = (1.0, 1.0, 1.0)
    if reference_axis_bounds is not None:
        with scope.batch(27) as ticket:
            ticket.consume(27)
            axis_scales = tuple(
                fraction_endpoint(
                    sum(
                        (abs(Fraction(float(frame.basis[axis, row]))) * Fraction(reference_axis_bounds[row]) for row in range(3)),
                        Fraction(0),
                    ),
                    upper=True,
                )
                for axis in range(3)
            )
    with scope.batch(15) as ticket:
        ticket.consume(15)
        placement_basis_bits = max(coefficient_bits(Fraction(float(value))) for value in frame.basis.flat)
        fixed_data_bytes = sum(
            array.nbytes for array in (
                frame.basis, frame.normals, frame.patch_axes,
                frame.alignment_residuals, frame.singular_vertices,
            )
        )
    layout = _prepare_grid_layout(
        features.knots, schedule, limits, target_size,
        desired_axis_scale=desired_axis_scale,
        placement_basis_bits=placement_basis_bits, fixed_data_bytes=fixed_data_bytes,
        placement_basis=frame.basis,
        maximum_size=maximum_size, axis_scales=axis_scales, scope=scope, budget=budget
    )
    del features
    with scope.host_workspace() as retained_workspace:
        # Extracted topology banks and the compact grid banks stay live until
        # the construction is returned.
        retained_workspace.set_bound(layout.retained_bytes)
        topology = _extract_integer_grid(
            layout, schedule, limits, frame, scope=scope, budget=budget
        )
        record_elapsed(record_phase, "topology_construction", topology_started)
        classification_started = phase_started(record_phase)
        regions = _dependency_phydrax_meshing__host_banks.family_host_array(
            (topology.cells.shape[0],), np.int64, scope=scope
        )
        regions.fill(-1)
        if source.mapped:
            classified = _classify_mapped_grid(
                source, topology, schedule, regions, scope=scope
            )
        else:
            classified = _classify_volume_grid(
                source, topology, schedule, regions, scope=scope
            )
        record_elapsed(record_phase, "region_classification", classification_started)
        retained_grid = _retain_integer_grid(
            topology, classified, source, limits, scope=scope, budget=budget
        )
        mesh, cells, regions = (
            retained_grid.mesh,
            retained_grid.cells,
            retained_grid.regions,
        )
        coordinates, integer_vertices = (
            retained_grid.coordinates,
            retained_grid.integer_vertices,
        )
        integer_cells = retained_grid.integer_cells
        exact_coordinates = retained_grid.exact_coordinates
        knots, subdivisions = layout.knots, layout.subdivisions
        octree_prefixes, octree_levels = topology.octree_prefixes, topology.octree_levels
        closure_rounds = topology.closure_rounds
        del topology, classified, retained_grid
        # Retained payload is exact before face classification: the two face
        # source banks hold one int64 per face each.
        data_bytes = 16 * mesh.topology.entities(2).count + sum(
            array.nbytes
            for array in (
                coordinates,
                cells,
                regions,
                integer_vertices,
                integer_cells,
                octree_prefixes,
                octree_levels,
                *knots,
                frame.basis,
                frame.normals,
                frame.patch_axes,
                frame.alignment_residuals,
                frame.singular_vertices,
            )
        )
        data_bytes += sum(
            16 + 2 * declared_integer_storage_bytes(coefficient_bits(value))
            for bank in (layout.exact_axis_knots, exact_coordinates)
            for row in bank for value in row
        )
        if data_bytes > limits.maximum_data_bytes:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.grid.data_bytes must fit maximum_data_bytes={limits.maximum_data_bytes}; got {data_bytes}",
                resource=True,
            )
        face_sources, face_polygons = _grid_face_sources(
            complex_,
            logical,
            integer_vertices,
            mesh,
            frame,
            knots,
            subdivisions,
            scope=scope,
            budget=budget,
        )
        return GridHexConstruction(
            mesh,
            regions,
            integer_vertices,
            exact_coordinates,
            integer_cells,
            face_sources,
            face_polygons,
            frame,
            knots,
            layout.exact_axis_knots,
            octree_prefixes,
            octree_levels,
            subdivisions,
            closure_rounds,
            initial_work - scope.remaining().remaining_work_units,
            canonical_fingerprint(
                {
                    "kind": "integer-grid-hex-construction",
                    "source": complex_.complex_id,
                    "schedule": schedule.schedule_id,
                    "mesh": mesh.mesh_id,
                    "regions": array_tree_fingerprint(regions),
                    "placement_basis": array_tree_fingerprint(frame.basis),
                    "integer_vertices": array_tree_fingerprint(integer_vertices),
                    "exact_axis_knots": tuple(
                        tuple((value.numerator, value.denominator) for value in axis)
                        for axis in layout.exact_axis_knots
                    ),
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


def _source_root_lipschitz_bound(
    polynomials: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...],
    inverse: list[list[Fraction]],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    root_axis_bounds: np.ndarray | None = None,
    reference_axis_bounds: np.ndarray | None = None,
) -> float:
    """Consume one root's derivative and Bernstein workspace before the next root."""
    # Lattice/weight preparations are consumed too, not only the Bernstein
    # coefficient memos covered by polynomial_preparation_scope. Only the
    # rounded bound escapes; no newly prepared exact owner remains live.
    with budget.discarded_preparation_scope():
        bounds = []
        local_squared = [Fraction(0), Fraction(0), Fraction(0)]
        if root_axis_bounds is not None:
            budget.reserve(24 * len(polynomials), 4096)
        if reference_axis_bounds is not None:
            budget.reserve(12 * len(polynomials), 4096)
        budget.reserve(9, 4096)
        columns = tuple(
            tuple(index for index in range(3) if inverse[index][axis])
            for axis in range(3)
        )
        for value in polynomials:
            # A physical component has three source derivatives, independent of
            # which column of the reference-frame solve consumes them.
            jets = tuple(
                _dependency_phydrax_discretization__coordinate_enclosure.derivative(
                    value, axis, budget=budget
                )
                for axis in range(3)
            )
            # Bernstein bounds commute with scalar multiplication. Reuse each
            # source-column magnitude for every one-term reference column,
            # rather than scaling, re-preparing and bounding the same jet.
            magnitudes: list[Fraction | None] = [None, None, None]
            if root_axis_bounds is not None:
                for axis, jet in enumerate(jets):
                    magnitude = _polynomial_magnitude(jet, budget=budget)
                    magnitudes[axis] = magnitude
                    local_squared[axis] += magnitude * magnitude
            row = []
            for axis, nonzero in enumerate(columns):
                if not nonzero:
                    row.append(Fraction(0))
                elif len(nonzero) == 1:
                    index = nonzero[0]
                    magnitude = magnitudes[index]
                    if magnitude is None:
                        magnitude = _polynomial_magnitude(jets[index], budget=budget)
                        magnitudes[index] = magnitude
                    coefficient = inverse[index][axis]
                    bits = (
                        abs(coefficient.numerator).bit_length()
                        + coefficient.denominator.bit_length()
                        + magnitude.numerator.bit_length()
                        + magnitude.denominator.bit_length()
                    )
                    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                        1, 1, 0, bits, budget=budget
                    )
                    row.append(abs(coefficient) * magnitude)
                else:
                    row.append(
                        _polynomial_magnitude(
                            _dependency_phydrax_discretization__coordinate_enclosure.add(
                                _dependency_phydrax_discretization__coordinate_enclosure.add(
                                    _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                        jets[0], inverse[0][axis], budget=budget
                                    ),
                                    _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                        jets[1], inverse[1][axis], budget=budget
                                    ),
                                    budget=budget,
                                ),
                                _dependency_phydrax_discretization__coordinate_enclosure.scale(
                                    jets[2], inverse[2][axis], budget=budget
                                ),
                                budget=budget,
                            ),
                            budget=budget,
                        )
                    )
            bounds.append(tuple(row))
        for axis in range(3):
            if root_axis_bounds is not None:
                root_axis_bounds[axis] = _sqrt_upper(local_squared[axis])
            if reference_axis_bounds is not None:
                reference_axis_bounds[axis] = max(
                    float(reference_axis_bounds[axis]),
                    _sqrt_upper(sum((row[axis] ** 2 for row in bounds), Fraction(0))),
                )
        return _sqrt_upper(_matrix_norm_product(tuple(bounds)))


@checked
def mapped_source_lipschitz_bound(
    reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    root_axis_bounds: np.ndarray | None = None,
    reference_axis_bounds: np.ndarray | None = None,
    root_reference_axis_bounds: np.ndarray | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> float:
    """Bound physical placement in source reference charts.

    Affine reference-frame solves and source Jacobians use exact arithmetic.
    Interval bounds cover the source expression throughout every reference cell.
    """
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.temporary_scope(),
    ):
        elements, routes, _ = source_geometry.resolve(
            reference_mesh, budget=budget, exact_source_prepared=False
        )
        root_count = sum(block.cell_count for block in reference_mesh.blocks)
        if root_axis_bounds is not None and root_axis_bounds.shape != (root_count, 3):
            raise ValueError("hex.source.root_axis_bounds must have shape (root_count, 3)")
        if root_reference_axis_bounds is not None and root_reference_axis_bounds.shape != (root_count, 3):
            raise ValueError("hex.source.root_reference_axis_bounds must have shape (root_count, 3)")
        if reference_axis_bounds is not None:
            if reference_axis_bounds.shape != (3,):
                raise ValueError("hex.source.reference_axis_bounds must have shape (3,)")
            reference_axis_bounds.fill(0.0)
        # Host corner, connectivity, and route copies are temporary.
        budget.reserve(
            0,
            24 * reference_mesh.coordinates.shape[0]
            + 64 * root_count
            + 16 * sum(route.size for route in routes)
            + 4096,
        )
        corners = np.asarray(reference_mesh.coordinates, dtype=np.float64)
        source_values = source_geometry.source_coordinates(budget=budget)
        family_execution_owner.count_geometry_queries(
            root_count, work_units=9 * root_count, budget=budget
        )
        host_routes = jax.device_get(routes)
        upper = 0.0
        root_row = 0
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
                if root_reference_axis_bounds is not None:
                    budget.reserve(24, 4096)
                    for axis in range(3):
                        root_reference_axis_bounds[root_row, axis] = _sqrt_upper(
                            sum((value[axis] ** 2 for value in matrix), Fraction(0))
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
                upper = max(
                    upper,
                    _source_root_lipschitz_bound(
                        polynomials, inverse, budget=budget,
                        root_axis_bounds=None if root_axis_bounds is None else root_axis_bounds[root_row],
                        reference_axis_bounds=reference_axis_bounds,
                    ),
                )
                root_row += 1
    if not np.isfinite(upper) or upper <= 0.0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.source.lipschitz_bound must be positive and finite; got {upper}"
        )
    return upper




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


class _TensorGridCharts(StrictModule):
    """Root-major tensor enumeration; only one chart's controls are materialized."""

    parents: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    nodes: tuple[int, ...]

    def integer(self, row: int) -> tuple[int, int, int]:
        parent = int(self.parents[row])
        _, ny, nz = self.counts[parent]
        index = row - int(self.starts[parent])
        return int(index // (ny * nz)), int((index // nz) % ny), int(index % nz)

    def controls(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        lower = self.integer(row)
        numerators = np.empty((8, 3), dtype=np.int64)
        denominators = np.empty((8, 3), dtype=np.int64)
        for corner, node in zip(
            reference_cell_topology("hexahedron").vertices, self.nodes, strict=True
        ):
            numerators[node] = tuple(lower[axis] + int(corner[axis]) for axis in range(3))
        denominators[:] = self.counts[int(self.parents[row])]
        return numerators, denominators


def _tensor_grid_charts(
    grid: _SourceBlockTensorGrid, *, scope: NativeExecutionScope
) -> _TensorGridCharts:
    chart = _dependency_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
        "hexahedron", 1
    )
    work = 5 * len(grid.root_intervals)
    with scope.batch(work) as ticket:
        ticket.consume(work)
        return _TensorGridCharts(
            grid.source_parent_rows, grid.root_intervals,
            np.cumsum((0, *(prod(int(n) for n in row) for row in grid.root_intervals)), dtype=np.int64),
            tuple(dofs[0] for dofs in chart.entity_dofs[0]),
        )


class _MappedRestrictionGroups(StrictModule):
    block_of_root: np.ndarray
    local_of_root: np.ndarray
    groups: dict[str, list[int]]


class _MappedGeometryPreparation(StrictModule):
    # Canonical blocks are built once: no dense host connectivity copy survives
    # beside the retained block that consumes it.
    blocks: tuple[CellBlock, ...]
    physical_points: np.ndarray
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    order: np.ndarray


def _prepare_mapped_source(
    reference_mesh: CellMesh,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_cell_regions: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    storage: _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage,
) -> _MappedSourcePreparation:
    source_elements, source_routes, _ = source_geometry.resolve(
        source_reference_mesh, budget=budget, exact_source_prepared=False
    )
    root_count = sum(block.cell_count for block in source_reference_mesh.blocks)
    target_cells = sum(block.cell_count for block in reference_mesh.blocks)
    target_vertices = reference_mesh.coordinates.shape[0]
    # This lease owns root preparation, grouping and composition banks only
    # until the independently owned result is transferred to the budget.
    # Chart controls and exact polynomial preparations have their own owners.
    storage.set_bound(
        1536 * root_count
        + 32 * source_reference_mesh.coordinates.shape[0]
        + 16 * sum(route.size for route in source_routes)
        + (24 + 4 * _HOST_ENTRY_BYTES) * target_vertices
        + (1280 + 16 * max(route.shape[1] for route in source_routes))
        * target_cells
        + 4096,
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
    exact_points: _RationalGridBank,
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
    # One hierarchy query per target cell. Exact chart solves charge one
    # query per tested candidate.
    family_execution_owner.count_geometry_queries(
        reference_cells.shape[0],
        work_units=reference_cells.shape[0],
        budget=budget,
    )
    target_corners = points[reference_cells]
    bvh = prepare_bvh(
        np.min(root_corners, axis=1),
        np.max(root_corners, axis=1),
        dtype=np.float64,
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
                tuple(exact_points[int(index)] for index in reference_cells[row]),
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
    grid: GridHexConstruction | SourceBlockIntegerGrid | _SourceBlockTensorGrid,
    prepared: _MappedSourcePreparation,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedRestrictionCharts | _TensorGridCharts:

    reference_cells, chart, source_cells = (
        prepared.reference_cells,
        prepared.chart,
        prepared.cells,
    )
    match grid:
        case _SourceBlockTensorGrid():
            if (
                grid.source_frame.reference_mesh.mesh_id != source_reference_mesh.mesh_id
                or grid.source_frame.source_id
                != _dependency_phydrax_discretization__cell_geometry_validity.cell_geometry_id(source_geometry)
            ):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.grid.source_frame must bind the unchanged source mesh and coordinate bank"
                )
            counts = grid.root_intervals
            if (
                counts.shape != (len(source_cells), 3) or counts.dtype.kind not in "iu"
                or np.any(counts <= 0)
            ):
                raise ValueError("tensor_grid.counts must define positive root-axis partitions")
            sizes = tuple(prod(int(n) for n in row) for row in counts)
            count = sum(sizes)
            budget.reserve(count + counts.size, 8 * count + 128 * len(sizes))
            if (
                grid.source_parent_rows.shape != (count,) or count != len(reference_cells)
                or not np.array_equal(
                    grid.source_parent_rows, np.repeat(np.arange(len(sizes)), sizes)
                )
            ):
                raise ValueError("tensor_grid.parents must retain exhaustive root-major order")
            charts = _tensor_grid_charts(grid, scope=scope)
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
            exact_points = grid.exact_vertex_coordinates
            if len(exact_points) != prepared.points.shape[0] or any(
                len(point) != 3 or any(type(value) is not Fraction for value in point)
                for point in exact_points
            ):
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.grid.exact_placement must cover every reference carrier vertex with rational provenance"
                )
            budget.reserve(3 * len(exact_points), 4096)
            for exact, carrier in zip(exact_points, prepared.points, strict=True):
                rounded = np.asarray(tuple(float(value) for value in exact), dtype=np.float64)
                if not np.array_equal(rounded.view(np.uint64), carrier.view(np.uint64)):
                    raise _dependency_phydrax_meshing__host_banks.failure(
                        "hex.grid.exact_placement must round to the actual reference carrier; got a different vertex image"
                    )
            charts = _locate_mapped_charts(
                prepared, exact_points, scope=scope, budget=budget
            )
        case _:
            raise TypeError(
                f"hex.grid must be GridHexConstruction or SourceBlockIntegerGrid; got {type(grid).__name__}"
            )
    budget.reserve(0)
    if np.any(prepared.regions[charts.parents] != grid.cell_regions):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.grid.regions must agree with source-reference regions; got different extracted and parent region IDs"
        )
    return charts


def _group_mapped_restrictions(
    prepared: _MappedSourcePreparation,
    charts: _MappedRestrictionCharts | _TensorGridCharts,
    source_reference_mesh: CellMesh,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedRestrictionGroups:
    budget.reserve(0)
    source_elements = prepared.elements
    parent_rows = charts.parents
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
    if isinstance(charts, _TensorGridCharts):
        families: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        for root, counts in enumerate(charts.counts):
            element = source_elements[block_of_root[root]]
            key = (element.element_id, tuple(int(n) for n in counts))
            families.setdefault(key, []).append(root)
        for (element_id, counts), roots in families.items():
            for local in range(prod(counts)):
                prototype = int(charts.starts[roots[0]]) + local
                budget.reserve(24)
                numerators, denominators = charts.controls(prototype)
                key = canonical_fingerprint({
                    "source_element": element_id,
                    "chart_numerators": array_tree_fingerprint(numerators),
                    "chart_denominators": array_tree_fingerprint(denominators),
                })
                groups[f"chart:{key}"] = [int(charts.starts[root]) + local for root in roots]
    else:
        for row, parent in enumerate(parent_rows.tolist()):
            element = source_elements[block_of_root[parent]]
            key = canonical_fingerprint(
                {
                    "source_element": element.element_id,
                    "chart_numerators": array_tree_fingerprint(charts.numerators[row]),
                    "chart_denominators": array_tree_fingerprint(charts.denominators[row]),
                }
            )
            groups.setdefault(f"chart:{key}", []).append(row)
    if len(groups) > limits.maximum_work_units:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.grid.restriction_groups must fit maximum_work_units={limits.maximum_work_units}; got {len(groups)}",
            resource=True,
        )
    return _MappedRestrictionGroups(block_of_root, local_of_root, groups)


class _MappedRootPlacement(StrictModule):
    """Private numerical operands consumed only by one mapped-placement loop."""

    polynomials: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...]
    coefficient_bits: int = eqx.field(static=True)
    degrees: tuple[int, ...] = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)


def _prepare_mapped_root_placement(
    element: CellGeometryElement,
    coefficients: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedRootPlacement | None:
    # The ordinary entry point authenticates the complete current public source
    # law and bank. No public tuple is promoted to an identity-only operand.
    polynomials = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
        element, coefficients, budget=budget
    )
    if polynomials is None or any(
        type(value) is not Fraction
        for polynomial in polynomials
        for value in polynomial.values()
    ):
        return None
    numerator, denominator = (
        _dependency_phydrax_discretization__coordinate_enclosure.coefficient_profile(
            polynomials, budget=budget
        )
    )
    terms = sum(len(polynomial) for polynomial in polynomials)
    budget.reserve(terms, 1024)
    dimension = reference_cell_topology(element.cell_kind).dimension
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        terms, terms, dimension,
        numerator + denominator, budget=budget,
    )
    # Copy every rational atom into private kernel state. The operand and its
    # image memo never escape this call's internal composition, and no callback
    # can expose them between preparation and consumption.
    private = cast(tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...], tuple(
        {
            index: Fraction(
                rational_coefficient(value, role="hex.placement.root").numerator,
                rational_coefficient(value, role="hex.placement.root").denominator,
            )
            for index, value in polynomial.items()
        }
        for polynomial in polynomials
    ))
    return _MappedRootPlacement(
        private,
        numerator + denominator,
        tuple(
            max(
                (index[axis] for polynomial in private for index in polynomial),
                default=0,
            )
            for axis in range(dimension)
        ),
        max((len(polynomial) for polynomial in private), default=0),
    )


def _mapped_root_point_image(
    placement: _MappedRootPlacement,
    point: tuple[Fraction, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    storage: _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage,
) -> tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]:
    # Admit the private memo record and integer arithmetic before evaluating.
    # A common source denominator and the point's maximum axis powers enclose
    # every numerator and denominator of the completed rational sum.
    bits = (
        placement.coefficient_bits
        + sum(
            degree
            * (abs(value.numerator).bit_length() + value.denominator.bit_length())
            for degree, value in zip(placement.degrees, point, strict=True)
        )
        + max(placement.maximum_terms, 1).bit_length()
    )
    point_bits = max(
        (abs(value.numerator).bit_length() + value.denominator.bit_length() for value in point),
        default=0,
    )
    storage.set_bound(
        storage.bound
        + 512
        + (len(point) + len(placement.polynomials))
        * (512 + 2 * declared_integer_storage_bytes(max(bits, point_bits, 1))),
        work=len(placement.polynomials),
    )
    return _dependency_phydrax_discretization__coordinate_enclosure.reference_image(
        placement.polynomials, "hexahedron", point, budget=budget
    )


@dataclass(slots=True)
class _MappedPointRecord:
    image: tuple[_dependency_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
    storage_bytes: int
    references: int = 1


def _place_mapped_geometry(
    reference_cells: np.ndarray,
    parent_rows: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
    elements: dict[str, _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement],
    placements: dict[int, _MappedRootPlacement | None],
    source_banks: dict[int, _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank],
    corner_nodes: tuple[int, ...] | None,
    physical_points: np.ndarray,
    /,
    *,
    storage: _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Visit roots, then chart groups, without changing published block order.

    Local memo ownership ends at the final use of an exact root point. A shared
    vertex independently owns its first image until the final incidence, even
    when that incidence belongs to a later root.
    """
    with budget.live_storage() as index_storage:
        # Admit schedules, incidence counters, memo/witness dictionary slots and
        # sorting scratch before allocating any of them. Rational atoms have a
        # separate chart bound, and evaluated records use the live image lease.
        index_storage.set_bound(
            4096 + 192 * reference_cells.shape[0]
            + 256 * physical_points.shape[0] + 512 * len(groups)
        )
        remaining = np.bincount(
            reference_cells.reshape(-1), minlength=physical_points.shape[0]
        )
        schedule: dict[int, list[tuple[str, int]]] = {}
        chart_points: dict[str, tuple[tuple[Fraction, ...], ...]] = {}
        for name, rows in groups:
            element = elements[name]
            if corner_nodes is not None:
                chart_bits = max(
                    abs(numerator).bit_length() + denominator.bit_length()
                    for row in element.chart_coefficients
                    for numerator, denominator in row
                )
                # Construct the actual authenticated chart's corner atoms once,
                # not once per target cell sharing that chart.
                _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                    len(corner_nodes) * 3, len(corner_nodes) * 3, 0,
                    chart_bits, budget=budget,
                )
                chart_points[name] = tuple(
                    tuple(Fraction(numerator, denominator) for numerator, denominator
                          in element.chart_coefficients[node])
                    for node in corner_nodes
                )
            for row in rows:
                schedule.setdefault(int(parent_rows[row]), []).append((name, int(row)))
        index_storage.set_bound(
            index_storage.bound
            + 256 * reference_cells.shape[1]
            * max((len(rows) for rows in schedule.values()), default=0)
        )
        witnesses: dict[int, _MappedPointRecord] = {}
        for parent in sorted(schedule):
            placement = placements[parent]
            local_remaining: dict[tuple[Fraction, ...], int] = {}
            cache: dict[tuple[Fraction, ...], _MappedPointRecord] = {}
            if placement is not None and corner_nodes is not None:
                for name, _ in schedule[parent]:
                    for point in chart_points[name]:
                        local_remaining[point] = local_remaining.get(point, 0) + 1
            for name, row in schedule[parent]:
                fallback = None
                if placement is None or corner_nodes is None:
                    fallback = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                        elements[name], source_banks[parent], budget=budget
                    )
                    if fallback is None:
                        raise _dependency_phydrax_meshing__host_banks.failure(
                            f"hex.source.restriction must support rational corner evaluation; got {elements[name].element_id}"
                        )
                for corner, vertex_ in enumerate(reference_cells[row]):
                    vertex = int(vertex_)
                    if fallback is None:
                        assert placement is not None
                        point = chart_points[name][corner]
                        record = cache.get(point)
                        if record is None:
                            before = storage.bound
                            image = _mapped_root_point_image(
                                placement, point, storage=storage, budget=budget
                            )
                            record = _MappedPointRecord(image, storage.bound - before)
                            cache[point] = record
                            del image
                    else:
                        # The generic evaluator owns its exact image storage;
                        # the admitted incidence index covers this record.
                        record = _MappedPointRecord(fallback[corner], 0)
                    previous = witnesses.get(vertex)
                    if previous is not None and previous.image != record.image:
                        raise _dependency_phydrax_meshing__host_banks.failure(
                            f"hex.source.restriction_traces must agree at shared coordinate vertices; got different images at vertex {vertex}"
                        )
                    physical_points[vertex] = (
                        _dependency_phydrax_discretization__coordinate_enclosure.rounded_point(
                            record.image
                        )
                    )
                    remaining[vertex] -= 1
                    retired_bytes = 0
                    if previous is None and remaining[vertex]:
                        witnesses[vertex] = record
                        record.references += 1
                    elif previous is not None and not remaining[vertex]:
                        del witnesses[vertex]
                        previous.references -= 1
                        if not previous.references:
                            retired_bytes += previous.storage_bytes
                    if fallback is None:
                        local_remaining[point] -= 1
                        if not local_remaining[point]:
                            del local_remaining[point], cache[point]
                            record.references -= 1
                    else:
                        record.references -= 1
                    if not record.references:
                        retired_bytes += record.storage_bytes
                    # No list, loop alias or closure may keep retired images
                    # alive after their storage admission is released.
                    del previous, record
                    if retired_bytes:
                        storage.set_bound(storage.bound - retired_bytes)
                del fallback
            assert not cache and not local_remaining
        assert not witnesses and not np.any(remaining)
        del schedule, chart_points, remaining, witnesses, cache, local_remaining


def _place_tensor_geometry(
    charts: _TensorGridCharts,
    reference_cells: np.ndarray,
    placements: dict[int, _MappedRootPlacement | None],
    physical_points: np.ndarray,
    *,
    storage: _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Evaluate each root-axis node once; retain exact shared-boundary witnesses."""
    corner_rows = {
        tuple(int(value) for value in corner): row
        for row, corner in enumerate(reference_cell_topology("hexahedron").vertices)
    }
    witnesses: dict[int, _MappedPointRecord] = {}
    with budget.live_storage() as index_storage:
        index_storage.set_bound(4096 + 96 * physical_points.shape[0])
        for parent, counts_ in enumerate(charts.counts):
            counts = tuple(int(n) for n in counts_)
            placement = placements[parent]
            assert placement is not None
            with budget.temporary_scope():
                _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                    sum(n + 1 for n in counts), sum(n + 1 for n in counts), 0,
                    2 * max(n.bit_length() for n in counts), budget=budget,
                )
                axes = tuple(tuple(Fraction(i, n) for i in range(n + 1)) for n in counts)
                for integer in product(*(range(n + 1) for n in counts)):
                    budget.reserve(9)
                    lower = tuple(min(i, n - 1) for i, n in zip(integer, counts, strict=True))
                    corner = tuple(i - lo for i, lo in zip(integer, lower, strict=True))
                    row = int(charts.starts[parent]) + (lower[0] * counts[1] + lower[1]) * counts[2] + lower[2]
                    vertex = int(reference_cells[row, corner_rows[corner]])
                    point = tuple(axis[i] for axis, i in zip(axes, integer, strict=True))
                    before = storage.bound
                    image = _mapped_root_point_image(
                        placement, point, storage=storage, budget=budget
                    )
                    previous = witnesses.get(vertex)
                    if previous is not None and previous.image != image:
                        raise _dependency_phydrax_meshing__host_banks.failure(
                            f"hex.source.restriction_traces must agree at shared coordinate vertices; got different images at vertex {vertex}"
                        )
                    physical_points[vertex] = _dependency_phydrax_discretization__coordinate_enclosure.rounded_point(image)
                    if previous is None and any(i in (0, n) for i, n in zip(integer, counts, strict=True)):
                        witnesses[vertex] = _MappedPointRecord(image, storage.bound - before)
                        del image
                    else:
                        del image
                        storage.set_bound(before)
                    del previous
                del axes, point
        retired = sum(record.storage_bytes for record in witnesses.values())
        witnesses.clear()
        storage.set_bound(storage.bound - retired)
    if not np.all(np.isfinite(physical_points)):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "hex.source.restriction must place every tensor vertex"
        )


def _compose_mapped_geometry(
    prepared: _MappedSourcePreparation,
    charts: _MappedRestrictionCharts | _TensorGridCharts,
    grouped: _MappedRestrictionGroups,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    placement_storage: _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage,
    tensor_counts: np.ndarray | None = None,
    reference_identity: bool = False,
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
    parent_rows = charts.parents
    block_of_root, local_of_root, groups = (
        grouped.block_of_root,
        grouped.local_of_root,
        grouped.groups,
    )
    if tensor_counts is not None and (
        tensor_counts.shape != (source_cells.shape[0], 3)
        or tensor_counts.dtype.kind not in "iu"
        or np.any(tensor_counts <= 0)
    ):
        raise ValueError("tensor_grid.counts must define positive integer root-axis partitions")
    chart_families: dict[
        tuple[int, tuple[int, int, int]],
        _dependency_phydrax_discretization__cell_geometry._TensorGridChartFamily,
    ] = {}
    blocks, elements, dofs, parent_ids, parent_vertices = [], {}, {}, {}, {}
    source_ids = np.asarray(
        source_reference_mesh.topology.entities(3).entity_ids, dtype=np.int64
    )
    source_vertex_ids = np.asarray(
        source_reference_mesh.vertex_global_ids, dtype=np.int64
    )
    # Physical placement consumes each root's immutable coefficient tuple once.
    # Reference points were already constructed directly from exact root axes;
    # their canonical theorem authenticates the source without preparing it here.
    source_banks = {}
    if not reference_identity:
        budget.reserve(0, 256 + 96 * source_cells.shape[0])
        for parent in range(source_cells.shape[0]):
            block = block_of_root[parent]
            local = local_of_root[parent]
            source_banks[int(parent)] = (
                _dependency_phydrax_discretization__coordinate_enclosure.coordinate_preparation_key(
                    source_elements[block],
                    tuple(
                        source_values[int(index)]
                        for index in np.asarray(source_routes[block], dtype=np.int64)[local]
                    ),
                    budget=budget,
                )[0]
            )
    placements = {} if reference_identity else {
        parent: _prepare_mapped_root_placement(
            source_elements[block_of_root[parent]], coefficients, budget=budget
        )
        for parent, coefficients in source_banks.items()
    }
    placement_groups: list[tuple[str, np.ndarray]] = []
    corner_nodes = (
        None if reference_identity
        else _dependency_phydrax_discretization__cell_geometry._cardinal_corner_nodes(
            chart, budget=budget
        )
    )
    target_ids = np.asarray(
        reference_mesh.topology.entities(3).entity_ids, dtype=np.int64
    )
    physical_points = (
        points if reference_identity
        else _dependency_phydrax_meshing__host_banks.family_host_array(
            points.shape, np.float64, scope=scope
        )
    )
    if not reference_identity:
        physical_points.fill(np.nan)
        # Physical placement remains independently authenticated.
        query_count = (
            sum(prod(int(n) + 1 for n in counts) for counts in charts.counts)
            if isinstance(charts, _TensorGridCharts) and all(value is not None for value in placements.values())
            else 8 * reference_cells.shape[0]
        )
        family_execution_owner.count_geometry_queries(
            query_count, work_units=query_count,
            budget=budget,
        )
    order = []
    for name, rows_ in sorted(groups.items()):
        rows = np.asarray(rows_, dtype=np.int64)
        rows = rows[np.argsort(target_ids[rows], kind="stable")]
        order.extend(rows.tolist())
        parents = parent_rows[rows]
        prototype = rows[0]
        controls = (
            None if isinstance(charts, _TensorGridCharts)
            else (charts.numerators[prototype], charts.denominators[prototype])
        )
        source_element = _dependency_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
            source_elements[block_of_root[parents[0]]], "mapped source"
        )
        family = None
        if tensor_counts is not None:
            count_row = tensor_counts[int(parents[0])]
            counts = (int(count_row[0]), int(count_row[1]), int(count_row[2]))
            key = (id(source_element), counts)
            family = chart_families.get(key)
            if family is None:
                family = _dependency_phydrax_discretization__cell_geometry._TensorGridChartFamily.prepare(
                    source_element, counts, budget=budget
                )
                chart_families[key] = family
        # Construction scratch (degree bounds, chart preparation) is dead once
        # the element exists; its chart arguments are retained by their owner.
        with budget.temporary_scope():
            if family is None:
                assert controls is not None
                element = _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
                    source_element, chart, controls[0], controls[1],
                    budget=budget,
                )
            else:
                if isinstance(charts, _TensorGridCharts):
                    integer = charts.integer(int(prototype))
                else:
                    assert controls is not None
                    budget.reserve(48, 4096)
                    start = controls[0][family.nodes[0]]
                    integer = (int(start[0]), int(start[1]), int(start[2]))
                    for corner, node in zip(
                        reference_cell_topology("hexahedron").vertices, family.nodes, strict=True
                    ):
                        if any(
                            int(controls[0][node, axis]) != integer[axis] + int(corner[axis])
                            or int(controls[1][node, axis]) != family.counts[axis]
                            for axis in range(3)
                        ):
                            raise ValueError("tensor_grid.chart must match the constructed exact partition")
                element = family.element(integer, budget=budget)
        route = np.stack(
            [
                np.asarray(source_routes[block_of_root[parent]], dtype=np.int64)[
                    local_of_root[parent]
                ]
                for parent in parents.tolist()
            ]
        )
        placement_groups.append((name, rows))
        with scope.host_workspace() as block_storage:
            block_storage.set_bound(cell_block_construction_bounds(len(rows), reference_cells.shape[1]))
            blocks.append(CellBlock(
                name, "hexahedron", reference_cells[rows], global_ids=target_ids[rows]
            ))
        elements[name], dofs[name] = element, route
        parent_ids[name] = source_ids[parents]
        parent_vertices[name] = source_vertex_ids[source_cells[parents]]
    if not reference_identity:
        if isinstance(charts, _TensorGridCharts) and all(value is not None for value in placements.values()):
            _place_tensor_geometry(
                charts, reference_cells, placements, physical_points,
                storage=placement_storage, budget=budget,
            )
        else:
            _place_mapped_geometry(
                reference_cells, parent_rows, placement_groups, elements, placements,
                source_banks, corner_nodes, physical_points,
                storage=placement_storage, budget=budget,
            )
    order_ = np.asarray(order, dtype=np.int64)
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
    return _MappedGeometryPreparation(tuple(blocks), physical_points, geometry, order_)


_ATOMIC_TENSOR_BUILDERS = (
    _prepare_mapped_source, _resolve_mapped_charts, _group_mapped_restrictions,
    _compose_mapped_geometry,
    _dependency_phydrax_discretization__cell_geometry._TensorGridChartFamily.element,
    _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity,
)


def _certify_source_grid_restrictions(
    mesh: CellMesh,
    source_mesh: CellMesh,
    parent_rows: np.ndarray,
    numerators: np.ndarray | None,
    denominators: np.ndarray | None,
    source_regions: np.ndarray,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    tensor_counts: np.ndarray | None = None,
    tensor_charts: _TensorGridCharts | None = None,
) -> None:
    """Prove exact source-grid restrictions without changing the grid carrier.

    Grouping is private proof staging. Original cell IDs, order, rounded
    coordinates and the exact chart banks remain the published grid owners.
    """
    root_shapes = tuple(
        (block.cell_kind, block.cell_count, block.arity) for block in source_mesh.blocks
    )
    with scope.host_workspace() as workspace:
        workspace.set_bound(
            _dependency_phydrax_discretization__cell_geometry.affine_geometry_construction_bounds(
                source_mesh.coordinates.shape[0], source_mesh.ambient_dimension, root_shapes
            )
        )
        root_geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(source_mesh)
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.discarded_preparation_scope(),
    ):
        with (
            family_execution_owner.coordinate_phase(scope=scope, budget=budget),
            budget.temporary_scope(),
            budget.live_storage() as source_storage,
        ):
            root_validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                root_geometry, mesh=source_mesh, budget=budget
            )
            if not root_validity.all_certified:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "source_grid.source_validity must certify every reference root"
                )
            prepared = _prepare_mapped_source(
                mesh, source_mesh, root_geometry, source_regions,
                budget=budget, storage=source_storage,
            )
            if tensor_charts is None:
                if numerators is None or denominators is None:
                    raise ValueError("source_grid.charts must provide exact controls")
                charts = _MappedRestrictionCharts(parent_rows, numerators, denominators)
            else:
                charts = tensor_charts
            grouped = _group_mapped_restrictions(
                prepared, charts, source_mesh, limits, budget=budget
            )
            with budget.live_storage() as placement_storage:
                realized = _compose_mapped_geometry(
                    prepared, charts, grouped, source_mesh, root_geometry,
                    scope=scope, budget=budget, placement_storage=placement_storage,
                    tensor_counts=tensor_counts,
                    reference_identity=tensor_charts is not None,
                )
            if tensor_charts is None:
                budget.reserve(mesh.coordinates.size)
                if not np.array_equal(
                    realized.physical_points.view(np.uint64),
                    np.asarray(mesh.coordinates, dtype=np.float64).view(np.uint64),
                ):
                    raise _dependency_phydrax_meshing__host_banks.failure(
                        "source_grid.reference_carrier must round the exact parent restrictions; got changed child coordinates"
                    )
            # The result has copied connectivity/ID/order banks, immutable chart
            # coefficients and canonical geometry routes. It does not borrow
            # the source lookup, grouping, or placement schedule.
            budget.retain_owner(realized, temporary_storage_credit=source_storage)
            del prepared, grouped, charts
        shapes = tuple(
            (block.cell_kind, block.cell_count, block.arity) for block in realized.blocks
        )
        with scope.host_workspace() as workspace:
            workspace.set_bound(cell_mesh_construction_bounds(
                mesh.coordinates.shape[0], mesh.ambient_dimension, shapes
            ))
            proof_mesh = CellMesh(
                mesh.coordinates, realized.blocks,
                vertex_global_ids=mesh.vertex_global_ids, numeric_version=mesh.numeric_version,
            )
        with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
            if tensor_counts is not None:
                budget.reserve(
                    tensor_counts.size + len(realized.order),
                    256 + 128 * len(tensor_counts) + 40 * len(realized.order),
                )
            validity = _dependency_phydrax_discretization__cell_geometry_validity._certify_restriction_cell_geometry_validity(
                source_mesh, root_geometry, root_validity,
                proof_mesh, realized.geometry, budget=budget,
                tensor_grid=(
                    None if tensor_counts is None
                    else _dependency_phydrax_discretization__cell_geometry_validity._TensorGridRestriction(
                        tuple(
                            (int(row[0]), int(row[1]), int(row[2]))
                            for row in tensor_counts
                        ),
                        tuple(int(row) for row in realized.order),
                    )
                ),
            )
            budget.reserve(len(realized.order), 8 * len(realized.order))
            status = np.empty(len(realized.order), dtype=np.int32)
            status[realized.order] = np.asarray(validity.status)
            failed = np.flatnonzero(
                status != _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
            )
            if failed.size:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    f"source_grid.validity must certify every exact reference restriction; got undecided original rows {failed.tolist()}"
                )


def certify_source_grid_restrictions(
    mesh: CellMesh,
    source_mesh: CellMesh,
    parent_rows: np.ndarray,
    numerators: np.ndarray,
    denominators: np.ndarray,
    source_regions: np.ndarray,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    tensor_counts: np.ndarray | None = None,
) -> None:
    """Certify the ordinary public materialized source-chart restrictions."""
    _certify_source_grid_restrictions(
        mesh, source_mesh, parent_rows, numerators, denominators, source_regions, limits,
        scope=scope, budget=budget, tensor_counts=tensor_counts,
    )


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
    """Realize public grids without changing their materialized chart contract."""
    return _realize_mapped_grid_hexes(
        grid, source_reference_mesh, source_geometry, source_cell_regions, limits,
        scope=scope, budget=budget,
    )


def _realize_mapped_grid_hexes(
    grid: GridHexConstruction | SourceBlockIntegerGrid | _SourceBlockTensorGrid,
    source_reference_mesh: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_cell_regions: np.ndarray,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    quality_policy: NativeHexGridSchedule | None = None,
    association_inputs: list[_dependency_phydrax_discretization__cell_geometry_validity._ConstructedTensorGrid] | None = None,
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
    target_cells = sum(block.cell_count for block in grid.mesh.blocks)
    target_vertices = grid.mesh.coordinates.shape[0]
    # Physical and reference points, connectivity, regions, parent and order
    # rows, and three quality bounds per cell are retained; their size is
    # known before composition. Coefficient routes are checked at publication.
    data_bytes = 48 * target_vertices + 112 * target_cells
    if data_bytes > limits.maximum_data_bytes:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.mapped.data_bytes must fit maximum_data_bytes={limits.maximum_data_bytes}; got {data_bytes}",
            resource=True,
        )
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.temporary_scope(),
        budget.live_storage() as source_storage,
    ):
        prepared = _prepare_mapped_source(
            grid.mesh,
            source_reference_mesh,
            source_geometry,
            source_cell_regions,
            budget=budget,
            storage=source_storage,
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
            prepared, charts, source_reference_mesh, limits, budget=budget
        )
        with budget.live_storage() as placement_storage:
            realized = _compose_mapped_geometry(
                prepared,
                charts,
                grouped,
                source_reference_mesh,
                source_geometry,
                scope=scope,
                budget=budget,
                placement_storage=placement_storage,
                tensor_counts=grid.root_intervals if isinstance(grid, _SourceBlockTensorGrid) else None,
            )
        budget.retain_owner(realized, temporary_storage_credit=source_storage)
        geometry, order_ = realized.geometry, realized.order
        parent_rows = charts.parents
        reference_mesh = prepared.reference_mesh
        reference_points = prepared.points
        # Root lookup, grouping and chart-construction banks have no consumer
        # after composition. Drop their owners before releasing phase scratch.
        del prepared, charts, grouped
    # The retained canonical blocks outlive preparation scratch. The physical
    # mesh only moves reference vertices, so it reuses the reference topology.
    vertex_count, ambient = reference_points.shape
    shapes = tuple(
        (block.cell_kind, block.cell_count, block.arity) for block in realized.blocks
    )
    mesh_bytes = cell_mesh_construction_bounds(vertex_count, ambient, shapes)
    reuse_bytes = cell_mesh_reuse_bounds(vertex_count, ambient, shapes)
    with scope.host_workspace() as workspace:
        workspace.set_bound(mesh_bytes)
        grouped_reference = CellMesh(
            reference_points,
            realized.blocks,
            vertex_global_ids=reference_mesh.vertex_global_ids,
            numeric_version=reference_mesh.numeric_version,
        )
        workspace.set_bound(reuse_bytes)
        mesh = grouped_reference.with_coordinates(
            realized.physical_points, numeric_version=reference_mesh.numeric_version
        )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        if isinstance(grid, _SourceBlockTensorGrid):
            # These are the full authenticated physical source maps on their
            # reference root domains, not affine fits through rounded vertices.
            root_validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                source_geometry, mesh=source_reference_mesh, budget=budget
            )
            if not root_validity.all_certified:
                raise _dependency_phydrax_meshing__host_banks.failure(
                    "hex.mapped.validity must certify every complete physical root"
                )
            budget.reserve(
                grid.root_intervals.size + 2 * len(order_),
                256 + 128 * len(grid.root_intervals) + 80 * len(order_),
            )
            theorem = _dependency_phydrax_discretization__cell_geometry_validity._TensorGridRestriction(
                tuple((int(row[0]), int(row[1]), int(row[2])) for row in grid.root_intervals),
                tuple(int(row) for row in order_),
            )
            construction = None
            if _ATOMIC_TENSOR_BUILDERS == (
                _prepare_mapped_source, _resolve_mapped_charts, _group_mapped_restrictions,
                _compose_mapped_geometry,
                _dependency_phydrax_discretization__cell_geometry._TensorGridChartFamily.element,
                _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity,
            ):
                construction = _dependency_phydrax_discretization__cell_geometry_validity._prepare_constructed_tensor_grid(
                    source_reference_mesh, source_geometry, mesh, geometry, theorem,
                    tuple(int(parent_rows[row]) for row in order_), budget,
                )
            association_authorized = construction is not None
            try:
                validity = _dependency_phydrax_discretization__cell_geometry_validity._certify_restriction_cell_geometry_validity(
                    source_reference_mesh, source_geometry, root_validity,
                    mesh, geometry, budget=budget, tensor_grid=theorem,
                    construction=construction,
                )
                association_fields = () if construction is None else construction.owner_fields[:-1]
            finally:
                if construction is not None:
                    construction.active = False
                    construction.owner_fields = ()
                    construction.parents = ()
                del construction
        else:
            validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                geometry, mesh=mesh, budget=budget
            )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.mapped.validity must certify physical cell maps; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved"
        )
    scaled, mean, aspect = _exact_mapped_volume_quality(
        mesh, geometry, scope=scope, budget=budget,
        tensor_policy=quality_policy if isinstance(grid, _SourceBlockTensorGrid) else None,
    )
    if association_inputs is not None and isinstance(grid, _SourceBlockTensorGrid) and association_authorized:
        with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
            source_fields = _dependency_phydrax_discretization__cell_geometry_validity._snapshot_immutable_owner_fields(
                (source_reference_mesh, source_geometry), budget,
            )
            if source_fields is not None:
                budget.reserve(0, 512 + 16 * (len(association_fields) + len(source_fields)))
                handoff = _dependency_phydrax_discretization__cell_geometry_validity._ConstructedTensorGrid(
                    budget, source_reference_mesh, source_geometry, mesh, geometry, theorem,
                    tuple(int(parent_rows[row]) for row in order_),
                    (*association_fields, *source_fields),
                )
                names = ("budget", "source_mesh", "source_geometry", "mesh", "geometry", "theorem", "parents")
                handoff.owner_fields = (
                    *handoff.owner_fields,
                    (handoff, type(handoff), names, tuple(getattr(handoff, name) for name in names)),
                )
                association_inputs.append(handoff)
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
    """Rigorous whole-cell bounds of the physical Jacobian ``J``.

    ``column_squared[k]`` bounds ``sup |J[:, k]|^2``, ``aspect_squared`` bounds
    ``sup |J|_2^2 * sup |cof J|_2^2`` and ``determinant_lower`` bounds ``inf det J``.
    """

    column_squared: tuple[Fraction, ...]
    aspect_squared: Fraction
    determinant_lower: Fraction


def _entrywise_quality_source(
    jacobian_bounds: tuple[tuple[Fraction, ...], ...],
    cofactor_bounds: tuple[tuple[Fraction, ...], ...],
    determinant_lower: Fraction,
    /,
) -> _MappedQualitySource:
    """Columns from entry magnitudes; ``|A|_2^2 <= |A|_1 |A|_inf`` for both factors."""
    return _MappedQualitySource(
        tuple(
            sum((row[axis] ** 2 for row in jacobian_bounds), Fraction(0))
            for axis in range(3)
        ),
        _matrix_norm_product(jacobian_bounds) * _matrix_norm_product(cofactor_bounds),
        determinant_lower,
    )


class _MappedQualityBounds(StrictModule):
    scaled: float
    mean: float
    aspect: float


def _box_binomial_weights(degrees: tuple[int, ...], /) -> np.ndarray:
    rows = [np.asarray([comb(n, i) for i in range(n + 1)], dtype=object) for n in degrees]
    return rows[0][:, None, None] * rows[1][None, :, None] * rows[2][None, None, :]


_BoxNet = tuple[np.ndarray, int]


def _box_convolve(
    first: _BoxNet,
    second: _BoxNet,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxNet:
    """Multiply two scaled tensor Bernstein nets; scaled products are convolutions.

    Each net carries an upper bound on its integer control bit lengths.
    """
    (left, left_bits), (right, right_bits) = first, second
    shape = tuple(a + b - 1 for a, b in zip(left.shape, right.shape, strict=True))
    bits = left_bits + right_bits + min(left.size, right.size).bit_length()
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        left.size * right.size, prod(shape), 3, bits, budget=budget
    )
    result = np.zeros(shape, dtype=object)
    rows, columns, layers = right.shape
    for (i, j, k), value in np.ndenumerate(left):
        if value:
            result[i : i + rows, j : j + columns, k : k + layers] += value * right
    return result, bits


def _box_combine(
    first: _BoxNet | None,
    second: _BoxNet | None,
    /,
    *,
    subtract: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxNet | None:
    """Add or subtract equal-degree nets; an absent net is the zero polynomial."""
    if second is None:
        return first
    net, bits = second
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        net.size, net.size, 3,
        max(bits, 0 if first is None else first[1]) + 1, budget=budget,
    )
    if first is None:
        return (-net if subtract else net), bits
    return (first[0] - net if subtract else first[0] + net), max(first[1], bits) + 1




_BoxQualityNet = tuple[np.ndarray, int, int]


class _BoxQualityOperands(StrictModule):
    """Private, unweighted integer Bernstein operands, not quality verdicts."""

    jacobian: tuple[tuple[_BoxQualityNet | None, ...], ...]
    minors: tuple[tuple[_BoxQualityNet | None, ...], ...]
    determinant: _BoxQualityNet | None


def _box_quality_net(
    net: _BoxNet | None,
    denominator: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxQualityNet | None:
    """Remove convolution weights once, preserving every nominal degree."""
    if net is None:
        return None
    values, bits = net
    degrees = tuple(length - 1 for length in values.shape)
    weight_bits = sum(degrees) + 1
    # The product of all weights bounds their least common multiple.
    bound = bits + values.size * weight_bits + denominator.bit_length()
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        11 * values.size, 8 * values.size, 3, bound, budget=budget
    )
    weights = _box_binomial_weights(degrees)
    common = 1
    for weight in weights.flat:
        common = lcm(common, int(weight))
    integers = np.asarray(
        [int(value) * (common // int(weight))
         for value, weight in zip(values.flat, weights.flat, strict=True)],
        dtype=object,
    ).reshape(values.shape)
    # Compact repeated planes only: their implicit nominal-degree repetitions
    # remain identical after restriction. Never lower a nonconstant elevation.
    for axis in range(3):
        first = np.take(integers, [0], axis=axis)
        if np.all(integers == first):
            integers = first
    return integers, denominator * common, bits + common.bit_length()


def _box_quality_source(
    operands: _BoxQualityOperands,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource:
    def extremum(net: _BoxQualityNet | None, *, lower: bool = False) -> Fraction:
        if net is None:
            return Fraction(0)
        values, denominator, bits = net
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            (1 if lower else 2) * values.size + 1, 2, 0,
            bits + denominator.bit_length(), budget=budget,
        )
        value = min(values.flat) if lower else max(abs(value) for value in values.flat)
        return Fraction(int(value), denominator)

    bits = max(
        (net[2] + net[1].bit_length()
         for row in (*operands.jacobian, *operands.minors, (operands.determinant,))
         for net in row if net is not None),
        default=1,
    )
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        65, 64, 0, 16 * bits + 32, budget=budget,
    )
    return _entrywise_quality_source(
        tuple(tuple(extremum(net) for net in row) for row in operands.jacobian),
        tuple(tuple(extremum(net) for net in row) for row in operands.minors),
        extremum(operands.determinant, lower=True),
    )


def _prepare_box_quality_operands(
    polynomials: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxQualityOperands | None:
    """Prepare a nonaffine hexahedron's rational tensor Bernstein operands.

    Coordinate controls ``b_k`` scaled by ``C(n, k)`` multiply by convolution,
    so every Jacobian entry, cofactor and determinant net is formed once at its
    nominal per-axis degree without monomial products or basis reconversion.
    A nominal-degree net is a degree elevation of the minimal one, whose
    controls are convex combinations of it: every extremum is the monomial
    route's bound or tighter, and each remains a rigorous enclosure.
    """
    nets: list[tuple[tuple[int, ...], tuple[Fraction, ...]]] = []
    for polynomial in polynomials:
        degrees = tuple(
            max((index[axis] for index in polynomial), default=0) for axis in range(3)
        )
        controls = _dependency_phydrax_discretization__coordinate_enclosure.bernstein_coefficients(
            polynomial, "box", 3, budget=budget
        )
        rational = tuple(value for value in controls if type(value) is Fraction)
        if len(rational) != len(controls):
            return None
        nets.append((degrees, rational))
    count = sum(len(controls) for _, controls in nets)
    denominator_bits = sum(value.denominator.bit_length() for _, controls in nets for value in controls)
    numerator_bits = max(abs(value.numerator).bit_length() for _, controls in nets for value in controls)
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        count, 2 * count, 3, denominator_bits + numerator_bits + 1, budget=budget,
    )
    common = 1
    for _, controls in nets:
        for value in controls:
            common = lcm(common, value.denominator)
    jacobian: list[list[_BoxNet | None]] = []
    # Extrema are evaluated by the shared consumer after preparation/restriction.
    for degrees, controls in nets:
        shape = tuple(degree + 1 for degree in degrees)
        integer_bits = (
            max(abs(value.numerator).bit_length() for value in controls)
            + common.bit_length()
        )
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            2 * len(controls),
            4 * len(controls),
            3,
            integer_bits + max(degrees).bit_length() + sum(degrees) + 1,
            budget=budget,
        )
        integers = np.asarray(
            [value.numerator * (common // value.denominator) for value in controls],
            dtype=object,
        ).reshape(shape)
        row: list[_BoxNet | None] = []
        for axis in range(3):
            if degrees[axis] == 0:
                row.append(None)
                continue
            lowered = tuple(
                degree - int(index == axis) for index, degree in enumerate(degrees)
            )
            # Difference, degree factor, magnitude and binomial scaling per control.
            budget.reserve(4 * prod(degree + 1 for degree in lowered))
            difference = np.diff(integers, axis=axis) * degrees[axis]
            # Binomial weights satisfy C(n, k) < 2**n on every axis.
            row.append(
                (
                    difference * _box_binomial_weights(lowered),
                    integer_bits + 1 + degrees[axis].bit_length() + sum(lowered),
                )
            )
        jacobian.append(row)

    def product_net(
        row: int, column: int, other_row: int, other_column: int
    ) -> _BoxNet | None:
        first, second = jacobian[row][column], jacobian[other_row][other_column]
        return (
            None
            if first is None or second is None
            else _box_convolve(first, second, budget=budget)
        )

    minors: list[list[_BoxNet | None]] = []
    for removed_row in range(3):
        a, b = [index for index in range(3) if index != removed_row]
        minor_row: list[_BoxNet | None] = []
        for removed_column in range(3):
            c, d = [index for index in range(3) if index != removed_column]
            minor = _box_combine(
                product_net(a, c, b, d),
                product_net(a, d, b, c),
                subtract=True,
                budget=budget,
            )
            minor_row.append(minor)
        minors.append(minor_row)
    determinant: _BoxNet | None = None
    for column in range(3):
        entry, minor = jacobian[0][column], minors[0][column]
        if entry is not None and minor is not None:
            determinant = _box_combine(
                determinant,
                _box_convolve(entry, minor, budget=budget),
                subtract=column == 1,
                budget=budget,
            )
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        2, 2, 0, 3 * common.bit_length() + 1, budget=budget,
    )
    squared = common * common
    cubed = squared * common
    return _BoxQualityOperands(
        tuple(tuple(_box_quality_net(net, common, budget=budget) for net in row)
              for row in jacobian),
        tuple(tuple(_box_quality_net(net, squared, budget=budget) for net in row)
              for row in minors),
        _box_quality_net(determinant, cubed, budget=budget),
    )


def _box_bernstein_quality_source(
    polynomials: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource | None:
    operands = _prepare_box_quality_operands(polynomials, budget=budget)
    return None if operands is None else _box_quality_source(operands, budget=budget)


def _split_box_quality_axis(
    net: _BoxQualityNet,
    axis: int,
    parameter: Fraction,
    /,
    *,
    right: bool,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxQualityNet:
    """Integer homogeneous de Casteljau, with one denominator per full net."""
    values, denominator, bits = net
    degree = values.shape[axis] - 1
    if degree == 0:
        return net
    numerator, divisor = parameter.numerator, parameter.denominator
    lines = values.size // (degree + 1)
    bound = bits + degree * (divisor.bit_length() + 1)
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        3 * lines * degree * (degree + 1) // 2 + values.size + (degree + 1)**2 + degree + 3,
        4 * values.size, 3, bound + denominator.bit_length(),
        budget=budget,
    )
    pending = np.moveaxis(values, axis, 0)
    result = np.empty_like(pending)
    result[degree if right else 0] = pending[-1 if right else 0] * divisor**degree
    for level in range(1, degree + 1):
        pending = (divisor - numerator) * pending[:-1] + numerator * pending[1:]
        result[degree - level if right else level] = (
            pending[-1 if right else 0] * divisor**(degree - level)
        )
    return np.moveaxis(result, 0, axis), denominator * divisor**degree, bound


def _restrict_box_quality_net(
    net: _BoxQualityNet | None,
    intervals: tuple[tuple[Fraction, Fraction], ...],
    scale: Fraction,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _BoxQualityNet | None:
    if net is None:
        return None
    for axis, (lower, upper) in enumerate(intervals):
        if net[0].shape[axis] == 1 or (lower == 0 and upper == 1):
            continue
        if lower != 0:
            net = _split_box_quality_axis(net, axis, lower, right=True, budget=budget)
        if upper != 1:
            _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
                3, 3, 0,
                2 * (lower.denominator.bit_length() + upper.denominator.bit_length()) + 4,
                budget=budget,
            )
            net = _split_box_quality_axis(
                net, axis, (upper - lower) / (1 - lower), right=False, budget=budget
            )
    if scale == 1:
        return net
    values, denominator, bits = net
    bound = bits + abs(scale.numerator).bit_length()
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        values.size + 1, values.size, 3,
        bound + denominator.bit_length() + scale.denominator.bit_length(), budget=budget,
    )
    return values * scale.numerator, denominator * scale.denominator, bound


def _root_box_quality_source(
    element: CellGeometryElement,
    local: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    whole_root: bool = False,
) -> _MappedQualitySource | None:
    """Authenticate a complete child, then restrict private actual-root operands."""
    composed = _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement
    if not isinstance(element, composed) or any(
        type(value) is not Fraction for row in local for value in row
    ):
        return None
    rational = cast(tuple[tuple[Fraction, ...], ...], local)
    count = sum(len(row) for row in local)
    bits = max(
        (abs(value.numerator).bit_length() + value.denominator.bit_length()
         for row in rational for value in row), default=1,
    )
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        count, count, 0, bits, budget=budget,
    )
    # Public Fraction atoms can be mutated through object.__setattr__. The
    # private cache keys must own fresh numerical atoms, never those objects.
    bank = tuple(tuple(Fraction(value.numerator, value.denominator) for value in row)
                 for row in rational)
    _dependency_phydrax_discretization__coordinate_enclosure.coordinate_preparation_key(
        element, bank, budget=budget
    )
    current = element
    origins = (Fraction(0),) * 3
    action = tuple((axis, Fraction(1)) for axis in range(3))
    while isinstance(current, composed):
        if current.cell_kind != "hexahedron" or current.source_element.cell_kind != "hexahedron":
            return None
        chart_bits = max(
            (abs(numerator).bit_length() + abs(denominator).bit_length()
             for row in current.chart_coefficients for numerator, denominator in row),
            default=1,
        )
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            0, 3 * len(current.chart_coefficients),
            0, chart_bits, budget=budget,
        )
        polynomials = _dependency_phydrax_discretization__coordinate_enclosure.reference_composition_arguments(
            current, budget=budget
        )
        if polynomials is None or len(polynomials) != 3:
            return None
        _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
            sum(len(polynomial) for polynomial in polynomials) + 24, 24, 0,
            4 * (chart_bits + bits) + 16, budget=budget,
        )
        next_origins, next_action = [], []
        used: set[int] = set()
        for polynomial in polynomials:
            if any(type(value) is not Fraction or sum(index) > 1
                   for index, value in polynomial.items() if value != 0):
                return None
            offset = polynomial.get((0, 0, 0), Fraction(0))
            linear = tuple(polynomial.get(tuple(int(i == axis) for i in range(3)), Fraction(0))
                           for axis in range(3))
            nonzero = tuple(axis for axis, value in enumerate(linear) if value != 0)
            if len(nonzero) != 1 or nonzero[0] in used:
                return None
            axis = nonzero[0]
            used.add(axis)
            scale = linear[axis]
            if min(offset, offset + scale) < 0 or max(offset, offset + scale) > 1:
                return None
            next_origins.append(offset + scale * origins[axis])
            next_action.append((action[axis][0], scale * action[axis][1]))
        origins, action = tuple(next_origins), tuple(next_action)
        bits = 4 * (chart_bits + bits) + 16
        current = current.source_element
    _, authority = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_preparation_key(
        current, bank, budget=budget
    )
    # Retain our private snapshot even when general preparation found an equal
    # public bank. The unchanged authority still binds the full current tree.
    key: _dependency_phydrax_discretization__coordinate_enclosure.CoordinatePreparationKey = (
        (authority[0], authority[1], bank)
        if len(authority) == 3
        else (authority[0], authority[1], bank, authority[3])
    )
    cache = budget.root_box_quality_operands_cache
    if key not in cache:
        polynomials = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
            current, bank, budget=budget
        )
        operands = (
            None if polynomials is None
            else _prepare_box_quality_operands(polynomials, budget=budget)
        )
        # Scientific integer nets use object arrays, which the generic model
        # owner profiler deliberately rejects. Retain their actual array/base
        # headers and integer owners through the coordinate graph profiler.
        # The extra tuple's declared 16-byte slots cover its own and the
        # object's 8-byte pointer banks; no payload is hidden as an opaque array.
        payloads: list[tuple[np.ndarray, tuple[object, ...]]] = []
        seen_arrays: set[int] = set()
        if operands is not None:
            nets = (
                *(net for row in operands.jacobian for net in row),
                *(net for row in operands.minors for net in row),
                operands.determinant,
            )
            for net in nets:
                if net is None:
                    continue
                array: object = net[0]
                while isinstance(array, np.ndarray) and id(array) not in seen_arrays:
                    seen_arrays.add(id(array))
                    payloads.append((array, tuple(array.flat)))
                    array = array.base
        budget.retain_basis((
            key, operands,
            None if operands is None else (
                operands.jacobian, operands.minors, operands.determinant,
            ),
            tuple(payloads),
        ))
        cache[key] = operands
    operands = cache[key]
    if operands is None:
        return None
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        40, 32, 0, 6 * bits + 32, budget=budget,
    )
    intervals = (
        ((Fraction(0), Fraction(1)),) * 3 if whole_root else tuple(
            (min(origin, origin + scale), max(origin, origin + scale))
            for origin, (_, scale) in zip(origins, action, strict=True)
        )
    )
    inverse = tuple(next(source for source, (target, _) in enumerate(action) if target == axis)
                    for axis in range(3))
    scales = tuple(scale for _, scale in action)
    permutation_sign = -1 if sum(action[i][0] > action[j][0]
                                 for i in range(3) for j in range(i + 1, 3)) % 2 else 1
    determinant_scale = permutation_sign * scales[0] * scales[1] * scales[2]
    # Axis reversals and permutations only reorder controls. Absolute extrema
    # need no materialized transpose/reversal; determinant keeps the signed scale.
    restricted = _BoxQualityOperands(
        tuple(tuple(_restrict_box_quality_net(
            row[source], intervals, abs(scales[source]), budget=budget
        ) for source in inverse) for row in operands.jacobian),
        tuple(tuple(_restrict_box_quality_net(
            row[source], intervals,
            abs(scales[(source + 1) % 3] * scales[(source + 2) % 3]),
            budget=budget,
        ) for source in inverse) for row in operands.minors),
        _restrict_box_quality_net(
            operands.determinant, intervals, determinant_scale, budget=budget
        ),
    )
    return _box_quality_source(restricted, budget=budget)


def _monomial_quality_source(
    polynomials: tuple[_dependency_phydrax_discretization__coordinate_enclosure.Polynomial, ...],
    cell_kind: str,
    domain: str,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource:
    """Enclose affine, simplex, pyramid or irrational maps from monomial algebra."""
    jacobian = _dependency_phydrax_discretization__coordinate_enclosure.physical_jacobian(
        polynomials, cell_kind, 3, budget=budget
    )
    if jacobian is None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.jacobian must support physical-reference interval evaluation; got unsupported cell kind {cell_kind}"
        )
    minors = []
    for row in range(3):
        a, b = [index for index in range(3) if index != row]
        minor_row = []
        for column in range(3):
            c, d = [index for index in range(3) if index != column]
            minor_row.append(
                _dependency_phydrax_discretization__coordinate_enclosure.add(
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
            )
        minors.append(tuple(minor_row))
    # Laplace expansion reuses the physical Jacobian and its minors. In a
    # pyramid the physical Jacobian already removes the collapsed-coordinate
    # factors, so no second determinant/collapse preparation is necessary.
    determinant = _dependency_phydrax_discretization__coordinate_enclosure.sum_polynomials(
        tuple(
            _dependency_phydrax_discretization__coordinate_enclosure.multiply(
                jacobian[0][column],
                _dependency_phydrax_discretization__coordinate_enclosure.scale(
                    minors[0][column], Fraction(-1 if column == 1 else 1), budget=budget
                ),
                budget=budget,
            )
            for column in range(3)
        ),
        budget=budget,
    )
    determinant_lower = min(
        rational_coefficient(value, role="hex.quality.determinant_controls")
        for value in _dependency_phydrax_discretization__coordinate_enclosure.bernstein_coefficients(
            determinant, domain, 3, budget=budget
        )
    )
    if determinant_lower <= 0:
        return _MappedQualitySource((), Fraction(0), determinant_lower)
    return _entrywise_quality_source(
        tuple(
            tuple(_polynomial_magnitude(value, domain, budget=budget) for value in row)
            for row in jacobian
        ),
        tuple(
            tuple(_polynomial_magnitude(minor, domain, budget=budget) for minor in row)
            for row in minors
        ),
        determinant_lower,
    )


_QuadraticForm = tuple[tuple[int, int, Fraction], ...]


class _MultilinearQualityChart(StrictModule):
    """Shared quality factors of one multilinear chart over affine tets.

    The chart is a hexahedral tensor chart or a rational collapsed pyramid,
    whose collapsed physical Jacobian is multilinear on the same unit box.
    ``forms`` hold the chart constants of ``|J_S v|^2 = v^T G v`` for every
    distinct chart column vertex vector, with ``G[i][j] = e_i . e_j`` the Gram
    matrix of a cell's source edges; ``determinant_lower`` is a rigorous
    ``inf det J_c``.
    """

    factors: _dependency_phydrax_discretization__cell_geometry_validity.MultilinearChartFactors
    forms: tuple[tuple[_QuadraticForm, ...], ...]
    determinant_lower: Fraction
    operations: int


def _multilinear_quality_chart(
    element: CellGeometryElement,
    cell_kind: str,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MultilinearQualityChart | None:
    """Earn one block's shared chart quality factors, or decline."""
    if cell_kind == "pyramid":
        factors = _dependency_phydrax_discretization__cell_geometry_validity.pyramid_chart_factors(
            element, budget=budget
        )
    elif cell_kind == "hexahedron" and isinstance(
        element,
        _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    ):
        factors = _dependency_phydrax_discretization__cell_geometry_validity.multilinear_chart_factors(
            element, cell_kind, budget=budget
        )
    else:
        return None
    if factors is None:
        return None
    forms: list[tuple[_QuadraticForm, ...]] = []
    for vectors in factors.columns:
        column: list[_QuadraticForm] = []
        for vector in dict.fromkeys(vectors):
            budget.reserve(2 * len(vector) * len(vector))
            column.append(
                tuple(
                    (i, j, (1 if i == j else 2) * vector[i] * vector[j])
                    for i in range(len(vector))
                    for j in range(i, len(vector))
                    if vector[i] and vector[j]
                )
            )
        forms.append(tuple(column))
    # The nominal-degree hexahedral chart net is a degree elevation of the
    # own-degree one; both enclose ``det J_c``, so the larger lower control
    # remains rigorous. The collapsed pyramid keeps its own-degree controls.
    lower = factors.determinant_lower
    if isinstance(
        element,
        _dependency_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    ):
        polynomials = _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
            element.chart_element,
            tuple(
                tuple(Fraction(numerator, denominator) for numerator, denominator in row)
                for row in element.chart_coefficients
            ),
            budget=budget,
        )
        if polynomials is not None and len(polynomials) == 3:
            nominal = _box_bernstein_quality_source(polynomials, budget=budget)
            if nominal is not None:
                lower = max(lower, nominal.determinant_lower)
    dimension = len(factors.source_nodes) - 1
    quadratic = sum(
        2 * len(form) - 1 for column in forms for form in column if form
    ) + sum(len(column) - 1 for column in forms)
    # Per cell, beyond its source simplex's shared frame and Gram matrix: the
    # quadratic forms and their maxima, the determinant product, the norm sum,
    # the pair products and their sum, and the product.
    operations = (
        quadratic
        + 1
        + (dimension - 1)
        + (dimension * (dimension - 1) - 1)
        + 1
    )
    return _MultilinearQualityChart(factors, tuple(forms), lower, operations)


def _multilinear_quality_source(
    chart: _MultilinearQualityChart,
    frame: _SourceFrame,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource | None:
    """Bound a multilinear chart over an affine simplex from its shared factors.

    ``J(xi) = J_S J_c(xi)``: each column ``J[:, k](xi)`` is a convex combination
    of ``J_S v`` over the chart's vertex vectors ``v``, so the norm's convexity
    gives ``sup |J[:, k]|^2 <= max_v v^T G v`` with ``G`` the source frame's
    Gram matrix. ``inf det J`` is ``det J_S`` times the chart's shared extreme.
    ``|J|_2 <= |J|_F`` and ``|cof J|_2 <= |cof J|_F``, whose squared columns
    ``|J[:, a] x J[:, b]|^2`` are at most ``c_a^2 c_b^2``. A nonpositive lower
    bound declines to the owning polynomial route.
    """
    factors = chart.factors
    dimension = len(frame.edges)
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        chart.operations,
        4 * dimension + 12,
        0,
        12 * frame.bits + 4 * factors.bits + 64,
        budget=budget,
    )
    measure = frame.measure
    determinant_lower = measure * (
        chart.determinant_lower if measure > 0 else factors.determinant_upper
    )
    if determinant_lower <= 0:
        return None
    gram = frame.gram
    column_squared = tuple(
        max(
            sum((weight * gram[i][j] for i, j, weight in form), Fraction(0))
            for form in column
        )
        for column in chart.forms
    )
    norm_squared = sum(column_squared, Fraction(0))
    pairs = sum(
        (
            column_squared[a] * column_squared[b]
            for a in range(dimension)
            for b in range(a + 1, dimension)
        ),
        Fraction(0),
    )
    return _MappedQualitySource(column_squared, norm_squared * pairs, determinant_lower)


# Edge frame ``R`` of the regular tetrahedron (0,0,0), (1,1,0), (1,0,1),
# (0,1,1); it is symmetric and ``2 R^{-1}`` has the signed unit entries below.
_REGULAR_FRAME = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
_REGULAR_INVERSE_SIGNS = ((1, 1, -1), (1, -1, 1), (-1, 1, 1))
# Bernstein product weights ``C(1, a) C(2, b) / C(3, a + b)`` per axis.
_CUBIC_PRODUCT_WEIGHTS = {
    (0, 0): Fraction(1),
    (0, 1): Fraction(2, 3),
    (1, 0): Fraction(1, 3),
    (0, 2): Fraction(1, 3),
    (1, 1): Fraction(2, 3),
    (1, 2): Fraction(1),
}
_SPECTRAL_MARGINS = (Fraction(1, 2**20), Fraction(1, 2**10))
_Matrix = tuple[tuple[Fraction, ...], ...]
# Exact Gram (30), one floating eigen estimate (27), the shifted diagonal (3),
# and the 1x1, 2x2 and 3x3 leading minors with their signs (1 + 3 + 14 + 3).
_SPECTRAL_BOUND_OPERATIONS = 30 + 27 + 3 + 1 + 3 + 14 + 3


def _cofactor_matrix(matrix: _Matrix, /) -> _Matrix:
    """Signed 2x2 minors: ``det M = sum_c M[0][c] C[0][c]``."""
    result = []
    for row in range(3):
        a, b = (index for index in range(3) if index != row)
        entries = []
        for column in range(3):
            p, q = (index for index in range(3) if index != column)
            minor = _dependency_phydrax_linalg__small_batched.exact_cofactor_determinant_2(
                (matrix[a][p], matrix[a][q]), (matrix[b][p], matrix[b][q])
            )
            entries.append(minor if (row + column) % 2 == 0 else -minor)
        result.append(tuple(entries))
    return tuple(result)


def _column_gram(matrix: _Matrix, /) -> _Matrix:
    """Exact ``M^T M`` of a 3x3 matrix."""
    products = {
        (i, j): sum((row[i] * row[j] for row in matrix), Fraction(0))
        for i in range(3)
        for j in range(i, 3)
    }
    return tuple(tuple(products[min(i, j), max(i, j)] for j in range(3)) for i in range(3))


def _positive_definite(matrix: _Matrix, /) -> bool:
    """Sylvester's criterion on the leading principal minors of a symmetric 3x3."""
    return (
        matrix[0][0] > 0
        and _dependency_phydrax_linalg__small_batched.exact_cofactor_determinant_2(
            (matrix[0][0], matrix[0][1]), (matrix[1][0], matrix[1][1])
        )
        > 0
        and _dependency_phydrax_linalg__small_batched.exact_cofactor_determinant_3(
            cast(tuple[Fraction, Fraction, Fraction], matrix[0]),
            cast(tuple[Fraction, Fraction, Fraction], matrix[1]),
            cast(tuple[Fraction, Fraction, Fraction], matrix[2]),
        )
        > 0
    )


def _eigen_estimates(gram: _Matrix, /) -> np.ndarray:
    """Floating eigenvalue estimates that only propose verified brackets."""
    return np.linalg.eigvalsh(np.asarray(gram, dtype=np.float64))


def _eigen_bound(gram: _Matrix, estimate: float, /, *, upper: bool) -> Fraction | None:
    """Exactly verified ``U > lambda_max`` or ``0 < L < lambda_min`` of a Gram matrix.

    A floating ``estimate`` only proposes the bound; ``U I - G`` or ``G - L I``
    must be positive definite by Sylvester's criterion in exact arithmetic.
    """
    proposal = Fraction(float(estimate))
    if proposal <= 0:
        return None
    for margin in _SPECTRAL_MARGINS:
        bound = proposal * (1 + margin if upper else 1 - margin)
        shifted = tuple(
            tuple(
                (bound - value if upper else value - bound)
                if i == j
                else (-value if upper else value)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(gram)
        )
        if _positive_definite(shifted):
            return bound
    return None


class _ChartCondition(StrictModule):
    """Rigorous ``sup kappa_2(R J_c)^2`` of one multilinear chart on its unit box.

    ``M = R J_c`` is the multilinear interpolation of its box-vertex matrices
    ``M_v`` over the axes it depends on. With positive vertex weights ``h_v``
    and ``h`` their multilinear interpolation, exact Bernstein products give
    the degree-two net of ``cof M = sum_{a,b} N_a N_b K(M_a, M_b)`` (``K`` the
    polarized cofactor), the degree-three net of ``det M = M[0] . cof M[0]``,
    and the nets of ``h^2`` and ``h^3``. Norm convexity over nonnegative
    Bernstein weights gives ``|M| <= S_1 h``, ``|cof M| <= S_2 h^2`` and
    ``|det M| >= D h^3``, so ``kappa_2(M) = |M| |cof M| / |det M| <= S_1 S_2 / D``
    everywhere. The weights ``h_v ~ |det M_v|^(1/3)`` only steer tightness.
    """

    condition_squared: Fraction


def _chart_condition_operations(active: int, /) -> int:
    """Rational operations of ``_chart_condition`` after its vertex matrices."""
    vertices, quadratic, cubic = 2**active, 3**active, 4**active
    pairs = vertices * (vertices - 1) // 2
    return (
        vertices * 27  # vertex cofactors
        + pairs * (9 + 27 + 18 + 18)  # polarized cofactors and weighted sums
        + vertices * 9  # diagonal cofactor terms
        + pairs * 4
        + vertices * 2  # h^2 net
        + vertices * quadratic * (5 + 2 + 2)  # det net: row-0 products, weights
        + vertices * quadratic * 3  # h^3 net
        + (vertices + quadratic) * (_SPECTRAL_BOUND_OPERATIONS + 2)
        + cubic * 2  # determinant ratios
        + 4
    )


def _chart_condition(
    factors: _dependency_phydrax_discretization__cell_geometry_validity.MultilinearChartFactors,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _ChartCondition | None:
    """Earn one chart's shared condition bound in the regular frame, or decline."""
    if len(factors.source_nodes) != 4 or len(factors.columns) != 3:
        return None
    corners = tuple(product((0, 1), repeat=3))
    bits = max(
        coefficient_bits(value)
        for column in factors.columns
        for vector in column
        for value in vector
    )
    # ``R M_v`` (one sum per entry) and the exact axis-independence tests.
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        8 * 9 + 3 * 4 * 9, 8 * 9, 3, 2 * bits + 8, budget=budget
    )
    all_matrices = {
        vertex: tuple(
            tuple(
                sum(
                    (
                        factors.columns[column][index][axis]
                        for axis in range(3)
                        if _REGULAR_FRAME[row][axis]
                    ),
                    Fraction(0),
                )
                for column in range(3)
            )
            for row in range(3)
        )
        for index, vertex in enumerate(corners)
    }
    # ``M`` is constant along an axis whose paired vertex matrices agree; the
    # interpolation over the remaining axes is then the same map.
    active = tuple(
        axis
        for axis in range(3)
        if any(
            all_matrices[vertex]
            != all_matrices[(*vertex[:axis], 1, *vertex[axis + 1 :])]
            for vertex in corners
            if not vertex[axis]
        )
    )
    vertices = tuple(
        vertex
        for vertex in corners
        if not any(value for axis, value in enumerate(vertex) if axis not in active)
    )
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        _chart_condition_operations(len(active)),
        len(vertices) * 9 + 3 ** len(active) * 10 + 4 ** len(active) * 2 + 8,
        3,
        8 * (bits + 64) + 64,
        budget=budget,
    )
    matrices: dict[tuple[int, ...], _Matrix] = {}
    weights: dict[tuple[int, ...], Fraction] = {}
    for vertex in vertices:
        matrix = all_matrices[vertex]
        weight = Fraction(
            float(abs(np.linalg.det(np.asarray(matrix, dtype=np.float64)))) ** (1.0 / 3.0)
        )
        if weight <= 0:
            return None
        matrices[vertex], weights[vertex] = matrix, weight
    cofactors = {vertex: _cofactor_matrix(matrix) for vertex, matrix in matrices.items()}
    zero = tuple(tuple(Fraction(0) for _ in range(3)) for _ in range(3))
    nets: dict[tuple[int, ...], _Matrix] = {}
    squares: dict[tuple[int, ...], Fraction] = {}

    def accumulate(node: tuple[int, ...], term: _Matrix, scale: Fraction) -> None:
        current = nets.get(node, zero)
        nets[node] = tuple(
            tuple(value + scale * add for value, add in zip(left, right, strict=True))
            for left, right in zip(current, term, strict=True)
        )

    for vertex in vertices:
        node = tuple(2 * value for value in vertex)
        accumulate(node, cofactors[vertex], Fraction(1))
        squares[node] = squares.get(node, Fraction(0)) + weights[vertex] ** 2
    for first, second in combinations(vertices, 2):
        node = tuple(a + b for a, b in zip(first, second, strict=True))
        scale = Fraction(1, 2 ** sum(1 for value in node if value == 1))
        combined = _cofactor_matrix(
            tuple(
                tuple(a + b for a, b in zip(left, right, strict=True))
                for left, right in zip(matrices[first], matrices[second], strict=True)
            )
        )
        polar = tuple(
            tuple(c - a - b for c, a, b in zip(rows[0], rows[1], rows[2], strict=True))
            for rows in zip(combined, cofactors[first], cofactors[second], strict=True)
        )
        accumulate(node, polar, scale)
        squares[node] = (
            squares.get(node, Fraction(0))
            + 2 * scale * weights[first] * weights[second]
        )
    determinants: dict[tuple[int, ...], Fraction] = {}
    cubes: dict[tuple[int, ...], Fraction] = {}
    for vertex in vertices:
        for node, cofactor in nets.items():
            target = tuple(a + b for a, b in zip(vertex, node, strict=True))
            scale = prod(
                (
                    _CUBIC_PRODUCT_WEIGHTS[a, b]
                    for a, b in zip(vertex, node, strict=True)
                ),
                start=Fraction(1),
            )
            row = sum(
                (
                    value * entry
                    for value, entry in zip(matrices[vertex][0], cofactor[0], strict=True)
                ),
                Fraction(0),
            )
            determinants[target] = determinants.get(target, Fraction(0)) + scale * row
            cubes[target] = (
                cubes.get(target, Fraction(0)) + scale * weights[vertex] * squares[node]
            )
    sign = 1 if determinants[(0, 0, 0)] > 0 else -1
    floor = min(sign * value / cubes[node] for node, value in determinants.items())
    if floor <= 0:
        return None
    column_bound = Fraction(0)
    for vertex, matrix in matrices.items():
        gram = _column_gram(matrix)
        bound = _eigen_bound(gram, _eigen_estimates(gram)[-1], upper=True)
        if bound is None:
            return None
        column_bound = max(column_bound, bound / weights[vertex] ** 2)
    cofactor_bound = Fraction(0)
    for node, cofactor in nets.items():
        gram = _column_gram(cofactor)
        bound = _eigen_bound(gram, _eigen_estimates(gram)[-1], upper=True)
        if bound is None:
            return None
        cofactor_bound = max(cofactor_bound, bound / squares[node] ** 2)
    return _ChartCondition(column_bound * cofactor_bound / (floor * floor))


_Rows = tuple[tuple[Fraction, ...], ...]


class _SourceFrame(StrictModule):
    """One affine source simplex's exact frame, shared by all cells over it.

    ``edges`` are the columns of ``J_S``, ``measure`` is ``det J_S`` and
    ``gram[i][j] = e_i . e_j``.
    """

    edges: _Rows
    measure: Fraction
    gram: _Matrix
    bits: int


def _exact_source_rows(
    local: _dependency_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    nodes: tuple[int, ...],
    /,
) -> _Rows | None:
    """A cell's exact rational source corners, or None to decline."""
    corners = tuple(local[node] for node in nodes)
    if any(
        len(row) != 3 or any(type(value) is not Fraction for value in row)
        for row in corners
    ):
        return None
    return cast(_Rows, corners)


def _source_frame(
    rows: _Rows,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _SourceFrame:
    bits = max(coefficient_bits(value) for row in rows for value in row)
    # The frame and the six Gram dot products.
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        _dependency_phydrax_discretization__cell_geometry_validity.affine_simplex_frame_operations(3)
        + 6 * 5,
        9 + 6 + 8,
        0,
        16 * 3 * (bits + 1) + 64,
        budget=budget,
    )
    edges, measure = _dependency_phydrax_discretization__cell_geometry_validity.affine_simplex_frame(
        rows
    )
    gram = _column_gram(tuple(tuple(edge[row] for edge in edges) for row in range(3)))
    return _SourceFrame(edges, measure, gram, 4 * (bits + 1))


# ``J_S R^{-1}`` as signed halved edge sums, its Gram matrix, one floating
# eigen estimate, both verified brackets (shift and leading minors) and the ratio.
_SOURCE_CONDITION_OPERATIONS = (
    9 * 3 + 30 + 27 + 2 * (_SPECTRAL_BOUND_OPERATIONS - 30 - 27) + 1
)


def _source_condition_squared(
    frame: _SourceFrame,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Fraction | None:
    """Rigorous ``kappa_2(J_S R^{-1})^2 <= U / L`` of one source simplex.

    ``U`` and ``L`` are exactly verified eigen brackets of the Gram matrix of
    ``J_S R^{-1}``, the map from the regular tetrahedron to the source cell.
    """
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        _SOURCE_CONDITION_OPERATIONS, 9 + 6 + 8, 0, 8 * frame.bits + 64, budget=budget
    )
    edges = frame.edges
    regular = tuple(
        tuple(
            sum(
                (
                    edges[axis][row] if signs[column] > 0 else -edges[axis][row]
                    for axis, signs in enumerate(_REGULAR_INVERSE_SIGNS)
                ),
                Fraction(0),
            )
            / 2
            for column in range(3)
        )
        for row in range(3)
    )
    gram = _column_gram(regular)
    estimates = _eigen_estimates(gram)
    upper = _eigen_bound(gram, estimates[-1], upper=True)
    lower = _eigen_bound(gram, estimates[0], upper=False)
    return None if upper is None or lower is None else upper / lower


def _conditioned_quality_source(
    chart: _ChartCondition,
    source_condition: Fraction,
    source: _MappedQualitySource,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource:
    """Tighten a multilinear cell's aspect bound through the regular frame.

    ``J = (J_S R^{-1})(R J_c)`` gives ``kappa_2(J) <= kappa_2(J_S R^{-1})
    sup kappa_2(R J_c)``; the smaller of this and the existing rigorous bound
    is kept.
    """
    budget.reserve(5)
    aspect_squared = (
        source_condition
        * chart.condition_squared
        * source.determinant_lower
        * source.determinant_lower
    )
    if aspect_squared >= source.aspect_squared:
        return source
    return _MappedQualitySource(
        source.column_squared, aspect_squared, source.determinant_lower
    )


def _affine_quality_source(
    chart: _dependency_phydrax_discretization__cell_geometry_validity.AffineChartFrame,
    frame: _SourceFrame,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MappedQualitySource | None:
    """Exact quality data of an affine simplex composed over an affine simplex.

    ``J = J_S J_c`` is constant, so its entries, unsigned minors and
    ``det J = det J_S det J_c`` are the monomial route's exact constant
    magnitudes; a nonpositive determinant declines to that route's refusal.
    """
    # The determinant product, nine three-term products, nine minors, the
    # column sums and both one/infinity norm products.
    _dependency_phydrax_discretization__coordinate_budget.reserve_polynomial(
        1 + 9 * 5 + 9 * 3 + 3 * 5 + 2 * (2 * (3 * 2 + 2) + 1) + 1,
        9 + 9 + 3 + 8,
        0,
        4 * (frame.bits + chart.bits) + 64,
        budget=budget,
    )
    determinant = frame.measure * chart.determinant
    if determinant <= 0:
        return None
    jacobian = tuple(
        tuple(
            sum(
                (
                    frame.edges[axis][row] * chart.columns[column][axis]
                    for axis in range(3)
                ),
                Fraction(0),
            )
            for column in range(3)
        )
        for row in range(3)
    )
    minors = tuple(
        tuple(abs(value) for value in row) for row in _cofactor_matrix(jacobian)
    )
    return _entrywise_quality_source(
        tuple(tuple(abs(value) for value in row) for row in jacobian),
        minors,
        determinant,
    )


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
    local = tuple(coordinates[int(index)] for index in route)
    source = (
        _root_box_quality_source(element, local, budget=budget)
        if cell_kind == "hexahedron" and domain == "box" else None
    )
    if source is None:
        polynomials = (
            _dependency_phydrax_discretization__coordinate_enclosure.coordinate_polynomials(
                element, local, budget=budget
            )
        )
        if polynomials is None:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"hex.quality.element must provide polynomial interval quality data; got {element.element_id}"
            )
        source = (
            _box_bernstein_quality_source(polynomials, budget=budget)
            if cell_kind == "hexahedron"
            and any(sum(index) > 1 for polynomial in polynomials for index in polynomial)
            else None
        )
        if source is None:
            source = _monomial_quality_source(polynomials, cell_kind, domain, budget=budget)
    if source.determinant_lower <= 0:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"hex.quality.determinant_lower must be positive; got {source.determinant_lower}"
        )
    return source


def _enclose_mapped_cell_quality(source: _MappedQualitySource, /) -> _MappedQualityBounds:
    column_squared, determinant_lower = source.column_squared, source.determinant_lower
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
    aspect = _sqrt_upper(source.aspect_squared / squared)
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
    using matrix one/infinity norms and polynomial cofactors. A multilinear
    hexahedral or collapsed-pyramid chart over an affine simplex uses its shared
    chart factors: the smaller of the Frobenius norms of ``J`` and ``cof J`` and
    the regular-frame product ``kappa_2(J_S R^{-1}) sup kappa_2(R J_c)``. An
    affine simplex composed over an affine simplex uses its exact constant
    Jacobian.
    """
    return _exact_mapped_volume_quality(mesh, geometry, scope=scope, budget=budget)


def _exact_mapped_volume_quality(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    tensor_policy: NativeHexGridSchedule | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Private policy enters only after the exhaustive tensor theorem authenticates
    # complete roots, counts, routes and every child chart.
    if any(
        block.cell_kind not in ("hexahedron", "tetrahedron", "pyramid")
        for block in mesh.blocks
    ):
        raise ValueError(
            f"hex.quality.blocks must contain hex, tet, or pyramid cells; got {tuple(block.cell_kind for block in mesh.blocks)}"
        )
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.temporary_scope(),
    ):
        elements, routes, _ = geometry.resolve(
            mesh, budget=budget, exact_source_prepared=False
        )
        count = sum(block.cell_count for block in mesh.blocks)
        # Host route copies are temporary; scalar arithmetic is not work pricing.
        budget.reserve(0, 16 * sum(route.size for route in routes) + 4096)
        coordinates = _dependency_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
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
        charts: dict[str, _MultilinearQualityChart | None] = {}
        conditions: dict[str, _ChartCondition | None] = {}
        affines: dict[
            str,
            _dependency_phydrax_discretization__cell_geometry_validity.AffineChartFrame
            | None,
        ] = {}
        # Cells over one affine source simplex share its exact frame and
        # regular-frame condition; both stay charged for this quality phase.
        source_frames: dict[_Rows, _SourceFrame] = {}
        source_conditions: dict[_Rows, Fraction | None] = {}
        root_bounds: dict[int, _MappedQualityBounds | None] = {}
        row_index = 0
        for block, element, block_routes in zip(
            mesh.blocks, elements, host_routes, strict=True
        ):
            domain = "simplex" if block.cell_kind == "tetrahedron" else "box"
            chart = condition = affine = None
            if block.cell_count:
                key = f"{block.cell_kind}:{element.element_id}"
                if key not in charts:
                    charts[key] = _multilinear_quality_chart(
                        element, block.cell_kind, budget=budget
                    )
                    earned = charts[key]
                    conditions[key] = (
                        None
                        if earned is None
                        else _chart_condition(earned.factors, budget=budget)
                    )
                    affines[key] = (
                        _dependency_phydrax_discretization__cell_geometry_validity.affine_chart_frame(
                            element, block.cell_kind, budget=budget
                        )
                        if block.cell_kind == "tetrahedron"
                        else None
                    )
                chart, condition, affine = charts[key], conditions[key], affines[key]
            nodes: tuple[int, ...] | None = None
            if chart is not None:
                nodes = chart.factors.source_nodes
            elif affine is not None:
                nodes = affine.source_nodes
            origins = (
                None if tensor_policy is None or geometry.restriction_source is None
                else np.asarray(geometry.restriction_source.block_parent_cell_ids[block.name])
            )
            for local_index, route in enumerate(np.asarray(block_routes, dtype=np.int64)):
                source = None
                local = tuple(coordinates[int(index)] for index in route)
                if origins is not None:
                    assert tensor_policy is not None
                    origin = int(origins[local_index])
                    if origin not in root_bounds:
                        budget.reserve(1, 256)
                        with budget.polynomial_preparation_scope():
                            whole = _root_box_quality_source(
                                element, local, budget=budget, whole_root=True
                            )
                            candidate = (
                                None if whole is None or whole.determinant_lower <= 0
                                else _enclose_mapped_cell_quality(whole)
                            )
                            root_bounds[origin] = (
                                candidate if candidate is not None
                                and candidate.scaled >= tensor_policy.minimum_scaled_jacobian
                                and candidate.mean >= tensor_policy.minimum_mean_ratio
                                and candidate.aspect <= tensor_policy.maximum_aspect_ratio
                                else None
                            )
                            if root_bounds[origin] is not None:
                                budget.retain_owner(root_bounds[origin])
                            del candidate, whole
                    root_bound = root_bounds[origin]
                    if root_bound is not None:
                        scaled[row_index], mean[row_index], aspect[row_index] = (
                            root_bound.scaled, root_bound.mean, root_bound.aspect
                        )
                        row_index += 1
                        continue
                rows = None if nodes is None else _exact_source_rows(local, nodes)
                if rows is not None:
                    budget.reserve(1)
                    frame = source_frames.get(rows)
                    if frame is None:
                        frame = source_frames[rows] = _source_frame(rows, budget=budget)
                    if condition is not None and rows not in source_conditions:
                        source_conditions[rows] = _source_condition_squared(
                            frame, budget=budget
                        )
                    source_condition = source_conditions.get(rows)
                    with budget.temporary_scope():
                        if chart is not None:
                            source = _multilinear_quality_source(
                                chart, frame, budget=budget
                            )
                            if (
                                source is not None
                                and condition is not None
                                and source_condition is not None
                            ):
                                source = _conditioned_quality_source(
                                    condition, source_condition, source, budget=budget
                                )
                        elif affine is not None:
                            source = _affine_quality_source(affine, frame, budget=budget)
                        if source is not None:
                            bounds = _enclose_mapped_cell_quality(source)
                if source is None:
                    # Only three scalar bounds survive this cell. Retire its
                    # polynomial workspaces and preparation memos together;
                    # source coefficients retained by the request remain charged.
                    with budget.polynomial_preparation_scope():
                        source = _prepare_mapped_quality_source(
                            element, coordinates, route, block.cell_kind, domain, budget=budget
                        )
                        bounds = _enclose_mapped_cell_quality(source)
                scaled[row_index], mean[row_index], aspect[row_index] = (
                    bounds.scaled,
                    bounds.mean,
                    bounds.aspect,
                )
                del source, bounds
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
