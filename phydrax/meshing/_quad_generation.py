#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, source-faithful quad extraction on authored triangle surface charts.

The barycentric-dual route partitions every triangle into three quads.
Shared edge identities, not coordinate coincidence, close the patches. Affine
sources use bilinear vertex maps; polynomial and rational spline source maps
retain their source coefficients through rational bilinear reference
composition. Spline triangle charts must remain within their source knot span.
Feature-separated cross fields use source chart jets when maps are supplied.
This route deliberately subdivides boundary chains: immutable-chain
recombination, arbitrary CAD projection and field-guided integer-grid extraction
remain distinct research operations, not properties inferred from these templates.

Association inheritance uses native predicate proofs of containment in the
affine source simplex. Rounded nodes outside that hull retain bounded
residuals; independent PLC coverage does not silently turn them into zero-residual rows.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax._meshcore as _owner_phydrax__meshcore
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_geometry_validity as _owner_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._exact_plc_geometry as _owner_phydrax_discretization__exact_plc_geometry
import phydrax.linalg._small_batched as _dependency_phydrax_linalg__small_batched
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._dual_extraction as dual_extraction_owner
import phydrax.meshing._family_execution as family_execution_owner
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope

from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from ..discretization import (
    CellBlock,
    CellMesh,
)
from ..optim import (
    Bounds,
    minimize,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ..typing import checked, ConvertibleToArray
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed


def _cross_energy(
    angles: Array, arguments: tuple[Array, Array, Array, Array], /
) -> Array:
    adjacent, transport, feature_rows, feature_angles = arguments
    mismatch = angles[adjacent[:, 1]] - angles[adjacent[:, 0]] - transport
    feature = angles[feature_rows] - feature_angles
    return jnp.sum(1.0 - jnp.cos(4.0 * mismatch)) + 4.0 * jnp.sum(
        1.0 - jnp.cos(4.0 * feature)
    )


def _surface_patch_ids(face_count: int, adjacent: np.ndarray, /) -> np.ndarray:
    """Label connected face patches in first-source-face order."""
    parent = np.arange(face_count, dtype=np.int64)

    def root(row: int, /) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = int(parent[row])
        return row

    for left, right in adjacent.tolist():
        first, second = root(left), root(right)
        parent[max(first, second)] = min(first, second)
    roots = np.asarray([root(row) for row in range(face_count)], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1].astype(np.int64, copy=False)


def _mapped_surface_directions(
    mesh: CellMesh,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, tuple[int, int]], np.ndarray]]:
    """Use source chart jets, not carrier chords, for curved face/feature frames."""

    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        elements, routes, coefficients = geometry.resolve(
            mesh, budget=budget, exact_source_prepared=False
        )
        reference = np.asarray(
            ((1.0 / 3.0, 1.0 / 3.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
            dtype=np.float64,
        )
        edge_vectors = np.asarray(
            ((1.0, 0.0), (-1.0, 1.0), (0.0, -1.0)), dtype=np.float64
        )
        values = np.asarray(coefficients, dtype=np.float64)
        first, second = [], []
        directions: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
        cursor = 0
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            scalar = _owner_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                element, "Surface frames"
            )
            point_queries = reference.shape[0] * block.cell_count
            family_execution_owner.count_geometry_queries(
                point_queries, work_units=point_queries, budget=budget
            )
            gradients = np.asarray(scalar.tabulate(reference)[1], dtype=np.float64)
            derivatives = (
                np.swapaxes(gradients, -1, -2)[None, ...]
                @ values[np.asarray(route)][:, None, ...]
            )
            if mesh.ambient_dimension == 2:
                derivatives = np.pad(derivatives, ((0, 0), (0, 0), (0, 0), (0, 1)))
            first.append(derivatives[:, 0, 0])
            second.append(derivatives[:, 0, 1])
            tangents = np.sum(derivatives[:, 1:] * edge_vectors[None, :, :, None], axis=2)
            for row, triangle in enumerate(np.asarray(block.vertices).tolist()):
                for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
                    key = tuple(sorted((triangle[a], triangle[b])))
                    directions[(cursor + row, key)] = tangents[row, edge] * (
                        1.0 if triangle[a] == key[0] else -1.0
                    )
            cursor += block.cell_count
    return np.concatenate(first), np.concatenate(second), directions


@checked
def prepare_surface_cross_field(
    mesh: CellMesh,
    /,
    *,
    feature_edges: ConvertibleToArray | None = None,
    maximum_steps: int = 100,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dual_extraction_owner.SurfaceCrossField:
    """Minimize a transported fourth-harmonic field using native optim/linalg.

    Boundary edges are always feature constraints. Additional feature edges are
    source connectivity edge *rows*, not inferred geometric identities. They cut
    the transport graph into patches, each with its own frame field; a protected
    crease must not smooth one patch's direction into another. Multiple
    incompatible directions are retained as achieved residuals, never certified
    as zero-residual alignment. The patches are not integer-grid parameterizations.
    """
    measurement_started = phase_started(record_phase)
    if mesh.topology.dimension != 2:
        raise TypeError(
            f"prepare_surface_cross_field.mesh.dimension must be 2; got {mesh.topology.dimension}."
        )
    if mesh.ambient_dimension not in (2, 3):
        raise ValueError(
            f"prepare_surface_cross_field.mesh.ambient_dimension must be 2 or 3; got {mesh.ambient_dimension}."
        )
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError(
            f"prepare_surface_cross_field.mesh.blocks must contain triangle cells; got {tuple(block.cell_kind for block in mesh.blocks)}."
        )
    triangles = _dependency_phydrax_meshing__host_banks.entities(mesh, 2)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    local = points[triangles]
    mapped_directions = None
    if source_geometry is None:
        first, second = local[:, 1] - local[:, 0], local[:, 2] - local[:, 0]
        geometry_id = mesh.geometry_id
    else:
        first, second, mapped_directions = _mapped_surface_directions(
            mesh, source_geometry, scope=scope, budget=budget
        )
        geometry_id = (
            _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                source_geometry
            )
        )
    normal = np.cross(first, second)
    lengths, areas = np.linalg.norm(first, axis=1), np.linalg.norm(normal, axis=1)
    if np.any(lengths == 0.0) or np.any(areas == 0.0):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"prepare_surface_cross_field.tangent_frame must have nonzero tangent length and area; got minimum length={float(np.min(lengths))}, area={float(np.min(areas))}."
        )
    tangent = first / lengths[:, None]
    normal /= areas[:, None]
    frames = np.stack((tangent, np.cross(normal, tangent)), axis=1)
    edges = _dependency_phydrax_meshing__host_banks.entities(mesh, 1)
    owners: dict[tuple[int, int], list[int]] = {
        tuple(sorted(edge)): [] for edge in edges.tolist()
    }
    for row, triangle in enumerate(triangles.tolist()):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            owners[tuple(sorted((triangle[a], triangle[b])))].append(row)
    if feature_edges is None:
        explicit: set[int] = set()
    else:
        features_ = np.asarray(feature_edges)
        if features_.ndim != 1 or features_.dtype.kind not in "iu":
            raise TypeError(
                f"prepare_surface_cross_field.feature_edges must be an integer vector of source edge rows; got shape={features_.shape}, dtype={features_.dtype}."
            )
        explicit = set(features_.tolist())
    if any(row < 0 or row >= edges.shape[0] for row in explicit):
        raise ValueError(
            f"prepare_surface_cross_field.feature_edges must index source edge rows; got {sorted(explicit)}, edge_count={edges.shape[0]}."
        )
    incidences = [owners[tuple(sorted(edge))] for edge in edges.tolist()]
    if any(len(faces) != 1 and len(faces) != 2 for faces in incidences):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"prepare_surface_cross_field.face_links must have one or two incident faces; got incidence counts {tuple(sorted({len(faces) for faces in incidences}))}."
        )
    counts = np.asarray([len(faces) for faces in incidences], dtype=np.int64)
    offsets = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts)))
    face_rows = np.asarray(
        [face for faces in incidences for face in faces], dtype=np.int64
    )
    edge_rows = np.repeat(np.arange(edges.shape[0], dtype=np.int64), counts)
    basis = frames[face_rows]
    direction = points[edges[edge_rows, 1]] - points[edges[edge_rows, 0]]
    if mapped_directions is not None:
        direction = np.asarray(
            [
                mapped_directions[(face, tuple(sorted(edges[edge].tolist())))]
                for face, edge in zip(face_rows.tolist(), edge_rows.tolist(), strict=True)
            ],
            dtype=np.float64,
        )
    gram = basis @ np.swapaxes(basis, -1, -2)
    right = np.sum(basis * direction[:, None, :], axis=-1)
    solve = _dependency_phydrax_linalg__small_batched.solve_small_linear(
        _dependency_phydrax_linalg__small_batched.SmallLinearSolvePlan(2), gram, right
    )
    coordinates, successful = jax.device_get((solve.value, solve.successful))
    if not np.all(np.asarray(successful, dtype=np.bool_)):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "prepare_surface_cross_field.frame_solve must succeed for every tangent frame; got singular or ill-conditioned frames."
        )
    all_angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    pairs, rotations, feature_rows, feature_angles, feature_edge_rows = [], [], [], [], []
    for row, faces in enumerate(incidences):
        edge_angles = all_angles[offsets[row] : offsets[row + 1]]
        if len(faces) == 2 and row not in explicit:
            pairs.append(faces)
            rotations.append(edge_angles[1] - edge_angles[0])
        if len(faces) == 1 or row in explicit:
            feature_edge_rows.append(row)
            feature_rows.extend(faces)
            feature_angles.extend(edge_angles.tolist())
    adjacent = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
    transport = np.asarray(rotations, dtype=np.float64)
    features = np.asarray(feature_rows, dtype=np.int32)
    directions = np.asarray(feature_angles, dtype=np.float64)
    cosine = np.bincount(
        features, weights=np.cos(4.0 * directions), minlength=triangles.shape[0]
    )
    sine = np.bincount(
        features, weights=np.sin(4.0 * directions), minlength=triangles.shape[0]
    )
    initial = 0.25 * np.arctan2(sine, cosine)
    arguments = (
        jnp.asarray(adjacent),
        jnp.asarray(transport),
        jnp.asarray(features),
        jnp.asarray(directions),
    )
    method = ProjectedLBFGS()
    work_rows = adjacent.shape[0] + features.shape[0]
    with family_execution_owner.field_work_allowance(
        method, work_rows, maximum_steps, scope=scope
    ) as field_ticket:
        result = minimize(
            _cross_energy,
            jnp.asarray(initial, dtype=jnp.float64),
            method=method,
            args=arguments,
            bounds=Bounds(
                jnp.full((triangles.shape[0],), -np.pi / 4.0, dtype=jnp.float64),
                jnp.full((triangles.shape[0],), np.pi / 4.0, dtype=jnp.float64),
            ),
            termination=OptimizationTermination(maximum_steps=maximum_steps),
        )
        family_execution_owner.charge_field_work(field_ticket, result, work_rows)
    angles = np.asarray(jax.device_get(result.parameters), dtype=np.float64)
    residuals = (
        np.abs(
            np.arctan2(
                np.sin(4.0 * (angles[features] - directions)),
                np.cos(4.0 * (angles[features] - directions)),
            )
        )
        / 4.0
    )
    record_elapsed(record_phase, "frame_solve", measurement_started)
    patch_ids = _surface_patch_ids(triangles.shape[0], adjacent)
    return dual_extraction_owner.SurfaceCrossField(
        frames,
        angles,
        adjacent,
        patch_ids,
        np.asarray(feature_edge_rows, dtype=np.int64),
        transport,
        features,
        directions,
        residuals,
        result,
        solve,
        mesh.topology_id,
        geometry_id,
    )


def _store_quad_corner(
    vertex: int,
    image: tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...],
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_bank: _owner_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    original: int,
    exact_vertices: dict[
        int, tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
    ],
    coordinates: np.ndarray,
) -> None:
    """Require one shared root and preserve source topology carriers."""
    if vertex in exact_vertices and exact_vertices[vertex] != image:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"CellGeometrySpec.shared_vertices must agree across source charts; got conflicting roots at vertex {vertex}."
        )
    exact_vertices[vertex] = image
    if (
        source_geometry.exact_source is not None
        and vertex < original
        and image == source_bank[vertex]
    ):
        return
    value = _owner_phydrax_discretization__coordinate_enclosure.rounded_point(image)
    if vertex < original and not np.array_equal(value, coordinates[vertex]):
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"CellGeometrySpec.corner_coordinates must match the topology corner carrier; got a source-map mismatch at vertex {vertex}."
        )
    coordinates[vertex] = value


def _realize_quad_corners(
    block_name: str,
    composed: _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement,
    cells: np.ndarray,
    routes: np.ndarray | Array,
    points: np.ndarray,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    source_bank: _owner_phydrax_discretization__coordinate_enclosure.CoordinateSourceBank,
    original: int,
    exact_vertices: dict[
        int, tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...]
    ],
    coordinates: np.ndarray,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Realize one chart while enforcing shared roots and source carriers."""
    for cell, route in zip(cells.tolist(), routes, strict=True):
        family_execution_owner.count_geometry_queries(4, work_units=4, budget=budget)
        local = (
            points[route]
            if source_geometry.exact_source is None
            else tuple(source_bank[int(index)] for index in route)
        )
        images = (
            _owner_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                composed, local, budget=budget
            )
        )
        if images is None:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"extract_surface_quads.coordinate_expressions must provide source expressions; got None for block {block_name!r}."
            )
        for vertex, image in zip(cell, images, strict=True):
            _store_quad_corner(
                vertex,
                image,
                source_geometry,
                source_bank,
                original,
                exact_vertices,
                coordinates,
            )


# Chart assembly owns source ancestry; corner realization preserves query, shared-root, and carrier-check order.
def _composed_quad_mesh(
    source: CellMesh,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    quads: np.ndarray,
    coordinates: np.ndarray,
    added: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    CellMesh,
    _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    np.ndarray,
]:
    """Retain source coefficients through rational barycentric charts."""

    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        elements, source_routes, coefficients = source_geometry.resolve(
            source, budget=budget, exact_source_prepared=False
        )
        source_bank = (
            ()
            if source_geometry.exact_source is None
            else source_geometry.source_coordinates(budget=budget)
        )
        numerators = np.asarray(
            (
                ((0, 0), (1, 0), (1, 1), (0, 1)),
                ((1, 0), (1, 1), (1, 1), (1, 0)),
                ((0, 1), (0, 1), (1, 1), (1, 1)),
            ),
            dtype=np.int64,
        )
        denominators = np.asarray(
            (
                ((1, 1), (2, 1), (3, 3), (1, 2)),
                ((1, 1), (2, 2), (3, 3), (2, 1)),
                ((1, 1), (1, 2), (3, 3), (2, 2)),
            ),
            dtype=np.int64,
        )
        canonical_triangle = (
            _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                "triangle", 1
            ).element_id
        )
        match source_geometry.exact_source:
            case _owner_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource():
                direct_plc = source_geometry.restriction_source is None and all(
                    element.element_id == canonical_triangle for element in elements
                )
            case _:
                direct_plc = False
        chart = _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
            "quadrilateral", 1
        )
        original = source.coordinates.shape[0]
        points = np.asarray(coefficients, dtype=np.float64)
        exact_vertices: dict[
            int,
            tuple[_owner_phydrax_discretization__coordinate_enclosure.ExactScalar, ...],
        ] = {}
        target_elements, target_routes, parent_ids, parent_vertices = {}, {}, {}, {}
        target_blocks, parent_rows = [], []
        origin = source_geometry.restriction_source
        origin_ids = {} if origin is None else origin.block_parent_cell_ids
        origin_vertices = {} if origin is None else origin.block_parent_vertex_ids
        cursor = 0
        if direct_plc:
            name = "cells"
            target_blocks.append(CellBlock(name, "quadrilateral", quads))
            target_routes[name] = np.empty((quads.shape[0], 3), dtype=np.int32)
            parent_ids[name] = np.empty(quads.shape[0], dtype=np.int64)
            parent_vertices[name] = np.empty((quads.shape[0], 3), dtype=np.int64)
        for block, element, routes in zip(
            source.blocks, elements, source_routes, strict=True
        ):
            for corner in range(3):
                name = "cells" if direct_plc else f"{block.name}/quad/{corner}"
                rows = np.arange(cursor, cursor + block.cell_count, dtype=np.int64)
                target_rows = 3 * rows + corner
                span = slice(3 * cursor + corner, 3 * (cursor + block.cell_count), 3)
                cells = quads[span] if direct_plc else quads[target_rows]
                chart_index = 0 if direct_plc else corner
                composed = _owner_phydrax_discretization__cell_geometry.PolynomialComposedCellGeometryElement(
                    _owner_phydrax_discretization__cell_geometry.require_scalar_coordinate_element(
                        element, "Quad source"
                    ),
                    chart,
                    numerators[chart_index],
                    denominators[chart_index],
                    budget=budget,
                )
                if direct_plc:
                    active_routes = target_routes[name][span]
                    source_vertices = np.asarray(source.vertex_global_ids)[
                        np.asarray(block.vertices)
                    ]
                    for column in range(3):
                        active_routes[:, column] = np.asarray(routes)[
                            :, (corner + column) % 3
                        ]
                        parent_vertices[name][span, column] = source_vertices[
                            :, (corner + column) % 3
                        ]
                    parent_ids[name][span] = np.asarray(block.global_ids)
                    target_elements[name] = composed
                else:
                    active_routes = np.asarray(routes)
                    target_blocks.append(
                        CellBlock(name, "quadrilateral", cells, global_ids=target_rows)
                    )
                    target_elements[name], target_routes[name] = composed, routes
                    parent_rows.append(rows)
                    parent_ids[name] = (
                        block.global_ids if origin is None else origin_ids[block.name]
                    )
                    parent_vertices[name] = (
                        np.asarray(source.vertex_global_ids)[np.asarray(block.vertices)]
                        if origin is None
                        else origin_vertices[block.name]
                    )
                _realize_quad_corners(
                    block.name,
                    composed,
                    cells,
                    active_routes,
                    points,
                    source_geometry,
                    source_bank,
                    original,
                    exact_vertices,
                    coordinates,
                    budget=budget,
                )
            cursor += block.cell_count
        ancestry = (
            _owner_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
                _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                    source_geometry
                )
                if origin is None
                else origin.source_geometry_id,
                source.topology_id if origin is None else origin.source_topology_id,
                parent_ids,
                parent_vertices,
            )
        )
        geometry = _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec(
            target_elements,
            target_routes,
            coefficients,
            restriction_source=ancestry,
            exact_source=source_geometry.exact_source,
            budget=budget,
        )
        mesh = CellMesh(
            coordinates,
            tuple(target_blocks),
            vertex_global_ids=family_execution_owner.vertex_ids(source, added),
            numeric_version=source.numeric_version,
        )
        target_parents = (
            np.repeat(np.arange(quads.shape[0] // 3, dtype=np.int64), 3)
            if direct_plc
            else np.concatenate(parent_rows)
        )
    return mesh, geometry, target_parents


@checked
def extract_surface_quads(
    source: CellMesh,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    boundary_subdivision: bool = True,
    cross_field: dual_extraction_owner.SurfaceCrossField | None = None,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> dual_extraction_owner.DualExtraction:
    """Partition source triangles with quad topology and certified mapped pullbacks."""

    if source.topology.dimension != 2:
        raise TypeError(
            f"extract_surface_quads.source.dimension must be 2; got {source.topology.dimension}."
        )
    if source.periodic_topology is not None:
        raise _dependency_phydrax_meshing__host_banks.failure(
            "extract_surface_quads.source.periodic_topology must provide quotient subdivision ancestry; got a periodic source without that ancestry."
        )
    if any(block.cell_kind != "triangle" for block in source.blocks):
        raise ValueError(
            f"extract_surface_quads.source.blocks must contain triangle cells; got {tuple(block.cell_kind for block in source.blocks)}."
        )

    geometry_id = (
        source.geometry_id
        if source_geometry is None
        else _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
            source_geometry
        )
    )
    if cross_field is not None and (
        cross_field.source_topology_id != source.topology_id
        or cross_field.source_geometry_id != geometry_id
    ):
        raise ValueError(
            f"extract_surface_quads.cross_field must bind the source topology and geometry; got topology={cross_field.source_topology_id}, geometry={cross_field.source_geometry_id}."
        )
    if not boundary_subdivision:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"extract_surface_quads.boundary_subdivision must permit barycentric edge subdivision; got {boundary_subdivision!r} requiring a parity-compatible recombination route."
        )
    cells = _dependency_phydrax_meshing__host_banks.entities(source, 2)
    count, original = cells.shape[0], source.coordinates.shape[0]
    family_execution_owner.check_family_limits(
        limits, 3 * count, original, 12 * count, 2048 * count + 128 * original, 32 * count
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        certificate = (
            _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                source, budget=budget
            )
            if source_geometry is None
            else _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                source_geometry, mesh=source, budget=budget
            )
        )
    if certificate.invalid_count or certificate.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"extract_surface_quads.source.validity must resolve valid mapped triangles; got invalid={certificate.invalid_count}, unresolved={certificate.unresolved_count}."
        )
    edges = _dependency_phydrax_meshing__host_banks.entities(source, 1)
    added = edges.shape[0] + count
    family_execution_owner.check_family_limits(
        limits,
        3 * count,
        original + added,
        12 * count,
        2048 * count + 128 * original,
        32 * count,
    )
    edge_lookup = {
        tuple(sorted(edge)): original + row for row, edge in enumerate(edges.tolist())
    }
    nodes = _dependency_phydrax_meshing__host_banks.family_host_array(
        (count, 7), np.int64, scope=scope
    )
    nodes[:, :3] = cells
    for row, triangle in enumerate(cells.tolist()):
        nodes[row, 3:6] = [
            edge_lookup[tuple(sorted((triangle[a], triangle[b])))]
            for a, b in ((0, 1), (1, 2), (2, 0))
        ]
    nodes[:, 6] = original + edges.shape[0] + np.arange(count, dtype=np.int64)
    points = np.asarray(source.coordinates, dtype=np.float64)
    coordinates = np.concatenate(
        (points, np.mean(points[edges], axis=1), np.mean(points[cells], axis=1))
    )
    quads = _owner_phydrax__meshcore.triangle_dual_quads(
        nodes, coordinates.shape[0], maximum_cells=limits.maximum_cells
    )
    geometry = None
    parent_cells = np.repeat(np.arange(count, dtype=np.int64), 3)
    if source_geometry is None:
        mesh = CellMesh(
            coordinates,
            (CellBlock("cells", "quadrilateral", quads),),
            vertex_global_ids=family_execution_owner.vertex_ids(source, added),
            numeric_version=source.numeric_version,
        )
    else:
        mesh, geometry, parent_cells = _composed_quad_mesh(
            source, source_geometry, quads, coordinates, added, scope=scope, budget=budget
        )
    supports = _dependency_phydrax_meshing__host_banks.family_host_array(
        (coordinates.shape[0], 3), np.int64, scope=scope
    )
    supports.fill(-1)
    supports[:original, 0] = np.arange(original, dtype=np.int64)
    supports[original : original + edges.shape[0], :2] = edges
    supports[original + edges.shape[0] :] = cells
    dimensions, rows = _dependency_phydrax_meshing__host_banks.ancestry(
        source, mesh, supports
    )
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        validity = (
            _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                mesh, budget=budget
            )
            if geometry is None
            else _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
                geometry, mesh=mesh, budget=budget
            )
        )
    if validity.invalid_count or validity.unresolved_count:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"extract_surface_quads.target.validity must resolve valid mapped cells; got invalid={validity.invalid_count}, unresolved={validity.unresolved_count}."
        )
    connectivity = _dependency_phydrax_meshing__host_banks.entities(mesh, 2)
    valences = np.bincount(
        connectivity.reshape(-1), minlength=coordinates.shape[0]
    ).astype(np.int64)
    child_edges = _dependency_phydrax_meshing__host_banks.entities(mesh, 1)
    source_edge_counts = np.bincount(
        np.asarray(
            [
                edge_lookup[tuple(sorted((triangle[a], triangle[b])))] - original
                for triangle in cells.tolist()
                for a, b in ((0, 1), (1, 2), (2, 0))
            ],
            dtype=np.int64,
        ),
        minlength=edges.shape[0],
    )
    boundary_edge = (dimensions[1] == 1) & (
        source_edge_counts[np.minimum(rows[1], edges.shape[0] - 1)] == 1
    )
    boundary = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    boundary[child_edges[boundary_edge].reshape(-1)] = True
    singular = np.flatnonzero((~boundary) & (valences != 4)).astype(np.int64)
    if (
        mesh.topology.entities(1).count > limits.maximum_edges
        or mesh.topology.entities(2).count > limits.maximum_faces
    ):
        raise _dependency_phydrax_meshing__host_banks.failure(
            "DualExtraction.entity_counts must fit the edge and face limits; got excessive edge or face counts.",
            resource=True,
        )
    data = (
        coordinates.nbytes
        + quads.nbytes
        + supports.nbytes
        + sum(array.nbytes for array in dimensions + rows)
    )
    if geometry is not None:
        data += np.asarray(geometry.coordinates).nbytes + sum(
            np.asarray(route).nbytes for route in geometry.geometry_dofs
        )
    if data > limits.maximum_data_bytes:
        raise _dependency_phydrax_meshing__host_banks.failure(
            f"DualExtraction.data_bytes must fit limit {limits.maximum_data_bytes}; got {data}.",
            resource=True,
        )
    return dual_extraction_owner.DualExtraction(
        mesh,
        source,
        supports,
        parent_cells,
        dimensions,
        rows,
        validity,
        valences,
        boundary,
        singular,
        cross_field,
        32 * count,
        geometry,
        source_geometry,
    )


def _exact_parent_containment(
    source: CellMesh,
    target: CellMesh,
    parent_dimension: int,
    child_dimension: int,
    parent_rows: np.ndarray,
    child_rows: np.ndarray,
    /,
) -> np.ndarray:
    """Prove represented child nodes lie in their affine source simplex.

    Convex coordinate weights of every dual child keep its entire map in this
    parent hull. Rounded means need not stay on an oblique source face, so
    ancestry alone never establishes this geometric containment criterion.
    """

    parent_vertices = _dependency_phydrax_meshing__host_banks.entities(
        source, parent_dimension
    )[parent_rows]
    if parent_vertices.shape[1] != parent_dimension + 1:
        raise RuntimeError(
            f"DualExtraction.parent_vertices must describe an affine simplex; got width={parent_vertices.shape[1]}, dimension={parent_dimension}."
        )
    child_vertices = _dependency_phydrax_meshing__host_banks.entities(
        target, child_dimension
    )[child_rows]
    valid = child_vertices >= 0
    points = np.asarray(target.coordinates, dtype=np.float64)[
        np.maximum(child_vertices, 0)
    ]
    parents = np.asarray(source.coordinates, dtype=np.float64)[parent_vertices]
    if parent_dimension == 0:
        contained = np.all(points == parents[:, :1, :], axis=-1)
    elif parent_dimension == 1:
        left, right = parents[:, 0, :], parents[:, 1, :]
        contained = np.all(
            (points >= np.minimum(left, right)[:, None, :])
            & (points <= np.maximum(left, right)[:, None, :]),
            axis=-1,
        )
        for first in range(points.shape[-1]):
            for second in range(first + 1, points.shape[-1]):
                axes = (first, second)
                contained &= (
                    _owner_phydrax__meshcore.exact_orient2d(
                        left[:, None, :][..., axes],
                        right[:, None, :][..., axes],
                        points[..., axes],
                    )
                    == 0
                )
    elif parent_dimension == 2:
        contained = np.zeros(valid.shape, dtype=np.bool_)
        for first in range(points.shape[-1]):
            for second in range(first + 1, points.shape[-1]):
                axes = (first, second)
                triangle = parents[..., axes]
                query = points[..., axes]
                orientation = _owner_phydrax__meshcore.exact_orient2d(
                    triangle[:, 0], triangle[:, 1], triangle[:, 2]
                )[:, None]
                inside = orientation != 0
                inside = inside & (
                    orientation
                    * _owner_phydrax__meshcore.exact_orient2d(
                        triangle[:, 0, None], triangle[:, 1, None], query
                    )
                    >= 0
                )
                inside &= (
                    orientation
                    * _owner_phydrax__meshcore.exact_orient2d(
                        triangle[:, 1, None], triangle[:, 2, None], query
                    )
                    >= 0
                )
                inside &= (
                    orientation
                    * _owner_phydrax__meshcore.exact_orient2d(
                        triangle[:, 2, None], triangle[:, 0, None], query
                    )
                    >= 0
                )
                contained |= inside
        if points.shape[-1] == 3:
            contained &= (
                _owner_phydrax__meshcore.exact_orient3d(
                    parents[:, 0, None], parents[:, 1, None], parents[:, 2, None], points
                )
                == 0
            )
    elif parent_dimension == 3:
        orientation = _owner_phydrax__meshcore.exact_orient3d(
            parents[:, 0], parents[:, 1], parents[:, 2], parents[:, 3]
        )[:, None]
        contained = np.broadcast_to(orientation != 0, valid.shape).copy()
        a, b, c, d = (
            parents[:, 0, None, :],
            parents[:, 1, None, :],
            parents[:, 2, None, :],
            parents[:, 3, None, :],
        )
        contained &= (
            orientation * _owner_phydrax__meshcore.exact_orient3d(points, b, c, d) >= 0
        )
        contained &= (
            orientation * _owner_phydrax__meshcore.exact_orient3d(a, points, c, d) >= 0
        )
        contained &= (
            orientation * _owner_phydrax__meshcore.exact_orient3d(a, b, points, d) >= 0
        )
        contained &= (
            orientation * _owner_phydrax__meshcore.exact_orient3d(a, b, c, points) >= 0
        )
    else:
        raise RuntimeError(
            f"DualExtraction.parent_dimension must be between 0 and 3; got {parent_dimension}."
        )
    return np.all(contained | ~valid, axis=1)


@checked
def remap_dual_metadata(
    extraction: dual_extraction_owner.DualExtraction,
    /,
    *,
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...] = (),
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...] = (),
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...] = (),
    associations: tuple[association_state_owner.GeometryAssociation, ...] = (),
    surface_source_cover: bool = False,
) -> tuple[
    tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...],
    tuple[association_state_owner.GeometryAssociation, ...],
]:
    """Propagate scientific identity by parent topology, never proximity."""
    source, target = extraction.source, extraction.mesh

    def scope(
        original: _dependency_phydrax_meshing__scope.MeshingScope, /
    ) -> _dependency_phydrax_meshing__scope.MeshingScope:
        dimension = original.entity_dimension
        entities = source.topology.entities(dimension)
        if (
            original.entity_set_id != entities.entity_set_id
            or original.source_id != source.mesh_id
            or original.source_revision != source.numeric_version
        ):
            raise ValueError(
                f"remap_dual_metadata.scope must bind the source mesh and entity set; got source={original.source_id}, revision={original.source_revision}, entity_set={original.entity_set_id}."
            )
        parent_ids = np.asarray(entities.entity_ids, dtype=np.int64)
        dimensions = extraction.entity_parent_dimensions[dimension]
        rows = extraction.entity_parent_rows[dimension]
        selected = (dimensions == dimension) & np.isin(
            parent_ids[np.minimum(rows, parent_ids.size - 1)],
            np.asarray(original.entity_ids),
        )
        child = target.topology.entities(dimension)
        return _dependency_phydrax_meshing__scope.MeshingScope(
            target.mesh_id,
            target.numeric_version,
            _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
            dimension,
            child.entity_set_id,
            np.asarray(child.entity_ids)[selected],
        )

    new_zones = tuple(
        _dependency_phydrax_meshing__organization.MeshZone(
            zone.name,
            zone.role,
            scope(zone.scope),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        for zone in zones
    )
    zone_ids = {
        old.zone_id: new.zone_id for old, new in zip(zones, new_zones, strict=True)
    }
    new_patches = tuple(
        _dependency_phydrax_meshing__organization.MeshPatch(
            patch.name,
            scope(patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
        )
        for patch in patches
    )
    new_labels = tuple(
        _dependency_phydrax_meshing__organization.MeshLabel(
            label.name, scope(label.scope)
        )
        for label in labels
    )
    new_associations = []
    coordinates = np.asarray(source.coordinates, dtype=np.float64)
    rounding = 64.0 * np.finfo(np.float64).eps * np.max(np.abs(coordinates), initial=1.0)
    for association in associations:
        if (
            association.association_kind
            is not association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
            and not (
                surface_source_cover
                and association.association_kind
                in (
                    association_state_owner.GeometryAssociationKind.SURFACE,
                    association_state_owner.GeometryAssociationKind.BREP,
                )
            )
        ):
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"remap_dual_metadata.associations must supply the source surface cover and continuous fidelity certification; got kind={association.association_kind!r}, surface_source_cover={surface_source_cover!r}."
            )
        source_dimensions = [
            degree
            for degree in range(source.topology.dimension + 1)
            if source.topology.entities(degree).entity_set_id
            == association.target_entity_set_id
        ]
        if len(source_dimensions) != 1:
            raise ValueError(
                f"GeometryAssociation.target_entity_set_id must identify one source entity set; got {association.target_entity_set_id!r}, matches={len(source_dimensions)}."
            )
        dimension = source_dimensions[0]
        parent_ids = np.asarray(
            source.topology.entities(dimension).entity_ids, dtype=np.int64
        )
        association_rows = {
            identifier: row
            for row, identifier in enumerate(
                np.asarray(association.target_global_ids).tolist()
            )
        }
        if np.any(~np.isin(np.asarray(association.target_global_ids), parent_ids)):
            raise ValueError(
                f"GeometryAssociation.target_global_ids must belong to the source entity set; got {np.asarray(association.target_global_ids).tolist()}."
            )
        for degree in range(dimension + 1):
            dimensions = extraction.entity_parent_dimensions[degree]
            parent_rows = extraction.entity_parent_rows[degree]
            selected, inherited = [], []
            for row in np.flatnonzero(dimensions == dimension).tolist():
                identifier = int(parent_ids[parent_rows[row]])
                if identifier in association_rows:
                    selected.append(row)
                    inherited.append(association_rows[identifier])
            if not selected:
                continue
            selected_ = np.asarray(selected, dtype=np.int64)
            inherited_ = np.asarray(inherited, dtype=np.int64)
            orientations = np.zeros((len(selected),), dtype=np.int8)
            if degree == dimension and degree in (1, 2):
                original = coordinates[
                    _dependency_phydrax_meshing__host_banks.entities(source, degree)[
                        parent_rows[selected_]
                    ]
                ]
                child = np.asarray(target.coordinates, dtype=np.float64)[
                    _dependency_phydrax_meshing__host_banks.entities(target, degree)[
                        selected_
                    ]
                ]
                if degree == 1:
                    left, right = (
                        original[:, 1] - original[:, 0],
                        child[:, 1] - child[:, 0],
                    )
                elif coordinates.shape[1] == 3:
                    left = np.cross(
                        original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]
                    )
                    right = np.cross(child[:, 1] - child[:, 0], child[:, 2] - child[:, 0])
                else:
                    left = np.ones((len(selected), 1), dtype=np.float64)
                    right = np.ones((len(selected), 1), dtype=np.float64)
                orientations = np.asarray(association.orientations, dtype=np.int8)[
                    inherited_
                ] * np.sign(np.sum(left * right, axis=1)).astype(np.int8)
            exact = (
                association.exact
                and association.association_kind
                is association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
                and bool(
                    np.all(
                        _exact_parent_containment(
                            source,
                            target,
                            dimension,
                            degree,
                            parent_rows[selected_],
                            selected_,
                        )
                    )
                )
            )
            residuals = np.asarray(association.residuals, dtype=np.float64)[
                inherited_
            ] + (0.0 if exact or dimension == source.topology.dimension else rounding)
            child_entities = target.topology.entities(degree)
            new_associations.append(
                association_state_owner.GeometryAssociation(
                    association.association_kind,
                    association.source_id,
                    association.source_revision,
                    child_entities.entity_set_id,
                    np.asarray(child_entities.entity_ids)[selected_],
                    tuple(
                        association.source_entities.entity_id(row) for row in inherited
                    ),
                    residuals,
                    resolved=np.asarray(association.resolved)[inherited_],
                    ambiguous=np.asarray(association.ambiguous)[inherited_],
                    exact=exact,
                    orientations=orientations,
                    source_dimensions=np.asarray(association.source_dimensions)[
                        inherited_
                    ]
                    if association.association_kind
                    in (
                        association_state_owner.GeometryAssociationKind.BREP,
                        association_state_owner.GeometryAssociationKind.SURFACE,
                    )
                    or association.source_entities.indexed_roles
                    else None,
                    source_indices=np.asarray(association.source_indices)[inherited_]
                    if association.association_kind
                    in (
                        association_state_owner.GeometryAssociationKind.BREP,
                        association_state_owner.GeometryAssociationKind.SURFACE,
                    )
                    or association.source_entities.indexed_roles
                    else None,
                    source_entity_roles=None
                    if not association.source_entities.indexed_roles
                    else tuple(
                        association.source_entities.role(row) for row in inherited
                    ),
                    source_occurrence_paths=tuple(
                        association.source_entities.occurrence_path(row)
                        for row in inherited
                    ),
                    parameters=np.asarray(association.parameters)[inherited_],
                    parent_dimensions=np.full(
                        (len(selected),), dimension, dtype=np.int64
                    ),
                    parent_ids=parent_ids[parent_rows[selected_]],
                    parent_association_id=association.association_id,
                    provenance=association_state_owner.GeometryAssociationProvenance.LINEAGE,
                )
            )
    _complete_vertex_associations(extraction, associations, new_associations, rounding)
    return (
        new_zones,
        new_patches,
        new_labels,
        _coalesce_generated_associations(tuple(new_associations)),
    )


@final
class _VertexIncidence(StrictModule):
    """Index source association incidence for uncovered target vertices."""

    target_ids: np.ndarray
    covered: set[int] = eqx.field(static=True)
    associations: dict[tuple[int, int], list[tuple[int, int]]] = eqx.field(static=True)
    incident: list[set[tuple[int, int]]] = eqx.field(static=True)


@final
class _VertexOwner(StrictModule):
    """Identify the lowest-dimensional unambiguous source association."""

    association_index: int = eqx.field(static=True)
    association_row: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    source_row: int = eqx.field(static=True)


@final
class _InheritedVertexRows(StrictModule):
    """Group target vertices by source association and incidence dimension."""

    entries: dict[tuple[int, int], list[tuple[int, int, int]]] = eqx.field(static=True)


@final
class _PublishedVertexAssociation(StrictModule):
    association: association_state_owner.GeometryAssociation


def _prepare_vertex_incidence(
    extraction: dual_extraction_owner.DualExtraction,
    originals: tuple[association_state_owner.GeometryAssociation, ...],
    outputs: list[association_state_owner.GeometryAssociation],
    /,
) -> _VertexIncidence | None:
    """Prepare incidence only when source cell associations cover all cells."""
    source, target = extraction.source, extraction.mesh
    source_cells = source.topology.entities(source.topology.dimension)
    cell_coverage = {
        identifier
        for association in originals
        if association.target_entity_set_id == source_cells.entity_set_id
        for identifier in np.asarray(association.target_global_ids).tolist()
    }
    if not set(np.asarray(source_cells.entity_ids).tolist()) <= cell_coverage:
        return None
    target_vertices = target.topology.entities(0)
    covered = {
        identifier
        for association in outputs
        if association.target_entity_set_id == target_vertices.entity_set_id
        for identifier in np.asarray(association.target_global_ids).tolist()
    }
    target_ids = np.asarray(target_vertices.entity_ids, dtype=np.int64)
    entity_associations: dict[tuple[int, int], list[tuple[int, int]]] = {}
    incident: list[set[tuple[int, int]]] = [
        set() for _ in range(source.coordinates.shape[0])
    ]
    for dimension in range(source.topology.dimension + 1):
        entities = source.topology.entities(dimension)
        ids = np.asarray(entities.entity_ids, dtype=np.int64)
        rows = {identifier: row for row, identifier in enumerate(ids.tolist())}
        vertices = _dependency_phydrax_meshing__host_banks.entities(source, dimension)
        for association_index, association in enumerate(originals):
            if association.target_entity_set_id != entities.entity_set_id:
                continue
            for association_row, identifier in enumerate(
                np.asarray(association.target_global_ids).tolist()
            ):
                row = rows[identifier]
                key = (dimension, row)
                entity_associations.setdefault(key, []).append(
                    (association_index, association_row)
                )
                for vertex in vertices[row].tolist():
                    if vertex >= 0:
                        incident[vertex].add(key)
    return _VertexIncidence(target_ids, covered, entity_associations, incident)


def _choose_vertex_owner(
    source: CellMesh,
    originals: tuple[association_state_owner.GeometryAssociation, ...],
    incidence: _VertexIncidence,
    support: np.ndarray,
    /,
) -> _VertexOwner:
    """Select one source identity by common support incidence, not proximity."""
    original_vertices = support[support >= 0].tolist()
    candidates = set(incidence.incident[original_vertices[0]])
    for vertex in original_vertices[1:]:
        candidates.intersection_update(incidence.incident[vertex])
    for dimension in range(source.topology.dimension + 1):
        values = [
            (association_index, association_row, row)
            for degree, row in sorted(candidates)
            if degree == dimension
            for association_index, association_row in incidence.associations[
                (degree, row)
            ]
        ]
        identities = {
            (
                originals[index].source_id,
                originals[index].source_revision,
                originals[index].source_entities.entity_id(association_row),
            )
            for index, association_row, _ in values
        }
        if len(identities) == 1:
            index, association_row, row = values[0]
            return _VertexOwner(index, association_row, dimension, row)
    raise _dependency_phydrax_meshing__host_banks.failure(
        "DualExtraction.vertex_supports must identify one feature/material source owner; "
        f"got ambiguous support {original_vertices}."
    )


def _inherit_vertex_rows(
    extraction: dual_extraction_owner.DualExtraction,
    originals: tuple[association_state_owner.GeometryAssociation, ...],
    incidence: _VertexIncidence,
    /,
) -> _InheritedVertexRows:
    """Group uncovered target rows in their existing deterministic traversal order."""
    inherited: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for target_row in range(extraction.mesh.coordinates.shape[0]):
        if incidence.target_ids[target_row] in incidence.covered:
            continue
        owner = _choose_vertex_owner(
            extraction.source,
            originals,
            incidence,
            extraction.vertex_supports[target_row],
        )
        inherited.setdefault((owner.association_index, owner.dimension), []).append(
            (target_row, owner.association_row, owner.source_row)
        )
    return _InheritedVertexRows(inherited)


def _publish_vertex_association(
    extraction: dual_extraction_owner.DualExtraction,
    association: association_state_owner.GeometryAssociation,
    dimension: int,
    entries: list[tuple[int, int, int]],
    target_ids: np.ndarray,
    rounding: float,
    /,
) -> _PublishedVertexAssociation:
    """Build a lineage association with parent containment and inherited evidence."""
    source, target = extraction.source, extraction.mesh
    target_vertices = target.topology.entities(0)
    child_rows = np.asarray([entry[0] for entry in entries], dtype=np.int64)
    association_rows = np.asarray([entry[1] for entry in entries], dtype=np.int64)
    source_rows = np.asarray([entry[2] for entry in entries], dtype=np.int64)
    parent_ids = np.asarray(
        source.topology.entities(dimension).entity_ids, dtype=np.int64
    )[source_rows]
    exact = (
        association.exact
        and association.association_kind
        is association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR
        and bool(
            np.all(
                _exact_parent_containment(
                    source, target, dimension, 0, source_rows, child_rows
                )
            )
        )
    )
    return _PublishedVertexAssociation(
        association_state_owner.GeometryAssociation(
            association.association_kind,
            association.source_id,
            association.source_revision,
            target_vertices.entity_set_id,
            target_ids[child_rows],
            tuple(
                association.source_entities.entity_id(row)
                for row in association_rows.tolist()
            ),
            np.asarray(association.residuals)[association_rows]
            + (0.0 if exact else rounding),
            resolved=np.asarray(association.resolved)[association_rows],
            ambiguous=np.asarray(association.ambiguous)[association_rows],
            exact=exact,
            parent_dimensions=np.full((child_rows.size,), dimension, dtype=np.int64),
            source_dimensions=np.asarray(association.source_dimensions)[association_rows]
            if association.association_kind
            in (
                association_state_owner.GeometryAssociationKind.BREP,
                association_state_owner.GeometryAssociationKind.SURFACE,
            )
            or association.source_entities.indexed_roles
            else None,
            source_indices=np.asarray(association.source_indices)[association_rows]
            if association.association_kind
            in (
                association_state_owner.GeometryAssociationKind.BREP,
                association_state_owner.GeometryAssociationKind.SURFACE,
            )
            or association.source_entities.indexed_roles
            else None,
            source_entity_roles=None
            if not association.source_entities.indexed_roles
            else tuple(
                association.source_entities.role(row) for row in association_rows.tolist()
            ),
            source_occurrence_paths=tuple(
                association.source_entities.occurrence_path(row)
                for row in association_rows.tolist()
            ),
            parameters=np.asarray(association.parameters)[association_rows],
            parent_ids=parent_ids,
            parent_association_id=association.association_id,
            provenance=association_state_owner.GeometryAssociationProvenance.LINEAGE,
        )
    )


def _complete_vertex_associations(
    extraction: dual_extraction_owner.DualExtraction,
    originals: tuple[association_state_owner.GeometryAssociation, ...],
    outputs: list[association_state_owner.GeometryAssociation],
    rounding: float,
    /,
) -> None:
    """Full uncovered coordinate vertices from unambiguous source incidence.

    Partial source cell selections stay partial; all-cell coverage activates
    completion. Material and feature junctions require a declared source identity.
    """
    incidence = _prepare_vertex_incidence(extraction, originals, outputs)
    if incidence is None:
        return
    inherited = _inherit_vertex_rows(extraction, originals, incidence)
    for (index, dimension), entries in sorted(inherited.entries.items()):
        published = _publish_vertex_association(
            extraction,
            originals[index],
            dimension,
            entries,
            incidence.target_ids,
            rounding,
        )
        outputs.append(published.association)


def _gathered_rows(
    blocks: list[Array], order: np.ndarray, dtype: type[np.generic] | None = None, /
) -> np.ndarray:
    """Concatenate per-association columns on the host once, in published row order."""
    values = np.concatenate([np.asarray(block) for block in blocks])[order]
    return values if dtype is None else values.astype(dtype, copy=False)


def _coalesce_generated_associations(
    associations: tuple[association_state_owner.GeometryAssociation, ...],
    /,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    """Publish one coalesced provider table per bound target/source binding.

    Different input association blocks remain measured per-row mesh genealogy.
    The generator owns the assembled provider table: it does not invent an
    aggregate parent association ID or attach rows to an unrelated parent.
    """
    groups: dict[
        tuple[
            association_state_owner.GeometryAssociationKind,
            str,
            str,
            str,
        ],
        list[association_state_owner.GeometryAssociation],
    ] = {}
    for association in associations:
        key = (
            association.association_kind,
            association.source_id,
            association.source_revision,
            association.target_entity_set_id,
        )
        groups.setdefault(key, []).append(association)
    output = []
    for key, members in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        # A stable sort by target ID keeps input block order among equal keys
        # before the uniqueness refusal.
        identifiers = np.concatenate(
            [
                np.asarray(association.target_global_ids, dtype=np.int64)
                for association in members
            ]
        )
        order = np.argsort(identifiers, kind="stable")
        identifiers = identifiers[order]
        if np.unique(identifiers).size != identifiers.size:
            raise _dependency_phydrax_meshing__host_banks.failure(
                "GeometryAssociation.target_global_ids must assign each target entity once; got repeated IDs across generated blocks."
            )
        kind, source_id, revision, target_set = key
        role_presence = {
            association.source_entities.indexed_roles for association in members
        }
        if len(role_presence) != 1:
            raise _dependency_phydrax_meshing__host_banks.failure(
                f"GeometryAssociation.source_entity_roles must use one indexing convention per source binding; got role presence {sorted(role_presence)}."
            )
        has_roles = next(iter(role_presence))
        indexed = (
            kind
            in (
                association_state_owner.GeometryAssociationKind.BREP,
                association_state_owner.GeometryAssociationKind.SURFACE,
            )
            or has_roles
        )
        rows = order.tolist()
        source_entity_ids = [
            value
            for association in members
            for value in association.source_entities.entity_ids()
        ]
        occurrence_paths = [
            value
            for association in members
            for value in association.source_entities.occurrence_paths()
        ]
        roles = [
            association.source_entities.role(row)
            for association in members
            if association.source_entities.indexed_roles
            for row in range(association.target_global_ids.size)
        ]
        output.append(
            association_state_owner.GeometryAssociation(
                kind,
                source_id,
                revision,
                target_set,
                identifiers,
                tuple(source_entity_ids[row] for row in rows),
                _gathered_rows(
                    [association.residuals for association in members], order, np.float64
                ),
                resolved=_gathered_rows(
                    [association.resolved for association in members], order, np.bool_
                ),
                ambiguous=_gathered_rows(
                    [association.ambiguous for association in members], order, np.bool_
                ),
                exact=all(association.exact for association in members),
                source_dimensions=_gathered_rows(
                    [association.source_dimensions for association in members],
                    order,
                    np.int8,
                )
                if indexed
                else None,
                source_indices=_gathered_rows(
                    [association.source_indices for association in members],
                    order,
                    np.int64,
                )
                if indexed
                else None,
                source_entity_roles=None
                if not has_roles
                else tuple(roles[row] for row in rows),
                source_occurrence_paths=tuple(occurrence_paths[row] for row in rows),
                parameters=_gathered_rows(
                    [association.parameters for association in members], order
                ),
                orientations=_gathered_rows(
                    [association.orientations for association in members], order, np.int8
                ),
                parent_dimensions=_gathered_rows(
                    [association.parent_dimensions for association in members],
                    order,
                    np.int8,
                ),
                parent_ids=_gathered_rows(
                    [association.parent_ids for association in members], order, np.int64
                ),
                provenance=association_state_owner.GeometryAssociationProvenance.PROVIDER,
            )
        )
    return tuple(output)
