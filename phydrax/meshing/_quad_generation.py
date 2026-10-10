#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, source-faithful quad extraction on authored triangle surface charts.

The barycentric-dual route partitions every triangle into three genuine quads.
Shared edge identities, not coordinate coincidence, close the patches. Affine
sources use bilinear vertex maps; polynomial and rational spline source maps
retain their original coefficients through exact rational bilinear reference
composition. Spline triangle charts must remain within their original knot span.
Feature-separated cross fields use original chart jets when maps are supplied.
This route deliberately subdivides boundary chains: immutable-chain
recombination, arbitrary CAD projection and field-guided integer-grid extraction
remain distinct research operations, not properties inferred from these templates.

Exact association inheritance uses native predicate proofs of containment in the
actual affine source simplex. Rounded nodes outside that hull retain bounded
residuals; independent PLC coverage does not silently turn them into exact rows.
"""

from __future__ import annotations

from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from .._meshcore import (
    charge_native_geometry_queries,
    current_native_execution_budget,
    NativeExecutionBudget,
)
from .._strict import StrictModule
from ..discretization import (
    CellBlock,
    CellMesh,
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_complex import _dense_rows, PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..linalg._small_batched import (
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
)
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
from ..optim._iterative._types import MinimizationResult
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
)
from ._contracts import MeshingFailure, MeshingFailureCategory, MeshingLimits
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from ._organization import MeshLabel, MeshPatch, MeshZone
from ._scope import MeshingEntityKind, MeshingScope


@final
class SurfaceCrossField(StrictModule):
    """C4 angles in feature-separated face patches with native solve evidence."""

    frames: np.ndarray
    angles: np.ndarray
    transport_edges: np.ndarray
    patch_ids: np.ndarray
    feature_edges: np.ndarray
    transport_angles: np.ndarray
    feature_rows: np.ndarray
    feature_angles: np.ndarray
    feature_residuals: np.ndarray
    optimization: MinimizationResult
    frame_solve: SmallLinearSolveResult
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)


@final
class DualExtraction(StrictModule):
    """Canonical candidate plus exact combinatorial subdivision ancestry.

    ``vertex_supports`` is a padded row of original vertex indices for each new
    vertex (``-1`` padding). ``entity_parent_dimensions`` and
    ``entity_parent_rows`` identify the smallest source entity containing each
    target entity. These maps are suitable for source association and material
    propagation; they do not certify global embedding of the input complex.
    """

    mesh: CellMesh
    source: CellMesh
    vertex_supports: np.ndarray
    parent_cells: np.ndarray
    entity_parent_dimensions: tuple[np.ndarray, ...]
    entity_parent_rows: tuple[np.ndarray, ...]
    validity: CellValidityCertificate
    valences: np.ndarray
    boundary_vertices: np.ndarray
    singular_vertices: np.ndarray
    cross_field: SurfaceCrossField | None
    work_units: int = eqx.field(static=True)
    geometry: CellGeometrySpec | None = None
    source_geometry: CellGeometrySpec | None = None


def _failure(message: str, /, *, resource: bool = False) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED
        if resource
        else MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        message,
        stage="family-extraction",
    )


def _budget(
    limits: MeshingLimits,
    cells: int,
    vertices: int,
    entries: int,
    scratch: int,
    work: int,
    /,
) -> None:
    for name, value, bound in (
        ("cells", cells, limits.maximum_cells),
        ("vertices", vertices, limits.maximum_vertices),
        ("connectivity entries", entries, limits.maximum_connectivity_entries),
        ("scratch bytes", scratch, limits.maximum_scratch_bytes),
        ("work units", work, limits.maximum_work_units),
    ):
        if value > bound:
            raise _failure(
                f"Dual extraction requires {value} {name}; limit is {bound}.",
                resource=True,
            )


def _family_host_array(shape: tuple[int, ...], dtype: DTypeLike, /) -> np.ndarray:
    """Allocate explicit family host banks in the current original native pool."""
    budget = current_native_execution_budget()
    return (
        np.empty(shape, dtype=dtype)
        if budget is None
        else budget.allocate_host_array(shape, dtype)
    )


def _entities(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    if dimension == 0:
        return np.arange(mesh.coordinates.shape[0], dtype=np.int64)[:, None]
    if dimension == mesh.topology.dimension:
        width = max(block.vertices.shape[1] for block in mesh.blocks)
        return np.concatenate(
            tuple(
                np.pad(
                    np.asarray(block.vertices, dtype=np.int64),
                    ((0, 0), (0, width - block.vertices.shape[1])),
                    constant_values=-1,
                )
                for block in mesh.blocks
            )
        )
    if dimension == 1:
        if not isinstance(
            mesh.connectivity,
            (
                PolygonalConnectivity,
                TetrahedralConnectivity,
                HexahedralConnectivity,
                PolyhedralConnectivity,
            ),
        ):
            raise ValueError(
                "Edge extraction requires a canonical surface/volume connectivity."
            )
        return np.asarray(mesh.connectivity.edges, dtype=np.int64)
    if dimension == 2 and mesh.topology.dimension == 3:
        if isinstance(mesh.connectivity, PolyhedralConnectivity):
            offsets = np.asarray(mesh.connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(mesh.connectivity.face_vertex_values, dtype=np.int64)
            lengths = np.diff(offsets)
            width = int(np.max(lengths))
            dense = _dense_rows(offsets, values, width)
            return np.where(np.arange(width)[None, :] < lengths[:, None], dense, -1)
        if not isinstance(
            mesh.connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
        ):
            raise ValueError("Face extraction requires a canonical volume connectivity.")
        return np.asarray(mesh.connectivity.faces, dtype=np.int64)
    raise ValueError("No entity vertex route for this dimension.")


def _ancestry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    lookup: dict[tuple[int, ...], tuple[int, int]] = {}
    for dimension in range(source.topology.dimension + 1):
        for row, vertices in enumerate(_entities(source, dimension).tolist()):
            key = tuple(sorted(value for value in vertices if value >= 0))
            lookup[key] = (dimension, row)
    dimensions, rows = [], []
    for dimension in range(target.topology.dimension + 1):
        vertices = _entities(target, dimension)
        parents = []
        for entity in vertices.tolist():
            key = tuple(
                sorted(
                    {
                        value
                        for vertex in entity
                        if vertex >= 0
                        for value in supports[vertex].tolist()
                        if value >= 0
                    }
                )
            )
            if key not in lookup:
                raise RuntimeError("Dual template has an entity with no source ancestor.")
            parents.append(lookup[key])
        parent = np.asarray(parents, dtype=np.int64).reshape((-1, 2))
        dimensions.append(parent[:, 0])
        rows.append(parent[:, 1])
    return tuple(dimensions), tuple(rows)


def _vertex_ids(source: CellMesh, added: int, /) -> np.ndarray:
    original = np.asarray(source.vertex_global_ids, dtype=np.int64)
    start = int(np.max(original)) + 1
    if start + added > np.iinfo(np.int64).max:
        raise _failure(
            "Dual extraction exhausts the scientific vertex ID range.", resource=True
        )
    return np.concatenate((original, np.arange(start, start + added, dtype=np.int64)))


def _cross_energy(
    angles: Array, arguments: tuple[Array, Array, Array, Array], /
) -> Array:
    adjacent, transport, feature_rows, feature_angles = arguments
    mismatch = angles[adjacent[:, 1]] - angles[adjacent[:, 0]] - transport
    feature = angles[feature_rows] - feature_angles
    return jnp.sum(1.0 - jnp.cos(4.0 * mismatch)) + 4.0 * jnp.sum(
        1.0 - jnp.cos(4.0 * feature)
    )


def _field_work_allowance(
    method: ProjectedLBFGS,
    rows: int,
    maximum_steps: int,
    /,
) -> NativeExecutionBudget | None:
    """Admit the declared pure-batch bound without publishing it as actual work."""
    budget = current_native_execution_budget()
    if budget is not None:
        # Each iteration has one value/gradient pair and at most the authored
        # Armijo trials. The optimizer owns one final value/gradient pair.
        bound = rows * (maximum_steps * (method.line_search.maximum_steps + 2) + 2)
        budget.admit_work_bound(bound)
    return budget


def _charge_field_work(
    budget: NativeExecutionBudget | None,
    result: MinimizationResult,
    rows: int,
    /,
) -> None:
    """Charge complete owning optimizer diagnostics once, before candidate use."""
    if budget is None:
        return
    if not result.diagnostics.counts_complete:
        raise RuntimeError(
            "A native family field requires complete optimizer work counts."
        )
    objectives, gradients = jax.device_get(
        (
            result.diagnostics.objective_evaluations,
            result.diagnostics.gradient_evaluations,
        )
    )
    budget.charge(work=rows * (int(objectives) + int(gradients)))


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
    geometry: CellGeometrySpec,
    /,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, tuple[int, int]], np.ndarray]]:
    """Use source chart jets, not carrier chords, for curved face/feature frames."""
    from ..discretization._cell_geometry import _require_scalar_coordinate_element

    elements, routes, coefficients = geometry.resolve(mesh)
    reference = np.asarray(
        ((1.0 / 3.0, 1.0 / 3.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
        dtype=np.float64,
    )
    edge_vectors = np.asarray(((1.0, 0.0), (-1.0, 1.0), (0.0, -1.0)), dtype=np.float64)
    values = np.asarray(coefficients, dtype=np.float64)
    first, second = [], []
    directions: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        scalar = _require_scalar_coordinate_element(element, "Surface frames")
        point_queries = reference.shape[0] * block.cell_count
        charge_native_geometry_queries(point_queries, work_units=point_queries)
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


def prepare_surface_cross_field(
    mesh: CellMesh,
    /,
    *,
    feature_edges: np.ndarray | None = None,
    maximum_steps: int = 100,
    source_geometry: CellGeometrySpec | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SurfaceCrossField:
    """Minimize a transported fourth-harmonic field using native optim/linalg.

    Boundary edges are always feature constraints. Additional feature edges are
    source connectivity edge *rows*, not inferred geometric identities. They cut
    the transport graph into patches, each with its own frame field; a protected
    crease must not smooth one patch's direction into another. Multiple
    incompatible directions are retained as achieved residuals, never certified
    as exact alignment. The patches are not integer-grid parameterizations.
    """
    measurement_started = phase_started(record_phase)
    if not isinstance(mesh, CellMesh) or mesh.topology.dimension != 2:
        raise TypeError("A surface cross field requires a two-dimensional CellMesh.")
    if mesh.ambient_dimension not in (2, 3):
        raise ValueError(
            "A surface cross field requires two or three ambient dimensions."
        )
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("A surface cross field requires triangle blocks.")
    triangles = _entities(mesh, 2)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    local = points[triangles]
    mapped_directions = None
    if source_geometry is None:
        first, second = local[:, 1] - local[:, 0], local[:, 2] - local[:, 0]
        geometry_id = mesh.geometry_id
    else:
        from ..discretization._cell_geometry_validity import cell_geometry_id

        first, second, mapped_directions = _mapped_surface_directions(
            mesh, source_geometry
        )
        geometry_id = cell_geometry_id(source_geometry)
    normal = np.cross(first, second)
    lengths, areas = np.linalg.norm(first, axis=1), np.linalg.norm(normal, axis=1)
    if np.any(lengths == 0.0) or np.any(areas == 0.0):
        raise _failure("Degenerate face has no regular tangent frame.")
    tangent = first / lengths[:, None]
    normal /= areas[:, None]
    frames = np.stack((tangent, np.cross(normal, tangent)), axis=1)
    edges = _entities(mesh, 1)
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
                "feature_edges must be an integer vector of source edge rows."
            )
        explicit = set(features_.tolist())
    if any(row < 0 or row >= edges.shape[0] for row in explicit):
        raise ValueError("feature_edges contains an unknown source edge row.")
    incidences = [owners[tuple(sorted(edge))] for edge in edges.tolist()]
    if any(len(faces) != 1 and len(faces) != 2 for faces in incidences):
        raise _failure("A surface cross field requires a manifold face link.")
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
    solve = solve_small_linear(SmallLinearSolvePlan(2), gram, right)
    coordinates, successful = jax.device_get((solve.value, solve.successful))
    if not np.all(np.asarray(successful, dtype=np.bool_)):
        raise _failure("Native tangent-frame solve is singular or ill-conditioned.")
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
    budget = _field_work_allowance(method, work_rows, maximum_steps)
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
    _charge_field_work(budget, result, work_rows)
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
    return SurfaceCrossField(
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


def _composed_quad_mesh(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    quads: np.ndarray,
    coordinates: np.ndarray,
    added: int,
    /,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray]:
    """Retain original source coefficients through exact barycentric charts."""
    from ..discretization._cell_geometry import (
        _require_scalar_coordinate_element,
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
        PolynomialComposedCellGeometryElement,
    )
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        rounded_point,
    )

    elements, source_routes, coefficients = source_geometry.resolve(source)
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
    chart = coordinate_lagrange_element("quadrilateral", 1)
    original = source.coordinates.shape[0]
    points = np.asarray(coefficients, dtype=np.float64)
    exact_vertices: dict[int, tuple[Fraction, ...]] = {}
    target_elements, target_routes, parent_ids, parent_vertices = {}, {}, {}, {}
    target_blocks, parent_rows = [], []
    origin = source_geometry.restriction_source
    origin_ids = {} if origin is None else origin.block_parent_cell_ids
    origin_vertices = {} if origin is None else origin.block_parent_vertex_ids
    cursor = 0
    for block, element, routes in zip(
        source.blocks, elements, source_routes, strict=True
    ):
        for corner in range(3):
            name = f"{block.name}/quad/{corner}"
            rows = np.arange(cursor, cursor + block.cell_count, dtype=np.int64)
            target_rows = 3 * rows + corner
            cells = quads[target_rows]
            composed = PolynomialComposedCellGeometryElement(
                _require_scalar_coordinate_element(element, "Quad source"),
                chart,
                numerators[corner],
                denominators[corner],
            )
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
            for cell, route in zip(cells.tolist(), np.asarray(routes), strict=True):
                charge_native_geometry_queries(4, work_units=4)
                images = coordinate_corner_images(composed, points[route])
                if images is None:
                    raise _failure(
                        "A quad source composition has no exact coordinate expressions."
                    )
                for vertex, image in zip(cell, images, strict=True):
                    if vertex in exact_vertices and exact_vertices[vertex] != image:
                        raise _failure(
                            "Curved triangle source charts disagree at a shared quad vertex."
                        )
                    exact_vertices[vertex] = image
                    value = rounded_point(image)
                    if vertex < original and not np.array_equal(
                        value, coordinates[vertex]
                    ):
                        raise _failure(
                            "Original source coordinate map differs from its topology corner carrier."
                        )
                    coordinates[vertex] = value
        cursor += block.cell_count
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry)
        if origin is None
        else origin.source_geometry_id,
        source.topology_id if origin is None else origin.source_topology_id,
        parent_ids,
        parent_vertices,
    )
    geometry = CellGeometrySpec(
        target_elements,
        target_routes,
        coefficients,
        restriction_source=ancestry,
    )
    mesh = CellMesh(
        coordinates,
        tuple(target_blocks),
        vertex_global_ids=_vertex_ids(source, added),
        numeric_version=source.numeric_version,
    )
    return mesh, geometry, np.concatenate(parent_rows)


def extract_surface_quads(
    source: CellMesh,
    limits: MeshingLimits,
    /,
    *,
    boundary_subdivision: bool = True,
    cross_field: SurfaceCrossField | None = None,
    source_geometry: CellGeometrySpec | None = None,
) -> DualExtraction:
    """Partition source triangles with genuine quad topology and exact mapped pullbacks."""
    from .._meshcore import triangle_dual_quads

    if not isinstance(source, CellMesh) or source.topology.dimension != 2:
        raise TypeError("source must be a two-dimensional CellMesh.")
    if source.periodic_topology is not None:
        raise _failure("Periodic dual extraction requires quotient subdivision ancestry.")
    if any(block.cell_kind != "triangle" for block in source.blocks):
        raise ValueError("Dual quad extraction requires triangle blocks.")
    from ..discretization._cell_geometry_validity import cell_geometry_id

    geometry_id = (
        source.geometry_id
        if source_geometry is None
        else cell_geometry_id(source_geometry)
    )
    if cross_field is not None and (
        cross_field.source_topology_id != source.topology_id
        or cross_field.source_geometry_id != geometry_id
    ):
        raise ValueError("The cross field is stale or belongs to another source mesh.")
    if not boundary_subdivision:
        raise _failure(
            "Barycentric dual quads require edge subdivision; immutable boundary chains need a parity-compatible recombination route."
        )
    cells = _entities(source, 2)
    count, original = cells.shape[0], source.coordinates.shape[0]
    _budget(
        limits, 3 * count, original, 12 * count, 2048 * count + 128 * original, 32 * count
    )
    certificate = (
        certify_cell_geometry_validity(source)
        if source_geometry is None
        else certify_cell_geometry_validity(source_geometry, mesh=source)
    )
    if certificate.invalid_count or certificate.unresolved_count:
        raise _failure("Source triangle validity is invalid or unresolved.")
    edges = _entities(source, 1)
    added = edges.shape[0] + count
    _budget(
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
    nodes = _family_host_array((count, 7), np.int64)
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
    quads = triangle_dual_quads(
        nodes, coordinates.shape[0], maximum_cells=limits.maximum_cells
    )
    geometry = None
    parent_cells = np.repeat(np.arange(count, dtype=np.int64), 3)
    if source_geometry is None:
        mesh = CellMesh(
            coordinates,
            (CellBlock("cells", "quadrilateral", quads),),
            vertex_global_ids=_vertex_ids(source, added),
            numeric_version=source.numeric_version,
        )
    else:
        mesh, geometry, parent_cells = _composed_quad_mesh(
            source, source_geometry, quads, coordinates, added
        )
    supports = _family_host_array((coordinates.shape[0], 3), np.int64)
    supports.fill(-1)
    supports[:original, 0] = np.arange(original, dtype=np.int64)
    supports[original : original + edges.shape[0], :2] = edges
    supports[original + edges.shape[0] :] = cells
    dimensions, rows = _ancestry(source, mesh, supports)
    validity = (
        certify_cell_geometry_validity(mesh)
        if geometry is None
        else certify_cell_geometry_validity(geometry, mesh=mesh)
    )
    if validity.invalid_count or validity.unresolved_count:
        raise _failure("Dual quad template has invalid or unresolved mapped cells.")
    connectivity = _entities(mesh, 2)
    valences = np.bincount(
        connectivity.reshape(-1), minlength=coordinates.shape[0]
    ).astype(np.int64)
    child_edges = _entities(mesh, 1)
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
        raise _failure("Dual quad entity count exceeds request limits.", resource=True)
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
        raise _failure(
            "Dual quad publication bytes exceed request limits.", resource=True
        )
    return DualExtraction(
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
    """Prove represented child nodes lie in their actual affine source simplex.

    Convex coordinate weights of every dual child keep its entire map in this
    parent hull. Rounded means need not stay on an oblique source face, so
    ancestry alone never establishes this exact geometric premise.
    """
    from .._meshcore import exact_orient2d, exact_orient3d

    parent_vertices = _entities(source, parent_dimension)[parent_rows]
    if parent_vertices.shape[1] != parent_dimension + 1:
        raise RuntimeError("Dual exact ancestry requires an affine simplex parent.")
    child_vertices = _entities(target, child_dimension)[child_rows]
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
                    exact_orient2d(
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
                orientation = exact_orient2d(
                    triangle[:, 0], triangle[:, 1], triangle[:, 2]
                )[:, None]
                inside = orientation != 0
                inside = inside & (
                    orientation
                    * exact_orient2d(triangle[:, 0, None], triangle[:, 1, None], query)
                    >= 0
                )
                inside &= (
                    orientation
                    * exact_orient2d(triangle[:, 1, None], triangle[:, 2, None], query)
                    >= 0
                )
                inside &= (
                    orientation
                    * exact_orient2d(triangle[:, 2, None], triangle[:, 0, None], query)
                    >= 0
                )
                contained |= inside
        if points.shape[-1] == 3:
            contained &= (
                exact_orient3d(
                    parents[:, 0, None], parents[:, 1, None], parents[:, 2, None], points
                )
                == 0
            )
    elif parent_dimension == 3:
        orientation = exact_orient3d(
            parents[:, 0], parents[:, 1], parents[:, 2], parents[:, 3]
        )[:, None]
        contained = np.broadcast_to(orientation != 0, valid.shape).copy()
        a, b, c, d = (
            parents[:, 0, None, :],
            parents[:, 1, None, :],
            parents[:, 2, None, :],
            parents[:, 3, None, :],
        )
        contained &= orientation * exact_orient3d(points, b, c, d) >= 0
        contained &= orientation * exact_orient3d(a, points, c, d) >= 0
        contained &= orientation * exact_orient3d(a, b, points, d) >= 0
        contained &= orientation * exact_orient3d(a, b, c, points) >= 0
    else:
        raise RuntimeError("A dual source simplex must have dimension zero to three.")
    return np.all(contained | ~valid, axis=1)


def remap_dual_metadata(
    extraction: DualExtraction,
    /,
    *,
    zones: tuple[MeshZone, ...] = (),
    patches: tuple[MeshPatch, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    surface_source_cover: bool = False,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    """Propagate scientific identity by exact parent topology, never proximity."""
    source, target = extraction.source, extraction.mesh

    def scope(original: MeshingScope, /) -> MeshingScope:
        dimension = original.entity_dimension
        entities = source.topology.entities(dimension)
        if (
            original.entity_set_id != entities.entity_set_id
            or original.source_id != source.mesh_id
            or original.source_revision != source.numeric_version
        ):
            raise ValueError("Dual metadata scope belongs to another source mesh.")
        parent_ids = np.asarray(entities.entity_ids, dtype=np.int64)
        dimensions = extraction.entity_parent_dimensions[dimension]
        rows = extraction.entity_parent_rows[dimension]
        selected = (dimensions == dimension) & np.isin(
            parent_ids[np.minimum(rows, parent_ids.size - 1)],
            np.asarray(original.entity_ids),
        )
        child = target.topology.entities(dimension)
        return MeshingScope(
            target.mesh_id,
            target.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            child.entity_set_id,
            np.asarray(child.entity_ids)[selected],
        )

    new_zones = tuple(
        MeshZone(
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
        MeshPatch(
            patch.name,
            scope(patch.scope),
            connected=patch.connected,
            adjacent_zone_ids=tuple(
                zone_ids[identifier] for identifier in patch.adjacent_zone_ids
            ),
        )
        for patch in patches
    )
    new_labels = tuple(MeshLabel(label.name, scope(label.scope)) for label in labels)
    new_associations = []
    coordinates = np.asarray(source.coordinates, dtype=np.float64)
    rounding = 64.0 * np.finfo(np.float64).eps * np.max(np.abs(coordinates), initial=1.0)
    for association in associations:
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            and not (
                surface_source_cover
                and association.association_kind
                in (GeometryAssociationKind.SURFACE, GeometryAssociationKind.BREP)
            )
        ):
            raise _failure(
                "Curved source association extraction requires the owning original surface cover and mandatory continuous fidelity certification."
            )
        source_dimensions = [
            degree
            for degree in range(source.topology.dimension + 1)
            if source.topology.entities(degree).entity_set_id
            == association.target_entity_set_id
        ]
        if len(source_dimensions) != 1:
            raise ValueError("Association names no unique source entity set.")
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
                "An association target ID is absent from its source entity set."
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
                original = coordinates[_entities(source, degree)[parent_rows[selected_]]]
                child = np.asarray(target.coordinates, dtype=np.float64)[
                    _entities(target, degree)[selected_]
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
                is GeometryAssociationKind.PIECEWISE_LINEAR
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
                GeometryAssociation(
                    association.association_kind,
                    association.source_id,
                    association.source_revision,
                    child_entities.entity_set_id,
                    np.asarray(child_entities.entity_ids)[selected_],
                    tuple(association.source_entity_ids[row] for row in inherited),
                    residuals,
                    resolved=np.asarray(association.resolved)[inherited_],
                    ambiguous=np.asarray(association.ambiguous)[inherited_],
                    exact=exact,
                    orientations=orientations,
                    source_dimensions=np.asarray(association.source_dimensions)[
                        inherited_
                    ]
                    if association.association_kind
                    in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
                    or association.source_entity_roles is not None
                    else None,
                    source_indices=np.asarray(association.source_indices)[inherited_]
                    if association.association_kind
                    in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
                    or association.source_entity_roles is not None
                    else None,
                    source_entity_roles=None
                    if association.source_entity_roles is None
                    else tuple(association.source_entity_roles[row] for row in inherited),
                    source_occurrence_paths=tuple(
                        association.source_occurrence_paths[row] for row in inherited
                    ),
                    parameters=np.asarray(association.parameters)[inherited_],
                    parent_dimensions=np.full(
                        (len(selected),), dimension, dtype=np.int64
                    ),
                    parent_ids=parent_ids[parent_rows[selected_]],
                    parent_association_id=association.association_id,
                    provenance=GeometryAssociationProvenance.LINEAGE,
                )
            )
    _complete_vertex_associations(extraction, associations, new_associations, rounding)
    return (
        new_zones,
        new_patches,
        new_labels,
        _coalesce_generated_associations(tuple(new_associations)),
    )


def _complete_vertex_associations(
    extraction: DualExtraction,
    originals: tuple[GeometryAssociation, ...],
    outputs: list[GeometryAssociation],
    rounding: float,
    /,
) -> None:
    """Lift an unassociated source vertex/edge through unique incidence owners.

    A barycenter of an unassociated interior source edge still belongs to the
    same authoritative region as every incident source cell. That identity is
    established by incidence, not by nearest geometry or centroid guessing.
    Ambiguous material/feature junctions require an explicit source association.
    Partial user association selections remain partial; only a complete input
    cell association activates completion of all coordinate vertices.
    """
    source, target = extraction.source, extraction.mesh
    source_cells = source.topology.entities(source.topology.dimension)
    cell_coverage = {
        identifier
        for association in originals
        if association.target_entity_set_id == source_cells.entity_set_id
        for identifier in np.asarray(association.target_global_ids).tolist()
    }
    if not set(np.asarray(source_cells.entity_ids).tolist()) <= cell_coverage:
        return
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
        vertices = _entities(source, dimension)
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
    inherited: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for target_row in range(target.coordinates.shape[0]):
        if target_ids[target_row] in covered:
            continue
        support = extraction.vertex_supports[target_row]
        original_vertices = support[support >= 0].tolist()
        candidates = set(incident[original_vertices[0]])
        for vertex in original_vertices[1:]:
            candidates.intersection_update(incident[vertex])
        chosen: tuple[int, int, int, int] | None = None
        for dimension in range(source.topology.dimension + 1):
            values = [
                (association_index, association_row, row)
                for degree, row in sorted(candidates)
                if degree == dimension
                for association_index, association_row in entity_associations[
                    (degree, row)
                ]
            ]
            identities = {
                (
                    originals[index].source_id,
                    originals[index].source_revision,
                    originals[index].source_entity_ids[association_row],
                )
                for index, association_row, _ in values
            }
            if len(identities) == 1:
                index, association_row, row = values[0]
                chosen = (index, association_row, dimension, row)
                break
        if chosen is None:
            raise _failure(
                "Coordinate vertex has ambiguous feature/material source owners; explicit source-junction ancestry is required."
            )
        index, association_row, dimension, row = chosen
        inherited.setdefault((index, dimension), []).append(
            (target_row, association_row, row)
        )
    for (index, dimension), entries in sorted(inherited.items()):
        association = originals[index]
        child_rows = np.asarray([entry[0] for entry in entries], dtype=np.int64)
        association_rows = np.asarray([entry[1] for entry in entries], dtype=np.int64)
        source_rows = np.asarray([entry[2] for entry in entries], dtype=np.int64)
        parent_ids = np.asarray(
            source.topology.entities(dimension).entity_ids, dtype=np.int64
        )[source_rows]
        exact = (
            association.exact
            and association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
            and bool(
                np.all(
                    _exact_parent_containment(
                        source, target, dimension, 0, source_rows, child_rows
                    )
                )
            )
        )
        outputs.append(
            GeometryAssociation(
                association.association_kind,
                association.source_id,
                association.source_revision,
                target_vertices.entity_set_id,
                target_ids[child_rows],
                tuple(
                    association.source_entity_ids[row]
                    for row in association_rows.tolist()
                ),
                np.asarray(association.residuals)[association_rows]
                + (0.0 if exact else rounding),
                resolved=np.asarray(association.resolved)[association_rows],
                ambiguous=np.asarray(association.ambiguous)[association_rows],
                exact=exact,
                parent_dimensions=np.full((child_rows.size,), dimension, dtype=np.int64),
                source_dimensions=np.asarray(association.source_dimensions)[
                    association_rows
                ]
                if association.association_kind
                in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
                or association.source_entity_roles is not None
                else None,
                source_indices=np.asarray(association.source_indices)[association_rows]
                if association.association_kind
                in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
                or association.source_entity_roles is not None
                else None,
                source_entity_roles=None
                if association.source_entity_roles is None
                else tuple(
                    association.source_entity_roles[row]
                    for row in association_rows.tolist()
                ),
                source_occurrence_paths=tuple(
                    association.source_occurrence_paths[row]
                    for row in association_rows.tolist()
                ),
                parameters=np.asarray(association.parameters)[association_rows],
                parent_ids=parent_ids,
                parent_association_id=association.association_id,
                provenance=GeometryAssociationProvenance.LINEAGE,
            )
        )


def _gathered_rows(
    blocks: list[Array], order: np.ndarray, dtype: type[np.generic] | None = None, /
) -> np.ndarray:
    """Concatenate per-association columns on the host once, in published row order."""
    values = np.concatenate([np.asarray(block) for block in blocks])[order]
    return values if dtype is None else values.astype(dtype, copy=False)


def _coalesce_generated_associations(
    associations: tuple[GeometryAssociation, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Publish one complete provider table per owning target/source binding.

    Different input association blocks remain actual per-row mesh genealogy.
    The generator owns the assembled provider table: it does not invent an
    aggregate parent association ID or attach rows to an unrelated parent.
    """
    groups: dict[
        tuple[GeometryAssociationKind, str, str, str], list[GeometryAssociation]
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
            raise _failure(
                "Generated association blocks assign a target entity more than once."
            )
        kind, source_id, revision, target_set = key
        role_presence = {
            association.source_entity_roles is not None for association in members
        }
        if len(role_presence) != 1:
            raise _failure(
                "Indexed and unindexed source association blocks cannot share one generated binding."
            )
        has_roles = next(iter(role_presence))
        indexed = (
            kind in (GeometryAssociationKind.BREP, GeometryAssociationKind.SURFACE)
            or has_roles
        )
        rows = order.tolist()
        source_entity_ids = [
            value for association in members for value in association.source_entity_ids
        ]
        occurrence_paths = [
            value
            for association in members
            for value in association.source_occurrence_paths
        ]
        roles = [
            value
            for association in members
            if association.source_entity_roles is not None
            for value in association.source_entity_roles
        ]
        output.append(
            GeometryAssociation(
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
                provenance=GeometryAssociationProvenance.PROVIDER,
            )
        )
    return tuple(output)
