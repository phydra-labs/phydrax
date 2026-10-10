#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native all-hex portfolio: dual closure, balanced grids and exact mapped blocks.

The dual route partitions a constrained affine tetrahedral complex into real
vertex-star hexes. Distinct balanced-grid and frame-grid routes solve a native
feature frame, prove a coordinate-plane block decomposition, and extract a
closed integer complex. The balanced route consumes the canonical spatial
octree and a bounded independent 1:8 face-parity template; the frame route
retains the feature-plane layout for element efficiency. Independent mapped
source charts retain actual coordinate expressions through exact restrictions,
including genuinely curved boundary/feature images, never node-only projection.

Nonpolycube cuts, general varying-frame singular graphs, and arbitrary CAD
projection without a source coverage chart remain explicit unclosed
research/capability gates. No route switches to dual after a
failure; none introduces a second spatial/runtime owner or relaxes pure families.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import prod
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._bvh import point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_integer
from ..discretization import (
    CellBlock,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from ..discretization._cell_geometry import (
    _require_full_p1_source,
    _require_scalar_coordinate_element,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_enclosure import (
    _solve_exact,
    add,
    bernstein_coefficients,
    coordinate_corner_images,
    coordinate_polynomials,
    derivative,
    determinant_polynomial,
    multiply,
    outward,
    physical_jacobian,
    Polynomial,
    prepared_coordinate_source_bank,
    rounded_point,
    scale,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._simplicial_locator import (
    CellLocationStatus,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from ..discretization.fem._cell_map import PreparedFiniteElementCellMap
from ..discretization.spatial._level_octree import refined_octree_leaves
from ..discretization.spatial._morton import morton_encode_integer_host, MortonAddressPlan
from ..geometry._atlas import PolygonTrimLoop, TrimDomain
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..linalg._small_batched import (
    SmallLinearSolvePlan,
    SmallLinearSolveResult,
    solve_small_linear,
)
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
from ..optim._iterative._types import MinimizationResult
from ..typing import parse
from ._contracts import MeshingLimits
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from ._quad_generation import (
    _ancestry,
    _budget,
    _charge_field_work,
    _entities,
    _failure,
    _family_host_array,
    _field_work_allowance,
    _vertex_ids,
    DualExtraction,
)
from ._reference_root_composition import (
    affine_reference_root_frame,
    exact_reference_chart_controls,
)
from ._volume_generation import PiecewiseLinearComplex, VolumeConstruction


if TYPE_CHECKING:
    from ._hex_frame_map import SourceBlockIntegerGrid


def extract_volume_hexes(
    source: CellMesh,
    limits: MeshingLimits,
    /,
    *,
    boundary_subdivision: bool = True,
    source_geometry: CellGeometrySpec | None = None,
) -> DualExtraction:
    """Partition a positive affine tetrahedral complex into actual hex cells.

    The result is a candidate. Independent source/global/coverage certification
    must still certify the *source* and the new mapped cells before publication.
    There is no inference that positive input tetrahedra cannot overlap.
    """
    from .._meshcore import tetrahedron_dual_hexes

    if not isinstance(source, CellMesh) or source.topology.dimension != 3:
        raise TypeError("source must be a three-dimensional CellMesh.")
    if source.coordinates.shape[1] != 3:
        raise ValueError("Hex extraction requires three physical coordinate axes.")
    if source.periodic_topology is not None:
        raise _failure(
            "Periodic dual hex extraction needs quotient subdivision ancestry."
        )
    if any(block.cell_kind != "tetrahedron" for block in source.blocks):
        raise ValueError("Dual hex extraction requires tetrahedral blocks.")
    if not boundary_subdivision:
        raise _failure(
            "Tetrahedral-dual all-hex closure requires boundary subdivision; immutable triangular facets cannot be retained as hex faces."
        )
    cells = _entities(source, 3)
    count, original = cells.shape[0], source.coordinates.shape[0]
    scratch, work = 8192 * count + 256 * original, 128 * count
    _budget(limits, 4 * count, original, 32 * count, scratch, work)
    source_geometry = (
        CellGeometrySpec.affine(source) if source_geometry is None else source_geometry
    )
    certificate = certify_cell_geometry_validity(source_geometry, mesh=source)
    if certificate.invalid_count or certificate.unresolved_count:
        raise _failure("Source tetrahedron validity is invalid or unresolved.")
    edges, faces = _entities(source, 1), _entities(source, 2)
    edge_count, face_count = edges.shape[0], faces.shape[0]
    added = edge_count + face_count + count
    output_edges, output_faces = (
        2 * edge_count + 3 * face_count + 4 * count,
        3 * face_count + 6 * count,
    )
    _budget(limits, 4 * count, original + added, 32 * count, scratch, work)
    if output_edges > limits.maximum_edges or output_faces > limits.maximum_faces:
        raise _failure("Dual hex edges/faces exceed request budgets.", resource=True)
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
    nodes = _family_host_array((count, 15), np.int64)
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
        raise _failure("A volume face has more than two incident tetrahedra.")
    nodes[:, 14] = original + edge_count + face_count + np.arange(count, dtype=np.int64)
    points = np.asarray(source.coordinates, dtype=np.float64)
    coordinates = _family_host_array((original + added, 3), np.float64)
    coordinates[:original] = points
    coordinates[original:] = np.nan
    hexes = tetrahedron_dual_hexes(
        nodes, coordinates.shape[0], maximum_cells=limits.maximum_cells
    )
    supports = _family_host_array((coordinates.shape[0], 4), np.int64)
    supports.fill(-1)
    supports[:original, 0] = np.arange(original, dtype=np.int64)
    supports[original : original + edge_count, :2] = edges
    supports[original + edge_count : original + edge_count + face_count, :3] = faces
    supports[original + edge_count + face_count :] = cells
    initial = (CellBlock("cells", "hexahedron", hexes),)
    mesh, geometry, parents = _parent_coordinate_geometry(
        source,
        source_geometry,
        initial,
        np.repeat(np.arange(count, dtype=np.int64), 4),
        coordinates,
        supports,
        _vertex_ids(source, added),
    )
    dimensions, rows = _ancestry(source, mesh, supports)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "Hex template positivity is invalid or unresolved after represented-coordinate construction."
        )
    boundary_faces = (dimensions[2] == 2) & (
        face_incidence[np.minimum(rows[2], face_count - 1)] == 1
    )
    boundary = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    boundary[_entities(mesh, 2)[boundary_faces].reshape(-1)] = True
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
        raise _failure("Dual hex publication bytes exceed request limits.", resource=True)
    return DualExtraction(
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


def _parent_coordinate_geometry(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target_blocks: tuple[CellBlock, ...],
    target_parents: np.ndarray,
    coordinates: np.ndarray,
    supports: np.ndarray,
    vertex_ids: np.ndarray,
    /,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray]:
    """Compose real target charts with unchanged original tetrahedral laws."""
    source_elements, source_routes, _ = source_geometry.resolve(source)
    host_routes = tuple(np.asarray(route, dtype=np.int64) for route in source_routes)
    source_values = prepared_coordinate_source_bank(source_geometry)
    source_cells = _entities(source, 3)
    root_blocks = np.repeat(
        np.arange(len(source.blocks)), [block.cell_count for block in source.blocks]
    )
    root_rows = np.concatenate([np.arange(block.cell_count) for block in source.blocks])
    groups: dict[str, list[tuple[np.ndarray, int, int]]] = {}
    group_charts = {}
    chart_controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    offset = 0
    for block in target_blocks:
        chart = coordinate_lagrange_element(block.cell_kind, 1)
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        for row in range(block_vertices.shape[0]):
            vertices = np.asarray(block_vertices[row], dtype=np.int64).reshape(-1)
            parent = int(target_parents[offset + row])
            tetrahedron = np.asarray(source_cells[parent], dtype=np.int64).reshape(-1)
            controls = [[(0, 1)] * 3 for _ in range(chart.local_dof_count)]
            for corner, vertex in enumerate(vertices):
                original = supports[vertex]
                original = original[original >= 0]
                if original.size == 0 or np.any(~np.isin(original, tetrahedron)):
                    raise _failure(
                        "Target source support does not belong to its original parent tetrahedron."
                    )
                dof = chart.entity_dofs[0][corner][0]
                for axis in range(3):
                    value = Fraction(
                        int(tetrahedron[axis + 1] in original), original.size
                    )
                    controls[dof][axis] = value.numerator, value.denominator
            element = source_elements[root_blocks[parent]]
            name = "source:" + canonical_fingerprint(
                {
                    "element": element.element_id,
                    "chart": chart.element_id,
                    "controls": controls,
                }
            )
            groups.setdefault(name, []).append(
                (vertices, parent, int(block.global_ids[row]))
            )
            if name not in chart_controls:
                numerators = _family_host_array((chart.local_dof_count, 3), np.int64)
                denominators = _family_host_array(numerators.shape, np.int64)
                for dof, values in enumerate(controls):
                    for axis, (numerator, denominator) in enumerate(values):
                        numerators[dof, axis], denominators[dof, axis] = (
                            numerator,
                            denominator,
                        )
                chart_controls[name] = numerators, denominators
                group_charts[name] = chart
        offset += block.cell_count
    if offset != target_parents.size:
        raise ValueError(
            "Target source parent rows must match the complete actual cell bank."
        )
    blocks, elements, dofs, parent_ids, parent_vertices = [], {}, {}, {}, {}
    source_ids = np.asarray(source.topology.entities(3).entity_ids, dtype=np.int64)
    source_vertices = np.asarray(source.vertex_global_ids, dtype=np.int64)
    source_images: dict[int, tuple[Fraction, ...]] = {}
    used_parents = np.unique(target_parents)
    charge_native_geometry_queries(
        4 * used_parents.size, work_units=4 * used_parents.size
    )
    for parent in used_parents:
        source_element = _require_full_p1_source(source_elements[root_blocks[parent]])
        route = host_routes[root_blocks[parent]][root_rows[parent]]
        images = coordinate_corner_images(
            source_element, tuple(source_values[int(index)] for index in route)
        )
        if images is None:
            raise _failure("Original P1 maps lack an owning exact corner operation.")
        for vertex, image in zip(source_cells[parent], images, strict=True):
            previous = source_images.setdefault(int(vertex), image)
            if previous != image:
                raise _failure(
                    "Original source traces disagree at a shared scientific vertex."
                )
    # For an authenticated P1 law this is exact evaluation at the authored
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
        represented = rounded_point(image)
        if original.size == 1:
            if not np.array_equal(coordinates[vertex], represented):
                raise _failure(
                    "Construction cannot move an original source corner to fit its map."
                )
        else:
            coordinates[vertex] = represented
    output_parents = _family_host_array((target_parents.size,), np.int64)
    output_offset = 0
    for name, cells in groups.items():
        vertices = np.stack([cell[0] for cell in cells])
        parents = np.asarray([cell[1] for cell in cells], dtype=np.int64)
        identities = np.asarray([cell[2] for cell in cells], dtype=np.int64)
        chart = group_charts[name]
        source_element = _require_full_p1_source(source_elements[root_blocks[parents[0]]])
        if chart.cell_kind == "pyramid":
            element = RationalComposedCellGeometryElement(
                source_element, chart, *chart_controls[name]
            )
        else:
            element = PolynomialComposedCellGeometryElement(
                source_element, chart, *chart_controls[name]
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
    try:
        mesh = CellMesh(
            coordinates,
            tuple(blocks),
            vertex_global_ids=vertex_ids,
            numeric_version=source.numeric_version,
        )
    except ValueError as error:
        raise _failure(f"Source-defined target topology is invalid: {error}") from error
    provenance = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry), source.topology_id, parent_ids, parent_vertices
    )
    geometry = CellGeometrySpec(
        elements,
        dofs,
        source_geometry.coordinates,
        exact_source=source_geometry.exact_source,
        restriction_source=provenance,
    )
    return mesh, geometry, output_parents


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
            raise ValueError("Hex grid maximum depth must be at most 20.")
        if (
            not np.isfinite(minimum_scaled_jacobian)
            or not 0.0 < minimum_scaled_jacobian <= 1.0
        ):
            raise ValueError("minimum_scaled_jacobian must lie in (0, 1].")
        if (
            not np.isfinite(minimum_mean_ratio)
            or not 0.0 < minimum_mean_ratio <= 1.0
            or not np.isfinite(maximum_aspect_ratio)
            or maximum_aspect_ratio < 1.0
        ):
            raise ValueError("Grid mean-ratio/aspect-ratio requirements are invalid.")
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
    solve: SmallLinearSolveResult
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


def _frame_rotation(angles: Array, /) -> Array:
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
    projected = normals @ (initial @ _frame_rotation(angles)).T
    # Fourth-order octahedral energy is invariant to axis signs/permutations.
    return jnp.sum(1.0 - jnp.sum(projected**4, axis=1))


def prepare_hex_frame(
    complex_: PiecewiseLinearComplex,
    schedule: NativeHexGridSchedule,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
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
        raise _failure("A feature patch has no regular orientation for its frame field.")
    normals /= lengths[:, None]
    first = normals[0]
    transverse = np.flatnonzero(np.abs(normals @ first) < 0.5)
    if transverse.size == 0:
        raise _failure("Feature normals do not span a volume frame.")
    second = normals[transverse[0]] - np.sum(normals[transverse[0]] * first) * first
    second /= np.linalg.norm(second)
    initial = np.stack((first, second, np.cross(first, second)))
    method = ProjectedLBFGS()
    budget = _field_work_allowance(method, normals.shape[0], schedule.maximum_field_steps)
    optimization = minimize(
        _frame_energy,
        jnp.zeros((3,), dtype=jnp.float64),
        method=method,
        args=(jnp.asarray(initial), jnp.asarray(normals)),
        bounds=Bounds(
            jnp.full((3,), -np.pi / 4.0, dtype=jnp.float64),
            jnp.full((3,), np.pi / 4.0, dtype=jnp.float64),
        ),
        termination=OptimizationTermination(maximum_steps=schedule.maximum_field_steps),
    )
    _charge_field_work(budget, optimization, normals.shape[0])
    angles = optimization.parameters
    if not isinstance(angles, Array):
        raise RuntimeError(
            "The native frame optimizer returned a non-array parameter tree."
        )
    basis = np.asarray(
        jax.device_get(jnp.asarray(initial) @ _frame_rotation(angles)), dtype=np.float64
    )
    projected = normals @ basis.T
    axes = np.argmax(np.abs(projected), axis=1).astype(np.int64)
    residuals = 1.0 - np.max(np.abs(projected), axis=1)
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        np.broadcast_to(basis.T, (complex_.vertices.shape[0], 3, 3)),
        complex_.vertices,
    )
    logical, successful = jax.device_get((solve.value, solve.successful))
    if not np.all(np.asarray(successful, dtype=np.bool_)):
        raise _failure("The native frame pullback is singular or ill-conditioned.")
    logical_ = np.asarray(logical, dtype=np.float64)
    # Exact postclassification, not epsilon snapping: every admitted source
    # polygon must lie on one represented coordinate plane of the solved frame.
    for row, loop in enumerate(loops):
        if np.any(logical_[loop, axes[row]] != logical_[loop[0], axes[row]]):
            raise _failure(
                "The solved frame has a non-grid planar patch: a positive integer-map block decomposition is unresolved for this feature arrangement."
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
) -> np.ndarray:
    # Logical integer identity is authoritative. Interpolation places vertices;
    # it never decides identity or merges two nearby geometry strata.
    logical = _family_host_array(vertices.shape, np.float64)
    for axis in range(3):
        position = vertices[:, axis] / subdivisions
        lower = np.floor(position).astype(np.int64)
        capped = np.minimum(lower, knots[axis].size - 2)
        fraction = position - capped
        logical[:, axis] = knots[axis][capped] + fraction * (
            knots[axis][capped + 1] - knots[axis][capped]
        )
    physical = _family_host_array(vertices.shape, np.float64)
    np.matmul(logical, basis, out=physical)
    return physical


def _grid_face_sources(
    complex_: PiecewiseLinearComplex,
    source_logical: np.ndarray,
    grid: np.ndarray,
    faces: np.ndarray,
    frame: HexFrameField,
    knots: tuple[np.ndarray, ...],
    subdivisions: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact trim membership of face centers after a boundary-plane exclusion proof."""
    face_points = np.mean(grid[faces], axis=1)
    source_faces = _family_host_array((faces.shape[0],), np.int64)
    source_faces.fill(-1)
    source_polygons = _family_host_array((faces.shape[0],), np.int64)
    source_polygons.fill(-1)
    for polygon in range(complex_.polygon_facets.size):
        loop = complex_.polygon_vertices[
            complex_.polygon_offsets[polygon] : complex_.polygon_offsets[polygon + 1]
        ]
        axis = int(frame.patch_axes[polygon])
        plane = source_logical[loop[0], axis]
        index = np.flatnonzero(knots[axis] == plane)
        if index.size != 1:
            raise RuntimeError("A source patch lost its authoritative grid plane.")
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
        trim = TrimDomain(PolygonTrimLoop(ordinal))
        classified = trim.classify(face_points[rows][:, remaining])
        if not np.all(np.asarray(classified.resolved)):
            raise _failure("A source patch trim has unresolved integer-grid membership.")
        chosen = rows[np.asarray(classified.inside) | np.asarray(classified.boundary)]
        facet = complex_.polygon_facets[polygon]
        occupied = source_faces[chosen]
        if np.any((occupied >= 0) & (occupied != facet)):
            raise _failure(
                "A grid face straddles incompatible authoritative source facets."
            )
        source_faces[chosen] = facet
        source_polygons[chosen] = polygon
    return source_faces, source_polygons


def generate_integer_grid_hexes(
    complex_: PiecewiseLinearComplex,
    construction: VolumeConstruction | MappedReferenceDomain,
    schedule: NativeHexGridSchedule,
    limits: MeshingLimits,
    target_size: float,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> GridHexConstruction:
    """Automatic polycube patch/block decomposition and closed integer extraction.

    Boundary and material polygons are first proved to lie on grid planes.
    Consequently no open integer cell can meet a domain interface: its located
    source region is constant on the whole cell, not guessed from its centroid.
    Balanced-grid selection consumes the canonical dyadic owner and closes
    every coarse/fine face by bounded repeated 1:8 templates. The resulting
    complete-subdivision closure is robust but may be less efficient than the
    feature-plane frame route; neither route is substituted after a failure.
    """
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise ValueError("target_size must be positive and finite.")
    frame, logical = prepare_hex_frame(complex_, schedule, record_phase=record_phase)
    topology_started = phase_started(record_phase)
    if isinstance(construction, MappedReferenceDomain):
        root_points = np.asarray(
            construction.reference_mesh.coordinates, dtype=np.float64
        )
        pullback = solve_small_linear(
            SmallLinearSolvePlan(3),
            np.broadcast_to(frame.basis.T, (root_points.shape[0], 3, 3)),
            root_points,
        )
        pulled, successful = jax.device_get((pullback.value, pullback.successful))
        if not np.all(successful):
            raise _failure("Independent source-reference chart feature pullback failed.")
        breaks = np.concatenate((logical, np.asarray(pulled, dtype=np.float64)))
    else:
        breaks = logical
    knots = tuple(np.unique(breaks[:, axis]) for axis in range(3))
    if any(values.size < 2 for values in knots):
        raise _failure("The feature-plane decomposition has a collapsed volume axis.")
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
            raise _failure(
                "An authoritative feature chain cuts an integer block face diagonally; parity-compatible cut templates remain unresolved."
            )
    for a, b in complex_.segments.tolist():
        if np.count_nonzero(logical[a] != logical[b]) != 1:
            raise _failure("An embedded source curve is not an integer-grid edge chain.")
    smallest = min(float(np.min(np.diff(values))) for values in knots)
    placement_size = min(target_size, smallest * schedule.maximum_aspect_ratio)
    axis_counts: list[list[int]] = []
    for values in knots:
        counts: list[int] = []
        axis_size = 0
        for first, last in zip(values[:-1], values[1:], strict=True):
            if placement_size < (last - first) / (1 << schedule.maximum_depth):
                raise _failure(
                    "Feature-grid interval subdivisions exceed the maximum dyadic address depth.",
                    resource=True,
                )
            count = max(1, int(np.ceil((last - first) / placement_size)))
            axis_size += count
            if axis_size > (1 << schedule.maximum_depth):
                raise _failure(
                    "Feature-grid axis capacity exceeds the maximum dyadic address depth.",
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
        raise _failure(
            "The requested feature/size integer grid exceeds its declared maximum depth.",
            resource=True,
        )
    required = (
        (1 << depth) ** 3
        if schedule.route == "balanced_grid"
        else prod(int(value) for value in sizes)
    )
    _budget(limits, required, 0, required * 8, required * 1024, required * 32)
    refined_knots = []
    for values, counts in zip(knots, axis_counts, strict=True):
        refined_axis = _family_host_array((sum(counts) + 1,), np.float64)
        offset = 0
        for first, last, count in zip(values[:-1], values[1:], counts, strict=True):
            refined_axis[offset : offset + count] = (
                first + (last - first) * np.arange(count, dtype=np.float64) / count
            )
            offset += count
        refined_axis[-1] = values[-1]
        refined_knots.append(refined_axis)
    knots = tuple(refined_knots)
    octree_prefixes = np.empty((0,), dtype=np.uint64)
    octree_levels = np.empty((0,), dtype=np.int64)
    closure_rounds = 0
    if schedule.route == "balanced_grid":
        address = MortonAddressPlan(
            (0.0, 0.0, 0.0), tuple(float(1 << depth) for _ in range(3)), depth
        )
        # Source feature points seed full fine-level paths; canonical balance
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
    else:
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
    if schedule.route == "balanced_grid":
        from .._meshcore import subdivide_hex_grid

        parent_corners = np.unique((integer_cells // 2) * 2, axis=0)
        grid_offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int64)
        integer_vertices, inverse = np.unique(
            (parent_corners[:, None, :] + grid_offsets[None, :, :]).reshape((-1, 3)),
            axis=0,
            return_inverse=True,
        )
        nodes = inverse.reshape((-1, 27)).astype(np.int64)
        cells = subdivide_hex_grid(
            nodes, integer_vertices.shape[0], maximum_cells=limits.maximum_cells
        )
        child_corners = (
            parent_corners[:, None, :]
            + np.asarray(list(product(range(2), repeat=3)), dtype=np.int64)[None, :, :]
        ).reshape((-1, 3))
        active = np.all(child_corners < sizes, axis=1)
        cells, integer_cells = cells[active], child_corners[active]
    else:
        integer_vertices, inverse = np.unique(
            (integer_cells[:, None, :] + offsets[None, :, :]).reshape((-1, 3)),
            axis=0,
            return_inverse=True,
        )
        cells = inverse.reshape((-1, 8))
    coordinates = _grid_vertex_coordinates(
        integer_vertices, knots, subdivisions, frame.basis
    )
    centers = np.mean(coordinates[cells], axis=1)
    # Reuse the existing inverse-cell-map/BVH owner; all candidate containing
    # source cells must have one region, with resource/nonconvergence explicit.
    record_elapsed(record_phase, "topology_construction", topology_started)
    classification_started = phase_started(record_phase)
    regions = _family_host_array((cells.shape[0],), np.int64)
    regions.fill(-1)
    charge_native_geometry_queries(cells.shape[0], work_units=cells.shape[0])
    if isinstance(construction, MappedReferenceDomain):
        root_cells = _entities(construction.reference_mesh, 3)
        root_corners = np.asarray(
            construction.reference_mesh.coordinates, dtype=np.float64
        )[root_cells]
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
            raise _failure(
                "Mapped grid source roots must be exact positive axis-aligned affine reference charts."
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
            raise _failure(
                "Mapped reference-root classification exceeds its bounded capacity.",
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
                raise _failure(
                    "An integer cell has overlapping independent source-reference roots."
                )
            if containing:
                regions[row] = construction.cell_regions[containing[0]]
            elif values.size:
                raise _failure(
                    "An integer cell crosses a declared source-reference chart boundary."
                )
        numeric_version = construction.source_revision
    else:
        space = FiniteElementPlan(
            construction.mesh,
            FiniteElementFieldSpec("grid-region", lagrange_element("tetrahedron", 1)),
        ).prepare()
        locator = PreparedSimplicialCellLocator(
            PreparedFiniteElementCellMap(space, 0),
            space.default_runtime.coordinates,
            SimplicialLocationPolicy(schedule.maximum_location_candidates, 16, 1),
        )
        located = locator.locate(centers)
        statuses, inside, candidates = jax.device_get(
            (located.status, located.inside, located.candidate_cells)
        )
        if np.any(
            (statuses != CellLocationStatus.LOCATED)
            & (statuses != CellLocationStatus.OUTSIDE)
        ):
            raise _failure(
                "Source region location exhausted its certified candidate/inverse-map workset.",
                resource=True,
            )
        for row in np.flatnonzero(inside).tolist():
            containing = np.asarray(candidates[row], dtype=np.int64)
            containing = containing[containing >= 0]
            values = construction.cell_regions[containing]
            if values.size == 0 or np.any(values != values[0]):
                raise _failure(
                    "An integer cell center lies on incompatible material regions despite the boundary-plane proof."
                )
            regions[row] = values[0]
        numeric_version = construction.mesh.numeric_version
    record_elapsed(record_phase, "region_classification", classification_started)
    kept = regions >= 0
    cells, regions, integer_cells = cells[kept], regions[kept], integer_cells[kept]
    if cells.shape[0] == 0:
        raise _failure("The integer-map extraction covers no source region.")
    used, compact = np.unique(cells, return_inverse=True)
    cells = compact.reshape(cells.shape)
    _budget(limits, cells.shape[0], used.size, cells.size, required * 1024, required * 32)
    if used.size != coordinates.shape[0]:
        retained_coordinates = _family_host_array((used.size, 3), coordinates.dtype)
        retained_vertices = _family_host_array((used.size, 3), integer_vertices.dtype)
        np.take(coordinates, used, axis=0, out=retained_coordinates)
        np.take(integer_vertices, used, axis=0, out=retained_vertices)
        coordinates, integer_vertices = retained_coordinates, retained_vertices
    # Count actual retained entities, not the bounding box: cavities and
    # disconnected components may discard most candidate cells. Admission must
    # nevertheless precede construction of the canonical connectivity owner.
    topology = reference_cell_topology("hexahedron")
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = np.asarray(topology.entities[dimension], dtype=np.int64)
        entity_rows = np.sort(
            cells[:, local_entities].reshape((-1, local_entities.shape[1])), axis=1
        )
        count = np.unique(entity_rows, axis=0).shape[0]
        if count > bound:
            raise _failure(
                f"Integer-grid extraction requires {count} dimension-{dimension} entities; limit is {bound}.",
                resource=True,
            )
    del entity_rows
    mesh = CellMesh(
        coordinates,
        (CellBlock("integer_blocks", "hexahedron", cells),),
        numeric_version=numeric_version,
    )
    validity = certify_cell_geometry_validity(mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "The integer-grid block maps lack globally positive per-cell determinant enclosures."
        )
    face_sources, face_polygons = _grid_face_sources(
        complex_,
        logical,
        integer_vertices,
        _entities(mesh, 2),
        frame,
        knots,
        subdivisions,
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
        raise _failure(
            f"Integer-grid publication requires {data_bytes} retained array bytes; limit is {limits.maximum_data_bytes}.",
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
    """Actual curved maps retaining independent declared source coefficients."""

    mesh: CellMesh
    reference_mesh: CellMesh
    geometry: CellGeometrySpec
    cell_regions: np.ndarray
    source_parent_rows: np.ndarray
    target_reference_rows: np.ndarray
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    mapped_id: str = eqx.field(static=True)


def mapped_source_lipschitz_bound(
    reference_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    /,
) -> float:
    """Source-level interval bound for physical placement in reference charts.

    Reference roots must be exactly affine. The actual reference-frame solve
    and the complete source Jacobian are rational; no sampled singular value
    or rounded inverse matrix supplies the physical placement bound.
    """
    elements, routes, _ = source_geometry.resolve(reference_mesh)
    corners = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    source_values = source_geometry.source_coordinates()
    root_count = sum(block.cell_count for block in reference_mesh.blocks)
    charge_native_geometry_queries(root_count, work_units=9 * root_count)
    upper = 0.0
    for block, element, route in zip(
        reference_mesh.blocks, elements, routes, strict=True
    ):
        cells = np.asarray(block.vertices, dtype=np.int64)
        block_routes = np.asarray(route, dtype=np.int64)
        for row in range(cells.shape[0]):
            matrix, _ = affine_reference_root_frame(corners[cells[row]], block.cell_kind)
            inverse = _solve_exact(
                matrix,
                [
                    [Fraction(int(axis == column)) for column in range(3)]
                    for axis in range(3)
                ],
            )
            polynomials = coordinate_polynomials(
                element, tuple(source_values[index] for index in block_routes[row])
            )
            if polynomials is None:
                raise _failure(
                    "Mapped source has no canonical exact coordinate expression."
                )
            bounds = tuple(
                tuple(
                    _polynomial_magnitude(
                        add(
                            add(
                                scale(derivative(value, 0), inverse[0][axis]),
                                scale(derivative(value, 1), inverse[1][axis]),
                            ),
                            scale(derivative(value, 2), inverse[2][axis]),
                        ),
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
        raise _failure("The mapped source placement bound is nonfinite or degenerate.")
    return upper


def source_block_interval_counts(
    reference_mesh: CellMesh,
    physical_lipschitz_bound: float,
    target_size: float,
    maximum_depth: int,
    /,
) -> np.ndarray:
    """Size authored local axes by exact original reference-edge norm bounds."""
    cells = _entities(reference_mesh, 3)
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    counts = _family_host_array((cells.shape[0], 3), np.int64)
    for root, vertices in enumerate(cells):
        matrix, _ = affine_reference_root_frame(points[vertices], "hexahedron")
        for axis in range(3):
            length = _sqrt_upper(sum((row[axis] ** 2 for row in matrix), Fraction(0)))
            ratio = (
                Fraction(physical_lipschitz_bound)
                * Fraction(length)
                / Fraction(target_size)
            )
            count = max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
            if count > 1 << maximum_depth:
                raise _failure(
                    "Original physical size requires a local source-axis count beyond its declared integer depth.",
                    resource=True,
                )
            counts[root, axis] = count
    return counts


def realize_mapped_grid_hexes(
    grid: GridHexConstruction | SourceBlockIntegerGrid,
    source_reference_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_cell_regions: np.ndarray,
    limits: MeshingLimits,
    /,
) -> MappedGridHexConstruction:
    """Restrict independently declared source maps without nodal reconstruction.

    Chart containment and inverse coordinates use exact source-root arithmetic.
    Rational reference-chart controls retain nonbinary inverse ratios without
    fitting or replacing any original physical source coefficients.
    Every shared coordinate value is checked, but global
    physical embedding and exact reference-chart coverage remain independent
    certification work, never inferred from this construction.
    """
    if any(block.cell_kind != "hexahedron" for block in source_reference_mesh.blocks):
        raise _failure("Mapped extraction needs hexahedral source reference roots.")
    source_elements, source_routes, _ = source_geometry.resolve(source_reference_mesh)
    source_values = prepared_coordinate_source_bank(source_geometry)
    source_cells = np.concatenate(
        [
            np.asarray(block.vertices, dtype=np.int64)
            for block in source_reference_mesh.blocks
        ]
    )
    source_regions = np.asarray(source_cell_regions, dtype=np.int64)
    if source_regions.shape != (source_cells.shape[0],):
        raise ValueError(
            "Mapped source region rows must match its declared reference cells."
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
        raise _failure("Source reference roots are not exact affine hexahedral charts.")
    reference_mesh = grid.mesh
    reference_cells = np.asarray(reference_mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    from ._hex_frame_map import SourceBlockIntegerGrid

    chart = coordinate_lagrange_element("hexahedron", 1)
    if isinstance(grid, SourceBlockIntegerGrid):
        if (
            grid.source_frame.reference_mesh.mesh_id != source_reference_mesh.mesh_id
            or grid.source_frame.source_id != cell_geometry_id(source_geometry)
        ):
            raise _failure(
                "Integer source charts name a different original root mesh or coordinate bank."
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
            raise _failure(
                "Integer source charts lack complete original parent/control rows."
            )
        if (
            np.any(denominators <= 0)
            or np.any(numerators < 0)
            or np.any(numerators > denominators)
        ):
            raise _failure(
                "An exact integer source chart leaves its original reference root."
            )
    else:
        target_corners = points[reference_cells]
        bvh = prepare_bvh(
            np.min(root_corners, axis=1), np.max(root_corners, axis=1), dtype=np.float64
        )
        charge_native_geometry_queries(
            reference_cells.shape[0], work_units=reference_cells.shape[0]
        )
        candidates, valid, complete = jax.device_get(
            point_select_leaf_items(
                np.mean(target_corners, axis=1), bvh=bvh, maximum_candidates=16
            )
        )
        if not np.all(complete):
            raise _failure(
                "Mapped reference-root lookup exceeds its bounded candidate capacity.",
                resource=True,
            )
        parent_rows = _family_host_array((reference_cells.shape[0],), np.int64)
        parent_rows.fill(-1)
        numerators = _family_host_array(
            (reference_cells.shape[0], chart.local_dof_count, 3), np.int64
        )
        denominators = _family_host_array(numerators.shape, np.int64)
        for row in range(reference_cells.shape[0]):
            containing = []
            for slot in np.flatnonzero(valid[row]).tolist():
                parent = int(candidates[row, slot])
                controls = exact_reference_chart_controls(
                    target_corners[row], root_corners[parent], "hexahedron", chart
                )
                if controls is not None:
                    containing.append((parent, controls))
            if len(containing) != 1:
                raise _failure(
                    "A target integer cell has no unique exact source-reference chart containment."
                )
            parent, controls = containing[0]
            if any(values.dtype.kind not in "iu" for values in controls):
                raise _failure(
                    "The exact reference chart exceeds the canonical signed-64-bit rational-control representation."
                )
            parent_rows[row] = parent
            numerators[row], denominators[row] = controls
    if np.any(source_regions[parent_rows] != grid.cell_regions):
        raise _failure(
            "Mapped source reference-region identities disagree with extracted block regions."
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
        raise _failure(
            "Mapped restriction groups exceed the declared work budget.", resource=True
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
    physical_points = _family_host_array(points.shape, np.float64)
    physical_points.fill(np.nan)
    exact_points: dict[int, tuple[Fraction, ...]] = {}
    charge_native_geometry_queries(
        8 * reference_cells.shape[0], work_units=8 * reference_cells.shape[0]
    )
    order = []
    for name, rows_ in sorted(groups.items()):
        rows = np.asarray(rows_, dtype=np.int64)
        rows = rows[np.argsort(target_ids[rows], kind="stable")]
        order.extend(rows.tolist())
        parents = parent_rows[rows]
        prototype = rows[0]
        source_element = _require_scalar_coordinate_element(
            source_elements[block_of_root[parents[0]]], "mapped source"
        )
        element = PolynomialComposedCellGeometryElement(
            source_element, chart, numerators[prototype], denominators[prototype]
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
            images = coordinate_corner_images(
                element, tuple(source_values[int(index)] for index in source_route)
            )
            if images is None:
                raise _failure(
                    "Mapped source restrictions have no exact corner-image operation."
                )
            for vertex, image in zip(cell, images, strict=True):
                previous = exact_points.get(vertex)
                if previous is not None and previous != image:
                    raise _failure(
                        "Independent source chart restrictions disagree exactly at a shared coordinate vertex."
                    )
                exact_points[vertex] = image
                physical_points[vertex] = rounded_point(image)
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
    provenance = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source_reference_mesh.topology_id,
        parent_ids,
        parent_vertices,
    )
    geometry = CellGeometrySpec(
        elements,
        dofs,
        source_geometry.coordinates,
        exact_source=source_geometry.exact_source,
        restriction_source=provenance,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "The mapped source restrictions have invalid or unresolved physical cell maps."
        )
    scaled, mean, aspect = exact_mapped_volume_quality(mesh, geometry)
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
                "source_geometry": cell_geometry_id(source_geometry),
                "source_topology": source_reference_mesh.topology_id,
                "mesh": mesh.mesh_id,
                "geometry": cell_geometry_id(geometry),
            }
        ),
    )


def exact_mapped_volume_quality(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enclose actual whole-cell volume quality using original source expressions.

    The aspect bound is the physical Jacobian's spectral condition number,
    enclosed by matrix one/infinity norms and exact polynomial cofactors. It is
    not the aspect of a corner-defined surrogate or a sampled Jacobian.
    """
    if any(
        block.cell_kind not in ("hexahedron", "tetrahedron", "pyramid")
        for block in mesh.blocks
    ):
        raise ValueError(
            "Exact mapped volume quality requires supported hex/tet/pyramid blocks."
        )
    elements, routes, _ = geometry.resolve(mesh)
    coordinates = prepared_coordinate_source_bank(geometry)
    count = sum(block.cell_count for block in mesh.blocks)
    scaled = _family_host_array((count,), np.float64)
    mean = _family_host_array((count,), np.float64)
    aspect = _family_host_array((count,), np.float64)
    charge_native_geometry_queries(count, work_units=count)
    row_index = 0
    for block, element, block_routes in zip(mesh.blocks, elements, routes, strict=True):
        domain = "simplex" if block.cell_kind == "tetrahedron" else "box"
        for route in np.asarray(block_routes, dtype=np.int64):
            polynomials = coordinate_polynomials(
                element, tuple(coordinates[int(index)] for index in route)
            )
            if polynomials is None:
                raise _failure(
                    "The mapped restriction lacks a canonical interval quality source."
                )
            jacobian = physical_jacobian(polynomials, block.cell_kind, 3)
            if jacobian is None:
                raise _failure(
                    "The actual physical-reference Jacobian interval source is unsupported."
                )
            determinant = determinant_polynomial(polynomials, block.cell_kind, 3)
            if determinant is None:
                raise _failure("The mapped determinant interval source is unsupported.")
            determinant_lower = min(bernstein_coefficients(determinant, domain, 3))
            if determinant_lower <= 0:
                raise _failure(
                    "Whole-cell mapped quality determinant positivity is unresolved."
                )
            bounds = tuple(
                tuple(_polynomial_magnitude(value, domain) for value in row)
                for row in jacobian
            )
            column_squared = tuple(
                sum((row[axis] ** 2 for row in bounds), Fraction(0)) for axis in range(3)
            )
            norm_squared = sum(column_squared, Fraction(0))
            product_squared = column_squared[0] * column_squared[1] * column_squared[2]
            column_product = _sqrt_upper(product_squared)
            scaled[row_index] = outward(
                determinant_lower / Fraction(column_product), -np.inf
            )
            root = float(determinant_lower) ** (2.0 / 3.0)
            squared = determinant_lower**2
            for _ in range(8):
                if Fraction(root) ** 3 <= squared:
                    break
                root = float(np.nextafter(root, -np.inf))
            if root < 0.0 or Fraction(root) ** 3 > squared:
                raise _failure(
                    "Directed mapped mean-ratio root construction is unresolved."
                )
            mean[row_index] = outward(3 * Fraction(root) / norm_squared, -np.inf)
            cofactors = []
            for row in range(3):
                a, b = [index for index in range(3) if index != row]
                cofactor_row = []
                for column in range(3):
                    c, d = [index for index in range(3) if index != column]
                    minor = add(
                        multiply(jacobian[a][c], jacobian[b][d]),
                        scale(multiply(jacobian[a][d], jacobian[b][c]), Fraction(-1)),
                    )
                    cofactor_row.append(_polynomial_magnitude(minor, domain))
                cofactors.append(tuple(cofactor_row))
            aspect[row_index] = _sqrt_upper(
                _matrix_norm_product(bounds)
                * _matrix_norm_product(tuple(cofactors))
                / squared
            )
            row_index += 1
    return scaled, mean, aspect


def _matrix_norm_product(bounds: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    row_norm = max(sum(row, Fraction(0)) for row in bounds)
    column_norm = max(
        sum((row[column] for row in bounds), Fraction(0)) for column in range(3)
    )
    return row_norm * column_norm


def _polynomial_magnitude(polynomial: Polynomial, domain: str = "box", /) -> Fraction:
    return max(abs(value) for value in bernstein_coefficients(polynomial, domain, 3))


def _sqrt_upper(value: Fraction, /) -> float:
    result = float(np.sqrt(outward(value, np.inf)))
    if np.isfinite(result) and Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
    if not np.isfinite(result) or Fraction(result) ** 2 < value:
        raise _failure("Directed mapped spectral-norm construction is unresolved.")
    return result
