"""Prepare source-chart varying frames and integer-map compatibility.

Frames are derivatives of retained coordinate expressions. Extraction restricts
those expressions and their source geometry identity to compatible integer cells.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization.fem._reference as reference_owner
import phydrax.meshing._hex_frame_records as frame_records_owner
from phydrax.meshing._host_banks import entities, failure, family_host_array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._strict import StrictModule
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    coordinate_polynomials,
    derivative,
    evaluate,
)
from ..discretization._reference_cell import reference_cell_topology
from ..ein import contract
from ..linalg._exact import determinant_exact
from ..linalg._small_batched import SmallLinearSolvePlan, solve_small_linear
from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
from ..optim._iterative._types import MinimizationResult
from ..typing import checked
from ._contracts import MeshingLimits
from ._family_execution import (
    charge_field_work,
    check_family_limits,
    coordinate_phase,
    count_geometry_queries,
    field_work_allowance,
)
from ._hex_generation import (
    frame_rotation,
    GridHexConstruction,
    MappedGridHexConstruction,
    realize_mapped_grid_hexes,
)
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from ._reference_root_composition import affine_reference_root_frame
from ._structured import logical_cell_bank


# Shared vertex IDs glue local tensor-face cycles without proximity matching.
_HEX_TOPOLOGY = reference_cell_topology("hexahedron")
_FACES = _HEX_TOPOLOGY.entities[2]
_POINTS = tuple(tuple(Fraction(v) for v in point) for point in _HEX_TOPOLOGY.vertices)
_INTEGER_POINTS = tuple(tuple(int(v) for v in point) for point in _HEX_TOPOLOGY.vertices)
_EDGE_AXES = tuple(
    tuple(
        edge
        for edge in _HEX_TOPOLOGY.entities[1]
        if _POINTS[edge[0]][axis] != _POINTS[edge[1]][axis]
    )
    for axis in range(3)
)


def chart_face_permutation(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the reversed tensor-face permutation from shared vertex IDs."""
    if (
        len(left) != 4
        or len(right) != 4
        or len(set(left)) != 4
        or set(left) != set(right)
    ):
        raise failure(
            "source_frame.face_connection must identify four distinct shared vertices; got incompatible tensor faces"
        )
    permutation = tuple(right.index(vertex) for vertex in left)
    if any(
        (permutation[(index + 1) % 4] - permutation[index]) % 4 != 3 for index in range(4)
    ):
        raise failure(
            "source_frame.boundary_parity must reverse shared chart cycles; got incompatible orientations"
        )
    return permutation


def _prepare_source_frame_corners(
    reference_mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    cells: np.ndarray,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.SourceFrameCornerData:
    with coordinate_phase(scope=scope, budget=budget):
        elements, routes, _ = geometry.resolve(
            reference_mesh, budget=budget, exact_source_prepared=False
        )
        values = geometry.source_coordinates(budget=budget)
        host_routes = jax.device_get(routes)
        frames = family_host_array((cells.shape[0], 8, 3, 3), np.float64, scope=scope)
        varying = []
        root = 0
        for _block, element, routes_ in zip(
            reference_mesh.blocks, elements, host_routes, strict=True
        ):
            routes_host = np.asarray(routes_, dtype=np.int64)
            for route in routes_host:
                polynomials = coordinate_polynomials(
                    element, tuple(values[index] for index in route), budget=budget
                )
                if polynomials is None:
                    raise failure(
                        "source_geometry.chart_derivatives must have a polynomial representation; got unsupported expressions"
                    )
                jets = tuple(
                    tuple(derivative(value, axis, budget=budget) for axis in range(3))
                    for value in polynomials
                )
                varying.append(
                    any(any(any(power) for power in jet) for row in jets for jet in row)
                )
                for corner, point in enumerate(_POINTS):
                    frames[root, corner] = [
                        [float(evaluate(jet, point, budget=budget)) for jet in row]
                        for row in jets
                    ]
                root += 1
        count_geometry_queries(root * 8, work_units=root * 72, budget=budget)
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        jnp.asarray(frames),
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), frames.shape),
    )
    successful = np.asarray(jax.device_get(solve.successful))
    if not np.all(successful):
        raise failure(
            "source_frame.corner_reciprocals must have successful solve status; got a singular corner"
        )
    return frame_records_owner.SourceFrameCornerData(
        frames, solve.value, np.asarray(varying, dtype=np.bool_)
    )


def _prepare_source_frame_faces(
    cells: np.ndarray,
) -> frame_records_owner.SourceFrameFaceData:
    bank: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    connections, permutations, boundary = [], [], []
    for root, cell in enumerate(cells):
        for face, local in enumerate(_FACES):
            cycle = tuple(int(cell[index]) for index in local)
            key = tuple(sorted(cycle))
            bank.setdefault(key, []).append((root, face, cycle))
    for occurrences in bank.values():
        if len(occurrences) == 1:
            boundary.append(occurrences[0][:2])
            continue
        if len(occurrences) != 2:
            raise failure(
                "source_frame.face_incidence must have one or two chart occurrences; got nonmanifold incidence"
            )
        left, right = occurrences
        permutation = chart_face_permutation(left[2], right[2])
        connections.append((left[0], left[1], right[0], right[1]))
        permutations.append(permutation)
    return frame_records_owner.SourceFrameFaceData(
        np.asarray(connections, dtype=np.int64).reshape((-1, 4)),
        np.asarray(permutations, dtype=np.int64).reshape((-1, 4)),
        np.asarray(boundary, dtype=np.int64).reshape((-1, 2)),
    )


@checked
def prepare_source_hex_frame(
    reference_mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.PreparedSourceHexFrame:
    """Prepare source-chart frames and their tensor-face connection.

    Physical trace continuity and global embedding require independent certificates.
    """
    started = phase_started(record_phase)
    if any(block.cell_kind != "hexahedron" for block in reference_mesh.blocks):
        raise failure(
            "reference_mesh.blocks must contain hexahedral charts; got another cell kind"
        )
    cells = np.concatenate(
        [
            np.asarray(jax.device_get(block.vertices), dtype=np.int64)
            for block in reference_mesh.blocks
        ]
    )
    points = np.asarray(jax.device_get(reference_mesh.coordinates), dtype=np.float64)
    for cell in cells:
        affine_reference_root_frame(points[cell], "hexahedron")
    with coordinate_phase(scope=scope, budget=budget):
        validity = certify_cell_geometry_validity(
            geometry, mesh=reference_mesh, budget=budget
        )
    if validity.invalid_count or validity.unresolved_count:
        raise failure(
            "source_geometry.validity must certify every chart determinant; got invalid or unresolved cells"
        )
    corners = _prepare_source_frame_corners(
        reference_mesh, geometry, cells, scope=scope, budget=budget
    )
    faces = _prepare_source_frame_faces(cells)
    source_id = cell_geometry_id(geometry)
    identity = canonical_fingerprint(
        {
            "kind": "source-integrable-hex-frame",
            "geometry": source_id,
            "topology": reference_mesh.topology_id,
            "connections": array_tree_fingerprint(faces.connections),
            "permutations": array_tree_fingerprint(faces.permutations),
        }
    )
    record_elapsed(record_phase, "source_frame_prepare", started)
    return frame_records_owner.PreparedSourceHexFrame(
        reference_mesh,
        geometry,
        cells,
        corners.frames,
        corners.reciprocal_frames,
        faces.connections,
        faces.permutations,
        faces.boundary,
        corners.varying_roots,
        source_id,
        identity,
    )


@checked
def extract_source_frame_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    grid: GridHexConstruction | frame_records_owner.SourceBlockIntegerGrid,
    regions: np.ndarray,
    limits: MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> MappedGridHexConstruction:
    """Restrict a compatible integer complex to the unchanged accepted source.

    Preparation is immutable: a refusal leaves both source and prior accepted
    grid untouched. Publication certification must still prove global embedding,
    whole-chart coverage, features, and declared quality bounds.
    """

    if cell_geometry_id(prepared.geometry) != prepared.source_id:
        raise failure(
            "prepared.source_id must match source_geometry; got changed coefficients"
        )
    if not np.array_equal(
        prepared.root_cells,
        np.concatenate(
            [
                np.asarray(block.vertices, dtype=np.int64)
                for block in prepared.reference_mesh.blocks
            ]
        ),
    ):
        raise failure(
            "prepared.root_cells must match reference_mesh; got changed source topology"
        )
    return realize_mapped_grid_hexes(
        grid,
        prepared.reference_mesh,
        prepared.geometry,
        regions,
        limits,
        scope=scope,
        budget=budget,
    )


@checked
def prepare_hex_frame_holonomy(
    source: cell_mesh_owner.CellMesh,
    connections: np.ndarray,
    transports: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> frame_records_owner.HexFrameHolonomy:
    """Decompose signed-permutation connections into a tree and cycle holonomy.

    Nonidentity cycle products retain singularity evidence. Every connection
    names a shared source-cell face; chart gluing is certified separately.
    """

    cells = entities(source, 3)
    edges = np.asarray(connections)
    rotations = np.asarray(transports)
    if edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"frame_connection.connections must be an integer matrix of source-cell pairs; got shape {edges.shape}, dtype {edges.dtype}"
        )
    if rotations.dtype.kind not in "iu" or rotations.shape != (edges.shape[0], 3, 3):
        raise ValueError(
            f"frame_connection.transports must be integer three-by-three actions per connection; got shape {rotations.shape}, dtype {rotations.dtype}"
        )
    identity = np.eye(3, dtype=np.int64)
    incidence = [set(row[row >= 0].tolist()) for row in cells]
    faces = {frozenset(row[row >= 0].tolist()) for row in entities(source, 2)}
    adjacency = [[] for _ in cells]
    seen = set()
    for row, ((left, right), action) in enumerate(
        zip(edges.tolist(), rotations, strict=True)
    ):
        if left == right or min(left, right) < 0 or max(left, right) >= len(cells):
            raise failure(
                f"frame_connection.connections must name distinct source-cell rows; got ({left}, {right})"
            )
        key = tuple(sorted((left, right)))
        if key in seen or frozenset(incidence[left] & incidence[right]) not in faces:
            raise failure(
                f"frame_connection.connections must name each shared source-cell face once; got duplicate or unshared pair ({left}, {right})"
            )
        seen.add(key)
        determinant = determinant_exact(
            tuple(tuple(Fraction(int(value)) for value in row) for row in action)
        )
        if (
            np.any(np.abs(action) > 1)
            or not np.array_equal(action.T @ action, identity)
            or determinant != 1
        ):
            raise failure(
                f"frame_connection.transports must be proper octahedral actions; got {action.tolist()}"
            )
        adjacency[left].append((right, row, action))
        adjacency[right].append((left, row, action.T))
    gauges = family_host_array((len(cells), 3, 3), np.int64, scope=scope)
    visited = np.zeros((len(cells),), dtype=np.bool_)
    tree = []
    for root in range(len(cells)):
        if visited[root]:
            continue
        visited[root] = True
        gauges[root] = identity
        queue = [root]
        position = 0
        while position < len(queue):
            left = queue[position]
            position += 1
            for right, row, action in adjacency[left]:
                if visited[right]:
                    continue
                gauges[right] = action @ gauges[left]
                visited[right] = True
                tree.append(row)
                queue.append(right)
    tree_set = set(tree)
    cycles = np.asarray(
        [row for row in range(len(edges)) if row not in tree_set], dtype=np.int64
    )
    products = np.asarray(
        [
            gauges[right].T @ rotations[row] @ gauges[left]
            for row in cycles
            for left, right in (edges[row],)
        ],
        dtype=np.int64,
    ).reshape((-1, 3, 3))
    singular = np.flatnonzero(np.any(products != identity, axis=(1, 2))).astype(np.int64)
    tree_array = np.asarray(tree, dtype=np.int64)
    identity_ = canonical_fingerprint(
        {
            "kind": "exact-source-frame-holonomy",
            "source": source.mesh_id,
            "connections": array_tree_fingerprint(edges),
            "transports": array_tree_fingerprint(rotations),
            "evidence": array_tree_fingerprint(
                (tree_array, cycles, products, singular, gauges)
            ),
        }
    )
    return frame_records_owner.HexFrameHolonomy(
        tree_array, cycles, products, singular, gauges, source.mesh_id, identity_
    )


class SolvedHexConnectionField(StrictModule):
    """Store the numeric field and immutable cycle-singularity preparation."""

    frames: jax.Array
    angles: jax.Array
    feature_residuals: jax.Array
    holonomy: frame_records_owner.HexFrameHolonomy
    optimization: MinimizationResult
    field_id: str = eqx.field(static=True)


def _connection_field_energy(
    angles: jax.Array,
    arguments: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    # All source/connection/constraint numeric state is dynamic optimizer args.

    initial, connections, transports, feature_rows, normals = arguments
    frames = initial @ jax.vmap(frame_rotation)(angles)
    left = frames[connections[:, 0]]
    right = frames[connections[:, 1]]
    relative = jnp.swapaxes(right, -1, -2) @ left @ jnp.swapaxes(transports, -1, -2)
    smooth = jnp.sum(jnp.square(relative - jnp.eye(3)))
    projected = contract("ni,nij->nj", normals, frames[feature_rows])
    alignment = jnp.sum(1.0 - jnp.sum(projected**4, axis=1))
    return smooth + 4.0 * alignment


@checked
def solve_hex_connection_field(
    source: cell_mesh_owner.CellMesh,
    initial_frames: np.ndarray,
    connections: np.ndarray,
    transports: np.ndarray,
    feature_cells: np.ndarray,
    feature_faces: np.ndarray,
    maximum_steps: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> SolvedHexConnectionField:
    """Solve feature alignment while retaining source-face and cycle evidence."""

    if (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps <= 0
    ):
        raise ValueError(
            f"frame_field.maximum_steps must be a positive integer; got {maximum_steps!r}"
        )
    holonomy = prepare_hex_frame_holonomy(source, connections, transports, scope=scope)
    cells = entities(source, 3)
    faces = entities(source, 2)
    initial = np.asarray(initial_frames, dtype=np.float64)
    if initial.shape != (len(cells), 3, 3) or not np.all(np.isfinite(initial)):
        raise ValueError(
            f"frame_field.initial_frames must be finite three-by-three matrices per source cell; got nonfinite values or shape {initial.shape}"
        )
    if not np.allclose(
        np.swapaxes(initial, -1, -2) @ initial, np.eye(3), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "frame_field.initial_frames must be orthonormal; got nonorthonormal frames"
        )
    if np.any(
        np.sum(np.cross(initial[:, 0], initial[:, 1]) * initial[:, 2], axis=1) <= 0.0
    ):
        raise ValueError(
            "frame_field.initial_frames must preserve orientation; got nonpositive determinants"
        )
    owners = np.asarray(feature_cells)
    selected = np.asarray(feature_faces)
    if (
        owners.dtype.kind not in "iu"
        or selected.dtype.kind not in "iu"
        or owners.ndim != 1
        or owners.shape != selected.shape
    ):
        raise ValueError(
            f"frame_field.feature_constraints must name integer source cell and face rows; got cell shape {owners.shape}, dtype {owners.dtype}, face shape {selected.shape}, dtype {selected.dtype}"
        )
    coordinates = np.asarray(jax.device_get(source.coordinates), dtype=np.float64)
    normals = family_host_array((len(owners), 3), np.float64, scope=scope)
    for row, (cell, face) in enumerate(
        zip(owners.tolist(), selected.tolist(), strict=True)
    ):
        if cell < 0 or cell >= len(cells) or face < 0 or face >= len(faces):
            raise failure(
                f"frame_field.feature_constraints must name available source entities; got cell {cell}, face {face}"
            )
        vertices = faces[face][faces[face] >= 0]
        if not set(vertices).issubset(set(cells[cell])):
            raise failure(
                f"frame_field.feature_faces must belong to their declared source cells; got face {face} outside cell {cell}"
            )
        points = coordinates[vertices]
        normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        length = np.linalg.norm(normal)
        if length == 0.0:
            raise failure(
                f"frame_field.feature_faces must have nonzero source normals; got zero normal at face {face}"
            )
        normals[row] = normal / length
    method = ProjectedLBFGS()
    with field_work_allowance(
        method, len(cells) + len(connections) + len(owners), maximum_steps, scope=scope
    ) as field_ticket:
        arguments = (
            jnp.asarray(initial),
            jnp.asarray(connections),
            jnp.asarray(transports),
            jnp.asarray(owners),
            jnp.asarray(normals),
        )
        angles = jnp.zeros((len(cells), 3), dtype=jnp.float64)
        result = minimize(
            _connection_field_energy,
            angles,
            method=method,
            args=arguments,
            bounds=Bounds(jnp.full_like(angles, -jnp.pi), jnp.full_like(angles, jnp.pi)),
            termination=OptimizationTermination(maximum_steps=maximum_steps),
        )
        charge_field_work(
            field_ticket, result, len(cells) + len(connections) + len(owners)
        )

    frames = jnp.asarray(initial) @ jax.vmap(frame_rotation)(result.parameters)
    projected = contract("ni,nij->nj", jnp.asarray(normals), frames[jnp.asarray(owners)])
    residuals = 1.0 - jnp.max(jnp.abs(projected), axis=1)
    identifier = canonical_fingerprint(
        {
            "kind": "source-connection-frame-solve",
            "connection": holonomy.connection_id,
            "features": array_tree_fingerprint((owners, selected)),
            "initial": array_tree_fingerprint(initial),
            "accepted_angles": array_tree_fingerprint(result.parameters),
            "maximum_steps": maximum_steps,
        }
    )
    return SolvedHexConnectionField(
        frames, result.parameters, residuals, holonomy, result, identifier
    )


def _prepare_source_grid_counts(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    workspace: NativeHostStorageWorkspace,
    *,
    scope: NativeExecutionScope,
) -> frame_records_owner.SourceIntegerGridCounts:
    roots = prepared.root_cells
    if len(roots) * 8192 > limits.maximum_scratch_bytes:
        raise failure(
            "source_grid.parity_scratch must respect limits.maximum_scratch_bytes; got oversized preparation",
            resource=True,
        )
    workspace.set_bound(len(roots) * 8192)
    counts = np.asarray(requested_intervals)
    material = np.asarray(regions)
    if (
        counts.dtype.kind not in "iu"
        or counts.shape != (len(roots), 3)
        or np.any(counts <= 0)
    ):
        raise ValueError(
            f"source_grid.requested_intervals must contain positive integer counts per source root and axis; got shape {counts.shape}, dtype {counts.dtype}, nonpositive or incompatible counts"
        )
    if material.dtype.kind not in "iu" or material.shape != (len(roots),):
        raise ValueError(
            f"source_grid.regions must contain one integer ID per source root; got shape {material.shape}, dtype {material.dtype}"
        )
    if (
        not isinstance(maximum_depth, int)
        or isinstance(maximum_depth, bool)
        or not 0 < maximum_depth <= 20
    ):
        raise ValueError(
            f"source_grid.maximum_depth must lie in 1..20; got {maximum_depth!r}"
        )
    if np.any(counts > (1 << maximum_depth)):
        raise failure(
            f"source_grid.requested_intervals must respect depth {maximum_depth}; got oversized counts",
            resource=True,
        )
    requested = counts
    counts = family_host_array(requested.shape, np.int64, scope=scope)
    counts[:] = requested
    source_ids, parity_work = _propagate_source_grid_counts(
        prepared, counts, limits, scope=scope
    )
    if np.any(counts > (1 << maximum_depth)):
        raise failure(
            "source_grid.counts must respect maximum_depth; got oversized compatible counts",
            resource=True,
        )
    total_cells = sum(prod(row) for row in counts.tolist())
    upper_vertices = sum(prod(value + 1 for value in row) for row in counts.tolist())
    scratch_bound = upper_vertices * 4096 + total_cells * 128 + len(roots) * 8192
    check_family_limits(
        limits,
        total_cells,
        upper_vertices,
        8 * total_cells,
        scratch_bound,
        total_cells * 128 + upper_vertices * 33 + parity_work,
    )
    data_bound = upper_vertices * 24 + total_cells * 488 + counts.nbytes
    if data_bound > limits.maximum_data_bytes:
        raise failure(
            "source_grid.data_bytes must respect limits.maximum_data_bytes; got oversized admitted banks",
            resource=True,
        )
    workspace.set_bound(scratch_bound)
    return frame_records_owner.SourceIntegerGridCounts(
        counts, material, source_ids, total_cells, upper_vertices, parity_work
    )


def _propagate_source_grid_counts(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    counts: np.ndarray,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, int]:
    """Take the finite maximum over source-edge root-axis equivalence classes."""
    roots = prepared.root_cells
    parent = list(range(len(roots) * 3))
    rank = [0] * len(parent)
    parity_work = 0
    parity_bound = len(roots) * (30 * (1 + 3 * max(1, len(parent).bit_length())) + 64)
    if parity_bound > limits.maximum_work_units:
        raise failure(
            "source_grid.parity_work must respect limits.maximum_work_units; got oversized preparation",
            resource=True,
        )
    parity_ticket = scope.batch(parity_bound)
    with parity_ticket:

        def find(row: int) -> int:
            nonlocal parity_work
            parity_ticket.consume(1)
            parity_work += 1
            while parent[row] != row:
                parent[row] = parent[parent[row]]
                row = parent[row]
                parity_ticket.consume(3)
                parity_work += 3
            return row

        edge_owner: dict[tuple[int, ...], int] = {}
        source_ids = np.asarray(
            jax.device_get(prepared.reference_mesh.vertex_global_ids), dtype=np.int64
        )
        for root, cell in enumerate(roots):
            for axis, edges in enumerate(_EDGE_AXES):
                row = 3 * root + axis
                for left, right in edges:
                    key = tuple(
                        sorted(
                            (int(source_ids[cell[left]]), int(source_ids[cell[right]]))
                        )
                    )
                    previous = edge_owner.setdefault(key, row)
                    a, b = find(row), find(previous)
                    parity_ticket.consume(4)
                    parity_work += 4
                    if a != b:
                        if rank[a] < rank[b]:
                            a, b = b, a
                        parent[b] = a
                        if rank[a] == rank[b]:
                            rank[a] += 1
        maxima: dict[int, int] = {}
        for row, count in enumerate(counts.reshape(-1).tolist()):
            representative = find(row)
            maxima[representative] = max(maxima.get(representative, 0), count)
        for row in range(counts.size):
            counts.reshape(-1)[row] = maxima[find(row)]
    return source_ids, parity_work


def _prepare_source_root_nodes(
    cell: np.ndarray,
    intervals: list[int],
    coordinates: np.ndarray,
    source_ids: np.ndarray,
    points: np.ndarray,
    node_bank: dict[tuple[tuple[int, Fraction], ...], int],
    node_count: int,
    node_ticket: NativeBatchTicket,
    *,
    scope: NativeExecutionScope,
) -> frame_records_owner.SourceIntegerRootNodes:
    shape = tuple(value + 1 for value in intervals)
    local_nodes = family_host_array((prod(shape),), np.int64, scope=scope)
    matrix, origin = affine_reference_root_frame(coordinates[cell], "hexahedron")
    for local_node, integer in enumerate(product(*(range(value) for value in shape))):
        uvw = tuple(Fraction(integer[axis], intervals[axis]) for axis in range(3))
        support = []
        for vertex, corner in zip(cell.tolist(), _POINTS, strict=True):
            weight = prod(
                (uvw[axis] if corner[axis] else 1 - uvw[axis] for axis in range(3)),
                start=Fraction(1),
            )
            if weight:
                support.append((int(source_ids[vertex]), weight))
        key = tuple(sorted(support))
        exact = tuple(
            origin[axis]
            + sum(
                (matrix[axis][column] * uvw[column] for column in range(3)), Fraction(0)
            )
            for axis in range(3)
        )
        node_ticket.consume(33)
        point = tuple(float(value) for value in exact)
        node = node_bank.get(key)
        if node is None:
            node = node_count
            node_count += 1
            node_bank[key] = node
            points[node] = point
        elif not np.array_equal(points[node], point):
            raise failure(
                "source_grid.shared_nodes must have equal source restrictions; got conflicting chart points"
            )
        local_nodes[local_node] = node
    return frame_records_owner.SourceIntegerRootNodes(local_nodes, node_count)


def _prepare_source_integer_banks(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    admission: frame_records_owner.SourceIntegerGridCounts,
    node_ticket: NativeBatchTicket,
    *,
    scope: NativeExecutionScope,
) -> frame_records_owner.SourceIntegerGridBanks:
    with node_ticket:
        coordinates = np.asarray(
            jax.device_get(prepared.reference_mesh.coordinates), dtype=np.float64
        )
        node_bank: dict[tuple[tuple[int, Fraction], ...], int] = {}
        points = family_host_array((admission.upper_vertices, 3), np.float64, scope=scope)
        cells = family_host_array((admission.total_cells, 8), np.int64, scope=scope)
        parent_rows = family_host_array((admission.total_cells,), np.int64, scope=scope)
        lower = family_host_array((admission.total_cells, 3), np.int64, scope=scope)
        chart = reference_owner.lagrange_element("hexahedron", 1)
        chart_numerators = family_host_array(
            (admission.total_cells, chart.local_dof_count, 3), np.int64, scope=scope
        )
        chart_denominators = family_host_array(
            chart_numerators.shape, np.int64, scope=scope
        )
        node_count = cell_count = 0
        for root, (cell, intervals) in enumerate(
            zip(prepared.root_cells, admission.counts.tolist(), strict=True)
        ):
            shape = tuple(value + 1 for value in intervals)
            root_nodes = _prepare_source_root_nodes(
                cell,
                intervals,
                coordinates,
                admission.source_ids,
                points,
                node_bank,
                node_count,
                node_ticket,
                scope=scope,
            )
            local_nodes, node_count = root_nodes.nodes, root_nodes.node_count
            local_cells = logical_cell_bank(shape, scope=scope)
            stop = cell_count + len(local_cells)
            np.take(local_nodes, local_cells, out=cells[cell_count:stop])
            parent_rows[cell_count:stop] = root
            for row, integer in enumerate(
                product(*(range(value) for value in intervals))
            ):
                lower[cell_count + row] = integer
                for corner, dofs in enumerate(chart.entity_dofs[0]):
                    chart_numerators[cell_count + row, dofs[0]] = tuple(
                        integer[axis] + _INTEGER_POINTS[corner][axis] for axis in range(3)
                    )
                    chart_denominators[cell_count + row, dofs[0]] = intervals
                node_ticket.consume(24)
            cell_count = stop
    return frame_records_owner.SourceIntegerGridBanks(
        points,
        cells,
        parent_rows,
        lower,
        chart_numerators,
        chart_denominators,
        node_count,
    )


def _assemble_source_integer_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    source_connection: frame_records_owner.SourceBlockConnection,
    admission: frame_records_owner.SourceIntegerGridCounts,
    banks: frame_records_owner.SourceIntegerGridBanks,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.SourceBlockIntegerGrid:
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = np.asarray(_HEX_TOPOLOGY.entities[dimension], dtype=np.int64)
        entity_rows = np.sort(
            banks.cells[:, local_entities].reshape((-1, local_entities.shape[1])), axis=1
        )
        if np.unique(entity_rows, axis=0).shape[0] > bound:
            raise failure(
                f"source_grid.entities must respect dimension {dimension} capacity {bound}; got oversized topology",
                resource=True,
            )
    mesh = cell_mesh_owner.CellMesh(
        banks.points[: banks.node_count],
        (
            reference_cell_types_owner.CellBlock(
                "source_integer_blocks", "hexahedron", banks.cells
            ),
        ),
        numeric_version=prepared.reference_mesh.numeric_version,
    )
    with coordinate_phase(scope=scope, budget=budget):
        validity = certify_cell_geometry_validity(mesh, budget=budget)
    if validity.invalid_count or validity.unresolved_count:
        raise failure(
            "source_grid.validity must certify every reference map; got invalid or unresolved cells"
        )
    identifier = canonical_fingerprint(
        {
            "kind": "source-block-integer-extraction",
            "frame": prepared.frame_id,
            "connection": source_connection.holonomy.connection_id,
            "offsets": array_tree_fingerprint(source_connection.offsets),
            "intervals": array_tree_fingerprint(admission.counts),
            "regions": array_tree_fingerprint(admission.material),
            "charts": array_tree_fingerprint(
                (banks.chart_numerators, banks.chart_denominators)
            ),
            "mesh": mesh.mesh_id,
        }
    )
    cell_regions = family_host_array((admission.total_cells,), np.int64, scope=scope)
    np.take(admission.material, banks.parent_rows, out=cell_regions)
    retained_bytes = sum(
        array.nbytes
        for array in (
            banks.points,
            banks.cells,
            cell_regions,
            banks.parent_rows,
            admission.counts,
            banks.lower,
            banks.chart_numerators,
            banks.chart_denominators,
        )
    )
    if retained_bytes > limits.maximum_data_bytes:
        raise failure(
            "source_grid.retained_bytes must respect limits.maximum_data_bytes; got oversized banks",
            resource=True,
        )
    return frame_records_owner.SourceBlockIntegerGrid(
        mesh,
        cell_regions,
        banks.parent_rows,
        admission.counts,
        banks.lower,
        banks.chart_numerators,
        banks.chart_denominators,
        prepared,
        source_connection,
        identifier,
    )


def _extract_source_block_integer_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    workspace: NativeHostStorageWorkspace,
    /,
    *,
    source_connection: frame_records_owner.SourceBlockConnection,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.SourceBlockIntegerGrid:
    """Extract source-supported integer maps with compatible root-axis counts."""
    started = phase_started(record_phase)
    if cell_geometry_id(prepared.geometry) != prepared.source_id:
        raise failure(
            "prepared.source_id must match source_geometry; got changed source coefficients"
        )
    _validate_source_block_connection(prepared, source_connection)
    if not np.array_equal(
        prepared.root_cells,
        np.concatenate(
            [
                np.asarray(jax.device_get(block.vertices), dtype=np.int64)
                for block in prepared.reference_mesh.blocks
            ]
        ),
    ):
        raise failure(
            "prepared.root_cells must match reference_mesh; got changed root identities"
        )
    admission = _prepare_source_grid_counts(
        prepared,
        requested_intervals,
        regions,
        limits,
        maximum_depth,
        workspace,
        scope=scope,
    )
    node_ticket = scope.batch(admission.total_cells * 24 + admission.upper_vertices * 33)
    record_elapsed(record_phase, "block_count_compatibility", started)
    started = phase_started(record_phase)
    banks = _prepare_source_integer_banks(prepared, admission, node_ticket, scope=scope)
    result = _assemble_source_integer_grid(
        prepared, source_connection, admission, banks, limits, scope=scope, budget=budget
    )
    record_elapsed(record_phase, "source_block_integer_extraction", started)
    return result


@checked
def extract_source_block_integer_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    /,
    *,
    source_connection: frame_records_owner.SourceBlockConnection | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.SourceBlockIntegerGrid:
    """Prepare the support-indexed workspace for one source-block extraction."""
    if source_connection is None:
        source_connection = prepare_source_block_connection(
            prepared, record_phase=record_phase, scope=scope
        )
    with scope.host_workspace() as workspace:
        return _extract_source_block_integer_grid(
            prepared,
            requested_intervals,
            regions,
            limits,
            maximum_depth,
            workspace,
            source_connection=source_connection,
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )


@checked
def prepare_source_block_connection(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> frame_records_owner.SourceBlockConnection:
    """Derive chart-axis transports from retained source-face identities.

    Tangential actions are determined by source face corners. The transverse
    sign is the positive continuation across opposite boundary orientations.
    These are topological tensor-chart actions, not fitted physical rotations.
    """
    started = phase_started(record_phase)
    transports = family_host_array(
        (len(prepared.face_connections), 3, 3), np.int64, scope=scope
    )
    offsets = family_host_array(
        (len(prepared.face_connections), 3), np.int64, scope=scope
    )
    corners = np.asarray(_POINTS, dtype=np.int64)
    for row, ((left, left_face, right, right_face), permutation) in enumerate(
        zip(prepared.face_connections.tolist(), prepared.face_permutations, strict=True)
    ):
        left_points = corners[np.asarray(_FACES[left_face])]
        right_points = corners[np.asarray(_FACES[right_face])][permutation]
        action = np.zeros((3, 3), dtype=np.int64)
        left_normal = int(
            np.flatnonzero(np.all(left_points == left_points[:1], axis=0))[0]
        )
        right_normal = int(
            np.flatnonzero(np.all(right_points == right_points[:1], axis=0))[0]
        )
        for axis in range(3):
            if axis == left_normal:
                continue
            for corner in range(1, 4):
                difference = left_points[corner] - left_points[0]
                if np.count_nonzero(difference) == 1 and difference[axis]:
                    action[:, axis] = (
                        right_points[corner] - right_points[0]
                    ) * difference[axis]
                    break
            else:
                raise failure(
                    "source_connection.tangent_axes must define a tensor-axis transition; got unavailable directions"
                )
        action[right_normal, left_normal] = (
            1 if left_points[0, left_normal] != right_points[0, right_normal] else -1
        )
        transports[row] = action
        offsets[row] = right_points[0] - action @ left_points[0]
        if not np.array_equal(left_points @ action.T + offsets[row], right_points):
            raise failure(
                "source_connection.face_map must match the integer affine corner action; got incompatible traces"
            )
    connections = prepared.face_connections[:, (0, 2)]
    holonomy = prepare_hex_frame_holonomy(
        prepared.reference_mesh, connections, transports, scope=scope
    )
    record_elapsed(record_phase, "frame_graph_decomposition", started)
    return frame_records_owner.SourceBlockConnection(
        transports, offsets, holonomy, prepared.frame_id
    )


def _validate_source_block_connection(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    connection: frame_records_owner.SourceBlockConnection,
    /,
) -> None:
    """Authenticate reused state without re-solving its prepared graph."""
    if (
        connection.frame_id != prepared.frame_id
        or connection.holonomy.source_id != prepared.reference_mesh.mesh_id
    ):
        raise failure(
            "source_connection.frame_id must match the prepared chart and mesh; got another source binding"
        )
    rows = len(prepared.face_connections)
    if connection.transports.shape != (rows, 3, 3) or connection.offsets.shape != (
        rows,
        3,
    ):
        raise failure(
            "source_connection.banks must contain one axis action per prepared face; got incompatible shapes"
        )
    expected = canonical_fingerprint(
        {
            "kind": "exact-source-frame-holonomy",
            "source": prepared.reference_mesh.mesh_id,
            "connections": array_tree_fingerprint(prepared.face_connections[:, (0, 2)]),
            "transports": array_tree_fingerprint(connection.transports),
            "evidence": array_tree_fingerprint(
                (
                    connection.holonomy.spanning_edges,
                    connection.holonomy.cycle_edges,
                    connection.holonomy.cycle_holonomy,
                    connection.holonomy.singular_cycles,
                    connection.holonomy.root_gauges,
                )
            ),
        }
    )
    if expected != connection.holonomy.connection_id:
        raise failure(
            "source_connection.holonomy must match the retained transport bank; got changed actions"
        )
    corners = np.asarray(_POINTS, dtype=np.int64)
    for row, (_, left_face, _, right_face) in enumerate(prepared.face_connections):
        left = corners[np.asarray(_FACES[left_face])]
        right = corners[np.asarray(_FACES[right_face])][prepared.face_permutations[row]]
        if not np.array_equal(
            left @ connection.transports[row].T + connection.offsets[row], right
        ):
            raise failure(
                "source_connection.offsets must preserve the face trace; got incompatible integer offsets"
            )
