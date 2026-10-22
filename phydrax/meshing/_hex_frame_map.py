"""Prepare source-chart varying frames and integer-map compatibility.

Frames are derivatives of retained coordinate expressions. Extraction restricts
those expressions and their source geometry identity to compatible integer cells.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import prod
from struct import pack, pack_into

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization.fem._reference as reference_owner
import phydrax.meshing._hex_frame_records as frame_records_owner
from phydrax.meshing._host_banks import entities, failure, family_host_array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from .._host_storage import declared_integer_storage_bytes
from .._meshcore import (
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from .._strict import StrictModule
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    _ConstructedTensorGrid,
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    coordinate_polynomials,
    derivative,
    evaluate,
    Polynomial,
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
    _certify_source_grid_restrictions,
    _HOST_ENTRY_BYTES,
    _realize_mapped_grid_hexes,
    _tensor_grid_charts,
    certify_source_grid_restrictions,
    frame_rotation,
    GridHexConstruction,
    MappedGridHexConstruction,
    NativeHexGridSchedule,
    realize_mapped_grid_hexes,
    source_block_interval_counts,
)
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from ._reference_root_composition import affine_reference_root_frame
from ._structured import _fill_tensor_cells, logical_cell_bank


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


def _fill_source_corner_frames(
    polynomials: tuple[Polynomial, ...],
    frames: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Write admitted numerical frames, then retire this root's exact jet scratch."""
    with budget.temporary_scope():
        jets = tuple(
            tuple(derivative(value, axis, budget=budget) for axis in range(3))
            for value in polynomials
        )
        varying = any(any(any(power) for power in jet) for row in jets for jet in row)
        for corner, point in enumerate(_POINTS):
            frames[corner] = [
                [float(evaluate(jet, point, budget=budget)) for jet in row]
                for row in jets
            ]
        return varying


def _prepare_source_frame_corners(
    reference_mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    cells: np.ndarray,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> frame_records_owner.SourceFrameCornerData:
    with coordinate_phase(scope=scope, budget=budget), budget.temporary_scope():
        elements, routes, _ = geometry.resolve(
            reference_mesh, budget=budget, exact_source_prepared=False
        )
        roots = cells.shape[0]
        # Host route copies and one varying flag per root are temporary.
        budget.reserve(0, 16 * sum(route.size for route in routes) + 8 * roots + 4096)
        # Eight corner evaluations of the nine derivative jets per root.
        count_geometry_queries(roots * 8, work_units=roots * 72, budget=budget)
        values = geometry.source_coordinates(budget=budget)
        host_routes = jax.device_get(routes)
        frames = family_host_array((roots, 8, 3, 3), np.float64, scope=scope)
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
                varying.append(
                    _fill_source_corner_frames(polynomials, frames[root], budget=budget)
                )
                root += 1
        varying_roots = family_host_array((roots,), np.bool_, scope=scope)
        varying_roots[:] = varying
    # The host status copy of the native reciprocal 3x3 solves is temporary.
    with scope.host_workspace() as workspace:
        workspace.set_bound(8 * roots + 4096)
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
        del successful
    return frame_records_owner.SourceFrameCornerData(
        frames, solve.value, varying_roots
    )


def _prepare_source_frame_faces(
    cells: np.ndarray, *, scope: NativeExecutionScope
) -> frame_records_owner.SourceFrameFaceData:
    occurrences_count = len(_FACES) * cells.shape[0]
    with scope.host_workspace() as workspace:
        # Face-key map entries with their occurrence tuples, connection,
        # permutation, and boundary lists (conservative CPython entries), and
        # the integer banks built from them.
        workspace.set_bound(
            (3 * _HOST_ENTRY_BYTES + 64) * occurrences_count + 4096
        )
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
        del bank
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
    root_count = sum(block.cell_count for block in reference_mesh.blocks)
    with scope.host_workspace() as workspace:
        # Host connectivity copies and their concatenation (128 bytes per
        # root) and the host point copy.
        workspace.set_bound(
            128 * root_count + 24 * reference_mesh.coordinates.shape[0] + 4096
        )
        cells = np.concatenate(
            [
                np.asarray(jax.device_get(block.vertices), dtype=np.int64)
                for block in reference_mesh.blocks
            ]
        )
        points = np.asarray(
            jax.device_get(reference_mesh.coordinates), dtype=np.float64
        )
        for cell in cells:
            affine_reference_root_frame(points[cell], "hexahedron")
        del points
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
    faces = _prepare_source_frame_faces(cells, scope=scope)
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

    cell_count = sum(block.cell_count for block in source.blocks)
    face_count = source.topology.entities(2).count
    connection_count = len(connections)
    with scope.host_workspace() as workspace:
        # Conservative CPython container bounds: incidence sets, queue, tree
        # and tree-set entries per cell; face keys per face; adjacency, seen,
        # and product entries per connection; plus their integer banks.
        workspace.set_bound(
            _HOST_ENTRY_BYTES
            * (6 * cell_count + 3 * face_count + 4 * connection_count)
            + 4096
        )
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
        del incidence, faces, seen
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
        del adjacency, visited
        tree_set = set(tree)
        cycles = np.asarray(
            [row for row in range(len(edges)) if row not in tree_set], dtype=np.int64
        )
        del tree_set
        products = np.asarray(
            [
                gauges[right].T @ rotations[row] @ gauges[left]
                for row in cycles
                for left, right in (edges[row],)
            ],
            dtype=np.int64,
        ).reshape((-1, 3, 3))
        singular = np.flatnonzero(np.any(products != identity, axis=(1, 2))).astype(
            np.int64
        )
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


def _source_tensor_storage_bounds(counts: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
    """Bound external construction owners separately from managed array payloads.

    Only boundary nodes enter ``shared``: corners have one support, edge
    interiors two, and face interiors four. Interior nodes have no Python hash
    entry. Native array metadata/alignment remain charged by their canonical
    allocator, not represented a second time as external Python payload.
    """
    cells = vertices = largest_nodes = largest_axes = 0
    external = 256 + 8192 * len(counts)
    for row in counts:
        nx, ny, nz = (int(n) for n in row)
        nodes = (nx + 1) * (ny + 1) * (nz + 1)
        cells += nx * ny * nz
        vertices += nodes
        largest_nodes = max(largest_nodes, nodes)
        largest_axes = max(largest_axes, nx + ny + nz + 3)
    node_bytes = declared_integer_storage_bytes(vertices.bit_length())
    source_id_bytes = declared_integer_storage_bytes(64)
    for row in counts:
        nx, ny, nz = (int(n) for n in row)
        integer_bytes = declared_integer_storage_bytes((nx * ny * nz).bit_length())
        fraction_bytes = 256 + 2 * integer_bytes
        support_bytes = 256 + 2 * 16 + source_id_bytes + fraction_bytes
        strata = (
            (8, 1),
            (4 * (nx + ny + nz - 3), 2),
            (2 * ((nx - 1) * (ny - 1) + (ny - 1) * (nz - 1)
                  + (nz - 1) * (nx - 1)), 4),
        )
        for entries, supports in strata:
            external += entries * (
                64 + node_bytes + 256 + supports * (16 + support_bytes)
            )
    # Finite binary64 affine root coordinates and depth <=20 have exact
    # numerators/denominators below 4096 bits after axis restriction. Keep one
    # root's separable axes, plus its affine frame and scalar construction
    # temporaries. The root loop explicitly retires these before its successor.
    scalar_bytes = 256 + 2 * declared_integer_storage_bytes(4096)
    external += largest_axes * (256 + 3 * (16 + scalar_bytes))
    external += 64 * scalar_bytes
    managed = (
        vertices * 3 * np.dtype(np.float64).itemsize
        + (cells * 9 + largest_nodes) * np.dtype(np.int64).itemsize
    )
    return external, managed


def _prepare_source_grid_counts(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray | None,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    workspace: NativeHostStorageWorkspace,
    *,
    scope: NativeExecutionScope,
    axis_bounds: np.ndarray | None = None,
    desired_axis_bounds: np.ndarray | None = None,
    physical_lipschitz_bound: float = 1.0,
    target_size: float | None = None,
    maximum_size: float | None = None,
    compact: bool = False,
) -> frame_records_owner.SourceIntegerGridCounts:
    roots = prepared.root_cells
    with scope.batch(24 * len(roots) + 1) as metadata_ticket:
        metadata_ticket.consume(24 * len(roots) + 1)
    if len(roots) * 8192 > limits.maximum_scratch_bytes:
        raise failure(
            "source_grid.parity_scratch must respect limits.maximum_scratch_bytes; got oversized preparation",
            resource=True,
        )
    workspace.set_bound(len(roots) * 8192)
    if requested_intervals is None:
        if axis_bounds is None or target_size is None or maximum_size is None:
            raise ValueError("source_grid.aimed_counts must supply directional bounds and the authored size band")
        counts = family_host_array((len(roots), 3), np.int64, scope=scope)
        counts.fill(1)
    else:
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
    lengths = None
    if axis_bounds is not None:
        if axis_bounds.shape != (len(roots), 3) or not np.all(np.isfinite(axis_bounds)) or np.any(axis_bounds <= 0):
            raise ValueError("source_grid.axis_bounds must contain positive finite bounds per root and local axis")
        lengths = family_host_array(axis_bounds.shape, np.float64, scope=scope)
        lengths[:] = axis_bounds
    desired_lengths = None
    if desired_axis_bounds is not None:
        if desired_axis_bounds.shape != (len(roots), 3) or not np.all(np.isfinite(desired_axis_bounds)) or np.any(desired_axis_bounds <= 0):
            raise ValueError("source_grid.desired_axis_bounds must contain positive finite reference-edge bounds")
        desired_lengths = family_host_array(desired_axis_bounds.shape, np.float64, scope=scope)
        desired_lengths[:] = desired_axis_bounds
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
    source_ids, parity_work, axis_classes = _propagate_source_grid_counts(
        prepared, counts, limits, scope=scope, axis_bounds=lengths,
        desired_axis_bounds=desired_lengths,
    )
    if requested_intervals is None:
        assert lengths is not None and target_size is not None and maximum_size is not None
        with scope.batch(2 * counts.size + 1) as metadata_ticket:
            metadata_ticket.consume(2 * counts.size + 1)
            grouped: dict[int, list[int]] = {}
            for row, representative in enumerate(axis_classes):
                grouped.setdefault(representative, []).append(row)
            compatible_groups = tuple(tuple(group) for group in grouped.values())

        def admitted(flat: tuple[int, ...]) -> bool:
            with scope.batch(24 * len(roots) + 1) as ticket:
                ticket.consume(24 * len(roots) + 1)
                rows = tuple(flat[row : row + 3] for row in range(0, len(flat), 3))
                cells = sum(prod(row) for row in rows)
                vertices = sum(prod(value + 1 for value in row) for row in rows)
                scratch = (
                    sum(_source_tensor_storage_bounds(rows))
                    if compact else vertices * 4096 + cells * 128 + len(roots) * 8192
                )
                work = cells * 128 + vertices * 33
                remaining = scope.remaining()
                return (
                    max(flat) <= 1 << maximum_depth
                    and cells <= limits.maximum_cells
                    and vertices <= limits.maximum_vertices
                    and 8 * cells <= limits.maximum_connectivity_entries
                    and scratch <= limits.maximum_scratch_bytes
                    and scratch <= remaining.remaining_managed_bytes
                    and work <= remaining.remaining_work_units
                    and cells * 128 + vertices * 33 + parity_work <= limits.maximum_work_units
                    and vertices * 24 + cells * 488 + counts.nbytes <= limits.maximum_data_bytes
                )

        selected = source_block_interval_counts(
            tuple(Fraction(float(value)) for value in lengths.flat),
            target_size, maximum_size, admitted, scope=scope,
            compatible_groups=compatible_groups,
            # Both the root-direction derivative enclosure and the global
            # Lipschitz bound times its reference edge enclose this physical
            # edge. Use their intersection, not the globally inflated grid.
            # Face-axis classes have already propagated both maxima.
            desired_lengths=None if desired_lengths is None else tuple(
                min(
                    Fraction(float(local)),
                    Fraction(float(reference)) * Fraction(physical_lipschitz_bound),
                )
                for local, reference in zip(lengths.flat, desired_lengths.flat, strict=True)
            ),
        )
        counts.flat = selected
    if np.any(counts > (1 << maximum_depth)):
        raise failure(
            "source_grid.counts must respect maximum_depth; got oversized compatible counts",
            resource=True,
        )
    with scope.batch(24 * len(roots) + 1) as metadata_ticket:
        metadata_ticket.consume(24 * len(roots) + 1)
    total_cells = sum(prod(row) for row in counts.tolist())
    upper_vertices = sum(prod(value + 1 for value in row) for row in counts.tolist())
    external_bound, managed_bound = (
        _source_tensor_storage_bounds(tuple(tuple(int(n) for n in row) for row in counts))
        if compact else (upper_vertices * 4096 + total_cells * 128 + len(roots) * 8192, 0)
    )
    scratch_bound = external_bound + managed_bound
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
    workspace.set_bound(external_bound)
    return frame_records_owner.SourceIntegerGridCounts(
        counts, material, source_ids, total_cells, upper_vertices, parity_work
    )


def _propagate_source_grid_counts(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    counts: np.ndarray,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    axis_bounds: np.ndarray | None = None,
    desired_axis_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, int, tuple[int, ...]]:
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
        length_maxima: dict[int, float] = {}
        desired_maxima: dict[int, float] = {}
        for row, count in enumerate(counts.reshape(-1).tolist()):
            representative = find(row)
            maxima[representative] = max(maxima.get(representative, 0), count)
            if axis_bounds is not None:
                parity_ticket.consume(1)
                parity_work += 1
                length_maxima[representative] = max(
                    length_maxima.get(representative, 0.0), float(axis_bounds.flat[row])
                )
            if desired_axis_bounds is not None:
                parity_ticket.consume(1)
                parity_work += 1
                desired_maxima[representative] = max(
                    desired_maxima.get(representative, 0.0), float(desired_axis_bounds.flat[row])
                )
        axis_classes = []
        for row in range(counts.size):
            representative = find(row)
            counts.flat[row] = maxima[representative]
            parity_ticket.consume(2)
            parity_work += 2
            axis_classes.append(representative)
            if axis_bounds is not None:
                parity_ticket.consume(1)
                parity_work += 1
                axis_bounds.flat[row] = length_maxima[representative]
            if desired_axis_bounds is not None:
                parity_ticket.consume(1)
                parity_work += 1
                desired_axis_bounds.flat[row] = desired_maxima[representative]
    return source_ids, parity_work, tuple(axis_classes)


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


def _prepare_source_tensor_banks(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    admission: frame_records_owner.SourceIntegerGridCounts,
    *,
    scope: NativeExecutionScope,
) -> frame_records_owner._SourceTensorGridBanks:
    """Construct roots in public node order from separable exact axis endpoints."""
    coordinates = np.asarray(prepared.reference_mesh.coordinates, dtype=np.float64)
    points = family_host_array((admission.upper_vertices, 3), np.float64, scope=scope)
    cells = family_host_array((admission.total_cells, 8), np.int64, scope=scope)
    parents = family_host_array((admission.total_cells,), np.int64, scope=scope)
    shared: dict[tuple[tuple[int, Fraction], ...], int] = {}
    node_count = cell_count = 0
    corner_rows = {corner: row for row, corner in enumerate(_INTEGER_POINTS)}
    for root, (cell, counts_) in enumerate(zip(prepared.root_cells, admission.counts, strict=True)):
        counts = tuple(int(n) for n in counts_)
        shape = tuple(n + 1 for n in counts)
        denominator = prod(counts)
        matrix, origin = affine_reference_root_frame(coordinates[cell], "hexahedron")
        axis_work = 9 * sum(shape)
        with scope.batch(axis_work) as ticket:
            ticket.consume(axis_work)
            terms = tuple(
                tuple(
                    tuple(matrix[row][axis] * Fraction(i, counts[axis]) for row in range(3))
                    for i in range(shape[axis])
                )
                for axis in range(3)
            )
        local = family_host_array(shape, np.int64, scope=scope)
        # Interior points have root-local support. Only boundary support can
        # identify a node of another conforming source root.
        with scope.batch(25 * prod(shape)) as ticket:
            for integer in product(*(range(n) for n in shape)):
                ticket.consume(9)
                point = tuple(float(
                    origin[row] + terms[0][integer[0]][row]
                    + terms[1][integer[1]][row] + terms[2][integer[2]][row]
                ) for row in range(3))
                key = None
                if any(integer[axis] in (0, counts[axis]) for axis in range(3)):
                    support = []
                    choices = tuple(
                        (0,) if i == 0 else (1,) if i == n else (0, 1)
                        for i, n in zip(integer, counts, strict=True)
                    )
                    for corner in product(*choices):
                        vertex = cell[corner_rows[corner]]
                        weight = prod(
                            integer[axis] if corner[axis] else counts[axis] - integer[axis]
                            for axis in range(3)
                        )
                        support.append((int(admission.source_ids[vertex]), Fraction(weight, denominator)))
                    ticket.consume(4 * len(support))
                    key = tuple(sorted(support))
                node = None if key is None else shared.get(key)
                if node is None:
                    node = node_count
                    node_count += 1
                    points[node] = point
                    if key is not None:
                        shared[key] = node
                elif not np.array_equal(points[node], point):
                    raise failure("source_grid.shared_nodes must have equal source restrictions; got conflicting chart points")
                local[integer] = node
        stop = cell_count + prod(counts)
        _fill_tensor_cells(local, cells[cell_count:stop], scope=scope)
        parents[cell_count:stop] = root
        cell_count = stop
        del local, terms
    return frame_records_owner._SourceTensorGridBanks(points, cells, parents, node_count)


def _tensor_chart_fingerprint(
    counts: np.ndarray, *, scope: NativeExecutionScope
) -> dict[str, object]:
    """Hash the public chart byte stream with a bounded partition workset."""
    chart = reference_owner.lagrange_element("hexahedron", 1)
    nodes = tuple(dofs[0] for dofs in chart.entity_dofs[0])
    total = sum(prod(int(n) for n in row) for row in counts)
    signature = [
        {"path": f"[{index}]", "shape": [total, 8, 3], "dtype": np.dtype(np.int64).str}
        for index in range(2)
    ]
    digest = hashlib.sha256()
    with scope.host_workspace() as workspace:
        largest = max(prod(int(n) for n in row) for row in counts)
        workspace.set_bound(6 * min(1024, largest) * 192 + 4096)

        @lru_cache(maxsize=4)
        def chunk(partition: tuple[int, ...], start: int, denominator: bool) -> bytes:
            size = min(1024, prod(partition) - start)
            if denominator:
                return pack("=qqq", *partition) * (8 * size)
            payload = bytearray(size * 192)
            with scope.batch(24 * size) as ticket:
                ticket.consume(24 * size)
                _, ny, nz = partition
                for local, index in enumerate(range(start, start + size)):
                    lower = (index // (ny * nz), (index // nz) % ny, index % nz)
                    for corner, node in zip(_INTEGER_POINTS, nodes, strict=True):
                        pack_into(
                            "=qqq", payload, (8 * local + node) * 24,
                            lower[0] + corner[0], lower[1] + corner[1], lower[2] + corner[2],
                        )
            return bytes(payload)

        try:
            for index, metadata in enumerate(signature):
                encoded = canonical_json(metadata).encode("ascii")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                digest.update((total * 192).to_bytes(8, "big"))
                for row in counts:
                    partition = tuple(int(n) for n in row)
                    for start in range(0, prod(partition), 1024):
                        digest.update(chunk(partition, start, bool(index)))
        finally:
            chunk.cache_clear()
    return {"signature": signature, "sha256": digest.hexdigest()}


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
    banks: frame_records_owner.SourceIntegerGridBanks | frame_records_owner._SourceTensorGridBanks,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> frame_records_owner.SourceBlockIntegerGrid | frame_records_owner._SourceBlockTensorGrid:
    cell_count = banks.cells.shape[0]
    # The root-node hash table is dead before this phase. Reuse its actual
    # workspace instead of retaining that envelope beside a second mesh one.
    workspace.set_bound(1024 * cell_count + 4096)
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = np.asarray(_HEX_TOPOLOGY.entities[dimension], dtype=np.int64)
        entity_rows = np.sort(
            banks.cells[:, local_entities].reshape((-1, local_entities.shape[1])),
            axis=1,
        )
        if np.unique(entity_rows, axis=0).shape[0] > bound:
            raise failure(
                f"source_grid.entities must respect dimension {dimension} capacity {bound}; got oversized topology",
                resource=True,
            )
        del entity_rows
    cell_regions = family_host_array((admission.total_cells,), np.int64, scope=scope)
    np.take(admission.material, banks.parent_rows, out=cell_regions)
    node_count, ambient = banks.node_count, banks.points.shape[1]
    shapes = (("hexahedron", cell_count, banks.cells.shape[1]),)
    block_bytes = cell_mesh_owner.cell_block_construction_bounds(
        cell_count, banks.cells.shape[1]
    )
    mesh_bytes = cell_mesh_owner.cell_mesh_construction_bounds(
        node_count, ambient, shapes
    )
    workspace.set_bound(block_bytes)
    block = reference_cell_types_owner.CellBlock(
        "source_integer_blocks", "hexahedron", banks.cells
    )
    workspace.set_bound(mesh_bytes)
    mesh = cell_mesh_owner.CellMesh(
        banks.points[:node_count],
        (block,),
        numeric_version=prepared.reference_mesh.numeric_version,
    )
    workspace.set_bound(0)
    identifier = canonical_fingerprint(
        {
            "kind": "source-block-integer-extraction",
            "frame": prepared.frame_id,
            "connection": source_connection.holonomy.connection_id,
            "offsets": array_tree_fingerprint(source_connection.offsets),
            "intervals": array_tree_fingerprint(admission.counts),
            "regions": array_tree_fingerprint(admission.material),
            "charts": (
                array_tree_fingerprint((banks.chart_numerators, banks.chart_denominators))
                if isinstance(banks, frame_records_owner.SourceIntegerGridBanks)
                else _tensor_chart_fingerprint(admission.counts, scope=scope)
            ),
            "mesh": mesh.mesh_id,
        }
    )
    retained_bytes = sum(
        array.nbytes
        for array in (
            banks.points,
            banks.cells,
            cell_regions,
            banks.parent_rows,
            admission.counts,
            *(
                (banks.lower, banks.chart_numerators, banks.chart_denominators)
                if isinstance(banks, frame_records_owner.SourceIntegerGridBanks)
                else ()
            ),
        )
    )
    if retained_bytes > limits.maximum_data_bytes:
        raise failure(
            "source_grid.retained_bytes must respect limits.maximum_data_bytes; got oversized banks",
            resource=True,
        )
    if isinstance(banks, frame_records_owner._SourceTensorGridBanks):
        return frame_records_owner._SourceBlockTensorGrid(
            mesh, cell_regions, banks.parent_rows, admission.counts,
            prepared, source_connection, identifier,
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
    requested_intervals: np.ndarray | None,
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
    axis_bounds: np.ndarray | None = None,
    desired_axis_bounds: np.ndarray | None = None,
    physical_lipschitz_bound: float = 1.0,
    target_size: float | None = None,
    maximum_size: float | None = None,
    compact: bool = False,
) -> frame_records_owner.SourceBlockIntegerGrid | frame_records_owner._SourceBlockTensorGrid:
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
        axis_bounds=axis_bounds, target_size=target_size, maximum_size=maximum_size,
        desired_axis_bounds=desired_axis_bounds, physical_lipschitz_bound=physical_lipschitz_bound,
        compact=compact,
    )
    node_ticket = None if compact else scope.batch(
        admission.total_cells * 24 + admission.upper_vertices * 33
    )
    reference_snapshot = None
    if record_phase is not None:
        reference_work = sum(
            1 + value.size
            for value in jax.tree.leaves(prepared.reference_mesh)
            if isinstance(value, (jax.Array, np.ndarray))
        )
        with scope.batch(2 * reference_work) as callback_ticket:
            callback_ticket.consume(2 * reference_work)
        reference_snapshot = array_tree_fingerprint(prepared.reference_mesh)
    record_elapsed(record_phase, "block_count_compatibility", started)
    # The recorder is user supplied. Its callback may mutate borrowed source
    # atoms, so the closed construction/proof phase reauthenticates afterward.
    if record_phase is not None:
        if cell_geometry_id(prepared.geometry) != prepared.source_id:
            raise failure(
                "prepared.source_id must match source_geometry after the phase callback"
            )
        if reference_snapshot != array_tree_fingerprint(prepared.reference_mesh):
            raise failure("prepared.reference_mesh must remain unchanged after the phase callback")
        if not np.array_equal(
            prepared.root_cells,
            np.concatenate([
                np.asarray(block.vertices, dtype=np.int64)
                for block in prepared.reference_mesh.blocks
            ]),
        ):
            raise failure("prepared.root_cells must match reference_mesh after the phase callback")
        _validate_source_block_connection(prepared, source_connection)
    started = phase_started(record_phase)
    if compact:
        banks = _prepare_source_tensor_banks(prepared, admission, scope=scope)
    else:
        assert node_ticket is not None
        banks = _prepare_source_integer_banks(prepared, admission, node_ticket, scope=scope)
    result = _assemble_source_integer_grid(
        prepared, source_connection, admission, banks, limits,
        scope=scope, workspace=workspace,
    )
    # Connectivity and rounded point construction banks have been copied into
    # their canonical mesh owners. They are not chart evidence; release their
    # last construction owner before staging the private restriction proof.
    del banks
    if isinstance(result, frame_records_owner._SourceBlockTensorGrid):
        _certify_source_grid_restrictions(
            result.mesh, prepared.reference_mesh, result.source_parent_rows,
            None, None, admission.material, limits,
            scope=scope, budget=budget, tensor_counts=admission.counts,
            tensor_charts=_tensor_grid_charts(result, scope=scope),
        )
    else:
        certify_source_grid_restrictions(
            result.mesh, prepared.reference_mesh, result.source_parent_rows,
            result.chart_numerators, result.chart_denominators, admission.material, limits,
            scope=scope, budget=budget,
            tensor_counts=admission.counts,
        )
    record_elapsed(record_phase, "source_block_integer_extraction", started)
    return result


def _extract_source_block_tensor_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray | None,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    /,
    *,
    source_connection: frame_records_owner.SourceBlockConnection | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    axis_bounds: np.ndarray | None = None,
    desired_axis_bounds: np.ndarray | None = None,
    physical_lipschitz_bound: float = 1.0,
    target_size: float | None = None,
    maximum_size: float | None = None,
) -> frame_records_owner._SourceBlockTensorGrid:
    """Close internal extraction without exporting materialized chart banks."""
    if source_connection is None:
        source_connection = prepare_source_block_connection(
            prepared, record_phase=record_phase, scope=scope
        )
    with scope.host_workspace() as workspace:
        result = _extract_source_block_integer_grid(
            prepared, requested_intervals, regions, limits, maximum_depth, workspace,
            source_connection=source_connection, record_phase=record_phase,
            scope=scope, budget=budget, axis_bounds=axis_bounds,
            desired_axis_bounds=desired_axis_bounds,
            physical_lipschitz_bound=physical_lipschitz_bound,
            target_size=target_size, maximum_size=maximum_size, compact=True,
        )
    assert isinstance(result, frame_records_owner._SourceBlockTensorGrid)
    return result


def _extract_source_tensor_frame_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    grid: frame_records_owner._SourceBlockTensorGrid,
    regions: np.ndarray,
    limits: MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    quality_policy: NativeHexGridSchedule | None = None,
    association_inputs: list[_ConstructedTensorGrid] | None = None,
) -> MappedGridHexConstruction:
    """Reauthenticate callback-visible source owners before physical realization."""
    if cell_geometry_id(prepared.geometry) != prepared.source_id:
        raise failure("prepared.source_id must match source_geometry; got changed coefficients")
    if not np.array_equal(
        prepared.root_cells,
        np.concatenate([np.asarray(block.vertices, dtype=np.int64) for block in prepared.reference_mesh.blocks]),
    ):
        raise failure("prepared.root_cells must match reference_mesh; got changed source topology")
    _validate_source_block_connection(prepared, grid.source_connection)
    current_grid_id = canonical_fingerprint({
        "kind": "source-block-integer-extraction",
        "frame": prepared.frame_id,
        "connection": grid.source_connection.holonomy.connection_id,
        "offsets": array_tree_fingerprint(grid.source_connection.offsets),
        "intervals": array_tree_fingerprint(grid.root_intervals),
        "regions": array_tree_fingerprint(np.asarray(regions, dtype=np.int64)),
        "charts": _tensor_chart_fingerprint(grid.root_intervals, scope=scope),
        "mesh": grid.mesh.mesh_id,
    })
    if grid.source_frame is not prepared or grid.grid_id != current_grid_id:
        raise failure("tensor_grid.plan must retain its authenticated source, counts and order")
    return _realize_mapped_grid_hexes(
        grid, prepared.reference_mesh, prepared.geometry, regions, limits,
        scope=scope, budget=budget, quality_policy=quality_policy,
        association_inputs=association_inputs,
    )


@checked
def extract_source_block_integer_grid(
    prepared: frame_records_owner.PreparedSourceHexFrame,
    requested_intervals: np.ndarray | None,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    /,
    *,
    source_connection: frame_records_owner.SourceBlockConnection | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    axis_bounds: np.ndarray | None = None,
    desired_axis_bounds: np.ndarray | None = None,
    physical_lipschitz_bound: float = 1.0,
    target_size: float | None = None,
    maximum_size: float | None = None,
) -> frame_records_owner.SourceBlockIntegerGrid:
    """Prepare the support-indexed workspace for one source-block extraction."""
    if source_connection is None:
        source_connection = prepare_source_block_connection(
            prepared, record_phase=record_phase, scope=scope
        )
    with scope.host_workspace() as workspace:
        result = _extract_source_block_integer_grid(
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
            axis_bounds=axis_bounds, target_size=target_size, maximum_size=maximum_size,
            desired_axis_bounds=desired_axis_bounds, physical_lipschitz_bound=physical_lipschitz_bound,
        )
    assert isinstance(result, frame_records_owner.SourceBlockIntegerGrid)
    return result


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
    connection_count = len(prepared.face_connections)
    with scope.host_workspace() as workspace:
        # Connection rows as host lists (conservative CPython entries) and
        # the cell-pair bank handed to the holonomy decomposition.
        workspace.set_bound((_HOST_ENTRY_BYTES + 16) * connection_count + 4096)
        transports = family_host_array((connection_count, 3, 3), np.int64, scope=scope)
        offsets = family_host_array((connection_count, 3), np.int64, scope=scope)
        corners = np.asarray(_POINTS, dtype=np.int64)
        for row, ((left, left_face, right, right_face), permutation) in enumerate(
            zip(
                prepared.face_connections.tolist(),
                prepared.face_permutations,
                strict=True,
            )
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
