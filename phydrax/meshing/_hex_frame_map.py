"""Exact source-chart varying frames and integer-map compatibility.

Frames are the derivatives of the original coordinate expressions, not a
rotation fitted to boundary nodes. The integrable chart field is retained by
its source geometry identity; extraction restricts those same expressions.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import charge_native_geometry_queries, current_native_execution_budget
from .._strict import StrictModule
from ..discretization import CellMesh
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_enclosure import (
    coordinate_polynomials,
    derivative,
    evaluate,
)
from ..discretization._reference_cell import reference_cell_topology
from ..ein import contract
from ..linalg._small_batched import SmallLinearSolvePlan, solve_small_linear
from ..optim._iterative._types import MinimizationResult
from ._contracts import MeshingLimits
from ._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from ._quad_generation import _failure, _family_host_array
from ._reference_root_composition import affine_reference_root_frame


if TYPE_CHECKING:
    from .._meshcore import NativeHostStorageWorkspace
    from ._hex_generation import GridHexConstruction, MappedGridHexConstruction

# Canonical local tensor face cycles; vertex identity, never proximity, glues charts.
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


def _chart_face_permutation(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Prove opposite face orientation using authoritative source vertices."""
    if (
        len(left) != 4
        or len(right) != 4
        or len(set(left)) != 4
        or set(left) != set(right)
    ):
        raise _failure(
            "Source frame connection does not identify one complete tensor face."
        )
    permutation = tuple(right.index(vertex) for vertex in left)
    if any(
        (permutation[(index + 1) % 4] - permutation[index]) % 4 != 3 for index in range(4)
    ):
        raise _failure(
            "Source frame boundary parity is incompatible: shared chart cycles are not reversed."
        )
    return permutation


class PreparedSourceHexFrame(StrictModule):
    """Prepared integrable frame with explicit source-root face connection."""

    reference_mesh: CellMesh
    geometry: CellGeometrySpec
    root_cells: np.ndarray
    corner_frames: np.ndarray
    reciprocal_frames: jax.Array
    face_connections: np.ndarray
    face_permutations: np.ndarray
    boundary_faces: np.ndarray
    varying_roots: np.ndarray
    source_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)


def prepare_source_hex_frame(
    reference_mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> PreparedSourceHexFrame:
    """Prepare an authored chart field once and prove its tensor face connection.

    Continuous physical traces and global embedding remain independent source
    certification obligations. No sampled frame is used as a geometry proxy.
    """
    started = phase_started(record_phase)
    if any(block.cell_kind != "hexahedron" for block in reference_mesh.blocks):
        raise _failure(
            "Varying-frame decomposition requires authored tensor volume charts."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in reference_mesh.blocks]
    )
    points = np.asarray(reference_mesh.coordinates, dtype=np.float64)
    for cell in cells:
        affine_reference_root_frame(points[cell], "hexahedron")
    validity = certify_cell_geometry_validity(geometry, mesh=reference_mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "Original varying-frame chart determinant is invalid or unresolved."
        )
    elements, routes, _ = geometry.resolve(reference_mesh)
    values = geometry.source_coordinates()
    frames = _family_host_array((cells.shape[0], 8, 3, 3), np.float64)
    varying = []
    root = 0
    for block, element, routes_ in zip(
        reference_mesh.blocks, elements, routes, strict=True
    ):
        for route in np.asarray(routes_, dtype=np.int64):
            polynomials = coordinate_polynomials(
                element, tuple(values[index] for index in route)
            )
            if polynomials is None:
                raise _failure(
                    "Original varying-frame source has no exact polynomial chart derivative."
                )
            jets = tuple(
                tuple(derivative(value, axis) for axis in range(3))
                for value in polynomials
            )
            varying.append(
                any(any(any(power) for power in jet) for row in jets for jet in row)
            )
            for corner, point in enumerate(_POINTS):
                frames[root, corner] = [
                    [float(evaluate(jet, point)) for jet in row] for row in jets
                ]
            root += 1
    charge_native_geometry_queries(root * 8, work_units=root * 72)
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        jnp.asarray(frames),
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), frames.shape),
    )
    successful = np.asarray(jax.device_get(solve.successful))
    if not np.all(successful):
        raise _failure(
            "A varying-frame source corner has singular reciprocal coordinates."
        )
    bank = {}
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
            raise _failure(
                "Source frame singularity has a nonmanifold chart-face incidence."
            )
        left, right = occurrences
        permutation = _chart_face_permutation(left[2], right[2])
        connections.append((left[0], left[1], right[0], right[1]))
        permutations.append(permutation)
    connection = np.asarray(connections, dtype=np.int64).reshape((-1, 4))
    permutation = np.asarray(permutations, dtype=np.int64).reshape((-1, 4))
    boundary_ = np.asarray(boundary, dtype=np.int64).reshape((-1, 2))
    varying_ = np.asarray(varying, dtype=np.bool_)
    source_id = cell_geometry_id(geometry)
    identity = canonical_fingerprint(
        {
            "kind": "source-integrable-hex-frame",
            "geometry": source_id,
            "topology": reference_mesh.topology_id,
            "connections": array_tree_fingerprint(connection),
            "permutations": array_tree_fingerprint(permutation),
        }
    )
    record_elapsed(record_phase, "source_frame_prepare", started)
    return PreparedSourceHexFrame(
        reference_mesh,
        geometry,
        cells,
        frames,
        solve.value,
        connection,
        permutation,
        boundary_,
        varying_,
        source_id,
        identity,
    )


def extract_source_frame_grid(
    prepared: PreparedSourceHexFrame,
    grid: GridHexConstruction | SourceBlockIntegerGrid,
    regions: np.ndarray,
    limits: MeshingLimits,
    /,
) -> MappedGridHexConstruction:
    """Restrict a compatible integer complex to the unchanged accepted source.

    Preparation is immutable: a refusal leaves both source and prior accepted
    grid untouched. Publication certification must still prove global embedding,
    whole-chart coverage, features and the original requested quality bounds.
    """
    from ._hex_generation import realize_mapped_grid_hexes

    if cell_geometry_id(prepared.geometry) != prepared.source_id:
        raise _failure(
            "The prepared varying-frame source coefficients changed before extraction."
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
        raise _failure("The prepared varying-frame original reference topology changed.")
    return realize_mapped_grid_hexes(
        grid, prepared.reference_mesh, prepared.geometry, regions, limits
    )


class HexFrameHolonomy(StrictModule):
    """Exact octahedral connection closure and source-root cut witnesses."""

    spanning_edges: np.ndarray
    cycle_edges: np.ndarray
    cycle_holonomy: np.ndarray
    singular_cycles: np.ndarray
    root_gauges: np.ndarray
    source_id: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)


def prepare_hex_frame_holonomy(
    source: CellMesh,
    connections: np.ndarray,
    transports: np.ndarray,
    /,
) -> HexFrameHolonomy:
    """Decompose an authored signed-permutation connection into tree and cycles.

    Nonidentity cycle products are explicit singularity witnesses, not discarded
    residuals. Cutting those cycle edges supplies single-valued block gauges;
    gluing or parity of the resulting integer charts is a separate obligation.
    Every connection must be an actual shared source-cell face.
    """
    from ._quad_generation import _entities

    cells = _entities(source, 3)
    edges = np.asarray(connections)
    rotations = np.asarray(transports)
    if edges.dtype.kind not in "iu" or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Frame connections must be integer source-cell pairs.")
    if rotations.dtype.kind not in "iu" or rotations.shape != (edges.shape[0], 3, 3):
        raise ValueError("Frame transport must retain exact integer octahedral actions.")
    identity = np.eye(3, dtype=np.int64)
    incidence = [set(row[row >= 0].tolist()) for row in cells]
    faces = {frozenset(row[row >= 0].tolist()) for row in _entities(source, 2)}
    adjacency = [[] for _ in cells]
    seen = set()
    for row, ((left, right), action) in enumerate(
        zip(edges.tolist(), rotations, strict=True)
    ):
        if left == right or min(left, right) < 0 or max(left, right) >= len(cells):
            raise _failure("Frame connection names an invalid original source-cell pair.")
        key = tuple(sorted((left, right)))
        if key in seen or frozenset(incidence[left] & incidence[right]) not in faces:
            raise _failure(
                "Frame connection is not one unique original shared cell face."
            )
        seen.add(key)
        determinant = (
            int(action[0, 0])
            * (
                int(action[1, 1]) * int(action[2, 2])
                - int(action[1, 2]) * int(action[2, 1])
            )
            - int(action[0, 1])
            * (
                int(action[1, 0]) * int(action[2, 2])
                - int(action[1, 2]) * int(action[2, 0])
            )
            + int(action[0, 2])
            * (
                int(action[1, 0]) * int(action[2, 1])
                - int(action[1, 1]) * int(action[2, 0])
            )
        )
        if (
            np.any(np.abs(action) > 1)
            or not np.array_equal(action.T @ action, identity)
            or determinant != 1
        ):
            raise _failure("Frame transport is not a proper exact octahedral action.")
        adjacency[left].append((right, row, action))
        adjacency[right].append((left, row, action.T))
    gauges = _family_host_array((len(cells), 3, 3), np.int64)
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
    return HexFrameHolonomy(
        tree_array, cycles, products, singular, gauges, source.mesh_id, identity_
    )


class SolvedHexConnectionField(StrictModule):
    """Accepted numeric field plus immutable exact singularity preparation."""

    frames: jax.Array
    angles: jax.Array
    feature_residuals: jax.Array
    holonomy: HexFrameHolonomy
    optimization: MinimizationResult
    field_id: str = eqx.field(static=True)


def _connection_field_energy(
    angles: jax.Array,
    arguments: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    # All source/connection/constraint numeric state is dynamic optimizer args.
    from ._hex_generation import _frame_rotation

    initial, connections, transports, feature_rows, normals = arguments
    frames = initial @ jax.vmap(_frame_rotation)(angles)
    left = frames[connections[:, 0]]
    right = frames[connections[:, 1]]
    relative = jnp.swapaxes(right, -1, -2) @ left @ jnp.swapaxes(transports, -1, -2)
    smooth = jnp.sum(jnp.square(relative - jnp.eye(3)))
    projected = contract("ni,nij->nj", normals, frames[feature_rows])
    alignment = jnp.sum(1.0 - jnp.sum(projected**4, axis=1))
    return smooth + 4.0 * alignment


def solve_hex_connection_field(
    source: CellMesh,
    initial_frames: np.ndarray,
    connections: np.ndarray,
    transports: np.ndarray,
    feature_cells: np.ndarray,
    feature_faces: np.ndarray,
    maximum_steps: int,
    /,
) -> SolvedHexConnectionField:
    """Solve feature alignment on a prepared source-cell connection graph.

    Features name original source faces, not arbitrary supplied normals. Exact
    cycle holonomy is retained even when the variational residual is small.
    This numerical field does not authorize replacing the original geometry.
    """
    from ..optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS
    from ._quad_generation import _charge_field_work, _entities, _field_work_allowance

    if (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps <= 0
    ):
        raise ValueError("maximum_steps must be a positive integer.")
    holonomy = prepare_hex_frame_holonomy(source, connections, transports)
    cells = _entities(source, 3)
    faces = _entities(source, 2)
    initial = np.asarray(initial_frames, dtype=np.float64)
    if initial.shape != (len(cells), 3, 3) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial frame state must match the original source cells.")
    if not np.allclose(
        np.swapaxes(initial, -1, -2) @ initial, np.eye(3), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Initial frames must be orthonormal, not coordinate-map Jacobians."
        )
    if np.any(
        np.sum(np.cross(initial[:, 0], initial[:, 1]) * initial[:, 2], axis=1) <= 0.0
    ):
        raise ValueError("Initial source frames must preserve orientation.")
    owners = np.asarray(feature_cells)
    selected = np.asarray(feature_faces)
    if (
        owners.dtype.kind not in "iu"
        or selected.dtype.kind not in "iu"
        or owners.ndim != 1
        or owners.shape != selected.shape
    ):
        raise ValueError("Feature constraints must name integer original cell/face rows.")
    coordinates = np.asarray(source.coordinates, dtype=np.float64)
    normals = _family_host_array((len(owners), 3), np.float64)
    for row, (cell, face) in enumerate(
        zip(owners.tolist(), selected.tolist(), strict=True)
    ):
        if cell < 0 or cell >= len(cells) or face < 0 or face >= len(faces):
            raise _failure(
                "Frame feature constraint names an unavailable original entity."
            )
        vertices = faces[face][faces[face] >= 0]
        if not set(vertices).issubset(set(cells[cell])):
            raise _failure(
                "Frame feature face does not belong to its declared source cell."
            )
        points = coordinates[vertices]
        normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        length = np.linalg.norm(normal)
        if length == 0.0:
            raise _failure("Frame feature face has a singular original orientation.")
        normals[row] = normal / length
    method = ProjectedLBFGS()
    budget = _field_work_allowance(
        method, len(cells) + len(connections) + len(owners), maximum_steps
    )
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
    _charge_field_work(budget, result, len(cells) + len(connections) + len(owners))
    from ._hex_generation import _frame_rotation

    frames = jnp.asarray(initial) @ jax.vmap(_frame_rotation)(result.parameters)
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


class SourceBlockConnection(StrictModule):
    """Source-authored tensor face transition, including integer offset."""

    transports: np.ndarray
    offsets: np.ndarray
    holonomy: HexFrameHolonomy
    frame_id: str = eqx.field(static=True)


class SourceBlockIntegerGrid(StrictModule):
    """Conforming chartwise integer mesh, not a global-rotation grid alias."""

    mesh: CellMesh
    cell_regions: np.ndarray
    source_parent_rows: np.ndarray
    root_intervals: np.ndarray
    cell_integer_lower: np.ndarray
    chart_numerators: np.ndarray
    chart_denominators: np.ndarray
    source_frame: PreparedSourceHexFrame
    source_connection: SourceBlockConnection
    grid_id: str = eqx.field(static=True)


def _extract_source_block_integer_grid(
    prepared: PreparedSourceHexFrame,
    requested_intervals: np.ndarray,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    workspace: NativeHostStorageWorkspace | None,
    /,
    *,
    source_connection: SourceBlockConnection,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SourceBlockIntegerGrid:
    """Extract exact local integer maps with authoritative source-entity gluing.

    Sizing interval requests are lower bounds, not immutable edge controls.
    Exact rational local chart controls, not rounded carrier corners, own the
    extracted maps. Original target/quality/resource contracts still apply.
    Parallel logical edges of each root share one count; shared source-edge
    identity propagates those counts across arbitrary chart face rotations.
    Vertex identity is exact rational source shape-function support,
    never physical proximity. Original affine reference charts remain intact.
    """
    from itertools import product
    from math import prod

    from ..discretization import CellBlock
    from ._quad_generation import _budget
    from ._structured import logical_cells

    started = phase_started(record_phase)
    if cell_geometry_id(prepared.geometry) != prepared.source_id:
        raise _failure("Original source coefficients changed after block preparation.")
    _validate_source_block_connection(prepared, source_connection)
    roots = prepared.root_cells
    if not np.array_equal(
        roots,
        np.concatenate(
            [
                np.asarray(block.vertices, dtype=np.int64)
                for block in prepared.reference_mesh.blocks
            ]
        ),
    ):
        raise _failure(
            "Original source root identities changed after fixed frame preparation."
        )
    if len(roots) * 8192 > limits.maximum_scratch_bytes:
        raise _failure(
            "Source root parity preparation exceeds original scratch capacity.",
            resource=True,
        )
    if workspace is not None:
        workspace.set_bound(len(roots) * 8192)
    counts = np.asarray(requested_intervals)
    material = np.asarray(regions)
    if (
        counts.dtype.kind not in "iu"
        or counts.shape != (len(roots), 3)
        or np.any(counts <= 0)
    ):
        raise ValueError(
            "Integer map intervals must name every original root and local axis."
        )
    if material.dtype.kind not in "iu" or material.shape != (len(roots),):
        raise ValueError("Integer map regions must name every original source root.")
    if (
        not isinstance(maximum_depth, int)
        or isinstance(maximum_depth, bool)
        or not 0 < maximum_depth <= 20
    ):
        raise ValueError("maximum_depth must retain the native schedule range 1..20.")
    if np.any(counts > (1 << maximum_depth)):
        raise _failure(
            "Requested source block counts exceed the original declared depth.",
            resource=True,
        )
    requested = counts
    counts = _family_host_array(requested.shape, np.int64)
    counts[:] = requested
    # Union root-axis classes through original edge topology. This is the
    # finite fixed point of exact face/edge parity, not balancing fallback.
    parent = list(range(len(roots) * 3))
    rank = [0] * len(parent)
    parity_work = 0
    execution = current_native_execution_budget()
    parity_bound = len(roots) * (30 * (1 + 3 * max(1, len(parent).bit_length())) + 64)
    if parity_bound > limits.maximum_work_units:
        raise _failure(
            "Source block parity work exceeds original work capacity.", resource=True
        )
    if execution is not None:
        execution.admit_work_bound(parity_bound)

    def find(row: int) -> int:
        nonlocal parity_work
        parity_work += 1
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
            parity_work += 3
        return row

    edge_axes = _EDGE_AXES
    edge_owner = {}
    source_ids = np.asarray(prepared.reference_mesh.vertex_global_ids, dtype=np.int64)
    for root, cell in enumerate(roots):
        for axis, edges in enumerate(edge_axes):
            row = 3 * root + axis
            for left, right in edges:
                key = tuple(
                    sorted((int(source_ids[cell[left]]), int(source_ids[cell[right]])))
                )
                previous = edge_owner.setdefault(key, row)
                a, b = find(row), find(previous)
                parity_work += 4
                if a != b:
                    if rank[a] < rank[b]:
                        a, b = b, a
                    parent[b] = a
                    if rank[a] == rank[b]:
                        rank[a] += 1
    maxima = {}
    for row, count in enumerate(counts.reshape(-1).tolist()):
        representative = find(row)
        maxima[representative] = max(maxima.get(representative, 0), count)
    for row in range(counts.size):
        counts.reshape(-1)[row] = maxima[find(row)]
    if execution is not None:
        execution.charge(work=parity_work)
    if np.any(counts > (1 << maximum_depth)):
        raise _failure(
            "Source block integer count exceeds the original declared depth.",
            resource=True,
        )
    total_cells = sum(prod(row) for row in counts.tolist())
    upper_vertices = sum(prod(value + 1 for value in row) for row in counts.tolist())
    scratch_bound = upper_vertices * 4096 + total_cells * 128 + len(roots) * 8192
    _budget(
        limits,
        total_cells,
        upper_vertices,
        8 * total_cells,
        scratch_bound,
        total_cells * 128 + upper_vertices * 33 + parity_work,
    )
    data_bound = upper_vertices * 24 + total_cells * 488 + counts.nbytes
    if data_bound > limits.maximum_data_bytes:
        raise _failure(
            "Source block integer banks exceed original retained-data capacity.",
            resource=True,
        )
    if workspace is not None:
        workspace.set_bound(scratch_bound)
    execution = current_native_execution_budget()
    if execution is not None:
        execution.admit_work_bound(total_cells * 128 + upper_vertices * 33)
    record_elapsed(record_phase, "block_count_compatibility", started)
    started = phase_started(record_phase)
    coordinates = np.asarray(prepared.reference_mesh.coordinates, dtype=np.float64)
    node_bank = {}
    points = _family_host_array((upper_vertices, 3), np.float64)
    cells = _family_host_array((total_cells, 8), np.int64)
    parent_rows = _family_host_array((total_cells,), np.int64)
    lower = _family_host_array((total_cells, 3), np.int64)
    from ..discretization import lagrange_element

    chart = lagrange_element("hexahedron", 1)
    chart_numerators = _family_host_array(
        (total_cells, chart.local_dof_count, 3), np.int64
    )
    chart_denominators = _family_host_array(chart_numerators.shape, np.int64)
    node_count = cell_count = 0
    for root, (cell, intervals) in enumerate(zip(roots, counts.tolist(), strict=True)):
        shape = tuple(value + 1 for value in intervals)
        local_nodes = _family_host_array((prod(shape),), np.int64)
        matrix, origin = affine_reference_root_frame(coordinates[cell], "hexahedron")
        for local_node, integer in enumerate(product(*(range(value) for value in shape))):
            uvw = tuple(Fraction(integer[axis], intervals[axis]) for axis in range(3))
            support = []
            for vertex, corner in zip(cell.tolist(), _POINTS, strict=True):
                weight = prod(
                    uvw[axis] if corner[axis] else 1 - uvw[axis] for axis in range(3)
                )
                if weight:
                    support.append((int(source_ids[vertex]), weight))
            key = tuple(sorted(support))
            exact = tuple(
                origin[axis]
                + sum(
                    (matrix[axis][column] * uvw[column] for column in range(3)),
                    Fraction(0),
                )
                for axis in range(3)
            )
            if execution is not None:
                execution.charge(work=33)
            point = tuple(float(value) for value in exact)
            node = node_bank.get(key)
            if node is None:
                node = node_count
                node_count += 1
                node_bank[key] = node
                points[node] = point
            elif not np.array_equal(points[node], point):
                raise _failure(
                    "Original shared source-entity integer maps disagree exactly."
                )
            local_nodes[local_node] = node
        local_cells = logical_cells(shape)
        stop = cell_count + len(local_cells)
        np.take(local_nodes, local_cells, out=cells[cell_count:stop])
        parent_rows[cell_count:stop] = root
        for row, integer in enumerate(product(*(range(value) for value in intervals))):
            lower[cell_count + row] = integer
            for corner, dofs in enumerate(chart.entity_dofs[0]):
                chart_numerators[cell_count + row, dofs[0]] = tuple(
                    integer[axis] + _INTEGER_POINTS[corner][axis] for axis in range(3)
                )
                chart_denominators[cell_count + row, dofs[0]] = intervals
            if execution is not None:
                execution.charge(work=24)
        cell_count = stop
    nodes = points[:node_count]
    topology = _HEX_TOPOLOGY
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = np.asarray(topology.entities[dimension], dtype=np.int64)
        entity_rows = np.sort(
            cells[:, local_entities].reshape((-1, local_entities.shape[1])), axis=1
        )
        if np.unique(entity_rows, axis=0).shape[0] > bound:
            raise _failure(
                f"Source block dimension-{dimension} topology exceeds original entity capacity.",
                resource=True,
            )
    mesh = CellMesh(
        nodes,
        (CellBlock("source_integer_blocks", "hexahedron", cells),),
        numeric_version=prepared.reference_mesh.numeric_version,
    )
    validity = certify_cell_geometry_validity(mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "Source block integer extraction has an invalid or unresolved reference map."
        )
    identifier = canonical_fingerprint(
        {
            "kind": "source-block-integer-extraction",
            "frame": prepared.frame_id,
            "connection": source_connection.holonomy.connection_id,
            "offsets": array_tree_fingerprint(source_connection.offsets),
            "intervals": array_tree_fingerprint(counts),
            "regions": array_tree_fingerprint(material),
            "charts": array_tree_fingerprint((chart_numerators, chart_denominators)),
            "mesh": mesh.mesh_id,
        }
    )
    cell_regions = _family_host_array((total_cells,), np.int64)
    np.take(material, parent_rows, out=cell_regions)
    retained_bytes = sum(
        array.nbytes
        for array in (
            points,
            cells,
            cell_regions,
            parent_rows,
            counts,
            lower,
            chart_numerators,
            chart_denominators,
        )
    )
    if retained_bytes > limits.maximum_data_bytes:
        raise _failure(
            "Source block integer extraction exceeds original retained-data capacity.",
            resource=True,
        )
    record_elapsed(record_phase, "source_block_integer_extraction", started)
    return SourceBlockIntegerGrid(
        mesh,
        cell_regions,
        parent_rows,
        counts,
        lower,
        chart_numerators,
        chart_denominators,
        prepared,
        source_connection,
        identifier,
    )


def extract_source_block_integer_grid(
    prepared: PreparedSourceHexFrame,
    requested_intervals: np.ndarray,
    regions: np.ndarray,
    limits: MeshingLimits,
    maximum_depth: int,
    /,
    *,
    source_connection: SourceBlockConnection | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SourceBlockIntegerGrid:
    """Own the finite exact-support workspace for one source-block extraction."""
    if source_connection is None:
        source_connection = prepare_source_block_connection(
            prepared, record_phase=record_phase
        )
    execution = current_native_execution_budget()
    if execution is None:
        return _extract_source_block_integer_grid(
            prepared,
            requested_intervals,
            regions,
            limits,
            maximum_depth,
            None,
            source_connection=source_connection,
            record_phase=record_phase,
        )
    with execution.host_workspace() as workspace:
        return _extract_source_block_integer_grid(
            prepared,
            requested_intervals,
            regions,
            limits,
            maximum_depth,
            workspace,
            source_connection=source_connection,
            record_phase=record_phase,
        )


def prepare_source_block_connection(
    prepared: PreparedSourceHexFrame,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SourceBlockConnection:
    """Derive exact chart-axis transports from retained original face identities.

    Tangential actions are determined by source face corners. The transverse
    sign is the positive continuation across opposite boundary orientations.
    These are topological tensor-chart actions, not fitted physical rotations.
    """
    started = phase_started(record_phase)
    transports = _family_host_array((len(prepared.face_connections), 3, 3), np.int64)
    offsets = _family_host_array((len(prepared.face_connections), 3), np.int64)
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
                raise _failure(
                    "Original source face has no complete tensor-axis transition."
                )
        action[right_normal, left_normal] = (
            1 if left_points[0, left_normal] != right_points[0, right_normal] else -1
        )
        transports[row] = action
        offsets[row] = right_points[0] - action @ left_points[0]
        if not np.array_equal(left_points @ action.T + offsets[row], right_points):
            raise _failure(
                "Original face connection does not admit one exact integer affine chart map."
            )
    connections = prepared.face_connections[:, (0, 2)]
    holonomy = prepare_hex_frame_holonomy(
        prepared.reference_mesh, connections, transports
    )
    record_elapsed(record_phase, "frame_graph_decomposition", started)
    return SourceBlockConnection(transports, offsets, holonomy, prepared.frame_id)


def _validate_source_block_connection(
    prepared: PreparedSourceHexFrame,
    connection: SourceBlockConnection,
    /,
) -> None:
    """Authenticate reused state without re-solving its prepared graph."""
    if (
        connection.frame_id != prepared.frame_id
        or connection.holonomy.source_id != prepared.reference_mesh.mesh_id
    ):
        raise _failure(
            "Source block connection belongs to a different original chart authority."
        )
    rows = len(prepared.face_connections)
    if connection.transports.shape != (rows, 3, 3) or connection.offsets.shape != (
        rows,
        3,
    ):
        raise _failure(
            "Prepared source block connection lost its original face-axis bank."
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
        raise _failure(
            "Prepared original source block transports changed after acceptance."
        )
    corners = np.asarray(_POINTS, dtype=np.int64)
    for row, (_, left_face, _, right_face) in enumerate(prepared.face_connections):
        left = corners[np.asarray(_FACES[left_face])]
        right = corners[np.asarray(_FACES[right_face])][prepared.face_permutations[row]]
        if not np.array_equal(
            left @ connection.transports[row].T + connection.offsets[row], right
        ):
            raise _failure(
                "Prepared source integer offset violates its original complete face trace."
            )
