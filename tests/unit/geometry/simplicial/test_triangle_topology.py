import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import phydrax as phx


def test_triangle_topology_traces_oriented_boundary_loops_and_components() -> None:
    index = np.arange(16).reshape((4, 4))
    triangles = []
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                continue
            a, b, c, d = (
                index[i, j],
                index[i + 1, j],
                index[i + 1, j + 1],
                index[i, j + 1],
            )
            triangles += [(a, b, c), (a, c, d)]
    faces = np.asarray(triangles + [(16, 17, 18)], dtype=np.int64)
    topology = phx.geometry.simplicial.TriangleTopology(faces)

    assert topology.num_face_components == 2
    np.testing.assert_array_equal(topology.face_component_ids, [0] * 16 + [1])
    assert topology.euler_characteristic == 1

    origin = np.asarray(topology.halfedge_origin)
    destination = np.asarray(topology.halfedge_destination)
    boundary = np.asarray(topology.boundary_halfedges)
    directed = {(int(origin[h]), int(destination[h])): int(h) for h in boundary}
    offsets = np.asarray(topology.boundary_loop_offsets)
    vertices = np.asarray(topology.boundary_loop_vertices)
    loops = [vertices[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]
    assert sorted(loop.size for loop in loops) == [3, 4, 12]
    traversed = []
    for loop in loops:
        # Consecutive loop vertices follow boundary half-edges in face orientation.
        halfedges = [
            directed[(int(start), int(stop))]
            for start, stop in zip(loop, np.roll(loop, -1), strict=True)
        ]
        # Each loop starts at the origin of its smallest boundary half-edge.
        assert halfedges[0] == min(halfedges)
        traversed.extend(halfedges)
    starts = [traversed[start] for start in offsets[:-1]]
    assert starts == sorted(starts)
    assert sorted(traversed) == sorted(boundary.tolist())


def _reference_topology(faces: np.ndarray, vertex_count: int) -> dict[str, np.ndarray]:
    """Construct the same complex independently from ordered Python incidences."""
    rows = [tuple(int(vertex) for vertex in row) for row in faces]
    origins = [vertex for row in rows for vertex in row]
    destinations = [vertex for a, b, c in rows for vertex in (b, c, a)]
    edge_members: dict[tuple[int, int], list[int]] = {}
    for halfedge, (origin, destination) in enumerate(
        zip(origins, destinations, strict=True)
    ):
        edge = (min(origin, destination), max(origin, destination))
        edge_members.setdefault(edge, []).append(halfedge)
    edges = sorted(edge_members)
    twins = [-1] * len(origins)
    halfedge_edges = [-1] * len(origins)
    edge_halfedges = []
    neighbors: list[list[int]] = [[] for _ in rows]
    for edge_id, edge in enumerate(edges):
        members = edge_members[edge]
        edge_halfedges.append((members[0], members[1] if len(members) == 2 else -1))
        for member in members:
            halfedge_edges[member] = edge_id
        if len(members) == 2:
            first, second = members
            twins[first], twins[second] = second, first
            neighbors[first // 3].append(second // 3)
            neighbors[second // 3].append(first // 3)

    boundary = [halfedge for halfedge, twin in enumerate(twins) if twin == -1]
    outgoing = {origins[halfedge]: halfedge for halfedge in boundary}
    remaining = set(boundary)
    loop_vertices: list[int] = []
    loop_offsets = [0]
    while remaining:
        current = min(remaining)
        while current in remaining:
            remaining.remove(current)
            loop_vertices.append(origins[current])
            current = outgoing[destinations[current]]
        loop_offsets.append(len(loop_vertices))

    offsets = [0]
    vertex_halfedges = []
    for vertex in range(vertex_count):
        vertex_halfedges.extend(
            halfedge for halfedge, origin in enumerate(origins) if origin == vertex
        )
        offsets.append(len(vertex_halfedges))
    components = [-1] * len(rows)
    component_count = 0
    for face in range(len(rows)):
        if components[face] != -1:
            continue
        components[face] = component_count
        pending = [face]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if components[neighbor] == -1:
                    components[neighbor] = component_count
                    pending.append(neighbor)
        component_count += 1
    result = {
        "faces": rows,
        "edges": edges,
        "halfedge_origin": origins,
        "halfedge_destination": destinations,
        "halfedge_face": [halfedge // 3 for halfedge in range(len(origins))],
        "halfedge_next": [3 * (h // 3) + (h + 1) % 3 for h in range(len(origins))],
        "halfedge_previous": [3 * (h // 3) + (h + 2) % 3 for h in range(len(origins))],
        "halfedge_twin": twins,
        "halfedge_edge": halfedge_edges,
        "edge_halfedges": edge_halfedges,
        "boundary_halfedges": boundary,
        "boundary_loop_vertices": loop_vertices,
        "boundary_loop_offsets": loop_offsets,
        "vertex_face_offsets": offsets,
        "vertex_faces": [halfedge // 3 for halfedge in vertex_halfedges],
        "vertex_halfedge_offsets": offsets,
        "vertex_halfedges": vertex_halfedges,
        "face_component_ids": components,
    }
    return {name: np.asarray(value, dtype=np.int32) for name, value in result.items()}


def _legacy_topology(faces: np.ndarray, vertex_count: int) -> dict[str, np.ndarray]:
    """Previous vectorized topology preparation, retained as a test oracle."""
    face_count = faces.shape[0]
    halfedge_count = 3 * face_count
    origin = faces.reshape((-1,))
    destination = faces[:, [1, 2, 0]].reshape((-1,))
    halfedge_face = np.repeat(np.arange(face_count, dtype=np.int32), 3)
    local = np.arange(halfedge_count, dtype=np.int32).reshape((-1, 3))
    low = np.minimum(origin, destination).astype(np.int64)
    high = np.maximum(origin, destination).astype(np.int64)
    order = np.argsort(low * vertex_count + high, kind="stable")
    starts = np.ones((halfedge_count,), dtype=np.bool_)
    starts[1:] = (low[order[1:]] != low[order[:-1]]) | (
        high[order[1:]] != high[order[:-1]]
    )
    first = np.flatnonzero(starts)
    group_sizes = np.diff(np.append(first, halfedge_count))
    edges = np.stack((low[order[first]], high[order[first]]), axis=1).astype(np.int32)
    halfedge_edge = np.empty((halfedge_count,), dtype=np.int32)
    halfedge_edge[order] = np.cumsum(starts) - 1
    paired = group_sizes == 2
    first_halfedges = order[first[paired]]
    second_halfedges = order[first[paired] + 1]
    edge_halfedges = np.full((first.size, 2), -1, dtype=np.int32)
    edge_halfedges[:, 0] = order[first]
    edge_halfedges[paired, 1] = second_halfedges
    twins = np.full((halfedge_count,), -1, dtype=np.int32)
    twins[first_halfedges] = second_halfedges
    twins[second_halfedges] = first_halfedges
    boundary = np.flatnonzero(twins < 0).astype(np.int32)

    boundary_starts, boundary_ends = origin[boundary], destination[boundary]
    count = boundary.size
    position = np.empty((vertex_count,), dtype=np.int64)
    position[boundary_starts] = np.arange(count, dtype=np.int64)
    successor = position[boundary_ends]
    indices = np.arange(count, dtype=np.int64)
    root, jump = indices.copy(), successor.copy()
    for _ in range(count.bit_length()):
        root = np.minimum(root, root[jump])
        jump = jump[jump]
    is_root = root == indices
    steps = np.where(is_root, 0, 1)
    jump = np.where(is_root, indices, successor)
    for _ in range(count.bit_length()):
        steps = steps + steps[jump]
        jump = jump[jump]
    lengths = np.bincount(root, minlength=count)
    loop_order = np.lexsort(((lengths[root] - steps) % lengths[root], root))
    loop_offsets = np.zeros((np.count_nonzero(is_root) + 1,), dtype=np.int32)
    np.cumsum(lengths[is_root], out=loop_offsets[1:])
    vertex_order = np.argsort(origin, kind="stable")
    counts = np.bincount(origin[vertex_order], minlength=vertex_count)
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.int32), np.cumsum(counts, dtype=np.int32))
    ).astype(np.int32)
    vertex_halfedges = np.arange(halfedge_count, dtype=np.int32)[vertex_order]
    adjacency = sp.csr_matrix(
        (
            np.ones((first_halfedges.size,), dtype=np.int8),
            (halfedge_face[first_halfedges], halfedge_face[second_halfedges]),
        ),
        shape=(face_count, face_count),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    _, smallest = np.unique(labels, return_index=True)
    relabel = np.empty((component_count,), dtype=np.int32)
    relabel[np.argsort(smallest)] = np.arange(component_count, dtype=np.int32)
    return {
        "faces": faces,
        "edges": edges,
        "halfedge_origin": origin,
        "halfedge_destination": destination,
        "halfedge_face": halfedge_face,
        "halfedge_next": local[:, [1, 2, 0]].reshape((-1,)),
        "halfedge_previous": local[:, [2, 0, 1]].reshape((-1,)),
        "halfedge_twin": twins,
        "halfedge_edge": halfedge_edge,
        "edge_halfedges": edge_halfedges,
        "boundary_halfedges": boundary,
        "boundary_loop_vertices": boundary_starts[loop_order].astype(np.int32),
        "boundary_loop_offsets": loop_offsets,
        "vertex_face_offsets": offsets,
        "vertex_faces": halfedge_face[vertex_halfedges],
        "vertex_halfedge_offsets": offsets,
        "vertex_halfedges": vertex_halfedges,
        "face_component_ids": relabel[labels],
    }


@pytest.mark.parametrize(
    "faces",
    [
        [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        [(0, 1, 2), (0, 2, 3)],
        [(4, 5, 6), (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        [(4, 5, 6), (0, 1, 2), (0, 2, 3), (7, 8, 9)],
    ],
)
def test_triangle_topology_all_fields_match_independent_incidence_oracle(
    faces: list[tuple[int, int, int]],
) -> None:
    faces_array = np.asarray(faces, dtype=np.int32)
    vertex_count = int(np.max(faces_array)) + 3
    expected = _reference_topology(faces_array, vertex_count)
    legacy = _legacy_topology(faces_array, vertex_count)
    for name, value in expected.items():
        np.testing.assert_array_equal(legacy[name], value, err_msg=f"legacy {name}")
    for face_view in (
        faces_array,
        np.asfortranarray(faces_array),
        np.repeat(faces_array, 2, axis=1)[:, ::2],
    ):
        topology = phx.geometry.simplicial.TriangleTopology(
            face_view, num_vertices=vertex_count
        )
        for name, value in expected.items():
            actual = np.asarray(getattr(topology, name))
            assert actual.dtype == value.dtype
            np.testing.assert_array_equal(actual, value, err_msg=name)
        assert topology.num_face_components == len(set(expected["face_component_ids"]))
        assert topology.num_vertices == vertex_count
        assert topology.watertight == (expected["boundary_halfedges"].size == 0)


def test_triangle_topology_detects_duplicates_across_comparison_blocks() -> None:
    faces = np.arange(3 * 65538, dtype=np.int32).reshape((-1, 3))
    faces[-1] = faces[65535, ::-1]
    with pytest.raises(ValueError, match="duplicate faces"):
        phx.geometry.simplicial.TriangleTopology(faces)


@pytest.mark.parametrize("explicit_values", [False, True])
def test_large_csr_preserves_stable_incidence_with_missing_vertices(
    explicit_values: bool,
) -> None:
    from phydrax.geometry.simplicial._topology import _csr

    owners = (np.arange(140000, dtype=np.int32) * 17 % 29) * 2 + 2
    values = np.arange(len(owners), dtype=np.int32)[::-1] if explicit_values else None
    offsets, incidence = _csr(owners, values, 64)
    order = np.argsort(owners, kind="stable")
    counts = np.bincount(owners, minlength=64)
    expected_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    np.testing.assert_array_equal(offsets, expected_offsets)
    np.testing.assert_array_equal(incidence, order if values is None else values[order])


@pytest.mark.parametrize(
    ("faces", "num_vertices", "message"),
    [
        ([(0, 1, -1)], None, "faces must contain non-negative indices"),
        ([(0, 0, 1)], None, "Every face must reference three distinct vertices"),
        ([(0, 1, 2), (2, 1, 0)], 2, "num_vertices does not cover"),
        ([(0, 1, 2), (2, 1, 0), (0, 1, 3)], None, "duplicate faces"),
        (
            [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
            None,
            "edge has more than two incident faces",
        ),
        ([(0, 1, 2), (0, 1, 3)], None, "inconsistent orientation"),
        ([(0, 1, 2), (0, 3, 4)], None, "Boundary is non-manifold at a vertex"),
    ],
)
def test_triangle_topology_rejects_invalid_inputs_in_order(
    faces: list[tuple[int, int, int]], num_vertices: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        phx.geometry.simplicial.TriangleTopology(
            np.asarray(faces, dtype=np.int32), num_vertices=num_vertices
        )
