import numpy as np

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
