#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._bvh import TriangleBVH
from ._ddg import DDGOperators, discrete_operators
from ._io import (
    mesh_region_from_source,
    planar_region_from_source,
    planar_region_from_triangles,
    triangle_arrays,
)
from ._mesh import MeshQueryResult, TriangleMesh, TriangleMeshQueryIndex
from ._regions import (
    MeshRegion,
    PlanarMeshRegion,
    SegmentMesh,
    SegmentQueryResult,
    TriangleSurface,
)
from ._topology import SegmentTopology, TriangleTopology


__all__ = [
    "DDGOperators",
    "MeshQueryResult",
    "MeshRegion",
    "mesh_region_from_source",
    "PlanarMeshRegion",
    "planar_region_from_source",
    "planar_region_from_triangles",
    "SegmentMesh",
    "discrete_operators",
    "SegmentQueryResult",
    "SegmentTopology",
    "TriangleBVH",
    "TriangleMesh",
    "TriangleMeshQueryIndex",
    "TriangleSurface",
    "TriangleTopology",
    "triangle_arrays",
]
