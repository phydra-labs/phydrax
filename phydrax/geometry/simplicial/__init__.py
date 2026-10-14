from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "AffineSimplexEvidence": "._affine",
    "AffineSimplexMap": "._affine",
    "SegmentBVH": "._bvh",
    "SegmentQueryExecution": "._bvh",
    "TriangleBVH": "._bvh",
    "WindingNumberResult": "._bvh",
    "WindingNumberRoute": "._bvh",
    "DDGOperators": "._ddg",
    "discrete_operators": "._ddg",
    "mesh_region_from_source": "._io",
    "planar_region_from_source": "._io",
    "planar_region_from_triangles": "._io",
    "triangle_arrays": "._io",
    "MeshQueryResult": "._mesh",
    "TriangleMesh": "._mesh",
    "TriangleMeshQueryIndex": "._mesh",
    "MeshRegion": "._regions",
    "PlanarMeshRegion": "._regions",
    "SegmentMesh": "._regions",
    "SegmentQueryResult": "._regions",
    "TriangleSurface": "._regions",
    "PreparedSimplicialSurfels": "._surfels",
    "SimplicialSurfelPlan": "._surfels",
    "SegmentTopology": "._topology",
    "TriangleTopology": "._topology",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._affine import AffineSimplexEvidence, AffineSimplexMap
    from ._bvh import (
        SegmentBVH,
        SegmentQueryExecution,
        TriangleBVH,
        WindingNumberResult,
        WindingNumberRoute,
    )
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
    from ._surfels import PreparedSimplicialSurfels, SimplicialSurfelPlan
    from ._topology import SegmentTopology, TriangleTopology


__all__ = [
    "AffineSimplexEvidence",
    "AffineSimplexMap",
    "DDGOperators",
    "MeshQueryResult",
    "MeshRegion",
    "mesh_region_from_source",
    "PreparedSimplicialSurfels",
    "PlanarMeshRegion",
    "planar_region_from_source",
    "planar_region_from_triangles",
    "SegmentBVH",
    "SegmentQueryExecution",
    "SegmentMesh",
    "discrete_operators",
    "SegmentQueryResult",
    "SimplicialSurfelPlan",
    "SegmentTopology",
    "TriangleBVH",
    "TriangleMesh",
    "TriangleMeshQueryIndex",
    "TriangleSurface",
    "TriangleTopology",
    "triangle_arrays",
    "WindingNumberResult",
    "WindingNumberRoute",
]
