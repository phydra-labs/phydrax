"""Source-plane cube fixtures with explicit chart and coedge provenance."""

import numpy as np

import phydrax as phx
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import MeshingDomain


def _cube_domain(
    *, parameter_scale: float = 1.0, source_revision: str = "authored-box-planes"
) -> MeshingDomain:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    edges = topology.entities[1]
    lookup = {tuple(sorted(edge)): index for index, edge in enumerate(edges)}
    curve_endpoints: dict[int, tuple[int, int]] = {}
    patches = []
    for face_index, face in enumerate(topology.entities[2]):
        scale = parameter_scale if face_index == 1 else 1.0
        uv = np.asarray(
            ((0.0, 0.0), (1.0 / scale, 0.0), (1.0 / scale, 1.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        uses = []
        for position, start in enumerate(face):
            end = face[(position + 1) % 4]
            edge = lookup[tuple(sorted((start, end)))]
            owner = curve_endpoints.setdefault(edge, (start, end))
            forward = owner == (start, end)
            first, last = (
                (position, (position + 1) % 4)
                if forward
                else ((position + 1) % 4, position)
            )
            curve = phx.geometry.LineCurve(uv[first], uv[last] - uv[first])
            uses.append(
                phx.geometry.PatchCurveUse(
                    edge, curve, 0.0 if forward else 1.0, 1.0 if forward else 0.0
                )
            )
        plane = phx.geometry.PlanePatch(
            points[face[0]],
            scale * (points[face[1]] - points[face[0]]),
            points[face[3]] - points[face[0]],
        )
        patches.append(phx.geometry.MeshingSurfacePatch(plane, (tuple(uses),)))
    return MeshingDomain(
        tuple(patches),
        tuple(
            phx.geometry.MeshingDomainCurve(*curve_endpoints[index])
            for index in range(len(edges))
        ),
        8,
        source_id="authored-layer-box",
        source_revision=source_revision,
        regions=(
            phx.geometry.MeshingDomainRegion(
                "fluid", tuple((index, 1) for index in range(6))
            ),
        ),
    )


def _cube_query(
    domain: MeshingDomain, *, parameter_scale: float = 1.0
) -> MeshingDomainBoundarySource:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    chains = []
    for patch, face in enumerate(topology.entities[2]):
        scale = parameter_scale if patch == 1 else 1.0
        charts = np.asarray(
            ((0.0, 0.0), (1.0 / scale, 0.0), (1.0 / scale, 1.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        provenance = []
        for index, use in enumerate(domain.patches[patch].loops[0]):
            if not isinstance(use, phx.geometry.PatchCurveUse):
                raise AssertionError(
                    "MeshingDomain.patches.loops must contain PatchCurveUse line coedges; got an unsupported coedge kind."
                )
            provenance.append((0, index, use.first, use.last))
        count = charts.shape[0]
        chains.append(
            (
                patch,
                charts,
                np.asarray(points[list(face)]),
                cells,
                boundary,
                np.asarray(provenance, dtype=np.float64),
                np.zeros((count,), dtype=np.bool_),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
            )
        )
    return MeshingDomainBoundarySource(
        domain, tuple(range(6)), resolution=4, chart_triangulations=tuple(chains)
    )
