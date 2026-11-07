#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified adaptive implicit publication of general, non-smooth sources.

Ported from the donor ``tests/unit/meshing/test_adaptive_implicit_volume.py``
general enclosed-source cases: an ellipsoid union with itself (a
non-distance field whose union branches tie everywhere) and two disjoint
spheres (one sharp union, two hidden components). Both are sharp CSG fields
with a ``NONSMOOTH`` certificate and no analytic profile; their root charts
rest on per-box branch decisions of the field's interval program. Every
bound is checked against exact ellipsoid or sphere geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfacePolicy,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
)
from phydrax.geometry.implicit._source_charts import (
    prepare_adaptive_implicit_root_charts,
)
from phydrax.meshing import CellMeshingResult
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_implicit import (
    _NativeImplicitBoundaryQuery,
    prepare_native_implicit,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource


_CONTRACT = SpatialCoordinateContract.si()
_DOMAIN = np.asarray(((-0.8, -0.6, -0.6), (0.8, 0.6, 0.6)), dtype=np.float64)
_TOLERANCE = 0.04
_ELLIPSOID_CENTER = np.asarray((0.0123, 0.0037, -0.0102), dtype=np.float64)
_ELLIPSOID_RADII = np.asarray((0.35, 0.28, 0.23), dtype=np.float64)
_SPHERES = (
    (np.asarray((-0.37, 0.013, -0.017), dtype=np.float64), 0.19),
    (np.asarray((0.36, -0.019, 0.021), dtype=np.float64), 0.14),
)


def _policy() -> AdaptiveImplicitSurfacePolicy:
    # The route tightens its discovery to half the deviation over the domain
    # diagonal; equal resolution lets it reuse the retained discovery.
    diagonal = float(np.linalg.norm(_DOMAIN[1] - _DOMAIN[0]))
    return AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.5 * _TOLERANCE / diagonal,
        root_tolerance=1.0e-10,
        allow_nonsmooth_field=True,
    )


def _source(
    geometry: CompiledGeometry, source_id: str, /
) -> tuple[NativeImplicitSource, AdaptiveImplicitSurface]:
    discovery = discover_adaptive_implicit_surface(
        geometry, domain=_DOMAIN, policy=_policy(), source_id=source_id
    )
    return (
        NativeImplicitSource(
            geometry, source_id, "scientific-revision", discovery=discovery
        ),
        discovery,
    )


def _ellipsoid_geometry(*, doubled: bool) -> CompiledGeometry:
    primitive = phx.geometry.Ellipsoid(
        tuple(float(value) for value in _ELLIPSOID_CENTER),
        tuple(float(value) for value in _ELLIPSOID_RADII),
    )
    return (primitive | primitive).compile() if doubled else primitive.compile()


def _spheres_geometry() -> CompiledGeometry:
    first, second = (
        phx.geometry.Sphere(tuple(float(value) for value in center), radius)
        for center, radius in _SPHERES
    )
    return (first | second).compile()


def _require_certified(result: CellMeshingResult, /) -> float:
    certificate = result.certification
    assert certificate is not None and certificate.passed
    assert certificate.coverage is not None and certificate.coverage.status == "certified"
    fidelity = certificate.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    assert 0.0 < fidelity.mesh_to_source_upper <= _TOLERANCE
    assert fidelity.source_to_mesh_upper <= _TOLERANCE
    return float(fidelity.mesh_to_source_upper)


def _boundary(result: CellMeshingResult, /) -> tuple[np.ndarray, np.ndarray]:
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks]
    )
    facets = np.sort(
        cells[:, ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))].reshape((-1, 3)), axis=1
    )
    unique, counts = np.unique(facets, axis=0, return_counts=True)
    assert np.all(counts <= 2)
    return np.asarray(result.mesh.coordinates, dtype=np.float64), unique[counts == 1]


def _component_count(faces: np.ndarray, /) -> int:
    parents = np.arange(int(faces.max()) + 1)

    def root(value: int, /) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = int(parents[value])
        return value

    for first, second, third in faces:
        for other in (second, third):
            left, right = root(int(first)), root(int(other))
            if left != right:
                parents[right] = left
    return len({root(int(vertex)) for vertex in np.unique(faces)})


def _volume(result: CellMeshingResult, /) -> float:
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    corners = np.concatenate(
        [coordinates[np.asarray(block.vertices)] for block in result.mesh.blocks]
    )
    return float(np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1]))) / 6.0


def _ellipsoid_scaled_radius(points: np.ndarray, /) -> np.ndarray:
    return np.linalg.norm((points - _ELLIPSOID_CENTER) / _ELLIPSOID_RADII, axis=1)


def _volume_specification(source: NativeImplicitSource, /) -> VolumeMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.2, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=_TOLERANCE,
    )


@pytest.mark.parametrize("source_kind", ("non-distance", "internal-tangency"))
def test_general_enclosed_scalar_source_publishes_and_rejects_stale_state(
    source_kind: str,
) -> None:
    if source_kind == "non-distance":
        geometry = _ellipsoid_geometry(doubled=True)
        components = 1
    else:
        geometry = _spheres_geometry()
        components = 2
    source, discovery = _source(geometry, "general-source")
    provider = NativeMeshingProvider(
        NativeMeshingOptions(
            "implicit_adaptive_tetrahedral",
            implicit_policy=_policy(),
            volume_schedule=NativeVolumeSchedule(),
        )
    )
    result = provider.plan(
        source, _volume_specification(source), coordinate_contract=_CONTRACT
    ).execute()
    epsilon = _require_certified(result)
    coordinates, faces = _boundary(result)
    assert _component_count(faces) == components
    boundary = coordinates[np.unique(faces)]
    volume = _volume(result)
    if source_kind == "non-distance":
        # The outer parallel body at epsilon lies in the ellipsoid scaled by
        # 1 + epsilon / c and the inner one contains the ellipsoid scaled by
        # 1 - epsilon / c, c the least semi-axis (support functions).
        spread = epsilon / float(np.min(_ELLIPSOID_RADII))
        assert np.all(np.abs(_ellipsoid_scaled_radius(boundary) - 1.0) <= spread)
        exact = 4.0 * math.pi / 3.0 * float(np.prod(_ELLIPSOID_RADII))
        assert (1.0 - spread) ** 3 * exact <= volume <= (1.0 + spread) ** 3 * exact
        # Exact distance from a point outside the major vertex.
        query = _ELLIPSOID_CENTER + np.asarray((_ELLIPSOID_RADII[0] + 0.05, 0.0, 0.0))
        exact_distance = 0.05
    else:
        distance = np.min(
            np.stack(
                [
                    np.abs(np.linalg.norm(boundary - center, axis=1) - radius)
                    for center, radius in _SPHERES
                ]
            ),
            axis=0,
        )
        assert np.all(distance <= epsilon)
        low = sum(4.0 * math.pi / 3.0 * (radius - epsilon) ** 3 for _, radius in _SPHERES)
        high = sum(
            4.0 * math.pi / 3.0 * (radius + epsilon) ** 3 for _, radius in _SPHERES
        )
        assert low <= volume <= high
        query = _SPHERES[1][0] + np.asarray((_SPHERES[1][1], 0.0, 0.0))
        exact_distance = 0.0
    bounds = AdaptiveImplicitBoundarySource(discovery).boundary_distance(query[None, :])
    assert bounds.semantics == "certified"
    assert bounds.lower[0] <= exact_distance <= bounds.upper[0]
    changed = phx.geometry.Ellipsoid((0.02, 0.03, 0.04), (0.35, 0.28, 0.23)).compile()
    with pytest.raises(ValueError, match="same compiled geometry"):
        NativeImplicitSource(
            changed, "general-source", "scientific-revision", discovery=discovery
        )


def test_ellipsoid_surface_publishes_certified_root_charts() -> None:
    geometry = _ellipsoid_geometry(doubled=False)
    source, _ = _source(geometry, "ellipsoid-surface")
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        MeshingScope.whole(source, 2),
        size_controls=(),
        maximum_deviation=_TOLERANCE,
    )
    result = (
        NativeMeshingProvider(
            NativeMeshingOptions("implicit_surface", implicit_policy=_policy())
        )
        .plan(source, specification, coordinate_contract=_CONTRACT)
        .execute()
    )
    certificate = result.certification
    assert certificate is not None and certificate.passed
    fidelity = certificate.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    epsilon = float(fidelity.mesh_to_source_upper)
    assert 0.0 < epsilon <= _TOLERANCE
    vertices = np.asarray(result.mesh.coordinates, dtype=np.float64)
    spread = epsilon / float(np.min(_ELLIPSOID_RADII))
    assert np.all(np.abs(_ellipsoid_scaled_radius(vertices) - 1.0) <= spread)


def _union_seam_distance(points: np.ndarray, /) -> np.ndarray:
    """Exact distance to the boundary of two overlapping balls of radius 0.3.

    The boundary is each sphere's cap outside the other ball; a cap's
    closest point is the radial projection when it lies on the cap, else a
    point of the seam circle (plane x = 0, radius sqrt(0.3^2 - 0.2^2)).
    """
    seam_radius = math.sqrt(0.3**2 - 0.2**2)
    seam = np.hypot(points[:, 0], np.hypot(points[:, 1], points[:, 2]) - seam_radius)
    candidates = [seam]
    for center_x, other_x in ((-0.2, 0.2), (0.2, -0.2)):
        offset = points - np.asarray((center_x, 0.0, 0.0))
        length = np.linalg.norm(offset, axis=1)
        projection = np.asarray((center_x, 0.0, 0.0)) + 0.3 * offset / length[:, None]
        on_cap = (
            np.linalg.norm(projection - np.asarray((other_x, 0.0, 0.0)), axis=1) >= 0.3
        )
        candidates.append(np.where(on_cap, np.abs(length - 0.3), np.inf))
    return np.min(np.stack(candidates), axis=0)


def test_union_seam_faces_take_the_root_witness_bound() -> None:
    geometry = (
        phx.geometry.Sphere((-0.2, 0.0, 0.0), 0.3)
        | phx.geometry.Sphere((0.2, 0.0, 0.0), 0.3)
    ).compile()
    source, _ = _source(geometry, "overlapping-union")
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        MeshingScope.whole(source, 2),
        size_controls=(),
        maximum_deviation=_TOLERANCE,
    )
    construction = prepare_native_implicit(
        source,
        specification,
        NativeMeshingOptions("implicit_surface", implicit_policy=_policy()),
        _CONTRACT,
    ).execute()
    query = construction.certification.fidelity_source
    witness = construction.certification.fidelity_witness
    assert isinstance(query, _NativeImplicitBoundaryQuery)
    assert isinstance(witness, SourceBoundaryChartWitness)
    root_map = prepare_adaptive_implicit_root_charts(
        construction.mesh,
        construction.geometry,
        query,
        witness,
        tolerance=_TOLERANCE,
    )
    kinds = {evidence.bound for evidence in root_map.chart_evidence}
    assert kinds == {"root_chart", "root_witness"}
    assert all(evidence.status == "certified" for evidence in root_map.chart_evidence)
    vertices = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    faces = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    seam_rows = [
        evidence.row
        for evidence in root_map.chart_evidence
        if evidence.bound == "root_witness"
    ]
    # Crease faces are those whose vertices straddle the seam plane x = 0 or
    # lie next to it; the witness bound encloses their exact distance.
    for row in seam_rows:
        corners = vertices[faces[row]]
        bound = root_map.chart_evidence[row].target_to_root_bound
        assert np.all(_union_seam_distance(corners) <= bound)
        assert float(np.min(np.abs(corners[:, 0]))) <= 0.1
    fidelity = certify_source_fidelity(
        construction.mesh,
        construction.geometry,
        query,
        tolerance=_TOLERANCE,
        witness=witness,
        root_charts=root_map,
    )
    assert fidelity.status == "certified"
    epsilon = float(fidelity.mesh_to_source_upper)
    assert epsilon >= max(
        evidence.target_to_root_bound for evidence in root_map.chart_evidence
    )
    assert np.all(_union_seam_distance(vertices) <= epsilon)
