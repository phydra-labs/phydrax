#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tetrahedral meshes of finite lattice specimens with mixed boundaries.

A specimen ``Intersection(region, envelope)`` is bounded by implicit faces of
the region and trimmed analytic caps of the envelope, joined along certified
creases (``construct_lattice_specimen_surface``). This module fills that fixed
mixed boundary with the native piecewise-linear-complex route, keeps creases
and sharp envelope edges as authored protected segments, and certifies the
result as a watertight, oriented, embedded solid.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np

from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..typing import checked
from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from ._canonical import certify_cell_mesh
from ._contracts import CellFamilyPolicy, CellMeshingTarget, VolumeFillStrategy
from ._controls import RegionSeed
from ._organization import RegionRole
from ._plc_source import PiecewiseLinearComplex
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import SizeControlStrength, UniformSizeControl
from ._surface_construction import LatticeSpecimenSurface
from ._volume_generation import NativeVolumeSchedule


# Faces of a positively oriented tetrahedron, each with an outward normal.
_OUTWARD_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
# Relative rounding allowance between summed cell volumes and the enclosed volume.
_COVERAGE_TOLERANCE = 1.0e-9


@final
class LatticeSpecimenVolume(StrictModule, NonTrainableState):
    """Certified tetrahedral mesh of a finite lattice specimen.

    ``result`` is the canonical certified mesh, audited with the open-boundary
    check rejected (not skipped). ``boundary_matches_surface`` records that the
    mesh boundary is exactly the specimen surface (implicit faces and trimmed
    caps), each triangle oriented outward. ``measure`` is the mesh volume and
    ``porosity`` its complement in the exact envelope volume; both are
    measured, with the surface's fidelity bounds as their source evidence.
    ``surface_volume`` is the volume the surface encloses (divergence theorem)
    and ``covers_surface`` records that the cells cover exactly that region:
    with the boundary matched and no inverted or overlapping cell (audit),
    the summed cell volumes equal it up to ``_COVERAGE_TOLERANCE`` relative
    rounding. ``construction`` is the native fixed-boundary fill with its
    compliance (unmet soft shape targets are reported there).
    """

    surface: LatticeSpecimenSurface
    result: CellMeshingResult
    construction: CellMeshingResult
    measure: float = eqx.field(static=True)
    envelope_volume: float = eqx.field(static=True)
    surface_volume: float = eqx.field(static=True)
    boundary_matches_surface: bool = eqx.field(static=True)

    def __init__(
        self,
        surface: LatticeSpecimenSurface,
        result: CellMeshingResult,
        construction: CellMeshingResult,
        /,
        *,
        measure: float,
        boundary_matches_surface: bool,
    ) -> None:
        self.surface = surface
        self.result = result
        self.construction = construction
        self.measure = float(measure)
        self.envelope_volume = float(surface.envelope.volume)
        self.surface_volume = surface.enclosed_volume()
        self.boundary_matches_surface = bool(boundary_matches_surface)

    @property
    def mesh(self) -> CellMesh:
        return self.result.mesh

    @property
    def solid_fraction(self) -> float:
        """Measured mesh volume over the exact envelope volume."""
        return self.measure / self.envelope_volume

    @property
    def porosity(self) -> float:
        return 1.0 - self.solid_fraction

    @property
    def covers_surface(self) -> bool:
        """The summed cell volumes equal the volume enclosed by the surface."""
        return abs(self.measure - self.surface_volume) <= _COVERAGE_TOLERANCE * abs(
            self.surface_volume
        )

    @property
    def certified_solid(self) -> bool:
        """Watertight, oriented, embedded solid bounded exactly by the surface."""
        audit = self.result.audit
        return (
            audit.passed
            and "open_boundary" in audit.evaluated_checks
            and "self_intersection" in audit.evaluated_checks
            and self.boundary_matches_surface
            and self.covers_surface
        )


def _point_records(points: np.ndarray, /) -> np.ndarray:
    """Bit patterns of float64 rows as sortable three-field records."""
    bits = np.ascontiguousarray(np.asarray(points, dtype=np.float64)).view(np.uint64)
    return bits.view(
        np.dtype([("x", np.uint64), ("y", np.uint64), ("z", np.uint64)])
    ).reshape(-1)


def _canonical_cycles(faces: np.ndarray, /) -> np.ndarray:
    """Rotate each oriented triangle so that its smallest vertex comes first."""
    start = np.argmin(faces, axis=1)
    columns = (start[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(faces, columns, axis=1)


def specimen_boundary_matches(mesh: CellMesh, surface: LatticeSpecimenSurface, /) -> bool:
    """Whether the mesh boundary is exactly the surface, each face oriented outward.

    Surface vertices are matched to mesh vertices by bit-identical
    coordinates. Each tetrahedron face is oriented outward from its cell by
    the sign of the cell's orientation; a boundary face has one incident cell.
    """
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    mesh_records = _point_records(coordinates)
    order = np.argsort(mesh_records, kind="stable")
    surface_records = _point_records(surface.vertices)
    position = np.searchsorted(mesh_records[order], surface_records)
    if np.any(position >= order.shape[0]):
        return False
    if not np.array_equal(mesh_records[order[position]], surface_records):
        return False
    vertex_rows = order[position]
    oriented: list[np.ndarray] = []
    for block in mesh.blocks:
        tetrahedra = np.asarray(block.vertices, dtype=np.int64)
        cells = coordinates[tetrahedra]
        positive = np.linalg.det(cells[:, 1:] - cells[:, :1]) > 0.0
        for corners in _OUTWARD_FACES:
            faces = tetrahedra[:, corners]
            oriented.append(np.where(positive[:, None], faces, faces[:, ::-1]))
    faces = np.concatenate(oriented)
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        return False
    boundary = np.unique(_canonical_cycles(faces[first[counts == 1]]), axis=0)
    expected = np.unique(
        _canonical_cycles(vertex_rows[np.asarray(surface.triangles, dtype=np.int64)]),
        axis=0,
    )
    return bool(np.array_equal(boundary, expected))


@checked
def mesh_lattice_specimen(
    surface: LatticeSpecimenSurface,
    /,
    *,
    maximum_edge_length: float | None = None,
    volume_schedule: NativeVolumeSchedule | None = None,
    coordinate_contract: SpatialCoordinateContract | None = None,
    source_revision: str = "revision-1",
) -> LatticeSpecimenVolume:
    """Fill a specimen's mixed boundary (implicit faces and trimmed caps) with tetrahedra.

    The boundary is a fixed piecewise-linear complex: the implicit lattice face
    and each envelope face are facets, and the creases and sharp envelope edges
    are authored protected segments. The native ``plc_tetrahedral`` route keeps
    every boundary vertex and facet. As for the periodic quotient fill, the
    fixed boundary is the discovery lattice's contour, so the default schedule
    makes shape targets soft and reports unmet shapes in the construction
    compliance. The result is certified with the open-boundary audit rejected.
    """
    from ._contracts import VolumeMeshingSpec
    from .providers._native import NativeMeshingProvider
    from .providers._native_options import NativeMeshingOptions
    from .providers._native_sources import NativePlcSource

    source_id = f"{surface.source_id}-specimen"
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    names = surface.facet_names
    segments = np.concatenate(
        (surface.crease_segments, surface.envelope_edge_segments)
    ).reshape(-1, 2)
    segment_ids = (
        *(f"{source_id}-crease-{row}" for row in range(surface.crease_segments.shape[0])),
        *(
            f"{source_id}-envelope-edge-{row}"
            for row in range(surface.envelope_edge_segments.shape[0])
        ),
    )
    # One facet per triangle (as the implicit carrier fill does): the fixed
    # boundary keeps every triangle, and facet identities name their face.
    groups = np.asarray(surface.triangle_facets, dtype=np.int64)
    count = groups.shape[0]
    complex_ = PiecewiseLinearComplex(
        vertices,
        np.asarray(surface.triangles, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (count, 1)),
        ("specimen",),
        segments=segments,
        vertex_ids=tuple(f"{source_id}-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(
            f"{source_id}-{names[group]}-{row}"
            for row, group in enumerate(groups.tolist())
        ),
        facet_occurrence_paths=((),) * count,
        segment_ids=segment_ids,
        segment_occurrence_paths=((),) * len(segment_ids),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, source_id, source_revision)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    cover = surface.discovery.cover
    if cover is None:
        raise ValueError("A specimen surface needs its discovery cover.")
    inside = np.flatnonzero(np.asarray(cover.value_upper) < 0.0)
    if not inside.size:
        raise ValueError("The specimen cover has no certified interior leaf to seed.")
    boxes = np.asarray(cover.boxes, dtype=np.float64)[inside]
    widest = int(np.argmax(np.prod(boxes[:, 1] - boxes[:, 0], axis=1)))
    seed = surface.envelope.map_host(0.5 * boxes[widest, 0] + 0.5 * boxes[widest, 1])
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            ()
            if maximum_edge_length is None
            else (UniformSizeControl(scope, float(maximum_edge_length)),)
        ),
        region_seeds=(
            RegionSeed(seed, "specimen", f"{source_id}-material", RegionRole.SOLID),
        ),
    )
    contract = (
        SpatialCoordinateContract.si()
        if coordinate_contract is None
        else coordinate_contract
    )
    construction = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "plc_tetrahedral",
                volume_schedule=(
                    NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT)
                    if volume_schedule is None
                    else volume_schedule
                ),
            )
        )
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )
    filled = construction.mesh
    result = certify_cell_mesh(
        CellMesh(
            np.asarray(filled.coordinates, dtype=np.float64),
            filled.blocks,
            vertex_global_ids=filled.vertex_global_ids,
        ),
        contract,
        audit_policy=CellMeshAuditPolicy(
            watertight_boundary=CellMeshAuditDisposition.REJECT
        ),
    )
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    volume = 0.0
    for block in result.mesh.blocks:
        cells = coordinates[np.asarray(block.vertices, dtype=np.int64)]
        volume += float(np.sum(np.abs(np.linalg.det(cells[:, 1:] - cells[:, :1])))) / 6.0
    return LatticeSpecimenVolume(
        surface,
        result,
        construction,
        measure=volume,
        boundary_matches_surface=specimen_boundary_matches(result.mesh, surface),
    )


__all__ = [
    "LatticeSpecimenVolume",
    "mesh_lattice_specimen",
    "specimen_boundary_matches",
]
