#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Solid STL/3MF export of accepted lattice meshes through the surface interchange owner.

The exported surface is the boundary of an accepted lattice volume: the
specimen surface of a certified `LatticeSpecimenVolume` (implicit faces and
trimmed caps), or the closed boundary of one lifted periodic cell of a
`PeriodicImplicitVolume` (contour and paired cell-face caps). A solid is
exported only under a `SurfaceValidityCertificate` from an audit that requires
every component closed, every edge on two consistently oriented triangles,
every vertex a single fan, nondegenerate triangles and an outward orientation.

3MF keeps the accepted surface exactly (binary64 coordinates, units) with the
source identity, result identity, certificate and fidelity bounds as package
metadata. STL keeps the triangles only, so it is declared lossy with respect to
the source: it needs ``allow_lossy=True`` and reports
``stl_lossy_with_respect_to_source``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.surface._contracts import (
    SurfaceAuditPolicy,
    SurfaceAuditReport,
    SurfaceMetadata,
    SurfaceValidityCertificate,
)
from ..geometry.surface._interop import (
    _loss_permission,
    _resolve_format,
    export_surface,
    SurfaceExportPolicy,
    SurfaceExportResult,
    SurfaceFileFormat,
    SurfaceSolidCertificateError,
)
from ..geometry.surface._model import SurfaceModel
from ._lattice_specimen import LatticeSpecimenVolume
from ._periodic_implicit_volume import PeriodicImplicitVolume


# Faces of a positively oriented tetrahedron, each with an outward normal.
_OUTWARD_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_STL_SOURCE_LOSS = "stl_lossy_with_respect_to_source"


@dataclass(frozen=True, slots=True)
class LatticeSolidExport:
    """A published lattice solid with its certificate and declared losses.

    ``export.report.losses`` are losses with respect to the accepted surface;
    ``source_losses`` are losses with respect to the lattice source (the
    implicit field and the certified fidelity evidence). ``fidelity`` names the
    certified source-fidelity bounds of the exported surface, which a 3MF
    package carries as metadata. ``enclosed_volume`` is the audited volume of
    the closed surface and ``mesh_measure`` the accepted volume mesh measure.
    """

    export: SurfaceExportResult
    audit: SurfaceAuditReport
    certificate: SurfaceValidityCertificate
    source_losses: tuple[str, ...]
    fidelity: tuple[tuple[str, float], ...]
    enclosed_volume: float
    mesh_measure: float

    @property
    def path(self) -> Path:
        return self.export.path

    @property
    def losses(self) -> tuple[str, ...]:
        return (*self.export.report.losses, *self.source_losses)

    @property
    def lossy(self) -> bool:
        return bool(self.losses)


def _first_appearance(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only triangle vertices, numbered in order of first appearance."""
    flat = triangles.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    order = flat[np.sort(first)]
    renumber = np.full(vertices.shape[0], -1, dtype=np.int64)
    renumber[order] = np.arange(order.size, dtype=np.int64)
    return vertices[order], renumber[triangles]


def _periodic_cell_boundary(volume: PeriodicImplicitVolume, /) -> np.ndarray:
    """Outward boundary triangles of the lifted cell (one incident tetrahedron)."""
    mesh = volume.mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    oriented = []
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
    return faces[np.sort(first[counts == 1])]


def lattice_solid_surface(
    volume: LatticeSpecimenVolume | PeriodicImplicitVolume, /
) -> tuple[SurfaceModel, tuple[tuple[str, float], ...]]:
    """The accepted closed boundary of a lattice volume and its fidelity bounds.

    A specimen must be a certified solid (``certified_solid``); a periodic cell
    must carry a passed audit. Only the boundary's vertices are kept, numbered
    in order of first appearance, so triangle-only formats reproduce them.
    """
    result = volume.result
    if isinstance(volume, LatticeSpecimenVolume):
        if not volume.certified_solid:
            raise ValueError("Only a certified specimen solid can be exported.")
        surface = volume.surface
        vertices = np.asarray(surface.vertices, dtype=np.float64)
        triangles = np.asarray(surface.triangles, dtype=np.int64)
        fidelity = (
            ("implicit_face_distance", surface.implicit_face_distance),
            ("cap_deviation", surface.cap_deviation),
            ("crease_root_distance", surface.crease_root_distance),
        )
        kind = "lattice-specimen"
        source_id = surface.source_id
    elif isinstance(volume, PeriodicImplicitVolume):
        if not result.audit.passed or volume.mesh.periodic_topology is None:
            raise ValueError("Only an audited periodic lattice cell can be exported.")
        vertices = np.asarray(volume.mesh.coordinates, dtype=np.float64)
        triangles = _periodic_cell_boundary(volume)
        fidelity = (
            (
                "vertex_root_distance",
                volume.surface.lattice_metrics().vertex_root_distance,
            ),
        )
        kind = "periodic-lattice-cell"
        source_id = volume.surface.source_id
    else:
        raise TypeError(
            "volume must be a LatticeSpecimenVolume or a PeriodicImplicitVolume."
        )
    points, faces = _first_appearance(vertices, triangles)
    metadata = SurfaceMetadata(
        source_id=source_id,
        source_revision=result.result_id,
        coordinate_contract=result.coordinate_contract,
        provenance=(
            f"phydrax.meshing:{kind}",
            f"result_id:{result.result_id}",
            *(f"fidelity:{name}={value!r}" for name, value in fidelity),
        ),
    )
    return SurfaceModel.from_triangles(points, faces, metadata), fidelity


def export_lattice_solid(
    path: str | Path,
    volume: LatticeSpecimenVolume | PeriodicImplicitVolume,
    policy: SurfaceExportPolicy,
    /,
    *,
    file_format: SurfaceFileFormat | None = None,
) -> LatticeSolidExport:
    """Export the accepted lattice solid as STL or 3MF through `export_surface`.

    The closed, oriented 2-manifold certificate is required for either format;
    a surface that fails it raises `SurfaceSolidCertificateError` with the
    audit. STL is lossy with respect to the source and needs
    ``policy.allow_lossy``.
    """
    destination = Path(path).expanduser().resolve()
    format_ = _resolve_format(destination, file_format)
    if format_ not in (SurfaceFileFormat.STL, SurfaceFileFormat.THREE_MF):
        raise ValueError("Lattice solids export as STL or 3MF.")
    source_losses = _loss_permission(
        (_STL_SOURCE_LOSS,) if format_ is SurfaceFileFormat.STL else (),
        policy.allow_lossy,
    )
    model, fidelity = lattice_solid_surface(volume)
    audit = model.audit(
        SurfaceAuditPolicy(require_closed=True, require_outward_orientation=True)
    )
    if not bool(np.asarray(audit.valid)):
        raise SurfaceSolidCertificateError(audit)
    certificate = SurfaceValidityCertificate(audit)
    exported = export_surface(destination, model, policy, file_format=format_)
    return LatticeSolidExport(
        exported,
        audit,
        certificate,
        source_losses,
        fidelity,
        float(np.sum(np.asarray(audit.component_signed_volumes))),
        float(volume.measure),
    )


__all__ = ["LatticeSolidExport", "export_lattice_solid", "lattice_solid_surface"]
