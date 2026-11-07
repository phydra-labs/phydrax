#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Solid STL/3MF export of accepted lattice meshes, read back independently."""

from __future__ import annotations

import hashlib
import zipfile
from functools import cache
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
)
from phydrax.geometry.surface import (
    export_surface,
    SurfaceExportPolicy,
    SurfaceLossyOperationError,
    SurfaceMetadata,
    SurfaceModel,
    SurfaceSolidCertificateError,
    SurfaceUnsupportedFormatError,
)
from phydrax.meshing import (
    export_lattice_solid,
    lattice_solid_surface,
    mesh_periodic_trigonometric_region,
    PeriodicImplicitVolume,
)
from phydrax.units import KILOMETER, METER, MILLIMETER


_CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"


@cache
def _schwarz_p_cell() -> PeriodicImplicitVolume:
    source = phx.geometry.TrigonometricLevelSet(
        phx.geometry.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )
    surface = discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.2, root_tolerance=1.0e-8
        ),
        source_id="schwarz_p-network",
    )
    return mesh_periodic_trigonometric_region(surface)


def _read_3mf(path: Path) -> tuple[np.ndarray, np.ndarray, str, dict[str, str]]:
    """Independent stdlib reader: relationships, then the model part."""
    with zipfile.ZipFile(path) as package:
        assert package.namelist() == [
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
        ]
        relationships = ElementTree.fromstring(package.read("_rels/.rels"))
        (target,) = [item.get("Target") for item in relationships]
        assert target == "/3D/3dmodel.model"
        root = ElementTree.fromstring(package.read("3D/3dmodel.model"))
    mesh = root.find(f"{_CORE}resources/{_CORE}object/{_CORE}mesh")
    assert mesh is not None
    points = np.asarray(
        [
            [float(vertex.attrib[axis]) for axis in "xyz"]
            for vertex in mesh.iter(f"{_CORE}vertex")
        ]
    )
    triangles = np.asarray(
        [
            [int(triangle.attrib[key]) for key in ("v1", "v2", "v3")]
            for triangle in mesh.iter(f"{_CORE}triangle")
        ]
    )
    metadata = {
        item.attrib["name"]: item.text or "" for item in root.iter(f"{_CORE}metadata")
    }
    return points, triangles, root.attrib["unit"], metadata


def _tetrahedron(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    triangles = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
    return points + offset, triangles


def _model(points: np.ndarray, triangles: np.ndarray) -> SurfaceModel:
    metadata = SurfaceMetadata(
        source_id="solid-fixture",
        source_revision="1",
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        provenance=("test",),
    )
    return SurfaceModel.from_triangles(points, triangles, metadata)


def test_periodic_cell_3mf_round_trips_bit_for_bit_and_deterministically(
    tmp_path: Path,
) -> None:
    volume = _schwarz_p_cell()
    model, fidelity = lattice_solid_surface(volume)
    policy = SurfaceExportPolicy(METER)

    first = export_lattice_solid(tmp_path / "cell.3mf", volume, policy)
    second = export_lattice_solid(tmp_path / "again.3mf", volume, policy)

    assert not first.lossy and first.source_losses == ()
    assert first.export.certificate is not None
    assert first.export.certificate.certificate_id == first.certificate.certificate_id
    # Equal inputs publish byte-identical packages.
    assert (tmp_path / "cell.3mf").read_bytes() == (tmp_path / "again.3mf").read_bytes()
    assert (
        first.export.report.artifact_digest
        == hashlib.sha256((tmp_path / "cell.3mf").read_bytes()).hexdigest()
    )
    del second
    points, triangles, unit, metadata = _read_3mf(tmp_path / "cell.3mf")
    np.testing.assert_array_equal(points, np.asarray(model.mesh.coordinates))
    assert points.tobytes() == np.asarray(model.mesh.coordinates).tobytes()
    np.testing.assert_array_equal(
        triangles, np.asarray(model.mesh.blocks[0].vertices, dtype=np.int64)
    )
    assert unit == "meter"
    assert metadata["phydrax:certificate_id"] == first.certificate.certificate_id
    assert metadata["phydrax:source_id"] == "schwarz_p-network"
    assert metadata["phydrax:source_revision"] == volume.result.result_id
    assert any(
        value.startswith("fidelity:vertex_root_distance=") for value in metadata.values()
    )
    assert fidelity[0][0] == "vertex_root_distance"
    # The closed cell boundary (contour and paired caps) encloses the mesh.
    assert first.enclosed_volume == pytest.approx(volume.measure, rel=1.0e-12)
    assert first.audit.component_count == 1


def test_3mf_scales_to_the_declared_unit_and_refuses_units_it_lacks(
    tmp_path: Path,
) -> None:
    points, triangles = _tetrahedron()
    model = _model(points, triangles)
    result = export_surface(
        tmp_path / "solid.3mf", model, SurfaceExportPolicy(MILLIMETER)
    )
    decoded, _, unit, _ = _read_3mf(tmp_path / "solid.3mf")
    assert unit == "millimeter"
    np.testing.assert_array_equal(decoded, points * 1000.0)
    assert result.report.coordinate_scale == 1000.0
    with pytest.raises(SurfaceUnsupportedFormatError, match="no model unit"):
        export_surface(tmp_path / "km.3mf", model, SurfaceExportPolicy(KILOMETER))


@pytest.mark.parametrize(
    ("case", "issue"),
    (
        ("open", "closed"),
        ("inward", "outward"),
        ("pinched", "separate fans"),
    ),
)
def test_3mf_refuses_surfaces_without_a_closed_oriented_manifold_certificate(
    tmp_path: Path, case: str, issue: str
) -> None:
    points, triangles = _tetrahedron()
    if case == "open":
        triangles = triangles[:3]
    elif case == "inward":
        triangles = triangles[:, ::-1].copy()
    else:
        # Two closed tetrahedra sharing one vertex: every edge has two
        # triangles, but the shared vertex joins two separate fans.
        other, other_triangles = _tetrahedron()
        other = -other[1:]
        points = np.concatenate((points, other))
        mapping = np.asarray((0, 4, 5, 6))
        triangles = np.concatenate((triangles, mapping[other_triangles[:, ::-1]]))
    model = _model(points, triangles)
    with pytest.raises(SurfaceSolidCertificateError) as refusal:
        export_surface(tmp_path / "bad.3mf", model, SurfaceExportPolicy(METER))
    assert not bool(np.asarray(refusal.value.report.valid))
    assert any(issue in text for text in refusal.value.report.issues)
    assert not (tmp_path / "bad.3mf").exists()


def test_lattice_stl_is_declared_lossy_with_respect_to_the_source(
    tmp_path: Path,
) -> None:
    meshio = pytest.importorskip("meshio")
    volume = _schwarz_p_cell()
    with pytest.raises(SurfaceLossyOperationError, match="lossy_with_respect_to_source"):
        export_lattice_solid(tmp_path / "cell.stl", volume, SurfaceExportPolicy(METER))
    exported = export_lattice_solid(
        tmp_path / "cell.stl", volume, SurfaceExportPolicy(METER, allow_lossy=True)
    )
    assert exported.lossy
    assert exported.source_losses == ("stl_lossy_with_respect_to_source",)
    assert "stl_cannot_retain_units_or_provenance" in exported.export.report.losses
    model, _ = lattice_solid_surface(volume)
    decoded = meshio.read(tmp_path / "cell.stl")
    # ASCII STL keeps every binary64 coordinate; the source evidence is gone.
    np.testing.assert_array_equal(decoded.points, np.asarray(model.mesh.coordinates))
