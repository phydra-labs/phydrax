"""TPMS lattice fill, specimen and solid export consumer types."""

from pathlib import Path
from typing import assert_type

import phydrax as phx


def periodic_lattice_fill(surface: phx.geometry.implicit.PeriodicImplicitSurface) -> None:
    volume = phx.meshing.mesh_periodic_trigonometric_region(surface)
    assert_type(volume, phx.meshing.PeriodicImplicitVolume)
    assert_type(volume.mesh, phx.discretization.CellMesh)
    assert_type(volume.result, phx.meshing.CellMeshingResult)
    assert_type(volume.quotient_region_components, int)
    assert_type(volume.measure, float)
    phx.meshing.mesh_periodic_trigonometric_region(surface, maximum_edge_length="0.1")  # ty: ignore[invalid-argument-type]


def lattice_specimen(region: phx.geometry.TrigonometricLevelSet) -> None:
    envelope = phx.geometry.Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5)
    surface = phx.meshing.construct_lattice_specimen_surface(
        phx.geometry.Intersection(region, envelope), source_id="specimen"
    )
    assert_type(surface, phx.meshing.LatticeSpecimenSurface)
    assert_type(surface.implicit_face_distance, float)
    volume = phx.meshing.mesh_lattice_specimen(surface)
    assert_type(volume, phx.meshing.LatticeSpecimenVolume)
    assert_type(volume.certified_solid, bool)
    assert_type(volume.porosity, float)
    phx.meshing.mesh_lattice_specimen(volume)  # ty: ignore[invalid-argument-type]


def lattice_solid_export(
    volume: phx.meshing.LatticeSpecimenVolume | phx.meshing.PeriodicImplicitVolume,
    path: Path,
) -> None:
    surface = phx.geometry.surface
    policy = surface.SurfaceExportPolicy(phx.units.MILLIMETER)
    exported = phx.meshing.export_lattice_solid(path, volume, policy)
    assert_type(exported, phx.meshing.LatticeSolidExport)
    assert_type(exported.certificate, surface.SurfaceValidityCertificate)
    assert_type(exported.source_losses, tuple[str, ...])
    assert_type(exported.lossy, bool)
    assert_type(exported.export.certificate, surface.SurfaceValidityCertificate | None)
    model, fidelity = phx.meshing.lattice_solid_surface(volume)
    assert_type(model, surface.SurfaceModel)
    assert_type(fidelity, tuple[tuple[str, float], ...])
    result = surface.export_surface(
        path, model, policy, file_format=surface.SurfaceFileFormat.THREE_MF
    )
    assert_type(result, surface.SurfaceExportResult)
    phx.meshing.export_lattice_solid(path, volume)  # ty: ignore[missing-argument]
