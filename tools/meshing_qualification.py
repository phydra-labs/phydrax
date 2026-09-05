#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import phydrax as phx


def qualify_manifold() -> dict[str, object]:
    import manifold3d

    def cube(offset):
        arrays = manifold3d.Manifold.cube().translate(offset).to_mesh64()
        return phx.geometry.SurfaceModel.from_triangles(
            arrays.vert_properties[:, :3],
            arrays.tri_verts,
            phx.geometry.SurfaceMetadata(
                source_id=str(offset),
                source_revision="0",
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                provenance=("qualification",),
            ),
        )

    result = phx.meshing.ManifoldProvider().execute(
        cube((0.0, 0.0, 0.0)),
        cube((0.5, 0.0, 0.0)),
        phx.meshing.SurfaceBooleanOperation.DIFFERENCE,
    )
    points = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    volume = float(
        np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6.0
    )
    if not np.isclose(volume, 0.5, atol=1e-12) or not result.audit.passed:
        raise RuntimeError("Manifold difference failed solid-volume qualification.")
    covered = np.concatenate(
        [np.asarray(item.target_global_ids) for item in result.associations]
    )
    if not np.array_equal(
        np.sort(covered), np.sort(np.asarray(result.mesh.entity_set(2).entity_ids))
    ):
        raise RuntimeError("Manifold result lacks complete source-face ancestry.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "volume": volume,
        "source_associations": len(result.associations),
        "audit_passed": result.audit.passed,
    }


def qualify_volume(
    provider_name: str, executable: str, extractor: str
) -> dict[str, object]:
    contract = phx.SpatialCoordinateContract.si()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    expected_volume = 1.0 / 6.0
    surface = phx.geometry.SurfaceModel.from_triangles(
        points,
        faces,
        phx.geometry.SurfaceMetadata(
            source_id="qualification-tetrahedron",
            source_revision="0",
            coordinate_contract=contract,
            provenance=("qualification",),
        ),
    )
    if provider_name == "mmg":
        mesh = phx.discretization.CellMesh.from_tetrahedra(
            points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        )
        provider = phx.meshing.MmgProvider()
        metric = phx.meshing.MeshMetricField(
            provider.vertex_scope(mesh),
            np.tile(np.eye(3) * 16.0, (4, 1, 1)),
            minimum_size=0.25,
            maximum_size=0.25,
        )
        result = provider.adapt(mesh, metric, contract)
    elif provider_name == "ftetwild":
        provider = phx.meshing.FTetWildProvider(
            phx.meshing.FTetWildOptions(envelope_distance=0.005, maximum_iterations=20)
        )
        scope = provider.whole_scope(surface)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            deterministic=False,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope, 0.3, strength=phx.meshing.SizeControlStrength.SOFT
                ),
            ),
        )
        result = provider.plan(surface, specification).execute()
    elif provider_name == "vorocrust":
        result = phx.meshing.VoroCrustProvider(executable, extractor).execute(
            surface, phx.meshing.VoroCrustOptions(1.0)
        )
    else:
        import build123d as bd

        with TemporaryDirectory(prefix="phydrax-gmsh-qualification-") as temporary:
            path = Path(temporary) / "cube.step"
            bd.export_step(bd.Box(1.0, 1.0, 1.0), path, unit=bd.Unit.MM)
            source = phx.geometry.BRepSource(
                phx.geometry.import_brep(
                    path, linear_deflection=0.05, angular_deflection=0.2
                )
            )
            provider = phx.meshing.GmshProvider(
                phx.meshing.GmshOptions(
                    coordinate_contract=phx.SpatialCoordinateContract(
                        phx.units.MILLIMETER
                    )
                )
            )
            scope = provider.whole_scope(source, 3)
            specification = phx.meshing.VolumeMeshingSpec(
                phx.meshing.CellMeshingTarget(
                    3,
                    3,
                    phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
                    geometry_order=2,
                ),
                scope,
                phx.meshing.VolumeFillStrategy.SIMPLEX,
                size_controls=(
                    phx.meshing.UniformSizeControl(scope, 0.3, maximum_growth_rate=2.0),
                ),
            )
            result = provider.plan(source, specification).execute()
            expected_volume = 1.0
    volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isclose(volume, expected_volume, rtol=0.03) or not result.audit.passed:
        raise RuntimeError(f"{provider_name} failed enclosed-volume qualification.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "cells": result.mesh.entity_set(3).count,
        "volume": volume,
        "audit_passed": result.audit.passed,
        "quality_scope": result.audit.quality_scope,
    }


def qualify_poisson() -> dict[str, object]:
    count = 400
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    angles = np.arange(count) * np.pi * (3.0 - np.sqrt(5.0))
    normals = np.column_stack(
        (np.sqrt(1.0 - z * z) * np.cos(angles), np.sqrt(1.0 - z * z) * np.sin(angles), z)
    )
    source = phx.meshing.OrientedPointCloud(
        normals,
        normals,
        phx.SpatialCoordinateContract.si(),
        source_id="sphere",
        source_revision="0",
    )
    result = phx.meshing.PoissonProvider().execute(
        source, phx.meshing.PoissonReconstructionSpec(depth=5)
    )
    error = float(
        np.max(np.abs(np.linalg.norm(np.asarray(result.mesh.coordinates), axis=1) - 1.0))
    )
    if error > 0.1 or not result.audit.passed:
        raise RuntimeError("Poisson reconstruction failed sphere qualification.")
    return {
        "provider": result.provider.name,
        "maximum_radius_error": error,
        "audit_passed": result.audit.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("manifold", "mmg", "ftetwild", "gmsh", "vorocrust", "poisson"),
        default="manifold",
    )
    parser.add_argument("--executable", default="vc_mesh")
    parser.add_argument("--extractor", default="phydrax-vorocrust")
    args = parser.parse_args()
    if args.provider == "manifold":
        result = qualify_manifold()
    elif args.provider == "poisson":
        result = qualify_poisson()
    else:
        result = qualify_volume(args.provider, args.executable, args.extractor)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
