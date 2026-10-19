#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Publish native spherical material fixtures with their source-bound atlas."""

import numpy as np

import phydrax as phx
from phydrax.geometry import MeshingDomain, SphereMaterialCellAtlas
from phydrax.geometry._sphere_material_atlas import sphere_material_atlas_from_result
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.qualification import sphere_surface


_PREPARATIONS: dict[
    float, tuple[MeshingDomain, CellMeshingResult, SphereMaterialCellAtlas]
] = {}


def accepted_sphere_material(
    size: float,
    *,
    reuse_preparation: bool = True,
) -> tuple[MeshingDomain, CellMeshingResult, SphereMaterialCellAtlas]:
    """Publish a sphere preparation, optionally reusing an accepted fixture.

    Cold callers execute the same producer and atlas preparation with a fresh
    request allowance. Reused fixtures retain the first preparation's evidence.
    """
    if not isinstance(reuse_preparation, bool):
        raise TypeError(
            "sphere_material.reuse_preparation must be bool; got nonboolean cache selector"
        )
    if reuse_preparation and size in _PREPARATIONS:
        return _PREPARATIONS[size]
    domain = sphere_surface().domain
    meshing = phx.meshing
    scope = meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    deviation = 0.25 * size * size
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            meshing.ProtectedFeature(
                scope,
                meshing.FeatureKind.SURFACE,
                maximum_deviation=deviation,
            ),
        ),
        limits=meshing.MeshingLimits(
            maximum_vertices=20_000,
            maximum_edges=160_000,
            maximum_faces=160_000,
            maximum_cells=160_000,
            maximum_work_units=640_000,
            maximum_geometry_queries=1_280_000,
            maximum_scratch_bytes=81_920_000,
        ),
    )
    result = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("parametric_surface"))
        .plan(
            meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    prepared = (
        domain,
        result,
        sphere_material_atlas_from_result(domain, result, maximum_fidelity=deviation),
    )
    if reuse_preparation:
        _PREPARATIONS[size] = prepared
    return prepared
