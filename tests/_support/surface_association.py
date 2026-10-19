# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Typed cylindrical sources for surface lineage and feature eligibility contracts."""

from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax.geometry import (
    CylinderPatch,
    LineCurve,
    MeshingDomain,
    MeshingDomainBoundarySource,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.meshing import CellMeshingResult


class SurfaceTransferFixture(phx.StrictModule):
    domain: MeshingDomain
    result: CellMeshingResult
    charts: np.ndarray
    boundary: MeshingDomainBoundarySource


def quarter_cylinder_domain() -> MeshingDomain:
    """Declare the unit quarter-cylinder chart with four source curve uses."""
    # Recipe provenance: phydrax/meshing/qualification/_surface_sources.py:cylinder_sheet_surface,
    # unit radius and height. This retains declaration data and source IDs;
    # curve and patch evaluation remain with the public geometry constructors.
    angle = 0.5 * np.pi
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    loop = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, angle),
        PatchCurveUse(1, LineCurve((angle, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((0.0, 1.0), (1.0, 0.0)), angle, 0.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 1.0, 0.0),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(3, 0),
        ),
        4,
        source_id="sheet",
        source_revision="r1",
    )


def prepare_surface_transfer_fixture() -> SurfaceTransferFixture:
    """Publish a bounded cylindrical mesh with its retained source chart bank."""
    domain = quarter_cylinder_domain()
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(len(domain.patches), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.75, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            phx.meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    source = result.surface_source
    if source is None:
        raise ValueError(
            "surface_transfer_fixture.source must retain source charts; got no surface source"
        )
    source.require_root(result.mesh, result.geometry)
    return SurfaceTransferFixture(
        domain=domain,
        result=result,
        charts=source.root_parameters,
        boundary=source.boundary_source,
    )
