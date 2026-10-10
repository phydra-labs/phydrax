#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import numpy as np
import pytest

import phydrax as phx
from examples._native_surface_sources import cylinder_sheet
from phydrax.discretization import CellGeometrySpec
from phydrax.geometry._mesh_certificates import certify_source_fidelity
from phydrax.geometry._surface_source_support import PreparedSurfaceSourceSupport
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    SurfaceAssociationTransfer,
)
from phydrax.meshing._lineage import MeshLineage


M = phx.meshing


def _root() -> tuple[Any, Any, Any, Any]:
    """Actual public surface result and its retained authoritative chart bank."""
    domain = cylinder_sheet().domain
    scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(len(domain.patches), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            # This is a lineage/certification fixture, not a dense surface campaign.
            M.UniformSizeControl(scope, 0.75, strength=M.SizeControlStrength.SOFT),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("parametric_surface"))
        .plan(
            M.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    source = result.surface_source
    if source is None:
        raise ValueError(
            "The public native surface must retain its original source chart premise."
        )
    source.require_root(result.mesh, result.geometry)
    return domain, result, source.root_parameters, source.boundary_source


def _association(result: Any, dimension: int) -> Any:
    entity_set = result.mesh.entity_set(dimension).entity_set_id
    return next(
        value for value in result.associations if value.target_entity_set_id == entity_set
    )


def test_serial_bisection_publishes_certified_original_surface_authority() -> None:
    domain, root, charts, boundary = _root()
    transfer = SurfaceAssociationTransfer(
        PreparedSurfaceSourceSupport(domain, root, charts, boundary)
    )
    adapted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            root,
            M.MarkedMeshAdaptation(
                np.asarray(root.mesh.blocks[0].global_ids, dtype=np.int64)[::3]
            ),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=M.BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=transfer,
            ),
        )
    )
    target = adapted.target
    assert target.mesh.entity_set(2).count > root.mesh.entity_set(2).count

    strata = {
        domain.entity_id(dimension, index)
        for dimension in range(3)
        for index in range(len(domain.source_indices[dimension]))
    }
    for dimension in range(3):
        before, after = _association(root, dimension), _association(target, dimension)
        assert after.association_kind is GeometryAssociationKind.SURFACE
        assert after.provenance is GeometryAssociationProvenance.LINEAGE
        assert bool(np.all(np.asarray(after.resolved)))
        assert set(after.source_entity_ids) <= strata
        old = dict(
            zip(
                np.asarray(before.target_global_ids).tolist(),
                before.source_entity_ids,
                strict=True,
            )
        )
        new = dict(
            zip(
                np.asarray(after.target_global_ids).tolist(),
                after.source_entity_ids,
                strict=True,
            )
        )
        assert all(
            new[identifier] == old[identifier] for identifier in set(old) & set(new)
        )
    preserved = set(np.asarray(_association(root, 0).target_global_ids).tolist())
    assert preserved <= set(
        np.asarray(_association(target, 0).target_global_ids).tolist()
    )

    certification, root_certification = target.certification, root.certification
    assert certification is not None and root_certification is not None
    fidelity, original = certification.fidelity, root_certification.fidelity
    assert fidelity is not None and original is not None
    assert fidelity.status == "certified"
    assert (fidelity.mesh_to_source_semantics, fidelity.source_to_mesh_semantics) == (
        "certified",
        "certified",
    )
    # Subdivision adds only the measured binary64 deviation from the exact
    # restricted root chords.
    slack = 64 * np.finfo(np.float64).eps * domain.scale
    assert fidelity.mesh_to_source_upper <= original.mesh_to_source_upper + slack
    assert fidelity.source_to_mesh_upper <= original.source_to_mesh_upper + slack

    source = certification.request.source
    assert source is not None
    moved = target.mesh.with_coordinates(
        np.asarray(target.mesh.coordinates) + np.asarray((0.0, 0.0, 1e-3)),
        numeric_version="moved",
    )
    refused = certify_source_fidelity(
        moved, CellGeometrySpec.affine(moved), source, tolerance=fidelity.tolerance
    )
    assert refused.status != "certified"

    lineage = adapted.lineage
    assert lineage is not None
    with pytest.raises(ValueError, match="topology lineage"):
        transfer.propagate(
            root,
            MeshLineage(
                target.mesh.topology_id, target.mesh.topology_id, lineage.entities
            ),
            target.mesh,
            geometry=target.geometry,
        )
