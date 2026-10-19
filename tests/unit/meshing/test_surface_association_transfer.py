#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellGeometrySpec
from phydrax.geometry import certify_source_fidelity, PreparedSurfaceSourceSupport
from phydrax.meshing import (
    CellMeshingResult,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    MeshLineage,
    SurfaceAssociationTransfer,
)
from phydrax.meshing._surface_association_transfer import (
    prepare_surface_curve_witness,
    PreparedSurfaceCurveWitness,
)
from tests._support.surface_association import (
    prepare_surface_transfer_fixture,
    SurfaceTransferFixture,
)
from tests._support.surface_metric import surface_resources


M = phx.meshing


@pytest.fixture(scope="module")
def surface_transfer_fixture() -> SurfaceTransferFixture:
    return prepare_surface_transfer_fixture()


def _association(result: CellMeshingResult, dimension: int) -> GeometryAssociation:
    entity_set = result.mesh.entity_set(dimension).entity_set_id
    return next(
        value for value in result.associations if value.target_entity_set_id == entity_set
    )


def test_serial_bisection_preserves_surface_support_and_fidelity(
    surface_transfer_fixture: SurfaceTransferFixture,
) -> None:
    domain, root, charts, boundary = (
        surface_transfer_fixture.domain,
        surface_transfer_fixture.result,
        surface_transfer_fixture.charts,
        surface_transfer_fixture.boundary,
    )
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
        assert set(after.source_entities.entity_ids()) <= strata
        old = dict(
            zip(
                np.asarray(before.target_global_ids).tolist(),
                before.source_entities.entity_ids(),
                strict=True,
            )
        )
        new = dict(
            zip(
                np.asarray(after.target_global_ids).tolist(),
                after.source_entities.entity_ids(),
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
    # The 64*eps allowance bounds the measured binary64 projection deviation
    # through this one-level restriction, scaled by the source domain extent.
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
    with pytest.raises(ValueError, match="SurfaceAssociation.lineage"):
        transfer.propagate(
            root,
            MeshLineage(
                target.mesh.topology_id, target.mesh.topology_id, lineage.entities
            ),
            target.mesh,
            geometry=target.geometry,
        )


@pytest.fixture(scope="module")
def surface_curve_witness(
    surface_transfer_fixture: SurfaceTransferFixture,
) -> PreparedSurfaceCurveWitness:
    root = surface_transfer_fixture
    with surface_resources() as (_, budget):
        return prepare_surface_curve_witness(
            PreparedSurfaceSourceSupport(
                root.domain, root.result, root.charts, root.boundary
            ),
            root.result,
            budget=budget,
        )


def test_feature_interval_eligibility_retains_curve_parameters(
    surface_curve_witness: PreparedSurfaceCurveWitness,
) -> None:
    witness = surface_curve_witness
    dimensions = np.asarray(witness.source_dimensions)
    row = np.flatnonzero(dimensions == 1)[0]
    kinds = np.repeat(dimensions[row : row + 1], 2)
    indices = np.repeat(np.asarray(witness.source_indices)[row : row + 1], 2)
    parameters = np.repeat(
        np.asarray(witness.source_parameters)[row : row + 1], 2, axis=0
    )
    curve = int(indices[0])
    owner = witness.support.domain.curve_uses(curve)[0][1]
    parameters[:, 0] = (owner.first, owner.last)
    strict = witness.edge_interval(kinds, indices, parameters)
    assert witness.admissible_edge_interval(kinds, indices, parameters) == strict
    assert strict == (curve, owner.first, owner.last)


def test_feature_interval_eligibility_refuses_patch_tokens_without_exception(
    surface_curve_witness: PreparedSurfaceCurveWitness,
) -> None:
    kinds = np.asarray((2, 2), dtype=np.int64)
    indices = np.zeros((2,), dtype=np.int64)
    parameters = np.zeros((2, 2), dtype=np.float64)
    assert (
        surface_curve_witness.admissible_edge_interval(kinds, indices, parameters) is None
    )
    with pytest.raises(ValueError, match="SurfaceCurveWitness.edge_interval"):
        surface_curve_witness.edge_interval(kinds, indices, parameters)


def test_feature_interval_eligibility_refuses_collapsed_curve_interval(
    surface_curve_witness: PreparedSurfaceCurveWitness,
) -> None:
    witness = surface_curve_witness
    dimensions = np.asarray(witness.source_dimensions)
    row = np.flatnonzero(dimensions == 1)[0]
    kinds = np.repeat(dimensions[row : row + 1], 2)
    indices = np.repeat(np.asarray(witness.source_indices)[row : row + 1], 2)
    parameters = np.repeat(
        np.asarray(witness.source_parameters)[row : row + 1], 2, axis=0
    )
    assert witness.admissible_edge_interval(kinds, indices, parameters) is None
    with pytest.raises(ValueError, match="SurfaceCurveWitness.edge_interval"):
        witness.edge_interval(kinds, indices, parameters)
