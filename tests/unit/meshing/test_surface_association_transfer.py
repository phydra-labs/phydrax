#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing._surface_association_transfer as owner
from phydrax.discretization import CellGeometrySpec
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
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


@pytest.mark.parametrize("reversed_orientation", (False, True))
def test_trim_chart_retains_coincident_nominal_boundary_occurrences(
    reversed_orientation: bool,
) -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    if reversed_orientation:
        points = points[[0, 2, 1]]
    charts = np.stack((points, points))
    cells = np.asarray(((0, 1, 2), (0, 1, 2)), dtype=np.int64)
    coedges = np.asarray(
        (((0, 0, 1), (0, 1, 1), (0, 2, 1)), ((2, 0, -1), (2, 1, -1), (2, 2, -1))),
        dtype=np.int32,
    )
    with surface_resources() as (_, budget):
        chart = owner._prepare_trim_chart(charts, cells, coedges, budget)
    assert chart.coordinates.shape == (6, 2)
    assert len(chart.edges) == 6
    assert all(len(uses) == 1 for uses in chart.edges.values())
    sides = [uses[0][3] for uses in chart.edges.values()]
    assert sides == ([2, 1, 0] if reversed_orientation else [0, 1, 2]) * 2


def test_trim_chart_rejects_missing_nominal_boundary_occurrences() -> None:
    charts = np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),))
    with surface_resources() as (_, budget):
        with pytest.raises(ValueError, match="label every noninterior"):
            owner._prepare_trim_chart(
                charts,
                np.asarray(((0, 1, 2),)),
                np.full((1, 3, 3), -1, dtype=np.int32),
                budget,
            )


@pytest.mark.parametrize("chart_offset", (0.0, 1.0e-12))
def test_curve_cover_consumes_nominal_cell_sides_without_chart_equality(
    surface_curve_witness: PreparedSurfaceCurveWitness,
    chart_offset: float,
) -> None:
    witness = surface_curve_witness
    support = witness.support
    domain = support.domain
    patches = np.asarray(support.root_patches)
    charts = SurfaceChartWitness(
        geometry_id=support.root_geometry_id,
        topology_id=witness.source_mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=support.root_cell_ids,
        patches=patches,
        charts=np.asarray(support.root_parameters) + chart_offset,
        geometry_entity_ids=tuple(domain.entity_id(2, int(patch)) for patch in patches),
        occurrence_paths=tuple(
            domain.source_occurrences[2][int(patch)] for patch in patches
        ),
    )
    ids = np.asarray(charts.cell_global_ids)
    coedges = witness.cell_coedges(ids, np.asarray(charts.patches))
    with surface_resources() as (_, budget):
        cover = owner.prepare_surface_curve_chart_cover(
            witness,
            witness.source_mesh,
            charts,
            np.asarray(witness.source_dimensions),
            np.asarray(witness.source_indices),
            np.asarray(witness.source_parameters),
            source_chart_cover=witness.boundary_source,
            cell_coedges=coedges,
            budget=budget,
        )
    np.testing.assert_array_equal(cover.boundary_cell_ids, ids)
    np.testing.assert_array_equal(cover.boundary_occurrences, coedges)
    for record in cover.chart_triangulations:
        for loop, ordinal, first, last in record[5]:
            use = domain.patches[record[0]].loops[int(loop)][int(ordinal)]
            assert first != last
            assert isinstance(use, phx.geometry.PatchCurveUse)


def test_curve_occurrence_lookup_uses_nominal_ids_and_rejects_missing_rows(
    surface_curve_witness: PreparedSurfaceCurveWitness,
) -> None:
    witness = surface_curve_witness
    ids = np.asarray(witness.boundary_source.boundary_cell_ids)
    expected = np.asarray(witness.boundary_source.boundary_occurrences)
    np.testing.assert_array_equal(witness.cell_coedges(ids[::-1]), expected[::-1])
    with pytest.raises(ValueError, match="repeat cell IDs"):
        witness.cell_coedges(np.repeat(ids[:1], 2))
    with pytest.raises(ValueError, match="include every source cell"):
        witness.cell_coedges(np.asarray((int(np.max(ids)) + 100,)))
    with pytest.raises(ValueError, match="owning patches"):
        witness.cell_coedges(ids, np.full(ids.shape, -1))
