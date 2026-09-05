import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._proposals import (
    mesh_proposal_scope,
    MeshCoordinateProposal,
    MeshMarkingProposal,
    MeshMetricProposal,
    MeshProposalSafetyPolicy,
    MeshSizeProposal,
    prepare_mesh_proposal,
    project_mesh_proposal,
)


def _source():
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((7, 3, 9, 1, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _policy(source, **kwargs):
    return MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
        **kwargs,
    )


def test_marking_preserves_protected_cell_through_conformity_closure_and_transfers_fields():
    source = _source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), np.ones(4), proposer_id="estimator"
    )
    policy = _policy(
        source, protected_scopes=(mesh_proposal_scope(source, 2, np.asarray((10,))),)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)
    result = transaction.commit(source)

    np.testing.assert_array_equal(transaction.projection.marked_cell_ids, (30,))
    cells = np.asarray(result.mesh.blocks[0].global_ids)
    row = int(np.flatnonzero(cells == 10)[0])
    np.testing.assert_array_equal(
        result.mesh.blocks[0].vertices[row], source.mesh.blocks[0].vertices[0]
    )
    np.testing.assert_array_equal(result.mesh.coordinates[:5], source.mesh.coordinates)
    assert transaction.transition is not None
    interpolated = transaction.transition.vertex_stencil.apply(
        source.mesh.vertex_global_ids, source.mesh.coordinates[:, 0]
    )
    np.testing.assert_allclose(interpolated, result.mesh.coordinates[:, 0])
    assert transaction.commit(source, accept=False) is source
    assert source.mesh.blocks[0].global_ids.size == 4


def test_mark_capacity_uses_priority_then_global_id_and_accounts_for_neighbor_closure():
    source = _source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        (2.0, 5.0, 5.0, 1.0),
        proposer_id="estimator",
    )
    limits = phx.meshing.MeshingLimits(maximum_cells=6, maximum_vertices=6)
    policy = _policy(source, limits=limits)
    transaction = prepare_mesh_proposal(source, proposal, policy)

    np.testing.assert_array_equal(transaction.projection.marked_cell_ids, (20,))
    assert transaction.commit(source).audit.entity_counts[-1] <= 6
    assert transaction.trusted_result.audit.vertex_count <= 6
    assert (
        project_mesh_proposal(source, proposal, policy).projection_id
        == transaction.projection.projection_id
    )


def test_revision_changes_reject_proposal_policy_and_commit_even_with_unchanged_topology():
    source = _source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), (1, 0, 0, 0), proposer_id="model"
    )
    policy = _policy(source)
    transaction = prepare_mesh_proposal(source, proposal, policy)
    refreshed = phx.meshing.certify_cell_mesh(
        source.mesh.with_coordinates(
            source.mesh.coordinates + 0.01, numeric_version="refreshed"
        ),
        source.coordinate_contract,
    )
    assert refreshed.mesh.topology_id == source.mesh.topology_id
    with pytest.raises(ValueError, match="revision"):
        project_mesh_proposal(refreshed, proposal, _policy(refreshed))
    fresh_proposal = MeshMarkingProposal(
        refreshed, mesh_proposal_scope(refreshed, 2), (1, 0, 0, 0), proposer_id="model"
    )
    with pytest.raises(ValueError, match="revision"):
        project_mesh_proposal(refreshed, fresh_proposal, policy)
    with pytest.raises(ValueError, match="revision"):
        transaction.commit(refreshed)


def test_size_projection_clamps_and_grades_in_sorted_global_id_order_then_refines():
    source = _source()
    scope = mesh_proposal_scope(source, 0)
    proposal = MeshSizeProposal(
        source, scope, (-5.0, 2.0, 0.2, 50.0, 1.0), proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_gradation=1.1)
    )
    field = transaction.projection.size_field
    sizes = np.asarray(field.values)
    assert np.all(sizes >= 0.1 - 1e-12)
    assert np.all(sizes <= 2.0 + 1e-12)
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    np.testing.assert_allclose(field.sample_points, source.mesh.coordinates[order])
    inverse = np.argsort(order)
    edges = np.asarray(source.mesh.connectivity.edges)
    ratios = sizes[inverse[edges[:, 0]]] / sizes[inverse[edges[:, 1]]]
    assert np.all(ratios <= 1.1 + 1e-12)
    assert np.all(ratios >= 1 / 1.1 - 1e-12)
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id
    np.testing.assert_array_equal(proposal.values, (-5.0, 2.0, 0.2, 50.0, 1.0))


def test_metric_projection_repairs_indefinite_asymmetric_tensors_with_bounded_gradation():
    source = _source()
    raw = np.asarray((((-3.0, 4.0), (0.0, 2.0)),) * 5)
    raw[0] = ((1.0e4, 0.0), (0.0, 4.0))
    proposal = MeshMetricProposal(
        source, mesh_proposal_scope(source, 0), raw, proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_anisotropy=2.0, maximum_gradation=1.05)
    )
    values = np.asarray(transaction.projection.metric.values)
    eigenvalues = np.linalg.eigvalsh(values)
    np.testing.assert_allclose(values, values.swapaxes(-1, -2), atol=1e-12)
    assert np.all(eigenvalues >= 0.25 - 1e-10)
    assert np.all(eigenvalues <= 100.0 + 1e-10)
    assert np.all(eigenvalues[:, -1] / eigenvalues[:, 0] <= 4.0 + 1e-10)
    sizes = np.linalg.det(values) ** (-0.25)
    inverse = np.argsort(np.argsort(np.asarray(source.mesh.vertex_global_ids)))
    edges = inverse[np.asarray(source.mesh.connectivity.edges)]
    ratios = sizes[edges[:, 0]] / sizes[edges[:, 1]]
    assert np.all(ratios <= 1.05 + 1e-10)
    assert np.all(ratios >= 1 / 1.05 - 1e-10)
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id


def test_coordinate_targets_and_optimization_respect_scope_protection_and_trust_region():
    source = _source()
    scope = mesh_proposal_scope(source, 0, np.asarray((5, 7)))
    proposal = MeshCoordinateProposal(
        source,
        scope,
        ((-2.0, 4.0), (9.0, 9.0)),
        source.coordinate_contract,
        proposer_id="model",
    )
    policy = _policy(
        source,
        protected_scopes=(mesh_proposal_scope(source, 0, np.asarray((7,))),),
        coordinate_bounds=((0.0, 0.0), (1.0, 1.0)),
        maximum_optimization_iterations=10,
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)
    result = transaction.commit(source)
    np.testing.assert_array_equal(
        result.mesh.coordinates[:4], source.mesh.coordinates[:4]
    )
    displacement = np.linalg.norm(result.mesh.coordinates[4] - source.mesh.coordinates[4])
    assert 0.0 < displacement <= 0.1 + 1e-12
    assert result.mesh.topology_id == source.mesh.topology_id
    assert result.quality.minimum_mean_ratio > 0
    assert (
        transaction.optimization.final_objective
        < transaction.optimization.initial_objective
    )


def test_candidate_that_fails_trusted_safety_audit_cannot_commit_and_rolls_back():
    source = _source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), (1, 0, 0, 0), proposer_id="model"
    )
    policy = _policy(
        source, audit_policy=phx.meshing.CellMeshAuditPolicy(minimum_mean_ratio=0.999)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)

    assert not transaction.admissible
    assert transaction.trusted_result.audit.passed
    with pytest.raises(ValueError, match="admissible"):
        transaction.commit(source)
    assert transaction.commit(source, accept=False) is source


def test_nonfinite_unknown_entities_and_coordinate_frame_mismatch_are_rejected():
    source = _source()
    with pytest.raises(ValueError, match="finite"):
        MeshSizeProposal(
            source,
            mesh_proposal_scope(source, 0),
            (1, 1, np.nan, 1, 1),
            proposer_id="model",
        )
    with pytest.raises(ValueError, match="unknown"):
        mesh_proposal_scope(source, 0, np.asarray((1234,)))
    other_frame = phx.SpatialCoordinateContract(
        source.coordinate_contract.length_unit, reference_frame="other"
    )
    with pytest.raises(ValueError, match="coordinate contract"):
        MeshCoordinateProposal(
            source,
            mesh_proposal_scope(source, 0, np.asarray((5,))),
            ((0.5, 0.5),),
            other_frame,
            proposer_id="model",
        )


def test_exhausted_capacity_produces_explicit_unchanged_source_without_refinement():
    source = _source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), np.ones(4), proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source,
        proposal,
        _policy(source, limits=phx.meshing.MeshingLimits(maximum_cells=4)),
    )

    assert transaction.projection.marked_cell_ids.size == 0
    assert transaction.transition is None
    assert transaction.commit(source) is source


def test_unit_gradation_reaches_beyond_sixty_four_adjacency_hops():
    points = np.asarray(
        [(float(column), float(row)) for column in range(72) for row in range(2)]
    )
    cells = np.asarray(
        [
            cell
            for column in range(71)
            for cell in (
                (2 * column, 2 * column + 2, 2 * column + 1),
                (2 * column + 2, 2 * column + 3, 2 * column + 1),
            )
        ],
        dtype=np.int32,
    )
    source = phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(points, cells),
        phx.SpatialCoordinateContract.si(),
    )
    sizes = np.full(points.shape[0], 2.0)
    sizes[0] = 0.1
    proposal = MeshSizeProposal(
        source, mesh_proposal_scope(source, 0), sizes, proposer_id="model"
    )
    projection = project_mesh_proposal(
        source, proposal, _policy(source, maximum_gradation=1.0)
    )

    np.testing.assert_allclose(projection.size_field.values, 0.1, atol=1e-12)
