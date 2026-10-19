from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing import (
    LearnedMeshProposer,
    mesh_proposal_scope,
    MeshCoordinateProposal,
    MeshMarkingProposal,
    MeshMetricProposal,
    MeshProposalFeatures,
    MeshProposalSafetyPolicy,
    MeshSizeProposal,
    prepare_mesh_proposal,
    project_mesh_proposal,
)
from tests._ported_models import full_port, in_order, PortedAffine
from tests._support.cell_meshes import proposal_source


# Fixed numerical qualification bounds from test_proposals.py at d2dfcb941:
# sparse transfer/coordinate projection, metric normalization, and edge lengths.
_TRANSFER_ROUNDOFF = 1e-12
_METRIC_ROUNDOFF = 1e-10
_METRIC_EDGE_ROUNDOFF = 1e-9


def _policy(source: Any, **kwargs: Any) -> Any:
    return MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
        **kwargs,
    )


def _assert_graded(sizes: Any, points: Any, edges: Any, growth: Any) -> None:
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    for first, second in ((0, 1), (1, 0)):
        bound = sizes[edges[:, first]] + (growth - 1.0) * lengths
        assert np.all(sizes[edges[:, second]] <= bound * (1.0 + _METRIC_ROUNDOFF))


def test_marking_projection_preserves_protected_cell_and_reports_withheld_marks() -> None:
    source = proposal_source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), np.ones(4), proposer_id="estimator"
    )
    policy = _policy(
        source, protected_scopes=(mesh_proposal_scope(source, 2, np.asarray((10,))),)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)
    result = transaction.commit(source)
    adaptation = transaction.adaptation
    assert adaptation is not None
    assert adaptation.transition is not None
    assert adaptation.transition.vertex_stencil is not None
    projection = transaction.projection
    requested = np.asarray(projection.proposal.scope.entity_ids)[
        np.asarray(projection.proposal.values) > 0.0
    ]
    np.testing.assert_array_equal(requested, (10, 20, 30, 40))
    np.testing.assert_array_equal(projection.marked_cell_ids, (20, 30, 40))
    # Protection is projected before native execution, with separate refusal evidence.
    np.testing.assert_array_equal(projection.protected_cell_ids, (10,))
    np.testing.assert_array_equal(projection.policy.protected_scopes[0].entity_ids, (10,))
    assert projection.proposal.proposer_id == "estimator"
    assert projection.proposal.evaluation_id is None

    cells = np.asarray(result.mesh.blocks[0].global_ids)
    row = int(np.flatnonzero(cells == 10)[0])
    np.testing.assert_array_equal(
        result.mesh.blocks[0].vertices[row], source.mesh.blocks[0].vertices[0]
    )
    np.testing.assert_array_equal(result.mesh.coordinates[:5], source.mesh.coordinates)
    interpolated = adaptation.transition.vertex_stencil.apply(
        source.mesh.vertex_global_ids, source.mesh.coordinates[:, 0]
    )
    np.testing.assert_allclose(interpolated, result.mesh.coordinates[:, 0])
    assert transaction.commit(source, accept=False) is source
    assert source.mesh.blocks[0].global_ids.size == 4


def test_marking_projection_respects_native_capacity_and_replays_deterministically() -> (
    None
):
    source = proposal_source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        np.asarray((2.0, 5.0, 5.0, 1.0), dtype=np.float64),
        proposer_id="estimator",
    )
    limits = phx.meshing.MeshingLimits(maximum_cells=6, maximum_vertices=6)
    policy = _policy(source, limits=limits)
    transaction = prepare_mesh_proposal(source, proposal, policy)

    np.testing.assert_array_equal(transaction.projection.marked_cell_ids, (20,))
    np.testing.assert_array_equal(transaction.projection.protected_cell_ids, ())
    assert transaction.commit(source).audit.entity_counts[-1] <= 6
    assert transaction.trusted_result.audit.vertex_count <= 6
    assert (
        project_mesh_proposal(source, proposal, policy).projection_id
        == transaction.projection.projection_id
    )


def test_proposal_projection_and_commit_refuse_stale_revisions() -> None:
    source = proposal_source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        np.asarray((1, 0, 0, 0), dtype=np.float64),
        proposer_id="model",
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
    with pytest.raises(ValueError, match="MeshProposal.scope"):
        project_mesh_proposal(refreshed, proposal, _policy(refreshed))
    fresh_proposal = MeshMarkingProposal(
        refreshed,
        mesh_proposal_scope(refreshed, 2),
        np.asarray((1, 0, 0, 0), dtype=np.float64),
        proposer_id="model",
    )
    with pytest.raises(ValueError, match="MeshProposalSafetyPolicy.source_result_id"):
        project_mesh_proposal(refreshed, fresh_proposal, policy)
    with pytest.raises(ValueError, match="MeshProposalTransaction.current"):
        transaction.commit(refreshed)


def test_size_projection_clamps_and_grades_scoped_vertex_values() -> None:
    source = proposal_source()
    scope = mesh_proposal_scope(source, 0)
    proposal = MeshSizeProposal(
        source,
        scope,
        np.asarray((-5.0, 2.0, 0.2, 50.0, 1.0), dtype=np.float64),
        proposer_id="model",
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_gradation=1.1)
    )
    field = transaction.projection.size_field
    assert field is not None
    assert field.sample_points is not None
    assert field.sample_entity_ids is not None
    sizes = np.asarray(field.values)
    assert np.all(sizes >= 0.1 - _TRANSFER_ROUNDOFF)
    assert np.all(sizes <= 2.0 + _TRANSFER_ROUNDOFF)
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    np.testing.assert_allclose(field.sample_points, source.mesh.coordinates[order])
    np.testing.assert_array_equal(field.sample_entity_ids, scope.entity_ids)
    inverse = np.argsort(order)
    assert isinstance(source.mesh.connectivity, phx.discretization.PolygonalConnectivity)
    edges = inverse[np.asarray(source.mesh.connectivity.edges)]
    _assert_graded(sizes, np.asarray(field.sample_points), edges, 1.1)
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id
    np.testing.assert_array_equal(proposal.values, (-5.0, 2.0, 0.2, 50.0, 1.0))


def test_metric_projection_reports_symmetric_and_indefinite_repair() -> None:
    source = proposal_source()
    raw = np.asarray((((-3.0, 4.0), (0.0, 2.0)),) * 5)
    raw[0] = ((1.0e4, 0.0), (0.0, 4.0))
    proposal = MeshMetricProposal(
        source, mesh_proposal_scope(source, 0), raw, proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_anisotropy=2.0, maximum_gradation=1.05)
    )
    assert transaction.projection.metric is not None
    values = np.asarray(transaction.projection.metric.values)
    eigenvalues = np.linalg.eigvalsh(values)
    np.testing.assert_allclose(values, values.swapaxes(-1, -2), atol=_TRANSFER_ROUNDOFF)
    assert np.all(eigenvalues >= 0.25 - _METRIC_ROUNDOFF)
    assert np.all(eigenvalues <= 100.0 + _METRIC_ROUNDOFF)
    assert np.all(eigenvalues[:, -1] / eigenvalues[:, 0] <= 4.0 + _METRIC_ROUNDOFF)
    sizes = np.linalg.det(values) ** (-0.25)
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    assert isinstance(source.mesh.connectivity, phx.discretization.PolygonalConnectivity)
    edges = np.argsort(order)[np.asarray(source.mesh.connectivity.edges)]
    _assert_graded(sizes, np.asarray(source.mesh.coordinates)[order], edges, 1.05)
    evidence = transaction.projection.metric_evidence
    assert evidence is not None
    assert evidence.symmetrized_count == 4 and evidence.projected_tensor_count == 4
    assert evidence.passed
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id


def test_anisotropic_proposal_uses_native_metric_adaptation_and_transfer() -> None:
    source = proposal_source()
    anisotropic = np.broadcast_to(np.diag((1.0 / 0.15**2, 1.0 / 0.6**2)), (5, 2, 2))
    proposal = MeshMetricProposal(
        source, mesh_proposal_scope(source, 0), anisotropic, proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source,
        proposal,
        _policy(source, maximum_anisotropy=20.0, maximum_gradation=2.0),
    )
    adaptation = transaction.adaptation
    assert adaptation is not None
    assert adaptation.metric is not None
    assert adaptation.transfer is not None
    result = transaction.commit(source)
    points = np.asarray(result.mesh.coordinates)
    assert isinstance(result.mesh.connectivity, phx.discretization.PolygonalConnectivity)
    edges = np.asarray(result.mesh.connectivity.edges)
    delta = np.abs(points[edges[:, 1]] - points[edges[:, 0]])
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            adaptation.metric.values[
                np.argsort(np.argsort(np.asarray(result.mesh.vertex_global_ids)))
            ],
            points,
            edges,
        )
    )

    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    assert adaptation.status.converged
    assert np.all(lengths <= np.sqrt(2.0) + _METRIC_EDGE_ROUNDOFF)
    assert np.max(delta[:, 0]) < np.max(delta[:, 1])
    np.testing.assert_allclose(
        adaptation.transfer.apply(source.mesh.coordinates),
        points,
        atol=_TRANSFER_ROUNDOFF,
    )


def test_coordinate_proposal_preserves_fixed_vertices_and_displacement_limit() -> None:
    source = proposal_source()
    scope = mesh_proposal_scope(source, 0, np.asarray((5, 7)))
    proposal = MeshCoordinateProposal(
        source,
        scope,
        np.asarray(((-2.0, 4.0), (9.0, 9.0)), dtype=np.float64),
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
    assert 0.0 < displacement <= 0.1 + _TRANSFER_ROUNDOFF
    assert result.mesh.topology_id == source.mesh.topology_id
    assert result.quality.minimum_mean_ratio > 0
    assert transaction.optimization is not None
    assert (
        transaction.optimization.final_objective
        < transaction.optimization.initial_objective
    )


def test_safety_audit_refuses_commit_without_discarding_native_result() -> None:
    source = proposal_source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        np.asarray((1, 0, 0, 0), dtype=np.float64),
        proposer_id="model",
    )
    policy = _policy(
        source, audit_policy=phx.meshing.CellMeshAuditPolicy(minimum_mean_ratio=0.999)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)

    assert not transaction.admissible
    assert transaction.trusted_result.audit.passed
    with pytest.raises(ValueError, match="MeshProposalTransaction.admissible"):
        transaction.commit(source)
    assert transaction.commit(source, accept=False) is source


def test_size_proposal_refuses_nonfinite_values() -> None:
    source = proposal_source()
    with pytest.raises(ValueError, match="MeshProposal.values"):
        MeshSizeProposal(
            source,
            mesh_proposal_scope(source, 0),
            np.asarray((1, 1, np.nan, 1, 1), dtype=np.float64),
            proposer_id="model",
        )


def test_proposal_scope_refuses_unknown_scientific_ids() -> None:
    source = proposal_source()
    with pytest.raises(ValueError, match="MeshProposal.scope.entity_ids"):
        mesh_proposal_scope(source, 0, np.asarray((1234,)))


def test_coordinate_proposal_refuses_changed_coordinate_contract() -> None:
    source = proposal_source()
    other_frame = phx.SpatialCoordinateContract(
        source.coordinate_contract.length_unit, reference_frame="other"
    )
    with pytest.raises(ValueError, match="MeshCoordinateProposal.coordinate_contract"):
        MeshCoordinateProposal(
            source,
            mesh_proposal_scope(source, 0, np.asarray((5,))),
            np.asarray(((0.5, 0.5),), dtype=np.float64),
            other_frame,
            proposer_id="model",
        )


def test_empty_capacity_projection_preserves_identical_source() -> None:
    source = proposal_source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), np.ones(4), proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source,
        proposal,
        _policy(source, limits=phx.meshing.MeshingLimits(maximum_cells=4)),
    )

    assert transaction.projection.marked_cell_ids.size == 0
    assert transaction.adaptation is None
    assert transaction.commit(source) is source


def test_size_gradation_reaches_across_long_connected_strip() -> None:
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

    assert projection.size_field is not None
    np.testing.assert_allclose(projection.size_field.values, 0.1, atol=_TRANSFER_ROUNDOFF)


def test_learned_proposal_keeps_author_and_realization_evidence() -> None:
    source = proposal_source()
    # One-hot cell features in sorted global-ID order; the model scores the
    # protected cell 10 highest.
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 2),
        np.eye(4, dtype=np.float64),
        feature_ids=(
            "cell-priority/0",
            "cell-priority/1",
            "cell-priority/2",
            "cell-priority/3",
        ),
        feature_owner_id="learned-marker-fixture",
    )
    proposer = LearnedMeshProposer(
        _Score((9.0, 1.0, 3.0, 2.0)),
        kind="marking",
        proposer_id="learned-marker",
    )
    proposal = proposer.propose(source, features)
    policy = _policy(
        source, protected_scopes=(mesh_proposal_scope(source, 2, np.asarray((10,))),)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)

    assert isinstance(proposal, MeshMarkingProposal)
    assert proposal.proposer_id == "learned-marker"
    assert proposal.evaluation_id is not None
    assert len(proposal.evaluation_id) == 64
    assert all(character in "0123456789abcdef" for character in proposal.evaluation_id)
    first_realization = proposer.propose(source, features, key=jax.random.key(71))
    other_realization = proposer.propose(source, features, key=jax.random.key(72))
    assert (
        first_realization.proposer_id == other_realization.proposer_id == "learned-marker"
    )
    assert first_realization.evaluation_id is not None
    assert other_realization.evaluation_id is not None
    assert first_realization.evaluation_id != other_realization.evaluation_id
    assert first_realization.proposal_id != other_realization.proposal_id
    np.testing.assert_array_equal(transaction.projection.marked_cell_ids, (20, 30, 40))
    np.testing.assert_array_equal(transaction.projection.protected_cell_ids, (10,))
    np.testing.assert_array_equal(proposal.values, (9.0, 1.0, 3.0, 2.0))
    # Identical values and evaluation evidence project identically: the learned
    # route has no trusted path of its own.
    untrusted = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        np.asarray((9.0, 1.0, 3.0, 2.0), dtype=np.float64),
        proposer_id=proposal.proposer_id,
        evaluation_id=proposal.evaluation_id,
    )
    assert (
        project_mesh_proposal(source, untrusted, policy).projection_id
        == transaction.projection.projection_id
    )
    result = transaction.commit(source)
    row = int(np.flatnonzero(np.asarray(result.mesh.blocks[0].global_ids) == 10)[0])
    np.testing.assert_array_equal(
        result.mesh.blocks[0].vertices[row], source.mesh.blocks[0].vertices[0]
    )
    contract = proposer.component_contract()
    assert contract.authority is phx.ComponentAuthority.DECISION


def test_learned_size_proposal_uses_shared_safety_projection() -> None:
    source = proposal_source()
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 0),
        np.eye(5, dtype=np.float64),
        feature_ids=(
            "size-channel/0",
            "size-channel/1",
            "size-channel/2",
            "size-channel/3",
            "size-channel/4",
        ),
        feature_owner_id="learned-size-fixture",
    )
    proposer = LearnedMeshProposer(
        _Score((-5.0, 2.0, 0.2, 50.0, 1.0)),
        kind="size",
        proposer_id="sizer",
    )
    transaction = prepare_mesh_proposal(
        source,
        proposer.propose(source, features),
        _policy(source, maximum_gradation=1.1),
    )

    assert transaction.projection.size_field is not None
    sizes = np.asarray(transaction.projection.size_field.values)
    assert np.all(sizes >= 0.1 - _TRANSFER_ROUNDOFF) and np.all(
        sizes <= 2.0 + _TRANSFER_ROUNDOFF
    )


def test_learned_metric_proposer_requires_spatial_dimension() -> None:
    with pytest.raises(ValueError, match="LearnedMeshProposer.spatial_dimension"):
        LearnedMeshProposer(_Score((1.0,)), kind="metric", proposer_id="metric")


def test_learned_proposer_requires_declared_port_binding() -> None:
    source = proposal_source()
    owner = phx.ModelPorts(
        inputs=(full_port("cell.indicators", (4,)),),
        outputs=(full_port("cell.marking-score", ()),),
    )
    model = PortedAffine(
        owner, out_size="scalar", weight=jnp.asarray([[9.0, 1.0, 3.0, 2.0]])
    )
    with pytest.raises(ValueError, match="mesh-proposer'.*owner_ports"):
        LearnedMeshProposer(model, kind="marking", proposer_id="ported")
    with pytest.raises(ValueError, match="input ports must declare the event shapes"):
        LearnedMeshProposer(
            model,
            kind="marking",
            proposer_id="ported",
            ports=phx.ModelPorts(
                inputs=(full_port("cell.indicators", (3,)),), outputs=owner.outputs
            ),
        )

    proposer = LearnedMeshProposer(
        model,
        kind="marking",
        proposer_id="ported",
        ports=owner,
        port_mapping=in_order(owner, owner),
    )
    evidence = proposer.component_contract().port_binding
    assert evidence is not None
    assert evidence.inputs == ((owner.inputs[0].port_id,) * 2,)
    assert evidence.unverified == ()
    proposal = proposer.propose(
        source,
        MeshProposalFeatures(
            source,
            mesh_proposal_scope(source, 2),
            np.eye(4, dtype=np.float64),
            feature_ids=(
                "cell-priority/0",
                "cell-priority/1",
                "cell-priority/2",
                "cell-priority/3",
            ),
            feature_owner_id="ported-marker-fixture",
        ),
    )
    np.testing.assert_array_equal(proposal.values, (9.0, 1.0, 3.0, 2.0))


def test_learned_proposal_refuses_nonfinite_model_output() -> None:
    source = proposal_source()
    # One-hot cell features in sorted global-ID order; the model scores the
    # protected cell 10 highest.
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 2),
        np.eye(4, dtype=np.float64),
        feature_ids=(
            "cell-priority/0",
            "cell-priority/1",
            "cell-priority/2",
            "cell-priority/3",
        ),
        feature_owner_id="learned-marker-fixture",
    )
    nonfinite = LearnedMeshProposer(
        _Score((np.nan, 0.0, 0.0, 0.0)),
        kind="marking",
        proposer_id="nan",
    )
    with pytest.raises(ValueError, match="MeshProposal.values"):
        nonfinite.propose(source, features)


class _AbstractScore(phx.AbstractArrayModel):
    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(self, weight: Any, bias: Any = 0.0) -> None:
        self.weight = jnp.asarray(weight, dtype=jnp.float64)
        self.bias = jnp.asarray(bias, dtype=jnp.float64)
        self.in_size = int(self.weight.size)
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        return self.weight @ x + self.bias


class _Score(_AbstractScore):
    def __init__(self, weight: Any, bias: Any = 0.0) -> None:
        super().__init__(weight, bias)


def test_learned_marker_trains_against_dual_weighted_residual_targets() -> None:
    source = proposal_source()
    residual = jnp.asarray([[1.0, 0.5], [0.2, 0.1], [2.0, -1.0], [0.0, 0.3]])
    correction = jnp.asarray([[0.5, 0.5], [1.0, 1.0], [0.25, -0.5], [1.0, 0.0]])
    targets = phx.discretization.fem.local_dual_weighted_residual(
        residual, correction
    ).absolute
    features = jnp.concatenate((residual, correction), axis=1)
    model = _Score(jnp.zeros(4))
    proposer = LearnedMeshProposer(
        model,
        kind="marking",
        proposer_id="dwr-marker",
    )
    parameters, model_state, fixed = phx.partition_parameters(proposer)

    def loss(parameters: Any) -> Any:
        bound = phx.combine_parameters(parameters, model_state, fixed)
        return jnp.mean((bound.evaluate(features) - targets) ** 2)

    gradient = jax.grad(loss)(parameters)
    assert {id(leaf) for leaf in jax.tree.leaves(parameters)} == {
        id(model.weight),
        id(model.bias),
    }
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    bound_features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 2),
        features,
        feature_ids=(
            "residual/0",
            "residual/1",
            "adjoint-correction/0",
            "adjoint-correction/1",
        ),
        feature_owner_id="local-dual-weighted-residual",
    )
    assert isinstance(proposer.propose(source, bound_features), MeshMarkingProposal)


class _StochasticScore(_AbstractScore):
    def __init__(self, weight: Any, bias: Any = 0.0) -> None:
        super().__init__(weight, bias)

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        if key is None:
            raise ValueError(
                f"_StochasticScore.key must supply an explicit realization; got {key!r}"
            )
        return self.weight @ x + self.bias + jax.random.normal(key, (), dtype=x.dtype)


def test_stochastic_proposal_addresses_survive_partition_and_row_permutation() -> None:
    model = _StochasticScore(jnp.asarray((1.0, 2.0)))
    proposer = LearnedMeshProposer(
        model,
        kind="marking",
        proposer_id="stochastic-marker",
    )
    rows = jnp.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=jnp.float64)
    identifiers = jnp.asarray((7, 7 + 2**32, -7), dtype=jnp.int64)
    key = jax.random.key(71)
    full = proposer.evaluate_addressed(
        rows, identifiers, key, address_id="source/physical-owner"
    )
    permutation = jnp.asarray((2, 0, 1))
    permuted = proposer.evaluate_addressed(
        rows[permutation],
        identifiers[permutation],
        key,
        address_id="source/physical-owner",
    )
    np.testing.assert_array_equal(permuted, full[permutation])
    partitioned = jnp.concatenate(
        tuple(
            proposer.evaluate_addressed(
                rows[index : index + 1],
                identifiers[index : index + 1],
                key,
                address_id="source/physical-owner",
            )
            for index in range(3)
        )
    )
    np.testing.assert_array_equal(partitioned, full)
    noise = full - rows @ model.weight
    assert noise[0] != noise[1]
    foreign = proposer.evaluate_addressed(
        rows, identifiers, key, address_id="new-source/physical-owner"
    )
    assert not np.array_equal(full, foreign)


def test_stochastic_learned_proposal_replays_only_explicit_realization() -> None:
    source = proposal_source()
    scope = mesh_proposal_scope(source, 2)
    features = MeshProposalFeatures(
        source,
        scope,
        np.ones((4, 1)),
        feature_ids=("physical-residual",),
        feature_owner_id="prepared-reaction-estimator",
    )
    proposer = LearnedMeshProposer(
        _StochasticScore((1.0,)),
        kind="marking",
        proposer_id="stochastic-marker",
    )
    with pytest.raises(ValueError, match="_StochasticScore.key"):
        proposer.propose(source, features)
    first = proposer.propose(source, features, key=jax.random.key(71))
    replay = proposer.propose(source, features, key=jax.random.key(71))
    other = proposer.propose(source, features, key=jax.random.key(72))
    np.testing.assert_array_equal(first.values, replay.values)
    assert first.proposal_id == replay.proposal_id
    assert first.proposal_id != other.proposal_id


def test_learned_model_update_invalidates_proposal_provenance_without_changing_scores() -> (
    None
):
    source = proposal_source()
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 2),
        np.zeros((4, 1)),
        feature_ids=("residual",),
        feature_owner_id="prepared-estimator",
    )
    model = _Score((1.0,))
    proposer = LearnedMeshProposer(model, kind="marking", proposer_id="trained-marker")
    updated_model = eqx.tree_at(lambda owner: owner.weight, model, jnp.asarray((2.0,)))
    updated = eqx.tree_at(lambda owner: owner.model, proposer, updated_model)
    first, second = proposer.propose(source, features), updated.propose(source, features)
    np.testing.assert_array_equal(first.values, second.values)
    assert proposer.model_id != updated.model_id
    assert first.proposal_id != second.proposal_id
