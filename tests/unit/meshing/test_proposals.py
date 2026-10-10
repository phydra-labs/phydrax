from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._proposals import (
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


def _source() -> Any:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((7, 3, 9, 1, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


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
        assert np.all(sizes[edges[:, second]] <= bound * (1.0 + 1e-10))


def test_proposals_scenario_1() -> None:
    source = _source()
    proposal = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 2), np.ones(4), proposer_id="estimator"
    )
    policy = _policy(
        source, protected_scopes=(mesh_proposal_scope(source, 2, np.asarray((10,))),)
    )
    transaction = prepare_mesh_proposal(source, proposal, policy)
    result = transaction.commit(source)
    adaptation = transaction.adaptation

    cells = np.asarray(result.mesh.blocks[0].global_ids)
    row = int(np.flatnonzero(cells == 10)[0])
    np.testing.assert_array_equal(
        result.mesh.blocks[0].vertices[row], source.mesh.blocks[0].vertices[0]
    )
    np.testing.assert_array_equal(result.mesh.coordinates[:5], source.mesh.coordinates)
    # ty: ignore[unresolved-attribute]
    interpolated = adaptation.transition.vertex_stencil.apply(
        source.mesh.vertex_global_ids, source.mesh.coordinates[:, 0]
    )
    np.testing.assert_allclose(interpolated, result.mesh.coordinates[:, 0])
    assert transaction.commit(source, accept=False) is source
    assert source.mesh.blocks[0].global_ids.size == 4
    source = _source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        # ty: ignore[invalid-argument-type]
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
    source = _source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        # ty: ignore[invalid-argument-type]
        (1, 0, 0, 0),
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
    with pytest.raises(ValueError, match="revision"):
        project_mesh_proposal(refreshed, proposal, _policy(refreshed))
    fresh_proposal = MeshMarkingProposal(
        refreshed,
        mesh_proposal_scope(refreshed, 2),
        # ty: ignore[invalid-argument-type]
        (1, 0, 0, 0),
        proposer_id="model",
    )
    with pytest.raises(ValueError, match="revision"):
        project_mesh_proposal(refreshed, fresh_proposal, policy)
    with pytest.raises(ValueError, match="revision"):
        transaction.commit(refreshed)


def test_proposals_scenario_2() -> None:
    source = _source()
    scope = mesh_proposal_scope(source, 0)
    proposal = MeshSizeProposal(
        source,
        scope,
        # ty: ignore[invalid-argument-type]
        (-5.0, 2.0, 0.2, 50.0, 1.0),
        proposer_id="model",
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_gradation=1.1)
    )
    field = transaction.projection.size_field
    # ty: ignore[unresolved-attribute]
    sizes = np.asarray(field.values)
    assert np.all(sizes >= 0.1 - 1e-12)
    assert np.all(sizes <= 2.0 + 1e-12)
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(field.sample_points, source.mesh.coordinates[order])
    # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(field.sample_entity_ids, scope.entity_ids)
    inverse = np.argsort(order)
    edges = inverse[np.asarray(source.mesh.connectivity.edges)]
    # ty: ignore[unresolved-attribute]
    _assert_graded(sizes, np.asarray(field.sample_points), edges, 1.1)
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id
    np.testing.assert_array_equal(proposal.values, (-5.0, 2.0, 0.2, 50.0, 1.0))
    source = _source()
    raw = np.asarray((((-3.0, 4.0), (0.0, 2.0)),) * 5)
    raw[0] = ((1.0e4, 0.0), (0.0, 4.0))
    proposal = MeshMetricProposal(
        source, mesh_proposal_scope(source, 0), raw, proposer_id="model"
    )
    transaction = prepare_mesh_proposal(
        source, proposal, _policy(source, maximum_anisotropy=2.0, maximum_gradation=1.05)
    )
    # ty: ignore[unresolved-attribute]
    values = np.asarray(transaction.projection.metric.values)
    eigenvalues = np.linalg.eigvalsh(values)
    np.testing.assert_allclose(values, values.swapaxes(-1, -2), atol=1e-12)
    assert np.all(eigenvalues >= 0.25 - 1e-10)
    assert np.all(eigenvalues <= 100.0 + 1e-10)
    assert np.all(eigenvalues[:, -1] / eigenvalues[:, 0] <= 4.0 + 1e-10)
    sizes = np.linalg.det(values) ** (-0.25)
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    edges = np.argsort(order)[np.asarray(source.mesh.connectivity.edges)]
    _assert_graded(sizes, np.asarray(source.mesh.coordinates)[order], edges, 1.05)
    evidence = transaction.projection.metric_evidence
    # ty: ignore[unresolved-attribute]
    assert evidence.symmetrized_count == 4 and evidence.projected_tensor_count == 4
    # ty: ignore[unresolved-attribute]
    assert evidence.passed
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id
    source = _source()
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
    result = transaction.commit(source)
    points = np.asarray(result.mesh.coordinates)
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(result.mesh.connectivity.edges)
    delta = np.abs(points[edges[:, 1]] - points[edges[:, 0]])
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            # ty: ignore[unresolved-attribute]
            adaptation.metric.values[
                np.argsort(np.argsort(np.asarray(result.mesh.vertex_global_ids)))
            ],
            points,
            edges,
        )
    )

    # ty: ignore[unresolved-attribute]
    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    # ty: ignore[unresolved-attribute]
    assert adaptation.status.converged
    assert np.all(lengths <= np.sqrt(2.0) + 1e-9)
    assert np.max(delta[:, 0]) < np.max(delta[:, 1])
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        adaptation.transfer.apply(source.mesh.coordinates),
        points,
        atol=1e-12,
    )
    source = _source()
    scope = mesh_proposal_scope(source, 0, np.asarray((5, 7)))
    proposal = MeshCoordinateProposal(
        source,
        scope,
        # ty: ignore[invalid-argument-type]
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
        # ty: ignore[unresolved-attribute]
        transaction.optimization.final_objective
        # ty: ignore[unresolved-attribute]
        < transaction.optimization.initial_objective
    )


def test_proposals_scenario_3() -> None:
    source = _source()
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        # ty: ignore[invalid-argument-type]
        (1, 0, 0, 0),
        proposer_id="model",
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
    source = _source()
    with pytest.raises(ValueError, match="finite"):
        MeshSizeProposal(
            source,
            mesh_proposal_scope(source, 0),
            # ty: ignore[invalid-argument-type]
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
            # ty: ignore[invalid-argument-type]
            ((0.5, 0.5),),
            other_frame,
            proposer_id="model",
        )
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
    assert transaction.adaptation is None
    assert transaction.commit(source) is source


def test_proposals_scenario_4() -> None:
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

    # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(projection.size_field.values, 0.1, atol=1e-12)
    source = _source()
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
    np.testing.assert_array_equal(proposal.values, (9.0, 1.0, 3.0, 2.0))
    # Identical values from any other proposer project identically: the learned
    # route has no trusted path of its own.
    untrusted = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        # ty: ignore[invalid-argument-type]
        (9.0, 1.0, 3.0, 2.0),
        proposer_id=proposal.proposer_id,
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

    nonfinite = LearnedMeshProposer(
        _Score((np.nan, 0.0, 0.0, 0.0)),
        kind="marking",
        proposer_id="nan",
    )
    with pytest.raises(ValueError, match="finite"):
        nonfinite.propose(source, features)
    source = _source()
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

    # ty: ignore[unresolved-attribute]
    sizes = np.asarray(transaction.projection.size_field.values)
    assert np.all(sizes >= 0.1 - 1e-12) and np.all(sizes <= 2.0 + 1e-12)
    with pytest.raises(ValueError, match="spatial_dimension"):
        LearnedMeshProposer(_Score((1.0,)), kind="metric", proposer_id="metric")
    source = _source()
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
    # ty: ignore[unresolved-attribute]
    assert evidence.inputs == ((owner.inputs[0].port_id,) * 2,)
    # ty: ignore[unresolved-attribute]
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


class _AbstractScore(phx.AbstractArrayModel):
    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

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
    source = _source()
    residual = jnp.asarray([[1.0, 0.5], [0.2, 0.1], [2.0, -1.0], [0.0, 0.3]])
    correction = jnp.asarray([[0.5, 0.5], [1.0, 1.0], [0.25, -0.5], [1.0, 0.0]])
    targets = phx.discretization.fem.local_dual_weighted_residual(
        residual, correction
    ).absolute
    features = jnp.concatenate((residual, correction), axis=1)
    proposer = LearnedMeshProposer(
        _Score(jnp.zeros(4)),
        kind="marking",
        proposer_id="dwr-marker",
    )
    parameters, model_state, fixed = phx.partition_parameters(proposer)

    def loss(parameters: Any) -> Any:
        bound = phx.combine_parameters(parameters, model_state, fixed)
        return jnp.mean((bound.evaluate(features) - targets) ** 2)

    gradient = jax.grad(loss)(parameters)
    assert {id(leaf) for leaf in jax.tree.leaves(parameters)} == {
        # ty: ignore[unresolved-attribute]
        id(proposer.model.weight),
        # ty: ignore[unresolved-attribute]
        id(proposer.model.bias),
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
            raise ValueError("Stochastic scoring requires an explicit realization.")
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
    source = _source()
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
    with pytest.raises(ValueError, match="explicit realization"):
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
    source = _source()
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
