from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
)
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._bisection import BisectionEvidence
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
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
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_implicit import (
    normalize_analytic_implicit_source,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeImplicitSource
from tests._ported_models import full_port, in_order, PortedAffine
from tests.unit.meshing.test_device_tetra_metric import _source as _tetrahedral_source
from tests.unit.meshing.test_native_implicit_routes import _geometry_and_grid


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


def _features(
    source: Any,
    dimension: int,
    values: Any,
    /,
    *,
    feature_ids: tuple[str, ...],
) -> MeshProposalFeatures:
    return MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, dimension),
        values,
        feature_ids=feature_ids,
        feature_owner_id="test-proposal-features",
    )


def _assert_exact_constant_stencil(stencil: Any, /) -> None:
    assert stencil is not None and stencil.preserves_constants
    sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    exact = stencil.exact_weights
    assert len(exact) == valid.shape[0]
    assert all(len(row) == valid.shape[1] for row in exact)
    for row in range(valid.shape[0]):
        assert np.any(valid[row])
        active_sources = sources[row, valid[row]]
        assert np.unique(active_sources).size == active_sources.size
        assert (
            sum(
                (
                    weight
                    for weight, is_valid in zip(exact[row], valid[row], strict=True)
                    if is_valid
                ),
                Fraction(0),
            )
            == 1
        )


def _assert_graded(sizes: Any, points: Any, edges: Any, growth: Any) -> None:
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    for first, second in ((0, 1), (1, 0)):
        bound = sizes[edges[:, first]] + (growth - 1.0) * lengths
        assert np.all(sizes[edges[:, second]] <= bound * (1.0 + 1e-10))


@pytest.fixture(scope="module")
def _implicit_volume_source() -> tuple[Any, Any, Any]:
    geometry, _ = _geometry_and_grid()
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        tube_radius=0.5,
        source_id="proposal-implicit-profile",
        source_revision="proposal-implicit-r1",
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08,
        root_tolerance=1.0e-8,
    )
    source = normalize_analytic_implicit_source(profile, policy, contract)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=0.8,
    )
    options = NativeMeshingOptions(
        "implicit_adaptive_tetrahedral",
        implicit_policy=policy,
        volume_schedule=NativeVolumeSchedule(),
    )
    result = (
        NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=contract,
        )
        .execute()
    )
    surface_specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(),
        maximum_deviation=0.8,
    )
    surface_options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=policy,
    )
    surface_result = (
        NativeMeshingProvider(surface_options)
        .plan(
            source,
            surface_specification,
            coordinate_contract=contract,
        )
        .execute()
    )
    return profile, result, surface_result


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

    np.testing.assert_array_equal(
        transaction.projection.marked_cell_ids, (10, 20, 30, 40)
    )
    # ty: ignore[unresolved-attribute]
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.PARTIAL
    # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(adaptation.evidence.rejected_refinement_ids, (10,))
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
    if adaptation is None:
        pytest.fail("Metric proposal did not prepare an adaptation result.")
    _assert_exact_constant_stencil(adaptation.stencil)
    metric = adaptation.metric
    if metric is None:
        pytest.fail("Metric proposal did not produce a target metric.")
    transfer = adaptation.transfer
    result = transaction.commit(source)
    points = np.asarray(result.mesh.coordinates)
    connectivity = result.mesh.connectivity
    if not all(
        block.cell_kind == "triangle" and np.asarray(block.vertices).shape[1] == 3
        for block in result.mesh.blocks
    ):
        pytest.fail("Metric proposal target is not triangular.")
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(connectivity.edges)
    delta = np.abs(points[edges[:, 1]] - points[edges[:, 0]])
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            metric.values[
                np.argsort(np.argsort(np.asarray(result.mesh.vertex_global_ids)))
            ],
            points,
            edges,
        )
    )

    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    assert adaptation.status.converged
    assert np.all(lengths <= np.sqrt(2.0) + 1e-9)
    assert np.max(delta[:, 0]) < np.max(delta[:, 1])
    if transfer is None:
        pytest.fail(
            "Metric proposal did not produce its requested interpolation transfer."
        )
    np.testing.assert_allclose(
        transfer.apply(source.mesh.coordinates),
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
    features = _features(
        source,
        2,
        np.eye(4),
        feature_ids=(
            "cell-indicator:10",
            "cell-indicator:20",
            "cell-indicator:30",
            "cell-indicator:40",
        ),
    )
    proposer = LearnedMeshProposer(
        # ty: ignore[invalid-argument-type, missing-argument]
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
    np.testing.assert_array_equal(proposal.values, (9.0, 1.0, 3.0, 2.0))
    # Evaluation identity remains separate from the numerical projection.
    untrusted = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        # ty: ignore[invalid-argument-type]
        (9.0, 1.0, 3.0, 2.0),
        proposer_id="learned-marker",
    )
    assert untrusted.evaluation_id is None
    assert proposal.proposal_id != untrusted.proposal_id
    manual_projection = project_mesh_proposal(source, untrusted, policy)
    np.testing.assert_array_equal(
        manual_projection.marked_cell_ids, transaction.projection.marked_cell_ids
    )
    # The protected cell's mark is rejected by native bisection, never executed.
    adaptation = transaction.adaptation
    if adaptation is None:
        pytest.fail("Marking proposal did not prepare an adaptation result.")
    assert isinstance(adaptation.evidence, BisectionEvidence)
    assert 10 in np.asarray(adaptation.evidence.rejected_refinement_ids)
    result = transaction.commit(source)
    _assert_exact_constant_stencil(adaptation.stencil)
    row = int(np.flatnonzero(np.asarray(result.mesh.blocks[0].global_ids) == 10)[0])
    np.testing.assert_array_equal(
        result.mesh.blocks[0].vertices[row], source.mesh.blocks[0].vertices[0]
    )
    contract = proposer.component_contract()
    assert contract.authority is phx.ComponentAuthority.DECISION

    nonfinite = LearnedMeshProposer(
        # ty: ignore[invalid-argument-type, missing-argument]
        _Score((np.nan, 0.0, 0.0, 0.0)),
        kind="marking",
        proposer_id="nan",
    )
    with pytest.raises(ValueError, match="finite"):
        nonfinite.propose(source, features)
    source = _source()
    features = _features(
        source,
        0,
        np.eye(5),
        feature_ids=tuple(f"vertex-indicator:{index}" for index in range(5)),
    )
    proposer = LearnedMeshProposer(
        # ty: ignore[invalid-argument-type, missing-argument]
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
        # ty: ignore[invalid-argument-type, missing-argument]
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
    proposal_features = _features(
        source,
        2,
        np.eye(4),
        feature_ids=(
            "cell-indicator:10",
            "cell-indicator:20",
            "cell-indicator:30",
            "cell-indicator:40",
        ),
    )
    proposal = proposer.propose(source, proposal_features)
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
    pass


class _CoordinateScore(phx.AbstractArrayModel):
    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, weight: Any, bias: Any) -> None:
        self.weight = jnp.asarray(weight, dtype=jnp.float64)
        self.bias = jnp.asarray(bias, dtype=jnp.float64)
        self.in_size = self.weight.shape[1]
        self.out_size = (self.weight.shape[0],)

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        del key
        return self.weight @ x + self.bias


def test_learned_coordinate_evaluation_binds_source_features_and_model() -> None:
    source = _source()
    scope = mesh_proposal_scope(source, 0)
    feature_ids = ("coordinate-feature:0", "coordinate-feature:1")
    feature_values = np.zeros((scope.entity_ids.size, len(feature_ids)))
    features = MeshProposalFeatures(
        source,
        scope,
        feature_values,
        feature_ids=feature_ids,
        feature_owner_id="coordinate-feature-owner",
    )
    proposer = LearnedMeshProposer(
        _CoordinateScore(np.eye(2), np.zeros(2)),
        kind="coordinate",
        proposer_id="learned-relocator",
        spatial_dimension=2,
    )
    proposal = proposer.propose(source, features)

    assert isinstance(proposal, MeshCoordinateProposal)
    assert proposal.evaluation_id is not None
    np.testing.assert_array_equal(proposal.values, feature_values)
    changed_features = MeshProposalFeatures(
        source,
        scope,
        feature_values + 1.0,
        feature_ids=feature_ids,
        feature_owner_id="coordinate-feature-owner",
    )
    changed = proposer.propose(source, changed_features)
    assert changed.proposal_id != proposal.proposal_id
    assert changed.evaluation_id != proposal.evaluation_id


def test_learned_marker_trains_against_dual_weighted_residual_targets() -> None:
    source = _source()
    residual = jnp.asarray([[1.0, 0.5], [0.2, 0.1], [2.0, -1.0], [0.0, 0.3]])
    correction = jnp.asarray([[0.5, 0.5], [1.0, 1.0], [0.25, -0.5], [1.0, 0.0]])
    targets = phx.discretization.fem.local_dual_weighted_residual(
        residual, correction
    ).absolute
    features = jnp.concatenate((residual, correction), axis=1)
    proposer = LearnedMeshProposer(
        # ty: ignore[invalid-argument-type, missing-argument]
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
    proposal_features = _features(
        source,
        2,
        features,
        feature_ids=("residual:0", "residual:1", "dual:0", "dual:1"),
    )
    assert isinstance(proposer.propose(source, proposal_features), MeshMarkingProposal)


def test_learned_features_cannot_be_reused_after_source_geometry_changes() -> None:
    source = _source()
    feature_rows = np.eye(5)
    features = _features(
        source,
        0,
        feature_rows,
        feature_ids=tuple(f"vertex-indicator:{index}" for index in range(5)),
    )
    moved_mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(source.mesh.coordinates) + np.asarray((0.05, 0.0)),
        np.asarray(source.mesh.blocks[0].vertices),
        vertex_global_ids=np.asarray(source.mesh.vertex_global_ids),
        cell_global_ids=np.asarray(source.mesh.blocks[0].global_ids),
    )
    moved = phx.meshing.certify_cell_mesh(moved_mesh, source.coordinate_contract)
    proposer = LearnedMeshProposer(
        # ty: ignore[invalid-argument-type, missing-argument]
        _Score(np.ones(5)),
        kind="size",
        proposer_id="source-bound-size-model",
    )

    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        proposer.propose(source, feature_rows)

    with pytest.raises(ValueError, match="stale for this source result"):
        proposer.propose(moved, features)


def test_tetrahedral_marking_size_and_metric_proposals_use_native_volume_routes() -> None:
    source = _tetrahedral_source()
    cell_ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
    scores = np.zeros(cell_ids.shape, dtype=np.float64)
    scores[[5, 0, 3]] = (2.0, 0.5, -1.0)
    marking = MeshMarkingProposal(
        source, mesh_proposal_scope(source, 3), scores, proposer_id="tetra-marker"
    )
    transaction = prepare_mesh_proposal(source, marking, _policy(source))
    np.testing.assert_array_equal(
        transaction.projection.marked_cell_ids, np.sort(cell_ids[[0, 5]])
    )
    adaptation = transaction.adaptation
    assert adaptation is not None
    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    assert adaptation.transition is not None
    _assert_exact_constant_stencil(adaptation.transition.vertex_stencil)
    refined = transaction.commit(source)
    assert refined.mesh.entity_set(3).count > cell_ids.size

    scope = mesh_proposal_scope(source, 0)
    count = scope.entity_ids.size
    order = np.argsort(np.asarray(source.mesh.vertex_global_ids))
    points = np.asarray(source.mesh.coordinates)[order]
    edges = np.argsort(order)[np.asarray(source.mesh.connectivity.edges)]
    sizes = np.full((count,), 0.3, dtype=np.float64)
    sizes[:2] = (50.0, -1.0)
    projection = project_mesh_proposal(
        source,
        MeshSizeProposal(source, scope, sizes, proposer_id="tetra-size"),
        _policy(source, maximum_gradation=1.2),
    )
    field = projection.size_field
    assert field is not None
    projected = np.asarray(field.values)
    assert np.all(projected >= 0.1 - 1e-12) and np.all(projected <= 2.0 + 1e-12)
    assert projected[1] == 0.1 and projected[0] < 2.0
    _assert_graded(projected, points, edges, 1.2)
    transaction = prepare_mesh_proposal(
        source,
        MeshSizeProposal(source, scope, np.full((count,), 0.3), proposer_id="tetra-size"),
        _policy(source),
    )
    adaptation = transaction.adaptation
    assert adaptation is not None and adaptation.metric is not None
    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id

    tensor = np.diag((1.0 / 0.25**2, 1.0 / 0.3**2, 1.0 / 0.35**2))
    raw = np.broadcast_to(tensor, (count, 3, 3)).copy()
    raw[0, 0, 1] += 1.0
    raw[1, 2, 2] = -4.0
    projection = project_mesh_proposal(
        source,
        MeshMetricProposal(source, scope, raw, proposer_id="tetra-metric"),
        _policy(source, maximum_anisotropy=4.0, maximum_gradation=1.5),
    )
    metric = projection.metric
    assert metric is not None
    values = np.asarray(metric.values)
    assert values.shape == (count, 3, 3)
    np.testing.assert_allclose(values, values.swapaxes(-1, -2), atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(values)
    assert np.all(eigenvalues >= 0.25 - 1e-10)
    assert np.all(eigenvalues <= 100.0 + 1e-10)
    assert np.all(eigenvalues[:, -1] / eigenvalues[:, 0] <= 16.0 + 1e-10)
    evidence = projection.metric_evidence
    assert evidence is not None and evidence.passed
    assert evidence.symmetrized_count == 1 and evidence.projected_tensor_count == 1
    transaction = prepare_mesh_proposal(
        source,
        MeshMetricProposal(
            source,
            scope,
            np.broadcast_to(tensor, (count, 3, 3)),
            proposer_id="tetra-metric",
        ),
        _policy(source, maximum_anisotropy=4.0),
    )
    adaptation = transaction.adaptation
    assert adaptation is not None and adaptation.metric is not None
    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D
    assert adaptation.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert transaction.commit(source).mesh.topology_id != source.mesh.topology_id


def test_implicit_marking_proposals_preserve_region_boundary_and_field(
    _implicit_volume_source: tuple[Any, Any, Any],
) -> None:
    profile, source, _ = _implicit_volume_source
    authority = source.certification.inputs.feature_source
    assert authority.authoring_profile.profile_id == profile.profile_id
    cell_ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 3, cell_ids),
        np.ones(cell_ids.shape, dtype=np.float64),
        proposer_id="implicit-marking-proposal",
    )
    transaction = prepare_mesh_proposal(source, proposal, _policy(source))
    adaptation = transaction.adaptation
    assert adaptation is not None
    target = transaction.commit(source)

    assert target.mesh.topology_id != source.mesh.topology_id
    assert target.certification is not None
    regions = target.certification.inputs.cell_regions
    assert regions is not None
    np.testing.assert_array_equal(
        np.asarray(regions),
        np.zeros(np.asarray(regions).shape, dtype=np.int64),
    )
    target_cells = np.asarray(target.mesh.entity_set(3).entity_ids, dtype=np.int64)
    assert len(target.zones) == 1
    np.testing.assert_array_equal(
        np.sort(np.asarray(target.zones[0].scope.entity_ids, dtype=np.int64)),
        np.sort(target_cells),
    )
    assert target.boundary is not None
    assert target.boundary.metadata.source_id == profile.source_id
    assert target.boundary.metadata.source_revision == profile.source_revision
    assert all(tag == "implicit-zero-set" for tag in target.boundary.metadata.cell_tags)
    cell_association = next(
        association
        for association in target.associations
        if association.target_entity_set_id == target.mesh.entity_set(3).entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.IMPLICIT
    assert cell_association.source_id == profile.source_id
    assert cell_association.source_revision == profile.source_revision
    assert cell_association.complete
    assert all(
        role is GeometrySourceEntityRole.REGION for role in cell_association.source_roles
    )
    assert adaptation.transition is not None
    stencil = adaptation.transition.vertex_stencil
    assert stencil is not None
    _assert_exact_constant_stencil(stencil)
    conserved = stencil.apply(
        source.mesh.vertex_global_ids,
        np.ones(source.mesh.vertex_global_ids.shape[0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(conserved),
        np.ones(target.mesh.vertex_global_ids.shape[0], dtype=np.float64),
    )


def test_implicit_size_proposals_use_three_dimensional_metric_adaptation(
    _implicit_volume_source: tuple[Any, Any, Any],
) -> None:
    profile, source, _ = _implicit_volume_source
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    # Native PLC volumes are 3-simplex blocks with simplicial entity incidence.
    connectivity = source.mesh.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    edges = np.asarray(connectivity.entities[1], dtype=np.int64)
    lengths = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    size = max(_policy(source).minimum_size, 0.6 * float(np.median(lengths)))
    proposal = MeshSizeProposal(
        source,
        mesh_proposal_scope(source, 0),
        np.full(source.mesh.vertex_global_ids.shape[0], size, dtype=np.float64),
        proposer_id="implicit-volume-size-proposal",
    )
    transaction = prepare_mesh_proposal(source, proposal, _policy(source))
    adaptation = transaction.adaptation
    assert adaptation is not None
    assert adaptation.metric is not None
    target = transaction.commit(source)

    assert target.certification is not None
    authority = target.certification.inputs.feature_source
    assert isinstance(authority, NativeImplicitSource)
    assert authority.authoring_profile is not None
    assert authority.authoring_profile.profile_id == profile.profile_id
    regions = target.certification.inputs.cell_regions
    assert regions is not None
    np.testing.assert_array_equal(
        np.asarray(regions),
        np.zeros(np.asarray(regions).shape, dtype=np.int64),
    )
    assert target.boundary is not None
    assert target.boundary.metadata.source_id == profile.source_id
    cell_association = next(
        association
        for association in target.associations
        if association.target_entity_set_id == target.mesh.entity_set(3).entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.IMPLICIT
    assert cell_association.complete


def test_implicit_surface_metric_proposals_preserve_exact_source_classes(
    _implicit_volume_source: tuple[Any, Any, Any],
) -> None:
    profile, _, source = _implicit_volume_source
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    edges = np.asarray(source.mesh.connectivity.edges, dtype=np.int64)
    lengths = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    size = max(_policy(source).minimum_size, 0.6 * float(np.median(lengths)))
    proposal = MeshSizeProposal(
        source,
        mesh_proposal_scope(source, 0),
        np.full(source.mesh.vertex_global_ids.shape[0], size, dtype=np.float64),
        proposer_id="implicit-surface-size-proposal",
    )
    transaction = prepare_mesh_proposal(source, proposal, _policy(source))
    adaptation = transaction.adaptation
    assert adaptation is not None
    assert adaptation.metric is not None
    target = transaction.commit(source)

    assert target.certification is not None
    authority = target.certification.inputs.feature_source
    assert isinstance(authority, NativeImplicitSource)
    assert authority.authoring_profile is not None
    assert authority.authoring_profile.profile_id == profile.profile_id
    assert target.boundary is not None
    assert target.boundary.metadata.source_id == profile.source_id
    assert target.boundary.metadata.source_revision == profile.source_revision
    cell_association = next(
        association
        for association in target.associations
        if association.target_entity_set_id == target.mesh.entity_set(2).entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.IMPLICIT
    assert cell_association.complete
    assert all(
        role is GeometrySourceEntityRole.FACET for role in cell_association.source_roles
    )
    assert adaptation.transition is not None
    stencil = adaptation.transition.vertex_stencil
    assert stencil is not None
    _assert_exact_constant_stencil(stencil)
    conserved = stencil.apply(
        source.mesh.vertex_global_ids,
        np.ones(source.mesh.vertex_global_ids.shape[0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(conserved),
        np.ones(target.mesh.vertex_global_ids.shape[0], dtype=np.float64),
    )
