# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    PointCloudPlan,
    TopologyEpoch,
    TransferGeometryBinding,
)
from phydrax.discretization._point_cloud import PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    adaptation_acceptance,
    commit_meshfree_epoch,
    degree_difference_indicator,
    flux_jump_indicator,
    LocalStencilPolicy,
    mark_points,
    meshfree_fill_measures,
    MeshfreeAdaptationPolicy,
    MeshfreeAdaptationStatus,
    MeshfreeAdaptiveSupport,
    MeshfreeEpochCandidate,
    MeshfreeEpochChange,
    MeshfreeErrorIndicator,
    MeshfreeMarking,
    MeshfreeMarkingPolicy,
    MeshfreeProbeJet,
    PointTransferStatus,
    prepare_adaptation_transfer,
    probe_residual_indicator,
    propose_adaptation,
    stage_meshfree_epoch,
    support_quality,
)
from phydrax.lifecycle import Composition, CompositionEntry


SOURCE = TopologyEpoch(0, "square-cloud-0", "bulk", "serial")
TARGET = TopologyEpoch(1, "square-cloud-1", "bulk", "serial")

type SquareFixture = tuple[
    np.ndarray, np.ndarray, np.ndarray, PreparedPointCloudDiscretization
]


def _square(count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered Cartesian unit-square cloud with exact boundary rows and normals."""
    axis = np.linspace(0.0, 1.0, count)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    boundary = np.any((points == 0.0) | (points == 1.0), axis=1)
    rng = np.random.default_rng(seed)
    spacing = 1.0 / (count - 1)
    points[~boundary] += rng.uniform(
        -0.15 * spacing, 0.15 * spacing, (np.sum(~boundary), 2)
    )
    normals = np.zeros_like(points)
    normals[:, 0] = np.where(
        points[:, 0] == 0.0, -1.0, np.where(points[:, 0] == 1.0, 1.0, 0.0)
    )
    normals[:, 1] = np.where(
        points[:, 1] == 0.0, -1.0, np.where(points[:, 1] == 1.0, 1.0, 0.0)
    )
    normals[boundary] /= np.linalg.norm(normals[boundary], axis=1, keepdims=True)
    return points, boundary, normals


def _project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on the unit-square boundary and its outward side normal."""
    gaps = np.stack(
        (points[:, 0], 1 - points[:, 0], points[:, 1], 1 - points[:, 1]), axis=1
    )
    side = np.argmin(gaps, axis=1)
    projected = points.copy()
    normals = np.zeros_like(points)
    for index, (axis, value, sign) in enumerate(
        ((0, 0.0, -1), (0, 1.0, 1), (1, 0.0, -1), (1, 1.0, 1))
    ):
        rows = side == index
        projected[rows, axis] = value
        normals[rows, axis] = sign
    return projected, normals


def _cloud(
    points: np.ndarray,
    boundary: np.ndarray,
    normals: np.ndarray,
    /,
    *,
    ids: np.ndarray | None = None,
    degree: int = 2,
    neighbors: int = 16,
) -> PreparedPointCloudDiscretization:
    return PointCloudPlan(
        jnp.asarray(points),
        jnp.asarray(meshfree_fill_measures(points, 1.0, intrinsic_dimension=2)),
        boundary_mask=jnp.asarray(boundary),
        boundary_normals=jnp.asarray(normals),
        stencil=LocalStencilPolicy(polynomial_degree=degree),
        neighbors=neighbors,
        point_ids=None if ids is None else jnp.asarray(ids),
    ).prepare()


def _laplace_residual(points: Array, jet: MeshfreeProbeJet) -> Array:
    """-Laplace(u) - f for u = x^2 + y^2 (f = -4)."""
    del points
    return -(jet.hessian[:, 0, 0] + jet.hessian[:, 1, 1]) + 4.0


def _layer(points: np.ndarray) -> np.ndarray:
    return np.exp(-points[:, 0] / 0.08)


def _layer_residual(points: Array, jet: MeshfreeProbeJet) -> Array:
    """-Laplace(u) - f for the boundary layer u = exp(-x / 0.08)."""
    return (
        -(jet.hessian[:, 0, 0] + jet.hessian[:, 1, 1])
        + jnp.exp(-points[:, 0] / 0.08) / 0.08**2
    )


@pytest.fixture(scope="module")
def square() -> SquareFixture:
    points, boundary, normals = _square(11)
    return points, boundary, normals, _cloud(points, boundary, normals)


def test_off_node_residual_vanishes_on_reproduced_fields_and_ranks_the_layer(
    square: SquareFixture,
) -> None:
    points, _, _, cloud = square
    quadratic = jnp.asarray(np.sum(points**2, axis=1))
    exact = probe_residual_indicator(cloud, quadratic, _laplace_residual)
    assert exact.kind == "probe-residual"
    assert exact.probe_count == 4 * points.shape[0]
    assert exact.aggregate < 1e-8

    layer = probe_residual_indicator(cloud, jnp.asarray(_layer(points)), _layer_residual)
    worst = int(np.argmax(np.asarray(layer.values)))
    assert layer.aggregate > 1e-2
    assert points[worst, 0] < 0.25


def test_flux_jump_and_degree_difference_vanish_on_quadratics(
    square: SquareFixture,
) -> None:
    points, boundary, normals, cloud = square
    quadratic = jnp.asarray(1.0 + points[:, 0] * points[:, 1] + points[:, 1] ** 2)
    layer = jnp.asarray(_layer(points))
    assert flux_jump_indicator(cloud, quadratic, diffusivity=2.0).aggregate < 1e-8
    assert flux_jump_indicator(cloud, layer).aggregate > 1e-2

    cubic = _cloud(points, boundary, normals, degree=3, neighbors=20)
    measures = cloud.quadrature_weights
    interior = ~boundary
    flat = degree_difference_indicator(
        cloud.laplacian,
        cubic.laplacian,
        quadratic,
        measures,
        cloud.stable_ids,
        rows=interior,
    )
    rough = degree_difference_indicator(
        cloud.laplacian, cubic.laplacian, layer, measures, cloud.stable_ids, rows=interior
    )
    assert flat.aggregate < 1e-8
    assert rough.aggregate > 1e-2
    assert np.all(np.asarray(rough.values)[boundary] == 0.0)


def test_support_quality_reads_the_cloud_stencil_evidence(square: SquareFixture) -> None:
    _, _, _, cloud = square
    quality = support_quality(cloud)
    assert np.isclose(
        float(np.max(np.asarray(quality.amplification))),
        cloud.report.maximum_amplification,
    )
    assert np.all(np.asarray(quality.spacing_ratio) >= 1.0)
    assert quality.indicator.kind == "support-amplification"


def test_marking_is_order_independent_bulk_and_capped() -> None:
    values = np.asarray([0.1, 3.0, 0.2, 2.0, 0.0, 1.0, 1.0])
    ids = np.asarray([40, 11, 7, 25, 3, 18, 9], dtype=np.int64)
    # Squared shares 9, 4, 1, 1, ... of 15.05: 90% needs the three largest.
    policy = MeshfreeMarkingPolicy("dorfler", fraction=0.9)
    marking = mark_points(
        MeshfreeErrorIndicator(values, ids, kind="flux-jump", probe_count=0), policy
    )
    permutation = np.asarray([6, 2, 0, 5, 1, 4, 3], dtype=np.intp)
    permuted = mark_points(
        MeshfreeErrorIndicator(
            values[permutation], ids[permutation], kind="flux-jump", probe_count=0
        ),
        policy,
    )
    np.testing.assert_array_equal(marking.marked_ids, permuted.marked_ids)
    assert marking.captured_fraction >= 0.9 and not marking.capped
    # Two equal indicators tie: the smaller stable ID (9) wins over 18.
    np.testing.assert_array_equal(marking.marked_ids, [9, 11, 25])

    capped = mark_points(
        MeshfreeErrorIndicator(values, ids, kind="flux-jump", probe_count=0),
        MeshfreeMarkingPolicy("dorfler", fraction=0.9, maximum_marked=1),
    )
    np.testing.assert_array_equal(capped.marked_ids, [11])
    assert capped.capped and capped.captured_fraction < 0.9


def _layer_proposal(
    points: np.ndarray,
    boundary: np.ndarray,
    normals: np.ndarray,
    ids: np.ndarray,
    /,
) -> tuple[
    MeshfreeAdaptiveSupport,
    MeshfreeErrorIndicator,
    MeshfreeMarking,
    MeshfreeAdaptationPolicy,
]:
    support = MeshfreeAdaptiveSupport(
        points,
        ids,
        kind="bulk",
        intrinsic_dimension=2,
        degree=2,
        neighbors=16,
        boundary_mask=boundary,
        boundary_normals=normals,
    )
    indicator = MeshfreeErrorIndicator(
        np.exp(-points[:, 0] / 0.1), ids, kind="probe-residual", probe_count=0
    )
    marking = mark_points(indicator, MeshfreeMarkingPolicy("dorfler", fraction=0.5))
    return support, indicator, marking, MeshfreeAdaptationPolicy(total_measure=1.0)


def test_children_are_deterministic_separated_projected_edge_midpoints(
    square: SquareFixture,
) -> None:
    points, boundary, normals, _ = square
    ids = np.arange(points.shape[0]) * 3 + 5
    support, indicator, marking, policy = _layer_proposal(points, boundary, normals, ids)
    proposal = propose_adaptation(
        support, indicator, marking, policy, boundary_projection=_project
    )
    assert proposal.admitted and proposal.changes == ("insertion",)

    target = np.asarray(proposal.target_points)
    target_ids = np.asarray(proposal.target_ids)
    inserted = np.asarray(proposal.inserted_ids)
    assert np.unique(target_ids).size == target_ids.size
    assert inserted.min() > ids.max()
    position = {int(identifier): row for row, identifier in enumerate(ids)}
    lookup = {int(identifier): row for row, identifier in enumerate(target_ids)}
    for child, (first, second) in zip(
        inserted, np.asarray(proposal.inserted_parents), strict=True
    ):
        a, b = points[position[int(first)]], points[position[int(second)]]
        location = target[lookup[int(child)]]
        length = np.linalg.norm(a - b)
        others = np.delete(target, lookup[int(child)], axis=0)
        assert (
            np.min(np.linalg.norm(others - location, axis=1))
            >= policy.minimum_separation * length - 1e-12
        )
        if boundary[position[int(first)]] and boundary[position[int(second)]]:
            assert np.any((location == 0.0) | (location == 1.0))
        else:
            np.testing.assert_allclose(location, 0.5 * (a + b), atol=1e-14)
    new_boundary = np.asarray(proposal.target_boundary)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(proposal.target_normals)[new_boundary], axis=1), 1.0
    )
    assert np.isclose(float(np.sum(np.asarray(proposal.target_measures))), 1.0)
    # Children concentrate where the indicator is large.
    assert np.mean(target[np.isin(target_ids, inserted), 0]) < 0.35

    permutation = np.random.default_rng(3).permutation(points.shape[0])
    again = propose_adaptation(
        *_layer_proposal(
            points[permutation],
            boundary[permutation],
            normals[permutation],
            ids[permutation],
        ),
        boundary_projection=_project,
    )
    original = dict(zip(target_ids.tolist(), map(tuple, target), strict=True))
    shuffled = dict(
        zip(
            np.asarray(again.target_ids).tolist(),
            map(tuple, np.asarray(again.target_points)),
            strict=True,
        )
    )
    assert original == shuffled

    unresolved = propose_adaptation(support, indicator, marking, policy)
    assert unresolved.unresolved_boundary_children > 0
    assert not np.any(np.asarray(unresolved.target_boundary)[points.shape[0] :])


def test_capacity_refusal_and_no_change_cannot_become_epochs(
    square: SquareFixture,
) -> None:
    points, boundary, normals, _ = square
    ids = np.arange(points.shape[0])
    support, indicator, marking, _ = _layer_proposal(points, boundary, normals, ids)
    refused = propose_adaptation(
        support,
        indicator,
        marking,
        MeshfreeAdaptationPolicy(total_measure=1.0, maximum_inserted=2),
        boundary_projection=_project,
    )
    assert (
        refused.status is MeshfreeAdaptationStatus.CAPACITY_REFUSED
        and not refused.admitted
    )
    with pytest.raises(ValueError, match="admitted adaptation proposal"):
        MeshfreeEpochChange(SOURCE, TARGET, cause="adaptive-refinement", proposal=refused)

    silent = MeshfreeErrorIndicator(
        np.zeros(ids.size), ids, kind="flux-jump", probe_count=0
    )
    nothing = propose_adaptation(
        support,
        silent,
        mark_points(silent, MeshfreeMarkingPolicy()),
        MeshfreeAdaptationPolicy(total_measure=1.0),
    )
    assert nothing.status is MeshfreeAdaptationStatus.NO_CHANGE

    degree = propose_adaptation(
        support,
        silent,
        mark_points(silent, MeshfreeMarkingPolicy()),
        MeshfreeAdaptationPolicy(total_measure=1.0),
        target_degree=3,
        target_neighbors=20,
    )
    assert degree.admitted and degree.changes == ("degree", "support")
    # Degree 5 needs 21 basis functions: 16 neighbors are not unisolvent.
    with pytest.raises(ValueError, match="needs between"):
        propose_adaptation(
            support,
            silent,
            mark_points(silent, MeshfreeMarkingPolicy()),
            MeshfreeAdaptationPolicy(total_measure=1.0),
            target_degree=5,
        )


def test_coarsening_removes_an_independent_set_of_quiet_interior_points(
    square: SquareFixture,
) -> None:
    points, boundary, normals, cloud = square
    ids = np.arange(points.shape[0])
    support, indicator, marking, _ = _layer_proposal(points, boundary, normals, ids)
    proposal = propose_adaptation(
        support,
        indicator,
        marking,
        MeshfreeAdaptationPolicy(
            total_measure=1.0, coarsen_fraction=0.2, maximum_removed=12
        ),
        boundary_projection=_project,
    )
    removed = np.asarray(proposal.removed_ids)
    assert 0 < removed.size <= 12 and "removal" in proposal.changes
    assert not np.any(boundary[removed])
    assert not np.any(np.isin(removed, np.asarray(marking.marked_ids)))
    assert np.all(
        np.asarray(indicator.values)[removed] <= 0.2 * float(np.max(indicator.values))
    )
    assert not np.any(np.isin(removed, np.asarray(proposal.inserted_parents)))
    neighbors = np.asarray(cloud.relation.source_indices)[removed]
    adjacency = np.isin(neighbors, removed) & (neighbors != removed[:, None])
    assert not np.any(adjacency[:, :8])


def _composition(cloud: PreparedPointCloudDiscretization, values: Array) -> Composition:
    epoch = CompositionEntry(
        SOURCE,
        entry_id="cloud/epoch",
        role="topology",
        owner_id="adaptive-test",
        structure_id=SOURCE.epoch_id,
        revision_id="initial",
        semantics_id="cloud",
    )
    field = CompositionEntry(
        values,
        entry_id="field/solution",
        role="physical-state",
        owner_id="adaptive-test",
        structure_id=SOURCE.epoch_id,
        revision_id="solution@0",
        semantics_id="field",
        dependencies=(epoch.binding("structure"),),
    )
    support = CompositionEntry(
        cloud,
        entry_id="cloud/support",
        role="discretization",
        owner_id="adaptive-test",
        structure_id=SOURCE.epoch_id,
        revision_id=cloud.prepared_id,
        semantics_id="support",
        dependencies=(epoch.binding("structure"),),
    )
    return Composition((epoch, field, support), boundary_id="adaptive-level-0")


def test_accepted_epoch_conserves_content_and_rejection_publishes_nothing(
    square: SquareFixture,
) -> None:
    points, boundary, normals, cloud = square
    support = MeshfreeAdaptiveSupport.from_point_cloud(cloud)
    indicator = probe_residual_indicator(
        cloud, jnp.asarray(_layer(points)), _layer_residual
    )
    marking = mark_points(indicator, MeshfreeMarkingPolicy("dorfler", fraction=0.4))
    proposal = propose_adaptation(
        support,
        indicator,
        marking,
        MeshfreeAdaptationPolicy(total_measure=1.0),
        boundary_projection=_project,
    )
    # Two fill quadratures differ in their first moments, so conservation with
    # linear reproduction is impossible: the owner certifies the obstruction.
    linear = prepare_adaptation_transfer(cloud, proposal, moment_degree=1)
    assert linear.evidence.status is PointTransferStatus.INFEASIBLE
    assert linear.evidence.witness_kind == "left-null" and linear.transfer is None

    transfer = prepare_adaptation_transfer(
        cloud,
        proposal,
        geometry=TransferGeometryBinding(
            SOURCE.geometry_id,
            TARGET.geometry_id,
            "topology-correspondence",
            source_topology_id=SOURCE.topology_id,
            target_topology_id=TARGET.topology_id,
            coverage_defect=None,
        ),
    )
    assert transfer.admitted
    assert transfer.evidence.conservative and transfer.evidence.constant_preserving
    target = np.asarray(proposal.target_points)
    np.testing.assert_allclose(
        transfer.apply(jnp.full(points.shape[0], 3.0)), 3.0, atol=1e-10
    )
    measures = np.asarray(cloud.quadrature_weights)
    field = jnp.asarray(_layer(points))
    np.testing.assert_allclose(
        float(jnp.sum(proposal.target_measures * transfer.apply(field))),
        float(np.sum(measures * np.asarray(field))),
        rtol=1e-10,
    )

    target_cloud = _cloud(
        target,
        np.asarray(proposal.target_boundary),
        np.asarray(proposal.target_normals),
        ids=np.asarray(proposal.target_ids),
    )
    change = MeshfreeEpochChange(
        SOURCE, TARGET, cause="adaptive-refinement", proposal=proposal
    )
    target_epoch = CompositionEntry(
        TARGET,
        entry_id="cloud/epoch",
        role="topology",
        owner_id="adaptive-test",
        structure_id=TARGET.epoch_id,
        revision_id=change.change_id,
        semantics_id="cloud",
    )
    rebuilt = CompositionEntry(
        target_cloud,
        entry_id="cloud/support",
        role="discretization",
        owner_id="adaptive-test",
        structure_id=TARGET.epoch_id,
        revision_id=target_cloud.prepared_id,
        semantics_id="support",
        dependencies=(target_epoch.binding("structure"),),
    )
    source = _composition(cloud, field)

    def stage() -> MeshfreeEpochCandidate:
        return stage_meshfree_epoch(
            source,
            change,
            epoch_entry="cloud/epoch",
            remap={"field/solution": transfer.epoch_transition(SOURCE, TARGET)},
            reprepare=(rebuilt,),
        )

    accepted = adaptation_acceptance(
        proposal, transfer, target_cloud.report, solve_successful=True
    )
    assert accepted.accepted and accepted.refusals == ()
    receipt = commit_meshfree_epoch(stage(), accepted_boundary=accepted.accepted)
    assert receipt.published and receipt.failed == ()
    assert receipt.composition.value("cloud/support") is target_cloud
    assert receipt.composition.value("field/solution").shape == (target.shape[0],)
    assert float(np.max(np.abs(np.asarray(receipt.conservation_residuals)))) <= float(
        np.max(np.asarray(receipt.content_tolerances))
    )

    worse = MeshfreeErrorIndicator(
        2.0 * np.ones(target.shape[0]) * indicator.aggregate,
        proposal.target_ids,
        kind="probe-residual",
        probe_count=0,
    )
    rejected = adaptation_acceptance(
        proposal,
        transfer,
        target_cloud.report,
        solve_successful=False,
        indicator_before=indicator,
        indicator_after=worse,
    )
    assert not rejected.accepted
    assert rejected.refusals == ("solve-failed", "indicator-not-reduced")
    refusal = commit_meshfree_epoch(stage(), accepted_boundary=rejected.accepted)
    assert not refusal.published and refusal.composition is source
