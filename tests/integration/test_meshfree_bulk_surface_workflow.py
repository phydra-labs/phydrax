# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real native coupled solve, analytic equilibrium, amounts and failure admission."""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_bulk_surface_exchange import (
    prepare_workflow,
    PreparedBulkSurfaceWorkflow,
    run_workflow,
)
from phydrax.discretization import (
    PointCloudPlan,
    prepare_point_cloud_field_reconstruction,
    PreparedPointCloudDiscretization,
    TopologyEpoch,
    TopologyEpochTransition,
)
from phydrax.discretization.meshfree._capacity import MeshfreeCapacityPolicy
from phydrax.discretization.meshfree._epochs import MeshfreeEpochChange
from phydrax.discretization.meshfree._resampling import SurfaceResamplingPolicy
from phydrax.discretization.meshfree._stencils import LocalStencilPolicy
from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
from phydrax.discretization.meshfree._surface_geometry import ImplicitSurfaceGeometry
from phydrax.discretization.meshfree._surface_quadrature import SurfaceQuadraturePolicy
from phydrax.discretization.meshfree._transfer import (
    PointTransferPlan,
    PointTransferRequest,
)
from phydrax.geometry import Rectangle
from phydrax.interfacial_transport import AdsorptionKinetics
from phydrax.interfacial_transport._film_evidence import FilmStepStatus
from phydrax.metrix import EuclideanStateGeometry, RegularLevelSetManifold
from phydrax.solver import (
    coupling as cpl,
    FixedStepProblem,
    FixedStepRolloutPlan,
    retry_fixed_step,
    RobustRetryPolicy,
)
from phydrax.solver.coupling import CouplingWindow, FixedStepParticipantEvidence
from phydrax.solver.coupling._contributions import (
    ContributionEndpoint,
    ResidualContribution,
)
from phydrax.solver.coupling._meshfree_components import MeshfreeComponent
from phydrax.solver.coupling._surface_exchange import (
    LangmuirAdsorptionFlux,
    MeshfreeBulkSurfaceEvidence,
    MeshfreeBulkSurfaceMethod,
    SurfaceExchangeLaw,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def test_analytic_langmuir_equilibrium_and_moving_query_ledger() -> None:
    result = run_workflow(size=128, dimension=3, seed=4)
    assert result["accepted"] is True
    assert float(result["equilibrium_error"]) < 1e-8
    assert float(result["flux_error"]) < 1e-8
    assert float(result["conservation_error"]) < 1e-10
    assert float(result["nonlinear_residual"]) < 1e-10
    assert int(result["nonlinear_iterations"]) > 0
    assert float(result["query_displacement"]) > 0
    assert 0 < float(result["window_lag"]) < 0.1
    assert float(result["window_lag_error"]) < 1e-14
    assert float(result["query_constant_error"]) < 1e-10
    assert result["geometry_differentiated_within_window"] is False


@pytest.fixture(scope="module")
def prepared() -> PreparedBulkSurfaceWorkflow:
    return prepare_workflow(size=128, dimension=3, seed=0)


def test_surface_numeric_revision_refuses_stale_reconstruction(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    original = prepared.surface
    revised = SurfacePointCloudPlan(
        original.points,
        original.plan.geometry,
        original.plan.neighbors,
        quadrature=SurfaceQuadraturePolicy("supplied", measures=2 * original.measures),
        stencil_policy=original.plan.stencil_policy,
    ).prepare()
    laplace = revised.laplace_beltrami
    operator = SparseCoordinateOperator(
        laplace.relation,
        -revised.measures[:, None] * laplace.coefficients,
        source=laplace.source,
        target=laplace.target,
    )
    with pytest.raises(ValueError, match="owner revision"):
        MeshfreeComponent(
            revised,
            operator,
            prepared.surface_component.reconstruction,
            revised.measures,
            name="surface",
            owner_id=revised.prepared_id,
        )


def test_surface_source_program_rebinding_refuses_stale_reconstruction(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    original = prepared.surface
    geometry = original.plan.geometry
    if not isinstance(geometry, ImplicitSurfaceGeometry):
        raise TypeError("This workflow requires its declared implicit source.")
    source = geometry.source
    if not isinstance(source, RegularLevelSetManifold):
        raise TypeError("This workflow requires its native level-set manifold.")
    # Keep nominal labels, sampled geometry, measures and operators unchanged;
    # only the actual bound source program is replaced. A source rebind cannot
    # reuse the old reconstruction just because cached point arrays coincide.
    replacement_source = eqx.tree_at(
        lambda manifold: manifold.constraint,
        source,
        lambda point: 3.0 * source.constraint(point),
    )
    replacement_geometry = eqx.tree_at(
        lambda owner: owner.source, geometry, replacement_source
    )
    revised = eqx.tree_at(
        lambda owner: owner.plan.geometry, original, replacement_geometry
    )
    with pytest.raises(ValueError, match="owner revision"):
        MeshfreeComponent(
            revised,
            prepared.surface_component.native_operator,
            prepared.surface_component.reconstruction,
            revised.measures,
            name="surface",
            owner_id=revised.prepared_id,
        )


def test_overcapacity_native_step_is_rejected_without_spending_amounts(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    method = prepared.method
    bulk, _ = prepared.initial
    surface = 1.01 * method.surface_measures
    result = method.step(
        jnp.asarray(0), jnp.asarray(0.0), (bulk, surface), jnp.asarray(1.0), None
    )
    assert not bool(result.successful)
    np.testing.assert_array_equal(result.accepted_state[0], bulk)
    np.testing.assert_array_equal(result.accepted_state[1], surface)


def test_signed_polynomial_query_cannot_claim_positive_native_amount_partition(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    reconstruction = prepared.method.query.reconstruction
    cloud = prepared.bulk.owner
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError(
            "The workflow bulk equation must use the native prepared point cloud."
        )
    kinetics = prepared.method.transport.structure.kinetics
    if kinetics is None:
        raise TypeError("Langmuir exchange requires its actual kinetic owner.")
    polynomial = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=reconstruction.support_geometry,
        radius=1.5,
        capacity=cloud.points.shape[0],
        reconstruction="polynomial",
    ).prepare_query(prepared.graph.points)
    assert np.any(np.asarray(polynomial.route.weights) < -1e-8)
    np.testing.assert_allclose(
        polynomial.apply(jnp.ones(cloud.state_shape)), 1.0, atol=1e-12, rtol=0.0
    )
    with pytest.raises(ValueError, match="Signed query weights"):
        MeshfreeBulkSurfaceMethod(
            polynomial, prepared.graph, prepared.method.bulk_volumes, kinetics
        )


def test_workflow_refuses_wrong_ambient_geometry() -> None:
    with pytest.raises(ValueError, match="dimension=3"):
        prepare_workflow(size=128, dimension=2)


def _method_evidence(evidence: object) -> MeshfreeBulkSurfaceEvidence:
    if not isinstance(evidence, FixedStepParticipantEvidence):
        raise TypeError("The fixed-step participant must publish its native evidence.")
    method = evidence.method
    if not isinstance(method, MeshfreeBulkSurfaceEvidence):
        raise TypeError("The bulk-surface owner must publish its typed evidence.")
    return method


def test_multi_substep_window_publishes_balanced_native_transfer_evidence(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    method = prepared.method
    bulk, surface = prepared.initial
    participant = method.participant(substeps=3)
    window = participant.advance_window(
        CouplingWindow(0, 0.0, 1.5), participant.initial_state(prepared.initial), (), None
    )
    assert bool(window.successful)
    evidence = window.evidence
    assert isinstance(evidence, FixedStepParticipantEvidence)
    np.testing.assert_array_equal(evidence.executed, [True, True, True])
    np.testing.assert_array_equal(evidence.successful, [True, True, True])
    reduced = _method_evidence(evidence)
    assert int(reduced.status) == int(FilmStepStatus.ACCEPTED)
    assert int(reduced.nonlinear_iterations) >= 3
    assert bool(reduced.query_complete)
    # The workflow admits its surface metric through the relaxed moment program.
    assert bool(reduced.metric_exact) == bool(prepared.graph.metric_result.exact)
    # Independent amount balance from the participant's own accepted amounts.
    bulk_after, surface_after = window.candidate_state.native
    transferred = float(jnp.sum(reduced.transferred_to_surface_mol))
    assert abs(transferred) > 1e-6
    np.testing.assert_allclose(
        float(jnp.sum(bulk) - jnp.sum(bulk_after)), transferred, rtol=1e-9
    )
    np.testing.assert_allclose(
        float(jnp.sum(surface_after) - jnp.sum(surface)), transferred, rtol=1e-9
    )
    np.testing.assert_allclose(
        reduced.candidate_transferred_to_surface_mol,
        reduced.transferred_to_surface_mol,
        rtol=1e-12,
    )
    assert 0.0 < float(reduced.maximum_coverage) < 1.0


def test_refused_window_keeps_refusal_evidence_and_the_accepted_checkpoint(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    method = prepared.method
    bulk, _ = prepared.initial
    overfilled = 1.01 * method.surface_measures
    participant = method.participant(substeps=3)
    checkpoint = participant.initial_state((bulk, overfilled))
    window = participant.advance_window(CouplingWindow(0, 0.0, 1.5), checkpoint, (), None)
    assert not bool(window.successful)
    evidence = window.evidence
    assert isinstance(evidence, FixedStepParticipantEvidence)
    # The first native substep refuses; no further owner work is performed.
    np.testing.assert_array_equal(evidence.executed, [True, False, False])
    np.testing.assert_array_equal(evidence.successful, [False, False, False])
    reduced = _method_evidence(evidence)
    assert int(reduced.status) != int(FilmStepStatus.ACCEPTED)
    np.testing.assert_array_equal(reduced.transferred_to_surface_mol, 0.0)
    np.testing.assert_array_equal(window.candidate_state.native[0], bulk)
    np.testing.assert_array_equal(window.candidate_state.native[1], overfilled)

    # The same refusal through the coupling runtime rolls the window back while
    # its participant evidence survives in the window result.
    space = participant.output_ports[0].space
    monitor = cpl.CallableCouplingSubsystem(
        lambda window_, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (), successful=True, status=0
        ),
        subsystem_id="monitor",
        input_ports=(
            cpl.CouplingPort("monitor/bulk", "input", space, reference_scale=1.0),
        ),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    graph = cpl.CouplingGraph(
        (participant, monitor),
        (cpl.CouplingExchange("observe-bulk", "bulk_concentration", "monitor/bulk"),),
    )
    coupled = cpl.prepare_coupling(
        graph,
        (checkpoint, jnp.zeros(1)),
        (space.zeros(),),
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
    )
    index = coupled.reference_state.subsystem_ids.index(participant.subsystem_id)
    result = cpl.advance_coupling_window(coupled, coupled.reference_state, 1.5)
    assert not bool(result.successful)
    assert int(result.status) == int(cpl.CouplingStatus.PARTICIPANT_FAILURE)
    held = result.accepted_state.participant_states[index]
    np.testing.assert_array_equal(held.native[0], bulk)
    np.testing.assert_array_equal(held.native[1], overfilled)
    assert int(held.accepted_windows) == 0
    retained = result.participant_evidence[index]
    assert isinstance(retained, FixedStepParticipantEvidence)
    np.testing.assert_array_equal(retained.executed, [True, False, False])
    assert int(_method_evidence(retained).status) == int(reduced.status)


def test_retry_reports_refused_attempts_and_selects_the_accepted_native_step(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    original = prepared.method
    kinetics = original.transport.structure.kinetics
    if kinetics is None:
        raise TypeError("Langmuir exchange requires its actual kinetic owner.")
    # A tight native Newton budget refuses long steps; shorter steps converge.
    method = MeshfreeBulkSurfaceMethod(
        original.query,
        prepared.graph,
        original.bulk_volumes,
        kinetics,
        surface_diffusivity=0.02,
        maximum_iterations=6,
    )
    state = prepared.initial
    retried = retry_fixed_step(
        method,
        RobustRetryPolicy(maximum_retries=4, reduction_factor=0.125),
        jnp.asarray(0),
        jnp.asarray(0.0),
        state,
        jnp.asarray(8.0),
    )
    assert bool(retried.successful)
    selected = int(retried.retry_count)
    assert selected >= 1
    np.testing.assert_array_equal(
        retried.attempt_successful[:selected], [False] * selected
    )
    attempts = retried.attempt_evidence
    assert isinstance(attempts, MeshfreeBulkSurfaceEvidence)
    first_status = int(attempts.status[0])
    assert first_status == int(FilmStepStatus.SOLVE_FAILED)
    assert int(attempts.nonlinear_iterations[0]) == 6
    np.testing.assert_array_equal(attempts.transferred_to_surface_mol[0], 0.0)
    chosen = retried.evidence
    assert isinstance(chosen, MeshfreeBulkSurfaceEvidence)
    assert int(chosen.status) == int(FilmStepStatus.ACCEPTED)
    # Independent recomputation of the selected native step and its balance.
    direct = method.step(
        jnp.asarray(0), jnp.asarray(0.0), state, retried.accepted_step_size, None
    )
    np.testing.assert_array_equal(retried.accepted_state[0], direct.accepted_state[0])
    np.testing.assert_array_equal(retried.accepted_state[1], direct.accepted_state[1])
    transferred = float(jnp.sum(chosen.transferred_to_surface_mol))
    np.testing.assert_allclose(
        float(jnp.sum(state[0]) - jnp.sum(retried.accepted_state[0])),
        transferred,
        rtol=1e-9,
    )


def test_rollout_retains_per_step_native_evidence_with_amount_balance(
    prepared: PreparedBulkSurfaceWorkflow,
) -> None:
    method = prepared.method
    problem = FixedStepProblem(
        method,
        prepared.initial,
        t0=0.0,
        t1=2.0,
        step_size=0.5,
        state_geometry=EuclideanStateGeometry(
            geometry_id="state-geometry:bulk-surface-amounts"
        ),
    )
    rollout = FixedStepRolloutPlan(evidence_retention="steps").rollout(problem)
    assert bool(rollout.successful)
    record = rollout.evidence
    assert record is not None
    assert int(record.accepted_step) == 3 and int(record.refused_step) == -1
    steps = record.steps
    assert isinstance(steps, MeshfreeBulkSurfaceEvidence)
    np.testing.assert_array_equal(record.step_committed, [True] * 4)
    np.testing.assert_array_equal(steps.status, int(FilmStepStatus.ACCEPTED))
    bulk, surface = prepared.initial
    total = float(jnp.sum(steps.transferred_to_surface_mol))
    np.testing.assert_allclose(
        total,
        float(jnp.sum(rollout.final_state[1]) - jnp.sum(surface)),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        float(jnp.sum(bulk) - jnp.sum(rollout.final_state[0])), total, rtol=1e-9
    )

    refused = FixedStepRolloutPlan().rollout(
        FixedStepProblem(
            method,
            (bulk, 1.01 * method.surface_measures),
            t0=0.0,
            t1=1.0,
            step_size=0.5,
            state_geometry=EuclideanStateGeometry(
                geometry_id="state-geometry:bulk-surface-amounts"
            ),
        )
    )
    assert not bool(refused.successful)
    assert refused.evidence is not None
    assert int(refused.evidence.accepted_step) == -1
    assert int(refused.evidence.refused_step) == 0
    refusal = refused.evidence.refused
    assert isinstance(refusal, MeshfreeBulkSurfaceEvidence)
    assert int(refusal.status) != int(FilmStepStatus.ACCEPTED)
    np.testing.assert_array_equal(refused.final_state[1], 1.01 * method.surface_measures)


# Moving-surface epoch cutover. This scenario owns its own 2-D bulk/curve
# fixture: it is independent of the sphere workflow's exterior surface metric.
RING_CENTER, RING_RADIUS = np.asarray([0.5, 0.5]), 0.3
RING_SUPPORT = Rectangle((0.5, 0.5), (1.0, 1.0))
EXCHANGE_STEP = 0.005


def _ring_component(points: np.ndarray, /) -> MeshfreeComponent:
    """Intrinsic meshfree curve |x - c| = r sampled at ``points``."""

    def constraint(point: Array) -> Array:
        offset = point - jnp.asarray(RING_CENTER)
        return jnp.asarray([jnp.dot(offset, offset) - RING_RADIUS**2])

    count = points.shape[0]
    source = RegularLevelSetManifold(
        constraint, ambient_dimension=2, codimension=1, manifold_id="exchange-ring"
    )
    surface = SurfacePointCloudPlan(
        jnp.asarray(points),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.1, geometry_id="exchange-ring"
        ),
        8,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(count, dtype=jnp.float64),
            total_area=2 * math.pi * RING_RADIUS,
        ),
        stencil_policy=LocalStencilPolicy(polynomial_degree=2),
        require_tube=True,
    ).prepare()
    laplace = surface.laplace_beltrami
    native = SparseCoordinateOperator(
        laplace.relation,
        -surface.measures[:, None] * laplace.coefficients,
        source=laplace.source,
        target=laplace.target,
    )
    return MeshfreeComponent(
        surface,
        native,
        surface.prepare_field_reconstruction(support_geometry=RING_SUPPORT.compile()),
        surface.measures,
        name="surface",
        owner_id=surface.prepared_id,
    )


def _shepard_bulk() -> MeshfreeComponent:
    axis = np.linspace(0.0, 1.0, 9)
    points = np.stack([item.ravel() for item in np.meshgrid(axis, axis)], axis=-1)
    volumes = np.full(points.shape[0], 1.0 / points.shape[0])
    cloud = PointCloudPlan(points, volumes, neighbors=20).prepare()
    view = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=RING_SUPPORT.compile(),
        radius=0.3,
        capacity=points.shape[0],
        reconstruction="shepard",
    )
    full = cloud.field_spaces[0].vector_space
    weights = sum(pair[1] for pair in cloud.derivative_weights)
    native = SparseCoordinateOperator(
        cloud.relation, -jnp.asarray(volumes)[:, None] * weights, source=full, target=full
    )
    return MeshfreeComponent(
        cloud, native, view, volumes, name="bulk", owner_id=cloud.prepared_id
    )


def _ring_normals(points: Array) -> np.ndarray:
    offset = np.asarray(points) - RING_CENTER
    return offset / np.linalg.norm(offset, axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class RingEpoch:
    """Bulk, source ring, sample-repaired target ring, and their epoch route."""

    bulk: MeshfreeComponent
    surface: MeshfreeComponent
    target: MeshfreeComponent
    law: SurfaceExchangeLaw
    change: MeshfreeEpochChange
    transition: TopologyEpochTransition

    @property
    def components(self) -> dict[str, MeshfreeComponent]:
        return {"bulk": self.bulk, "surface": self.surface}


@pytest.fixture(scope="module")
def ring_epoch() -> RingEpoch:
    angles = 2 * math.pi * (np.arange(24, dtype=np.float64) + 0.25) / 24
    circle = RING_CENTER + RING_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
    # A gap at sample 5: the resampling proposal inserts its probe.
    surface = _ring_component(np.delete(circle, 5, axis=0))
    bulk = _shepard_bulk()
    law = SurfaceExchangeLaw(
        ContributionEndpoint("bulk", "concentration"),
        ContributionEndpoint("surface", "concentration"),
        bulk.reconstruction.prepare_query(surface.owner.points),
        surface,
        _ring_normals(surface.owner.points),
        LangmuirAdsorptionFlux(AdsorptionKinetics(2.0, 0.5, 1.0)),
        deposition="positive",
    )

    def project(points: Array) -> Array:
        offset = points - jnp.asarray(RING_CENTER)
        return jnp.asarray(RING_CENTER) + RING_RADIUS * offset / jnp.linalg.norm(
            offset, axis=1, keepdims=True
        )

    def residual(points: Array) -> Array:
        return jnp.linalg.norm(points - jnp.asarray(RING_CENTER), axis=1) - RING_RADIUS

    probe_angles = 2 * math.pi * (0.5 * np.arange(48, dtype=np.float64) + 0.25) / 24
    probes = RING_CENTER + RING_RADIUS * np.column_stack(
        (np.cos(probe_angles), np.sin(probe_angles))
    )
    repair = SurfaceResamplingPolicy(maximum_fill=0.06, minimum_separation=0.01).repair(
        surface.owner.points,
        probes,
        MeshfreeCapacityPolicy((32,)),
        project,
        residual,
        lambda points: (jnp.ones(points.shape[0]), jnp.ones(points.shape[0])),
    )
    assert repair.converged and repair.inserted_probes == (10,)
    target = _ring_component(np.asarray(repair.points))
    source_epoch = TopologyEpoch(0, "exchange-ring:23", "ring", "serial")
    target_epoch = TopologyEpoch(1, "exchange-ring:24", "ring", "serial")
    # Independent exactly normalized nonnegative allocation between the two
    # samplings: conservative by construction, without a solve.
    old, new = np.asarray(surface.mass_diagonal), np.asarray(target.mass_diagonal)
    gap = (
        np.asarray(target.owner.points)[:, None, :]
        - np.asarray(surface.owner.points)[None, :, :]
    )
    affinity = np.exp(-np.sum(gap * gap, axis=-1) / 0.01)
    coefficients = affinity * old[None, :] / (new @ affinity)[None, :]
    rows, columns = np.divmod(np.arange(new.size * old.size), old.size)
    transfer = PointTransferPlan(
        EdgeRelation(columns, rows, source_size=old.size, target_size=new.size),
        coefficients.reshape(-1),
        old,
        new,
        source_id=surface.owner_id,
        target_id=target.owner_id,
        request=PointTransferRequest("conservative-positive"),
    ).prepare()
    assert transfer.admitted
    return RingEpoch(
        bulk,
        surface,
        target,
        law,
        MeshfreeEpochChange(
            source_epoch, target_epoch, cause="sample-repair", proposal=repair
        ),
        transfer.epoch_transition(source_epoch, target_epoch),
    )


def _exchange_steps(
    law: SurfaceExchangeLaw,
    components: dict[str, MeshfreeComponent],
    amounts: tuple[Array, Array],
    steps: int,
) -> tuple[Array, Array]:
    """Explicit amount steps on the law's own rows, certified at every state."""
    prepared = law.prepare(components, ())
    contribution = prepared.contributions[0]
    if not isinstance(contribution, ResidualContribution):
        raise TypeError("Surface exchange lowers one residual contribution.")
    bulk_measures = components["bulk"].mass_diagonal
    bulk_amount, surface_amount = amounts
    for _ in range(steps):
        bulk = bulk_amount / bulk_measures
        surface = surface_amount / law.measures
        report = prepared.certificate.defects(
            {("bulk", "concentration"): bulk, ("surface", "concentration"): surface},
            (),
            {},
        )
        assert bool(report.accepted(1e-10))
        bulk_rows, surface_rows = contribution.residual.evaluate((bulk, surface), None)
        bulk_amount = bulk_amount - EXCHANGE_STEP * bulk_rows
        surface_amount = surface_amount - EXCHANGE_STEP * surface_rows
    return bulk_amount, surface_amount


def test_moving_surface_epoch_rebinds_exchange_and_state_atomically(
    ring_epoch: RingEpoch,
) -> None:
    law, surface, target = ring_epoch.law, ring_epoch.surface, ring_epoch.target
    initial = (ring_epoch.bulk.mass_diagonal, 0.1 * surface.mass_diagonal)
    total = float(jnp.sum(initial[0]) + jnp.sum(initial[1]))
    bulk_amount, surface_amount = _exchange_steps(law, ring_epoch.components, initial, 3)
    history = surface_amount / law.measures
    bulk_amount, surface_amount = _exchange_steps(
        law, ring_epoch.components, (bulk_amount, surface_amount), 1
    )
    current = surface_amount / law.measures
    relocation = law.relocate_at_epoch(
        ring_epoch.components,
        target,
        _ring_normals(target.owner.points),
        (current, history),
        (ring_epoch.transition, ring_epoch.transition),
        ring_epoch.change,
        window_start=4 * EXCHANGE_STEP,
        accepted_boundary=True,
    )
    assert relocation.published and relocation.epoch.failed == ()
    relocated = relocation.law
    assert relocation.surface is target
    assert relocated.query.admitted_count == 24 != law.query.admitted_count
    assert relocated.deposition == relocated.evidence.deposition == "positive"
    assert relocated.evidence.geometry_epoch == 1
    assert relocation.epoch.composition.value("surface/epoch").epoch_id == (
        ring_epoch.change.target.epoch_id
    )
    # Every live surface history keeps its content through its own route.
    for moved, source in zip(relocation.surface_states, (current, history), strict=True):
        assert moved.shape == (24,)
        np.testing.assert_allclose(
            jnp.vdot(target.mass_diagonal, moved),
            jnp.vdot(law.measures, source),
            rtol=1e-13,
        )
    assert bool(
        jnp.all(
            jnp.abs(relocation.conservation_residuals) <= relocation.content_tolerances
        )
    )
    rebound = {"bulk": ring_epoch.bulk, "surface": target}
    with pytest.raises(ValueError, match="owner revision"):
        relocated.prepare(ring_epoch.components, ())
    with pytest.raises(ValueError, match="owner revision"):
        law.prepare(rebound, ())
    # Integration continues on the target epoch with the total amount conserved.
    bulk_amount, surface_amount = _exchange_steps(
        relocated,
        rebound,
        (bulk_amount, target.mass_diagonal * relocation.surface_states[0]),
        3,
    )
    adsorbed = float(jnp.sum(surface_amount) - jnp.sum(initial[1]))
    assert adsorbed > 1e-4
    np.testing.assert_allclose(
        float(jnp.sum(bulk_amount) + jnp.sum(surface_amount)), total, rtol=1e-13
    )


def test_refused_surface_epoch_publishes_nothing(ring_epoch: RingEpoch) -> None:
    law, surface, target = ring_epoch.law, ring_epoch.surface, ring_epoch.target
    current = jnp.full(law.measures.shape, 0.2)
    history = jnp.full(law.measures.shape, 0.1)
    routes = (ring_epoch.transition, ring_epoch.transition)
    normals = _ring_normals(target.owner.points)
    unaccepted = law.relocate_at_epoch(
        ring_epoch.components,
        target,
        normals,
        (current, history),
        routes,
        ring_epoch.change,
        window_start=1.0,
        accepted_boundary=False,
    )
    assert not unaccepted.published and unaccepted.epoch.failed == ()
    assert unaccepted.law is law and unaccepted.surface is surface
    np.testing.assert_array_equal(unaccepted.surface_states[0], current)
    np.testing.assert_array_equal(unaccepted.surface_states[1], history)
    failed = law.relocate_at_epoch(
        ring_epoch.components,
        target,
        normals,
        (current, history.at[3].set(jnp.nan)),
        routes,
        ring_epoch.change,
        window_start=1.0,
        accepted_boundary=True,
    )
    assert not failed.published and failed.epoch.failed == ("surface/state/1",)
    assert failed.law is law and failed.surface is surface
    np.testing.assert_array_equal(failed.surface_states[0], current)
    # The source exchange stays current against its own surface.
    law.prepare(ring_epoch.components, ())
    with pytest.raises(ValueError, match="target surface capacity"):
        law.relocate_at_epoch(
            ring_epoch.components,
            ring_epoch.surface,
            _ring_normals(surface.owner.points),
            (current,),
            (ring_epoch.transition,),
            ring_epoch.change,
            window_start=1.0,
            accepted_boundary=True,
        )
